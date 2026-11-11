# =====================================================================
# File: neuralnetwork_v1.py
# Dự án: Driver Guardian - Module DataSystem
# Tác vụ: Mô hình suy luận phát hiện buồn ngủ kết hợp trích xuất khuôn mặt,
#         68 điểm mốc (FaceBoxesV2 + PIPNet) và mạng nơ-ron LiteFat Multimodal
#         (MobileNetV3-Small + STGL) với chế độ use_visual_context=True.
# Cấu hình: Đồng bộ 100% siêu tham số căn chỉnh khuôn mặt theo Stage 2 (run_stage2.py):
#   - letterbox_size: (640, 640)
#   - face_size: (224, 224)
#   - bbox_margin: 0.15 (15% mở rộng cực trị landmark và bbox)
#   - box_smooth_alpha: 0.7 (bộ lọc EMA làm mượt hộp cắt)
#   - visual_frame_stride: 8 (chỉ cắt 1 ảnh mặt mỗi 8 frame để tối ưu 87.5% CPU/VRAM)
#   - use_visual_context: True
# Checkpoint mặc định: ai/LiteFat/checkpoints/LiteFat_v6_stage2_visual/best.pt
# =====================================================================
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ai.LiteFat.config import TrainConfig

# Tự động cấu hình sys.path để nạp đúng các module LiteFat và DataSystem
_CURRENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CURRENT_DIR.parents[2]
_LITEFAT_DIR = _CURRENT_DIR.parents[1] / "LiteFat"

for _p in (_REPO_ROOT, _LITEFAT_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from ai.LiteFat.model import LiteFat
from ai.LiteFat.dataset import FastPIPNetExtractor
from ai.LiteFat.ultis import crop_faces_from_boxes, _parse_size
from ai.LiteFat.pipnet_68_camera import draw_68_landmarks_custom

try:
    from ai.OptimalAlgorithsm.DataSystem.loadvideo2handle import read_video
except ImportError:
    from loadvideo2handle import read_video

from ai.OptimalAlgorithsm.DataSystem.config import (
    DEFAULT_CONFIG,
    DataSystemConfig,
    BENCHMARK_RUNS,
    BENCHMARK_WARMUP,
    LETTERBOX_COLOR,
    MODEL_IMAGE_SIZE,
    VIDEO_ROOT,
)

# Đường dẫn mặc định tới checkpoint Stage 2 Multimodal
DEFAULT_CHECKPOINT_PATH = _LITEFAT_DIR / "checkpoints" / "LiteFat_v6_stage2_visual" / "best.pt"


def load_model_from_checkpoint(ckpt_dir: Path, device: torch.device) -> Tuple[LiteFat, TrainConfig, float]:
    """Nạp mô hình LiteFat, cấu hình và ngưỡng tối ưu từ thư mục checkpoint."""
    cfg_path = ckpt_dir / "config.json"
    weights_path = ckpt_dir / "best.pt"
    if not weights_path.is_file():
        weights_path = ckpt_dir / "last.pt"

    # Đọc cấu hình
    if cfg_path.is_file():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg = TrainConfig(**cfg_dict)
    else:
        cfg = TrainConfig()

    # Đọc ngưỡng tối ưu đã lưu
    optimal_thr = 0.50
    test_m_path = ckpt_dir / "test_metrics.json"
    best_m_path = ckpt_dir / "best_metrics.json"

    if test_m_path.is_file():
        try:
            with open(test_m_path, "r", encoding="utf-8") as f:
                tm = json.load(f)
            optimal_thr = float(tm.get("optimal_threshold", 0.50))
        except Exception:
            pass
    elif best_m_path.is_file():
        try:
            with open(best_m_path, "r", encoding="utf-8") as f:
                bm = json.load(f)
            optimal_thr = float(bm.get("metrics", {}).get("optimal_threshold", 0.50))
        except Exception:
            pass

    # Xây dựng model
    model = LiteFat(
        num_nodes=68,
        feature_dim=cfg.feature_dim,
        mobilenet_proj_hidden=cfg.mobilenet_proj_hidden,
        node_embed_dim=cfg.node_embed_dim,
        residual_channels=cfg.residual_channels,
        dilation_channels=cfg.dilation_channels,
        skip_channels=cfg.skip_channels,
        end_channels=cfg.end_channels,
        num_classes=cfg.num_classes,
        num_blocks=cfg.num_blocks,
        layers_per_block=cfg.layers_per_block,
        kernel_size=cfg.kernel_size,
        pretrained_backbone=False,
        dropout=cfg.dropout,
        static_shape_ratio=getattr(cfg, "static_shape_ratio", 0.25),
    )

    ckpt_data = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = ckpt_data.get("model_state_dict", ckpt_data)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, cfg, optimal_thr


def letterbox(
        image: np.ndarray,
        new_size: Optional[Union[int, Tuple[int, int], List[int]]] = (640, 640),
        color: Tuple[int, int, int] = LETTERBOX_COLOR,
) -> np.ndarray:
    """
    Căn chỉnh kích thước ảnh về new_size giữ nguyên tỷ lệ khung hình và chèn viền màu.
    Hỗ trợ cả định dạng HWC [H, W, 3] và CHW [3, H, W].
    """
    is_chw = image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3
    if is_chw:
        image = np.transpose(image, (1, 2, 0))

    target_w, target_h = _parse_size(new_size)
    h, w = image.shape[:2]
    if w == target_w and h == target_h:
        return np.transpose(image, (2, 0, 1)) if is_chw else image

    scale = min(target_w / float(w), target_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    canvas_color = color
    if np.issubdtype(image.dtype, np.floating):
        canvas_color = tuple(c / 255.0 for c in color)

    canvas = np.full((target_h, target_w, 3), canvas_color, dtype=image.dtype)
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    canvas[pad_top: pad_top + new_h, pad_left: pad_left + new_w] = resized

    if is_chw:
        return np.transpose(canvas, (2, 0, 1))
    return canvas


class NeuralNetwork(nn.Module):
    """
    Module NeuralNetwork_v1 thay thế DrowsyClassifier bằng LiteFat Multimodal:
      - Nhận đầu vào là chuỗi ảnh [T, 3, H, W] (RGB, uint8 hoặc float).
      - Sử dụng FastPIPNetExtractor trích xuất khuôn mặt và 68 điểm mốc theo lô trên GPU.
      - Sử dụng Stride-aware Cropping để chỉ cắt khuôn mặt tại các bước nhảy visual_frame_stride=8,
        giúp loại bỏ 87.5% chi phí CPU/VRAM.
      - Chạy suy luận LiteFat Multimodal (use_visual_context=True) và trả về xác suất buồn ngủ P(drowsy) in [0, 1].
    """

    def __init__(
            self,
            checkpoint_path: Optional[Union[str, Path]] = None,
            device: Optional[Union[str, torch.device]] = None,
            visual_frame_stride: Optional[int] = None,
            letterbox_size: Optional[Tuple[int, int]] = None,
            face_size: Optional[Tuple[int, int]] = None,
            conf_threshold: Optional[float] = None,
            nms_threshold: Optional[float] = None,
            bbox_margin: Optional[float] = None,
            box_smooth_alpha: Optional[float] = None,
            pipnet_backbone: Optional[str] = None,
            enforce_frame_count: bool = False,
            chunk_size: Optional[int] = None,
            *,
            config: DataSystemConfig = DEFAULT_CONFIG,
            model: Optional[LiteFat] = None,
            extractor: Optional[FastPIPNetExtractor] = None,
            **kwargs,
    ):
        super().__init__()
        device_str = config.device if device is None else device
        self.device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))

        # 1. Nạp mô hình LiteFat
        if model is None:
            self.model, ckpt_cfg, _ = load_model_from_checkpoint(
                ckpt_dir=checkpoint_path,
                device=self.device,
                # prefer_ema=True,
            )
            ckpt_cfg = ckpt_cfg.to_dict()
            print(f"[INFO] Đã nạp xong trọng số mô hình LiteFat ở config {checkpoint_path}")
        else:
            self.model = model.to(self.device).eval()
            ckpt_cfg = {}

        # 2. Cấu hình siêu tham số (đồng bộ theo run_stage2.py và checkpoint)
        self.visual_frame_stride = int(
            visual_frame_stride
            if visual_frame_stride is not None
            else ckpt_cfg.get("visual_frame_stride", 8)
        )
        if self.visual_frame_stride < 1:
            raise ValueError("visual_frame_stride phải >= 1")

        self.letterbox_size = _parse_size(
            letterbox_size if letterbox_size is not None else ckpt_cfg.get("letterbox_size", (640, 640))
        )
        self.face_size = _parse_size(
            (224, 224)
        )
        self.conf_threshold = float(conf_threshold if conf_threshold is not None else 0.5)
        self.nms_threshold = float(nms_threshold if nms_threshold is not None else 0.3)
        self.bbox_margin = float(bbox_margin if bbox_margin is not None else 0.15)
        self.box_smooth_alpha = float(box_smooth_alpha if box_smooth_alpha is not None else 0.7)
        self.pipnet_backbone = str(pipnet_backbone if pipnet_backbone is not None else "resnet18")

        self.enforce_frame_count = bool(enforce_frame_count)
        self.frame_count = config.neural_count
        self.fps = config.neural_fps

        # 3. Khởi tạo FastPIPNetExtractor
        if extractor is None:
            self.extractor = FastPIPNetExtractor(
                backbone=self.pipnet_backbone,
                device=str(self.device),
                letterbox_size=self.letterbox_size,
                face_size=self.face_size,
                conf_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
                bbox_margin=self.bbox_margin,
                box_smooth_alpha=self.box_smooth_alpha,
            )
        else:
            self.extractor = extractor
        self.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Optional[Union[str, Path]] = None, **kwargs) -> NeuralNetwork:
        return cls(checkpoint_path=checkpoint_path, **kwargs)

    @torch.inference_mode()
    def forward(
            self,
            x: Union[np.ndarray, torch.Tensor],
            return_details: bool = False,
    ) -> Union[float, Dict[str, Any]]:
        """
        Thực hiện suy luận từ chuỗi khung hình [T, 3, H, W] -> P(drowsy) in [0, 1].

        Args:
            x: Chuỗi ảnh RGB có shape [T, 3, H, W] (torch tensor hoặc numpy array).
            return_details: Nếu True, trả về dict gồm xác suất, clip_logits, frame_logits,
                            landmarks và face_frames. Mặc định False trả về float.

        Returns:
            Xác suất buồn ngủ P(drowsy) dạng float trong khoảng [0.0, 1.0].
        """
        # 1. Xác thực định dạng đầu vào
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x)
        elif isinstance(x, torch.Tensor):
            x_tensor = x
        else:
            raise TypeError("Đầu vào x phải là một numpy array hoặc torch.Tensor")

        if x_tensor.ndim != 4 or x_tensor.shape[1] != 3 or any(s == 0 for s in x_tensor.shape):
            raise ValueError(f"Đầu vào x phải có kích thước [T, 3, H, W] không rỗng, nhận được {x_tensor.shape}")

        T, _, H, W = x_tensor.shape
        if self.enforce_frame_count and T != self.frame_count:
            raise ValueError(f"Yêu cầu chính xác {self.frame_count} frames, nhận được {T}")

        if x_tensor.dtype != torch.uint8 and (
                not x_tensor.is_floating_point()
                or not torch.isfinite(x_tensor).all()
                or x_tensor.min() < 0.0
                or x_tensor.max() > 1.0
        ):
            raise ValueError("Ảnh đầu vào phải là kiểu uint8 [0, 255] hoặc float hữu hạn trong [0, 1]")

        # 2. Chuyển đổi tensor sang list frames BGR numpy uint8 cho FastPIPNetExtractor
        x_cpu = x_tensor.detach().cpu()
        if x_cpu.is_floating_point():
            x_cpu = (x_cpu * 255.0).clamp(0.0, 255.0).to(torch.uint8)

        # Chuyển CHW RGB -> HWC BGR
        raw_frames_bgr = [
            np.ascontiguousarray(x_cpu[i].permute(1, 2, 0).numpy()[:, :, ::-1])
            for i in range(T)
        ]

        target_w, target_h = self.letterbox_size
        uniform_frames: List[np.ndarray] = []
        for f in raw_frames_bgr:
            if f.shape[1] != target_w or f.shape[0] != target_h:
                uniform_frames.append(letterbox(f, new_size=(target_w, target_h)))
            else:
                uniform_frames.append(f)

        # 3. Trích xuất khuôn mặt & 68 điểm mốc theo lô (Không cắt ảnh mặt toàn bộ để tránh nghẽn CPU)
        extract_res = self.extractor.extract_from_frames(
            uniform_frames,
            crop_faces=False,
            return_crop_boxes=True,
        )
        _, landmarks_tensor, chosen_bboxes, found, crop_boxes = extract_res

        # 4. Cắt khuôn mặt theo bước nhảy (Stride-aware Cropping) tối ưu 87.5% chi phí
        if self.visual_frame_stride > 1:
            sel_indices = list(range(0, T, self.visual_frame_stride))
            sel_frames = [uniform_frames[i] for i in sel_indices]
            sel_boxes = crop_boxes[sel_indices]
        else:
            sel_frames = uniform_frames
            sel_boxes = crop_boxes

        face_frames_tensor = crop_faces_from_boxes(
            sel_frames,
            sel_boxes,
            face_size=self.face_size,
        )  # [T_v, 3, 224, 224]

        # 5. Đưa dữ liệu lên thiết bị suy luận và gọi mô hình LiteFat
        landmarks_gpu = landmarks_tensor.unsqueeze(0).to(self.device, non_blocking=True)  # [1, T, 68, 2]
        face_frames_gpu = face_frames_tensor.unsqueeze(0).to(self.device, non_blocking=True)  # [1, T_v, 3, 224, 224]
        # batch_lengths = torch.tensor([240]).unsqueeze(0).to(self.device)
        self.model.eval()
        frame_logits, clip_logits = self.model(
            video_frames=face_frames_gpu,
            landmarks=landmarks_gpu,
            # lengths=batch_lengths,
            return_clip_logits=True,
        )

        if clip_logits.shape != (1, 2) or not torch.isfinite(clip_logits).all():
            raise ValueError(f"LiteFat trả về clip_logits không hợp lệ: {clip_logits}")

        prob = torch.softmax(clip_logits, dim=-1)[:, 1].cpu().numpy().item()

        if return_details:
            return {
                "prob": prob,
                "clip_logits": clip_logits.cpu(),
                "frame_logits": frame_logits.cpu(),
                "landmarks": landmarks_tensor,
                "face_frames": face_frames_tensor,
                "bboxes": chosen_bboxes,
                "found": found,
                "crop_boxes": crop_boxes,
            }
        return prob


# Định danh alias để hỗ trợ cả 2 cách đặt tên
NeuralNetworkV1 = NeuralNetwork


def draw_hud_dashboard(
        canvas: np.ndarray,
        prob: Optional[float],
        window_range: Tuple[float, float],
        timestamp: float,
        fps: float,
        latency_ms: Optional[float] = None,
        is_buffering: bool = False,
        is_paused: bool = False,
        buffer_progress: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Vẽ bảng điều khiển HUD Dashboard hiển thị trực quan:
      - Trạng thái và điểm xác suất buồn ngủ P(drowsy), thanh đo (gauge meter),
        nhãn phân loại [ DROWSY ] (đỏ) hoặc [ ALERT ] (xanh lá).
      - Thông tin cửa sổ trượt: Window [start - end], timestamp video, FPS hiển thị, độ trễ suy luận.
      - Hướng dẫn phím tắt ở cạnh dưới canvas.
    """
    h, w = canvas.shape[:2]

    # 1. Header banner nền tối phía trên (y: 0 -> 72)
    header_h = 72
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 24, 30), -1)
    cv2.rectangle(overlay, (0, h - 26), (w, h), (20, 24, 30), -1)
    # Blend bán trong suốt
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
    cv2.line(canvas, (0, header_h), (w, header_h), (50, 60, 75), 1)
    cv2.line(canvas, (0, h - 26), (w, h - 26), (50, 60, 75), 1)

    # 2. Tiêu đề ứng dụng
    cv2.putText(
        canvas,
        "DRIVER GUARDIAN - LITEFAT v1",
        (14, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # 3. Trạng thái & Điểm số buồn ngủ
    if is_buffering:
        curr, total = buffer_progress if buffer_progress else (0, 240)
        pct = curr / max(1, total) * 100.0
        cv2.putText(
            canvas,
            f"BUFFERING: {curr}/{total} frames ({pct:.0f}%)",
            (14, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )
        # Thanh tiến trình buffer
        bar_x1, bar_x2, bar_y1, bar_y2 = 280, 430, 40, 52
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (40, 48, 60), -1)
        fill_w = int((bar_x2 - bar_x1) * min(1.0, curr / max(1, total)))
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), (0, 215, 255), -1)
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (80, 95, 115), 1)
    elif prob is not None:
        p_val = float(np.clip(prob, 0.0, 1.0))
        is_drowsy = p_val >= 0.5
        badge_text = "[ DROWSY ]" if is_drowsy else "[ ALERT ]"
        badge_color = (0, 0, 255) if is_drowsy else (0, 255, 0)

        cv2.putText(
            canvas,
            badge_text,
            (14, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            badge_color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"P(Drowsy): {p_val * 100:.1f}%",
            (135, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        # Thanh đo xác suất (Gauge bar)
        bar_x1, bar_x2, bar_y1, bar_y2 = 280, 430, 40, 52
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (40, 48, 60), -1)
        fill_w = int((bar_x2 - bar_x1) * p_val)
        fill_color = (0, 0, 255) if p_val >= 0.6 else ((0, 200, 255) if p_val >= 0.4 else (0, 230, 0))
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), fill_color, -1)
        # Vạch ngưỡng 0.5
        thresh_x = bar_x1 + int((bar_x2 - bar_x1) * 0.5)
        cv2.line(canvas, (thresh_x, bar_y1 - 2), (thresh_x, bar_y2 + 2), (255, 255, 255), 1)
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (80, 95, 115), 1)

    # 4. Metadata (bên phải banner)
    meta_x = 445
    cv2.putText(
        canvas,
        f"Time: {timestamp:.1f}s | FPS: {fps:.1f}",
        (meta_x, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 215, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Win: [{window_range[0]:.1f}s - {window_range[1]:.1f}s]",
        (meta_x, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 215, 230),
        1,
        cv2.LINE_AA,
    )
    if latency_ms is not None:
        cv2.putText(
            canvas,
            f"Infer: {latency_ms:.1f}ms",
            (meta_x, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 5. Footer hướng dẫn phím tắt
    cv2.putText(
        canvas,
        "[SPACE] Pause/Resume  |  [Q/ESC] Quit",
        (14, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (160, 175, 190),
        1,
        cv2.LINE_AA,
    )
    if is_paused:
        cv2.putText(
            canvas,
            "[ PAUSED ]",
            (w - 105, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


def resolve_video_path(video_arg: Optional[str] = None) -> Path:
    """
    Tự động xác định đường dẫn video dataset:
      1. Nếu người dùng truyền video_arg: ưu tiên đường dẫn tuyệt đối hoặc tương đối với VIDEO_ROOT.
      2. Mặc định 1: Tìm video trong dataset UTA-RLDD (E:\\LSTM\\drowsy_dataset\\UTA-RLDD\\Fold1_part1\\04\\0.mp4).
      3. Mặc định 2: Tìm video mẫu cục bộ trong repo (ai/LSTM/docs/video/video.mp4).
      4. Mặc định 3: Tìm bất kỳ video .mp4 nào trong thư mục VIDEO_ROOT.
    """
    if video_arg:
        p = Path(video_arg).expanduser().resolve()
        if p.is_file():
            return p
        try:
            from ai.OptimalAlgorithsm.DataSystem.config import VIDEO_ROOT
            p_rel = (Path(VIDEO_ROOT) / video_arg).resolve()
            if p_rel.is_file():
                return p_rel
        except Exception:
            pass
        raise FileNotFoundError(f"Không tìm thấy video tại đường dẫn: {video_arg}")

    default_uta = Path(r"E:\LSTM\drowsy_dataset\UTA-RLDD\Fold1_part1\04\0.mp4")
    if default_uta.is_file():
        return default_uta.resolve()

    local_sample = _REPO_ROOT / "ai" / "LSTM" / "docs" / "video" / "video.mp4"
    if local_sample.is_file():
        return local_sample.resolve()

    try:
        from ai.OptimalAlgorithsm.DataSystem.config import VIDEO_ROOT
        v_root = Path(VIDEO_ROOT)
        if v_root.is_dir():
            for f in v_root.rglob("*.mp4"):
                if f.is_file():
                    return f.resolve()
    except Exception:
        pass

    raise FileNotFoundError(
        "Không tìm thấy video dataset mặc định. Vui lòng thiết lập biến VIDEO_PATH ở đầu hàm main()."
    )


def main():
    # =========================================================================
    # CÁC HẰNG SỐ CẤU HÌNH (TÙY CHỈNH TRỰC TIẾP TẠI ĐÂY)
    # =========================================================================
    VIDEO_PATH: Optional[
        str] = r"E:\LSTM\drowsy_dataset\UTA-RLDD\Fold1_part2\09\10.mp4"
    CHECKPOINT_PATH: Union[
        str, Path] = Path(
        r"D:\Project\DATN\driver-guardian\ai\LiteFat\checkpoints\LiteFat_v6_stage2_visual")  # Trọng số mô hình LiteFat Stage 2
    DEVICE: Optional[str] = None  # "cuda" hoặc "cpu"; None để tự động chọn CUDA nếu có
    FPS: float = 20.0  # Tốc độ khung hình mục tiêu (20.0 FPS khớp huấn luyện LiteFat)
    WINDOW_SEC: float = 12.0  # Thời lượng cửa sổ trượt (12.0s = 240 frame @ 20 FPS)
    STRIDE_SEC: float = 1.0  # Bước dịch cửa sổ trượt (1.0s = 20 frame)
    VISUAL_FRAME_STRIDE: int = 8  # Lấy 1 ảnh mặt mỗi 8 frame cho MobileNetV3 (tối ưu CPU/VRAM)
    LETTERBOX_SIZE: int = 640  # Kích thước ảnh vuông letterbox (640x640)
    CONF_THRESHOLD: float = 0.5  # Ngưỡng tin cậy nhận diện khuôn mặt PIPNet
    DISPLAY_DELAY_MS: int = 1  # Thời gian chờ cv2.waitKey (1ms cho tốc độ mượt mà)
    MAX_FRAMES: Optional[int] = None  # Giới hạn số frame tối đa; None để chạy hết video
    NO_DISPLAY: bool = False  # True nếu chạy môi trường không có màn hình (headless)
    # =========================================================================

    dev = torch.device(DEVICE or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] Khởi tạo NeuralNetwork_v1 trên thiết bị: {dev}")
    print(f"[*] Nạp checkpoint: {CHECKPOINT_PATH}")

    nn_v1 = NeuralNetwork(
        checkpoint_path=CHECKPOINT_PATH,
        device=dev,
        visual_frame_stride=VISUAL_FRAME_STRIDE,
        letterbox_size=(LETTERBOX_SIZE, LETTERBOX_SIZE),
        conf_threshold=CONF_THRESHOLD,
    ).eval()
    print("[*] Nạp mô hình và bộ trích xuất PIPNet thành công!")

    # Nạp video dataset
    video_path = resolve_video_path(VIDEO_PATH)
    print(f"\n[*] Đang nạp video dataset: {video_path}")

    window_frames = round(WINDOW_SEC * FPS)
    stride_frames = max(1, round(STRIDE_SEC * FPS))

    print(f"[*] Cấu hình WindowSlide:")
    print(f"    - Target FPS: {FPS:.1f}")
    print(f"    - Cửa sổ trượt: {WINDOW_SEC:.1f}s ({window_frames} frames)")
    print(f"    - Bước trượt: {STRIDE_SEC:.1f}s ({stride_frames} frames)")
    print(f"    - Visual Frame Stride: {VISUAL_FRAME_STRIDE}")
    print(f"    - Letterbox Canvas: {LETTERBOX_SIZE}x{LETTERBOX_SIZE}")

    nn_frames: deque = deque(maxlen=window_frames)
    latest_score: Optional[float] = None
    latest_landmarks: Optional[np.ndarray] = None
    latest_bbox: Optional[np.ndarray] = None
    latest_found: bool = False
    latest_latency_ms: Optional[float] = None
    window_start_sec: float = 0.0
    window_end_sec: float = WINDOW_SEC

    paused = False
    total_inferences = 0
    total_frames = 0
    scores_history: List[float] = []
    latencies_history: List[float] = []

    window_name = "Driver Guardian - LiteFat NeuralNetwork v1 (WindowSlide)"
    if not NO_DISPLAY:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, LETTERBOX_SIZE, LETTERBOX_SIZE)

    video_reader = read_video(
        str(video_path),
        target_fps=FPS,
        target_luma=DEFAULT_CONFIG.target_luma,
        min_duration_sec=0,
    )

    prev_time = time.perf_counter()
    print("[*] Bắt đầu xử lý luồng video... Nhấn [SPACE] để tạm dừng, [Q/ESC] để thoát.\n")

    try:
        for i, (frame_rgb_chw, timestamp) in enumerate(video_reader):
            total_frames = i + 1
            count = total_frames
            if MAX_FRAMES is not None and count > MAX_FRAMES:
                break

            # 1. Chuẩn bị ảnh letterbox: [3, H, W] RGB -> [H, W, 3] RGB -> Letterbox (size, size)
            rgb_hwc = np.moveaxis(frame_rgb_chw, 0, -1)
            target_size = (LETTERBOX_SIZE, LETTERBOX_SIZE)
            if rgb_hwc.shape[1] != LETTERBOX_SIZE or rgb_hwc.shape[0] != LETTERBOX_SIZE:
                lb_rgb = letterbox(rgb_hwc, target_size)
            else:
                lb_rgb = rgb_hwc

            lb_bgr = cv2.cvtColor(lb_rgb, cv2.COLOR_RGB2BGR)

            # Lưu CHW RGB uint8 vào hàng đợi buffer cho NeuralNetwork
            chw_for_nn = np.moveaxis(lb_rgb, -1, 0)
            nn_frames.append(chw_for_nn)

            is_buffering = count < window_frames

            # 2. Điều kiện kích hoạt WindowSlide (giống engine.py)
            if count >= window_frames and (count - window_frames) % stride_frames == 0:
                window_start_sec = (count - window_frames) / FPS
                window_end_sec = count / FPS
                window_tensor = np.stack(nn_frames)  # [window_frames, 3, size, size]

                t_infer_start = time.perf_counter()
                with torch.inference_mode():
                    res = nn_v1.forward(window_tensor, return_details=True)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                latest_latency_ms = (time.perf_counter() - t_infer_start) * 1000.0

                latest_score = float(res["prob"])
                scores_history.append(latest_score)
                latencies_history.append(latest_latency_ms)
                total_inferences += 1

                # Lấy kết quả landmarks và bbox của frame mới nhất từ chuỗi trích xuất
                latest_landmarks = res["landmarks"][-1].cpu().numpy()
                latest_bbox = res["bboxes"][-1]
                latest_found = bool(res["found"][-1])

                status_label = "DROWSY" if latest_score >= 0.5 else "ALERT"
                print(
                    f"[Slide #{total_inferences:03d}] Frame {count:05d} ({timestamp:6.1f}s) | "
                    f"Win: [{window_start_sec:5.1f}s - {window_end_sec:5.1f}s] | "
                    f"P(drowsy): {latest_score:.4f} [{status_label}] | "
                    f"Latency: {latest_latency_ms:5.1f} ms",
                    flush=True,
                )
            else:
                # Trích xuất tức thời landmarks frame hiện tại để bám sát chuyển động khuôn mặt
                extract_res = nn_v1.extractor.extract_from_frames(
                    [lb_bgr], crop_faces=False, return_crop_boxes=True
                )
                _, lms_batch, bboxes_batch, found_batch, _ = extract_res
                latest_landmarks = lms_batch[0].cpu().numpy()
                latest_bbox = bboxes_batch[0]
                latest_found = bool(found_batch[0])

            # 3. Vẽ 68 điểm landmark và bounding box qua hàm LiteFat draw_68_landmarks_custom
            vis_frame = lb_bgr.copy()
            target_h, target_w = vis_frame.shape[:2]

            if latest_landmarks is not None and latest_found:
                pts_px = latest_landmarks[:, :2].copy()
                pts_px[:, 0] *= float(target_w)
                pts_px[:, 1] *= float(target_h)

                if latest_bbox is not None and len(latest_bbox) >= 5 and float(latest_bbox[4]) > 0:
                    bboxes_to_draw = np.array([latest_bbox[:5]], dtype=np.float32)
                else:
                    lx1, ly1 = float(np.min(pts_px[:, 0])), float(np.min(pts_px[:, 1]))
                    lx2, ly2 = float(np.max(pts_px[:, 0])), float(np.max(pts_px[:, 1]))
                    bboxes_to_draw = np.array([[lx1, ly1, lx2, ly2, 0.99]], dtype=np.float32)

                draw_68_landmarks_custom(
                    vis_frame,
                    np.expand_dims(pts_px, 0),
                    bboxes_to_draw,
                    conf_threshold=CONF_THRESHOLD,
                )

            # 4. Tính toán FPS hiển thị và vẽ HUD Dashboard
            curr_now = time.perf_counter()
            disp_fps = 1.0 / max(1e-5, curr_now - prev_time)
            prev_time = curr_now

            vis_frame = draw_hud_dashboard(
                vis_frame,
                prob=latest_score,
                window_range=(window_start_sec, window_end_sec),
                timestamp=timestamp,
                fps=disp_fps,
                latency_ms=latest_latency_ms,
                is_buffering=is_buffering,
                is_paused=paused,
                buffer_progress=(count, window_frames) if is_buffering else None,
            )

            # 5. Hiển thị cửa sổ OpenCV và xử lý phím bấm
            if not NO_DISPLAY:
                cv2.imshow(window_name, vis_frame)

                while True:
                    wait_ms = 0 if paused else max(1, DISPLAY_DELAY_MS)
                    key = cv2.waitKey(wait_ms) & 0xFF

                    if key in (27, ord('q'), ord('Q')):
                        print("\n[*] Người dùng yêu cầu dừng phát video.")
                        return
                    elif key == 32:  # Phím SPACE
                        paused = not paused
                        if paused:
                            paused_frame = draw_hud_dashboard(
                                vis_frame.copy(),
                                prob=latest_score,
                                window_range=(window_start_sec, window_end_sec),
                                timestamp=timestamp,
                                fps=disp_fps,
                                latency_ms=latest_latency_ms,
                                is_buffering=is_buffering,
                                is_paused=True,
                                buffer_progress=(count, window_frames) if is_buffering else None,
                            )
                            cv2.imshow(window_name, paused_frame)
                            continue
                        else:
                            break
                    else:
                        break

    except KeyboardInterrupt:
        print("\n[*] Đã nhận tín hiệu ngắt (Ctrl+C).")
    finally:
        if not NO_DISPLAY:
            cv2.destroyAllWindows()

        print(f"\n{'=' * 65}")
        print("[*] TỔNG KẾT KẾT QUẢ SUY LUẬN WINDOWSLIDE VIDEO")
        print(f"{'=' * 65}")
        print(f"[*] Tổng số frame đã đọc : {total_frames}")
        print(f"[*] Tổng số cửa sổ trượt: {total_inferences}")
        if scores_history:
            scores_arr = np.array(scores_history)
            lats_arr = np.array(latencies_history)
            print(
                f"[*] Điểm P(drowsy) trung bình: {scores_arr.mean():.4f} (Min: {scores_arr.min():.4f}, Max: {scores_arr.max():.4f})")
            print(f"[*] Tỷ lệ cảnh báo buồn ngủ : {(scores_arr >= 0.5).mean() * 100:.1f}%")
            print(f"[*] Độ trễ suy luận trung bình: {lats_arr.mean():.1f} ms/cửa sổ")
        print(f"{'=' * 65}\n")

        print(f"\n{'=' * 65}")
        print("[*] TỔNG KẾT KẾT QUẢ SUY LUẬN WINDOWSLIDE VIDEO")
        print(f"{'=' * 65}")
        print(f"[*] Tổng số frame đã đọc : {total_frames}")
        print(f"[*] Tổng số cửa sổ trượt: {total_inferences}")
        if scores_history:
            scores_arr = np.array(scores_history)
            lats_arr = np.array(latencies_history)
            print(
                f"[*] Điểm P(drowsy) trung bình: {scores_arr.mean():.4f} (Min: {scores_arr.min():.4f}, Max: {scores_arr.max():.4f})")
            print(f"[*] Tỷ lệ cảnh báo buồn ngủ : {(scores_arr >= 0.5).mean() * 100:.1f}%")
            print(f"[*] Độ trễ suy luận trung bình: {lats_arr.mean():.1f} ms/cửa sổ")
        print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
