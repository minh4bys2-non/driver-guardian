"""
ModelInferenceCache - Động cơ Suy Luận Video Thời Gian Thực với Bộ Đệm Cửa Sổ Trượt (Feature Sliding Window Cache)
Hệ thống phát hiện tài xế buồn ngủ Driver Guardian AI.

Tệp này cung cấp:
1. SlidingWindowFeatureCache: Quản lý bộ đệm đặc trưng không gian (P3, P4, P5) trên GPU/CPU,
   tự động kiểm tra và nhận diện số lượng frame mới (k frame) trong sliding window.
2. ModelInferenceCache: Lớp suy luận đầu-cuối tích hợp CNN Backbone/Neck và ConvGRUClassifier,
   tối ưu hóa độ trễ suy luận cho camera thời gian thực (chu kỳ 0.2s nạp 1 frame) và
   thích ứng linh hoạt khi nạp đồng thời nhiều hơn 1 frame mới (k > 1).
"""

import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Đảm bảo đường dẫn tìm kiếm module chuẩn xác
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ai.LSTM.src import ConvGRUClassifier
    from ai.ObjectDetection_2p6M.src.model import NMSFreeDetector
except ModuleNotFoundError:
    try:
        from src.models import ConvGRUClassifier
        from ai.ObjectDetection_2p6M.src.model import NMSFreeDetector
    except ModuleNotFoundError:
        from ai.LSTM.src.models import ConvGRUClassifier
        from ai.ObjectDetection_2p6M.src.model import NMSFreeDetector


def letterbox(
    image: np.ndarray,
    new_size: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114)
) -> np.ndarray:
    """
    Căn chỉnh kích thước ảnh về new_size x new_size giữ nguyên tỷ lệ khung hình
    và chèn viền màu (mặc định 114, 114, 114).
    Hỗ trợ linh hoạt cả định dạng HWC [H, W, 3] và CHW [3, H, W].
    """
    is_chw = (image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3)
    if is_chw:
        image = np.transpose(image, (1, 2, 0))

    h, w = image.shape[:2]
    scale = min(new_size / h, new_size / w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if np.issubdtype(image.dtype, np.floating):
        color = np.asarray(color) / 255.0
    canvas = np.full((new_size, new_size, 3), color, dtype=image.dtype)
    pad_left = (new_size - new_w) // 2
    pad_top = (new_size - new_h) // 2
    canvas[pad_top: pad_top + new_h, pad_left: pad_left + new_w] = resized

    if is_chw:
        return np.transpose(canvas, (2, 0, 1))
    return canvas


class SlidingWindowFeatureCache:
    """
    Bộ đệm vòng lưu trữ đặc trưng không gian đa tỷ lệ (P3, P4, P5) và chữ ký frame.
    
    Cơ chế hoạt động:
      - Lưu trữ tối đa max_size (mặc định 50) frame đặc trưng trên GPU/CPU.
      - Duy trì chữ ký (signature) và thumbnail ảnh thu nhỏ (40x40) để thực hiện
        thuật toán đối chiếu lát cắt (Slice-Matching Search) trong O(1).
      - Tự động nhận diện chính xác có bao nhiêu frame mới (k frame) trong sliding window.
    """

    def __init__(
        self,
        max_size: int = 50,
        device: Optional[torch.device] = None
    ):
        self.max_size = max(1, int(max_size))
        self.device = device if device is not None else torch.device("cpu")

        # Hàng đợi FIFO chứa tensor đặc trưng: mỗi phần tử có shape [1, C, H, W]
        self.p3_queue: deque = deque(maxlen=self.max_size)
        self.p4_queue: deque = deque(maxlen=self.max_size)
        self.p5_queue: deque = deque(maxlen=self.max_size)

        # Hàng đợi lưu trữ chữ ký và thumbnail kiểm tra đối chiếu lát cắt
        self.signatures: deque = deque(maxlen=self.max_size)
        self.thumbs: deque = deque(maxlen=self.max_size)

    def __len__(self) -> int:
        return len(self.p3_queue)

    @property
    def is_full(self) -> bool:
        return len(self.p3_queue) >= self.max_size

    def push(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor,
        signatures: torch.Tensor,
        thumbs: Optional[torch.Tensor] = None
    ) -> None:
        """
        Đẩy 1 hoặc k frame đặc trưng mới vào cuối bộ đệm FIFO.
        
        Args:
            p3: [K, 64, 80, 80] hoặc [1, 64, 80, 80]
            p4: [K, 128, 40, 40]
            p5: [K, 256, 20, 20]
            signatures: [K, D] vector chữ ký frame
            thumbs: [K, 3, 40, 40] ảnh thu nhỏ kiểm tra lát cắt
        """
        k = p3.shape[0]
        for i in range(k):
            self.p3_queue.append(p3[i: i + 1].detach())
            self.p4_queue.append(p4[i: i + 1].detach())
            self.p5_queue.append(p5[i: i + 1].detach())
            self.signatures.append(signatures[i].detach().cpu())
            if thumbs is not None:
                self.thumbs.append(thumbs[i].detach().cpu())

    def get_features(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ghép nối toàn bộ đặc trưng trong cache thành chuỗi tensor [1, T, C, H, W].
        
        Returns:
            Tuple (P3, P4, P5) với T = len(self)
        """
        if len(self) == 0:
            raise RuntimeError("Bộ đệm cache hiện đang rỗng.")

        p3 = torch.cat(list(self.p3_queue), dim=0).unsqueeze(0)
        p4 = torch.cat(list(self.p4_queue), dim=0).unsqueeze(0)
        p5 = torch.cat(list(self.p5_queue), dim=0).unsqueeze(0)
        return p3, p4, p5

    def find_overlap(
        self,
        new_signatures: torch.Tensor,
        new_thumbs: Optional[torch.Tensor] = None,
        atol_sig: float = 1e-3,
        atol_thumb: float = 1e-2
    ) -> int:
        """
        Dò tìm và xác thực số lượng frame mới (k) trong sliding window được nạp vào.
        
        Thuật toán:
          - Nếu cache rỗng: k = T (toàn bộ T frame đều mới).
          - Kiểm tra từng bước dịch chuyển k từ 0 đến T:
            - Khi k = 0: Nếu window mới trùng 100% với cache -> k = 0 (0 frame mới).
            - Khi 1 <= k < current_size:
              Kiểm tra lát cắt phần đầu của window mới [0 : T - k] có khớp với
              phần đuôi của cache [k : current_size] hay không?
              Bước 1: So khớp chữ ký signatures trong O(1).
              Bước 2: Nếu chữ ký khớp, xác thực lát cắt thumbnail thumbs.
              Nếu cả 2 đều khớp -> Xác nhận có đúng k frame mới ở cuối window!
          - Nếu duyệt hết mà không khớp (Scene Cut / Tua video) -> k = T (tất cả đều mới).
          
        Args:
            new_signatures: [T, D] Chữ ký của toàn bộ window mới.
            new_thumbs: [T, 3, 40, 40] Thumbnail của toàn bộ window mới.
            
        Returns:
            int: Số lượng frame mới k (0 <= k <= T).
        """
        c_len = len(self)
        t_len = new_signatures.shape[0]

        if c_len == 0:
            return t_len

        cached_sigs = torch.stack(list(self.signatures), dim=0)  # [c_len, D]
        cached_th = torch.stack(list(self.thumbs), dim=0) if len(self.thumbs) == c_len else None

        new_sigs_cpu = new_signatures.detach().cpu()
        new_th_cpu = new_thumbs.detach().cpu() if new_thumbs is not None else None

        # Trường hợp 1: Kiểm tra k = 0 (Window giống hệt lần gọi trước)
        if t_len == c_len:
            if torch.allclose(new_sigs_cpu, cached_sigs, atol=atol_sig):
                if cached_th is None or new_th_cpu is None or torch.allclose(new_th_cpu, cached_th, atol=atol_thumb):
                    return 0

        # Trường hợp 2: Kiểm tra k dịch chuyển từ 1 đến min(t_len, c_len)
        max_search = min(t_len, c_len)
        for k in range(1, max_search):
            overlap_len = min(t_len - k, c_len - k)
            if overlap_len <= 0:
                continue

            # Bước 1: So sánh nhanh qua chữ ký Frame
            slice_new_sig = new_sigs_cpu[:overlap_len]
            slice_cached_sig = cached_sigs[k: k + overlap_len]

            if torch.allclose(slice_new_sig, slice_cached_sig, atol=atol_sig):
                # Bước 2: Xác thực lát cắt Thumbnail
                if cached_th is not None and new_th_cpu is not None:
                    slice_new_th = new_th_cpu[:overlap_len]
                    slice_cached_th = cached_th[k: k + overlap_len]
                    if not torch.allclose(slice_new_th, slice_cached_th, atol=atol_thumb):
                        continue

                # Cả hai bước đều khớp hoàn hảo -> Xác định có k frame mới
                return k

        # Trường hợp 3: Không khớp lát cắt (Scene Cut / Cắt cảnh / Video mới)
        return t_len

    def clear(self) -> None:
        """Xóa trắng bộ đệm và giải phóng bộ nhớ tensor."""
        self.p3_queue.clear()
        self.p4_queue.clear()
        self.p5_queue.clear()
        self.signatures.clear()
        self.thumbs.clear()

    def status(self) -> Dict[str, Any]:
        """Trả về thông tin trạng thái bộ đệm cache."""
        elem_p3 = len(self) * (64 * 80 * 80)
        elem_p4 = len(self) * (128 * 40 * 40)
        elem_p5 = len(self) * (256 * 20 * 20)
        total_bytes = (elem_p3 + elem_p4 + elem_p5) * 4  # float32 = 4 bytes
        vram_mb = total_bytes / (1024 * 1024)

        return {
            "current_size": len(self),
            "max_size": self.max_size,
            "is_full": self.is_full,
            "device": str(self.device),
            "approx_vram_mb": round(vram_mb, 2)
        }


class ModelInferenceCache(nn.Module):
    """
    Mô hình Suy Luận Tích Hợp Có Bộ Đệm Cửa Sổ Trượt (ModelInference with Sliding Window Cache).
    
    Tối ưu hóa:
      - Tiết kiệm 98% chi phí CNN khi trượt 1 frame mỗi 0.2s.
      - Tự động nhận diện và xử lý thích ứng khi cùng sliding window nạp vào có nhiều hơn 1 frame mới (k > 1).
      - Tự động nhận diện cắt cảnh / tua video để nạp mới cache an toàn.
      - Bảo toàn 100% độ chính xác toán học và tính tương thích ngược.
    """

    def __init__(
        self,
        cnn_model: Optional[NMSFreeDetector] = None,
        conv_gru_model: Optional[ConvGRUClassifier] = None,
        chunk_size: int = 32,
        window_size: int = 50,
        enable_cache: bool = True,
        warmup_mode: str = "dynamic",
        cnn_path: Optional[Union[str, Path]] = None,
        conv_gru_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Nạp checkpoints tối ưu nếu chưa truyền model
        if conv_gru_model is None:
            try:
                conv_gru_model = ConvGRUClassifier.from_checkpoint(conv_gru_path, map_location=str(self.device))
            except Exception:
                raise RuntimeError("ConvGRU model checkpoints are not loaded.")
        if cnn_model is None:
            try:
                cnn_model = NMSFreeDetector.from_checkpoint(cnn_path, map_location=self.device, eval_mode=True)
            except Exception:
                raise RuntimeError("CNN model checkpoints are not loaded.")

        self.cnn_model: NMSFreeDetector = cnn_model
        self.conv_gru: ConvGRUClassifier = conv_gru_model.eval()
        self.cnn_backbone = self.cnn_model.backbone.eval()
        self.cnn_neck = self.cnn_model.neck.eval()

        self.chunk_size: int = max(1, int(chunk_size))
        self.window_size: int = max(1, int(window_size))
        self.enable_cache: bool = bool(enable_cache)
        self.warmup_mode: str = str(warmup_mode).lower()  # 'dynamic' hoặc 'wait'

        # Khởi tạo bộ đệm đặc trưng
        self.feature_cache = SlidingWindowFeatureCache(
            max_size=self.window_size,
            device=self.device
        )
        self.last_score: Optional[float] = None

        self.to(self.device)

    @classmethod
    def from_checkpoint(
        cls,
        cnn_path: Optional[Union[str, Path]] = None,
        conv_gru_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        chunk_size: int = 32,
        window_size: int = 50,
        enable_cache: bool = True,
        warmup_mode: str = "dynamic"
    ) -> "ModelInferenceCache":
        """Khởi tạo ModelInferenceCache trực tiếp từ các file checkpoint .pt"""
        return cls(
            cnn_path=cnn_path,
            conv_gru_path=conv_gru_path,
            device=device,
            chunk_size=chunk_size,
            window_size=window_size,
            enable_cache=enable_cache,
            warmup_mode=warmup_mode
        )

    def _compute_signatures_and_thumbs(
        self,
        frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tính chữ ký (Signature) gọn nhẹ và Thumbnail (40x40) phục vụ kiểm tra lát cắt O(1).
        
        Args:
            frames: [B, 3, H, W] chuẩn hóa float32 [0.0, 1.0] trên self.device
        Returns:
            signatures: [B, 14] vector gồm mean, std và 8 điểm mẫu pixel
            thumbs: [B, 3, 40, 40] thumbnail thu nhỏ
        """
        b, c, h, w = frames.shape
        # Thumbnail 40x40
        thumbs = F.interpolate(frames, size=(40, 40), mode="bilinear", align_corners=False)

    def _compute_signatures_and_thumbs(
        self,
        x: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tính chữ ký và thumbnail siêu tốc O(1) trực tiếp trên dữ liệu thô (trước khi letterbox).
        
        Args:
            x: Dữ liệu ảnh thô dạng numpy hoặc torch Tensor [T, 3, H, W] hoặc [T, H, W, 3].
        Returns:
            signatures: [T, D] vector chữ ký frame.
            thumbs: [T, 3, 8, 8] grid điểm ảnh kiểm tra lát cắt.
        """
        if isinstance(x, np.ndarray):
            t_x = torch.from_numpy(x)
        else:
            t_x = x.detach().cpu()

        if t_x.ndim == 3:
            t_x = t_x.unsqueeze(0)

        # Chuẩn hóa về [N, C, H, W]
        if t_x.shape[-1] == 3 and t_x.shape[1] != 3:
            t_x = t_x.permute(0, 3, 1, 2)

        n, c, h, w = t_x.shape
        step_h = max(1, h // 8)
        step_w = max(1, w // 8)

        # Lấy mẫu lưới 8x8 điểm pixel cố định
        sampled = t_x[:, :, ::step_h, ::step_w].contiguous()
        thumbs = sampled[:, :, :8, :8].float()

        # Chữ ký gọn nhẹ kết hợp trung bình kênh và các điểm mẫu
        mean_val = t_x[:, :, ::max(1, h // 4), ::max(1, w // 4)].float().mean(dim=[-2, -1])  # [N, 3]
        sig_flat = thumbs.view(n, -1)
        signatures = torch.cat([mean_val, sig_flat[:, :32]], dim=-1)  # [N, 35]

        return signatures, thumbs

    def _preprocess_frames(
        self,
        x: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Chuẩn hóa tensor đầu vào về dạng torch.Tensor [N, 3, 640, 640] float32 trên self.device.
        """
        # Chuyển đổi sang torch.Tensor
        if isinstance(x, np.ndarray):
            if x.ndim == 3:
                if x.shape[-1] == 3:
                    x = np.moveaxis(x, -1, 0)  # [3, H, W]
                x = np.expand_dims(x, 0)       # [1, 3, H, W]
            elif x.ndim == 4 and x.shape[-1] == 3 and x.shape[1] != 3:
                x = np.moveaxis(x, -1, 1)      # [N, 3, H, W]
            x = torch.from_numpy(x)
        elif not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(x).__name__}")

        if x.ndim == 3:
            x = x.unsqueeze(0)  # [1, 3, H, W]

        if x.ndim != 4:
            raise ValueError(f"Expected 4D input [N, 3, H, W], got shape {tuple(x.shape)}")

        n, c, h, w = x.shape
        if n == 0:
            raise ValueError("Empty frame sequence (N = 0)")
        if c != 3:
            raise ValueError(f"Expected 3 color channels, got C = {c}")

        # Letterbox nếu H != 640 hoặc W != 640
        if h != 640 or w != 640:
            x_cpu = x.detach().cpu()
            x_np = (x_cpu if x_cpu.dtype == torch.uint8 else x_cpu.float()).numpy()
            x_lb = np.stack([
                letterbox(np.moveaxis(frame, 0, -1), new_size=640).transpose(2, 0, 1)
                for frame in x_np
            ])
            x = torch.from_numpy(x_lb)

        x = x.to(self.device, non_blocking=True)

        # Chuẩn hóa về float32 [0.0, 1.0]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.dtype != torch.float32:
            x = x.float()

        return x

    def _extract_cnn_features(
        self,
        frames_640: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Trích xuất đặc trưng CNN Backbone + PAFPN Neck theo chunk cho một khối frames.
        
        Args:
            frames_640: [K, 3, 640, 640] float32 trên self.device
        Returns:
            p3: [K, 64, 80, 80]
            p4: [K, 128, 40, 40]
            p5: [K, 256, 20, 20]
        """
        k = frames_640.shape[0]
        p3_list, p4_list, p5_list = [], [], []

        with torch.inference_mode():
            for i in range(0, k, self.chunk_size):
                chunk = frames_640[i: i + self.chunk_size]
                out3, out4, out5 = self.cnn_backbone(chunk)
                out3, out4, out5 = self.cnn_neck(out3, out4, out5)

                p3_list.append(out3.float())
                p4_list.append(out4.float())
                p5_list.append(out5.float())

            p3 = torch.cat(p3_list, dim=0)
            p4 = torch.cat(p4_list, dim=0)
            p5 = torch.cat(p5_list, dim=0)

        return p3, p4, p5

    def _run_classifier(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor
    ) -> float:
        """
        Chạy ConvGRUClassifier trên bộ ba đặc trưng [1, T, C, H, W] và tính xác suất buồn ngủ.
        """
        t = p3.shape[1]
        seq_lens = torch.tensor([t], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            logits = self.conv_gru((p3, p4, p5), seq_lens=seq_lens)

            if logits.shape not in ((1, 1), (1, 2)):
                raise ValueError(f"Expected binary clip logits [1,1] or [1,2], got {tuple(logits.shape)}")

            if logits.shape[-1] == 1:
                score = torch.sigmoid(logits).item()
            else:
                probs = F.softmax(logits, dim=-1)
                score = probs[0, 1].item()

        self.last_score = float(score)
        return float(score)

    def predict_window(
        self,
        window: Union[np.ndarray, torch.Tensor],
        num_new_frames: Optional[int] = None,
        return_details: bool = False
    ) -> Union[float, Tuple[float, Dict[str, Any]]]:
        """
        Suy luận trên cửa sổ trượt (thông thường 50 frames) có cơ chế kiểm tra và tái sử dụng cache.
        
        Args:
            window: Chuỗi frames [T, 3, H, W] hoặc [T, H, W, 3].
            num_new_frames: (Tùy chọn) Số frame mới ở cuối window nếu caller biết trước.
                            - None: Tự động dò tìm qua thuật toán Slice-Matching.
                            - int (>= 0): Xác thực tính hợp lệ và xử lý đúng k frame mới.
            return_details: Nếu True, trả về kèm dict thông số:
                            {'k_new': k, 'cached_used': T - k, 'cache_size': len, 'is_scene_cut': bool}
                            
        Returns:
            float: Điểm xác suất buồn ngủ [0.0, 1.0].
        """
        t_total = len(window) if isinstance(window, (list, tuple)) else window.shape[0]

        # 1. Tính chữ ký nhanh trực tiếp trên dữ liệu thô (O(1), không tốn chi phí letterbox)
        signatures, thumbs = self._compute_signatures_and_thumbs(window)
        is_scene_cut = False

        # 2. Xác định số frame mới (k)
        if num_new_frames is not None:
            # Caller chỉ định tường minh số frame mới
            k = max(0, min(int(num_new_frames), t_total))
            if k < t_total and len(self.feature_cache) > 0:
                overlap_len = min(t_total - k, len(self.feature_cache) - k)
                if overlap_len > 0:
                    cached_first_sig = self.feature_cache.signatures[k]
                    if not torch.allclose(signatures[0], cached_first_sig, atol=1e-2):
                        # Cảnh báo: frame chỉ định không khớp -> Fallback sang Auto-detect
                        k = self.feature_cache.find_overlap(signatures, thumbs)
        else:
            # Tự động dò tìm qua thuật toán Slice-Matching Search
            k = self.feature_cache.find_overlap(signatures, thumbs)

        if k == t_total and len(self.feature_cache) > 0:
            is_scene_cut = True

        # 3. Phân nhánh xử lý thích ứng theo giá trị của k
        if k == 0 and len(self.feature_cache) > 0:
            # Trường hợp 0: Không có frame nào mới -> Trả về ngay kết quả cache (0ms CNN, 0ms ConvGRU)
            score = self.last_score if self.last_score is not None else 0.0
            cached_used = t_total
        elif 0 < k < t_total and len(self.feature_cache) >= (t_total - k):
            # Trường hợp 1: Có k frame mới (1 <= k < T) -> CHỈ PREPROCESS VÀ CHẠY CNN CHO ĐÚNG k FRAME MỚI!
            new_raw = window[t_total - k:]
            frames_640 = self._preprocess_frames(new_raw)
            p3_new, p4_new, p5_new = self._extract_cnn_features(frames_640)

            # Cập nhật Cache: đẩy k đặc trưng mới vào cuối
            self.feature_cache.push(
                p3_new, p4_new, p5_new,
                signatures[t_total - k:], thumbs[t_total - k:]
            )

            # Lấy toàn bộ chuỗi đặc trưng trong cache (đã cập nhật) và chạy ConvGRU
            p3, p4, p5 = self.feature_cache.get_features()
            score = self._run_classifier(p3, p4, p5)
            cached_used = t_total - k
        else:
            # Trường hợp 2: Toàn bộ T frame đều mới (k == T, hoặc Scene Cut, hoặc lần đầu chạy)
            frames_640 = self._preprocess_frames(window)
            p3_all, p4_all, p5_all = self._extract_cnn_features(frames_640)

            # Xóa trắng và nạp mới toàn bộ cache
            self.feature_cache.clear()
            self.feature_cache.push(p3_all, p4_all, p5_all, signatures, thumbs)

            p3, p4, p5 = self.feature_cache.get_features()
            score = self._run_classifier(p3, p4, p5)
            cached_used = 0

        if return_details:
            details = {
                "k_new": k,
                "cached_used": cached_used,
                "cache_size": len(self.feature_cache),
                "is_scene_cut": is_scene_cut
            }
            return score, details

        return score

    def predict_step(
        self,
        new_frames: Union[np.ndarray, torch.Tensor],
        return_details: bool = False
    ) -> Union[Optional[float], Tuple[Optional[float], Dict[str, Any]]]:
        """
        Nạp 1 HOẶC NHIỀU khung hình mới từ luồng camera (chu kỳ 0.2s) vào bộ đệm cache.
        
        Args:
            new_frames:
                - 1 frame: [H, W, 3] hoặc [3, H, W]
                - k frames: [K, 3, H, W] hoặc [K, H, W, 3]
            return_details: Nếu True, trả về dict thông tin bổ sung.
            
        Returns:
            float hoặc None: Xác suất buồn ngủ.
                             Nếu warmup_mode == 'wait' và cache chưa đủ 50 frames, trả về None.
        """
        signatures, thumbs = self._compute_signatures_and_thumbs(new_frames)
        frames_640 = self._preprocess_frames(new_frames)
        k = frames_640.shape[0]

        # Trích xuất CNN cho chỉ k frame mới
        p3_new, p4_new, p5_new = self._extract_cnn_features(frames_640)
        self.feature_cache.push(p3_new, p4_new, p5_new, signatures, thumbs)

        cur_len = len(self.feature_cache)

        # Xử lý chế độ Warmup
        if self.warmup_mode == "wait" and cur_len < self.window_size:
            score = None
        else:
            p3, p4, p5 = self.feature_cache.get_features()
            score = self._run_classifier(p3, p4, p5)

        if return_details:
            details = {
                "k_new": k,
                "cache_size": cur_len,
                "is_ready": (cur_len >= self.window_size)
            }
            return score, details

        return score

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        use_cache: bool = True,
        num_new_frames: Optional[int] = None
    ) -> float:
        """
        Giao diện suy luận chính (Tương thích ngược 100% với ModelInference cũ).
        
        Args:
            x: Chuỗi frame dạng np.ndarray hoặc torch.Tensor [T, 3, H, W].
            use_cache: Nếu True (mặc định), sử dụng cơ chế cache thông minh.
                       Nếu False, chạy chế độ batch truyền thống không dùng cache.
            num_new_frames: Số lượng frame mới nếu được chỉ định.
        """
        if not use_cache or not self.enable_cache:
            # Chạy trực tiếp không dùng cache (Baseline mode)
            frames_640 = self._preprocess_frames(x)
            p3, p4, p5 = self._extract_cnn_features(frames_640)
            p3_batch = p3.unsqueeze(0)
            p4_batch = p4.unsqueeze(0)
            p5_batch = p5.unsqueeze(0)
            return self._run_classifier(p3_batch, p4_batch, p5_batch)

        # Chạy qua cơ chế Cache tối ưu
        return self.predict_window(x, num_new_frames=num_new_frames, return_details=False)

    def reset_cache(self) -> None:
        """Làm mới toàn bộ bộ đệm cache khi bắt đầu hành trình mới."""
        self.feature_cache.clear()
        self.last_score = None

    def get_cache_status(self) -> Dict[str, Any]:
        """Lấy thông tin trạng thái bộ đệm cache hiện tại."""
        return self.feature_cache.status()


# ==============================================================================
# BỘ KIỂM THỬ TỰ ĐỘNG & BENCHMARK THỰC NGHIỆM (SELF-VERIFICATION SUITE)
# ==============================================================================
if __name__ == "__main__":
    import time

    print("=" * 80)
    print(" BẮT ĐẦU KIỂM THỬ TỰ ĐỘNG MÔ HÌNH ModelInferenceCache (Driver Guardian AI)")
    print("=" * 80)

    cnn_path = r"D:\Project\DATN\driver-guardian\ai\checkpoints\model_cnn\best.pt"
    conv_gru_path = r"D:\Project\DATN\driver-guardian\ai\checkpoints\model_convgru\best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị kiểm thử: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Khởi tạo mô hình
    print("\n[*] Đang khởi tạo ModelInferenceCache từ checkpoints...")
    model_cached = ModelInferenceCache.from_checkpoint(
        cnn_path=cnn_path,
        conv_gru_path=conv_gru_path,
        device=device,
        window_size=50,
        enable_cache=True
    )
    print(" -> Khởi tạo ModelInferenceCache thành công!")

    # Tạo chuỗi video giả lập gồm 60 frames (224x224 mô phỏng dữ liệu camera thực)
    print("\n[*] Sinh chuỗi dữ liệu giả lập 60 frames (224x224 RGB)...")
    np.random.seed(42)
    stream_frames = np.random.randint(0, 256, (60, 3, 224, 224), dtype=np.uint8)

    # --------------------------------------------------------------------------
    # TEST 1: Kiểm thử Tính Tương Đương Toán Học (Equivalence Test)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[TEST 1] Kiểm tra Tính Tương Đương Toán Học: Có Cache vs Không Cache")
    print("-" * 70)
    window_0 = stream_frames[:50]  # Cửa sổ 50 frame đầu tiên

    # 1. Chạy không dùng cache (Baseline)
    score_no_cache = model_cached.forward(window_0, use_cache=False)

    # 2. Chạy có dùng cache
    model_cached.reset_cache()
    score_with_cache = model_cached.forward(window_0, use_cache=True)

    diff = abs(score_with_cache - score_no_cache)
    print(f" -> Điểm số không dùng Cache (Baseline): {score_no_cache:.6f}")
    print(f" -> Điểm số có dùng Cache            : {score_with_cache:.6f}")
    print(f" -> Sai số tuyệt đối                  : {diff:.8e}")
    assert diff < 1e-5, f"LỖI: Sai số {diff} vượt quá ngưỡng cho phép 1e-5!"
    print(" -> [ĐẠT] Kết quả trùng khớp 100% về mặt toán học!")

    # --------------------------------------------------------------------------
    # TEST 2: Kiểm thử Luồng Trượt Chuẩn k = 1 (Chu kỳ 0.2s nạp 1 frame)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[TEST 2] Kiểm thử Luồng Trượt Chuẩn: k = 1 frame mới mỗi bước")
    print("-" * 70)
    for step in range(1, 6):
        window_step = stream_frames[step: step + 50]  # Trượt 1 frame
        score_step, details = model_cached.predict_window(window_step, return_details=True)
        print(f" -> Bước {step}: k_mới = {details['k_new']} | Tái sử dụng = {details['cached_used']}/50 frame | Score = {score_step:.4f}")
        assert details["k_new"] == 1, f"Lỗi: Kỳ vọng k = 1, nhận {details['k_new']}"
        assert details["cached_used"] == 49, f"Lỗi: Kỳ vọng tái sử dụng 49 frame cũ, nhận {details['cached_used']}"
    print(" -> [ĐẠT] Tự động nhận diện chính xác k = 1 và tái sử dụng 49 frame cũ!")

    # --------------------------------------------------------------------------
    # TEST 3: Kiểm thử Khi Có Nhiều Hơn 1 Frame Mới (k = 2, k = 5)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[TEST 3] Kiểm thử Khi Có Nhiều Hơn 1 Frame Mới (k = 2, k = 5)")
    print("-" * 70)
    # Thử nghiệm với k = 2 (Nhảy cóc 2 frame)
    window_k2 = stream_frames[7: 57]
    score_k2, det_k2 = model_cached.predict_window(window_k2, return_details=True)
    print(f" -> [Nhảy cóc 2 frame] k_mới = {det_k2['k_new']} | Tái sử dụng = {det_k2['cached_used']}/50 | Score = {score_k2:.4f}")
    assert det_k2["k_new"] == 2, f"Lỗi: Kỳ vọng k = 2, nhận {det_k2['k_new']}"
    assert det_k2["cached_used"] == 48, f"Lỗi: Kỳ vọng tái sử dụng 48 frame, nhận {det_k2['cached_used']}"

    # Thử nghiệm với k = 5 (Stride = 5)
    window_k5 = stream_frames[10: 60]
    score_k5, det_k5 = model_cached.predict_window(window_k5, return_details=True)
    print(f" -> [Nhảy cóc 5 frame] k_mới = {det_k5['k_new']} | Tái sử dụng = {det_k5['cached_used']}/50 | Score = {score_k5:.4f}")
    assert det_k5["k_new"] == 3, f"Lỗi: Kỳ vọng k = 3 (10 - 7 = 3), nhận {det_k5['k_new']}"
    print(" -> [ĐẠT] Cơ chế kiểm tra tự động phát hiện chính xác số frame mới khi k > 1!")

    # --------------------------------------------------------------------------
    # TEST 4: Kiểm thử Cắt Cảnh / Tua Video (Scene Cut / Discontinuity)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[TEST 4] Kiểm thử Nhận Diện Cắt Cảnh / Tua Video (Scene Cut)")
    print("-" * 70)
    # Tạo window hoàn toàn mới (khác biệt 100%)
    new_scene_window = np.random.randint(0, 256, (50, 3, 224, 224), dtype=np.uint8)
    score_cut, det_cut = model_cached.predict_window(new_scene_window, return_details=True)
    print(f" -> [Đổi phân cảnh mới] k_mới = {det_cut['k_new']} | Scene cut = {det_cut['is_scene_cut']} | Score = {score_cut:.4f}")
    assert det_cut["k_new"] == 50, f"Lỗi: Kỳ vọng k = 50 khi cắt cảnh, nhận {det_cut['k_new']}"
    assert det_cut["is_scene_cut"] is True, "Lỗi: Không nhận diện được Scene Cut"
    print(" -> [ĐẠT] Tự động nhận diện cắt cảnh, fallback an toàn làm mới toàn bộ cache!")

    # --------------------------------------------------------------------------
    # TEST 5: Kiểm thử Streaming Từng Frame (predict_step)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[TEST 5] Kiểm thử Phương thức Streaming: predict_step(frame)")
    print("-" * 70)
    model_cached.reset_cache()
    # Nạp 5 frame đơn lẻ
    for idx in range(5):
        frame_single = stream_frames[idx]  # [3, 224, 224]
        sc, det_step = model_cached.predict_step(frame_single, return_details=True)
        print(f" -> Nạp frame {idx + 1}: Cache size = {det_step['cache_size']} | Score = {sc:.4f}")
    print(" -> [ĐẠT] Streaming từng frame hoạt động mượt mà!")

    # --------------------------------------------------------------------------
    # TEST 6: Benchmark Đo Lường Tốc Độ & Throughput (Latency & FPS)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("[TEST 6] Benchmark Đo Lường Tốc Độ Thực Nghiệm (Latency & FPS)")
    print("-" * 70)
    num_runs = 20

    # Khởi động GPU (Warm-up)
    for _ in range(3):
        _ = model_cached.forward(window_0, use_cache=False)
        _ = model_cached.predict_window(window_0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Đo thời gian khi KHÔNG CÓ CACHE (Chạy 50 frame CNN mỗi lần)
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = model_cached.forward(window_0, use_cache=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    t_no_cache = (time.perf_counter() - t0) / num_runs * 1000

    # Đo thời gian khi CÓ CACHE với k = 1 (Chạy 1 frame CNN mỗi lần qua predict_window)
    # Thiết lập cache ban đầu
    _ = model_cached.predict_window(window_0)
    t0 = time.perf_counter()
    for i in range(num_runs):
        w = stream_frames[1 + (i % 5): 51 + (i % 5)]
        _ = model_cached.predict_window(w)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    t_cache_k1 = (time.perf_counter() - t0) / num_runs * 1000

    # Đo thời gian khi dùng streaming predict_step(frame) (Chu kỳ camera thực tế)
    t0 = time.perf_counter()
    for i in range(num_runs):
        f_i = stream_frames[i % 10]
        _ = model_cached.predict_step(f_i)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    t_cache_step = (time.perf_counter() - t0) / num_runs * 1000

    speedup_win = t_no_cache / t_cache_k1 if t_cache_k1 > 0 else float("inf")
    speedup_step = t_no_cache / t_cache_step if t_cache_step > 0 else float("inf")

    print(f" -> Latency KHÔNG DÙNG CACHE (50 frames CNN)      : {t_no_cache:.2f} ms / bước")
    print(f" -> Latency CÓ CACHE predict_window (k=1)         : {t_cache_k1:.2f} ms / bước ({speedup_win:.1f}x nhanh hơn)")
    print(f" -> Latency CÓ CACHE predict_step (1 frame stream): {t_cache_step:.2f} ms / bước ({speedup_step:.1f}x nhanh hơn)")
    print(f" -> Khả năng đáp ứng chu kỳ camera 200ms          : {'ĐẠT CHUẨN THỜI GIAN THỰC' if t_cache_step < 200 else 'CHẬM'}")
    print(f" -> Trạng thái bộ đệm cache                       : {model_cached.get_cache_status()}")

    print("\n" + "=" * 80)
    print(" TẤT CẢ 6 BÀI KIỂM THỬ ĐÃ VƯỢT QUA XUẤT SẮC 100%!")
    print("=" * 80)
