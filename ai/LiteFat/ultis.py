# =====================================================================
# File: ultis.py
# Dự án: Driver Guardian - Module LiteFat
# Tác vụ: Thư viện tiện ích dùng chung: xử lý ảnh, cắt vùng khuôn mặt,
#         tăng cường dữ liệu đa phương thức đối xứng 68 điểm mốc,
#         thiết lập hạt giống tái lập, logging và lưu trữ checkpoint.
# =====================================================================
from __future__ import annotations

import logging
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Bảng hoán vị đối xứng trái - phải cho 68 điểm mốc khuôn mặt chuẩn 300-W
FLIP_MAP_68 = [
    # 0..16: Đường viền hàm dưới (17 điểm) hoán vị đối xứng
    16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    # 17..26: Lông mày trái (17..21) <-> Lông mày phải (22..26)
    26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
    # 27..35: Sống mũi (27..30 đối xứng chính nó) & Cánh mũi (31..35)
    27, 28, 29, 30, 35, 34, 33, 32, 31,
    # 36..47: Mắt trái (36..41) <-> Mắt phải (42..47)
    45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40,
    # 48..67: Viền môi ngoài (48..59) & Viền môi trong (60..67)
    54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55,
    64, 63, 62, 61, 60, 67, 66, 65,
]


def _parse_size(size: Optional[Union[int, Tuple[int, int], List[int]]]) -> Tuple[int, int]:
    """
    Chuẩn hóa kích thước về tuple (width, height).
    """
    if size is None:
        return (640, 640)
    if isinstance(size, int):
        return (size, size)
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return (int(size[0]), int(size[1]))
    raise ValueError(f"Kích thước không hợp lệ: {size}. Phải là int hoặc tuple (w, h).")


def letterbox(
        image: np.ndarray,
        new_size: Optional[int | Tuple[int, int] | List[int]] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """
    Resize ảnh giữ nguyên tỷ lệ khung hình gốc với đệm màu viền về kích thước đích (w, h).
    - image: Ảnh BGR numpy array shape (H, W, 3).
    - new_size: Kích thước đích theo (w, h) (vd: (640, 480)) hoặc số nguyên S tương ứng (S, S).
    - color: Màu đệm viền BGR, mặc định (114, 114, 114).
    Trả về ảnh numpy array có shape (target_h, target_w, 3).
    """
    target_w, target_h = _parse_size(new_size)
    if target_w <= 0 or target_h <= 0:
        return image

    h, w = image.shape[:2]
    if w == target_w and h == target_h:
        return image

    scale = min(target_w / float(w), target_h / float(h))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (nw, nh), interpolation=interp)

    if nw == target_w and nh == target_h:
        return resized

    canvas = np.full((target_h, target_w, 3), color, dtype=image.dtype)
    x, y = (target_w - nw) // 2, (target_h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Tính Intersection over Union (IoU) giữa 2 hộp giới hạn [x1, y1, x2, y2].
    """
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def crop_faces_from_boxes(
        frames: List[np.ndarray],
        crop_boxes: Union[np.ndarray, torch.Tensor],
        face_size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """
    Cắt vùng khuôn mặt chuẩn hóa 224x224 cho nhánh MobileNetV3-Small từ crop_boxes:
      - Nhận vào danh sách frames BGR (đã qua letterbox) và crop_boxes [T, 4] dạng [x1, y1, x2, y2].
      - Resize về face_size (w, h), chuyển sang RGB [0, 1] và chuẩn hóa ImageNet mean/std.
      - Trả về torch.Tensor [T, 3, crop_h, crop_w].
    """
    T = len(frames)
    crop_w, crop_h = _parse_size(face_size)
    if T == 0:
        return torch.empty((0, 3, crop_h, crop_w), dtype=torch.float32)

    if isinstance(crop_boxes, torch.Tensor):
        crop_boxes = crop_boxes.detach().cpu().numpy()
    crop_boxes = np.asarray(crop_boxes, dtype=np.int32)

    face_crops = np.zeros((T, crop_h, crop_w, 3), dtype=np.uint8)

    for t in range(T):
        if t < len(crop_boxes):
            x1, y1, x2, y2 = crop_boxes[t]
        else:
            x1, y1, x2, y2 = 0, 0, frames[t].shape[1], frames[t].shape[0]

        crop = frames[t][y1:y2, x1:x2, :]
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            crop = frames[t]

        resized = cv2.resize(crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        face_crops[t] = resized[:, :, ::-1]  # Chuyển BGR sang RGB

    # Chuẩn hóa ImageNet vector hóa cho toàn bộ chuỗi T ảnh
    face_tensor = torch.from_numpy(face_crops).permute(0, 3, 1, 2).float() / 255.0
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    face_tensor = (face_tensor - norm_mean) / norm_std
    return face_tensor


def augment_multimodal_clip(
        face_frames: Optional[torch.Tensor],
        landmarks: torch.Tensor,
        is_train: bool = True,
        flip_prob: float = 0.5,
        scale_jitter: float = 0.08,
        shift_jitter: float = 0.05,
        roll_deg: float = 12.0,
        temporal_dropout: float = 0.05,
        landmark_noise_std: float = 0.002,
        color_jitter: float = 0.2,
        spatial_dropout: float = 0.15,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    Tăng cường dữ liệu không gian - thời gian đồng bộ cho cả chuỗi điểm mốc và ảnh mặt:
    1. Lật ngang đối xứng (Horizontal Flip) đồng bộ cả mốc và ảnh:
       - Tọa độ x -> (1.0 - x).
       - Hoán vị đỉnh giải phẫu 68 điểm theo FLIP_MAP_68 (mắt trái <-> mắt phải).
       - Lật trục ngang chiều rộng của face_frames.
    2. Quay 2D trong mặt phẳng (In-plane Roll Rotation) góc ngẫu nhiên quanh tâm mặt.
    3. Co giãn bất đẳng hướng (Anisotropic Scale: scale_x != scale_y) và dịch chuyển (Shift Jitter).
    4. Thêm nhiễu Gaussian nhẹ (Landmark Noise) trên tọa độ điểm mốc hợp lệ.
    5. Che khuất tạm thời vài khung hình (Temporal Frame Dropout) mô phỏng mất tracking.
    6. Biến đổi độ sáng/độ tương phản (Color Jitter) nhất quán thời gian trên face_frames.
    7. Cắt ngẫu nhiên vùng ảnh (Random Spatial Cutout / Erasing) trên face_frames.
    """
    if not is_train:
        return face_frames, landmarks

    lms = landmarks.clone()
    frames = face_frames.clone() if face_frames is not None else None

    # Xác định các điểm hợp lệ (không phải đệm 1.0)
    valid_pts = (lms < 0.999).all(dim=-1, keepdim=True)  # [T, 68, 1]

    # 1. Lật ngang đối xứng (Horizontal Flip)
    if random.random() < flip_prob:
        # Lật tọa độ x của điểm mốc
        lms[..., 0] = torch.where(valid_pts.squeeze(-1), 1.0 - lms[..., 0], lms[..., 0])
        # Hoán vị đỉnh giải phẫu đối xứng chuẩn 300W
        flip_idx = torch.as_tensor(FLIP_MAP_68, dtype=torch.long, device=lms.device)
        lms = lms[:, flip_idx, :]

        # Lật ngang chiều rộng của ảnh khuôn mặt
        if frames is not None:
            frames = torch.flip(frames, dims=[-1])

    # 2. Quay 2D góc nghiêng đầu (In-plane Roll Rotation quanh tâm mặt)
    if roll_deg > 0 and random.random() < 0.5:
        angle_deg = random.uniform(-roll_deg, roll_deg)
        theta = math.radians(angle_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        valid_coords = lms[valid_pts.squeeze(-1)]
        if valid_coords.numel() > 0:
            center_x = valid_coords[:, 0].mean().item()
            center_y = valid_coords[:, 1].mean().item()

            dx = lms[..., 0] - center_x
            dy = lms[..., 1] - center_y
            rot_x = dx * cos_t - dy * sin_t + center_x
            rot_y = dx * sin_t + dy * cos_t + center_y

            rot_x = torch.clamp(rot_x, 0.0, 1.0)
            rot_y = torch.clamp(rot_y, 0.0, 1.0)

            lms[..., 0] = torch.where(valid_pts.squeeze(-1), rot_x, lms[..., 0])
            lms[..., 1] = torch.where(valid_pts.squeeze(-1), rot_y, lms[..., 1])

    # 3. Co giãn gần như đẳng hướng (Near-Isotropic Scale Jitter) và dịch chuyển (Shift)
    # Giới hạn biến dạng tỷ lệ trục y/x trong ngưỡng ±1.5% để bảo toàn độ mở mắt EAR và miệng MAR
    if scale_jitter > 0 or shift_jitter > 0:
        base_scale = 1.0 + random.uniform(-scale_jitter, scale_jitter)
        aspect_ratio = 1.0 + random.uniform(-0.015, 0.015)
        scale_x = base_scale
        scale_y = base_scale * aspect_ratio
        shift_x = random.uniform(-shift_jitter, shift_jitter)
        shift_y = random.uniform(-shift_jitter, shift_jitter)

        # Tính tâm trung bình của các điểm hợp lệ
        valid_coords = lms[valid_pts.squeeze(-1)]
        if valid_coords.numel() > 0:
            center_x = valid_coords[:, 0].mean().item()
            center_y = valid_coords[:, 1].mean().item()

            new_x = (lms[..., 0] - center_x) * scale_x + center_x + shift_x
            new_y = (lms[..., 1] - center_y) * scale_y + center_y + shift_y

            new_x = torch.clamp(new_x, 0.0, 1.0)
            new_y = torch.clamp(new_y, 0.0, 1.0)

            lms[..., 0] = torch.where(valid_pts.squeeze(-1), new_x, lms[..., 0])
            lms[..., 1] = torch.where(valid_pts.squeeze(-1), new_y, lms[..., 1])

    # 4. Nhiễu Gaussian mượt theo thời gian (Temporally Smoothed Landmark Noise)
    # Bộ lọc 3-tap [0.25, 0.50, 0.25] tránh tạo ra gai đạo hàm bậc 2 (acc) giả,
    # và giữ nguyên tọa độ gốc của 2 khóe mắt (36, 45) để bảo vệ mẫu số IOD
    if landmark_noise_std > 0:
        raw_noise = torch.randn_like(lms) * landmark_noise_std
        if lms.shape[0] >= 3:
            smooth_noise = 0.5 * raw_noise
            smooth_noise[1:-1] += 0.25 * (raw_noise[:-2] + raw_noise[2:])
            smooth_noise[0] += 0.25 * raw_noise[0]
            smooth_noise[-1] += 0.25 * raw_noise[-1]
            noise = smooth_noise
        else:
            noise = raw_noise

        # Không làm nhiễu 2 nút gốc đo khoảng cách mắt (36, 45)
        noise[:, 36, :] = 0.0
        noise[:, 45, :] = 0.0

        noisy_lms = torch.clamp(lms + noise, 0.0, 1.0)
        lms = torch.where(valid_pts, noisy_lms, lms)

    # 5. Temporal Frame Dropout: che khuất ngẫu nhiên 1-3 khung hình mô phỏng mất tracking
    if temporal_dropout > 0 and random.random() < 0.3:
        T_len = lms.shape[0]
        num_drop = min(max(1, int(round(T_len * temporal_dropout))), 4)
        if T_len > num_drop:
            start_drop = random.randint(0, T_len - num_drop)
            lms[start_drop:start_drop + num_drop] = 1.0
            if frames is not None:
                frames[start_drop:start_drop + num_drop] = 0.0

    # 6. Tăng cường trên khung hình khuôn mặt (nhất quán thời gian)
    if frames is not None and frames.numel() > 0:
        # Biến đổi độ sáng & tương phản nhất quán cho toàn bộ T frames
        if color_jitter > 0 and random.random() < 0.5:
            brightness_factor = 1.0 + random.uniform(-color_jitter, color_jitter)
            contrast_factor = 1.0 + random.uniform(-color_jitter, color_jitter)
            mean_val = frames.mean(dim=(-2, -1), keepdim=True)
            frames = (frames - mean_val) * contrast_factor + mean_val
            frames = frames * brightness_factor

        # Spatial Dropout / Cutout: Che khuất một vùng nhỏ ngẫu nhiên trên mặt
        if spatial_dropout > 0 and random.random() < spatial_dropout:
            _, _, H, W = frames.shape
            cut_h = int(H * random.uniform(0.1, 0.25))
            cut_w = int(W * random.uniform(0.1, 0.25))
            y1 = random.randint(0, max(0, H - cut_h))
            x1 = random.randint(0, max(0, W - cut_w))
            frames[:, :, y1:y1 + cut_h, x1:x1 + cut_w] = 0.0

    return frames, lms


def seed_everything(seed: int = 42) -> None:
    """Thiết lập hạt giống ngẫu nhiên để tái lập kết quả thực nghiệm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_logging(output_dir: Union[str, Path]) -> logging.Logger:
    """Cấu hình logging đồng thời ra Console và File log."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_file = out_path / "train.log"

    logger = logging.getLogger("DriverGuardian.LiteFat")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Xóa các handler cũ nếu có
    for h in list(logger.handlers):
        logger.removeHandler(h)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_h = logging.StreamHandler(sys.stdout)
    console_h.setLevel(logging.INFO)
    console_h.setFormatter(formatter)
    logger.addHandler(console_h)

    file_h = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
    file_h.setLevel(logging.INFO)
    file_h.setFormatter(formatter)
    logger.addHandler(file_h)

    return logger


def atomic_save(obj: Any, path: Union[str, Path]) -> None:
    """Lưu tệp an toàn (Atomic Save) chống lỗi tệp hỏng khi bị ngắt đột ngột."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=str(p.parent), suffix=".tmp")
    tmp_path = Path(tmp_file.name)
    try:
        tmp_file.close()
        torch.save(obj, tmp_path)
        if p.exists():
            p.unlink()
        tmp_path.rename(p)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise e


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Tính toán các chỉ số đánh giá phân loại buồn ngủ:
    Accuracy, Precision, Recall, Specificity, F1-Score, F2-Score và ROC-AUC (nếu có probs).
    """
    labels = np.asarray(labels, dtype=int).ravel()
    preds = np.asarray(preds, dtype=int).ravel()

    tp = int(np.sum((labels == 1) & (preds == 1)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    tn = int(np.sum((labels == 0) & (preds == 0)))
    total = len(labels)

    acc = (tp + tn) / max(1, total)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    f1 = (2.0 * prec * rec) / max(1e-8, prec + rec) if (prec + rec) > 0 else 0.0
    f2 = (5.0 * prec * rec) / max(1e-8, 4.0 * prec + rec) if (4.0 * prec + rec) > 0 else 0.0

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
        "f2": float(f2),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

    if probs is not None:
        try:
            from sklearn.metrics import roc_auc_score
            probs_arr = np.asarray(probs, dtype=float).ravel()
            if len(np.unique(labels)) > 1:
                metrics["auc"] = float(roc_auc_score(labels, probs_arr))
            else:
                metrics["auc"] = 0.5
        except Exception:
            metrics["auc"] = 0.5

    return metrics
