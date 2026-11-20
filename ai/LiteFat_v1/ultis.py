# =====================================================================
# File: ultis.py
# Dự án: Driver Guardian - Module LiteFat_v1 (KSS 1-9, Sliding 60s/30s)
# Tác vụ: Thư viện hàm phụ trợ dùng chung:
#           1. Tiền xử lý ảnh, cắt vùng mặt & augmentation đối xứng 68 mốc.
#           2. Khởi tạo seed, logging, lưu tệp an toàn (atomic_save) & ModelEMA.
#           3. Tính toán chỉ số đánh giá KSS 1-9 (MAE, RMSE, Within-1, Exact,
#              QWK, Pearson/Spearman, 3-Regime, Binary Drowsy F1/AUC, val_score).
#           4. Nạp & khôi phục checkpoint (load_checkpoint, safe_load_optimizer,
#              load_pretrained_weights).
# =====================================================================
from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

# Bảng hoán vị đối xứng trái - phải cho 68 điểm mốc khuôn mặt chuẩn 300-W
FLIP_MAP_68 = [
    16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
    26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
    27, 28, 29, 30, 35, 34, 33, 32, 31,
    45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40,
    54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55,
    64, 63, 62, 61, 60, 67, 66, 65,
]


# =====================================================================
# 1. TIỆN ÍCH XỬ LÝ ẢNH & TĂNG CƯỜNG DỮ LIỆU ĐA PHƯƠNG THỨC
# =====================================================================
def _parse_size(size: Optional[Union[int, Tuple[int, int], List[int]]]) -> Tuple[int, int]:
    """Chuẩn hóa kích thước về tuple (width, height)."""
    if size is None:
        return (640, 640)
    if isinstance(size, int):
        return (size, size)
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return (int(size[0]), int(size[1]))
    raise ValueError(f"Kích thước không hợp lệ: {size}. Phải là int hoặc tuple (w, h).")


def letterbox(
        image: np.ndarray,
        new_size: Optional[Union[int, Tuple[int, int], List[int]]] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """Resize ảnh giữ nguyên tỷ lệ khung hình gốc kèm đệm màu viền về kích thước đích (w, h)."""
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
    """Tính Intersection over Union (IoU) giữa 2 hộp giới hạn [x1, y1, x2, y2]."""
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
    """Cắt vùng khuôn mặt từ crop_boxes [T, 4], chuyển RGB và chuẩn hóa ImageNet."""
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
        face_crops[t] = resized[:, :, ::-1]

    face_tensor = torch.from_numpy(face_crops).permute(0, 3, 1, 2).float() / 255.0
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (face_tensor - norm_mean) / norm_std


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
    """Tăng cường dữ liệu không gian - thời gian đồng bộ cho cả chuỗi điểm mốc và ảnh mặt."""
    if not is_train:
        return face_frames, landmarks

    lms = landmarks.clone()
    frames = face_frames.clone() if face_frames is not None else None
    valid_pts = (lms < 0.999).all(dim=-1, keepdim=True)

    if random.random() < flip_prob:
        lms[..., 0] = torch.where(valid_pts.squeeze(-1), 1.0 - lms[..., 0], lms[..., 0])
        flip_idx = torch.as_tensor(FLIP_MAP_68, dtype=torch.long, device=lms.device)
        lms = lms[:, flip_idx, :]
        if frames is not None:
            frames = torch.flip(frames, dims=[-1])

    if roll_deg > 0 and random.random() < 0.5:
        angle_deg = random.uniform(-roll_deg, roll_deg)
        theta = math.radians(angle_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        valid_coords = lms[valid_pts.squeeze(-1)]
        if valid_coords.numel() > 0:
            center_x = valid_coords[:, 0].mean().item()
            center_y = valid_coords[:, 1].mean().item()
            dx = lms[..., 0] - center_x
            dy = lms[..., 1] - center_y
            rot_x = torch.clamp(dx * cos_t - dy * sin_t + center_x, 0.0, 1.0)
            rot_y = torch.clamp(dx * sin_t + dy * cos_t + center_y, 0.0, 1.0)
            lms[..., 0] = torch.where(valid_pts.squeeze(-1), rot_x, lms[..., 0])
            lms[..., 1] = torch.where(valid_pts.squeeze(-1), rot_y, lms[..., 1])

    if scale_jitter > 0 or shift_jitter > 0:
        base_scale = 1.0 + random.uniform(-scale_jitter, scale_jitter)
        aspect_ratio = 1.0 + random.uniform(-0.015, 0.015)
        scale_x, scale_y = base_scale, base_scale * aspect_ratio
        shift_x = random.uniform(-shift_jitter, shift_jitter)
        shift_y = random.uniform(-shift_jitter, shift_jitter)
        valid_coords = lms[valid_pts.squeeze(-1)]
        if valid_coords.numel() > 0:
            center_x = valid_coords[:, 0].mean().item()
            center_y = valid_coords[:, 1].mean().item()
            new_x = torch.clamp((lms[..., 0] - center_x) * scale_x + center_x + shift_x, 0.0, 1.0)
            new_y = torch.clamp((lms[..., 1] - center_y) * scale_y + center_y + shift_y, 0.0, 1.0)
            lms[..., 0] = torch.where(valid_pts.squeeze(-1), new_x, lms[..., 0])
            lms[..., 1] = torch.where(valid_pts.squeeze(-1), new_y, lms[..., 1])

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

        noise[:, 36, :] = 0.0
        noise[:, 45, :] = 0.0
        noisy_lms = torch.clamp(lms + noise, 0.0, 1.0)
        lms = torch.where(valid_pts, noisy_lms, lms)

    if temporal_dropout > 0 and random.random() < 0.3:
        T_len = lms.shape[0]
        num_drop = min(max(1, int(round(T_len * temporal_dropout))), 4)
        if T_len > num_drop:
            start_drop = random.randint(0, T_len - num_drop)
            lms[start_drop:start_drop + num_drop] = 1.0
            if frames is not None and frames.shape[0] > 0:
                T_v = frames.shape[0]
                f_start = min(T_v - 1, int(round(start_drop * T_v / float(T_len))))
                f_num = max(1, int(round(num_drop * T_v / float(T_len))))
                frames[f_start:min(T_v, f_start + f_num)] = 0.0

    if frames is not None and frames.numel() > 0:
        if color_jitter > 0 and random.random() < 0.5:
            brightness_factor = 1.0 + random.uniform(-color_jitter, color_jitter)
            contrast_factor = 1.0 + random.uniform(-color_jitter, color_jitter)
            mean_val = frames.mean(dim=(-2, -1), keepdim=True)
            frames = (frames - mean_val) * contrast_factor + mean_val
            frames = frames * brightness_factor

        if spatial_dropout > 0 and random.random() < spatial_dropout:
            _, _, H, W = frames.shape
            cut_h = int(H * random.uniform(0.1, 0.25))
            cut_w = int(W * random.uniform(0.1, 0.25))
            y1 = random.randint(0, max(0, H - cut_h))
            x1 = random.randint(0, max(0, W - cut_w))
            frames[:, :, y1:y1 + cut_h, x1:x1 + cut_w] = 0.0

    return frames, lms


# =====================================================================
# 2. HỆ THỐNG, LOGGING, ATOMIC SAVE & MODEL EMA
# =====================================================================
def seed_everything(seed: int = 42) -> None:
    """Thiết lập hạt giống ngẫu nhiên để tái lập kết quả thực nghiệm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_logging(
        output_dir: Union[str, Path],
        logger_name: str = "DriverGuardian.LiteFat_v1",
) -> logging.Logger:
    """Cấu hình logging đồng thời ra Console và File log `train.log`."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_file = out_path / "train.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for h in list(logger.handlers):
        try:
            h.close()
        except Exception:
            pass
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
    """Lưu tệp an toàn (Atomic Save) chống hỏng tệp khi bị ngắt giữa chừng."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=str(p.parent), suffix=".tmp")
    tmp_path = Path(tmp_file.name)
    try:
        tmp_file.close()
        torch.save(obj, tmp_path)
        tmp_path.replace(p)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise e


class ModelEMA:
    """Theo dõi trọng số mô hình bằng Exponential Moving Average (EMA) có warmup và đồng bộ buffer."""

    def __init__(self, model: nn.Module, decay: float = 0.990):
        self.decay = float(decay)
        self.num_updates = 0
        self.param_keys: Set[str] = {k for k, _ in model.named_parameters()}
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.clone().detach() for k, v in model.state_dict().items()
        }

    def _effective_decay(self) -> float:
        """Warmup decay để tránh giữ trọng số khởi tạo ngẫu nhiên quá lâu khi số step/epoch ít."""
        return min(self.decay, float(1 + self.num_updates) / float(5 + self.num_updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        eff_decay = self._effective_decay()
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                continue
            target = v.detach().to(self.shadow[k].device)
            if v.dtype.is_floating_point:
                # Làm mượt EMA cho cả tham số và floating-point buffers (BatchNorm running_mean/running_var)
                self.shadow[k].lerp_(target, 1.0 - eff_decay)
            else:
                # Các buffer số nguyên (num_batches_tracked, ...) sao chép trực tiếp
                self.shadow[k].copy_(target)

    def state_dict(self) -> Dict[str, Any]:
        return {"shadow": self.shadow, "num_updates": self.num_updates}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if isinstance(state_dict, dict) and "shadow" in state_dict and isinstance(state_dict["shadow"], dict):
            src = state_dict["shadow"]
            self.num_updates = int(state_dict.get("num_updates", 0))
        else:
            src = state_dict
        for k, v in src.items():
            if k in self.shadow and isinstance(v, torch.Tensor) and self.shadow[k].shape == v.shape:
                self.shadow[k].copy_(v.to(self.shadow[k].device))

    def apply_shadow(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Sao lưu trọng số hiện tại và nạp trọng số EMA shadow vào mô hình."""
        backup = {k: v.clone().detach() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)
        return backup

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        """Khôi phục lại trọng số ban đầu từ backup."""
        model.load_state_dict(backup, strict=False)


# =====================================================================
# 3. CÁC HÀM PHỤ TRỢ TÍNH TOÁN CHỈ SỐ ĐÁNH GIÁ KSS 1-9 & NHỊ PHÂN
# =====================================================================
def _average_ranks(arr: np.ndarray) -> np.ndarray:
    """Tính thứ hạng trung bình có xử lý giá trị trùng (ties)."""
    x = np.asarray(arr, dtype=np.float64).ravel()
    n = len(x)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    unique_vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    if len(unique_vals) < n:
        sum_ranks = np.bincount(inv, weights=ranks)
        ranks = (sum_ranks / counts)[inv]
    return ranks


def binary_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Tính ROC-AUC nhị phân thuần NumPy tương đương kiểm định Mann-Whitney U."""
    y = np.asarray(labels, dtype=int).ravel()
    p = np.asarray(probs, dtype=np.float64).ravel()
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = _average_ranks(p)
    sum_pos_ranks = float(ranks[y == 1].sum())
    u_stat = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / float(n_pos * n_neg))


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tính hệ số tương quan tuyến tính Pearson."""
    a = np.asarray(y_true, dtype=np.float64).ravel()
    b = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(a) < 2:
        return 0.0
    a_c = a - np.mean(a)
    b_c = b - np.mean(b)
    denom = math.sqrt(float(np.sum(a_c ** 2) * np.sum(b_c ** 2)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(a_c * b_c) / denom)


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tính hệ số tương quan thứ hạng Spearman."""
    a = np.asarray(y_true, dtype=np.float64).ravel()
    b = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(a) < 2:
        return 0.0
    return pearson_correlation(_average_ranks(a), _average_ranks(b))


def quadratic_weighted_kappa(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        min_rating: int = 1,
        max_rating: int = 9,
) -> float:
    """
    Tính Quadratic Weighted Kappa (QWK) cho thang điểm thứ tự KSS [min_rating..max_rating].
    """
    yt = np.clip(np.rint(np.asarray(y_true, dtype=np.float64)).astype(int), min_rating, max_rating) - min_rating
    yp = np.clip(np.rint(np.asarray(y_pred, dtype=np.float64)).astype(int), min_rating, max_rating) - min_rating
    n_samples = len(yt)
    n_ratings = max(2, max_rating - min_rating + 1)
    if n_samples == 0:
        return 0.0

    O = np.zeros((n_ratings, n_ratings), dtype=np.float64)
    np.add.at(O, (yt, yp), 1.0)

    idx = np.arange(n_ratings, dtype=np.float64)
    W = ((idx[:, None] - idx[None, :]) ** 2) / float((n_ratings - 1) ** 2)

    hist_t = O.sum(axis=1)
    hist_p = O.sum(axis=0)
    E = np.outer(hist_t, hist_p) / float(n_samples)

    denom = float(np.sum(W * E))
    if denom <= 1e-12:
        return 1.0 if np.array_equal(yt, yp) else 0.0
    return float(1.0 - (np.sum(W * O) / denom))


def compute_binary_metrics(
        labels: np.ndarray,
        preds: np.ndarray,
        probs: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Tính các chỉ số phân loại nhị phân (Accuracy, Precision, Recall, Specificity, F1, F2, AUC)."""
    y = np.asarray(labels, dtype=int).ravel()
    p = np.asarray(preds, dtype=int).ravel()

    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tn = int(np.sum((y == 0) & (p == 0)))
    total = len(y)

    acc = (tp + tn) / max(1, total)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    f1 = (2.0 * prec * rec) / max(1e-8, prec + rec) if (prec + rec) > 0 else 0.0
    f2 = (5.0 * prec * rec) / max(1e-8, 4.0 * prec + rec) if (4.0 * prec + rec) > 0 else 0.0

    metrics: Dict[str, float] = {
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
        "auc": 0.5,
    }

    if probs is not None:
        try:
            probs_arr = np.asarray(probs, dtype=np.float64).ravel()
            if len(np.unique(y)) > 1:
                metrics["auc"] = binary_auc(y, probs_arr)
        except Exception:
            metrics["auc"] = 0.5

    return metrics


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x_shift = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True).clip(min=1e-12)


def compute_kss_metrics(
        targets: np.ndarray,
        pred_kss: np.ndarray,
        pred_logits: Optional[np.ndarray] = None,
        pred_probs: Optional[np.ndarray] = None,
        drowsy_threshold: float = 7.0,
        kss_min: float = 1.0,
        kss_max: float = 9.0,
) -> Dict[str, float]:
    """
    Tính bộ chỉ số toàn diện cho bài toán dự đoán KSS (thang 1..9):
      - Hồi quy liên tục: mae, rmse, within_1_acc, within_05_acc, exact_acc, pearson_r, spearman_r.
      - Thứ tự & Phân nhóm: qwk (Quadratic Weighted Kappa), regime_acc, regime_macro_f1 (3 vùng 1..4 / 5..6 / 7..9).
      - Phát hiện buồn ngủ nhị phân (KSS >= drowsy_threshold): accuracy, precision, recall, specificity, f1, f2, auc.
      - Điểm tổng hợp val_score = 0.30*(1 - mae/4) + 0.15*(1 - rmse/4.5) + 0.20*within_1_acc + 0.20*max(0, qwk) + 0.15*regime_macro_f1.
    """
    y_true = np.asarray(targets, dtype=np.float64).ravel()
    y_pred = np.asarray(pred_kss, dtype=np.float64).ravel()
    n = len(y_true)
    if n == 0:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "within_1_acc": 0.0,
            "within_05_acc": 0.0,
            "exact_acc": 0.0,
            "qwk": 0.0,
            "pearson_r": 0.0,
            "spearman_r": 0.0,
            "regime_acc": 0.0,
            "regime_macro_f1": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "f1": 0.0,
            "f2": 0.0,
            "auc": 0.5,
            "val_score": 0.0,
            "pred_mean": 0.0,
            "pred_std": 0.0,
            "pred_min": 0.0,
            "pred_max": 0.0,
            "num_samples": 0,
        }

    abs_err = np.abs(y_pred - y_true)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    within_1_acc = float(np.mean(abs_err <= 1.0))
    within_05_acc = float(np.mean(abs_err <= 0.5))

    min_int = int(round(kss_min))
    max_int = int(round(kss_max))
    y_true_int = np.clip(np.rint(y_true), min_int, max_int).astype(int)
    y_pred_int = np.clip(np.rint(y_pred), min_int, max_int).astype(int)
    exact_acc = float(np.mean(y_pred_int == y_true_int))

    qwk = quadratic_weighted_kappa(y_true_int, y_pred_int, min_rating=min_int, max_rating=max_int)
    pearson_r = pearson_correlation(y_true, y_pred)
    spearman_r = spearman_correlation(y_true, y_pred)

    # 3 vùng KSS: 0=Alert (<=4), 1=Mid (5..6), 2=Drowsy (>=7)
    reg_true = np.where(y_true >= 6.5, 2, np.where(y_true >= 4.5, 1, 0))
    reg_pred = np.where(y_pred >= 6.5, 2, np.where(y_pred >= 4.5, 1, 0))
    regime_acc = float(np.mean(reg_true == reg_pred))
    reg_f1s: List[float] = []
    for r_id in (0, 1, 2):
        if np.any(reg_true == r_id) or np.any(reg_pred == r_id):
            tp_r = float(np.sum((reg_true == r_id) & (reg_pred == r_id)))
            fp_r = float(np.sum((reg_true != r_id) & (reg_pred == r_id)))
            fn_r = float(np.sum((reg_true == r_id) & (reg_pred != r_id)))
            denom_r = 2.0 * tp_r + fp_r + fn_r
            reg_f1s.append((2.0 * tp_r / denom_r) if denom_r > 0 else 0.0)
    regime_macro_f1 = float(np.mean(reg_f1s)) if reg_f1s else 0.0

    # Chỉ số nhị phân (KSS >= drowsy_threshold)
    y_bin_true = (y_true >= drowsy_threshold).astype(int)
    y_bin_pred = (y_pred >= (drowsy_threshold - 0.5)).astype(int)

    probs_arr = None
    if pred_probs is not None:
        probs_mat = np.asarray(pred_probs, dtype=np.float64)
        if probs_mat.ndim == 2 and probs_mat.shape[1] >= 2:
            drowsy_start_idx = max(0, min(probs_mat.shape[1] - 1, int(round(drowsy_threshold - kss_min))))
            probs_arr = probs_mat[:, drowsy_start_idx:].sum(axis=-1)
    elif pred_logits is not None:
        logits_mat = np.asarray(pred_logits, dtype=np.float64)
        if logits_mat.ndim == 2 and logits_mat.shape[1] >= 2:
            probs_mat = _softmax_np(logits_mat)
            drowsy_start_idx = max(0, min(probs_mat.shape[1] - 1, int(round(drowsy_threshold - kss_min))))
            probs_arr = probs_mat[:, drowsy_start_idx:].sum(axis=-1)

    if probs_arr is None:
        probs_arr = np.clip((y_pred - kss_min) / max(1e-6, kss_max - kss_min), 0.0, 1.0)

    bin_m = compute_binary_metrics(y_bin_true, y_bin_pred, probs_arr)

    val_score = (
        0.30 * max(0.0, 1.0 - mae / 4.0)
        + 0.15 * max(0.0, 1.0 - rmse / 4.5)
        + 0.20 * within_1_acc
        + 0.20 * max(0.0, qwk)
        + 0.15 * regime_macro_f1
    )

    metrics: Dict[str, float] = {
        "mae": mae,
        "rmse": rmse,
        "within_1_acc": within_1_acc,
        "within_05_acc": within_05_acc,
        "exact_acc": exact_acc,
        "qwk": qwk,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "regime_acc": regime_acc,
        "regime_macro_f1": regime_macro_f1,
        "val_score": float(val_score),
        "pred_mean": float(np.mean(y_pred)),
        "pred_std": float(np.std(y_pred)),
        "pred_min": float(np.min(y_pred)),
        "pred_max": float(np.max(y_pred)),
        "num_samples": int(n),
    }
    metrics.update(bin_m)
    return metrics


def compute_metrics(
        labels: np.ndarray,
        preds: np.ndarray,
        probs: Optional[np.ndarray] = None,
        pred_logits: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Hàm bọc tương thích ngược:
      - Nếu mảng labels/preds nằm trong thang KSS (giá trị > 1 hoặc float), gọi compute_kss_metrics.
      - Nếu là nhãn nhị phân {0, 1}, gọi compute_binary_metrics.
    """
    y_arr = np.asarray(labels).ravel()
    p_arr = np.asarray(preds).ravel()
    is_kss = (
        pred_logits is not None
        or np.any(y_arr > 1)
        or np.any(p_arr > 1)
        or np.issubdtype(p_arr.dtype, np.floating)
    )
    if is_kss:
        return compute_kss_metrics(y_arr, p_arr, pred_logits=pred_logits, pred_probs=probs)
    return compute_binary_metrics(y_arr, p_arr, probs=probs)


# =====================================================================
# 4. CÁC HÀM PHỤ TRỢ NẠP CHECKPOINT & WARM-START
# =====================================================================
def safe_load_optimizer(
        optimizer: torch.optim.Optimizer,
        opt_state_dict: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Nạp an toàn optimizer state dict, xử lý tương thích khi số lượng param_groups
    hoặc cấu trúc nhóm tham số giữa checkpoint và mô hình hiện tại khác nhau.
    """
    saved_groups = len(opt_state_dict.get("param_groups", []))
    curr_groups = len(optimizer.param_groups)

    try:
        if saved_groups == curr_groups:
            optimizer.load_state_dict(opt_state_dict)
            return True
        if saved_groups == 1 and curr_groups == 2:
            if logger is not None:
                logger.warning(
                    "Checkpoint optimizer state chỉ có 1 param_group. "
                    "Tự động ánh xạ nhóm 0 và giữ nguyên khởi tạo mới cho nhóm 1."
                )
            adapted_dict = {
                "state": opt_state_dict.get("state", {}),
                "param_groups": [
                    opt_state_dict["param_groups"][0],
                    optimizer.state_dict()["param_groups"][1],
                ],
            }
            optimizer.load_state_dict(adapted_dict)
            return True
        if logger is not None:
            logger.warning(
                f"Không tương thích số lượng param_groups (checkpoint: {saved_groups}, hiện tại: {curr_groups}). "
                "Bỏ qua khôi phục optimizer state."
            )
        return False
    except Exception as e:
        if logger is not None:
            logger.warning(f"Không thể khôi phục optimizer_state_dict ({e}). Sử dụng optimizer mới.")
        return False


def _resolve_checkpoint_path(
        checkpoint_path: Union[str, Path],
        save_dir: Optional[Union[str, Path]] = None,
) -> Path:
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_file():
        return ckpt_path

    if save_dir is not None:
        save_p = Path(save_dir)
        alt_path = save_p / checkpoint_path
        if alt_path.is_file():
            return alt_path
        available_names = [c.name for c in sorted(save_p.glob("*.pt"))] if save_p.is_dir() else []
        raise FileNotFoundError(
            f"Không tìm thấy tệp checkpoint tại '{checkpoint_path}' hoặc '{alt_path}'. "
            f"Các checkpoint hiện có trong {save_p}: {available_names}"
        )

    raise FileNotFoundError(f"Không tìm thấy tệp checkpoint tại: '{checkpoint_path}'")


def load_checkpoint(
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        ema: Optional[ModelEMA] = None,
        save_dir: Optional[Union[str, Path]] = None,
        cfg: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        logger: Optional[logging.Logger] = None,
        strict: bool = True,
) -> Dict[str, Any]:
    """
    Nạp toàn bộ trạng thái checkpoint (model, EMA, optimizer, scheduler, GradScaler,
    best metrics & trạng thái đóng băng backbone) để tiếp tục huấn luyện hoặc đánh giá.
    """
    ckpt_path = _resolve_checkpoint_path(checkpoint_path, save_dir=save_dir)
    map_loc = device if device is not None else next(model.parameters()).device

    if logger is not None:
        logger.info(f"Đang nạp checkpoint từ: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=map_loc, weights_only=False)

    # 1. Nạp trọng số mô hình
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        except RuntimeError as e:
            if strict:
                incompat = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                non_aux_missing = [k for k in incompat.missing_keys if not k.startswith("domain_discriminator.")]
                if non_aux_missing or incompat.unexpected_keys:
                    raise e
                if logger is not None:
                    logger.warning(
                        f"Checkpoint cũ không chứa nhánh phụ trợ ({incompat.missing_keys}); "
                        "đã nạp tương thích với strict=False."
                    )
            else:
                raise
    elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        model.load_state_dict(checkpoint, strict=strict)
        checkpoint = {"model_state_dict": checkpoint, "epoch": 0}
    else:
        if logger is not None:
            logger.warning("Checkpoint không chứa 'model_state_dict' hợp lệ!")

    # 2. Khôi phục EMA shadow nếu có
    if ema is not None and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])
        if "ema_num_updates" in checkpoint:
            ema.num_updates = int(checkpoint["ema_num_updates"])

    # 3. Khôi phục epoch
    ckpt_epoch = int(checkpoint.get("epoch", 0))
    start_epoch = ckpt_epoch + 1

    # 4. Khôi phục Optimizer, Scheduler, GradScaler
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        safe_load_optimizer(optimizer, checkpoint["optimizer_state_dict"], logger=logger)

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        saved_cfg_epochs = checkpoint.get("config", {}).get("epochs", None)
        target_epochs = getattr(cfg, "epochs", None) if cfg is not None else None
        if (
            optimizer is not None
            and saved_cfg_epochs is not None
            and target_epochs is not None
            and int(saved_cfg_epochs) != int(target_epochs)
        ):
            if logger is not None:
                logger.warning(
                    f"Phát hiện thay đổi tổng số epochs ({saved_cfg_epochs} -> {target_epochs}). "
                    f"Tái lập lịch trình Cosine mượt từ Epoch {start_epoch} đến Epoch {target_epochs}."
                )
            remaining_epochs = max(1, int(target_epochs) - start_epoch + 1)
            ckpt_groups = checkpoint.get("optimizer_state_dict", {}).get("param_groups", [])
            base_lr_ref = float(
                (ckpt_groups[0].get("initial_lr") if ckpt_groups else None)
                or optimizer.param_groups[0].get("initial_lr")
                or getattr(cfg, "lr", 3e-4)
            )
            saved_lr_ref = float(
                ckpt_groups[0].get("lr", optimizer.param_groups[0].get("lr", base_lr_ref))
                if ckpt_groups
                else optimizer.param_groups[0].get("lr", base_lr_ref)
            )
            raw_ratio = saved_lr_ref / max(1e-12, base_lr_ref)
            start_ratio = min(1.0, max(0.02, raw_ratio))
            peak_ratio = min(1.0, max(0.15, start_ratio))

            def resume_lr_lambda(ep_offset: int) -> float:
                if remaining_epochs > 2 and start_ratio < peak_ratio:
                    if ep_offset == 0:
                        return start_ratio
                    progress = float(ep_offset - 1) / float(max(1, remaining_epochs - 1))
                    return peak_ratio * 0.5 * (1.0 + math.cos(math.pi * progress))
                progress = float(ep_offset) / float(max(1, remaining_epochs))
                return peak_ratio * 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=resume_lr_lambda)
        else:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                if logger is not None:
                    logger.warning(f"Không thể khôi phục scheduler_state_dict ({e}).")

    if scaler is not None and "scaler_state_dict" in checkpoint and getattr(scaler, "is_enabled", lambda: False)():
        try:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        except Exception as e:
            if logger is not None:
                logger.warning(f"Không thể khôi phục scaler_state_dict ({e}).")

    # 5. Khôi phục Best Metrics
    best_epoch = int(checkpoint.get("best_epoch", -1))
    best_val_score = float(checkpoint.get("best_val_score", -1.0))
    best_val_mae = float(checkpoint.get("best_val_mae", float("inf")))
    best_val_f1 = float(checkpoint.get("best_val_f1", -1.0))
    early_stop_counter = int(checkpoint.get("early_stop_counter", 0))

    if best_epoch < 0 and save_dir is not None:
        best_json_path = Path(save_dir) / "best_metrics.json"
        if best_json_path.is_file():
            try:
                with open(best_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if int(data.get("best_epoch", 0)) <= ckpt_epoch:
                    best_epoch = int(data["best_epoch"])
                    m = data.get("metrics", {})
                    best_val_f1 = float(m.get("f1", -1.0))
                    best_val_mae = float(m.get("mae", float("inf")))
                    best_val_score = float(m.get("val_score", -1.0))
            except Exception:
                pass

    if best_val_score < 0 and isinstance(checkpoint.get("val_metrics"), dict):
        vm = checkpoint["val_metrics"]
        best_val_f1 = float(vm.get("f1", -1.0))
        best_val_mae = float(vm.get("mae", float("inf")))
        best_val_score = float(vm.get("val_score", -1.0))
        best_epoch = ckpt_epoch

    # 6. Đồng bộ trạng thái đóng/mở băng backbone MobileNetV3 theo start_epoch
    if cfg is not None and hasattr(model, "set_backbone_trainable"):
        freeze_epochs = int(getattr(cfg, "freeze_backbone_epochs", 0))
        unfreeze_blocks = getattr(cfg, "unfreeze_last_n_blocks", 3)
        if start_epoch <= freeze_epochs:
            model.set_backbone_trainable(False)
        else:
            model.set_backbone_trainable(True, unfreeze_last_n_blocks=unfreeze_blocks)

    if logger is not None:
        logger.info("=" * 70)
        logger.info(f"===> KHÔI PHỤC CHECKPOINT THÀNH CÔNG TỪ: {ckpt_path.name}")
        logger.info(f"===> Đã hoàn thành Epoch {ckpt_epoch} | Tiếp tục từ Epoch {start_epoch}")
        logger.info(
            f"===> Best Val Score: {best_val_score:.4f} | Best MAE: {best_val_mae:.4f} | Best Epoch: {best_epoch}"
        )
        logger.info("=" * 70)

    return {
        "checkpoint_path": str(ckpt_path),
        "ckpt_epoch": ckpt_epoch,
        "start_epoch": start_epoch,
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "best_val_mae": best_val_mae,
        "best_val_f1": best_val_f1,
        "early_stop_counter": early_stop_counter,
        "scheduler": scheduler,
        "history": checkpoint.get("history", []),
        "checkpoint": checkpoint,
    }


def load_pretrained_weights(
        pretrained_path: Union[str, Path],
        model: nn.Module,
        use_visual_context: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        logger: Optional[logging.Logger] = None,
) -> Set[str]:
    """
    Nạp trọng số từng phần (Warm-start) từ checkpoint trước đó cho các tầng khớp kích thước.
    Hỗ trợ:
      - Ánh xạ block STGL theo cặp (block_idx, layer_idx).
      - Thích nghi kernel Conv2d cho TCN khi kernel_size thay đổi.
      - Tự động reset FiLM về identity khi chuyển từ Stage 1 (Landmark-Only) sang Stage 2 (Visual).
    Trả về tập tên các tham số đã được nạp từ checkpoint tiền huấn luyện.
    """
    p = Path(pretrained_path)
    if not p.is_file():
        if logger is not None:
            logger.warning(f"Không tìm thấy tệp pretrained_from tại '{pretrained_path}'. Bỏ qua warm-start.")
        return set()

    map_loc = device if device is not None else next(model.parameters()).device
    if logger is not None:
        logger.info(f"Đang nạp trọng số tiền huấn luyện (Warm-start) từ: {p}")

    checkpoint = torch.load(p, map_location=map_loc, weights_only=False)
    saved_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    curr_dict = model.state_dict()

    ckpt_cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    ckpt_has_visual = bool(ckpt_cfg.get("use_visual_context", False))
    is_landmark_to_visual = bool(use_visual_context) and (not ckpt_has_visual)
    if is_landmark_to_visual and hasattr(model, "fusion") and hasattr(model.fusion, "reset_film_to_identity"):
        if logger is not None:
            logger.info(
                "===> [Warm-Start] Chuyển từ Landmark-Only sang Visual Context: "
                "Giữ ImageNet cho MobileNetV3 và kích hoạt Zero-init Identity FiLM."
            )
        model.fusion.reset_film_to_identity()

    saved_stgl_indices = sorted({int(k.split(".")[1]) for k in saved_dict if k.startswith("stgl_blocks.")})
    curr_stgl_indices = sorted({int(k.split(".")[1]) for k in curr_dict if k.startswith("stgl_blocks.")})

    num_blocks = int(getattr(model, "num_blocks", 3))
    saved_lpb = len(saved_stgl_indices) // max(1, num_blocks) if saved_stgl_indices else 0
    curr_lpb = len(curr_stgl_indices) // max(1, num_blocks) if curr_stgl_indices else 0

    block_mapping: Dict[int, int] = {}
    if saved_lpb > 0 and curr_lpb > 0:
        for b in range(num_blocks):
            for l in range(min(saved_lpb, curr_lpb)):
                block_mapping[b * curr_lpb + l] = b * saved_lpb + l

    matched_keys: List[str] = []
    adapted_keys: List[str] = []
    skipped_keys: List[str] = []
    pretrained_param_names: Set[str] = set()

    with torch.no_grad():
        for name, curr_param in curr_dict.items():
            if is_landmark_to_visual and (name.startswith("context_encoder.") or "film_" in name):
                skipped_keys.append(f"{name} (bảo toàn ImageNet / zero-init FiLM)")
                continue

            if name.startswith("stgl_blocks."):
                parts = name.split(".")
                c_idx = int(parts[1])
                suffix = ".".join(parts[2:])
                saved_name = f"stgl_blocks.{block_mapping[c_idx]}.{suffix}" if c_idx in block_mapping else None
            else:
                saved_name = name

            if saved_name is not None and saved_name in saved_dict:
                saved_param = saved_dict[saved_name]
                if curr_param.shape == saved_param.shape:
                    curr_param.copy_(saved_param.to(curr_param.device))
                    matched_keys.append(name)
                    pretrained_param_names.add(name)
                elif (
                    curr_param.ndim == 4
                    and saved_param.ndim == 4
                    and curr_param.shape[:3] == saved_param.shape[:3]
                ):
                    curr_param.zero_()
                    k_saved = saved_param.shape[-1]
                    k_curr = curr_param.shape[-1]
                    k_copy = min(k_saved, k_curr)
                    curr_param[..., -k_copy:].copy_(saved_param[..., -k_copy:].to(curr_param.device))
                    adapted_keys.append(f"{name} ({k_saved}->{k_curr})")
                    pretrained_param_names.add(name)
                else:
                    skipped_keys.append(f"{name} (shape lệch: {tuple(saved_param.shape)} vs {tuple(curr_param.shape)})")
            else:
                if name.startswith("stgl_blocks.") and "skip_conv" in name:
                    curr_param.zero_()
                elif name.startswith("stgl_blocks.") and "res_scale" in name:
                    curr_param.fill_(1.0)
                skipped_keys.append(f"{name} (mới)")

    if logger is not None:
        logger.info("=" * 70)
        logger.info(f"===> KẾT QUẢ WARM-START TỪ {p.name}:")
        logger.info(f"===> Khớp hoàn toàn: {len(matched_keys)} | TCN adaptation: {len(adapted_keys)} | Mới/Bỏ qua: {len(skipped_keys)}")
        logger.info("=" * 70)

    return pretrained_param_names
