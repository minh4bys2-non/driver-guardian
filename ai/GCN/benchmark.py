from __future__ import annotations

import argparse
import csv
import dataclasses
from dataclasses import dataclass, field, asdict
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Đảm bảo mã hóa UTF-8 trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Cấu hình Matplotlib backend không tương tác để tránh treo giao diện khi chạy dòng lệnh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# Import các thành phần nội bộ từ hệ thống GCN
from config import TrainConfig
from dataset import collate_gcn, GCNDataset
from model import DriverSTGCN, DriverSTGCNTransformer

logger = logging.getLogger("DriverGuardian.Benchmark")


# =====================================================================
# 1. DATACLASS CẤU HÌNH ĐO LƯỜNG RIÊNG BIỆT: BenchmarkConfig
# =====================================================================
@dataclass
class BenchmarkConfig:
    """
    Cấu hình quản lý tập trung toàn bộ tiến trình Benchmark và Phân tích Ngưỡng Tối ưu.
    Hỗ trợ xuất/nạp JSON/YAML, ánh xạ tham số CLI, và tự động tìm kiếm checkpoint phù hợp.
    """
    # 1. Đường dẫn Checkpoint & Thư mục xuất báo cáo
    checkpoint_path: str = "checkpoints/GCN1/checkpoints/best.pt"
    output_dir: Optional[str] = None          # None: tự động tạo thư mục 'benchmark' cùng cấp checkpoint
    splits: List[str] = field(default_factory=lambda: ["val", "test"])

    # 2. Dữ liệu & Thư mục Cache Landmark
    dataset_dir: str = r"E:\LSTM\data_processed"
    manifest_filename: str = "dataset_merged_split.csv"
    use_cache: bool = True                    # Ưu tiên nạp từ file .pt đã trích xuất trước để suy luận siêu tốc (~15s)
    cache_dir: str = r"E:\LSTM\cache_landmarks"

    # 3. Siêu tham số Dataloader & Môi trường thực thi
    batch_size: int = 32
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True                          # Bật Automatic Mixed Precision (FP16) khi chạy CUDA
    time_interval: float = 0.1                # Khoảng lấy mẫu thời gian thực (0.1s = 10 FPS)
    window_sec: Optional[float] = 10.0        # Cửa sổ thời gian cắt clip (giây). None = toàn bộ thời lượng
    letterbox_size: int = 480

    # 4. Phân tích Ngưỡng Phân loại (Threshold Analysis & Tuning)
    default_threshold: float = 0.5            # Ngưỡng chuẩn mực mặc định
    threshold_step: float = 0.01              # Bước quét ngưỡng từ 0.00 đến 1.00 (101 điểm)
    target_metric: str = "f1"                 # Tiêu chuẩn chọn ngưỡng tối ưu: 'f1', 'f2', 'accuracy', 'youden'

    # 5. Tùy chọn Xuất Kết quả & Đồ thị
    save_plots: bool = True                   # Lưu 4 tệp biểu đồ phân giải cao (300 DPI)
    save_predictions: bool = True             # Lưu predictions_val.csv, predictions_test.csv
    save_summary_json: bool = True            # Lưu benchmark_summary.json
    dpi: int = 300                            # Độ phân giải ảnh xuất bản
    class_names: Dict[int, str] = field(default_factory=lambda: {0: "Alert", 1: "Drowsy"})

    def __post_init__(self):
        """Đồng bộ hóa các thiết lập phụ thuộc và chuẩn hóa kiểu dữ liệu."""
        self.device = self.device.lower().strip()
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("[WARNING] CUDA không khả dụng trên máy hiện tại. Chuyển thiết bị sang 'cpu'.")
            self.device = "cpu"

        # Chuẩn hóa danh sách splits
        if isinstance(self.splits, str):
            self.splits = [s.strip().lower() for s in self.splits.replace(",", " ").split() if s.strip()]
        else:
            self.splits = [str(s).strip().lower() for s in self.splits]

        # Tự động định vị thư mục xuất kết quả nếu chưa được chỉ định
        if not self.output_dir:
            ckpt_p = Path(self.checkpoint_path)
            if ckpt_p.parent.exists():
                self.output_dir = str(ckpt_p.parent / "benchmark")
            else:
                self.output_dir = "benchmark_results"

    def auto_resolve_checkpoint(self) -> None:
        """Tự động tìm kiếm checkpoint best.pt gần nhất nếu đường dẫn được chỉ định không tồn tại."""
        p = Path(self.checkpoint_path)
        if not p.is_file():
            candidates = sorted(Path("checkpoints").glob("**/best.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
            if candidates:
                logger.info(f"Không tìm thấy '{self.checkpoint_path}'. Tự động chọn checkpoint mới nhất: {candidates[0]}")
                self.checkpoint_path = str(candidates[0])
                if self.output_dir is None or "checkpoints" in self.output_dir:
                    self.output_dir = str(candidates[0].parent / "benchmark")

    def to_dict(self) -> Dict[str, Any]:
        """Xuất toàn bộ cấu hình sang dạng dictionary thuần túy."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Lưu cấu hình ra file JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkConfig:
        """Nạp cấu hình từ tệp JSON."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Không tìm thấy file cấu hình benchmark: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "class_names" in data and isinstance(data["class_names"], dict):
            data["class_names"] = {int(k): v for k, v in data["class_names"].items()}
        return cls(**data)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> BenchmarkConfig:
        """Khởi tạo cấu hình từ tham số dòng lệnh CLI."""
        kwargs = {}
        for key in (
            "checkpoint_path",
            "output_dir",
            "splits",
            "dataset_dir",
            "cache_dir",
            "use_cache",
            "batch_size",
            "device",
            "amp",
            "time_interval",
            "window_sec",
            "default_threshold",
            "target_metric",
            "save_plots",
            "save_predictions",
            "save_summary_json",
            "dpi",
        ):
            val = getattr(args, key, None)
            if val is not None:
                kwargs[key] = val

        cfg = cls(**kwargs)
        cfg.auto_resolve_checkpoint()
        return cfg


# =====================================================================
# 2. CẤU HÌNH LOGGING & TIỆN ÍCH AN TOÀN
# =====================================================================
def configure_logging(log_file: Path | str) -> None:
    """Cấu hình đồng bộ hệ thống log ra cả Console và file benchmark.log."""
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def load_checkpoint_safely(
    checkpoint_path: str | Path,
    device: torch.device,
) -> Tuple[nn.Module, TrainConfig, Dict[str, Any]]:
    """
    Nạp checkpoint một cách an toàn và tương thích ngược với mọi phiên bản TrainConfig:
    - Tự động lọc các trường thừa/thiếu giữa các bản cập nhật để không ném lỗi TypeError.
    - Khởi tạo kiến trúc mô hình chính xác (DriverSTGCN hoặc DriverSTGCNTransformer).
    - Nạp weights_state_dict và chuyển sang chế độ đánh giá eval().
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file checkpoint tại: {ckpt_path}")

    logger.info(f"Đang nạp checkpoint từ: {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)

    # 1. Tái tạo TrainConfig an toàn qua việc lọc các trường hợp lệ
    saved_cfg_dict = checkpoint.get("config", {})
    valid_train_fields = {f.name for f in dataclasses.fields(TrainConfig)}
    filtered_cfg_dict = {k: v for k, v in saved_cfg_dict.items() if k in valid_train_fields}
    cfg = TrainConfig(**filtered_cfg_dict)

    # 2. Khởi tạo mô hình tương ứng
    model_name = checkpoint.get("model_name", cfg.model_name).lower().strip()
    if model_name in ("stgcn", "driverstgcn", "light"):
        logger.info("Khởi tạo kiến trúc DriverSTGCN (Spatio-Temporal Graph CNN)...")
        model = DriverSTGCN(**cfg.get_stgcn_kwargs())
    elif model_name in ("transformer", "driverstgcntransformer", "trans"):
        logger.info("Khởi tạo kiến trúc DriverSTGCNTransformer (STGCN + Transformer)...")
        model = DriverSTGCNTransformer(**cfg.get_transformer_kwargs())
    else:
        raise ValueError(f"Không nhận dạng được kiến trúc mô hình: '{model_name}'")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    logger.info(f"Nạp mô hình thành công: {num_params:,} tham số ({model_size_mb:.2f} MB)")
    if "epoch" in checkpoint:
        logger.info(f"Checkpoint được lưu tại Epoch: {checkpoint['epoch']}")

    return model, cfg, checkpoint


# =====================================================================
# 3. DATASET VÀ BỘ TẢI DỮ LIỆU ĐO LƯỜNG (BENCHMARK DATASET & LOADER)
# =====================================================================
class BenchmarkDataset(Dataset):
    """
    Dataset tối ưu hóa tốc độ suy luận cho quá trình Benchmark:
    - Nếu có sẵn cache landmark (.pt) trong cache_dir: Nạp toàn bộ dữ liệu vào RAM chỉ một lần,
      cắt cửa sổ thời gian (Temporal Window Slicing) chính xác ở tâm clip (center-crop).
    - Nếu không có cache: Tự động fallback về đọc video gốc thông qua GCNDataset trong dataset.py.
    """
    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        manifest_filename: str = "dataset_merged_split.csv",
        time_interval: float = 0.1,
        window_sec: Optional[float] = 10.0,
        use_cache: bool = True,
        cache_dir: Optional[str | Path] = None,
        letterbox_size: int = 480,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = split.lower().strip()
        self.time_interval = float(time_interval)
        self.window_sec = float(window_sec) if window_sec is not None else None
        self.letterbox_size = letterbox_size
        self.use_cache = bool(use_cache)
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.cached: Optional[Dict[str, Dict[str, Any]]] = None
        self.samples: List[Dict[str, Any]] = []
        self._init_samples_and_cache(manifest_filename)

    def _init_samples_and_cache(self, manifest_filename: str) -> None:
        """Đọc danh sách mẫu từ manifest CSV và nạp cache landmark nếu khả dụng."""
        manifest = self.dataset_dir / manifest_filename
        if manifest.is_file():
            with open(manifest, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("split", "").strip().lower() == self.split:
                        rel_path = row.get("rel_path") or row.get("processed_path")
                        norm_key = Path(rel_path.replace("\\", "/")).as_posix()
                        self.samples.append({
                            "key": norm_key,
                            "path": self.dataset_dir / norm_key,
                            "label": int(row["label"]),
                            "video_id": row.get("video_id") or row.get("clip_id") or Path(rel_path).stem,
                            "fps": float(row.get("fps", 30.0)),
                            "total_frames": int(float(row.get("frame_count", 0))),
                            "duration_sec": float(row.get("duration_sec", 0.0)),
                        })
        else:
            # Fallback quét thư mục nếu thiếu manifest
            for label, folder in ((0, "0_alert"), (1, "1_drowsy")):
                split_dir = self.dataset_dir / self.split / folder
                if split_dir.is_dir():
                    for p in sorted(split_dir.glob("*.mp4")):
                        key = p.relative_to(self.dataset_dir).as_posix()
                        self.samples.append({
                            "key": key,
                            "path": p,
                            "label": label,
                            "video_id": p.stem,
                            "fps": 30.0,
                            "total_frames": 0,
                            "duration_sec": 0.0,
                        })

        # Nạp cache .pt vào RAM
        if self.use_cache and self.cache_dir is not None:
            cache_file = self.cache_dir / f"{self.split}.pt"
            if cache_file.is_file():
                try:
                    payload = torch.load(cache_file, map_location="cpu", weights_only=False)
                except TypeError:
                    payload = torch.load(cache_file, map_location="cpu")
                self.cached = payload.get("samples", payload)
                hit_count = sum(1 for s in self.samples if s["key"] in self.cached)
                logger.info(f"[{self.split.upper()}] Đã nạp cache '{cache_file.name}': {len(self.cached)} video trong cache | Khớp {hit_count}/{len(self.samples)} mẫu.")
            else:
                logger.warning(f"[{self.split.upper()}] Không tìm thấy file cache {cache_file}. Sẽ trích xuất trực tiếp từ video gốc.")

        # Nếu không có cache, khởi tạo fallback qua GCNDataset
        if self.cached is None:
            self.fallback_ds = GCNDataset(
                dataset_dir=self.dataset_dir,
                split=self.split,
                time_interval=self.time_interval,
                window_sec=self.window_sec,
                manifest_filename=manifest_filename,
                letterbox_size=self.letterbox_size,
            )
        else:
            self.fallback_ds = None

    def _slice_temporal_window(self, full_lm: torch.Tensor, fps: float, duration_sec: float) -> torch.Tensor:
        """
        Cắt cửa sổ thời gian window_sec trên chuỗi landmark (C=2, T_full, V=68).
        Đối với đánh giá Benchmark, vị trí cắt luôn được cố định ở chính giữa thời lượng (center-crop)
        để bảo đảm tính xác định và có thể tái lập 100%.
        """
        T = full_lm.shape[1]
        if self.window_sec is None or T <= 1:
            return full_lm.clone()

        win = min(self.window_sec, duration_sec) if duration_sec > 0 else self.window_sec
        if self.time_interval > 0:
            win_frames = len(np.arange(0, win, self.time_interval))
        else:
            win_frames = int(round(win * (fps if fps > 0 else 30.0)))
        win_frames = max(1, min(win_frames, T))

        max_start = T - win_frames
        start = max_start // 2  # Cắt tâm cố định cho Validation / Test
        return full_lm[:, start:start + win_frames, :].clone()

    def __len__(self) -> int:
        return len(self.samples) if self.samples else (len(self.fallback_ds) if self.fallback_ds else 0)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # Đường nhanh: Trả về từ Cache RAM
        if self.cached is not None:
            sample = self.samples[index]
            key = sample["key"]
            if key in self.cached:
                item = self.cached[key]
                fps = float(item.get("fps", sample.get("fps", 30.0)))
                dur = float(item.get("duration_sec", sample.get("duration_sec", 0.0)))
                lm = item["landmarks"]  # (2, T, 68)
                lm_sliced = self._slice_temporal_window(lm, fps, dur)
                label_tensor = torch.tensor(sample["label"], dtype=torch.long)
                meta = {
                    "video_id": sample["video_id"],
                    "path": str(sample["path"]),
                    "fps": fps,
                    "duration_sec": dur,
                    "num_sampled_frames": lm_sliced.shape[1],
                    "time_interval": self.time_interval,
                }
                return lm_sliced, label_tensor, meta

        # Fallback đọc từ video gốc
        if self.fallback_ds is not None:
            return self.fallback_ds[index]

        raise RuntimeError(f"Không thể tải mẫu dữ liệu tại index {index}")


def build_benchmark_dataloader(
    cfg: BenchmarkConfig,
    split: str,
) -> DataLoader:
    """Xây dựng DataLoader cho một split cụ thể."""
    ds = BenchmarkDataset(
        dataset_dir=cfg.dataset_dir,
        split=split,
        manifest_filename=cfg.manifest_filename,
        time_interval=cfg.time_interval,
        window_sec=cfg.window_sec,
        use_cache=cfg.use_cache,
        cache_dir=cfg.cache_dir,
        letterbox_size=cfg.letterbox_size,
    )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        collate_fn=collate_gcn,
    )


# =====================================================================
# 4. ĐỘNG CƠ TÍNH TOÁN CHỈ SỐ & PHÂN TÍCH QUÉT NGƯỠNG (THRESHOLD ENGINE)
# =====================================================================
class ThresholdAnalyzer:
    """
    Động cơ phân tích đường cong biến thiên của các chỉ số qua dải ngưỡng phân loại:
    - Quét toàn bộ ngưỡng tau trong [0.00, 1.00].
    - Tính toán Accuracy, Recall, Precision, Specificity, F1, F2, Balanced Accuracy, Youden's J.
    - Tìm kiếm ngưỡng tối ưu theo tiêu chí chỉ định trên tập Validation.
    """
    def __init__(self, step: float = 0.01):
        self.thresholds = np.linspace(0.0, 1.0, int(round(1.0 / step)) + 1)

    def evaluate_at_threshold(
        self,
        y_true: np.ndarray,
        y_prob_drowsy: np.ndarray,
        tau: float,
    ) -> Dict[str, float]:
        """Tính toán toàn bộ các chỉ số phân loại tại một ngưỡng phân loại cụ thể tau."""
        y_pred = (y_prob_drowsy >= tau).astype(int)
        total = len(y_true)
        if total == 0:
            return {"accuracy": 0.0, "recall": 0.0, "precision": 0.0, "f1": 0.0, "f2": 0.0, "specificity": 0.0}

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        accuracy = float((tp + tn) / total)
        recall = float(tp / max(1, tp + fn))        # Sensitivity (Khả năng phát hiện Drowsy)
        precision = float(tp / max(1, tp + fp))
        specificity = float(tn / max(1, tn + fp))  # True Negative Rate (Độ nhận diện Alert)
        f1 = float((2 * precision * recall) / max(1e-12, precision + recall))
        f2 = float((5 * precision * recall) / max(1e-12, 4 * precision + recall))
        balanced_acc = float((recall + specificity) / 2.0)
        youden_j = float(recall + specificity - 1.0)

        return {
            "threshold": float(tau),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "specificity": specificity,
            "f1": f1,
            "f2": f2,
            "balanced_accuracy": balanced_acc,
            "youden_j": youden_j,
        }

    def sweep_thresholds(
        self,
        y_true: np.ndarray,
        y_prob_drowsy: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Thực hiện quét dải ngưỡng trên toàn bộ tập dữ liệu."""
        metrics_history = {
            "thresholds": self.thresholds,
            "accuracy": [],
            "recall": [],
            "precision": [],
            "specificity": [],
            "f1": [],
            "f2": [],
            "balanced_accuracy": [],
            "youden_j": [],
        }

        for tau in self.thresholds:
            m = self.evaluate_at_threshold(y_true, y_prob_drowsy, tau)
            metrics_history["accuracy"].append(m["accuracy"])
            metrics_history["recall"].append(m["recall"])
            metrics_history["precision"].append(m["precision"])
            metrics_history["specificity"].append(m["specificity"])
            metrics_history["f1"].append(m["f1"])
            metrics_history["f2"].append(m["f2"])
            metrics_history["balanced_accuracy"].append(m["balanced_accuracy"])
            metrics_history["youden_j"].append(m["youden_j"])

        for k in metrics_history:
            metrics_history[k] = np.array(metrics_history[k])

        return metrics_history

    def find_optimal_thresholds(self, sweep_results: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Xác định các điểm ngưỡng tối ưu theo từng tiêu chí mục tiêu."""
        th = sweep_results["thresholds"]

        best_f1_idx = int(np.argmax(sweep_results["f1"]))
        best_f2_idx = int(np.argmax(sweep_results["f2"]))
        best_acc_idx = int(np.argmax(sweep_results["accuracy"]))
        best_youden_idx = int(np.argmax(sweep_results["youden_j"]))

        return {
            "best_f1": {
                "threshold": float(th[best_f1_idx]),
                "f1": float(sweep_results["f1"][best_f1_idx]),
                "f2": float(sweep_results["f2"][best_f1_idx]),
                "recall": float(sweep_results["recall"][best_f1_idx]),
                "precision": float(sweep_results["precision"][best_f1_idx]),
                "accuracy": float(sweep_results["accuracy"][best_f1_idx]),
                "specificity": float(sweep_results["specificity"][best_f1_idx]),
            },
            "best_f2": {
                "threshold": float(th[best_f2_idx]),
                "f1": float(sweep_results["f1"][best_f2_idx]),
                "f2": float(sweep_results["f2"][best_f2_idx]),
                "recall": float(sweep_results["recall"][best_f2_idx]),
                "precision": float(sweep_results["precision"][best_f2_idx]),
                "accuracy": float(sweep_results["accuracy"][best_f2_idx]),
                "specificity": float(sweep_results["specificity"][best_f2_idx]),
            },
            "best_accuracy": {
                "threshold": float(th[best_acc_idx]),
                "accuracy": float(sweep_results["accuracy"][best_acc_idx]),
                "f1": float(sweep_results["f1"][best_acc_idx]),
                "recall": float(sweep_results["recall"][best_acc_idx]),
            },
            "best_youden": {
                "threshold": float(th[best_youden_idx]),
                "youden_j": float(sweep_results["youden_j"][best_youden_idx]),
                "recall": float(sweep_results["recall"][best_youden_idx]),
                "specificity": float(sweep_results["specificity"][best_youden_idx]),
            },
        }


# =====================================================================
# 5. BỘ TRỰC QUAN HÓA BIỂU ĐỒ CHUYÊN NGHIỆP: BenchmarkPlotter
# =====================================================================
class BenchmarkPlotter:
    """
    Trực quan hóa kết quả đo lường với chất lượng đồ họa xuất bản (300 DPI):
    1. threshold_comparison_val_test.png: Đường cong biến thiên các chỉ số đối chiếu Val và Test.
    2. confusion_matrix_val_test.png: Ma trận nhầm lẫn 2x2 ở ngưỡng 0.5 và ngưỡng tối ưu.
    3. roc_pr_curves_val_test.png: Đồ thị ROC-AUC và PR-AUC đối chiếu.
    4. metrics_comparison_val_test.png: Biểu đồ cột nhóm so sánh các chỉ số hiệu năng.
    """
    def __init__(self, output_dir: Path | str, dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        # Cấu hình phong cách đồ họa Seaborn
        sns.set_theme(style="whitegrid", font="sans-serif")
        plt.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
        })

    def plot_threshold_comparison(
        self,
        val_sweep: Optional[Dict[str, np.ndarray]],
        test_sweep: Optional[Dict[str, np.ndarray]],
        val_opt: Optional[Dict[str, Any]],
        test_opt: Optional[Dict[str, Any]],
        default_threshold: float = 0.5,
    ) -> Path:
        """
        Vẽ biểu đồ phân tích ngưỡng 4 khung hình hoàn chỉnh:
        - Subplot 1: Validation Metrics vs Threshold.
        - Subplot 2: Test Metrics vs Threshold (đối chiếu ngưỡng tối ưu của Val sang Test).
        - Subplot 3: Khả năng tổng quát hóa F1 và F2 (Val vs Test Across Thresholds).
        - Subplot 4: Đánh đổi Precision - Recall qua dải ngưỡng (Trade-off Curve).
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)
        fig.suptitle("PHÂN TÍCH ĐƯỜNG CONG BIẾN THIÊN VÀ TỐI ƯU HÓA NGƯỠNG PHÂN LOẠI (THRESHOLD TUNING)", fontsize=16, fontweight="bold", y=0.98)

        colors = {
            "accuracy": "#2ca02c",     # Green
            "recall": "#d62728",       # Red
            "precision": "#1f77b4",    # Blue
            "f1": "#ff7f0e",           # Orange
            "f2": "#9467bd",           # Purple
            "specificity": "#7f7f7f",  # Gray
        }

        # 1. Khung hình (0, 0): Tập Validation
        ax1 = axes[0, 0]
        if val_sweep is not None:
            th = val_sweep["thresholds"]
            ax1.plot(th, val_sweep["accuracy"], label="Accuracy", color=colors["accuracy"], linewidth=2.0)
            ax1.plot(th, val_sweep["recall"], label="Recall (Drowsy)", color=colors["recall"], linewidth=2.2)
            ax1.plot(th, val_sweep["precision"], label="Precision", color=colors["precision"], linewidth=2.0)
            ax1.plot(th, val_sweep["f1"], label="F1-Score", color=colors["f1"], linewidth=2.5)
            ax1.plot(th, val_sweep["f2"], label="F2-Score (Recall Priority)", color=colors["f2"], linewidth=2.2, linestyle="--")

            # Đánh dấu các ngưỡng đặc biệt
            ax1.axvline(default_threshold, color="#333333", linestyle=":", linewidth=1.5, label=f"Mặc định ({default_threshold})")
            if val_opt:
                tau_f1 = val_opt["best_f1"]["threshold"]
                f1_val = val_opt["best_f1"]["f1"]
                ax1.axvline(tau_f1, color=colors["f1"], linestyle="-.", linewidth=1.8, label=f"Best F1 (tau={tau_f1:.2f})")
                ax1.scatter([tau_f1], [f1_val], color=colors["f1"], s=100, zorder=5, marker="*")

                tau_f2 = val_opt["best_f2"]["threshold"]
                f2_val = val_opt["best_f2"]["f2"]
                ax1.axvline(tau_f2, color=colors["f2"], linestyle="-.", linewidth=1.8, label=f"Best F2 (tau={tau_f2:.2f})")
                ax1.scatter([tau_f2], [f2_val], color=colors["f2"], s=100, zorder=5, marker="P")

            ax1.set_title("Tập Validation: Sự biến thiên của các chỉ số theo Ngưỡng", fontweight="bold")
            ax1.set_xlabel("Ngưỡng phân loại (Classification Threshold tau)")
            ax1.set_ylabel("Giá trị chỉ số (Score)")
            ax1.set_ylim(-0.02, 1.05)
            ax1.legend(loc="lower left", framealpha=0.9)
        else:
            ax1.text(0.5, 0.5, "Không có dữ liệu Validation", ha="center", va="center")

        # 2. Khung hình (0, 1): Tập Test
        ax2 = axes[0, 1]
        if test_sweep is not None:
            th = test_sweep["thresholds"]
            ax2.plot(th, test_sweep["accuracy"], label="Accuracy", color=colors["accuracy"], linewidth=2.0)
            ax2.plot(th, test_sweep["recall"], label="Recall (Drowsy)", color=colors["recall"], linewidth=2.2)
            ax2.plot(th, test_sweep["precision"], label="Precision", color=colors["precision"], linewidth=2.0)
            ax2.plot(th, test_sweep["f1"], label="F1-Score", color=colors["f1"], linewidth=2.5)
            ax2.plot(th, test_sweep["f2"], label="F2-Score (Recall Priority)", color=colors["f2"], linewidth=2.2, linestyle="--")

            ax2.axvline(default_threshold, color="#333333", linestyle=":", linewidth=1.5, label=f"Mặc định ({default_threshold})")
            if val_opt:
                tau_f1 = val_opt["best_f1"]["threshold"]
                ax2.axvline(tau_f1, color=colors["f1"], linestyle="-.", linewidth=1.8, label=f"Val Best F1 (tau={tau_f1:.2f})")
                tau_f2 = val_opt["best_f2"]["threshold"]
                ax2.axvline(tau_f2, color=colors["f2"], linestyle="-.", linewidth=1.8, label=f"Val Best F2 (tau={tau_f2:.2f})")

            ax2.set_title("Tập Test: Kiểm chứng chuyển dịch Ngưỡng từ Tập Val", fontweight="bold")
            ax2.set_xlabel("Ngưỡng phân loại (Classification Threshold tau)")
            ax2.set_ylabel("Giá trị chỉ số (Score)")
            ax2.set_ylim(-0.02, 1.05)
            ax2.legend(loc="lower left", framealpha=0.9)
        else:
            ax2.text(0.5, 0.5, "Không có dữ liệu Test", ha="center", va="center")

        # 3. Khung hình (1, 0): Đối chiếu Val F1/F2 vs Test F1/F2
        ax3 = axes[1, 0]
        if val_sweep is not None and test_sweep is not None:
            th = val_sweep["thresholds"]
            ax3.plot(th, val_sweep["f1"], label="Validation F1", color="#ff7f0e", linewidth=2.5)
            ax3.plot(th, test_sweep["f1"], label="Test F1", color="#ff7f0e", linewidth=2.0, linestyle="--")
            ax3.plot(th, val_sweep["f2"], label="Validation F2", color="#9467bd", linewidth=2.5)
            ax3.plot(th, test_sweep["f2"], label="Test F2", color="#9467bd", linewidth=2.0, linestyle="--")

            if val_opt:
                tau_f1 = val_opt["best_f1"]["threshold"]
                ax3.axvline(tau_f1, color="#ff7f0e", linestyle=":", label=f"Opt F1 tau={tau_f1:.2f}")
                tau_f2 = val_opt["best_f2"]["threshold"]
                ax3.axvline(tau_f2, color="#9467bd", linestyle=":", label=f"Opt F2 tau={tau_f2:.2f}")

            ax3.set_title("Khả năng Tổng quát hóa: So sánh F1 & F2 (Val vs Test)", fontweight="bold")
            ax3.set_xlabel("Ngưỡng phân loại (Threshold tau)")
            ax3.set_ylabel("Điểm F-score")
            ax3.set_ylim(-0.02, 1.05)
            ax3.legend(loc="lower left", framealpha=0.9)
        else:
            ax3.text(0.5, 0.5, "Cần cả dữ liệu Val và Test", ha="center", va="center")

        # 4. Khung hình (1, 1): Precision vs Recall Trade-off across Thresholds
        ax4 = axes[1, 1]
        if val_sweep is not None and test_sweep is not None:
            ax4.plot(val_sweep["recall"], val_sweep["precision"], label="Validation (PR Curve)", color="#1f77b4", linewidth=2.5)
            ax4.plot(test_sweep["recall"], test_sweep["precision"], label="Test (PR Curve)", color="#2ca02c", linewidth=2.5, linestyle="--")

            # Đánh dấu điểm hoạt động tại ngưỡng 0.5 và ngưỡng tối ưu
            if val_opt:
                tau_f1 = val_opt["best_f1"]["threshold"]
                idx_v = int(np.argmin(np.abs(val_sweep["thresholds"] - tau_f1)))
                ax4.scatter([val_sweep["recall"][idx_v]], [val_sweep["precision"][idx_v]], color="#ff7f0e", s=120, zorder=5, label=f"Val @ Opt F1 (tau={tau_f1:.2f})")

                idx_t = int(np.argmin(np.abs(test_sweep["thresholds"] - tau_f1)))
                ax4.scatter([test_sweep["recall"][idx_t]], [test_sweep["precision"][idx_t]], color="#d62728", s=120, zorder=5, marker="X", label=f"Test @ Opt F1 (tau={tau_f1:.2f})")

            ax4.set_title("Đánh đổi Precision - Recall qua Dải ngưỡng", fontweight="bold")
            ax4.set_xlabel("Recall (Độ nhạy - Drowsy)")
            ax4.set_ylabel("Precision (Độ chuẩn xác)")
            ax4.set_xlim(-0.02, 1.05)
            ax4.set_ylim(-0.02, 1.05)
            ax4.legend(loc="lower left", framealpha=0.9)
        else:
            ax4.text(0.5, 0.5, "Cần cả dữ liệu Val và Test", ha="center", va="center")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = self.output_dir / "threshold_comparison_val_test.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Đã lưu biểu đồ phân tích ngưỡng: {out_path.name}")
        return out_path

    def plot_confusion_matrices(
        self,
        cm_val_default: Optional[np.ndarray],
        cm_val_opt: Optional[np.ndarray],
        cm_test_default: Optional[np.ndarray],
        cm_test_opt: Optional[np.ndarray],
        opt_threshold: float,
        default_threshold: float = 0.5,
        class_labels: Tuple[str, str] = ("Alert", "Drowsy"),
    ) -> Path:
        """
        Vẽ ma trận nhầm lẫn 2x2 đối chiếu:
        - Hàng trên: Ngưỡng mặc định 0.5 (Val và Test).
        - Hàng dưới: Ngưỡng tối ưu tau_opt (Val và Test).
        Mỗi ô thể hiện rõ cả số lượng tuyệt đối và phần trăm chuẩn hóa (%).
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=self.dpi)
        fig.suptitle("ĐỐI CHIẾU MA TRẬN NHẦM LẪN (CONFUSION MATRIX): NGƯỠNG MẶC ĐỊNH VS NGƯỠNG TỐI ƯU", fontsize=15, fontweight="bold", y=0.98)

        def draw_cm(ax, cm, title, cmap):
            if cm is None:
                ax.text(0.5, 0.5, "Không có dữ liệu", ha="center", va="center")
                return
            cm_norm = cm.astype(float) / np.maximum(1, cm.sum(axis=1, keepdims=True)) * 100.0
            sns.heatmap(
                cm_norm,
                annot=False,
                cmap=cmap,
                cbar=True,
                ax=ax,
                vmin=0,
                vmax=100,
                square=True,
            )
            # Điền text hiển thị song song Count và %
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    cnt = cm[i, j]
                    pct = cm_norm[i, j]
                    color = "white" if pct > 50 else "black"
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        f"{cnt:,}\n({pct:.1f}%)",
                        ha="center",
                        va="center",
                        color=color,
                        fontweight="bold",
                        fontsize=12,
                    )
            ax.set_title(title, fontweight="bold", fontsize=12)
            ax.set_xlabel("Nhãn Dự đoán (Predicted)", fontweight="bold")
            ax.set_ylabel("Nhãn Thực tế (Actual)", fontweight="bold")
            ax.set_xticklabels(class_labels)
            ax.set_yticklabels(class_labels)

        # 1. Val @ Default 0.5
        draw_cm(axes[0, 0], cm_val_default, f"Validation (Ngưỡng mặc định = {default_threshold:.2f})", "Blues")
        # 2. Test @ Default 0.5
        draw_cm(axes[0, 1], cm_test_default, f"Test (Ngưỡng mặc định = {default_threshold:.2f})", "Blues")
        # 3. Val @ Optimal Threshold
        draw_cm(axes[1, 0], cm_val_opt, f"Validation (Ngưỡng tối ưu = {opt_threshold:.2f})", "Greens")
        # 4. Test @ Optimal Threshold
        draw_cm(axes[1, 1], cm_test_opt, f"Test (Ngưỡng tối ưu từ Val = {opt_threshold:.2f})", "Greens")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = self.output_dir / "confusion_matrix_val_test.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Đã lưu ma trận nhầm lẫn: {out_path.name}")
        return out_path

    def plot_roc_pr_curves(
        self,
        val_true: Optional[np.ndarray],
        val_prob: Optional[np.ndarray],
        test_true: Optional[np.ndarray],
        test_prob: Optional[np.ndarray],
    ) -> Path:
        """Vẽ đường cong ROC (với AUC) và Precision-Recall Curve (với Average Precision) cho Val và Test."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
        fig.suptitle("ĐƯỜNG CONG ROC-AUC VÀ PRECISION-RECALL (PR-AUC) ĐỐI CHIẾU", fontsize=15, fontweight="bold", y=1.02)

        # 1. ROC Curves
        ax_roc = axes[0]
        if val_true is not None and val_prob is not None:
            fpr_v, tpr_v, _ = roc_curve(val_true, val_prob)
            auc_v = auc(fpr_v, tpr_v)
            ax_roc.plot(fpr_v, tpr_v, label=f"Validation (AUC = {auc_v:.4f})", color="#1f77b4", linewidth=2.5)

        if test_true is not None and test_prob is not None:
            fpr_t, tpr_t, _ = roc_curve(test_true, test_prob)
            auc_t = auc(fpr_t, tpr_t)
            ax_roc.plot(fpr_t, tpr_t, label=f"Test (AUC = {auc_t:.4f})", color="#2ca02c", linewidth=2.5, linestyle="--")

        ax_roc.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.5, label="Random Guess (0.5000)")
        ax_roc.set_title("Đường cong ROC (Receiver Operating Characteristic)", fontweight="bold")
        ax_roc.set_xlabel("Tỷ lệ Báo động Giả (False Positive Rate - FPR)")
        ax_roc.set_ylabel("Tỷ lệ Phát hiện Đúng (True Positive Rate / Recall - TPR)")
        ax_roc.set_xlim(-0.02, 1.02)
        ax_roc.set_ylim(-0.02, 1.02)
        ax_roc.legend(loc="lower right", framealpha=0.9)

        # 2. PR Curves
        ax_pr = axes[1]
        if val_true is not None and val_prob is not None:
            p_v, r_v, _ = precision_recall_curve(val_true, val_prob)
            ap_v = average_precision_score(val_true, val_prob)
            ax_pr.plot(r_v, p_v, label=f"Validation (AP = {ap_v:.4f})", color="#1f77b4", linewidth=2.5)

        if test_true is not None and test_prob is not None:
            p_t, r_t, _ = precision_recall_curve(test_true, test_prob)
            ap_t = average_precision_score(test_true, test_prob)
            ax_pr.plot(r_t, p_t, label=f"Test (AP = {ap_t:.4f})", color="#2ca02c", linewidth=2.5, linestyle="--")

        ax_pr.set_title("Đường cong Precision - Recall (PR Curve)", fontweight="bold")
        ax_pr.set_xlabel("Recall (Độ nhạy - Drowsy)")
        ax_pr.set_ylabel("Precision (Độ chuẩn xác)")
        ax_pr.set_xlim(-0.02, 1.02)
        ax_pr.set_ylim(-0.02, 1.02)
        ax_pr.legend(loc="lower left", framealpha=0.9)

        plt.tight_layout()
        out_path = self.output_dir / "roc_pr_curves_val_test.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Đã lưu biểu đồ ROC và PR curves: {out_path.name}")
        return out_path

    def plot_metrics_comparison_bar(
        self,
        val_default_m: Dict[str, float],
        test_default_m: Dict[str, float],
        val_opt_m: Dict[str, float],
        test_opt_m: Dict[str, float],
        opt_threshold: float,
    ) -> Path:
        """Vẽ biểu đồ cột nhóm so sánh các chỉ số giữa Val và Test ở ngưỡng mặc định vs tối ưu."""
        metric_keys = ["accuracy", "recall", "precision", "specificity", "f1", "f2"]
        metric_names = ["Accuracy", "Recall", "Precision", "Specificity", "F1-Score", "F2-Score"]

        df_list = []
        for k, name in zip(metric_keys, metric_names):
            df_list.append({"Metric": name, "Value": val_default_m.get(k, 0.0) * 100, "Group": "Val @ 0.50"})
            df_list.append({"Metric": name, "Value": val_opt_m.get(k, 0.0) * 100, "Group": f"Val @ {opt_threshold:.2f}"})
            df_list.append({"Metric": name, "Value": test_default_m.get(k, 0.0) * 100, "Group": "Test @ 0.50"})
            df_list.append({"Metric": name, "Value": test_opt_m.get(k, 0.0) * 100, "Group": f"Test @ {opt_threshold:.2f}"})

        df = pd.DataFrame(df_list)

        fig, ax = plt.subplots(figsize=(14, 6), dpi=self.dpi)
        palette = ["#9ecae1", "#3182bd", "#a1d99b", "#31a354"]
        sns.barplot(data=df, x="Metric", y="Value", hue="Group", palette=palette, ax=ax)

        ax.set_title("SO SÁNH CÁC CHỈ SỐ PHÂN LOẠI (NGƯỠNG MẶC ĐỊNH 0.5 VS NGƯỠNG TỐI ƯU)", fontweight="bold")
        ax.set_ylabel("Tỷ lệ phần trăm (%)")
        ax.set_xlabel("Chỉ số đo lường")
        ax.set_ylim(0, 115)

        for p in ax.patches:
            val = p.get_height()
            if val > 0:
                ax.annotate(
                    f"{val:.1f}%",
                    (p.get_x() + p.get_width() / 2.0, val + 1.5),
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    rotation=0,
                )

        ax.legend(title="Thiết lập đánh giá", framealpha=0.9, loc="upper right")
        plt.tight_layout()
        out_path = self.output_dir / "metrics_comparison_val_test.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Đã lưu biểu đồ so sánh chỉ số: {out_path.name}")
        return out_path


# =====================================================================
# 6. TRÌNH ĐIỀU PHỐI ĐO LƯỜNG TOÀN DIỆN (BENCHMARK RUNNER)
# =====================================================================
class BenchmarkRunner:
    """
    Trình điều phối toàn bộ vòng đời thực thi Benchmark:
    1. Quản lý cấu hình BenchmarkConfig.
    2. Nạp an toàn mô hình và xây dựng DataLoaders.
    3. Thực thi suy luận thu thập xác suất dự đoán và đo lường độ trễ (latency).
    4. Kích hoạt ThresholdAnalyzer tìm ngưỡng tối ưu trên tập Validation.
    5. Kiểm chứng ngưỡng tối ưu trên tập Test.
    6. Xuất 4 bộ biểu đồ PNG, JSON tóm tắt, CSV dự đoán và hiển thị bảng tổng kết trên console.
    """
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        if config is None:
            config = BenchmarkConfig()
        self.cfg = config
        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        configure_logging(self.output_dir / "benchmark.log")

    def _run_inference_split(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        desc: str = "Inference",
    ) -> Dict[str, Any]:
        """Thực thi suy luận trích xuất xác suất softmax, nhãn thực tế, metadata và đo đạc độ trễ."""
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_samples = 0
        all_targets = []
        all_probs = []
        all_metas = []

        batch_times = []
        start_eval_time = time.time()

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True, leave=False):
                landmarks, labels, lengths, metas = batch[0], batch[1], batch[2], batch[3]
                landmarks = landmarks.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                count = labels.size(0)

                t0 = time.perf_counter()
                with torch.autocast(device_type=device.type, enabled=self.cfg.amp and device.type == "cuda"):
                    logits = model(landmarks, lengths=lengths)
                    loss = criterion(logits, labels)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                batch_times.append(t1 - t0)
                probs = torch.softmax(logits, dim=1)

                total_loss += loss.item() * count
                total_samples += count

                all_targets.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_metas.extend(metas)

        total_time = time.time() - start_eval_time
        y_true = np.concatenate(all_targets, axis=0) if all_targets else np.array([], dtype=int)
        y_probs = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 2), dtype=float)
        avg_loss = float(total_loss / max(1, total_samples))

        mean_batch_latency_ms = float(np.mean(batch_times) * 1000) if batch_times else 0.0
        latency_ms_per_clip = float((total_time / max(1, total_samples)) * 1000)
        throughput_clips_sec = float(total_samples / max(1e-6, total_time))

        return {
            "y_true": y_true,
            "y_probs": y_probs,
            "metas": all_metas,
            "avg_loss": avg_loss,
            "total_samples": total_samples,
            "total_time_sec": total_time,
            "mean_batch_latency_ms": mean_batch_latency_ms,
            "latency_ms_per_clip": latency_ms_per_clip,
            "throughput_clips_sec": throughput_clips_sec,
        }

    def _save_predictions_csv(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        metas: List[Dict[str, Any]],
        split_name: str,
        default_th: float,
        opt_th: float,
    ) -> Path:
        """Lưu toàn bộ chi tiết dự đoán từng clip ra tệp CSV."""
        out_csv = self.output_dir / f"predictions_{split_name}.csv"
        rows = []
        for i in range(len(y_true)):
            true_lbl = int(y_true[i])
            prob_alert = float(y_probs[i, 0])
            prob_drowsy = float(y_probs[i, 1])

            pred_default = 1 if prob_drowsy >= default_th else 0
            pred_opt = 1 if prob_drowsy >= opt_th else 0

            meta = metas[i] if i < len(metas) else {}
            rows.append({
                "split": split_name,
                "video_id": meta.get("video_id", f"sample_{i}"),
                "path": meta.get("path", ""),
                "true_label": true_lbl,
                "true_class": self.cfg.class_names.get(true_lbl, str(true_lbl)),
                "prob_alert": prob_alert,
                "prob_drowsy": prob_drowsy,
                "pred_default_0.5": pred_default,
                "pred_default_class": self.cfg.class_names.get(pred_default, str(pred_default)),
                "is_correct_default": bool(true_lbl == pred_default),
                "pred_opt": pred_opt,
                "pred_opt_class": self.cfg.class_names.get(pred_opt, str(pred_opt)),
                "is_correct_opt": bool(true_lbl == pred_opt),
                "valid_frames": meta.get("num_sampled_frames", meta.get("valid_frames", 0)),
                "duration_sec": meta.get("duration_sec", 0.0),
            })

        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False, encoding="utf-8")
        logger.info(f"Đã lưu danh sách dự đoán: {out_csv.name} ({len(df):,} dòng)")
        return out_csv

    def run(self) -> Dict[str, Any]:
        """Thực thi toàn bộ quy trình đo lường benchmark."""
        logger.info("=" * 80)
        logger.info("BẮT ĐẦU CHẠY BENCHMARK & TỐI ƯU HÓA NGƯỠNG CHO MÔ HÌNH DRIVER GUARDIAN GCN")
        logger.info("=" * 80)
        logger.info(f"Checkpoint        : {self.cfg.checkpoint_path}")
        logger.info(f"Thư mục lưu kết quả: {self.output_dir}")
        logger.info(f"Thiết bị sử dụng  : {self.cfg.device} (AMP: {self.cfg.amp})")
        logger.info(f"Tập đánh giá      : {self.cfg.splits}")
        logger.info(f"Sử dụng Cache     : {self.cfg.use_cache} (Thư mục: {self.cfg.cache_dir})")
        logger.info(f"Tiêu chí tối ưu   : {self.cfg.target_metric.upper()}")

        # 1. Nạp mô hình từ checkpoint
        device = torch.device(self.cfg.device)
        model, train_cfg, ckpt_dict = load_checkpoint_safely(self.cfg.checkpoint_path, device)
        analyzer = ThresholdAnalyzer(step=self.cfg.threshold_step)
        plotter = BenchmarkPlotter(self.output_dir, dpi=self.cfg.dpi)

        eval_data = {}
        # 2. Chạy suy luận qua các splits chỉ định (Val / Test)
        for split in self.cfg.splits:
            logger.info(f"\n--- Đang xử lý tập: {split.upper()} ---")
            loader = build_benchmark_dataloader(self.cfg, split=split)
            logger.info(f"[{split.upper()}] Tổng số mẫu dataset: {len(loader.dataset):,} mẫu.")
            res = self._run_inference_split(model, loader, device, desc=f"Eval [{split.upper()}]")
            eval_data[split] = res

        # 3. Phân tích quét ngưỡng trên tập Validation (nếu có)
        val_sweep = None
        val_opt = None
        target_opt_th = self.cfg.default_threshold

        if "val" in eval_data:
            val_res = eval_data["val"]
            val_sweep = analyzer.sweep_thresholds(val_res["y_true"], val_res["y_probs"][:, 1])
            val_opt = analyzer.find_optimal_thresholds(val_sweep)

            # Chọn ngưỡng tối ưu theo cấu hình target_metric
            if self.cfg.target_metric.lower() == "f2":
                target_opt_th = val_opt["best_f2"]["threshold"]
            elif self.cfg.target_metric.lower() == "accuracy":
                target_opt_th = val_opt["best_accuracy"]["threshold"]
            elif self.cfg.target_metric.lower() == "youden":
                target_opt_th = val_opt["best_youden"]["threshold"]
            else:
                target_opt_th = val_opt["best_f1"]["threshold"]

            logger.info(f"\n[KẾT QUẢ TỐI ƯU NGƯỠNG TRÊN TẬP VALIDATION]")
            logger.info(f"  - Ngưỡng tối ưu F1       : tau = {val_opt['best_f1']['threshold']:.2f} (F1 = {val_opt['best_f1']['f1']*100:.2f}%, Recall = {val_opt['best_f1']['recall']*100:.2f}%)")
            logger.info(f"  - Ngưỡng tối ưu F2 (Safety): tau = {val_opt['best_f2']['threshold']:.2f} (F2 = {val_opt['best_f2']['f2']*100:.2f}%, Recall = {val_opt['best_f2']['recall']*100:.2f}%)")
            logger.info(f"  - Ngưỡng tối ưu Accuracy : tau = {val_opt['best_accuracy']['threshold']:.2f} (Acc = {val_opt['best_accuracy']['accuracy']*100:.2f}%)")
            logger.info(f"  - Đã chọn ngưỡng mục tiêu: tau = {target_opt_th:.2f} ({self.cfg.target_metric.upper()} Criterion)")

        # 4. Quét ngưỡng trên tập Test (để so sánh đối chiếu)
        test_sweep = None
        test_opt = None
        if "test" in eval_data:
            test_res = eval_data["test"]
            test_sweep = analyzer.sweep_thresholds(test_res["y_true"], test_res["y_probs"][:, 1])
            test_opt = analyzer.find_optimal_thresholds(test_sweep)

        # 5. Tính toán ma trận nhầm lẫn và metrics chi tiết tại ngưỡng mặc định & tối ưu
        summary_report: Dict[str, Any] = {
            "model_info": {
                "model_name": ckpt_dict.get("model_name", train_cfg.model_name),
                "checkpoint_path": str(Path(self.cfg.checkpoint_path).resolve()),
                "epoch": ckpt_dict.get("epoch", None),
                "num_parameters": sum(p.numel() for p in model.parameters()),
            },
            "benchmark_config": self.cfg.to_dict(),
            "optimal_thresholds_found_on_val": val_opt,
            "chosen_optimal_threshold": float(target_opt_th),
        }

        # Lưu predictions CSV và tổng hợp metrics cho từng split
        cm_dict = {}
        metrics_default_dict = {}
        metrics_opt_dict = {}

        for split in self.cfg.splits:
            s_res = eval_data[split]
            y_true = s_res["y_true"]
            prob_drowsy = s_res["y_probs"][:, 1]

            m_default = analyzer.evaluate_at_threshold(y_true, prob_drowsy, self.cfg.default_threshold)
            m_opt = analyzer.evaluate_at_threshold(y_true, prob_drowsy, target_opt_th)

            # ROC-AUC & PR-AUC
            fpr, tpr, _ = roc_curve(y_true, prob_drowsy)
            roc_auc_val = float(auc(fpr, tpr))
            pr_auc_val = float(average_precision_score(y_true, prob_drowsy))

            m_default["roc_auc"] = roc_auc_val
            m_default["pr_auc"] = pr_auc_val
            m_opt["roc_auc"] = roc_auc_val
            m_opt["pr_auc"] = pr_auc_val

            metrics_default_dict[split] = m_default
            metrics_opt_dict[split] = m_opt

            cm_def = np.array([[m_default["tn"], m_default["fp"]], [m_default["fn"], m_default["tp"]]])
            cm_opt = np.array([[m_opt["tn"], m_opt["fp"]], [m_opt["fn"], m_opt["tp"]]])
            cm_dict[f"{split}_default"] = cm_def
            cm_dict[f"{split}_opt"] = cm_opt

            # Xuất predictions CSV
            if self.cfg.save_predictions:
                self._save_predictions_csv(
                    y_true=y_true,
                    y_probs=s_res["y_probs"],
                    metas=s_res["metas"],
                    split_name=split,
                    default_th=self.cfg.default_threshold,
                    opt_th=target_opt_th,
                )

            summary_report[split] = {
                "n_samples": s_res["total_samples"],
                "loss": s_res["avg_loss"],
                "default_threshold_0.5": m_default,
                "optimal_threshold_applied": m_opt,
                "latency_ms_per_clip": s_res["latency_ms_per_clip"],
                "throughput_clips_sec": s_res["throughput_clips_sec"],
            }

        # 6. Vẽ và lưu 4 bộ đồ thị biểu diễn kết quả (300 DPI)
        if self.cfg.save_plots:
            logger.info("\n--- Đang kết xuất các biểu đồ trực quan hóa (300 DPI) ---")
            plotter.plot_threshold_comparison(
                val_sweep=val_sweep,
                test_sweep=test_sweep,
                val_opt=val_opt,
                test_opt=test_opt,
                default_threshold=self.cfg.default_threshold,
            )
            plotter.plot_confusion_matrices(
                cm_val_default=cm_dict.get("val_default"),
                cm_val_opt=cm_dict.get("val_opt"),
                cm_test_default=cm_dict.get("test_default"),
                cm_test_opt=cm_dict.get("test_opt"),
                opt_threshold=target_opt_th,
                default_threshold=self.cfg.default_threshold,
            )
            plotter.plot_roc_pr_curves(
                val_true=eval_data.get("val", {}).get("y_true"),
                val_prob=eval_data.get("val", {}).get("y_probs")[:, 1] if "val" in eval_data else None,
                test_true=eval_data.get("test", {}).get("y_true"),
                test_prob=eval_data.get("test", {}).get("y_probs")[:, 1] if "test" in eval_data else None,
            )
            if "val" in eval_data and "test" in eval_data:
                plotter.plot_metrics_comparison_bar(
                    val_default_m=metrics_default_dict["val"],
                    test_default_m=metrics_default_dict["test"],
                    val_opt_m=metrics_opt_dict["val"],
                    test_opt_m=metrics_opt_dict["test"],
                    opt_threshold=target_opt_th,
                )

        # 7. Lưu báo cáo tóm tắt JSON
        if self.cfg.save_summary_json:
            summary_json_path = self.output_dir / "benchmark_summary.json"
            with open(summary_json_path, "w", encoding="utf-8") as f:
                json.dump(summary_report, f, indent=4, ensure_ascii=False)
            logger.info(f"Đã lưu tóm tắt benchmark JSON: {summary_json_path.name}")

        # 8. Hiển thị bảng tổng kết nghiệm thu đẹp mắt trên Console
        self._print_final_summary_table(summary_report, target_opt_th)

        return summary_report

    def _print_final_summary_table(self, rep: Dict[str, Any], opt_th: float) -> None:
        """In bảng tóm tắt kết quả đo lường đối chiếu theo định dạng chuẩn đẹp mắt."""
        logger.info("\n" + "=" * 90)
        logger.info("                      BÁO CÁO KẾT QUẢ ĐO LƯỜNG VÀ TỐI ƯU HÓA NGƯỠNG")
        logger.info(f"           Mô hình: {rep['model_info']['model_name']} | Epoch: {rep['model_info'].get('epoch', 'N/A')} | Ngưỡng Tối ưu: tau = {opt_th:.2f}")
        logger.info("=" * 90)

        header = f"{'Chỉ số (Metric)':<26} | {'Val @ 0.50':<12} | {f'Val @ {opt_th:.2f}':<12} | {'Test @ 0.50':<12} | {f'Test @ {opt_th:.2f}':<12}"
        logger.info(header)
        logger.info("-" * 90)

        val_d = rep.get("val", {}).get("default_threshold_0.5", {})
        val_o = rep.get("val", {}).get("optimal_threshold_applied", {})
        tst_d = rep.get("test", {}).get("default_threshold_0.5", {})
        tst_o = rep.get("test", {}).get("optimal_threshold_applied", {})

        metrics_rows = [
            ("Accuracy (Độ chính xác)", "accuracy", True),
            ("Recall (Độ nhạy - Drowsy)", "recall", True),
            ("Precision (Độ chuẩn xác)", "precision", True),
            ("Specificity (Tỉnh táo)", "specificity", True),
            ("F1-Score (Cân bằng beta=1)", "f1", True),
            ("F2-Score (Ưu tiên Recall)", "f2", True),
            ("ROC-AUC", "roc_auc", False),
            ("PR-AUC (Avg Precision)", "pr_auc", False),
        ]

        for title, key, is_pct in metrics_rows:
            def fmt(d):
                if not d or key not in d:
                    return "N/A"
                val = d[key]
                return f"{val * 100:.2f}%" if is_pct else f"{val:.4f}"

            row = f"{title:<26} | {fmt(val_d):<12} | {fmt(val_o):<12} | {fmt(tst_d):<12} | {fmt(tst_o):<12}"
            logger.info(row)

        logger.info("-" * 90)
        val_lat = rep.get("val", {}).get("latency_ms_per_clip", 0.0)
        test_lat = rep.get("test", {}).get("latency_ms_per_clip", 0.0)
        val_fps = rep.get("val", {}).get("throughput_clips_sec", 0.0)
        test_fps = rep.get("test", {}).get("throughput_clips_sec", 0.0)
        logger.info(f"{'Độ trễ suy luận (ms/clip)':<26} | {val_lat:<12.2f} | {'-':<12} | {test_lat:<12.2f} | {'-':<12}")
        logger.info(f"{'Thông lượng (Clips/sec)':<26} | {val_fps:<12.1f} | {'-':<12} | {test_fps:<12.1f} | {'-':<12}")
        logger.info("=" * 90)


# =====================================================================
# 7. GIAO DIỆN DÒNG LỆNH (CLI) VÀ KHỞI CHẠY TRỰC TIẾP
# =====================================================================
def parse_arguments() -> argparse.Namespace:
    """Thiết lập bộ phân tích cú pháp tham số dòng lệnh CLI."""
    parser = argparse.ArgumentParser(
        description="Benchmark & Threshold Tuning Suite cho mô hình Driver Guardian GCN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", dest="checkpoint_path", type=str, default="checkpoints/GCN3/checkpoints/best.pt", help="Đường dẫn tới file checkpoint .pt cần đánh giá.")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default=None, help="Thư mục lưu báo cáo kết quả và đồ thị.")
    parser.add_argument("--splits", dest="splits", nargs="+", default=["val", "test"], help="Danh sách các tập cần đánh giá: 'val', 'test' hoặc cả hai.")
    parser.add_argument("--dataset_dir", dest="dataset_dir", type=str, default=r"E:\LSTM\data_processed", help="Thư mục gốc chứa bộ dữ liệu.")
    parser.add_argument("--cache_dir", dest="cache_dir", type=str, default=r"E:\LSTM\cache_landmarks", help="Thư mục chứa cache landmark (.pt).")
    parser.add_argument("--use_cache", dest="use_cache", action="store_true", default=True, help="Bật nạp cache landmark để tăng tốc suy luận.")
    parser.add_argument("--no_cache", dest="use_cache", action="store_false", help="Tắt nạp cache, đọc và trích xuất trực tiếp từ video gốc.")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=32, help="Kích thước batch khi suy luận.")
    parser.add_argument("--device", dest="device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị chạy ('cuda' hoặc 'cpu').")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True, help="Bật Automatic Mixed Precision.")
    parser.add_argument("--target_metric", dest="target_metric", type=str, default="f1", choices=["f1", "f2", "accuracy", "youden"], help="Tiêu chí tìm ngưỡng tối ưu trên tập Validation.")
    parser.add_argument("--default_threshold", dest="default_threshold", type=float, default=0.5, help="Ngưỡng phân loại mặc định.")
    parser.add_argument("--dpi", dest="dpi", type=int, default=300, help="Độ phân giải DPI của các biểu đồ PNG.")
    return parser.parse_args()


def main():
    """Hàm khởi chạy chính của công cụ benchmark."""
    config = BenchmarkConfig()
    runner = BenchmarkRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
