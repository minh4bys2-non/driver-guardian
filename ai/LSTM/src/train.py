#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Tệp: src/train.py
Mục đích:
    Pipeline hoàn chỉnh huấn luyện mô hình Deep GRU (Drowsiness Detection),
    đánh giá kết quả mô hình (Model Evaluation) và chẩn đoán quá trình huấn luyện
    (Training Process Diagnostics) sử dụng tập dữ liệu video thô trích xuất trực tiếp
    bản đồ đặc trưng qua mô hình PyTorch native BackboneNeck (src/dataset2.py)
    và hỗ trợ tăng cường dữ liệu thời gian (src/augment.py).

Luồng dữ liệu:
    - Tập Huấn luyện (Train): RawVideoBackboneNeckDataset từ src/dataset2.py (nạp từ video thô qua OpenCV & BackboneNeck)
    - Tập Kiểm định (Val): RawVideoBackboneNeckDataset từ src/dataset2.py (nạp từ video thô qua OpenCV & BackboneNeck)
    - Tăng cường dữ liệu: DetectionAugmenter từ src/augment.py (Temporal Consistency qua shared seed)
    - Mô hình: DeepGRUClassifier từ src/models.py (CNNAdapter + Deep GRU + TemporalAttentionPooling + FC)
    - Hàm mất mát: DrowsinessLoss từ src/loss.py (CrossEntropyLoss hỗ trợ pos_weight)

Cơ chế cấu hình:
    - 100% CẤU HÌNH TẬP TRUNG QUA configs/config.py (TrainConfig) & configs/config.yaml
    - KHÔNG DÙNG GIAO DIỆN DÒNG LỆNH (No CLI / No Argparse)
    - Thực thi trực tiếp: python train.py hoặc python src/train.py
"""

import os
import sys
import time
import json
import csv
import math
import random
import re
import shutil
import tempfile
import logging
import gc
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union, Sequence

# Đảm bảo console Windows hỗ trợ in tiếng Việt UTF-8 không lỗi charmap
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Tắt triệt để các thông báo rác từ FFmpeg/swscaler ra màn hình console
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

# Tối ưu hóa bộ cấp phát CUDA Caching Allocator: Chống phân mảnh bộ nhớ khi chuỗi video có độ dài biến thiên
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Thêm thư mục gốc dự án vào sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Thiết lập môi trường tối ưu cho OpenMP và ONNX Runtime
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["ORT_LOG_LEVEL"] = "3"

import cv2
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Sử dụng backend Agg cho matplotlib để tránh lỗi hiển thị GUI trên Windows/Headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report,
        roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Import các module nội bộ của dự án
from configs.config import TrainConfig, load_config
from src.dataset2 import (
    RawVideoBackboneNeckDataset,
    collate_raw_video_features,
    DEFAULT_CHECKPOINT_PATH
)
from src.models import DeepGRUClassifier
from src.loss import DrowsinessLoss, build_loss


# ==============================================================================
# 1. HÀM THIẾT LẬP REPRODUCIBILITY & LOGGING
# ==============================================================================
def seed_everything(seed: int = 42) -> None:
    """Cố định hạt giống ngẫu nhiên cho toàn bộ hệ thống để đảm bảo tính tái lập."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    """Hàm worker_init_fn đảm bảo tính tái lập cho PyTorch DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_logger(log_dir: Union[str, Path], experiment_name: str) -> logging.Logger:
    """Khởi tạo logger ghi log đồng thời ra console và tệp tin .log."""
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = log_dir_path / f"{experiment_name}.log"

    logger = logging.getLogger(f"Trainer1_{experiment_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Formatter chuẩn
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Handler ghi ra file (UTF-8)
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # Handler in ra màn hình console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger


def get_vram_info(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Truy vấn thông số chi tiết về tình trạng bộ nhớ VRAM của GPU NVIDIA.

    Args:
        device: Thiết bị PyTorch (nếu None sẽ tự phát hiện cuda:0).

    Returns:
        Dict chứa:
            allocated_gb: VRAM hiện tại đang dùng bởi tensors (GB).
            reserved_gb: VRAM được PyTorch Caching Allocator bảo lưu (GB).
            peak_gb: Đỉnh VRAM cao nhất đã đạt được kể từ lần reset gần nhất (GB).
            total_gb: Tổng dung lượng VRAM vật lý của GPU (GB).
            percent_used: Tỷ lệ phần trăm VRAM bảo lưu trên tổng VRAM (%).
    """
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "peak_gb": 0.0,
            "total_gb": 0.0,
            "percent_used": 0.0
        }
    dev = device if device is not None and device.type == "cuda" else torch.device("cuda:0")
    try:
        dev_idx = dev.index if dev.index is not None else 0
        total_bytes = torch.cuda.get_device_properties(dev_idx).total_memory
        allocated_bytes = torch.cuda.memory_allocated(dev_idx)
        reserved_bytes = torch.cuda.memory_reserved(dev_idx)
        peak_bytes = torch.cuda.max_memory_allocated(dev_idx)

        total_gb = total_bytes / (1024 ** 3)
        allocated_gb = allocated_bytes / (1024 ** 3)
        reserved_gb = reserved_bytes / (1024 ** 3)
        peak_gb = peak_bytes / (1024 ** 3)
        percent_used = (reserved_gb / total_gb * 100.0) if total_gb > 0 else 0.0

        return {
            "allocated_gb": round(allocated_gb, 2),
            "reserved_gb": round(reserved_gb, 2),
            "peak_gb": round(peak_gb, 2),
            "total_gb": round(total_gb, 2),
            "percent_used": round(percent_used, 1)
        }
    except Exception:
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "peak_gb": 0.0,
            "total_gb": 0.0,
            "percent_used": 0.0
        }


def cleanup_cuda_memory(force_gc: bool = True) -> None:
    """
    Thu hồi rác Python và giải phóng bộ nhớ đệm (caching allocator) của CUDA.

    Args:
        force_gc: Cờ bật thu gom rác Python gc.collect() trước khi xả cache GPU.
    """
    if force_gc:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==============================================================================
# 2. BỘ ĐO LƯỜNG VÀ ĐÁNH GIÁ CHỈ SỐ: MetricsTracker
# ==============================================================================
class MetricsTracker:
    """Bộ tích lũy và tính toán toàn diện các chỉ số đánh giá kết quả mô hình."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Làm rỗng bộ đệm trước khi sang epoch mới."""
        self.targets: List[int] = []
        self.preds: List[int] = []
        self.probs_drowsy: List[float] = []
        self.total_loss: float = 0.0
        self.total_samples: int = 0
        self.batch_count: int = 0
        self.correct_samples: int = 0

    @property
    def running_loss(self) -> float:
        """Giá trị loss trung bình lũy tiến."""
        return self.total_loss / max(1, self.total_samples)

    @property
    def running_acc(self) -> float:
        """Độ chính xác trung bình lũy tiến (%)."""
        return (self.correct_samples / max(1, self.total_samples)) * 100.0

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        probs: torch.Tensor,
        loss_val: float,
        batch_size: int
    ) -> None:
        """
        Gom kết quả của một batch.
        Args:
            preds: Tensor [B] nhãn dự đoán (0 hoặc 1)
            targets: Tensor [B] nhãn thực tế Ground Truth (0 hoặc 1)
            probs: Tensor [B, 2] xác suất softmax
            loss_val: Giá trị scalar loss của batch
            batch_size: Số lượng mẫu trong batch
        """
        self.targets.extend(targets.detach().cpu().numpy().tolist())
        self.preds.extend(preds.detach().cpu().numpy().tolist())
        # Lưu xác suất của lớp 1 (Drowsy) phục vụ tính ROC-AUC và PR-AUC
        self.probs_drowsy.extend(probs[:, 1].detach().cpu().numpy().tolist())
        self.total_loss += loss_val * batch_size
        self.total_samples += batch_size
        self.batch_count += 1
        self.correct_samples += int((preds == targets).sum().item())

    def compute(self) -> Dict[str, float]:
        """Tính toán và trả về từ điển các chỉ số phân loại tổng hợp."""
        if self.total_samples == 0:
            return {"loss": 0.0, "accuracy": 0.0, "f1": 0.0}

        avg_loss = self.total_loss / self.total_samples
        y_true = np.array(self.targets)
        y_pred = np.array(self.preds)
        y_prob = np.array(self.probs_drowsy)

        # Tính toán Accuracy & F1
        if HAS_SKLEARN:
            acc = float(accuracy_score(y_true, y_pred))
            f1_drowsy = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
            f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            prec_drowsy = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
            recall_drowsy = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))

            # Tính Specificity (True Negative Rate) cho lớp 0 (Alert)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

            # Tính ROC-AUC và PR-AUC an toàn khi có cả 2 lớp
            try:
                if len(np.unique(y_true)) > 1:
                    auc_roc = float(roc_auc_score(y_true, y_prob))
                    auc_pr = float(average_precision_score(y_true, y_prob))
                else:
                    auc_roc = 0.5
                    auc_pr = 0.0
            except Exception:
                auc_roc = 0.5
                auc_pr = 0.0
        else:
            # Fallback tính thủ công nếu không có scikit-learn
            correct = (y_true == y_pred).sum()
            acc = float(correct / len(y_true))
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            prec_drowsy = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall_drowsy = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            f1_drowsy = (2 * prec_drowsy * recall_drowsy / (prec_drowsy + recall_drowsy)) if (prec_drowsy + recall_drowsy) > 0 else 0.0
            f1_macro = f1_drowsy
            f1_weighted = f1_drowsy
            auc_roc = 0.5
            auc_pr = 0.0

        return {
            "loss": avg_loss,
            "accuracy": acc,
            "f1": f1_drowsy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "precision": prec_drowsy,
            "recall": recall_drowsy,
            "specificity": specificity,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr
        }

    def get_confusion_matrix(self) -> np.ndarray:
        """Trả về ma trận nhầm lẫn kích thước [2, 2]."""
        if HAS_SKLEARN and len(self.targets) > 0:
            return confusion_matrix(self.targets, self.preds, labels=[0, 1])
        return np.zeros((2, 2), dtype=int)

    def get_classification_report(self) -> str:
        """Trả về chuỗi báo cáo phân loại chi tiết theo từng lớp."""
        if HAS_SKLEARN and len(self.targets) > 0:
            return classification_report(
                self.targets, self.preds,
                labels=[0, 1],
                target_names=["0_alert", "1_drowsy"],
                digits=4,
                zero_division=0
            )
        return "Scikit-learn chưa được cài đặt hoặc tập dữ liệu rỗng."


# ==============================================================================
# 3. BỘ TRỰC QUAN HÓA & XUẤT ARTIFACTS: TrainingVisualizer
# ==============================================================================
class TrainingVisualizer:
    """Theo dõi quá trình huấn luyện, phát hiện Overfitting và xuất biểu đồ chẩn đoán."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.history: List[Dict[str, Any]] = []

        # Khởi tạo TensorBoard SummaryWriter
        self.tb_writer = None
        if HAS_TENSORBOARD and config.tb_log_dir:
            tb_dir = Path(config.tb_log_dir) / config.experiment_name
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))

        # Đảm bảo các thư mục đích tồn tại
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
        lr: float,
        grad_norm: float,
        epoch_time_s: float
    ) -> Dict[str, Any]:
        """Ghi nhận thông số của một Epoch vào lịch sử và TensorBoard."""
        entry: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_metrics.get("loss", 0.0),
            "train_acc": train_metrics.get("accuracy", 0.0),
            "train_f1": train_metrics.get("f1", 0.0),
            "lr": lr,
            "grad_norm": grad_norm,
            "epoch_time_s": epoch_time_s,
        }

        if val_metrics is not None:
            entry.update({
                "val_loss": val_metrics.get("loss", 0.0),
                "val_acc": val_metrics.get("accuracy", 0.0),
                "val_f1": val_metrics.get("f1", 0.0),
                "val_precision": val_metrics.get("precision", 0.0),
                "val_recall": val_metrics.get("recall", 0.0),
                "val_specificity": val_metrics.get("specificity", 0.0),
                "val_auc_roc": val_metrics.get("auc_roc", 0.0),
                "val_auc_pr": val_metrics.get("auc_pr", 0.0),
                "gap_loss": val_metrics.get("loss", 0.0) - train_metrics.get("loss", 0.0),
                "gap_f1": train_metrics.get("f1", 0.0) - val_metrics.get("f1", 0.0),
            })
        else:
            entry.update({
                "val_loss": float("nan"),
                "val_acc": float("nan"),
                "val_f1": float("nan"),
                "gap_loss": 0.0,
                "gap_f1": 0.0,
            })

        self.history.append(entry)

        # Ghi sang TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("Loss/Train", entry["train_loss"], epoch)
            self.tb_writer.add_scalar("Accuracy/Train", entry["train_acc"], epoch)
            self.tb_writer.add_scalar("F1_Drowsy/Train", entry["train_f1"], epoch)
            self.tb_writer.add_scalar("LearningRate", lr, epoch)
            self.tb_writer.add_scalar("GradNorm", grad_norm, epoch)

            if val_metrics is not None:
                self.tb_writer.add_scalar("Loss/Val", entry["val_loss"], epoch)
                self.tb_writer.add_scalar("Accuracy/Val", entry["val_acc"], epoch)
                self.tb_writer.add_scalar("F1_Drowsy/Val", entry["val_f1"], epoch)
                self.tb_writer.add_scalar("Precision_Drowsy/Val", entry["val_precision"], epoch)
                self.tb_writer.add_scalar("Recall_Drowsy/Val", entry["val_recall"], epoch)
                self.tb_writer.add_scalar("Specificity_Alert/Val", entry["val_specificity"], epoch)
                self.tb_writer.add_scalar("AUC_ROC/Val", entry["val_auc_roc"], epoch)
                self.tb_writer.add_scalar("AUC_PR/Val", entry["val_auc_pr"], epoch)
                self.tb_writer.add_scalar("Diagnostics/Generalization_Gap_Loss", entry["gap_loss"], epoch)

            self.tb_writer.flush()

        return entry

    def save_history_csv(self, output_path: Optional[str] = None) -> Path:
        """Xuất toàn bộ bảng lịch sử huấn luyện qua các epoch ra file CSV."""
        path = Path(output_path or self.config.history_csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if len(self.history) == 0:
            return path

        fieldnames = list(self.history[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)
        return path

    def sync_history_from_csv(self, resumed_epoch: int, csv_path: Optional[Union[str, Path]] = None) -> int:
        """
        Đồng bộ lại lịch sử huấn luyện từ file CSV cũ cho các epoch <= resumed_epoch.
        Đảm bảo đồ thị loss/accuracy vẽ tiếp tục liền mạch từ Epoch 1.
        """
        path = Path(csv_path or self.config.history_csv_path)
        if not path.exists():
            return 0
        try:
            loaded_entries: List[Dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ep_str = row.get("epoch", "")
                    if not ep_str:
                        continue
                    try:
                        ep = int(float(ep_str))
                    except ValueError:
                        continue
                    if ep <= resumed_epoch:
                        typed_row: Dict[str, Any] = {}
                        for k, v in row.items():
                            if k == "epoch":
                                typed_row[k] = int(float(v))
                            else:
                                try:
                                    typed_row[k] = float(v)
                                except (ValueError, TypeError):
                                    typed_row[k] = v
                        loaded_entries.append(typed_row)
            self.history = loaded_entries
            return len(self.history)
        except Exception:
            return 0

    def plot_learning_curves(self, output_path: Optional[str] = None) -> Path:
        """
        Vẽ đồ thị 2 trục biểu diễn song song Train/Val Loss và Train/Val F1 & Accuracy.
        Tô sáng vùng Generalization Gap để phát hiện Overfitting trực quan.
        """
        path = Path(output_path or self.config.plot_curves_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if len(self.history) == 0:
            return path

        epochs = [h["epoch"] for h in self.history]
        t_loss = [h["train_loss"] for h in self.history]
        v_loss = [h["val_loss"] for h in self.history]
        t_f1 = [h["train_f1"] for h in self.history]
        v_f1 = [h["val_f1"] for h in self.history]
        t_acc = [h["train_acc"] for h in self.history]
        v_acc = [h["val_acc"] for h in self.history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

        # 1. Đồ thị Loss (Train vs Val)
        ax1.plot(epochs, t_loss, label="Train Loss", color="#1f77b4", linewidth=2.0, marker="o", markersize=4)
        ax1.plot(epochs, v_loss, label="Val Loss", color="#d62728", linewidth=2.0, marker="s", markersize=4)
        ax1.fill_between(epochs, t_loss, v_loss, color="#d62728", alpha=0.12, label="Generalization Gap")
        ax1.set_title("Động học Hàm Mất Mát (Loss Curves)", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Loss", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", frameon=True)

        # 2. Đồ thị Hiệu năng Phân loại (F1-Score & Accuracy)
        ax2.plot(epochs, t_f1, label="Train F1 (Drowsy)", color="#2ca02c", linewidth=2.0, marker="^", markersize=4)
        ax2.plot(epochs, v_f1, label="Val F1 (Drowsy)", color="#ff7f0e", linewidth=2.2, marker="*", markersize=6)
        ax2.plot(epochs, t_acc, label="Train Acc", color="#9467bd", linestyle=":", linewidth=1.5)
        ax2.plot(epochs, v_acc, label="Val Acc", color="#8c564b", linestyle="--", linewidth=1.5)
        ax2.set_title("Chỉ số Đánh giá (F1-Score & Accuracy)", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Score (0.0 - 1.0)", fontsize=11)
        ax2.set_ylim(-0.02, 1.02)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="lower right", frameon=True)

        plt.suptitle(f"Chẩn đoán Quá trình Huấn luyện Mô hình — Bài thử nghiệm: {self.config.experiment_name}", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str] = ["0_alert", "1_drowsy"],
        output_path: Optional[str] = None
    ) -> Path:
        """Vẽ biểu đồ Heatmap Ma trận nhầm lẫn (Counts và Normalized %)."""
        path = Path(output_path or self.config.plot_cm_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm.astype("float"), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)

        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        labels = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                labels[i, j] = f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)"

        if HAS_SEABORN:
            sns.heatmap(cm_norm, annot=labels, fmt="", cmap="Blues", cbar=True,
                        xticklabels=class_names, yticklabels=class_names, ax=ax, annot_kws={"size": 12, "fontweight": "bold"})
        else:
            im = ax.imshow(cm_norm, cmap="Blues")
            fig.colorbar(im, ax=ax)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, labels[i, j], ha="center", va="center", color="black", fontweight="bold")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)

        ax.set_title("Ma trận Nhầm lẫn tại Checkpoint Tốt nhất (Confusion Matrix)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Nhãn Dự Đoán (Predicted Label)", fontsize=11, labelpad=8)
        ax.set_ylabel("Nhãn Thực Tế (Ground Truth)", fontsize=11, labelpad=8)
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_roc_pr_curves(
        self,
        y_true: List[int],
        y_probs: List[float],
        output_path: Optional[str] = None
    ) -> Optional[Path]:
        """Vẽ biểu đồ Đường cong ROC và Precision-Recall Curve."""
        if not HAS_SKLEARN or len(np.unique(y_true)) < 2:
            return None

        path = Path(output_path or self.config.plot_roc_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        y_true_arr = np.array(y_true)
        y_prob_arr = np.array(y_probs)

        fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
        roc_auc = roc_auc_score(y_true_arr, y_prob_arr)
        prec, rec, _ = precision_recall_curve(y_true_arr, y_prob_arr)
        pr_auc = average_precision_score(y_true_arr, y_prob_arr)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

        # 1. Đường cong ROC
        ax1.plot(fpr, tpr, color="#1f77b4", linewidth=2.2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        ax1.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.2, label="Random Guess (AUC = 0.50)")
        ax1.set_title("Đường cong ROC (Receiver Operating Characteristic)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("False Positive Rate (FPR)", fontsize=11)
        ax1.set_ylabel("True Positive Rate (Recall)", fontsize=11)
        ax1.set_xlim(-0.02, 1.02)
        ax1.set_ylim(-0.02, 1.02)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="lower right", frameon=True)

        # 2. Đường cong Precision-Recall
        ax2.plot(rec, prec, color="#2ca02c", linewidth=2.2, label=f"PR Curve (AUC = {pr_auc:.4f})")
        base_rate = sum(y_true_arr) / len(y_true_arr)
        ax2.axhline(y=base_rate, color="gray", linestyle="--", linewidth=1.2, label=f"Baseline = {base_rate:.2f}")
        ax2.set_title("Đường cong Precision-Recall (PR Curve)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Recall (Độ nhạy)", fontsize=11)
        ax2.set_ylabel("Precision (Độ chuẩn xác)", fontsize=11)
        ax2.set_xlim(-0.02, 1.02)
        ax2.set_ylim(-0.02, 1.02)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="lower left", frameon=True)

        plt.suptitle("Đánh giá Khả năng Phân tách Ngưỡng Quyết định", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def save_summary_json(self, summary_dict: Dict[str, Any], output_path: Optional[str] = None) -> Path:
        """Lưu bản tóm tắt kết quả huấn luyện ra file JSON."""
        path = Path(output_path or self.config.summary_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=4, ensure_ascii=False)
        return path

    def close(self) -> None:
        """Đóng an toàn phiên TensorBoard."""
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
            self.tb_writer = None


# ==============================================================================
# 4. BỘ QUẢN LÝ CHECKPOINTS & RESUME: CheckpointManager
# ==============================================================================
class CheckpointManager:
    """Quản lý lưu trữ checkpoint, khôi phục trạng thái huấn luyện và Early Stopping."""

    def __init__(self, config: TrainConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = config.monitor_metric.lower()
        self.mode = config.monitor_mode.lower()
        self.min_delta = config.min_delta
        self.patience = config.patience
        self.early_stopping_enabled = config.early_stopping

        self.patience_counter = 0
        self.best_epoch = 0
        self.periodic_ckpts: List[Path] = []

        if self.mode == "min":
            self.best_score = float("inf")
        else:
            self.best_score = float("-inf")

    def _extract_metric_score(self, metrics: Dict[str, float]) -> Optional[float]:
        """Trích xuất giá trị chỉ số giám sát từ từ điển metrics."""
        for k, v in metrics.items():
            if k.lower() == self.monitor or f"val_{k.lower()}" == self.monitor:
                return float(v)
        return None

    def _find_available_epoch_checkpoints(self) -> Dict[int, Path]:
        """Quét và lập danh mục tất cả các file checkpoint epoch sẵn có trong thư mục."""
        found: Dict[int, Path] = {}
        for p in self.checkpoint_dir.glob("*.pt"):
            m = re.search(r"epoch_(\d+)", p.name, re.IGNORECASE)
            if m:
                found[int(m.group(1))] = p
        return dict(sorted(found.items()))

    def resolve_checkpoint_path(
        self,
        resume_epoch: Optional[int] = None,
        checkpoint_target: Optional[Union[str, Path, int]] = None
    ) -> Path:
        """
        Phân giải đường dẫn tệp checkpoint từ tham số resume_epoch hoặc chuỗi cấu hình.
        Hỗ trợ số epoch cụ thể (vd: 10), bí danh ('last', 'best') hoặc tên file trực tiếp.
        """
        target_str = ""
        target_epoch: Optional[int] = resume_epoch

        if checkpoint_target is not None:
            if isinstance(checkpoint_target, int):
                target_epoch = checkpoint_target
            elif isinstance(checkpoint_target, Path):
                target_str = str(checkpoint_target)
            else:
                target_str = str(checkpoint_target).strip()
                if target_str.isdigit():
                    target_epoch = int(target_str)
                elif target_str.lower().startswith("epoch_"):
                    suffix = target_str.lower().replace("epoch_", "").replace(".pt", "")
                    if suffix.isdigit():
                        target_epoch = int(suffix)

        # 1. Trường hợp tìm theo target_epoch cụ thể
        if target_epoch is not None:
            candidate_names = [
                f"{self.config.experiment_name}_epoch_{target_epoch:03d}.pt",
                f"{self.config.experiment_name}_epoch_{target_epoch}.pt",
                f"epoch_{target_epoch:03d}.pt",
                f"epoch_{target_epoch}.pt",
            ]
            for cand in candidate_names:
                cand_path = self.checkpoint_dir / cand
                if cand_path.exists():
                    return cand_path

            # Tìm kiếm mở rộng (Glob pattern)
            glob_matches = list(self.checkpoint_dir.glob(f"*epoch_{target_epoch:03d}*.pt"))
            if not glob_matches:
                glob_matches = list(self.checkpoint_dir.glob(f"*epoch_{target_epoch}*.pt"))
            if glob_matches:
                return sorted(glob_matches)[-1]

            # Không tìm thấy: Quét tất cả các epoch hiện có để báo lỗi chi tiết
            available_epochs: List[int] = []
            for p in self.checkpoint_dir.glob("*.pt"):
                m = re.search(r"epoch_(\d+)", p.name, re.IGNORECASE)
                if m:
                    available_epochs.append(int(m.group(1)))
            available_epochs = sorted(list(set(available_epochs)))

            avail_str = f"Epoch {available_epochs}" if available_epochs else "[Thư mục chưa có checkpoint nào]"
            raise FileNotFoundError(
                f"[CheckpointManager] Không tìm thấy file checkpoint cho Epoch {target_epoch} "
                f"trong thư mục: '{self.checkpoint_dir.resolve()}'.\n"
                f"-> Các checkpoint epoch sẵn có: {avail_str}\n"
                f"-> Gợi ý: Hãy thiết lập 'resume_epoch' là một trong các epoch trên, hoặc đặt 'enable_resume: false' để huấn luyện từ đầu."
            )

        # 2. Trường hợp target_str là bí danh 'best'
        if target_str.lower() == "best":
            best_path = self.checkpoint_dir / f"{self.config.experiment_name}_best.pt"
            if best_path.exists():
                return best_path
            best_matches = list(self.checkpoint_dir.glob("*best.pt"))
            if best_matches:
                return best_matches[0]
            raise FileNotFoundError(
                f"[CheckpointManager] Không tìm thấy checkpoint tốt nhất ('best') trong: '{self.checkpoint_dir.resolve()}'."
            )

        # 3. Trường hợp target_str là bí danh 'last' hoặc rỗng
        if target_str.lower() in ["last", ""]:
            last_path = self.checkpoint_dir / f"{self.config.experiment_name}_last.pt"
            if last_path.exists():
                return last_path
            # Tìm epoch lớn nhất hiện có
            epoch_ckpts: List[Tuple[int, Path]] = []
            for p in self.checkpoint_dir.glob("*.pt"):
                m = re.search(r"epoch_(\d+)", p.name, re.IGNORECASE)
                if m:
                    epoch_ckpts.append((int(m.group(1)), p))
            if epoch_ckpts:
                epoch_ckpts.sort(key=lambda x: x[0])
                latest_epoch, latest_path = epoch_ckpts[-1]
                self.logger.info(f"[*] Tự động tìm thấy checkpoint epoch lớn nhất hiện có: Epoch {latest_epoch} ({latest_path.name})")
                return latest_path
            raise FileNotFoundError(
                f"[CheckpointManager] Không tìm thấy checkpoint gần nhất ('last') trong: '{self.checkpoint_dir.resolve()}'."
            )

        # 4. Trường hợp đường dẫn tệp cụ thể
        direct_path = Path(target_str)
        if direct_path.exists():
            return direct_path
        relative_path = self.checkpoint_dir / target_str
        if relative_path.exists():
            return relative_path

        raise FileNotFoundError(
            f"[CheckpointManager] Không tìm thấy tệp checkpoint: '{target_str}' "
            f"(đã kiểm tra cả đường dẫn trực tiếp và trong '{self.checkpoint_dir.resolve()}')."
        )

    def step(
        self,
        current_metrics: Optional[Dict[str, float]] = None,
        epoch: int = 1,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        train_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Cập nhật trạng thái và lưu checkpoint tại cuối mỗi epoch.
        Lưu ý: Luôn lưu checkpoint cho epoch (dù có chạy validation hay không).
        Returns:
            should_stop (bool): True nếu kích hoạt Early Stopping, ngược lại False.
        """
        eval_metrics = val_metrics if val_metrics is not None else current_metrics
        score = self._extract_metric_score(eval_metrics) if eval_metrics else None

        # 1. Đóng gói checkpoint data
        checkpoint_data: Dict[str, Any] = {
            "epoch": epoch,
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "patience_counter": self.patience_counter,
            "monitor_metric": self.monitor,
            "monitor_mode": self.mode,
            "train_metrics": train_metrics,
            "val_metrics": eval_metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.config.to_dict()
        }

        if model is not None:
            checkpoint_data["model_state_dict"] = model.state_dict()
        if optimizer is not None:
            checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None:
            checkpoint_data["scaler_state_dict"] = scaler.state_dict()

        # 2. Luôn lưu checkpoint mới nhất (last.pt)
        last_ckpt_path = self.checkpoint_dir / f"{self.config.experiment_name}_last.pt"
        torch.save(checkpoint_data, str(last_ckpt_path))

        # 3. Đánh giá xem có phải Checkpoint Tốt nhất (best.pt) hay không
        is_best = False
        if score is not None:
            if self.mode == "min":
                improved = score < (self.best_score - self.min_delta)
            else:
                improved = score > (self.best_score + self.min_delta)

            if improved:
                self.best_score = score
                self.best_epoch = epoch
                self.patience_counter = 0
                is_best = True

                # Cập nhật best_epoch trong payload trước khi lưu best.pt
                checkpoint_data["best_epoch"] = self.best_epoch
                checkpoint_data["best_score"] = self.best_score
                best_ckpt_path = self.checkpoint_dir / f"{self.config.experiment_name}_best.pt"
                torch.save(checkpoint_data, str(best_ckpt_path))
                self.logger.info(
                    f"    [*] [KỶ LỤC MỚI] Chỉ số '{self.monitor}' đạt mức tối ưu: {score:.4f} "
                    f"-> Đã lưu Checkpoint tốt nhất: {best_ckpt_path.name}"
                )
            else:
                self.patience_counter += 1

        # 4. Lưu checkpoint theo Epoch:
        save_all = getattr(self.config, "save_all_epochs", True)
        if save_all:
            # Lưu TẤT CẢ các epoch riêng biệt
            epoch_ckpt_path = self.checkpoint_dir / f"{self.config.experiment_name}_epoch_{epoch:03d}.pt"
            torch.save(checkpoint_data, str(epoch_ckpt_path))
            self.logger.info(f"    [+] Đã lưu checkpoint Epoch {epoch:02d}: {epoch_ckpt_path.name}")
        elif not self.config.save_best_only and (epoch % self.config.save_ckpt_interval_epochs == 0):
            # Lưu theo chu kỳ save_ckpt_interval_epochs và giữ ckpt_keep_last
            interval_ckpt = self.checkpoint_dir / f"{self.config.experiment_name}_epoch_{epoch:03d}.pt"
            torch.save(checkpoint_data, str(interval_ckpt))
            self.periodic_ckpts.append(interval_ckpt)
            if self.config.ckpt_keep_last and len(self.periodic_ckpts) > self.config.ckpt_keep_last:
                old_ckpt = self.periodic_ckpts.pop(0)
                if old_ckpt.exists():
                    old_ckpt.unlink()

        # 5. Kiểm tra điều kiện Dừng sớm (Early Stopping)
        if score is not None and self.early_stopping_enabled and (self.patience_counter >= self.patience):
            self.logger.warning(
                f"\n[!] EARLY STOPPING ĐƯỢC KÍCH HOẠT: Chỉ số '{self.monitor}' không cải thiện "
                f"sau {self.patience} epochs liên tiếp. Dừng sớm tại Epoch {epoch}!"
            )
            return True

        return False

    def load_checkpoint(
        self,
        checkpoint_path: Optional[Union[str, Path, int]] = None,
        resume_epoch: Optional[int] = None,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None
    ) -> int:
        """
        Nạp checkpoint phục vụ tiếp tục huấn luyện (Resume).
        Tự động phân giải đường dẫn qua resolve_checkpoint_path.
        Returns:
            start_epoch (int): Epoch bắt đầu tiếp theo (epoch_nạp + 1).
        """
        resolved_path = self.resolve_checkpoint_path(
            resume_epoch=resume_epoch,
            checkpoint_target=checkpoint_path
        )

        self.logger.info(f"[*] Đang nạp checkpoint từ: {resolved_path.resolve()}")
        checkpoint = torch.load(str(resolved_path), map_location="cpu")

        if model is not None and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            self.logger.info("    [✓] Đã khôi phục trọng số mô hình.")

        if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.logger.info("    [✓] Đã khôi phục trạng thái Optimizer.")
            except Exception as e:
                self.logger.warning(f"    [!] Không thể nạp optimizer_state_dict ({e}), tiếp tục với optimizer mới.")

        if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                self.logger.info("    [✓] Đã khôi phục trạng thái LR Scheduler.")
            except Exception as e:
                self.logger.warning(f"    [!] Không thể nạp scheduler_state_dict ({e}), tiếp tục với scheduler mới.")

        if scaler is not None and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
            try:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
                self.logger.info("    [✓] Đã khôi phục trạng thái GradScaler.")
            except Exception as e:
                self.logger.warning(f"    [!] Không thể nạp scaler_state_dict ({e}).")

        resumed_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = resumed_epoch + 1
        self.best_score = float(checkpoint.get("best_score", self.best_score))
        self.best_epoch = int(checkpoint.get("best_epoch", resumed_epoch))
        self.patience_counter = int(checkpoint.get("patience_counter", 0))

        self.logger.info(
            f"[✓] Đã khôi phục trạng thái thành công từ Epoch {resumed_epoch}! "
            f"Sẽ bắt đầu huấn luyện từ Epoch {start_epoch} "
            f"(Kỷ lục '{self.monitor}': {self.best_score:.4f} tại Epoch {self.best_epoch}, Patience: {self.patience_counter})."
        )
        return start_epoch


# ==============================================================================
# 5. ĐỘNG CƠ HUẤN LUYỆN LÕI: DrowsinessTrainer1 (Raw Video PyTorch BackboneNeck)
# ==============================================================================
class DrowsinessTrainer1:
    """Động cơ điều phối toàn diện pipeline huấn luyện, kiểm định và chẩn đoán với video thô."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config

        # 1. Cố định tính ngẫu nhiên (Reproducibility)
        seed_everything(config.seed)

        # 2. Khởi tạo Logger và Visualizer
        self.logger = setup_logger(config.log_dir, config.experiment_name)
        self.logger.info("=" * 80)
        self.logger.info(f"   KHỞI TẠO PIPELINE HUẤN LUYỆN VIDEO THÔ: {config.experiment_name.upper()}")
        self.logger.info("=" * 80)

        # 3. Xác định Thiết bị Tính toán (CUDA vs CPU)
        if config.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.logger.info(f"[+] Thiết bị tính toán: GPU NVIDIA ({torch.cuda.get_device_name(0)})")
            v_info = get_vram_info(self.device)
            self.logger.info(
                f"    -> Tổng VRAM: {v_info['total_gb']:.1f} GB | Đang cấp phát: {v_info['allocated_gb']:.2f} GB | "
                f"Gradient Accumulation: {config.gradient_accumulation_steps}"
            )
        else:
            self.device = torch.device("cpu")
            self.logger.info("[+] Thiết bị tính toán: CPU")

        self.amp = config.amp and (self.device.type == "cuda")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        # 4. Khởi tạo DataLoaders (Train & Val từ video thô qua src.dataset2)
        self.train_loader, self.val_loader = self._build_dataloaders()

        # 5. Khởi tạo Kiến trúc Mô hình (DeepGRUClassifier)
        self.logger.info(
            f"[*] Khởi tạo mô hình DeepGRUClassifier: input_dim={config.input_dim}, "
            f"hidden_dim={config.hidden_dim}, num_layers={config.num_layers}, fusion={config.spatial_fusion}"
        )
        self.model = DeepGRUClassifier.from_config(config).to(self.device)

        # 6. Khởi tạo Hàm Mất Mát (DrowsinessLoss)
        self.criterion = build_loss(config).to(self.device)
        self.logger.info(f"[+] Hàm mất mát: {self.criterion.__class__.__name__} (loss_type='{config.loss_type}')")

        # 7. Khởi tạo Optimizer & LR Scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # 8. Khởi tạo CheckpointManager & Visualizer
        self.checkpoint_manager = CheckpointManager(config, self.logger)
        self.visualizer = TrainingVisualizer(config)
        self.metrics_tracker = MetricsTracker()

        # Biến theo dõi trạng thái và xử lý Resume
        self.start_epoch = 1
        enable_resume = getattr(self.config, "enable_resume", False)
        resume_epoch = getattr(self.config, "resume_epoch", None)
        resume_target = getattr(self.config, "resume", "")

        if enable_resume:
            self.logger.info("[*] Cờ 'enable_resume=True': Kích hoạt nạp lại trạng thái huấn luyện...")
            self.start_epoch = self.checkpoint_manager.load_checkpoint(
                checkpoint_path=resume_target,
                resume_epoch=resume_epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler
            )
            # Đồng bộ lịch sử visualizer từ file CSV cũ
            synced_count = self.visualizer.sync_history_from_csv(resumed_epoch=self.start_epoch - 1)
            if synced_count > 0:
                self.logger.info(f"    [+] Đã đồng bộ thành công {synced_count} mốc lịch sử từ {self.config.history_csv_path}.")
        else:
            if resume_epoch is not None or resume_target:
                self.logger.info(
                    f"[*] [Thông báo] Cờ 'enable_resume=False' (mặc định an toàn): "
                    f"Bỏ qua checkpoint đã cấu hình (epoch={resume_epoch}, target='{resume_target}'). "
                    f"Huấn luyện bắt đầu mới từ Epoch 1."
                )
            else:
                self.logger.info("[*] Chế độ huấn luyện mới từ đầu (Epoch 1).")

    def _build_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Khởi tạo Train DataLoader và Val DataLoader từ video thô qua src.dataset2."""
        raw_train_dir = Path(getattr(self.config, "dataset_dir", "dataset"))
        raw_val_dir_str = getattr(self.config, "val_dataset_dir", None)
        raw_val_dir = Path(raw_val_dir_str) if raw_val_dir_str else raw_train_dir

        manifest_file = (
            getattr(self.config, "manifest_file", None)
            or getattr(self.config, "train_manifest_csv", None)
        )
        val_manifest = (
            getattr(self.config, "val_manifest", None)
            or getattr(self.config, "val_manifest_csv", None)
            or manifest_file
        )

        ckpt_path = (
            getattr(self.config, "backbone_neck_checkpoint", None)
            or getattr(self.config, "checkpoint_path", None)
            or DEFAULT_CHECKPOINT_PATH
        )

        sample_interval = float(getattr(self.config, "sample_interval", 0.1))
        seq_len = getattr(self.config, "seq_len", None)
        chunk_size = int(getattr(self.config, "chunk_size", 16))
        min_frames = int(getattr(self.config, "min_frames", 1))
        use_aug = bool(getattr(self.config, "use_augmentation", True))
        effective_augmenter = None
        if use_aug:
            try:
                from src.augment import get_video_augmenter
                effective_augmenter = get_video_augmenter()
                self.logger.info("[+] Đã kích hoạt bộ tăng cường dữ liệu: DetectionAugmenter (src/augment.py)")
            except Exception as e:
                self.logger.warning(f"[!] Không thể nạp DetectionAugmenter: {e}. Tiếp tục không augment.")
                effective_augmenter = None
        else:
            self.logger.info("[-] Tăng cường dữ liệu: ĐÃ TẮT (use_augmentation=False)")

        self.logger.info(f"[*] Khởi tạo tập huấn luyện video thô từ: {raw_train_dir.resolve()}")
        train_dataset = RawVideoBackboneNeckDataset(
            dataset_dir=raw_train_dir,
            manifest_file=manifest_file,
            split="train",
            sample_interval=sample_interval,
            seq_len=seq_len,
            checkpoint_path=ckpt_path,
            chunk_size=chunk_size,
            device="auto",
            augmenter=effective_augmenter,
            window_sampling="random",
            min_frames=min_frames
        )

        self.logger.info(f"[*] Khởi tạo tập kiểm định video thô từ: {raw_val_dir.resolve()}")
        val_dataset = RawVideoBackboneNeckDataset(
            dataset_dir=raw_val_dir,
            manifest_file=val_manifest,
            split="val",
            sample_interval=sample_interval,
            seq_len=seq_len,
            checkpoint_path=ckpt_path,
            chunk_size=chunk_size,
            device="auto",
            window_sampling="center",
            min_frames=min_frames
        )

        # Nếu tập val rỗng do cấu trúc thư mục phẳng, chia tự động bằng random_split
        if len(val_dataset) == 0 and len(train_dataset) > 1:
            self.logger.info(
                f"[*] Không phát hiện thư mục val tách biệt. Tự động chia dataset theo tỷ lệ train_ratio={train_ratio:.2f}."
            )
            all_dataset = RawVideoBackboneNeckDataset(
                dataset_dir=raw_train_dir,
                manifest_file=manifest_file,
                split="all",
                sample_interval=sample_interval,
                seq_len=seq_len,
                checkpoint_path=ckpt_path,
                chunk_size=chunk_size,
                device="auto",
                min_frames=min_frames
            )
            total_len = len(all_dataset)
            tr_len = max(1, int(round(total_len * train_ratio)))
            va_len = max(1, total_len - tr_len)
            generator = torch.Generator().manual_seed(self.config.seed)
            from torch.utils.data import random_split
            train_dataset, val_dataset = random_split(all_dataset, [tr_len, va_len], generator=generator)

        self.logger.info(f"    -> Đã nạp thành công {len(train_dataset)} mẫu huấn luyện.")
        self.logger.info(f"    -> Đã nạp thành công {len(val_dataset)} mẫu kiểm định.")

        g = torch.Generator().manual_seed(self.config.seed)
        val_workers = getattr(self.config, "val_num_workers", self.config.num_workers)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            collate_fn=collate_raw_video_features,
            pin_memory=self.config.pin_memory if self.device.type == "cuda" else False,
            persistent_workers=self.config.persistent_workers if self.config.num_workers > 0 else False,
            worker_init_fn=_seed_worker,
            generator=g,
            drop_last=self.config.drop_last
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=getattr(self.config, "val_batch_size", self.config.batch_size),
            shuffle=False,
            num_workers=val_workers,
            collate_fn=collate_raw_video_features,
            pin_memory=self.config.pin_memory if self.device.type == "cuda" else False,
            persistent_workers=self.config.persistent_workers if val_workers > 0 else False,
            worker_init_fn=_seed_worker,
            drop_last=False
        )

        return train_loader, val_loader

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Khởi tạo bộ tối ưu hóa theo cấu hình."""
        opt_name = self.config.optimizer.lower()
        lr = self.config.lr0
        wd = self.config.weight_decay

        if opt_name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd, betas=self.config.betas)
        elif opt_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd, betas=self.config.betas)
        elif opt_name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd, momentum=self.config.momentum)
        else:
            raise ValueError(f"Không hỗ trợ bộ tối ưu hóa: '{opt_name}'")

    def _build_scheduler(self) -> Optional[Any]:
        """Khởi tạo bộ điều chỉnh tốc độ học (LR Scheduler)."""
        if not self.config.use_scheduler:
            return None

        sched_type = self.config.scheduler_type.lower()
        if sched_type == "cosine":
            t_max = self.config.epochs
            eta_min = self.config.lr0 * self.config.lr_min_factor
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)
        elif sched_type == "plateau":
            mode = "min" if self.config.monitor_mode == "min" else "max"
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode=mode, factor=0.5, patience=self.config.patience // 2
            )
        return None

    def _autocast_context(self):
        """Tạo context manager autocast tương thích đa phiên bản PyTorch."""
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", enabled=self.amp)
        return torch.cuda.amp.autocast(enabled=self.amp)

    def _handle_cuda_oom(self, epoch: int, batch_idx: int, total_batches: int, phase: str = "Train") -> None:
        """
        Xử lý sự cố quá tải bộ nhớ VRAM (CUDA Out Of Memory):
        - Xả sạch bộ đệm gradient: zero_grad(set_to_none=True).
        - Thu gom rác Python gc.collect() và dọn dẹp cache GPU torch.cuda.empty_cache().
        - Ghi log cảnh báo mức ERROR kèm chi tiết tài nguyên VRAM.
        - Bỏ qua batch lỗi để duy trì tiến trình huấn luyện mà không bị sập (crash).
        """
        v_info = get_vram_info(self.device)
        self.logger.error("!" * 80)
        self.logger.error(
            f"[!] PHÁT HIỆN SỰ CỐ CUDA OUT OF MEMORY (OOM) TRONG PHA {phase.upper()}!\n"
            f"    - Mốc thời gian: Epoch {epoch:02d}/{self.config.epochs:02d} | Batch {batch_idx:03d}/{total_batches:03d}\n"
            f"    - Trạng thái VRAM: Đã cấp phát {v_info['allocated_gb']:.2f}GB / {v_info['total_gb']:.1f}GB "
            f"(Bảo lưu: {v_info['reserved_gb']:.2f}GB, Đỉnh: {v_info['peak_gb']:.2f}GB)\n"
            f"    - Hành động: Đang xả sạch gradients, giải phóng caching allocator và bỏ qua batch lỗi an toàn..."
        )
        self.logger.error("!" * 80)

        # 1. Xả sạch optimizer gradients
        try:
            self.optimizer.zero_grad(set_to_none=True)
        except Exception:
            pass

        # 2. Xả scaler nếu có
        if hasattr(self, "scaler") and self.scaler is not None:
            try:
                self.scaler.update()
            except Exception:
                pass

        # 3. Thu dọn bộ nhớ VRAM
        cleanup_cuda_memory(force_gc=True)

    def train_one_epoch(self, epoch: int) -> Tuple[Dict[str, float], float]:
        """
        Thực hiện một epoch huấn luyện với giám sát trực quan qua tqdm,
        hỗ trợ Gradient Accumulation và cơ chế tự phục hồi khi gặp CUDA OOM.
        Returns:
            (train_metrics, avg_grad_norm)
        """
        self.model.train()
        self.metrics_tracker.reset()
        grad_norms: List[float] = []

        total_batches = len(self.train_loader)
        current_lr = self.optimizer.param_groups[0]["lr"]
        use_tqdm = getattr(self.config, "use_tqdm", True)

        accum_steps = max(1, getattr(self.config, "gradient_accumulation_steps", 1))
        empty_cache_interval = getattr(self.config, "empty_cache_interval", 0)

        # Đảm bảo gradients sạch trước khi bắt đầu epoch
        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            self.train_loader,
            desc=f"Train [{epoch:02d}/{self.config.epochs:02d}]",
            total=total_batches,
            dynamic_ncols=True,
            leave=False,
            disable=not use_tqdm,
            file=sys.stdout
        )

        for batch_idx, (features, targets, seq_lens, metas) in enumerate(pbar, start=1):
            try:
                p3, p4, p5 = features
                p3 = p3.to(self.device, non_blocking=True)
                p4 = p4.to(self.device, non_blocking=True)
                p5 = p5.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                seq_lens = seq_lens.to(self.device, non_blocking=True)

                # Chạy forward pass với Automatic Mixed Precision (AMP)
                with self._autocast_context():
                    logits = self.model((p3, p4, p5), seq_lens=seq_lens)
                    loss = self.criterion(logits, targets)
                    loss_scaled = loss / accum_steps

                # Backward pass có điều chỉnh tỷ lệ qua GradScaler
                self.scaler.scale(loss_scaled).backward()

                # Cập nhật trọng số theo chu kỳ tích lũy gradient
                is_accum_step = (batch_idx % accum_steps == 0) or (batch_idx == total_batches)
                if is_accum_step:
                    # Unscale trước khi cắt gradient
                    self.scaler.unscale_(self.optimizer)
                    gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                    grad_norms.append(float(gnorm.item()))

                    # Cập nhật trọng số
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                cur_gnorm = grad_norms[-1] if grad_norms else 0.0

                # Tính toán xác suất và cập nhật metrics (dùng loss gốc không scale)
                with torch.no_grad():
                    probs = F.softmax(logits, dim=-1)
                    preds = torch.argmax(probs, dim=-1)
                    self.metrics_tracker.update(preds, targets, probs, float(loss.item()), p3.size(0))

                temp_acc = (preds == targets).float().mean().item()

                # Truy vấn thông tin VRAM real-time
                vram_postfix = ""
                if self.device.type == "cuda":
                    v_info = get_vram_info(self.device)
                    vram_postfix = f"{v_info['allocated_gb']:.1f}/{v_info['total_gb']:.0f}G"

                if use_tqdm:
                    pbar_kwargs = {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{self.metrics_tracker.running_acc:.1f}%",
                        "gnorm": f"{cur_gnorm:.3f}",
                        "lr": f"{current_lr:.2e}"
                    }
                    if vram_postfix:
                        pbar_kwargs["vram"] = vram_postfix
                    pbar.set_postfix(**pbar_kwargs)
                else:
                    if batch_idx % self.config.log_interval == 0 or batch_idx == total_batches:
                        vram_str = f" | VRAM: {vram_postfix}" if vram_postfix else ""
                        self.logger.info(
                            f"  [Train] Epoch {epoch:02d}/{self.config.epochs:02d} | "
                            f"Batch {batch_idx:03d}/{total_batches:03d} | "
                            f"Loss: {loss.item():.4f} | Acc: {temp_acc * 100:.1f}% | "
                            f"GradNorm: {cur_gnorm:.3f} | LR: {current_lr:.6f}{vram_str}"
                        )

            except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_err:
                if isinstance(oom_err, torch.cuda.OutOfMemoryError) or "out of memory" in str(oom_err).lower():
                    self._handle_cuda_oom(epoch, batch_idx, total_batches, phase="Train")
                    continue
                raise oom_err

            finally:
                # Chủ động thu hồi biến tensor khỏi scope vòng lặp Python
                if "p3" in locals(): del p3
                if "p4" in locals(): del p4
                if "p5" in locals(): del p5
                if "targets" in locals(): del targets
                if "seq_lens" in locals(): del seq_lens
                if "features" in locals(): del features
                if "logits" in locals(): del logits
                if "loss" in locals(): del loss
                if "loss_scaled" in locals(): del loss_scaled
                if "probs" in locals(): del probs
                if "preds" in locals(): del preds

                # Dọn cache định kỳ nếu bật cấu hình empty_cache_interval > 0
                if empty_cache_interval > 0 and (batch_idx % empty_cache_interval == 0):
                    cleanup_cuda_memory(force_gc=False)

        pbar.close()
        train_metrics = self.metrics_tracker.compute()
        avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
        return train_metrics, avg_grad_norm

    def validate(self, epoch: int) -> Tuple[Dict[str, float], np.ndarray, List[int], List[float], float]:
        """
        Thực hiện đánh giá trên tập kiểm định video thô với giám sát qua tqdm,
        tự động dọn dẹp cache GPU và bọc cơ chế xử lý ngoại lệ OOM an toàn.
        Returns:
            (val_metrics, confusion_matrix, targets, probs_drowsy, avg_latency_ms)
        """
        self.model.eval()
        self.metrics_tracker.reset()

        # Giải phóng triệt để cache GPU trước khi validate
        cleanup_cuda_memory(force_gc=True)

        total_batches = len(self.val_loader)
        batch_latencies: List[float] = []
        use_tqdm = getattr(self.config, "use_tqdm", True)

        val_pbar = tqdm(
            self.val_loader,
            desc=f"Val   [{epoch:02d}/{self.config.epochs:02d}]",
            total=total_batches,
            dynamic_ncols=True,
            leave=False,
            disable=not use_tqdm,
            file=sys.stdout
        )

        with torch.no_grad():
            for batch_idx, (features, targets, seq_lens, metas) in enumerate(val_pbar, start=1):
                try:
                    start_t = time.perf_counter()

                    p3, p4, p5 = features
                    p3 = p3.to(self.device, non_blocking=True)
                    p4 = p4.to(self.device, non_blocking=True)
                    p5 = p5.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    seq_lens = seq_lens.to(self.device, non_blocking=True)

                    with self._autocast_context():
                        logits = self.model((p3, p4, p5), seq_lens=seq_lens)
                        loss = self.criterion(logits, targets)

                    probs = F.softmax(logits, dim=-1)
                    preds = torch.argmax(probs, dim=-1)

                    end_t = time.perf_counter()
                    latency_ms = (end_t - start_t) * 1000.0 / max(1, p3.size(0))
                    batch_latencies.append(latency_ms)

                    self.metrics_tracker.update(preds, targets, probs, float(loss.item()), p3.size(0))

                    if use_tqdm:
                        val_pbar.set_postfix(
                            val_loss=f"{self.metrics_tracker.running_loss:.4f}",
                            val_acc=f"{self.metrics_tracker.running_acc:.1f}%",
                            ms=f"{latency_ms:.1f}"
                        )
                    else:
                        if batch_idx % self.config.log_interval == 0 or batch_idx == total_batches:
                            self.logger.info(
                                f"  [Val] Epoch {epoch:02d}/{self.config.epochs:02d} | "
                                f"Batch {batch_idx:03d}/{total_batches:03d} | "
                                f"Loss: {loss.item():.4f} | Acc: {self.metrics_tracker.running_acc:.1f}% | "
                                f"Latency: {latency_ms:.2f}ms/clip"
                            )
                except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_err:
                    if isinstance(oom_err, torch.cuda.OutOfMemoryError) or "out of memory" in str(oom_err).lower():
                        self._handle_cuda_oom(epoch, batch_idx, total_batches, phase="Val")
                        continue
                    raise oom_err
                finally:
                    if "p3" in locals(): del p3
                    if "p4" in locals(): del p4
                    if "p5" in locals(): del p5
                    if "targets" in locals(): del targets
                    if "seq_lens" in locals(): del seq_lens
                    if "features" in locals(): del features
                    if "logits" in locals(): del logits
                    if "loss" in locals(): del loss
                    if "probs" in locals(): del probs
                    if "preds" in locals(): del preds

        val_pbar.close()

        # Giải phóng cache GPU sau khi hoàn tất kiểm định
        cleanup_cuda_memory(force_gc=True)

        val_metrics = self.metrics_tracker.compute()
        cm = self.metrics_tracker.get_confusion_matrix()
        avg_latency = float(np.mean(batch_latencies)) if batch_latencies else 0.0

        return val_metrics, cm, self.metrics_tracker.targets, self.metrics_tracker.probs_drowsy, avg_latency

    def fit(self) -> Dict[str, Any]:
        """Vòng lặp chính điều phối quá trình huấn luyện và kiểm định qua các epoch."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"   BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN (TỔNG {self.config.epochs} EPOCHS)")
        self.logger.info("=" * 80)

        total_train_start = time.time()
        stopping_reason = "Completed all epochs"

        best_val_metrics: Dict[str, float] = {}
        best_cm: Optional[np.ndarray] = None
        best_targets: List[int] = []
        best_probs: List[float] = []

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            epoch_start = time.time()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Reset thống kê đỉnh VRAM đầu mỗi epoch nếu dùng GPU CUDA
            if self.device.type == "cuda":
                try:
                    torch.cuda.reset_peak_memory_stats(self.device)
                except Exception:
                    pass

            # 1. Chạy Pha Huấn Luyện (Train Pass)
            train_metrics, grad_norm = self.train_one_epoch(epoch)

            # Dọn dẹp cache sau train pass trước khi sang validation
            cleanup_cuda_memory(force_gc=True)

            # 2. Chạy Pha Kiểm Định (Validation Pass) nếu đến chu kỳ
            val_metrics = None
            avg_latency = 0.0
            if epoch % self.config.val_interval_epochs == 0 or epoch == self.config.epochs:
                val_metrics, cm, targets, probs, avg_latency = self.validate(epoch)

            # Dọn dẹp cache sau val pass trước khi cập nhật scheduler và lưu checkpoint
            cleanup_cuda_memory(force_gc=True)

            # 3. Cập nhật Scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics is not None:
                        monitor_val = self.checkpoint_manager._extract_metric_score(val_metrics) or 0.0
                        self.scheduler.step(monitor_val)
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # 4. Ghi nhận và Trực quan hóa (Visualizer & Diagnostics)
            log_entry = self.visualizer.log_epoch(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=current_lr,
                grad_norm=grad_norm,
                epoch_time_s=epoch_time
            )

            # 5. In Bảng Tổng Kết Epoch & Chẩn đoán Overfitting / VRAM
            self.logger.info("-" * 80)
            log_str = (
                f"[EPOCH {epoch:02d}/{self.config.epochs:02d}] "
                f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy'] * 100:.1f}% | Train F1: {train_metrics['f1']:.4f} | "
            )
            if val_metrics is not None:
                log_str += (
                    f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy'] * 100:.1f}% | "
                    f"Val F1: {val_metrics['f1']:.4f} | Val Rec: {val_metrics['recall']:.4f} | "
                    f"Gap: {log_entry['gap_loss']:+.4f} | Latency: {avg_latency:.1f}ms/clip"
                )
            log_str += f" | Time: {epoch_time:.1f}s"

            if self.device.type == "cuda":
                v_info = get_vram_info(self.device)
                log_str += f" | Peak VRAM: {v_info['peak_gb']:.2f}/{v_info['total_gb']:.1f}GB ({v_info['percent_used']:.1f}%)"
                if v_info["percent_used"] > 90.0:
                    self.logger.warning(
                        f"  [!] CẢNH BÁO VRAM: Mức chiếm dụng VRAM ({v_info['percent_used']:.1f}%) vượt ngưỡng 90%! "
                        f"Khuyến nghị giảm batch_size hoặc tăng gradient_accumulation_steps."
                    )

            self.logger.info(log_str)

            # Chẩn đoán Overfitting nếu Gap Loss tăng quá cao
            if val_metrics is not None and log_entry["gap_loss"] > 0.5:
                self.logger.warning(
                    f"  [!] CẢNH BÁO OVERFITTING: Khoảng cách tổng quát hóa (Gap Loss = {log_entry['gap_loss']:.4f}) "
                    f"vượt ngưỡng 0.50! Val Loss đang cao hơn đáng kể so với Train Loss."
                )

            # 6. Quản lý Checkpoints & Kiểm tra Dừng sớm (Early Stopping)
            should_stop = self.checkpoint_manager.step(
                current_metrics=val_metrics,
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                train_metrics=train_metrics,
                val_metrics=val_metrics
            )

            if val_metrics is not None and self.checkpoint_manager.best_epoch == epoch:
                best_val_metrics = val_metrics
                best_cm = cm
                best_targets = targets
                best_probs = probs

            if should_stop:
                stopping_reason = f"Early Stopping triggered at epoch {epoch} (patience={self.config.patience})"
                break

        total_train_time = time.time() - total_train_start
        self.logger.info("=" * 80)
        self.logger.info(f"   KẾT THÚC HUẤN LUYỆN — Tổng thời gian: {total_train_time:.1f}s")
        self.logger.info(f"   Lý do kết thúc: {stopping_reason}")
        self.logger.info(f"   Checkpoint tốt nhất tại Epoch {self.checkpoint_manager.best_epoch} "
                         f"(Best {self.config.monitor_metric} = {self.checkpoint_manager.best_score:.4f})")
        self.logger.info("=" * 80)

        # 7. Lưu bảng lịch sử CSV và vẽ đồ thị học tập
        csv_path = self.visualizer.save_history_csv()
        curves_path = self.visualizer.plot_learning_curves()
        self.logger.info(f"[+] Đã lưu lịch sử huấn luyện vào: {csv_path.resolve()}")
        self.logger.info(f"[+] Đã xuất biểu đồ học tập vào: {curves_path.resolve()}")

        return {
            "total_time_s": total_train_time,
            "best_epoch": self.checkpoint_manager.best_epoch,
            "best_score": self.checkpoint_manager.best_score,
            "stopping_reason": stopping_reason,
            "best_metrics": best_val_metrics,
            "best_cm": best_cm,
            "best_targets": best_targets,
            "best_probs": best_probs
        }

    def evaluate_final(self, fit_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Nạp lại Checkpoint tốt nhất (best.pt) và thực hiện đánh giá chuyên sâu
        kết quả mô hình, xuất các biểu đồ chẩn đoán (CM Heatmap, ROC/PR Curves) và JSON summary.
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("   ĐÁNH GIÁ CHUYÊN SÂU KẾT QUẢ MÔ HÌNH VỚI CHECKPOINT TỐT NHẤT")
        self.logger.info("=" * 80)

        best_ckpt_path = Path(self.config.checkpoint_dir) / f"{self.config.experiment_name}_best.pt"
        if best_ckpt_path.exists():
            self.logger.info(f"[*] Nạp trọng số tối ưu từ: {best_ckpt_path.resolve()}")
            ckpt = torch.load(str(best_ckpt_path), map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.logger.warning("[!] Không tìm thấy best checkpoint, sử dụng trọng số hiện tại của mô hình.")

        # Chạy kiểm định toàn diện
        val_metrics, cm, targets, probs, latency = self.validate(epoch=self.checkpoint_manager.best_epoch)

        # In Báo cáo Phân loại Chi tiết
        report_str = self.metrics_tracker.get_classification_report()
        self.logger.info("\n[*] BÁO CÁO PHÂN LOẠI CHI TIẾT (CLASSIFICATION REPORT):")
        self.logger.info("\n" + report_str)

        # In Ma trận Nhầm lẫn
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        self.logger.info("[*] CHI TIẾT MA TRẬN NHẦM LẪN:")
        self.logger.info(f"    - True Negatives  (Alert đoán đúng Alert):    {tn:4d}")
        self.logger.info(f"    - False Positives (Alert đoán nhầm Drowsy):   {fp:4d} (Báo động giả)")
        self.logger.info(f"    - False Negatives (Drowsy đoán nhầm Alert):   {fn:4d} (Bỏ sót nguy hiểm)")
        self.logger.info(f"    - True Positives  (Drowsy đoán đúng Drowsy):  {tp:4d}")

        # Xuất biểu đồ Heatmap Confusion Matrix
        cm_path = self.visualizer.plot_confusion_matrix(cm, class_names=["0_Alert", "1_Drowsy"])
        self.logger.info(f"[+] Đã xuất biểu đồ Ma trận nhầm lẫn: {cm_path.resolve()}")

        # Xuất biểu đồ ROC & PR Curves
        roc_path = self.visualizer.plot_roc_pr_curves(targets, probs)
        if roc_path:
            self.logger.info(f"[+] Đã xuất biểu đồ ROC & PR Curves: {roc_path.resolve()}")

        # Lưu bản tóm tắt JSON toàn diện
        summary_payload = {
            "experiment_name": self.config.experiment_name,
            "best_epoch": self.checkpoint_manager.best_epoch,
            "monitor_metric": self.config.monitor_metric,
            "best_score": self.checkpoint_manager.best_score,
            "val_metrics": val_metrics,
            "confusion_matrix": {
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
            },
            "avg_latency_ms_per_clip": round(latency, 2),
            "config": self.config.to_dict()
        }
        json_path = self.visualizer.save_summary_json(summary_payload)
        self.logger.info(f"[+] Đã lưu bản tóm tắt cấu hình & kết quả: {json_path.resolve()}")

        # Đóng tài nguyên
        self.close()

        self.logger.info("\n" + "=" * 80)
        self.logger.info("   [✓] HOÀN TẤT THÀNH CÔNG TOÀN BỘ PIPELINE HUẤN LUYỆN VÀ ĐÁNH GIÁ!")
        self.logger.info("=" * 80)
        return summary_payload

    def close(self) -> None:
        """Đóng an toàn các tài nguyên mở (Datasets, Visualizer, Logger handlers, giải phóng VRAM)."""
        cleanup_cuda_memory(force_gc=True)
        if hasattr(self, "train_loader") and hasattr(self.train_loader, "dataset"):
            if hasattr(self.train_loader.dataset, "close"):
                try:
                    self.train_loader.dataset.close()
                except Exception:
                    pass
        if hasattr(self, "val_loader") and hasattr(self.val_loader, "dataset"):
            if hasattr(self.val_loader.dataset, "close"):
                try:
                    self.val_loader.dataset.close()
                except Exception:
                    pass
        if hasattr(self, "visualizer") and hasattr(self.visualizer, "close"):
            try:
                self.visualizer.close()
            except Exception:
                pass
        if hasattr(self, "logger"):
            for h in list(self.logger.handlers):
                try:
                    h.close()
                    self.logger.removeHandler(h)
                except Exception:
                    pass
        cleanup_cuda_memory(force_gc=True)


# Alias hỗ trợ gọi tương đương DrowsinessTrainer
DrowsinessTrainer = DrowsinessTrainer1


# ==============================================================================
# 6. CƠ CHẾ KIỂM THỬ ĐỘC LẬP TỰ ĐỘNG: run_dry_run_test (Video Thô)
# ==============================================================================
def run_dry_run_test() -> None:
    """
    Chạy thử nghiệm độc lập toàn diện quy trình huấn luyện và đánh giá
    trên dữ liệu giả lập (mock raw video files .mp4), xác nhận 100% các thành phần hoạt động trơn tru.
    """
    print("\n" + "=" * 80)
    print("   [DRY-RUN TEST] BẮT ĐẦU KIỂM THỬ TỰ ĐỘNG TOÀN DIỆN PIPELINE HUẤN LUYỆN (RAW VIDEO)")
    print("=" * 80)

    temp_dir = tempfile.mkdtemp(prefix="driver_guardian_train1_dryrun_")
    temp_path = Path(temp_dir)

    try:
        # 1. Tạo video clips giả lập cho cả train và val
        print("\n[*] [Bước 1] Khởi tạo các tệp video giả lập (Mock Video Files)...")
        train_alert_dir = temp_path / "train" / "0_alert"
        train_drowsy_dir = temp_path / "train" / "1_drowsy"
        val_alert_dir = temp_path / "val" / "0_alert"
        val_drowsy_dir = temp_path / "val" / "1_drowsy"

        for d in (train_alert_dir, train_drowsy_dir, val_alert_dir, val_drowsy_dir):
            d.mkdir(parents=True, exist_ok=True)

        def create_dummy_video(file_path: Path, num_frames: int = 15, fps: float = 30.0, color=(100, 150, 200)):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(file_path), fourcc, fps, (320, 240))
            for f_i in range(num_frames):
                img = np.full((240, 320, 3), color, dtype=np.uint8)
                cv2.putText(img, f"Frame {f_i}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                writer.write(img)
            writer.release()

        v1_path = train_alert_dir / "sample_alert_1.mp4"
        v2_path = train_drowsy_dir / "sample_drowsy_1.mp4"
        v3_path = val_alert_dir / "sample_alert_val.mp4"
        v4_path = val_drowsy_dir / "sample_drowsy_val.mp4"

        create_dummy_video(v1_path, num_frames=15, fps=30.0, color=(50, 100, 150))
        create_dummy_video(v2_path, num_frames=20, fps=30.0, color=(150, 50, 50))
        create_dummy_video(v3_path, num_frames=12, fps=20.0, color=(50, 150, 50))
        create_dummy_video(v4_path, num_frames=15, fps=25.0, color=(150, 150, 50))

        print(f"    [✓] Đã tạo thành công 4 video mock tại: {temp_dir}")

        # 2. Cấu hình bài test mô phỏng
        print("\n[*] [Bước 2] Thiết lập cấu hình TrainConfig thử nghiệm...")
        cfg = TrainConfig(
            dataset_dir=str(temp_path),
            val_dataset_dir=str(temp_path),
            checkpoint_dir=str(temp_path / "checkpoints"),
            log_dir=str(temp_path / "logs"),
            tb_log_dir=str(temp_path / "runs"),
            history_csv_path=str(temp_path / "logs" / "training_history.csv"),
            summary_json_path=str(temp_path / "logs" / "training_summary.json"),
            plot_curves_path=str(temp_path / "logs" / "loss_accuracy_curves.png"),
            plot_cm_path=str(temp_path / "logs" / "confusion_matrix_best.png"),
            plot_roc_path=str(temp_path / "logs" / "roc_pr_curves.png"),
            experiment_name="dryrun_test1",
            epochs=2,
            batch_size=2,
            val_batch_size=2,
            sample_interval=0.1,
            num_workers=0,
            val_num_workers=0,
            gradient_accumulation_steps=2,
            device="cuda" if torch.cuda.is_available() else "cpu",
            amp=False,
            use_tqdm=False,
            save_all_epochs=True,
            enable_resume=False,
            dry_run=True
        )

        # 3. Khởi tạo Trainer và chạy huấn luyện 2 epochs
        print("[*] [Bước 3] Khởi tạo DrowsinessTrainer1 và thực thi 2 epochs (save_all_epochs=True)...")
        trainer = DrowsinessTrainer1(config=cfg)
        fit_results = trainer.fit()

        # 4. Chạy đánh giá cuối cùng
        print("[*] [Bước 4] Thực thi đánh giá chuyên sâu evaluate_final()...")
        summary_payload = trainer.evaluate_final(fit_results)
        trainer.close()

        # 5. Kiểm tra tính toàn vẹn của các file kết quả và Checkpoint
        print("\n[*] [Bước 5] Xác thực lưu trữ Checkpoint tất cả các epoch (save_all_epochs):")
        ckpt_ep1 = temp_path / "checkpoints" / "dryrun_test1_epoch_001.pt"
        ckpt_ep2 = temp_path / "checkpoints" / "dryrun_test1_epoch_002.pt"
        ckpt_last = temp_path / "checkpoints" / "dryrun_test1_last.pt"
        ckpt_best = temp_path / "checkpoints" / "dryrun_test1_best.pt"

        assert ckpt_ep1.exists(), "LỖI: Thiếu checkpoint Epoch 1: dryrun_test1_epoch_001.pt!"
        assert ckpt_ep2.exists(), "LỖI: Thiếu checkpoint Epoch 2: dryrun_test1_epoch_002.pt!"
        assert ckpt_last.exists(), "LỖI: Thiếu checkpoint gần nhất: dryrun_test1_last.pt!"
        assert ckpt_best.exists(), "LỖI: Thiếu checkpoint tốt nhất: dryrun_test1_best.pt!"
        print(f"    [✓] Đã tạo thành công: {ckpt_ep1.name}")
        print(f"    [✓] Đã tạo thành công: {ckpt_ep2.name}")
        print(f"    [✓] Đã tạo thành công: {ckpt_last.name}")
        print(f"    [✓] Đã tạo thành công: {ckpt_best.name}")

        # Kiểm tra nội dung checkpoint
        ckpt_data = torch.load(str(ckpt_ep1), map_location="cpu")
        assert ckpt_data["epoch"] == 1, "Checkpoint epoch 1 không đúng epoch!"
        assert "best_epoch" in ckpt_data, "Thiếu trường 'best_epoch' trong checkpoint!"
        assert "patience_counter" in ckpt_data, "Thiếu trường 'patience_counter' trong checkpoint!"
        print("    [✓] Cấu trúc checkpoint hợp lệ và đầy đủ metadata trạng thái.")

        assert Path(cfg.history_csv_path).exists(), "Thiếu file training_history.csv!"
        assert Path(cfg.summary_json_path).exists(), "Thiếu file training_summary.json!"
        assert Path(cfg.plot_curves_path).exists(), "Thiếu file loss_accuracy_curves.png!"
        assert Path(cfg.plot_cm_path).exists(), "Thiếu file confusion_matrix_best.png!"
        print(f"    [✓] Đã tạo thành công: {cfg.history_csv_path}")
        print(f"    [✓] Đã tạo thành công: {cfg.summary_json_path}")
        print(f"    [✓] Đã tạo thành công: {cfg.plot_curves_path}")
        print(f"    [✓] Đã tạo thành công: {cfg.plot_cm_path}")

        # 6. Kiểm tra cơ chế bật/tắt Resume
        print("\n[*] [Bước 6] Kiểm tra cơ chế Bật/Tắt Resume (enable_resume):")
        from dataclasses import replace

        # 6a. Thử nghiệm khi enable_resume=False: Phải luôn bắt đầu từ Epoch 1
        cfg_no_resume = replace(cfg, enable_resume=False, resume_epoch=1)
        trainer_no_resume = DrowsinessTrainer1(config=cfg_no_resume)
        assert trainer_no_resume.start_epoch == 1, (
            f"LỖI: Khi enable_resume=False, start_epoch phải là 1 nhưng lại là {trainer_no_resume.start_epoch}!"
        )
        trainer_no_resume.close()
        print("    [✓] [enable_resume=False]: Huấn luyện khởi tạo mới an toàn từ Epoch 1.")

        # 6b. Thử nghiệm khi enable_resume=True và resume_epoch=1: Bắt đầu từ Epoch 2
        print("\n[*] [Bước 7] Kiểm tra nạp lại từ Epoch 1 (enable_resume=True, resume_epoch=1):")
        cfg_resume_ep1 = replace(cfg, enable_resume=True, resume_epoch=1)
        trainer_resume_ep1 = DrowsinessTrainer1(config=cfg_resume_ep1)
        assert trainer_resume_ep1.start_epoch == 2, (
            f"LỖI: Khi nạp lại từ Epoch 1, start_epoch phải là 2 nhưng lại là {trainer_resume_ep1.start_epoch}!"
        )
        assert len(trainer_resume_ep1.visualizer.history) == 1, (
            f"LỖI: Lịch sử visualizer chưa được đồng bộ chính xác (kỳ vọng 1 dòng, thực tế {len(trainer_resume_ep1.visualizer.history)})!"
        )
        trainer_resume_ep1.close()
        print("    [✓] [enable_resume=True, resume_epoch=1]: Khôi phục thành công, tiếp tục từ Epoch 2.")

        # 6c. Thử nghiệm nạp epoch không tồn tại (vd: epoch 99)
        print("\n[*] [Bước 8] Kiểm tra xử lý ngoại lệ khi nạp Epoch không tồn tại:")
        dummy_logger = logging.getLogger("TestCheckpointManager1")
        test_cm = CheckpointManager(cfg, dummy_logger)
        try:
            test_cm.resolve_checkpoint_path(resume_epoch=99)
            raise AssertionError("LỖI: Hệ thống không báo lỗi khi nạp epoch 99 không tồn tại!")
        except FileNotFoundError as fnf_err:
            assert "Epoch 99" in str(fnf_err), "Ngoại lệ không đề cập đến Epoch 99!"
            assert "Các checkpoint epoch sẵn có" in str(fnf_err), "Ngoại lệ thiếu danh sách các epoch sẵn có!"
            print(f"    [✓] Báo lỗi ngoại lệ thân thiện thành công:\n        {fnf_err.args[0].splitlines()[0]}")

        # 7. Kiểm tra đo đạc thông số VRAM & dọn dẹp bộ nhớ
        print("\n[*] [Bước 9] Kiểm tra chức năng giám sát VRAM (get_vram_info) & cleanup_cuda_memory:")
        v_test = get_vram_info()
        assert isinstance(v_test, dict) and "allocated_gb" in v_test and "total_gb" in v_test, "LỖI cấu trúc get_vram_info!"
        cleanup_cuda_memory(force_gc=True)
        print(f"    [✓] get_vram_info hoạt động chuẩn xác: {v_test}")

        # 8. Kiểm tra cơ chế tự phục hồi OOM (_handle_cuda_oom)
        print("\n[*] [Bước 10] Kiểm tra cơ chế tự phục hồi CUDA OOM (_handle_cuda_oom):")
        cfg_oom_test = replace(cfg, enable_resume=False)
        mock_trainer = DrowsinessTrainer1(config=cfg_oom_test)
        try:
            mock_trainer._handle_cuda_oom(epoch=1, batch_idx=1, total_batches=2, phase="Mock_OOM_Test")
            print("    [✓] _handle_cuda_oom xả gradients, dọn dẹp cache VRAM và xử lý an toàn không gây sập!")
        finally:
            mock_trainer.close()

        print("\n" + "=" * 80)
        print("   >>> [THÀNH CÔNG 100%] KIỂM THỬ DRY-RUN TOÀN BỘ CƠ CHẾ TRAIN ĐẠT CHUẨN! <<<")
        print("=" * 80)

    finally:
        # Dọn dẹp an toàn tài nguyên trainer trước khi xóa thư mục tạm
        if "trainer" in locals():
            try:
                trainer.close()
            except Exception:
                pass
        if "trainer_no_resume" in locals():
            try:
                trainer_no_resume.close()
            except Exception:
                pass
        if "trainer_resume_ep1" in locals():
            try:
                trainer_resume_ep1.close()
            except Exception:
                pass
        logging.shutdown()
        time.sleep(0.1)
        # Dọn dẹp sạch sẽ thư mục tạm
        if temp_path.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"[*] Đã dọn dẹp an toàn thư mục tạm: {temp_dir}")


# ==============================================================================
# 7. HÀM FACTORY TIỆN ÍCH CHO EXTERNAL IMPORT & ENTRY POINT
# ==============================================================================
def train_pipeline(
    config: Optional[TrainConfig] = None,
    enable_resume: Optional[bool] = None,
    resume_epoch: Optional[int] = None
) -> Dict[str, Any]:
    """
    Hàm giao diện tiện ích cấp cao để khởi chạy toàn bộ pipeline huấn luyện video thô.
    Cho phép gọi trực tiếp từ script Python khác hoặc Jupyter Notebooks.
    """
    cfg = config if config is not None else load_config()
    if enable_resume is not None:
        cfg.enable_resume = enable_resume
    if resume_epoch is not None:
        cfg.resume_epoch = resume_epoch
        if enable_resume is None:
            cfg.enable_resume = True

    if cfg.dry_run:
        run_dry_run_test()
        return {"status": "dry_run_completed"}

    trainer = DrowsinessTrainer1(config=cfg)
    fit_results = trainer.fit()
    summary = trainer.evaluate_final(fit_results)
    return summary


def main() -> None:
    """Điểm khởi chạy chính khi thực thi tệp: python src/train1.py."""
    config = load_config()
    if config.dry_run:
        run_dry_run_test()
    else:
        trainer = DrowsinessTrainer1(config=config)
        fit_results = trainer.fit()
        trainer.evaluate_final(fit_results)


if __name__ == "__main__":
    main()
