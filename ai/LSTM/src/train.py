#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Tệp: src/train.py
Mục đích:
    Pipeline hoàn chỉnh huấn luyện mô hình Deep GRU (Drowsiness Detection),
    đánh giá kết quả mô hình (Model Evaluation) và chẩn đoán quá trình huấn luyện
    (Training Process Diagnostics).

Luồng dữ liệu:
    - Tập Huấn luyện (Train): HDF5FeatureDataset từ src/dataset.py (nạp từ file HDF5 .h5)
    - Tập Kiểm định (Val): HDF5FeatureDataset từ src/dataset.py (nạp từ file HDF5 .h5)
    - Mô hình: DeepGRUClassifier từ src/models.py (CNNAdapter + Deep GRU + TemporalAttentionPooling + FC)
    - Hàm mất mát: DrowsinessLoss từ src/loss.py (CrossEntropyLoss hỗ trợ pos_weight)

Cơ chế cấu hình:
    - 100% CẤU HÌNH TẬP TRUNG QUA configs/config.py (TrainConfig)
    - KHÔNG DÙNG GIAO DIỆN DÒNG LỆNH (No CLI / No Argparse)
    - Thực thi trực tiếp: python train.py
"""

import os
import sys
import time
import json
import csv
import math
import random
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union

# Đảm bảo console Windows hỗ trợ in tiếng Việt UTF-8 không lỗi charmap
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Thêm thư mục gốc dự án vào sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Thiết lập môi trường tối ưu cho OpenMP và ONNX Runtime
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["ORT_LOG_LEVEL"] = "3"

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
from src.dataset import HDF5FeatureDataset, collate_h5_features, seed_worker, create_mock_h5_dataset
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


def setup_logger(log_dir: Union[str, Path], experiment_name: str) -> logging.Logger:
    """Khởi tạo logger ghi log đồng thời ra console và tệp tin .log."""
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = log_dir_path / f"{experiment_name}.log"

    logger = logging.getLogger(f"Trainer_{experiment_name}")
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

        # Tính toán Accuracy
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
        """
        Ghi nhận tiến trình của 1 epoch vào bộ nhớ lịch sử và TensorBoard.
        Tính toán khoảng cách tổng quát hóa (Generalization Gap) để chẩn đoán Overfitting.
        """
        val_m = val_metrics if val_metrics is not None else {}
        gap_loss = (val_m.get("loss", 0.0) - train_metrics.get("loss", 0.0)) if val_metrics else 0.0
        gap_f1 = (train_metrics.get("f1", 0.0) - val_m.get("f1", 0.0)) if val_metrics else 0.0

        entry = {
            "epoch": epoch,
            "train_loss": train_metrics.get("loss", 0.0),
            "train_acc": train_metrics.get("accuracy", 0.0),
            "train_f1": train_metrics.get("f1", 0.0),
            "val_loss": val_m.get("loss", 0.0),
            "val_acc": val_m.get("accuracy", 0.0),
            "val_f1": val_m.get("f1", 0.0),
            "val_recall": val_m.get("recall", 0.0),
            "val_precision": val_m.get("precision", 0.0),
            "val_specificity": val_m.get("specificity", 0.0),
            "val_auc_roc": val_m.get("auc_roc", 0.0),
            "val_auc_pr": val_m.get("auc_pr", 0.0),
            "lr": lr,
            "grad_norm": grad_norm,
            "gap_loss": gap_loss,
            "gap_f1": gap_f1,
            "epoch_time_s": round(epoch_time_s, 2)
        }
        self.history.append(entry)

        # Ghi vào TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("Loss/train", train_metrics.get("loss", 0.0), epoch)
            self.tb_writer.add_scalar("Accuracy/train", train_metrics.get("accuracy", 0.0), epoch)
            self.tb_writer.add_scalar("F1/train", train_metrics.get("f1", 0.0), epoch)
            self.tb_writer.add_scalar("Optimizer/lr", lr, epoch)
            self.tb_writer.add_scalar("Optimizer/grad_norm", grad_norm, epoch)

            if val_metrics is not None:
                self.tb_writer.add_scalar("Loss/val", val_m.get("loss", 0.0), epoch)
                self.tb_writer.add_scalar("Accuracy/val", val_m.get("accuracy", 0.0), epoch)
                self.tb_writer.add_scalar("F1/val", val_m.get("f1", 0.0), epoch)
                self.tb_writer.add_scalar("Recall/val_drowsy", val_m.get("recall", 0.0), epoch)
                self.tb_writer.add_scalar("Precision/val_drowsy", val_m.get("precision", 0.0), epoch)
                self.tb_writer.add_scalar("AUC/val_roc", val_m.get("auc_roc", 0.0), epoch)
                self.tb_writer.add_scalar("Diagnostics/gap_loss", gap_loss, epoch)

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


# ==============================================================================
# 4. BỘ QUẢN LÝ CHECKPOINTS & EARLY STOPPING: CheckpointManager
# ==============================================================================
class CheckpointManager:
    """Quản lý lưu trữ checkpoint mô hình và cơ chế dừng sớm (Early Stopping)."""

    def __init__(self, config: TrainConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = config.monitor_metric.lower()
        self.mode = config.monitor_mode.lower()
        self.patience = config.patience
        self.min_delta = config.min_delta
        self.early_stopping_enabled = config.early_stopping

        self.patience_counter = 0
        self.best_epoch = 0
        self.best_score = -float("inf") if self.mode == "max" else float("inf")
        self.periodic_ckpts: List[Path] = []

    def _is_better(self, score: float) -> bool:
        """So sánh điểm số hiện tại với kỷ lục tốt nhất."""
        if self.mode == "max":
            return score > (self.best_score + self.min_delta)
        else:
            return score < (self.best_score - self.min_delta)

    def step(
        self,
        current_metrics: Dict[str, float],
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None
    ) -> bool:
        """
        Cập nhật trạng thái sau mỗi epoch kiểm định.
        Returns:
            should_stop (bool): True nếu kích hoạt Early Stopping, ngược lại False.
        """
        score = current_metrics.get(self.monitor, 0.0)

        # Chuẩn bị state dictionary đầy đủ
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "metrics": current_metrics,
            "best_score": self.best_score,
            "config": self.config.to_dict()
        }

        # 1. Luôn lưu checkpoint gần nhất (last_model.pt) để chống mất mát
        last_ckpt_path = self.checkpoint_dir / f"{self.config.experiment_name}_last.pt"
        torch.save(checkpoint_data, str(last_ckpt_path))

        # 2. Kiểm tra kỷ lục tốt nhất (best_model.pt)
        is_best = self._is_better(score)
        if is_best:
            old_best = self.best_score
            self.best_score = score
            self.best_epoch = epoch
            self.patience_counter = 0
            best_ckpt_path = self.checkpoint_dir / f"{self.config.experiment_name}_best.pt"
            torch.save(checkpoint_data, str(best_ckpt_path))
            self.logger.info(
                f"[★ BEST CHECKPOINT] Epoch {epoch}: Chỉ số '{self.monitor}' cải thiện từ "
                f"{old_best:.4f} -> {score:.4f}. Đã lưu: {best_ckpt_path.name}"
            )
        else:
            self.patience_counter += 1
            self.logger.info(
                f"[i] Epoch {epoch}: Chỉ số '{self.monitor}'={score:.4f} không cải thiện "
                f"(Best: {self.best_score:.4f} tại epoch {self.best_epoch}). Patience: {self.patience_counter}/{self.patience}"
            )

        # 3. Lưu checkpoint định kỳ theo save_ckpt_interval_epochs
        if not self.config.save_best_only and (epoch % self.config.save_ckpt_interval_epochs == 0):
            interval_ckpt = self.checkpoint_dir / f"{self.config.experiment_name}_epoch_{epoch:03d}.pt"
            torch.save(checkpoint_data, str(interval_ckpt))
            self.periodic_ckpts.append(interval_ckpt)
            # Giữ lại số checkpoint gần nhất theo cấu hình ckpt_keep_last
            if len(self.periodic_ckpts) > self.config.ckpt_keep_last:
                old_ckpt = self.periodic_ckpts.pop(0)
                if old_ckpt.exists():
                    old_ckpt.unlink()

        # 4. Kiểm tra điều kiện Dừng sớm (Early Stopping)
        if self.early_stopping_enabled and (self.patience_counter >= self.patience):
            self.logger.warning(
                f"\n[!] EARLY STOPPING ĐƯỢC KÍCH HOẠT: Chỉ số '{self.monitor}' không cải thiện "
                f"sau {self.patience} epochs liên tiếp. Dừng sớm tại Epoch {epoch}!"
            )
            return True

        return False

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None
    ) -> int:
        """Nạp checkpoint phục vụ tiếp tục huấn luyện (Resume)."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file checkpoint: {path}")

        self.logger.info(f"[*] Đang nạp checkpoint từ: {path.resolve()}")
        checkpoint = torch.load(str(path), map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        self.best_score = float(checkpoint.get("best_score", self.best_score))
        self.best_epoch = int(checkpoint.get("epoch", 0))
        self.logger.info(f"[✓] Đã khôi phục trạng thái thành công! Sẽ bắt đầu huấn luyện từ Epoch {start_epoch}.")
        return start_epoch


# ==============================================================================
# 5. ĐỘNG CƠ HUẤN LUYỆN LÕI: DrowsinessTrainer
# ==============================================================================
class DrowsinessTrainer:
    """Động cơ điều phối toàn diện pipeline huấn luyện, kiểm định và chẩn đoán."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config

        # 1. Cố định tính ngẫu nhiên (Reproducibility)
        seed_everything(config.seed)

        # 2. Khởi tạo Logger và Visualizer
        self.logger = setup_logger(config.log_dir, config.experiment_name)
        self.logger.info("=" * 80)
        self.logger.info(f"   KHỞI TẠO PIPELINE HUẤN LUYỆN: {config.experiment_name.upper()}")
        self.logger.info("=" * 80)

        # 3. Xác định Thiết bị Tính toán (CUDA vs CPU)
        if config.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.logger.info(f"[+] Thiết bị tính toán: GPU NVIDIA ({torch.cuda.get_device_name(0)})")
        else:
            self.device = torch.device("cpu")
            self.logger.info("[+] Thiết bị tính toán: CPU")

        self.amp = config.amp and (self.device.type == "cuda")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        # 4. Khởi tạo DataLoaders (Train HDF5 + Val Raw Video ONNX)
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

        # Biến theo dõi trạng thái
        self.start_epoch = 1
        if config.resume:
            self.start_epoch = self.checkpoint_manager.load_checkpoint(
                config.resume, self.model, self.optimizer, self.scheduler, self.scaler
            )

    def _build_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Khởi tạo Train DataLoader và Val DataLoader từ tệp HDF5 (HDF5FeatureDataset)."""
        self.logger.info(f"[*] Nạp tập huấn luyện HDF5: {self.config.train_h5}")
        train_h5_path = Path(self.config.train_h5)
        if not train_h5_path.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp HDF5 tập train tại: {train_h5_path.resolve()}")

        # Khởi tạo Train Dataset (HDF5FeatureDataset)
        train_dataset = HDF5FeatureDataset(
            h5_path=train_h5_path,
            manifest_csv=self.config.train_manifest_csv,
            split="train",
            seq_len=self.config.seq_len,
            include_augmented=self.config.include_augmented_train,
            window_sampling="random"
        )
        self.logger.info(f"    -> Đã nạp thành công {len(train_dataset)} mẫu huấn luyện từ tệp HDF5.")

        g = torch.Generator()
        g.manual_seed(self.config.seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            collate_fn=collate_h5_features,
            pin_memory=self.config.pin_memory if self.device.type == "cuda" else False,
            persistent_workers=self.config.persistent_workers if self.config.num_workers > 0 else False,
            worker_init_fn=seed_worker,
            generator=g,
            drop_last=self.config.drop_last
        )

        # Khởi tạo Validation Dataset (HDF5FeatureDataset)
        val_h5_str = getattr(self.config, "val_h5", None) or self.config.train_h5
        val_h5_path = Path(val_h5_str)
        self.logger.info(f"[*] Nạp tập kiểm định HDF5: {val_h5_path.resolve()}")
        if not val_h5_path.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp HDF5 tập validation tại: {val_h5_path.resolve()}")

        val_manifest = getattr(self.config, "val_manifest_csv", None) or self.config.train_manifest_csv
        val_dataset = HDF5FeatureDataset(
            h5_path=val_h5_path,
            manifest_csv=val_manifest,
            split="val",
            seq_len=self.config.seq_len,
            stride=1,
            include_augmented=False,  # Tuyệt đối không dùng augmentation cho validation (chống rò rỉ dữ liệu)
            window_sampling="center"  # Lấy cửa sổ trung tâm có tính xác định cao
        )
        self.logger.info(f"    -> Đã nạp thành công {len(val_dataset)} mẫu kiểm định từ tệp HDF5: {val_h5_path.name}")

        val_workers = getattr(self.config, "val_num_workers", self.config.num_workers)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=val_workers,
            collate_fn=collate_h5_features,
            pin_memory=self.config.pin_memory if self.device.type == "cuda" else False,
            persistent_workers=self.config.persistent_workers if val_workers > 0 else False,
            worker_init_fn=seed_worker,
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

    def train_one_epoch(self, epoch: int) -> Tuple[Dict[str, float], float]:
        """
        Thực hiện một epoch huấn luyện với giám sát trực quan qua tqdm.
        Returns:
            (train_metrics, avg_grad_norm)
        """
        self.model.train()
        self.metrics_tracker.reset()
        grad_norms: List[float] = []

        total_batches = len(self.train_loader)
        current_lr = self.optimizer.param_groups[0]["lr"]
        use_tqdm = getattr(self.config, "use_tqdm", True)

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
            p3, p4, p5 = features
            p3 = p3.to(self.device, non_blocking=True)
            p4 = p4.to(self.device, non_blocking=True)
            p5 = p5.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            seq_lens = seq_lens.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Chạy forward pass với Automatic Mixed Precision (AMP)
            with self._autocast_context():
                logits = self.model((p3, p4, p5), seq_lens=seq_lens)
                loss = self.criterion(logits, targets)

            # Backward pass có điều chỉnh tỷ lệ qua GradScaler
            self.scaler.scale(loss).backward()

            # Unscale trước khi cắt gradient
            self.scaler.unscale_(self.optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            grad_norms.append(float(gnorm.item()))

            # Cập nhật trọng số
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Tính toán xác suất và cập nhật metrics
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                self.metrics_tracker.update(preds, targets, probs, float(loss.item()), p3.size(0))

            temp_acc = (preds == targets).float().mean().item()

            if use_tqdm:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{self.metrics_tracker.running_acc:.1f}%",
                    gnorm=f"{gnorm.item():.3f}",
                    lr=f"{current_lr:.2e}"
                )
            else:
                if batch_idx % self.config.log_interval == 0 or batch_idx == total_batches:
                    self.logger.info(
                        f"  [Train] Epoch {epoch:02d}/{self.config.epochs:02d} | "
                        f"Batch {batch_idx:03d}/{total_batches:03d} | "
                        f"Loss: {loss.item():.4f} | Acc: {temp_acc * 100:.1f}% | "
                        f"GradNorm: {gnorm.item():.3f} | LR: {current_lr:.6f}"
                    )

        pbar.close()
        train_metrics = self.metrics_tracker.compute()
        avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
        return train_metrics, avg_grad_norm

    def validate(self, epoch: int) -> Tuple[Dict[str, float], np.ndarray, List[int], List[float], float]:
        """
        Thực hiện đánh giá trên tập kiểm định HDF5 (HDF5FeatureDataset) với giám sát qua tqdm.
        Returns:
            (val_metrics, confusion_matrix, targets, probs_drowsy, avg_latency_ms)
        """
        self.model.eval()
        self.metrics_tracker.reset()

        # Giải phóng cache GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        val_pbar.close()
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

            # 1. Chạy Pha Huấn Luyện (Train Pass)
            train_metrics, grad_norm = self.train_one_epoch(epoch)

            # 2. Chạy Pha Kiểm Định (Validation Pass) nếu đến chu kỳ
            val_metrics = None
            avg_latency = 0.0
            if epoch % self.config.val_interval_epochs == 0 or epoch == self.config.epochs:
                val_metrics, cm, targets, probs, avg_latency = self.validate(epoch)

            # 3. Cập nhật Scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics is not None:
                        monitor_val = val_metrics.get(self.config.monitor_metric, 0.0)
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

            # 5. In Bảng Tổng Kết Epoch & Chẩn đoán Overfitting
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
            self.logger.info(log_str)

            # Chẩn đoán Overfitting nếu Gap Loss tăng quá cao
            if val_metrics is not None and log_entry["gap_loss"] > 0.5:
                self.logger.warning(
                    f"  [!] CẢNH BÁO OVERFITTING: Khoảng cách tổng quát hóa (Gap Loss = {log_entry['gap_loss']:.4f}) "
                    f"vượt ngưỡng 0.50! Val Loss đang cao hơn đáng kể so với Train Loss."
                )

            # 6. Kiểm tra và Lưu Checkpoint / Early Stopping
            if val_metrics is not None:
                should_stop = self.checkpoint_manager.step(
                    current_metrics=val_metrics,
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler
                )

                if self.checkpoint_manager.best_epoch == epoch:
                    best_val_metrics = val_metrics
                    best_cm = cm
                    best_targets = targets
                    best_probs = probs

                if should_stop:
                    stopping_reason = f"Early Stopping triggered at epoch {epoch} (patience={self.config.patience})"
                    break

            self.logger.info("-" * 80)

        total_train_time = time.time() - total_train_start
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"   KẾT THÚC HUẤN LUYỆN! Tổng thời gian: {total_train_time / 60.0:.2f} phút.")
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
        Nạp lại Checkpoint tốt nhất (best_model.pt) và thực hiện đánh giá chuyên sâu
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
        """Đóng an toàn các tài nguyên mở (HDF5 datasets, Visualizer)."""
        if hasattr(self, "train_loader") and hasattr(self.train_loader, "dataset"):
            if hasattr(self.train_loader.dataset, "close"):
                self.train_loader.dataset.close()
        if hasattr(self, "val_loader") and hasattr(self.val_loader, "dataset"):
            if hasattr(self.val_loader.dataset, "close"):
                self.val_loader.dataset.close()
        if hasattr(self, "visualizer") and hasattr(self.visualizer, "close"):
            self.visualizer.close()


# ==============================================================================
# 6. CƠ CHẾ KIỂM THỬ ĐỘC LẬP TỰ ĐỘNG: run_dry_run_test
# ==============================================================================
def run_dry_run_test() -> None:
    """
    Chạy thử nghiệm độc lập toàn diện quy trình huấn luyện và đánh giá
    trên dữ liệu giả lập (mock data HDF5), xác nhận 100% các thành phần hoạt động trơn tru.
    """
    print("\n" + "=" * 80)
    print("   [DRY-RUN TEST] BẮT ĐẦU KIỂM THỬ TỰ ĐỘNG TOÀN DIỆN PIPELINE HUẤN LUYỆN")
    print("=" * 80)

    temp_dir = tempfile.mkdtemp(prefix="driver_guardian_train_dryrun_")
    temp_path = Path(temp_dir)

    try:
        # 1. Tạo tệp HDF5 mock chứa đồng thời cả split train và val
        mock_h5 = temp_path / "mock_features.h5"
        mock_csv = temp_path / "mock_manifest.csv"
        print(f"[*] [Bước 1] Sinh dữ liệu mock HDF5 cho cả tập train và val: {mock_h5.name}")
        create_mock_h5_dataset(mock_h5, mock_csv)

        # 2. Tạo cấu hình TrainConfig thử nghiệm
        print("[*] [Bước 2] Cấu hình TrainConfig độc lập cho chế độ Dry-Run...")
        cfg = TrainConfig(
            train_h5=str(mock_h5),
            val_h5=str(mock_h5),
            train_manifest_csv=str(mock_csv),
            val_manifest_csv=str(mock_csv),
            checkpoint_dir=str(temp_path / "checkpoints"),
            log_dir=str(temp_path / "logs"),
            tb_log_dir=str(temp_path / "tb_logs"),
            history_csv_path=str(temp_path / "logs" / "history.csv"),
            summary_json_path=str(temp_path / "logs" / "summary.json"),
            plot_curves_path=str(temp_path / "logs" / "curves.png"),
            plot_cm_path=str(temp_path / "logs" / "cm.png"),
            plot_roc_path=str(temp_path / "logs" / "roc.png"),
            epochs=2,
            batch_size=2,
            val_batch_size=2,
            num_workers=0,
            val_num_workers=0,
            patience=2,
            val_interval_epochs=1,
            sample_interval=0.1,
            seq_len=20,
            input_dim=128,
            hidden_dim=96,
            num_layers=2,
            use_tqdm=True,
            experiment_name="dryrun_test",
            dry_run=True
        )

        # 3. Khởi tạo Trainer và chạy huấn luyện
        print("[*] [Bước 3] Khởi tạo DrowsinessTrainer và thực thi 2 epochs...")
        trainer = DrowsinessTrainer(config=cfg)
        fit_results = trainer.fit()

        # 4. Chạy đánh giá cuối cùng
        print("[*] [Bước 4] Thực thi đánh giá chuyên sâu evaluate_final()...")
        summary_payload = trainer.evaluate_final(fit_results)

        # 5. Kiểm tra tính toàn vẹn của các file kết quả
        print("\n[*] [Bước 5] Xác thực các tệp xuất xưởng (Output Artifacts):")
        assert Path(cfg.history_csv_path).exists(), "Thiếu file training_history.csv!"
        assert Path(cfg.summary_json_path).exists(), "Thiếu file training_summary.json!"
        assert Path(cfg.plot_curves_path).exists(), "Thiếu file loss_accuracy_curves.png!"
        assert Path(cfg.plot_cm_path).exists(), "Thiếu file confusion_matrix_best.png!"
        print(f"    [✓] Đã tạo thành công: {cfg.history_csv_path}")
        print(f"    [✓] Đã tạo thành công: {cfg.summary_json_path}")
        print(f"    [✓] Đã tạo thành công: {cfg.plot_curves_path}")
        print(f"    [✓] Đã tạo thành công: {cfg.plot_cm_path}")

        print("\n" + "=" * 80)
        print("   >>> [THÀNH CÔNG 100%] KIỂM THỬ DRY-RUN HOÀN TOÀN ĐẠT CHUẨN CHẤT LƯỢNG! <<<")
        print("=" * 80)

    finally:
        # Dọn dẹp an toàn tài nguyên trainer trước khi xóa thư mục tạm
        if "trainer" in locals():
            try:
                trainer.close()
            except Exception:
                pass
        # Dọn dẹp sạch sẽ thư mục tạm
        if temp_path.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"[*] Đã dọn dẹp an toàn thư mục tạm: {temp_dir}")


# ==============================================================================
# 7. HÀM FACTORY TIỆN ÍCH CHO EXTERNAL IMPORT & ENTRY POINT
# ==============================================================================
def train_pipeline(config: Optional[TrainConfig] = None) -> Dict[str, Any]:
    """
    Hàm giao diện tiện ích cấp cao để khởi chạy toàn bộ pipeline huấn luyện.
    Cho phép gọi trực tiếp từ script Python khác hoặc Jupyter Notebooks.
    """
    cfg = config if config is not None else load_config()
    if cfg.dry_run:
        run_dry_run_test()
        return {"status": "dry_run_completed"}

    trainer = DrowsinessTrainer(config=cfg)
    fit_results = trainer.fit()
    summary = trainer.evaluate_final(fit_results)
    return summary


def main() -> None:
    """Điểm khởi chạy chính khi thực thi tệp: python train.py."""
    config = load_config()
    if config.dry_run:
        run_dry_run_test()
    else:
        trainer = DrowsinessTrainer(config=config)
        fit_results = trainer.fit()
        trainer.evaluate_final(fit_results)


if __name__ == "__main__":
    main()
