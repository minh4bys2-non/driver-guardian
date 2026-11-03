# File: train.py
"""
Kịch bản huấn luyện hướng đối tượng (OOP) mô hình ST-GCN Giám sát Tài xế (Driver Guardian).
Tích hợp trực tiếp config.py, dataset.py và model.py:
- Lớp điều phối chính: GCNTrainer (cùng các hàm wrapper tương thích ngược: train, run_epoch, build_model, ...)
- Hàm mất mát: CrossEntropyLoss (hỗ trợ Label Smoothing trên tập Train, CrossEntropy thuần trên Val/Test)
- Theo dõi chỉ số qua TensorBoard: Loss, Accuracy, Recall, F1, F2, Precision cho cả Train & Validation
- Quản lý và lưu checkpoint nguyên tử (epoch_XX.pt, best.pt, last.pt) dựa trên chỉ số Validation F1-score
- Cấu hình thuần túy qua đối tượng TrainConfig.
"""

from __future__ import annotations
import json
import logging
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional

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

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from config import TrainConfig
from dataset import build_gcn_dataloaders
from model import DriverSTGCN, DriverSTGCNTransformer

logger = logging.getLogger("DriverGuardian.GCN")


# =====================================================================
# 1. TIỆN ÍCH HỆ THỐNG & HÀM ĐỘC LẬP (BACKWARD-COMPATIBLE HELPERS)
# =====================================================================
def seed_everything(seed: int = 42) -> None:
    """Cố định seed để đảm bảo khả năng tái lập kết quả thực nghiệm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_logging(log_file: Path | str) -> None:
    """Cấu hình logging đồng thời ra Console và file train.log."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

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


def atomic_save(state: Any, path: str | Path) -> None:
    """Lưu tệp an toàn (Atomic Save qua file tạm) để tránh hỏng checkpoint khi bị ngắt đột ngột."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_file = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, path)
    finally:
        Path(tmp_file).unlink(missing_ok=True)


def build_model(cfg: TrainConfig) -> nn.Module:
    """Khởi tạo mô hình dựa trên TrainConfig."""
    model_name = cfg.model_name.lower().strip()
    if model_name in ("stgcn", "driverstgcn", "light"):
        logger.info("Khởi tạo kiến trúc DriverSTGCN (ST-GCN + AttentiveDrowsinessOutputHead)...")
        return DriverSTGCN(**cfg.get_stgcn_kwargs())
    elif model_name in ("transformer", "driverstgcntransformer", "trans"):
        logger.info("Khởi tạo kiến trúc DriverSTGCNTransformer (STGCN + TransformerEncoder)...")
        return DriverSTGCNTransformer(**cfg.get_transformer_kwargs())
    else:
        raise ValueError(f"Không hỗ trợ model_name='{cfg.model_name}'. Chọn 'stgcn' hoặc 'transformer'.")


def get_cosine_warmup_scheduler(
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        warmup_epochs: int = 5,
        min_lr_ratio: float = 0.05,
) -> LambdaLR:
    """Tạo bộ điều tốc Cosine Annealing kèm giai đoạn Linear Warmup."""
    warmup_epochs = max(0, min(warmup_epochs, total_epochs - 1))

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs - 1))
        progress = min(1.0, max(0.0, progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        avg_loss: float,
        num_classes: int = 2,
) -> Dict[str, float]:
    """
    Tính toán chi tiết các chỉ số phân loại: Loss, Accuracy, Recall, F1-score, F2-score, Precision.
    - Với bài toán 2 lớp (0: Alert, 1: Drowsy): tập trung vào nhãn dương (1: Drowsy).
    - Với bài toán đa lớp (> 2 lớp): tính trung bình Macro trên tất cả các lớp.
    """
    total = len(y_true)
    if total == 0:
        return {
            "loss": float(avg_loss),
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "f2": 0.0,
            "n_samples": 0,
        }

    accuracy = float((y_true == y_pred).sum() / total)

    if num_classes == 2:
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        precision = float(tp / max(1, tp + fp))
        recall = float(tp / max(1, tp + fn))
        f1 = float((2 * tp) / max(1, 2 * tp + fp + fn))
        f2 = float((5 * tp) / max(1, 5 * tp + 4 * fn + fp))
    else:
        precisions, recalls, f1s, f2s = [], [], [], []
        for c in range(num_classes):
            tp_c = int(((y_pred == c) & (y_true == c)).sum())
            fp_c = int(((y_pred == c) & (y_true != c)).sum())
            fn_c = int(((y_pred != c) & (y_true == c)).sum())

            p_c = tp_c / max(1, tp_c + fp_c)
            r_c = tp_c / max(1, tp_c + fn_c)
            f1_c = (2 * p_c * r_c) / max(1e-12, p_c + r_c)
            f2_c = (5 * p_c * r_c) / max(1e-12, 4 * p_c + r_c)

            precisions.append(p_c)
            recalls.append(r_c)
            f1s.append(f1_c)
            f2s.append(f2_c)

        precision = float(np.mean(precisions))
        recall = float(np.mean(recalls))
        f1 = float(np.mean(f1s))
        f2 = float(np.mean(f2s))

    return {
        "loss": float(avg_loss),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "n_samples": total,
    }


def run_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        amp: bool = True,
        max_grad_norm: float = 1.0,
        desc: str = "Train",
        num_classes: int = 2,
) -> Dict[str, float]:
    """Chạy 1 epoch đầy đủ qua DataLoader và trả về bộ chỉ số tổng hợp."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_preds = []

    progress = tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True, leave=False)

    for batch in progress:
        landmarks, labels, lengths = batch[0], batch[1], batch[2]
        landmarks = landmarks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        count = labels.size(0)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                outputs = model(landmarks, lengths=lengths)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Gặp giá trị Loss không hợp lệ (NaN/Inf) tại {desc}")

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    optimizer.step()

        preds = logits.detach().argmax(dim=1)
        total_loss += loss.detach().item() * count
        total_samples += count

        all_targets.append(labels.detach().cpu().numpy())
        all_preds.append(preds.cpu().numpy())

        current_loss = total_loss / max(1, total_samples)
        progress.set_postfix(loss=f"{current_loss:.4f}")

    if total_samples == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "f2": 0.0,
            "n_samples": 0,
        }

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    avg_loss = total_loss / total_samples

    return compute_metrics(y_true, y_pred, avg_loss, num_classes=num_classes)


def log_epoch_to_tensorboard(
        writer: SummaryWriter,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        current_lr: float,
        epoch: int,
) -> None:
    """Ghi nhận chi tiết các scalar Loss, Accuracy, Recall, F1, F2 cho cả Train & Val."""
    for metric_name in ("loss", "accuracy", "recall", "f1", "f2", "precision"):
        if metric_name in train_metrics:
            writer.add_scalar(f"train/{metric_name}", train_metrics[metric_name], epoch)
        if metric_name in val_metrics:
            writer.add_scalar(f"val/{metric_name}", val_metrics[metric_name], epoch)

    writer.add_scalar("train/learning_rate", current_lr, epoch)

    for metric_name in ("loss", "accuracy", "recall", "f1", "f2"):
        writer.add_scalars(
            f"compare/{metric_name}",
            {"train": train_metrics.get(metric_name, 0.0), "val": val_metrics.get(metric_name, 0.0)},
            epoch,
        )
    writer.flush()


# =====================================================================
# 2. LỚP QUẢN LÝ HUẤN LUYỆN HƯỚNG ĐỐI TƯỢNG (OOP GCN TRAINER)
# =====================================================================
class GCNTrainer:
    """
    Lớp Trainer hướng đối tượng (OOP) quản lý toàn bộ vòng đời huấn luyện mô hình Driver Guardian GCN:
    1. Khởi tạo và đồng bộ siêu tham số cấu hình (TrainConfig).
    2. Xây dựng DataLoaders (Train, Val, Test) tích hợp Cache .pt & InsightFace 68 3D Landmarks.
    3. Khởi tạo kiến trúc mô hình (DriverSTGCN / DriverSTGCNTransformer), Hàm mất mát, Optimizer, Scheduler và Scaler (AMP).
    4. Điều phối vòng lặp huấn luyện từng epoch, tính toán chỉ số chi tiết (Loss, Accuracy, Recall, F1, F2).
    5. Quản lý lưu trữ checkpoint nguyên tử (Atomic Save) cho từng epoch (epoch_XX.pt), best.pt và last.pt.
    6. Ghi nhận log trực quan theo thời gian thực ra Console, train.log và TensorBoard.
    7. Tự động đánh giá kiểm thử nghiệm thu trên tập Test bằng checkpoint tốt nhất sau khi kết thúc.
    """

    seed_everything = staticmethod(seed_everything)
    configure_logging = staticmethod(configure_logging)
    atomic_save = staticmethod(atomic_save)
    compute_metrics = staticmethod(compute_metrics)

    def __init__(self, cfg: Optional[TrainConfig] = None):
        self.cfg = cfg if cfg is not None else TrainConfig()
        self.cfg.__post_init__()

        self.save_dir = Path(self.cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(self.cfg.log_dir) if self.cfg.log_dir else (self.save_dir / "tensorboard")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.configure_logging(self.save_dir / "train.log")
        self.seed_everything(self.cfg.seed)

        self.device = torch.device(self.cfg.device if torch.cuda.is_available() or self.cfg.device == "cpu" else "cpu")
        self.writer: Optional[SummaryWriter] = None

        self.cfg.save(self.save_dir / "config.json")
        self._setup_components()

    def _setup_components(self) -> None:
        """Khởi tạo DataLoaders, Mô hình, Hàm mất mát, Optimizer, Scheduler và Scaler."""
        logger.info("=" * 70)
        logger.info("KHỞI TẠO PIPELINE HUẤN LUYỆN GCNTRAINER (OOP) - DRIVER GUARDIAN GCN")
        logger.info("=" * 70)
        logger.info(f"Thiết bị sử dụng (Device)      : {self.device}")
        logger.info(f"Kiến trúc mô hình (Model)      : {self.cfg.model_name}")
        logger.info(f"Số lớp phân loại (Classes)     : {self.cfg.num_classes} ({self.cfg.class_names})")
        logger.info(f"Mô hình Landmark (InsightFace) : {self.cfg.landmark_model_name} (Det: {self.cfg.det_size})")
        logger.info(f"Sử dụng Cache Landmark (.pt)   : {self.cfg.use_cache} (Dir: {self.cfg.cache_dir})")
        logger.info(f"Hệ số Label Smoothing          : {self.cfg.label_smoothing}")
        logger.info(f"Thư mục lưu Checkpoint         : {self.save_dir}")
        logger.info(f"Thư mục log TensorBoard        : {self.log_dir}")

        self.train_loader, self.val_loader, self.test_loader = self._build_dataloaders()
        logger.info(
            f"Số lượng mẫu Dataset           : Train={len(self.train_loader.dataset)} | "
            f"Val={len(self.val_loader.dataset)} | Test={len(self.test_loader.dataset)}"
        )

        self.model = self._build_model().to(self.device)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Tổng số tham số học được       : {num_params:,} parameters")

        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smoothing)
        self.val_criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        self.scheduler = self._build_scheduler()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.amp and self.device.type == "cuda")

        self.start_epoch = 1
        self.best_epoch = 0
        self.best_metrics: Optional[Dict[str, float]] = None

        if self.cfg.resume:
            self.load_checkpoint(self.cfg.resume)

    def _build_dataloaders(self):
        return build_gcn_dataloaders(**self.cfg.get_dataloader_kwargs())

    def _build_model(self) -> nn.Module:
        return build_model(self.cfg)

    def _build_scheduler(self) -> LambdaLR:
        return get_cosine_warmup_scheduler(
            self.optimizer,
            total_epochs=self.cfg.epochs,
            warmup_epochs=self.cfg.warmup_epochs,
            min_lr_ratio=self.cfg.min_lr_ratio,
        )

    def load_checkpoint(self, resume_path: str | Path) -> None:
        """Khôi phục trạng thái mô hình và optimizer từ checkpoint đã lưu."""
        resume_path = Path(resume_path)
        if not resume_path.is_file():
            logger.warning(f"Không tìm thấy file checkpoint resume tại: {resume_path}. Bỏ qua nạp lại.")
            return

        logger.info(f"Nạp lại checkpoint từ: {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(resume_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint and self.scaler is not None and checkpoint["scaler_state_dict"] is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.best_metrics = checkpoint.get("best_metrics", None)
        logger.info(f"Đã khôi phục thành công! Tiếp tục từ Epoch {self.start_epoch} (Best Epoch trước đó: {self.best_epoch})")

    def save_checkpoints(
            self,
            epoch: int,
            train_metrics: Dict[str, float],
            val_metrics: Dict[str, float],
            is_best: bool,
    ) -> None:
        """Lưu trữ an toàn mọi checkpoint của từng epoch (epoch_XX.pt), best.pt và last.pt."""
        checkpoint_state = {
            "format_version": 1,
            "epoch": epoch,
            "model_name": self.cfg.model_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_epoch": self.best_epoch if not is_best else epoch,
            "best_metrics": self.best_metrics if not is_best else val_metrics,
            "config": self.cfg.to_dict(),
        }

        epoch_ckpt_path = self.save_dir / f"epoch_{epoch:02d}.pt"
        self.atomic_save(checkpoint_state, epoch_ckpt_path)
        self.atomic_save(checkpoint_state, self.save_dir / "last.pt")

        if is_best:
            self.best_epoch = epoch
            self.best_metrics = val_metrics
            best_state = {
                "format_version": 1,
                "epoch": self.best_epoch,
                "model_name": self.cfg.model_name,
                "model_state_dict": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                "train_metrics": train_metrics,
                "val_metrics": self.best_metrics,
                "config": self.cfg.to_dict(),
            }
            self.atomic_save(best_state, self.save_dir / "best.pt")
            with open(self.save_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                json.dump(self.best_metrics, f, indent=4, ensure_ascii=False)

    def log_epoch_to_tensorboard(
            self,
            train_metrics: Dict[str, float],
            val_metrics: Dict[str, float],
            current_lr: float,
            epoch: int,
    ) -> None:
        if self.writer is None:
            return
        log_epoch_to_tensorboard(self.writer, train_metrics, val_metrics, current_lr, epoch)

    def run_epoch(
            self,
            loader: DataLoader,
            training: bool = True,
            desc: str = "Train",
    ) -> Dict[str, float]:
        """Chạy 1 epoch đầy đủ qua DataLoader (dùng cho cả Train, Val, Test)."""
        return run_epoch(
            model=self.model,
            loader=loader,
            criterion=self.criterion if training else self.val_criterion,
            device=self.device,
            optimizer=self.optimizer if training else None,
            scaler=self.scaler if training else None,
            amp=self.cfg.amp,
            max_grad_norm=self.cfg.max_grad_norm,
            desc=desc,
            num_classes=self.cfg.num_classes,
        )

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        if hasattr(self.model, "output_head") and hasattr(self.model.output_head, "grl"):
            progress = float(epoch - 1) / float(max(1, self.cfg.epochs - 1))
            self.model.output_head.grl.set_lambda(progress)

        return self.run_epoch(
            loader=self.train_loader,
            training=True,
            desc=f"Epoch {epoch:02d}/{self.cfg.epochs:02d} [Train]",
        )

    def validate(self, epoch: int) -> Dict[str, float]:
        return self.run_epoch(
            loader=self.val_loader,
            training=False,
            desc=f"Epoch {epoch:02d}/{self.cfg.epochs:02d} [Val]",
        )

    def evaluate_test(self) -> Optional[Dict[str, float]]:
        """Đánh giá kiểm thử nghiệm thu trên tập Test bằng checkpoint tốt nhất (best.pt)."""
        best_pt_path = self.save_dir / "best.pt"
        if not best_pt_path.is_file() or len(self.test_loader.dataset) == 0:
            return None

        logger.info("\nĐang tải best.pt để đánh giá nghiệm thu trên tập Test...")
        try:
            checkpoint = torch.load(best_pt_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(best_pt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics = self.run_epoch(
            loader=self.test_loader,
            training=False,
            desc="Final Test",
        )

        if self.writer is not None:
            for m_name, m_val in test_metrics.items():
                self.writer.add_scalar(f"test/{m_name}", m_val, self.best_epoch)
            self.writer.flush()

        with open(self.save_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=4, ensure_ascii=False)

        logger.info(
            f"Kết quả Test (Best Epoch {self.best_epoch}): "
            f"Loss={test_metrics['loss']:.4f} | Acc={test_metrics['accuracy']:.4f} | "
            f"Recall={test_metrics['recall']:.4f} | F1={test_metrics['f1']:.4f} | F2={test_metrics['f2']:.4f}"
        )
        return test_metrics

    def fit(self) -> nn.Module:
        """Thực thi toàn bộ quy trình huấn luyện và đánh giá mô hình."""
        self.writer = SummaryWriter(str(self.log_dir))
        self.writer.add_text(
            "Config",
            f"```json\n{json.dumps(self.cfg.to_dict(), indent=2, ensure_ascii=False)}\n```",
            0
        )

        logger.info("=" * 70)
        logger.info("BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN (FIT LOOP)")
        logger.info("=" * 70)

        start_train_time = time.time()

        try:
            for epoch in range(self.start_epoch, self.cfg.epochs + 1):
                epoch_start = time.time()
                current_lr = float(self.optimizer.param_groups[0]["lr"])

                train_metrics = self.train_one_epoch(epoch)
                val_metrics = self.validate(epoch)

                self.scheduler.step()
                epoch_duration = time.time() - epoch_start

                self.log_epoch_to_tensorboard(train_metrics, val_metrics, current_lr, epoch)

                is_best = False
                if self.best_metrics is None:
                    is_best = True
                else:
                    val_score = (val_metrics["f1"], val_metrics["f2"], -val_metrics["loss"])
                    best_score = (self.best_metrics["f1"], self.best_metrics["f2"], -self.best_metrics["loss"])
                    if val_score > best_score:
                        is_best = True

                self.save_checkpoints(epoch, train_metrics, val_metrics, is_best)

                best_tag = " >>> [BEST RECORD UPDATED] <<<" if is_best else ""
                logger.info(
                    f"Epoch {epoch:02d}/{self.cfg.epochs:02d} ({epoch_duration:.1f}s) | LR: {current_lr:.6g} [Checkpoint: epoch_{epoch:02d}.pt]\n"
                    f"  - Train: Loss={train_metrics['loss']:.4f} | Acc={train_metrics['accuracy']:.4f} | Recall={train_metrics['recall']:.4f} | F1={train_metrics['f1']:.4f} | F2={train_metrics['f2']:.4f}\n"
                    f"  - Val  : Loss={val_metrics['loss']:.4f} | Acc={val_metrics['accuracy']:.4f} | Recall={val_metrics['recall']:.4f} | F1={val_metrics['f1']:.4f} | F2={val_metrics['f2']:.4f}{best_tag}"
                )

        except KeyboardInterrupt:
            logger.warning("\nQuá trình huấn luyện bị ngắt bởi người dùng. Toàn bộ checkpoint đã được bảo lưu an toàn.")
        finally:
            total_time = time.time() - start_train_time
            logger.info(f"\nHoàn tất huấn luyện sau {total_time / 60:.2f} phút.")
            if self.best_metrics:
                logger.info(f"Checkpoint tốt nhất tại Epoch {self.best_epoch}:")
                logger.info(f"  - Val F1      : {self.best_metrics['f1']:.4f}")
                logger.info(f"  - Val F2      : {self.best_metrics['f2']:.4f}")
                logger.info(f"  - Val Recall  : {self.best_metrics['recall']:.4f}")
                logger.info(f"  - Val Accuracy: {self.best_metrics['accuracy']:.4f}")
                logger.info(f"  - Val Loss    : {self.best_metrics['loss']:.4f}")

            self.evaluate_test()

            if self.writer is not None:
                self.writer.close()

        return self.model


def train(cfg: Optional[TrainConfig] = None) -> nn.Module:
    """Hàm thực thi huấn luyện toàn bộ pipeline thông qua GCNTrainer."""
    trainer = GCNTrainer(cfg)
    return trainer.fit()


# =====================================================================
# 3. KHỞI CHẠY TRỰC TIẾP (THUẦN PYTHON, KHÔNG DÙNG CLI)
# =====================================================================
if __name__ == "__main__":
    config = TrainConfig()
    train(config)
