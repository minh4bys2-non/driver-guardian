# File: train.py
"""
Kịch bản huấn luyện mô hình ST-GCN Giám sát Tài xế (Driver Guardian).
Tích hợp trực tiếp config.py, dataset.py và model.py:
- Hàm mất mát: CrossEntropyLoss
- Theo dõi chỉ số qua TensorBoard: Loss, Accuracy, Recall, F1, F2 cho cả Train & Validation
- Quản lý và lưu checkpoint (best.pt, last.pt) dựa trên chỉ số Validation F1-score
- Cấu hình thuần túy qua đối tượng TrainConfig (không dùng CLI/argparse).
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
# 1. TIỆN ÍCH HỆ THỐNG: SEED, LOGGING & CHECKPOINT
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


# =====================================================================
# 2. KHỞI TẠO MÔ HÌNH VÀ BỘ ĐIỀU TỐC LEARNING RATE
# =====================================================================
def build_model(cfg: TrainConfig) -> nn.Module:
    """Khởi tạo mô hình dựa trên TrainConfig."""
    model_name = cfg.model_name.lower().strip()
    if model_name in ("stgcn", "driverstgcn", "light"):
        logger.info("Khoi tao kien truc DriverSTGCN (Spatio-Temporal Graph CNN)...")
        model = DriverSTGCN(**cfg.get_stgcn_kwargs())
    elif model_name in ("transformer", "driverstgcntransformer", "trans"):
        logger.info("Khoi tao kien truc DriverSTGCNTransformer (STGCN + TransformerEncoder)...")
        model = DriverSTGCNTransformer(**cfg.get_transformer_kwargs())
    else:
        raise ValueError(f"Khong ho tro model_name='{cfg.model_name}'. Chon 'stgcn' hoac 'transformer'.")
    return model


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


# =====================================================================
# 3. TÍNH TOÁN CHỈ SỐ: LOSS, ACCURACY, RECALL, F1, F2
# =====================================================================
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
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        precision = float(tp / max(1, tp + fp))
        recall = float(tp / max(1, tp + fn))
        f1 = float((2 * tp) / max(1, 2 * tp + fp + fn))
        # F2-score: beta=2 -> uu tien Recall cao hon Precision (giam bo sot tai xe ngu gat)
        f2 = float((5 * tp) / max(1, 5 * tp + 4 * fn + fp))
    else:
        # Macro average cho đa lớp
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


# =====================================================================
# 4. THỰC THI 1 EPOCH (TRAIN HOẶC EVALUATION)
# =====================================================================
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
        # Nhận tuple 4 phần tử từ collate_gcn: (padded_landmarks, labels, lengths, metas)
        landmarks, labels, lengths = batch[0], batch[1], batch[2]
        landmarks = landmarks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        count = labels.size(0)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(landmarks, lengths=lengths)
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


# =====================================================================
# 5. GHI LOG TENSORBOARD ĐỒNG BỘ
# =====================================================================
def log_epoch_to_tensorboard(
        writer: SummaryWriter,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        current_lr: float,
        epoch: int,
) -> None:
    """Ghi nhận chi tiết các scalar Loss, Accuracy, Recall, F1, F2 cho cả Train & Val."""
    # 1. Các nhánh độc lập
    for metric_name in ("loss", "accuracy", "recall", "f1", "f2", "precision"):
        if metric_name in train_metrics:
            writer.add_scalar(f"train/{metric_name}", train_metrics[metric_name], epoch)
        if metric_name in val_metrics:
            writer.add_scalar(f"val/{metric_name}", val_metrics[metric_name], epoch)

    writer.add_scalar("train/learning_rate", current_lr, epoch)

    # 2. Đồ thị so sánh đối chiếu trực tiếp giữa Train và Validation
    for metric_name in ("loss", "accuracy", "recall", "f1", "f2"):
        writer.add_scalars(
            f"compare/{metric_name}",
            {"train": train_metrics.get(metric_name, 0.0), "val": val_metrics.get(metric_name, 0.0)},
            epoch,
        )
    writer.flush()


# =====================================================================
# 6. HÀM CHÍNH: QUẢN LÝ HUẤN LUYỆN & LƯU CHECKPOINT THEO F1
# =====================================================================
def train(cfg: Optional[TrainConfig] = None) -> nn.Module:
    """
    Hàm thực thi huấn luyện toàn bộ pipeline.
    Nhận đối tượng TrainConfig trực tiếp từ mã nguồn hoặc khởi tạo mặc định.
    """
    if cfg is None:
        cfg = TrainConfig()
    cfg.__post_init__()

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(save_dir / "train.log")

    log_dir = Path(cfg.log_dir) if cfg.log_dir else (save_dir / "tensorboard")
    log_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    logger.info("=" * 70)
    logger.info("BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH DRIVER GUARDIAN GCN")
    logger.info("=" * 70)
    logger.info(f"Thiết bị sử dụng (Device) : {device}")
    logger.info(f"Kiến trúc mô hình (Model) : {cfg.model_name}")
    logger.info(f"Số lớp phân loại (Classes): {cfg.num_classes} ({cfg.class_names})")
    logger.info(f"Thư mục lưu Checkpoint     : {save_dir}")
    logger.info(f"Thư mục log TensorBoard   : {log_dir}")

    # Lưu lại file cấu hình json tại thư mục checkpoint
    cfg.save(save_dir / "config.json")

    # 1. Khởi tạo DataLoaders
    train_loader, val_loader, test_loader = build_gcn_dataloaders(**cfg.get_dataloader_kwargs())
    logger.info(
        f"Số lượng mẫu Dataset      : Train={len(train_loader.dataset)} | Val={len(val_loader.dataset)} | Test={len(test_loader.dataset)}")

    # 2. Khởi tạo Mô hình & Hàm Mất mát CrossEntropyLoss
    model = build_model(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Tổng số tham số học được  : {num_params:,} parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = get_cosine_warmup_scheduler(
        optimizer,
        total_epochs=cfg.epochs,
        warmup_epochs=cfg.warmup_epochs,
        min_lr_ratio=cfg.min_lr_ratio,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    # 3. Quản lý khôi phục (Resume) nếu có
    start_epoch = 1
    best_epoch = 0
    best_metrics: Optional[Dict[str, float]] = None

    if cfg.resume:
        resume_path = Path(cfg.resume)
        if resume_path.is_file():
            logger.info(f"Nạp lại checkpoint từ: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "scaler_state_dict" in checkpoint and scaler is not None and checkpoint["scaler_state_dict"] is not None:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_epoch = checkpoint.get("best_epoch", 0)
            best_metrics = checkpoint.get("best_metrics", None)
            logger.info(f"Đã khôi phục thành công! Tiếp tục từ Epoch {start_epoch} (Best Epoch trước đó: {best_epoch})")

    # 4. Vòng lặp huấn luyện từng Epoch & Đánh giá Checkpoint theo F1
    writer = SummaryWriter(str(log_dir))
    writer.add_text("Config", f"```json\n{json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False)}\n```", 0)

    start_train_time = time.time()

    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            epoch_start = time.time()
            current_lr = float(optimizer.param_groups[0]["lr"])

            # Huấn luyện trên tập Train
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                amp=cfg.amp,
                max_grad_norm=cfg.max_grad_norm,
                desc=f"Epoch {epoch:02d}/{cfg.epochs:02d} [Train]",
                num_classes=cfg.num_classes,
            )

            # Đánh giá trên tập Validation
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                scaler=None,
                amp=cfg.amp,
                max_grad_norm=cfg.max_grad_norm,
                desc=f"Epoch {epoch:02d}/{cfg.epochs:02d} [Val]",
                num_classes=cfg.num_classes,
            )

            scheduler.step()
            epoch_duration = time.time() - epoch_start

            # Ghi TensorBoard
            log_epoch_to_tensorboard(writer, train_metrics, val_metrics, current_lr, epoch)

            # Đánh giá tiêu chí lưu Checkpoint tốt nhất: DỰA TRÊN CHỈ SỐ F1 CỦA TẬP VALIDATION
            is_best = False
            if best_metrics is None:
                is_best = True
            else:
                # Ưu tiên F1 cao hơn; nếu F1 bằng nhau xét tiếp F2 và sau đó là Loss thấp hơn
                val_score = (val_metrics["f1"], val_metrics["f2"], -val_metrics["loss"])
                best_score = (best_metrics["f1"], best_metrics["f2"], -best_metrics["loss"])
                if val_score > best_score:
                    is_best = True

            if is_best:
                best_epoch = epoch
                best_metrics = val_metrics
                best_state = {
                    "format_version": 1,
                    "epoch": best_epoch,
                    "model_name": cfg.model_name,
                    "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "val_metrics": best_metrics,
                    "config": cfg.to_dict(),
                }
                atomic_save(best_state, save_dir / "best.pt")
                with open(save_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                    json.dump(best_metrics, f, indent=4, ensure_ascii=False)

            # Lưu checkpoint cuối cùng sau mỗi epoch để hỗ trợ resume
            last_state = {
                "format_version": 1,
                "epoch": epoch,
                "model_name": cfg.model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "config": cfg.to_dict(),
            }
            atomic_save(last_state, save_dir / "last.pt")

            # In thông tin tổng kết Epoch
            best_tag = " >>> [BEST CHECKPOINT SAVED] <<<" if is_best else ""
            logger.info(
                f"Epoch {epoch:02d}/{cfg.epochs:02d} ({epoch_duration:.1f}s) | LR: {current_lr:.6g}\n"
                f"  - Train: Loss={train_metrics['loss']:.4f} | Acc={train_metrics['accuracy']:.4f} | Recall={train_metrics['recall']:.4f} | F1={train_metrics['f1']:.4f} | F2={train_metrics['f2']:.4f}\n"
                f"  - Val  : Loss={val_metrics['loss']:.4f} | Acc={val_metrics['accuracy']:.4f} | Recall={val_metrics['recall']:.4f} | F1={val_metrics['f1']:.4f} | F2={val_metrics['f2']:.4f}{best_tag}"
            )

    except KeyboardInterrupt:
        logger.warning("\nQuá trình huấn luyện bị ngắt bởi người dùng. Checkpoint 'last.pt' đã được bảo lưu an toàn.")
    finally:
        total_time = time.time() - start_train_time
        logger.info(f"\nHoàn tất huấn luyện sau {total_time / 60:.2f} phút.")
        if best_metrics:
            logger.info(f"Checkpoint tốt nhất tại Epoch {best_epoch}:")
            logger.info(f"  - Val F1      : {best_metrics['f1']:.4f}")
            logger.info(f"  - Val F2      : {best_metrics['f2']:.4f}")
            logger.info(f"  - Val Recall  : {best_metrics['recall']:.4f}")
            logger.info(f"  - Val Accuracy: {best_metrics['accuracy']:.4f}")
            logger.info(f"  - Val Loss    : {best_metrics['loss']:.4f}")

        # 5. Đánh giá kiểm thử cuối cùng trên tập Test bằng model tốt nhất
        best_pt_path = save_dir / "best.pt"
        if best_pt_path.is_file() and len(test_loader.dataset) > 0:
            logger.info("\nĐang tải best.pt để đánh giá nghiệm thu trên tập Test...")
            checkpoint = torch.load(best_pt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            test_metrics = run_epoch(
                model=model,
                loader=test_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                scaler=None,
                amp=cfg.amp,
                desc="Final Test",
                num_classes=cfg.num_classes,
            )
            for m_name, m_val in test_metrics.items():
                writer.add_scalar(f"test/{m_name}", m_val, best_epoch)
            writer.flush()
            with open(save_dir / "test_metrics.json", "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, indent=4, ensure_ascii=False)
            logger.info(
                f"Kết quả Test (Best Epoch {best_epoch}): "
                f"Loss={test_metrics['loss']:.4f} | Acc={test_metrics['accuracy']:.4f} | "
                f"Recall={test_metrics['recall']:.4f} | F1={test_metrics['f1']:.4f} | F2={test_metrics['f2']:.4f}"
            )

        writer.close()

    return model


# =====================================================================
# 7. KHỞI CHẠY TRỰC TIẾP (THUẦN PYTHON, KHÔNG DÙNG CLI)
# =====================================================================
if __name__ == "__main__":
    # Cấu hình trực tiếp các tham số huấn luyện qua lớp TrainConfig tại đây:
    config = TrainConfig()

    # Bắt đầu quá trình huấn luyện
    train(config)
