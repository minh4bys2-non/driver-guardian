#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Huấn luyện Mô hình SpatioTemporal ConvGRUClassifier (Driver Guardian AI)
Phiên bản: 2.0 (Đã gia cố toàn diện theo tiêu chuẩn Code Review & Quality Gates)

Đặc tính kỹ thuật đã nâng cấp:
1. Chuẩn hóa hàm mất mát (Loss) và nhãn dự đoán cho cả Binary BCE và Multi-class CE,
   triệt tiêu hoàn toàn lỗi không nhận gradient của kênh 0.
2. Tách biệt hoàn toàn việc lưu checkpoint last.pt khỏi điều kiện validation,
   ngăn ngừa mất mát dữ liệu khi val_interval_epochs > 1.
3. Bảo toàn trạng thái AMP GradScaler khi lưu và nạp checkpoint (Resumable AMP).
4. Đồng bộ cơ chế bắt lỗi CUDA Out Of Memory (OOM) cho cả vòng lặp Train và Val,
   giải phóng triệt để traceback frame để tránh OOM lặp lại.
5. Khắc phục sai lệch gradient tích lũy (Gradient Accumulation) ở batch cuối epoch.
6. Tích hợp Early Stopping và tự động ghi nhật ký huấn luyện ra file training_history.csv.
7. Hỗ trợ phân giải bí danh checkpoint ('last', 'best') và cấu hình val_batch_size/val_num_workers riêng.
"""

import os
import sys
import time
import random
import argparse
import csv
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, recall_score, f1_score

# Đảm bảo đường dẫn gốc dự án được khai báo
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import các thành phần dự án
from configs.config import load_config, TrainConfig
from src.models1 import ConvGRUClassifier
from src.dataset2 import build_raw_video_dataloaders, ChunkedBackboneNeckExtractor


def seed_everything(seed: int = 42) -> None:
    """Cố định seed ngẫu nhiên cho toàn bộ các thư viện để đảm bảo tính tái lập (reproducibility)."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Tính toán Accuracy, Recall, F1-Score (macro) an toàn từ mảng NumPy."""
    if len(preds) == 0 or len(targets) == 0:
        return {"acc": 0.0, "recall": 0.0, "f1": 0.0}
    acc = accuracy_score(targets, preds)
    recall = recall_score(targets, preds, average='macro', zero_division=0)
    f1 = f1_score(targets, preds, average='macro', zero_division=0)
    return {"acc": float(acc), "recall": float(recall), "f1": float(f1)}


class EarlyStopping:
    """Bộ giám sát dừng sớm Early Stopping dựa trên metric chỉ định."""
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
        monitor: str = "val_f1"
    ) -> None:
        self.patience = max(1, patience)
        self.min_delta = float(min_delta)
        self.mode = mode.lower().strip()
        self.monitor = monitor
        self.best_score = -float("inf") if self.mode == "max" else float("inf")
        self.counter = 0
        self.best_epoch = 0
        self.early_stop = False

    def step(self, current_val: float, epoch: int) -> bool:
        if self.mode == "max":
            improved = current_val > (self.best_score + self.min_delta)
        else:
            improved = current_val < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_val
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class Trainer:
    """Pipeline Huấn luyện và Kiểm định Chuyên nghiệp cho ConvGRUClassifier."""
    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        print(f"[*] Đang sử dụng thiết bị: {self.device}")

        # Kiểm tra tính tương thích của supervision_mode
        supervision = getattr(self.cfg, "supervision_mode", "attention_pooling")
        if supervision not in ("attention_pooling", "clip"):
            print(f"[CẢNH BÁO] supervision_mode='{supervision}' hiện chưa có nhãn frame-level từ dataset2. "
                  f"Tự động đặt về 'attention_pooling' (Clip-level).")
            self.cfg.supervision_mode = "attention_pooling"

        # Khởi tạo DataLoader
        print("[*] Đang khởi tạo DataLoader từ dataset video thô...")
        val_batch_size = getattr(self.cfg, "val_batch_size", self.cfg.batch_size)
        val_num_workers = getattr(self.cfg, "val_num_workers", self.cfg.num_workers)
        min_frames = getattr(self.cfg, "min_frames", 10)
        img_size = getattr(self.cfg, "image_size", (640, 640))[0]

        self.train_loader, self.val_loader = build_raw_video_dataloaders(
            dataset_dir=self.cfg.dataset_dir,
            manifest_file=self.cfg.manifest_file,
            sample_interval=self.cfg.sample_interval,
            seq_len=self.cfg.seq_len,
            batch_size=self.cfg.batch_size,
            val_batch_size=val_batch_size,
            num_workers=self.cfg.num_workers,
            val_num_workers=val_num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle_train=self.cfg.shuffle,
            use_augmentation=self.cfg.use_augmentation,
            min_frames=min_frames,
            img_size=img_size
        )

        # Kiểm tra tính hợp lệ của tập dữ liệu
        train_samples_count = len(self.train_loader.dataset)
        val_samples_count = len(self.val_loader.dataset)
        print(f"[*] Tập Train: {train_samples_count} video | Tập Val: {val_samples_count} video")
        if train_samples_count == 0:
            raise RuntimeError(f"Tập huấn luyện (train) không có mẫu video nào trong '{self.cfg.dataset_dir}'. Kiểm tra cấu hình dataset!")
        if val_samples_count == 0:
            print("[CẢNH BÁO] Tập kiểm định (val) không có mẫu nào! Quá trình validation sẽ bị bỏ qua.")

        # Khởi tạo mô hình trích xuất đặc trưng BackboneNeck trên GPU
        print("[*] Đang nạp mô hình trích xuất BackboneNeck (NMSFreeDetector)...")
        self.extractor = ChunkedBackboneNeckExtractor(
            checkpoint_path=self.cfg.backbone_neck_checkpoint,
            device=self.cfg.device,
            chunk_size=self.cfg.chunk_size,
            use_fp16=self.cfg.amp,
            img_size=img_size
        ).to(self.device)
        self.extractor.eval()
        for param in self.extractor.parameters():
            param.requires_grad = False

        # Khởi tạo mô hình ConvGRUClassifier chính
        print("[*] Đang nạp ConvGRUClassifier...")
        self.model = ConvGRUClassifier.from_config(self.cfg).to(self.device)

        # Cấu hình hàm mất mát (Loss) và cơ chế tính nhãn dự đoán
        loss_type = self.cfg.loss_type.lower()
        num_classes = getattr(self.cfg, "num_classes", 2)
        if loss_type == "bce":
            if num_classes == 1:
                pos_weight = torch.tensor([self.cfg.pos_weight], device=self.device) if self.cfg.pos_weight else None
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=self.cfg.bce_reduction)
                self.is_binary_bce = True
                print("[*] Sử dụng BCEWithLogitsLoss cho bài toán Binary Classification 1-Logit.")
            else:
                print(f"[CẢNH BÁO] Config yêu cầu loss_type='bce' nhưng num_classes={num_classes} (>1). "
                      f"Tự động chuyển đổi sang CrossEntropyLoss để đảm bảo tính đúng đắn của gradient trên tất cả các lớp.")
                self.criterion = nn.CrossEntropyLoss()
                self.is_binary_bce = False
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.is_binary_bce = False
            print("[*] Sử dụng CrossEntropyLoss chuẩn.")

        # Tối ưu hóa (Optimizer) & Lịch trình (Scheduler)
        if self.cfg.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.cfg.lr0,
                momentum=self.cfg.momentum,
                weight_decay=self.cfg.weight_decay
            )
        else:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.lr0,
                betas=self.cfg.betas,
                weight_decay=self.cfg.weight_decay
            )

        if self.cfg.use_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.epochs,
                eta_min=self.cfg.lr0 * self.cfg.lr_min_factor
            )
        else:
            self.scheduler = None

        # Thư mục TensorBoard và Checkpoint
        log_dir = Path(self.cfg.tb_log_dir) / self.cfg.experiment_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir))
        print(f"[*] TensorBoard log_dir: {log_dir}")

        self.ckpt_dir = Path(self.cfg.checkpoint_dir) / self.cfg.experiment_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Cấu hình Mix Precision (AMP)
        self.scaler = torch.amp.GradScaler("cuda") if (self.cfg.amp and self.device.type == "cuda") else None

        # Cấu hình Early Stopping
        if getattr(self.cfg, "early_stopping", False):
            self.early_stopper = EarlyStopping(
                patience=getattr(self.cfg, "patience", 10),
                min_delta=getattr(self.cfg, "min_delta", 1e-4),
                mode=getattr(self.cfg, "monitor_mode", "max"),
                monitor=getattr(self.cfg, "monitor_metric", "val_f1")
            )
        else:
            self.early_stopper = None

        # Khởi tạo tệp training_history.csv
        history_path = getattr(self.cfg, "history_csv_path", None)
        if history_path:
            self.history_csv_path = Path(history_path)
        else:
            self.history_csv_path = self.ckpt_dir / "training_history.csv"
        self._init_history_csv()

        self.global_step = 0
        self.best_val_f1 = 0.0
        self.start_epoch = 1

    def _init_history_csv(self) -> None:
        """Khởi tạo tiêu đề cho tệp CSV lịch sử huấn luyện."""
        self.history_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_csv_path.exists():
            with open(self.history_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "train_loss", "train_acc", "train_recall", "train_f1",
                    "val_loss", "val_acc", "val_recall", "val_f1", "lr", "time_sec"
                ])

    def _log_history_csv(
        self,
        epoch: int,
        train_m: Dict[str, float],
        val_m: Optional[Dict[str, float]],
        elapsed_sec: float
    ) -> None:
        """Ghi nhận thông số sau mỗi epoch vào tệp CSV."""
        lr = self.optimizer.param_groups[0]['lr']
        val_loss = f"{val_m['loss']:.4f}" if val_m else ""
        val_acc = f"{val_m['acc']:.4f}" if val_m else ""
        val_rec = f"{val_m['recall']:.4f}" if val_m else ""
        val_f1 = f"{val_m['f1']:.4f}" if val_m else ""

        with open(self.history_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_m['loss']:.4f}",
                f"{train_m['acc']:.4f}",
                f"{train_m['recall']:.4f}",
                f"{train_m['f1']:.4f}",
                val_loss,
                val_acc,
                val_rec,
                val_f1,
                f"{lr:.6e}",
                f"{elapsed_sec:.2f}"
            ])

    def compute_loss_and_preds(self, logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tính toán Loss và Nhãn dự đoán theo cơ chế chuẩn hóa (triệt tiêu lỗi 0-gradient)."""
        if self.is_binary_bce:
            logits_1d = logits.view(-1)
            loss = self.criterion(logits_1d, labels.float().view(-1))
            preds = (torch.sigmoid(logits_1d) >= 0.5).long()
        else:
            loss = self.criterion(logits, labels.long())
            preds = torch.argmax(logits, dim=1)
        return loss, preds

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        valid_batches_count = 0

        accum_steps = max(1, getattr(self.cfg, "gradient_accumulation_steps", 1))
        accum_count = 0
        self.optimizer.zero_grad()

        num_batches = len(self.train_loader)
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}/{self.cfg.epochs}", disable=not self.cfg.use_tqdm)
        
        for batch_idx, batch in enumerate(pbar):
            frames_t, labels, seq_lens, metas = batch

            # Bỏ qua batch rỗng do video lỗi
            if frames_t.size(0) == 0:
                continue

            frames_t = frames_t.to(self.device)
            labels = labels.to(self.device)
            seq_lens = seq_lens.to(self.device)

            try:
                autocast_enabled = (self.scaler is not None and self.device.type == "cuda")
                with torch.amp.autocast(device_type=self.device.type, enabled=autocast_enabled):
                    # 1. Trích xuất đặc trưng (p3, p4, p5) qua Extractor không tính gradient
                    p3, p4, p5 = self.extractor(frames_t)

                    # 2. Forward pass qua ConvGRU
                    logits = self.model((p3, p4, p5), seq_lens=seq_lens)

                    # 3. Tính toán Loss và Nhãn dự đoán
                    loss, preds = self.compute_loss_and_preds(logits, labels)

                    # Chuẩn hóa loss theo số bước tích lũy thực tế
                    loss_scaled = loss / accum_steps

                # 4. Backward & Cập nhật tham số
                accum_count += 1
                if self.scaler is not None:
                    self.scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                is_step_boundary = (accum_count == accum_steps) or ((batch_idx + 1) == num_batches)
                if is_step_boundary:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    accum_count = 0

                # 5. Ghi nhận dữ liệu thống kê
                unscaled_loss_item = loss.item()
                total_loss += unscaled_loss_item
                valid_batches_count += 1

                preds_np = preds.detach().cpu().numpy()
                targets_np = labels.detach().cpu().numpy()
                all_preds.extend(preds_np)
                all_targets.extend(targets_np)

                self.global_step += 1

                if self.cfg.enable_step_logging and self.global_step % self.cfg.log_step_interval == 0:
                    step_metrics = calculate_metrics(preds_np, targets_np)
                    self.writer.add_scalar("Train_Step/Loss", unscaled_loss_item, self.global_step)
                    self.writer.add_scalar("Train_Step/Accuracy", step_metrics["acc"], self.global_step)
                    self.writer.add_scalar("Train_Step/Recall", step_metrics["recall"], self.global_step)
                    self.writer.add_scalar("Train_Step/F1", step_metrics["f1"], self.global_step)

                pbar.set_postfix({"loss": f"{unscaled_loss_item:.4f}"})

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n[CẢNH BÁO] CUDA Out of Memory tại batch {batch_idx}. Đang dọn dẹp bộ nhớ và bỏ qua batch...")
                    self.optimizer.zero_grad()
                    accum_count = 0
                    try:
                        del frames_t, labels, seq_lens, metas
                    except Exception:
                        pass
                    del e
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        # Tính toán metric trung bình toàn epoch
        avg_loss = total_loss / max(valid_batches_count, 1)
        epoch_metrics = calculate_metrics(np.array(all_preds), np.array(all_targets))
        epoch_metrics["loss"] = avg_loss
        return epoch_metrics

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        valid_batches_count = 0

        pbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch}/{self.cfg.epochs}", disable=not self.cfg.use_tqdm)
        for batch_idx, batch in enumerate(pbar):
            frames_t, labels, seq_lens, metas = batch
            if frames_t.size(0) == 0:
                continue

            frames_t = frames_t.to(self.device)
            labels = labels.to(self.device)
            seq_lens = seq_lens.to(self.device)

            try:
                autocast_enabled = (self.cfg.amp and self.device.type == "cuda")
                with torch.amp.autocast(device_type=self.device.type, enabled=autocast_enabled):
                    p3, p4, p5 = self.extractor(frames_t)
                    logits = self.model((p3, p4, p5), seq_lens=seq_lens)
                    loss, preds = self.compute_loss_and_preds(logits, labels)

                loss_item = loss.item()
                total_loss += loss_item
                valid_batches_count += 1

                preds_np = preds.cpu().numpy()
                targets_np = labels.cpu().numpy()
                all_preds.extend(preds_np)
                all_targets.extend(targets_np)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n[CẢNH BÁO] CUDA Out of Memory tại validation batch {batch_idx}. Đang bỏ qua batch...")
                    try:
                        del frames_t, labels, seq_lens, metas
                    except Exception:
                        pass
                    del e
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        avg_loss = total_loss / max(valid_batches_count, 1)
        epoch_metrics = calculate_metrics(np.array(all_preds), np.array(all_targets))
        epoch_metrics["loss"] = avg_loss
        return epoch_metrics

    def _resolve_checkpoint_path(self, checkpoint_path: str) -> Path:
        """Phân giải bí danh checkpoint ('last', 'best') hoặc đường dẫn cụ thể."""
        alias = str(checkpoint_path).strip().lower()
        if alias in ("last", "last.pt"):
            return self.ckpt_dir / "last.pt"
        elif alias in ("best", "best.pt"):
            return self.ckpt_dir / "best.pt"

        target = Path(checkpoint_path)
        if not target.is_absolute():
            candidate = self.ckpt_dir / target
            if candidate.exists():
                return candidate
        return target

    def load_checkpoint(self, checkpoint_path: str) -> None:
        target_path = self._resolve_checkpoint_path(checkpoint_path)
        print(f"[*] Đang khôi phục từ checkpoint: {target_path}")
        if not target_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint tại: {target_path}")

        ckpt = torch.load(target_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_val_f1 = ckpt.get("best_val_f1", 0.0)
        self.global_step = ckpt.get("global_step", (self.start_epoch - 1) * len(self.train_loader))

        if self.scheduler is not None and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            print("[*] Đã khôi phục thành công trạng thái GradScaler (AMP).")

        print(f"[*] Đã khôi phục thành công. Sẽ tiếp tục từ epoch {self.start_epoch} (Best F1 trước đó: {self.best_val_f1:.4f})")

    def _atomic_save(self, state: Dict[str, Any], target_path: Path) -> None:
        """Lưu file checkpoint theo cơ chế Atomic Rename để chống hỏng file khi bị ngắt đột ngột."""
        tmp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
        torch.save(state, tmp_path)
        if target_path.exists():
            target_path.unlink()
        tmp_path.rename(target_path)

    def save_checkpoint(self, epoch: int, is_best: bool = False, val_f1: Optional[float] = None) -> None:
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_f1": self.best_val_f1,
            "val_f1": val_f1,
            "config": self.cfg.to_dict()
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler is not None:
            state["scaler_state_dict"] = self.scaler.state_dict()

        # 1. Luôn lưu file last.pt cuối mỗi epoch
        last_path = self.ckpt_dir / "last.pt"
        self._atomic_save(state, last_path)

        # 2. Lưu best.pt nếu đạt F1 cao nhất
        if is_best:
            best_path = self.ckpt_dir / "best.pt"
            self._atomic_save(state, best_path)
            print(f"[*] Đã lưu mô hình tốt nhất (best.pt) tại epoch {epoch} với F1: {self.best_val_f1:.4f}")

        # 3. Lưu epoch cụ thể theo cấu hình
        save_all = getattr(self.cfg, "save_all_epochs", False)
        save_interval = getattr(self.cfg, "save_ckpt_interval_epochs", 1)
        if save_all or (save_interval > 0 and epoch % save_interval == 0):
            epoch_path = self.ckpt_dir / f"epoch_{epoch}.pt"
            self._atomic_save(state, epoch_path)

        # 4. Tỉa bớt checkpoint cũ nếu cấu hình ckpt_keep_last
        keep_last = getattr(self.cfg, "ckpt_keep_last", None)
        if keep_last and keep_last > 0:
            epoch_files = sorted(
                self.ckpt_dir.glob("epoch_*.pt"),
                key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else 0
            )
            if len(epoch_files) > keep_last:
                for old_f in epoch_files[:-keep_last]:
                    try:
                        old_f.unlink()
                    except Exception:
                        pass

    def train(self) -> None:
        print("[*] Bắt đầu quá trình huấn luyện...")
        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            t_epoch_start = time.time()
            train_metrics = self.train_epoch(epoch)

            # Ghi log Train Epoch
            self.writer.add_scalar("Train_Epoch/Loss", train_metrics["loss"], epoch)
            self.writer.add_scalar("Train_Epoch/Accuracy", train_metrics["acc"], epoch)
            self.writer.add_scalar("Train_Epoch/Recall", train_metrics["recall"], epoch)
            self.writer.add_scalar("Train_Epoch/F1", train_metrics["f1"], epoch)
            self.writer.add_scalar("Train_Epoch/LearningRate", self.optimizer.param_groups[0]['lr'], epoch)

            # Đánh giá tập Validation
            run_val = (len(self.val_loader.dataset) > 0) and (
                (epoch % self.cfg.val_interval_epochs == 0) or (epoch == self.cfg.epochs)
            )

            is_best = False
            val_metrics = None
            if run_val:
                val_metrics = self.validate_epoch(epoch)
                self.writer.add_scalar("Val_Epoch/Loss", val_metrics["loss"], epoch)
                self.writer.add_scalar("Val_Epoch/Accuracy", val_metrics["acc"], epoch)
                self.writer.add_scalar("Val_Epoch/Recall", val_metrics["recall"], epoch)
                self.writer.add_scalar("Val_Epoch/F1", val_metrics["f1"], epoch)

                print(f"Epoch {epoch:02d}/{self.cfg.epochs} | Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
                      f"Val Acc: {val_metrics['acc']:.4f}")

                if val_metrics["f1"] > self.best_val_f1:
                    self.best_val_f1 = val_metrics["f1"]
                    is_best = True
            else:
                print(f"Epoch {epoch:02d}/{self.cfg.epochs} | Train Loss: {train_metrics['loss']:.4f} | "
                      f"Train F1: {train_metrics['f1']:.4f} | (Val: Bỏ qua)")

            # ĐẢM BẢO LUÔN LƯU CHECKPOINT CUỐI MỖI EPOCH
            self.save_checkpoint(
                epoch=epoch,
                is_best=is_best,
                val_f1=val_metrics["f1"] if val_metrics else None
            )

            # Ghi nhật ký vào file training_history.csv
            elapsed_sec = time.time() - t_epoch_start
            self._log_history_csv(epoch, train_metrics, val_metrics, elapsed_sec)

            # Kiểm tra Early Stopping
            if run_val and self.early_stopper is not None and val_metrics is not None:
                monitor_key = self.early_stopper.monitor.replace("val_", "")
                target_score = val_metrics.get(monitor_key, val_metrics.get(self.early_stopper.monitor, val_metrics["f1"]))
                if self.early_stopper.step(target_score, epoch):
                    print(f"[*] Early Stopping được kích hoạt tại epoch {epoch}! (Kỷ nguyên tối ưu nhất: {self.early_stopper.best_epoch})")
                    break

            if self.scheduler is not None:
                self.scheduler.step()

            # Dọn rác bộ nhớ định kỳ
            if self.cfg.empty_cache_interval > 0 and epoch % self.cfg.empty_cache_interval == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print("[*] Đã hoàn thành quá trình huấn luyện.")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Script Huấn luyện Driver Guardian ConvGRUClassifier (v2.0)")
    parser.add_argument("--config", type=str, default=None, help="Đường dẫn đến file cấu hình yaml/json")
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn file .pt hoặc bí danh ('last', 'best') để resume")
    args = parser.parse_args()

    # Tải cấu hình
    config = load_config(args.config)

    # Cố định seed
    seed_everything(config.seed)

    # Khởi tạo Trainer
    trainer = Trainer(config)

    # Nạp lại checkpoint nếu có yêu cầu
    if args.resume:
        trainer.load_checkpoint(args.resume)
    elif config.enable_resume and config.resume:
        trainer.load_checkpoint(config.resume)

    # Chạy vòng lặp huấn luyện
    trainer.train()


if __name__ == "__main__":
    main()
