# =====================================================================
# File: train.py
# Dự án: Driver Guardian - Module LiteFat (IEEE/RSJ IROS 2025 Upgraded)
# Tác vụ: Lớp huấn luyện LiteFatTrainer thực thi pipeline huấn luyện
#         theo cấu hình TrainConfig trong config.py.
# =====================================================================
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config import TrainConfig
from dataset import build_litefat_dataloaders
from loss import LiteFatLoss
from model import LiteFat
from ultis import atomic_save, compute_metrics, configure_logging, seed_everything


class LiteFatTrainer:
    """
    Trình quản lý và thực thi huấn luyện mô hình LiteFat:
      - Hỗ trợ Differential Learning Rate (tốc độ học riêng cho Backbone và GCN/TCN).
      - Đóng băng Backbone theo giai đoạn (Staged Backbone Freezing).
      - Linear Warmup + Cosine Annealing Learning Rate.
      - Tăng cường dữ liệu không gian - thời gian đối xứng trên Landmark & Video Frames.
      - Lưu checkpoint an toàn (Atomic Save) và theo dõi qua TensorBoard.
    """

    def __init__(self, config: Optional[TrainConfig] = None):
        self.cfg = config if config is not None else TrainConfig()

        # 1. Thiết lập hạt giống ngẫu nhiên để tái lập kết quả
        seed_everything(self.cfg.seed)

        # 2. Thiết lập thư mục lưu trữ và Logging
        self.save_dir = Path(self.cfg.checkpoint_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = configure_logging(self.save_dir)

        # Lưu lại cấu hình JSON của phiên huấn luyện
        self.cfg.save_json(self.save_dir / "config.json")

        # Khởi tạo TensorBoard SummaryWriter
        log_path = Path(self.cfg.log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_path))

        # 3. Khởi tạo Thiết bị tính toán (CUDA GPU / CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info("=" * 70)
        self.logger.info(f"KHỞI TẠO LITEFAT TRAINER (Thiết bị: {self.device})")
        self.logger.info(f"Chế độ Bối cảnh Thị giác (use_visual_context): {self.cfg.use_visual_context}")
        self.logger.info(f"Thời gian lấy mẫu (time_interval): {self.cfg.time_interval}s | Cửa sổ: {self.cfg.window_sec}s")
        self.logger.info(f"Bộ dữ liệu chọn lọc: {self.cfg.selected_datasets or 'Tất cả'}")
        self.logger.info("=" * 70)

        # 4. Xây dựng DataLoaders (Train, Val, Test)
        self.logger.info("Đang khởi tạo DataLoaders...")
        self.train_loader, self.val_loader, self.test_loader = build_litefat_dataloaders(
            dataset_dir=self.cfg.dataset_dir,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            time_interval=self.cfg.time_interval,
            window_sec=self.cfg.window_sec,
            letterbox_size=self.cfg.letterbox_size,
            face_size=self.cfg.face_size,
            manifest_filename=self.cfg.manifest_filename,
            use_cache=self.cfg.use_cache,
            cache_dir=self.cfg.cache_dir,
            pin_memory=self.cfg.pin_memory,
            shuffle_train=True,
            device=str(self.device),
            extract_crops_only=(not self.cfg.use_visual_context),
            selected_datasets=self.cfg.selected_datasets,
            use_visual_context=self.cfg.use_visual_context,
            augment_train=self.cfg.augment_train,
            flip_prob=self.cfg.flip_prob,
            scale_jitter=self.cfg.scale_jitter,
            shift_jitter=self.cfg.shift_jitter,
            landmark_noise_std=self.cfg.landmark_noise_std,
            color_jitter=self.cfg.color_jitter,
            spatial_dropout=self.cfg.spatial_dropout,
        )
        self.logger.info(
            f"Kích thước tập dữ liệu: Train={len(self.train_loader.dataset)} | "
            f"Val={len(self.val_loader.dataset)} | Test={len(self.test_loader.dataset)}"
        )

        # 5. Khởi tạo Mô hình LiteFat
        self.logger.info("Đang khởi tạo mô hình LiteFat...")
        self.model = LiteFat(
            num_nodes=self.cfg.num_nodes,
            feature_dim=self.cfg.feature_dim,
            mobilenet_proj_hidden=self.cfg.mobilenet_proj_hidden,
            node_embed_dim=self.cfg.node_embed_dim,
            residual_channels=self.cfg.residual_channels,
            dilation_channels=self.cfg.dilation_channels,
            skip_channels=self.cfg.skip_channels,
            end_channels=self.cfg.end_channels,
            num_classes=self.cfg.num_classes,
            num_blocks=self.cfg.num_blocks,
            layers_per_block=self.cfg.layers_per_block,
            kernel_size=self.cfg.kernel_size,
            pretrained_backbone=self.cfg.pretrained_backbone,
            dropout=self.cfg.dropout,
            freeze_backbone=(self.cfg.freeze_backbone_epochs > 0),
            vertical_pairs=self.cfg.vertical_pairs,
            cross_region_pairs=self.cfg.cross_region_pairs,
        ).to(self.device)

        stats = self.model.get_trainable_params_count()
        self.logger.info(
            f"Tổng số tham số: {stats['total_params']:,} | Khả học: {stats['trainable_params']:,} "
            f"(MobileNetV3: {stats['backbone_trainable']:,} / {stats['backbone_total']:,})"
        )

        # 6. Khởi tạo Hàm mất mát LiteFatLoss
        self.criterion = LiteFatLoss(
            clip_loss_weight=self.cfg.clip_loss_weight,
            frame_loss_weight=self.cfg.frame_loss_weight,
            label_smoothing=self.cfg.label_smoothing,
        ).to(self.device)

        # 7. Thiết lập Bộ tối ưu hóa Differential Learning Rate (AdamW)
        self.optimizer = self._build_optimizer()

        # 8. Thiết lập Bộ điều chỉnh Tốc độ học (Warmup + CosineAnnealingLR)
        self.scheduler = self._build_scheduler()

        # 9. Bộ khuếch đại độ chính xác hỗn hợp Automatic Mixed Precision (AMP)
        amp_enabled = self.cfg.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        # Các biến theo dõi thực nghiệm
        self.best_val_f1 = -1.0
        self.best_epoch = -1
        self.history: List[Dict[str, Any]] = []

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Thiết lập tốc độ học riêng biệt (Differential LR) cho Backbone và Graph/Head."""
        backbone_params = []
        graph_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "context_encoder.features" in name:
                backbone_params.append(param)
            else:
                graph_params.append(param)

        param_groups = [
            {"params": graph_params, "lr": self.cfg.lr, "weight_decay": self.cfg.weight_decay},
        ]
        if backbone_params:
            param_groups.append(
                {"params": backbone_params, "lr": self.cfg.backbone_lr, "weight_decay": self.cfg.weight_decay}
            )

        optimizer = torch.optim.AdamW(param_groups)
        return optimizer

    def _build_scheduler(self) -> Any:
        """Thiết lập lịch trình học Cosine Annealing kèm Linear Warmup."""
        warmup_epochs = max(0, self.cfg.warmup_epochs)
        total_epochs = max(1, self.cfg.epochs)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        import math
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Thực thi một epoch huấn luyện."""
        self.model.train()

        # Quản lý đóng/mở khóa backbone MobileNetV3 theo epoch
        if epoch <= self.cfg.freeze_backbone_epochs:
            self.model.set_backbone_trainable(False)
        else:
            self.model.set_backbone_trainable(True, unfreeze_last_n_blocks=3)

        total_loss = 0.0
        all_targets: List[int] = []
        all_preds: List[int] = []
        num_batches = len(self.train_loader)

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Nhận dạng các thành phần của batch từ collate_litefat
            if len(batch) == 4:
                batch_frames, batch_landmarks, batch_labels, batch_lengths = batch
            elif len(batch) >= 5:
                batch_frames, batch_landmarks, batch_labels, batch_lengths = batch[:4]
            else:
                batch_frames, batch_landmarks, batch_labels = batch[:3]
                batch_lengths = None

            batch_landmarks = batch_landmarks.to(self.device, non_blocking=True)
            batch_labels = batch_labels.to(self.device, non_blocking=True)
            if batch_frames is not None:
                batch_frames = batch_frames.to(self.device, non_blocking=True)
            if batch_lengths is not None:
                batch_lengths = batch_lengths.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Chạy Forward Pass với AMP
            with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                frame_logits, clip_logits = self.model(
                    input1=batch_frames,
                    input2=batch_landmarks,
                    lengths=batch_lengths,
                    return_clip_logits=True,
                )
                loss = self.criterion(
                    frame_logits=frame_logits,
                    targets=batch_labels,
                    clip_logits=clip_logits,
                    lengths=batch_lengths,
                )

            # Backward Pass với GradScaler
            self.scaler.scale(loss).backward()

            if self.cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            preds = torch.argmax(clip_logits, dim=-1).detach().cpu().numpy()
            targets = batch_labels.detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

        elapsed = time.time() - start_time
        metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
        metrics["loss"] = total_loss / max(1, num_batches)
        metrics["time"] = elapsed

        return metrics

    @torch.no_grad()
    def _evaluate_epoch(self, dataloader: Any, split_name: str = "val") -> Dict[str, float]:
        """Đánh giá mô hình trên tập validation hoặc test."""
        self.model.eval()
        total_loss = 0.0
        all_targets: List[int] = []
        all_preds: List[int] = []
        num_batches = len(dataloader)

        start_time = time.time()

        for batch in dataloader:
            if len(batch) == 4:
                batch_frames, batch_landmarks, batch_labels, batch_lengths = batch
            elif len(batch) >= 5:
                batch_frames, batch_landmarks, batch_labels, batch_lengths = batch[:4]
            else:
                batch_frames, batch_landmarks, batch_labels = batch[:3]
                batch_lengths = None

            batch_landmarks = batch_landmarks.to(self.device, non_blocking=True)
            batch_labels = batch_labels.to(self.device, non_blocking=True)
            if batch_frames is not None:
                batch_frames = batch_frames.to(self.device, non_blocking=True)
            if batch_lengths is not None:
                batch_lengths = batch_lengths.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                frame_logits, clip_logits = self.model(
                    input1=batch_frames,
                    input2=batch_landmarks,
                    lengths=batch_lengths,
                    return_clip_logits=True,
                )
                loss = self.criterion(
                    frame_logits=frame_logits,
                    targets=batch_labels,
                    clip_logits=clip_logits,
                    lengths=batch_lengths,
                )

            total_loss += loss.item()
            preds = torch.argmax(clip_logits, dim=-1).cpu().numpy()
            targets = batch_labels.cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

        elapsed = time.time() - start_time
        metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
        metrics["loss"] = total_loss / max(1, num_batches)
        metrics["time"] = elapsed

        return metrics

    def train(self) -> None:
        """Vòng lặp huấn luyện chính qua toàn bộ các epoch."""
        self.logger.info("\n>>> BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN LITEFAT <<<")

        for epoch in range(1, self.cfg.epochs + 1):
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._evaluate_epoch(self.val_loader, split_name="val")

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log chỉ số ra Console
            self.logger.info(
                f"Epoch [{epoch:02d}/{self.cfg.epochs:02d}] "
                f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']*100:.2f}% | Train F1: {train_metrics['f1']:.4f} || "
                f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}% | Val F1: {val_metrics['f1']:.4f} (Rec: {val_metrics['recall']*100:.1f}%) | "
                f"LR: {current_lr:.2e} | Time: {train_metrics['time']:.1f}s"
            )

            # Ghi chỉ số vào TensorBoard
            self.writer.add_scalar("Loss/Train", train_metrics["loss"], epoch)
            self.writer.add_scalar("Loss/Val", val_metrics["loss"], epoch)
            self.writer.add_scalar("Accuracy/Train", train_metrics["accuracy"], epoch)
            self.writer.add_scalar("Accuracy/Val", val_metrics["accuracy"], epoch)
            self.writer.add_scalar("F1/Train", train_metrics["f1"], epoch)
            self.writer.add_scalar("F1/Val", val_metrics["f1"], epoch)
            self.writer.add_scalar("Recall/Val", val_metrics["recall"], epoch)
            self.writer.add_scalar("LR", current_lr, epoch)

            record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": current_lr,
            }
            self.history.append(record)

            # Lưu checkpoint cuối cùng (last.pt)
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "val_metrics": val_metrics,
                "config": self.cfg.to_dict(),
            }
            atomic_save(checkpoint_payload, self.save_dir / "last.pt")

            # Lưu checkpoint định kỳ theo save_freq
            if epoch % self.cfg.save_freq == 0:
                atomic_save(checkpoint_payload, self.save_dir / f"epoch_{epoch:02d}.pt")

            # Lưu checkpoint tốt nhất (best.pt) theo Val F1
            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                self.best_epoch = epoch
                atomic_save(checkpoint_payload, self.save_dir / "best.pt")
                with open(self.save_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                    json.dump({"best_epoch": epoch, "metrics": val_metrics}, f, indent=2)
                self.logger.info(f"===> ĐÃ LƯU CHECKPOINT TỐT NHẤT TẠI EPOCH {epoch} (Val F1: {self.best_val_f1:.4f})")

        self.logger.info("\n>>> HOÀN THÀNH HUẤN LUYỆN! <<<")
        self.logger.info(f"Checkpoint tốt nhất: Epoch {self.best_epoch} (Val F1: {self.best_val_f1:.4f})")

        # Đánh giá tự động trên tập Test
        self.evaluate_test()
        self.writer.close()

    def evaluate_test(self) -> Dict[str, float]:
        """Tải checkpoint best.pt và đánh giá trên tập Test độc lập."""
        best_pt_path = self.save_dir / "best.pt"
        if not best_pt_path.is_file():
            self.logger.warning(f"Không tìm thấy checkpoint {best_pt_path} để đánh giá test.")
            return {}

        self.logger.info(f"\n--- ĐÁNH GIÁ TRÊN TẬP TEST VỚI CHECKPOINT TỐT NHẤT (Epoch {self.best_epoch}) ---")
        checkpoint = torch.load(best_pt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics = self._evaluate_epoch(self.test_loader, split_name="test")
        self.logger.info(
            f"Test Loss: {test_metrics['loss']:.4f} | "
            f"Test Acc: {test_metrics['accuracy']*100:.2f}% | "
            f"Test Precision: {test_metrics['precision']*100:.2f}% | "
            f"Test Recall: {test_metrics['recall']*100:.2f}% | "
            f"Test F1: {test_metrics['f1']:.4f} | "
            f"Test F2: {test_metrics['f2']:.4f}"
        )

        with open(self.save_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

        return test_metrics


def main():
    """Hàm chạy thực thi dòng lệnh: python train.py."""
    config = TrainConfig()
    trainer = LiteFatTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
