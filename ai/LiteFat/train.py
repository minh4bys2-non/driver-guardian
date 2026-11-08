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
from tqdm import tqdm

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
            roll_deg=getattr(self.cfg, "roll_deg", 12.0),
            temporal_dropout=getattr(self.cfg, "temporal_dropout", 0.05),
            landmark_noise_std=self.cfg.landmark_noise_std,
            color_jitter=self.cfg.color_jitter,
            spatial_dropout=self.cfg.spatial_dropout,
            subject_balanced_sampling=getattr(self.cfg, "subject_balanced_sampling", True),
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

        # Các biến theo dõi thực nghiệm
        self.start_epoch = 1
        self.best_val_f1 = -1.0
        self.best_val_score = -1.0
        self.best_epoch = -1
        self.history: List[Dict[str, Any]] = []
        self.pretrained_param_names: set[str] = set()

        # Nạp trọng số tiền huấn luyện (Warm-start) nếu được yêu cầu trước khi tạo Optimizer
        if getattr(self.cfg, "pretrained_from", None) and not self.cfg.resume:
            self.load_pretrained_weights(self.cfg.pretrained_from)

        # 7. Thiết lập Bộ tối ưu hóa Differential Learning Rate (AdamW)
        self.optimizer = self._build_optimizer()

        # 8. Thiết lập Bộ điều chỉnh Tốc độ học (Warmup + CosineAnnealingLR)
        self.scheduler = self._build_scheduler()

        # 9. Bộ khuếch đại độ chính xác hỗn hợp Automatic Mixed Precision (AMP)
        amp_enabled = self.cfg.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        # Tự động khôi phục nếu cfg.resume được cấu hình
        if self.cfg.resume:
            self.load_checkpoint(self.cfg.resume)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Thiết lập tốc độ học riêng biệt (Differential LR) cho Backbone, Pretrained STGL và New STGL."""
        backbone_params = []
        matched_graph_params = []
        new_graph_params = []

        pretrained_scale = float(getattr(self.cfg, "pretrained_lr_scale", 1.0))
        has_pretrained = bool(self.pretrained_param_names) and (pretrained_scale < 1.0)

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "context_encoder.features" in name:
                backbone_params.append(param)
            elif has_pretrained and name in self.pretrained_param_names:
                matched_graph_params.append(param)
            else:
                new_graph_params.append(param)

        param_groups = []
        if matched_graph_params:
            param_groups.append({
                "params": matched_graph_params,
                "lr": self.cfg.lr * pretrained_scale,
                "weight_decay": self.cfg.weight_decay,
            })
            self.logger.info(
                f"Đã gán Warm-start LR={self.cfg.lr * pretrained_scale:.2e} (scale={pretrained_scale}) cho {len(matched_graph_params)} tham số pretrained."
            )
        if new_graph_params:
            param_groups.append({
                "params": new_graph_params,
                "lr": self.cfg.lr,
                "weight_decay": self.cfg.weight_decay,
            })
        if backbone_params and self.cfg.use_visual_context:
            param_groups.append({
                "params": backbone_params,
                "lr": self.cfg.backbone_lr,
                "weight_decay": self.cfg.weight_decay,
            })

        optimizer = torch.optim.AdamW(param_groups)
        return optimizer

    def _safe_load_optimizer(self, opt_state_dict: Dict[str, Any]) -> None:
        """Nạp an toàn optimizer state dict, xử lý tương thích khi số lượng param_groups khác nhau."""
        saved_groups = len(opt_state_dict.get("param_groups", []))
        curr_groups = len(self.optimizer.param_groups)

        if saved_groups == curr_groups:
            self.optimizer.load_state_dict(opt_state_dict)
        elif saved_groups == 1 and curr_groups == 2:
            self.logger.warning(
                "Checkpoint optimizer state chỉ có 1 nhóm tham số (Graph/Head). "
                "Tự động ánh xạ nhóm 0 và khởi tạo nhóm Backbone ở trạng thái mới."
            )
            adapted_dict = {
                "state": opt_state_dict.get("state", {}),
                "param_groups": [
                    opt_state_dict["param_groups"][0],
                    self.optimizer.state_dict()["param_groups"][1],
                ],
            }
            self.optimizer.load_state_dict(adapted_dict)
        else:
            self.logger.warning(
                f"Không tương thích số lượng param_groups (checkpoint: {saved_groups}, hiện tại: {curr_groups}). "
                "Bỏ qua khôi phục optimizer state."
            )

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Nạp toàn bộ trạng thái từ tệp checkpoint để tiếp tục huấn luyện."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            alt_path = self.save_dir / checkpoint_path
            if alt_path.is_file():
                ckpt_path = alt_path
            else:
                available_ckpts = list(self.save_dir.glob("*.pt"))
                available_names = [c.name for c in available_ckpts]
                raise FileNotFoundError(
                    f"Không tìm thấy tệp checkpoint tại '{checkpoint_path}' hoặc '{alt_path}'. "
                    f"Các checkpoint hiện có trong {self.save_dir}: {available_names}"
                )

        self.logger.info(f"Đang nạp checkpoint để tiếp tục huấn luyện từ: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # 1. Khôi phục trọng số mô hình
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.logger.warning("Checkpoint không chứa 'model_state_dict'!")

        # 2. Khôi phục Optimizer & Scheduler & GradScaler
        if "optimizer_state_dict" in checkpoint:
            self._safe_load_optimizer(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            saved_cfg_epochs = checkpoint.get("config", {}).get("epochs", None)
            if saved_cfg_epochs is not None and saved_cfg_epochs != self.cfg.epochs:
                self.logger.warning(
                    f"Phát hiện thay đổi tổng số epochs ({saved_cfg_epochs} -> {self.cfg.epochs}). "
                    f"Tự động tái lập lịch trình Cosine mượt từ Epoch {self.start_epoch} đến Epoch {self.cfg.epochs}."
                )
                remaining_epochs = max(1, self.cfg.epochs - self.start_epoch + 1)
                def resume_lr_lambda(ep_offset: int) -> float:
                    progress = float(ep_offset) / float(max(1, remaining_epochs))
                    return 0.5 * (1.0 + math.cos(math.pi * progress))
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=resume_lr_lambda)
            else:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint and self.scaler.is_enabled():
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # 3. Khôi phục Epochs & Metrics
        ckpt_epoch = checkpoint.get("epoch", 0)
        self.start_epoch = ckpt_epoch + 1

        if "best_epoch" in checkpoint:
            self.best_epoch = checkpoint.get("best_epoch", -1)
            self.best_val_f1 = checkpoint.get("best_val_f1", -1.0)
            self.best_val_score = checkpoint.get("best_val_score", self.best_val_f1)
        else:
            # Dự phòng từ best_metrics.json hoặc metrics của checkpoint
            best_json_path = self.save_dir / "best_metrics.json"
            if best_json_path.is_file():
                try:
                    with open(best_json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("best_epoch", 0) <= ckpt_epoch:
                        self.best_epoch = data["best_epoch"]
                        self.best_val_f1 = data.get("metrics", {}).get("f1", -1.0)
                        self.best_val_score = data.get("metrics", {}).get("val_score", self.best_val_f1)
                except Exception:
                    pass
            if self.best_val_f1 < 0 and "val_metrics" in checkpoint:
                self.best_val_f1 = checkpoint["val_metrics"].get("f1", -1.0)
                self.best_val_score = checkpoint["val_metrics"].get("val_score", self.best_val_f1)
                self.best_epoch = ckpt_epoch

        # 4. Đồng bộ trạng thái đóng/mở băng backbone theo start_epoch
        if self.start_epoch <= self.cfg.freeze_backbone_epochs:
            self.model.set_backbone_trainable(False)
        else:
            self.model.set_backbone_trainable(True, unfreeze_last_n_blocks=3)

        self.logger.info("=" * 70)
        self.logger.info(f"===> KHÔI PHỤC CHECKPOINT THÀNH CÔNG TỪ: {ckpt_path.name}")
        self.logger.info(
            f"===> Đã hoàn thành Epoch {ckpt_epoch}. "
            f"Sẽ tiếp tục huấn luyện từ Epoch {self.start_epoch} đến Epoch {self.cfg.epochs}."
        )
        self.logger.info(
            f"===> Kỷ lục Best Val F1 hiện tại: {self.best_val_f1:.4f} (tại Epoch {self.best_epoch})"
        )
        self.logger.info("=" * 70)

    def load_pretrained_weights(self, pretrained_path: str | Path) -> None:
        """Nạp trọng số từng phần (Warm-start) từ checkpoint trước đó cho các tầng khớp kích thước, hỗ trợ ánh xạ theo block."""
        p = Path(pretrained_path)
        if not p.is_file():
            self.logger.warning(f"Không tìm thấy file pretrained_from tại '{pretrained_path}'. Bỏ qua nạp warm-start.")
            return

        self.logger.info(f"Đang nạp trọng số tiền huấn luyện (Warm-start) từ: {p}")
        checkpoint = torch.load(p, map_location=self.device, weights_only=False)
        saved_dict = checkpoint.get("model_state_dict", checkpoint)
        curr_dict = self.model.state_dict()

        # Xác định cấu trúc số tầng STGL của checkpoint cũ vs mô hình hiện tại
        saved_stgl_indices = sorted(list(set(int(k.split(".")[1]) for k in saved_dict if k.startswith("stgl_blocks."))))
        curr_stgl_indices = sorted(list(set(int(k.split(".")[1]) for k in curr_dict if k.startswith("stgl_blocks."))))

        num_blocks = getattr(self.model, "num_blocks", 3)
        saved_lpb = len(saved_stgl_indices) // num_blocks if saved_stgl_indices else 0
        curr_lpb = len(curr_stgl_indices) // num_blocks if curr_stgl_indices else 0

        # Ánh xạ theo từng Block logic: (block b, layer l) -> (block b, layer l)
        block_mapping = {}
        if saved_lpb > 0 and curr_lpb > 0:
            for b in range(num_blocks):
                for l in range(min(saved_lpb, curr_lpb)):
                    s_idx = b * saved_lpb + l
                    c_idx = b * curr_lpb + l
                    block_mapping[c_idx] = s_idx

        matched_keys = []
        adapted_keys = []
        skipped_keys = []
        self.pretrained_param_names = set()

        for name, curr_param in curr_dict.items():
            if name.startswith("stgl_blocks."):
                parts = name.split(".")
                c_idx = int(parts[1])
                suffix = ".".join(parts[2:])
                if c_idx in block_mapping:
                    saved_name = f"stgl_blocks.{block_mapping[c_idx]}.{suffix}"
                else:
                    saved_name = None
            else:
                saved_name = name

            if saved_name is not None and saved_name in saved_dict:
                saved_param = saved_dict[saved_name]
                if curr_param.shape == saved_param.shape:
                    curr_param.copy_(saved_param)
                    matched_keys.append(f"{name} <- {saved_name}")
                    self.pretrained_param_names.add(name)
                elif (
                    curr_param.ndim == 4
                    and saved_param.ndim == 4
                    and curr_param.shape[:3] == saved_param.shape[:3]
                ):
                    # TCN Conv2d: shape [C_out, C_in, 1, kernel_size]
                    # Khởi tạo 0 toàn bộ kernel mới, sao chép kernel cũ vào các vị trí nhân quả cuối
                    curr_param.zero_()
                    k_saved = saved_param.shape[-1]
                    curr_param[..., -k_saved:].copy_(saved_param)
                    adapted_keys.append(f"{name} (kernel {k_saved} -> {curr_param.shape[-1]})")
                    self.pretrained_param_names.add(name)
                else:
                    skipped_keys.append(f"{name} (shape lệch: {saved_param.shape} vs {curr_param.shape})")
            else:
                # Tầng mới: Zero-init skip_conv và res_scale để tầng mới bắt đầu như identity pass-through
                if name.startswith("stgl_blocks.") and ("skip_conv" in name or "res_scale" in name):
                    curr_param.zero_()
                    skipped_keys.append(f"{name} (tầng mới - zero-init)")
                else:
                    skipped_keys.append(f"{name} (mới)")

        self.logger.info("=" * 70)
        self.logger.info(f"===> KẾT QUẢ WARM-START TỪ {p.name}:")
        self.logger.info(f"===> Khớp hoàn toàn: {len(matched_keys)} tensors")
        self.logger.info(f"===> Tương thích thích nghi (TCN kernel adaptation): {len(adapted_keys)} tensors")
        self.logger.info(f"===> Khởi tạo mới / Zero-init: {len(skipped_keys)} tensors")
        self.logger.info("=" * 70)

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
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:02d}/{self.cfg.epochs:02d} [Train]",
            dynamic_ncols=True,
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
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
                    video_frames=batch_frames,
                    landmarks=batch_landmarks,
                    lengths=batch_lengths,
                    return_clip_logits=True,
                )
                loss = self.criterion(
                    frame_logits=frame_logits,
                    targets=batch_labels,
                    clip_logits=clip_logits,
                    lengths=batch_lengths,
                )

            # Backward Pass với GradScaler (kèm kiểm tra an toàn số học)
            if not torch.isfinite(loss):
                self.logger.warning(
                    f"Epoch {epoch:02d} | Batch {batch_idx + 1}/{num_batches}: Loss không hữu hạn ({loss.item()})! "
                    "Bỏ qua cập nhật gradient để bảo vệ trọng số mô hình."
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue

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

            current_lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
                "lr": f"{current_lr:.2e}",
            })

        elapsed = time.time() - start_time
        metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
        metrics["loss"] = total_loss / max(1, num_batches)
        metrics["time"] = elapsed

        return metrics

    @torch.no_grad()
    def _evaluate_epoch(
            self,
            dataloader: Any,
            split_name: str = "val",
            epoch: Optional[int] = None,
            return_raw: bool = False,
    ) -> Any:
        """Đánh giá mô hình trên tập validation hoặc test."""
        self.model.eval()
        total_loss = 0.0
        all_targets: List[int] = []
        all_preds: List[int] = []
        all_probs: List[float] = []
        num_batches = len(dataloader)

        start_time = time.time()
        desc = (
            f"Epoch {epoch:02d}/{self.cfg.epochs:02d} [{split_name.capitalize()}]"
            if epoch is not None
            else f"[{split_name.capitalize()}]"
        )
        pbar = tqdm(
            dataloader,
            desc=desc,
            dynamic_ncols=True,
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
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
                    video_frames=batch_frames,
                    landmarks=batch_landmarks,
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
            probs = torch.softmax(clip_logits, dim=-1)[:, 1].cpu().numpy()
            preds = torch.argmax(clip_logits, dim=-1).cpu().numpy()
            targets = batch_labels.cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())
            all_probs.extend(probs.tolist())

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
            })

        elapsed = time.time() - start_time
        targets_np = np.array(all_targets)
        preds_np = np.array(all_preds)
        probs_np = np.array(all_probs)

        metrics = compute_metrics(targets_np, preds_np, probs_np)
        metrics["loss"] = total_loss / max(1, num_batches)
        metrics["time"] = elapsed

        # Quét ngưỡng tối ưu (Threshold Calibration) cân bằng Recall & Specificity
        best_th = 0.50
        candidates = []
        for th in np.linspace(0.35, 0.65, 31):
            m_th = compute_metrics(targets_np, (probs_np >= th).astype(int), probs_np)
            candidates.append((float(th), m_th))

        valid_candidates = [c for c in candidates if c[1]["specificity"] >= 0.50 and c[1]["recall"] >= 0.50]
        if valid_candidates:
            best_th, best_m = max(valid_candidates, key=lambda c: c[1]["f1"])
        else:
            best_th, best_m = max(candidates, key=lambda c: c[1]["f1"])

        metrics["optimal_threshold"] = best_th
        metrics["calibrated_f1"] = best_m["f1"]
        metrics["calibrated_accuracy"] = best_m["accuracy"]
        metrics["calibrated_recall"] = best_m["recall"]
        metrics["calibrated_precision"] = best_m["precision"]
        metrics["calibrated_specificity"] = best_m["specificity"]

        auc = metrics.get("auc", 0.5)
        # Composite validation score kết hợp hài hòa F1, AUC và Accuracy
        metrics["val_score"] = 0.50 * metrics["calibrated_f1"] + 0.30 * auc + 0.20 * metrics["accuracy"]

        if return_raw:
            return metrics, targets_np, probs_np
        return metrics

    def train(self) -> None:
        """Vòng lặp huấn luyện chính qua toàn bộ các epoch."""
        if self.start_epoch > self.cfg.epochs:
            self.logger.warning(
                f"Checkpoint đã hoàn thành Epoch {self.start_epoch - 1}, trong khi cfg.epochs là {self.cfg.epochs}. "
                f"Không có epoch mới để huấn luyện. Vui lòng tăng 'epochs' trong TrainConfig "
                f"(ví dụ: epochs={self.start_epoch + 4}) để tiếp tục."
            )
            self.evaluate_test()
            self.writer.close()
            return

        self.logger.info("\n>>> BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN LITEFAT <<<")
        if self.start_epoch > 1:
            self.logger.info(f"Tiếp tục huấn luyện từ Epoch {self.start_epoch:02d} đến {self.cfg.epochs:02d}")

        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._evaluate_epoch(self.val_loader, split_name="val", epoch=epoch)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log chỉ số ra Console
            self.logger.info(
                f"Epoch [{epoch:02d}/{self.cfg.epochs:02d}] \n"
                f"\t Train: Loss={train_metrics['loss']:.4f} | Acc={train_metrics['accuracy']*100:.2f}% | F1={train_metrics['f1']:.4f} |\n"
                f"\t Val  : Loss={val_metrics['loss']:.4f} | Acc={val_metrics['accuracy']*100:.2f}% | AUC={val_metrics.get('auc', 0.5):.4f} | Score={val_metrics.get('val_score', 0.0):.4f} |\n"
                f"\t Val @0.50: F1={val_metrics['f1']:.4f} (Prec={val_metrics['precision']*100:.1f}%, Rec={val_metrics['recall']*100:.1f}%) |\n"
                f"\t Val @{val_metrics.get('optimal_threshold', 0.50):.2f}: F1={val_metrics.get('calibrated_f1', 0.0):.4f} (Prec={val_metrics.get('calibrated_precision', 0.0)*100:.1f}%, Rec={val_metrics.get('calibrated_recall', 0.0)*100:.1f}%, Spec={val_metrics.get('calibrated_specificity', 0.0)*100:.1f}%) |\n"
                f"\t LR: {current_lr:.2e} | Time: {train_metrics['time']:.1f}s"
            )

            # Ghi chỉ số vào TensorBoard
            self.writer.add_scalar("Loss/Train", train_metrics["loss"], epoch)
            self.writer.add_scalar("Loss/Val", val_metrics["loss"], epoch)
            self.writer.add_scalar("Accuracy/Train", train_metrics["accuracy"], epoch)
            self.writer.add_scalar("Accuracy/Val", val_metrics["accuracy"], epoch)
            self.writer.add_scalar("F1/Train", train_metrics["f1"], epoch)
            self.writer.add_scalar("F1/Val", val_metrics["f1"], epoch)
            self.writer.add_scalar("AUC/Val", val_metrics.get("auc", 0.5), epoch)
            self.writer.add_scalar("Score/Val", val_metrics.get("val_score", 0.0), epoch)
            self.writer.add_scalar("Calibrated_F1/Val", val_metrics.get("calibrated_f1", 0.0), epoch)
            self.writer.add_scalar("Optimal_Threshold/Val", val_metrics.get("optimal_threshold", 0.50), epoch)
            self.writer.add_scalar("Recall/Val", val_metrics["recall"], epoch)
            self.writer.add_scalar("LR", current_lr, epoch)

            record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": current_lr,
            }
            self.history.append(record)

            # 1. Cập nhật kỷ lục best.pt theo composite val_score (tránh chọn các epoch FP ảo)
            cur_val_score = val_metrics.get("val_score", val_metrics["f1"])
            is_new_best = cur_val_score > self.best_val_score
            if is_new_best:
                self.best_val_score = cur_val_score
                self.best_val_f1 = val_metrics.get("calibrated_f1", val_metrics["f1"])
                self.best_epoch = epoch

            # 2. Lưu checkpoint cuối cùng (last.pt)
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "best_epoch": self.best_epoch,
                "best_val_score": self.best_val_score,
                "best_val_f1": self.best_val_f1,
                "val_metrics": val_metrics,
                "train_metrics": train_metrics,
                "config": self.cfg.to_dict(),
            }
            atomic_save(checkpoint_payload, self.save_dir / "last.pt")

            # 3. Lưu checkpoint định kỳ theo save_freq
            if epoch % self.cfg.save_freq == 0:
                atomic_save(checkpoint_payload, self.save_dir / f"epoch_{epoch:02d}.pt")

            # 4. Lưu checkpoint tốt nhất (best.pt)
            if is_new_best:
                atomic_save(checkpoint_payload, self.save_dir / "best.pt")
                with open(self.save_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                    json.dump({"best_epoch": epoch, "metrics": val_metrics}, f, indent=2)
                self.logger.info(
                    f"===> ĐÃ LƯU CHECKPOINT TỐT NHẤT TẠI EPOCH {epoch} "
                    f"(Score: {self.best_val_score:.4f} | Calib F1: {self.best_val_f1:.4f} | AUC: {val_metrics.get('auc', 0.5):.4f})"
                )

        self.logger.info("\n>>> HOÀN THÀNH HUẤN LUYỆN! <<<")
        self.logger.info(f"Checkpoint tốt nhất: Epoch {self.best_epoch} (Val F1: {self.best_val_f1:.4f})")

        # Đánh giá tự động trên tập Test
        self.evaluate_test()
        self.writer.close()

    def evaluate_test(self) -> Dict[str, float]:
        """Tải checkpoint best.pt và đánh giá trên tập Test độc lập kèm Calibration ngưỡng."""
        best_pt_path = self.save_dir / "best.pt"
        if not best_pt_path.is_file():
            self.logger.warning(f"Không tìm thấy checkpoint {best_pt_path} để đánh giá test.")
            return {}

        self.logger.info(f"\n--- ĐÁNH GIÁ TRÊN TẬP TEST VỚI CHECKPOINT TỐT NHẤT (Epoch {self.best_epoch}) ---")
        checkpoint = torch.load(best_pt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # 1. Tìm ngưỡng tối ưu trên tập Validation
        val_metrics, val_targets, val_probs = self._evaluate_epoch(self.val_loader, split_name="val", return_raw=True)
        best_th = float(val_metrics.get("optimal_threshold", 0.50))
        best_val_f1 = float(val_metrics.get("calibrated_f1", val_metrics["f1"]))

        self.logger.info(
            f"Ngưỡng tối ưu trên Validation: {best_th:.2f} "
            f"(Val Calib F1: {best_val_f1:.4f} vs thr 0.50: {val_metrics['f1']:.4f} | AUC: {val_metrics.get('auc', 0.5):.4f})"
        )

        # 2. Đánh giá trên tập Test với cả ngưỡng mặc định và ngưỡng tối ưu
        test_metrics, test_targets, test_probs = self._evaluate_epoch(self.test_loader, split_name="test", return_raw=True)
        self.logger.info(
            f"[Ngưỡng mặc định 0.50] Test Loss: {test_metrics['loss']:.4f} | "
            f"Test Acc: {test_metrics['accuracy']*100:.2f}% | "
            f"Test AUC: {test_metrics.get('auc', 0.5):.4f} | "
            f"Test Precision: {test_metrics['precision']*100:.2f}% | "
            f"Test Recall: {test_metrics['recall']*100:.2f}% | "
            f"Test Specificity: {test_metrics.get('specificity', 0.0)*100:.2f}% | "
            f"Test F1: {test_metrics['f1']:.4f} | "
            f"Test F2: {test_metrics['f2']:.4f}"
        )

        calibrated_test_metrics = compute_metrics(test_targets, (test_probs >= best_th).astype(int), test_probs)
        calibrated_test_metrics["loss"] = test_metrics["loss"]
        calibrated_test_metrics["optimal_threshold"] = best_th

        self.logger.info(
            f"[Ngưỡng tối ưu {best_th:.2f}] Test Acc: {calibrated_test_metrics['accuracy']*100:.2f}% | "
            f"Test AUC: {calibrated_test_metrics.get('auc', 0.5):.4f} | "
            f"Test Precision: {calibrated_test_metrics['precision']*100:.2f}% | "
            f"Test Recall: {calibrated_test_metrics['recall']*100:.2f}% | "
            f"Test Specificity: {calibrated_test_metrics.get('specificity', 0.0)*100:.2f}% | "
            f"Test F1: {calibrated_test_metrics['f1']:.4f} | "
            f"Test F2: {calibrated_test_metrics['f2']:.4f}"
        )

        output_payload = {
            "default_metrics": test_metrics,
            "calibrated_metrics": calibrated_test_metrics,
            "optimal_threshold": best_th,
            "best_epoch": self.best_epoch,
        }
        with open(self.save_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

        return test_metrics


def main():
    """Hàm chạy thực thi dòng lệnh: python train.py."""
    config = TrainConfig()
    trainer = LiteFatTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
