# =====================================================================
# File: train.py
# Dự án: Driver Guardian - Module LiteFat_v1 (KSS 1-9, Sliding 60s/30s)
# Tác vụ: Lớp LiteFatTrainer thực thi pipeline huấn luyện, lưu checkpoint,
#         đánh giá Validation/Test (KSS + phân rã dataset/subject) và
#         ghi log TensorBoard theo cấu hình TrainConfig trong config.py.
# =====================================================================
from __future__ import annotations

import csv
import datetime
import json
import math
import multiprocessing as mp
import subprocess
import sys

try:
    mp.set_start_method("fork", force=True)
except Exception:
    pass

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2

try:
    cv2.setNumThreads(0)
except Exception:
    pass

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import TrainConfig
from dataset import build_litefat_dataloaders
from loss import LiteFatLoss
from model import LiteFatV1
from ultis import (
    ModelEMA,
    atomic_save,
    binary_auc,
    compute_kss_metrics,
    configure_logging,
    load_checkpoint,
    load_pretrained_weights,
    quadratic_weighted_kappa,
    seed_everything,
)


class LiteFatTrainer:
    """
    Trình quản lý và thực thi huấn luyện mô hình LiteFatV1 (dự đoán điểm KSS 1..9):
      - Hỗ trợ Differential Learning Rate (Backbone MobileNetV3 vs Đồ thị STGL/KSS Head).
      - Đóng băng Backbone theo giai đoạn (Staged Backbone Freezing).
      - Linear Warmup + Cosine Annealing Learning Rate + GradScaler AMP.
      - Exponential Moving Average (EMA) cho đánh giá ổn định.
      - Lưu checkpoint an toàn (last.pt, best.pt, epoch_XX.pt) và theo dõi TensorBoard.
    """

    def __init__(
            self,
            config: Optional[TrainConfig] = None,
            dataloaders: Optional[Tuple[Any, Any, Any]] = None,
    ):
        self.cfg = config if config is not None else TrainConfig()

        # 1. Thiết lập hạt giống ngẫu nhiên
        seed_everything(self.cfg.seed)

        # 2. Thư mục lưu trữ, Logging & TensorBoard
        self.save_dir = Path(self.cfg.checkpoint_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = configure_logging(self.save_dir)
        self.cfg.save_json(self.save_dir / "config.json")
        self._write_run_meta()

        log_path = Path(self.cfg.log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_path))

        # 3. Thiết bị tính toán
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info("=" * 72)
        self.logger.info(f"KHỞI TẠO LITEFAT_V1 TRAINER (Thiết bị: {self.device})")
        self.logger.info(f"Ngữ cảnh thị giác (use_visual_context): {self.cfg.use_visual_context}")
        self.logger.info(
            f"Lấy mẫu: {self.cfg.fps} FPS (dt={self.cfg.time_interval}s) | "
            f"Model Window={self.cfg.window_sec}s / Stride={self.cfg.stride_sec}s | "
            f"Dataset Window={self.cfg.dataset_window_sec}"
        )
        self.grad_accum_steps = max(1, int(getattr(self.cfg, "grad_accum_steps", 1)))
        self.logger.info(
            f"Batch size: {self.cfg.batch_size} | Grad accum steps: {self.grad_accum_steps} | "
            f"Effective batch size: {self.cfg.batch_size * self.grad_accum_steps}"
        )
        self.logger.info(f"Bộ dữ liệu: {self.cfg.selected_datasets or 'Tất cả'} ({self.cfg.manifest_filename})")
        self.logger.info("=" * 72)

        # 4. Xây dựng DataLoaders (Train, Val, Test)
        if dataloaders is not None:
            self.train_loader, self.val_loader, self.test_loader = dataloaders
        else:
            self.logger.info("Đang khởi tạo DataLoaders...")
            self.train_loader, self.val_loader, self.test_loader = build_litefat_dataloaders(
                dataset_dir=self.cfg.dataset_dir,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_workers,
                time_interval=self.cfg.time_interval,
                window_sec=self.cfg.dataset_window_sec,
                letterbox_size=self.cfg.letterbox_size,
                face_size=self.cfg.face_size,
                manifest_filename=self.cfg.manifest_filename,
                use_cache=self.cfg.use_cache,
                cache_dir=self.cfg.cache_dir,
                pin_memory=self.cfg.pin_memory,
                shuffle_train=True,
                device=str(self.device),
                pipnet_backbone=self.cfg.pipnet_backbone,
                conf_threshold=self.cfg.conf_threshold,
                nms_threshold=self.cfg.nms_threshold,
                bbox_margin=self.cfg.bbox_margin,
                box_smooth_alpha=self.cfg.box_smooth_alpha,
                extract_crops_only=(not self.cfg.use_visual_context),
                selected_datasets=self.cfg.selected_datasets,
                use_visual_context=self.cfg.use_visual_context,
                augment_train=self.cfg.augment_train,
                flip_prob=self.cfg.flip_prob,
                scale_jitter=self.cfg.scale_jitter,
                shift_jitter=self.cfg.shift_jitter,
                roll_deg=self.cfg.roll_deg,
                temporal_dropout=self.cfg.temporal_dropout,
                landmark_noise_std=self.cfg.landmark_noise_std,
                color_jitter=self.cfg.color_jitter,
                spatial_dropout=self.cfg.spatial_dropout,
                subject_balanced_sampling=self.cfg.subject_balanced_sampling,
                use_manifest_sample_weights=self.cfg.use_manifest_sample_weights,
                eval_window_mode=self.cfg.eval_window_mode,
                drop_single_class_subjects=self.cfg.drop_single_class_subjects,
                visual_frame_stride=self.cfg.visual_frame_stride,
                sampler_uta_quota=float(getattr(self.cfg, "sampler_uta_quota", 0.40)),
                sampler_weight_exponents=tuple(getattr(self.cfg, "sampler_weight_exponents", (1.0, 1.0, 0.5))),
            )

        if hasattr(self.train_loader, "dataset") and hasattr(self.train_loader.dataset, "return_meta"):
            self.train_loader.dataset.return_meta = True

        self.logger.info(
            f"Kích thước tập dữ liệu: Train={len(self.train_loader.dataset)} | "
            f"Val={len(self.val_loader.dataset)} | Test={len(self.test_loader.dataset)}"
        )

        # 5. Khởi tạo Mô hình LiteFatV1 & EMA
        self.logger.info("Đang khởi tạo mô hình LiteFatV1 (KSS 1-9)...")
        use_dann = bool(getattr(self.cfg, "use_dann", False))
        self.model = LiteFatV1(
            num_nodes=self.cfg.num_nodes,
            feature_dim=self.cfg.feature_dim,
            mobilenet_proj_hidden=self.cfg.mobilenet_proj_hidden,
            node_embed_dim=self.cfg.node_embed_dim,
            residual_channels=self.cfg.residual_channels,
            dilation_channels=self.cfg.dilation_channels,
            skip_channels=self.cfg.skip_channels,
            end_channels=self.cfg.end_channels,
            num_classes=self.cfg.num_classes,
            kss_min=self.cfg.kss_min,
            kss_max=self.cfg.kss_max,
            fps=self.cfg.fps,
            window_sec=self.cfg.window_sec,
            stride_sec=self.cfg.stride_sec,
            num_blocks=self.cfg.num_blocks,
            layers_per_block=self.cfg.layers_per_block,
            kernel_size=self.cfg.kernel_size,
            pretrained_backbone=self.cfg.pretrained_backbone,
            dropout=self.cfg.dropout,
            freeze_backbone=(self.cfg.freeze_backbone_epochs > 0),
            vertical_pairs=self.cfg.vertical_pairs,
            cross_region_pairs=self.cfg.cross_region_pairs,
            static_shape_ratio=self.cfg.static_shape_ratio,
            use_dann=use_dann,
            eye_vel_smooth=bool(getattr(self.cfg, "eye_vel_smooth", False)),
            eye_y_scale_mode=str(getattr(self.cfg, "eye_y_scale_mode", "v6")),
            window_attn_weighted_logits=bool(getattr(self.cfg, "window_attn_weighted_logits", False)),
        ).to(self.device)
        self.model.set_visual_context_enabled(self.cfg.use_visual_context)
        self.model.set_dann_enabled(use_dann)

        self.ema = ModelEMA(self.model, decay=self.cfg.ema_decay)
        self.early_stop_counter = 0

        stats = self.model.get_trainable_params_count()
        self.logger.info(
            f"Tổng số tham số: {stats['total_params']:,} | Khả học: {stats['trainable_params']:,} "
            f"(MobileNetV3: {stats['backbone_trainable']:,} / {stats['backbone_total']:,})"
        )

        # 6. Khởi tạo Hàm mất mát Hybrid KSS Loss
        self.criterion = LiteFatLoss(
            num_classes=self.cfg.num_classes,
            kss_min=self.cfg.kss_min,
            kss_max=self.cfg.kss_max,
            clip_reg_weight=self.cfg.clip_reg_weight,
            clip_cls_weight=self.cfg.clip_cls_weight,
            frame_loss_weight=self.cfg.frame_loss_weight,
            window_loss_weight=self.cfg.window_loss_weight,
            window_smooth_weight=self.cfg.window_smooth_weight,
            boundary_margin_weight=float(getattr(self.cfg, "boundary_margin_weight", 0.15)),
            uta_ordinal_sigma=float(getattr(self.cfg, "uta_ordinal_sigma", 1.25)),
            dann_weight=float(getattr(self.cfg, "dann_weight", 0.15)),
            huber_beta=self.cfg.huber_beta,
            ordinal_sigma=self.cfg.ordinal_sigma,
            label_smoothing=self.cfg.label_smoothing,
            alert_kss_threshold=self.cfg.alert_kss_threshold,
            mid_kss_threshold=self.cfg.mid_kss_threshold,
            topk_mid_ratio=self.cfg.topk_mid_ratio,
            topk_drowsy_ratio=self.cfg.topk_drowsy_ratio,
            fps=self.cfg.fps,
            window_sec=self.cfg.window_sec,
            stride_sec=self.cfg.stride_sec,
            smooth_margin=self.cfg.smooth_margin,
            uta_reg_mode=str(getattr(self.cfg, "uta_reg_mode", "point")),
            uta_interval_center_weight=float(getattr(self.cfg, "uta_interval_center_weight", 0.50)),
            boundary_drowsy_min=float(getattr(self.cfg, "boundary_drowsy_min", 6.70)),
            boundary_nondrowsy_max=float(getattr(self.cfg, "boundary_nondrowsy_max", 6.20)),
        ).to(self.device)

        # Biến theo dõi thực nghiệm
        self.start_epoch = 1
        self.best_val_score = -1.0
        self.best_val_mae = float("inf")
        self.best_val_f1 = -1.0
        self.best_epoch = -1
        self.history: List[Dict[str, Any]] = []
        self.pretrained_param_names: Set[str] = set()

        # Warm-start nếu được yêu cầu và không resume
        if self.cfg.pretrained_from and not self.cfg.resume:
            self.load_pretrained_weights(self.cfg.pretrained_from)

        # 7. Optimizer, Scheduler & GradScaler AMP
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        amp_enabled = self.cfg.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        # Khôi phục checkpoint nếu có cfg.resume
        if self.cfg.resume:
            self.load_checkpoint(self.cfg.resume)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Thiết lập AdamW với tốc độ học phân biệt (Differential LR) và tách nhóm no_decay."""
        no_decay_keywords = (
            "norm",
            "bn",
            "node_embed",
            "E1",
            "E2",
            "alpha",
            "temp_scale",
            "kss_mix_logit",
            "res_scale",
            "init_",
        )

        def _is_no_decay(param_name: str, p: torch.Tensor) -> bool:
            if p.ndim <= 1 or param_name.endswith(".bias"):
                return True
            return any(kw in param_name for kw in no_decay_keywords)

        buckets: Dict[str, List[torch.Tensor]] = {
            "pretrained_graph_decay": [],
            "pretrained_graph_no_decay": [],
            "new_graph_decay": [],
            "new_graph_no_decay": [],
            "backbone_decay": [],
            "backbone_no_decay": [],
        }

        pretrained_scale = float(self.cfg.pretrained_lr_scale)
        has_pretrained = bool(self.pretrained_param_names) and (pretrained_scale < 1.0)

        for name, param in self.model.named_parameters():
            no_decay = _is_no_decay(name, param)
            suffix = "no_decay" if no_decay else "decay"
            if "context_encoder.features" in name:
                buckets[f"backbone_{suffix}"].append(param)
                continue
            if not param.requires_grad:
                continue
            if has_pretrained and name in self.pretrained_param_names:
                buckets[f"pretrained_graph_{suffix}"].append(param)
            else:
                buckets[f"new_graph_{suffix}"].append(param)

        param_groups: List[Dict[str, Any]] = []
        group_specs = [
            ("pretrained_graph_decay", self.cfg.lr * pretrained_scale, self.cfg.weight_decay, "pretrained_graph"),
            ("pretrained_graph_no_decay", self.cfg.lr * pretrained_scale, 0.0, "pretrained_graph_no_decay"),
            ("new_graph_decay", self.cfg.lr, self.cfg.weight_decay, "new_graph"),
            ("new_graph_no_decay", self.cfg.lr, 0.0, "new_graph_no_decay"),
        ]
        for key, lr_val, wd_val, gname in group_specs:
            if buckets[key]:
                param_groups.append({
                    "params": buckets[key],
                    "lr": lr_val,
                    "weight_decay": wd_val,
                    "name": gname,
                })

        if self.cfg.use_visual_context and (buckets["backbone_decay"] or buckets["backbone_no_decay"]):
            initial_backbone_lr = (
                self.cfg.backbone_lr if self.start_epoch > self.cfg.freeze_backbone_epochs else 0.0
            )
            if buckets["backbone_decay"]:
                param_groups.append({
                    "params": buckets["backbone_decay"],
                    "lr": initial_backbone_lr,
                    "weight_decay": self.cfg.weight_decay,
                    "name": "backbone",
                })
            if buckets["backbone_no_decay"]:
                param_groups.append({
                    "params": buckets["backbone_no_decay"],
                    "lr": initial_backbone_lr,
                    "weight_decay": 0.0,
                    "name": "backbone_no_decay",
                })

        return torch.optim.AdamW(param_groups)

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Thiết lập lịch trình học Linear Warmup + Cosine Annealing."""
        warmup_epochs = max(0, int(self.cfg.warmup_epochs))
        total_epochs = max(1, int(self.cfg.epochs))

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def load_checkpoint(self, checkpoint_path: str | Path) -> Dict[str, Any]:
        """Nạp checkpoint thông qua hàm phụ trợ `load_checkpoint` trong ultis.py."""
        info = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            ema=self.ema,
            save_dir=self.save_dir,
            cfg=self.cfg,
            device=self.device,
            logger=self.logger,
        )
        self.start_epoch = info["start_epoch"]
        self.best_epoch = info["best_epoch"]
        self.best_val_score = info["best_val_score"]
        self.best_val_mae = info["best_val_mae"]
        self.best_val_f1 = info["best_val_f1"]
        self.early_stop_counter = int(info.get("early_stop_counter", 0))
        if info.get("scheduler") is not None:
            self.scheduler = info["scheduler"]
        if isinstance(info.get("history"), list):
            self.history = info["history"]
        return info

    def load_pretrained_weights(self, pretrained_path: str | Path) -> Set[str]:
        """Nạp trọng số warm-start thông qua `load_pretrained_weights` trong ultis.py."""
        self.pretrained_param_names = load_pretrained_weights(
            pretrained_path=pretrained_path,
            model=self.model,
            use_visual_context=self.cfg.use_visual_context,
            device=self.device,
            logger=self.logger,
        )
        return self.pretrained_param_names

    @staticmethod
    def _unpack_batch(
            batch: Tuple[Any, ...],
    ) -> Tuple[
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        batch_frames = batch[0]
        batch_landmarks = batch[1]
        batch_labels = batch[2]
        batch_lengths = batch[3] if len(batch) >= 4 else None
        sample_weights = None
        domain_targets = None
        if len(batch) >= 5 and isinstance(batch[4], (list, tuple)) and batch[4] and isinstance(batch[4][0], dict):
            metas = batch[4]
            sample_weights = torch.tensor(
                [max(1e-6, float(m.get("sample_weight", 1.0))) for m in metas],
                dtype=torch.float32,
            )
            domain_targets = torch.tensor(
                [int(m.get("domain_id", -1)) for m in metas],
                dtype=torch.long,
            )
        return batch_frames, batch_landmarks, batch_labels, batch_lengths, sample_weights, domain_targets

    def _step_optimizer(self) -> None:
        """Thực hiện gradient clipping (nếu bật), bước tối ưu AMP và cập nhật EMA."""
        if self.cfg.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.update(self.model)
        self.optimizer.zero_grad(set_to_none=True)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Thực thi một epoch huấn luyện cho LiteFatV1 (hỗ trợ tích lũy gradient)."""
        self.model.train()

        # Đóng/mở khóa backbone MobileNetV3 theo epoch (chỉ áp dụng khi bật ngữ cảnh thị giác)
        if self.cfg.use_visual_context:
            if epoch <= self.cfg.freeze_backbone_epochs:
                self.model.set_backbone_trainable(False)
                for g in self.optimizer.param_groups:
                    if g.get("name") == "backbone":
                        g["lr"] = 0.0
            else:
                self.model.set_backbone_trainable(True, unfreeze_last_n_blocks=self.cfg.unfreeze_last_n_blocks)
                base_lr_ref = self.optimizer.param_groups[0]["lr"]
                base_ratio = base_lr_ref / max(1e-9, self.cfg.lr)
                for g in self.optimizer.param_groups:
                    if g.get("name") == "backbone":
                        g["lr"] = self.cfg.backbone_lr * base_ratio
        else:
            self.model.set_backbone_trainable(False)

        total_loss = 0.0
        comp_sums: Dict[str, float] = {
            "loss_clip_reg": 0.0,
            "loss_boundary": 0.0,
            "loss_clip_cls": 0.0,
            "loss_frame": 0.0,
            "loss_window": 0.0,
            "loss_smooth": 0.0,
            "loss_domain": 0.0,
        }
        all_targets: List[float] = []
        all_pred_kss: List[float] = []
        all_pred_logits: List[np.ndarray] = []

        total_batches = len(self.train_loader)
        num_batches = max(1, total_batches)
        total_train_steps = max(1, int(self.cfg.epochs) * num_batches)
        use_dann = bool(getattr(self.cfg, "use_dann", False))
        dann_max_alpha = float(getattr(self.cfg, "dann_max_alpha", 0.25))
        # Khi đã bật subject_balanced_sampling thì sampler đã cân bằng lớp/miền, tránh nhân chồng trọng số loss
        use_sample_weights = bool(getattr(self.cfg, "use_manifest_sample_weights", True)) and (
            not bool(getattr(self.cfg, "subject_balanced_sampling", False))
        )
        accum_steps = max(1, int(getattr(self.cfg, "grad_accum_steps", 1)))
        valid_steps = 0
        opt_steps = 0
        accum_count = 0
        current_accum_target = min(accum_steps, num_batches)
        start_time = time.time()

        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:02d}/{self.cfg.epochs:02d} [Train]",
            dynamic_ncols=True,
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            # Tính kích thước cửa sổ tích lũy hiện tại để giữ đúng trung bình gradient ở nhóm lẻ cuối epoch
            if accum_count == 0:
                current_accum_target = min(accum_steps, max(1, total_batches - batch_idx))

            if use_dann:
                progress = min(1.0, max(0.0, float((epoch - 1) * num_batches + batch_idx) / float(total_train_steps)))
                dann_alpha = dann_max_alpha * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
                self.model.set_dann_alpha(dann_alpha)
            else:
                self.model.set_dann_alpha(0.0)

            batch_frames, batch_landmarks, batch_labels, batch_lengths, batch_weights, batch_domains = self._unpack_batch(batch)
            batch_landmarks = batch_landmarks.to(self.device, non_blocking=True)
            batch_labels = batch_labels.to(self.device, non_blocking=True)
            if batch_frames is not None:
                batch_frames = batch_frames.to(self.device, non_blocking=True)
            if batch_lengths is not None:
                batch_lengths = batch_lengths.to(self.device, non_blocking=True)
            batch_weights = batch_weights.to(self.device, non_blocking=True) if (use_sample_weights and batch_weights is not None) else None
            batch_domains = batch_domains.to(self.device, non_blocking=True) if (use_dann and batch_domains is not None) else None

            with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                out_dict = self.model(
                    video_frames=batch_frames,
                    landmarks=batch_landmarks,
                    lengths=batch_lengths,
                    return_dict=True,
                )
                loss, loss_details = self.criterion(
                    out_dict,
                    batch_labels,
                    lengths=batch_lengths,
                    sample_weights=batch_weights,
                    domain_targets=batch_domains,
                    return_dict=True,
                )

            if not torch.isfinite(loss):
                self.logger.warning(
                    f"Epoch {epoch:02d} | Batch {batch_idx + 1}/{num_batches}: Loss không hữu hạn ({loss.item()}). Bỏ qua bước cập nhật."
                )
                del out_dict, loss, loss_details, batch_frames, batch_landmarks, batch_labels, batch_lengths, batch_weights, batch_domains
                continue

            loss_to_backward = loss if current_accum_target == 1 else (loss / float(current_accum_target))
            self.scaler.scale(loss_to_backward).backward()
            accum_count += 1

            last_loss = float(loss.item())
            valid_steps += 1
            total_loss += last_loss
            for k in comp_sums:
                comp_sums[k] += float(loss_details.get(k, 0.0))

            clip_kss_np = out_dict["clip_kss"].detach().float().cpu().numpy()
            clip_logits_np = out_dict["clip_logits"].detach().float().cpu().numpy()
            targets_np = batch_labels.detach().float().cpu().numpy()

            all_pred_kss.extend(clip_kss_np.tolist())
            all_targets.extend(targets_np.tolist())
            all_pred_logits.append(clip_logits_np)

            del out_dict, loss, loss_details, loss_to_backward
            del batch_frames, batch_landmarks, batch_labels, batch_lengths, batch_weights, batch_domains

            if accum_count >= current_accum_target:
                self._step_optimizer()
                accum_count = 0
                opt_steps += 1

            current_lr = self.optimizer.param_groups[0]["lr"]
            postfix = {
                "loss": f"{last_loss:.4f}",
                "avg_loss": f"{total_loss / valid_steps:.4f}",
                "lr": f"{current_lr:.2e}",
            }
            if accum_steps > 1:
                postfix["opt_steps"] = opt_steps
            pbar.set_postfix(postfix)

        # Flush phần gradient hợp lệ còn lại nếu kết thúc epoch mà vẫn còn tích lũy
        if accum_count > 0:
            self._step_optimizer()
            opt_steps += 1

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        logits_cat = np.concatenate(all_pred_logits, axis=0) if all_pred_logits else None
        metrics = compute_kss_metrics(
            np.array(all_targets),
            np.array(all_pred_kss),
            pred_logits=logits_cat,
            kss_min=self.cfg.kss_min,
            kss_max=self.cfg.kss_max,
        )
        denom = max(1, valid_steps)
        metrics["loss"] = total_loss / denom
        for k, v in comp_sums.items():
            metrics[k] = v / denom
        metrics["opt_steps"] = float(opt_steps)
        metrics["time"] = elapsed
        return metrics

    @torch.no_grad()
    def _evaluate_epoch(
            self,
            dataloader: Any,
            split_name: str = "val",
            epoch: Optional[int] = None,
            return_raw: bool = False,
            use_ema: bool = False,
    ) -> Any:
        """Đánh giá mô hình trên tập validation hoặc test (hỗ trợ EMA & phân rã theo dataset)."""
        backup = self.ema.apply_shadow(self.model) if (use_ema and hasattr(self, "ema")) else None
        self.model.eval()

        total_loss = 0.0
        all_targets: List[float] = []
        all_pred_kss: List[float] = []
        all_pred_logits: List[np.ndarray] = []
        num_batches = max(1, len(dataloader))
        start_time = time.time()

        desc = (
            f"Epoch {epoch:02d}/{self.cfg.epochs:02d} [{split_name.capitalize()}{'(EMA)' if use_ema else ''}]"
            if epoch is not None
            else f"[{split_name.capitalize()}{'(EMA)' if use_ema else ''}]"
        )
        pbar = tqdm(dataloader, desc=desc, dynamic_ncols=True, leave=False)

        try:
            for batch_idx, batch in enumerate(pbar):
                batch_frames, batch_landmarks, batch_labels, batch_lengths, _, _ = self._unpack_batch(batch)
                batch_landmarks = batch_landmarks.to(self.device, non_blocking=True)
                batch_labels = batch_labels.to(self.device, non_blocking=True)
                if batch_frames is not None:
                    batch_frames = batch_frames.to(self.device, non_blocking=True)
                if batch_lengths is not None:
                    batch_lengths = batch_lengths.to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                    out_dict = self.model(
                        video_frames=batch_frames,
                        landmarks=batch_landmarks,
                        lengths=batch_lengths,
                        return_dict=True,
                    )
                    loss = self.criterion(
                        out_dict,
                        batch_labels,
                        lengths=batch_lengths,
                    )

                total_loss += float(loss.item())
                all_pred_kss.extend(out_dict["clip_kss"].float().cpu().numpy().tolist())
                all_targets.extend(batch_labels.float().cpu().numpy().tolist())
                all_pred_logits.append(out_dict["clip_logits"].float().cpu().numpy())

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
                })
        finally:
            if backup is not None:
                self.ema.restore(self.model, backup)

        elapsed = time.time() - start_time
        targets_np = np.array(all_targets, dtype=np.float64)
        pred_kss_np = np.array(all_pred_kss, dtype=np.float64)
        logits_np = np.concatenate(all_pred_logits, axis=0) if all_pred_logits else np.empty((0, self.cfg.num_classes))

        metrics = compute_kss_metrics(
            targets_np,
            pred_kss_np,
            pred_logits=logits_np if len(logits_np) > 0 else None,
            kss_min=self.cfg.kss_min,
            kss_max=self.cfg.kss_max,
        )
        metrics["loss"] = total_loss / max(1, len(dataloader))
        metrics["time"] = elapsed

        # Phân rã chỉ số theo từng bộ dữ liệu con (UL-DD, UTA-RLDD, ...)
        samples = getattr(dataloader.dataset, "samples", [])
        per_dataset_metrics: Dict[str, Dict[str, float]] = {}
        if len(samples) == len(targets_np) and len(samples) > 0:
            ds_arr = np.array([str(s.get("dataset", "unknown")) for s in samples])
            for d_name in np.unique(ds_arr):
                m_d = (ds_arr == d_name)
                d_metrics = compute_kss_metrics(
                    targets_np[m_d],
                    pred_kss_np[m_d],
                    pred_logits=logits_np[m_d] if len(logits_np) == len(targets_np) else None,
                    kss_min=self.cfg.kss_min,
                    kss_max=self.cfg.kss_max,
                )
                per_dataset_metrics[str(d_name)] = {
                    "mae": d_metrics["mae"],
                    "rmse": d_metrics["rmse"],
                    "within_1_acc": d_metrics["within_1_acc"],
                    "qwk": d_metrics["qwk"],
                    "auc": d_metrics["auc"],
                    "f1": d_metrics["f1"],
                    "num_samples": int(np.sum(m_d)),
                }
        metrics["per_dataset_metrics"] = per_dataset_metrics

        if return_raw:
            return metrics, targets_np, pred_kss_np, logits_np
        return metrics

    def _log_tensorboard(self, epoch: int, train_m: Dict[str, float], val_m: Dict[str, float], lr: float) -> None:
        """Ghi toàn bộ chỉ số KSS & loss vào TensorBoard."""
        self.writer.add_scalar("Loss/Train", train_m["loss"], epoch)
        self.writer.add_scalar("Loss/Val", val_m["loss"], epoch)
        for comp in ("loss_clip_reg", "loss_boundary", "loss_clip_cls", "loss_frame", "loss_window", "loss_smooth", "loss_domain"):
            if comp in train_m:
                self.writer.add_scalar(f"Loss/{comp}", train_m[comp], epoch)

        self.writer.add_scalar("KSS_MAE/Train", train_m["mae"], epoch)
        self.writer.add_scalar("KSS_MAE/Val", val_m["mae"], epoch)
        self.writer.add_scalar("KSS_RMSE/Train", train_m["rmse"], epoch)
        self.writer.add_scalar("KSS_RMSE/Val", val_m["rmse"], epoch)
        self.writer.add_scalar("KSS_Within1/Train", train_m["within_1_acc"], epoch)
        self.writer.add_scalar("KSS_Within1/Val", val_m["within_1_acc"], epoch)
        self.writer.add_scalar("KSS_ExactAcc/Val", val_m["exact_acc"], epoch)
        self.writer.add_scalar("KSS_QWK/Train", train_m["qwk"], epoch)
        self.writer.add_scalar("KSS_QWK/Val", val_m["qwk"], epoch)
        self.writer.add_scalar("KSS_RegimeAcc/Val", val_m["regime_acc"], epoch)
        self.writer.add_scalar("Drowsy_F1/Val", val_m["f1"], epoch)
        self.writer.add_scalar("Drowsy_AUC/Val", val_m["auc"], epoch)
        self.writer.add_scalar("Score/Val", val_m["val_score"], epoch)
        self.writer.add_scalar("LR", lr, epoch)

        for d_name, d_m in val_m.get("per_dataset_metrics", {}).items():
            self.writer.add_scalar(f"KSS_MAE/Val_{d_name}", d_m["mae"], epoch)
            self.writer.add_scalar(f"KSS_Within1/Val_{d_name}", d_m["within_1_acc"], epoch)
            self.writer.add_scalar(f"KSS_QWK/Val_{d_name}", d_m["qwk"], epoch)

    def train(self) -> Dict[str, float]:
        """Vòng lặp huấn luyện chính qua toàn bộ các epoch."""
        if self.start_epoch > self.cfg.epochs:
            self.logger.warning(
                f"Checkpoint đã hoàn thành Epoch {self.start_epoch - 1}, trong khi cfg.epochs={self.cfg.epochs}. "
                "Vui lòng tăng 'epochs' trong TrainConfig để tiếp tục train."
            )
            test_m = self.evaluate_test()
            self.close()
            return test_m

        self.logger.info("\n>>> BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN LITEFAT_V1 (KSS 1-9) <<<")
        if self.start_epoch > 1:
            self.logger.info(f"Tiếp tục huấn luyện từ Epoch {self.start_epoch:02d} đến {self.cfg.epochs:02d}")

        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            train_m = self._train_epoch(epoch)
            val_m = self._evaluate_epoch(
                self.val_loader,
                split_name="val",
                epoch=epoch,
                use_ema=self.cfg.use_ema_eval,
            )

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            per_ds_str = " | ".join(
                f"{k}(MAE={v['mae']:.3f}, W1={v['within_1_acc']*100:.1f}%, QWK={v['qwk']:.3f})"
                for k, v in val_m.get("per_dataset_metrics", {}).items()
            )
            self.logger.info(
                f"Epoch [{epoch:02d}/{self.cfg.epochs:02d}]\n"
                f"  Train: Loss={train_m['loss']:.4f} (Dom={train_m.get('loss_domain', 0.0):.3f}) | MAE={train_m['mae']:.4f} | RMSE={train_m['rmse']:.4f} | "
                f"Within1={train_m['within_1_acc']*100:.1f}% | QWK={train_m['qwk']:.4f} | "
                f"Pred=[{train_m['pred_min']:.2f}..{train_m['pred_max']:.2f}, {train_m['pred_mean']:.2f}±{train_m['pred_std']:.2f}]\n"
                f"  Val  : Loss={val_m['loss']:.4f} | MAE={val_m['mae']:.4f} | RMSE={val_m['rmse']:.4f} | "
                f"Within1={val_m['within_1_acc']*100:.1f}% | Exact={val_m['exact_acc']*100:.1f}% | "
                f"QWK={val_m['qwk']:.4f} | RegimeAcc={val_m['regime_acc']*100:.1f}% | "
                f"DrowsyAUC={val_m['auc']:.4f} | Score={val_m['val_score']:.4f} | "
                f"Pred=[{val_m['pred_min']:.2f}..{val_m['pred_max']:.2f}, {val_m['pred_mean']:.2f}±{val_m['pred_std']:.2f}]\n"
                f"  Val Per-Dataset: {per_ds_str or 'N/A'}\n"
                f"  LR: {current_lr:.2e} | Time: {train_m['time']:.1f}s"
            )

            self._log_tensorboard(epoch, train_m, val_m, current_lr)

            self.history.append({
                "epoch": epoch,
                "train": train_m,
                "val": val_m,
                "lr": current_lr,
            })

            cur_val_score = float(val_m["val_score"])
            is_new_best = cur_val_score > self.best_val_score
            if is_new_best:
                self.best_val_score = cur_val_score
                self.best_val_mae = float(val_m["mae"])
                self.best_val_f1 = float(val_m["f1"])
                self.best_epoch = epoch
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "ema_state_dict": self.ema.shadow,
                "ema_num_updates": self.ema.num_updates,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "best_epoch": self.best_epoch,
                "best_val_score": self.best_val_score,
                "best_val_mae": self.best_val_mae,
                "best_val_f1": self.best_val_f1,
                "early_stop_counter": self.early_stop_counter,
                "val_metrics": val_m,
                "train_metrics": train_m,
                "history": self.history,
                "config": self.cfg.to_dict(),
            }
            atomic_save(checkpoint_payload, self.save_dir / "last.pt")

            if self.cfg.save_freq > 0 and (epoch % self.cfg.save_freq == 0):
                atomic_save(checkpoint_payload, self.save_dir / f"epoch_{epoch:02d}.pt")

            if is_new_best:
                atomic_save(checkpoint_payload, self.save_dir / "best.pt")
                with open(self.save_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                    json.dump({"best_epoch": epoch, "metrics": val_m}, f, indent=2)
                self.logger.info(
                    f"===> ĐÃ LƯU BEST.PT TẠI EPOCH {epoch} "
                    f"(Score: {self.best_val_score:.4f} | MAE: {self.best_val_mae:.4f} | QWK: {val_m['qwk']:.4f})"
                )

            if self.early_stop_counter >= int(self.cfg.early_stop_patience):
                self.logger.info(
                    f"===> DỪNG SỚM (Early Stopping) tại Epoch {epoch} "
                    f"vì val_score không cải thiện sau {self.early_stop_counter} epoch liên tiếp."
                )
                break

        self.logger.info("\n>>> HOÀN THÀNH HUẤN LUYỆN! <<<")
        self.logger.info(
            f"Checkpoint tốt nhất: Epoch {self.best_epoch} "
            f"(Score: {self.best_val_score:.4f} | MAE: {self.best_val_mae:.4f})"
        )

        test_metrics = self.evaluate_test()
        self.close()
        return test_metrics

    def close(self) -> None:
        """Đóng TensorBoard writer và các file handler của logger."""
        if hasattr(self, "writer") and self.writer is not None:
            try:
                self.writer.close()
            except Exception:
                pass
        if hasattr(self, "logger") and self.logger is not None:
            for h in list(self.logger.handlers):
                try:
                    h.close()
                except Exception:
                    pass
                self.logger.removeHandler(h)

    def evaluate_test(self, checkpoint_path: Optional[str | Path] = None) -> Dict[str, Any]:
        """Nạp checkpoint tốt nhất (best.pt) và đánh giá trên tập Test độc lập."""
        ckpt_target = Path(checkpoint_path) if checkpoint_path is not None else (self.save_dir / "best.pt")
        if not ckpt_target.is_file():
            fallback_last = self.save_dir / "last.pt"
            if fallback_last.is_file():
                ckpt_target = fallback_last
            else:
                self.logger.warning(f"Không tìm thấy checkpoint {ckpt_target} để đánh giá test.")
                return {}

        checkpoint = torch.load(ckpt_target, map_location=self.device, weights_only=False)
        eval_epoch = int(checkpoint.get("epoch", self.best_epoch))

        if self.cfg.use_ema_eval and "ema_state_dict" in checkpoint:
            ema_sd = checkpoint["ema_state_dict"]
            if isinstance(ema_sd, dict) and "shadow" in ema_sd and isinstance(ema_sd["shadow"], dict):
                ema_sd = ema_sd["shadow"]
            self.model.load_state_dict(ema_sd, strict=False)
            self.logger.info(f"\n--- ĐÁNH GIÁ TEST BẰNG EMA CHECKPOINT ({ckpt_target.name}, Epoch {eval_epoch}) ---")
        elif "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.logger.info(f"\n--- ĐÁNH GIÁ TEST BẰNG CHECKPOINT ({ckpt_target.name}, Epoch {eval_epoch}) ---")

        test_metrics, test_targets, test_pred_kss, test_logits = self._evaluate_epoch(
            self.test_loader,
            split_name="test",
            return_raw=True,
            use_ema=False,
        )

        self.logger.info(
            f"[TEST] Loss={test_metrics['loss']:.4f} | MAE={test_metrics['mae']:.4f} | "
            f"RMSE={test_metrics['rmse']:.4f} | Within1={test_metrics['within_1_acc']*100:.2f}% | "
            f"Exact={test_metrics['exact_acc']*100:.2f}% | QWK={test_metrics['qwk']:.4f} | "
            f"Pearson={test_metrics['pearson_r']:.4f} | RegimeAcc={test_metrics['regime_acc']*100:.2f}% | "
            f"DrowsyAUC={test_metrics['auc']:.4f} | DrowsyF1={test_metrics['f1']:.4f}"
        )

        test_samples = getattr(self.test_loader.dataset, "samples", [])
        per_sub_test: Dict[str, Dict[str, float]] = {}
        if len(test_samples) == len(test_targets) and len(test_samples) > 0:
            for d_name, d_m in test_metrics.get("per_dataset_metrics", {}).items():
                self.logger.info(
                    f"   [Dataset {d_name}] MAE={d_m['mae']:.4f} | RMSE={d_m['rmse']:.4f} | "
                    f"Within1={d_m['within_1_acc']*100:.2f}% | QWK={d_m['qwk']:.4f} | "
                    f"AUC={d_m['auc']:.4f} (N={d_m['num_samples']})"
                )

            test_ds_arr = np.array([str(s.get("dataset", "unknown")) for s in test_samples])
            test_sub_arr = np.array([str(s.get("subject_id", "unknown")) for s in test_samples])
            for sub_id in np.unique(test_sub_arr):
                m_s = (test_sub_arr == sub_id)
                s_ds = test_ds_arr[m_s][0]
                s_m = compute_kss_metrics(
                    test_targets[m_s],
                    test_pred_kss[m_s],
                    pred_logits=test_logits[m_s] if len(test_logits) == len(test_targets) else None,
                    kss_min=self.cfg.kss_min,
                    kss_max=self.cfg.kss_max,
                )
                per_sub_test[f"{s_ds}:{sub_id}"] = {
                    "mae": s_m["mae"],
                    "within_1_acc": s_m["within_1_acc"],
                    "qwk": s_m["qwk"],
                    "auc": s_m["auc"],
                    "num_samples": int(np.sum(m_s)),
                }

        output_payload = {
            "best_epoch": self.best_epoch if self.best_epoch > 0 else eval_epoch,
            "checkpoint": str(ckpt_target),
            "test_metrics": test_metrics,
            "per_subject_test": per_sub_test,
            "bootstrap_ci_95": self._compute_bootstrap_ci(
                targets_np=test_targets,
                pred_kss_np=test_pred_kss,
                logits_np=test_logits,
                samples=test_samples,
            ),
        }
        with open(self.save_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

        self._save_predictions_csv(
            self.save_dir / "test_predictions.csv",
            samples=test_samples,
            targets_np=test_targets,
            pred_kss_np=test_pred_kss,
            logits_np=test_logits,
        )

        if self.val_loader is not None and len(getattr(self.val_loader, "dataset", [])) > 0:
            _, val_targets, val_pred_kss, val_logits = self._evaluate_epoch(
                self.val_loader,
                split_name="val",
                return_raw=True,
                use_ema=False,
            )
            val_samples = getattr(self.val_loader.dataset, "samples", [])
            self._save_predictions_csv(
                self.save_dir / "val_predictions.csv",
                samples=val_samples,
                targets_np=val_targets,
                pred_kss_np=val_pred_kss,
                logits_np=val_logits,
            )

        ov_ci = output_payload["bootstrap_ci_95"].get("overall", {})
        if ov_ci:
            mae_ci = ov_ci.get("mae") or [0.0, 0.0]
            qwk_ci = ov_ci.get("qwk") or [0.0, 0.0]
            auc_ci = ov_ci.get("auc") or [0.0, 0.0]
            self.logger.info(
                f"[TEST Bootstrap 95% CI] MAE=[{mae_ci[0]:.3f}, {mae_ci[1]:.3f}] | "
                f"QWK=[{qwk_ci[0]:.3f}, {qwk_ci[1]:.3f}] | AUC=[{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]"
            )

        return test_metrics

    def _write_run_meta(self) -> None:
        """Ghi lại metadata truy vết môi trường, git commit và các cờ ablation vào run_meta.json."""
        repo_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
        git_commit = None
        git_dirty_stat = None
        try:
            res_hash = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=3,
            )
            if res_hash.returncode == 0:
                git_commit = res_hash.stdout.strip()
            res_stat = subprocess.run(
                ["git", "diff", "--stat"],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=3,
            )
            if res_stat.returncode == 0:
                git_dirty_stat = res_stat.stdout.strip()
        except Exception:
            pass

        ablation_keys = (
            "uta_reg_mode",
            "uta_interval_center_weight",
            "boundary_drowsy_min",
            "boundary_nondrowsy_max",
            "sampler_uta_quota",
            "sampler_weight_exponents",
            "eye_vel_smooth",
            "eye_y_scale_mode",
            "window_attn_weighted_logits",
            "boundary_margin_weight",
            "dann_weight",
            "dann_max_alpha",
            "static_shape_ratio",
        )
        meta = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git_commit": git_commit,
            "git_dirty_stat": git_dirty_stat,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "seed": int(self.cfg.seed),
            "ablation_flags": {k: getattr(self.cfg, k, None) for k in ablation_keys},
        }
        with open(self.save_dir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _logits_to_drowsy_prob(self, pred_kss_np: np.ndarray, logits_np: np.ndarray) -> np.ndarray:
        """Tính xác suất buồn ngủ (KSS >= 7) từ logits hoặc fallback tuyến tính từ pred_kss."""
        if logits_np is not None and len(logits_np) == len(pred_kss_np) and logits_np.ndim == 2 and logits_np.shape[1] >= 2:
            x = logits_np.astype(np.float64)
            x = x - np.max(x, axis=-1, keepdims=True)
            exp_x = np.exp(x)
            probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True).clip(min=1e-12)
            start_idx = max(0, min(probs.shape[1] - 1, int(round(7.0 - self.cfg.kss_min))))
            return probs[:, start_idx:].sum(axis=-1)
        span = max(1e-6, float(self.cfg.kss_max - self.cfg.kss_min))
        return np.clip((pred_kss_np.astype(np.float64) - self.cfg.kss_min) / span, 0.0, 1.0)

    def _save_predictions_csv(
            self,
            out_path: Path,
            samples: List[Dict[str, Any]],
            targets_np: np.ndarray,
            pred_kss_np: np.ndarray,
            logits_np: np.ndarray,
    ) -> None:
        """Lưu dự đoán từng mẫu ra tệp CSV phục vụ chẩn đoán sai số."""
        drowsy_probs = self._logits_to_drowsy_prob(pred_kss_np, logits_np)
        min_int = int(round(self.cfg.kss_min))
        max_int = int(round(self.cfg.kss_max))
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "index", "key", "dataset", "subject_id", "video_id",
                "target_kss", "pred_kss", "abs_err", "pred_kss_int", "drowsy_prob",
            ])
            for i in range(len(targets_np)):
                s = samples[i] if i < len(samples) else {}
                t_val = float(targets_np[i])
                p_val = float(pred_kss_np[i])
                p_int = int(np.clip(np.rint(p_val), min_int, max_int))
                writer.writerow([
                    i,
                    str(s.get("key", "")),
                    str(s.get("dataset", "")),
                    str(s.get("subject_id", "")),
                    str(s.get("video_id", "")),
                    f"{t_val:.4f}",
                    f"{p_val:.4f}",
                    f"{abs(p_val - t_val):.4f}",
                    p_int,
                    f"{float(drowsy_probs[i]):.6f}",
                ])

    def _compute_bootstrap_ci(
            self,
            targets_np: np.ndarray,
            pred_kss_np: np.ndarray,
            logits_np: np.ndarray,
            samples: List[Dict[str, Any]],
            n_boot: int = 1000,
    ) -> Dict[str, Any]:
        """Tính Bootstrap 95% CI (MAE, QWK, AUC) trên tổng thể và từng bộ dữ liệu."""
        n = len(targets_np)
        if n == 0:
            return {}
        drowsy_probs = self._logits_to_drowsy_prob(pred_kss_np, logits_np)
        min_int = int(round(self.cfg.kss_min))
        max_int = int(round(self.cfg.kss_max))
        rng = np.random.default_rng(int(self.cfg.seed))

        def _boot_slice(mask: np.ndarray) -> Dict[str, Any]:
            y = targets_np[mask]
            p = pred_kss_np[mask]
            pr = drowsy_probs[mask]
            m_n = len(y)
            if m_n == 0:
                return {}
            maes: List[float] = []
            qwks: List[float] = []
            aucs: List[float] = []
            for _ in range(n_boot):
                idx = rng.integers(0, m_n, size=m_n)
                yb = y[idx]
                pb = p[idx]
                prb = pr[idx]
                maes.append(float(np.mean(np.abs(pb - yb))))
                qwks.append(quadratic_weighted_kappa(yb, pb, min_rating=min_int, max_rating=max_int))
                ybin = (yb >= 7.0).astype(int)
                if len(np.unique(ybin)) > 1:
                    aucs.append(binary_auc(ybin, prb))
            return {
                "n": int(m_n),
                "n_boot": int(n_boot),
                "mae": [float(x) for x in np.percentile(maes, [2.5, 97.5])],
                "qwk": [float(x) for x in np.percentile(qwks, [2.5, 97.5])],
                "auc": [float(x) for x in np.percentile(aucs, [2.5, 97.5])] if aucs else None,
            }

        res: Dict[str, Any] = {
            "overall": _boot_slice(np.ones(n, dtype=bool)),
            "per_dataset": {},
        }
        if len(samples) == n:
            ds_arr = np.array([str(s.get("dataset", "unknown")) for s in samples])
            for d_name in np.unique(ds_arr):
                res["per_dataset"][str(d_name)] = _boot_slice(ds_arr == d_name)
        return res


def main() -> None:
    """Điểm vào chính khi chạy `python train.py` với cấu hình mặc định từ TrainConfig."""
    config = TrainConfig()
    trainer = LiteFatTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
