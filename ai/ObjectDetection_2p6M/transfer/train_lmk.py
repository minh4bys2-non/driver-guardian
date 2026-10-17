import logging
import math
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from ai.ObjectDetection_2p6M.transfer.config_lmk import TrainConfig
from ai.ObjectDetection_2p6M.transfer.dataloader_lmk import build_loaders
from ai.ObjectDetection_2p6M.transfer.checkpoint import CheckpointManager, read_checkpoint
from ai.ObjectDetection_2p6M.utils.seed import set_seed
from ai.ObjectDetection_2p6M.train.ema import ModelEMA
from ai.ObjectDetection_2p6M.transfer.loss_lmk import FaceLandmarkDetectionLoss
from ai.ObjectDetection_2p6M.transfer.model_lmk import FaceLmkDetector

logger = logging.getLogger('train_face_lmk')

class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = torch.device(cfg.device if cfg.device == 'cpu' or torch.cuda.is_available() else 'cpu')
        self.train_loader, self.val_loader, data_signature = build_loaders(cfg)
        self.ckpt = CheckpointManager(cfg, data_signature, len(self.train_loader))
        self.model = FaceLmkDetector(cfg).to(self.device)
        if cfg.resume:
            pass  # Resume chỉ cần checkpoint hiện tại, không phụ thuộc pretrained cũ.
        elif cfg.pretrained_ckpt:
            checkpoint = read_checkpoint(cfg.pretrained_ckpt, cfg)
            self.model.load_state_dict(checkpoint.get('ema') or checkpoint['model'], strict=True)
        elif cfg.trunk_ckpt:
            self.model.load_trunk(cfg.trunk_ckpt, map_location=self.device)
        elif cfg.require_pretrained_trunk:
            raise ValueError('Điền trunk_ckpt, pretrained_ckpt hoặc resume trong config_lmk.py.')

        self.loss_fn = FaceLandmarkDetectionLoss(cfg.face).to(self.device)
        self.optimizer = self._build_optimizer()
        self.scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp and self.device.type == 'cuda')
        self.ema = ModelEMA(self.model, cfg.ema_decay, cfg.ema_warmup_updates) if cfg.use_ema else None
        self.start_epoch, self.next_batch, self.global_step, self.best_val = 0, 0, 0, float('inf')
        if cfg.resume:
            self.start_epoch, self.next_batch, self.global_step, self.best_val = self.ckpt.load(
                cfg.resume, self.model, self.optimizer, self.scaler, self.ema)
        self.ckpt.prepare()
        self._setup_logging()
        self.writer = SummaryWriter(log_dir=str(self.ckpt.tb_dir),
                                    purge_step=self.global_step + 1 if cfg.resume else None)
        self.writer.add_text('config', str(cfg), self.global_step)
        self._active_stage_name = None
        self._trunk_frozen = False
        if self.start_epoch < cfg.epochs:
            self._configure_stage(self.start_epoch, force=True)
        logger.info('Run: %s | device=%s | epoch=%s batch=%s step=%s',
                    self.ckpt.directory, self.device, self.start_epoch, self.next_batch, self.global_step)

    def _setup_logging(self):
        from datetime import datetime

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        logger.propagate = False
        logger.setLevel(self.cfg.log_level.upper())
        fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        path = self.ckpt.log_dir / f'{datetime.now():%Y%m%d_%H%M%S_%f}.log'
        handlers = [logging.FileHandler(path, encoding='utf-8')]
        if self.cfg.log_stdout:
            handlers.append(logging.StreamHandler())
        for handler in handlers:
            handler.setFormatter(fmt)
            logger.addHandler(handler)

    def _build_optimizer(self):
        cfg = self.cfg
        # Luôn giữ đủ hai param group để optimizer state của head không bị mất khi chuyển stage.
        head_params = list(self.model.head.parameters())
        trunk_params = list(self.model.backbone.parameters()) + list(self.model.neck.parameters())
        param_groups = [
            {'params': head_params, 'lr': cfg.stage1.head_lr, 'name': 'head'},
            {'params': trunk_params, 'lr': 0.0, 'name': 'trunk'},
        ]
        if cfg.optimizer == 'adamw':
            return torch.optim.AdamW(param_groups, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)
        if cfg.optimizer == 'sgd':
            return torch.optim.SGD(
                param_groups,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
                nesterov=cfg.sgd_nesterov,
            )
        raise ValueError(f'optimizer không hỗ trợ: {cfg.optimizer}')

    def _configure_stage(self, epoch: int, force: bool = False) -> None:
        stage_name, stage_cfg, local_epoch = self.cfg.stage_for_epoch(epoch)
        if not force and stage_name == self._active_stage_name:
            return
        freeze_trunk = stage_name == 'stage1_head_only'
        self.model.freeze_trunk(freeze_trunk)
        self._trunk_frozen = freeze_trunk
        self._active_stage_name = stage_name
        trunk_trainable = sum(p.numel() for group in (self.model.backbone, self.model.neck)
                              for p in group.parameters() if p.requires_grad)
        head_trainable = sum(p.numel() for p in self.model.head.parameters() if p.requires_grad)
        logger.info(
            f'Bắt đầu {stage_name}: local_epoch={local_epoch}, epochs={stage_cfg.epochs}, '
            f'freeze_trunk={freeze_trunk}, head_lr={stage_cfg.head_lr:.2e}, '
            f'trunk_lr={stage_cfg.trunk_lr:.2e}, trainable head={head_trainable:,}, '
            f'trunk={trunk_trainable:,}.'
        )

    def _set_stage_learning_rates(self, epoch: int, step: int) -> tuple:
        _, stage_cfg, local_epoch = self.cfg.stage_for_epoch(epoch)
        steps_per_epoch = len(self.train_loader)
        local_step = local_epoch * steps_per_epoch + step
        total_steps = max(stage_cfg.epochs * steps_per_epoch, 1)
        warmup_steps = int(stage_cfg.warmup_epochs * steps_per_epoch)
        if warmup_steps > 0 and local_step < warmup_steps:
            factor = (local_step + 1) / warmup_steps
        else:
            decay_step = max(local_step - warmup_steps, 0)
            decay_steps = max(total_steps - warmup_steps - 1, 1)
            progress = min(decay_step / decay_steps, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = stage_cfg.min_lr_factor + (1.0 - stage_cfg.min_lr_factor) * cosine

        lr_head = stage_cfg.head_lr * factor
        lr_trunk = stage_cfg.trunk_lr * factor
        for group in self.optimizer.param_groups:
            group['lr'] = lr_head if group['name'] == 'head' else lr_trunk
        return lr_head, lr_trunk

    def _train_one_epoch(self, epoch: int) -> None:
        cfg = self.cfg
        self.model.train()
        t_epoch = time.time()
        running_loss, n_steps_done = 0.0, 0
        augmentation_counts = {}
        use_amp = cfg.amp and self.device.type == 'cuda'
        total_norm = 0.0

        sampler = self.train_loader.batch_sampler
        sampler.epoch, sampler.start_batch = epoch, self.next_batch
        train_progress = tqdm(
            self.train_loader,
            initial=self.next_batch, total=len(self.train_loader),
            desc=f'Train epoch {epoch + 1}/{cfg.epochs}',
            unit='batch',
            dynamic_ncols=True,
            mininterval=1.0,
        )
        for step, batch in enumerate(train_progress, start=self.next_batch):
            lr_head, lr_trunk = self._set_stage_learning_rates(epoch, step)
            images = batch['image'].to(self.device, non_blocking=True)
            for aug_name in batch.get('geometric_aug', []):
                augmentation_counts[aug_name] = augmentation_counts.get(aug_name, 0) + 1
            targets = batch['targets']
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=use_amp):
                preds = self.model(images)
                total_loss, items = self.loss_fn(preds, targets)
            if not torch.isfinite(total_loss):
                logger.warning(f'[epoch {epoch} step {step}] loss không hữu hạn ({total_loss.item()}) - bỏ qua step.')
                continue

            optimizer_step_succeeded = True
            if self.scaler.is_enabled():
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_norm).item()
                prev_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                optimizer_step_succeeded = self.scaler.get_scale() >= prev_scale
            else:
                total_loss.backward()
                total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_norm).item()
                self.optimizer.step()

            if optimizer_step_succeeded:
                if self.ema is not None:
                    self.ema.update(self.model)

            self.global_step += 1
            running_loss += items['loss']
            n_steps_done += 1
            train_progress.set_postfix(
                loss=f"{items['loss']:.4f}",
                lr_head=f'{lr_head:.2e}',
                lr_trunk=f'{lr_trunk:.2e}',
                refresh=False,
            )

            if self.global_step % cfg.log_loss_interval == 0:
                logger.info(f"epoch {epoch} step {step}/{len(self.train_loader)} (global {self.global_step}) "
                            f"stage={self._active_stage_name} loss={items['loss']:.4f} "
                            f"lr_head={lr_head:.2e} lr_trunk={lr_trunk:.2e}")
                self.writer.add_scalar('train/loss_total', items['loss'], self.global_step)
                self.writer.add_scalar('train/lr_head', lr_head, self.global_step)
                self.writer.add_scalar('train/lr_trunk', lr_trunk, self.global_step)
            if self.global_step % cfg.log_interval == 0:
                self.writer.add_scalars('train/loss_o2m', {'iou': items['o2m/iou'], 'cls': items['o2m/cls'], 'dfl': items['o2m/dfl'], 'lmk': items['o2m/lmk']}, self.global_step)
                self.writer.add_scalars('train/loss_o2o', {'iou': items['o2o/iou'], 'cls': items['o2o/cls'], 'dfl': items['o2o/dfl'], 'lmk': items['o2o/lmk']}, self.global_step)
                self.writer.add_scalars('train/n_pos', {'o2m': items['o2m/n_pos'], 'o2o': items['o2o/n_pos'], 'o2m_lmk': items['o2m/n_lmk_pos'], 'o2o_lmk': items['o2o/n_lmk_pos']}, self.global_step)
                if cfg.log_gradients:
                    self.writer.add_scalar('train/grad_norm', total_norm, self.global_step)
                if self.device.type == 'cuda':
                    self.writer.add_scalar('train/gpu_mem_gb', torch.cuda.max_memory_allocated() / 1e9, self.global_step)

            self.next_batch = step + 1
            # Cuối epoch đã validate/save bên ngoài, không ghi hai lần cùng step.
            epoch_finished = self.next_batch == len(self.train_loader)
            if (not epoch_finished and cfg.val_interval_steps
                    and self.global_step % cfg.val_interval_steps == 0):
                del preds, total_loss, images
                self._validate_and_save(epoch, self.next_batch)
            if (not epoch_finished and cfg.save_ckpt_interval_steps
                    and self.global_step % cfg.save_ckpt_interval_steps == 0):
                self.ckpt.save_periodic(self.model, self.optimizer, self.scaler, self.ema,
                                        epoch, self.global_step, self.best_val, self.next_batch)

        avg_loss = running_loss / max(n_steps_done, 1)
        logger.info(
            f'== epoch {epoch} xong: loss trung bình={avg_loss:.4f}, '
            f'{n_steps_done} step, {time.time() - t_epoch:.1f}s | '
            f'geometric_aug={augmentation_counts} =='
        )

    @torch.no_grad()
    def _validate(self, epoch: int) -> float | None:
        if self.val_loader is None:
            return None
        eval_model = self.ema.ema if self.ema is not None else self.model
        eval_model.eval()
        total, n = 0.0, 0
        val_progress = tqdm(
            self.val_loader,
            desc=f'Validate epoch {epoch + 1}/{self.cfg.epochs}',
            unit='batch',
            dynamic_ncols=True,
            mininterval=1.0,
            leave=False,
        )
        for batch in val_progress:
            images = batch['image'].to(self.device, non_blocking=True)
            preds = eval_model(images, return_o2m=True)
            _, items = self.loss_fn(preds, batch['targets'])
            total += items['loss']
            n += 1
            val_progress.set_postfix(val_loss=f'{total / n:.4f}', refresh=False)
        val_loss = total / max(n, 1)
        logger.info(f'== epoch {epoch} validate: loss={val_loss:.4f} ({n} batch) ==')
        self.writer.add_scalar('val/loss_total', val_loss, self.global_step)
        return val_loss

    def _validate_and_save(self, epoch: int, next_batch: int = 0) -> None:
        if self.val_loader is None:
            return
        try:
            val_loss = self._validate(epoch)
            if not math.isfinite(val_loss):
                raise FloatingPointError('Validation loss không hữu hạn.')
            if val_loss < self.best_val:
                self.best_val = val_loss
                self.ckpt.save('best.pt', self.model, self.optimizer, self.scaler, self.ema,
                               epoch if next_batch else epoch + 1,
                               self.global_step, self.best_val, next_batch)
        finally:
            # Quan trọng khi không dùng EMA: validation đã chuyển model sang eval.
            self.model.train()

    def fit(self):
        cfg = self.cfg
        logger.info('TrainConfig: %s', cfg)
        try:
            for epoch in range(self.start_epoch, cfg.epochs):
                self._configure_stage(epoch)
                self._train_one_epoch(epoch)
                self.next_batch = 0
                if self.val_loader is not None and (epoch + 1) % cfg.val_interval == 0:
                    self._validate_and_save(epoch)
                args = (self.model, self.optimizer, self.scaler, self.ema,
                        epoch + 1, self.global_step, self.best_val)
                self.ckpt.save_periodic(*args)
                if epoch + 1 == cfg.stage1.epochs:
                    self.ckpt.save('stage1_head_only_final.pt', *args)
                if epoch + 1 == cfg.epochs:
                    self.ckpt.save(f'{self._active_stage_name}_final.pt', *args)
            logger.info('Training hoàn tất.')
            return self.model, self.ema
        finally:
            self.writer.close()
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()


if __name__ == '__main__':
    Trainer(TrainConfig()).fit()
