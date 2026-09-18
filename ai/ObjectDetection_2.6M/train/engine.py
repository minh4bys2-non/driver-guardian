import os
import glob
import math
import time
import logging
from tempfile import mkdtemp

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.model import NMSFreeDetector
from src.config import TrainConfig
from train.loss import DetectionLoss
from train.ema import ModelEMA
from train.dataloader_ import build_dataloaders, EpochBatchSampler
from evaluation.mAPEvaluation import MetricAccumulator
from utils.seed import set_seed
from utils.artifacts import build_metadata, checkpoint_dir, write_metadata
from utils.checkpoint import load as load_checkpoint, save as save_checkpoint
from utils.logging_setup import setup_logging
from utils.tb_logger import TrainingLogger

logger = logging.getLogger("train")


def get_dataloader(cfg):
    train, val, classes, nc = build_dataloaders(cfg)
    if cfg.nc != nc:
        raise ValueError(f"nc={cfg.nc} != dataset nc={nc}")
    return train, val, classes


def get_model(cfg):
    return NMSFreeDetector(
        nc=cfg.nc, reg_max=cfg.reg_max,
        backbone_w=cfg.backbone_w, backbone_n=cfg.backbone_n,
        neck_n=cfg.neck_n, strides=cfg.strides,
        img_size=cfg.img_size
    )


def get_criterion(cfg):
    return DetectionLoss(
        nc=cfg.nc, reg_max=cfg.reg_max,
        topk_o2m=getattr(cfg, "topk_o2m", 10),
        topk_o2o=getattr(cfg, "topk_o2o", 1),
        alpha=getattr(cfg, "alpha", .5),
        beta=getattr(cfg, "beta", 6.),
        box_gain=getattr(cfg, "box_gain", 7.5),
        cls_gain=getattr(cfg, "cls_gain", 1.),
        dfl_gain=getattr(cfg, "dfl_gain", 1.5),
        o2m_weight=getattr(cfg, "w_o2m", 1.),
        o2o_weight=getattr(cfg, "w_o2o", 1.)
    )


def get_optimizer(model, cfg, groups=None):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith("bias") else decay).append(p)

    groups = groups or [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.}
    ]

    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr0, betas=getattr(cfg, "betas", (.9, .999)))
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr0, momentum=cfg.momentum, nesterov=True)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def lr_lambda_factory(cfg, steps_per_epoch):
    warmup = max(1, int(cfg.warmup_epochs * steps_per_epoch))
    total = max(warmup + 1, cfg.epochs * steps_per_epoch)

    def fn(step):
        if step < warmup:
            return step / warmup
        p = min((step - warmup) / max(1, total - warmup), 1.)
        cosine = .5 * (1 + math.cos(math.pi * p))
        return cfg.lr_min_factor + (1 - cfg.lr_min_factor) * cosine

    return fn


def move_batch(images, targets, device):
    images = images.to(device, non_blocking=True)
    targets = [{
        "boxes": t["boxes"].to(device, non_blocking=True),
        "labels": t["labels"].to(device, non_blocking=True)
    } for t in targets]
    return images, targets


def save_periodic_checkpoint(
    cfg, model, optimizer, scheduler, ema,
    epoch, step, best_map, scaler=None, next_batch=0
):
    out = checkpoint_dir(cfg)
    path = os.path.join(out, f"ckpt_step{step:08d}.pt")

    for p in (path, os.path.join(out, "last.pt")):
        save_checkpoint(
            p, model, optimizer, scheduler, ema,
            epoch, step, best_map, cfg,
            scaler=scaler, next_batch=next_batch
        )

    keep = getattr(cfg, "ckpt_keep_last", 3)
    if keep > 0:
        ckpts = sorted(glob.glob(os.path.join(out, "ckpt_step*.pt")))
        for p in ckpts[:-keep]:
            os.remove(p)

    logger.info(f"[step {step}] saved {path}")
    return path


@torch.no_grad()
def validate(model, criterion, loader, device, tb_logger=None, step=0):
    model.eval()
    totals, n = {}, 0
    metrics = MetricAccumulator(nc=criterion.nc)
    started = time.perf_counter()
    logger.info("[val %s] starting validation: %s batches", step, len(loader))

    for images, targets in tqdm(loader, desc=f"Validation [{step}]", leave=False, ncols=100):
        images, targets = move_batch(images, targets, device)
        preds = model(images)
        _, items = criterion(preds, targets)

        for k, v in items.items():
            totals[k] = totals.get(k, 0.) + v

        metrics.update(preds, targets)
        n += 1

    if not n:
        raise ValueError("Validation loader rỗng")

    losses = {k: v / n for k, v in totals.items()}
    logger.info("[val %s] batches done in %.1fs; computing mAP for %s images",
                step, time.perf_counter() - started, metrics.img_id)
    scores = metrics.compute()

    if tb_logger:
        tb_logger.log_losses(losses, step=step, phase="val")
        tb_logger.log_scalars({
            f"val/{k}": scores[k]
            for k in ("map_50", "map_50_95", "precision", "recall")
        }, step)

    logger.info(
        "[val %s] loss=%.4f mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f time=%.1fs",
        step, losses["loss"], scores["map_50"], scores["map_50_95"],
        scores["precision"], scores["recall"], time.perf_counter() - started
    )
    return losses["loss"], scores


def train_one_epoch(
    model, criterion, loader, val_loader,
    optimizer, scheduler, scaler, ema, device, cfg, epoch,
    global_step=0, best_map=float("-inf"), tb_logger=None, start_batch=0
):
    model.train()
    t0, running = time.time(), 0.
    n_batches = len(loader)
    use_amp = scaler is not None

    if not 0 <= start_batch <= n_batches:
        raise ValueError("Resume batch không hợp lệ")

    sampler = getattr(loader, "batch_sampler", None)
    if isinstance(sampler, EpochBatchSampler):
        sampler.epoch, sampler.start_batch = epoch, start_batch
    elif start_batch:
        raise ValueError("Resume giữa epoch yêu cầu EpochBatchSampler")

    val_every = getattr(cfg, "val_interval_steps", 500)
    save_every = getattr(cfg, "save_ckpt_interval_steps", 1000)
    log_every = getattr(cfg, "log_interval", 0)
    loss_every = getattr(cfg, "log_loss_interval", 50)
    log_grad = tb_logger is not None and getattr(cfg, "log_gradients", True)
    log_weight = tb_logger is not None and getattr(cfg, "log_weights", True)

    pbar = tqdm(
        enumerate(loader, start=start_batch),
        initial=start_batch, total=n_batches,
        desc=f"Epoch [{epoch + 1}/{cfg.epochs}]",
        ncols=100
    )

    for step, (images, targets) in pbar:
        images, targets = move_batch(images, targets, device)
        global_step += 1

        snapshot = (
            TrainingLogger.snapshot_params(model)
            if log_weight and tb_logger.should_log_scalar(global_step)
            else None
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=str(device).split(":")[0], enabled=use_amp):
            preds = model(images)
            loss, items = criterion(preds, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm).item()

        if not math.isfinite(grad_norm):
            logger.warning(f"[step {global_step}] gradient NaN/Inf: {grad_norm}")

        if log_grad:
            tb_logger.log_gradients(model, global_step, total_norm=grad_norm)

        if use_amp:
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < scale
        else:
            optimizer.step()
            skipped = False

        if log_weight:
            tb_logger.log_weights(model, global_step)
            tb_logger.log_weight_updates(model, snapshot, global_step)

        if not skipped:
            scheduler.step()
            if ema:
                ema.update(model=model)

        running += items["loss"]

        if tb_logger:
            if loss_every > 0 and global_step % loss_every == 0:
                tb_logger.log_losses(items, global_step, "train")
            tb_logger.log_learning_rate(optimizer, global_step, epoch)
            tb_logger.log_ema(ema, global_step)
            tb_logger.log_gpu_memory(global_step)

        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{items['loss']:.4f}", lr=f"{lr:.1e}")

        if loss_every > 0 and global_step % loss_every == 0:
            logger.info(
                f"[epoch {epoch}] step={step}/{n_batches} "
                f"global={global_step} loss={items['loss']:.4f} lr={lr:.6f}"
            )

        if log_every > 0 and global_step % log_every == 0:
            mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.
            logger.info(
                f"[step {global_step}] "
                f"o2m(iou={items['o2m/iou']:.3f} cls={items['o2m/cls']:.3f} "
                f"dfl={items['o2m/dfl']:.3f} n={items['o2m/n_pos']}) "
                f"o2o(iou={items['o2o/iou']:.3f} cls={items['o2o/cls']:.3f} "
                f"dfl={items['o2o/dfl']:.3f} n={items['o2o/n_pos']}) "
                f"lr={lr:.6f} mem={mem:.2f}GB t={time.time() - t0:.1f}s"
            )

        if val_loader is not None and val_every > 0 and global_step % val_every == 0:
            eval_model = ema.ema if ema else model
            val_loss, scores = validate(
                eval_model, criterion, val_loader, device,
                tb_logger=tb_logger, step=global_step
            )
            map95 = scores["map_50_95"]

            logger.info(
                f"[step {global_step}] train={items['loss']:.4f} "
                f"val={val_loss:.4f} mAP50-95={map95:.4f}"
            )

            if map95 > best_map:
                best_map = map95
                save_checkpoint(
                    os.path.join(checkpoint_dir(cfg), "best.pt"),
                    model, optimizer, scheduler, ema,
                    epoch, global_step, best_map, cfg,
                    scaler=scaler, next_batch=step + 1
                )
                logger.info(f"[step {global_step}] new best mAP50-95={best_map:.4f}")

            model.train()

        if not cfg.save_best_only and save_every > 0 and global_step % save_every == 0:
            save_periodic_checkpoint(
                cfg, model, optimizer, scheduler, ema,
                epoch, global_step, best_map,
                scaler=scaler, next_batch=step + 1
            )

    return running / max(1, n_batches - start_batch), global_step, best_map


def run_training(cfg,
                 model_factory=None, optimizer_factory=None, on_epoch_start=None):
    setup_logging(cfg)
    set_seed(cfg.seed)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    if device != cfg.device:
        logger.warning(f"{cfg.device} unavailable -> {device}")

    train_loader, val_loader, classes = get_dataloader(cfg)
    if not len(train_loader):
        raise ValueError("Training loader rỗng")
    if val_loader is not None and not len(val_loader):
        raise ValueError("Validation loader rỗng")

    cfg.checkpoint_metadata = build_metadata(cfg, classes)
    out = checkpoint_dir(cfg)
    write_metadata(out, cfg.checkpoint_metadata)

    logger.info(
        "[data] train=%s val=%s classes=%s",
        len(train_loader.dataset),
        len(val_loader.dataset) if val_loader else 0,
        len(classes)
    )

    model = (model_factory or get_model)(cfg).to(device)
    criterion = get_criterion(cfg).to(device)
    optimizer = (optimizer_factory or get_optimizer)(model, cfg)

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(cfg, steps_per_epoch)
    )

    use_amp = cfg.amp and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    ema = (
        ModelEMA(model, cfg.ema_decay, cfg.ema_warmup_updates)
        if cfg.use_ema else None
    )

    start_epoch = start_batch = global_step = 0
    best_map = float("-inf")

    if cfg.resume:
        start_epoch, start_batch, global_step, best_map = load_checkpoint(
            cfg.resume, model, optimizer, scheduler, ema,
            map_location=device, scaler=scaler, cfg=cfg
        )

        if start_batch == steps_per_epoch:
            start_epoch += 1
            start_batch = 0

        logger.info(
            f"[resume] epoch={start_epoch} step={global_step} "
            f"best_map={best_map:.4f}"
        )

    if not 0 <= start_epoch <= cfg.epochs or not 0 <= start_batch <= steps_per_epoch:
        raise ValueError("Resume state không hợp lệ")

    tb_root = getattr(cfg, "tb_log_dir", "runs")
    writer = None
    if tb_root:
        os.makedirs(tb_root, exist_ok=True)
        tb_dir = mkdtemp(prefix=f"{cfg.run_name}_", dir=tb_root)
        writer = SummaryWriter(tb_dir)
        logger.info("[tensorboard] %s | start_step=%s", tb_dir, global_step)
    tb = TrainingLogger(writer, cfg) if writer else None

    if tb:
        tb.log_hparams(cfg)

    try:
        for epoch in range(start_epoch, cfg.epochs):
            if on_epoch_start:
                on_epoch_start(model, cfg, epoch)

            train_loss, global_step, best_map = train_one_epoch(
                model, criterion, train_loader, val_loader,
                optimizer, scheduler, scaler, ema, device, cfg, epoch,
                global_step, best_map, tb, start_batch
            )

            start_batch = 0
            logger.info(f"[epoch {epoch}] loss={train_loss:.4f} step={global_step}")

        if val_loader is not None:
            eval_model = ema.ema if ema else model
            val_loss, scores = validate(
                eval_model, criterion, val_loader, device,
                tb_logger=tb, step=global_step
            )
            map95 = scores["map_50_95"]

            if map95 > best_map:
                best_map = map95
                save_checkpoint(
                    os.path.join(out, "best.pt"),
                    model, optimizer, scheduler, ema,
                    cfg.epochs, global_step, best_map, cfg,
                    scaler=scaler
                )

            logger.info(f"[final] val={val_loss:.4f} mAP50-95={map95:.4f}")

        if cfg.save_best_only:
            save_checkpoint(
                os.path.join(out, "last.pt"),
                model, optimizer, scheduler, ema,
                cfg.epochs, global_step, best_map, cfg,
                scaler=scaler
            )
        else:
            save_periodic_checkpoint(
                cfg, model, optimizer, scheduler, ema,
                cfg.epochs, global_step, best_map, scaler
            )

    finally:
        if writer:
            writer.close()

    logger.info(f"Training done | best mAP50-95={best_map:.4f}")
    logger.info(f"Best checkpoint: {os.path.join(out, 'best.pt')}")
    return best_map

if __name__ == "__main__":
    run_training(TrainConfig())
