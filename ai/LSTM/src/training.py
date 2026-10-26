"""Train from scratch or train a classifier over a frozen pretrained trunk."""
import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from .checkpoint import atomic_save, load_checkpoint, restore_rng, rng_state
from .config import TrainConfig, load_config
from .dataloader import build_raw_video_dataloaders
from .model import DrowsyClassifier
from .tensorboard import log_predictions

logger = logging.getLogger("ai.LSTM")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def configure_logging(path):
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)


def cosine_warmup(optimizer, epochs, warmup_epochs=5, min_lr_ratio=0.05):
    if not 0 <= warmup_epochs < epochs:
        raise ValueError("warmup_epochs must be in [0, epochs)")

    def factor(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs - 1)
        progress = min(1, progress)
        return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2

    return LambdaLR(optimizer, lr_lambda=factor)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, accumulation=1,
              desc="Eval", amp=True, max_grad_norm=1.0, writer=None, epoch=0, split="val"):
    if accumulation < 1:
        raise ValueError("accumulation must be positive")
    epoch_step = epoch * 10
    batches = len(loader)
    log_steps = {math.ceil(batches * i / 10) for i in range(1, 11)}
    started = time.perf_counter()
    targets, scores, predictions = [], [], []
    grad_norms = []
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    total_loss = total = correct = tp = fp = fn = group_samples = 0
    progress = tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True, leave=False)
    for step, (frames, labels, lengths, _) in enumerate(progress):
        frames, labels, lengths = (x.to(device, non_blocking=True) for x in (frames, labels, lengths))
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(frames, seq_lens=lengths)
            loss = criterion(logits, labels)
        count = labels.numel()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss in {desc}, batch {step}")
        if training:
            # Accumulate summed losses, then normalize by samples (including a short final batch).
            scaler.scale(loss * count).backward()
            group_samples += count
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(group_samples)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm,
                                         error_if_nonfinite=not scaler.is_enabled())
                if writer is not None:
                    grad_norms.append(float(grad_norm))
                    if step + 1 == len(loader):
                        for name, parameter in model.named_parameters():
                            if parameter.grad is not None and torch.isfinite(parameter.grad).all():
                                writer.add_histogram(f"gradients/{name}", parameter.grad, epoch_step)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                group_samples = 0
        if writer is not None:
            targets.append(labels.detach().cpu())
            scores.append(logits.detach().float().softmax(dim=1)[:, 1].cpu())
        pred = logits.detach().argmax(dim=1)
        if writer is not None:
            predictions.append(pred.cpu())
        total_loss += loss.detach().item() * count
        total += count
        correct += (pred == labels).sum().item()
        tp += ((pred == 1) & (labels == 1)).sum().item()
        fp += ((pred == 1) & (labels == 0)).sum().item()
        fn += ((pred == 0) & (labels == 1)).sum().item()
        progress.set_postfix(loss=f"{total_loss / total:.4f}", acc=f"{correct / total:.3f}")
        if (training and step + 1 in log_steps) or (not training and step + 1 == batches):
            metrics = {"loss": total_loss / total, "acc": correct / total,
                       "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
                       "f1": 2 * tp / max(1, 2 * tp + fp + fn), "n_samples": total}
            if training:
                logger.info("Epoch %d batch %d/%d train=%s", epoch, step + 1, batches,
                            json.dumps(metrics))
            if writer is not None:
                log_step = epoch_step - 10 + (step + 1) * 10 // batches if training else epoch_step
                for name, value in metrics.items():
                    writer.add_scalar(f"{split}/{name}", value, log_step)
                elapsed = time.perf_counter() - started
                writer.add_scalar(f"{split}/duration_seconds", elapsed, log_step)
                writer.add_scalar(f"{split}/samples_per_second", total / elapsed, log_step)
                finite_norms = [value for value in grad_norms if math.isfinite(value)]
                if finite_norms:
                    writer.add_scalar("train/grad_norm_before_clip", sum(finite_norms) / len(finite_norms), log_step)
                if training:
                    writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], log_step)
                    writer.add_scalar("train/amp_scale", scaler.get_scale(), log_step)
                writer.flush()
    if not total:
        raise ValueError(f"Empty loader: {desc}")
    if writer is not None:
        log_predictions(writer, split, torch.cat(targets).numpy(), torch.cat(scores).numpy(),
                        torch.cat(predictions).numpy(), epoch_step)
    return metrics

def train(cfg: TrainConfig | None = None):
    cfg = cfg or load_config()
    cfg.__post_init__()
    save_dir = Path(cfg.save_dir)
    saved = load_checkpoint(cfg.resume) if cfg.resume else None
    if saved:
        if not {"optimizer", "scheduler", "scaler", "rng", "best_model", "best_metrics", "best_epoch"} <= saved.keys():
            raise ValueError("Resume requires a full last.pt checkpoint")
        allowed = {"resume", "save_dir", "device", "num_workers"}
        changed = [key for key, value in cfg.to_dict().items()
                   if key not in allowed and value != saved["config"].get(key)]
        if changed:
            raise ValueError(f"Resume configuration differs: {changed}")
        if save_dir.resolve() != Path(cfg.resume).resolve().parent:
            raise ValueError("Resume must use the original run directory")
    elif save_dir.exists() and any(save_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {save_dir}; choose a new save_dir or resume last.pt")
    save_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(save_dir / "train.log")
    try:
        from torch.utils.tensorboard import SummaryWriter

        with SummaryWriter(str(save_dir / "tensorboard"),
                           purge_step=saved["epoch"] * 10 + 1 if saved else None) as writer:
            writer.add_text("config", "```json\n" + json.dumps(cfg.to_dict(), indent=2) + "\n```",
                            saved["epoch"] * 10 if saved else 0)
            logger.info("TensorBoard logs: %s", save_dir / "tensorboard")
            return _train(cfg, saved, save_dir, writer)
    except Exception:
        logger.exception("Training failed; the previous complete checkpoint is retained")
        raise


def _train(cfg, saved, save_dir, writer):
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else torch.device(cfg.device)
    logger.info("Configuration: %s", json.dumps(cfg.to_dict()))
    (save_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    if saved:
        model = DrowsyClassifier(**saved["model_config"])
        model.load_state_dict(saved["model"], strict=True)
    elif cfg.finetune_from:
        pretrained = load_checkpoint(cfg.finetune_from)
        model = DrowsyClassifier(**(pretrained["model_config"] | dict(freeze_backbone=True, freeze_bn=True)))
        model.load_state_dict(pretrained["model"], strict=True)
        logger.info("Finetuning weights from %s with a fresh optimizer/schedule", cfg.finetune_from)
    else:
        model = DrowsyClassifier(
            backbone_checkpoint=cfg.backbone_checkpoint, freeze_backbone=cfg.backbone_mode == "frozen",
            freeze_bn=cfg.backbone_mode == "frozen", backbone_w=cfg.backbone_w, backbone_n=cfg.backbone_n,
            neck_n=cfg.neck_n, input_dim=cfg.input_dim, hidden_dim=cfg.hidden_dim,
            chunk_size=cfg.chunk_size, dropout=cfg.dropout, gradient_checkpointing=cfg.gradient_checkpointing)
    model.to(device)
    loaders = build_raw_video_dataloaders(
        cfg.dataset_dir, batch_size=cfg.batch_size, num_workers=cfg.num_workers, img_size=cfg.image_size,
        window_sec=cfg.window_sec, target_fps=cfg.target_fps, use_augmentation=cfg.use_augmentation,
        pin_memory=device.type == "cuda")
    train_loader, val_loader, test_loader = loaders
    logger.info("Device=%s mode=%s samples(train/val/test)=%s trainable_parameters=%d", device,
                cfg.backbone_mode, [len(loader.dataset) for loader in loaders],
                sum(p.numel() for p in model.parameters() if p.requires_grad))
    logger.info("Trainable parameters: backbone/neck=%d classifier=%d; backbone/neck eval=%s",
                sum(p.numel() for p in model.backbone_neck.parameters() if p.requires_grad),
                sum(p.numel() for p in model.conv_gru.parameters() if p.requires_grad),
                not model.backbone_neck.training)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = cosine_warmup(optimizer, cfg.epochs, cfg.warmup_epochs, cfg.min_lr_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    start_epoch, best_epoch, best_metrics, best_model = 1, 0, None, None
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch = saved["epoch"] + 1
        best_epoch, best_metrics, best_model = saved["best_epoch"], saved["best_metrics"], saved["best_model"]
        restore_rng(saved["rng"])
        logger.info("Resumed epoch %d; best epoch=%d", saved["epoch"], best_epoch)
    for epoch in range(start_epoch, cfg.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        tr = run_epoch(model, train_loader, criterion, device, optimizer, scaler, cfg.accumulation,
                       desc=f"Train {epoch}/{cfg.epochs}", amp=cfg.amp, max_grad_norm=cfg.max_grad_norm,
                       writer=writer, epoch=epoch, split="train")
        va = run_epoch(model, val_loader, criterion, device, desc=f"Val {epoch}/{cfg.epochs}", amp=cfg.amp,
                       writer=writer, epoch=epoch, split="val")
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                writer.add_histogram(f"weights/{name}", parameter.detach(), epoch * 10)
        writer.flush()
        scheduler.step()
        improved = best_metrics is None or (va["f1"], -va["loss"]) > (best_metrics["f1"], -best_metrics["loss"])
        if improved:
            best_epoch, best_metrics = epoch, va
            best_model = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        state = {"format_version": 1, "epoch": epoch, "model": model.state_dict(),
                 "model_config": model.model_config, "config": cfg.to_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(), "rng": rng_state(), "best_epoch": best_epoch,
                 "best_metrics": best_metrics, "best_model": best_model}
        atomic_save(state, save_dir / "last.pt")
        if improved:
            _save_best(save_dir, state)
        logger.info("Epoch %d/%d lr=%.6g train=%s val=%s%s", epoch, cfg.epochs, lr,
                    json.dumps(tr), json.dumps(va), " BEST" if improved else "")
    # Rebuild best.pt from last.pt's embedded best weights after an interrupted write.
    state = load_checkpoint(save_dir / "last.pt")
    _save_best(save_dir, state)
    model.load_state_dict(state["best_model"])
    te = run_epoch(model, test_loader, criterion, device, desc="Test (best)", amp=cfg.amp,
                   writer=writer, epoch=cfg.epochs + 1, split="test")
    writer.add_scalar("test/checkpoint_epoch", state["best_epoch"], (cfg.epochs + 1) * 10)
    logger.info("Test best_epoch=%d metrics=%s", state["best_epoch"], json.dumps(te))
    (save_dir / "test_metrics.json").write_text(json.dumps(te, indent=2), encoding="utf-8")
    return model, te


def _save_best(save_dir, state):
    atomic_save({"format_version": 1, "epoch": state["best_epoch"], "model": state["best_model"],
                 "model_config": state["model_config"], "config": state["config"],
                 "val_metrics": state["best_metrics"]}, save_dir / "best.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", help="Optional saved config.json; defaults live in src/config.py")
    source.add_argument("--resume", help="Resume last.pt with its saved configuration")
    args = parser.parse_args()
    if args.resume:
        cfg = TrainConfig(**load_checkpoint(args.resume)["config"])
        cfg.resume, cfg.save_dir = args.resume, str(Path(args.resume).parent)
    else:
        cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
