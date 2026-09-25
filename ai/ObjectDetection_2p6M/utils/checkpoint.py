import os
import random
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch

from utils.artifacts import architecture, validate_metadata, write_metadata


def rng_state():
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (name, keys.tolist(), pos, has_gauss, cached),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save(path, model, optimizer, scheduler, ema, epoch, global_step, best_val, cfg,
         scaler=None, next_batch=0):
    _save(path, model, cfg, {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": ema.state_dict() if ema else None,
        "ema_updates": ema.updates if ema else 0,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": rng_state(),
        "epoch": epoch,
        "next_batch": next_batch,
        "global_step": global_step,
        "best_val": best_val,
        "cfg": {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init},
        "sampling_sha256": getattr(cfg, "sampling_sha256", None),
    })


def save_only_model(path, model, ema=None, *, cfg):
    _save(path, model, cfg, {
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema else None,
    })


def _save(path, model, cfg, checkpoint):
    metadata = cfg.checkpoint_metadata
    if architecture(model) != metadata["architecture"] or architecture(cfg) != metadata["architecture"]:
        raise ValueError("Kiến trúc model/config đã thay đổi so với chữ ký ban đầu")
    path = Path(path)
    write_metadata(path.parent, metadata)
    checkpoint["metadata"] = metadata
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load(path, model, optimizer=None, scheduler=None, ema=None, map_location="cpu",
         scaler=None, cfg=None):
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    validate_metadata(path, ckpt, model, cfg.checkpoint_metadata if cfg is not None else None)
    required = ("epoch", "next_batch", "global_step", "rng", "ema_updates", "optimizer", "scheduler")
    if any(key not in ckpt for key in required):
        raise ValueError("Checkpoint không có đủ trạng thái để resume; chỉ dùng nạp pretrained")
    if cfg is not None:
        if ckpt.get("sampling_sha256") != getattr(cfg, "sampling_sha256", None):
            raise ValueError("Sampling đã thay đổi hoặc checkpoint thiếu chữ ký sampling; không thể resume chính xác")
        resume_fields = (
            "batch_size", "img_size", "seed", "shuffle", "drop_last", "epochs", "optimizer",
            "head_only_epochs", "trunk_lr_factor", "lr0", "lr_min_factor", "warmup_epochs", "weight_decay", "betas", "momentum",
            "grad_clip_norm", "use_ema", "ema_decay", "ema_warmup_updates",
            "horizontalFlip", "shiftScaleRotate",
            "randomBrightnessContrast", "hueSaturationValue", "gaussNoise", "blur",
            "train_class_sampling", "class_sampling_path", "skip_iscrowd", "skip_isfake",
            "include_images_without_annotations", "cls_gain", "box_gain", "dfl_gain",
            "w_o2o", "w_o2m", "topk_o2m", "topk_o2o", "alpha", "beta",
        )
        _norm = lambda v: tuple(v) if isinstance(v, (list, tuple)) else v
        changed = [key for key in resume_fields if _norm(ckpt["cfg"].get(key)) != _norm(getattr(cfg, key, None))]
        if changed:
            raise ValueError(f"Cấu hình resume đã thay đổi: {', '.join(changed)}")
    if scaler is not None and not ckpt.get("scaler"):
        raise ValueError("Checkpoint không có GradScaler để resume AMP")
    if ema is not None and ckpt.get("ema") is None:
        raise ValueError("Checkpoint không có EMA để resume")
    model.load_state_dict(ckpt["model"])

    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler:
        scheduler.load_state_dict(ckpt["scheduler"])
    if ema and ckpt.get("ema"):
        ema.load_state_dict(ckpt["ema"])
        ema.updates = ckpt["ema_updates"]
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    if "rng" in ckpt:
        restore_rng(ckpt["rng"])

    return (
        ckpt.get("epoch", 0),
        ckpt.get("next_batch", 0),
        ckpt.get("global_step", 0),
        ckpt.get("best_val", float("-inf")),
    )


def load_only_model(path, model, map_location="cpu", cfg=None):
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    validate_metadata(path, ckpt, model, cfg.checkpoint_metadata if cfg is not None else None)
    model.load_state_dict(ckpt.get("ema") or ckpt["model"])
