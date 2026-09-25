import logging
import torch

from ai.ObjectDetection_2p6M.finetune_.finetune_config import FineTuneConfig
from ai.ObjectDetection_2p6M.src.model import NMSFreeDetector
from ai.ObjectDetection_2p6M.train_.engine import get_optimizer, run_training
from ai.ObjectDetection_2p6M.utils.artifacts import architecture, validate_metadata

class FineTuneDetector(NMSFreeDetector):
    def train(self, mode=True):
        super().train(mode)
        for module in (self.backbone, self.neck):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()
        return self

def get_finetune_model(cfg: FineTuneConfig):
    if cfg.resume:
        model = FineTuneDetector(**architecture(cfg), img_size=cfg.img_size)
    else:
        if not cfg.tfl_pretrained_pth:
            raise ValueError("Cần đặt tfl_pretrained_pth hoặc resume để fine-tune")
        checkpoint = torch.load(cfg.tfl_pretrained_pth, map_location="cpu", weights_only=True)
        source = validate_metadata(cfg.tfl_pretrained_pth, checkpoint)
        target = cfg.checkpoint_metadata
        target_architecture = architecture(cfg)
        if target["architecture"] != target_architecture:
            raise ValueError("Cấu hình model đã thay đổi sau khi tạo metadata dataset đích")
        if any(value != target_architecture[key] for key, value in source["architecture"].items() if key != "nc"):
            raise ValueError("Fine-tune chỉ cho phép đổi nc; cấu hình backbone/neck/reg_max/strides phải giữ nguyên")
        model = FineTuneDetector(**source["architecture"], img_size=cfg.img_size)
        model.load_state_dict(checkpoint.get("ema") or checkpoint["model"])
        if source["categories_sha256"] != target["categories_sha256"]:
            model.replace_head(nc=cfg.nc, replace_all=False)
            logging.getLogger("train_").info(
                "[finetune_] Giữ pretrained, chỉ khởi tạo output cls mới: nc=%s -> %s; categories đã đổi",
                source["architecture"]["nc"], cfg.nc)

    set_finetune_stage(model, cfg, 0)
    return model

def set_finetune_stage(model, cfg, epoch):
    frozen = epoch < cfg.head_only_epochs
    model.freeze_trunk(frozen)
    model.head.requires_grad_(True)
    model.head.dfl.requires_grad_(False)
    model.train(model.training)
    logging.getLogger("train_").info(
        "[finetune_] epoch=%s stage=%s trunk_lr_factor=%s",
        epoch + 1, 1 if frozen else 2, 0 if frozen else cfg.trunk_lr_factor)

def get_finetune_optimizer(model, cfg):
    groups = []
    # Đăng ký cả trunk từ đầu để layout optimizer ổn định khi mở băng/resume.
    for name in ("head", "backbone", "neck"):
        decay, no_decay = [], []
        for parameter in getattr(model, name).parameters():
            (no_decay if parameter.ndim <= 1 else decay).append(parameter)
        lr = cfg.lr0 * (1 if name == "head" else cfg.trunk_lr_factor)
        groups.extend([
            {"params": decay, "lr": lr, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ])
    return get_optimizer(model, cfg, groups=groups)

def run_finetuning(cfg: FineTuneConfig):
    return run_training(cfg, model_factory=get_finetune_model,
                        optimizer_factory=get_finetune_optimizer, on_epoch_start=set_finetune_stage)

if __name__ == "__main__":
    run_finetuning(FineTuneConfig())
