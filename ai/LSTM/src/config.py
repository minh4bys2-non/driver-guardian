"""Edit DEFAULT_CONFIG here; each run saves its resolved configuration."""
import math
from dataclasses import asdict, dataclass
from pathlib import Path

@dataclass
class TrainConfig:
    dataset_dir: str = "/home/tranmanhduy/Workspace/dataE/DataLSTM_v2"
    # dataset_dir: str = "/home/tranmanhduy/Workspace/dataE/UL-DD_Processed"
    save_dir: str = "ai/LSTM/checkpoints/head_landmarkv2"
    backbone_mode: str = "frozen"  # scratch | frozen
    backbone_checkpoint: str | None = str(
        Path(__file__).resolve().parents[2] / "ObjectDetection_2p6M/checkpoints"
        / "231e35bb4061257f9bcb7cc5e3a0d0064e5e1beb6351a5e0adfc8d0c91ba4e11"
        / "da92589d1b67339be4491beaf147575e81a2699bbde06d654c05f852d7f0e9c9"
        / "landmark_finetune2/best.pt"
    )
    backbone_w: tuple = (16, 32, 64, 128, 256)
    backbone_n: tuple = (1, 2, 2, 1)
    neck_n: int = 1
    input_dim: int = 96
    hidden_dim: int = 96
    dropout: float = 0.35
    chunk_size: int = 8
    gradient_checkpointing: bool = False
    image_size: int = 224
    window_sec: float = 10
    target_fps: float = 5
    use_augmentation: bool = True
    batch_size: int = 2
    num_workers: int = 2
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.05
    accumulation: int = 4
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "auto"
    amp: bool = True
    resume: str | None = None  # Path to last.pt; keep the original epoch budget.
    finetune_from: str | None = None  # Full classifier weights; start a new optimizer/schedule.

    def __post_init__(self):
        self.backbone_w, self.backbone_n = tuple(self.backbone_w), tuple(self.backbone_n)
        for name in ("lr", "weight_decay", "max_grad_norm", "window_sec", "target_fps", "dropout", "min_lr_ratio"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("num_workers", "chunk_size", "warmup_epochs", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.backbone_mode not in ("scratch", "frozen"):
            raise ValueError("backbone_mode must be scratch or frozen")
        if self.finetune_from and self.backbone_mode != "frozen":
            raise ValueError("finetune_from requires frozen backbone_mode")
        if self.backbone_mode == "frozen" and not (self.backbone_checkpoint or self.finetune_from):
            raise ValueError("frozen mode requires backbone_checkpoint")
        if self.backbone_mode == "scratch" and self.backbone_checkpoint:
            raise ValueError("scratch mode must not load backbone_checkpoint")
        if self.gradient_checkpointing and self.backbone_mode == "scratch":
            raise ValueError("Disable gradient_checkpointing when training BatchNorm from scratch")
        for name in ("batch_size", "epochs", "accumulation", "image_size", "input_dim", "hidden_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.window_sec <= 0 or self.target_fps <= 0 or round(self.window_sec * self.target_fps) < 1:
            raise ValueError("Invalid video window or target FPS")
        if not 0 <= self.warmup_epochs < self.epochs or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("Invalid warmup or minimum LR ratio")
        if self.lr <= 0 or self.weight_decay < 0 or self.max_grad_norm <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid optimizer or dropout configuration")

    def to_dict(self):
        return asdict(self)

DEFAULT_CONFIG = TrainConfig()

def load_config(path=None):
    if path is None:
        return TrainConfig(**DEFAULT_CONFIG.to_dict())
    import json
    return TrainConfig(**json.loads(Path(path).read_text(encoding="utf-8")))
