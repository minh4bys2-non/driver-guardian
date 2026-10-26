"""Finetune a classifier on a new dataset with frozen CNN and one-tenth the saved LR."""
from pathlib import Path

from .checkpoint import load_checkpoint
from .config import TrainConfig
from .training import train

CHECKPOINT = "/home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/LSTM/checkpoints/head_landmarkv1/best.pt"
DATASET_DIR = "/home/tranmanhduy/Workspace/dataE/DataLSTM_v2"  # Sửa thành đường dẫn dataset mới.
SAVE_DIR = "ai/LSTM/checkpoints/finetuned"
FINETUNE_CONFIG = dict(
    batch_size=2,     # None: kế thừa checkpoint; số clip mỗi batch.
    epochs=5,         # Tổng số epoch của run finetune mới.
    warmup_epochs=1,  # Phải nhỏ hơn epochs.
    num_workers=None,    # Số worker đọc dữ liệu; 0 để đọc ở tiến trình chính.
    accumulation=4,   # Số batch tích lũy gradient mỗi lần cập nhật.
    device="cuda",         # "auto", "cuda", "cuda:0" hoặc "cpu".
    amp=None,            # True/False: mixed precision trên CUDA.
)


def finetune(checkpoint, dataset_dir, save_dir, *, batch_size=None, epochs=None,
             warmup_epochs=None, num_workers=None, accumulation=None, device=None, amp=None):
    checkpoint = str(Path(checkpoint).expanduser().resolve())
    saved = load_checkpoint(checkpoint)
    overrides = dict(batch_size=batch_size, epochs=epochs, warmup_epochs=warmup_epochs,
                     num_workers=num_workers, accumulation=accumulation, device=device, amp=amp)
    cfg = TrainConfig(**(saved["config"] | {k: v for k, v in overrides.items() if v is not None} | dict(
        dataset_dir=str(dataset_dir), save_dir=str(save_dir),
        backbone_mode="frozen", backbone_checkpoint=None,
        finetune_from=checkpoint, resume=None, lr=saved["config"]["lr"] / 10,
    )))
    return train(cfg)


if __name__ == "__main__":
    finetune(CHECKPOINT, DATASET_DIR, SAVE_DIR, **FINETUNE_CONFIG)
