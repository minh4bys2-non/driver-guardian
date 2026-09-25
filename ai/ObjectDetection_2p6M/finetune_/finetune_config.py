from dataclasses import dataclass

from ai.ObjectDetection_2p6M.src.config import TrainConfig


@dataclass
class FineTuneConfig(TrainConfig):
    nc: int = 80  # Đặt bằng số categories của dataset đích; head được tạo lại khi mapping đổi.
    labels_root: str = "/run/media/tranmanhduy/Data/MSCOCO/labels"
    images_root_dir: str = "/run/media/tranmanhduy/Data/MSCOCO/images"
    index_cache_dir: str = "/run/media/tranmanhduy/Data/MSCOCO/cache"
    batch_size: int = 128
    epochs: int = 30
    lr0: float = 1e-4
    head_only_epochs: int = 5  # Giai đoạn 1; epochs là tổng cả hai giai đoạn.
    trunk_lr_factor: float = 0.1  # LR backbone/neck giai đoạn 2 = LR head * hệ số này.
    warmup_epochs: float = 1.0
    class_sampling_path: str = "/run/media/tranmanhduy/Data/MSCOCO/labels/train/class_sampling.jsonl"  # id phải khớp categories đích.

    tfl_pretrained_pth: str = "/home/tranmanhduy/Workspace/ptithcm/TTTN/NewVersionObDetect/checkpoints/2bfb6cc36ef5ab822a944006c746e5d348d67e387501c9027df5b957cf124031/d9afe7f332a0080b29fdd068bbb94f32147807d062bc8dbfeb01f8b968f5ad22/train_/best.pt"  # Đường dẫn .pt cùng thư mục architecture.json/categories.json.
    resume: str = ""  # Rỗng: bắt đầu từ pretrained; tiếp tục fine-tune: đặt đường dẫn checkpoint fine-tune.

    run_name: str = "finetune_"
    tb_log_dir: str = "runs/finetune_"
    log_dir: str = "logs/finetune_"
    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.head_only_epochs, int) or not 0 < self.head_only_epochs < self.epochs:
            raise ValueError("head_only_epochs phải là số nguyên > 0 và < epochs")
        if not 0 < self.trunk_lr_factor < 1:
            raise ValueError("trunk_lr_factor phải > 0 và < 1")
        if self.lr0 <= 0:
            raise ValueError("lr0 phải > 0")
