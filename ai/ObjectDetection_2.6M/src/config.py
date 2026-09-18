from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    # ---- Data ----
    # Labels và ảnh có thư mục gốc riêng, mỗi gốc gồm train/ và val/.
    labels_root: str = "/home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/DataPretrain1/Object365/labels"
    train_subdir: str = "train"
    val_subdir: str = "val"
    images_info_filename: str = "images_info.jsonl"
    annotations_filename: str = "annotations.jsonl"
    categories_filename: str = "categories.jsonl"
    train_image_path_map_filename: str = "images_train.jsonl"
    val_image_path_map_filename: str = "images_val.jsonl"
    
    images_root_dir: str = "/home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/DataPretrain1/Object365/images"
    images_train_subdir: str = "train"
    images_val_subdir: str = "val"
    
    index_cache_dir: str = "/home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/DataPretrain1/Object365/cache"     # nơi lưu byte-offset index (pickle cache)
    rebuild_index: bool = False          # ép build lại index dù đã có cache
    max_load_retries: int = 10
    data_error_log_filename: str = "bad_samples_{split}.log"

    skip_iscrowd: bool = True            # bỏ annotation iscrowd=1 khi train
    skip_isfake: bool = True             # bỏ annotation isfake=1 khi train
    include_images_without_annotations: bool = False  # Giữ ảnh không còn nhãn hợp lệ sau lọc.

    img_size: int = 640
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = False
    persistent_workers: bool = True
    prefetch_factor: int = 4
    seed: int = 42

    # ---- Data Augment ----
    horizontalFlip: float = 0.5
    shiftScaleRotate: tuple = (0.08, 0.10, 15, 0.8)
    randomBrightnessContrast: float = 0.4
    hueSaturationValue: tuple = (15, 20, 15, 0.3)
    gaussNoise: tuple = (5.0, 25.0, 0.25)
    blur: tuple = (5, 0.15)
    class_sampling_path: str = "/home/tranmanhduy/Workspace/ptithcm/TTTN/CNNModel/DataPretrain1/Object365/labels/train/class_sampling.jsonl"  # id gốc; probability=140 nghĩa là hệ số lặp 1.4.
    train_class_sampling: bool = True
    
    # ---- Model ----
    nc: int = 80
    reg_max: int = 16
    backbone_w: tuple = (16, 32, 64, 128, 256)
    backbone_n: tuple = (1, 2, 2, 1)
    neck_n: int = 1
    strides: tuple = (8, 16, 32)

    # ---- Optim ----
    epochs: int = 100
    lr0: float = 1e-3             # LR sau warmup
    lr_min_factor: float = 0.01   # LR cuối = lr0 * lr_min_factor (cosine)
    weight_decay: float = 5e-4
    warmup_epochs: float = 3.0
    optimizer: str = "adamw"      # "adamw" | "sgd"
    grad_clip_norm: float = 10.0
    betas: tuple = (0.9, 0.98)
    momentum: float = 0.937       # dùng khi optimizer="sgd"

    # ---- Loss weights (truyền thẳng xuống DetectionLoss) ----
    cls_gain: float = 0.5
    box_gain: float = 7.5
    dfl_gain: float = 1.5
    w_o2o: float = 1.0
    w_o2m: float = 1.0
    topk_o2m: int = 10
    topk_o2o: int = 1
    alpha: float = 0.5
    beta: float = 6.0

    # ---- EMA ----
    use_ema: bool = True
    ema_decay: float = 0.9998
    ema_warmup_updates: int = 2000

    # ---- TensorBoard / Logging ----
    tb_log_dir: str = "runs"      # Thư mục lưu log TensorBoard
    log_dir: str = "./logs"       # Thư mục lưu file .log (text logging)
    run_name: str = "train"       # Tiền tố tên file .log (train_{timestamp}.log)
    log_level: str = "INFO"
    log_stdout: bool = False
    log_window_size: int = 150
    log_gradients: bool = True    # Log gradient histogram & RMSNorm
    log_weights: bool = True      # Log weight/bias histogram, STD & RMSNorm
    log_hist_interval: int = 1000  # Số step giữa 2 lần log histogram (đặt lớn như 500/1000 để giảm nhe/tăng tốc training, đặt <= 0 để tắt hẳn)

    # ---- Runtime ----
    device: str = "cuda"          # sẽ tự fallback về cpu nếu không có GPU
    amp: bool = True              # mixed precision
    log_interval: int = 500        # số step giữa 2 lần log chi tiết (breakdown loss, lr, ema, gpu)
    log_loss_interval: int = 50    # số step giữa 2 lần log loss (nhẹ, ghi thường xuyên hơn log_interval)
    val_interval_steps: int = 1000 # số step giữa 2 lần validate
    save_ckpt_interval_steps: int = 1000 # số step giữa 2 lần lưu checkpoint định kỳ
    ckpt_dir: str = "./checkpoints"
    resume: str = ""    # path checkpoint để resume (vd: checkpoints/last.pt), rỗng = train từ đầu
    save_best_only: bool = False  # False -> lưu thêm checkpoint định kỳ
    ckpt_keep_last: int = 3       # số checkpoint định kỳ (theo global_step) giữ lại, <=0 = giữ hết

    # ---- Runtime (tự động gán bởi engine/dataloader, KHÔNG cấu hình) ----
    checkpoint_metadata: dict = field(default=None, init=False, repr=False, compare=False)
    sampling_sha256: str = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        assert len(self.shiftScaleRotate) == 4, \
            "shiftScaleRotate cần đúng 4 phần tử: (shift_limit, scale_limit, rotate_limit, p)"
        assert len(self.hueSaturationValue) == 4, \
            "hueSaturationValue cần đúng 4 phần tử: (hue_shift_limit, sat_shift_limit, val_shift_limit, p)"
        assert len(self.gaussNoise) == 3, \
            "gaussNoise cần đúng 3 phần tử: (var_min, var_max, p)"
        assert len(self.blur) == 2, \
            "blur cần đúng 2 phần tử: (blur_limit, p)"
        assert self.log_interval > 0, "log_interval phải > 0"
        assert self.log_loss_interval > 0, "log_loss_interval phải > 0"
        assert self.log_window_size > 0, "log_window_size phải > 0"
        assert self.max_load_retries > 0, "max_load_retries phải > 0"
        assert self.log_level.upper() in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}, \
            "log_level không hợp lệ"
        assert self.num_workers >= 0, "num_workers không được âm"
        assert self.prefetch_factor is None or self.prefetch_factor >= 1, \
            "prefetch_factor phải >= 1 (hoặc None nếu num_workers=0)"

        if self.num_workers == 0:
            if self.persistent_workers:
                print("[Config][Warning] persistent_workers=True yêu cầu num_workers > 0. "
                      "Tự động đặt lại persistent_workers=False.")
                self.persistent_workers = False
            if self.prefetch_factor is not None:
                print("[Config][Warning] prefetch_factor chỉ có tác dụng khi num_workers > 0. "
                      "Tự động đặt lại prefetch_factor=None.")
                self.prefetch_factor = None
