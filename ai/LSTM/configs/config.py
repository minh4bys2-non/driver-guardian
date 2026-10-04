from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, List, Dict, Any, Union
import os
import json
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


# Xác định thư mục gốc dự án (driver-guardian)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Checkpoint PyTorch tối ưu của NMSFreeDetector
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "ai"
    / "ObjectDetection_2p6M"
    / "checkpoints"
    / "2bfb6cc36ef5ab822a944006c746e5d348d67e387501c9027df5b957cf124031"
    / "de0c5b23e1ae749b972f10cb5f3e21e2ccf37bd334eb7a7c3091c460073ff310"
    / "finetune"
    / "best.pt"
)

# Thư mục video thô mặc định đã chuẩn hóa
DEFAULT_DATASET_DIR = (
    "/home/riftuser/.cache/kagglehub/datasets/nyvantran6634/dataset-datn4ni/versions/1/data_processed"
)


@dataclass
class TrainConfig:
    """
    Cấu hình toàn bộ hệ thống huấn luyện cho mô hình Deep GRU (Drowsiness Detection).
    Hỗ trợ luồng huấn luyện trực tiếp từ dữ liệu video thô (Raw Video)
    qua mô hình trích xuất đặc trưng BackboneNeck (NMSFreeDetector).
    """

    # ---- 1. DATASET CONFIGURATION (RAW VIDEO & NMSFreeDetector) ----
    dataset_dir: str = DEFAULT_DATASET_DIR  # Đường dẫn thư mục video thô
    manifest_file: Optional[str] = "dataset_merged_split.csv"  # Tệp CSV/JSON manifest tập dữ liệu
    val_dataset_dir: Optional[str] = None  # Thư mục video thô val riêng biệt (None = dùng dataset_dir/val hoặc split)
    val_manifest: Optional[str] = None  # Tệp CSV/JSON manifest tập val riêng biệt
    backbone_neck_checkpoint: str = str(DEFAULT_CHECKPOINT_PATH)  # Checkpoint .pt của NMSFreeDetector
    sample_interval: float = 0.1  # Khoảng thời gian (giây) giữa 2 khung hình lấy mẫu (Target FPS = 10.0)
    seq_len: Optional[int] = 50  # Số lượng khung hình cố định cho mỗi video clip
    image_size: Tuple[int, int] = (640, 640)  # Kích thước khung hình (Height, Width) input NMSFreeDetector
    video_exts: Tuple[str, ...] = (".avi", ".mp4", ".mkv", ".mov")  # Các định dạng video hợp lệ
    train_ratio: float = 0.8  # Tỷ lệ chia tập huấn luyện (80% train, 20% validation) khi không có thư mục val
    split_by_subject: bool = True  # Chia dataset theo người tham gia (Subject-independent split)
    min_frames: int = 10  # Số khung hình tối thiểu cho mỗi video clip hợp lệ
    use_augmentation: bool = True  # Bật/tắt tăng cường dữ liệu thời gian cho tập huấn luyện (src/augment.py)

    # ---- 2. DATALOADER CONFIGURATION ----
    batch_size: int = 16  # Kích thước batch size cho tập train
    val_batch_size: int = 16  # Kích thước batch size cho tập val
    num_workers: int = 2  # Số worker tiến trình nạp train (2 tối ưu song song hóa I/O trên Linux)
    val_num_workers: int = 2  # Số worker tiến trình nạp val (2 tối ưu song song hóa I/O trên Linux)
    pin_memory: bool = False  # Tránh lỗi CUDA host alloc OOM khi xử lý video lớn
    shuffle: bool = True  # Trộn ngẫu nhiên tập huấn luyện
    drop_last: bool = False  # Không bỏ rơi batch cuối cùng để đánh giá trọn vẹn tập dữ liệu
    persistent_workers: bool = False  # Giữ worker giữa các epoch (False khi num_workers=0)
    prefetch_factor: Optional[int] = None  # Số batch prefetch cho mỗi worker
    seed: int = 42  # Seed khởi tạo ngẫu nhiên để đảm bảo tính lặp lại (reproducibility)
    chunk_size: int = 16  # Kích thước chunk chia nhỏ khung hình khi suy luận NMSFreeDetector chống tràn VRAM

    # ---- 3. MODEL ARCHITECTURE CONFIGURATION (NMSFreeDetector PAFPN + Deep GRU) ----
    cnn_neck_channels: Tuple[int, int, int] = (64, 128, 256)  # Kênh thực tế của (p3, p4, p5) từ NMSFreeDetector PAFPN
    cnn_strides: Tuple[int, int, int] = (8, 16, 32)  # Strides tương ứng của (p3, p4, p5)
    cnn_num_features: int = 3  # Số lượng tầng đặc trưng đầu vào (p3, p4, p5)
    cnn_out_channels: int = 448  # Tổng số kênh khi ghép nối (64 + 128 + 256 = 448)
    cnn_spatial_size: Tuple[int, int] = (20, 20)  # Độ phân giải đặc trưng không gian tầng sâu nhất p5 (640 / 32 = 20)
    spatial_fusion: str = "concat"  # Phương thức kết hợp: 'concat' | 'attention'
    adapter_dropout: float = 0.25  # Tỷ lệ Dropout sau Spatial Feature Adapter
    use_norm: bool = True  # Sử dụng LayerNorm trong Spatial Adapter
    input_dim: int = 256  # Kích thước vector đặc trưng x^t sau Spatial Adapter đưa vào GRU
    hidden_dim: int = 192  # Số lượng đơn vị ẩn (hidden units) trong từng khối GRU
    num_layers: int = 2  # Số lớp GRU xếp chồng (Deep GRU - 2 lớp)
    num_classes: int = 2  # Số lượng lớp đầu ra phân loại (2: Tỉnh táo vs Buồn ngủ)
    dropout: float = 0.35  # Tỷ lệ Dropout giữa các lớp GRU
    supervision_mode: str = "attention_pooling"  # 'attention_pooling' (phương án chuẩn)

    # ---- 4. LOSS CONFIGURATION ----
    loss_type: str = "ce"  # Loại loss: "ce" (CrossEntropyLoss chuẩn) hoặc "bce" (BCELoss)
    bce_eps: float = 1e-7  # Hằng số epsilon kẹp giá trị tránh log(0) (nếu dùng bce)
    bce_reduction: str = "mean"  # Cách gom nhóm loss ('mean' | 'sum' | 'none')
    pos_weight: Optional[float] = None  # Trọng số cho lớp dương (1: Buồn ngủ) khi mất cân bằng dữ liệu

    # ---- 5. OPTIMIZER & SCHEDULER CONFIGURATION ----
    epochs: int = 40  # Tổng số epoch huấn luyện
    lr0: float = 1e-3  # Learning rate khởi tạo
    lr_min_factor: float = 0.01  # Hệ số lr tối thiểu: lr_min = lr0 * lr_min_factor (1e-5)
    weight_decay: float = 1e-4  # Trọng số phạt L2 regularization
    warmup_epochs: float = 1.0  # Số epoch khởi động mềm (Warmup)
    optimizer: str = "adamw"  # Bộ tối ưu hóa: "adamw" | "adam" | "sgd"
    betas: Tuple[float, float] = (0.9, 0.999)  # Tham số betas cho Adam/AdamW
    momentum: float = 0.9  # Động lượng Momentum khi optimizer="sgd"
    grad_clip_norm: float = 1.0  # Ngưỡng cắt gradient (Gradient Clipping) tránh bùng nổ gradient
    gradient_accumulation_steps: int = 1  # Số bước tích lũy gradient trước khi cập nhật trọng số (giảm đỉnh VRAM)
    empty_cache_interval: int = 0  # Số step giữa các lần giải phóng cache GPU (0: chỉ dọn mốc epoch; >0: dọn định kỳ)
    use_scheduler: bool = True  # Bật/Tắt bộ điều chỉnh Learning Rate Scheduler
    scheduler_type: str = "cosine"  # Loại Scheduler: "cosine" | "step" | "plateau"

    # ---- 6. CHECKPOINTS, EARLY STOPPING & DIAGNOSTICS CONFIGURATION ----
    tb_log_dir: str = "logs/tensorboard"  # Thư mục lưu log cho TensorBoard
    log_dir: str = "logs"  # Thư mục lưu log text (.log)
    experiment_name: str = "deepgru_raw_nmsfree"  # Tên bài thử nghiệm (experiment)
    checkpoint_dir: str = "checkpoints/experiments"  # Thư mục lưu checkpoint mô hình (.pt)
    save_ckpt_interval_epochs: int = 1  # Số epoch giữa 2 lần lưu checkpoint định kỳ
    save_best_only: bool = False  # True: Chỉ lưu best checkpoint | False: Lưu định kỳ + last/best
    ckpt_keep_last: Optional[int] = None  # Số lượng checkpoint định kỳ giữ lại (None = giữ toàn bộ)
    save_all_epochs: bool = True  # True: Lưu checkpoint cho TẤT CẢ các epoch
    enable_resume: bool = False  # Cờ bật/tắt nạp lại: True = Cho phép resume | False = Luôn train mới từ Epoch 1
    resume_epoch: Optional[int] = None  # Số epoch cụ thể cần nạp lại
    resume: str = ""  # Đường dẫn file checkpoint, số epoch hoặc bí danh ('last', 'best') để huấn luyện tiếp

    # Cấu hình Giám sát Phân giải cao theo từng Step (Per-Step Metrics trên TensorBoard)
    enable_step_logging: bool = True  # Bật/tắt tính toán và ghi nhận per-step metrics lên TensorBoard
    log_step_interval: int = 5  # Số batch steps giữa các lần ghi TensorBoard (mặc định = 5)
    flush_step_interval: int = 50  # Số batch steps giữa các lần flush đĩa SummaryWriter (mặc định = 50)
    ema_beta: float = 0.95  # Hệ số làm mượt hàm mất mát Exponential Moving Average (EMA)
    
    # Cơ chế Dừng sớm (Early Stopping)
    early_stopping: bool = True  # Bật/tắt Early Stopping
    patience: int = 10  # Số epoch chờ không cải thiện trước khi dừng sớm
    monitor_metric: str = "val_f1"  # Chỉ số theo dõi: "val_f1", "val_loss", "val_acc", "val_recall"
    monitor_mode: str = "max"  # Chế độ theo dõi: "max" (cho f1/acc/recall) hoặc "min" (cho loss)
    min_delta: float = 1e-4  # Mức cải thiện tối thiểu để tính là tiến bộ

    # File kết quả và biểu đồ chẩn đoán (Diagnostic Plots & Reports)
    history_csv_path: str = "logs/training_history.csv"  # File lưu bảng lịch sử các epoch
    summary_json_path: str = "logs/training_summary.json"  # File lưu kết quả tóm tắt cuối cùng
    plot_curves_path: str = "logs/loss_accuracy_curves.png"  # File ảnh biểu đồ loss / accuracy
    plot_cm_path: str = "logs/confusion_matrix_best.png"  # File ảnh ma trận nhầm lẫn
    plot_roc_path: str = "logs/roc_pr_curves.png"  # File ảnh đường cong ROC & PR
    dry_run: bool = False  # Cờ bật chế độ kiểm thử nhanh mô phỏng (mock data)

    # ---- 7. RUNTIME & HARDWARE CONFIGURATION ----
    device: str = "cuda"  # Thiết bị tính toán ("cuda" | "cpu")
    amp: bool = True  # Bật Tự động ép kiểu chính xác hỗn hợp (Automatic Mixed Precision FP16)
    log_interval: int = 10  # Số step giữa các lần in log tiến trình chi tiết
    val_interval_epochs: int = 1  # Số epoch giữa 2 lần đánh giá tập Validation
    use_tqdm: bool = True  # Bật/tắt thanh tiến trình giám sát trực quan qua tqdm

    def __post_init__(self):
        """Kiểm tra tính hợp lệ của tham số cấu hình và tự động điều chỉnh theo môi trường."""
        # 1. Chuẩn hóa đường dẫn val_dataset_dir
        if self.val_dataset_dir == "":
            self.val_dataset_dir = None

        # 2. Tự động kiểm tra và giải quyết đường dẫn checkpoint NMSFreeDetector
        if self.backbone_neck_checkpoint:
            ckpt_path = Path(self.backbone_neck_checkpoint)
            if not ckpt_path.exists():
                # Thử tìm tương đối so với PROJECT_ROOT
                alt_path = PROJECT_ROOT / self.backbone_neck_checkpoint
                if alt_path.exists():
                    self.backbone_neck_checkpoint = str(alt_path)
                elif DEFAULT_CHECKPOINT_PATH.exists():
                    self.backbone_neck_checkpoint = str(DEFAULT_CHECKPOINT_PATH)

        # 3. Ràng buộc các tham số dữ liệu & dataloader
        if self.seq_len is not None:
            assert self.seq_len > 0, "seq_len phải > 0"
        assert self.batch_size > 0, "batch_size phải > 0"
        assert self.val_batch_size > 0, "val_batch_size phải > 0"
        assert 0.0 < self.train_ratio < 1.0, "train_ratio phải nằm trong khoảng (0.0, 1.0)"
        assert self.num_workers >= 0, "num_workers không được âm"
        assert self.val_num_workers >= 0, "val_num_workers không được âm"
        assert self.epochs > 0, "epochs phải > 0"
        assert self.lr0 > 0.0, "lr0 phải > 0"
        assert self.gradient_accumulation_steps >= 1, "gradient_accumulation_steps phải là số nguyên >= 1"
        assert self.empty_cache_interval >= 0, "empty_cache_interval không được âm"
        assert self.patience > 0, "patience phải > 0"
        if self.resume_epoch is not None:
            assert self.resume_epoch >= 1, "resume_epoch phải là số nguyên dương >= 1"

        # 4. Cơ chế tự động bảo vệ tài nguyên (Safety Guard) trên Windows và GPU
        if os.name == "nt":
            if self.num_workers > 0 and self.batch_size > 8:
                print(f"[!] [Safety Guard] Windows: tự động chuyển num_workers: {self.num_workers} -> 0.")
                self.num_workers = 0
            if self.val_num_workers > 0 and self.val_batch_size > 8:
                self.val_num_workers = 0
            if self.pin_memory:
                self.pin_memory = False

        # Kiểm tra giới hạn VRAM GPU nếu dùng CUDA
        if self.device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    if total_vram_gb <= 4.5 and self.batch_size > 16:
                        print(f"[!] [Safety Guard] GPU VRAM {total_vram_gb:.1f}GB. Tự động hạ batch_size từ {self.batch_size} xuống 16.")
                        self.batch_size = 16
            except Exception:
                pass

        # 5. Xử lý logic khi num_workers == 0
        if self.num_workers == 0:
            if self.persistent_workers:
                self.persistent_workers = False
            if self.prefetch_factor is not None:
                self.prefetch_factor = None

        # 6. Ràng buộc tham số mô hình
        assert self.input_dim > 0, "input_dim phải > 0"
        assert self.hidden_dim > 0, "hidden_dim phải > 0"
        assert self.num_layers > 0, "num_layers phải > 0"
        assert self.num_classes > 0, "num_classes phải > 0"

        # 7. Ràng buộc tham số ghi nhận log cấp độ Step
        assert self.log_step_interval >= 1, "log_step_interval phải >= 1"
        assert self.flush_step_interval >= 1, "flush_step_interval phải >= 1"
        assert 0.0 < self.ema_beta < 1.0, "ema_beta phải nằm trong khoảng (0, 1)"

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi Config sang dạng Dictionary."""
        return asdict(self)

    def save_json(self, json_path: str) -> None:
        """Lưu cấu hình ra file JSON."""
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)
        print(f"[TrainConfig] Đã lưu cấu hình vào: {path.resolve()}")

    @classmethod
    def load_json(cls, json_path: str) -> "TrainConfig":
        """Nạp cấu hình từ file JSON."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file config JSON: {json_path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tuple_fields = ["image_size", "video_exts", "cnn_neck_channels", "cnn_strides", "cnn_spatial_size", "betas"]
        for fld in tuple_fields:
            if fld in data and isinstance(data[fld], list):
                data[fld] = tuple(data[fld])

        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def load_yaml(cls, yaml_path: str) -> "TrainConfig":
        """Nạp cấu hình từ file YAML."""
        if yaml is None:
            raise ImportError("Vui lòng cài đặt pyyaml (`pip install pyyaml`) để đọc file YAML.")
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file config YAML: {yaml_path}")
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        flattened: Dict[str, Any] = {}
        if isinstance(raw_data, dict):
            for section, values in raw_data.items():
                if isinstance(values, dict):
                    flattened.update(values)
                else:
                    flattened[section] = values

        tuple_fields = ["image_size", "video_exts", "cnn_neck_channels", "cnn_strides", "cnn_spatial_size", "betas"]
        for fld in tuple_fields:
            if fld in flattened and isinstance(flattened[fld], list):
                flattened[fld] = tuple(flattened[fld])

        # Lọc các key không thuộc TrainConfig fields
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in flattened.items() if k in valid_keys}
        return cls(**filtered)


def load_config(config_path: Optional[str] = None) -> TrainConfig:
    """Tải cấu hình từ đường dẫn YAML hoặc JSON; mặc định nạp configs/config.yaml nếu có."""
    if config_path is None:
        default_yaml = Path(__file__).parent / "config.yaml"
        if default_yaml.exists():
            return TrainConfig.load_yaml(str(default_yaml))
        return TrainConfig()

    p = Path(config_path)
    if p.suffix.lower() in [".yaml", ".yml"]:
        return TrainConfig.load_yaml(str(p))
    elif p.suffix.lower() == ".json":
        return TrainConfig.load_json(str(p))
    else:
        raise ValueError(f"Định dạng tệp cấu hình không được hỗ trợ: {p.suffix}")
