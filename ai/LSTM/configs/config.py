from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, List, Dict, Any
import os
import json
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class TrainConfig:
    """
    Cấu hình toàn bộ hệ thống huấn luyện cho mô hình Deep LSTM / GRU (Drowsiness Detection).
    Hỗ trợ 2 chế độ:
    1. Chế độ Tensor .pt siêu tốc (Fast Preloaded Tensor Mode - Khuyến nghị & mặc định).
    2. Chế độ Video thô trích xuất qua CNN (Legacy Raw Video Mode).
    """

    # ---- 1. DATASET CONFIGURATION (HDF5 TRAIN & RAW VIDEO ONNX VAL) ----
    use_preloaded_pt: bool = True  # True: Dùng Tensor .pt đã trích xuất | False: Đọc video thô qua OpenCV
    train_pt: str = "extracted_features_pt/features_sust_train.pt"  # Đường dẫn tệp tensor train cũ
    val_pt: str = "extracted_features_pt/features_sust_val.pt"  # Đường dẫn tệp tensor validation cũ
    
    # Tập huấn luyện HDF5 (src/dataset.py: HDF5FeatureDataset)
    train_h5: str = "dataset_features.h5"  # Đường dẫn tệp HDF5 trích xuất sẵn (.h5)
    train_manifest_csv: Optional[str] = None  # Đường dẫn file CSV manifest tập train (tùy chọn)
    include_augmented_train: bool = True  # Cho phép nạp các mẫu tăng cường cho tập train
    
    # Tập kiểm định HDF5 (src/dataset.py: HDF5FeatureDataset)
    val_h5: Optional[str] = None  # Đường dẫn tệp HDF5 tập validation (None: dùng chung train_h5 với split='val')
    val_manifest_csv: Optional[str] = None  # Đường dẫn file CSV manifest tập validation (tùy chọn)

    # Tập kiểm định Video thô (Legacy: RawVideoONNXDataset)
    val_dataset_dir: str = r"D:\Project\AI\dataset\filtered_SUST\in_threshold"  # Thư mục video thô validation
    val_manifest: Optional[str] = None  # Đường dẫn tệp CSV/JSON manifest tập validation (tùy chọn)
    val_batch_size: int = 16  # Kích thước batch size cho tập val
    val_num_workers: int = 0  # Số worker tiến trình nạp tập val (0 để chạy tối ưu với ONNX trên GPU)
    min_frames: int = 10  # Số khung hình tối thiểu cho mỗi video clip
    dataset_dir: str = r"D:\Project\AI\dataset\filtered_SUST\in_threshold"  # Đường dẫn tới thư mục video thô (filtered_SUST)
    manifest_file: Optional[str] = None  # Đường dẫn tệp CSV/JSON manifest cho dataset video thô (tùy chọn)
    backbone_neck_checkpoint: Optional[str] = None  # Đường dẫn checkpoint .pt của BackboneNeck (None = dùng mặc định)
    
    # Thông số khung hình chung
    seq_len: Optional[int] = None  # Số lượng khung hình cố định cho mỗi video (None: lấy toàn bộ video)
    sample_interval: float = 0.1  # Khoảng thời gian t (giây) giữa 2 khung hình lấy mẫu (Target FPS = 10.0)
    image_size: Tuple[int, int] = (640, 640)  # Kích thước khung hình (Height, Width) theo backbone_neck.onnx
    video_exts: Tuple[str, ...] = (".avi", ".mp4", ".mkv")  # Các định dạng video hợp lệ
    train_ratio: float = 0.8  # Tỷ lệ chia tập huấn luyện (80% train, 20% validation)
    split_by_subject: bool = True  # Chia dataset theo người tham gia (Subject-independent split)

    # ---- 2. DATALOADER CONFIGURATION ----
    batch_size: int = 16  # Kích thước batch size cho tập train (16 tối ưu cho GPU 4GB)
    num_workers: int = 0  # Số lượng worker nạp tập train (0 tối ưu trên Windows với HDF5)
    pin_memory: bool = False  # Tắt pin_memory trên Windows/tensor lớn để tránh lỗi CUDA driver host alloc OOM
    shuffle: bool = True  # Trộn ngẫu nhiên tập huấn luyện
    drop_last: bool = False  # Không bỏ rơi batch cuối cùng để đánh giá trọn vẹn tập dữ liệu
    persistent_workers: bool = False  # Giữ nguyên worker giữa các epoch
    prefetch_factor: Optional[int] = None  # Số lượng batch prefetch cho mỗi worker
    seed: int = 42  # Seed khởi tạo ngẫu nhiên để đảm bảo tính lặp lại (reproducibility)
    chunk_size: int = 16  # Kích thước chunk chia nhỏ khung hình khi trích xuất qua CNN/ONNX chống tràn VRAM
    use_dummy_cnn: bool = False  # Tự động trích xuất đặc trưng qua Dummy CNN trong collate_fn (chỉ cho raw video)

    # ---- 3. MODEL ARCHITECTURE CONFIGURATION ----
    backbone_onnx_path: str = "checkpoints/backbone.onnx"
    backbone_neck_onnx_path: str = "checkpoints/backbone_neck.onnx"
    cnn_manifest_path: str = r"/outsrc/myCNN\checkpoints_ftCOCO\model_mainfest.json"
    cnn_weights_path: str = r"/outsrc/myCNN\checkpoints_ftCOCO\ft_step00091000.pt"
    cnn_neck_channels: Tuple[int, int, int] = (64, 128, 256)  # Kênh thực tế của (p3, p4, p5) từ backbone_neck.onnx
    cnn_strides: Tuple[int, int, int] = (8, 16, 32)  # Strides tương ứng của (p3, p4, p5)
    cnn_num_features: int = 3  # Số lượng tầng đặc trưng đầu vào (p3, p4, p5)
    cnn_out_channels: int = 448  # Tổng số kênh khi ghép nối (64 + 128 + 256 = 448)
    cnn_spatial_size: Tuple[int, int] = (20, 20)  # Độ phân giải đặc trưng không gian tầng sâu nhất p5 (640 / 32 = 20)
    spatial_fusion: str = "concat"  # Phương thức kết hợp: 'concat' | 'sum' | 'mean' | 'attention'
    adapter_dropout: float = 0.25  # Tỷ lệ Dropout sau Spatial Feature Adapter
    use_norm: bool = True  # Sử dụng LayerNorm trong Spatial Adapter và Dropout trong FC head
    input_dim: int = 256  # Kích thước vector đặc trưng x^t sau Spatial Adapter đưa vào LSTM/GRU
    hidden_dim: int = 192  # Số lượng đơn vị ẩn (hidden units) trong từng khối LSTM/GRU
    num_layers: int = 2  # Số lớp LSTM/GRU xếp chồng (Deep GRU - 2 lớp)
    num_classes: int = 2  # Số lượng lớp đầu ra phân loại (2: Tỉnh táo vs Buồn ngủ)
    dropout: float = 0.35  # Tỷ lệ Dropout giữa các lớp LSTM/GRU
    supervision_mode: str = "attention_pooling"  # 'attention_pooling' (phương án A chuẩn) | 'sequence'

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
    use_scheduler: bool = True  # Bật/Tắt bộ điều chỉnh Learning Rate Scheduler
    scheduler_type: str = "cosine"  # Loại Scheduler: "cosine" | "step" | "plateau"

    # ---- 6. CHECKPOINTS, EARLY STOPPING & DIAGNOSTICS CONFIGURATION ----
    tb_log_dir: str = "logs/tensorboard"  # Thư mục lưu log cho TensorBoard
    log_dir: str = "logs"  # Thư mục lưu log text (.log)
    experiment_name: str = "deepgru_h5train_rawval"  # Tên bài thử nghiệm (experiment)
    checkpoint_dir: str = "checkpoints/experiments"  # Thư mục lưu checkpoint mô hình (.pt)
    save_ckpt_interval_epochs: int = 1  # Số epoch giữa 2 lần lưu checkpoint định kỳ (khi save_all_epochs=False)
    save_best_only: bool = False  # True: Chỉ lưu best checkpoint | False: Lưu định kỳ + last/best
    ckpt_keep_last: Optional[int] = None  # Số lượng checkpoint định kỳ giữ lại (None hoặc 0 = giữ toàn bộ)
    save_all_epochs: bool = True  # True: Lưu checkpoint cho TẤT CẢ các epoch | False: Lưu theo chu kỳ
    enable_resume: bool = False  # Cờ bật/tắt nạp lại: True = Cho phép resume | False = Luôn train mới từ Epoch 1
    resume_epoch: Optional[int] = None  # Số epoch cụ thể cần nạp lại (vd: 10). Chỉ có tác dụng khi enable_resume=True
    resume: str = ""  # Đường dẫn file checkpoint, số epoch hoặc bí danh ('last', 'best') để huấn luyện tiếp
    
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
        # 0. Tự động đồng bộ đường dẫn tập val nếu để trống
        if not self.val_h5:
            self.val_h5 = self.train_h5
        if not self.val_manifest_csv and self.train_manifest_csv:
            self.val_manifest_csv = self.train_manifest_csv

        # 1. Tự động nhận diện môi trường Kaggle và tệp dữ liệu thực tế
        if os.path.exists("/kaggle"):
            if not os.path.exists(self.train_pt):
                kaggle_merged_train = "/kaggle/input/datasets/nyvantran6634/sust4ni1/features_merged_train.pt"
                kaggle_merged_val = "/kaggle/input/datasets/nyvantran6634/sust4ni1/features_merged_val.pt"
                kaggle_sust_train = "/kaggle/input/datasets/nyvantran6634/sust4ni1/features_sust_train.pt"
                kaggle_sust_val = "/kaggle/input/datasets/nyvantran6634/sust4ni1/features_sust_val.pt"
                if os.path.exists(kaggle_merged_train):
                    self.train_pt = kaggle_merged_train
                    self.val_pt = kaggle_merged_val
                elif os.path.exists(kaggle_sust_train):
                    self.train_pt = kaggle_sust_train
                    self.val_pt = kaggle_sust_val
            self.checkpoint_dir = "/kaggle/working/checkpoints"
            self.tb_log_dir = "/kaggle/working/runs"
        else:
            if not os.path.exists(self.train_pt):
                cand_merged_train = "extracted_features_pt/features_merged_train.pt"
                cand_merged_val = "extracted_features_pt/features_merged_val.pt"
                if os.path.exists(cand_merged_train):
                    self.train_pt = cand_merged_train
                    self.val_pt = cand_merged_val

        # 2. Ràng buộc các tham số dữ liệu & dataloader
        if self.seq_len is not None:
            assert self.seq_len > 0, "seq_len phải > 0"
        assert self.batch_size > 0, "batch_size phải > 0"
        assert self.val_batch_size > 0, "val_batch_size phải > 0"
        assert 0.0 < self.train_ratio < 1.0, "train_ratio phải nằm trong khoảng (0.0, 1.0)"
        assert self.num_workers >= 0, "num_workers không được âm"
        assert self.val_num_workers >= 0, "val_num_workers không được âm"
        assert self.epochs > 0, "epochs phải > 0"
        assert self.lr0 > 0.0, "lr0 phải > 0"
        assert self.patience > 0, "patience phải > 0"
        if self.resume_epoch is not None:
            assert self.resume_epoch >= 1, "resume_epoch phải là số nguyên dương >= 1"
        if not self.enable_resume and (self.resume_epoch is not None or self.resume):
            # Cảnh báo nhẹ người dùng rằng cờ enable_resume đang tắt
            pass

        # 3. Cơ chế tự động bảo vệ tài nguyên (Safety Guard) trên Windows và GPU
        if os.name == "nt":
            # Trên Windows, nếu dùng HDF5 với tensor lớn mà num_workers > 0 thì dễ gây lỗi Windows 1455
            if self.num_workers > 0 and self.batch_size > 8:
                print(f"[!] [Safety Guard] Phát hiện hệ điều hành Windows với tensor HDF5 lớn. Tự động chuyển num_workers: {self.num_workers} -> 0 để tránh lỗi IPC Shared Memory (error 1455).")
                self.num_workers = 0
            if self.val_num_workers > 0 and self.val_batch_size > 8:
                self.val_num_workers = 0

            # Trên Windows với tensor 4D/5D dung lượng hàng GB, pin_memory=True kích hoạt cudaHostAlloc gây cạn bộ nhớ đệm
            if self.pin_memory:
                print("[!] [Safety Guard] Tự động tắt pin_memory trên Windows khi xử lý tensor đặc trưng lớn để ngăn lỗi cudaErrorMemoryAllocation.")
                self.pin_memory = False

        # Kiểm tra giới hạn VRAM GPU nếu dùng CUDA
        if self.device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    if total_vram_gb <= 4.5 and self.batch_size > 16:
                        print(f"[!] [Safety Guard] Phát hiện GPU VRAM {total_vram_gb:.1f}GB (<= 4.5GB). Tự động hạ batch_size từ {self.batch_size} xuống 16 để ngăn ngừa CUDA Out of Memory.")
                        self.batch_size = 16
            except Exception:
                pass

        # 4. Xử lý logic khi num_workers == 0
        if self.num_workers == 0:
            if self.persistent_workers:
                self.persistent_workers = False
            if self.prefetch_factor is not None:
                self.prefetch_factor = None

        # 4. Ràng buộc tham số mô hình
        assert self.input_dim > 0, "input_dim phải > 0"
        assert self.hidden_dim > 0, "hidden_dim phải > 0"
        assert self.num_layers > 0, "num_layers phải > 0"
        assert self.num_classes > 0, "num_classes phải > 0"

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

        return cls(**data)

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
