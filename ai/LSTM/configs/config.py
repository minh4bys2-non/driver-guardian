#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tệp: configs/config.py
Mục đích:
    Định nghĩa cấu hình hệ thống huấn luyện và kiểm định mô hình SpatioTemporal ConvGRU.
    Hỗ trợ luồng huấn luyện trực tiếp từ dữ liệu video thô (Raw Video)
    qua mô hình trích xuất đặc trưng BackboneNeck (NMSFreeDetector PAFPN).

Ghi chú:
    Toàn bộ tham số và giá trị mặc định được rà soát và đồng bộ chuẩn xác từ checkpoint:
    checkpoints/experiments/deepgru_raw_nmsfree/best.pt.
    Loại bỏ toàn bộ các tham số dư thừa/không sử dụng.
"""

from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict, Any, Union
import os
import json
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


# Xác định thư mục gốc dự án (driver-guardian)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Checkpoint PyTorch tối ưu của NMSFreeDetector BackboneNeck
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "ai"
    / "ObjectDetection_2p6M"
    / "checkpoints"
    / "231e35bb4061257f9bcb7cc5e3a0d0064e5e1beb6351a5e0adfc8d0c91ba4e11"
    / "da92589d1b67339be4491beaf147575e81a2699bbde06d654c05f852d7f0e9c9"
    / "landmark_train"
    / "best.pt"
)

# Thư mục video thô mặc định
DEFAULT_DATASET_DIR = "E:/LSTM/data_processed"


@dataclass
class TrainConfig:
    """
    Cấu hình toàn bộ hệ thống huấn luyện cho mô hình ConvGRU (Drowsiness Detection).
    Chỉ lưu trữ các tham số thực tế được sử dụng trong codebase và mô hình.
    """

    # ---- 1. DATASET CONFIGURATION (RAW VIDEO & NMSFreeDetector) ----
    dataset_dir: str = DEFAULT_DATASET_DIR  # Đường dẫn thư mục video thô
    manifest_file: Optional[str] = "dataset_merged_split.csv"  # Tệp CSV/JSON manifest tập dữ liệu
    backbone_neck_checkpoint: str = str(DEFAULT_CHECKPOINT_PATH)  # Checkpoint .pt của BackboneNeck
    sample_interval: float = 0.2  # Khoảng thời gian (giây) giữa 2 khung hình lấy mẫu (FPS = 5.0)
    seq_len: Optional[int] = None  # Số khung hình cố định (None = dynamic padding theo độ dài clip)
    image_size: Tuple[int, int] = (640, 640)  # Kích thước khung hình (Height, Width) input Backbone
    video_exts: Tuple[str, ...] = (".avi", ".mp4", ".mkv", ".mov")  # Định dạng video hợp lệ
    min_frames: int = 10  # Số khung hình tối thiểu cho mỗi video clip hợp lệ
    use_augmentation: bool = True  # Bật/tắt tăng cường dữ liệu video cho tập huấn luyện

    # ---- 2. DATALOADER CONFIGURATION ----
    batch_size: int = 16  # Kích thước batch size cho tập train
    val_batch_size: int = 16  # Kích thước batch size cho tập val
    num_workers: int = 0  # Số worker tiến trình nạp train (0 là an toàn nhất trên Windows)
    val_num_workers: int = 0  # Số worker tiến trình nạp val
    pin_memory: bool = False  # Tránh lỗi CUDA host alloc OOM khi xử lý video lớn
    shuffle: bool = True  # Trộn ngẫu nhiên tập huấn luyện
    seed: int = 42  # Seed khởi tạo ngẫu nhiên để đảm bảo tính lặp lại (reproducibility)
    chunk_size: int = 32  # Kích thước chunk chia nhỏ khung hình khi chạy Backbone chống tràn VRAM

    # ---- 3. MODEL ARCHITECTURE CONFIGURATION (NMSFreeDetector PAFPN + ConvGRU) ----
    cnn_neck_channels: Tuple[int, int, int] = (64, 128, 256)  # Kênh thực tế của (p3, p4, p5) từ PAFPN
    adapter_dropout: float = 0.25  # Tỷ lệ Dropout sau cổ giảm kênh SpatialReductionNeck
    input_dim: int = 128  # Kích thước kênh sau SpatialReductionNeck đưa vào ConvGRU
    convgru_in_dim: int = 128  # Bí danh đồng bộ cho input_dim được đọc bởi models.py
    hidden_dim: int = 64  # Số kênh ẩn trong từng tầng ConvGRU (chuẩn hóa theo best.pt)
    convgru_hidden_dim: int = 64  # Bí danh đồng bộ cho hidden_dim được đọc bởi models.py
    num_layers: int = 2  # Số tầng ConvGRU xếp chồng
    convgru_num_layers: int = 2  # Bí danh đồng bộ cho num_layers được đọc bởi models.py
    num_classes: int = 2  # Số lượng lớp đầu ra phân loại (2: Tỉnh táo vs Buồn ngủ)
    dropout: float = 0.35  # Tỷ lệ Dropout cho đầu phân loại FC Head
    supervision_mode: str = "attention_pooling"  # 'attention_pooling' (phương án chuẩn)

    # ---- 4. LOSS CONFIGURATION ----
    loss_type: str = "ce"  # Loại loss: "ce" (CrossEntropyLoss chuẩn) hoặc "bce" (BCEWithLogitsLoss)
    bce_eps: float = 1e-7  # Hằng số epsilon kẹp giá trị tránh log(0) khi dùng bce
    bce_reduction: str = "mean"  # Cách gom nhóm loss ('mean' | 'sum')
    pos_weight: Optional[float] = None  # Trọng số cho lớp dương (1: Buồn ngủ) khi dữ liệu lệch

    # ---- 5. OPTIMIZER & SCHEDULER CONFIGURATION ----
    epochs: int = 20  # Tổng số epoch huấn luyện
    lr0: float = 0.001  # Learning rate khởi tạo
    lr_min_factor: float = 0.01  # Hệ số lr tối thiểu: lr_min = lr0 * lr_min_factor (1e-5)
    weight_decay: float = 0.0001  # Trọng số phạt L2 regularization
    optimizer: str = "adamw"  # Bộ tối ưu hóa: "adamw" | "adam" | "sgd"
    betas: Tuple[float, float] = (0.9, 0.999)  # Tham số betas cho Adam/AdamW
    momentum: float = 0.9  # Động lượng Momentum khi optimizer="sgd"
    grad_clip_norm: float = 1.0  # Ngưỡng cắt gradient (Gradient Clipping) tránh bùng nổ gradient
    gradient_accumulation_steps: int = 1  # Số bước tích lũy gradient trước khi cập nhật trọng số
    empty_cache_interval: int = 2  # Số epoch giữa các lần dọn cache GPU
    use_scheduler: bool = True  # Bật/Tắt bộ điều chỉnh Learning Rate Scheduler (CosineAnnealingLR)

    # ---- 6. CHECKPOINTS, EARLY STOPPING & LOGGING CONFIGURATION ----
    tb_log_dir: str = "logs/tensorboard"  # Thư mục lưu log TensorBoard
    experiment_name: str = "deepgru_raw_nmsfree"  # Tên bài thử nghiệm
    checkpoint_dir: str = "checkpoints/experiments"  # Thư mục lưu checkpoint mô hình (.pt)
    save_ckpt_interval_epochs: int = 1  # Số epoch giữa 2 lần lưu checkpoint định kỳ
    save_all_epochs: bool = True  # True: Lưu checkpoint cho TẤT CẢ các epoch
    ckpt_keep_last: Optional[int] = None  # Số lượng checkpoint định kỳ giữ lại (None = giữ toàn bộ)
    enable_resume: bool = False  # Bật/tắt nạp lại: True = Cho phép resume | False = Train mới
    resume: str = ""  # Đường dẫn file checkpoint hoặc bí danh ('last', 'best') để resume

    # Cấu hình Giám sát Phân giải cao theo từng Step (Per-Step Metrics trên TensorBoard)
    enable_step_logging: bool = True  # Bật/tắt tính toán và ghi nhận per-step metrics
    log_step_interval: int = 1  # Số batch steps giữa các lần ghi TensorBoard

    # Cơ chế Dừng sớm (Early Stopping)
    early_stopping: bool = True  # Bật/tắt Early Stopping
    patience: int = 10  # Số epoch chờ không cải thiện trước khi dừng sớm
    monitor_metric: str = "val_f1"  # Chỉ số theo dõi: "val_f1", "val_loss", "val_acc", "val_recall"
    monitor_mode: str = "max"  # Chế độ theo dõi: "max" hoặc "min"
    min_delta: float = 0.0001  # Mức cải thiện tối thiểu để tính là tiến bộ

    # File lịch sử huấn luyện
    history_csv_path: str = "logs/training_history.csv"  # File lưu bảng lịch sử các epoch

    # ---- 7. RUNTIME & HARDWARE CONFIGURATION ----
    device: str = "cuda"  # Thiết bị tính toán ("cuda" | "cpu")
    amp: bool = True  # Bật Tự động ép kiểu chính xác hỗn hợp (FP16)
    val_interval_epochs: int = 1  # Số epoch giữa 2 lần đánh giá tập Validation
    use_tqdm: bool = True  # Bật/tắt thanh tiến trình trực quan qua tqdm

    def __post_init__(self):
        """Kiểm tra tính hợp lệ của tham số cấu hình và tự động điều chỉnh theo môi trường."""
        # 1. Tự động kiểm tra và giải quyết đường dẫn checkpoint Backbone
        if self.backbone_neck_checkpoint:
            ckpt_path = Path(self.backbone_neck_checkpoint)
            if not ckpt_path.exists():
                alt_path = PROJECT_ROOT / self.backbone_neck_checkpoint
                if alt_path.exists():
                    self.backbone_neck_checkpoint = str(alt_path)
                elif DEFAULT_CHECKPOINT_PATH.exists():
                    self.backbone_neck_checkpoint = str(DEFAULT_CHECKPOINT_PATH)

        # 2. Đồng bộ các bí danh đọc bởi models.py
        if self.convgru_hidden_dim != self.hidden_dim:
            self.convgru_hidden_dim = self.hidden_dim
        if self.convgru_in_dim != self.input_dim:
            self.convgru_in_dim = self.input_dim
        if self.convgru_num_layers != self.num_layers:
            self.convgru_num_layers = self.num_layers

        # 3. Ràng buộc các tham số dữ liệu & dataloader
        if self.seq_len is not None:
            assert self.seq_len > 0, "seq_len phải > 0"
        assert self.batch_size > 0, "batch_size phải > 0"
        assert self.val_batch_size > 0, "val_batch_size phải > 0"
        assert self.num_workers >= 0, "num_workers không được âm"
        assert self.val_num_workers >= 0, "val_num_workers không được âm"
        assert self.epochs > 0, "epochs phải > 0"
        assert self.lr0 > 0.0, "lr0 phải > 0"
        assert self.gradient_accumulation_steps >= 1, "gradient_accumulation_steps phải là số nguyên >= 1"
        assert self.empty_cache_interval >= 0, "empty_cache_interval không được âm"
        assert self.patience > 0, "patience phải > 0"

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

        # 5. Ràng buộc tham số mô hình
        assert self.input_dim > 0, "input_dim phải > 0"
        assert self.hidden_dim > 0, "hidden_dim phải > 0"
        assert self.num_layers > 0, "num_layers phải > 0"
        assert self.num_classes > 0, "num_classes phải > 0"

        # 6. Ràng buộc tham số ghi nhận log
        assert self.log_step_interval >= 1, "log_step_interval phải >= 1"

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

        tuple_fields = ["image_size", "video_exts", "cnn_neck_channels", "betas"]
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

        tuple_fields = ["image_size", "video_exts", "cnn_neck_channels", "betas"]
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
