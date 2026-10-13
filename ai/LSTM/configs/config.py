#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tệp: configs/config.py
Mục đích:
    Định nghĩa cấu hình hệ thống huấn luyện và kiểm định mô hình SpatioTemporal ConvGRU
    và mô hình phân loại toàn diện End-to-End DrowsyClassifier.
    Hỗ trợ cả 2 luồng:
      1. Huấn luyện trích xuất 2 giai đoạn (Offline Backbone PAFPN + ConvGRUClassifier) qua src/train.py.
      2. Huấn luyện toàn diện End-to-End từ video thô (DrowsyClassifier) qua src/train1.py.

Ghi chú:
    Toàn bộ tham số dùng chung giữa 2 mô hình/luồng được chú thích rõ là [DÙNG CHUNG].
    Các tham số đặc thù được chú thích [DÀNH RIÊNG DrowsyClassifier] hoặc [DÀNH RIÊNG ConvGRUClassifier].
"""

from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict, Any, Union
import os
import sys
import json
from pathlib import Path

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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
    Cấu hình toàn bộ hệ thống huấn luyện cho mô hình ConvGRU và DrowsyClassifier (Drowsiness Detection).
    Chỉ lưu trữ các tham số thực tế được sử dụng trong codebase và mô hình.
    """

    # ---- 1. DATASET CONFIGURATION (RAW VIDEO) ----
    dataset_dir: str = DEFAULT_DATASET_DIR  # [DÙNG CHUNG] Đường dẫn thư mục video thô
    manifest_file: Optional[str] = "dataset_merged_split.csv"  # [DÙNG CHUNG] Tệp CSV/JSON manifest tập dữ liệu
    backbone_neck_checkpoint: str = str(DEFAULT_CHECKPOINT_PATH)  # [DÀNH RIÊNG train.py] Checkpoint .pt của BackboneNeck
    sample_interval: float = 0.2  # [DÙNG CHUNG] Khoảng thời gian (giây) giữa 2 khung hình lấy mẫu (FPS = 5.0)
    seq_len: Optional[int] = None  # [DÙNG CHUNG] Số khung hình cố định (None = dynamic padding theo độ dài clip)
    image_size: Tuple[int, int] = (640, 640)  # [DÙNG CHUNG] Kích thước khung hình (Height, Width) input Backbone
    video_exts: Tuple[str, ...] = (".avi", ".mp4", ".mkv", ".mov")  # [DÙNG CHUNG] Định dạng video hợp lệ
    min_frames: int = 10  # [DÙNG CHUNG] Số khung hình tối thiểu cho mỗi video clip hợp lệ
    use_augmentation: bool = True  # [DÙNG CHUNG] Bật/tắt tăng cường dữ liệu video cho tập huấn luyện

    # ---- 2. DATALOADER CONFIGURATION ----
    batch_size: int = 16  # [DÙNG CHUNG] Kích thước batch size cho tập train
    val_batch_size: int = 16  # [DÙNG CHUNG] Kích thước batch size cho tập val
    num_workers: int = 0  # [DÙNG CHUNG] Số worker tiến trình nạp train (0 là an toàn nhất trên Windows)
    val_num_workers: int = 0  # [DÙNG CHUNG] Số worker tiến trình nạp val
    pin_memory: bool = False  # [DÙNG CHUNG] Tránh lỗi CUDA host alloc OOM khi xử lý video lớn
    shuffle: bool = True  # [DÙNG CHUNG] Trộn ngẫu nhiên tập huấn luyện
    seed: int = 42  # [DÙNG CHUNG] Seed khởi tạo ngẫu nhiên để đảm bảo tính lặp lại (reproducibility)
    chunk_size: int = 32  # [DÀNH RIÊNG train.py] Kích thước chunk chia nhỏ khung hình khi chạy Backbone tách rời

    # ---- 3. MODEL ARCHITECTURE CONFIGURATION ----
    # [DÙNG CHUNG] Số lượng lớp đầu ra phân loại (2: Tỉnh táo vs Buồn ngủ)
    num_classes: int = 2
    # [DÙNG CHUNG] Kích thước kênh sau SpatialReductionNeck đưa vào ConvGRU
    input_dim: int = 128
    # [DÙNG CHUNG] Bí danh đồng bộ cho input_dim được đọc bởi models.py
    convgru_in_dim: int = 128
    # [DÙNG CHUNG] Số kênh ẩn trong từng tầng ConvGRU
    hidden_dim: Union[int, Tuple[int, ...]] = 128
    # [DÙNG CHUNG] Bí danh đồng bộ cho hidden_dim được đọc bởi models.py
    convgru_hidden_dim: Union[int, Tuple[int, ...]] = 128
    # [DÙNG CHUNG] Số tầng ConvGRU xếp chồng
    num_layers: int = 2
    # [DÙNG CHUNG] Bí danh đồng bộ cho num_layers được đọc bởi models.py
    convgru_num_layers: int = 2
    # [DÙNG CHUNG] Tỷ lệ Dropout cho đầu phân loại FC Head
    dropout: float = 0.35
    # [DÙNG CHUNG] Cơ chế giám sát chuỗi thời gian ('attention_pooling' | 'sequence')
    supervision_mode: str = "attention_pooling"
    # [DÙNG CHUNG] Kích thước lưới không gian hội tụ (Height, Width) sau SpatialReductionNeck
    target_size: Tuple[int, int] = (14, 14)

    # ---- THAM SỐ DÀNH RIÊNG CHO DROWSYCLASSIFIER (src/models1.py & src/train1.py) ----
    # [DÀNH RIÊNG DrowsyClassifier] Độ rộng kênh các tầng Backbone (c1, c2, c3, c4, c5)
    backbone_w: Tuple[int, ...] = (16, 32, 64, 128, 256)
    # [DÀNH RIÊNG DrowsyClassifier] Số lượng khối lặp lại trong từng tầng Backbone
    backbone_n: Tuple[int, ...] = (1, 2, 2, 1)
    # [DÀNH RIÊNG DrowsyClassifier] Số lượng khối lặp lại trong PAFPN
    neck_n: int = 1
    # [DÀNH RIÊNG DrowsyClassifier/train1.py] Bật gradient checkpointing cho Backbone/Neck tiết kiệm VRAM
    gradient_checkpointing: bool = True
    # [DÀNH RIÊNG DrowsyClassifier/train1.py] Khóa running mean/var của BatchNorm trong Backbone/Neck
    freeze_bn: bool = False

    # ---- THAM SỐ DÀNH RIÊNG CHO CONVGRU TÁCH RỜI (src/models.py & src/train.py) ----
    # [DÀNH RIÊNG ConvGRUClassifier] Kênh thực tế của (p3, p4, p5) từ Backbone PAFPN ngoài
    cnn_neck_channels: Tuple[int, int, int] = (64, 128, 256)
    # [DÀNH RIÊNG ConvGRUClassifier] Tỷ lệ Dropout sau cổ giảm kênh SpatialReductionNeck
    adapter_dropout: float = 0.25

    # ---- 4. LOSS CONFIGURATION ----
    loss_type: str = "ce"  # [DÙNG CHUNG] Loại loss: "ce" (CrossEntropyLoss chuẩn) hoặc "bce" (BCEWithLogitsLoss)
    bce_eps: float = 1e-7  # [DÙNG CHUNG] Hằng số epsilon kẹp giá trị tránh log(0) khi dùng bce
    bce_reduction: str = "mean"  # [DÙNG CHUNG] Cách gom nhóm loss ('mean' | 'sum')
    pos_weight: Optional[float] = None  # [DÙNG CHUNG] Trọng số cho lớp dương (1: Buồn ngủ) khi dữ liệu lệch

    # ---- 5. OPTIMIZER & SCHEDULER CONFIGURATION ----
    epochs: int = 20  # [DÙNG CHUNG] Tổng số epoch huấn luyện
    lr0: float = 0.001  # [DÙNG CHUNG] Learning rate khởi tạo
    lr_min_factor: float = 0.01  # [DÙNG CHUNG] Hệ số lr tối thiểu: lr_min = lr0 * lr_min_factor (1e-5)
    weight_decay: float = 0.0001  # [DÙNG CHUNG] Trọng số phạt L2 regularization
    optimizer: str = "adamw"  # [DÙNG CHUNG] Bộ tối ưu hóa: "adamw" | "adam" | "sgd"
    betas: Tuple[float, float] = (0.9, 0.999)  # [DÙNG CHUNG] Tham số betas cho Adam/AdamW
    momentum: float = 0.9  # [DÙNG CHUNG] Động lượng Momentum khi optimizer="sgd"
    grad_clip_norm: float = 1.0  # [DÙNG CHUNG] Ngưỡng cắt gradient (Gradient Clipping) tránh bùng nổ gradient
    gradient_accumulation_steps: int = 1  # [DÙNG CHUNG] Số bước tích lũy gradient trước khi cập nhật trọng số
    empty_cache_interval: int = 2  # [DÙNG CHUNG] Số epoch giữa các lần dọn cache GPU
    use_scheduler: bool = True  # [DÙNG CHUNG] Bật/Tắt bộ điều chỉnh Learning Rate Scheduler (CosineAnnealingLR)

    # ---- 6. CHECKPOINTS, EARLY STOPPING & LOGGING CONFIGURATION ----
    tb_log_dir: str = "logs/tensorboard"  # [DÙNG CHUNG] Thư mục lưu log TensorBoard
    experiment_name: str = "deepgru_raw_nmsfree"  # [DÙNG CHUNG] Tên bài thử nghiệm
    checkpoint_dir: str = "checkpoints/experiments"  # [DÙNG CHUNG] Thư mục lưu checkpoint mô hình (.pt)
    save_ckpt_interval_epochs: int = 1  # [DÙNG CHUNG] Số epoch giữa 2 lần lưu checkpoint định kỳ
    save_all_epochs: bool = True  # [DÙNG CHUNG] True: Lưu checkpoint cho TẤT CẢ các epoch
    ckpt_keep_last: Optional[int] = None  # [DÙNG CHUNG] Số lượng checkpoint định kỳ giữ lại (None = giữ toàn bộ)
    enable_resume: bool = False  # [DÙNG CHUNG] Bật/tắt nạp lại: True = Cho phép resume | False = Train mới
    resume: str = ""  # [DÙNG CHUNG] Đường dẫn file checkpoint hoặc bí danh ('last', 'best') để resume

    # Cấu hình Giám sát Phân giải cao theo từng Step (Per-Step Metrics trên TensorBoard)
    enable_step_logging: bool = True  # [DÙNG CHUNG] Bật/tắt tính toán và ghi nhận per-step metrics
    log_step_interval: int = 1  # [DÙNG CHUNG] Số batch steps giữa các lần ghi TensorBoard

    # Cơ chế Dừng sớm (Early Stopping)
    early_stopping: bool = True  # [DÙNG CHUNG] Bật/tắt Early Stopping
    patience: int = 10  # [DÙNG CHUNG] Số epoch chờ không cải thiện trước khi dừng sớm
    monitor_metric: str = "val_f1"  # [DÙNG CHUNG] Chỉ số theo dõi: "val_f1", "val_loss", "val_acc", "val_recall"
    monitor_mode: str = "max"  # [DÙNG CHUNG] Chế độ theo dõi: "max" hoặc "min"
    min_delta: float = 0.0001  # [DÙNG CHUNG] Mức cải thiện tối thiểu để tính là tiến bộ

    # File lịch sử huấn luyện
    history_csv_path: str = "logs/training_history.csv"  # [DÙNG CHUNG] File lưu bảng lịch sử các epoch

    # ---- 7. RUNTIME & HARDWARE CONFIGURATION ----
    device: str = "cuda"  # [DÙNG CHUNG] Thiết bị tính toán ("cuda" | "cpu")
    amp: bool = True  # [DÙNG CHUNG] Bật Tự động ép kiểu chính xác hỗn hợp (FP16)
    val_interval_epochs: int = 1  # [DÙNG CHUNG] Số epoch giữa 2 lần đánh giá tập Validation
    use_tqdm: bool = True  # [DÙNG CHUNG] Bật/tắt thanh tiến trình trực quan qua tqdm

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

        # 5. Ràng buộc tham số mô hình chung
        assert self.input_dim > 0, "input_dim phải > 0"
        if isinstance(self.hidden_dim, int):
            assert self.hidden_dim > 0, "hidden_dim phải > 0"
        else:
            assert all(h > 0 for h in self.hidden_dim), "tất cả các kênh ẩn trong hidden_dim phải > 0"
        assert self.num_layers > 0, "num_layers phải > 0"
        assert self.num_classes > 0, "num_classes phải > 0"

        # 6. Ràng buộc tham số mô hình DrowsyClassifier
        assert len(self.target_size) == 2 and all(s > 0 for s in self.target_size), "target_size phải có dạng (H, W) với H, W > 0"
        assert len(self.backbone_w) >= 5 and all(w > 0 for w in self.backbone_w), "backbone_w phải có ít nhất 5 tầng kênh > 0"
        assert len(self.backbone_n) >= 4 and all(n > 0 for n in self.backbone_n), "backbone_n phải có ít nhất 4 tầng lặp > 0"
        assert self.neck_n >= 1, "neck_n phải >= 1"

        # 7. Ràng buộc tham số ghi nhận log
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

        tuple_fields = ["image_size", "video_exts", "cnn_neck_channels", "betas", "backbone_w", "backbone_n", "target_size"]
        for fld in tuple_fields:
            if fld in data and isinstance(data[fld], list):
                data[fld] = tuple(data[fld])
        if "hidden_dim" in data and isinstance(data["hidden_dim"], list):
            data["hidden_dim"] = tuple(data["hidden_dim"])

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

        tuple_fields = ["image_size", "video_exts", "cnn_neck_channels", "betas", "backbone_w", "backbone_n", "target_size"]
        for fld in tuple_fields:
            if fld in flattened and isinstance(flattened[fld], list):
                flattened[fld] = tuple(flattened[fld])
        if "hidden_dim" in flattened and isinstance(flattened["hidden_dim"], list):
            flattened["hidden_dim"] = tuple(flattened["hidden_dim"])

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
