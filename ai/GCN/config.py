# File: config.py
"""
Hệ thống cấu hình trung tâm (Centralized Configuration) cho mô hình GCN và Dataset.
Quản lý toàn bộ siêu tham số không - thời gian, cấu trúc đồ thị, mô hình phân loại,
và bộ dữ liệu trích xuất điểm mốc MediaPipe / Dlib.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import torch


# =====================================================================
# 1. CẤU HÌNH ĐỒ THỊ VÀ TRÍCH XUẤT ĐỘNG HỌC (GRAPH & KINEMATICS)
# =====================================================================
@dataclass
class GraphConfig:
    """Cấu hình đồ thị khuôn mặt và các chỉ số điểm mốc chuẩn hóa."""
    num_nodes: int = 68
    nose_idx: int = 30           # Mốc tâm chóp mũi (gốc tọa độ cục bộ)
    left_eye_idx: int = 36       # Khóe mắt ngoài trái (đo IOD)
    right_eye_idx: int = 45      # Khóe mắt ngoài phải (đo IOD)
    eps: float = 1e-6            # Tránh chia cho 0 khi tính khoảng cách IOD
    adaptive_init_scale: float = 1e-5  # Hệ số khởi tạo ma trận kề thích ứng

    # Cặp chỉ số đo độ mở mắt (EAR) & miệng (MAR)
    vertical_pairs: List[Tuple[int, int]] = field(default_factory=lambda: [
        (37, 41), (38, 40), (43, 47), (44, 46),  # Mí mắt trên - dưới
        (50, 58), (51, 57), (52, 56), (61, 67), (62, 66), (63, 65)  # Môi trên - dưới
    ])

    # Cặp chỉ số tiên nghiệm liên vùng (Mắt <-> Miệng, Mũi <-> Cằm)
    cross_region_pairs: List[Tuple[int, int]] = field(default_factory=lambda: [
        (37, 48), (41, 48), (38, 62), (40, 66),  # Mắt trái <-> Miệng
        (44, 54), (46, 54), (43, 62), (47, 66),  # Mắt phải <-> Miệng
        (30, 8), (30, 36), (30, 45), (27, 8)     # Chóp mũi <-> Cằm & Đuôi mắt
    ])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# 2. CẤU HÌNH MÔ HÌNH DRIVERSTGCN (MODEL 1 - LIGHTWEIGHT)
# =====================================================================
@dataclass
class DriverSTGCNConfig:
    """Cấu hình mô hình ST-GCN tiêu chuẩn tối ưu cho Edge AI."""
    in_channels: int = 6                      # [x, y, dx, dy, d2x, d2y]
    num_classes: int = 2                      # Mặc định 2 lớp (Alert vs Drowsy)
    num_nodes: int = 68                       # Số điểm mốc
    hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    strides: List[int] = field(default_factory=lambda: [1, 2, 2])
    temporal_kernel: int = 5                  # Độ dài kernel tích chập thời gian
    gat_heads: int = 4                        # Số attention heads của DenseGATConv
    dropout: float = 0.25                     # Tỷ lệ Dropout
    classifier_hidden_dim: Optional[int] = None  # None: mặc định curr_dim // 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# 3. CẤU HÌNH MÔ HÌNH DRIVERSTGCNTRANSFORMER (MODEL 2 - HYBRID)
# =====================================================================
@dataclass
class DriverSTGCNTransformerConfig:
    """Cấu hình mô hình lai ST-GCN + Transformer."""
    in_channels: int = 6
    num_classes: int = 2
    num_nodes: int = 68
    d_model: int = 128                        # Kích thước biểu diễn của Transformer
    nhead: int = 4                            # Số Attention Heads trong Transformer
    num_transformer_layers: int = 2           # Số lớp TransformerEncoder
    max_len: int = 300                        # Chiều dài chuỗi thời gian tối đa
    dropout: float = 0.2
    stem_channels: List[int] = field(default_factory=lambda: [64, 128])
    stem_strides: List[int] = field(default_factory=lambda: [1, 2])
    dim_feedforward: Optional[int] = None     # None: mặc định d_model * 2
    init_std: float = 0.02                    # Độ lệch chuẩn khởi tạo CLS token & PosEmbedding

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# 4. CẤU HÌNH BỘ DỮ LIỆU (DATASET CONFIG)
# =====================================================================
@dataclass
class DatasetConfig:
    """Cấu hình bộ dữ liệu và tiền xử lý khung hình."""
    dataset_dir: str = r"E:\LSTM\data_processed"
    manifest_filename: str = "dataset_merged_split.csv"
    time_interval: float = 0.2                # 0.2s = 5 FPS lấy mẫu thời gian thực
    window_sec: Optional[float] = 10.0        # Cửa sổ thời gian (giây), None: toàn bộ video
    landmark_model_path: Optional[str] = None # None: tự tìm face_landmarker.task trong project
    num_classes: int = 2
    class_names: Dict[int, str] = field(default_factory=lambda: {
        0: "Alert",
        1: "Drowsy"
    })
    letterbox_size: int = 480

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# 5. CẤU HÌNH DATALOADER & HUẤN LUYỆN (DATALOADER & TRAINING)
# =====================================================================
@dataclass
class DataLoaderConfig:
    """Cấu hình DataLoader."""
    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = True
    shuffle_train: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingConfig:
    """Cấu hình huấn luyện mô hình."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# 6. CẤU HÌNH TỔNG HỢP TOÀN HỆ THỐNG (MASTER CONFIG)
# =====================================================================
@dataclass
class MasterConfig:
    """Tổng hợp cấu hình toàn hệ thống Driver Guardian GCN."""
    graph: GraphConfig = field(default_factory=GraphConfig)
    stgcn: DriverSTGCNConfig = field(default_factory=DriverSTGCNConfig)
    transformer: DriverSTGCNTransformerConfig = field(default_factory=DriverSTGCNTransformerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Lưu cấu hình sang tệp JSON hoặc YAML (nếu có PyYAML)."""
        save_config(self, path)


# =====================================================================
# 7. CÁC HÀM TIỆN ÍCH KHỞI TẠO, LƯU VÀ NẠP CẤU HÌNH
# =====================================================================
def get_default_config(num_classes: int = 2) -> MasterConfig:
    """
    Tạo đối tượng MasterConfig với số nhãn phân loại tùy chọn (2 hoặc 3 lớp).
    - num_classes = 2: Alert (0), Drowsy (1)
    - num_classes = 3: Alert (0), Fatigue/Yawn (1), Drowsy (2)
    """
    if num_classes == 3:
        class_names = {0: "Alert", 1: "Fatigue", 2: "Drowsy"}
    elif num_classes == 2:
        class_names = {0: "Alert", 1: "Drowsy"}
    else:
        class_names = {i: f"Class_{i}" for i in range(num_classes)}

    stgcn_cfg = DriverSTGCNConfig(num_classes=num_classes)
    trans_cfg = DriverSTGCNTransformerConfig(num_classes=num_classes)
    dataset_cfg = DatasetConfig(num_classes=num_classes, class_names=class_names)

    return MasterConfig(
        graph=GraphConfig(),
        stgcn=stgcn_cfg,
        transformer=trans_cfg,
        dataset=dataset_cfg,
        dataloader=DataLoaderConfig(),
        training=TrainingConfig(),
    )


def save_config(config: MasterConfig | Any, path: str | Path) -> None:
    """Lưu đối tượng cấu hình ra tệp JSON hoặc YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = config.to_dict() if hasattr(config, "to_dict") else asdict(config)

    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
            with open(p, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
            return
        except ImportError:
            pass  # Fallback to json if PyYAML is not installed

    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_config(path: str | Path) -> MasterConfig:
    """Nạp tệp cấu hình từ JSON hoặc YAML và trả về MasterConfig."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Không tìm thấy file config tại: {p}")

    content = None
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
            with open(p, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
        except ImportError:
            pass

    if content is None:
        with open(p, "r", encoding="utf-8") as f:
            content = json.load(f)

    # Khôi phục các dataclass con
    return MasterConfig(
        graph=GraphConfig(**content.get("graph", {})),
        stgcn=DriverSTGCNConfig(**content.get("stgcn", {})),
        transformer=DriverSTGCNTransformerConfig(**content.get("transformer", {})),
        dataset=DatasetConfig(**content.get("dataset", {})),
        dataloader=DataLoaderConfig(**content.get("dataloader", {})),
        training=TrainingConfig(**content.get("training", {})),
    )


if __name__ == "__main__":
    cfg = get_default_config(num_classes=2)
    print("=== MasterConfig (Default 2 Classes) ===")
    print(f"Num classes : {cfg.stgcn.num_classes}")
    print(f"ST-GCN dims : {cfg.stgcn.hidden_dims}")
    print(f"Transformer : d_model={cfg.transformer.d_model}, heads={cfg.transformer.nhead}")
    print(f"Dataset dir : {cfg.dataset.dataset_dir}")
    print(f"Classes     : {cfg.dataset.class_names}")
