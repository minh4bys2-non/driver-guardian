# File: config.py
"""
Hệ thống cấu hình trung tâm (Centralized Configuration) cho mô hình GCN và Dataset.
Quản lý toàn bộ siêu tham số không - thời gian, cấu trúc đồ thị, mô hình phân loại,
và bộ dữ liệu trích xuất điểm mốc InsightFace / Cache (.pt) trong một lớp duy nhất: TrainConfig.
"""

from __future__ import annotations
import dataclasses
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import torch


@dataclass
class TrainConfig:
    """
    Cấu hình tổng hợp toàn hệ thống Driver Guardian GCN.
    Tất cả các siêu tham số được quản lý tập trung và phân nhóm rõ ràng.
    """
    # =====================================================================
    # 1. CẤU HÌNH ĐỒ THỊ VÀ TRÍCH XUẤT ĐỘNG HỌC (GRAPH & KINEMATICS)
    # =====================================================================
    num_nodes: int = 68
    nose_idx: int = 30                 # Mốc tâm chóp mũi (gốc tọa độ cục bộ)
    left_eye_idx: int = 36             # Khóe mắt ngoài trái (đo IOD)
    right_eye_idx: int = 45            # Khóe mắt ngoài phải (đo IOD)
    eps: float = 1e-6                  # Tránh chia cho 0 khi tính khoảng cách IOD
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

    # =====================================================================
    # 2. CẤU HÌNH MÔ HÌNH (MODEL ARCHITECTURE)
    # =====================================================================
    model_name: str = "stgcn"                 # "stgcn" | "transformer"
    in_channels: int = 6                      # [x, y, dx, dy, d2x, d2y]
    num_classes: int = 2                      # Mặc định 2 lớp (Alert vs Drowsy)

    # Siêu tham số riêng cho Model 1: DriverSTGCN (với AttentiveDrowsinessOutputHead)
    num_domains: int = 0                      # > 0 để bật Domain Generalization (MSTGCN)
    enable_regression: bool = False           # True để bật dự đoán điểm buồn ngủ liên tục
    stgcn_hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    stgcn_strides: List[int] = field(default_factory=lambda: [1, 2, 2])
    temporal_kernel: int = 5                  # Độ dài kernel tích chập thời gian
    gat_heads: int = 4                        # Số attention heads của DenseGATConv
    dropout: float = 0.25                     # Tỷ lệ Dropout cho STGCN
    classifier_hidden_dim: Optional[int] = None  # None: mặc định curr_dim // 2

    # Siêu tham số riêng cho Model 2: DriverSTGCNTransformer
    d_model: int = 128                        # Kích thước biểu diễn của Transformer
    nhead: int = 4                            # Số Attention Heads trong Transformer
    num_transformer_layers: int = 2           # Số lớp TransformerEncoder
    max_len: int = 300                        # Chiều dài chuỗi thời gian tối đa
    transformer_dropout: float = 0.2
    stem_channels: List[int] = field(default_factory=lambda: [64, 128])
    stem_strides: List[int] = field(default_factory=lambda: [1, 2])
    dim_feedforward: Optional[int] = None     # None: mặc định d_model * 2
    init_std: float = 0.02                    # Độ lệch chuẩn khởi tạo CLS token & PosEmbedding

    # =====================================================================
    # 3. CẤU HÌNH BỘ DỮ LIỆU, INSIGHTFACE & CACHE (DATASET, INSIGHTFACE & CACHE)
    # =====================================================================
    dataset_dir: str = r"E:\LSTM\data_processed"
    manifest_filename: str = "dataset_merged_split.csv"
    time_interval: float = 0.1                # Khoảng lấy mẫu thời gian thực (0.1s = 10 FPS; 0: đọc mọi frame)
    window_sec: Optional[float] = 10.0        # Cửa sổ thời gian (giây), None: toàn bộ video
    landmark_model_path: Optional[str] = None # None: tự tìm hoặc nạp mặc định của InsightFace
    landmark_model_name: str = "buffalo_l"    # Model InsightFace 68 3D landmarks
    det_size: Tuple[int, int] = (320, 320)    # Kích thước đầu vào SCRFD (320x320 hoặc 640x640)
    class_names: Dict[int, str] = field(default_factory=lambda: {
        0: "Alert",
        1: "Drowsy"
    })
    letterbox_size: int | Tuple[int, int] = (640, 480)  # Kích thước (w, h) hoặc int S; (0, 0) / 0 để tắt

    # Cấu hình trích xuất & nạp cache landmark (.pt)
    use_cache: bool = True                    # Bật nạp chuỗi landmark từ file {cache_dir}/{split}.pt trong GCNDataset
    cache_dir: str = r"E:\LSTM\cache_landmarks"  # Thư mục chứa train.pt, val.pt, test.pt
    cache_splits: List[str] = field(default_factory=lambda: ["train", "val", "test"])
    landmark_batch_size: int = 64             # Batch size suy luận 1k3d68.onnx (Batched GPU Inference)
    extract_num_workers: int = 4              # Số luồng CPU giải mã video song song khi tạo cache
    extract_max_side: int = 0                 # Thu nhỏ frame khi tạo cache (nếu tắt letterbox)
    cache_save_every: int = 100               # Lưu tạm (atomic save) file .pt sau mỗi N video
    resume_cache: bool = True                 # Tiếp tục từ các video đã có trong file .pt

    # =====================================================================
    # 4. CẤU HÌNH DATALOADER & HUẤN LUYỆN (DATALOADER & TRAINING)
    # =====================================================================
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True
    shuffle_train: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.05
    max_grad_norm: float = 1.0
    amp: bool = True
    label_smoothing: float = 0.15             # Hệ số làm mịn nhãn Label Smoothing (0.0: tắt)
    save_dir: str = "ai/GCN/checkpoints"
    log_dir: Optional[str] = None
    resume: Optional[str] = None
    seed: int = 42

    def __post_init__(self):
        """Tự động điều chỉnh và đồng bộ các thuộc tính phụ thuộc."""
        if not (0.0 <= float(self.label_smoothing) < 1.0):
            raise ValueError(f"label_smoothing phải nằm trong khoảng [0.0, 1.0), nhận được: {self.label_smoothing}")

        # Chuẩn hóa letterbox_size về tuple (w, h) hoặc int tương thích
        if isinstance(self.letterbox_size, int):
            self.letterbox_size = (self.letterbox_size, self.letterbox_size) if self.letterbox_size > 0 else (0, 0)
        elif isinstance(self.letterbox_size, (list, tuple)) and len(self.letterbox_size) == 2:
            w, h = int(self.letterbox_size[0]), int(self.letterbox_size[1])
            self.letterbox_size = (w, h) if (w > 0 and h > 0) else (0, 0)
        else:
            self.letterbox_size = (0, 0)

        if isinstance(self.det_size, (list, tuple)) and len(self.det_size) == 2:
            self.det_size = (int(self.det_size[0]), int(self.det_size[1]))

        if self. vertical_pairs is not None:
            self.vertical_pairs = [tuple(p) for p in self.vertical_pairs]
        if self.cross_region_pairs is not None:
            self.cross_region_pairs = [tuple(p) for p in self.cross_region_pairs]

        if self.class_names is None or len(self.class_names) != self.num_classes:
            if self.num_classes == 2:
                self.class_names = {0: "Alert", 1: "Drowsy"}
            elif self.num_classes == 3:
                self.class_names = {0: "Alert", 1: "Fatigue", 2: "Drowsy"}
            else:
                self.class_names = {i: f"Class_{i}" for i in range(self.num_classes)}

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi toàn bộ cấu hình sang dictionary."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Lưu cấu hình sang tệp JSON hoặc YAML (nếu có PyYAML)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()

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

    @classmethod
    def load(cls, path: str | Path) -> TrainConfig:
        """Nạp cấu hình từ tệp JSON hoặc YAML (tương thích ngược với các phiên bản cũ/mới)."""
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

        # Chuyển class_names key sang int nếu nạp từ json (json lưu key dạng string)
        if "class_names" in content and isinstance(content["class_names"], dict):
            content["class_names"] = {int(k): v for k, v in content["class_names"].items()}

        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered_content = {k: v for k, v in content.items() if k in valid_fields}
        return cls(**filtered_content)

    # -----------------------------------------------------------------
    # Helper methods trích xuất kwargs để truyền vào các components
    # -----------------------------------------------------------------
    def get_stgcn_kwargs(self) -> Dict[str, Any]:
        """Trích xuất tham số thuần túy cho mô hình DriverSTGCN."""
        return {
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "num_nodes": self.num_nodes,
            "num_domains": self.num_domains,
            "enable_regression": self.enable_regression,
            "hidden_dims": list(self.stgcn_hidden_dims),
            "strides": list(self.stgcn_strides),
            "dropout": self.dropout,
            "temporal_kernel": self.temporal_kernel,
            "gat_heads": self.gat_heads,
            "classifier_hidden_dim": self.classifier_hidden_dim,
            "nose_idx": self.nose_idx,
            "left_eye_idx": self.left_eye_idx,
            "right_eye_idx": self.right_eye_idx,
            "eps": self.eps,
            "adaptive_init_scale": self.adaptive_init_scale,
            "vertical_pairs": list(self.vertical_pairs) if self.vertical_pairs is not None else None,
            "cross_region_pairs": list(self.cross_region_pairs) if self.cross_region_pairs is not None else None,
        }

    def get_transformer_kwargs(self) -> Dict[str, Any]:
        """Trích xuất tham số thuần túy cho mô hình DriverSTGCNTransformer."""
        return {
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "num_nodes": self.num_nodes,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_transformer_layers": self.num_transformer_layers,
            "max_len": self.max_len,
            "dropout": self.transformer_dropout,
            "stem_channels": list(self.stem_channels),
            "stem_strides": list(self.stem_strides),
            "dim_feedforward": self.dim_feedforward,
            "init_std": self.init_std,
            "nose_idx": self.nose_idx,
            "left_eye_idx": self.left_eye_idx,
            "right_eye_idx": self.right_eye_idx,
            "eps": self.eps,
            "adaptive_init_scale": self.adaptive_init_scale,
            "vertical_pairs": list(self.vertical_pairs) if self.vertical_pairs is not None else None,
            "cross_region_pairs": list(self.cross_region_pairs) if self.cross_region_pairs is not None else None,
        }

    def get_extractor_kwargs(self) -> Dict[str, Any]:
        """Trích xuất tham số thuần túy cho FaceLandmarkExtractor."""
        return {
            "model_path": self.landmark_model_path,
            "time_interval": self.time_interval,
            "device": self.device,
            "det_size": self.det_size,
            "model_name": self.landmark_model_name,
            "landmark_batch_size": self.landmark_batch_size,
            "letterbox_size": self.letterbox_size,
        }

    def get_dataset_kwargs(self, split: str = "train") -> Dict[str, Any]:
        """Trích xuất tham số thuần túy cho GCNDataset."""
        return {
            "dataset_dir": self.dataset_dir,
            "split": split,
            "time_interval": self.time_interval,
            "window_sec": self.window_sec,
            "manifest_filename": self.manifest_filename,
            "landmark_model_path": self.landmark_model_path,
            "letterbox_size": self.letterbox_size,
            "device": self.device,
            "det_size": self.det_size,
            "model_name": self.landmark_model_name,
            "use_cache": self.use_cache,
            "cache_dir": self.cache_dir,
            "landmark_batch_size": self.landmark_batch_size,
        }

    def get_dataloader_kwargs(self) -> Dict[str, Any]:
        """Trích xuất tham số thuần túy cho build_gcn_dataloaders."""
        return {
            "dataset_dir": self.dataset_dir,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "time_interval": self.time_interval,
            "window_sec": self.window_sec,
            "manifest_filename": self.manifest_filename,
            "pin_memory": self.pin_memory,
            "shuffle_train": self.shuffle_train,
            "landmark_model_path": self.landmark_model_path,
            "letterbox_size": self.letterbox_size,
            "device": self.device,
            "det_size": self.det_size,
            "model_name": self.landmark_model_name,
            "use_cache": self.use_cache,
            "cache_dir": self.cache_dir,
            "landmark_batch_size": self.landmark_batch_size,
        }


# =====================================================================
# HÀM TIỆN ÍCH CẤP MODULE
# =====================================================================
DEFAULT_CONFIG = TrainConfig()


def get_default_config(num_classes: int = 2) -> TrainConfig:
    """Tạo đối tượng TrainConfig với số nhãn phân loại tùy chọn (2 hoặc 3 lớp)."""
    return TrainConfig(num_classes=num_classes)


def save_config(config: TrainConfig, path: str | Path) -> None:
    """Lưu đối tượng TrainConfig ra tệp JSON hoặc YAML."""
    config.save(path)


def load_config(path: str | Path) -> TrainConfig:
    """Nạp tệp cấu hình từ JSON hoặc YAML và trả về TrainConfig."""
    return TrainConfig.load(path)


if __name__ == "__main__":
    if hasattr(torch.sys.stdout, "reconfigure"):
        torch.sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = get_default_config(num_classes=2)
    print("=== TrainConfig (Default 2 Classes) ===")
    print(f"Num classes : {cfg.num_classes}")
    print(f"ST-GCN dims : {cfg.stgcn_hidden_dims}")
    print(f"Transformer : d_model={cfg.d_model}, heads={cfg.nhead}")
    print(f"Dataset dir : {cfg.dataset_dir}")
    print(f"Cache dir   : {cfg.cache_dir} (use_cache={cfg.use_cache})")
    print(f"Classes     : {cfg.class_names}")

    stgcn_kw = cfg.get_stgcn_kwargs()
    print(f"\nST-GCN Kwargs Keys ({len(stgcn_kw)} params): {list(stgcn_kw.keys())}")
