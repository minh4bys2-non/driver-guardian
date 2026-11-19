# =====================================================================
# File: config.py
# Dự án: Driver Guardian - Module LiteFat_v1 (KSS 1-9, Sliding 60s/30s)
# Tác vụ: Tập trung toàn bộ cấu hình (TrainConfig) cho dữ liệu, tiền xử lý,
#         lấy mẫu 10 FPS, kiến trúc LiteFatV1, tối ưu hóa và LiteFatLoss.
# =====================================================================
from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class TrainConfig:
    """
    Cấu hình tập trung cho toàn bộ pipeline huấn luyện, kiểm định và suy luận LiteFatV1 (KSS 1-9):
      - Dữ liệu dạng E:\\LSTM\\data_processed_v1 đọc từ manifest .csv hoặc .json.
      - Lấy mẫu chuẩn 10 FPS (time_interval = 0.1s), giữ toàn bộ chiều dài mẫu (>60s)
        để LiteFatV1 thực thi Sliding Window (60s window / 30s stride) tích lũy đặc trưng.
      - Hỗ trợ đầy đủ cả Landmark-Only và Multimodal (MobileNetV3 + FiLM).
    """

    # =================================================================
    # 1. ĐƯỜNG DẪN DỮ LIỆU, MANIFEST & LƯU TRỮ
    # =================================================================
    dataset_dir: str = r"E:\LSTM\data_processed_v1"
    manifest_filename: str = "dataset_merged_split.csv"
    cache_dir: Optional[str] = r"E:\LSTM\LiteFat_v1\cache_pt"
    use_cache: bool = False
    selected_datasets: Optional[List[str]] = None

    checkpoint_dir: str = "./checkpoints/LiteFat_v1"
    log_dir: str = "./checkpoints/LiteFat_v1/logs"

    # =================================================================
    # 2. CHẾ ĐỘ ĐA PHƯƠNG THỨC & TRÍCH XUẤT PIPNET
    # =================================================================
    use_visual_context: bool = False
    visual_frame_stride: int = 5
    letterbox_size: Tuple[int, int] = (640, 640)
    face_size: Tuple[int, int] = (224, 224)

    pipnet_backbone: str = "resnet18"
    conf_threshold: float = 0.5
    nms_threshold: float = 0.3
    bbox_margin: float = 0.15
    box_smooth_alpha: float = 0.7

    # =================================================================
    # 3. LẤY MẪU THỜI GIAN & SLIDING WINDOW (10 FPS, 60s / 30s)
    # =================================================================
    fps: int = 10
    time_interval: float = 0.1
    # None: lấy toàn bộ video dài >60s để model tự chạy sliding window 60s/30s.
    # Có thể đặt giá trị (vd: 120.0) nếu muốn cắt cửa sổ cố định khi train.
    dataset_window_sec: Optional[float] = None
    eval_window_mode: str = "full"

    # Tham số cửa sổ trượt của kiến trúc LiteFatV1 và LiteFatLoss
    window_sec: float = 60.0
    stride_sec: float = 45.0

    # =================================================================
    # 4. TĂNG CƯỜNG DỮ LIỆU ĐA PHƯƠNG THỨC (DATA AUGMENTATION)
    # =================================================================
    augment_train: bool = True
    flip_prob: float = 0.5
    scale_jitter: float = 0.08
    shift_jitter: float = 0.05
    roll_deg: float = 12.0
    temporal_dropout: float = 0.05
    landmark_noise_std: float = 0.0003
    color_jitter: float = 0.2
    spatial_dropout: float = 0.15

    # =================================================================
    # 5. THAM SỐ KIẾN TRÚC MÔ HÌNH LITEFAT_V1 (KSS 1-9)
    # =================================================================
    num_nodes: int = 68
    feature_dim: int = 64
    mobilenet_proj_hidden: int = 192
    node_embed_dim: int = 16
    residual_channels: int = 64
    dilation_channels: int = 64
    skip_channels: int = 128
    end_channels: int = 256
    num_classes: int = 9
    kss_min: float = 1.0
    kss_max: float = 9.0
    num_blocks: int = 3
    layers_per_block: int = 4
    kernel_size: int = 3
    dropout: float = 0.3
    pretrained_backbone: bool = True
    static_shape_ratio: float = 0.55

    vertical_pairs: List[Tuple[int, int]] = field(default_factory=lambda: [
        (37, 41), (38, 40), (43, 47), (44, 46),
        (50, 58), (51, 57), (52, 56), (61, 67), (62, 66), (63, 65),
    ])
    cross_region_pairs: List[Tuple[int, int]] = field(default_factory=lambda: [
        (19, 37), (24, 44), (30, 39), (30, 42), (57, 8), (51, 33),
    ])

    # =================================================================
    # 6. THAM SỐ HUẤN LUYỆN, TỐI ƯU HÓA & CHECKPOINT
    # =================================================================
    epochs: int = 40
    batch_size: int = 2
    grad_accum_steps: int = 8
    num_workers: int = 0
    pin_memory: bool = True
    seed: int = 42

    subject_balanced_sampling: bool = True
    use_manifest_sample_weights: bool = True
    drop_single_class_subjects: bool = False

    ema_decay: float = 0.990
    use_ema_eval: bool = True
    early_stop_patience: int = 10

    lr: float = 3e-4
    backbone_lr: float = 3e-5
    weight_decay: float = 2e-3
    freeze_backbone_epochs: int = 5
    unfreeze_last_n_blocks: int = 3
    warmup_epochs: int = 2
    pretrained_lr_scale: float = 0.25
    grad_clip: float = 1.0
    use_amp: bool = True

    save_freq: int = 1
    resume: Optional[str] = None
    pretrained_from: Optional[str] = None

    # =================================================================
    # 7. THAM SỐ HÀM MẤT MÁT HYBRID KSS LOSS (LITEFATLOSS)
    # =================================================================
    clip_reg_weight: float = 1.0
    clip_cls_weight: float = 0.5
    frame_loss_weight: float = 0.3
    window_loss_weight: float = 0.25
    window_smooth_weight: float = 0.05

    huber_beta: float = 0.5
    ordinal_sigma: float = 0.85
    label_smoothing: float = 0.05
    alert_kss_threshold: float = 4.0
    mid_kss_threshold: float = 6.0
    topk_mid_ratio: float = 0.70
    topk_drowsy_ratio: float = 0.45
    smooth_margin: float = 1.0

    # =================================================================
    # 8. THAM SỐ DOMAIN-ADVERSARIAL TRAINING (REGIME-CONDITIONED DANN)
    # =================================================================
    use_dann: bool = True
    dann_weight: float = 0.02
    dann_max_alpha: float = 0.15

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển cấu hình thành từ điển thuần JSON-serializable."""
        return asdict(self)

    def save_json(self, path: Union[str, Path]) -> None:
        """Lưu cấu hình ra tệp JSON để tái lập thực nghiệm."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> TrainConfig:
        """Nạp cấu hình từ tệp JSON, tự động chuẩn hóa kiểu tuple cho kích thước và cặp đỉnh."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered: Dict[str, Any] = {k: v for k, v in data.items() if k in valid_fields}

        for size_key in ("letterbox_size", "face_size"):
            if size_key in filtered and isinstance(filtered[size_key], list):
                filtered[size_key] = tuple(int(x) for x in filtered[size_key])

        for pairs_key in ("vertical_pairs", "cross_region_pairs"):
            if pairs_key in filtered and isinstance(filtered[pairs_key], list):
                filtered[pairs_key] = [(int(p[0]), int(p[1])) for p in filtered[pairs_key]]

        return cls(**filtered)
