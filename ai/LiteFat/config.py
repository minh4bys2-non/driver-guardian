# =====================================================================
# File: config.py
# Dự án: Driver Guardian - Module LiteFat (IEEE/RSJ IROS 2025 Upgraded)
# Tác vụ: Định nghĩa lớp cấu hình TrainConfig và tài liệu hướng dẫn
#         chi tiết bằng Tiếng Việt cho toàn bộ pipeline huấn luyện.
# =====================================================================
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TrainConfig:
    """
    Cấu hình toàn diện cho quá trình huấn luyện và kiểm định mô hình LiteFat.

    Hướng dẫn chọn chế độ huấn luyện:
    --------------------------------
    1. Chế độ Đa phương thức hoàn chỉnh (Multimodal Mode - Khuyến nghị):
       - Đặt: `use_visual_context = True`
       - Hoạt động: Kết hợp cả chuỗi tọa độ 68 điểm mốc mặt (động học IOD) và
         chuỗi ảnh crop khuôn mặt 224x224 qua MobileNetV3-Small.
       - Chiến lược tối ưu:
         + Đặt `freeze_backbone_epochs = 5` để đóng băng MobileNetV3 trong 5 epoch đầu,
           buộc nhánh Động học & Đồ thị STGL học hội tụ trước.
         + Sau đó mở khóa với `backbone_lr = 5e-5` (nhỏ hơn 10 lần so với `lr = 5e-4` của GCN).

    2. Chế độ Chỉ dùng Landmark siêu tốc (Fast Landmark-Only Mode):
       - Đặt: `use_visual_context = False`
       - Hoạt động: Tắt hoàn toàn nhánh MobileNetV3, bỏ qua việc đọc giải mã video .mp4.
         Dataset đọc trực tiếp 100% từ cache .pt trên RAM/NVMe.
       - Tốc độ: Huấn luyện cực nhanh (~10-15 giây / epoch), không ngốn VRAM GPU.
    """

    # =================================================================
    # 1. ĐƯỜNG DẪN DỮ LIỆU & BỘ DỮ LIỆU CON
    # =================================================================
    # Thư mục gốc chứa các video đã qua tiền xử lý (chứa dataset_merged_split.csv)
    dataset_dir: str = r"E:\LSTM\data_processed"

    # Thư mục chứa các tệp cache đặc trưng .pt đã trích xuất sẵn (train.pt, val.pt, test.pt)
    cache_dir: str = r"E:\LSTM\LiteFat\cache_pt"

    # Cờ sử dụng cache .pt (True: nạp từ cache để tăng tốc; False: trích xuất on-the-fly)
    use_cache: bool = True

    # Danh sách các bộ dữ liệu con được sử dụng để huấn luyện và đánh giá.
    # Đặt `None` để sử dụng tất cả các bộ dữ liệu có trong manifest/cache.
    # Ví dụ: ["SUST-DDD", "UL-DD", "VBDDD"]
    selected_datasets: Optional[List[str]] = None

    # Tên file manifest CSV phân chia tập dữ liệu
    manifest_filename: str = "dataset_merged_split.csv"

    # Thư mục lưu kết quả huấn luyện (checkpoints, metrics json, cấu hình)
    checkpoint_dir: str = "checkpoints/LiteFat"

    # Thư mục lưu log TensorBoard
    log_dir: str = "checkpoints/LiteFat/logs"

    # =================================================================
    # 2. CHẾ ĐỘ ĐA PHƯƠNG THỨC & NGỮ CẢNH THỊ GIÁC (VISUAL CONTEXT)
    # =================================================================
    # Bật/tắt nhánh MobileNetV3-Small trích xuất ngữ cảnh thị giác từ ảnh khuôn mặt.
    # True: Huấn luyện đa phương thức (Landmark + Face Crops).
    # False: Chỉ huấn luyện đồ thị từ Landmark (không cần giải mã video, siêu nhanh).
    use_visual_context: bool = False

    # Kích thước ảnh khung hình đưa vào trích xuất điểm mốc letterbox (w, h)
    letterbox_size: Tuple[int, int] = (640, 480)

    # Kích thước ảnh khuôn mặt crop chuẩn hóa cho MobileNetV3-Small (w, h)
    face_size: Tuple[int, int] = (224, 224)

    # =================================================================
    # 3. LẤY MẪU THỜI GIAN (TEMPORAL SAMPLING)
    # =================================================================
    # Khoảng cách lấy mẫu thời gian giữa các khung hình (giây).
    # 0.05s tương ứng 20 FPS (khớp chuẩn cache .pt, ghi nhận đầy đủ chớp mắt 100-300ms).
    # 0.1s tương ứng 10 FPS.
    time_interval: float = 0.05

    # Độ dài cửa sổ trượt (giây).
    # 4.0s tương ứng T = 80 frames ở 20 FPS (đủ để quan sát hành vi ngáp và mắt nhắm kéo dài).
    # Đặt None để lấy toàn bộ độ dài video.
    window_sec: Optional[float] = 10

    # =================================================================
    # 4. TĂNG CƯỜNG DỮ LIỆU ĐA PHƯƠNG THỨC (DATA AUGMENTATION)
    # =================================================================
    # Bật/tắt tăng cường dữ liệu trên tập Train
    augment_train: bool = True

    # Xác suất lật ngang đối xứng (Horizontal Flip) đồng thời cả Landmark 68 điểm và ảnh khuôn mặt
    flip_prob: float = 0.5

    # Biên độ co giãn tỉ lệ khuôn mặt ngẫu nhiên (Scale Jitter)
    scale_jitter: float = 0.08

    # Biên độ dịch chuyển tọa độ khuôn mặt ngẫu nhiên (Shift Jitter)
    shift_jitter: float = 0.05

    # Độ lệch chuẩn của nhiễu Gaussian cộng vào tọa độ điểm mốc
    landmark_noise_std: float = 0.002

    # Biên độ biến đổi độ sáng và độ tương phản nhất quán thời gian trên ảnh khuôn mặt
    color_jitter: float = 0.2

    # Xác suất che khuất một phần vùng ảnh ngẫu nhiên (Spatial Erasing / Cutout)
    spatial_dropout: float = 0.15

    # =================================================================
    # 5. THAM SỐ KIẾN TRÚC MÔ HÌNH LITEFAT
    # =================================================================
    num_nodes: int = 68
    feature_dim: int = 64
    mobilenet_proj_hidden: int = 192
    node_embed_dim: int = 16
    residual_channels: int = 64
    dilation_channels: int = 64
    skip_channels: int = 128
    end_channels: int = 256
    num_classes: int = 2
    num_blocks: int = 3
    layers_per_block: int = 2
    kernel_size: int = 2
    dropout: float = 0.3
    pretrained_backbone: bool = True

    # Cặp chỉ số đỉnh giải phẫu dọc mắt (EAR) và dọc miệng (MAR) để tạo cạnh kề cơ sở A_prior
    # Cặp mắt trái: (37, 41), (38, 40); Cặp mắt phải: (43, 47), (44, 46)
    # Cặp miệng: (50, 58), (51, 57), (52, 56), (61, 67), (62, 66), (63, 65)
    vertical_pairs: List[Tuple[int, int]] = field(default_factory=lambda: [
        (37, 41), (38, 40), (43, 47), (44, 46),
        (50, 58), (51, 57), (52, 56), (61, 67), (62, 66), (63, 65)
    ])

    # Cặp chỉ số liên vùng (Mày <-> Mắt, Mũi <-> Mắt, Miệng <-> Cằm)
    cross_region_pairs: List[Tuple[int, int]] = field(default_factory=lambda: [
        (19, 37), (24, 44), (30, 39), (30, 42), (57, 8), (51, 33)
    ])

    # =================================================================
    # 6. THAM SỐ HUẤN LUYỆN & TỐI ƯU HÓA (OPTIMIZATION)
    # =================================================================
    epochs: int = 5
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = True
    seed: int = 42

    # Tốc độ học cơ bản cho nhánh Đồ thị STGL, Dung hợp FiLM và Đầu phân loại
    lr: float = 5e-4

    # Tốc độ học riêng biệt cho xương sống MobileNetV3-Small (thường nhỏ hơn 10 lần)
    backbone_lr: float = 5e-5

    # Hệ số phạt suy giảm trọng số (Weight Decay L2)
    weight_decay: float = 1e-3

    # Số epoch đầu tiên đóng băng hoàn toàn MobileNetV3-Small (chỉ huấn luyện STGL)
    freeze_backbone_epochs: int = 5

    # Số epoch làm nóng tốc độ học (Linear Warmup)
    warmup_epochs: int = 3

    # Giới hạn chuẩn đạo hàm để chống nổ gradient (Gradient Clipping)
    grad_clip: float = 1.0

    # Sử dụng bộ nhớ hỗn hợp Automatic Mixed Precision (AMP float16)
    use_amp: bool = True

    # =================================================================
    # 7. HÀM MẤT MÁT (LOSS CONFIGURATION)
    # =================================================================
    # Trọng số của hàm mất mát cấp đoạn video (Clip-level CrossEntropy)
    clip_loss_weight: float = 1.0

    # Trọng số của hàm mất mát từng khung hình có trọng số thời gian nhân quả (Causal Frame Loss)
    frame_loss_weight: float = 0.3

    # Hệ số làm mượt nhãn (Label Smoothing) chống overfitting
    label_smoothing: float = 0.1

    # Lưu lại checkpoint sau mỗi save_freq epoch
    save_freq: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển cấu hình thành từ điển (Dict)."""
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        """Lưu cấu hình ra file JSON phục vụ tái lập thực nghiệm."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str | Path) -> TrainConfig:
        """Nạp cấu hình từ file JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
