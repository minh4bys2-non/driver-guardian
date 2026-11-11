"""DataSystem kế thừa cấu hình vật lý và bổ sung trích xuất/chuẩn hóa."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from ai.PhysicalBranch.config import PhysicalConfig

MODEL_IMAGE_SIZE = 640  # Cạnh ảnh vuông đầu vào FaceBoxesV2 & PIPNet, pixel; dùng chung cho letterbox và buffer.
LITEFAT_DEFAULT_CHECKPOINT = str(
    Path(__file__).resolve().parents[2] / "LiteFat" / "checkpoints" / "LiteFat_v6_stage1"
)

@dataclass(frozen=True)
class DataSystemConfig(PhysicalConfig):
    # Kế thừa toàn bộ ngưỡng vật lý; chỉ bổ sung tham số tạo dataset và chạy mạng.
    target_luma: float = 70  # Nâng sáng frame tối tới mức Y mục tiêu [0,255]; 0 để tắt, gamma tối thiểu 0.5.
    stride_sec: float = PhysicalConfig.window_sec  # Bước dịch cửa sổ xuất mẫu, giây; nhỏ hơn window_sec tạo mẫu chồng lấn.
    neural_sec: float = 12.0  # Cửa sổ thời gian đưa vào LiteFat (khớp 12s chuẩn của LiteFat); không vượt window_sec.
    neural_fps: float = 20.0  # FPS lấy ảnh cho mạng (30 % 10 == 0 -> step = 3).
    neural_size: int = MODEL_IMAGE_SIZE  # Cạnh ảnh buffer, pixel (640x640 cho FaceBoxes/PIPNet).
    chunk_size: int = 0  # Tương thích ngược; 0 cho LiteFat.
    visual_frame_stride: int = 8  # Lấy 1 ảnh mặt mỗi 8 frame cho MobileNetV3 (giảm 87.5% chi phí).
    face_size: Tuple[int, int] = (224, 224)  # Kích thước ảnh cắt khuôn mặt cho MobileNetV3.
    letterbox_size: Tuple[int, int] = (640, 640)  # Kích thước letterbox cho FaceBoxes & PIPNet.

    checkpoint_path: str = LITEFAT_DEFAULT_CHECKPOINT

    # Chất lượng và tổng hợp số đo; hệ số chuẩn hóa tự suy ra từ detector.
    min_observed_ratio: float = 0.8  # Coverage tối thiểu từng nhánh mắt/miệng/đầu trên window_sec, trong [0,1].
    blink_duration_percentile: float = 90  # Percentile thời lượng nháy mắt hợp lệ trong cửa sổ; 90 là P90.
    nod_duration_percentile: float = 100  # Percentile thời lượng gật hợp lệ; 100 lấy sự kiện dài nhất.
    device: str | None = None  # Thiết bị suy luận; None tự chọn CUDA nếu có, nếu không dùng CPU.

    @property
    def neural_count(self) -> int:
        return round(self.neural_sec * self.neural_fps)

DEFAULT_CONFIG = DataSystemConfig()
# Tiền xử lý ảnh, dữ liệu và checkpoint dùng chung.
LETTERBOX_COLOR = (114, 114, 114)  # Màu RGB viền padding khi giữ tỷ lệ ảnh; các kênh trong [0,255].
VIDEO_ROOT = r"E:\LSTM\drowsy_dataset\UTA-RLDD"  # Gốc chứa video; video_path trong manifest phải tương đối với thư mục này.
INPUT_JSONL = r"E:\LSTM\drowsy_dataset\UTA-RLDD/dataset.jsonl"  # Manifest đầu vào: mỗi dòng có video_path và label nguyên 0/1.
OUTPUT_JSONL = "ai/OptimalAlgorithsm/DataSystem/drowsiness_physical_30.jsonl"  # Dataset physical_v1; các entrypoint TrainingSystem dùng chung đường dẫn này.
# Tùy chọn khi chạy python -m ai.OptimalAlgorithsm.DataSystem.engine.
RUN_DEVICE = None  # Thiết bị của entrypoint; ghi đè DataSystemConfig.device khi chạy module engine.
STRICT = False  # True: dừng ngay khi lỗi xử lý; False: ghi lỗi và chuyển dòng manifest tiếp theo.
RESUME = True  # Chạy lại sau khi dừng/crash: giữ mẫu đã lưu, nối mẫu còn thiếu. False để tạo lại từ đầu.
PROGRESS_INTERVAL_SEC = 10  # Chu kỳ báo tiến độ theo thời gian thực; 0 để tắt log định kỳ.
BENCHMARK_WARMUP = 3
BENCHMARK_RUNS = 20
