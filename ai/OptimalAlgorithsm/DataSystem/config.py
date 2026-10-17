"""DataSystem kế thừa cấu hình vật lý và bổ sung trích xuất/chuẩn hóa."""

from dataclasses import dataclass
from pathlib import Path

from ai.PhysicalBranch.config import PhysicalConfig
from ai.LSTM.src.config import DEFAULT_CONFIG as TRAIN_CONFIG

MODEL_IMAGE_SIZE = TRAIN_CONFIG.image_size  # Cạnh ảnh vuông đầu vào CNN, pixel; dùng chung cho letterbox và suy luận.

@dataclass(frozen=True)
class DataSystemConfig(PhysicalConfig):
    # Kế thừa toàn bộ ngưỡng vật lý; chỉ bổ sung tham số tạo dataset và chạy mạng.
    target_luma: float = 70  # Nâng sáng frame tối tới mức Y mục tiêu [0,255]; 0 để tắt, gamma tối thiểu 0.5.
    stride_sec: float = 60  # Bước dịch cửa sổ xuất mẫu, giây; nhỏ hơn window_sec tạo mẫu chồng lấn.
    neural_sec: float = TRAIN_CONFIG.window_sec  # Số giây cuối mỗi cửa sổ đưa vào CNN–ConvGRU; không vượt window_sec.
    neural_fps: float = TRAIN_CONFIG.target_fps  # FPS lấy ảnh cho mạng; FPS vật lý phải chia hết cho giá trị này.
    neural_size: int = MODEL_IMAGE_SIZE  # Cạnh ảnh buffer, pixel; Dùng chung cho buffer và DrowsyClassifier.
    chunk_size: int = TRAIN_CONFIG.chunk_size  # Số frame mỗi lượt CNN; 0 xử lý toàn bộ frame; không đổi độ dài chuỗi ConvGRU.

    checkpoint_path: str = str(Path(__file__).resolve().parents[3] / TRAIN_CONFIG.save_dir / "best.pt")

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
BASE_DIR = Path(__file__).resolve().parent.parent  # Gốc OptimalAlgorithsm, suy ra từ file để không phụ thuộc thư mục chạy.
VIDEO_ROOT = BASE_DIR / "ULDD"  # Gốc chứa video; video_path trong manifest phải tương đối với thư mục này.
INPUT_JSONL = VIDEO_ROOT / "dataset.jsonl"  # Manifest đầu vào: mỗi dòng có video_path và label nguyên 0/1.
OUTPUT_JSONL = BASE_DIR / "drowsiness_physical_60.jsonl"  # Dataset physical_v1; các entrypoint TrainingSystem dùng chung đường dẫn này.
# Tùy chọn khi chạy python -m ai.OptimalAlgorithsm.DataSystem.engine.
RUN_DEVICE = None  # Thiết bị của entrypoint; ghi đè DataSystemConfig.device khi chạy module engine.
STRICT = False  # True: dừng ngay khi lỗi xử lý; False: ghi lỗi và chuyển dòng manifest tiếp theo.
RESUME = False  # True: kiểm tra rồi nối mẫu chưa có; False: ghi đè output khi run().
PROGRESS_INTERVAL_SEC = 10  # Chu kỳ báo tiến độ theo thời gian thực; 0 để tắt log định kỳ.
BENCHMARK_WARMUP = 3
BENCHMARK_RUNS = 20
