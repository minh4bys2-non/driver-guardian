import json
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class AIResult:
    timestamp_sec: float  # Thời điểm đo, tính bằng giây từ đầu phiên/video.
    window_sec: float = 60.0  # Cửa sổ trượt tính tần suất và PERCLOS, đơn vị giây.

    ear: float | None = None  # Tỷ lệ hình học của mắt (EAR), phản ánh độ mở mắt.
    blink_rate_per_min: float | None = None  # Tần suất nháy mắt, lần/phút.
    eye_closure_duration_ms: float | None = None  # Thời gian nhắm mắt liên tục hiện tại, ms.
    perclos_pct: float | None = None  # Phần trăm thời gian mắt đóng ít nhất 80% (P80), 0–100.

    mar: float | None = None  # Tỷ lệ hình học của miệng (MAR), phản ánh độ mở miệng.
    mouth_open_duration_ms: float | None = None  # Thời gian mở miệng liên tục hiện tại, ms; dùng nhận diện ngáp.

    pitch_deg: float | None = None  # Góc cúi/ngửa đầu so với tư thế nhìn thẳng đã hiệu chuẩn, độ.
    pitch_deviation_duration_ms: float | None = None  # Thời gian pitch liên tục vượt dải bình thường hiện tại, ms.
    pitch_deviation_rate_per_min: float | None = None  # Tần suất pitch vượt dải bình thường, lần/phút.

    lstm_drowsiness_probability: float | None = None  # Xác suất buồn ngủ từ LSTM sau softmax, 0–1.
    warning_score: float | None = None  # Mức cần cảnh báo tổng hợp từ các thang đo và LSTM: 0 = không cần, 1 = cao nhất.

    def to_json(self) -> str:
        return json.dumps(asdict(self), allow_nan=False)