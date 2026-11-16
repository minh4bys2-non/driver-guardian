"""Cấu hình chung cho nhánh vật lý; ghi đè bằng dataclasses.replace."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PhysicalConfig:
    # Vận hành camera/video trực tiếp qua CameraMetrics.run().
    camera_source: int | str = 0  # Chỉ số webcam hoặc đường dẫn video cho CameraMetrics.run().
    display: bool = True  # Hiển thị cửa sổ camera và bảng metric khi chạy trực tiếp.
    logdir: str | None = None  # Thư mục TensorBoard; None để tắt ghi log.
    output_interval_sec: float = 1  # Chu kỳ in metric JSON, giây; 0 để in mọi frame.
    max_frames: int | None = None  # Giới hạn số frame mỗi lần run(); None để đọc đến hết.
    use_video_time: bool = False  # True: timestamp = chỉ số frame/FPS nguồn; False: đồng hồ thực.

    # Timeline, bộ lọc thời lượng và cửa sổ thống kê.
    fps: int = 30  # FPS quy ước của detector; DataSystem dùng làm FPS lấy mẫu video.
    window_sec: float = 60  # Cửa sổ thống kê chung, giây; có thể ghi đè từng metric qua windows.
    max_gap_sec: float = 0.25  # Gap tối đa giữa hai quan sát, giây; vượt mức này ngắt sự kiện/chuỗi liên tục.
    blink_duration_ms: tuple = (100, 2000)  # (Tối thiểu, tối đa), ms; chỉ đếm nháy mắt hoàn tất trong khoảng này.
    yawn_duration_ms: tuple = (4000, 7500)  # (Tối thiểu, tối đa), ms; lọc sự kiện mở miệng được tính là ngáp.
    nod_duration_ms: tuple = (800, 3500)  # (Tối thiểu, tối đa), ms; lọc sự kiện gật đầu hoàn tất.
    windows: dict = field(default_factory=dict)  # Cửa sổ riêng, giây: blink, perclos, yawn, pom, nod, over_angle, head_motion.
    eye_options: dict = field(default_factory=dict)  # Keyword ghi đè AdaptiveHMM_FSM mắt, ví dụ min_duration_ms/adapt_interval.
    mouth_options: dict = field(default_factory=dict)  # Keyword ghi đè AdaptiveHMM_FSM miệng; ưu tiên hơn cấu hình chung.

    # HMM hai trạng thái: khởi tạo, thích nghi và mô hình dự phòng.
    init_duration_sec: float = 10  # Quy đổi thành số mẫu hợp lệ khởi tạo HMM: max(4, round(fps × giây)).
    learning_rate: float = 0.01  # Hệ số trộn tham số HMM mới với cũ; lớn hơn thích nghi nhanh hơn.
    adapt_interval: int = 300  # Số quan sát hợp lệ tích lũy trước mỗi lần cập nhật HMM.
    min_adapt_state_weight: float = 2  # Tổng posterior tối thiểu ở mỗi trạng thái để cho phép cập nhật HMM.
    min_variance: float = 1e-5  # Sàn phương sai Gaussian, tránh phân phối co về một điểm.
    em_iterations: int = 20  # Số vòng EM khi fit HMM ban đầu từ hai cụm dữ liệu.
    kmeans_iterations: int = 25  # Số vòng tối đa K-means một chiều để khởi tạo hai trạng thái.
    transition_smoothing: float = 1e-3  # Số giả đếm cộng vào tần suất trạng thái/chuyển trạng thái khi khởi tạo.
    init_percentiles: tuple = (5, 95)  # (Percentile thấp, cao) dùng kiểm tra độ phân tách mẫu khởi tạo.
    eye_baseline_percentile: float = 80  # Percentile EAR lấy baseline mắt mở; giữ cố định tới khi reset.
    other_baseline_percentile: float = 20  # Percentile baseline nghỉ cho mode mouth/pitch.
    eye_separation: float = 0.06  # Độ chênh EAR tối thiểu giữa hai percentile để fit hai cụm.
    mouth_separation: float = 0.15  # Độ chênh MAR tối thiểu; cũng dùng tạo mean mở miệng dự phòng.
    pitch_separation: float = 10  # Độ chênh pitch, độ, cho HMM mode pitch; CameraMetrics gật bằng hysteresis.
    eye_distinct_ratio: float = 0.65  # Cần percentile EAR thấp < baseline × tỷ lệ này để nhận hai cụm rõ.
    eye_open_ratio: float = 0.75  # EAR >= baseline × tỷ lệ này thì ép trạng thái mắt về mở.
    eye_closed_ratio: float = 0.2  # Mean EAR mắt đóng dự phòng = baseline mắt mở × tỷ lệ này.
    event_separation_multiplier: float = 2  # Mean sự kiện mouth/pitch dự phòng = baseline + separation × hệ số.
    fallback_sigma_divisor: float = 4  # Độ lệch chuẩn dự phòng = chênh hai mean / số chia, có áp dụng sàn.
    fallback_min_sigma: float = 0.01  # Sàn độ lệch chuẩn Gaussian khi khởi tạo dự phòng.
    fallback_transition: tuple = ((0.97, 0.03), (0.1, 0.9))  # Ma trận P(trạng thái kế | hiện tại); thứ tự 0=nghỉ, 1=sự kiện.
    resting_prior: tuple = (0.99, 0.01)  # Xác suất (nghỉ, sự kiện) dự phòng và khi ép mắt về mở.
    perclos_closed_fraction: float = 0.8  # Mức đóng trên khoảng EAR mở–đóng để tính PERCLOS; không phải ngưỡng risk.

    # Tư thế đầu: hiệu chuẩn, hysteresis gật và FFT.
    calibration_frames: int = 30  # Số mẫu góc hợp lệ để lấy trung vị tư thế đầu trung tính.
    angle_limits: tuple = (20, 25, 20)  # Ngưỡng tuyệt đối (pitch, yaw, roll), độ; vượt một trục là over_angle.
    nod_pitch_deg: float = 14  # Vào trạng thái gật khi tín hiệu pitch vượt ngưỡng này, độ.
    nod_release_deg: float = 8  # Thoát gật khi pitch <= ngưỡng này; phải nhỏ hơn nod_pitch_deg.
    nod_direction: int | None = None  # None: dùng |pitch|; ±1: chỉ xét pitch theo hướng đã chọn.
    head_motion_min_amplitude_deg: float = 0.1  # Biên độ đỉnh–đỉnh pitch tối thiểu, độ, để trả tần số FFT.

    # FaceMesh và hình học camera.
    detection_confidence: float = 0.6  # Ngưỡng confidence phát hiện mặt của MediaPipe, trong [0,1].
    tracking_confidence: float = 0.6  # Ngưỡng confidence tracking mặt của MediaPipe, trong [0,1].
    max_num_faces: int = 1  # Số mặt tối đa FaceMesh tìm; pipeline hiện chỉ sử dụng mặt đầu tiên.
    refine_landmarks: bool = True  # Bật tinh chỉnh landmark mắt/môi của FaceMesh.
    left_eye: tuple = (362, 385, 387, 263, 373, 380)  # Sáu chỉ số landmark mắt trái, đúng thứ tự công thức EAR.
    right_eye: tuple = (33, 160, 158, 133, 153, 144)  # Sáu chỉ số landmark mắt phải; EAR chung là trung bình hai mắt.
    mouth: tuple = (61, 37, 267, 291, 314, 84)  # Sáu chỉ số landmark môi, đúng thứ tự công thức MAR.
    pose_landmarks: tuple = (1, 152, 263, 33, 291, 61)  # Sáu landmark 2D tương ứng từng điểm 3D trong face_model.
    # Mô hình mặt 3D chuẩn cho PnP; thứ tự điểm khớp pose_landmarks, không phải số đo từng người.
    face_model: tuple = ((0, 0, 0), (0, 330, 65), (225, -170, 135),
                         (-225, -170, 135), (150, 150, 125), (-150, 150, 125))
    distortion_coeffs: tuple = (0, 0, 0, 0)  # Hệ số méo ống kính theo quy ước OpenCV; mặc định giả định không méo.
    camera_matrix: tuple | None = None  # Ma trận nội tại K 3×3; None dùng tiêu cự xấp xỉ chiều rộng ảnh.
    face_landmarker_model_path: str | None = r"D:\Project\DATN\tempt\driver-guardian\ai\PhysicalBranch\models\face_landmarker.task"  # Đường dẫn tới face_landmarker.task; None để tự động dò tìm.


DEFAULT_CONFIG = PhysicalConfig()

# Thang riêng của brand.normalize_metrics(): các cặp (số đo thô, điểm 0–1).
# Không có thang thì trả None; DataSystem không dùng bảng này mà dùng physical_v1.
DEFAULT_SCALES = dict.fromkeys((
    "ear", "blink_rate_per_min", "mar", "mouth_open_duration_ms", "yawn_duration_ms",
    "yawning_frequency_per_min", "pom_pct", "nod_duration_ms", "nod_event_duration_ms",
    "nodding_frequency_per_min", "pitch_mean_amplitude_deg", "head_motion_frequency_hz",
    "over_angle_pct",
))
DEFAULT_SCALES.update(
    eye_closure_duration_ms=((400, 0), (800, 1)),
    blink_duration_ms=((400, 0), (800, 1)),  # (Tối thiểu, tối đa), ms; chỉ đếm nháy mắt hoàn tất trong khoảng này.
    perclos_pct=((0, 0), (15, 1)),
    pitch_amplitude_deg=((8, 0), (20, 1)),
)
