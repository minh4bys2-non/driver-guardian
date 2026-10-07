# PhysicalBranch

Triển khai các metric trong `Metric_revised.md` (tài liệu nằm tại
`/home/tranmanhduy/Workspace/ptithcm/Documents/Metric_revised.md`).
Nhánh này đo tín hiệu mắt, miệng và đầu; không tự kết luận buồn ngủ hoặc chạy LSTM.

## 1. Cấu trúc

| File | Trách nhiệm |
| --- | --- |
| [camera_metrics.py](camera_metrics.py) | `FaceTracker`, `CameraMetrics`, HUD, camera/video và log TensorBoard |
| [adaptive_hmm_fsm.py](adaptive_hmm_fsm.py) | `AdaptiveHMM` phân loại hai trạng thái; `AdaptiveHMM_FSM` ghép HMM với thống kê thời gian |
| [temporal_metrics.py](temporal_metrics.py) | `StateMachine` đếm sự kiện, `WindowRatio` tính tỷ lệ thời gian, `HeadMotionWindow` tính trung bình và FFT pitch |
| [head_pose_estimation.py](head_pose_estimation.py) | Ước lượng pitch/yaw/roll bằng SQPnP + tinh chỉnh LM |
| [interface.py](interface.py) | `AIResult`, chuyển dictionary sang DTO và xuất JSON |
| [brand.py](brand.py) | Chuẩn hóa chỉ số theo các thang điểm có thể ghi đè |
| [tests/test_metrics.py](tests/test_metrics.py) | Kiểm thử metric, cửa sổ, mất tín hiệu, FFT và tích hợp |

Đọc từ `CameraMetrics.process_landmarks()`. FSM và tỷ lệ cửa sổ được dùng chung;
không còn demo `PitchFSM.py` hay một bộ đếm gật đầu độc lập.

```text
Frame BGR → FaceTracker → landmark pixel
                          ├─ EAR → HMM mắt → FSM nháy mắt
                          │      └─ ngưỡng P80 → tỷ lệ PERCLOS
                          ├─ MAR → HMM miệng → FSM ngáp + tỷ lệ POM
                          └─ PnP → góc so với baseline
                                   ├─ biên độ pitch + FSM → sự kiện gật
                                   ├─ giới hạn góc → tỷ lệ lệch góc
                                   └─ chuỗi pitch có dấu → trung bình + FFT
                                     ↓
                          Dictionary chỉ số + windows_sec
                                     ↓
                       JSON / callback / HUD / TensorBoard
```

## 2. Điều chỉnh cửa sổ thời gian

`window_sec` đặt giá trị chung, mặc định **60 giây**. `windows` ghi đè riêng
cho từng loại thống kê, không buộc tất cả metric dùng cùng một chiều dài.

| Khóa cửa sổ | Các chỉ số sử dụng |
| --- | --- |
| `blink` | Số lần và tần suất nháy mắt |
| `perclos` | PERCLOS và `eye_observed_sec` |
| `yawn` | Số lần và tần suất ngáp |
| `pom` | POM và `mouth_observed_sec` |
| `nod` | Số lần và tần suất gật đầu |
| `over_angle` | Tỷ lệ lệch góc và `head_observed_sec` |
| `head_motion` | Pitch trung bình, biên độ trung bình và tần số FFT |

```python
from ai.PhysicalBranch.camera_metrics import CameraMetrics

monitor = CameraMetrics(
    window_sec=30,
    windows={"perclos": 60, "yawn": 90, "nod": 30, "head_motion": 20},
)

# Đổi trong khi đang chạy; các cửa sổ không được chỉ định giữ nguyên.
monitor.set_windows(perclos=45, head_motion=80)

# Đổi tất cả thành 60 giây, riêng FFT dùng 80 giây.
monitor.set_windows(window_sec=60, head_motion=80)
```

Giá trị phải hữu hạn và lớn hơn 0. Tên cửa sổ sai hoặc giá trị không hợp lệ
bị từ chối trước khi thay đổi trạng thái.

Khi chiều dài thay đổi, lịch sử của metric đó được xóa, gồm sự kiện đang đo
nếu là FSM. HMM và baseline đầu vẫn giữ; không khôi phục dữ liệu cũ đã loại
khi tăng cửa sổ. Đặt lại đúng giá trị hiện tại không xóa lịch sử.
`reset()` mới là thao tác xóa toàn bộ mô hình, hiệu chuẩn và thống kê.

`windows_sec` trong mỗi kết quả là các chiều dài thực tế đang dùng.
`window_sec` chỉ là giá trị cấu hình chung gần nhất, không thay thế các giá trị
đã ghi đè. `eye_options`/`mouth_options` vẫn nhận `window_size_sec` và
`ratio_window_sec`; `windows` có ưu tiên cao hơn khi khởi tạo.

## 3. Định nghĩa và các hiệu chỉnh

### Mắt và miệng

EAR/MAR dùng chung tỷ lệ hình học của 6 landmark; EAR lấy trung bình hai mắt.
HMM gán mắt `0 = mở`, `1 = đóng`; miệng `0 = nghỉ`, `1 = mở`.
Khởi tạo cần `max(4, round(fps × init_duration_sec))` mẫu hợp lệ. Mặc định là
150 mẫu ở 30 FPS; nếu mẫu không tách rõ hai trạng thái, dùng phân phối dự phòng.

FSM chỉ xác nhận **0 → 1 → 0**, với thời lượng nằm trong dải cấu hình.
Nếu bắt đầu quan sát khi tín hiệu đã ở trạng thái 1, hoặc mất tín hiệu giữa
chu kỳ, lần trở về 0 không được tính là một sự kiện hoàn chỉnh.
Một lần đóng mắt, mở miệng hoặc cúi đầu kéo dài chỉ có thể tạo tối đa một sự kiện.

- Tần suất luôn dùng **lần/phút**: `count × 60 / T`, với `T` là cửa sổ của bộ đếm. Ví dụ 2 lần trong cửa sổ 20 giây cho 6 lần/phút.
- PERCLOS dùng `EAR <= A0 - 0.8 × (A0 - A100)`, gồm đúng biên P80; các mốc lấy từ khởi tạo, độc lập với trạng thái HMM.
- POM dùng trạng thái mở miệng của HMM; không đồng nghĩa toàn bộ thời gian đó là ngáp.
- Các tỷ lệ dùng thời lượng theo timestamp, không đếm frame. Mẫu số chỉ gồm khoảng quan sát hợp lệ trong cửa sổ; chưa có khoảng hợp lệ thì trả `None`.

### Đầu và FFT

Góc đầu xuất ra là góc tương đối với trung vị hiệu chuẩn ban đầu, quy về
`[-180, 180)`. `pitch_amplitude_deg = abs(pitch_deg)`.
Mặc định FSM gật dùng biên độ này ở cả hai phía baseline; đặt `nod_direction=1`
hoặc `-1` nếu thiết lập camera chỉ muốn theo dõi một hướng.

Kích hoạt khi biên độ vượt `nod_pitch_deg`, duy trì đến khi về bằng hoặc dưới
`nod_release_deg`, rồi kiểm tra thời lượng để xác nhận sự kiện.

`nodding_frequency_per_min` là số sự kiện/phút (**F_nod**).
`head_motion_frequency_hz` là tần số trội của chuỗi pitch có dấu (**H_F**).
FFT không dùng `abs(pitch)` vì thao tác đó có thể làm thay đổi tần số tín hiệu.

`HeadMotionWindow` nội suy chuỗi theo timestamp lên lưới đều, trừ trung bình,
áp cửa sổ Hann, rồi lấy đỉnh phổ không gồm thành phần DC. Chỉ trả tần số khi:

- Đã có đủ chiều dài `head_motion` liên tục và ít nhất 4 mẫu.
- Không có mẫu mất tín hiệu hoặc khoảng cách mẫu vượt `max_gap_sec` trong đoạn đang dùng.
- Độ dao động đỉnh–đỉnh lớn hơn `head_motion_min_amplitude_deg`, mặc định 0.1° để bỏ tín hiệu gần như đứng yên.

`pitch_mean_deg` và `pitch_mean_amplitude_deg` là trung bình theo thời gian
của pitch có dấu và trị tuyệt đối; có thể trả từ cửa sổ chưa đầy.
`head_motion_observed_sec` cho biết thời lượng liên tục hiện có.
`head_motion_resolution_hz` là khoảng cách các bin FFT, xấp xỉ `1 / T`.
Cửa sổ 20 giây có độ phân giải khoảng 0.05 Hz; muốn khảo sát dao động chậm hơn,
cần tăng cửa sổ, ví dụ 80 giây. Nội suy hỗ trợ FPS dao động, không bù được
chuyển động vượt khả năng lấy mẫu của camera. Code không tự gán Type I/II
hay kết luận buồn ngủ từ các mốc 0.05/0.2 Hz trong tài liệu.

### Mặc định vận hành

| Tham số | Giá trị |
| --- | --- |
| Thời lượng nháy mắt hợp lệ | 100–2000 ms |
| Thời lượng ngáp hợp lệ | 3500–7500 ms |
| Thời lượng gật hợp lệ | 800–3500 ms |
| Ngưỡng gật / nhả | 14° / 8° |
| Giới hạn pitch / yaw / roll | 20° / 25° / 20° |
| Hiệu chuẩn đầu | 30 mẫu góc hợp lệ |
| Khoảng cách mẫu tối đa | 0.25 giây |
| Học HMM trực tuyến | Xét cập nhật mỗi 300 mẫu, learning rate 0.01 |

Các dải thời lượng là cấu hình vận hành kế thừa, không phải hằng số sinh lý
được tài liệu mới quy định. Có thể đổi qua `eye_options`, `mouth_options`,
`nod_duration_ms`; các ngưỡng gật và giới hạn góc cũng là tham số constructor.
Ngưỡng phân nhóm thời lượng 400/800 ms khác với giới hạn xác nhận sự kiện.

## 4. Dữ liệu đầu ra

| Nhóm | Trường chính |
| --- | --- |
| Mắt | `ear`, `eye_state`, `blink_count`, `blink_rate_per_min`, `blink_detected`, `blink_duration_ms`, `eye_closure_duration_ms`, `last_eye_closure_duration_ms`, `perclos_pct`, `p80_ear_threshold`, `eye_observed_sec` |
| Miệng | `mar`, `mouth_state`, `yawn_count`, `yawning_frequency_per_min`, `yawn_detected`, `yawn_duration_ms`, `mouth_open_duration_ms`, `last_mouth_open_duration_ms`, `pom_pct`, `mouth_observed_sec` |
| Đầu | `pitch_deg`, `yaw_deg`, `roll_deg`, `pitch_amplitude_deg`, `nodding`, `nod_count`, `nod_detected`, `nodding_frequency_per_min`, `nod_duration_ms`, `last_nod_duration_ms`, `nod_event_duration_ms`, `over_angle`, `over_angle_pct`, `head_observed_sec` |
| Chuỗi đầu | `pitch_mean_deg`, `pitch_mean_amplitude_deg`, `head_motion_frequency_hz`, `head_motion_resolution_hz`, `head_motion_observed_sec` |
| Trạng thái | `timestamp_sec`, `window_sec`, `windows_sec`, `face_detected`, `eye_ready`, `mouth_ready`, `head_ready`, `head_calibrated`, `pose_valid`, `pose_error` |

`blink_duration_ms`, `yawn_duration_ms`, `nod_event_duration_ms` giữ thời lượng
sự kiện **hợp lệ gần nhất**, hoặc `None` nếu chưa có. Chúng không phải trung bình
cửa sổ và có thể còn giá trị sau khi sự kiện đã hết hạn khỏi bộ đếm.
Các trường `last_*_duration_ms` giữ lần kết thúc gần nhất, kể cả không hợp lệ.
`eye_closure_duration_ms`, `mouth_open_duration_ms`, `nod_duration_ms` đo pha
hiện đang diễn ra; khi mất tín hiệu chúng trả `None`.

Số đếm chỉ gồm sự kiện kết thúc trong `(timestamp - T, timestamp]`.
Tần suất dùng đủ `T` làm mẫu số kể cả lúc mới chạy hoặc có khoảng mất tín hiệu.
Mất tín hiệu ngắt sự kiện đang đo nhưng không xóa sự kiện hợp lệ còn trong cửa sổ;
FFT thì xóa đoạn liên tục để tránh nội suy qua khoảng mất dữ liệu.

`eye_ready`/`mouth_ready` chỉ phản ánh HMM đã khởi tạo, không bảo đảm frame hiện
tại có tín hiệu. `head_ready` cần góc hợp lệ đã hiệu chuẩn tại frame hiện tại.
Đổi kích thước ảnh chỉ reset phần đầu, gồm FFT; mắt/miệng vẫn giữ mô hình.

## 5. DTO và chuẩn hóa

```python
from ai.PhysicalBranch.interface import AIResult
from ai.PhysicalBranch.brand import normalize_metrics

metrics = monitor.process_landmarks(None, (640, 480), timestamp=0.0)
result = AIResult.from_metrics(metrics)
print(result.to_json())
scores = normalize_metrics(result, scales={"perclos_pct": ((0, 0), (20, 1))})
```

`AIResult.from_metrics()` chọn các trường DTO, bỏ thông tin phụ như `fps` và
`processing_ms`. `to_json()` từ chối NaN/Inf. Các thuộc tính mới có trong DTO;
`pitch_deviation_*` cũ được bỏ để dùng đúng các chỉ số gật đầu và biên độ.
`lstm_drowsiness_probability` và `warning_score` vẫn dành cho tầng tích hợp,
PhysicalBranch không tính chúng.

`normalize_metrics()` trả điểm quy ước 0–1, nội suy giữa các mốc và chặn hai đầu.
Mặc định giữ các thang tham khảo: thời lượng mắt 400→800 ms, PERCLOS 0→15%,
biên độ pitch 8→20°. Phép nội suy thành điểm là quy ước triển khai, không phải
xác suất được các nghiên cứu cung cấp. Cần ghi đè `scales` khi hiệu chỉnh thiết lập.

Tần suất nháy mắt thấp không còn tự nhận điểm nguy hiểm cao. EAR, MAR, các tần
suất, thời lượng ngáp/gật và FFT mặc định chưa có thang, trả `None`.
Muốn dùng các mốc MAR 0.35/0.73 trong tài liệu, truyền thang tường minh.
Đã bỏ `pitch_safe_range` và điểm của `pitch_deg`; dùng `pitch_amplitude_deg`
tương đối với baseline. Thiếu số đo, NaN/Inf hoặc thiếu thang trả `None`;
miền giá trị sai hoặc thang không hợp lệ gây `ValueError`.

## 6. Chạy và kiểm thử

Cần Python ≥ 3.10, NumPy, OpenCV. Xử lý frame cần MediaPipe có
`solutions.face_mesh`; `process_landmarks()` không cần MediaPipe.
TensorBoard chỉ cần khi bật `logdir`.

Chạy từ thư mục gốc repository:

```bash
python -m ai.PhysicalBranch.camera_metrics --window-sec 30 --metric-window perclos=60 --metric-window head_motion=20
python -m ai.PhysicalBranch.camera_metrics --source video.mp4 --video-time --no-display --window-sec 60
python -m unittest discover -s ai/PhysicalBranch/tests -v
```

Mặc định mở camera 0; `r` reset, `q` thoát. `--metric-window NAME=SECONDS`
có thể lặp lại. Trong Python, `run(source=..., on_metrics=..., logdir=...)`
nhận callback mỗi frame, tự giải phóng tài nguyên khi kết thúc hoặc gặp lỗi.
`output_interval_sec` chỉ điều chỉnh in JSON, không thay đổi cửa sổ thống kê.

`process(frame, timestamp)` nhận ảnh BGR; `process_landmarks(points, (w, h), timestamp)`
nhận mảng landmark pixel hữu hạn `N × 2`, `N ≥ 468`. Timestamp tính bằng giây,
phải hữu hạn và tăng nghiêm ngặt. Khi không truyền timestamp vào `process()`,
hệ thống dùng `time.monotonic()`; `run()` dùng thời gian từ đầu phiên hoặc
`frame_index / FPS_video` khi bật `use_video_time`.
FPS cấu hình quyết định số mẫu khởi tạo HMM; thống kê dùng timestamp thực tế.
Khi gọi `process()` trực tiếp, đóng tracker bằng `close()` trong `finally`.
