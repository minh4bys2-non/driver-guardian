# PhysicalBranch

Triển khai các metric trong `Metric_revised.md` (tài liệu nằm tại
`/home/tranmanhduy/Workspace/ptithcm/Documents/Metric_revised.md`).
Nhánh này đo tín hiệu mắt, miệng và đầu; không tự kết luận buồn ngủ hoặc chạy LSTM.

## 1. Cấu trúc

```text
ai/PhysicalBranch/
├── Structure.md
├── camera_metrics.py
├── adaptive_hmm_fsm.py
├── temporal_metrics.py
├── head_pose_estimation.py
├── interface.py
├── brand.py
└── tests/
    └── test_metrics.py
```

`__pycache__/` là cache Python sinh khi chạy, không thuộc mã nguồn.

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
`[-180, 180)`. Chuỗi góc hiệu chuẩn được unwrap trước khi lấy trung vị để
các mẫu gần +180° và -180° không tạo baseline sai gần 0°.
Khi chưa đủ mẫu hiệu chuẩn, mất mặt, lỗi pose hoặc khoảng cách mẫu vượt
`max_gap_sec` sẽ xóa các mẫu baseline đang gom; baseline đã hoàn tất vẫn giữ.
`pitch_amplitude_deg = abs(pitch_deg)`.
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
Tích phân dùng các đoạn nội suy tuyến tính, cắt đúng biên cửa sổ và tách tại
điểm pitch bằng 0 khi tính trị tuyệt đối. Ví dụ đoạn từ -10° đến +10° có
biên độ trung bình 5°. Phép tính dùng trực tiếp tổng diện tích, không phụ
thuộc API `np.trapz` đã thiếu trong môi trường NumPy được kiểm thử.
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

### Thuộc tính cấu hình ảnh hưởng đến độ chính xác

Các tham số dưới đây ảnh hưởng đến phân loại mắt/miệng, xác nhận sự kiện
hoặc chỉ số thống kê. Tăng giá trị không đồng nghĩa tăng độ chính xác;
các mặc định chưa được chứng minh là tối ưu trên video thực tế.

#### Khởi tạo và thích nghi mắt/miệng

Cấu hình riêng qua `eye_options` và `mouth_options` của `CameraMetrics`;
hai dictionary này ghi đè cấu hình chung `fps`, `window_size_sec`, `max_gap_sec`.
Nguồn: [AdaptiveHMM_FSM](adaptive_hmm_fsm.py).

| Thuộc tính | Mặc định | Ảnh hưởng |
| --- | --- | --- |
| `init_duration_sec` | 5 giây | Quyết định lượng mẫu khởi tạo HMM và mốc EAR/MAR bình thường; ít mẫu dễ làm mốc thiếu ổn định. |
| `fps` | 30 | Số mẫu khởi tạo là `max(4, round(fps × init_duration_sec))`; dùng suy ra thời gian nếu gọi detector trực tiếp mà không truyền timestamp. |
| `learning_rate` | 0.01 | Mức cập nhật HMM mỗi lần thích nghi; cao thì thích nghi nhanh nhưng dễ bị dữ liệu gần đây kéo lệch. |
| `adapt_interval` | 300 mẫu | Số mẫu gom trước khi thử cập nhật; nhỏ thì xét cập nhật thường xuyên hơn, lớn thì thích nghi chậm hơn. |

`init_duration_sec=5` ở `fps=30` nghĩa là 150 mẫu hợp lệ, không bảo đảm
hoàn tất sau đúng 5 giây thực tế. Mất mặt hoặc xử lý chậm làm khởi tạo lâu hơn.
HMM chỉ cập nhật online nếu tổng trọng số posterior của mỗi trạng thái
trong lô đạt ít nhất 2; đủ `adapt_interval` mẫu chưa chắc đã cập nhật.

#### Thời lượng sự kiện và khoảng mất dữ liệu

Nguồn: [StateMachine](temporal_metrics.py), [CameraMetrics](camera_metrics.py).

| Thuộc tính | Mặc định | Ảnh hưởng |
| --- | --- | --- |
| `min_duration_ms` | Mắt: 100 ms; miệng: 3500 ms | Thời lượng tối thiểu để đếm sự kiện; tăng sẽ loại sự kiện ngắn nhưng có thể bỏ sót sự kiện thật. |
| `max_duration_ms` | Mắt: 2000 ms; miệng: 7500 ms | Thời lượng tối đa để đếm sự kiện; vượt giới hạn sẽ không được tính vào bộ đếm. |
| `nod_duration_ms` | `(800, 3500)` ms | Cặp thời lượng tối thiểu/tối đa cho gật đầu, truyền trực tiếp vào `CameraMetrics`. |
| `max_gap_sec` | 0.25 giây | Khoảng cách tối đa giữa hai mẫu; vượt ngưỡng ngắt sự kiện, loại khoảng trống khỏi tỷ lệ và xóa chuỗi FFT liên tục. |

`min_duration_ms`/`max_duration_ms` lọc sự kiện để đếm, không trực tiếp
quyết định mắt đóng hay miệng mở. Ví dụ nhắm mắt quá 2000 ms không được tính
là một lần nháy mắt hợp lệ nhưng vẫn có thể đóng góp vào PERCLOS.
Một mẫu mất tín hiệu cũng ngắt sự kiện, kể cả khoảng cách mẫu chưa vượt ngưỡng.

#### Cửa sổ thống kê

| Thuộc tính | Mặc định | Ảnh hưởng |
| --- | --- | --- |
| `window_sec` | 60 giây | Cửa sổ chung trong `CameraMetrics`. |
| `window_size_sec` | 60 giây | Cửa sổ đếm sự kiện trong detector mắt/miệng. |
| `ratio_window_sec` | `None` → dùng `window_size_sec` | Cửa sổ riêng cho PERCLOS hoặc POM. |
| `windows` | `None` | Ghi đè từng cửa sổ `blink`, `perclos`, `yawn`, `pom`, `nod`, `over_angle`, `head_motion`; ưu tiên hơn các tùy chọn cửa sổ mắt/miệng. |

Cửa sổ ngắn phản ứng nhanh nhưng dao động nhiều; cửa sổ dài ổn định hơn
nhưng phản ánh thay đổi chậm hơn. Tần suất dùng toàn bộ chiều dài cửa sổ
làm mẫu số ngay từ đầu, nên lúc mới chạy có thể thấp hơn nhịp sự kiện thực tế.
`head_motion` mặc định 60 giây khi dùng `CameraMetrics`; lớp
`HeadMotionWindow` dùng độc lập mặc định 20 giây.

#### Hiệu chuẩn và chuyển động đầu

Nguồn: [CameraMetrics](camera_metrics.py), [HeadMotionWindow](temporal_metrics.py).

| Thuộc tính | Mặc định | Ảnh hưởng |
| --- | --- | --- |
| `calibration_frames` | 30 mẫu | Số mẫu ước lượng tư thế trung tính; mọi góc tương đối phụ thuộc mốc này. |
| `nod_pitch_deg` | 14° | Ngưỡng bắt đầu trạng thái gật; giảm ngưỡng làm tăng độ nhạy. |
| `nod_release_deg` | 8° | Ngưỡng kết thúc trạng thái gật; phải thấp hơn ngưỡng bắt đầu để giảm dao động trạng thái. |
| `nod_direction` | `None` | `None` xét cả hai chiều pitch qua trị tuyệt đối; `-1` hoặc `1` chỉ xét một chiều. |
| `angle_limits` | `(20, 25, 20)`° | Giới hạn pitch/yaw/roll, quyết định `over_angle` và `over_angle_pct`. |
| `head_motion_min_amplitude_deg` | 0.1° | Biên độ đỉnh–đỉnh phải vượt ngưỡng này mới xuất tần số FFT; tên tương ứng trong `HeadMotionWindow` là `min_amplitude_deg`. |

#### Landmark và hình học camera

Nguồn: [FaceTracker](camera_metrics.py), [HeadPoseEstimator](head_pose_estimation.py).

| Thuộc tính | Mặc định | Ảnh hưởng |
| --- | --- | --- |
| `detection_confidence` | 0.6 | Ngưỡng tin cậy phát hiện mặt, tác động đến landmark đầu vào. |
| `tracking_confidence` | 0.6 | Ngưỡng tin cậy theo dõi mặt, tác động đến tính liên tục của dữ liệu. |
| `camera_matrix` | Ước lượng từ kích thước ảnh | Tham số nội tại camera dùng ước lượng góc đầu. |
| `distortion_coeffs` | Các hệ số bằng 0 | Hiệu chỉnh méo ống kính khi ước lượng góc đầu. |
| `image_width`, `image_height` | Kích thước ảnh đầu vào | Dùng dựng ma trận camera mặc định và hệ tọa độ ảnh. |

`camera_matrix` và `distortion_coeffs` có trong constructor của
`HeadPoseEstimator`, chưa được đưa ra constructor của `CameraMetrics`.
`FaceTracker` đang cố định `max_num_faces=1`, `refine_landmarks=True`;
các bộ chỉ số landmark mắt/miệng và mô hình mặt 3D `MODEL` cũng cố định trong code.

#### HMM cấp thấp và ngưỡng cố định

Các tham số sau thuộc [AdaptiveHMM](adaptive_hmm_fsm.py), chưa được truyền
ra qua `eye_options`/`mouth_options`:

| Thuộc tính | Mặc định | Ảnh hưởng |
| --- | --- | --- |
| `min_variance` | `1e-5` | Sàn phương sai Gaussian, tránh phân phối quá hẹp khi dữ liệu ít biến thiên. |
| `em_iterations` | 20 | Số vòng EM khi khởi tạo bằng dữ liệu có hai trạng thái đủ tách biệt. |
| `positive_state` | Mắt: `"low"`; miệng: `"high"` | Quy ước trạng thái sự kiện ở phía giá trị thấp hay cao; wrapper tự chọn theo `mode`. |
| `n_states` | 2 | Code chỉ chấp nhận hai trạng thái, không hỗ trợ tùy ý tăng số trạng thái. |

Các hằng số trong `_initialize()` và `process()` cũng ảnh hưởng kết quả,
nhưng hiện muốn điều chỉnh phải sửa code:

| Hằng số | Giá trị hiện tại | Vai trò |
| --- | --- | --- |
| Percentile kiểm tra độ tách biệt | 5 và 95 | Xác định miền biến thiên dùng kiểm tra dữ liệu khởi tạo. |
| Ngưỡng tách hai trạng thái | Mắt: 0.06; miệng: 0.15; chế độ pitch: 10 | Quyết định học từ dữ liệu hay dùng phân phối dự phòng. `CameraMetrics` dùng FSM ngưỡng cho đầu, không dùng HMM pitch. |
| Percentile xác định mức bình thường | Mắt: 80; miệng/pitch: 20 | Xác định mốc `normal` ban đầu. |
| Điều kiện mắt đóng đủ khác biệt | `low < normal × 0.65` | Kiểm tra dữ liệu khởi tạo có trạng thái đóng mắt rõ ràng. |
| Mốc sự kiện dự phòng | Mắt: `normal × 0.2`; miệng/pitch: `normal + separation × 2` | Ước lượng trạng thái sự kiện khi dữ liệu chưa đủ đa dạng. |
| Độ lệch chuẩn dự phòng | `max(abs(normal - event) / 4, 0.01)` | Quyết định độ rộng phân phối ban đầu. |
| Ma trận chuyển trạng thái dự phòng `A` | `[[0.97, 0.03], [0.1, 0.9]]` | Chi phối xu hướng giữ hoặc chuyển trạng thái. |
| Xác suất ban đầu dự phòng `pi` | `[0.99, 0.01]` | Ưu tiên trạng thái bình thường khi bắt đầu suy luận. |
| Ngưỡng ép mắt mở | `EAR >= normal × 0.75` | Ghi đè trạng thái HMM thành mắt mở, posterior thành `[0.99, 0.01]`. |
| Hệ số PERCLOS P80 | 0.8 | Ngưỡng EAR là `opened - 0.8 × (opened - closed)`. |

Mốc `normal` và `event_reference` dùng tính P80 giữ nguyên sau khởi tạo.
Do đó `learning_rate` có thể thay đổi phân loại HMM nhưng không thay đổi
ngưỡng P80 dùng tính PERCLOS.

`scales` trong [brand.py](brand.py) chỉ ảnh hưởng điểm chuẩn hóa 0–1,
không thay đổi nhận diện thô; các thang mặc định được mô tả tại mục 5.
Ngoài constructor, `timestamp` và `run(use_video_time=...)` quyết định
thời lượng đo được. Khi xử lý video lưu sẵn, dùng timeline video để thời
lượng sự kiện không phụ thuộc tốc độ xử lý của máy.

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

### Giải thích từng thông số trên HUD

Các giá trị minh họa dưới đây lấy từ ảnh HUD người dùng cung cấp.
HUD thay dấu `_` trong tên trường bằng dấu cách và làm tròn số đến hai chữ
số thập phân; vì vậy số đếm nguyên cũng hiển thị như `15.00`.
EAR/MAR không có đơn vị; góc dùng độ, thời lượng sự kiện dùng ms,
thời gian quan sát dùng giây, tỷ lệ dùng %, tần suất sự kiện dùng lần/phút,
tần số chuyển động dùng Hz (chu kỳ/giây).

#### EYES: READY — mắt

`READY` cho biết HMM mắt đã đủ mẫu khởi tạo; không phải chứng nhận độ chính
xác hay kết luận người lái tỉnh táo. HUD có thể thay bằng `NO FACE` hoặc
`NO SIGNAL` khi frame hiện tại không có dữ liệu phù hợp.

| Nhãn trên HUD | Giá trị trong ảnh | Ý nghĩa và cách đọc |
| --- | --- | --- |
| `ear` | 0.29 | Eye Aspect Ratio: tỷ lệ độ mở mắt tính từ landmark, lấy trung bình hai mắt. Giá trị thấp biểu thị mắt khép hơn so với mốc của người đang đo. |
| `eye state` | OPEN | Trạng thái mắt hiện tại sau HMM và quy tắc ép mắt mở: `OPEN` = mở, `CLOSED` = đóng. |
| `perclos pct` | 2.97 | Phần trăm thời gian quan sát hợp lệ mà EAR bằng hoặc dưới ngưỡng P80. Trong ảnh tương đương khoảng 1.78 giây trên 60 giây; không phải xác suất buồn ngủ. |
| `blink rate per min` | 15.00 | Tần suất nháy mắt hợp lệ, tính bằng `blink_count × 60 / windows_sec["blink"]`. |
| `blink count` | 15.00 | Có 15 sự kiện nháy mắt hợp lệ kết thúc trong cửa sổ đếm hiện tại; không phải tổng cả phiên. |
| `blink duration ms` | 200.04 | Thời lượng lần nháy mắt hợp lệ gần nhất. Giữ nguyên đến khi có lần hợp lệ mới hoặc reset bộ đếm. |
| `eye closure duration ms` | 0.00 | Thời lượng pha nhắm mắt đang diễn ra; bằng 0 vì hiện tại mắt mở. |
| `last eye closure duration ms` | 35.99 | Thời lượng pha nhắm mắt vừa kết thúc, kể cả không đủ điều kiện tính là nháy mắt. Nếu dùng mặc định tối thiểu 100 ms, lần này không được đếm. |
| `p80 ear threshold` | 0.10 | Ngưỡng EAR tính PERCLOS: `normal - 0.8 × (normal - event_reference)`. Đây không phải ngưỡng duy nhất quyết định `eye state`; giá trị hiển thị đã làm tròn. |
| `eye observed sec` | 60.00 | Tổng thời gian quan sát hợp lệ trong cửa sổ PERCLOS, dùng làm mẫu số tính tỷ lệ; loại khoảng mất dữ liệu. |

#### MOUTH: READY — miệng

`READY` cho biết HMM miệng đã khởi tạo. `RESTING` là nhãn trạng thái nghỉ
của mô hình, không nhất thiết có nghĩa hai môi khép hoàn toàn.

| Nhãn trên HUD | Giá trị trong ảnh | Ý nghĩa và cách đọc |
| --- | --- | --- |
| `mar` | 0.38 | Mouth Aspect Ratio: tỷ lệ độ mở miệng từ landmark. So sánh theo mốc cá nhân; code không dùng một ngưỡng MAR cố định chung để nhận diện ngáp. |
| `mouth state` | RESTING | Trạng thái hiện tại: `RESTING` = nghỉ, `OPEN` = mở theo HMM. Một pha mở miệng phải đủ chu kỳ và thời lượng mới được tính là sự kiện ngáp. |
| `pom pct` | 0.00 | Phần trăm thời gian miệng ở trạng thái mở trên thời gian quan sát hợp lệ trong cửa sổ POM. Không phải tỷ lệ thời gian ngáp đã được xác nhận. |
| `yawning frequency per min` | 0.00 | Tần suất sự kiện ngáp hợp lệ trong cửa sổ đếm, đơn vị lần/phút. |
| `yawn count` | 0.00 | Không có sự kiện ngáp hợp lệ còn nằm trong cửa sổ đếm hiện tại. |
| `yawn duration ms` | 5320.29 | Thời lượng lần ngáp hợp lệ gần nhất, khoảng 5.32 giây. Có thể vẫn còn giá trị dù sự kiện đã ra khỏi cửa sổ và `yawn count` về 0. |
| `mouth open duration ms` | 0.00 | Thời lượng pha mở miệng đang diễn ra; bằng 0 vì trạng thái hiện tại là nghỉ. |
| `last mouth open duration ms` | 100.01 | Thời lượng pha mở miệng vừa kết thúc, kể cả không hợp lệ. Nếu dùng mặc định tối thiểu 3500 ms, lần này không được tính là ngáp. |
| `mouth observed sec` | 60.00 | Tổng thời gian quan sát hợp lệ trong cửa sổ POM, dùng làm mẫu số tính tỷ lệ. |

`yawn duration ms = 5320.29` cùng `yawn count = 0` không mâu thuẫn:
một trường nhớ sự kiện hợp lệ gần nhất, trường còn lại chỉ đếm trong cửa sổ.
Tương tự, `pom pct = 0.00` là số đã làm tròn, không nhất thiết bằng 0 tuyệt đối;
cửa sổ POM và cửa sổ ngáp cũng có thể được cấu hình khác nhau.

#### HEAD: READY — đầu

`READY` nghĩa là frame hiện tại có góc đầu hợp lệ và đã có baseline hiệu
chuẩn. Pitch/yaw/roll đều là góc tương đối so với baseline, không phải góc
tuyệt đối so với đường hoặc camera. Chiều dấu phụ thuộc hệ tọa độ đang dùng.

| Nhãn trên HUD | Giá trị trong ảnh | Ý nghĩa và cách đọc |
| --- | --- | --- |
| `pitch deg` | 5.55 | Góc cúi/ngẩng đầu so với baseline, có dấu. |
| `yaw deg` | 50.02 | Góc quay đầu sang hai bên so với baseline. Nếu dùng giới hạn mặc định 25°, frame này đang vượt giới hạn yaw. |
| `roll deg` | -3.25 | Góc nghiêng đầu sang vai so với baseline; dấu âm biểu thị một chiều quay trong hệ tọa độ. |
| `pitch amplitude deg` | 5.55 | Độ lệch pitch hiện tại so với baseline: `abs(pitch_deg)`; không phải biên độ đỉnh–đỉnh của cả cửa sổ. |
| `over angle pct` | 82.65 | Phần trăm thời gian quan sát hợp lệ có ít nhất một góc vượt giới hạn pitch/yaw/roll. Với 60 giây hợp lệ, tương đương khoảng 49.59 giây; không phải mức độ lệch góc hiện tại. |
| `nodding` | NO | Hiện không ở pha gật theo ngưỡng pitch và ngưỡng nhả. Vượt yaw không kích hoạt trạng thái này. |
| `nod detected` | NO | Frame này không vừa xác nhận một chu kỳ gật hợp lệ. `YES` chỉ xuất hiện ở frame kết thúc sự kiện, không giữ suốt pha gật. |
| `nodding frequency per min` | 5.00 | Tần suất gật hợp lệ trong cửa sổ đếm, đơn vị lần/phút. |
| `nod count` | 5.00 | Có 5 sự kiện gật hợp lệ kết thúc trong cửa sổ đếm hiện tại. |
| `nod duration ms` | 0.00 | Thời lượng pha gật đang diễn ra; bằng 0 vì hiện không ở trạng thái gật. |
| `nod event duration ms` | 964.03 | Thời lượng sự kiện gật hợp lệ gần nhất, khoảng 0.964 giây; được giữ ngay cả khi sự kiện ra khỏi cửa sổ. |
| `head observed sec` | 60.00 | Thời gian quan sát góc đầu hợp lệ trong cửa sổ `over_angle`, dùng tính tỷ lệ lệch góc. |
| `pitch mean deg` | 7.57 | Trung bình pitch có dấu theo thời gian trong cửa sổ `head_motion`. Các độ lệch trái dấu có thể triệt tiêu nhau. |
| `pitch mean amplitude deg` | 7.96 | Trung bình `abs(pitch)` theo thời gian trong cửa sổ `head_motion`; hai chiều lệch đều đóng góp giá trị dương. |
| `head motion frequency hz` | 0.02 | Tần số trội của chuỗi pitch có dấu, tính bằng FFT sau khi trừ trung bình và áp cửa sổ Hann. Không phải số lần gật mỗi giây. |
| `head motion resolution hz` | 0.02 | Khoảng cách giữa các bin FFT, xấp xỉ `1 / T`. Với cửa sổ 60 giây, giá trị khoảng 0.0167 Hz được làm tròn thành 0.02. |
| `head motion observed sec` | 60.00 | Thời lượng chuỗi pitch liên tục hiện có trong cửa sổ FFT; khác với tổng thời gian hợp lệ dùng cho `head observed sec`. |

Không nên suy ra chính xác chu kỳ 50 giây chỉ từ `head motion frequency hz = 0.02`
trên HUD vì số đã làm tròn. Khi tần số trội nằm gần bin thấp nhất, nó có
thể phản ánh biến thiên chậm hoặc xu hướng trong cửa sổ; riêng giá trị này
chưa chứng minh có gật đầu tuần hoàn. `nodding frequency per min = 5` và
`head motion frequency hz = 0.02` đo hai khái niệm khác nhau nên không cần
quy đổi ra cùng một kết quả.

#### Hiệu năng xử lý

| Nhãn trên HUD | Giá trị trong ảnh | Ý nghĩa và cách đọc |
| --- | --- | --- |
| `FPS / processing` | `27.8 / 7.9 ms` | 27.8 là FPS tức thời tính từ khoảng cách giữa hai mốc bắt đầu xử lý frame liên tiếp; 7.9 ms là thời gian gọi `process()` cho frame hiện tại. |

FPS phản ánh cả nhịp đọc frame và công việc giữa các lần xử lý; `processing`
không bao gồm toàn bộ thời gian đọc camera, vẽ HUD, callback và ghi log.
Vì vậy FPS không nhất thiết bằng `1000 / processing_ms`, và FPS hiển thị
không phải giá trị `fps` cấu hình dùng tính số mẫu khởi tạo.

Ảnh HUD mô tả các chỉ số đo được, chưa đủ để kết luận buồn ngủ hay xác nhận
độ chính xác mô hình. Ví dụ yaw lớn có thể do quay đầu thật hoặc baseline/
ước lượng góc chưa phù hợp; cần đối chiếu hình ảnh và dữ liệu thực tế.

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
Giá trị trả về của `run()` là số frame đã xử lý, gồm cả frame hiển thị lúc nhấn `q`.
`output_interval_sec` chỉ điều chỉnh in JSON, không thay đổi cửa sổ thống kê.

`process(frame, timestamp)` nhận ảnh BGR; `process_landmarks(points, (w, h), timestamp)`
nhận mảng landmark pixel hữu hạn `N × 2`, `N ≥ 468`. Timestamp tính bằng giây,
phải hữu hạn và tăng nghiêm ngặt. Khi không truyền timestamp vào `process()`,
hệ thống dùng `time.monotonic()`; `run()` dùng thời gian từ đầu phiên hoặc
`frame_index / FPS_video` khi bật `use_video_time`.
FPS cấu hình quyết định số mẫu khởi tạo HMM; thống kê dùng timestamp thực tế.
Khi gọi `process()` trực tiếp, đóng tracker bằng `close()` trong `finally`.

### Kết quả rà soát ngày 2026-10-07

18 bài kiểm thử vượt qua trên Python 3.12.13, NumPy 2.4.4 và OpenCV 5.0.0.
Các ca hồi quy bổ sung kiểm tra tích phân biên độ khi pitch đổi dấu, cắt cửa
sổ, baseline qua ±180°, gián đoạn hiệu chuẩn và số frame/tài nguyên khi thoát.
Kiểm thử dùng landmark tổng hợp và camera/tracker giả lập; chưa kiểm chứng
camera thực, MediaPipe thực hoặc chất lượng phát hiện trên video thực tế.
