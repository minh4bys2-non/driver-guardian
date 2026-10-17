# Chuẩn hóa theo miền bộ đo — physical_v1

DataSystem bỏ quy tắc nguy hiểm `risk_v2`: không còn `safe/danger`, đường cong
nháy mắt/FFT hai nhánh hoặc nhân tần suất nháy mắt với chỉ số mắt khác.
Tám đặc trưng là số đo đã đổi thang, không phải tám điểm nguy hiểm hay xác suất.

## Lựa chọn và cơ sở

Dùng phép tuyến tính `z = (x - L) / (U - L)` với `L = 0` và `U` lấy từ miền
bộ trích xuất. Đây là lựa chọn thiết kế cho pipeline này, không phải khẳng định
đã tối ưu độ chính xác phân loại bằng thực nghiệm.

Phép biến đổi giữ thứ tự và tỷ lệ so với 0, có thể đảo bằng `x = z * U`.
Không dùng nhãn, không fit thống kê trên toàn dataset, nên không phát sinh
rò rỉ train/test từ bước chuẩn hóa. Chạy train và suy luận phải dùng cùng cấu hình.

So với các lựa chọn khác:

- Min–Max theo dữ liệu phụ thuộc cực trị tập train và nhạy với outlier;
  dữ liệu mới có thể vượt miền đã fit. [Tài liệu MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).
- QuantileTransformer đưa dữ liệu về phân phối đều, nhưng thay đổi khoảng cách
  và có thể làm biến dạng tương quan tuyến tính. [Tài liệu QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html).
- Ở đây mỗi đặc trưng có cận suy ra được từ tỷ lệ, FSM hoặc tốc độ lấy mẫu,
  nên chưa cần thêm một bộ fit/load thống kê. Cận FSM là cận kỹ thuật, không phải
  ngưỡng sinh lý hoặc mức nguy hiểm được công bố trong nghiên cứu.

## Công thức cho tám đặc trưng

| Đặc trưng | Số đo x | Hệ số U |
| --- | --- | --- |
| `blink_frequency` | Nháy mắt/phút | Cận tần suất FSM mắt ở công thức dưới |
| `blink_duration` | P90 thời lượng nháy mắt hợp lệ, ms | Giới hạn thời lượng tối đa FSM mắt |
| `perclos` | `perclos_pct / 100` | 1 |
| `yawn_frequency` | Ngáp/phút | Cận tần suất FSM miệng |
| `nod_duration` | Thời lượng gật hợp lệ lớn nhất, giây | Giới hạn thời lượng tối đa FSM gật / 1000 |
| `nod_frequency` | Gật/phút | Cận tần suất FSM gật |
| `dominant_head_motion_frequency` | Tần số đỉnh FFT pitch, Hz | Nyquist lưới nội suy; mặc định `fps / 2` |
| `neuralnetwork_score` | Score lớp 1 của CNN–ConvGRU | 1 |

P90 và max là mặc định của `blink_duration_percentile=90`,
`nod_duration_percentile=100` trong `DataSystem/config.py`. Không có sự kiện hợp
lệ thì thời lượng là 0. Thời lượng vẫn chỉ mô tả sự kiện hoàn tất được FSM chấp
nhận; sự kiện kéo dài vượt giới hạn không được đưa vào thống kê này.

Ví dụ mặc định: P90 nháy mắt 680 ms → 0,34; gật 1,5 giây → 1,5/3,5;
PERCLOS 10% → 0,10; FFT 0,1 Hz ở 30 FPS → 0,1/15.
Tần suất nháy mắt không thay đổi khi chỉ thay PERCLOS.

### Cận tần suất sự kiện

Với cửa sổ FSM dài W giây, thời lượng tối thiểu m ms và FPS xử lý F:

```text
d = max((m - 0.000001) / 1000, 1/F)
N_upper = ceil(W / d) + 1
U_rate = N_upper * 60 / W
z_rate = rate_per_minute / U_rate
```

Đây là cận trên bảo thủ, không phải tần suất tối đa thực tế đạt được. Một sự kiện
hoàn tất chiếm ít nhất d giây; sự kiện kế tiếp không chồng lên sự kiện trước.
Cộng một sự kiện cho trường hợp bắt đầu trước biên trái nhưng kết thúc trong
cửa sổ. Khoảng nghỉ giữa sự kiện chỉ làm số đếm thực tế nhỏ hơn.
Dung sai 0,000001 ms khớp FSM; `1/F` xử lý cả cấu hình thời lượng tối thiểu bằng 0.
Dùng cửa sổ thực tế của từng FSM sau khi gộp `windows`, `eye_options`, `mouth_options`.

### Cận FFT

FFT của chuỗi lấy mẫu đều có tần số không âm tới Nyquist, F/2.
[NumPy rfftfreq](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftfreq.html)
mô tả các bin tần số, bao gồm Nyquist khi số mẫu chẵn.
DataSystem lấy mẫu lại video theo timeline F rồi PhysicalBranch nội suy chuỗi pitch.
Với cửa sổ FFT W không chia hết cho bước frame, code làm tròn số mẫu nội suy.
Cận bảo thủ dùng `max(3, ceil(W * F)) / (2 * W)` để bao cả sai lệch làm tròn
và yêu cầu ít nhất bốn mẫu FFT. Với mặc định W=60, F=30, cận đúng bằng 15 Hz.
Cận này mô tả miền tính toán; tăng FPS bằng lặp frame không tạo thêm thông tin
chuyển động hay tăng băng thông thực của video nguồn.

Bộ lọc FFT yêu cầu tần số hữu hạn không âm, độ phân giải dương và quan sát
liên tục đủ cửa sổ. Đầu đứng yên hoặc dao động không vượt ngưỡng biên độ
trả 0 Hz và giữ mẫu. Mất tín hiệu hoặc chưa đủ cửa sổ trả None và loại mẫu.

## Kiểm tra dữ liệu và phiên bản

`normalization.py` suy ra hệ số từ một `CameraMetrics(config=...)` chưa mở
FaceMesh/camera. Vì vậy dùng đúng giới hạn detector kế thừa, không lặp bộ ngưỡng.
Giá trị âm, thiếu, NaN/Inf hoặc vượt cận bị từ chối; chỉ kẹp dung sai số học 1e-6
ở biên trên. Không thay dữ liệu thiếu bằng 0 và không làm tròn 4 chữ số.

Mỗi dòng JSONL lưu:

- Tám giá trị chuẩn hóa trong `[0,1]` và `raw_features` theo đơn vị ở bảng.
- `normalization_scales`: hệ số U của từng đặc trưng.
- `normalization_id`: SHA-256 của encoding và toàn bộ cấu hình kế thừa đã áp dụng.
- `feature_encoding="physical_v1"`, video nguồn, mốc cửa sổ và nhãn manifest.

Output mặc định: `drowsiness_physical_60.jsonl`. Resume từ chối encoding, mã cấu
hình hoặc hệ số khác. TrainingSystem đọc phiên bản mới và từ chối trộn cấu hình.
Hash không bao gồm nội dung checkpoint hoặc mã nguồn; lưu các phiên bản này riêng.
Dữ liệu `risk_v1/risk_v2` cũ đã mất thông tin do bão hòa và ánh xạ nhiều-một:
phải trích xuất lại từ video, không đổi tên encoding của các điểm cũ.

## Giới hạn và đánh giá tiếp theo

Cận kỹ thuật có thể rộng hơn dữ liệu thực tế: tần suất nháy mắt hoặc FFT thường
nằm gần 0 sau khi chia cận. `[0,1]` không có nghĩa các cột có phân phối đều,
phương sai bằng nhau hay mức ảnh hưởng bằng nhau. Chưa chứng minh cách này tốt
hơn Min–Max/quantile học trên tập train về chất lượng dự đoán.

TrainingSystem hiện dùng weighted mean, noisy-OR và cooccurrence với trọng số
dương. Những phép gộp đó đơn điệu theo từng đầu vào, không học được quan hệ
hình chữ U hoặc ảnh hưởng ngược chiều của một số số đo. Phải đánh giá lại mô hình
và huấn luyện lại trọng số; không tái sử dụng kết quả tối ưu từ các điểm nguy hiểm.
Nếu cần mô hình học quan hệ đó, thay bộ phân loại ở một bước riêng thay vì đưa
quy tắc bệnh lý trở lại chuẩn hóa.

Khi có đủ dữ liệu, so sánh trên cùng split theo người lái; chỉ fit mọi scaler
thống kê trên train và dùng nguyên tham số cho validation/test. Đây là quy tắc
tránh leakage trong [hướng dẫn scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
