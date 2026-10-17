# TrainingSystem — Tối ưu trọng số và chọn đặc trưng

TrainingSystem tối ưu `[w1, ..., w8, T_cls]` trong `[0,1]^9` từ JSONL của DataSystem. Không huấn luyện lại CNN/ConvGRU. Bốn optimizer là các biến thể liên tục, đa mục tiêu; kiểm thử phần mềm không xác nhận tương đương nguyên vẹn bài báo gốc.

## Chạy

Từ thư mục gốc repository, sau khi DataSystem hoàn tất ghi dữ liệu:

```bash
python -m ai.OptimalAlgorithsm.TrainingSystem.cmdpsofs
python -m ai.OptimalAlgorithsm.TrainingSystem.IBGWO4
python -m ai.OptimalAlgorithsm.TrainingSystem.MOHHO
python -m ai.OptimalAlgorithsm.TrainingSystem.NSPSOFS
```

Cả bốn entrypoint lấy `DATA_PATH` từ `DataSystem.config.OUTPUT_JSONL`, mặc định là `drowsiness_physical_60.jsonl`. Chỉnh đường dẫn chung tại DataSystem; `OUTPUT_DIR`, `FITNESS_METHOD`, `CONFIG` nằm trong từng entrypoint. Cấu hình hiện tại dùng `cooccurrence`, quần thể 50, 100 mốc generation, seed 42. `generations` bao gồm quần thể khởi tạo: 100 mốc tương ứng 99 bước cập nhật.

Mỗi lần chạy tạo `output/<algorithm>/<algorithm>_<fitness>/`, ví dụ `output/nspsofs/nspsofs_cooccurrence/`. Nếu thư mục đã tồn tại, thêm hậu tố `_2`, `_3`, … để không ghi đè kết quả cũ. Đọc đường dẫn được in cuối lần chạy; các file nằm trực tiếp trong thư mục output cũ không được cập nhật.

## Đánh giá theo subject

`engine.run_experiment()` dùng chung cho cả bốn entrypoint:

1. Đọc một snapshot JSONL, kiểm tra dữ liệu và lưu SHA-256.
2. Chia 60%/20%/20% **số subject** vào train/validation/test với `split_seed=42`, độc lập seed optimizer. Tỷ lệ số dòng có thể khác vì video dài ngắn khác nhau.
3. Optimizer chỉ nhận `X_train, y_train`.
4. Chấm archive cuối trên validation, cùng hai baseline luôn đoán 0 và luôn đoán 1. Chọn theo balanced accuracy cao nhất, rồi F1, rồi ít đặc trưng hơn. Baseline thắng khi candidate không hơn và chỉ dùng nhiều đặc trưng hơn.
5. Chốt một mô hình rồi mới dự đoán test; không dùng test để chọn candidate hoặc threshold. Báo thêm hai baseline cố định trên test để đối chiếu.

Balanced accuracy = `(recall + specificity) / 2`; luôn đoán một lớp đạt 0.5 khi tập có cả hai nhãn. Không tự cấm threshold 0: nghiệm luôn báo buồn ngủ vẫn có thể tồn tại trong Pareto, nhưng không được trình bày thành mô hình hoàn hảo.

Metadata bắt buộc có `source_video`. Ưu tiên `subject_id`; nếu thiếu, chỉ suy ra từ đường dẫn UTA-RLDD `FoldN_partN/<subject>/<video>`. ID subject được dùng xuyên fold/part. ID số `1`, `"1"`, `"01"` được gom cùng nhóm; `subject_id` trên đường dẫn UTA phải khớp ID trong đường dẫn. Đường dẫn tương đương được chuẩn hóa trước kiểm tra nhóm. Dataset khác cần bổ sung `subject_id`, hoặc chủ động truyền `group_by="video"` vào `run_experiment()`; chia theo video không bảo đảm người độc lập. Cần ít nhất 5 nhóm và mỗi tập có cả hai nhãn; không tự rơi về chia ngẫu nhiên theo dòng.

DataSystem mới mặc định dùng cửa sổ và stride đều 60 giây. Nếu dùng stride nhỏ hơn cửa sổ, chia theo dòng sẽ làm nội dung chồng lấn giữa các tập. Giữ toàn bộ subject trong một tập tránh vấn đề này. Việc chia này chỉ bảo vệ tầng tối ưu hiện tại: vẫn cần kiểm tra CNN/ConvGRU và các ngưỡng chuẩn hóa trước đó chưa được huấn luyện/tinh chỉnh bằng người thuộc test.

## Hàm dự đoán và mục tiêu

Thứ tự tám đặc trưng nằm trong `fitnessFunction.FEATURES`. Feature phải là scalar số hữu hạn trong `[0,1]` (không nhận boolean hay chuỗi số), nhãn là số nguyên 0 hoặc 1. Dòng trống bỏ qua; dataset rỗng hoặc sai schema bị từ chối. Hỗ trợ `physical_v1` và các phiên bản cũ `scale_v1`, `risk_v1`, `risk_v2`, không cho trộn encoding. Với `physical_v1`, mã cấu hình và hệ số chuẩn hóa phải đồng nhất; chọn dataset mới `drowsiness_physical_60.jsonl` sau khi tạo lại dữ liệu. Không fit lại scaler trong TrainingSystem: `physical_v1` đã chia số đo cho cận bộ trích xuất, không sử dụng nhãn hay thống kê test. Các phép gộp trọng số dương hiện tại chưa học được quan hệ không đơn điệu từ số đo trung lập. Thiếu encoding được xem là `scale_v1`.

Cột thứ 8 là `neuralnetwork_score` (xác suất lớp drowsy từ CNN–ConvGRU).
JSONL và `normalization_scales` phải dùng đúng khóa này. Cả bốn thuật toán,
metadata và báo cáo `selected_features` dùng chung tên từ `FEATURES`.

Chọn feature khi `wj >= 0.5`, chỉ feature được chọn tham gia:

```text
weighted_mean: score = sum(Xj * wj) / sum(wj)
noisy_or:      score = 1 - product(1 - Xj * wj)
cooccurrence:  score = a + (1 - a) * a * b
y_pred = (score >= T_cls)
objectives = [1 - recall, 1 - precision, số feature / 8]
```

Với `cooccurrence`, `a`, `b` là hai giá trị `Xj * wj` lớn nhất; chỉ chọn một feature thì `b = 0`. Một giá trị lớn vẫn được giữ, cặp giá trị lớn tăng score, nhiều giá trị nhỏ không cộng dồn. Đặt `FITNESS_METHOD = "cooccurrence"` để sử dụng. Hàm tăng score khi feature tăng; với `physical_v1`, đây chỉ là một giả thuyết phân loại cần kiểm chứng, vì feature lớn hơn chỉ có nghĩa là số đo lớn hơn; chỉ xét cùng một mẫu/cửa sổ, không mô hình hóa thứ tự thời gian hay chứng minh các biểu hiện xảy ra cùng thời điểm. Đây là giả thuyết tính điểm cần đánh giá trên validation/test, không phải xác suất đã hiệu chuẩn.

Chuẩn hóa mới không tự khắc phục giới hạn của cả ba phép gộp: trọng số dương
không học được ảnh hưởng ngược chiều hoặc hình chữ U. Cận vật lý rộng cũng có thể
khiến một số cột tập trung gần 0. Không diễn giải score là mức nguy hiểm đã biết,
không đổi lại hướng đặc trưng bằng quy tắc thủ công. Cần tối ưu và đánh giá lại
trên dataset mới; không tái sử dụng candidate/threshold từ dataset risk cũ.
Đổi sang bộ phân loại học được quan hệ phi tuyến là thay đổi mô hình riêng.

Cả ba mục tiêu được tối thiểu hóa trên train. `noisy_or` không mặc nhiên là xác suất đã hiệu chuẩn. Không chọn feature thì dự đoán toàn 0, optimizer phạt `[1,1,1]`. Recall/precision có mẫu số bằng 0 được quy ước bằng 0.

Archive loại vị trí trùng, giữ nghiệm không bị trội, giới hạn bằng crowding distance. Archive hữu hạn không bảo đảm giữ mọi nghiệm Pareto từng gặp. NSPSOFS lấy leader từ quần thể; archive của nó chỉ phục vụ báo cáo.

## Đọc kết quả

- `split.json`: `input_schema`, hash dữ liệu, seed chia, số mẫu/nhãn, subject và chỉ số dòng dữ liệu (0-based, bỏ dòng trống) của từng tập.
- `config.json`: cấu hình, `input_schema` và phạm vi đánh giá; `n_samples` là số mẫu train, `n_samples_total` là toàn bộ dữ liệu.
- `generation_log.jsonl`: `representative` là **một candidate** được chọn theo balanced accuracy/F1/số feature trên train; mọi metric hiển thị cùng thuộc candidate đó. Đây chưa phải mô hình cuối.
- `archive_extrema` trong generation log giữ các cực trị riêng để phân tích Pareto, không ghép thành thành tích của một mô hình. `best_ideal_distance` chỉ là chẩn đoán archive, không dùng chọn mô hình cuối.
- `population_log.jsonl`, `pareto_archive.jsonl`: kết quả tối ưu trên train. Archive sắp theo balanced accuracy/F1/số feature trên train, không theo khoảng cách tới điểm lý tưởng.
- `validation_archive.jsonl`: kết quả validation của archive và baseline, không chứa metric test.
- **`evaluation.json`**: `fitness_method`, `input_schema`, mô hình đã chọn (`candidate` hoặc `constant_0/constant_1`), metric validation/test, baseline test, tổng số lượt đánh giá và thời gian toàn pipeline. Có TP/TN/FP/FN, recall, precision, F1, specificity, FPR, balanced accuracy và cờ dự đoán một lớp.
- Sáu biểu đồ PNG: recall/precision/số feature theo cùng representative mỗi generation; khoảng cách lý tưởng, Pareto và tần suất feature trên train.

`input_schema` lưu thứ tự tám feature, `feature_encoding`, `normalization_id` và `normalization_scales`. Khi suy luận phải dùng đúng thứ tự/cấu hình này; các số đo không phải xác suất. Dataset cũ không có cấu hình chuẩn hóa sẽ lưu hai trường normalization là `null`. File kết quả cũ không được tự chuyển đổi hoặc ghi đè.

Chỉ gọi trực tiếp lớp optimizer sẽ chỉ tối ưu trên dữ liệu được truyền vào. Muốn đánh giá đầy đủ cần dùng entrypoint hoặc `run_experiment()`.

`evaluations` đếm lượt gọi mục tiêu trong tìm kiếm, gồm candidate bị phạt vì không chọn feature; không gồm phép tính lại phục vụ log, validation/test. Với 50 × 100, CMDPSOFS và NSPSOFS có 5.000 lượt, IBGWO4 có 9.950; MOHHO tùy số nhánh dive. Cùng số generation không có nghĩa cùng ngân sách tính toán; cần kiểm soát số lượt đánh giá, thời gian và nhiều seed khi so sánh nghiên cứu.

Chạy vài giây với 9k dòng có thể bình thường: fitness chỉ xử lý ma trận tám feature đã trích xuất. Số dòng lớn không đồng nghĩa số quan sát độc lập lớn. Không chọn thuật toán/seed hoặc tiếp tục tinh chỉnh bằng kết quả test; dùng validation cho quyết định đó.

## Kiểm thử

```bash
MPLCONFIGDIR=/tmp/driver-guardian-mpl MPLBACKEND=Agg \
python -m unittest discover -s ai/OptimalAlgorithsm/TrainingSystem/tests -v
```

Kiểm thử gồm công thức fitness, confusion matrix, dữ liệu lỗi, Pareto/crowding, cấu hình, 4 optimizer × 2 fitness với tái lập theo seed, luồng `physical_v1` qua cả bốn optimizer và metadata trong kết quả, bộ đếm đánh giá, log cùng một candidate, chia subject không trùng, chọn baseline và kiểm tra nhãn test không ảnh hưởng mô hình được chọn.
