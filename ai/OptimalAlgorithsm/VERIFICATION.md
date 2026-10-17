# Kiểm tra PhysicalBranch, DataSystem và TrainingSystem

Kiểm tra ngày 2026-10-14, giữ nguyên `neuralnetwork.py` trả 0.5.

## Lỗi đã sửa

- FFT đủ dữ liệu nhưng đầu đứng yên từng trả `None`, làm DataSystem loại mẫu.
  Nay trả 0 Hz khi biên độ không vượt ngưỡng; thiếu dữ liệu vẫn trả `None`.
- Coverage từng nhánh từng so với cửa sổ chung, khiến cấu hình cửa sổ riêng
  có thể không bao giờ đạt. Nay dùng cửa sổ thực của perclos/pom/over_angle.
- DataSystem từng bỏ `subject_id` trong manifest. Nay giữ metadata và kiểm tra
  xung đột khi nối dữ liệu hoặc gặp lại video.
- TrainingSystem nhận diện subject trong đường dẫn ULDD, giữ mọi góc IR/L3D/R3D
  và cả hai trạng thái A/D của cùng người trong một tập.
- Resume từ chối bool/list ở trường đặc trưng, bool ở timestamp/thang chuẩn hóa,
  phù hợp với kiểm tra đầu vào của TrainingSystem.
- Hai test cũ kỳ vọng warm-up 150 mẫu và suy luận mạng thật được cập nhật thành
  300 mẫu (10 giây ở 30 FPS) và điểm mạng cố định 0.5.

## Các đoạn bị bỏ qua

TrainingSystem đọc toàn bộ dòng JSONL hợp lệ; dòng lỗi báo lỗi kèm số dòng.
Train/validation/test chia hết các mẫu theo nhóm. Chỉ train tham gia tối ưu,
validation chọn mô hình, test đánh giá sau cùng. Cả bốn optimizer được test
về số mẫu đánh giá, số thế hệ và số cá thể ghi log.

DataSystem chỉ xuất cửa sổ đầy. Mặc định FFT cần 60 giây pitch liên tục sau
hiệu chuẩn; một lần mất pose xóa chuỗi FFT. Coverage 80% không thay điều kiện
này. Với stride 60 giây, cửa sổ đầu bị loại và cửa sổ thứ hai mới có thể đạt.
Đầu đứng yên đủ dữ liệu không còn bị loại.

Đã chạy FaceMesh/PnP thật trên 120 giây đầu của `A/A/A_IR_A.mp4` và
`A/D/A_IR_D.mp4`, tổng 7.200 frame ở 30 FPS. Mỗi video giữ `[60,120)` và loại
`[0,60)` vì FFT mới quan sát 59 giây. NeuralNetwork được thay bằng callable
trả 0.5 trong lần kiểm tra này để không cần nạp checkpoint; không ghi dataset.

## Test

```bash
MPLBACKEND=Agg python -m pytest ai/PhysicalBranch/tests ai/OptimalAlgorithsm/DataSystem/tests ai/OptimalAlgorithsm/TrainingSystem/tests -q
```

Kết quả: **93 passed, 191 subtests passed**. `git diff --check` sạch.

Test gồm đối chiếu forward-backward HMM với liệt kê mọi chuỗi trạng thái,
chuẩn hóa xác suất sau thích nghi, PnP nhiều góc quay với intrinsics đã biết,
tích phân coverage độc lập, biên thời lượng FSM, FFT nhiều FPS/tần số và mất
pose, giải mã video thật, đổi RGB/BGR, đóng tài nguyên, lịch cửa sổ, resume,
chuẩn hóa, công thức fitness, Pareto, bốn optimizer, seed tái lập, chống rò rỉ
tập test, xuất JSONL và sáu biểu đồ thật cho mỗi optimizer.

Đây là xác minh logic và kiểm tra tích hợp có giới hạn. Chưa đo độ chính xác
phát hiện blink/yawn/nod trên toàn bộ dataset có nhãn sự kiện thủ công; chưa
đánh giá mô hình neural đang phát triển hoặc khẳng định độ chính xác buồn ngủ.
