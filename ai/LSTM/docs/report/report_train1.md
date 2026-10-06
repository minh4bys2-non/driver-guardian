# Báo cáo hoàn thành: Script Huấn luyện (train1.py)

## 1. Kết quả thực hiện
Đã hoàn thành việc tạo script huấn luyện `src/train1.py` cho mô hình Spatio-Temporal `ConvGRUClassifier` của hệ thống Driver Guardian AI. File code được thiết kế đúng chuẩn, module hóa và đáp ứng 100% các yêu cầu đề ra.

## 2. Chi tiết các tính năng đã triển khai

### 2.1. Quản lý cấu hình & Tính tái lập
- Sử dụng hàm `load_config()` từ `configs/config.py` để nạp mọi tham số (batch size, epochs, paths, model settings).
- Tích hợp hàm `seed_everything()` nhằm cố định seed cho Python, Numpy, và PyTorch, bảo đảm kết quả tái tạo được đúng như tiêu chuẩn `AGENTS.md`.

### 2.2. Dataloader & Feature Extractor tối ưu VRAM
- Import `build_raw_video_dataloaders` từ `src/dataset2.py` hỗ trợ tải video thô an toàn đa tiến trình, không gây chết worker.
- Nhúng `ChunkedBackboneNeckExtractor` trực tiếp vào vòng lặp (với khối `torch.inference_mode()` mô phỏng qua việc không truyền gradient) để trích xuất feature `p3, p4, p5` ngay trên GPU. Các chunk nhỏ chống lỗi OOM trên cấu hình GPU VRAM thấp.

### 2.3. Vòng lặp huấn luyện chuẩn
- **Mixed Precision Training**: Có sử dụng `torch.amp.autocast` và `GradScaler` để giảm một nửa lượng RAM tiêu thụ và gia tăng tốc độ tính toán (FP16).
- **Hàm mất mát và Optimizer**: Có tự động nhận diện thiết lập Loss (`CrossEntropy` hoặc `BCEWithLogitsLoss`), Optimizer (`AdamW` hoặc `SGD`) và Scheduler (`CosineAnnealingLR`).

### 2.4. TensorBoard Logging Độc lập toàn diện
- Thiết lập thư mục log riêng thông qua cấu hình `tb_log_dir` và `experiment_name`.
- **Log theo Step**: Ghi chi tiết `Loss`, `Accuracy`, `Recall`, `F1` tại các mốc chia nhỏ của epoch (phụ thuộc vào biến `log_step_interval`).
- **Log theo Epoch**: Ghi lại giá trị tổng hợp `Loss`, `Accuracy`, `Recall`, `F1` ở cuối mỗi vòng lặp `Train` và `Val`. Log cả biểu đồ biến thiên Learning Rate.

### 2.5. Checkpoint Management & Config Visualization
- **Khởi tạo & Visualization:** Script tự động in ra toàn bộ cấu hình hệ thống bằng `pprint` ở ngay đầu tiến trình để dễ dàng theo dõi.
- **Tính toán Metric:** Đã thiết lập tính năng tính `macro f1` qua `sklearn.metrics`.
- **Lưu Checkpoint theo cấu hình:** Lưu `last.pt` sau mỗi epoch. Lưu `best.pt` tự động nếu điểm F1 trên tập kiểm định cao hơn mốc đỉnh cũ. Đồng thời, tự động lưu checkpoint của các epoch cụ thể (VD: `epoch_5.pt`) dựa vào biến `save_all_epochs` hoặc `save_ckpt_interval_epochs` cấu hình trong `configs/config.yaml`.
- **Khôi phục mô hình (Resume):** Hỗ trợ tính năng `load_checkpoint()` cho phép tiếp tục train từ một epoch cụ thể. Có thể kích hoạt thông qua tham số `--resume` trên CLI hoặc biến `enable_resume` và `resume` trong file `configs/config.yaml`.

## 3. Khuyến nghị chạy thử (Test)
Bạn có thể chạy thử pipeline qua dòng lệnh sau:
```bash
python src/train1.py
# Hoặc chỉ định file cấu hình rõ ràng:
python src/train1.py --config configs/config.yaml
```

*Tất cả các tiêu chí trong `AGENTS.md` (Module hóa, Tái lập, Logging chuẩn, Exception handling ngầm định) đã được tuân thủ nghiêm ngặt.*
