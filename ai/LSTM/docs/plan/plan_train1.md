# Kế hoạch thực hiện: Viết script huấn luyện (train1.py)

## 1. Cấu trúc tổng quan của file `src/train1.py`
File sẽ được thiết kế dạng module hoàn chỉnh, có class `Trainer` hoặc hàm `main()` quản lý toàn bộ pipeline.
Các thành phần chính trong file:
- Hàm `seed_everything()`: Đảm bảo tính tái lập (reproducibility).
- Hàm `calculate_metrics()`: Tính toán Accuracy, Recall, F1-score từ `sklearn.metrics`.
- Class `Trainer`: Bao bọc logic huấn luyện, lưu checkpoint và logging.
- Hàm `main()`: Entry point thực thi file.

## 2. Chi tiết các module được sử dụng từ các file khác
Để đảm bảo rõ ràng, script `src/train1.py` sẽ import trực tiếp các module/class từ các file có sẵn trong project:

1. **Từ `configs/config.py`**:
   - `load_config`: Hàm tải cấu hình từ file `.yaml` hoặc `.json`.
   - `TrainConfig`: Dataclass chứa các tham số cấu hình hệ thống huấn luyện.

2. **Từ `src/models1.py`**:
   - `ConvGRUClassifier`: Lớp mô hình Spatio-Temporal ConvGRU mới với cơ chế Spatial Reduction Neck. Sử dụng `ConvGRUClassifier.from_config(config)` để khởi tạo.

3. **Từ `src/dataset2.py`**:
   - `build_raw_video_dataloaders`: Hàm factory cung cấp `train_loader` và `val_loader` để tải các file video thô lên (Raw Video Frames).
   - `ChunkedBackboneNeckExtractor`: Mô hình trích xuất đặc trưng NMSFreeDetector qua cơ chế Mini-Chunk (tránh OOM trên GPU). Được sử dụng ngay bên trong vòng lặp training để chuyển tensor `frames_t` thành `(p3, p4, p5)`.

## 3. Cấu hình TensorBoard Logging độc lập
- Khởi tạo đối tượng `SummaryWriter(log_dir)` từ `torch.utils.tensorboard`.
- **Ghi theo Step (Global Step):**
  - Trong vòng lặp train batch: Ở mỗi khoảng `log_step_interval` batch, tính metrics và dùng `writer.add_scalar('Train_Step/Loss', loss, global_step)`. Tương tự với Acc, Recall, F1.
- **Ghi theo Epoch (Global Epoch):**
  - Tại cuối mỗi epoch, tính giá trị trung bình toàn epoch cho Train và Validation.
  - Sử dụng `writer.add_scalar('Train_Epoch/Loss', avg_train_loss, epoch)`. Tương tự với Validation_Epoch.

## 4. Vòng lặp Huấn luyện (Training Loop) & Kiểm định
- Nạp data: `frames_t, label_t, seq_len_t, _ = batch` từ dataloader.
- Trích xuất đặc trưng Backbone (chặn gradient): `p3, p4, p5 = extractor(frames_t.to(device))`
- Forward pass ConvGRU: `logits = model((p3, p4, p5), seq_lens=seq_len_t.to(device))`
- Tính Loss (CrossEntropyLoss) và Backward pass.
- Ghi log TensorBoard tại từng step/epoch. Đánh giá Validation Loop cuối mỗi epoch.
- Quản lý Checkpoints: Lưu `best_model.pt` và `last.pt`.

## 5. Các bước thực hiện tiếp theo (sau khi xác nhận)
- **Bước 3**: Tiến hành viết code Python cho file `src/train1.py` dựa trên kế hoạch này.
- **Bước 4 & 5**: Hoàn thiện và tạo báo cáo report.

Xin người dùng xác nhận kế hoạch đã bổ sung rõ các module để tiến hành bước code.
