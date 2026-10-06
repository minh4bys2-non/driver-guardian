# Báo cáo Hoàn thành: Tích hợp Cơ chế Tiếp tục Huấn luyện (Resume Training) từ một Epoch Chỉ định

**Mã tài liệu:** `report_resume_train.md`  
**Dự án:** Driver Guardian AI — SpatioTemporal ConvGRU Training Pipeline  
**Tệp mục tiêu:** [`notebooks/02_train_convgru_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/02_train_convgru_kaggle.ipynb)  
**Quy chuẩn áp dụng:** [AGENTS.md](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md) — Bước 3: Thực hiện kế hoạch

---

## 1. Tổng quan Kết quả Thực hiện

Đã hoàn thành toàn diện việc nâng cấp tệp Jupyter Notebook [`notebooks/02_train_convgru_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/02_train_convgru_kaggle.ipynb), bổ sung cơ chế **Resume Training** mạnh mẽ, an toàn và tối ưu cho môi trường đám mây Kaggle. 

Giờ đây, người dùng có thể:
1. Tiếp tục huấn luyện từ bất kỳ epoch nào đã lưu (ví dụ: `epoch_5.pt`, `epoch_10.pt`).
2. Khôi phục nhanh từ checkpoint gần nhất (`"last"`) hoặc checkpoint có điểm kiểm định cao nhất (`"best"`).
3. Đính kèm và nạp checkpoint từ một Kaggle Dataset bên ngoài (`/kaggle/input/...`).
4. Bảo toàn nguyên vẹn toàn bộ trạng thái học tập (weights, optimizer, scheduler, AMP scaler) và dữ liệu nhật ký lịch sử mà không lo bị ghi đè hay tràn bộ nhớ đĩa Kaggle.

---

## 2. Chi tiết các Nâng cấp Kỹ thuật Đã Triển khai

### 2.1. Nâng cấp Cấu hình Siêu tham số (Section 2 - `KaggleTrainConfig`)
Bổ sung các trường cấu hình trực quan:
- `enable_resume: bool = False`: Cho phép tự động khôi phục ngay khi khởi tạo Trainer.
- `resume: Optional[Union[str, int]] = None`: Nhận số nguyên epoch (ví dụ: `5`), bí danh (`'last'`, `'best'`), hoặc đường dẫn file `.pt`.
- `resume_epoch: Optional[int] = None`: Tùy chọn ép buộc số thứ tự epoch bắt đầu nếu muốn thay đổi so với số epoch lưu trong file checkpoint.
- `save_epoch_interval: int = 1`: Tần suất lưu checkpoint định kỳ theo từng epoch (`epoch_N.pt`).
- `max_keep_ckpts: int = 5`: Tự động dọn dẹp xoay vòng, chỉ duy trì 5 file checkpoint epoch gần nhất trên `/kaggle/working` để bảo vệ dung lượng đĩa (giới hạn 20GB của Kaggle).

### 2.2. Nâng cấp Lớp Huấn luyện (Section 6 - `KaggleTrainer`)
1. **Bộ Phân giải Đường dẫn Thông minh (`_resolve_checkpoint_path`)**:
   - Nếu truyền vào số nguyên hoặc chuỗi số (`5` hoặc `"5"`): Tự động tìm `epoch_5.pt` trong thư mục `checkpoint_dir` hoặc tìm kiếm đệ quy trong thư mục dữ liệu đầu vào `/kaggle/input`.
   - Nếu truyền bí danh (`"last"` hoặc `"best"`): Trỏ chính xác về `last.pt` hoặc `best.pt`.
   - Nếu truyền đường dẫn file: Kiểm tra sự tồn tại của tệp và ném ngoại lệ rõ ràng nếu không tìm thấy.
2. **Khôi phục Trọn vẹn Trạng thái (`load_checkpoint`)**:
   - Nạp lại `model_state_dict` cho mô hình.
   - Nạp lại `optimizer_state_dict` cho AdamW.
   - Nạp lại `scheduler_state_dict` cho CosineAnnealingLR.
   - Nạp lại `scaler_state_dict` cho PyTorch AMP `GradScaler` (tránh lỗi bùng nổ gradient khi resume ở độ chính xác hỗn hợp FP16).
   - Khôi phục `self.best_val_f1` và tự động cập nhật `self.start_epoch = ckpt["epoch"] + 1`.
   - In thông báo chi tiết: Epoch tiếp tục, Learning Rate hiện tại và Best Val F1.
3. **Lưu trữ Checkpoint Đa dạng & Dọn dẹp Xoay vòng (`save_checkpoint`)**:
   - Lưu đầy đủ từ điển trạng thái (bao gồm `scheduler_state_dict` và `scaler_state_dict`).
   - Luôn lưu `last.pt` và `best.pt`.
   - Lưu định kỳ `epoch_{epoch}.pt`.
   - Tự động gọi `_prune_old_checkpoints()` để xóa các checkpoint epoch cũ hơn khi vượt quá `max_keep_ckpts`, giúp ổ đĩa Kaggle không bị đầy.
4. **Bảo toàn Lịch sử Huấn luyện (`_init_history_csv`)**:
   - Khi tiếp tục chạy, file `training_history.csv` không bị xóa trắng mà tiếp tục ghi nối tiếp (append) từ `self.start_epoch`.
5. **Cập nhật Vòng lặp (`train`)**:
   - Chạy chính xác từ `self.start_epoch` đến `self.cfg.epochs`.
   - Thông báo rõ ràng: `BẮT ĐẦU HUẤN LUYỆN SPATIOTEMPORAL CONVGRU (TỪ EPOCH X ĐẾN Y)`.

### 2.3. Bổ sung Cell Giao diện Tương tác Tiếp tục Huấn luyện (Section 7 - Cell 2)
Ngay trước khi gọi `trainer.train()`, người dùng được cung cấp một cell cấu hình nhanh:
```python
# [Tùy chọn A] Tiếp tục từ checkpoint gần nhất (last.pt):
# trainer.load_checkpoint("last")

# [Tùy chọn B] Tiếp tục từ checkpoint có điểm F1 cao nhất (best.pt):
# trainer.load_checkpoint("best")

# [Tùy chọn C] Tiếp tục từ một epoch cụ thể đã lưu (ví dụ: Epoch 5 -> nạp epoch_5.pt):
# trainer.load_checkpoint(5)

# [Tùy chọn D] Tiếp tục từ tệp checkpoint đính kèm từ Kaggle Dataset (Input):
# trainer.load_checkpoint("/kaggle/input/my-driver-guardian-weights/epoch_10.pt")
```

---

## 3. Kết quả Kiểm thử & Chất lượng

1. **Kiểm tra Cú pháp Toàn diện**: Toàn bộ 25 cells (16 code cells) của notebook [`notebooks/02_train_convgru_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/02_train_convgru_kaggle.ipynb) được biên dịch (`compile`) thành công mà không phát sinh bất kỳ lỗi cú pháp nào.
2. **Kiểm tra Chuẩn JSON Notebook**: Tệp `.ipynb` tuân thủ nghiêm ngặt chuẩn Jupyter Notebook format v4 (`nbformat=4, nbformat_minor=5`).
3. **Kiểm tra Tính Độc lập (Self-Contained)**: Toàn bộ cơ chế resume, pruning và phân giải checkpoint đều được nhúng trực tiếp trong notebook, không phụ thuộc vào script ngoài.

---

## 4. Hướng dẫn Người dùng Sử dụng Tính năng Resume trên Kaggle

Khi phiên làm việc trên Kaggle bị ngắt quãng hoặc bạn muốn train thêm các epoch tiếp theo:

1. **Trường hợp 1: Huấn luyện tiếp trong cùng một phiên (Session)**:
   - Trong Section 7, tại ô code thứ 2, bỏ dấu comment `#` ở dòng:
     ```python
     trainer.load_checkpoint("last") # hoặc trainer.load_checkpoint(5)
     ```
   - Chạy cell đó và chạy tiếp cell `trainer.train()`. Hệ thống sẽ tiếp tục chạy từ epoch tiếp theo.
2. **Trường hợp 2: Huấn luyện trong một phiên mới (Mở Notebook mới)**:
   - Đính kèm thư mục checkpoint của phiên trước dưới dạng Kaggle Dataset (Input).
   - Chỉ định đường dẫn tới file checkpoint:
     ```python
     trainer.load_checkpoint("/kaggle/input/<dataset-weights>/last.pt")
     ```
   - Điều chỉnh tổng số `cfg.epochs` trong Section 2 lên mốc mong muốn (ví dụ từ 30 lên 50) và chạy huấn luyện tiếp tục bình thường.
