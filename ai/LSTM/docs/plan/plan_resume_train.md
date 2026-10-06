# Kế hoạch Thực hiện: Tích hợp Cơ chế Tiếp tục Huấn luyện (Resume Training) từ một Epoch Chỉ định

**Mã tài liệu:** `plan_resume_train.md`  
**Dự án:** Driver Guardian AI — SpatioTemporal ConvGRU Training Pipeline  
**Tệp mục tiêu:** [`notebooks/02_train_convgru_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/02_train_convgru_kaggle.ipynb)  
**Quy chuẩn áp dụng:** [AGENTS.md](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md) — Bước 2: Lên kế hoạch thực hiện

---

## 1. Mục tiêu Kế hoạch

Triển khai chi tiết các thay đổi trong tệp notebook [`notebooks/02_train_convgru_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/02_train_convgru_kaggle.ipynb) để cung cấp khả năng tiếp tục huấn luyện (Resume) mạnh mẽ, an toàn và dễ sử dụng cho người dùng trên nền tảng Kaggle.

---

## 2. Các Bước Triển khai Kỹ thuật Chi tiết

### Bước 2.1: Cập nhật Cấu hình Siêu tham số trong Section 2 (`KaggleTrainConfig`)
Bổ sung các trường quản lý checkpoint và resume vào class `KaggleTrainConfig`:
1. `enable_resume: bool = False`: Cờ điều khiển việc tự động nạp checkpoint khi khởi tạo.
2. `resume: Optional[Union[str, int]] = None`: Đường dẫn file checkpoint, số nguyên đại diện cho epoch (ví dụ: `5`), hoặc bí danh (`"last"`, `"best"`).
3. `resume_epoch: Optional[int] = None`: Tùy chọn ép buộc số epoch bắt đầu lại nếu muốn bỏ qua số epoch lưu trong metadata checkpoint.
4. `save_epoch_interval: int = 1`: Tần suất lưu các file checkpoint theo từng epoch (`epoch_{epoch}.pt`).
5. `max_keep_ckpts: int = 5`: Số lượng checkpoint epoch tối đa được lưu lại trên ổ đĩa. Khi vượt quá số lượng này, hệ thống tự động dọn dẹp các epoch cũ nhất để bảo vệ bộ nhớ đĩa Kaggle (/kaggle/working 20GB), nhưng luôn giữ nguyên `best.pt` và `last.pt`.

---

### Bước 2.2: Nâng cấp Lớp `KaggleTrainer` trong Section 6
Cải tiến các phương thức quản lý trạng thái, lưu và nạp checkpoint:

1. **Khởi tạo biến trạng thái**:
   - Thêm `self.start_epoch = 1` trong hàm `__init__`.
   - Nếu `self.cfg.enable_resume and self.cfg.resume`: tự động gọi `self.load_checkpoint(self.cfg.resume)`.

2. **Xây dựng Phương thức `_resolve_checkpoint_path(self, checkpoint_path: Union[str, Path, int]) -> Path`**:
   - Nếu truyền vào số nguyên hoặc chuỗi số (VD: `5` hoặc `"5"`): tìm tệp `epoch_5.pt` trong `self.ckpt_dir` hoặc tìm đệ quy trong thư mục dữ liệu đầu vào `/kaggle/input`.
   - Nếu là bí danh `"last"` hoặc `"best"`: phân giải về `self.ckpt_dir / "last.pt"` hoặc `self.ckpt_dir / "best.pt"`.
   - Nếu là đường dẫn tệp cụ thể (ví dụ `/kaggle/input/.../best.pt`): kiểm tra sự tồn tại của file.
   - Ném ngoại lệ `FileNotFoundError` rõ ràng nếu không tìm thấy tệp.

3. **Xây dựng Phương thức `load_checkpoint(self, checkpoint_path: Union[str, Path, int]) -> None`**:
   - Nạp checkpoint bằng `torch.load(target_path, map_location=self.device, weights_only=False)`.
   - Khôi phục `model_state_dict` cho `self.model`.
   - Khôi phục `optimizer_state_dict` cho `self.optimizer`.
   - Khôi phục `scheduler_state_dict` cho `self.scheduler` (CosineAnnealingLR).
   - Khôi phục `scaler_state_dict` cho `self.scaler` (bảo toàn thang đo độ dốc AMP FP16).
   - Khôi phục `self.best_val_f1 = ckpt.get("best_val_f1", 0.0)`.
   - Tính toán `self.start_epoch`:
     - Nếu `self.cfg.resume_epoch` được thiết lập: dùng `self.cfg.resume_epoch`.
     - Ngược lại: `self.start_epoch = ckpt.get("epoch", 0) + 1`.
   - In thông báo xác nhận: số epoch tiếp tục, learning rate hiện tại, điểm Best F1 đã lưu.

4. **Nâng cấp Phương thức `save_checkpoint(self, epoch: int, is_best: bool, val_f1: Optional[float])`**:
   - Lưu đầy đủ các từ điển trạng thái (`model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `best_val_f1`, `epoch`, `val_f1`).
   - Luôn lưu `last.pt`.
   - Lưu `best.pt` khi đạt kỷ lục F1.
   - Nếu `epoch % self.cfg.save_epoch_interval == 0`: lưu `epoch_{epoch}.pt`.
   - Cơ chế Pruning: Kiểm tra danh sách các file `epoch_*.pt` trong thư mục checkpoint; nếu số lượng vượt quá `max_keep_ckpts`, tự động xóa các file có số epoch nhỏ hơn để tránh đầy đĩa.

5. **Bảo toàn Tệp Nhật ký `_init_history_csv(self)` & Ghi nối**:
   - Nếu `self.history_csv_path` đã tồn tại và đang chạy resume: không xóa đè mà đọc các dòng cũ, chỉ ghi tiếp các epoch từ `self.start_epoch`.
   - Nếu là lần chạy đầu tiên: tạo mới và ghi tiêu đề cột (header).

6. **Cập nhật Vòng lặp `train(self)`**:
   - Đổi phạm vi epoch: `for epoch in range(self.start_epoch, self.cfg.epochs + 1):`.
   - Kiểm tra nếu `self.start_epoch > self.cfg.epochs`: thông báo mô hình đã hoàn thành đủ số epoch cần thiết.

---

### Bước 2.3: Bổ sung Giao diện Tương tác Tiếp tục Huấn luyện trong Section 7
Thêm 1 Code cell tương tác chuyên biệt ngay trước khi gọi `trainer.train()`:
- Cung cấp đoạn mã mẫu rõ ràng với các kịch bản thực tế:
  ```python
  # ============================================================================
  # CẤU HÌNH TIẾP TỤC HUẤN LUYỆN (RESUME CHECKPOINT SELECTION)
  # ============================================================================
  # Bỏ comment một trong các dòng bên dưới để tiếp tục từ checkpoint tương ứng:
  
  # Kịch bản 1: Tiếp tục từ epoch gần nhất vừa lưu trong phiên làm việc
  # trainer.load_checkpoint("last")
  
  # Kịch bản 2: Tiếp tục từ mô hình có F1 cao nhất
  # trainer.load_checkpoint("best")
  
  # Kịch bản 3: Tiếp tục từ một epoch cụ thể (ví dụ: Epoch 5)
  # trainer.load_checkpoint(5)
  
  # Kịch bản 4: Tiếp tục từ file checkpoint tải lên qua Kaggle Dataset
  # trainer.load_checkpoint("/kaggle/input/my-driver-guardian-weights/epoch_10.pt")
  ```

---

## 3. Tiêu chí Kiểm định & Đánh giá Chất lượng (Quality Criteria)

Sau khi cập nhật, notebook phải vượt qua các kiểm tra:
1. **Kiểm tra Cú pháp & Cấu trúc JSON**: Tệp notebook vẫn duy trì định dạng chuẩn JSON Jupyter v4, không có lỗi cú pháp.
2. **Kiểm tra Logic Phân giải Checkpoint**: Hàm `_resolve_checkpoint_path` giải quyết chính xác các trường hợp: số nguyên (ví dụ `5`), bí danh (`"last"`, `"best"`), và đường dẫn tệp trực tiếp.
3. **Kiểm tra Bảo toàn Trạng thái**: Sau khi gọi `load_checkpoint`, `start_epoch`, `best_val_f1`, và các tham số tối ưu hóa được khôi phục chính xác.
4. **Kiểm tra Giới hạn Đĩa Kaggle**: Cơ chế dọn dẹp (pruning) đảm bảo không tồn đọng quá `max_keep_ckpts` file checkpoint phụ.

---

## 4. Hành động Tiếp theo

👉 **Kính mời bạn duyệt kế hoạch triển khai trên (`plan_resume_train.md`). Khi bạn chấp thuận, tôi sẽ tiến hành Bước 3: Cập nhật tệp notebook `notebooks/02_train_convgru_kaggle.ipynb` và tạo tệp báo cáo tổng kết `docs/report/report_resume_train.md`.**
