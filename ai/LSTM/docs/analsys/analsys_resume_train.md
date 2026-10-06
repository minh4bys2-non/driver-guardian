# Báo cáo Phân tích Yêu cầu: Bổ sung Cơ chế Tiếp tục Huấn luyện (Resume Training) từ một Epoch Chỉ định

**Mã tài liệu:** `analsys_resume_train.md`  
**Dự án:** Driver Guardian AI — SpatioTemporal ConvGRU Training Pipeline  
**Mục tiêu áp dụng:** [`notebooks/02_train_convgru_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/02_train_convgru_kaggle.ipynb) (và đồng bộ chuẩn hóa cùng [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py))  
**Quy chuẩn áp dụng:** [AGENTS.md](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md) — Bước 1: Khảo sát & Phân tích (Discovery)

---

## 1. Bối cảnh & Mục tiêu (Context & Objectives)

### 1.1. Bối cảnh
Khi huấn luyện các mô hình Video Deep Learning (như `ConvGRUClassifier` kết hợp trích xuất đặc trưng GPU từ Backbone PAFPN), thời gian chạy thường kéo dài nhiều giờ hoặc nhiều ngày:
- Trên môi trường **Kaggle**, một phiên làm việc (session) giới hạn tối đa **12 giờ liên tục** hoặc có thể bị gián đoạn bất ngờ do timeout, mất kết nối mạng hoặc người dùng muốn chia nhỏ quá trình huấn luyện thành nhiều đợt.
- Nếu không có cơ chế khôi phục (Resume) từ một epoch cụ thể, người dùng buộc phải huấn luyện lại từ đầu ($Epoch = 1$), gây lãng phí lớn tài nguyên tính toán GPU và thời gian.

### 1.2. Mục tiêu kỹ thuật
Bổ sung cơ chế **Resume Training** toàn diện và linh hoạt, cho phép:
1. Tiếp tục huấn luyện từ một epoch bất kỳ (ví dụ: epoch 5, epoch 10) hoặc từ các bí danh chuẩn (`"last"`, `"best"`).
2. Khôi phục hoàn toàn và bảo toàn tính toàn vẹn trạng thái của:
   - Trọng số mô hình (`model_state_dict`).
   - Trạng thái tối ưu hóa (`optimizer_state_dict`).
   - Lịch trình học (`scheduler_state_dict` Cosine Annealing).
   - Bộ chia tỷ lệ FP16 Mixed Precision (`scaler_state_dict` của AMP GradScaler).
   - Điểm số kiểm định kỷ lục (`best_val_f1`) và chỉ số epoch bắt đầu (`start_epoch`).
3. Bảo toàn tệp nhật ký `training_history.csv` (không ghi đè xóa trắng lịch sử cũ khi resume).
4. Lưu định kỳ các checkpoint theo từng epoch (`epoch_1.pt`, `epoch_2.pt`,...) kèm cấu hình dọn dẹp hoặc giới hạn số lượng checkpoint để không làm tràn ổ đĩa `/kaggle/working` (giới hạn 20GB).

---

## 2. Khảo sát Hiện trạng Code trong Notebook & Script

### 2.1. Trong notebook hiện tại ([`notebooks/02_train_convgru_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/02_train_convgru_kaggle.ipynb))
- Phương thức `save_checkpoint` hiện tại chỉ lưu 2 file: `last.pt` và `best.pt`.
- Chưa lưu các file checkpoint theo từng epoch cụ thể (`epoch_{N}.pt`).
- Chưa có hàm `load_checkpoint()` để nạp lại trạng thái.
- Vòng lặp `train()` luôn cố định chạy: `for epoch in range(1, self.cfg.epochs + 1):`.
- Phương thức khởi tạo `_init_history_csv()` mở file với mode `"w"` (ghi đè), sẽ xóa sạch nhật ký các epoch trước khi chạy lại.

### 2.2. Trong script gốc ([`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py))
- Đã có phương thức `load_checkpoint` và `_resolve_checkpoint_path` hỗ trợ phân giải alias `'last'`, `'best'`.
- Lưu checkpoint nguyên tử (`_atomic_save`) chống hỏng file khi bị ngắt điện/tiến trình.
- Đã hỗ trợ khôi phục trạng thái `scaler_state_dict` cho PyTorch AMP.

---

## 3. Phân tích Các Thành phần Cần Nâng cấp Chi tiết

### 3.1. Cấu hình Siêu tham số (`KaggleTrainConfig`)
Cần bổ sung các trường cấu hình trực quan trong Section 2:
```python
@dataclass
class KaggleTrainConfig:
    ...
    # Cấu hình Khôi phục Huấn luyện (Resume)
    enable_resume: bool = False             # Bật/tắt chế độ tiếp tục huấn luyện
    resume: Optional[Union[str, int]] = None # Đường dẫn file .pt, số nguyên epoch (VD: 5), hoặc bí danh ('last', 'best')
    resume_epoch: Optional[int] = None       # Tùy chọn ép buộc số thứ tự epoch bắt đầu
    
    # Cấu hình Lưu trữ Checkpoint
    save_epoch_interval: int = 1            # Khoảng cách lưu checkpoint epoch_N.pt (1: lưu mỗi epoch)
    save_best_only: bool = False            # Nếu True: chỉ lưu best.pt và last.pt để tiết kiệm đĩa
    max_keep_ckpts: int = 5                 # Tự động giữ lại N checkpoint gần nhất tránh đầy 20GB disk Kaggle
```

### 3.2. Lưu Checkpoint Đa dạng & Tiết kiệm Dung lượng (`save_checkpoint`)
- Lưu `state` chứa đầy đủ:
  ```python
  state = {
      "epoch": epoch,
      "model_state_dict": self.model.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
      "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
      "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
      "best_val_f1": self.best_val_f1,
      "val_f1": val_f1,
  }
  ```
- Luôn lưu `last.pt` và `best.pt`.
- Lưu `epoch_{epoch}.pt` nếu `epoch % self.cfg.save_epoch_interval == 0`.
- Cơ chế dọn dẹp (Pruning): Nếu tổng số checkpoint `epoch_*.pt` vượt quá `max_keep_ckpts`, tự động xóa các epoch cũ hơn (trừ `best.pt` và `last.pt`) để bảo vệ bộ nhớ đĩa `/kaggle/working`.

### 3.3. Bộ Phân giải Đường dẫn & Nạp Checkpoint (`load_checkpoint`)
- Hỗ trợ các định dạng linh hoạt:
  - **Số nguyên hoặc chuỗi số** (`resume=5` hoặc `resume="5"`): Tự động tìm `epoch_5.pt` trong thư mục `checkpoint_dir` (hoặc trong `/kaggle/input/...`).
  - **Bí danh** (`resume="last"` hoặc `resume="best"`): Tự động trỏ về `last.pt` hoặc `best.pt`.
  - **Đường dẫn tệp cụ thể**: Nhận diện đường dẫn đầy đủ từ `/kaggle/input/...` hoặc thư mục làm việc.
- Nạp lại đầy đủ các tensor weights và scheduler/scaler, sau đó cập nhật:
  - `self.start_epoch = ckpt["epoch"] + 1` (nếu người dùng không chỉ định ghi đè `resume_epoch`).
  - `self.best_val_f1 = ckpt.get("best_val_f1", 0.0)`.

### 3.4. Xử lý An toàn Nhật ký Huấn luyện (`training_history.csv`)
- Khi `enable_resume == True` và file CSV đã tồn tại:
  - Đọc các dòng lịch sử cũ, chỉ giữ lại các bản ghi từ epoch $\le \text{ckpt\_epoch}$.
  - Mở file ở chế độ append (`"a"`) từ `start_epoch`, ngăn chặn việc ghi đè mất mát lịch sử các epoch trước đó.

### 3.5. Cập nhật Vòng lặp Huấn luyện (`train`)
- Khởi chạy vòng lặp từ `self.start_epoch` thay vì cố định `1`:
  ```python
  for epoch in range(self.start_epoch, self.cfg.epochs + 1):
  ```
- Hiển thị thông báo tiến trình rõ ràng:
  `[*] Đang tiếp tục huấn luyện từ Epoch X đến Epoch Y...`

---

## 4. Đề xuất Phương án Triển khai & Trải nghiệm Người dùng (UX)

Trong notebook [`notebooks/02_train_convgru_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/02_train_convgru_kaggle.ipynb):
1. **Tại Section 2**: Bổ sung đầy đủ các tham số resume vào `KaggleTrainConfig`.
2. **Tại Section 6**: Bổ sung các phương thức `_resolve_checkpoint_path()`, `load_checkpoint()`, logic dọn dẹp checkpoint xoay vòng và sửa logic khởi tạo CSV.
3. **Tại Section 7**: 
   - Thêm một cell cấu hình nhanh (Toggle Resume Cell) ngay trước khi gọi `trainer.train()`.
   - Cung cấp ví dụ trực quan bằng code được comment mẫu:
     ```python
     # Ví dụ 1: Tiếp tục từ epoch 10
     # trainer.load_checkpoint(10)
     # Ví dụ 2: Tiếp tục từ checkpoint gần nhất
     # trainer.load_checkpoint("last")
     # Ví dụ 3: Tiếp tục từ file nạp qua Kaggle Dataset
     # trainer.load_checkpoint("/kaggle/input/my-checkpoints/best.pt")
     ```

---

## 5. Kết luận & Hành động Tiếp theo

Việc bổ sung cơ chế tiếp tục huấn luyện từ một epoch cụ thể là tính năng cực kỳ quan trọng cho người dùng chạy trên nền tảng đám mây Kaggle, giúp đảm bảo an toàn dữ liệu, tiết kiệm chi phí và tăng tính chủ động trong thực nghiệm.

👉 **Kính mời bạn duyệt nội dung phân tích (`analsys_resume_train.md`). Khi bạn chấp thuận, tôi sẽ tiếp tục thực hiện Bước 2: Tạo kế hoạch thực hiện (`plan_resume_train.md`).**
