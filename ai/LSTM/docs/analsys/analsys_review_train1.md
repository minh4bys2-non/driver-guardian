# Báo cáo Phân tích & Kiểm thử Lỗ hổng Huấn luyện: `src/train1.py`

**Mã tài liệu:** `analsys_review_train1.md`  
**Đối tượng kiểm tra:** [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py)  
**Tiêu chuẩn đánh giá:** Code Review & Quality Gates (5 trục: *Correctness, Architecture, Readability, Security, Performance*).

---

## 1. Bối cảnh & Mục tiêu Kiểm tra (Context & Scope)
File [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py) đảm nhiệm vai trò **Pipeline Huấn luyện Chính** cho mô hình không gian - thời gian `ConvGRUClassifier` ([`src/models1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models1.py)), trích xuất đặc trưng on-the-fly từ video thô thông qua `ChunkedBackboneNeckExtractor` ([`src/dataset2.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/dataset2.py)).

Mục tiêu kiểm tra: Rà soát toàn bộ các **lỗ hổng tiềm ẩn (bugs, silent failures, memory leaks, data corruption, training pitfalls)** có thể làm sập tiến trình huấn luyện, làm sai lệch số học/gradient hoặc gây suy thoái hiệu năng khi chạy thực tế trên máy tính người dùng.

---

## 2. Bảng Tổng hợp Phân loại Lỗ hổng (Findings Summary)

| Mức độ | Số lượng | Vấn đề trọng tâm |
|---|:---:|---|
| **Critical** | **4** | Lỗi số học BCE Loss & Argmax, mất checkpoint khi `val_interval > 1`, thiếu OOM handling trong Validation, lệch tỷ trọng gradient accumulation. |
| **Required** | **3** | Mất trạng thái AMP `GradScaler` khi resume, nguy cơ Data Leakage tiềm tàng / rỗng tập mẫu, không tương thích `supervision_mode="sequence"`. |
| **Consider / Optional** | **3** | Bùng nổ VRAM khi `seq_len: Null`, overhead chuyển kiểu FP16-FP32-FP16, chưa đồng bộ các cờ config (Early Stopping, alias resume, `val_batch_size`). |
| **Nit** | **1** | Chuẩn hóa `device_type` trong `autocast` và an toàn OpenCV đa tiến trình trên Windows. |

---

## 3. Chi tiết Phân tích Lỗ hổng theo 5 Trục Đánh giá

### Trục 1: Tính Đúng đắn & Huấn luyện (Correctness & Training Integrity)

#### 🔴 Lỗ hổng 1.1 [Critical]: Xung đột số học giữa Binary Cross-Entropy (BCE) Loss và Argmax
- **Vị trí code:** Dòng 90–93, 156–160, 186, 239–243, 247.
- **Hiện tượng:**
  ```python
  if self.cfg.loss_type.lower() == "bce":
      # Tính loss chỉ dựa trên logits[:, 1]
      loss = self.criterion(logits[:, 1].unsqueeze(1), labels.float().unsqueeze(1))
  ...
  # Dự đoán nhãn lại so sánh argmax giữa 2 kênh
  preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
  ```
- **Hậu quả nghiêm trọng:**
  1. Mô hình `ConvGRUClassifier` có đầu ra 2 kênh (`num_classes=2`). Khi dùng BCE, hàm mất mát chỉ truyền gradient qua kênh 1 (`logits[:, 1]`). Kênh 0 (`logits[:, 0]`) nhận **gradient = 0**, giữ nguyên các giá trị ngẫu nhiên khởi tạo ban đầu!
  2. Khi tính nhãn dự đoán `torch.argmax(logits, dim=1)`, mô hình so sánh kênh 1 (được train) với kênh 0 (nhiễu ngẫu nhiên chưa hề được train). Điều này dẫn đến **suy thoái toán học âm thầm (silent failure)**, dự đoán sai hoàn toàn và chỉ số F1/Accuracy không phản ánh năng lực thực tế.
  3. Nếu người dùng chỉnh `num_classes = 1` trong config theo đúng chuẩn Binary Classification: dòng code `logits[:, 1]` sẽ lập tức ném lỗi **`IndexError: index 1 is out of bounds for dimension 1 with size 1`** làm sập toàn bộ script.
- **Biện pháp khắc phục (Remedy):**
  - Nếu cấu hình `loss_type == "bce"`: đầu ra mô hình phải là 1 kênh (`num_classes=1`), loss tính bằng `BCEWithLogitsLoss(logits.view(-1), labels.float())`, và nhãn dự đoán tính bằng ngưỡng xác suất Sigmoid: `preds = (torch.sigmoid(logits.view(-1)) > 0.5).long()`.
  - Nếu `num_classes == 2`: bắt buộc chuẩn hóa về `CrossEntropyLoss(logits, labels)` và dùng `argmax(dim=1)`.

---

#### 🔴 Lỗ hổng 1.2 [Critical]: Sai lệch Tỷ trọng Gradient tích lũy (Gradient Accumulation Bias) ở Cuối Epoch
- **Vị trí code:** Dòng 163–179.
- **Hiện tượng:**
  ```python
  loss = loss / accum_steps
  ...
  if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
      # Cập nhật optimizer
  ```
- **Hậu quả:** 
  - Tại batch cuối cùng của epoch, nếu số lượng batch tích lũy thực tế $k < \text{accum\_steps}$ (ví dụ còn 1 batch lẻ trong khi `accum_steps = 4`), loss của batch này vẫn bị chia cho 4 thay vì chia cho 1.
  - Hậu quả là gradient của nhóm dữ liệu cuối cùng bị thu nhỏ bất thường ($\times \frac{1}{4}$ thay vì đúng chuẩn trung bình), làm méo mó bước tối ưu hóa ở cuối mỗi epoch.
- **Biện pháp khắc phục (Remedy):**
  - Đếm số lượng micro-batches thực tế trong chu kỳ tích lũy hiện tại và chuẩn hóa loss chính xác theo số bước đó.

---

#### 🟠 Lỗ hổng 1.3 [Required]: Bỏ quên trạng thái của `GradScaler` khi lưu và khôi phục Checkpoint (AMP Resume)
- **Vị trí code:** Dòng 257–273, 274–285.
- **Hiện tượng:**
  - `save_checkpoint` chỉ lưu `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`.
  - Hoàn toàn **không lưu** `self.scaler.state_dict()`.
- **Hậu quả:** 
  - Khi resume training với AMP (`--resume`), `GradScaler` bị khởi tạo lại với scale factor mặc định (65536.0). 
  - Nếu mô hình đã huấn luyện qua nhiều epoch với dải gradient nhỏ, việc reset scale factor sẽ gây tràn số số học (Inf/NaN gradient) trong các batch đầu tiên sau resume, khiến optimizer bỏ qua hàng loạt bước cập nhật trọng số.
- **Biện pháp khắc phục (Remedy):**
  - Thêm `state["scaler_state_dict"] = self.scaler.state_dict()` khi lưu và nạp lại trong `load_checkpoint`.

---

#### 🟠 Lỗ hổng 1.4 [Required]: Không tương thích khi cấu hình `supervision_mode = "sequence"`
- **Vị trí code:** Dòng 153–160, 237–243.
- **Hiện tượng:**
  - Trong `models1.py`, nếu `supervision_mode` là `"sequence"` hoặc `"frame"`, mô hình trả về tensor logits dạng 3D: `[B, T, num_classes]`.
  - Nhưng DataLoader trong `dataset2.py` trả về `labels` chỉ có dạng 1D: `[B]`.
  - Khi chạy qua `CrossEntropyLoss(logits, labels)`, PyTorch sẽ ném ngoại lệ kích thước tensor không khớp (`ValueError: Expected target size [B, C], got [B]`).
- **Biện pháp khắc phục (Remedy):**
  - Thêm assertion kiểm tra `supervision_mode == "attention_pooling"` trong `train1.py`, hoặc bổ sung logic chuyển đổi shape nhãn nếu hỗ trợ giám sát từng khung hình (frame-level).

---

### Trục 2: Kiến trúc & Tính Ổn định Tiến trình (Architecture & Operational Reliability)

#### 🔴 Lỗ hổng 2.1 [Critical]: Nguy cơ Mất Checkpoint toàn diện khi `val_interval_epochs > 1`
- **Vị trí code:** Dòng 314–330.
- **Hiện tượng:**
  ```python
  if epoch % self.cfg.val_interval_epochs == 0:
      val_metrics = self.validate_epoch(epoch)
      ...
      if val_metrics["f1"] > self.best_val_f1:
          self.save_checkpoint(epoch, is_best=True)
      else:
          self.save_checkpoint(epoch, is_best=False)
  ```
- **Hậu quả nghiêm trọng:**
  - Lệnh gọi `save_checkpoint` nằm **bên trong** điều kiện kiểm tra validation.
  - Nếu người dùng đặt `val_interval_epochs = 5`, thì tại các epoch 1, 2, 3, 4: **không có bất kỳ checkpoint nào được lưu** (kể cả file `last.pt` hay `epoch_X.pt`).
  - Nếu xảy ra sự cố sập nguồn, quá nhiệt GPU hoặc lỗi hệ thống ở epoch 4, toàn bộ công sức huấn luyện của 4 epoch sẽ mất trắng.
- **Biện pháp khắc phục (Remedy):**
  - Đưa việc lưu `last.pt` và checkpoint định kỳ ra ngoài phạm vi kiểm tra validation (chạy cuối mỗi epoch), chỉ việc cập nhật `best.pt` mới phụ thuộc vào validation.

---

#### 🔴 Lỗ hổng 2.2 [Critical]: Thiếu cơ chế Bắt lỗi OOM trong Validation Loop & Dò rỉ Tham chiếu Exception
- **Vị trí code:** Dòng 202–210 (Train loop) so với Dòng 219–255 (Val loop).
- **Hiện tượng:**
  - Vòng lặp `validate_epoch` hoàn toàn không có khối `try...except RuntimeError` bắt CUDA OOM. Nếu tập validation xuất hiện video có độ phân giải hoặc độ dài lớn, toàn bộ tiến trình train nhiều ngày sẽ bị hủy hoại.
  - Trong `train_epoch`: khi bắt `except RuntimeError as e:`, biến `e` lưu giữ traceback chứa toàn bộ các tensor GPU trung gian (`frames_t`, `p3, p4, p5`, `logits`, `loss`). Lệnh gọi `torch.cuda.empty_cache()` không thể giải phóng bộ nhớ nếu các biến cục bộ chưa được xóa (`del`), dẫn đến OOM lặp lại ngay batch tiếp theo.
- **Biện pháp khắc phục (Remedy):**
  - Bọc khối `try...except` OOM đồng bộ cho cả `validate_epoch`.
  - Thực hiện thu hồi bộ nhớ chuẩn xác: `del frames_t, labels, seq_lens; torch.cuda.empty_cache()`.

---

#### 🟠 Lỗ hổng 2.3 [Required]: Nguy cơ Data Leakage tiềm tàng & Crash khi Tập Mẫu Rỗng
- **Vị trí code:** `src/dataset2.py` (Dòng 298–319) và `src/train1.py` (Dòng 213–214, 252–254).
- **Hiện tượng:**
  1. Trong `RawVideoFramesDataset`, nếu thư mục không có cấu trúc `train/` và `val/` riêng biệt và không có manifest, logic quét thư mục fallback về quét toàn bộ thư mục gốc, khiến tập Train và tập Val sử dụng chung 100% video giống hệt nhau (Data Leakage nghiêm trọng).
  2. Nếu tập dữ liệu hoặc tập val bị rỗng (`len(all_preds) == 0`), hàm `calculate_metrics(np.array([]), np.array([]))` sẽ ném lỗi `ValueError: Found array with 0 sample(s)` làm sập script.
  3. Công thức tính `avg_loss = total_loss / max(len(self.train_loader), 1)` sẽ bị sai (hạ thấp loss giả tạo) nếu có bất kỳ batch nào bị skip (do lỗi video hoặc do OOM).
- **Biện pháp khắc phục (Remedy):**
  - Thêm cảnh báo/kiểm tra số lượng mẫu và tính độc lập của `train_loader.dataset` và `val_loader.dataset`.
  - Tính loss trung bình dựa trên số batch thực tế đã nạp thành công (`num_valid_batches`).
  - Xử lý mảng rỗng trước khi gọi `calculate_metrics`.

---

### Trục 3: Hiệu năng & Tối ưu Tài nguyên (Performance & Resource Management)

#### 🟡 Lỗ hổng 3.1 [Consider]: Bùng nổ VRAM cực đại khi `seq_len: Null`
- **Vị trí code:** `configs/config.yaml` (Dòng 20), `src/dataset2.py` (Dòng 456–462), `src/train1.py` (Dòng 143).
- **Hiện tượng:**
  - Trong `config.yaml`, cấu hình `seq_len: Null`.
  - Khi đó, các video được đọc nguyên vẹn độ dài. Tại hàm `collate_video_frames`, toàn bộ batch được zero-pad lên $T_{\max}$ của video dài nhất trong batch.
  - Nếu xuất hiện 1 video dài 60s (300 frames) trong batch 8 mẫu:
    - Kích thước tensor khung hình thô trên GPU: $8 \times 300 \times 3 \times 640 \times 640 \times 1 \text{ byte} \approx 2.95 \text{ GB}$.
    - Kích thước đặc trưng $(p3, p4, p5)$ sinh ra: $\approx 6.88 \text{ GB}$.
    - Tổng VRAM vượt quá 9.8 GB $\rightarrow$ Tràn bộ nhớ (OOM) lập tức trên các GPU phổ thông (RTX 3050 4GB/6GB, RTX 3060 6GB/8GB).
- **Biện pháp khắc phục (Remedy):**
  - Luôn khuyến nghị đặt ngưỡng trần an toàn `max_seq_len` (ví dụ 30 hoặc 50 khung hình) hoặc cảnh báo rõ ràng trong tài liệu cấu hình.

---

#### 🟡 Lỗ hổng 3.2 [Consider]: Chu kỳ Chuyển đổi Kiểu Dữ liệu Gây lãng phí VRAM (FP16 $\rightarrow$ FP32 $\rightarrow$ FP16)
- **Vị trí code:** `src/dataset2.py` (Dòng 232–234) & `src/train1.py` (Dòng 148–150).
- **Hiện tượng:**
  - `ChunkedBackboneNeckExtractor` chạy bằng FP16 nhưng lại ép kiểu đầu ra `.float()` (FP32).
  - Sang `train1.py`, các tensor FP32 này lại đi vào `torch.amp.autocast("cuda")` và bị ép ngược lại FP16.
  - Việc lưu trữ đặc trưng $(p3, p4, p5)$ của toàn bộ chuỗi ở FP32 làm tăng gấp đôi dung lượng VRAM trung gian trên GPU (~600MB - 1.2GB VRAM lãng phí).
- **Biện pháp khắc phục (Remedy):**
  - Cho phép giữ đặc trưng ở kiểu FP16 trực tiếp từ extractor khi AMP được kích hoạt.

---

### Trục 4: Tính Khả dụng & Đồng bộ Cấu hình (Config Sync & Code Quality)

#### 🟡 Lỗ hổng 4.1 [Consider]: Bỏ sót Tham số Cấu hình Quan trọng
- **Vị trí code:** Dòng 344–394.
- **Chi tiết:**
  - `TrainConfig` có định nghĩa `early_stopping`, `patience`, `val_batch_size`, `val_num_workers`, `history_csv_path`, nhưng `train1.py` **không sử dụng** các tham số này.
  - `TrainConfig.resume` cho phép truyền bí danh `'last'` hoặc `'best'`, nhưng `load_checkpoint` trong `train1.py` chỉ kiểm tra `os.path.exists()`, dẫn đến lỗi không tìm thấy file nếu người dùng dùng bí danh.
- **Biện pháp khắc phục (Remedy):**
  - Thêm logic phân giải bí danh `'last'` $\rightarrow$ `checkpoint_dir/last.pt`, `'best'` $\rightarrow$ `checkpoint_dir/best.pt`.
  - Tích hợp ghi nhận file `training_history.csv` định kỳ sau mỗi epoch để người dùng dễ theo dõi tiến độ ngoài TensorBoard.

---

#### ⚪ Lỗ hổng 4.2 [Nit]: Chuẩn hóa Device Type và OpenCV Multi-processing trên Windows
- **Vị trí code:** Dòng 148, 235 trong `train1.py` và `_seed_worker` trong `dataset2.py`.
- **Chi tiết:**
  - Nên sử dụng `device_type=self.device.type` trong `torch.amp.autocast` để tránh xung đột khi chạy trên máy không có CUDA.
  - Thêm `cv2.setNumThreads(0)` trong hàm khởi tạo worker của DataLoader để tránh xung đột thread pool đa tiến trình trên môi trường Windows.

---

## 4. Kế hoạch Đề xuất Tiếp theo (Next Steps)

Sau khi Người dùng duyệt báo cáo phân tích này, Agent sẽ tiến hành Bước 2:
1. Tạo tài liệu kế hoạch chi tiết: `docs/plan/plan_review_train1.md`.
2. Lập danh sách từng khối code cụ thể cần tái cấu trúc và sửa chữa trong `src/train1.py`.
3. Thiết lập bộ kiểm thử tự động (Unit Test / Smoke Test) để kiểm chứng việc khắc phục triệt để các lỗ hổng đã nêu.
