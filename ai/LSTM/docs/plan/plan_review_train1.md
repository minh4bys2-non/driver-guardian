# Kế hoạch Thực hiện Khắc phục Lỗ hổng: `src/train1.py`

**Mã tài liệu:** `plan_review_train1.md`  
**Dựa trên phân tích:** [`docs/analsys/analsys_review_train1.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/analsys/analsys_review_train1.md)  
**Mục tiêu chính:** Tái cấu trúc và hoàn thiện mã nguồn [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py) nhằm khắc phục triệt để 4 lỗi Critical, 3 lỗi Required và tích hợp các khuyến nghị về hiệu năng, bộ nhớ VRAM và tính ổn định.

---

## 1. Mục tiêu Cốt lõi & Tiêu chuẩn Nghiệm thu

1. **Chuẩn hóa Số học Loss & Suy luận:** Loại bỏ hoàn toàn mâu thuẫn giữa BCE Loss và Argmax, đồng bộ số kênh đầu ra và cơ chế tính nhãn dự đoán.
2. **Bảo toàn Checkpoint Huấn luyện:** Đảm bảo `last.pt` luôn được lưu cuối mỗi epoch bất kể giá trị `val_interval_epochs`. Lưu và khôi phục đầy đủ trạng thái AMP `GradScaler`.
3. **Chống sập do OOM (VRAM Resilience):** Đồng bộ cơ chế bắt ngoại lệ OOM và giải phóng bộ nhớ triệt để cho cả vòng lặp Huấn luyện (Train) lẫn Kiểm định (Validation).
4. **Chuẩn hóa Tích lũy Gradient:** Tính toán hệ số chia loss chính xác cho các batch dư lẻ ở cuối epoch khi dùng Gradient Accumulation.
5. **Đồng bộ Tham số Cấu hình:** Hỗ trợ Early Stopping, nạp checkpoint theo bí danh (`"last"`, `"best"`), xuất file tổng hợp `training_history.csv`.
6. **Kiểm thử Tự động (Self-Verification):** Xây dựng bộ test script độc lập kiểm chứng 100% các tình huống biên đã được sửa.

---

## 2. Kế hoạch Thực hiện Chi tiết theo Từng Giai đoạn

### Giai đoạn 1: Chuẩn hóa Cơ chế Loss, Dự đoán & Tích lũy Gradient
- **Nhiệm vụ 1.1: Tái cấu trúc Logic Loss và Preds (BCE vs CrossEntropy)**
  - Tách bạch hàm tính toán nhãn dự đoán:
    - Nếu `loss_type == "bce"` và `num_classes == 1`:
      - Logits có shape `[B, 1]`.
      - Loss: `criterion(logits.view(-1), labels.float())`.
      - Nhãn dự đoán: `(torch.sigmoid(logits.view(-1)) > 0.5).long()`.
    - Nếu `loss_type == "ce"` hoặc `num_classes == 2`:
      - Logits có shape `[B, 2]`.
      - Loss: `criterion(logits, labels.long())`.
      - Nhãn dự đoán: `torch.argmax(logits, dim=1)`.
    - Nếu người dùng cấu hình `loss_type == "bce"` nhưng `num_classes == 2`: tự động hiển thị cảnh báo và chuyển đổi an toàn sang `CrossEntropyLoss` hoặc sử dụng binary logits thích hợp, ngăn chặn tình trạng kênh 0 không nhận gradient.
- **Nhiệm vụ 1.2: Ràng buộc `supervision_mode`**
  - Thêm assert / kiểm tra: nếu `supervision_mode == "sequence"`, làm phẳng trục thời gian `[B*T, num_classes]` và `labels` tương ứng nếu có nhãn frame-level; nếu không có, yêu cầu dùng `attention_pooling`.
- **Nhiệm vụ 1.3: Sửa lỗi Tích lũy Gradient (Gradient Accumulation)**
  - Xác định kích thước nhóm tích lũy thực tế $k$ cho từng bước:
    - Các bước thông thường: $k = \text{accum\_steps}$.
    - Bước cuối cùng của epoch: $k = (\text{total\_batches} \pmod{\text{accum\_steps}})$ nếu dư lẻ, ngược lại $k = \text{accum\_steps}$.
  - Chia loss chính xác cho $k$ tương ứng để không làm méo mó độ dốc gradient.

---

### Giai đoạn 2: Tái cấu trúc Hệ thống Checkpoint & Phân giải Bí danh
- **Nhiệm vụ 2.1: Tách rời Lưu `last.pt` khỏi Điều kiện Validation**
  - Đưa logic lưu `last.pt` và lưu epoch định kỳ (`save_all_epochs`, `save_ckpt_interval_epochs`) ra khỏi khối `if epoch % val_interval_epochs == 0:`.
  - Luôn cập nhật `last.pt` sau khi kết thúc epoch để bảo vệ toàn vẹn tiến trình khi gặp sự cố đột ngột.
  - Chỉ cập nhật `best.pt` khi validation diễn ra và chỉ số F1 đạt đỉnh mới.
- **Nhiệm vụ 2.2: Tích hợp Trạng thái `GradScaler` vào Checkpoint**
  - Trong `save_checkpoint`: bổ sung `if self.scaler is not None: state["scaler_state_dict"] = self.scaler.state_dict()`.
  - Trong `load_checkpoint`: bổ sung nạp `self.scaler.load_state_dict(ckpt["scaler_state_dict"])`.
- **Nhiệm vụ 2.3: Phân giải Bí danh Checkpoint (Checkpoint Alias Resolver)**
  - Cho phép người dùng truyền `--resume last`, `--resume best` hoặc cấu hình `config.resume: "last"` / `"best"`.
  - Tự động map sang đường dẫn tệp thực tế: `<checkpoint_dir>/<experiment_name>/last.pt` hoặc `best.pt`.

---

### Giai đoạn 3: Gia cố Vòng lặp Validation, Bắt lỗi OOM & Chống Crash Dữ liệu Rỗng
- **Nhiệm vụ 3.1: Bổ sung Bắt lỗi OOM cho `validate_epoch`**
  - Bao bọc toàn bộ khối xử lý batch trong `validate_epoch` bằng `try...except RuntimeError as e:`.
  - Nếu gặp OOM: in cảnh báo, giải phóng bộ nhớ và tiếp tục batch tiếp theo thay vì làm sập toàn bộ script.
- **Nhiệm vụ 3.2: Dọn dẹp Tham chiếu VRAM Triệt để khi OOM**
  - Trong cả `train_epoch` và `validate_epoch`: khi bắt được lỗi OOM, thực hiện xóa biến tạm (`del frames_t, labels, seq_lens, p3, p4, p5, logits, loss`), giải phóng traceback `del e`, sau đó mới gọi `torch.cuda.empty_cache()`.
- **Nhiệm vụ 3.3: Xử lý Mảng Rỗng & Tính Trung bình Loss Chuẩn xác**
  - Đếm số lượng batch hợp lệ thực tế đã xử lý (`valid_batches`). Tính `avg_loss = total_loss / max(valid_batches, 1)`.
  - Trong `calculate_metrics`: kiểm tra nếu mảng nhãn/dự đoán có độ dài bằng 0, trả về `{"acc": 0.0, "recall": 0.0, "f1": 0.0}` thay vì gọi `sklearn.metrics` gây crash.

---

### Giai đoạn 4: Đồng bộ Tham số Cấu hình & Giám sát Tiến trình
- **Nhiệm vụ 4.1: Tích hợp Early Stopping**
  - Xây dựng lớp / logic `EarlyStopping`:
    - Theo dõi metric được cấu hình (`config.monitor_metric`, mặc định `val_f1`).
    - Đếm `patience` khi không có cải thiện vượt quá `min_delta`.
    - Tự động dừng sớm và thông báo epoch đạt đỉnh tốt nhất.
- **Nhiệm vụ 4.2: Tích hợp Ghi nhận Lịch sử Huấn luyện CSV**
  - Tạo file `training_history.csv` tại thư mục chỉ định trong config (`config.history_csv_path`).
  - Ghi nhận định kỳ sau mỗi epoch: `epoch, train_loss, train_acc, train_recall, train_f1, val_loss, val_acc, val_recall, val_f1, lr, elapsed_time_s`.
- **Nhiệm vụ 4.3: Hỗ trợ `val_batch_size` và `val_num_workers` riêng biệt**
  - Cập nhật hàm tạo DataLoader để sử dụng đúng `val_batch_size` và `val_num_workers` từ cấu hình.
- **Nhiệm vụ 4.4: Tối ưu OpenCV Đa tiến trình trên Windows**
  - Bổ sung `cv2.setNumThreads(0)` trong `_seed_worker` để ngăn ngừa xung đột thread pool CPU khi `num_workers > 0`.

---

### Giai đoạn 5: Kiểm thử Toàn diện & Báo cáo
- **Nhiệm vụ 5.1: Xây dựng Script Kiểm thử Giả lập (Mock Unit Test Suite)**
  - Viết script test độc lập kiểm tra tất cả các trường hợp:
    1. Kiểm thử tính đúng đắn của BCE và CrossEntropy (loss, backprop, argmax/sigmoid).
    2. Kiểm thử Gradient Accumulation với số batch dư lẻ.
    3. Kiểm thử Save & Resume Checkpoint (kèm `GradScaler` và alias `last`/`best`).
    4. Kiểm thử lưu `last.pt` khi `val_interval_epochs = 3`.
    5. Kiểm thử cơ chế phục hồi khi gặp OOM trong cả Train và Val.
    6. Kiểm thử Early Stopping dừng đúng hạn.
- **Nhiệm vụ 5.2: Tạo Báo cáo Kết quả Hoàn thành**
  - Tạo file `docs/report/report_review_train1.md` ghi nhận toàn bộ các sửa đổi mã nguồn, kết quả chạy kiểm thử và hướng dẫn vận hành pipeline.

---

## 3. Thứ tự Triển khai (Execution Order)

```text
[Giai đoạn 1] Fix Loss / Preds / Grad Accumulation
      │
      ▼
[Giai đoạn 2] Fix Checkpoints / Resume / Scaler State
      │
      ▼
[Giai đoạn 3] Harden OOM Handling / Val Loop / Empty Arrays
      │
      ▼
[Giai đoạn 4] Sync Config / Early Stopping / History CSV
      │
      ▼
[Giai đoạn 5] Run Test Suite & Generate report_review_train1.md
```

---

## 4. Yêu cầu Phê duyệt (Approval Request)

Xin người dùng xem xét và phê duyệt kế hoạch trên (`docs/plan/plan_review_train1.md`). Khi nhận được sự đồng ý, Agent sẽ tiến hành **Bước 3: Thực hiện kế hoạch** (chỉnh sửa mã nguồn và chạy kiểm thử).
