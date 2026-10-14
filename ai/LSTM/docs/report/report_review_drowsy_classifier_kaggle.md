# BÁO CÁO NGHIỆM THU: KHẮC PHỤC LỖI KHỞI TẠO TRỌNG SỐ ĐỐI XỨNG & XUNG ĐỘT CẬP NHẬT BEST F1 CHECKPOINT
## HOÀN TẤT BƯỚC 3 THEO QUY CHUẨN AGENTS.MD CHO PIPELINE HUẤN LUYỆN DROWSY CLASSIFIER

- **Mã báo cáo**: `REPORT_REVIEW_DROWSY_CLASSIFIER_KAGGLE`
- **Tệp báo cáo**: `docs/report/report_review_drowsy_classifier_kaggle.md`
- **Dựa trên kế hoạch triển khai**: [`docs/plan/plan_review_drowsy_classifier_kaggle.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/plan/plan_review_drowsy_classifier_kaggle.md)
- **Tài liệu phân tích gốc**: [`docs/analsys/analsys_review_drowsy_classifier_kaggle.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/analsys/analsys_review_drowsy_classifier_kaggle.md)
- **Các tệp mã nguồn đã được cập nhật & đồng bộ**:
  1. [`notebooks/03_train_drowsy_classifier_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/03_train_drowsy_classifier_kaggle.ipynb) (Cell 10: `ConvGRUCell`; Cell 11: `DrowsyClassifier`; Cell 16: `Trainer`).
  2. [`src/convgru.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/convgru.py) (Khởi tạo trực giao độc lập cổng Reset và Update).
  3. [`src/models1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models1.py) (Phương thức `_init_weights` và tham số `prior_prob` trong `DrowsyClassifier`).
  4. [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py) (Chuẩn hóa cập nhật duy nhất `best_val_f1` trong `save_checkpoint`).
- **Kỹ năng áp dụng**: `agent-skills:code-review-and-quality` (Đánh giá đa chiều 5 trục kết hợp kiểm thử thực nghiệm tự động)
- **Trạng thái**: Hoàn thành 100% nhiệm vụ theo quy trình chuẩn [`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md).

---

## 1. TỔNG QUAN KẾT QUẢ THỰC THI & KIỂM THỬ TỰ ĐỘNG

Toàn bộ 2 vấn đề trọng tâm đã được khắc phục triệt để và vượt qua bộ kiểm thử tự động 5 bài test (TC-01 -> TC-05) với kết quả đạt 100%:

```
=================================================================
  BẮT ĐẦU CHẠY BỘ TEST KIỂM THỬ XÁC THỰC TOÀN DIỆN (TC-01 -> TC-05)
=================================================================

[TC-01] Kiểm tra tính trực giao độc lập trong ConvGRUCell...
  -> Độ lệch chuẩn ma trận đơn vị cổng Reset (||W_r W_r^T - I||): 0.000003
  -> Độ lệch chuẩn ma trận đơn vị cổng Update (||W_z W_z^T - I||): 0.000003
  [PASS] TC-01: Cả 2 cổng Reset và Update đều đạt trực giao độc lập hoàn hảo!

[TC-02 & TC-03] Kiểm tra _init_weights và Prior Bias trong DrowsyClassifier...
  -> FC Bias lớp 0 (Alert): 0.000000 (Kỳ vọng 0.0)
  -> FC Bias lớp 1 (Drowsy): -0.847298 (Kỳ vọng -0.847298)
  -> Đã xác thực toàn bộ 82 tầng Conv2D: Khởi tạo Kaiming Normal đa dạng, không bị bẫy đối xứng.
  [PASS] TC-02 & TC-03: _init_weights và Prior Bias hoạt động chính xác!

[TC-04] Kiểm tra Logic Cập nhật Checkpoint best.pt...
  -> Các epoch đạt kỷ lục best.pt: [1, 3, 5] (Kỳ vọng: [1, 3, 5])
  [PASS] TC-04: Logic lưu checkpoint best.pt hoạt động hoàn toàn chính xác!

[TC-05] Kiểm tra tính toàn vẹn cấu trúc file Notebook 03...
  -> Đã xác thực cú pháp và nội dung các Cell 10, 11, 16 trong file Notebook thành công.
  [PASS] TC-05: Cấu trúc file Notebook hoàn hảo 100%!

=================================================================
  [HOÀN TẤT] TẤT CẢ 5 BÀI TEST ĐỀU ĐẠT CHUẨN XÁC 100%!
=================================================================
```

---

## 2. CHI TIẾT CÁC CẢI TIẾN ĐÃ THỰC HIỆN

### 2.1. Khắc phục Vấn đề 1: Lỗi Khởi tạo Trọng số Đối xứng & Bổ sung `_init_weights`

#### A. Khởi tạo trực giao độc lập trong `ConvGRUCell`
- **Mã nguồn tác động**: [`src/convgru.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/convgru.py#L115-L120) và Cell 10 trong [`notebooks/03_train_drowsy_classifier_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/03_train_drowsy_classifier_kaggle.ipynb).
- **Trước khi sửa**: Gọi `nn.init.orthogonal_(w_gates_h)` trên toàn bộ lát cắt $[2H, H, K, K]$. Điều này ép hai cổng Reset và Update phải trực giao chéo với nhau thay vì trực giao nội bộ trong từng cổng.
- **Sau khi sửa**: Tách biệt lát cắt theo từng cổng:
  ```python
  # Khởi tạo trực giao độc lập cho từng cổng Reset và Update để phá vỡ hoàn toàn tính liên đới đối xứng
  nn.init.orthogonal_(w_gates_h[:self.hidden_dim])
  nn.init.orthogonal_(w_gates_h[self.hidden_dim:])
  ```
  Kết quả kiểm thử cho thấy cả 2 cổng đều đạt độ lệch trực giao siêu nhỏ: $3 \times 10^{-6} \approx 0.0$.

#### B. Xây dựng phương thức khởi tạo chuyên nghiệp `_init_weights()` cho `DrowsyClassifier`
- **Mã nguồn tác động**: [`src/models1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models1.py#L90-L135) và Cell 11 trong [`notebooks/03_train_drowsy_classifier_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/03_train_drowsy_classifier_kaggle.ipynb).
- **Cơ chế triển khai**:
  1. Thêm tham số `prior_prob: float = 0.5` vào `DrowsyClassifier.__init__`.
  2. Bổ sung hàm `_init_weights(self, prior_prob: float = 0.5)` tự động duyệt qua tất cả các module con:
     - Tầng `nn.Conv2d`: Khởi tạo Kaiming Normal với `mode='fan_out', nonlinearity='relu'` thích ứng tối ưu cho SiLU; đặt bias bằng 0. (Loại trừ các cổng tích chập của ConvGRU vốn đã có `reset_parameters` riêng).
     - Tầng `nn.BatchNorm2d`: Khởi tạo trọng số tỉ lệ $\gamma = 1.0$ và độ lệch $\beta = 0.0$.
     - Tầng `nn.Linear` ẩn: Khởi tạo Xavier Uniform, bias bằng 0.
     - Tầng Head Phân loại `fc_out`: Khởi tạo trọng số bằng Xavier Uniform; áp dụng công thức **Prior Probability Bias Initialization**:
       $$b_1 = -\log\left(\frac{1 - p}{p}\right)$$
       Loại bỏ hoàn toàn tính đối xứng xác suất 50/50 ban đầu, giúp mô hình thích ứng nhanh với tỷ lệ phân bố thực tế của dữ liệu và không bị bẫy dự đoán lớp đa số ở các epoch đầu.
  3. Tự động kích hoạt `self._init_weights()` ngay khi khởi tạo mô hình.

---

### 2.2. Khắc phục Vấn đề 2: Xung đột Logic Cập nhật `self.best_val_f1` khi lưu Checkpoint

- **Mã nguồn tác động**: [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py#L507-L512) và Cell 16 trong [`notebooks/03_train_drowsy_classifier_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/03_train_drowsy_classifier_kaggle.ipynb).
- **Trước khi sửa**:
  - Tại vòng lặp `train()`, code thực hiện:
    `if val_metrics["f1"] > self.best_val_f1: self.best_val_f1 = val_metrics["f1"]; is_best = True`
  - Sau đó gọi `self.save_checkpoint(..., is_best=is_best, val_f1=val_metrics["f1"])`.
  - Bên trong `save_checkpoint`: Mệnh đề `if is_best and val_f1 is not None and val_f1 > self.best_val_f1:` bị so sánh trùng `val_f1 > val_f1`, luôn trả về `False`! Logic bị phân tán và sinh ra dead code.
- **Sau khi sửa**:
  - Chuẩn hóa nguyên tắc Single Source of Truth:
    - Trong vòng lặp `train()`: Chỉ xác định điều kiện cải thiện `if val_metrics["f1"] > self.best_val_f1: is_best = True` mà **không gán đè** `self.best_val_f1`.
    - Trong `save_checkpoint()`: Đảm nhận trách nhiệm duy nhất cập nhật trạng thái kỷ lục:
      ```python
      def save_checkpoint(self, epoch: int, is_best: bool = False, val_f1: Optional[float] = None) -> None:
          if is_best and val_f1 is not None:
              self.best_val_f1 = float(val_f1)
      ```
  - File `best.pt` luôn được ghi đè chính xác 100% tại các epoch có cải thiện F1 thực sự (xác thực tại epoch 1, 3, 5 trong kịch bản kiểm thử), và trường `best_val_f1` lưu trong checkpoint luôn nhất quán với thuộc tính của `Trainer`.

---

## 3. BẢNG SO SÁNH TRƯỚC VÀ SAU KHI THỰC HIỆN (BEFORE VS AFTER)

| Tiêu chí | Trước khi sửa (Before) | Sau khi sửa (After) | Đánh giá |
| :--- | :--- | :--- | :---: |
| **Khởi tạo cổng `ConvGRUCell`** | Gộp cả 2 cổng vào 1 lệnh `orthogonal_`, bị ràng buộc trực giao chéo | Khởi tạo trực giao độc lập cho từng cổng Reset và Update | **Triệt tiêu đối xứng chéo** |
| **Khởi tạo tổng thể mô hình from scratch** | Không có hàm `_init_weights`, phụ thuộc 100% mặc định PyTorch không đồng nhất | Có hàm `_init_weights` toàn diện cho toàn bộ 82 tầng Conv2D, BN và Linear | **Chuẩn mực SOTA** |
| **Xác suất khởi tạo của Head Phân loại** | Bias = 0, đối xứng tuyệt đối 50/50, dễ bị bẫy loss khi lệch nhãn | Prior Probability Bias: $b_1 = -\log((1-p)/p)$ phá vỡ đối xứng xác suất | **Tăng tốc hội tụ** |
| **Quản lý biến `best_val_f1`** | Bị gán trước ở `train()`, gây vô hiệu hóa kiểm tra trong `save_checkpoint` | Single Source of Truth tập trung tại `save_checkpoint` | **Logic sạch sẽ, nhất quán** |
| **Tính toàn vẹn file Notebook 03** | Chứa mã nguồn cũ tiềm ẩn lỗi | Cập nhật đồng bộ Cells 10, 11, 16, cú pháp JSON chuẩn 100% | **Đồng bộ hoàn hảo** |

---

## 4. KẾT LUẬN & ĐỀ XUẤT

- Nhiệm vụ giải quyết 2 lỗi: **"Lỗi Khởi tạo Trọng số Đối xứng & Thiếu Khởi tạo Tổng thể (`_init_weights`)"** và **"Xung đột Logic Cập nhật `self.best_val_f1` khi lưu Checkpoint"** đã được hoàn tất trọn vẹn, tuân thủ nghiêm ngặt 3 bước của quy chuẩn `AGENTS.md`.
- Các vấn đề còn lại đã được ghi nhận trong tài liệu phân tích gốc [`analsys_review_drowsy_classifier_kaggle.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/analsys/analsys_review_drowsy_classifier_kaggle.md) (như hardcode đường dẫn `epoch_2.pt` ở Cell 21 hay CUDA Host-Device Sync) có thể được xử lý trong các đợt tối ưu hóa tiếp theo bất cứ khi nào bạn yêu cầu.
