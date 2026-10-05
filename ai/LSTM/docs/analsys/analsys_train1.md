# Phân tích yêu cầu: Viết script huấn luyện ConvGRUClassifier (train1.py)

## 1. Mục tiêu
- Viết file `src/train1.py` thực hiện pipeline huấn luyện (training pipeline) cho mô hình `ConvGRUClassifier` được định nghĩa trong `src/models1.py`.
- Thiết lập ghi nhận thông số quá trình huấn luyện vào TensorBoard.

## 2. Các yêu cầu chi tiết
- **Mô hình**: Sử dụng `ConvGRUClassifier` thay vì mô hình cũ. Nhận config khởi tạo mô hình từ file cấu hình chung.
- **Cấu hình**: Đọc các tham số cấu hình cho mô hình và quá trình huấn luyện (batch_size, epochs, learning rate,...) từ config (ví dụ: `configs/config.yaml`).
- **TensorBoard Logging độc lập**:
  - Ghi nhận đầy đủ các thông số: `loss`, `accuracy` (acc), `recall`, và `F1-score` cho cả hai tập Train và Validation.
  - Các thông số phải được ghi theo hai đơn vị thời gian: **epoch** (từng kỷ nguyên) và **step** (từng bước cập nhật batch).

## 3. Các thành phần cần triển khai trong script `train1.py`
1. **Khởi tạo và đọc cấu hình**: Parse yaml/json hoặc dùng hệ thống config.
2. **DataLoader**: Chuẩn bị dataset để truyền (p3, p4, p5) feature maps và seq_lens vào mô hình, tương thích với output của backbone NMSFreeDetector.
3. **Loss & Optimizer**: Thiết lập CrossEntropy/BCE Loss, AdamW optimizer, Scheduler (Cosine Annealing...).
4. **Metric calculation**: Hàm tính toán Acc, Recall, F1 theo metric tiêu chuẩn.
5. **TensorBoard Writer**: Khởi tạo `SummaryWriter` cho quá trình training, log thông số vào thư mục được chỉ định trong config.
6. **Vòng lặp huấn luyện (Training Loop)**:
   - Cập nhật trọng số theo batch.
   - Tính toán và ghi nhận log TensorBoard tại từng step.
7. **Kiểm định (Validation Loop)**: 
   - Đánh giá mô hình trên tập validation sau mỗi epoch và/hoặc step.
8. **Checkpoints**: Lưu model trọng số (best model, last model).

## 4. Hành động
Xin người dùng duyệt nội dung phân tích (analsys_train1.md) này để tiếp tục tạo kế hoạch (plan_train1.md).
