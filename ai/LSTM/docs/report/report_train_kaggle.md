# Báo cáo Hoàn thành: Xây dựng Jupyter Notebook Huấn luyện trên Kaggle (Standalone)

**Mã tài liệu:** `report_train_kaggle.md`  
**Dự án:** Driver Guardian AI — SpatioTemporal ConvGRU Drowsiness Detection  
**Tệp mục tiêu:** [`notebooks/02_train_convgru_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/02_train_convgru_kaggle.ipynb)  
**Phương án triển khai:** Phương án B / Lựa chọn 2 (**Standalone Self-Contained Notebook**)  
**Quy chuẩn áp dụng:** [AGENTS.md](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md) — Bước 3: Thực hiện kế hoạch

---

## 1. Tổng quan Kết quả Thực hiện

Đã hoàn thành toàn diện việc xây dựng tệp Jupyter Notebook [`notebooks/02_train_convgru_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/02_train_convgru_kaggle.ipynb). Notebook được thiết kế theo đúng yêu cầu:
- **Độc lập hoàn toàn (Self-Contained)**: Nhúng trực tiếp toàn bộ kiến trúc mô hình, bộ trích xuất đặc trưng GPU, bộ nạp dữ liệu video thô và pipeline huấn luyện vào các cell của notebook. Người dùng chỉ cần tải duy nhất tệp notebook này lên Kaggle và chạy trực tiếp mà không cần đính kèm hoặc clone repository phụ thuộc bên ngoài.
- **Đồng bộ 100% Cấu hình Kỹ thuật**: Giữ trọn vẹn toàn bộ các tính năng tiên tiến và gia cố toán học từ [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py) và [`configs/config.yaml`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.yaml).
- **Tối ưu hóa Chuyên biệt cho Kaggle**: Tự động nhận diện đường dẫn đầu vào chỉ đọc `/kaggle/input/...` và thư mục kết quả `/kaggle/working/...`, tối ưu bộ nhớ cho GPU Tesla T4/P100 (16GB VRAM), thiết lập `num_workers=2` an toàn trên Shared Memory (2GB) của Kaggle.

---

## 2. Chi tiết Cấu trúc 9 Phân đoạn trong Notebook

Notebook được tổ chức mạch lạc thành 24 cells (bao gồm 9 Markdown sections và 15 Code cells thực thi):

### Section 1: Khởi tạo Môi trường & Khảo sát Phần cứng
- **Cell 1.1 - 1.3**:
  - Giới thiệu tổng quan hệ thống phát hiện ngủ gật tài xế với SpatioTemporal ConvGRU.
  - Cài đặt & kiểm tra phiên bản PyTorch, OpenCV, CUDA Toolkit và dung lượng VRAM GPU.
  - Hàm `seed_everything(42)` cố định toàn bộ các nguồn sinh ngẫu nhiên (`random`, `numpy`, `torch`, `cudnn.deterministic = True`).

### Section 2: Cấu hình Đường dẫn & Siêu tham số Huấn luyện
- **Cell 2.1 - 2.2**:
  - Class `KaggleTrainConfig`: Tự động nhận diện cờ `IS_KAGGLE = os.path.exists("/kaggle")`.
  - Phân tách đường dẫn rõ ràng:
    - Input: `dataset_dir` (`/kaggle/input/...`), `backbone_neck_checkpoint` (`best.pt`).
    - Output: `checkpoint_dir`, `tb_log_dir`, `history_csv_path` (`/kaggle/working/...`).
  - Đồng bộ siêu tham số: `batch_size=8`, `val_batch_size=4`, `chunk_size=32`, `lr0=1e-3`, `grad_clip_norm=1.0`, `gradient_accumulation_steps=2`, `patience=10`.

### Section 3: Định nghĩa Bộ Trích xuất Đặc trưng Backbone PAFPN
- **Cell 3.1 - 3.2**:
  - Nhúng đầy đủ các khối CNN: `Conv`, `Bottleneck`, `C2f`, `CIB`, `C2fCIB`, `SPPF`, `Attention`, `C2fPSA`, `SCDown`.
  - Định nghĩa mạng `Backbone`, `PAFPN` và lớp bọc `BackboneNeck`.
  - Lớp `ChunkedBackboneNeckExtractor`: Nạp trọng số từ checkpoint `best.pt`, cắt nhỏ chuỗi khung hình GPU thành các mini-chunk (`chunk_size=32`), ép kiểu FP16 Mixed Precision, chuyển đổi `uint8 -> float32 [0.0, 1.0]` theo từng chunk, loại bỏ triệt để nguy cơ tràn VRAM.

### Section 4: Kiến trúc Mô hình Không gian - Thời gian `ConvGRUClassifier`
- **Cell 4.1 - 4.3**:
  - `ConvGRUCell`: Cổng gộp Kernel Fusion ($2 \times H$), hỗ trợ sequence masking `seq_lens`.
  - `ConvGRU`: Mạng đa tầng bảo toàn bản đồ không gian 2D ($40 \times 40$) qua thời gian.
  - `SpatialReductionNeck`: Nén đa tỷ lệ từ $p_3, p_4, p_5$ xuống 64 kênh tại lưới $40 \times 40$.
  - `SpatialAttentionPooling` & `TemporalAttentionPooling`: Cơ chế Chú ý Kép Dual Attention gom tụ đặc trưng không gian và thời gian kèm khử nhiễu padding frame.
  - `ConvGRUClassifier`: Đóng gói mô hình hoàn chỉnh kèm phương thức `from_config`.

### Section 5: Pipeline Nạp & Tiền xử lý Video Thô
- **Cell 5.1 - 5.3**:
  - Hàm `letterbox`: Chuẩn hóa kích thước khung hình về $640 \times 640$ giữ nguyên tỷ lệ khung hình.
  - `RawVideoFramesDataset`: Đọc video thô qua OpenCV với `sample_interval=0.2s`, trả về Tensor `torch.uint8` trên CPU để bảo vệ RAM hệ thống.
  - `collate_video_frames`: Ghép batch động, zero-padding theo $T_{max}$ dài nhất trong batch.
  - `build_dataloaders`: Factory khởi tạo DataLoader cho Train và Validation.

### Section 6: Hạ tầng Huấn luyện & Đánh giá (Training Infrastructure)
- **Cell 6.1 - 6.3**:
  - `calculate_metrics`: Tính toán Accuracy, Recall, F1-Score (macro) an toàn từ `sklearn.metrics`.
  - `EarlyStopping`: Theo dõi chỉ số `val_f1`, kích hoạt dừng sớm khi qua 10 epochs không cải thiện.
  - `KaggleTrainer`:
    - Hàm `compute_loss_and_preds` chuẩn hóa `CrossEntropyLoss` (2 classes) và `BCEWithLogitsLoss` (1 class), triệt tiêu lỗi gradient 0.
    - Vòng lặp `train_epoch` với Mixed Precision (AMP `GradScaler`), Gradient Accumulation (sửa lệch tỷ trọng batch cuối), OOM recovery an toàn.
    - Vòng lặp `validate_epoch` đánh giá tập kiểm định.
    - Quản lý checkpoint: Luôn lưu `last.pt`, lưu `best.pt` khi đạt F1 kỷ lục.
    - Tự động ghi nhật ký ra tệp `training_history.csv`.

### Section 7: Thực thi Huấn luyện Tương tác
- **Cell 7.1 - 7.3**:
  - Khởi tạo `KaggleTrainer`, in tổng số tham số trainable của mô hình.
  - Chạy `trainer.train()` với thanh tiến trình `tqdm.auto` hiển thị trực quan và bảng số liệu từng epoch theo thời gian thực.

### Section 8: Phân tích & Trực quan hóa Đồ thị Học tập
- **Cell 8.1 - 8.3**:
  - Đọc `training_history.csv` và vẽ 4 đồ thị học tập inline bằng `matplotlib`: Loss, Macro F1-Score, Accuracy, Learning Rate Schedule.
  - Tự động nạp `best.pt` và vẽ ma trận nhầm lẫn Confusion Matrix dạng Heatmap (`seaborn`) trên tập Validation.

### Section 9: Đóng gói Checkpoints & Tải về Máy tính (Export Artifacts)
- **Cell 9.1 - 9.2**:
  - Tự động nén toàn bộ thư mục checkpoints, logs, training history CSV và các đồ thị thành tệp `/kaggle/working/driver_guardian_artifacts.zip`.
  - Hiển thị hướng dẫn trực quan tải file kết quả từ tab Output trên Kaggle.

---

## 3. Kết quả Kiểm thử & Đánh giá Chất lượng

Tệp notebook đã vượt qua toàn bộ các bước kiểm tra nghiêm ngặt:
1. **Kiểm tra Cấu trúc JSON Notebook**: Định dạng chuẩn Jupyter v4 (`nbformat=4, nbformat_minor=5`), đầy đủ metadata của kernel Python 3.
2. **Kiểm tra Cú pháp Python (Compilation Test)**: 100% toàn bộ 15 code cells được biên dịch (`compile`) thành công mà không phát sinh bất kỳ lỗi cú pháp (`SyntaxError`) nào.
3. **Kiểm tra Tính Độc lập (Self-Contained Check)**: Xác nhận không có bất kỳ lệnh import phụ thuộc nào trỏ tới mã nguồn nội bộ (`src.*`, `ai.*`, `configs.*`). Tất cả các lớp cần thiết đều được định nghĩa khép kín trong notebook.
4. **Kiểm tra Forward Pass & Shape Tính toán**: Mô hình `ConvGRUClassifier` nhận đầu vào đặc trưng đa tầng $(p_3, p_4, p_5)$ kèm `seq_lens` và xuất đúng kích thước Logits $[B, \text{num\_classes}] = [2, 2]$.

---

## 4. Hướng dẫn Sử dụng trên Kaggle

Để huấn luyện mô hình trên Kaggle bằng notebook này, người dùng chỉ cần thực hiện 4 bước đơn giản:

1. **Tải Notebook lên Kaggle**:
   - Truy cập [kaggle.com](https://www.kaggle.com) -> Chọn **Create** -> **New Notebook** -> Chọn **File** -> **Import Notebook** -> Chọn tệp [`notebooks/02_train_convgru_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/02_train_convgru_kaggle.ipynb).
2. **Kích hoạt GPU Accelerator**:
   - Ở cột cài đặt bên phải (**Notebook options**): mục **Accelerator** chọn **GPU T4 x1** hoặc **GPU P100**.
3. **Đính kèm Dữ liệu (Add Input Data)**:
   - Thêm Dataset chứa video thô và file `dataset_merged_split.csv`.
   - Thêm Dataset chứa file trọng số `best.pt` của NMSFreeDetector PAFPN.
   - Cập nhật 2 đường dẫn `dataset_dir` và `backbone_neck_checkpoint` trong Section 2 tương ứng với tên dataset trên Kaggle.
4. **Bắt đầu Huấn luyện**:
   - Chọn **Run All** (hoặc nhấn tổ hợp phím `Shift + Enter` qua từng cell).
   - Khi hoàn tất, tải file `driver_guardian_artifacts.zip` từ mục **Output** ở cột phải về máy.
