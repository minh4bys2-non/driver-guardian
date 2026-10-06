# Kế hoạch Thực hiện: Xây dựng Jupyter Notebook Huấn luyện trên Kaggle (Standalone)

**Mã tài liệu:** `plan_train_kaggle.md`  
**Dự án:** Driver Guardian AI — SpatioTemporal ConvGRU Drowsiness Detection  
**Tệp mục tiêu:** [`notebooks/02_train_convgru_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/02_train_convgru_kaggle.ipynb)  
**Phương án đã chọn:** Phương án B / Lựa chọn 2 (**Standalone Self-Contained Notebook**)  
**Quy chuẩn áp dụng:** [AGENTS.md](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md) — Bước 2: Lên kế hoạch thực hiện

---

## 1. Mục tiêu Kế hoạch

Triển khai chi tiết từng cell cho tệp Jupyter Notebook [`notebooks/02_train_convgru_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/02_train_convgru_kaggle.ipynb) theo chuẩn **Self-Contained**. Người dùng chỉ cần tải một tệp `.ipynb` duy nhất lên nền tảng Kaggle, đính kèm dataset video thô và file trọng số BackboneNeck (`best.pt`), sau đó nhấn **Run All** là có thể huấn luyện hoàn chỉnh mà không cần cài thêm repo ngoài hay cấu hình phức tạp.

---

## 2. Thiết kế Cấu trúc Chi tiết Các Cell trong Notebook

Notebook được chia thành **9 Section logic** với các Markdown headers phân đoạn rõ ràng:

### Section 1: Khởi tạo Môi trường & Khảo sát Phần cứng
- **Cell 1.1 (Markdown)**: Giới thiệu kiến trúc SpatioTemporal ConvGRU, cơ chế trích xuất Backbone PAFPN on-the-fly, mục tiêu phát hiện ngủ gật của tài xế.
- **Cell 1.2 (Code - Imports & Dependencies)**:
  - Cài đặt/Kiểm tra các thư viện: `torch`, `torchvision`, `opencv-python`, `albumentations`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`.
- **Cell 1.3 (Code - Reproducibility & Device Detection)**:
  - Hàm `seed_everything(42)` cố định seed cho `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `cudnn.deterministic = True`.
  - Kiểm tra GPU (NVIDIA Tesla T4 hoặc P100), in thông tin VRAM và kích hoạt `torch.cuda.amp`.

### Section 2: Cấu hình Đường dẫn & Siêu tham số (Kaggle Configuration)
- **Cell 2.1 (Code - KaggleTrainConfig)**:
  - Tự động nhận diện môi trường Kaggle (`IS_KAGGLE = os.path.exists('/kaggle')`).
  - Thiết lập đường dẫn động:
    - `DATASET_DIR`: `/kaggle/input/<dataset-slug>/data_processed` hoặc trỏ trực tiếp vào thư mục video thô.
    - `MANIFEST_FILE`: Tự động tìm `dataset_merged_split.csv` trong dataset input.
    - `BACKBONE_CKPT`: Đường dẫn file trọng số `best.pt` của NMSFreeDetector PAFPN.
    - `OUTPUT_DIR`: `/kaggle/working/runs/deepgru_raw_nmsfree`.
    - `CHECKPOINT_DIR`: `/kaggle/working/checkpoints`.
    - `LOGS_DIR`: `/kaggle/working/logs`.
  - Đồng bộ 100% siêu tham số từ [`configs/config.yaml`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.yaml):
    - `batch_size = 8`, `val_batch_size = 4`, `num_workers = 2`, `val_num_workers = 0`.
    - `epochs = 30`, `lr0 = 1e-3`, `lr_min_factor = 0.01`, `weight_decay = 1e-4`.
    - `chunk_size = 32`, `gradient_accumulation_steps = 2`, `grad_clip_norm = 1.0`.
    - `early_stopping = True`, `patience = 10`, `monitor = 'val_f1'`, `mode = 'max'`.

### Section 3: Định nghĩa Kiến trúc Trích xuất Đặc trưng (Backbone PAFPN Extractor)
*(Nhúng trực tiếp theo Phương án B / Lựa chọn 2 - Standalone)*
- **Cell 3.1 (Code - CNN Blocks & PAFPN)**:
  - Các khối tích chập chuẩn: `Conv`, `Bottleneck`, `C2f`, `CIB`, `C2fCIB`, `SPPF`, `Attention`, `C2fPSA`, `SCDown`.
  - Lớp `Backbone` và `PAFPN` (trích xuất 3 mức đặc trưng đa tỷ lệ: $p_3, p_4, p_5$).
  - Lớp `BackboneNeck` bọc Backbone + PAFPN.
- **Cell 3.2 (Code - ChunkedBackboneNeckExtractor)**:
  - Lớp trích xuất đặc trưng Mini-Chunk trên GPU (`chunk_size=32`), hỗ trợ FP16 Mixed Precision, chuyển đổi `uint8 -> float32 [0.0, 1.0]` theo từng chunk, triệt tiêu trần tràn VRAM trên GPU.

### Section 4: Định nghĩa Mô hình SpatioTemporal ConvGRUClassifier
- **Cell 4.1 (Code - ConvGRU Core)**:
  - `ConvGRUCell` với Kernel Fusion (gộp cổng Reset và Update vào 1 tầng tích chập $2 \times H$), hỗ trợ sequence masking `seq_lens`.
  - `ConvGRU` đa tầng (Multi-layer ConvGRU) hỗ trợ dynamic spatial resolution và bidirectional/unidirectional.
- **Cell 4.2 (Code - Dual Attention & SpatialReductionNeck)**:
  - `SpatialReductionNeck`: Nén kênh đa tỷ lệ $p_3$ (giảm $2\times$), $p_4$ (giữ nguyên), $p_5$ (tăng $2\times$) rồi ghép kênh và nén $448 \rightarrow 64$ kênh tại lưới $40 \times 40$.
  - `SpatialAttentionPooling`: Cơ chế chú ý không gian 2D tập trung vào vùng mắt/mặt.
  - `TemporalAttentionPooling`: Cơ chế gom tụ chú ý thời gian kèm masking `seq_lens`.
- **Cell 4.3 (Code - ConvGRUClassifier Complete Model)**:
  - Đóng gói toàn bộ mô hình `ConvGRUClassifier`, tích hợp hàm `from_config` và phương thức khởi tạo trọng số trực giao an toàn.

### Section 5: Pipeline Nạp Dữ liệu Video Thô (Raw Video DataLoaders)
- **Cell 5.1 (Code - Video Augmenter & Letterbox)**:
  - Hàm `letterbox` giữ tỷ lệ khung hình video về $640 \times 640$.
  - Augmentation cho video: biến đổi ngẫu nhiên đồng nhất trên toàn bộ khung hình trong clip.
- **Cell 5.2 (Code - RawVideoFramesDataset & Collate Function)**:
  - `RawVideoFramesDataset`: Đọc video thô qua OpenCV với `sample_interval=0.2s`, trích xuất chuỗi frame, chuyển thành Tensor `uint8` trên CPU để bảo vệ RAM hệ thống.
  - `collate_video_frames`: Gom batch động, zero-padding theo độ dài chuỗi dài nhất trong batch ($T_{max}$), loại bỏ video lỗi, trả về `(frames, labels, seq_lens, metas)`.
  - Hàm factory `build_raw_video_dataloaders`.

### Section 6: Cơ sở Hạ tầng Huấn luyện & Đánh giá (Training Infrastructure)
- **Cell 6.1 (Code - Metrics & Early Stopping)**:
  - Hàm `calculate_metrics`: Tính Accuracy, Recall, F1-Score (macro) an toàn từ `sklearn.metrics`.
  - Lớp `EarlyStopping`: Theo dõi chỉ số `val_f1`, kích hoạt dừng sớm khi không cải thiện qua $patience=10$ epochs.
- **Cell 6.2 (Code - Loss Function & Prediction Resolver)**:
  - Hàm `compute_loss_and_preds`: Chuẩn hóa `CrossEntropyLoss` (cho 2 lớp) và `BCEWithLogitsLoss` (cho 1 lớp), loại bỏ triệt để lỗi gradient 0 của kênh 0.
- **Cell 6.3 (Code - KaggleTrainer Class)**:
  - Đóng gói toàn bộ logic huấn luyện:
    - Vòng lặp `train_epoch`: Gradient Accumulation, AMP GradScaler, Clip Grad Norm, OOM Recovery với giải phóng traceback frame.
    - Vòng lặp `validate_epoch`: Đánh giá trung thực, tính Confusion Matrix.
    - Lưu trữ checkpoint: Luôn lưu `last.pt`, lưu `best.pt` khi đạt F1 cao nhất, quản lý dung lượng đĩa `/kaggle/working`.
    - Ghi nhận nhật ký: `training_history.csv` và TensorBoard SummaryWriter.

### Section 7: Thực thi Huấn luyện Tương tác (Interactive Execution)
- **Cell 7.1 (Code - Khởi tạo & Kiểm thử Dữ liệu/Mô hình)**:
  - Khởi tạo DataLoader, kiểm tra 1 batch mẫu, in kích thước Tensor và nhãn.
  - Khởi tạo Extractor và Model, in bảng tổng hợp số lượng tham số trainable.
- **Cell 7.2 (Code - Chạy Vòng lặp Huấn luyện)**:
  - Chạy `trainer.train()` với thanh tiến trình `tqdm.notebook`.
  - Hiển thị bảng kết quả Epoch Metrics theo thời gian thực.

### Section 8: Phân tích & Trực quan hóa Kết quả Huấn luyện (Visualizations)
- **Cell 8.1 (Code - Vẽ Đồ thị Học tập Inline)**:
  - Đọc file `training_history.csv`.
  - Vẽ đồ thị 2 hàng 2 cột:
    1. Train Loss vs Val Loss.
    2. Train Accuracy vs Val Accuracy.
    3. Train F1-Score vs Val F1-Score.
    4. Learning Rate schedule.
- **Cell 8.2 (Code - Vẽ Ma trận Nhầm lẫn Confusion Matrix)**:
  - Vẽ Heatmap Confusion Matrix trên tập validation của mô hình tốt nhất (`best.pt`).

### Section 9: Đóng gói & Tải về Checkpoints (Artifacts Export)
- **Cell 9.1 (Code - Zip Artifacts Helper)**:
  - Tự động nén toàn bộ thư mục checkpoints, logs, training_history.csv, và biểu đồ thành file `/kaggle/working/driver_guardian_artifacts.zip`.
  - Cung cấp liên kết tải trực tiếp tiện lợi cho người dùng từ giao diện Kaggle.

---

## 3. Tiêu chí Chất lượng & Kiểm định (Quality Checklist)

Trước khi bàn giao, notebook cần đạt các tiêu chí sau:
1. **Tính Tự lập (Self-Contained)**: Không phụ thuộc vào bất kỳ file `.py` ngoài nào của dự án; chỉ cần cài đặt thư viện pip chuẩn.
2. **Cấu trúc JSON Notebook Hợp lệ**: Tệp `.ipynb` được sinh đúng định dạng chuẩn Jupyter v4 (`nbformat=4, nbformat_minor=5`).
3. **Tính Đúng đắn Toán học & Gradient**:
   - Sử dụng `CrossEntropyLoss` cho `num_classes=2`, đồng bộ `compute_loss_and_preds`.
   - Chuẩn hóa tỷ lệ Gradient Accumulation ở batch cuối epoch.
   - Bảo toàn trạng thái AMP GradScaler.
4. **An toàn Bộ nhớ**:
   - `ChunkedBackboneNeckExtractor` cắt chunk 32 khung hình trên GPU.
   - Bắt ngoại lệ CUDA OOM và dọn sạch bộ nhớ đệm cache.
   - Giới hạn `num_workers=2` an toàn trên Shared Memory 2GB của Kaggle.

---

## 4. Hành động Tiếp theo

👉 **Kính mời bạn duyệt kế hoạch triển khai trên (`plan_train_kaggle.md`). Khi bạn đồng ý (hoặc có chỉnh sửa), tôi sẽ tiến hành Bước 3: Tạo tệp notebook `notebooks/02_train_convgru_kaggle.ipynb` và tệp báo cáo tổng kết `docs/report/report_train_kaggle.md`.**
