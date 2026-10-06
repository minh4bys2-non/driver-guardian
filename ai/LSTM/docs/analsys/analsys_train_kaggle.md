# Báo cáo Phân tích Yêu cầu: Xây dựng Jupyter Notebook Huấn luyện trên Kaggle (train1_kaggle)

**Mã tài liệu:** `analsys_train_kaggle.md`  
**Dự án:** Driver Guardian AI — SpatioTemporal ConvGRU Drowsiness Detection  
**File tham chiếu cấu hình:** [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py), [`configs/config.yaml`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.yaml)  
**Quy chuẩn áp dụng:** [AGENTS.md](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md) — Bước 1: Khảo sát & Phân tích (Discovery)

---

## 1. Mục tiêu & Bối cảnh (Objectives & Scope)

### 1.1. Mục tiêu tổng quát
Xây dựng một tệp Jupyter Notebook hoàn chỉnh, đạt chuẩn mã nguồn sạch và tái lập (reproducible), thực hiện toàn bộ quy trình huấn luyện mô hình **`ConvGRUClassifier`** tương đương với pipeline trong [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py), được thiết kế chuyên biệt và tối ưu hóa để vận hành trơn tru trên nền tảng **Kaggle Notebooks (GPU Kernel)**.

### 1.2. Vị trí & Quy tắc đặt tên
- **Thư mục đích:** `notebooks/`
- **Tên tệp đề xuất:** [`notebooks/02_train_convgru_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/02_train_convgru_kaggle.ipynb) (tiếp nối notebook `01_demo_img_preprocess.ipynb` theo quy tắc tiền tố số thứ tự tại Section 3 AGENTS.md).

---

## 2. Phân tích Hiện trạng Kỹ thuật từ `src/train1.py` & `configs/config.yaml`

Pipeline huấn luyện trong [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py) là phiên bản v2.0 đã qua chuẩn hóa khắt khe với các thành phần kỹ thuật then chốt:

| Thành phần | Đặc tả kỹ thuật trong `src/train1.py` | Yêu cầu tương ứng trong Notebook |
|---|---|---|
| **Mô hình chính** | `ConvGRUClassifier` ([`src/models1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models1.py)) với `SpatialReductionNeck`, `ConvGRU`, `SpatialAttentionPooling`, `TemporalAttentionPooling`. | Nạp kiến trúc mô hình đầy đủ, khởi tạo từ thông số cấu hình. |
| **Trích xuất đặc trưng** | `ChunkedBackboneNeckExtractor` ([`src/dataset2.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/dataset2.py)), cắt nhỏ chuỗi khung hình GPU on-the-fly (`chunk_size=32`), đông băng trọng số. | Nạp trọng số PAFPN NMSFreeDetector (`.pt`), trích xuất feature maps `(p3, p4, p5)`. |
| **Dữ liệu & DataLoader** | `build_raw_video_dataloaders` đọc trực tiếp video thô (`.mp4`, `.avi`,...) qua `RawVideoFramesDataset`, chia theo subject, augment và letterbox 640x640. | Điều chỉnh cấu hình đường dẫn dữ liệu tương thích cấu trúc thư mục Kaggle `/kaggle/input/...`. |
| **Hàm mất mát (Loss)** | `CrossEntropyLoss` (chuẩn cho `num_classes=2`) hoặc `BCEWithLogitsLoss` (khi `num_classes=1`), xử lý triệt để lỗi gradient. | Đồng bộ cơ chế tính loss và nhãn dự đoán theo số lớp (`compute_loss_and_preds`). |
| **Bộ tối ưu & Lịch trình** | AdamW (`lr0=1e-3`, `weight_decay=1e-4`, `betas=[0.9, 0.999]`), `CosineAnnealingLR` (`eta_min=1e-5`). | Thiết lập optimizer và scheduler đồng nhất. |
| **Tăng tốc & Bộ nhớ** | Mixed Precision (AMP `GradScaler('cuda')`), Gradient Accumulation (`steps=2`), giải phóng bộ nhớ OOM an toàn (`torch.cuda.empty_cache()`). | Giữ nguyên cơ chế AMP và dọn rác OOM để chạy ổn định trên GPU Kaggle. |
| **Kiểm soát & Giám sát** | Early Stopping (`monitor=val_f1`, `patience=10`, `mode=max`), ghi nhận `training_history.csv`, lưu checkpoint `best.pt`, `last.pt`. | Tích hợp Early Stopping, xuất CSV và vẽ biểu đồ trực quan trực tiếp trên output notebook. |

---

## 3. Khảo sát Đặc thù Môi trường Kaggle & Các Giải pháp Kỹ thuật

Môi trường Kaggle Notebooks có nhiều điểm khác biệt lớn so với máy tính cục bộ (Local Windows), đòi hỏi các giải pháp thích ứng cụ thể:

### 3.1. Hệ thống Tập tin (File System Constraints)
- **Kaggle Input (`/kaggle/input/...`)**: Chỉ cho phép đọc (Read-only).
  - Dữ liệu video thô và file manifest (`dataset_merged_split.csv`) sẽ nằm trong `/kaggle/input/<dataset-name>/...`.
  - Trọng số BackboneNeck NMSFreeDetector (`best.pt`) cũng được nạp qua dataset đầu vào.
- **Kaggle Working (`/kaggle/working/...`)**: Thư mục duy nhất có quyền ghi (Read-Write).
  - Toàn bộ Checkpoints (`best.pt`, `last.pt`), file nhật ký `training_history.csv`, logs TensorBoard, và hình ảnh biểu đồ phải được ghi vào `/kaggle/working/...`.
- **Giải pháp**: Xây dựng khối biến cấu hình đường dẫn `KaggleConfig` linh hoạt, tự động nhận diện nếu đang chạy trên Kaggle (`os.path.exists('/kaggle')`) hoặc cho phép ghi đè đường dẫn dễ dàng.

### 3.2. Quản lý Bộ nhớ & Giới hạn Tài nguyên (Resource & VRAM Limits)
- **GPU**: Kaggle thường cấp NVIDIA Tesla T4 (15–16 GB VRAM) hoặc Tesla P100 (16 GB VRAM).
  - Cấu hình `batch_size: 8`, `val_batch_size: 4`, `chunk_size: 32` kết hợp `amp: true` hoàn toàn phù hợp và an toàn trên VRAM 16GB.
- **Shared Memory (`/dev/shm`)**: Giới hạn ở 2GB.
  - Đặt `num_workers: 2` (hoặc tối đa 4) trong DataLoader, tránh dùng quá nhiều worker dẫn đến lỗi `Bus error` hoặc OOM RAM hệ thống do đa tiến trình giải mã video OpenCV.
- **Giới hạn Lưu trữ đĩa (`/kaggle/working` ~ 20GB)**:
  - Nếu lưu checkpoint mọi epoch (`save_all_epochs: true`), dung lượng đĩa có thể bị tràn.
  - Cần mặc định lưu `best.pt` (F1 cao nhất) và `last.pt`, kèm tùy chọn dọn dẹp các checkpoint phụ nếu người dùng huấn luyện dài ngày.

### 3.3. Phụ thuộc Mã nguồn Dự án (Project Dependencies & Code Access)
- Để chạy được `ConvGRUClassifier` và `ChunkedBackboneNeckExtractor`, notebook cần truy cập các module trong `src/` và `configs/`, đồng thời cần mã nguồn `ai.ObjectDetection_2p6M` để nạp `BackboneNeck`.
- **Phương án tích hợp mã nguồn**:
  1. *Phương án A (Import từ Repository/Utility Dataset - Khuyến nghị)*:
     - Notebook có cell đầu tiên cấu hình `sys.path` trỏ về thư mục dự án (khi người dùng clone Git hoặc đính kèm dự án dưới dạng Kaggle Dataset).
  2. *Phương án B (Standalone / Nhúng trực tiếp các định nghĩa cốt lõi)*:
     - Đóng gói đầy đủ code các lớp kiến trúc và extractor vào các cell của notebook để notebook có khả năng chạy độc lập hoàn toàn mà không sợ lỗi import module ngoại vi.
  3. *Phương án C (Hybrid)*: 
     - Ưu tiên import từ thư mục mã nguồn gốc nếu có; nếu không tìm thấy sẽ sử dụng định nghĩa trực tiếp trong notebook.

### 3.4. Trực quan hóa & Tương tác trong Notebook
- Khác với chạy script nền [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py), người dùng Kaggle cần theo dõi tiến trình trực quan:
  - Sử dụng thanh tiến trình `tqdm.notebook` hiển thị mượt mà trên giao diện Web.
  - Hiển thị bảng tổng hợp chỉ số sau mỗi epoch (`IPython.display`).
  - Hỗ trợ TensorBoard trực tiếp qua extension `%load_ext tensorboard`.
  - Tự động vẽ biểu đồ trực tiếp (Inline Plots): Loss curves, Accuracy/F1 curves, Confusion Matrix sau khi kết thúc hoặc ngắt huấn luyện.
  - Cung cấp cell nén kết quả (Zip Archive) để người dùng tải toàn bộ weights và log về máy cá nhân chỉ bằng một cú nhấp chuột.

---

## 4. Cấu trúc Dự kiến của Jupyter Notebook (`02_train_convgru_kaggle.ipynb`)

Tuân thủ nghiêm ngặt **Mục 3 AGENTS.md** (Quy chuẩn làm việc với Jupyter Notebooks):

1. **Header & Cell Giới thiệu (Cell 1 - Markdown & Imports)**:
   - Mô tả mục tiêu, kiến trúc ConvGRU, sơ đồ luồng dữ liệu và danh sách thư viện cần thiết (`torch`, `cv2`, `albumentations`, `sklearn`, `matplotlib`,...).
2. **Cấu hình Môi trường & Thiết bị (Cell 2 - Environment & Seed)**:
   - Hàm `seed_everything(42)` đảm bảo tính tái lập.
   - Kiểm tra GPU (NVIDIA Tesla T4/P100), dung lượng VRAM, cấu hình CUDA và AMP.
3. **Cấu hình Đường dẫn & Siêu tham số (Cell 3 - Config Setup)**:
   - Khai báo class/dict `KaggleTrainConfig` đồng bộ 100% với [`configs/config.yaml`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.yaml):
     - `dataset_dir`, `manifest_file`, `backbone_neck_checkpoint` (được trỏ chuẩn về `/kaggle/input/...`).
     - `output_dir`, `checkpoint_dir`, `tb_log_dir` (trỏ về `/kaggle/working/...`).
     - Hyperparameters: `batch_size=8`, `epochs=30`, `lr0=1e-3`, `grad_clip_norm=1.0`, `chunk_size=32`, `patience=10`.
4. **Khởi tạo Dữ liệu (Cell 4 - Dataset & DataLoader)**:
   - Thiết lập DataLoader với `build_raw_video_dataloaders`.
   - In kiểm tra số lượng mẫu tập Train/Val và hiển thị thông số kích thước batch thử nghiệm.
5. **Khởi tạo Mô hình & Bộ trích xuất (Cell 5 - Model & Extractor)**:
   - Nạp `ChunkedBackboneNeckExtractor` với checkpoint PAFPN.
   - Nạp `ConvGRUClassifier`, in tổng số tham số (Trainable vs Non-trainable params).
6. **Bộ Tối ưu, Hàm Mất mát & Giám sát (Cell 6 - Optimizer, Loss & EarlyStopping)**:
   - `CrossEntropyLoss` / `BCEWithLogitsLoss`.
   - AdamW + `CosineAnnealingLR`.
   - `EarlyStopping` theo `val_f1`.
7. **Pipeline Huấn luyện Chuyên biệt (Cell 7 - KaggleTrainer)**:
   - Đóng gói logic huấn luyện sạch (`train_epoch`, `validate_epoch`, `save_checkpoint`, `OOM recovery`).
8. **Thực thi Huấn luyện (Cell 8 - Execution)**:
   - Chạy `trainer.train()`, hiển thị tiến trình và cập nhật thời gian thực.
9. **Trực quan hóa Đồ thị & Đánh giá (Cell 9 - Inline Visualizations)**:
   - Đọc `training_history.csv` và vẽ biểu đồ Train vs Val Loss/Accuracy/F1 bằng `matplotlib`.
   - Vẽ Confusion Matrix của kỷ nguyên tốt nhất.
10. **Đóng gói Checkpoint Tải về (Cell 10 - Checkpoint Archiving)**:
    - Zip thư mục checkpoint và logs để người dùng tải file nén `.zip` từ output Kaggle.

---

## 5. Đề xuất Lựa chọn cho Người dùng (Options for Clarification)

Trước khi chuyển sang **Bước 2: Lập Kế hoạch Thực hiện (`plan_train_kaggle.md`)**, kính mời bạn xem xét và xác nhận phương án cấu trúc mã nguồn trong Notebook:

- **Lựa chọn 1 (Khuyến nghị - Hybrid / Tái sử dụng Module sạch)**:
  Notebook sẽ thêm đường dẫn project vào `sys.path` để import trực tiếp từ `src.models1`, `src.dataset2`, `configs.config` (yêu cầu người dùng clone repo hoặc thêm repo vào Kaggle Input/Working). Đây là cách làm chuẩn mực của kỹ thuật phần mềm, mã nguồn ngắn gọn, dễ bảo trì.
- **Lựa chọn 2 (Standalone Self-Contained Notebook)**:
  Tích hợp trực tiếp toàn bộ code của `ConvGRUClassifier`, `SpatialReductionNeck`, `ChunkedBackboneNeckExtractor`, `RawVideoDataset` vào chính các cell của Notebook. Ưu điểm là notebook có thể copy-paste chạy độc lập trên bất kỳ tài khoản Kaggle nào mà không cần đính kèm repo, chỉ cần cung cấp link dataset và weight backbone.

---

## 6. Kết luận & Hành động Tiếp theo

Tài liệu phân tích này đã khảo sát toàn diện các yêu cầu kỹ thuật từ [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py) và các ràng buộc đặc thù của môi trường Kaggle.

👉 **Kính mời bạn duyệt và phản hồi về nội dung phân tích trên (hoặc chọn Lựa chọn 1 hay 2) để Agent tiến hành Bước 2: Tạo kế hoạch thực hiện (`plan_train_kaggle.md`).**
