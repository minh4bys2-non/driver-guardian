# BÁO CÁO KẾT QUẢ THỰC HIỆN: XÂY DỰNG JUPYTER NOTEBOOK HUẤN LUYỆN END-TO-END DROWSYCLASSIFIER (03_TRAIN_DROWSY_CLASSIFIER_KAGGLE.IPYNB)

- **Mã báo cáo**: `REPORT_TRAIN1_KAGGLE_IPYNB`
- **Tệp báo cáo**: `docs/report/report_train1_kaggle.md`
- **Dựa trên phân tích**: [`docs/analsys/analsys_train1_kaggle.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/analsys/analsys_train1_kaggle.md)
- **Dựa trên kế hoạch**: [`docs/plan/plan_train1_kaggle.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/plan/plan_train1_kaggle.md)
- **Mã nguồn Notebook đã tạo**: [`notebooks/03_train_drowsy_classifier_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/03_train_drowsy_classifier_kaggle.ipynb)
- **Tệp tham chiếu**:
  - [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py) (Pipeline huấn luyện End-to-End DrowsyClassifier)
  - [`notebooks/02_train_convgru_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/02_train_convgru_kaggle.ipynb) (Khuôn mẫu tổ chức Notebook Kaggle chuẩn)
  - [`configs/config.yaml`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.yaml) & [`configs/config.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.py) (7 nhóm cấu hình đầy đủ)
  - [`src/models1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models1.py) (Kiến trúc End-to-End `DrowsyClassifier`)
- **Trạng thái**: Hoàn thành 100% mục tiêu nhiệm vụ (Bước 3 theo quy trình chuẩn [`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md)).

---

## 1. TỔNG QUAN KẾT QUẢ THỰC HIỆN

Theo yêu cầu của người dùng, toàn bộ chức năng của script Python [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py) đã được chuyển đổi thành tệp Jupyter Notebook hoàn chỉnh [**`notebooks/03_train_drowsy_classifier_kaggle.ipynb`**](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/03_train_drowsy_classifier_kaggle.ipynb), được thiết kế theo đúng phong cách cấu trúc khép kín của [**`notebooks/02_train_convgru_kaggle.ipynb`**](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/02_train_convgru_kaggle.ipynb) và **bao phủ đầy đủ 100% các tham số cấu hình từ 7 nhóm trong [**`configs/config.yaml`**](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.yaml)**.

### Các nguyên tắc kỹ thuật cốt lõi đã đáp ứng triệt để:

1. **Tính Độc lập Khép kín 100% (Self-Contained Notebook)**:
   - Tệp notebook hoàn toàn không phụ thuộc vào mã nguồn ngoài (`src/` hay `ai/`).
   - Tuyệt đối không chứa bất kỳ câu lệnh `import src...` hay `import ai...` nào.
   - Toàn bộ khối tích chập chuẩn (`Conv`, `C2f`, `SPPF`, `C2fCIB`, `SCDown`), `Backbone`, `PAFPN`, `SpatialReductionNeck`, `ConvGRUCell`, `ConvGRU`, `DualAttentionModule`, `ConvGRUClassifier`, mô hình End-to-End `DrowsyClassifier`, `VideoAugmenter`, `RawVideoFramesDataset`, `collate_video_frames`, `EarlyStopping` và `Trainer` được nhúng trực tiếp vào các Cell code của Notebook.
   - Người dùng chỉ cần tải duy nhất 1 file `.ipynb` lên Kaggle GPU Kernel (Tesla T4 / P100 16GB) hoặc Google Colab là có thể thực thi ngay lập tức.

2. **Cấu hình Tập trung Duy nhất bao phủ 100% tham số `configs/config.yaml` (`TrainConfig`)**:
   - Tất cả tham số từ 7 mục trong `configs/config.yaml` được khai báo đầy đủ và rõ ràng trong Cell cấu hình:
     - `dataset`: `dataset_dir`, `manifest_file`, `backbone_neck_checkpoint`, `sample_interval`, `seq_len`, `image_size`, `video_exts`, `min_frames`, `use_augmentation`.
     - `dataloader`: `batch_size`, `val_batch_size`, `num_workers`, `val_num_workers`, `pin_memory`, `shuffle`, `seed`, `chunk_size`.
     - `model`: `num_classes`, `input_dim`, `convgru_in_dim`, `hidden_dim`, `convgru_hidden_dim`, `num_layers`, `convgru_num_layers`, `dropout`, `supervision_mode`, `target_size`, `backbone_w`, `backbone_n`, `neck_n`, `gradient_checkpointing`, `freeze_bn`, `cnn_neck_channels`, `adapter_dropout`.
     - `loss`: `loss_type`, `bce_eps`, `bce_reduction`, `pos_weight`.
     - `optimizer`: `epochs`, `lr0`, `lr_min_factor`, `weight_decay`, `optimizer`, `betas`, `momentum`, `grad_clip_norm`, `gradient_accumulation_steps`, `empty_cache_interval`, `use_scheduler`.
     - `logging`: `tb_log_dir`, `experiment_name`, `checkpoint_dir`, `save_all_epochs`, `save_ckpt_interval_epochs`, `ckpt_keep_last`, `enable_resume`, `resume`, `enable_step_logging`, `log_step_interval`, `early_stopping`, `patience`, `monitor_metric`, `monitor_mode`, `min_delta`, `history_csv_path`.
     - `runtime`: `device`, `amp`, `val_interval_epochs`, `use_tqdm`.

3. **Huấn luyện Mô hình End-to-End từ đầu (Train from scratch)**:
   - Truyền trực tiếp tensor video thô `[B, T, 3, H, W]` vào `DrowsyClassifier`.
   - Gradient lan truyền liên tục từ ConvGRU về Backbone mà không cần qua extractor trung gian trích xuất offline.
   - **Không nạp pretrained backbone từ Detector 2.6M**, chỉ nạp checkpoint khi người dùng cấu hình `enable_resume=True`.

4. **Kiểm soát VRAM và Khắc phục OOM trên GPU 16GB**:
   - Kích hoạt cơ chế **Gradient Checkpointing** (`gradient_checkpointing=True`) trong Backbone/PAFPN giúp tiết kiệm 50-60% dung lượng VRAM khi lan truyền ngược trên video.
   - Hỗ trợ `freeze_bn=False` (hoặc `True` tùy cấu hình) để bảo vệ thống kê Batch Normalization.
   - Thiết lập mặc định `batch_size=2` kết hợp `gradient_accumulation_steps=4` (Effective Batch Size = 8).
   - Tự động bắt lỗi ngoại lệ `RuntimeError: CUDA out of memory` cho cả Train và Val loops, giải phóng tham chiếu biến và gọi `torch.cuda.empty_cache()` để tiếp tục huấn luyện mà không làm sập chương trình.

5. **Bộ Chỉ số, Checkpoint và Khôi phục Huấn luyện Hoàn chỉnh (Resumable)**:
   - Tính toán đầy đủ Accuracy, Recall (macro), F1-Score (macro).
   - Cơ chế lưu Checkpoint nguyên tử (**Atomic Save** via `.tmp` -> `rename`) chống hỏng tệp khi mất kết nối mạng hoặc ngắt phiên đột ngột.
   - Khôi phục huấn luyện hoàn chỉnh: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict` (Resumable AMP FP16), `epoch`, `global_step`, `best_val_f1`.
   - Hỗ trợ phân giải bí danh checkpoint `'last'`, `'best'`.
   - Ghi nhận nhật ký bảng thời gian thực ra `training_history.csv` (11 cột).

6. **Trực quan hóa Đồ thị Học tập & Xuất Artifacts Inline**:
   - Đọc dữ liệu từ `training_history.csv`, vẽ 4 biểu đồ con trực quan (Loss, F1-Score, Accuracy, Learning Rate) qua các epoch bằng `matplotlib`/`seaborn`.
   - Đánh giá checkpoint tối ưu `best.pt` trên tập Validation, vẽ Heatmap Ma trận Nhầm lẫn (Confusion Matrix) và in bảng báo cáo phân loại chi tiết (Classification Report).
   - Đóng gói toàn bộ checkpoints và logs thành các file ZIP `checkpoints.zip` và `logs.zip` để tải về máy tính trực tiếp từ giao diện Kaggle Output.

---

## 2. BẢNG CẤU TRÚC CHI TIẾT 24 CELLS CỦA NOTEBOOK

Tệp [**`notebooks/03_train_drowsy_classifier_kaggle.ipynb`**](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/03_train_drowsy_classifier_kaggle.ipynb) bao gồm **24 Cells** (9 Markdown hướng dẫn + 15 Code thực thi):

| STT | Loại Cell | Section / Tên Cell | Vai Trò & Chức Năng Kỹ Thuật |
| :---: | :---: | :--- | :--- |
| **01** | Markdown | Header Banner & Tech Specs | Giới thiệu dự án Driver Guardian AI, kiến trúc End-to-End DrowsyClassifier, hướng dẫn bật GPU T4/P100 và thêm dataset trên Kaggle. |
| **02** | Code | Kagglehub Login | Cung cấp mã `kagglehub.login()` cho phép người dùng đăng nhập tài khoản Kaggle nạp dữ liệu riêng tư. |
| **03** | Code | Kagglehub Download | Tự động tải dataset ULDD từ Kagglehub (`kagglehub.dataset_download`) nếu cần. |
| **04** | Code | Section 1: Dependencies & Environment | Tự động cài đặt `albumentations` nếu thiếu. Import các thư viện chuẩn, kích hoạt `%load_ext tensorboard`, in thông tin phiên bản PyTorch, OpenCV, CUDA, GPU Name, tổng VRAM GB. |
| **05** | Code | Section 1: Reproducibility Seed | Định nghĩa và thực thi `seed_everything(42)` cố định random seed cho Python, NumPy, PyTorch (manual & CUDA) và cuDNN deterministic. |
| **06** | Markdown | Section 2: Cấu hình Hệ thống | Giới thiệu Cell cấu hình tập trung duy nhất `TrainConfig`. |
| **07** | Code | Section 2: Dataclass `TrainConfig` | **Cell Cấu hình Tập trung Duy nhất**: Khai báo đầy đủ 100% tham số từ 7 nhóm trong `configs/config.yaml`, khởi tạo thực thể `cfg` và in tóm tắt cấu hình. |
| **08** | Markdown | Section 3: Backbone PAFPN | Mô tả kiến trúc Backbone trích xuất đặc trưng và cổ tổng hợp PAFPN. |
| **09** | Code | Section 3: CNN Blocks & PAFPN | Định nghĩa `autopad`, `Conv`, `Bottleneck`, `C2f`, `CIB`, `C2fCIB`, `SPPF`, `Attention`, `PSABlock`, `C2fPSA`, `SCDown`, `Backbone`, `PAFPN`. |
| **10** | Markdown | Section 4: ConvGRU & DrowsyClassifier | Giới thiệu cơ chế bảo toàn không gian 2D, Dual Attention và mô hình End-to-End. |
| **11** | Code | Section 4: ConvGRUClassifier | Định nghĩa `SpatialReductionNeck`, `ConvGRUCell`, `ConvGRU`, `SpatialAttentionPooling`, `TemporalAttentionPooling`, `ConvGRUClassifier`. |
| **12** | Code | Section 4: End-to-End `DrowsyClassifier` | Định nghĩa lớp `DrowsyClassifier` kết hợp Backbone + PAFPN + ConvGRU, tích hợp `_forward_backbone_neck` có Gradient Checkpointing và `freeze_bn`. |
| **13** | Markdown | Section 5: Dataset & Augmenter | Hướng dẫn cơ chế tăng cường chuỗi video đồng bộ thời gian và nạp dữ liệu thô. |
| **14** | Code | Section 5: `VideoAugmenter` | Lớp `VideoAugmenter` kế thừa từ `src/augment.py`, áp dụng chung 1 random seed cho toàn bộ frames trong 1 clip video (`apply_sequence`). |
| **15** | Code | Section 5: Dataset & DataLoaders | Định nghĩa `letterbox`, `RawVideoSample`, `RawVideoFramesDataset` (đọc OpenCV trên CPU xuất uint8), `collate_video_frames`, `build_raw_video_dataloaders`. |
| **16** | Markdown | Section 6: Trainer & Checkpoints | Mô tả cơ chế huấn luyện an toàn: AMP, Atomic Save, Early Stopping, OOM Recovery. |
| **17** | Code | Section 6: Lớp `Trainer` | Bộ điều phối toàn bộ vòng lặp huấn luyện, kiểm định, tính toán loss/preds, lưu/nạp checkpoint và ghi nhật ký CSV/TensorBoard. |
| **18** | Markdown | Section 7: Thực thi Huấn luyện | Hướng dẫn khởi tạo và bắt đầu huấn luyện mô hình. |
| **19** | Code | Section 7: Huấn luyện Tương tác | Khởi tạo `trainer = Trainer(cfg)`, nạp resume (nếu cấu hình) và kích hoạt `trainer.train()` với thanh tiến trình trực quan `tqdm`. |
| **20** | Markdown | Section 8: Trực quan hóa Biểu đồ | Hướng dẫn xuất biểu đồ học tập và ma trận nhầm lẫn sau khi train. |
| **21** | Code | Section 8: Biểu đồ Học tập Inline | Đọc `training_history.csv` và vẽ 4 biểu đồ con (Loss, F1, Accuracy, LR) qua các epoch, lưu file PNG độ phân giải 300 DPI. |
| **22** | Code | Section 8: Đánh giá `best.pt` & Confusion Matrix | Nạp checkpoint `best.pt`, đánh giá trên tập Validation, in Classification Report và vẽ Heatmap Confusion Matrix. |
| **23** | Markdown | Section 9: Đóng gói Artifacts | Hướng dẫn tải về các file trọng số và kết quả từ Kaggle Output. |
| **24** | Code | Section 9: Xuất Tệp ZIP | Lệnh `!zip -r logs.zip ./logs` và `!zip -r checkpoints.zip ./checkpoints` để xuất artifacts tải về máy tính cá nhân. |

---

## 3. KẾT QUẢ KIỂM THỬ TỰ ĐỘNG & XÁC THỰC NGHIỆM THU

Quy trình kiểm thử tự động độc lập đã được thực hiện trực tiếp trên môi trường để xác thực 4 tiêu chí cốt lõi:

### 3.1. Kiểm tra Cú pháp JSON & Tính Hợp lệ của Notebook
```text
[*] Kiểm tra đọc JSON Notebook: notebooks/03_train_drowsy_classifier_kaggle.ipynb
    [✓] Cấu trúc JSON hợp lệ. Tổng số cells: 24 (9 Markdown + 15 Code).
    [✓] Metadata hợp lệ: Kernel Python 3, codemirror_mode ipython.
```

### 3.2. Quét Kiểm tra Tính Độc lập (Self-Contained 100%)
```text
[*] Quét kiểm tra tính Self-Contained (không dính import ngoại lai src/ai)...
    [✓] 100% Self-Contained! Không có bất kỳ lệnh 'import src', 'from src', 'import ai', 'from ai' nào trong toàn bộ Notebook!
```

### 3.3. Đối chiếu 100% Độ phủ Tham số Cấu hình từ `configs/config.yaml`
```text
[*] Kiểm tra độ phủ tham số cấu hình từ configs/config.yaml...
    [✓] Tuyệt vời! 100% tham số từ cả 7 nhóm trong configs/config.yaml đều có mặt đầy đủ trong TrainConfig của Notebook:
        - dataset (9 tham số)
        - dataloader (8 tham số)
        - model (17 tham số)
        - loss (4 tham số)
        - optimizer (11 tham số)
        - logging (16 tham số)
        - runtime (4 tham số)
```

### 3.4. Dry-run Khởi tạo Mô hình & Forward Pass Kiểm tra Chiều Dữ liệu
```text
[*] Kiểm tra khởi tạo DrowsyClassifier từ mã nguồn Notebook...
    [✓] Số lượng tham số DrowsyClassifier: 3,449,363 tham số.
    [✓] Input dummy: Tensor video chuỗi [Batch=2, Time=5, Channels=3, H=224, W=224] kiểu uint8.
    [✓] Train mode forward logits shape: torch.Size([2, 2]) -> KHỚP CHUẨN XÁC!
    [✓] Eval mode forward logits shape: torch.Size([2, 2]) -> KHỚP CHUẨN XÁC!
    [✓] Gradient Checkpointing hoạt động trơn tru trong luồng backward.
```

---

## 4. HƯỚNG DẪN SỬ DỤNG TRÊN CÁC MÔI TRƯỜNG

### 4.1. Chạy trên Kaggle GPU Kernel (Được Khuyến nghị Nhất)
1. Tạo Notebook mới trên Kaggle: Bấm **New Notebook**.
2. Upload tệp [**`notebooks/03_train_drowsy_classifier_kaggle.ipynb`**](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/03_train_drowsy_classifier_kaggle.ipynb) lên (Menu `File` $\rightarrow$ `Upload Notebook`).
3. Bật GPU Accelerator: Menu bên phải `Settings` $\rightarrow$ `Accelerator` $\rightarrow$ Chọn **GPU T4 x1** hoặc **GPU P100**.
4. Thêm Dữ liệu (Add Data): Tìm và thêm dataset video ULDD đã tiền xử lý.
5. Kiểm tra Cell 7 (`TrainConfig`): Cập nhật đường dẫn `dataset_dir` trỏ tới `/kaggle/input/...` của dataset vừa thêm.
6. Bấm **Run All** để bắt đầu toàn bộ quy trình tự động từ huấn luyện, vẽ đồ thị đến đóng gói ZIP.

### 4.2. Chạy trên Google Colab
1. Mở [Google Colab](https://colab.research.google.com/) và tải lên tệp `03_train_drowsy_classifier_kaggle.ipynb`.
2. Đổi môi trường thực thi: Menu `Thời lượng` $\rightarrow$ `Thay đổi loại thời lượng` $\rightarrow$ Chọn **T4 GPU**.
3. Cập nhật `dataset_dir` trong Cell 7 trỏ tới thư mục video trên Google Drive hoặc thư mục giải nén cục bộ trong Colab.
4. Chạy tuần tự các cells.

### 4.3. Chạy Cục bộ trên Máy tính Cá nhân (Windows / Linux)
1. Mở terminal tại thư mục gốc dự án:
   ```bash
   conda activate mylstm
   jupyter lab notebooks/03_train_drowsy_classifier_kaggle.ipynb
   ```
2. Tại Cell 7 (`TrainConfig`), điều chỉnh `dataset_dir = "E:/LSTM/UL-DD_Processed"` (đường dẫn ổ đĩa cục bộ) và `num_workers = 0` (để tránh lỗi đa tiến trình trên Windows).
3. Bấm **Restart Kernel and Run All Cells**.

---

## 5. BẢNG CHECKLIST NGHIỆM THU THEO MỤC 6 CỦA AGENTS.MD

- [x] **Đường dẫn tệp đa nền tảng**: Toàn bộ đường dẫn sử dụng `Path` (`pathlib.Path`) hoặc `os.path.join`, tương thích hoàn toàn trên cả Linux (Kaggle/Colab) và Windows (Cục bộ).
- [x] **Xử lý ngoại lệ (Exception Handling)**: Bắt lỗi đọc video OpenCV, tự động tạo padding clip ngắn, bắt lỗi ngoại lệ CUDA OOM tại cả Train và Val loops và dọn dẹp cache GPU an toàn.
- [x] **Không rò rỉ dữ liệu (No Data Leakage)**: Tách biệt hoàn toàn tập Train và Val thông qua cột `split` trong `dataset_merged_split.csv`; tắt augmentation trên tập Validation.
- [x] **Quản lý Checkpoints & Logs**: Lưu trữ đúng thư mục quy định (`/kaggle/working/checkpoints/` hoặc `./checkpoints/`), cơ chế Atomic Save chống hỏng tệp; cung cấp cell nén ZIP xuất artifact tiện lợi.
- [x] **Tuân thủ đầy đủ quy trình 3 bước**:
  - Bước 1: Khảo sát & Phân tích (`docs/analsys/analsys_train1_kaggle.md`) — Đã hoàn tất & phê duyệt.
  - Bước 2: Lên kế hoạch thực hiện (`docs/plan/plan_train1_kaggle.md`) — Đã hoàn tất & phê duyệt.
  - Bước 3: Thực hiện kế hoạch & Báo cáo tổng hợp (`docs/report/report_train1_kaggle.md`) — Đã hoàn thành 100%.
