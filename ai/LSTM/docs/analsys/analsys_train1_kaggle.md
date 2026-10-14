# TÀI LIỆU PHÂN TÍCH: XÂY DỰNG JUPYTER NOTEBOOK HUẤN LUYỆN END-TO-END DROWSYCLASSIFIER (KAGGLE EDITION)

- **Mã yêu cầu**: `TRAIN1_KAGGLE_IPYNB`
- **Tệp phân tích**: `docs/analsys/analsys_train1_kaggle.md`
- **Mã nguồn tham chiếu**:
  - [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py) (Pipeline huấn luyện End-to-End `DrowsyClassifier`)
  - [`notebooks/02_train_convgru_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/02_train_convgru_kaggle.ipynb) (Chuẩn mực kiến trúc Notebook Kaggle, cấu trúc các Cell, tính khép kín Self-Contained, trực quan hóa inline)
  - [`configs/config.yaml`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.yaml) & [`configs/config.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.py) (Hệ thống cấu hình 7 nhóm tham số chuẩn)
  - [`src/models1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models1.py) & [`src/models.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py) (Kiến trúc mô hình `DrowsyClassifier` và `ConvGRUClassifier`)
  - [`src/dataset.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/dataset.py) & [`src/augment.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/augment.py) (Pipeline nạp video thô và tăng cường dữ liệu video đồng bộ thời gian)
- **Tệp đầu ra dự kiến**: [`notebooks/03_train_drowsy_classifier_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/03_train_drowsy_classifier_kaggle.ipynb)
- **Quy chuẩn áp dụng**: [`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md) — **Bước 1: Khảo sát & Phân tích (Discovery)**
- **Trạng thái**: Đang chờ người dùng phê duyệt trước khi chuyển sang Bước 2 (Lên kế hoạch).

---

## 1. YÊU CẦU CỐT LÕI VÀ BỐI CẢNH DỰ ÁN

### 1.1. Yêu cầu từ người dùng
> *"phiên bản .ipynb của file @[src/train1.py] dựa trên các chức năng và cách config giống như file @[notebooks/02_train_convgru_kaggle.ipynb] . yêu cầu config đầy đủ các thông số như trong file @[configs/config.yaml] đã có"*

### 1.2. Phân tích mục tiêu kỹ thuật
1. **Chuyển đổi hoàn chỉnh `src/train1.py` sang định dạng Jupyter Notebook (`.ipynb`)**:
   - Tên tệp đề xuất: `notebooks/03_train_drowsy_classifier_kaggle.ipynb`.
   - Vận hành mượt mà trên môi trường đám mây **Kaggle GPU Kernel (Tesla T4 16GB / P100 16GB)** hoặc Google Colab cũng như máy tính cục bộ.
   - Huấn luyện mô hình phân loại trạng thái buồn ngủ toàn diện **End-to-End `DrowsyClassifier`** trực tiếp từ chuỗi video thô (Backbone + PAFPN + ConvGRUClassifier).
   - **Huấn luyện từ đầu (Train from scratch)** theo đúng định hướng kiến trúc của `src/train1.py`, không nạp pretrained detector backbone ngoại vi.

2. **Kế thừa triết lý thiết kế từ `notebooks/02_train_convgru_kaggle.ipynb`**:
   - **Tính tự lập và khép kín 100% (Self-Contained Notebook)**: Tích hợp đầy đủ các định nghĩa mạng CNN (`Conv`, `C2f`, `PAFPN`, `Backbone`), `ConvGRUClassifier`, `DrowsyClassifier`, `VideoAugmenter`, `RawVideoFramesDataset`, `collate_video_frames`, `EarlyStopping`, `Trainer` ngay trong các Cell của Notebook mà không cần phụ thuộc vào việc clone repository hay module Python ngoài.
   - **Cell cấu hình tập trung duy nhất (`TrainConfig`)**: Được tổ chức khoa học, dễ dàng tùy chỉnh đường dẫn thư mục làm việc `/kaggle/working/`, dataset `/kaggle/input/...` hoặc tải qua `kagglehub`.
   - **Trực quan hóa và đánh giá tương tác (Interactive Visualizations)**:
     - Ghi nhận `SummaryWriter` TensorBoard và thanh tiến trình `tqdm`.
     - Vẽ biểu đồ Loss, Accuracy, F1, Learning Rate inline ngay sau khi huấn luyện.
     - Đánh giá ma trận nhầm lẫn (Confusion Matrix) và bảng phân tích phân loại (Classification Report) của checkpoint tối ưu nhất (`best.pt`).
   - **Đóng gói Artifacts (Zipping & Download)**: Cell tự động nén `logs/` và `checkpoints/` thành file zip để người dùng tải về máy cá nhân tiện lợi.

3. **Cấu hình ĐẦY ĐỦ 100% các thông số từ `configs/config.yaml`**:
   - Đồng bộ trọn vẹn toàn bộ 7 nhóm cấu hình với đầy đủ các trường, kiểu dữ liệu, giá trị mặc định và chú thích rõ ràng.
   - Đảm bảo tính linh hoạt khi chạy trên Kaggle (ví dụ: `dataset_dir`, `checkpoint_dir`, `batch_size`, `gradient_accumulation_steps`).

---

## 2. PHÂN TÍCH ĐỐI CHIẾU: `02_train_convgru_kaggle.ipynb` VS `03_train_drowsy_classifier_kaggle.ipynb`

| Tiêu chí kỹ thuật | `02_train_convgru_kaggle.ipynb` (Hiện tại) | `03_train_drowsy_classifier_kaggle.ipynb` (Dự kiến tạo) | Ghi chú & Rationale kỹ thuật |
| :--- | :--- | :--- | :--- |
| **Mô hình mục tiêu** | `ConvGRUClassifier` ([`src/models.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py)) | `DrowsyClassifier` ([`src/models1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models1.py)) | `DrowsyClassifier` tích hợp toàn diện Backbone + PAFPN + ConvGRU. |
| **Luồng xử lý dữ liệu** | Giai đoạn 2: Cắt mini-chunk qua `ChunkedBackboneNeckExtractor` đóng băng, đưa $(p_3, p_4, p_5)$ vào `ConvGRU`. | End-to-End: Truyền trực tiếp tensor video `[B, T, 3, H, W]` vào `DrowsyClassifier`. Gradient lan truyền xuyên suốt toàn mạng. | Loại bỏ hoàn toàn lớp trung gian `ChunkedBackboneNeckExtractor`. |
| **Khởi tạo trọng số** | Yêu cầu nạp checkpoint `backbone_neck_checkpoint` (`best.pt` của NMSFreeDetector 2.6M). | **Huấn luyện từ đầu (Train from scratch)**. Tuyệt đối không nạp checkpoint detector. Checkpoint chỉ nạp khi resume. | Tuân thủ chỉ đạo kỹ thuật cốt lõi của `src/train1.py`. |
| **Tối ưu bộ nhớ VRAM** | Dùng mini-chunk 32 frames vì Backbone đóng băng không cần lưu đồ thị gradient. | Tích hợp **Gradient Checkpointing** (`gradient_checkpointing=True`) và tùy chọn `freeze_bn` trong `DrowsyClassifier`. | Gradient Checkpointing giúp giảm ~50-60% dung lượng VRAM khi lan truyền ngược qua toàn bộ mạng video. |
| **Kích thước ảnh mặc định** | `(640, 640)` theo detector | Hỗ trợ cấu hình `(224, 224)` hoặc `(640, 640)` theo `config.yaml` | `target_size` tự động co giãn `(H//16, W//16)` hoặc nhận trực tiếp cấu hình `(14, 14)`. |
| **Độ phủ tham số cấu hình** | Chỉ khai báo tập con tham số cần thiết cho ConvGRU | **Bao phủ 100% tất cả các tham số từ 7 mục trong `configs/config.yaml`**. | Đáp ứng chính xác yêu cầu của người dùng. |
| **Bộ nạp dữ liệu (Dataset)** | `RawVideoFramesDataset` + `VideoAugmenter` | Kế thừa nguyên vẹn `RawVideoFramesDataset` + `VideoAugmenter` đồng bộ thời gian | Tối ưu đọc OpenCV video thô, lấy mẫu đều đặn (`sample_interval=0.2s`). |
| **Cơ chế lưu Checkpoint** | Atomic Save (`.tmp` -> rename), `last.pt`, `best.pt`, `epoch_{N}.pt`, `ckpt_keep_last`. | Giữ nguyên 100% cơ chế Atomic Save và logic xoay vòng checkpoint của `src/train1.py`. | Chống hỏng tệp khi mất kết nối mạng hoặc phiên Kaggle kết thúc đột ngột. |
| **Khôi phục (Resume)** | Hỗ trợ nạp checkpoint `resume` (path hoặc alias `'last'`, `'best'`). | Kế thừa 100% cơ chế Resume đầy đủ (Model, Optimizer, Scheduler, Scaler, Epoch, Best F1). | Giúp dễ dàng tiếp tục huấn luyện trên Kaggle qua nhiều phiên (sessions). |
| **Khắc phục lỗi OOM** | Bắt lỗi CUDA OOM, `torch.cuda.empty_cache()` và bỏ qua batch. | Giữ nguyên cho cả 2 vòng lặp Train và Validation. | Đảm bảo tính bền bỉ tuyệt đối trong môi trường GPU đám mây. |
| **Trực quan hóa Inline** | Đồ thị Loss/F1/LR và Ma trận nhầm lẫn Confusion Matrix. | Giữ nguyên toàn bộ các cell vẽ đồ thị và đánh giá sau huấn luyện. | Phục vụ báo cáo và phân tích kết quả trực quan ngay trong notebook. |

---

## 3. BẢNG PHÂN TÍCH ĐỒNG BỘ 100% THAM SỐ CẤU HÌNH TỪ `CONFIGS/CONFIG.YAML`

Lớp `@dataclass class TrainConfig` trong Cell Cấu hình tập trung của Notebook sẽ chứa đầy đủ 7 nhóm tham số chuẩn xác từ [`configs/config.yaml`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.yaml):

### Nhóm 1: `dataset` (Cấu hình Dữ liệu Video Thô)
| Tham số | Kiểu dữ liệu | Giá trị mặc định trong `config.yaml` | Giá trị đề xuất trên Kaggle / Notebook | Chú thích & Ý nghĩa |
| :--- | :--- | :--- | :--- | :--- |
| `dataset_dir` | `str` | `"E:/LSTM/UL-DD_Processed"` | `r"./kagglehub/nyvantran6634/ul-dd4ni/UL-DD_Processed"` | Thư mục gốc chứa video thô (đã quét nhãn hoặc có train/val) |
| `manifest_file` | `Optional[str]` | `"dataset_merged_split.csv"` | `"dataset_merged_split.csv"` | Tệp CSV manifest danh sách video, nhãn và cột split |
| `backbone_neck_checkpoint` | `Optional[str]` | `"D:/.../best.pt"` | `""` hoặc `None` | [Tham chiếu `train.py`] Checkpoint BackboneNeck (trong `train1` không dùng nạp pretrained) |
| `sample_interval` | `float` | `0.2` | `0.2` | Khoảng thời gian lấy mẫu (giây) -> FPS hiệu dụng = 5.0 |
| `seq_len` | `Optional[int]` | `None` | `None` | Số khung hình cố định (`None` = dynamic padding theo độ dài clip) |
| `image_size` | `Tuple[int, int]` | `[224, 224]` | `(224, 224)` | Kích thước khung hình (Height, Width) đưa vào mạng |
| `video_exts` | `Tuple[str, ...]` | `[".avi", ".mp4", ".mkv", ".mov"]` | `(".avi", ".mp4", ".mkv", ".mov")` | Các định dạng video được hỗ trợ đọc |
| `min_frames` | `int` | `10` | `10` | Số khung hình tối thiểu để coi là một video hợp lệ |
| `use_augmentation` | `bool` | `true` | `True` | Bật/tắt tăng cường dữ liệu chuỗi video nhất quán thời gian |

### Nhóm 2: `dataloader` (Cấu hình Bộ Nạp Dữ Liệu)
| Tham số | Kiểu dữ liệu | Giá trị mặc định trong `config.yaml` | Giá trị đề xuất trên Kaggle / Notebook | Chú thích & Ý nghĩa |
| :--- | :--- | :--- | :--- | :--- |
| `batch_size` | `int` | `2` | `4` hoặc `2` | Kích thước batch huấn luyện (phù hợp VRAM khi train End-to-End) |
| `val_batch_size` | `int` | `2` | `4` hoặc `2` | Kích thước batch kiểm định |
| `num_workers` | `int` | `0` | `2` | Số tiến trình worker tải dữ liệu (trên Linux Kaggle đặt 2, Windows đặt 0) |
| `val_num_workers` | `int` | `0` | `2` | Số tiến trình worker tập validation |
| `pin_memory` | `bool` | `false` | `False` | Khóa vùng nhớ GPU (`False` tránh tràn pinned host memory) |
| `shuffle` | `bool` | `true` | `True` | Xáo trộn ngẫu nhiên tập huấn luyện |
| `seed` | `int` | `42` | `42` | Hạt giống ngẫu nhiên để tái lập kết quả |
| `chunk_size` | `int` | `32` | `32` | [Dành riêng `train.py`] Mini-chunk chia nhỏ frames khi trích xuất |

### Nhóm 3: `model` (Cấu hình Kiến trúc Mô hình)
| Tham số | Kiểu dữ liệu | Giá trị mặc định trong `config.yaml` | Giá trị đề xuất trên Kaggle / Notebook | Chú thích & Ý nghĩa |
| :--- | :--- | :--- | :--- | :--- |
| `num_classes` | `int` | `2` | `2` | Số lượng lớp phân loại (0: Tỉnh táo, 1: Buồn ngủ) |
| `input_dim` | `int` | `128` | `128` | Số kênh sau cổ giảm kênh `SpatialReductionNeck` |
| `convgru_in_dim` | `int` | `128` | `128` | Bí danh đồng bộ cho `input_dim` |
| `hidden_dim` | `int` | `128` | `128` | Số kênh ẩn của từng tầng ConvGRU |
| `convgru_hidden_dim` | `int` | `128` | `128` | Bí danh đồng bộ cho `hidden_dim` |
| `num_layers` | `int` | `2` | `2` | Số tầng ConvGRU xếp chồng |
| `convgru_num_layers` | `int` | `2` | `2` | Bí danh đồng bộ cho `num_layers` |
| `dropout` | `float` | `0.35` | `0.35` | Tỷ lệ Dropout cho đầu phân loại FC Head |
| `supervision_mode` | `str` | `"attention_pooling"` | `"attention_pooling"` | Cơ chế giám sát chuỗi thời gian (`'attention_pooling'` hoặc `'sequence'`) |
| `target_size` | `Tuple[int, int]` | `[14, 14]` | `(14, 14)` | Kích thước lưới không gian hội tụ (Height, Width) |
| **`backbone_w`** | `Tuple[int, ...]` | `[16, 32, 64, 128, 256]` | `(16, 32, 64, 128, 256)` | **[DÀNH RIÊNG DrowsyClassifier]** Độ rộng kênh các tầng Backbone (c1, c2, c3, c4, c5) |
| **`backbone_n`** | `Tuple[int, ...]` | `[1, 2, 2, 1]` | `(1, 2, 2, 1)` | **[DÀNH RIÊNG DrowsyClassifier]** Số khối lặp lại trong từng tầng Backbone |
| **`neck_n`** | `int` | `1` | `1` | **[DÀNH RIÊNG DrowsyClassifier]** Số khối lặp lại trong PAFPN |
| **`gradient_checkpointing`** | `bool` | `true` | `True` | **[DÀNH RIÊNG DrowsyClassifier]** Bật Gradient Checkpointing tiết kiệm VRAM |
| **`freeze_bn`** | `bool` | `false` | `False` | **[DÀNH RIÊNG DrowsyClassifier]** Khóa running mean/var của BatchNorm |
| `cnn_neck_channels` | `Tuple[int, int, int]` | `[64, 128, 256]` | `(64, 128, 256)` | [Tham chiếu `train.py`] Kênh đặc trưng (p3, p4, p5) từ PAFPN ngoài |
| `adapter_dropout` | `float` | `0.25` | `0.25` | [Tham chiếu `train.py`] Tỷ lệ Dropout sau cổ giảm kênh |

### Nhóm 4: `loss` (Cấu hình Hàm Mất Mát)
| Tham số | Kiểu dữ liệu | Giá trị mặc định trong `config.yaml` | Giá trị đề xuất trên Kaggle / Notebook | Chú thích & Ý nghĩa |
| :--- | :--- | :--- | :--- | :--- |
| `loss_type` | `str` | `"ce"` | `"ce"` | Loại loss: `"ce"` (CrossEntropyLoss) hoặc `"bce"` (BCEWithLogitsLoss) |
| `bce_eps` | `float` | `1.0e-7` | `1e-7` | Epsilon kẹp giá trị số học tránh lỗi `log(0)` |
| `bce_reduction` | `str` | `"mean"` | `"mean"` | Cách gom nhóm loss (`'mean'` hoặc `'sum'`) |
| `pos_weight` | `Optional[float]` | `null` | `None` | Trọng số cho lớp dương (1: Buồn ngủ) khi dữ liệu lệch |

### Nhóm 5: `optimizer` (Cấu hình Tối Ưu Hóa & Lịch Trình Học)
| Tham số | Kiểu dữ liệu | Giá trị mặc định trong `config.yaml` | Giá trị đề xuất trên Kaggle / Notebook | Chú thích & Ý nghĩa |
| :--- | :--- | :--- | :--- | :--- |
| `epochs` | `int` | `20` | `20` | Tổng số epoch huấn luyện |
| `lr0` | `float` | `0.001` | `0.001` | Learning rate khởi tạo |
| `lr_min_factor` | `float` | `0.01` | `0.01` | Tỷ lệ lr tối thiểu: $lr_{min} = lr0 \times 0.01 = 10^{-5}$ |
| `weight_decay` | `float` | `0.0001` | `0.0001` | Trọng số phạt L2 Regularization (Weight Decay) |
| `optimizer` | `str` | `"adamw"` | `"adamw"` | Bộ tối ưu hóa: `"adamw"`, `"adam"`, hoặc `"sgd"` |
| `betas` | `Tuple[float, float]` | `[0.9, 0.999]` | `(0.9, 0.999)` | Hệ số betas cho Adam/AdamW |
| `momentum` | `float` | `0.9` | `0.9` | Động lượng Momentum khi `optimizer="sgd"` |
| `grad_clip_norm` | `float` | `1.0` | `1.0` | Ngưỡng cắt chuẩn gradient chống nổ gradient |
| `gradient_accumulation_steps` | `int` | `4` | `4` | Số bước tích lũy gradient trước khi cập nhật trọng số |
| `empty_cache_interval` | `int` | `2` | `2` | Khoảng cách epoch dọn dẹp cache VRAM định kỳ |
| `use_scheduler` | `bool` | `true` | `True` | Bật/tắt bộ điều chỉnh học tốc CosineAnnealingLR |

### Nhóm 6: `logging` (Cấu hình Checkpoints, Early Stopping & Nhật Ký)
| Tham số | Kiểu dữ liệu | Giá trị mặc định trong `config.yaml` | Giá trị đề xuất trên Kaggle / Notebook | Chú thích & Ý nghĩa |
| :--- | :--- | :--- | :--- | :--- |
| `tb_log_dir` | `str` | `"logs/tensorboard"` | `"./logs/tensorboard"` | Thư mục lưu nhật ký TensorBoard |
| `experiment_name` | `str` | `"deepgru_raw_nmsfree"` | `"drowsy_classifier_end2end"` | Tên bài thử nghiệm huấn luyện |
| `checkpoint_dir` | `str` | `"checkpoints/experiments"` | `"./checkpoints/drowsy_end2end"` | Thư mục lưu trữ các tệp checkpoint (.pt) |
| `save_all_epochs` | `bool` | `true` | `True` | Lưu checkpoint cho tất cả các epoch |
| `save_ckpt_interval_epochs` | `int` | `1` | `1` | Khoảng epoch giữa 2 lần lưu định kỳ |
| `ckpt_keep_last` | `Optional[int]` | `null` | `5` hoặc `None` | Số lượng checkpoint định kỳ tối đa giữ lại |
| `enable_resume` | `bool` | `false` | `False` | Bật/tắt tiếp tục huấn luyện từ checkpoint |
| `resume` | `str` | `""` | `""` | Đường dẫn file .pt hoặc bí danh (`'last'`, `'best'`) để resume |
| `enable_step_logging` | `bool` | `true` | `True` | Ghi nhận TensorBoard theo từng Batch Step |
| `log_step_interval` | `int` | `1` | `5` | Số batch steps giữa các lần ghi TensorBoard |
| `early_stopping` | `bool` | `true` | `True` | Bật/tắt cơ chế dừng sớm Early Stopping |
| `patience` | `int` | `10` | `10` | Số epoch chờ không cải thiện trước khi dừng sớm |
| `monitor_metric` | `str` | `"val_f1"` | `"val_f1"` | Chỉ số theo dõi: `"val_f1"`, `"val_loss"`, `"val_acc"` |
| `monitor_mode` | `str` | `"max"` | `"max"` | Chế độ theo dõi (`"max"` hoặc `"min"`) |
| `min_delta` | `float` | `0.0001` | `0.0001` | Mức cải thiện tối thiểu để tính là tiến bộ |
| `history_csv_path` | `str` | `"logs/training_history.csv"` | `"./logs/training_history.csv"` | Tệp CSV lưu bảng lịch sử các epoch huấn luyện |

### Nhóm 7: `runtime` (Cấu hình Phần Cứng & Môi Trường Chạy)
| Tham số | Kiểu dữ liệu | Giá trị mặc định trong `config.yaml` | Giá trị đề xuất trên Kaggle / Notebook | Chú thích & Ý nghĩa |
| :--- | :--- | :--- | :--- | :--- |
| `device` | `str` | `"cuda"` | `"cuda"` | Thiết bị tính toán (`"cuda"` hoặc `"cpu"`) |
| `amp` | `bool` | `true` | `True` | Tự động ép kiểu chính xác hỗn hợp FP16 (AMP) |
| `val_interval_epochs` | `int` | `1` | `1` | Số epoch giữa 2 lần đánh giá tập Validation |
| `use_tqdm` | `bool` | `true` | `True` | Bật thanh tiến trình trực quan `tqdm` |

---

## 4. THIẾT KẾ CẤU TRÚC CHI TIẾT CÁC CELLS TRONG NOTEBOOK

Notebook dự kiến bao gồm **9 Section logic** được cấu trúc rõ ràng:

```text
notebooks/03_train_drowsy_classifier_kaggle.ipynb
│
├── [Markdown] Header Banner & Giới thiệu Kỹ thuật End-to-End DrowsyClassifier
├── [Code]     Cell 1: kagglehub.login() (Hỗ trợ nạp dataset private)
├── [Code]     Cell 2: kagglehub.dataset_download (Tải dataset video ULDD nếu cần)
├── [Code]     Cell 3: Kiểm tra môi trường (PyTorch, OpenCV, Albumentations, CUDA, GPU Info)
├── [Code]     Cell 4: Cố định Hạt giống Ngẫu nhiên (seed_everything = 42)
│
├── [Markdown] Section 2: Cấu hình Hệ thống Tập trung Duy nhất (TrainConfig)
├── [Code]     Cell 5: Dataclass TrainConfig (100% tham số config.yaml) & Khởi tạo cfg
│
├── [Markdown] Section 3: Định nghĩa Kiến trúc Backbone PAFPN
├── [Code]     Cell 6: Các khối tích chập: Conv, Bottleneck, C2f, CIB, C2fCIB, SPPF, C2fPSA, SCDown, Backbone, PAFPN
│
├── [Markdown] Section 4: Định nghĩa Kiến trúc Không gian - Thời gian & Mô hình End-to-End
├── [Code]     Cell 7: SpatialReductionNeck, ConvGRUCell, ConvGRU, TemporalAttentionPooling, ConvGRUClassifier
├── [Code]     Cell 8: Lớp mô hình hợp nhất End-to-End DrowsyClassifier (forward, gradient checkpointing, freeze_bn)
│
├── [Markdown] Section 5: Pipeline Tăng cường Dữ liệu & Nạp Video Thô
├── [Code]     Cell 9: VideoAugmenter (Temporal Consistency đồng bộ seed)
├── [Code]     Cell 10: letterbox, RawVideoFramesDataset, collate_video_frames, build_raw_video_dataloaders
│
├── [Markdown] Section 6: Hạ tầng Huấn luyện & Cơ chế Tiếp tục Huấn luyện (Trainer)
├── [Code]     Cell 11: calculate_metrics, EarlyStopping, Lớp Trainer chuyên biệt cho DrowsyClassifier:
│              ├── __init__, _init_history_csv, _log_history_csv
│              ├── compute_loss_and_preds, train_epoch, validate_epoch
│              ├── _resolve_checkpoint_path, load_checkpoint, _atomic_save, save_checkpoint
│              └── train (Vòng lặp chính với TensorBoard, CSV logging, Early Stopping, OOM recovery)
│
├── [Markdown] Section 7: Thực thi Huấn luyện Tương tác
├── [Code]     Cell 12: Khởi tạo Trainer, in tổng số tham số mô hình và bắt đầu train!
│
├── [Markdown] Section 8: Trực quan hóa & Đánh giá mô hình Sau Huấn luyện
├── [Code]     Cell 13: Vẽ biểu đồ Loss, Accuracy, F1, LR qua các epoch từ training_history.csv
├── [Code]     Cell 14: Đánh giá checkpoint tối ưu (best.pt): Confusion Matrix heatmap & Classification Report
│
└── [Markdown] Section 9: Đóng gói Checkpoints & Xuất Dữ liệu (Export Artifacts)
    └── [Code] Cell 15: Tạo tệp nén logs.zip và checkpoints.zip để tải về máy tính cá nhân
```

---

## 5. ĐÁNH GIÁ RỦI RO & PHƯƠNG ÁN XỬ LÝ (RISK MITIGATION)

1. **Rủi ro Tràn VRAM GPU (CUDA OOM) khi Huấn luyện Video End-to-End**:
   - *Nguyên nhân*: Truyền toàn bộ gradient qua cả mạng Backbone + Neck + ConvGRU trên chuỗi video tiêu tốn nhiều bộ nhớ hơn so với khi đóng băng Backbone.
   - *Giải pháp*:
     - Bật mặc định `gradient_checkpointing=True` trong `DrowsyClassifier`.
     - Sử dụng `image_size=(224, 224)` (hoặc `target_size=(14, 14)`).
     - Thiết lập `batch_size=2` kết hợp `gradient_accumulation_steps=4` (Effective batch size = 8).
     - Duy trì khối `try...except RuntimeError as e` bắt lỗi `"out of memory"`, tự động gọi `torch.cuda.empty_cache()` và bỏ qua batch mà không làm sập toàn bộ phiên huấn luyện.

2. **Rủi ro Mất mát Kết nối & Quá giới hạn Thời gian Phiên Kaggle (Kaggle Session Timeout)**:
   - *Nguyên nhân*: Phiên Kaggle GPU có giới hạn 12 tiếng liên tục hoặc có thể ngắt kết nối trình duyệt.
   - *Giải pháp*:
     - Cơ chế **Atomic Save** lưu `last.pt`, `best.pt` sau mỗi epoch.
     - Cấu hình `enable_resume=True` và `resume="last"` tại Cell cấu hình để người dùng dễ dàng bật lại và tiếp tục từ epoch bị ngắt quãng.

3. **Tính Độc lập và Tương thích Môi trường (Compatibility)**:
   - *Nguyên nhân*: Kaggle Kernel có thể khác biệt phiên bản PyTorch / Albumentations so với máy cục bộ.
   - *Giải pháp*:
     - Cell đầu tiên kiểm tra và tự động cài đặt `albumentations` nếu thiếu.
     - `VideoAugmenter` có cơ chế tương thích tự động chọn `A.Affine` hoặc `A.ShiftScaleRotate`.

---

## 6. KẾT LUẬN & ĐỀ XUẤT BƯỚC TIẾP THEO

Tài liệu phân tích trên đã xác định đầy đủ:
- Chức năng huấn luyện End-to-End từ [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py).
- Phong cách tổ chức khép kín, trực quan hóa và đóng gói từ [`notebooks/02_train_convgru_kaggle.ipynb`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/02_train_convgru_kaggle.ipynb).
- Toàn bộ 100% tham số cấu hình trong 7 nhóm từ [`configs/config.yaml`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.yaml).

Theo quy trình làm việc chuẩn trong [`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md):
- **Bước 1 (Khảo sát & Phân tích)**: Đã hoàn tất và được lưu tại `docs/analsys/analsys_train1_kaggle.md`.
- **Bước tiếp theo (Bước 2: Lên kế hoạch)**: Sẽ được tiến hành để tạo file `docs/plan/plan_train1_kaggle.md` ngay sau khi người dùng phê duyệt tài liệu phân tích này.
