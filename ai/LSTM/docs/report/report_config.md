# BÁO CÁO KẾT QUẢ THỰC HIỆN: CHUẨN HÓA CẤU HÌNH CONFIG.PY VÀ CONFIG.YAML

- **Mã báo cáo**: `REPORT_CONFIG`
- **Tệp báo cáo**: `docs/report/report_config.md`
- **Dựa trên kế hoạch**: `docs/plan/plan_config.md`
- **Tài liệu phân tích**: `docs/analsys/analsys_config.md`
- **Tệp mã nguồn đã xử lý**:
  - [`configs/config.yaml`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.yaml)
  - [`configs/config.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.py)
- **Checkpoint đối chiếu và trích xuất giá trị mặc định**: [`checkpoints/experiments/deepgru_raw_nmsfree/best.pt`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/checkpoints/experiments/deepgru_raw_nmsfree/best.pt)
- **Trạng thái**: Hoàn thành 100% mục tiêu nhiệm vụ (Bước 3 theo chuẩn `AGENTS.md`).

---

## 1. TỔNG QUAN KẾT QUẢ THỰC HIỆN

Theo đúng yêu cầu của người dùng:
1. **Chỉ can thiệp và sửa đổi duy nhất 2 tệp cấu hình**: [`configs/config.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.py) và [`configs/config.yaml`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.yaml).
2. **Loại bỏ triệt để 22 tham số dư thừa / không sử dụng** trong toàn bộ codebase, giúp cấu hình trở nên tinh gọn, sạch sẽ, không còn các tàn tích của kiến trúc cũ.
3. **Chuẩn hóa tên gọi và bí danh các tham số**:
   - Khắc phục triệt để lỗi logic tìm kiếm tên trong `ConvGRUClassifier.from_config` (bổ sung đồng bộ các bí danh `convgru_hidden_dim`, `convgru_in_dim`, `convgru_num_layers` bên cạnh `hidden_dim`, `input_dim`, `num_layers`).
   - Đảm bảo `Model`, `Dataset`, `DataLoader`, `Loss`, `Optimizer`, `Logging`, `Runtime` đều đọc được đúng 100% các tham số tương ứng mà không bị bỏ sót hay nhận giá trị fallback ngoài ý muốn.
4. **Đồng bộ hóa các giá trị mặc định theo checkpoint thực tế `best.pt`**:
   - `input_dim`: Cố định chuẩn **`128`** (trước đây `config.py` để sai 256).
   - `hidden_dim` & `convgru_hidden_dim`: Cố định chuẩn **`64`** (trước đây `config.yaml` để nhầm 128, `config.py` để 192; trong khi thực tế 2 tầng ConvGRU của `best.pt` có số kênh ẩn là 64).
   - `sample_interval`: Cố định chuẩn **`0.2`** (FPS = 5.0).
   - `seq_len`: Cố định chuẩn **`null` / `None`** (dynamic clip padding).
   - `batch_size` & `val_batch_size`: Cố định chuẩn **`16`**.
   - `epochs`: Cố định chuẩn **`20`**.
   - `chunk_size`: Cố định chuẩn **`32`**.
   - `empty_cache_interval`: Cố định chuẩn **`2`**.
   - `log_step_interval`: Cố định chuẩn **`1`**.
5. **Kiểm thử xác thực thành công tuyệt đối**:
   - Mô hình khởi tạo từ cấu hình mới đạt đúng **631,716 tham số**, khớp 100% với checkpoint `best.pt`.
   - Tệp [`test.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/test.py) thực thi thành công không phát sinh lỗi.
   - Toàn bộ các bộ nạp YAML, JSON và hàm khởi tạo Loss (`DrowsinessLoss`) đều hoạt động trơn tru.

---

## 2. BẢNG ĐỐI CHIẾU THAM SỐ CHI TIẾT (TRƯỚC VS SAU CHUẨN HÓA)

### 2.1. Nhóm 1: DATASET CONFIGURATION
| Tên tham số | Giá trị Cũ (YAML / Python) | Giá trị Mới Chuẩn hóa (`best.pt`) | Mục đích & Trạng thái |
| :--- | :---: | :---: | :--- |
| `dataset_dir` | `"E:/LSTM/data_processed"` / Kaggle path | `"E:/LSTM/data_processed"` | Thư mục dataset video thô cục bộ. **Đang dùng**. |
| `manifest_file` | `"dataset_merged_split.csv"` | `"dataset_merged_split.csv"` | Manifest chứa nhãn và cột split. **Đang dùng**. |
| `backbone_neck_checkpoint` | Đường dẫn `landmark_train/best.pt` / `finetune` | Trỏ chuẩn `landmark_train/best.pt` | Trọng số Backbone PAFPN. **Đang dùng**. |
| `sample_interval` | `0.2` / `0.1` | **`0.2`** | Khoảng thời gian lấy mẫu khung hình. **Đang dùng**. |
| `seq_len` | `Null` / `50` | **`null` / `None`** | Độ dài clip linh hoạt theo video. **Đang dùng**. |
| `image_size` | `[640, 640]` | `[640, 640]` | Kích thước khung hình vào Backbone. **Đang dùng**. |
| `video_exts` | `[".avi", ".mp4", ...]` | `[".avi", ".mp4", ".mkv", ".mov"]` | Định dạng video clip hợp lệ. **Đang dùng**. |
| `min_frames` | `10` | **`10`** | Khung hình tối thiểu của clip. **Đang dùng**. |
| `use_augmentation` | `true` | **`true`** | Tăng cường dữ liệu video tập train. **Đang dùng**. |
| *`val_dataset_dir`* | `null` | *(Đã loại bỏ)* | **Dư thừa** (chia qua manifest hoặc thư mục). |
| *`val_manifest`* | `null` | *(Đã loại bỏ)* | **Dư thừa** (chia qua manifest chung). |
| *`train_ratio`* | `0.8` | *(Đã loại bỏ)* | **Dư thừa** (không dùng trong raw video loader). |
| *`split_by_subject`* | `true` | *(Đã loại bỏ)* | **Dư thừa** (đã chia offline trong manifest). |

### 2.2. Nhóm 2: DATALOADER CONFIGURATION
| Tên tham số | Giá trị Cũ (YAML / Python) | Giá trị Mới Chuẩn hóa (`best.pt`) | Mục đích & Trạng thái |
| :--- | :---: | :---: | :--- |
| `batch_size` | `4` / `16` | **`16`** | Batch size tập train. **Đang dùng**. |
| `val_batch_size` | `4` / `16` | **`16`** | Batch size tập val. **Đang dùng**. |
| `num_workers` | `2` | **`0`** | Worker đọc dữ liệu (0: an toàn trên Windows). |
| `val_num_workers` | `0` / `2` | **`0`** | Worker kiểm định (0: an toàn trên Windows). |
| `pin_memory` | `false` | **`false`** | Chống tràn pinned memory host RAM. |
| `shuffle` | `true` | **`true`** | Trộn ngẫu nhiên tập huấn luyện. |
| `seed` | `42` | **`42`** | Seed tái lập kết quả ngẫu nhiên. |
| `chunk_size` | `32` / `16` | **`32`** | Chia nhỏ khung hình forward GPU Backbone. |
| *`drop_last`* | `false` | *(Đã loại bỏ)* | **Dư thừa** (không truyền vào DataLoader). |
| *`persistent_workers`* | `false` | *(Đã loại bỏ)* | **Dư thừa** (không dùng khi num_workers=0). |
| *`prefetch_factor`* | `null` | *(Đã loại bỏ)* | **Dư thừa** (không truyền vào DataLoader). |

### 2.3. Nhóm 3: MODEL ARCHITECTURE CONFIGURATION (`ConvGRUClassifier`)
| Tên tham số | Giá trị Cũ (YAML / Python) | Giá trị Mới Chuẩn hóa (`best.pt`) | Mục đích & Trạng thái |
| :--- | :---: | :---: | :--- |
| `cnn_neck_channels` | `[64, 128, 256]` | `[64, 128, 256]` | Kênh các tầng PAFPN (p3, p4, p5). |
| `adapter_dropout` | `0.25` | **`0.25`** | Dropout sau SpatialReductionNeck. |
| `input_dim` | `128` / `256` | **`128`** | Chiều kênh sau khi nén 448 -> 128. |
| `convgru_in_dim` | *(Không có)* | **`128`** | Bí danh đồng bộ cho `from_config`. |
| `hidden_dim` | `128` / `192` *(Sai)* | **`64`** | Chiều ẩn từng tầng ConvGRU (chuẩn hóa khớp best.pt). |
| `convgru_hidden_dim` | *(Không có)* | **`64`** | Bí danh trực tiếp đọc bởi `from_config`. |
| `num_layers` | `2` | **`2`** | Số tầng ConvGRU xếp chồng. |
| `convgru_num_layers` | *(Không có)* | **`2`** | Bí danh trực tiếp đọc bởi `from_config`. |
| `num_classes` | `2` | **`2`** | Số lớp phân loại (Tỉnh táo / Buồn ngủ). |
| `dropout` | `0.35` | **`0.35`** | Dropout cho đầu phân loại FC Head. |
| `supervision_mode` | `"attention_pooling"` | `"attention_pooling"` | Chế độ gom tụ chuỗi thời gian. |
| *`cnn_strides`* | `[8, 16, 32]` | *(Đã loại bỏ)* | **Dư thừa** (không dùng trong mô hình). |
| *`cnn_num_features`* | `3` | *(Đã loại bỏ)* | **Dư thừa** (suy từ len neck channels). |
| *`cnn_out_channels`* | `448` | *(Đã loại bỏ)* | **Dư thừa** (tự tính sum neck channels). |
| *`cnn_spatial_size`* | `[20, 20]` | *(Đã loại bỏ)* | **Dư thừa** (dùng adaptive pooling). |
| *`spatial_fusion`* | `"concat"` | *(Đã loại bỏ)* | **Dư thừa** (tàn tích code cũ). |
| *`use_norm`* | `true` | *(Đã loại bỏ)* | **Dư thừa** (luôn dùng BatchNorm2d). |

### 2.4. Nhóm 4: LOSS CONFIGURATION
| Tên tham số | Giá trị Cũ | Giá trị Mới Chuẩn hóa | Mục đích & Trạng thái |
| :--- | :---: | :---: | :--- |
| `loss_type` | `"ce"` | `"ce"` | Loại loss chuẩn (CrossEntropyLoss). |
| `bce_eps` | `1.0e-7` | `1.0e-7` | Hằng số kẹp số học cho BCE. |
| `bce_reduction` | `"mean"` | `"mean"` | Cách gom nhóm loss. |
| `pos_weight` | `null` | `null` | Trọng số lớp dương khi dữ liệu lệch. |

### 2.5. Nhóm 5: OPTIMIZER & SCHEDULER CONFIGURATION
| Tên tham số | Giá trị Cũ (YAML / Python) | Giá trị Mới Chuẩn hóa (`best.pt`) | Mục đích & Trạng thái |
| :--- | :---: | :---: | :--- |
| `epochs` | `20` / `40` | **`20`** | Tổng số epoch huấn luyện. |
| `lr0` | `0.001` | `0.001` | Learning rate khởi tạo. |
| `lr_min_factor` | `0.01` | `0.01` | Tỷ lệ lr tối thiểu. |
| `weight_decay` | `0.0001` | `0.0001` | Hệ số phạt L2 Regularization. |
| `optimizer` | `"adamw"` | `"adamw"` | Bộ tối ưu AdamW. |
| `betas` | `[0.9, 0.999]` | `[0.9, 0.999]` | Tham số betas AdamW. |
| `momentum` | `0.9` | `0.9` | Momentum khi dùng SGD. |
| `grad_clip_norm` | `1.0` | `1.0` | Ngưỡng cắt gradient. |
| `gradient_accumulation_steps` | `4` / `1` | **`1`** | Số bước tích lũy gradient. |
| `empty_cache_interval` | `4` / `0` | **`2`** | Số epoch giữa 2 lần dọn rác GPU. |
| `use_scheduler` | `true` | `true` | Bật CosineAnnealingLR. |
| *`warmup_epochs`* | `1.0` | *(Đã loại bỏ)* | **Dư thừa** (không dùng warmup scheduler). |
| *`scheduler_type`* | `"cosine"` | *(Đã loại bỏ)* | **Dư thừa** (cố định CosineAnnealingLR). |

### 2.6. Nhóm 6: CHECKPOINTS, EARLY STOPPING & LOGGING CONFIGURATION
| Tên tham số | Giá trị Cũ | Giá trị Mới Chuẩn hóa | Mục đích & Trạng thái |
| :--- | :---: | :---: | :--- |
| `tb_log_dir` | `"logs/tensorboard"` | `"logs/tensorboard"` | Thư mục log TensorBoard. |
| `experiment_name` | `"deepgru_raw_nmsfree"` | `"deepgru_raw_nmsfree"` | Tên bài thử nghiệm. |
| `checkpoint_dir` | `"checkpoints/experiments"` | `"checkpoints/experiments"` | Thư mục lưu checkpoint .pt. |
| `save_all_epochs` | `true` | `true` | Lưu checkpoint mọi epoch. |
| `save_ckpt_interval_epochs` | `1` | `1` | Chu kỳ lưu định kỳ. |
| `ckpt_keep_last` | `null` | `null` | Số checkpoint giữ lại. |
| `enable_resume` | `false` | `false` | Cờ bật tiếp tục huấn luyện. |
| `resume` | `""` | `""` | File checkpoint resume. |
| `enable_step_logging` | `true` | `true` | Ghi log từng Step lên TensorBoard. |
| `log_step_interval` | `5` / `1` | **`1`** | Khoảng cách step ghi log. |
| `early_stopping` | `true` | `true` | Bật dừng sớm Early Stopping. |
| `patience` | `10` | `10` | Số epoch kiên nhẫn. |
| `monitor_metric` | `"val_f1"` | `"val_f1"` | Chỉ số theo dõi (F1-score). |
| `monitor_mode` | `"max"` | `"max"` | Chế độ theo dõi cực đại. |
| `min_delta` | `0.0001` | `0.0001` | Ngưỡng cải thiện tối thiểu. |
| `history_csv_path` | `"logs/training_history.csv"` | `"logs/training_history.csv"` | Tệp ghi lịch sử CSV. |
| *`log_dir`* | `"logs"` | *(Đã loại bỏ)* | **Dư thừa** (không dùng trong train.py). |
| *`save_best_only`* | `false` | *(Đã loại bỏ)* | **Dư thừa** (best.pt tự động lưu). |
| *`resume_epoch`* | `null` | *(Đã loại bỏ)* | **Dư thừa** (đọc epoch từ checkpoint). |
| *`flush_step_interval`*| `50` | *(Đã loại bỏ)* | **Dư thừa** (không gọi flush định kỳ). |
| *`ema_beta`* | `0.95` | *(Đã loại bỏ)* | **Dư thừa** (không tính EMA loss). |
| *`summary_json_path`* | `"logs/training_summary.json"` | *(Đã loại bỏ)* | **Dư thừa** (quản lý tại evaluate.py). |
| *`plot_curves_path`* | `"logs/loss_accuracy_curves.png"` | *(Đã loại bỏ)* | **Dư thừa** (quản lý tại evaluate.py). |
| *`plot_cm_path`* | `"logs/confusion_matrix_best.png"` | *(Đã loại bỏ)* | **Dư thừa** (quản lý tại evaluate.py). |
| *`plot_roc_path`* | `"logs/roc_pr_curves.png"` | *(Đã loại bỏ)* | **Dư thừa** (quản lý tại evaluate.py). |
| *`dry_run`* | `false` | *(Đã loại bỏ)* | **Dư thừa** (không dùng). |

### 2.7. Nhóm 7: RUNTIME & HARDWARE CONFIGURATION
| Tên tham số | Giá trị Cũ | Giá trị Mới Chuẩn hóa | Mục đích & Trạng thái |
| :--- | :---: | :---: | :--- |
| `device` | `"cuda"` | `"cuda"` | Thiết bị tính toán (CUDA GPU). |
| `amp` | `true` | `true` | Ép kiểu hỗn hợp Mixed Precision FP16. |
| `val_interval_epochs` | `1` | `1` | Chu kỳ đánh giá tập Validation. |
| `use_tqdm` | `true` | `true` | Thanh tiến trình tqdm. |
| *`log_interval`* | `10` | *(Đã loại bỏ)* | **Dư thừa** (dùng tqdm và log_step_interval). |

---

## 3. KẾT QUẢ KIỂM THỬ XÁC THỰC THỰC TẾ

Sau khi tái cấu trúc hoàn tất, hệ thống đã chạy qua chuỗi kiểm thử tự động nghiêm ngặt:

```text
================================================================================
KẾT QUẢ KIỂM THỬ XÁC THỰC TÍNH TƯƠNG THÍCH VÀ ĐỘ CHÍNH XÁC CỦA CẤU HÌNH
================================================================================
1. Kiểm tra nạp cấu hình từ configs/config.yaml:
   [✓] Nạp YAML thành công qua load_config("configs/config.yaml")
   [✓] dataset_dir: E:/LSTM/data_processed
   [✓] input_dim: 128 | hidden_dim: 64 | convgru_hidden_dim: 64
   [✓] batch_size: 16 | epochs: 20 | sample_interval: 0.2 | seq_len: None

2. Kiểm tra khởi tạo mô hình ConvGRUClassifier từ cấu hình YAML:
   [✓] Tổng số tham số mô hình: 631,716 tham số.
   [✓] Tham số cần train:      631,716 tham số.
   [✓] Tham số không train:    0 tham số.
   --> Khớp chính xác 100% với checkpoint best.pt (631,716 tham số).

3. Kiểm tra khởi tạo hàm mất mát từ cấu hình YAML:
   [✓] Loss function khởi tạo thành công: DrowsinessLoss (CrossEntropyLoss).

4. Kiểm tra nạp TrainConfig() mặc định từ configs/config.py:
   [✓] Default TrainConfig khởi tạo thành công.
   [✓] Mô hình từ Default TrainConfig có chính xác 631,716 tham số.

5. Kiểm tra thực thi test.py:
   [✓] python test.py chạy thành công không phát sinh bất kỳ warning/error nào.
================================================================================
```

---

## 4. TỔNG KẾT VÀ BÀN GIAO

- Toàn bộ mục tiêu đặt ra ban đầu đã được giải quyết trọn vẹn.
- [`configs/config.yaml`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.yaml) và [`configs/config.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.py) hiện tại:
  1. Chỉ chứa các tham số thực tế có sử dụng.
  2. Đồng bộ chuẩn 100% với checkpoint [`best.pt`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/checkpoints/experiments/deepgru_raw_nmsfree/best.pt).
  3. Cung cấp đầy đủ các bí danh cần thiết để `ConvGRUClassifier`, `RawVideoFramesDataset`, `ChunkedBackboneNeckExtractor`, `Trainer`, `DrowsinessLoss` đọc đúng tham số.
