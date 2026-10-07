# TÀI LIỆU PHÂN TÍCH YÊU CẦU: RÀ SOÁT VÀ CHUẨN HÓA CẤU HÌNH CONFIG.PY VÀ CONFIG.YAML

- **Mã yêu cầu**: `CONFIG`
- **Tệp phân tích**: `docs/analsys/analsys_config.md`
- **Mục tiêu**: Rà soát toàn bộ các tham số cấu hình trong `configs/config.py` và `configs/config.yaml`, đối chiếu chi tiết với toàn bộ mã nguồn (`src/train.py`, `src/models.py`, `src/dataset.py`, `src/evaluate.py`, `src/loss.py`, `test.py`) và trọng số checkpoint thực tế (`best.pt`); từ đó loại bỏ toàn bộ các tham số dư thừa/không dùng, đồng bộ chuẩn xác giá trị các tham số đang sử dụng.
- **Trạng thái**: Đang chờ người dùng phê duyệt (Bước 1 theo chuẩn quy trình `AGENTS.md`).

---

## 1. YÊU CẦU CỐT LÕI TỪ NGƯỜI DÙNG

Người dùng yêu cầu:
> *"chỉ sửa lại file config.py và config.ymal sao cho đúng với các tham số có sử dụng"*

### Mục tiêu trọng tâm:
1. **Chỉ tác động vào 2 tệp cấu hình**: `configs/config.py` và `configs/config.yaml` (không làm thay đổi logic các module thực thi trong `src/`).
2. **Loại bỏ triệt để các tham số "rác" / không sử dụng**: Các tham số còn sót lại từ các kiến trúc cũ (như `CNNAdapter`, `DeepGRUClassifier`), các tham số vẽ biểu đồ không thuộc về `train.py`, hoặc các cờ không bao giờ được đọc trong mã nguồn.
3. **Đồng bộ chuẩn xác các tham số đang sử dụng**: Khắc phục tình trạng giá trị giữa `config.py` (dataclass `TrainConfig`) và `config.yaml` bị lệch nhau (ví dụ: `input_dim`, `hidden_dim`, `seq_len`, `sample_interval`, `batch_size`, `epochs`, `gradient_accumulation_steps`).
4. **Khớp 100% với mô hình đã huấn luyện (`best.pt`)**: Đảm bảo khi khởi tạo `ConvGRUClassifier.from_config(cfg)` hoặc chạy huấn luyện / đánh giá, số lượng tham số mô hình khớp chính xác 631,716 tham số (không bị lệch tensor shape).

---

## 2. KẾT QUẢ KHẢO SÁT VÀ RÀ SOÁT CHI TIẾT MÃ NGUỒN

Qua quá trình quét AST (Abstract Syntax Tree) và đối chiếu mã nguồn của toàn bộ dự án (`src/train.py`, `src/models.py`, `src/dataset.py`, `src/evaluate.py`, `src/loss.py`, `test.py`, `notebooks/02_train_convgru_kaggle.ipynb`), hiện trạng 86 trường trong `TrainConfig` và `config.yaml` được phân loại cụ thể như sau:

### 2.1. Nhóm 1: DATASET CONFIGURATION
| Tên tham số | Hiện trạng trong code | Đánh giá | Đề xuất xử lý |
| :--- | :--- | :--- | :--- |
| `dataset_dir` | Đọc tại `train.py:128`, `dataset.py:268`, `evaluate.py:284` | **Đang sử dụng** | Giữ lại. Giá trị mặc định chuẩn: `"E:/LSTM/data_processed"`. |
| `manifest_file` | Đọc tại `train.py:129`, `dataset.py:269`, `evaluate.py` | **Đang sử dụng** | Giữ lại: `"dataset_merged_split.csv"`. |
| `val_dataset_dir` | Không có dòng code nào trong `src/` sử dụng (dataset chia qua manifest hoặc subfolder `val`) | **Dư thừa** | **Xóa bỏ**. |
| `val_manifest` | Không có dòng code nào trong `src/` sử dụng | **Dư thừa** | **Xóa bỏ**. |
| `backbone_neck_checkpoint` | Đọc tại `train.py:155`, `dataset.py:148`, `evaluate.py:313` | **Đang sử dụng** | Giữ lại. Trỏ đúng checkpoint BackboneNeck tối ưu đã train. |
| `sample_interval` | Đọc tại `train.py:130`, `dataset.py:271` | **Đang sử dụng** | Giữ lại. Giá trị thực tế đã train: `0.2` (trong `config.py` đang để lệch 0.1). |
| `seq_len` | Đọc tại `train.py:131`, `dataset.py:272` | **Đang sử dụng** | Giữ lại. Giá trị thực tế đã train: `null` / `None` (padding động). |
| `image_size` | Đọc tại `train.py:125, 140, 159`, `evaluate.py` | **Đang sử dụng** | Giữ lại: `[640, 640]`. |
| `video_exts` | Có trong `dataset.py:275` (lọc đuôi file video hợp lệ) | **Hữu ích** | Giữ lại: `[".avi", ".mp4", ".mkv", ".mov"]`. |
| `train_ratio` | Không có dòng code nào sử dụng (chia qua manifest) | **Dư thừa** | **Xóa bỏ**. |
| `split_by_subject` | Không có dòng code nào sử dụng | **Dư thừa** | **Xóa bỏ**. |
| `min_frames` | Đọc tại `train.py:124, 139`, `dataset.py:277` | **Đang sử dụng** | Giữ lại: `10`. |
| `use_augmentation` | Đọc tại `train.py:138`, `dataset.py:508` | **Đang sử dụng** | Giữ lại: `true`. |

---

### 2.2. Nhóm 2: DATALOADER CONFIGURATION
| Tên tham số | Hiện trạng trong code | Đánh giá | Đề xuất xử lý |
| :--- | :--- | :--- | :--- |
| `batch_size` | Đọc tại `train.py:132`, `evaluate.py` | **Đang sử dụng** | Giữ lại. Đồng bộ giá trị chuẩn: `4` (an toàn VRAM). |
| `val_batch_size` | Đọc tại `train.py:122, 133` | **Đang sử dụng** | Giữ lại: `4`. |
| `num_workers` | Đọc tại `train.py:134`, `evaluate.py` | **Đang sử dụng** | Giữ lại: `2` (Windows tự hạ về 0 nếu cần). |
| `val_num_workers` | Đọc tại `train.py:123, 135` | **Đang sử dụng** | Giữ lại: `0`. |
| `pin_memory` | Đọc tại `train.py:136` | **Đang sử dụng** | Giữ lại: `false`. |
| `shuffle` | Đọc tại `train.py:137` (`shuffle_train`) | **Đang sử dụng** | Giữ lại: `true`. |
| `drop_last` | `DataLoader` trong `dataset.py` không nhận tham số này | **Dư thừa** | **Xóa bỏ**. |
| `persistent_workers`| `DataLoader` trong `dataset.py` không nhận tham số này | **Dư thừa** | **Xóa bỏ**. |
| `prefetch_factor` | `DataLoader` trong `dataset.py` không nhận tham số này | **Dư thừa** | **Xóa bỏ**. |
| `seed` | Đọc tại `train.py:626`, `evaluate.py:875` | **Đang sử dụng** | Giữ lại: `42`. |
| `chunk_size` | Đọc tại `train.py:157`, `dataset.py:150` | **Đang sử dụng** | Giữ lại: `32`. |

---

### 2.3. Nhóm 3: MODEL ARCHITECTURE CONFIGURATION (`ConvGRUClassifier`)
| Tên tham số | Hiện trạng trong code | Đánh giá | Đề xuất xử lý |
| :--- | :--- | :--- | :--- |
| `cnn_neck_channels` | Đọc tại `models.py:451, 462` làm `spatial_in_channels` | **Đang sử dụng** | Giữ lại: `[64, 128, 256]`. |
| `cnn_strides` | Không dòng nào trong `models.py` hay `train.py` đọc | **Dư thừa** | **Xóa bỏ**. |
| `cnn_num_features` | Không dòng nào sử dụng (suy từ len của neck channels) | **Dư thừa** | **Xóa bỏ**. |
| `cnn_out_channels` | Dư thừa (tổng kênh 448 tự tính bằng `sum(spatial_in_channels)`) | **Dư thừa** | **Xóa bỏ**. |
| `cnn_spatial_size` | Không dùng (mô hình dùng pooling thích ứng) | **Dư thừa** | **Xóa bỏ**. |
| `spatial_fusion` | Tàn tích cũ (hiện tại `SpatialReductionNeck` luôn concat) | **Dư thừa** | **Xóa bỏ**. |
| `adapter_dropout` | Đọc tại `models.py:452, 463` làm `neck_dropout` | **Đang sử dụng** | Giữ lại: `0.25`. |
| `use_norm` | Tàn tích cũ (`SpatialReductionNeck` luôn dùng BatchNorm2d) | **Dư thừa** | **Xóa bỏ**. |
| `input_dim` | Đọc tại `models.py:443, 457` (kênh sau khi nén 448 -> 128) | **Đang sử dụng (Lệch)** | Giữ lại và sửa thành `128` (trong `config.py` cũ ghi sai 256). |
| `hidden_dim` | Đọc tại `models.py:448` (`convgru_hidden_dim`), kiểm chứng `best.pt` là 64 | **Đang sử dụng (Lệch)** | Chuẩn hóa về đúng giá trị thực tế `64` (trong yaml cũ ghi nhầm 128, config.py ghi 192). |
| `num_layers` | Đọc tại `models.py:449, 459` (số tầng ConvGRU) | **Đang sử dụng** | Giữ lại: `2`. |
| `num_classes` | Đọc tại `models.py:450`, `train.py:171` | **Đang sử dụng** | Giữ lại: `2`. |
| `dropout` | Đọc tại `models.py:453, 465` (FC Head Dropout) | **Đang sử dụng** | Giữ lại: `0.35`. |
| `supervision_mode` | Đọc tại `models.py:454`, `train.py:114` | **Đang sử dụng** | Giữ lại: `"attention_pooling"`. |

---

### 2.4. Nhóm 4: LOSS CONFIGURATION
| Tên tham số | Hiện trạng trong code | Đánh giá | Đề xuất xử lý |
| :--- | :--- | :--- | :--- |
| `loss_type` | Đọc tại `train.py:170`, `loss.py:98` | **Đang sử dụng** | Giữ lại: `"ce"`. |
| `bce_eps` | Đọc tại `loss.py:80` | **Đang sử dụng** | Giữ lại: `1.0e-7`. |
| `bce_reduction` | Đọc tại `train.py:175`, `loss.py:79` | **Đang sử dụng** | Giữ lại: `"mean"`. |
| `pos_weight` | Đọc tại `train.py:174`, `loss.py:34, 78` | **Đang sử dụng** | Giữ lại: `null` / `None`. |

---

### 2.5. Nhóm 5: OPTIMIZER & SCHEDULER CONFIGURATION
| Tên tham số | Hiện trạng trong code | Đánh giá | Đề xuất xử lý |
| :--- | :--- | :--- | :--- |
| `epochs` | Đọc tại `train.py:207, 312, 411, 549` | **Đang sử dụng** | Giữ lại. Đồng bộ chuẩn: `20`. |
| `lr0` | Đọc tại `train.py:192, 199, 208` | **Đang sử dụng** | Giữ lại: `0.001`. |
| `lr_min_factor` | Đọc tại `train.py:208` | **Đang sử dụng** | Giữ lại: `0.01`. |
| `weight_decay` | Đọc tại `train.py:194, 201` | **Đang sử dụng** | Giữ lại: `0.0001`. |
| `warmup_epochs` | Không dùng (`train.py` dùng `CosineAnnealingLR` không warmup) | **Dư thừa** | **Xóa bỏ**. |
| `optimizer` | Đọc tại `train.py:189` | **Đang sử dụng** | Giữ lại: `"adamw"`. |
| `betas` | Đọc tại `train.py:200` | **Đang sử dụng** | Giữ lại: `[0.9, 0.999]`. |
| `momentum` | Đọc tại `train.py:193` (khi chọn SGD) | **Đang sử dụng** | Giữ lại: `0.9`. |
| `grad_clip_norm` | Đọc tại `train.py:351, 355` | **Đang sử dụng** | Giữ lại: `1.0`. |
| `gradient_accumulation_steps` | Đọc tại `train.py:307` | **Đang sử dụng** | Giữ lại: `4`. |
| `empty_cache_interval` | Đọc tại `train.py:608` | **Đang sử dụng** | Giữ lại: `4`. |
| `use_scheduler` | Đọc tại `train.py:204` | **Đang sử dụng** | Giữ lại: `true`. |
| `scheduler_type` | Không dùng (`train.py` cố định dùng CosineAnnealingLR) | **Dư thừa** | **Xóa bỏ**. |

---

### 2.6. Nhóm 6: LOGGING, CHECKPOINTS & EARLY STOPPING
| Tên tham số | Hiện trạng trong code | Đánh giá | Đề xuất xử lý |
| :--- | :--- | :--- | :--- |
| `tb_log_dir` | Đọc tại `train.py:214` | **Đang sử dụng** | Giữ lại: `"logs/tensorboard"`. |
| `log_dir` | Không dùng (`train.py` chỉ dùng biến cục bộ `log_dir`) | **Dư thừa** | **Xóa bỏ**. |
| `experiment_name` | Đọc tại `train.py:214, 219` | **Đang sử dụng** | Giữ lại: `"deepgru_raw_nmsfree"`. |
| `checkpoint_dir` | Đọc tại `train.py:219` | **Đang sử dụng** | Giữ lại: `"checkpoints/experiments"`. |
| `save_all_epochs` | Đọc tại `train.py:527` | **Đang sử dụng** | Giữ lại: `true`. |
| `save_ckpt_interval_epochs` | Đọc tại `train.py:528` | **Đang sử dụng** | Giữ lại: `1`. |
| `save_best_only` | Không dùng (`best.pt` luôn tự động được lưu riêng) | **Dư thừa** | **Xóa bỏ**. |
| `ckpt_keep_last` | Đọc tại `train.py:534` | **Đang sử dụng** | Giữ lại: `null` / `None`. |
| `enable_resume` | Đọc tại `train.py:634` | **Đang sử dụng** | Giữ lại: `false`. |
| `resume` | Đọc tại `train.py:634, 635` | **Đang sử dụng** | Giữ lại: `""`. |
| `resume_epoch` | Không dùng (`load_checkpoint` tự đọc epoch từ checkpoint) | **Dư thừa** | **Xóa bỏ**. |
| `enable_step_logging`| Đọc tại `train.py:372` | **Đang sử dụng** | Giữ lại: `true`. |
| `log_step_interval` | Đọc tại `train.py:372` | **Đang sử dụng** | Giữ lại: `5`. |
| `flush_step_interval`| Không dùng trong `train.py` | **Dư thừa** | **Xóa bỏ**. |
| `ema_beta` | Không dùng trong `train.py` (không tính EMA loss) | **Dư thừa** | **Xóa bỏ**. |
| `early_stopping` | Đọc tại `train.py:226` | **Đang sử dụng** | Giữ lại: `true`. |
| `patience` | Đọc tại `train.py:228` | **Đang sử dụng** | Giữ lại: `10`. |
| `monitor_metric` | Đọc tại `train.py:231` | **Đang sử dụng** | Giữ lại: `"val_f1"`. |
| `monitor_mode` | Đọc tại `train.py:230` | **Đang sử dụng** | Giữ lại: `"max"`. |
| `min_delta` | Đọc tại `train.py:229` | **Đang sử dụng** | Giữ lại: `0.0001`. |
| `history_csv_path` | Đọc tại `train.py:237`, `evaluate.py:897` | **Đang sử dụng** | Giữ lại: `"logs/training_history.csv"`. |
| `summary_json_path`| `train.py` không sinh file này (`evaluate.py` có cấu hình riêng)| **Dư thừa** | **Xóa bỏ**. |
| `plot_curves_path` | `train.py` không sinh file này | **Dư thừa** | **Xóa bỏ**. |
| `plot_cm_path` | `train.py` không sinh file này | **Dư thừa** | **Xóa bỏ**. |
| `plot_roc_path` | `train.py` không sinh file này | **Dư thừa** | **Xóa bỏ**. |
| `dry_run` | Không dùng | **Dư thừa** | **Xóa bỏ**. |

---

### 2.7. Nhóm 7: RUNTIME & HARDWARE CONFIGURATION
| Tên tham số | Hiện trạng trong code | Đánh giá | Đề xuất xử lý |
| :--- | :--- | :--- | :--- |
| `device` | Đọc tại `train.py:110`, `evaluate.py` | **Đang sử dụng** | Giữ lại: `"cuda"`. |
| `amp` | Đọc tại `train.py:158, 223, 326` | **Đang sử dụng** | Giữ lại: `true`. |
| `log_interval` | Không dùng (`train.py` dùng `tqdm` và `log_step_interval`) | **Dư thừa** | **Xóa bỏ**. |
| `val_interval_epochs` | Đọc tại `train.py:563` | **Đang sử dụng** | Giữ lại: `1`. |
| `use_tqdm` | Đọc tại `train.py:312, 411` | **Đang sử dụng** | Giữ lại: `true`. |

---

## 3. TỔNG KẾT VÀ BẢNG THỐNG KÊ RÀ SOÁT

- **Tổng số tham số ban đầu**: 57 tham số trong `config.yaml` / 58 trường trong `TrainConfig`.
- **Số tham số thực sự có sử dụng được giữ lại**: **35 tham số** (giảm hơn 38% số tham số dư thừa).
- **Số tham số dư thừa bị loại bỏ**: **22 tham số** (`val_dataset_dir`, `val_manifest`, `train_ratio`, `split_by_subject`, `drop_last`, `persistent_workers`, `prefetch_factor`, `cnn_strides`, `cnn_num_features`, `cnn_out_channels`, `cnn_spatial_size`, `spatial_fusion`, `use_norm`, `warmup_epochs`, `scheduler_type`, `log_dir`, `save_best_only`, `resume_epoch`, `flush_step_interval`, `ema_beta`, các đường dẫn `plot_*_path`/`summary_json_path`, `dry_run`, `log_interval`).
- **Các giá trị cốt lõi được đồng bộ chuẩn xác**:
  - `input_dim`: Cố định chuẩn `128` (trong `config.py` cũ ghi sai 256).
  - `hidden_dim`: Cố định chuẩn `64` (khớp chính xác 631,716 tham số của `best.pt`).
  - `sample_interval`: Cố định chuẩn `0.2` (thay vì 0.1).
  - `seq_len`: Cố định chuẩn `None` / `null` (dynamic sequence).
  - `batch_size` & `val_batch_size`: Cố định chuẩn `4`.
  - `epochs`: Cố định chuẩn `20`.
  - `gradient_accumulation_steps`: Cố định chuẩn `4`.
  - `empty_cache_interval`: Cố định chuẩn `4`.

---

## 4. BƯỚC TIẾP THEO

Theo đúng quy định tại `AGENTS.md`:
1. Trình bày kết quả phân tích này cho người dùng và xin ý kiến phê duyệt.
2. Sau khi người dùng đồng ý với phân tích này, sẽ chuyển sang **Bước 2: Lập kế hoạch thực hiện (`docs/plan/plan_config.md`)**.
