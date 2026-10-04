# BÁO CÁO PHÂN TÍCH: RÀ SOÁT LỖI ĐƯỜNG DẪN, LOẠI BỎ THAM SỐ THỪA VÀ ĐỒNG BỘ CẤU HÌNH TRAIN TỪ RAW DATA CHO NMSFreeDetector

> **Tài liệu:** `docs/analsys/analsys_config_nmsfree.md`  
> **Nhiệm vụ:** Rà soát lỗi đường dẫn trong cấu hình, loại bỏ các tham số không hỗ trợ huấn luyện trực tiếp từ dữ liệu video thô (Raw Video) qua mô hình `NMSFreeDetector`, và đồng bộ hóa tuyệt đối giữa `configs/config.py` và `configs/config.yaml`.  
> **Tuân thủ quy trình:** Bước 1 - Khảo sát & Phân tích (Discovery) theo `AGENTS.md`.

---

## 1. Tổng quan & Bối cảnh Kỹ thuật

### 1.1. Bối cảnh Hệ thống
Hệ thống phát hiện tài xế buồn ngủ (**Driver Guardian AI**) sử dụng kiến trúc hai giai đoạn tích hợp:
1. **Trích xuất đặc trưng không gian (Spatial Feature Extraction):** Sử dụng thân mạng `BackboneNeck` (Backbone + PAFPN Neck) của mô hình `NMSFreeDetector` (định nghĩa trong `ai/ObjectDetection_2p6M`), nạp trọng số tối ưu từ checkpoint PyTorch `.pt` (`best.pt`).
2. **Học chuỗi thời gian & Phân loại (Temporal Modeling & Classification):** Sử dụng mô hình `DeepGRUClassifier` (gồm `CNNAdapter` + Deep GRU 2 lớp + `TemporalAttentionPooling` + Linear Classifier).

Dữ liệu đầu vào hiện tại là tập video thô đã chuẩn hóa tại:
`/home/riftuser/.cache/kagglehub/datasets/nyvantran6634/dataset-datn4ni/versions/1/data_processed`
Bao gồm:
- Tệp manifest: `dataset_merged_split.csv` và `dataset_merged_split.json`.
- Cấu trúc thư mục con: `train/` (`0_alert/`, `1_drowsy/`) và `val/` (`0_alert/`, `1_drowsy/`).

Pipeline nạp dữ liệu trực tiếp từ video thô được hiện thực trong `src/dataset2.py` thông qua lớp `RawVideoBackboneNeckDataset` và hàm factory `build_raw_video_dataloaders`.

### 1.2. Vấn đề Hiện tại của Hệ thống Cấu hình
Hai tệp cấu hình trung tâm của dự án là `configs/config.py` (`TrainConfig` dataclass) và `configs/config.yaml` đang gặp phải các vấn đề nghiêm trọng:
1. **Lỗi đường dẫn nghiêm trọng (Path Errors):** 
   - Xuất hiện các đường dẫn Windows tuyệt đối (`E:\LSTM\...`, `D:\Project\...`, `/outsrc/myCNN\...`).
   - Đường dẫn checkpoint NMSFreeDetector bị sai prefix (`/workspace/...` thay vì `/home/riftuser/workspace/...`).
   - Đường dẫn tệp ONNX trỏ tới các tệp không tồn tại (`checkpoints/backbone.onnx`, `checkpoints/backbone_neck.onnx`).
2. **Tham số dư thừa và xung đột (Dead & Conflicting Parameters):**
   - Còn tồn tại nhiều tham số của các chế độ cũ đã bị khai tử:
     - Chế độ tensor trích xuất sẵn (`use_preloaded_pt`, `train_pt`, `val_pt`).
     - Chế độ file HDF5 trích xuất sẵn (`train_h5`, `val_h5`, `train_manifest_csv`, `val_manifest_csv`, `include_augmented_train`).
     - Chế độ ONNX Runtime suy luận cũ (`backbone_onnx_path`, `backbone_neck_onnx_path`, `cnn_manifest_path`).
     - Chế độ CNN giả lập (`use_dummy_cnn`).
   - Xung đột giữa 2 biến trỏ checkpoint: `backbone_neck_checkpoint` (đang mang giá trị `None`) và `cnn_weights_path` (đang trỏ đường dẫn lỗi).
3. **Bất đồng bộ sâu sắc giữa `config.py` và `config.yaml`:**
   - `config.yaml` hoàn toàn thiếu các tham số then chốt để nạp raw video: `dataset_dir`, `manifest_file`, `backbone_neck_checkpoint`.
   - `config.yaml` chứa lỗi chính tả: `video_exits` thay vì `video_exts`.
   - Khác biệt về siêu tham số mặc định: `batch_size` (16 vs 64), `chunk_size` (16 vs 100), `input_dim` (256 vs 512), `hidden_dim` (192 vs 256).

---

## 2. Rà soát Chi tiết các Lỗi Đường dẫn (Path Errors Audit)

Dưới đây là bảng rà soát chi tiết tất cả các đường dẫn trong cấu hình hiện tại và trạng thái thực tế trên môi trường Linux Ubuntu:

| Tham số cấu hình | Giá trị trong `config.py` | Giá trị trong `config.yaml` | Tình trạng thực tế trên hệ thống | Đánh giá & Hướng khắc phục |
| :--- | :--- | :--- | :--- | :--- |
| `train_h5` | `"dataset_features.h5"` | `E:\LSTM\checkpoints\dataset_features.h5` | **Không tồn tại** (Đường dẫn Windows ổ `E:\`) | ❌ **Lỗi nghiêm trọng**: Xóa bỏ tham số vì không dùng HDF5. |
| `val_h5` | `None` | `null` | Không dùng | ❌ Xóa bỏ tham số. |
| `train_pt` / `val_pt` | `"extracted_features_pt/..."` | *(Không có)* | **Không tồn tại** | ❌ Xóa bỏ tham số. |
| `dataset_dir` | `/home/riftuser/.cache/kagglehub/.../data_processed` | *(BỊ THIẾU)* | **TỒN TẠI HỢP LỆ** (Chứa `train/`, `val/`, `dataset_merged_split.csv`) | ⚠️ **Thiếu trong YAML**: Cần giữ lại và bổ sung vào `config.yaml`. |
| `val_dataset_dir` | `r""` (chuỗi rỗng) | *(BỊ THIẾU)* | Mặc định dùng chung `dataset_dir` (có thư mục `val/`) | ⚠️ Chuẩn hóa thành `Optional[str] = None` trong cả Python và YAML. |
| `manifest_file` | `None` | *(BỊ THIẾU)* | Tồn tại `dataset_merged_split.csv` trong `dataset_dir` | ⚠️ Bổ sung vào `config.yaml`, mặc định trỏ hoặc tự động nhận diện file CSV. |
| `cnn_weights_path` | `/workspace/driver-guardian/.../best.pt` | *(BỊ THIẾU)* | **SAI ĐƯỜNG DẪN**: Thiếu `/home/riftuser` ở đầu | ❌ **Lỗi nghiêm trọng**: File thực tế tại `/home/riftuser/workspace/driver-guardian/ai/ObjectDetection_2p6M/checkpoints/.../best.pt`. |
| `backbone_neck_checkpoint` | `None` | *(BỊ THIẾU)* | Là biến chuẩn được `src/dataset2.py` đọc | ⚠️ Chuẩn hóa thành đường dẫn trỏ tới checkpoint `best.pt` của NMSFreeDetector. |
| `backbone_onnx_path` | `"checkpoints/backbone.onnx"` | `"checkpoints/backbone.onnx"` | **Không tồn tại** | ❌ Xóa bỏ vì huấn luyện qua PyTorch native module. |
| `backbone_neck_onnx_path`| `"checkpoints/backbone_neck.onnx"` | `"checkpoints/backbone_neck.onnx"` | **Không tồn tại** | ❌ Xóa bỏ vì không dùng ONNX. |
| `cnn_manifest_path` | `r"/outsrc/myCNN\checkpoints_ftCOCO\model_mainfest.json"` | *(Không có)* | **Không tồn tại** (Đường dẫn Windows rác) | ❌ Xóa bỏ hoàn toàn. |
| `video_exts` | `(".avi", ".mp4", ".mkv")` | `video_exits: [".avi", ".mp4", ".mkv"]` | Hợp lệ, nhưng YAML **SAI CHÍNH TẢ** (`video_exits`) | ⚠️ Sửa chính tả thành `video_exts` và bổ sung đuôi `".mov"`. |

---

## 3. Phân loại & Quyết định Loại bỏ Tham số (Parameter Elimination)

Dựa trên yêu cầu cốt lõi **"chỉ hỗ trợ huấn luyện trực tiếp từ dữ liệu video thô qua mô hình NMSFreeDetector"**, toàn bộ các tham số sau đây sẽ được **XÓA BỎ HOÀN TOÀN** khỏi `configs/config.py` và `configs/config.yaml`:

### 3.1. Danh sách Tham số Bị Xóa Bỏ
1. **Tham số nạp Tensor trích xuất sẵn (.pt):**
   - `use_preloaded_pt`: Cờ lựa chọn chế độ nạp tensor `.pt`.
   - `train_pt`: Đường dẫn file tensor tập train.
   - `val_pt`: Đường dẫn file tensor tập validation.
2. **Tham số nạp tập đặc trưng HDF5 (.h5):**
   - `train_h5`: Đường dẫn file HDF5 train.
   - `val_h5`: Đường dẫn file HDF5 validation.
   - `train_manifest_csv`: File CSV manifest cho HDF5 train.
   - `val_manifest_csv`: File CSV manifest cho HDF5 val.
   - `include_augmented_train`: Cờ nạp mẫu tăng cường offline trong HDF5.
3. **Tham số suy luận ONNX Runtime:**
   - `backbone_onnx_path`: Đường dẫn file ONNX của Backbone.
   - `backbone_neck_onnx_path`: Đường dẫn file ONNX của BackboneNeck.
   - `cnn_manifest_path`: Đường dẫn file cấu hình json của mô hình ONNX cũ.
4. **Tham số Dummy CNN:**
   - `use_dummy_cnn`: Cờ chạy trích xuất CNN giả lập khi kiểm thử ban đầu.
5. **Hợp nhất biến Checkpoint dư thừa:**
   - Xóa bỏ `cnn_weights_path` (đang mang đường dẫn sai) và hợp nhất thành một biến duy nhất: `backbone_neck_checkpoint` đại diện cho trọng số PyTorch của `NMSFreeDetector`.

---

## 4. Đặc tả Cấu trúc Cấu hình Chuẩn Hóa cho Raw Video + NMSFreeDetector

Sau khi dọn dẹp, cấu hình sẽ tinh gọn thành **7 nhóm tham số rõ ràng, thống nhất 100% giữa `config.py` và `config.yaml`**:

### Group 1: Dataset Configuration (Tập dữ liệu Video Thô)
- `dataset_dir: str`: Thư mục gốc chứa video thô (`/home/riftuser/.cache/kagglehub/datasets/nyvantran6634/dataset-datn4ni/versions/1/data_processed`).
- `manifest_file: Optional[str]`: Tệp manifest CSV/JSON định tuyến mẫu (mặc định trỏ đến `dataset_merged_split.csv` hoặc `None` để tự động dò).
- `val_dataset_dir: Optional[str]`: Thư mục video thô validation riêng biệt (`None` = dùng chung cấu trúc `dataset_dir/val`).
- `val_manifest: Optional[str]`: Tệp manifest cho tập val (`None` = tự động phân tách theo trường `split` trong manifest chính).
- `backbone_neck_checkpoint: str`: Đường dẫn tuyệt đối đến checkpoint `.pt` của `NMSFreeDetector` (`/home/riftuser/workspace/driver-guardian/ai/ObjectDetection_2p6M/checkpoints/2bfb6cc36ef5ab822a944006c746e5d348d67e387501c9027df5b957cf124031/de0c5b23e1ae749b972f10cb5f3e21e2ccf37bd334eb7a7c3091c460073ff310/finetune/best.pt`).
- `sample_interval: float = 0.1`: Chu kỳ lấy mẫu khung hình (0.1 giây ~ 10 FPS).
- `seq_len: Optional[int] = 50`: Số lượng khung hình cố định mỗi clip (`50` hoặc `None` để dynamic padding).
- `image_size: Tuple[int, int] = (640, 640)`: Kích thước frame chuẩn hóa qua letterbox trước khi đưa vào NMSFreeDetector.
- `video_exts: Tuple[str, ...] = (".avi", ".mp4", ".mkv", ".mov")`: Các định dạng video hợp lệ.
- `train_ratio: float = 0.8`: Tỷ lệ chia tập huấn luyện nếu dataset không chia sẵn thư mục train/val.
- `split_by_subject: bool = True`: Chia tập theo đối tượng người lái xe.
- `min_frames: int = 10`: Số khung hình tối thiểu hợp lệ của một video clip.

### Group 2: DataLoader Configuration (Nạp dữ liệu Đa tiến trình)
- `batch_size: int = 16`: Kích thước batch huấn luyện (16 tối ưu cho GPU VRAM từ 4GB - 8GB khi chạy cả NMSFreeDetector trích xuất).
- `val_batch_size: int = 16`: Kích thước batch kiểm định.
- `num_workers: int = 0`: Số worker tiến trình nạp train (0 an toàn tuyệt đối với CUDA lazy context).
- `val_num_workers: int = 0`: Số worker nạp val.
- `pin_memory: bool = False`: Tắt pin_memory để tránh OOM bộ nhớ host.
- `shuffle: bool = True`: Trộn ngẫu nhiên tập train.
- `drop_last: bool = False`: Không bỏ rơi batch cuối cùng.
- `persistent_workers: bool = False`: Tắt khi num_workers=0.
- `prefetch_factor: Optional[int] = None`: Số batch prefetch.
- `seed: int = 42`: Hạt giống ngẫu nhiên đảm bảo tính tái lập.
- `chunk_size: int = 16`: Kích thước mini-chunk khi trích xuất khung hình qua NMSFreeDetector để chống tràn VRAM.

### Group 3: Model Architecture Configuration (NMSFreeDetector PAFPN + Deep GRU)
- `cnn_neck_channels: Tuple[int, int, int] = (64, 128, 256)`: Kênh đặc trưng đầu ra (p3, p4, p5) từ PAFPN Neck của NMSFreeDetector.
- `cnn_strides: Tuple[int, int, int] = (8, 16, 32)`: Strides tương ứng của các tầng đặc trưng.
- `cnn_num_features: int = 3`: Số tầng đặc trưng không gian đa tỷ lệ.
- `cnn_out_channels: int = 448`: Tổng số kênh kết hợp (64 + 128 + 256).
- `cnn_spatial_size: Tuple[int, int] = (20, 20)`: Kích thước bản đồ đặc trưng tầng sâu nhất sau khi pooling/alignment.
- `spatial_fusion: str = "concat"`: Phương thức dung hợp đặc trưng không gian ('concat' / 'attention').
- `adapter_dropout: float = 0.25`: Tỷ lệ Dropout sau Spatial Feature Adapter.
- `use_norm: bool = True`: Sử dụng LayerNorm trong Spatial Adapter.
- `input_dim: int = 256`: Kích thước vector đặc trưng đưa vào GRU.
- `hidden_dim: int = 192`: Số hidden units trong mỗi tầng GRU.
- `num_layers: int = 2`: Số lớp GRU xếp chồng (Deep GRU).
- `num_classes: int = 2`: Số lớp phân loại (0: Tỉnh táo, 1: Buồn ngủ).
- `dropout: float = 0.35`: Tỷ lệ Dropout giữa các lớp GRU.
- `supervision_mode: str = "attention_pooling"`: Cơ chế Temporal Attention Pooling chuẩn.

### Group 4: Loss Configuration
- `loss_type: str = "ce"`: "ce" (CrossEntropyLoss) hoặc "bce" (BCELoss).
- `bce_eps: float = 1e-7`: Epsilon chống log(0).
- `bce_reduction: str = "mean"`: Gom nhóm loss.
- `pos_weight: Optional[float] = None`: Trọng số lớp dương khi mất cân bằng dữ liệu.

### Group 5: Optimizer & Scheduler Configuration
- `epochs: int = 40`: Tổng số epoch huấn luyện.
- `lr0: float = 1e-3`: Learning rate khởi tạo.
- `lr_min_factor: float = 0.01`: Hệ số lr tối thiểu (lr_min = 1e-5).
- `weight_decay: float = 1e-4`: Regularization phạt L2.
- `warmup_epochs: float = 1.0`: Khởi động mềm Warmup.
- `optimizer: str = "adamw"`: Bộ tối ưu AdamW.
- `betas: Tuple[float, float] = (0.9, 0.999)`: Tham số AdamW.
- `momentum: float = 0.9`: Momentum.
- `grad_clip_norm: float = 1.0`: Ngưỡng cắt gradient.
- `use_scheduler: bool = True`: Bật Cosine Annealing Scheduler.
- `scheduler_type: str = "cosine"`: Loại Scheduler.

### Group 6: Checkpoints, Early Stopping & Diagnostics
- `tb_log_dir: str = "logs/tensorboard"`: Thư mục log TensorBoard.
- `log_dir: str = "logs"`: Thư mục log text.
- `experiment_name: str = "deepgru_raw_nmsfree"`: Tên bài thí nghiệm chuẩn hóa.
- `checkpoint_dir: str = "checkpoints/experiments"`: Thư mục lưu checkpoint.
- `save_all_epochs: bool = True`: Lưu toàn bộ checkpoint các epoch.
- `save_ckpt_interval_epochs: int = 1`: Chu kỳ lưu checkpoint.
- `save_best_only: bool = False`: Lưu định kỳ + best/last.
- `ckpt_keep_last: Optional[int] = None`: Giữ toàn bộ checkpoint.
- `enable_resume: bool = False`: Cờ tiếp tục huấn luyện.
- `resume_epoch: Optional[int] = None`: Epoch cần khôi phục.
- `resume: str = ""`: Đường dẫn checkpoint cần khôi phục.
- `early_stopping: bool = True`: Cơ chế dừng sớm.
- `patience: int = 10`: Số epoch chịu đựng không cải thiện.
- `monitor_metric: str = "val_f1"`: Chỉ số giám sát chính.
- `monitor_mode: str = "max"`: Chiều tối ưu ("max").
- `min_delta: float = 1e-4`: Ngưỡng tiến bộ tối thiểu.
- Các file báo cáo & đồ thị: `history_csv_path`, `summary_json_path`, `plot_curves_path`, `plot_cm_path`, `plot_roc_path`, `dry_run`.

### Group 7: Runtime & Hardware Configuration
- `device: str = "cuda"`: Thiết bị phần cứng ("cuda" | "cpu").
- `amp: bool = True`: Tự động ép kiểu hỗn hợp Automatic Mixed Precision (FP16).
- `log_interval: int = 10`: Tần suất ghi log step.
- `val_interval_epochs: int = 1`: Tần suất đánh giá tập val.
- `use_tqdm: bool = True`: Thanh tiến trình tqdm.

---

## 5. Kế hoạch Cập nhật & Đồng bộ Logic Code

1. **Cập nhật phương thức `__post_init__` trong `configs/config.py`:**
   - Xóa bỏ logic gán Kaggle `.pt` paths và `val_h5 = self.train_h5`.
   - Bổ sung xác thực kiểm tra đường dẫn `dataset_dir` và `backbone_neck_checkpoint` tồn tại (báo lỗi rõ ràng nếu không tìm thấy checkpoint hoặc dataset).
   - Thêm cơ chế fallback tự động: Nếu `backbone_neck_checkpoint` là đường dẫn tương đối, tự động ghép với thư mục gốc dự án.
2. **Cập nhật hàm `load_yaml` trong `configs/config.py`:**
   - Xử lý chuyển đổi kiểu dữ liệu mảng danh sách (list) sang tuple cho các trường: `image_size`, `video_exts`, `cnn_neck_channels`, `cnn_strides`, `cnn_spatial_size`, `betas`.
3. **Cập nhật toàn diện tệp `configs/config.yaml`:**
   - Cấu trúc theo đúng 7 nhóm section tương ứng với dataclass `TrainConfig`: `dataset`, `dataloader`, `model`, `loss`, `optimizer`, `logging`, `runtime`.
   - Đồng bộ hoàn toàn giá trị mặc định của tất cả các siêu tham số.
   - Sửa lỗi chính tả `video_exits` thành `video_exts`.

---

## 6. Kết luận & Đề xuất Bước tiếp theo

Tài liệu phân tích này đã xác định đầy đủ:
- **Nguyên nhân và vị trí chính xác của tất cả các lỗi đường dẫn** trong cấu hình.
- **Danh sách cụ thể các tham số thừa cần loại bỏ** để chỉ tập trung vào luồng huấn luyện raw video với `NMSFreeDetector`.
- **Đặc tả cấu trúc và giá trị đồng bộ chuẩn 100%** giữa `config.py` và `config.yaml`.

Theo Quy trình làm việc tại Mục 5 của `AGENTS.md`:
> *Agent dừng tại Bước 1 và gửi báo cáo phân tích tới Người dùng. Sau khi Người dùng duyệt báo cáo phân tích này, Agent sẽ tiến hành Bước 2 (Lập kế hoạch thực hiện `plan_config_nmsfree.md`).*
