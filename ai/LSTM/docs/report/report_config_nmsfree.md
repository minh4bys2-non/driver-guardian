# BÁO CÁO KẾT QUẢ: RÀ SOÁT LỖI ĐƯỜNG DẪN, LOẠI BỎ THAM SỐ THỪA VÀ ĐỒNG BỘ CẤU HÌNH CHO NMSFreeDetector

> **Tài liệu:** `docs/report/report_config_nmsfree.md`  
> **Kế thừa từ:** `docs/analsys/analsys_config_nmsfree.md` và `docs/plan/plan_config_nmsfree.md`  
> **Nhiệm vụ:** Tổng hợp kết quả thực hiện rà soát lỗi đường dẫn, loại bỏ các tham số không hỗ trợ huấn luyện trực tiếp từ dữ liệu video thô (Raw Video) qua mô hình `NMSFreeDetector`, và đồng bộ hóa 100% giữa `configs/config.py` và `configs/config.yaml`.  
> **Tuân thủ quy trình:** Bước 3 - Báo cáo kết quả thực hiện theo `AGENTS.md`.

---

## 1. Tổng quan Kết quả Thực hiện

Nhiệm vụ đã được triển khai và hoàn thành thành công 100% với các kết quả cụ thể:
1. **Khắc phục toàn diện các lỗi đường dẫn:**
   - Xóa bỏ triệt để đường dẫn Windows tuyệt đối (`E:\...`).
   - Sửa lỗi đường dẫn checkpoint của `NMSFreeDetector`: Thay thế đường dẫn thiếu prefix `/workspace/...` bằng đường dẫn chuẩn xác thực tế trên hệ thống Linux Ubuntu:  
     `/home/riftuser/workspace/driver-guardian/ai/ObjectDetection_2p6M/checkpoints/2bfb6cc36ef5ab822a944006c746e5d348d67e387501c9027df5b957cf124031/de0c5b23e1ae749b972f10cb5f3e21e2ccf37bd334eb7a7c3091c460073ff310/finetune/best.pt`.
   - Xóa bỏ các đường dẫn trỏ tới tệp/thư mục không tồn tại: `dataset_features.h5`, `extracted_features_pt/`, `checkpoints/backbone.onnx`, `checkpoints/backbone_neck.onnx`, `cnn_manifest_path`.
   - Sửa lỗi chính tả trong file YAML: `video_exits` -> `video_exts`.
2. **Loại bỏ toàn bộ tham số dư thừa (Không hỗ trợ raw video + NMSFreeDetector):**
   - Đã loại bỏ 11 tham số legacy thuộc các chế độ HDF5, Tensor .pt, ONNX Runtime và Dummy CNN.
   - Hợp nhất hai biến checkpoint trùng lặp (`cnn_weights_path` và `backbone_neck_checkpoint`) thành một biến chuẩn duy nhất: `backbone_neck_checkpoint`.
3. **Đồng bộ hóa tuyệt đối (Parity 100%) giữa Python và YAML:**
   - Cả `configs/config.py` và `configs/config.yaml` đều chia thành 7 nhóm section thống nhất: `dataset`, `dataloader`, `model`, `loss`, `optimizer`, `logging`, `runtime`.
   - Khớp 1-1 về tên tham số, giá trị mặc định, và kiểu dữ liệu chuyển đổi.
4. **Kiểm thử tích hợp thành công trên tập dữ liệu thực tế:**
   - Đã kiểm thử `RawVideoBackboneNeckDataset` quét thành công **4.046 mẫu train** và **1.026 mẫu val** trực tiếp từ cấu hình mới.

---

## 2. Bảng Đối chiếu Chi tiết (Before vs After)

| Thành phần cấu hình | Trạng thái Trước khi sửa (Before) | Trạng thái Sau khi chuẩn hóa (After) | Ghi chú & Lợi ích |
| :--- | :--- | :--- | :--- |
| **`train_h5` / `val_h5`** | `E:\LSTM\checkpoints\dataset_features.h5` trong YAML, `"dataset_features.h5"` trong Python | **ĐÃ XÓA HOÀN TOÀN** | Loại bỏ lỗi đường dẫn Windows và phụ thuộc file HDF5 |
| **`train_pt` / `val_pt`** | `extracted_features_pt/features_sust_train.pt` (không tồn tại) | **ĐÃ XÓA HOÀN TOÀN** | Loại bỏ chế độ nạp Tensor .pt offline |
| **`backbone_onnx_path` / `backbone_neck_onnx_path`** | `"checkpoints/backbone.onnx"` (không tồn tại) | **ĐÃ XÓA HOÀN TOÀN** | `NMSFreeDetector` chạy trực tiếp qua PyTorch native module |
| **`use_dummy_cnn`** | `false` trong cả Python và YAML | **ĐÃ XÓA HOÀN TOÀN** | Bỏ cờ mock testing cũ trong collate |
| **`cnn_manifest_path`** | `r"/outsrc/myCNN\checkpoints_ftCOCO\model_mainfest.json"` | **ĐÃ XÓA HOÀN TOÀN** | Loại bỏ đường dẫn Windows không tồn tại |
| **`cnn_weights_path`** | `/workspace/driver-guardian/.../best.pt` (sai prefix) | **HỢP NHẤT VÀO `backbone_neck_checkpoint`** | Thống nhất 1 biến checkpoint duy nhất |
| **`backbone_neck_checkpoint`** | `None` trong Python, thiếu trong YAML | Đường dẫn tuyệt đối chuẩn xác tới `best.pt` (tồn tại 100%) | Đảm bảo nạp đúng trọng số NMSFreeDetector |
| **`dataset_dir`** | Có trong Python, **BỊ THIẾU** trong YAML | Đồng bộ đường dẫn tập video thô trong cả Python và YAML | Cho phép cấu hình linh hoạt thư mục dữ liệu |
| **`manifest_file`** | `None` trong Python, **BỊ THIẾU** trong YAML | Đồng bộ `"dataset_merged_split.csv"` trong cả Python và YAML | Định tuyến rõ ràng nhãn và video clip |
| **`video_exts`** | `video_exits` trong YAML (sai chính tả) | `video_exts: [".avi", ".mp4", ".mkv", ".mov"]` | Sửa lỗi chính tả, bổ sung đuôi `.mov` |
| **`batch_size`** | 16 (Python) vs 64 (YAML) | **Đồng bộ = 16** | Tối ưu VRAM GPU khi chạy cả NMSFreeDetector |
| **`chunk_size`** | 16 (Python) vs 100 (YAML) | **Đồng bộ = 16** | Ngăn ngừa tràn CUDA OOM khi inference mini-chunk |
| **`input_dim` / `hidden_dim`** | 256/192 (Python) vs 512/256 (YAML) | **Đồng bộ = 256 / 192** | Thống nhất kiến trúc mạng Deep GRU chuẩn |
| **`resume_epoch`** | `None` (Python) vs `2` (YAML) | **Đồng bộ = `null` (`None`)** | Đồng nhất trạng thái khi `enable_resume: false` |

---

## 3. Chi tiết Mã Nguồn Thay đổi

### 3.1. Tệp `configs/config.py`
1. **Định nghĩa đường dẫn chuẩn động qua `PROJECT_ROOT`**:
   ```python
   PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
   DEFAULT_CHECKPOINT_PATH = (
       PROJECT_ROOT
       / "ai"
       / "ObjectDetection_2p6M"
       / "checkpoints"
       / "2bfb6cc36ef5ab822a944006c746e5d348d67e387501c9027df5b957cf124031"
       / "de0c5b23e1ae749b972f10cb5f3e21e2ccf37bd334eb7a7c3091c460073ff310"
       / "finetune"
       / "best.pt"
   )
   ```
2. **Cập nhật dataclass `TrainConfig`**:
   - Nhóm Dataset chỉ giữ lại các tham số phục vụ raw video: `dataset_dir`, `manifest_file`, `val_dataset_dir`, `val_manifest`, `backbone_neck_checkpoint`, `sample_interval`, `seq_len`, `image_size`, `video_exts`, `train_ratio`, `split_by_subject`, `min_frames`.
   - Nhóm Model chỉ giữ lại các tham số liên quan đến feature maps (p3, p4, p5) và Deep GRU: `cnn_neck_channels`, `cnn_strides`, `cnn_num_features`, `cnn_out_channels`, `cnn_spatial_size`, `spatial_fusion`, `adapter_dropout`, `use_norm`, `input_dim`, `hidden_dim`, `num_layers`, `num_classes`, `dropout`, `supervision_mode`.
3. **Cập nhật `__post_init__`**:
   - Loại bỏ kiểm tra `train_pt`, `val_pt`, `train_h5`, Kaggle paths.
   - Thêm cơ chế tự động tìm fallback cho `backbone_neck_checkpoint` nếu đường dẫn tương đối.
4. **Cập nhật `load_yaml` và `load_json`**:
   - Tự động chuyển đổi các trường danh sách thành tuple: `image_size`, `video_exts`, `cnn_neck_channels`, `cnn_strides`, `cnn_spatial_size`, `betas`.
   - Lọc bỏ các key dư thừa không thuộc dataclass fields.

### 3.2. Tệp `configs/config.yaml`
- Viết lại toàn bộ cấu trúc theo 7 nhóm chuẩn, loại bỏ các trường thừa và đồng bộ 100% giá trị mặc định với `TrainConfig`.

---

## 4. Kết quả Kiểm thử & Xác thực Tự động

### 4.1. Kiểm thử Tính Đồng nhất Cấu hình (Config Parity Test)
Đã chạy kiểm tra tự động so sánh giữa thực thể `TrainConfig()` và `load_config("configs/config.yaml")`:
```text
SUCCESS: TrainConfig() and load_config() are 100% identical!
Dataset dir: /home/riftuser/.cache/kagglehub/datasets/nyvantran6634/dataset-datn4ni/versions/1/data_processed
Backbone checkpoint: /home/riftuser/workspace/driver-guardian/ai/ObjectDetection_2p6M/checkpoints/2bfb6cc36ef5ab822a944006c746e5d348d67e387501c9027df5b957cf124031/de0c5b23e1ae749b972f10cb5f3e21e2ccf37bd334eb7a7c3091c460073ff310/finetune/best.pt
Tuple fields correctly converted:
  image_size: (640, 640) <class 'tuple'>
  video_exts: ('.avi', '.mp4', '.mkv', '.mov') <class 'tuple'>
  cnn_neck_channels: (64, 128, 256) <class 'tuple'>
  cnn_strides: (8, 16, 32) <class 'tuple'>
  betas: (0.9, 0.999) <class 'tuple'>
```
*Kết quả:* **PASS 100%**.

### 4.2. Kiểm thử Tuần tự hóa JSON (JSON Round-Trip Test)
Kiểm tra lưu cấu hình qua `save_json` và nạp lại qua `load_json`:
```text
[TrainConfig] Đã lưu cấu hình vào: /tmp/tmpk0ovuq1f/test_cfg.json
All Config Serialization & Loading Unit Tests Passed!
```
*Kết quả:* **PASS 100%**.

### 4.3. Kiểm thử Tồn tại của Đường dẫn & Khởi tạo Mô hình
Kiểm tra các đường dẫn thực tế trên ổ đĩa và khả năng tích hợp mô hình:
```text
Checking dataset_dir exists: True
Checking backbone_neck_checkpoint exists: True
Checking manifest dataset_merged_split.csv exists: True
DeepGRUClassifier instantiated successfully: <class 'src.models.DeepGRUClassifier'>
All path and model verification passed successfully!
```
*Kết quả:* **PASS 100%**.

### 4.4. Kiểm thử Khám phá Dữ liệu với `RawVideoBackboneNeckDataset`
Kiểm tra nạp trực tiếp dữ liệu từ cấu hình mới:
```text
Testing RawVideoBackboneNeckDataset discovery on actual dataset...
Train samples discovered: 4046
Val samples discovered: 1026
SUCCESS: RawVideoBackboneNeckDataset discovered samples directly from config!
```
*Kết quả:* **PASS 100%** (Quét đủ 4.046 clip train và 1.026 clip val).

---

## 5. Danh mục Tiêu chí Chất lượng (Quality Checklist)

Tuân thủ Mục 6 của `AGENTS.md`:
- [x] Toàn bộ đường dẫn file sử dụng `pathlib.Path` hoặc `os.path.join`, tương thích hoàn hảo Linux/Windows.
- [x] Đã xóa bỏ toàn bộ lỗi đường dẫn tuyệt đối Windows và tệp không tồn tại.
- [x] Đường dẫn trọng số checkpoint NMSFreeDetector chính xác tuyệt đối và đã được xác thực tồn tại.
- [x] Đã loại bỏ 100% các tham số legacy không thuộc luồng raw video qua NMSFreeDetector.
- [x] Đã sửa lỗi chính tả `video_exits` -> `video_exts`.
- [x] Đã đồng bộ 100% giữa `configs/config.py` và `configs/config.yaml`.
- [x] Đã kiểm thử nạp dữ liệu và khởi tạo mô hình thành công không phát sinh lỗi.
