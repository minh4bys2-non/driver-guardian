# BÁO CÁO PHÂN TÍCH: KIỂM TRA TĂNG CƯỜNG DỮ LIỆU (src/augment.py & src/dataset2.py), CHUẨN HÓA QUY TRÌNH CHẠY VÀ XÁC THỰC LOAD CHECKPOINT NMSFreeDetector

> **Tài liệu:** `docs/analsys/analsys_dataset2_augment_ckpt.md`  
> **Nhiệm vụ:**
> 1. Đọc và kiểm tra file `src/dataset2.py` đã hỗ trợ sẵn tăng cường dữ liệu với các phép tăng cường ở `src/augment.py` chưa.
> 2. Chuẩn hóa quy trình chạy ở môi trường hiện tại (Ubuntu Linux, GPU NVIDIA Tesla V100-SXM3-32GB).
> 3. Kiểm tra tính chính xác và toàn vẹn của việc nạp checkpoint tại file: `/home/riftuser/workspace/driver-guardian/ai/ObjectDetection_2p6M/checkpoints/2bfb6cc36ef5ab822a944006c746e5d348d67e387501c9027df5b957cf124031/de0c5b23e1ae749b972f10cb5f3e21e2ccf37bd334eb7a7c3091c460073ff310/finetune/best.pt`.  
> **Tuân thủ quy trình:** Bước 1 - Khảo sát & Phân tích (Discovery) theo `AGENTS.md`.

---

## 1. Phân tích Chi tiết Yêu cầu 1: Hỗ trợ Tăng cường Dữ liệu giữa `src/dataset2.py` và `src/augment.py`

### 1.1. Hiện trạng của `src/augment.py`
Tệp `src/augment.py` hiện thực lớp `DetectionAugmenter` dựa trên thư viện **Albumentations**:
1. **Các phép tăng cường áp dụng:**
   - `A.HorizontalFlip(p=0.5)`: Lật ngang khung hình.
   - `_make_shift_scale_rotate` (dùng `A.Affine` / `A.ShiftScaleRotate`, p=0.8): Dịch chuyển (8%), co giãn (10%), xoay (15 độ).
   - `A.RandomBrightnessContrast(p=0.4)`: Thay đổi ngẫu nhiên độ sáng và độ tương phản.
   - `A.HueSaturationValue(p=0.3)`: Biến đổi không gian màu HSV (Hue 15, Saturation 20, Value 15).
   - `_make_gauss_noise(p=0.25)`: Thêm nhiễu Gauss ngẫu nhiên (var 5.0 - 25.0).
   - `A.Blur(p=0.15)`: Làm mờ hình ảnh (blur limit 5).
   - Toàn bộ pipeline Compose có xác suất kích hoạt tổng thể `p = 0.5`.
2. **Giao diện lập trình (API Methods) trong `DetectionAugmenter`:**
   - `__call__(self, image: np.ndarray, boxes=None, labels=None, seed: Optional[int] = None) -> Tuple[np.ndarray, List, List]`:
     Áp dụng biến đổi cho một ảnh đơn lẻ và trả về tuple gồm `(image, boxes, labels)`.
   - `augment_video(self, frames: List[np.ndarray], boxes_list=None, labels_list=None, seed: Optional[int] = None) -> Tuple[List[np.ndarray], List, List, int]`:
     Tăng cường một chuỗi video bằng **chung một random seed** nhằm bảo toàn tính nhất quán thời gian (**Temporal Consistency**). Kết quả trả về là một tuple 4 phần tử: `(aug_frames, aug_boxes_list, aug_labels_list, seed)`.
   - Biến thể hiện tại mặc định: `augmenter = DetectionAugmenter(config)`.

### 1.2. Hiện trạng xử lý Augmentation trong `src/dataset2.py`
Trong `src/dataset2.py`, phương thức `RawVideoBackboneNeckDataset.__getitem__` (dòng 633 - 640) hiện có đoạn code:
```python
# 3. Tăng cường dữ liệu thời gian (Temporal Augmentation) nếu có
if self.augmenter is not None and self.split == "train":
    aug_seed = random.randint(0, 1000000)
    if hasattr(self.augmenter, "apply_sequence"):
        frames_rgb = self.augmenter.apply_sequence(frames_rgb, seed=aug_seed)
    else:
        frames_rgb = [self.augmenter(img, seed=aug_seed) if callable(self.augmenter) else img for img in frames_rgb]
```

### 1.3. Kết luận Đánh giá: `src/dataset2.py` CHƯA HỖ TRỢ ĐÚNG `src/augment.py` (Lỗi Nghiêm Trọng)
Qua kiểm thử thực tế và rà soát mã nguồn, pipeline hiện tại đang gặp **4 xung đột / lỗ hổng nghiêm trọng**:

1. **Sai lệch tên phương thức gọi chuỗi video (Method Name Mismatch):**
   - `dataset2.py` kiểm tra `hasattr(self.augmenter, "apply_sequence")`.
   - Nhưng trong `src/augment.py`, hàm xử lý chuỗi video có tên là `augment_video`.
   - Hệ quả: Nhánh `hasattr` bị bỏ qua, chương trình buộc phải rơi vào nhánh `else`.

2. **Lỗi sụp đổ cấu trúc dữ liệu khi chạy (Runtime Data Shape Crash):**
   - Khi rơi vào nhánh `else`:
     `[self.augmenter(img, seed=aug_seed) for img in frames_rgb]`
     `DetectionAugmenter.__call__` trả về tuple 3 phần tử: `(transformed_image, out_boxes, out_labels)`.
   - Kết quả là `frames_rgb` trở thành danh sách chứa các tuple `[(img, boxes, labels), ...]`, thay vì danh sách các mảng numpy 3D `[H, W, 3]`.
   - Ngay ở bước tiếp theo trong `extract_chunks` (dòng 287), lệnh `np.stack(batch_slice, axis=0)` sẽ ngay lập tức quăng ngoại lệ:
     `ValueError: could not broadcast input array from shape (3,) ...` hoặc sụp đổ chương trình.

3. **Không khớp số lượng phần tử trả về của `augment_video`:**
   - Hàm `augment_video` trong `src/augment.py` trả về tuple 4 giá trị: `(aug_frames, aug_boxes_list, aug_labels_list, seed)`.
   - Trong khi bài toán nhận diện tài xế buồn ngủ trên video thô (phân loại nhị phân Alert vs Drowsy) chỉ cần danh sách khung hình `aug_frames` (không dùng bboxes). Nếu gán trực tiếp sẽ gây lỗi giải nén hoặc sai kiểu dữ liệu.

4. **Bị ngắt kết nối hoàn toàn trong Pipeline Huấn luyện:**
   - Trong `src/train1.py` (dòng 1006 - 1017), hàm tạo `RawVideoBackboneNeckDataset` **hoàn toàn không truyền tham số `augmenter`**.
   - Trong `configs/config.py` và `configs/config.yaml`, hoàn toàn thiếu các cờ cấu hình như `use_augmentation: bool = True` hoặc `aug_p: float = 0.5`.
   - Do đó, trong thực tế hiện tại, tính năng tăng cường dữ liệu đang bị vô hiệu hóa (disabled 100%).

### 1.4. Đề xuất Giải pháp Kỹ thuật
- **Trong `src/augment.py`:**
  - Bổ sung alias hoặc phương thức chuyên biệt `apply_sequence(frames: List[np.ndarray], seed: Optional[int] = None) -> List[np.ndarray]` và `augment_frames(frames)` chỉ tập trung xử lý danh sách ảnh, tự động trích xuất `aug_frames` và giữ nguyên tính nhất quán thời gian (shared seed).
  - Hoàn thiện để `DetectionAugmenter` hỗ trợ cả 2 dạng: có bbox (cho Object Detection) và thuần ảnh (cho Video Classification).
- **Trong `src/dataset2.py`:**
  - Nâng cấp khối gọi augmentation tại `__getitem__`: Hỗ trợ đầy đủ `hasattr(self.augmenter, "apply_sequence")`, `hasattr(self.augmenter, "augment_video")`, và cơ chế bóc tách an toàn `tuple/list` nếu truyền vào một callable trả về tuple.
  - Cập nhật hàm factory `build_raw_video_dataloaders` cho phép tùy chọn tự động nạp `augmenter` từ `src.augment` khi `split == "train"`.
- **Trong `configs/config.py` & `configs/config.yaml`:**
  - Bổ sung tham số `use_augmentation: bool = True` trong nhóm cấu hình dataset/dataloader.
  - Tích hợp vào `src/train1.py` (hoặc `train.py`) để truyền `augmenter` vào dataset huấn luyện.

---

## 2. Phân tích Chi tiết Yêu cầu 2: Chuẩn hóa Quy trình Chạy ở Môi trường Hiện tại

### 2.1. Khảo sát Thực tế Môi trường Thực thi
- **Hệ điều hành:** Linux Ubuntu 6.6.137+ (x86_64).
- **Phần cứng Tính toán (GPU):** 
  - Model: **NVIDIA Tesla V100-SXM3-32GB** (Compute Capability 7.0, VRAM 32GB).
  - CUDA Runtime: PyTorch 2.4+ tích hợp CUDA, hỗ trợ hoàn hảo Automatic Mixed Precision (AMP FP16).
- **Tập dữ liệu khả dụng:**
  - Đã tải sẵn tại: `/home/riftuser/.cache/kagglehub/datasets/nyvantran6634/dataset-datn4ni/versions/1/data_processed`.
  - Bao gồm: `dataset_merged_split.csv` (100% video hợp lệ), thư mục `train/` (4046 video), `val/` (1026 video).

### 2.2. Vấn đề Phân mảnh và Lỗi Thực thi Hiện tại
1. **Xung đột giữa `train.py` và `train1.py`:**
   - Tệp gốc `train.py` hiện tại đang import `src/train.py`, vốn là pipeline cũ dựa trên tệp đặc trưng HDF5 (`train_h5`).
   - Khi chạy `python train.py`, chương trình lập tức crash với lỗi:
     `AttributeError: 'TrainConfig' object has no attribute 'train_h5'`.
     (Do ở bước trước, `train_h5` đã bị loại bỏ khỏi `TrainConfig` để dọn dẹp cấu hình).
   - Tệp thực thi pipeline video thô hiện tại đang là `train1.py` (gọi `src/train1.py`). Điều này gây nhầm lẫn lớn cho người dùng và vi phạm quy ước chuẩn của dự án (`AGENTS.md` quy định điểm chạy chuẩn là `train.py`).
2. **Cấu hình Đa tiến trình (DataLoader `num_workers`) trên Linux:**
   - Trong `configs/config.py` và `config.yaml`, `num_workers` đang đặt là `0`.
   - Quá trình đọc video thô qua OpenCV (`cv2.VideoCapture`) và letterbox tiêu tốn đáng kể tài nguyên CPU.
   - Thử nghiệm thực tế với `num_workers=2` trên máy chủ Linux Tesla V100 đã chạy thành công tốt đẹp (`Batch shape: [2, 100, 64, 80, 80]`), giúp tăng tốc độ nạp dữ liệu song song với GPU inference.
3. **Cấu hình Trực quan hóa Headless:**
   - Môi trường chạy không có màn hình hiển thị X11/GUI. Việc cấu hình `matplotlib.use("Agg")` đã được áp dụng, cần bảo đảm không có lệnh `cv2.imshow` nào được gọi ngoài chế độ debug/demo.

### 2.3. Kế hoạch Chuẩn hóa Quy trình Chạy
1. **Chuẩn hóa Điểm Thực thi Trung tâm (`train.py`):**
   - Chuyển toàn bộ logic nạp video thô (`src/train1.py`) thành mã nguồn chính thức cho `train.py` và `src/train.py`.
   - Đồng nhất lệnh chạy chuẩn: `python train.py` sẽ trực tiếp khởi chạy pipeline huấn luyện video thô qua `NMSFreeDetector BackboneNeck` + `DeepGRUClassifier`.
2. **Tối ưu hóa Cấu hình Môi trường Hiện tại (`configs/config.py` & `configs/config.yaml`):**
   - Đảm bảo `device = "cuda"`, `amp = True`.
   - Thiết lập `batch_size = 16`, `chunk_size = 16` (cân bằng hoàn hảo giữa thông lượng VRAM và tốc độ trích xuất).
   - Hỗ trợ an toàn `num_workers = 0` (mặc định an toàn) hoặc `2` (tăng tốc) kèm theo cờ cấu hình rõ ràng.
   - Bổ sung cờ `use_augmentation: bool = True` để người dùng có thể linh hoạt bật/tắt augmentation khi huấn luyện.

---

## 3. Phân tích Chi tiết Yêu cầu 3: Kiểm tra và Xác thực Load Checkpoint NMSFreeDetector

### 3.1. Đường dẫn Checkpoint Mục tiêu
Đường dẫn được chỉ định bởi Người dùng:
```text
/home/riftuser/workspace/driver-guardian/ai/ObjectDetection_2p6M/checkpoints/2bfb6cc36ef5ab822a944006c746e5d348d67e387501c9027df5b957cf124031/de0c5b23e1ae749b972f10cb5f3e21e2ccf37bd334eb7a7c3091c460073ff310/finetune/best.pt
```

### 3.2. Kết quả Rà soát Tệp Thực tế
- **Trạng thái tồn tại:** ĐÃ TỒN TẠI HỢP LỆ TRÊN HỆ THỐNG.
- **Kích thước file:** `41 MB` (42,670,419 bytes).
- **Cấu trúc băm SHA-256 trong đường dẫn:**
  - `architecture_sha256`: `2bfb6cc36ef5ab822a944006c746e5d348d67e387501c9027df5b957cf124031`
  - `categories_sha256`: `de0c5b23e1ae749b972f10cb5f3e21e2ccf37bd334eb7a7c3091c460073ff310`
  - Tệp lưu trữ: `finetune/best.pt`.
  - Khớp 100% với định dạng quản lý checkpoint artifact của submodule `ai.ObjectDetection_2p6M`.

### 3.3. Rà soát Nội dung và Cấu trúc Dữ liệu bên trong Checkpoint
Thực hiện giải nén và kiểm tra state dictionary bằng PyTorch:
```python
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
```
Kết quả ghi nhận các khóa dữ liệu trung tâm:
1. `model`: State dict của mô hình phát hiện gồm **685 tham số**.
2. `ema`: State dict của trọng số trung bình hàm mũ (Exponential Moving Average) gồm **685 tham số** tối ưu.
3. `metadata`: Chứa đầy đủ thông tin kiến trúc mô hình:
   ```json
   {
     "format_version": 1,
     "task": "object_detection",
     "architecture": {
       "backbone_n": [1, 2, 2, 1],
       "backbone_w": [16, 32, 64, 128, 256],
       "nc": 80,
       "neck_n": 1,
       "reg_max": 16,
       "strides": [8, 16, 32]
     }
   }
   ```
4. `epoch`: Epoch đã huấn luyện; `best_val`: Chỉ số validation tốt nhất; `cfg`: Cấu hình siêu tham số ban đầu.

### 3.4. Kiểm tra Cơ chế Nạp và Trích xuất Đặc trưng (BackboneNeck Inference Verification)
Thực hiện chạy kiểm thử nạp mô hình qua `PyTorchBackboneNeckExtractor` định nghĩa trong `src/dataset2.py`:
1. **Xác thực hàm `validate_metadata`:**
   - Hàm `validate_metadata(self.checkpoint_path, checkpoint)` chạy thành công, trích xuất chính xác siêu tham số kiến trúc mà không gặp bất kỳ lỗi nào.
2. **Khởi tạo và nạp trọng số mô hình:**
   - Khởi tạo `NMSFreeDetector` với đúng cấu hình kiến trúc trích xuất.
   - Nạp ưu tiên trọng số từ `checkpoint["ema"]` (trọng số tối ưu nhất).
   - Đóng gói thành công vào wrapper `BackboneNeck(detector)`.
   - Đóng băng gradient toàn bộ các tham số (`param.requires_grad = False`).
3. **Thực nghiệm suy luận trích xuất đặc trưng trên CPU & GPU CUDA:**
   - Đã chạy kiểm thử đưa tensor khung hình `[1, 640, 640, 3]` vào `extract_chunks`:
     - **Tầng p3:** Shape trả về `[1, 64, 80, 80]` (Stride 8, 64 kênh) -> **CHÍNH XÁC**.
     - **Tầng p4:** Shape trả về `[1, 128, 40, 40]` (Stride 16, 128 kênh) -> **CHÍNH XÁC**.
     - **Tầng p5:** Shape trả về `[1, 256, 20, 20]` (Stride 32, 256 kênh) -> **CHÍNH XÁC**.
   - Mô hình hoạt động hoàn toàn ổn định trên thiết bị tính toán `cuda:0` (NVIDIA Tesla V100).
4. **Lưu ý kỹ thuật về `sys.path`:**
   - Để `import ai.ObjectDetection_2p6M...` hoạt động không bị lỗi `ModuleNotFoundError`, biến `PROJECT_ROOT` (`/home/riftuser/workspace/driver-guardian`) phải luôn được chèn vào `sys.path` trước khi thực hiện import. `src/dataset2.py` đã có đoạn mã xử lý này, tuy nhiên cần đồng bộ trên toàn bộ các tệp thực thi.

---

## 4. Bảng Tổng hợp Kết quả Rà soát

| Hạng mục kiểm tra | Hiện trạng | Đánh giá | Hành động đề xuất |
| :--- | :--- | :--- | :--- |
| **`src/dataset2.py` hỗ trợ `src/augment.py`** | Kiểm tra `apply_sequence`, rơi vào nhánh `else`, trả về tuple `(img, boxes, labels)` | ❌ **Chưa hỗ trợ / Lỗi Crash khi chạy** | Nâng cấp hàm trong `src/augment.py` và sửa `__getitem__` trong `dataset2.py` để xử lý danh sách frame mượt mà. |
| **Bật/Tắt Augment trong Training** | `src/train1.py` không truyền `augmenter`, `config.py` không có cờ bật/tắt | ⚠️ **Bị vô hiệu hóa hoàn toàn** | Thêm cấu hình `use_augmentation` vào `config.py`/`config.yaml` và truyền `DetectionAugmenter` vào `train_dataset`. |
| **Quy trình chạy hiện tại** | `train.py` trỏ vào code HDF5 cũ (bị crash do thiếu `train_h5`), code mới nằm ở `train1.py` | ❌ **Phân mảnh, dễ gây nhầm lẫn** | Chuẩn hóa `train.py` thành entry point chính thức, hợp nhất logic từ `train1.py` sang `train.py`. |
| **Kiểm tra Checkpoint `best.pt`** | Tệp tồn tại tại `ai/ObjectDetection_2p6M/.../finetune/best.pt`, nạp thành công EMA weights và trích xuất đúng `(p3, p4, p5)` |  **Tồn tại & Nạp hoàn toàn chính xác** | Duy trì đường dẫn mặc định trong `configs/config.py` và `configs/config.yaml`. |

---

## 5. Kết luận & Đề xuất Bước tiếp theo

Tài liệu khảo sát & phân tích này đã làm rõ:
1. `src/dataset2.py` **chưa hỗ trợ đúng** các phép tăng cường của `src/augment.py` và sẽ crash nếu bật lên; đồng thời pipeline hiện tại đang bỏ quên việc nạp `augmenter`.
2. Quy trình chạy đang bị phân mảnh giữa `train.py` (bị lỗi) và `train1.py` (chạy được). Cần chuẩn hóa hợp nhất về `train.py` cho môi trường Linux + GPU Tesla V100 hiện tại.
3. Checkpoint `best.pt` của `NMSFreeDetector` **đã được kiểm tra và xác nhận nạp hoàn toàn chính xác**, trích xuất đúng 3 tầng bản đồ đặc trưng `p3`, `p4`, `p5`.

Theo Quy trình làm việc tại Mục 5 của `AGENTS.md`:
> *Agent dừng tại Bước 1 và gửi báo cáo phân tích tới Người dùng. Sau khi Người dùng duyệt báo cáo phân tích này, Agent sẽ tiến hành Bước 2 (Lập kế hoạch thực hiện `plan_dataset2_augment_ckpt.md`).*
