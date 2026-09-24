# Cấu Trúc Tổ Chức Module ObjectDetection_2p6M

Tài liệu này mô tả chi tiết kiến trúc, cấu trúc thư mục, cơ chế quản lý trọng số (weights/checkpoints) và hướng dẫn người dùng sau khi clone repository từ GitHub biết cách tải và bố trí các thành phần checkpoint, metadata và mô hình ONNX để hệ thống có thể vận hành chính xác.

---

## 1. Tổng Quan

`ObjectDetection_2p6M` là module phát hiện vật thể (Object Detection) siêu nhẹ (~2.6M parameters), được thiết kế chuyên biệt cho các thiết bị biên (Edge Devices / Mobile / Camera nhúng).

* **Mô hình**: `NMSFreeDetector`
* **Đặc tính nổi bật**:
  * **End-to-End NMS-Free**: Không cần thuật toán hậu xử lý Non-Maximum Suppression (NMS) bên ngoài, giúp giảm độ trễ và tránh nghẽn CPU/NPU.
  * **Dual Assignment Head**: Áp dụng phân bổ One-to-One (o2o) cho inference thời gian thực và One-to-Many (o2m) hỗ trợ hội tụ khi training.
  * **DFL (Distribution Focal Loss)**: Tối ưu hóa dự đoán bounding box với độ chính xác cao dưới dạng phân phối xác suất.
  * **Two-stage Fine-tuning**: Hỗ trợ đóng băng thân mạng (backbone/neck) ở giai đoạn 1 và mở băng toàn bộ ở giai đoạn 2.
  * **Edge Runtime Ready**: Hỗ trợ phân tách và export mô hình ONNX thành 2 phần (`backbone_neck.onnx` và `head.onnx`) để chạy đa luồng hoặc tối ưu trên NPU/DSP.

---

## 2. Cây Thư Mục Toàn Bộ Module

Do chính sách lưu trữ mã nguồn và quy định trong `.gitignore`, **các file trọng số (`.pt`, `.pth`, `.onnx`,...) và thư mục checkpoint (`checkpoints/`) KHÔNG được đẩy lên GitHub**.

Dưới đây là sơ đồ toàn bộ cây thư mục, phân biệt rõ các file đã có trên GitHub và các file cần tải/tự sinh cục bộ:

```text
ObjectDetection_2p6M/
├── Structure.md                     # [GitHub] Tài liệu hướng dẫn cấu trúc & sắp xếp checkpoint (file này)
├── TRAINING_FIXES.md                # [GitHub] Lịch sử fix bug, ghi chú kỹ thuật về training/fine-tune
│
├── src/                             # [GitHub] Mã nguồn định nghĩa kiến trúc mô hình
│   ├── model.py                     # NMSFreeDetector chính (kết nối backbone_neck và head)
│   ├── backbone_neck.py             # Mạng trích xuất đặc trưng Backbone & PAN-FPN Neck
│   ├── blocks.py                    # Các khối tích chập cơ bản (Conv, Bottleneck, SPPF,...)
│   ├── head.py                      # Detection Head (DFL + One-to-One / One-to-Many)
│   └── config.py                    # Dataclass TrainConfig (siêu tham số train, augment, model)
│
├── finetune_/                       # [GitHub] Huấn luyện tinh chỉnh (Fine-tuning 2 giai đoạn)
│   ├── finetune_config.py           # Dataclass FineTuneConfig (kế thừa TrainConfig)
│   └── finetune_engine.py           # Engine fine-tune (freeze/unfreeze trunk, nạp pretrained)
│
├── inference/                       # [GitHub] Dự đoán với PyTorch (.pt)
│   ├── infer.py                     # Script inference PyTorch trên ảnh hoặc Camera/Webcam
│   ├── image.jpg                    # Ảnh mẫu test
│   └── girl.png                     # Ảnh mẫu test
│
├── runtime/                         # [GitHub + Local] Triển khai ONNX Runtime trên thiết bị biên
│   ├── convertor.py                 # [GitHub] Script xuất model PyTorch sang 2 file ONNX
│   ├── inferenceOnnx.py             # [GitHub] Script inference ONNX Runtime & benchmark FPS
│   ├── categories.json              # [GitHub] Mapping nhãn mặc định (80 lớp COCO)
│   ├── backbone_neck.onnx           # [LOCAL - Cần export hoặc tải về] Model ONNX trích xuất đặc trưng
│   ├── head.onnx                    # [LOCAL - Cần export hoặc tải về] Model ONNX dự đoán bbox/class
│   └── results/                     # [LOCAL] Thư mục chứa ảnh output sau inference ONNX
│
├── evaluation/                      # [GitHub] Đánh giá chất lượng mô hình (mAP)
│   ├── runeval.py                   # Script chạy đánh giá mAP trên tập validation
│   ├── ultralytics_evaluation.py    # Metric mAP@50, mAP@50-95 theo chuẩn COCO/Ultralytics
│   └── mAPEvaluation.py             # Tiện ích matching dự đoán và ground-truth
│
├── utils/                           # [GitHub] Tiện ích phụ trợ
│   ├── artifacts.py                 # Quản lý SHA-256 hash kiến trúc/nhãn và metadata
│   ├── checkpoint.py                # Lưu/nạp checkpoint (.pt), bảo toàn RNG, Scaler, EMA
│   ├── init_weights.py              # Khởi tạo trọng số mạng
│   ├── logging_setup.py             # Cấu hình log console & file
│   ├── seed.py                      # Cố định random seed
│   └── tb_logger.py                 # Tiện ích ghi log TensorBoard
│
└── checkpoints/                     # [LOCAL - BỊ GITIGNORE] Thư mục chứa trọng số mô hình
    └── <architecture_sha256>/       # Mã băm SHA-256 của kiến trúc (ví dụ: 2bfb6cc36ef5ab822...)
        └── <categories_sha256>/     # Mã băm SHA-256 của danh sách nhãn (ví dụ: de0c5b23e1ae...)
            └── <run_name>/          # Tên lần chạy (ví dụ: train, finetune)
                ├── architecture.json# [BẮT BUỘC] Metadata kiến trúc
                ├── categories.json  # [BẮT BUỘC] Danh mục nhãn đã chuẩn hóa
                ├── best.pt          # [BẮT BUỘC NẾU CÓ] Trọng số tốt nhất
                ├── last.pt          # Trọng số checkpoint cuối cùng
                └── ckpt_step*.pt    # Các checkpoint lưu định kỳ theo step
```

---

## 3. Cơ Chế Ràng Buộc Checkpoint & Metadata (Cực Kỳ Quan Trọng)

Module này được trang bị cơ chế bảo toàn tính toàn vẹn (integrity validation) trong `utils/artifacts.py` và `utils/checkpoint.py`.

### 3.1. Tại sao không thể chỉ copy một file `.pt` lẻ loi?

Khi gọi `torch.load` qua `validate_metadata(checkpoint_path, checkpoint)`:
1. Checkpoint `.pt` tự lưu metadata bên trong (gồm `architecture`, `categories`, mã băm SHA-256).
2. **Hệ thống bắt buộc phải tìm thấy 2 file metadata nằm cùng thư mục với file `.pt`**:
   * `architecture.json`: Mô tả 6 tham số kiến trúc (`nc`, `reg_max`, `backbone_w`, `backbone_n`, `neck_n`, `strides`) kèm mã hash SHA-256.
   * `categories.json`: Chứa danh sách các class được sắp xếp tăng dần theo ID dataset (`index`, `id`, `name`).
3. Nếu thiếu 1 trong 2 file này, hoặc nội dung không khớp với chữ ký đã lưu trong file `.pt`, chương trình sẽ lập tức báo lỗi:
   ```text
   ValueError: Thiếu hoặc sai metadata đi kèm checkpoint: .../architecture.json
   ```

### 3.2. Chữ ký SHA-256 mặc định

* **Kiến trúc mặc định** (`nc=80, reg_max=16, backbone_w=(16, 32, 64, 128, 256), backbone_n=(1, 2, 2, 1), neck_n=1, strides=(8, 16, 32)`):
  * **`architecture_sha256`**: `2bfb6cc36ef5ab822a944006c746e5d348d67e387501c9027df5b957cf124031`
* **Tập nhãn COCO 80 lớp**:
  * **`categories_sha256`**: `de0c5b23e1ae749b972f10cb5f3e21e2ccf37bd334eb7a7c3091c460073ff310`
* **Tập nhãn Object365**:
  * **`categories_sha256`**: `d9afe7f332a0080b29fdd068bbb94f32147807d062bc8dbfeb01f8b968f5ad22`

---

## 4. Hướng Dẫn Đặt Các Thành Phần Sau Khi Tải Về

Khi bạn nhận được checkpoint từ người khác hoặc tải từ Google Drive / Hugging Face / GitHub Releases:

### Trường hợp 1: Đặt theo cấu trúc chuẩn của hệ thống (Khuyên dùng cho Train / Fine-tune / Resume)

Tạo cấu trúc thư mục chuẩn trong `checkpoints/` như sau:

```text
ai/ObjectDetection_2p6M/checkpoints/
└── 2bfb6cc36ef5ab822a944006c746e5d348d67e387501c9027df5b957cf124031/
    └── de0c5b23e1ae749b972f10cb5f3e21e2ccf37bd334eb7a7c3091c460073ff310/
        └── finetune/
            ├── architecture.json
            ├── categories.json
            └── best.pt
```

*(Nếu dùng checkpoint pretrain từ Object365, thay thư mục hash `de0c5b2...` bằng `d9afe7f...` và thư mục con `train/`)*.

### Trường hợp 2: Đặt tự do để chạy Inference PyTorch hoặc Convert ONNX

Nếu bạn muốn đặt checkpoint ở một thư mục ngắn gọn (ví dụ `checkpoints/best.pt` hoặc `inference/weights/best.pt`), bạn **BẮT BUỘC** phải copy cả 2 file `architecture.json` và `categories.json` vào cùng thư mục đó:

```text
checkpoints/weights/
├── architecture.json
├── categories.json
└── best.pt
```

### Trường hợp 3: Đặt các file ONNX để chạy ONNX Runtime (`runtime/`)

Nếu bạn tải trực tiếp các file ONNX đã được convert sẵn:
* Đặt `backbone_neck.onnx` vào: `ai/ObjectDetection_2p6M/runtime/backbone_neck.onnx`
* Đặt `head.onnx` vào: `ai/ObjectDetection_2p6M/runtime/head.onnx`
* Đảm bảo file `runtime/categories.json` đã có và khớp với số lớp của mô hình (file này đã có sẵn trên Git cho bộ 80 lớp COCO).

---

## 5. Cấu Hình Đường Dẫn Trong Các Script Sau Khi Đặt File

Sau khi đã tải và đặt các file vào đúng vị trí, hãy kiểm tra và chỉnh sửa đường dẫn trong các file sau trước khi chạy:

### 5.1. Chạy Fine-tuning (`finetune_/finetune_config.py`)
Mở [finetune_config.py](file:///home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/ObjectDetection_2p6M/finetune_/finetune_config.py) và cấu hình lại:
```python
# Đường dẫn checkpoint pretrained (kèm architecture.json và categories.json cùng thư mục)
tfl_pretrained_pth: str = "checkpoints/2bfb6cc36ef5ab822a944006c746e5d348d67e387501c9027df5b957cf124031/d9afe7f332a0080b29fdd068bbb94f32147807d062bc8dbfeb01f8b968f5ad22/train/best.pt"

# Thư mục dữ liệu trên máy bạn
labels_root: str = "/path/to/your/labels"
images_root_dir: str = "/path/to/your/images"
index_cache_dir: str = "/path/to/your/cache"
```
Chạy lệnh:
```bash
python -m ai.ObjectDetection_2p6M.finetune_.finetune_engine
```

### 5.2. Chạy Inference PyTorch (`inference/infer.py`)
Mở [infer.py](file:///home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/ObjectDetection_2p6M/inference/infer.py), trong hàm `main()`:
```python
checkpoint_path = "checkpoints/2bfb6cc36ef5ab822a944006c746e5d348d67e387501c9027df5b957cf124031/de0c5b23e1ae749b972f10cb5f3e21e2ccf37bd334eb7a7c3091c460073ff310/finetune/best.pt"
images = ["inference/image.jpg"]   # hoặc để images = [] và camera_id = 0 để mở webcam
```
Chạy lệnh:
```bash
python -m ai.ObjectDetection_2p6M.inference.infer
```

### 5.3. Xuất Mô Hình sang ONNX (`runtime/convertor.py`)
Mở [convertor.py](file:///home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/ObjectDetection_2p6M/runtime/convertor.py), trong hàm `main()`:
```python
checkpoint_path = "checkpoints/path_to_your/best.pt"
output_dir = Path(__file__).resolve().parent  # sẽ xuất ra runtime/backbone_neck.onnx và runtime/head.onnx
```
Chạy lệnh:
```bash
python -m ai.ObjectDetection_2p6M.runtime.convertor
```

### 5.4. Chạy Inference ONNX Runtime (`runtime/inferenceOnnx.py`)
Mở [inferenceOnnx.py](file:///home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/ObjectDetection_2p6M/runtime/inferenceOnnx.py), trong hàm `main()`:
```python
backbone_neck_path = "runtime/backbone_neck.onnx"
head_path = "runtime/head.onnx"
categories_path = "runtime/categories.json"
images = ["inference/image.jpg"]   # hoặc camera_id = 0
```
Chạy lệnh:
```bash
python -m ai.ObjectDetection_2p6M.runtime.inferenceOnnx
```

### 5.5. Chạy Đánh Giá Model (`evaluation/runeval.py`)
Mở [runeval.py](file:///home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/ObjectDetection_2p6M/evaluation/runeval.py), trong hàm `main()`:
```python
checkpoint_path = "checkpoints/path_to_your/best.pt"
cfg = TrainConfig(
    labels_root="/path/to/dataset/labels",
    images_root_dir="/path/to/dataset/images",
    index_cache_dir="/path/to/dataset/cache",
    batch_size=4,
    device="cuda",
)
```
Chạy lệnh:
```bash
python -m ai.ObjectDetection_2p6M.evaluation.runeval
```

---

## 6. Mẫu Cấu Trúc Các File Metadata

Nếu bạn tải được file checkpoint `.pt` nhưng người gửi quên không đính kèm file metadata, bạn có thể tự tạo 2 file này và đặt ngay cạnh file `.pt`:

### 6.1. File `architecture.json` (Chuẩn cho mô hình 2.6M, 80 classes)
```json
{
  "model": {
    "backbone_n": [1, 2, 2, 1],
    "backbone_w": [16, 32, 64, 128, 256],
    "nc": 80,
    "neck_n": 1,
    "reg_max": 16,
    "strides": [8, 16, 32]
  },
  "sha256": "2bfb6cc36ef5ab822a944006c746e5d348d67e387501c9027df5b957cf124031"
}
```

### 6.2. File `categories.json` (Ví dụ định dạng COCO)
File là một JSON array, các class phải được sắp xếp tăng dần theo `id`, `index` bắt đầu từ `0`:
```json
[
  {
    "index": 0,
    "id": 0,
    "name": "aeroplane"
  },
  {
    "index": 1,
    "id": 1,
    "name": "apple"
  },
  ...
]
```
*(Tham khảo nội dung đầy đủ của 80 lớp tại [runtime/categories.json](file:///home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/ObjectDetection_2p6M/runtime/categories.json))*.

---

## 7. Tóm Tắt Quy Trình Bắt Đầu Nhanh (Quick Start Checklist)

- [ ] **Bước 1**: Clone repository về máy.
- [ ] **Bước 2**: Tải checkpoint `.pt` cùng `architecture.json` và `categories.json`.
- [ ] **Bước 3**: Đặt vào thư mục `checkpoints/` (theo cấu trúc SHA-256 hoặc một thư mục riêng biệt).
- [ ] **Bước 4**: Mở script cần dùng (`infer.py`, `convertor.py`, hoặc `finetune_config.py`) và cập nhật lại đường dẫn `checkpoint_path`.
- [ ] **Bước 5**: Chạy thử nghiệm và kiểm tra kết quả!
