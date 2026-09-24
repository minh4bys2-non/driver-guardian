# Cấu Trúc Tổ Chức Module ObjectDetection_24M

Tài liệu này mô tả chi tiết cấu trúc thư mục, kiến trúc mô hình, cơ chế quản lý trọng số (weights/checkpoints) và hướng dẫn người dùng sau khi tải về từ GitHub biết cách bố trí các thành phần checkpoint, file cấu hình và metadata để chạy inference hoặc huấn luyện mô hình Object Detection 24M.

---

## 1. Tổng Quan Mô Hình

`ObjectDetection_24M` là phiên bản mô hình phát hiện vật thể (Object Detection) tầm trung với khoảng **24 triệu tham số (~24M parameters)**, cung cấp độ chính xác cao hơn, khả năng nhận diện các vật thể nhỏ và đa dạng ngữ cảnh tốt hơn so với phiên bản siêu nhẹ 2.6M.

* **Kiến trúc chính**: `NMSFreeDetector`
* **Kích thước ảnh đầu vào mặc định**: `480 × 480` (pixel)
* **Đặc tính kỹ thuật**:
  * **End-to-End NMS-Free**: Tích hợp cơ chế One-to-One (o2o) loại bỏ hoàn toàn bước hậu xử lý NMS (Non-Maximum Suppression), cho phép suy luận trực tiếp từ tensor đầu ra mà không tốn tài nguyên lọc box trùng lặp.
  * **Dual Assignment Head**: Sử dụng nhánh One-to-One (`topk=1`) cho suy luận và One-to-Many (`topk=10`) trong quá trình huấn luyện nhằm tối đa hóa gradient học được từ ground truth.
  * **DFL (Distribution Focal Loss)**: Biểu diễn tọa độ hộp giới hạn dưới dạng phân phối xác suất rời rạc (`reg_max=16`), giúp tối ưu hóa định vị bounding box chính xác.
  * **Thân mạng sâu và rộng**:
    * Backbone width: `(56, 112, 224, 448, 640)`
    * Backbone depth (số block lặp lại): `(3, 6, 6, 3)`
    * PAFPN Neck depth: `3`
    * Strides: `(8, 16, 32)` tương ứng các feature maps P3, P4, P5

---

## 2. Cây Thư Mục Toàn Bộ Module

Theo quy định trong `.gitignore`, **các file trọng số (`.pt`, `.pth`, `.tar.xz`,...) và các thư mục checkpoint (`checkpoints*/`) KHÔNG được lưu trên GitHub** vì kích thước vượt quá giới hạn (file checkpoint `.pt` của model 24M khoảng **363MB**).

Dưới đây là sơ đồ chi tiết toàn bộ thư mục `ObjectDetection_24M`:

```text
ObjectDetection_24M/
├── Structure.md                  # [GitHub] Tài liệu cấu trúc thư mục & hướng dẫn đặt checkpoint (file này)
├── README.md                     # [GitHub] Hướng dẫn tóm tắt sử dụng
├── checkpoints_ftCOCO.tar.xz     # [LOCAL/TẢI VỀ - Bị Gitignore] File nén chứa bộ weights đã fine-tune trên COCO (~332MB)
│
├── checkpoints_ftCOCO/           # [LOCAL/TẢI VỀ - Bị Gitignore] Thư mục chứa weights & metadata đã fine-tune trên COCO
│   ├── model_mainfest.json       # [BẮT BUỘC] Metadata cấu hình kiến trúc model (dùng cho NMSFreeDetector.from_config)
│   ├── categories.jsonl          # [BẮT BUỘC] Danh mục 80 lớp đối tượng định dạng JSON Lines
│   └── ft_step00091000.pt        # [BẮT BUỘC] Checkpoint trọng số PyTorch (~363MB)
│
├── src/                          # [GitHub] Mã nguồn chính của mô hình
│   ├── model.py                  # Định nghĩa NMSFreeDetector (ghép nối Backbone, PAFPN Neck và DetectHead)
│   ├── backbone_neck.py          # Chi tiết mạng trích xuất đặc trưng Backbone & Path Aggregation FPN (PAFPN)
│   ├── blocks.py                 # Các khối tích chập cơ sở (Conv, Bottleneck, SPPF, RepNCSP,...)
│   ├── head.py                   # NMS-Free Detection Head (DFL, scale prediction, phân nhánh o2o/o2m)
│   ├── config.py                 # Dataclass TrainConfig (cấu hình siêu tham số, data pipeline, transfer learning)
│   │
│   ├── runtime/                  # Module suy luận (Inference)
│   │   ├── infer.py              # Script chạy nhận diện trên ảnh, lưu kết quả vẽ bounding box
│   │   ├── runcamera.py          # Script chạy nhận diện thời gian thực qua Webcam / Camera
│   │   ├── image.jpg             # Ảnh mẫu kiểm thử 1
│   │   ├── image.png             # Ảnh mẫu kiểm thử 2
│   │   ├── image3.jpg            # Ảnh mẫu kiểm thử 3
│   │   └── result/               # [LOCAL] Thư mục chứa ảnh output sau khi chạy infer.py
│   │
│   ├── train/                    # Module xử lý dữ liệu huấn luyện
│   │   └── dataloader1_obj365.py # Dataloader đọc nhãn JSONL, tiền xử lý ảnh letterbox và data augmentation
│   │
│   └── utils/                    # Tiện ích bổ trợ
│       ├── checkpoint.py         # Hàm save_checkpoint, load_checkpoint, load_model_only
│       └── init_weights.py       # Khởi tạo trọng số (Kaiming Normal, BatchNorm, focal prior bias)
│
└── delete/                       # [Tham khảo] Mã nguồn training và tài liệu phân tích luồng loss cũ
    ├── engine.py                 # Logic vòng lặp training cũ
    ├── training.py               # Script thực thi huấn luyện cũ
    ├── loss.py                   # Định nghĩa TaskAlignedAssigner, CIoU Loss, DFL Loss
    ├── loss_flow.md              # Sơ đồ Mermaid chi tiết luồng gán nhãn và tính toán Loss
    └── ema.py                    # Cập nhật Exponential Moving Average (EMA) cho trọng số
```

---

## 3. Cơ Chế Checkpoint & Metadata của Mô Hình 24M

Không giống như các mô hình thông thường chỉ nạp file `.pt` đơn thuần, mã nguồn `ObjectDetection_24M` thiết kế kiến trúc động thông qua file cấu hình manifest và danh mục nhãn dạng JSON Lines:

### 3.1. File cấu hình kiến trúc: `model_mainfest.json`
* Tên file mặc định trong repo là `model_mainfest.json` (chú ý ký tự `mainfest`).
* File này cho phép khởi tạo mô hình một cách linh hoạt mà không cần hardcode tham số trong Python:
  ```python
  model = NMSFreeDetector.from_config("checkpoints_ftCOCO/model_mainfest.json")
  ```
* Nội dung chứa định nghĩa cấu hình mạng 24M:
  ```json
  {
      "format_version": 1,
      "model": {
          "type": "NMSFreeDetector",
          "backbone_w": [56, 112, 224, 448, 640],
          "backbone_n": [3, 6, 6, 3],
          "neck_n": 3,
          "reg_max": 16,
          "num_classes": 80,
          "strides": [8, 16, 32]
      }
  }
  ```

### 3.2. File danh mục nhãn: `categories.jsonl`
* Định dạng: **JSON Lines (`.jsonl`)** — Mỗi dòng là một đối tượng JSON hợp lệ gồm `id` và `name`.
* Hàm `load_categories()` trong `src/train/dataloader1_obj365.py` đọc từng dòng và tự động sắp xếp theo thứ tự `id` tăng dần để ánh xạ sang chỉ số lớp (0 đến 79 cho COCO).
* Ví dụ:
  ```jsonl
  {"name": "aeroplane", "id": 0}
  {"name": "apple", "id": 1}
  {"name": "backpack", "id": 2}
  ...
  ```

### 3.3. File checkpoint PyTorch: `ft_step00091000.pt`
* File nhị phân lưu `state_dict` của mô hình PyTorch (kích thước ~363MB).
* Được nạp thông qua hàm `load_model_only()` trong `src/utils/checkpoint.py`.

---

## 4. Hướng Dẫn Sắp Xếp File Khi Tải Về

Khi bạn clone repository từ GitHub, thư mục `checkpoints_ftCOCO/` sẽ **chưa có file `.pt`** (hoặc chưa tồn tại).

### Bước 1: Chuẩn bị thư mục và giải nén / đặt file
Tùy theo nguồn nhận dữ liệu:

* **Nếu bạn nhận được file nén `checkpoints_ftCOCO.tar.xz`**:
  Chạy lệnh sau tại thư mục gốc `ai/ObjectDetection_24M`:
  ```bash
  tar -xf checkpoints_ftCOCO.tar.xz
  ```
  Lệnh này sẽ tự động giải nén ra thư mục `checkpoints_ftCOCO/` với đầy đủ 3 file cần thiết.

* **Nếu bạn tải các file lẻ từ Google Drive / Cloud Storage**:
  Tạo thư mục `checkpoints_ftCOCO/` ngay trong `ai/ObjectDetection_24M/` và đặt 3 file vào:
  ```text
  ai/ObjectDetection_24M/
  └── checkpoints_ftCOCO/
      ├── model_mainfest.json
      ├── categories.jsonl
      └── ft_step00091000.pt    (hoặc file checkpoint .pt bạn có)
  ```

### Bước 2: Kiểm tra tính sẵn sàng của các file
Đảm bảo các file sau đã có mặt trước khi chạy code:
1. `checkpoints_ftCOCO/model_mainfest.json`
2. `checkpoints_ftCOCO/categories.jsonl`
3. `checkpoints_ftCOCO/ft_step00091000.pt`

---

## 5. Hướng Dẫn Cấu Hình Đường Dẫn & Chạy Thử Nghiệm

Trước khi chạy, hãy mở các file runtime và cập nhật lại đường dẫn tuyệt đối hoặc tương đối phù hợp với môi trường máy của bạn:

### 5.1. Nhận diện trên ảnh tĩnh (`src/runtime/infer.py`)
Mở file [infer.py](file:///home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/ObjectDetection_24M/src/runtime/infer.py), trong khối `if __name__ == "__main__":` (khoảng dòng 155):

```python
# Cấu hình đường dẫn phù hợp với vị trí repo của bạn
config_path = "checkpoints_ftCOCO/model_mainfest.json"
categories_path = "checkpoints_ftCOCO/categories.jsonl"
checkpoint_path = "checkpoints_ftCOCO/ft_step00091000.pt"

model = NMSFreeDetector.from_config(config_path)
detector = NMSFreeInference(
    model=model,
    categories_path=categories_path,
    img_size=480,
    device="cuda",          # Chọn "cuda" hoặc "cpu"
    checkpoint_path=checkpoint_path,
    score_thres=0.25,
    max_det=30,
)
```

Chạy lệnh từ thư mục `ai/ObjectDetection_24M`:
```bash
python -m src.runtime.infer
```
Ảnh kết quả sau khi vẽ bounding box sẽ được lưu vào thư mục `src/runtime/result/`.

### 5.2. Nhận diện thời gian thực qua Camera/Webcam (`src/runtime/runcamera.py`)
Mở file [runcamera.py](file:///home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/ObjectDetection_24M/src/runtime/runcamera.py), kiểm tra phần cấu hình:

```python
model = NMSFreeDetector()   # Hoặc NMSFreeDetector.from_config("checkpoints_ftCOCO/model_mainfest.json")
detector = NMSFreeInference(
    model=model,
    categories_path="checkpoints_ftCOCO/categories.jsonl",
    checkpoint_path="checkpoints_ftCOCO/ft_step00091000.pt",
    img_size=480,
    device="cuda",          # "cuda" hoặc "cpu"
    score_thres=0.4,
    use_nms=False
)

# Chạy với Webcam ID 0
run_camera_detection(detector, camera_id=0)
```

Chạy lệnh:
```bash
python -m src.runtime.runcamera
```
*Nhấn phím `q` trên cửa sổ OpenCV để thoát luồng camera.*

---

## 6. Mẫu Cấu Hình Tự Khôi Phục (Ngoại Lệ)

Nếu bạn chỉ tải được duy nhất file checkpoint `.pt` mà không có các file metadata đi kèm, bạn có thể tự tạo chúng theo mẫu dưới đây:

### 6.1. Tạo file `checkpoints_ftCOCO/model_mainfest.json`
Tạo file với nội dung chuẩn của mô hình 24M:
```json
{
    "format_version": 1,
    "model": {
        "type": "NMSFreeDetector",
        "backbone_w": [56, 112, 224, 448, 640],
        "backbone_n": [3, 6, 6, 3],
        "neck_n": 3,
        "reg_max": 16,
        "num_classes": 80,
        "strides": [8, 16, 32]
    }
}
```

### 6.2. Tạo file `checkpoints_ftCOCO/categories.jsonl`
Mỗi dòng là một đối tượng JSON (chú ý đúng 80 lớp COCO, bắt đầu từ id 0):
```jsonl
{"name": "aeroplane", "id": 0}
{"name": "apple", "id": 1}
{"name": "backpack", "id": 2}
{"name": "banana", "id": 3}
{"name": "baseball bat", "id": 4}
...
```

---

## 7. Bảng Tóm Tắt Quy Trình Bắt Đầu Nhanh (Quick Checklist)

| Bước | Hành động | Chi tiết |
| :--- | :--- | :--- |
| **1** | Clone repository | `git clone ...` |
| **2** | Giải nén hoặc tạo folder | Giải nén `checkpoints_ftCOCO.tar.xz` hoặc tạo folder `checkpoints_ftCOCO/` |
| **3** | Đặt file checkpoint | Đặt `ft_step00091000.pt`, `model_mainfest.json`, `categories.jsonl` vào `checkpoints_ftCOCO/` |
| **4** | Cài đặt dependencies | `pip install torch torchvision opencv-python matplotlib numpy albumentations` |
| **5** | Kiểm tra đường dẫn | Sửa đường dẫn file trong `src/runtime/infer.py` hoặc `runcamera.py` |
| **6** | Chạy thử nghiệm | `python -m src.runtime.infer` |
