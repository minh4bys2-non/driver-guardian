# KẾ HOẠCH TÁI CẤU TRÚC THƯ MỤC DỰ ÁN (PROJECT FOLDER RESTRUCTURING PLAN)
**Mã kế hoạch:** `plan_folder_structure.md`  
**Dự án:** Hệ Thống Nhận Diện Tài Xế Buồn Ngủ (Driver Guardian - Deep LSTM / GRU)  
**Đường dẫn gốc:** `D:\Project\DATN\driver-guardian\ai\LSTM`  
**Căn cứ phân tích:** [`analsys/analsys_folder_structure.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/analsys_folder_structure.md)  
**Quy chuẩn áp dụng:** [`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md) (Mục 2, 3, 4, 6)  
**Ngày lập kế hoạch:** 01/10/2026  

---

## 1. MỤC TIÊU VÀ NGUYÊN TẮC THỰC HIỆN

### 1.1. Mục tiêu
Chuyển đổi toàn diện cấu trúc thư mục phẳng hiện tại sang cấu trúc phân tầng chuẩn hóa công nghiệp cho dự án AI/ML theo đúng quy định tại `AGENTS.md`, bao gồm:
- Gom nhóm rõ ràng các tài liệu (`../../../../docs`), sổ tay thí nghiệm (`notebooks/`), mã nguồn tái sử dụng (`src/`), cấu hình (`configs/`), trọng số (`checkpoints/`) và nhật ký huấn luyện (`logs/`).
- Chuẩn hóa đặt tên Notebook có tiền tố thể hiện thứ tự pipeline.
- Đóng gói mã nguồn `src/` thành Python package hoàn chỉnh có `__init__.py`, thống nhất các module mô hình (LSTM và GRU).
- Đảm bảo **Zero Breaking Changes**: Toàn bộ đường dẫn import và file trọng số được cập nhật đồng bộ, không phát sinh lỗi khi chạy lại.

### 1.2. Nguyên tắc an toàn
1. **Bảo toàn dữ liệu & mã nguồn gốc:** Không xóa file nguồn khi chưa kiểm tra và di chuyển an toàn.
2. **Không commit weights lớn vào Git:** Quản lý `checkpoints/*.onnx` qua `../../../../.gitignore` và chỉ lưu trữ tệp `.gitkeep` trên kho mã nguồn.
3. **Độc lập nền tảng (Cross-platform paths):** Sử dụng `pathlib.Path` cho toàn bộ các cập nhật đường dẫn.

---

## 2. CẤU TRÚC MỤC TIÊU SAU KHI TÁI CẤU TRÚC

```text
D:\Project\DATN\driver-guardian\ai\LSTM\
│
├── notebooks/                       # Sổ tay Jupyter theo thứ tự pipeline
│   ├── 01_train_lstm.ipynb          # (Nguyên bản datn4ni2.ipynb)
│   ├── 02_train_gru.ipynb           # (Nguyên bản datn4ni3.ipynb)
│   ├── 03_train_experiments.ipynb   # (Nguyên bản datn4ni4.ipynb)
│   ├── 04_evaluate_lstm.ipynb       # (Nguyên bản test_model.ipynb)
│   └── 05_evaluate_gru.ipynb        # (Nguyên bản test_model1.ipynb)
│
├── docs/                            # Tài liệu phân tích và kiến trúc
│   ├── images/
│   │   └── img.png                  # Tệp hình ảnh sơ đồ/minh họa
│   ├── analysis_data_processed.md   # Phân tích tập dữ liệu 5,072 clips
│   ├── struct_dataset.md            # Cấu trúc thư mục dataset trên đĩa E:\
│   ├── analsys_folder_structure.md  # Báo cáo phân tích hiện trạng (Bước 1)
│   └── plan_folder_structure.md     # Bản kế hoạch thực hiện này (Bước 2)
│
├── src/                             # Mã nguồn tái sử dụng (.py)
│   ├── __init__.py                  # Đóng gói package src
│   ├── augment.py                   # Data Augmentation pipeline (Albumentations)
│   ├── loss.py                      # DrowsinessLoss (CrossEntropy hỗ trợ Frame/Sequence)
│   ├── models.py                    # SpatialFeatureAdapter, DeepLSTMClassifier, DeepGRUClassifier
│   └── evaluate.py                  # Đo lường & kiểm thử ONNX Runtime (từ test_onnx.py)
│
├── configs/                         # Quản lý siêu tham số và cấu hình hệ thống
│   ├── config.yaml                  # File cấu hình chuẩn hóa định dạng YAML
│   └── config.py                    # Dataclass TrainConfig nạp từ YAML hoặc default
│
├── checkpoints/                     # Trọng số mô hình và checkpoint huấn luyện
│   ├── .gitkeep                     # Giữ thư mục trên git
│   ├── backbone.onnx                # Trọng số PAFPN Backbone
│   └── backbone_neck.onnx           # Trọng số PAFPN Backbone + Neck
│
├── logs/                            # Nhật ký huấn luyện và TensorBoard
│   └── .gitkeep
│
├── .gitignore                       # Cấu hình bỏ qua checkpoint lớn & pycache
├── requirements.txt                 # Danh sách thư viện và dependencies
├── README.md                        # Hướng dẫn dự án và cách sử dụng
└── AGENTS.md                        # Bộ quy tắc chỉ đạo hoạt động của Agent
```

---

## 3. CÁC GIAI ĐOẠN TRIỂN KHAI CHI TIẾT (IMPLEMENTATION PHASES)

### Giai đoạn 1: Khởi tạo cấu trúc thư mục & thiết lập Git protection
- **Bước 1.1:** Tạo các thư mục con: `notebooks/`, `../../../../docs`, `docs/images/`, `src/`, `configs/`, `checkpoints/`, `logs/`.
- **Bước 1.2:** Tạo các tệp `.gitkeep` trong `logs/` và `checkpoints/`.
- **Bước 1.3:** Cập nhật / tạo file `../../../../.gitignore` tại root để ngăn chặn commit các tệp `.onnx`, `.pt`, `.pth`, `.pkl`, `.bin` trong thư mục `checkpoints/` cũng như `__pycache__/`.

### Giai đoạn 2: Di chuyển tài liệu và hình ảnh vào `../../../../docs`
- **Bước 2.1:** Di chuyển `../analsys/analysis_data_processed.md` và `struct_dataset.md` vào `docs/`.
- **Bước 2.2:** Di chuyển `../analsys/analsys_folder_structure.md` và `plan_folder_structure.md` vào `docs/`.
- **Bước 2.3:** Di chuyển `img.png` vào `docs/images/img.png`.
- **Bước 2.4:** Kiểm tra và cập nhật các liên kết tương đối trong các tệp tài liệu nếu có tham chiếu hình ảnh.

### Giai đoạn 3: Di chuyển tệp trọng số vào `checkpoints/`
- **Bước 3.1:** Di chuyển `backbone.onnx` và `backbone_neck.onnx` vào thư mục `checkpoints/`.
- **Bước 3.2:** Đảm bảo kích thước và checksum của 2 tệp ONNX được bảo toàn nguyên vẹn sau khi di chuyển.

### Giai đoạn 4: Chuẩn hóa thư mục cấu hình `configs/`
- **Bước 4.1:** Tạo `configs/config.yaml` chứa đầy đủ cấu hình dataset paths, sequence length, image size, batch size, learning rate, loss config.
- **Bước 4.2:** Di chuyển `config.py` vào `configs/config.py`, bổ sung hàm tiện ích `load_config(yaml_path)` để đọc từ YAML hoặc dùng giá trị mặc định.
- **Bước 4.3:** Tạo cầu nối tương thích (wrapper / proxy) nếu cần để các script cũ không bị lỗi import.

### Giai đoạn 5: Đóng gói và module hóa mã nguồn trong `src/`
- **Bước 5.1:** Tạo `src/__init__.py` export các lớp và hàm chính (`CNNAdapter`, `DeepLSTMClassifier`, `DeepGRUClassifier`, `DrowsinessLoss`, `TrainConfig`).
- **Bước 5.2:** Di chuyển `augment.py` vào `src/augment.py`.
- **Bước 5.3:** Di chuyển `loss.py` vào `src/loss.py`, cập nhật import sang `configs.config`.
- **Bước 5.4:** Hợp nhất `model.py` (Deep LSTM) và `model1.py` (Deep GRU) vào `src/models.py`:
  - Dùng chung lớp `CNNAdapter` (tránh duplicate code).
  - Tách bạch 2 kiến trúc `DeepLSTMClassifier` và `DeepGRUClassifier`.
- **Bước 5.5:** Di chuyển `test_onnx.py` thành `src/evaluate.py`, cập nhật đường dẫn `backbone_neck_path` tự động trỏ tới `checkpoints/backbone_neck.onnx` bằng `pathlib.Path`.

### Giai đoạn 6: Di chuyển và chuẩn hóa thứ tự `notebooks/`
- **Bước 6.1:** Sao chép/Di chuyển có đổi tên chuẩn:
  - `datn4ni2.ipynb` $\rightarrow$ `notebooks/01_train_lstm.ipynb`
  - `datn4ni3.ipynb` $\rightarrow$ `notebooks/02_train_gru.ipynb`
  - `datn4ni4.ipynb` $\rightarrow$ `notebooks/03_train_experiments.ipynb`
  - `test_model.ipynb` $\rightarrow$ `notebooks/04_evaluate_lstm.ipynb`
  - `test_model1.ipynb` $\rightarrow$ `notebooks/05_evaluate_gru.ipynb`
- **Bước 6.2:** Thêm cell cấu hình `sys.path.append(str(Path.cwd().parent))` ở đầu mỗi notebook (nếu notebook cần import từ `src` và `configs`).

### Giai đoạn 7: Bổ sung các tệp chuẩn dự án
- **Bước 7.1:** Tạo `requirements.txt` chuẩn hóa các thư viện phụ thuộc (`torch`, `torchvision`, `albumentations`, `opencv-python`, `onnxruntime`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `pyyaml`).
- **Bước 7.2:** Tạo `../../../../README.md` mô tả tổng quan dự án, cấu trúc thư mục, quy trình huấn luyện và kiểm thử.
- **Bước 7.3:** Dọn dẹp các tệp `__pycache__` thừa ở thư mục gốc.

### Giai đoạn 8: Kiểm tra toàn diện & Nghiệm thu (Verification Gate)
- **Bước 8.1 (Import Test):** Kiểm tra lệnh import từ Python:
  ```bash
  python -c "from src.models import DeepLSTMClassifier, DeepGRUClassifier; from src.loss import DrowsinessLoss; from configs.config import TrainConfig; print('Import test: PASSED')"
  ```
- **Bước 8.2 (Model Loading Test):** Kiểm tra suy luận thử nghiệm với `checkpoints/backbone_neck.onnx`:
  ```bash
  python src/evaluate.py
  ```
- **Bước 8.3 (Git Status Check):** Kiểm tra trạng thái git, xác nhận cấu trúc sạch sẽ, các file trọng số không bị tracked bừa bãi.

---

## 4. MA TRẬN ÁNH XẠ FILE (FILE MIGRATION MATRIX)

| Tệp Hiện Tại (Root) | Vị Trí & Tên Mới Sau Tái Cấu Trúc | Hành Động | Ghi Chú |
| :--- | :--- | :---: | :--- |
| `../analsys/analysis_data_processed.md` | `docs/analysis_data_processed.md` | Move | Tài liệu phân tích dataset |
| `../struct_dataset.md` | `docs/struct_dataset.md` | Move | Tài liệu cấu trúc thư mục dữ liệu |
| `../analsys/analsys_folder_structure.md` | `docs/analsys_folder_structure.md` | Move | Tài liệu phân tích Bước 1 |
| `img.png` | `docs/images/img.png` | Move | Hình ảnh minh họa |
| `backbone.onnx` | `checkpoints/backbone.onnx` | Move | Trọng số CNN backbone |
| `backbone_neck.onnx` | `checkpoints/backbone_neck.onnx` | Move | Trọng số CNN backbone+neck |
| `config.py` | `configs/config.py` | Move + Refactor | Hỗ trợ YAML & dataclass |
| *(Mới)* | `configs/config.yaml` | Create | Cấu hình siêu tham số dạng YAML |
| *(Mới)* | `src/__init__.py` | Create | Khởi tạo package src |
| `augment.py` | `src/augment.py` | Move | Data augmentation module |
| `loss.py` | `src/loss.py` | Move + Update import | Cập nhật import configs |
| `model.py` & `model1.py` | `src/models.py` | Merge & Refactor | Gom chung Adapter + LSTM + GRU |
| `test_onnx.py` | `src/evaluate.py` | Move + Refactor | Cập nhật đường dẫn tới checkpoints/ |
| `datn4ni2.ipynb` | `notebooks/01_train_lstm.ipynb` | Move & Rename | Chuẩn hóa theo thứ tự luồng |
| `datn4ni3.ipynb` | `notebooks/02_train_gru.ipynb` | Move & Rename | Chuẩn hóa theo thứ tự luồng |
| `datn4ni4.ipynb` | `notebooks/03_train_experiments.ipynb` | Move & Rename | Chuẩn hóa theo thứ tự luồng |
| `test_model.ipynb` | `notebooks/04_evaluate_lstm.ipynb` | Move & Rename | Chuẩn hóa theo thứ tự luồng |
| `test_model1.ipynb` | `notebooks/05_evaluate_gru.ipynb` | Move & Rename | Chuẩn hóa theo thứ tự luồng |
| *(Mới)* | `requirements.txt` | Create | Danh sách dependencies |
| *(Mới)* | `../../../../README.md` | Create | Tài liệu hướng dẫn dự án |
| *(Mới)* | `../../../../.gitignore` | Create/Update | Loại trừ file checkpoint lớn |
| `AGENTS.md` | `AGENTS.md` | Giữ nguyên tại Root | Theo đúng quy chuẩn Agent |

---

## 5. BƯỚC TIẾP THEO

Theo đúng **Bước 3** trong quy trình chuẩn tại [`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md):
- Sau khi bạn phê duyệt bản kế hoạch này, Agent sẽ tiến hành **Bước 3: Thực hiện kế hoạch** bằng cách tạo các thư mục, di chuyển và chuẩn hóa code theo đúng ma trận trên, sau đó chạy kiểm thử nghiệm thu để đảm bảo hệ thống hoạt động hoàn hảo.
