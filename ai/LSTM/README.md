# Driver Guardian — Hệ Thống Nhận Diện Tài Xế Buồn Ngủ (Deep LSTM & GRU Pipeline)

Dự án nghiên cứu và phát triển mô hình học sâu hai giai đoạn (Two-Stage Decoupled Pipeline) nhằm phát hiện sớm trạng thái buồn ngủ của người lái xe, kết hợp giữa mạng tích chập trích xuất đặc trưng không gian (CNN Backbone + PAFPN Neck qua ONNX) và mạng nơ-ron hồi quy chuỗi thời gian (Spatial Feature Adapter + Deep LSTM / Deep GRU).

---

## 1. Cấu Trúc Thư Mục Chuẩn Hóa

Cấu trúc dự án tuân thủ nghiêm ngặt quy chuẩn kiến trúc AI/ML Pipeline được quy định trong [`AGENTS.md`](AGENTS.md):

```text
LSTM/
│
├── notebooks/                       # Sổ tay Jupyter theo thứ tự thực thi pipeline
│   ├── 01_train_lstm.ipynb          # Pipeline huấn luyện mô hình Deep LSTM
│   ├── 02_train_gru.ipynb           # Pipeline huấn luyện mô hình Deep GRU
│   ├── 03_train_experiments.ipynb   # Thử nghiệm mở rộng & siêu tham số
│   ├── 04_evaluate_lstm.ipynb       # Đánh giá toàn diện mô hình LSTM (Confusion matrix, ROC, F1)
│   └── 05_evaluate_gru.ipynb        # Đánh giá toàn diện mô hình GRU
│
├── docs/                            # Tài liệu nghiên cứu, phân tích và kiến trúc
│   ├── images/
│   │   └── img.png                  # Sơ đồ và hình ảnh minh họa
│   ├── analysis_data_processed.md   # Phân tích tập dữ liệu chuẩn hóa 5,072 clips
│   ├── struct_dataset.md            # Cấu trúc lưu trữ dữ liệu thực tế
│   ├── analsys_folder_structure.md  # Báo cáo phân tích hiện trạng thư mục (Bước 1)
│   └── plan_folder_structure.md     # Kế hoạch chi tiết tái cấu trúc thư mục (Bước 2)
│
├── src/                             # Mã nguồn Python module hóa
│   ├── __init__.py                  # Khởi tạo package
│   ├── models.py                    # SpatialFeatureAdapter, DeepLSTMClassifier, DeepGRUClassifier
│   ├── loss.py                      # DrowsinessLoss (CrossEntropy) & DrowsinessBCELoss
│   ├── augment.py                   # Data Augmentation pipeline (Albumentations)
│   └── evaluate.py                  # Đo lường & kiểm thử trích xuất đặc trưng qua ONNX Runtime
│
├── configs/                         # Cấu hình siêu tham số và đường dẫn
│   ├── __init__.py
│   ├── config.yaml                  # Cấu hình chuẩn định dạng YAML
│   └── config.py                    # Dataclass TrainConfig hỗ trợ load từ YAML/JSON
│
├── checkpoints/                     # Trọng số mô hình (.onnx, .pt, .pth)
│   ├── .gitkeep
│   ├── backbone.onnx                # Trọng số CNN Backbone
│   └── backbone_neck.onnx           # Trọng số CNN Backbone + Neck (PAFPN)
│
├── logs/                            # Nhật ký huấn luyện và TensorBoard runs
│   └── .gitkeep
│
├── .gitignore                       # Loại trừ checkpoints lớn và cache
├── requirements.txt                 # Danh sách thư viện phụ thuộc
├── README.md                        # Giới thiệu và tài liệu dự án
└── AGENTS.md                        # Nguyên tắc làm việc và quy chuẩn của Agent
```

---

## 2. Cài Đặt Môi Trường

Khuyến nghị sử dụng Python 3.10+ cùng môi trường ảo (Conda hoặc venv):

```bash
# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

---

## 3. Hướng Dẫn Sử Dụng

### 3.1. Nạp Cấu Hình
```python
from configs.config import load_config, TrainConfig

# Nạp từ file YAML chuẩn
cfg = load_config("configs/config.yaml")
print(f"Batch size: {cfg.batch_size}, Epochs: {cfg.epochs}")
```

### 3.2. Khởi Tạo Mô Hình (Deep LSTM hoặc Deep GRU)
```python
import torch
from src.models import DeepLSTMClassifier, DeepGRUClassifier, build_model

# Cách 1: Khởi tạo từ config
model = build_model(cfg, model_type="lstm")

# Cách 2: Khởi tạo trực tiếp
model_gru = DeepGRUClassifier(input_dim=256, hidden_dim=256, num_layers=3, num_classes=2)
```

### 3.3. Kiểm Thử Suy Luận ONNX
```bash
python src/evaluate.py
```

### 3.4. Chạy Sổ Tay Huấn Luyện & Đánh Giá
Mở Jupyter Notebook và thực thi các sổ tay theo thứ tự tại thư mục `notebooks/`:

