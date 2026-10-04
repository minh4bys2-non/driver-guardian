# AGENTS.MD — AI & Machine Learning Pipeline Guidelines

Tài liệu này thiết lập các nguyên tắc làm việc, cấu trúc dự án và quy trình thực hiện dành cho **Antigravity** khi triển
khai dự án xử lý dữ liệu, huấn luyện mô hình (AI/ML) thông qua các file Jupyter Notebook (`.ipynb`) và Python script
(`.py`).

---

## 1. Mục tiêu cốt lõi của Agent

- Xây dựng pipeline dữ liệu và huấn luyện mô hình AI ổn định, có khả năng tái lập (reproducible).
- **Thử nghiệm linh hoạt trong Notebooks (`.ipynb`)**: Dùng để EDA (Khám phá dữ liệu), trực quan hóa, và chạy thử nghiệm
  nhanh.
- **Chuẩn hóa vào Scripts (`.py`)**: Đóng gói mã nguồn sạch, module hóa, sẵn sàng cho automation, batch processing hoặc
  deployment.
- Đảm bảo tuân thủ cấu trúc thư mục, quản lý dependencies và ghi log đầy đủ.

---

## 2. Cấu trúc thư mục dự án tiêu chuẩn

Agent cần tuân thủ hoặc khởi tạo cấu trúc thư mục như sau:

```text
project-root
│
├── notebooks/           # Jupyter Notebooks theo từng giai đoạn
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_prototyping.ipynb
│
├── docs/           # chứ tài liệu thực hiện các nhiệm vụ
│   ├── images        # chứa ảnh demo hoặc ảnh trong file tài liệu
│   ├── video        # chứa video demo hoặc video trong file tài liệu
│   ├── plan        # chứa các tài liệu về kế hoạch thực hiện nhiệm vụ 
│   ├── analsys     # chứa các tài liệu về kết quả phân tích nhiệm vụ
│   └── report      # chứa các tài liệu về kết quả thực hiện nhiệm vụ   
│
├── src/                 # Mã nguồn tái sử dụng (.py)
│   ├── __init__.py
│   ├── data_loader.py   # Tải & kiểm tra cấu trúc dữ liệu
│   ├── preprocess.py    # Feature engineering & transform
│   ├── models.py        # Kiến trúc mô hình / wrapper
│   ├── train.py         # Pipeline huấn luyện
│   └── evaluate.py      # Đánh giá metric, visual validation
│
├── configs/             # Hyperparameters, paths (YAML hoặc JSON)
│   └── config.yaml
│
├── checkpoints/         # Weights, model artifacts (.pt, .onnx, .pkl)
├── logs/                # TensorBoard / Log training
├── requirements.txt     # Dependencies
├── README.md
└── agents.md
```

---

## 3. Quy chuẩn làm việc với Jupyter Notebooks (`.ipynb`)

1. **Quy tắc đặt tên**: Đặt tên có tiền tố số thứ tự để thể hiện luồng xử lý:
    - `01_data_exploration.ipynb`
    - `02_feature_engineering.ipynb`
    - `03_baseline_model.ipynb`
    - `04_hyperparameter_tuning.ipynb`
2. **Cấu trúc mỗi Notebook**:
    - **Cell đầu tiên**: Mô tả mục tiêu của notebook và các thư viện cần import.
    - **Cell cấu hình**: Đặt seed, đường dẫn path, biến toàn cục (`SEED = 42`,
      `DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'`).
    - Phân chia các section rõ ràng bằng Markdown headers (`#`, `##`, `###`).
3. **Quy chuẩn Code trong Notebook**:
    - Tránh biến toàn cục lộn xộn; gói các bước logic vào hàm nhỏ.
    - Hiển thị trực quan dữ liệu (plots, charts, sample dataframes, metrics).
    - Trước khi chốt kết quả, đảm bảo notebook có thể chạy mượt mà từ đầu đến cuối (**Restart Kernel and Run All Cells**
      không phát sinh lỗi).

---

## 4. Quy chuẩn làm việc với Python Scripts (`.py`)

1. **Tính Module hóa & Clean Code**:
    - Mỗi file đảm nhận một trách nhiệm duy nhất (Single Responsibility Principle).
    - Bắt buộc dùng **Type Hints** và **Docstrings** chuẩn (Google/NumPy style).
2. **Quản lý cấu hình & Hyperparameters**:
    - Không hardcode tham số trong code logic.
    - Sử dụng `argparse`, `click`, hoặc đọc cấu hình từ file `.yaml`/`.json`.
3. **Reproducibility (Tính tái lập)**:
    - Luôn định nghĩa hàm cố định seed cho Random, NumPy, PyTorch/TensorFlow:
      ```python
      def seed_everything(seed: int = 42):
          import random, os, numpy as np, torch
          random.seed(seed)
          os.environ['PYTHONHASHSEED'] = str(seed)
          np.random.seed(seed)
          torch.manual_seed(seed)
          torch.cuda.manual_seed_all(seed)
      ```
4. **Logging & Device handling**:
    - Dùng thư viện `logging` thay vì lạm dụng `print()`.
    - Tự động phát hiện và cấu hình tăng tốc phần cứng (`cuda`, `mps`, hoặc `cpu`).
5. **Entry Point Execution**:
    - Bọc code thực thi trong khối:
      ```python
      if __name__ == "__main__":
          main()
      ```

---

## 5. Quy trình làm việc từng bước của Agent (Action Workflow)

Khi nhận một nhiệm vụ mới từ người dùng, Agent cần tuân theo 5 bước sau:

### Bước 1: Khảo sát & Phân tích (Discovery)

- Tạo file `analsys_<viết tắt yêu cầu>.md` chứa các phân tích yêu cầu của user 
- Khi user chấp nhận `analsys_<viết tắt yêu cầu>.md` ở bước 1 thì mới thực hiện bước 2

### Bước 2: Lên kế hoạch thực hiện

- Tạo file `plan_<viết tắt yêu cầu>.md` chứa kế hoạch tiếp theo từ những phân tích của file
  `analsys_<viết tắt yêu cầu>.md` ở bước 1
- Khi user chấp nhận `plan_<viết tắt yêu cầu>.md` ở bước 2 thì mới thực hiện bước 3

### Bước 3: Thực hiện kế hoạch

- Tạo file các file tương ứng và thực hiện theo kế hoạch ở file `plan_<viết tắt yêu cầu>.md` ở bước 2
- Tạo file `report_<viết tắt yêu cầu>.md` chứa tổng hợp những gì đã thực hiện để hoàn thành nhiệm vụ

---

## 6. Tiêu chí chất lượng (Checklist trước khi hoàn tất)

- [ ] Toàn bộ đường dẫn file sử dụng `pathlib.Path` hoặc `os.path.join`, tương thích đa nền tảng.
- [ ] Xử lý ngoại lệ (Exception Handling) tại các bước đọc file và tải mô hình.
- [ ] Không rò rỉ dữ liệu (Data Leakage) giữa tập Train, Validation và Test.
- [ ] Checkpoint và file trọng số lớn không bị commit vào Git mà được lưu trữ đúng thư mục quy định.