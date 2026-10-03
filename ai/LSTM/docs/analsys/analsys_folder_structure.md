# BÁO CÁO PHÂN TÍCH HIỆN TRẠNG TỔ CHỨC THƯ MỤC DỰ ÁN (FOLDER STRUCTURE ANALYSIS)
**Mã tài liệu:** `analsys_folder_structure.md`  
**Dự án:** Hệ Thống Nhận Diện Tài Xế Buồn Ngủ (Driver Guardian - Deep LSTM / GRU)  
**Đường dẫn gốc:** `D:\Project\DATN\driver-guardian\ai\LSTM`  
**Ngày thực hiện:** 01/10/2026  
**Trạng thái kiểm tra:** ❌ **CHƯA ĐẠT CHUẨN** (Cấu trúc hiện tại ở dạng phẳng, chưa phân tầng theo quy chuẩn `AGENTS.md`)

---

## 1. TỔNG QUAN & MỤC TIÊU PHÂN TÍCH (EXECUTIVE SUMMARY)

Yêu cầu từ người dùng: **"Kiểm tra xem cách tổ chức folder đã đúng với định dạng chưa"**.

Dựa trên tiêu chuẩn được quy định trong tài liệu nguyên tắc của dự án ([`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md) - Mục 2 & Mục 6), cấu trúc thư mục chuẩn cần đáp ứng:
1. **Phân tách rõ ràng** giữa mã nguồn tái sử dụng (`src/`), sổ tay thử nghiệm (`notebooks/`), tài liệu kỹ thuật (`../../../../docs`), tệp cấu hình (`configs/`), trọng số mô hình (`checkpoints/`) và nhật ký huấn luyện (`logs/`).
2. **Quy chuẩn đặt tên** Notebooks có tiền tố số thứ tự theo từng giai đoạn xử lý dữ liệu và huấn luyện.
3. **Tiêu chí an toàn mã nguồn**: Các tệp trọng số nhị phân lớn (`.pt`, `.onnx`, `.pkl`) phải được lưu trữ riêng trong `checkpoints/`, tránh commit nhầm vào Git.

---

## 2. HIỆN TRẠNG THỰC TẾ THƯ MỤC HIỆN TẠI (CURRENT STATE AUDIT)

Hiện tại, tất cả các tệp trong thư mục `D:\Project\DATN\driver-guardian\ai\LSTM\` đều nằm trải phẳng (flat) tại thư mục gốc, chưa có bất kỳ thư mục con phân loại chức năng nào:

```text
D:\Project\DATN\driver-guardian\ai\LSTM\
│
├── __pycache__/                 # Cache bytecode Python
├── AGENTS.md                    # Hướng dẫn quy chuẩn Agent & Pipeline AI/ML
├── analysis_data_processed.md   # Báo cáo phân tích tập dữ liệu đã xử lý
├── struct_dataset.md            # Tài liệu cấu trúc cây thư mục dataset
├── augment.py                   # Module Data Augmentation cho video/ảnh
├── backbone.onnx                # Trọng số mô hình ONNX (Backbone)
├── backbone_neck.onnx           # Trọng số mô hình ONNX (Backbone + Neck)
├── config.py                    # Cấu hình huấn luyện (Dataclass TrainConfig)
├── datn4ni2.ipynb               # Sổ tay huấn luyện / thử nghiệm LSTM (Phiên bản 2)
├── datn4ni3.ipynb               # Sổ tay thử nghiệm / tối ưu (Phiên bản 3)
├── datn4ni4.ipynb               # Sổ tay thử nghiệm / mở rộng (Phiên bản 4)
├── img.png                      # Tệp hình ảnh minh họa/kiểm thử
├── loss.py                      # Hàm mất mát DrowsinessLoss
├── model.py                     # Kiến trúc mạng SpatialFeatureAdapter & Deep LSTM
├── model1.py                    # Kiến trúc mạng SpatialFeatureAdapter & Deep GRU
├── test_model.ipynb             # Sổ tay kiểm thử mô hình LSTM
├── test_model1.ipynb            # Sổ tay kiểm thử mô hình GRU
└── test_onnx.py                 # Mã kiểm thử suy luận qua ONNX Runtime
```

---

## 3. ĐỐI CHIẾU CHI TIẾT: CHUẨN QUY ĐỊNH VS. THỰC TẾ

| Hạng mục / Thư mục | Quy định tại `AGENTS.md` | Hiện trạng thực tế | Đánh giá | Rủi ro / Điểm cần khắc phục |
| :--- | :--- | :--- | :---: | :--- |
| **`notebooks/`** | Gom toàn bộ `.ipynb`, đặt tên theo luồng có thứ tự (`01_...`, `02_...`) | Đang nằm trực tiếp ở thư mục gốc (`datn4ni2.ipynb`, `test_model.ipynb`, v.v.) | ❌ **Chưa đạt** | Gây rối mắt ở root; không thể hiện rõ thứ tự thực thi pipeline cho người mới tiếp cận. |
| **`../../../../docs`** | Chứa tài liệu kỹ thuật, báo cáo phân tích, kiến trúc | Đang để các file `.md` ở thư mục gốc (`analysis_data_processed.md`, `struct_dataset.md`) | ❌ **Chưa đạt** | Làm ô nhiễm không gian thư mục gốc dự án. |
| **`src/`** | Chứa mã nguồn Python tái sử dụng (`__init__.py`, `data_loader.py`, `models.py`, `train.py`, `evaluate.py`) | Đang để các file `.py` ở thư mục gốc (`augment.py`, `model.py`, `model1.py`, `loss.py`, `test_onnx.py`) | ❌ **Chưa đạt** | Chưa đóng gói thành package; không có `__init__.py`; khó import tái sử dụng; dễ xung đột namespace. |
| **`configs/`** | Lưu cấu hình tham số dạng YAML/JSON (`configs/config.yaml`) | Đang dùng file script Python phẳng `config.py` ở root | ❌ **Chưa đạt** | Chưa tách biệt giữa cấu hình thuần túy (data/hyperparams) và logic code. |
| **`checkpoints/`** | Lưu trữ weights, model artifacts (`.pt`, `.onnx`, `.pkl`) | Hai tệp ONNX lớn (`backbone.onnx`, `backbone_neck.onnx`) đang nằm ở root | ❌ **Chưa đạt** | Vi phạm Checklist Tiêu chí chất lượng (Mục 6); nguy cơ commit nhầm file nhị phân lớn vào git. |
| **`logs/`** | Lưu log training, TensorBoard logs | Chưa tồn tại | ❌ **Chưa đạt** | Chưa có nơi gom log tập trung cho quá trình huấn luyện tự động. |
| **`requirements.txt`** | Quản lý các dependencies của môi trường | Chưa tồn tại | ❌ **Chưa đạt** | Gây khó khăn cho tính tái lập (reproducibility) khi triển khai trên máy khác. |
| **`../../../../README.md`** | Tài liệu giới thiệu tổng quan dự án | Chưa tồn tại | ❌ **Chưa đạt** | Thiếu hướng dẫn cài đặt và chạy hệ thống. |

---

## 4. KẾT LUẬN & ĐÁNH GIÁ MỨC ĐỘ TUÂN THỦ

- **Tỷ lệ tuân thủ cấu trúc phân tầng:** **0/8 thư mục/tệp thành phần tiêu chuẩn**.
- **Kết luận:** Cách tổ chức thư mục hiện tại **CHƯA ĐÚNG** với định dạng chuẩn đã quy ước trong [`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md).

---

## 5. ĐỀ XUẤT ÁNH XẠ CHUYỂN ĐỔI (PROPOSED MIGRATION MAPPING)

Để chuyển đổi cấu trúc hiện tại sang cấu trúc chuẩn mực mà **không làm gãy đường dẫn import hoặc gián đoạn luồng làm việc**, đề xuất sơ đồ ánh xạ như sau:

```text
LSTM (project-root)/
│
├── notebooks/
│   ├── 01_eda_and_data_check.ipynb         # (Nếu có hoặc tách từ quá trình trích xuất)
│   ├── 02_feature_extraction_pipeline.ipynb # Các notebook trích xuất đặc trưng
│   ├── 03_train_lstm_datn4ni2.ipynb        # <- datn4ni2.ipynb
│   ├── 04_train_gru_datn4ni3.ipynb         # <- datn4ni3.ipynb
│   ├── 05_train_experiments_datn4ni4.ipynb  # <- datn4ni4.ipynb
│   ├── 06_evaluate_lstm.ipynb              # <- test_model.ipynb
│   └── 07_evaluate_gru.ipynb               # <- test_model1.ipynb
│
├── docs/
│   ├── analysis_data_processed.md          # <- Chuyển từ root vào
│   ├── struct_dataset.md                   # <- Chuyển từ root vào
│   ├── analsys_folder_structure.md         # <- Tài liệu phân tích này
│   └── images/
│       └── img.png                         # <- Chuyển từ root vào
│
├── src/
│   ├── __init__.py                         # Khởi tạo package
│   ├── augment.py                          # <- augment.py
│   ├── loss.py                             # <- loss.py
│   ├── models.py                           # <- Hợp nhất model.py (LSTM) & model1.py (GRU)
│   └── evaluate.py                         # <- Tái cấu trúc từ test_onnx.py
│
├── configs/
│   ├── config.yaml                         # Cấu hình siêu tham số & đường dẫn
│   └── config.py                           # Dataclass load từ YAML hoặc giữ tạm thời
│
├── checkpoints/
│   ├── backbone.onnx                       # <- Chuyển từ root vào
│   └── backbone_neck.onnx                  # <- Chuyển từ root vào
│
├── logs/                                   # Thư mục trống kèm .gitkeep
├── requirements.txt                        # Danh sách thư viện cần thiết
├── README.md                               # Hướng dẫn tổng quan dự án
└── AGENTS.md                               # Quy chuẩn agent (giữ nguyên tại root)
```

---

## 6. BƯỚC TIẾP THEO (NEXT STEP)

Theo **Bước 2** trong quy trình chuẩn tại [`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md):
- Sau khi bạn xem xét và chốt bản phân tích này, Agent sẽ tạo file **`../plan/plan_folder_structure.md`** để lập kế hoạch chi tiết từng bước:
  1. Tạo các thư mục con tiêu chuẩn (`notebooks/`, `../../../../docs`, `src/`, `configs/`, `checkpoints/`, `logs/`).
  2. Di chuyển các tệp về đúng vị trí.
  3. Cập nhật các đường dẫn `import` trong code Python và Notebooks để đảm bảo mọi thứ vẫn chạy trơn tru không phát sinh lỗi.
  4. Tạo các file bổ trợ còn thiếu (`requirements.txt`, `../../../../README.md`, `__init__.py`).
