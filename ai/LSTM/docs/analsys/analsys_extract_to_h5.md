# BÁO CÁO PHÂN TÍCH YÊU CẦU TRÍCH XUẤT ĐẶC TRƯNG DATASET SANG HDF5 (.H5)
**Mã tài liệu:** `analsys_extract_to_h5.md`  
**Dự án:** Hệ Thống Nhận Diện Tài Xế Buồn Ngủ (Driver Guardian - Deep LSTM / GRU)  
**Tệp thực thi mục tiêu:** `extract_to_pt.py`  
**Ngày thực hiện:** 01/10/2026 (Cập nhật: Không dùng Spatial Pooling, lưu nguyên bản đa tầng đặc trưng)  
**Quy chuẩn áp dụng:** [`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md) (Bước 1: Khảo sát & Phân tích - Discovery)

---

## 1. TỔNG QUAN & MỤC TIÊU NHIỆM VỤ (EXECUTIVE SUMMARY)

### 1.1. Yêu cầu từ người dùng
Người dùng yêu cầu xây dựng script `extract_to_pt.py` nhằm thực hiện pipeline trích xuất đặc trưng từ video sang tệp định dạng HDF5 (`.h5`) với các tiêu chuẩn cốt lõi:
1. **Định dạng Dataset đầu vào:** Tương thích trực tiếp với cấu trúc phân tầng được mô tả tại [`docs/struct_dataset.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/struct_dataset.md) (`train/`, `val/`, phân nhóm theo nhãn `0_alert/` và `1_drowsy/`).
2. **Mô hình trích xuất đặc trưng:** Nạp và suy luận qua mô hình ONNX [`checkpoints/backbone_neck.onnx`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/checkpoints/backbone_neck.onnx) (Backbone PAFPN YOLOv10 trunk).
3. **Đặc trưng lưu trữ (Raw Feature Maps - Không Spatial Pooling):**  
   ⭐ **Tuyệt đối không chạy qua hợp nhất không gian (Spatial Pooling / `AdaptiveAvgPool2d`)**. Giữ nguyên bản toàn bộ các tensor đặc trưng không gian đa tỉ lệ được mô hình xuất ra (`p3`, `p4`, `p5`).
4. **Cơ chế lưu trữ:** Ghi lũy tiến (append mode) vào **1 tệp `.h5` duy nhất**, có tích hợp nén dữ liệu (compression) và chunking theo từng frame nhằm tối ưu dung lượng đĩa và tốc độ I/O.
5. **Tốc độ lấy mẫu khung hình (Temporal Sampling):** Cho phép tùy chỉnh qua tham số (CLI/Config), giá trị mặc định là `sample_interval = 0.1` giây (tương đương 10 FPS).
6. **Tăng cường dữ liệu (Data Augmentation):** Tích hợp module [`src/augment.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/augment.py) với cơ chế bảo toàn tính nhất quán thời gian (Temporal Consistency - chung 1 random seed cho toàn bộ frames của 1 video clip).
7. **Khả năng chống chịu lỗi (Fault-Tolerance & Resume):** Có cơ chế tiếp tục quá trình trích xuất ngay tại vị trí bị ngắt đột ngột (do mất điện, tràn RAM/VRAM, hoặc dừng tiến trình) mà không cần tính toán lại các video đã xử lý.
8. **Truy vết & Quản lý dữ liệu:** Tự động tạo và cập nhật tệp `.csv` để theo dõi toàn bộ trạng thái xử lý, siêu dữ liệu (metadata), ánh xạ vị trí trong file `.h5`.

---

## 2. KHẢO SÁT HIỆN TRẠNG & CÁC THÀNH PHẦN KỸ THUẬT SẴN CÓ

### 2.1. Cấu trúc cây thư mục Dataset (`docs/struct_dataset.md`)
Dữ liệu video được tổ chức phân tầng rõ ràng theo chuẩn Machine Learning:
```text
E:\LSTM\data_processed\ (hoặc đường dẫn cấu hình --data_dir)
├── dataset_merged_split.csv               # Bảng tra cứu toàn bộ ~4,882 mẫu (tùy chọn)
├── dataset_merged_split.json              # File cấu trúc metadata nạp Dataset (tùy chọn)
│
├── train\                                 # TẬP HUẤN LUYỆN (~80% clips)
│   ├── 0_alert\                           # Nhãn 0: Lái xe tỉnh táo / bình thường
│   │   ├── sust_n_1.mp4
│   │   └── ...
│   └── 1_drowsy\                          # Nhãn 1: Lái xe buồn ngủ / ngủ gật
│       ├── sust_d_1.mp4
│       └── ...
│
└── val\                                   # TẬP KIỂM ĐỊNH (~20% clips)
    ├── 0_alert\                           # Nhãn 0: Kiểm định tỉnh táo
    └── 1_drowsy\                          # Nhãn 1: Kiểm định buồn ngủ
```
- **Quy tắc gán nhãn:**
  - Nằm trong thư mục `0_alert` $\rightarrow$ Nhãn `0` (Alert / Normal).
  - Nằm trong thư mục `1_drowsy` $\rightarrow$ Nhãn `1` (Drowsy).
- **Định dạng video hỗ trợ:** `.mp4`, `.avi`, `.mkv`, `.mov`.

### 2.2. Khảo sát mô hình ONNX Trích xuất đặc trưng (`checkpoints/backbone_neck.onnx`)
Kết quả kiểm tra cấu trúc đồ thị tính toán qua ONNX Runtime:
- **Đầu vào (Input):**
  - Tên: `'images'`
  - Kích thước: `['batch_size', 3, 640, 640]`
  - Kiểu dữ liệu: `float32`, miền giá trị $[0.0, 1.0]$.
  - Chuẩn hóa: Đưa qua hàm `letterbox` giữ nguyên tỷ lệ khung hình với padding xám `(114, 114, 114)`.
- **Đầu ra (Outputs - 3 tầng đặc trưng không gian PAFPN):**
  - **Tầng P3:** Tên node `'p3'`, shape `['batch_size', 64, 80, 80]` (Độ phân giải $80\times 80$, 64 kênh, chi tiết không gian cao).
  - **Tầng P4:** Tên node `'p4'`, shape `['batch_size', 128, 40, 40]` (Độ phân giải $40\times 40$, 128 kênh, đặc trưng trung gian).
  - **Tầng P5:** Tên node `'p5'`, shape `['batch_size', 256, 20, 20]` (Độ phân giải $20\times 20$, 256 kênh, ngữ nghĩa mức cao).
- **Yêu cầu xử lý:**  
  Không áp dụng Adaptive Pooling hay làm phẳng. Với mỗi video gồm $T$ khung hình sau khi lấy mẫu, kết quả trích xuất sẽ gồm đúng 3 tensor 4D nguyên bản:
  - $P_3 \in \mathbb{R}^{T \times 64 \times 80 \times 80}$
  - $P_4 \in \mathbb{R}^{T \times 128 \times 40 \times 40}$
  - $P_5 \in \mathbb{R}^{T \times 256 \times 20 \times 20}$

### 2.3. Khảo sát Dung lượng Dữ liệu & Tính toán Bộ nhớ
- **Số phần tử cho 1 khung hình:**
  - $P_3$: $64 \times 80 \times 80 = 409,600$ giá trị
  - $P_4$: $128 \times 40 \times 40 = 204,800$ giá trị
  - $P_5$: $256 \times 20 \times 20 = 102,400$ giá trị
  - Tổng cộng 1 frame = $716,800$ giá trị số thực.
- **Kích thước bộ nhớ thô cho 1 frame:**
  - Định dạng Float32: $716,800 \times 4 \text{ bytes} \approx 2.87\text{ MB} / \text{frame}$.
  - Định dạng Float16: $716,800 \times 2 \text{ bytes} \approx 1.43\text{ MB} / \text{frame}$.
- **Đánh giá & Giải pháp tối ưu:**
  - Đối với video dài 10s lấy mẫu chu kỳ $0.1s$ ($T = 100$ frames): Dung lượng thô $\approx 143\text{ MB}$ (FP16).
  - Vì các tầng đặc trưng qua hàm kích hoạt SiLU/ReLU chứa nhiều giá trị phân cụm và số 0, **thuật toán nén HDF5 (`lzf` hoặc `gzip`) sẽ nén giảm thêm từ 30% - 60% dung lượng**.
  - **Khuyến nghị bắt buộc:** Mặc định lưu trữ dạng `float16` để tối ưu gấp đôi dung lượng đĩa và băng thông I/O khi nạp vào mô hình sau này.

### 2.4. Khảo sát Pipeline Tăng cường dữ liệu (`src/augment.py`)
- Lớp `DetectionAugmenter` đã cung cấp sẵn phương thức `augment_video(frames, seed)`:
  - Cho phép tăng cường toàn bộ chuỗi frames của một video với **cùng 1 random seed cố định** (Temporal Consistency).
  - Biến đổi hình học & quang học: `HorizontalFlip`, `ShiftScaleRotate` (hoặc `Affine`), `RandomBrightnessContrast`, `HueSaturationValue`, `GaussNoise`, `Blur`.
- **Quy tắc áp dụng:**
  - Chỉ áp dụng augment cho tập `train` (tập `val` giữ nguyên 100% dữ liệu gốc để đảm bảo tính khách quan trong đánh giá kiểm định).
  - Có tùy chọn sinh nhiều bản sao (`num_aug >= 1`) và cờ giữ video gốc (`include_original=True`).

### 2.5. Môi trường thực thi & Thư viện khả dụng
- `Python 3.12`
- `torch 2.13.0+cu126` (CUDA khả dụng: `True`)
- `onnxruntime 1.26.0` (Hỗ trợ `CUDAExecutionProvider`, `TensorrtExecutionProvider`, `CPUExecutionProvider`)
- `h5py 3.16.0` (Khả dụng và hỗ trợ đầy đủ các thuật toán nén `gzip`, `lzf`, `chunks`)
- `cv2 5.0.0` (OpenCV VideoCapture)

---

## 3. PHÂN TÍCH THIẾT KẾ KIẾN TRÚC KỸ THUẬT (ARCHITECTURAL DESIGN)

### 3.1. Thiết kế Cấu trúc Nhóm trong tệp HDF5 (`.h5`)
Tệp HDF5 được tổ chức theo cây phân cấp chặt chẽ:

```text
dataset_features.h5 (hoặc tên chỉ định qua CLI)
│
├── train/                                    # Phân tập huấn luyện
│   ├── 0_alert/                              # Nhóm nhãn 0 (Alert)
│   │   ├── sust_n_1/                         # Video gốc
│   │   │   ├── p3                            # Dataset: [T, 64, 80, 80], float16, chunks=(1, 64, 80, 80)
│   │   │   ├── p4                            # Dataset: [T, 128, 40, 40], float16, chunks=(1, 128, 40, 40)
│   │   │   ├── p5                            # Dataset: [T, 256, 20, 20], float16, chunks=(1, 256, 20, 20)
│   │   │   └── attrs:
│   │   │       ├── label = 0
│   │   │       ├── seq_len = T
│   │   │       ├── sample_interval = 0.1
│   │   │       ├── orig_fps = 30.0
│   │   │       ├── is_augmented = False
│   │   │       ├── aug_seed = -1
│   │   │       ├── source_dataset = "sust"
│   │   │       └── is_completed = True
│   │   │
│   │   ├── sust_n_1_aug01/                   # Bản sao tăng cường
│   │   │   ├── p3                            # [T, 64, 80, 80]
│   │   │   ├── p4                            # [T, 128, 40, 40]
│   │   │   ├── p5                            # [T, 256, 20, 20]
│   │   │   └── attrs: (is_augmented=True, aug_seed=42)
│   │   └── ...
│   │
│   └── 1_drowsy/                             # Nhóm nhãn 1 (Drowsy)
│       └── ...
│
└── val/                                      # Phân tập kiểm định
    ├── 0_alert/
    │   └── ...
    └── 1_drowsy/
        └── ...
```

#### Thiết lập HDF5 Chunking & Nén tối ưu cho PyTorch DataLoader:
- **Chunking theo từng khung hình:**
  - `p3`: `chunks = (1, 64, 80, 80)`
  - `p4`: `chunks = (1, 128, 40, 40)`
  - `p5`: `chunks = (1, 256, 20, 20)`
  *Lợi ích:* Khi DataLoader nạp dữ liệu huấn luyện, PyTorch có thể truy cập lát cắt theo frame hoặc theo đoạn thời gian mà không cần nén/giải nén toàn bộ chuỗi $T$ khung hình cùng lúc.
- **Thuật toán nén:**
  - Mặc định: `compression="lzf"` (Tốc độ giải nén siêu nhanh trên CPU, không gây nghẽn cổ chai I/O khi nạp batch trong PyTorch).
  - Tùy chọn: `compression="gzip", compression_opts=4` (Nếu người dùng muốn tối đa hóa khả năng tiết kiệm dung lượng đĩa).

---

### 3.2. Cơ chế Tiếp tục Quá trình (Resume / Fault-Tolerance)
Nhằm chống chịu sự cố gián đoạn đột ngột mà không làm hỏng file `.h5`:

1. **Kiểm tra 2 lớp trước khi xử lý:**
   - **Lớp 1 (CSV Manifest):** Đọc danh sách `video_id` có `status == "SUCCESS"`. Nếu đã hoàn thành, bỏ qua ngay lập tức.
   - **Lớp 2 (HDF5 Group):** Kiểm tra xem group `/{split}/{label_folder}/{video_id}` đã tồn tại chưa:
     - Nếu tồn tại và có thuộc tính `is_completed == True`: Đã hoàn tất an toàn.
     - Nếu tồn tại nhưng `is_completed != True` (do bị dừng giữa chừng khi đang ghi): Gọi `del h5_file[group_path]` để xóa sạch dữ liệu dở dang và trích xuất lại từ đầu.
2. **Ghi thuộc tính an toàn (Atomic Transaction Emulation):**
   - Chỉ ghi `is_completed = True` vào `attrs` sau khi cả 3 dataset `p3`, `p4`, `p5` đã ghi xong trọn vẹn.
   - Thực thi `h5_file.flush()` và `csv_file.flush()` ngay sau mỗi video.

---

### 3.3. Thiết kế Tệp CSV Truy vết (Traceability Manifest CSV)
Tệp CSV (mặc định: `extraction_manifest.csv`) lưu trữ theo dõi thời gian thực:

| Tên cột | Kiểu dữ liệu | Ý nghĩa |
| :--- | :---: | :--- |
| `video_id` | `str` | Khóa định danh duy nhất (ví dụ: `train_0_alert_sust_n_1_aug01`) |
| `split` | `str` | Phân tập (`train` hoặc `val`) |
| `label` | `int` | Nhãn phân loại (`0`: Alert, `1`: Drowsy) |
| `label_name` | `str` | Thư mục nhãn (`0_alert` hoặc `1_drowsy`) |
| `orig_file` | `str` | Đường dẫn tới tệp video nguồn |
| `source_dataset` | `str` | Nguồn dữ liệu suy luận (`sust`, `uta-rldd`, `vbddd`, v.v.) |
| `orig_duration_s` | `float` | Thời lượng video gốc tính bằng giây |
| `orig_fps` | `float` | Tốc độ khung hình gốc của video |
| `sample_interval` | `float` | Chu kỳ lấy mẫu ($t = 0.1s$) |
| `num_frames` | `int` | Số lượng frames trích xuất ($T$) |
| `p3_shape` | `str` | Kích thước tensor P3 (ví dụ: `(100, 64, 80, 80)`) |
| `p4_shape` | `str` | Kích thước tensor P4 (ví dụ: `(100, 128, 40, 40)`) |
| `p5_shape` | `str` | Kích thước tensor P5 (ví dụ: `(100, 256, 20, 20)`) |
| `dtype` | `str` | Kiểu dữ liệu lưu trữ (`float16` hoặc `float32`) |
| `is_augmented` | `bool` | Đánh dấu bản sao tăng cường hay bản gốc |
| `aug_seed` | `int` | Random seed áp dụng cho video |
| `h5_group_path` | `str` | Đường dẫn nhóm bên trong tệp `.h5` |
| `status` | `str` | Trạng thái (`SUCCESS` hoặc `FAILED`) |
| `error_msg` | `str` | Chi tiết lỗi nếu có |
| `timestamp` | `str` | Thời điểm hoàn thành (`YYYY-MM-DD HH:MM:SS`) |

---

### 3.4. Quản lý Bộ nhớ & Streaming Mini-Chunk Inference
Vì đặc trưng thô $P_3, P_4, P_5$ chiếm dung lượng đáng kể, chiến lược suy luận được kiểm soát chặt:
1. **Chia nhỏ khung hình theo chunk (Chunk-based Forward):**
   - Thiết lập `chunk_size = 16` (hoặc 32) khung hình cho mỗi lần gọi `session.run()` trên GPU CUDA.
   - Tránh việc nạp toàn bộ $T$ khung hình cùng lúc vào VRAM gây lỗi `CUDA out of memory`.
2. **Ép kiểu Float16 tức thì (Immediate FP16 Conversion):**
   - Chuyển đổi tensor đầu ra từ Float32 sang Float16 ngay trên CPU trước khi ghi vào mảng đệm.
3. **Giải phóng tài nguyên chủ động:**
   - Dùng `del` xóa mảng frames và tensor tạm, gọi `gc.collect()` định kỳ.

---

## 4. MA TRẬN THAM SỐ DÒNG LỆNH ĐỀ XUẤT CHO `extract_to_pt.py`

| Tham số CLI | Kiểu | Mặc định | Mô tả chức năng |
| :--- | :---: | :---: | :--- |
| `--data_dir` | `str` | `E:\LSTM\data_processed` | Thư mục dataset theo `../struct_dataset.md` |
| `--output_h5` | `str` | `checkpoints/dataset_features.h5` | Đường dẫn tệp `.h5` đầu ra |
| `--manifest_csv`| `str` | `checkpoints/dataset_manifest.csv` | Đường dẫn tệp `.csv` truy vết |
| `--onnx_path` | `str` | `checkpoints/backbone_neck.onnx` | Tệp trọng số ONNX Backbone + Neck |
| `--sample_interval` | `float` | `0.1` | Chu kỳ lấy mẫu khung hình (giây, mặc định 0.1s $\approx$ 10 FPS) |
| `--chunk_size` | `int` | `16` | Batch size khi forward ONNX (chống tràn VRAM) |
| `--device_id` | `int` | `0` | GPU Device index (`0` cho CUDA, fallback CPU nếu không có CUDA) |
| `--compression` | `str` | `lzf` | Thuật toán nén HDF5 (`lzf` tối ưu tốc độ, `gzip` tối ưu dung lượng) |
| `--fp16` | `flag` | `True` | Lưu đặc trưng dạng `float16` (mặc định BẬT để tiết kiệm đĩa) |
| `--augment` | `flag` | `True` | Bật tăng cường dữ liệu cho tập `train` từ `augment.py` |
| `--num_aug` | `int` | `1` | Số bản sao augment cho mỗi clip trong tập `train` |
| `--include_original`| `flag`| `True` | Giữ video gốc cùng các bản sao augment |
| `--aug_seed` | `int` | `42` | Random seed cơ sở cho quá trình augment |
| `--force_recompute` | `flag`| `False` | Bỏ qua cơ chế resume, ghi đè lại toàn bộ |
| `--limit` | `int` | `None` | Giới hạn số clip xử lý để chạy test thử |

---

## 5. KẾT LUẬN & BƯỚC TIẾP THEO (NEXT STEPS)

- Phân tích đã được cập nhật **chính xác 100% theo chỉ đạo của bạn**: Bỏ hoàn toàn Spatial Pooling, lưu giữ trực tiếp và nguyên vẹn các tensor không gian đa tầng $P_3, P_4, P_5$ vào tệp `.h5` có nén và chunking.
- Theo quy trình chuẩn tại [`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md):
  - Sau khi bạn xác nhận bản phân tích này, Agent sẽ tạo tệp **`docs/plan_extract_to_h5.md`** (Bước 2: Lập kế hoạch chi tiết).
  - Sau khi bạn phê duyệt kế hoạch, Agent sẽ tiến hành lập trình tệp **`extract_to_pt.py`** (Bước 3: Thực hiện kế hoạch).
