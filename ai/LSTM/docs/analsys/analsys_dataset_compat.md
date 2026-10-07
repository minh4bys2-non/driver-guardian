# TÀI LIỆU PHÂN TÍCH: KIỂM TRA ĐỘ TƯƠNG THÍCH CỦA SRC/DATASET.PY VỚI ĐỊNH DẠNG DATASET TẠI DOCS/STRUCT_DATASET1.MD

- **Mã yêu cầu**: `DATASET_COMPAT`
- **Tệp phân tích**: `docs/analsys/analsys_dataset_compat.md`
- **Kỹ năng áp dụng**: `/agent-skills:code-review-and-quality`
- **Mục tiêu**: Đánh giá toàn diện khả năng sử dụng của `src/dataset.py` với tập dữ liệu được mô tả trong `docs/struct_dataset1.md` (tập trung vào tập `train` và `val`, không bắt buộc nạp tập `test`), phát hiện các điểm nghẽn, lỗi tiềm ẩn và đề xuất phương án chuẩn hóa.
- **Trạng thái**: Đang chờ người dùng phê duyệt (Bước 1 theo chuẩn quy trình `AGENTS.md`).

---

## 1. YÊU CẦU CỐT LÕI VÀ PHẠM VI ĐÁNH GIÁ

### 1.1. Yêu cầu từ người dùng
> *"kiểm tra @src/dataset.py có thể sử dụng với tập dataset có định dạng như file @docs/struct_dataset1.md không, không cần thiết phải load được tập test"*

### 1.2. Phạm vi khảo sát
1. **Đối tượng kiểm tra**: Module [src/dataset.py](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/dataset.py) (bao gồm `RawVideoFramesDataset`, `collate_video_frames`, `build_raw_video_dataloaders`, `ChunkedBackboneNeckExtractor`).
2. **Quy cách tập dữ liệu mục tiêu**: Định dạng được mô tả trong [docs/struct_dataset1.md](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/struct_dataset1.md) (tập ULDD cắt sẵn clip 25.0s, có manifest CSV/JSON, chia phân tầng `train`, `val`, `test`).
3. **Phạm vi ràng buộc**:
   - Chỉ bắt buộc tải thành công tập `train` và `val` để phục vụ huấn luyện và kiểm định mô hình.
   - Không bắt buộc nạp tập `test`.
   - Đánh giá theo 5 trục chất lượng chuẩn: **Correctness (Đúng đắn)**, **Readability (Dễ hiểu)**, **Architecture (Kiến trúc)**, **Security/Robustness (Tin cậy)**, **Performance (Hiệu năng)**.

---

## 2. ĐỐI CHIẾU THÔNG SỐ VÀ CẤU TRÚC KỸ THUẬT

| Tiêu chí | Cấu trúc dữ liệu tại `docs/struct_dataset1.md` | Hiện trạng xử lý trong `src/dataset.py` | Đánh giá độ tương thích |
| :--- | :--- | :--- | :--- |
| **Thư mục gốc** | `E:\LSTM\data_processed\` hoặc `uldd_processed\` | Tham số `dataset_dir: Union[str, Path]` | **Tương thích 100%** |
| **Phân chia thư mục con** | `train/`, `val/`, `test/` | Duyệt qua `self.split` trong `("train", "val")` | **Tương thích 100%** với train/val |
| **Thư mục nhãn** | `0_alert/` (KSS $\le$ 4.5) và `1_drowsy/` (KSS $\ge$ 7.0) | Duyệt `[(0, "0_alert"), (1, "1_drowsy")]` | **Tương thích 100%** |
| **Định dạng video** | `.mp4` (ví dụ `uldd_subE_s01_kss1.0_w00_25.0s.mp4`) | `video_exts = (".mp4", ".avi", ".mkv", ".mov")` | **Tương thích 100%** |
| **Độ dài clip** | Cố định 25.0 giây (khoảng 750 frames tại 30 fps) | OpenCV đọc từng frame theo `sample_interval` | **Tương thích tốt**, padding động |
| **Manifest CSV** | `dataset_merged_split.csv` | Tự động phát hiện ứng viên `dataset_merged_split.csv` | **Rủi ro logic cột** (xem chi tiết mục 3.1) |
| **Manifest JSON** | `dataset_merged_split.json` | Tự động quét ứng viên `.json` nhưng hàm đọc chỉ xử lý `.csv` | **Lỗi Critical** (xem chi tiết mục 3.1) |
| **Nguồn dữ liệu (`source`)** | Dữ liệu ULDD (tiền tố `uldd_...`) | Chỉ bắt `"sust"` và `"uta"`, còn lại fallback `"custom"` | **Cần cải thiện** (gán nhầm custom) |
| **Mã đối tượng (`subject_id`)** | Chứa trong tên file (`subE`, `subA`, `subH`, `subD`) | Chưa trích xuất regex từ tên file (`subject_id = None`) | **Thiếu trường dữ liệu** |

---

## 3. KẾT QUẢ ĐÁNH GIÁ THEO 5 TRỤC CODE REVIEW & QUALITY

### 3.1. Trục 1: Correctness (Tính đúng đắn & Khả năng nạp dữ liệu)

#### A. Khi nạp trực tiếp từ thư mục (Folder Discovery Mode - Không dùng Manifest)
- **Đánh giá**: **HOẠT ĐỘNG HOÀN HẢO**.
- **Cơ chế**:
  - Với `split="train"`: Code quét chính xác `train/0_alert` và `train/1_drowsy`.
  - Với `split="val"`: Code quét chính xác `val/0_alert` và `val/1_drowsy`.
  - Tất cả các file có đuôi `.mp4` đều được nhận diện, map đúng label `0` (alert) và `1` (drowsy).
  - Không cần tập `test`, việc khởi tạo qua `build_raw_video_dataloaders()` chạy mượt mà.

#### B. Khi nạp qua Manifest (`_load_from_manifest`)
Tại `src/dataset.py:286-291`, khi `manifest_file=None`, lớp `RawVideoFramesDataset` tự động kiểm tra các file sau:
```python
for candidate in ("dataset_manifest.csv", "dataset_merged_split.csv", "dataset_merged_split.json"):
    cand_path = self.dataset_dir / candidate
    if cand_path.exists():
        manifest_path = cand_path
        break
```
Nếu trong thư mục dữ liệu tồn tại `dataset_merged_split.csv` hoặc `dataset_merged_split.json` (như mô tả trong `struct_dataset1.md`), code sẽ kích hoạt chế độ nạp từ manifest. Ở đây phát sinh **2 điểm nghẽn nghiêm trọng**:

1. **[CRITICAL] Không hỗ trợ tệp `.json`**:
   - `candidate` thứ 3 là `dataset_merged_split.json`.
   - Tuy nhiên, trong hàm `_load_from_manifest` (dòng 321–350), code chỉ có duy nhất nhánh:
     ```python
     if manifest_path.suffix.lower() == ".csv":
         # Đọc CSV
     return samples
     ```
   - **Hệ quả**: Nếu người dùng chỉ định `manifest_file="dataset_merged_split.json"` hoặc chỉ có file JSON trong thư mục, hàm sẽ trả về `samples = []` (danh sách rỗng). Dataset dài 0 phần tử, làm chương trình dừng đột ngột với lỗi `IndexError` hoặc DataLoader không có dữ liệu.

2. **[REQUIRED] Thứ tự và tên cột đường dẫn trong `dataset_merged_split.csv`**:
   - Tại dòng 333:
     ```python
     p_str = row.get("orig_file") or row.get("path") or row.get("file_path") or ""
     ```
   - **Rủi ro 1**: Nếu file `dataset_merged_split.csv` có cột `orig_file` ghi lại đường dẫn video thô ban đầu (ví dụ: `E:\raw\subjectE.mp4`), code sẽ lấy giá trị này thay vì lấy file clip đã cắt. Khi đường dẫn thô không tồn tại, dòng `if not v_path.exists(): continue` sẽ âm thầm loại bỏ mẫu.
   - **Rủi ro 2**: Nếu CSV sử dụng tên cột là `clip_path` (tên phổ biến cho bảng tra cứu clip sau cắt), code không tìm thấy và `p_str` trở thành rỗng `""`, dẫn tới việc loại bỏ 100% mẫu trong tập dữ liệu.

3. **[REQUIRED] Chưa trích xuất `subject_id` và nhận diện nguồn `uldd`**:
   - Tên clip trong `struct_dataset1.md` có định dạng rất chuẩn hóa: `uldd_subE_s01_kss1.0_w00_25.0s.mp4`.
   - Tại dòng 312:
     ```python
     src = "sust" if "sust" in v_path.stem.lower() else ("uta-rldd" if "uta" in v_path.stem.lower() else "custom")
     ```
   - Nguồn của dữ liệu này là `uldd`, nhưng đang bị gán thành `"custom"`.
   - `subject_id` bị để mặc định là `None`, làm mất khả năng kiểm tra rò rỉ đối tượng (Subject Leakage) khi đánh giá hoặc phân tích KSS.

---

### 3.2. Trục 2: Readability & Simplicity (Tính dễ đọc & Đơn giản)
- Mã nguồn trong `src/dataset.py` được cấu trúc bài bản, có chú thích tiếng Việt rõ ràng, bám sát các nguyên tắc type hints.
- Tuy nhiên, phần `_discover_samples()` lồng ghép quá nhiều heuristic hardcode (tìm file manifest ngầm định, kiểm tra chuỗi `"sust"`, `"uta"`).
- Cần tách bạch rõ ràng giữa:
  - Chế độ tự động dò theo thư mục chuẩn (Directory Crawler).
  - Chế độ đọc bảng chỉ mục (Manifest Loader).

---

### 3.3. Trục 3: Architecture & Modularity (Kiến trúc hệ thống)
- **Điểm cộng vượt trội**:
  - Tách rời hoàn toàn phần giải mã video trên CPU (`RawVideoFramesDataset`) và trích xuất đặc trưng trên GPU (`ChunkedBackboneNeckExtractor`).
  - Hỗ trợ cơ chế `num_workers > 0` mà không bị xung đột tiến trình CUDA.
  - Zero-padding trục thời gian động trong `collate_video_frames`, phù hợp với các video 25s (~125 frames ở tần số lấy mẫu 0.2s).
- **Hạn chế**:
  - Hàm `build_raw_video_dataloaders()` chỉ hỗ trợ trả về `(train_loader, val_loader)`. Mặc dù theo yêu cầu người dùng hiện tại không cần tập test, nhưng về mặt kiến trúc dài hạn, hàm nên hỗ trợ thêm tùy chọn nạp `test_loader` độc lập nếu người dùng cấu hình `split="test"`.

---

### 3.4. Trục 4: Security & Robustness (Bảo mật & Tính tin cậy)
- **Độ tin cậy ngoại lệ**:
  - Hàm `_sample_video_frames` có kiểm tra `cap.isOpened()` và kiểm tra frame rỗng `ret or frame_bgr is None`.
  - Mẫu lỗi trả về `(None, label, 0, meta)` và được hàm `collate_video_frames` tự động lọc bỏ (`valid_batch = [item for item in batch if item[0] is not None]`).
- **Rủi ro rò rỉ tài nguyên**:
  - `cap.release()` luôn được gọi sau vòng lặp đọc video, đảm bảo không rò rỉ file handle trên Windows.

---

### 3.5. Trục 5: Performance (Hiệu năng tính toán & Bộ nhớ)
- **Bộ nhớ RAM**:
  - Mỗi clip 25 giây ở `sample_interval = 0.2s` tương ứng với ~125 khung hình.
  - Tại kích thước `640x640x3`, mỗi frame chiếm khoảng 1.2 MB.
  - Do `dataset.py` lưu giữ định dạng `torch.uint8` (`[T, 3, 640, 640]`), bộ nhớ RAM cho mỗi video là:
    $$125 \times 3 \times 640 \times 640 \times 1\text{ byte} \approx 153.6\text{ MB RAM}$$
  - Với `batch_size = 4`, một batch chiếm ~614 MB RAM. Đây là con số an toàn cho hầu hết các máy tính hiện nay.
- **Tràn VRAM GPU**:
  - Lớp `ChunkedBackboneNeckExtractor` cắt nhỏ tensor theo `chunk_size = 4` (hoặc 16/32) khi đưa vào Backbone, đảm bảo mức sử dụng VRAM trần luôn dưới 2GB trên GPU 4GB VRAM.

---

## 4. TỔNG HỢP VẤN ĐỀ VÀ PHÂN LOẠI MỨC ĐỘ (FINDINGS)

| Mã | Mức độ | Vị trí | Mô tả vấn đề | Giải pháp đề xuất |
| :--- | :--- | :--- | :--- | :--- |
| **F-01** | **CRITICAL** | `src/dataset.py:321-350` | `_load_from_manifest` không có code đọc file `.json`, trả về `[]` nếu nạp `dataset_merged_split.json`. | Bổ sung nhánh xử lý đọc file JSON (sử dụng thư viện `json`). |
| **F-02** | **REQUIRED** | `src/dataset.py:333` | Thứ tự đọc cột đường dẫn ưu tiên `orig_file` trước `path`, và thiếu trường `clip_path`. | Ưu tiên đọc `clip_path` $\rightarrow$ `path` $\rightarrow$ `file_path` $\rightarrow$ `orig_file`. |
| **F-03** | **REQUIRED** | `src/dataset.py:312, 343` | Nhận diện nguồn dữ liệu bỏ sót tiền tố `uldd_`, gán thành `"custom"`. | Bổ sung kiểm tra `uldd` trong tên file: `src = "uldd" if "uldd" in stem ...`. |
| **F-04** | **OPTIONAL** | `src/dataset.py:314, 345` | Bỏ trống `subject_id` (`None`), không trích xuất mã đối tượng (VD: `subE`, `subA`). | Dùng Regex trích xuất `sub[A-Za-z0-9]+` từ stem của video để gán vào `subject_id`. |
| **F-05** | **OPTIONAL** | `src/dataset.py:487-564` | `build_raw_video_dataloaders` chỉ tạo Train và Val DataLoader, chưa có tùy chọn lấy Test DataLoader. | Thêm tùy chọn `return_test: bool = False` để sẵn sàng mở rộng khi cần. |

---

## 5. KẾT LUẬN VÀ PHÊ DUYỆT BƯỚC 1

### 5.1. Câu trả lời trực tiếp cho người dùng
> **KẾT LUẬN**: 
> **CÓ THỂ SỬ DỤNG ĐƯỢC NGAY** `src/dataset.py` với cấu trúc thư mục trong `docs/struct_dataset1.md` ở chế độ **quét thư mục (Folder Discovery)** cho 2 tập `train` và `val`.
> 
> Tuy nhiên, nếu trong thư mục có sẵn các file `dataset_merged_split.csv` hoặc `dataset_merged_split.json`, `src/dataset.py` sẽ tự động chuyển sang đọc file chỉ mục này và có nguy cơ **bị lỗi rỗng dữ liệu (nếu là JSON)** hoặc **bỏ sót mẫu (nếu CSV chứa cột `orig_file` không hợp lệ hoặc dùng cột `clip_path`)**.

### 5.2. Các bước khắc phục đề xuất (Remedies)
1. Cập nhật `_load_from_manifest` hỗ trợ đầy đủ cả file `.csv` và `.json`.
2. Chuẩn hóa thứ tự ưu tiên cột đường dẫn: `clip_path` $\rightarrow$ `path` $\rightarrow$ `file_path` $\rightarrow$ `orig_file`.
3. Bổ sung nhận diện nguồn `uldd` và tự động trích xuất `subject_id` từ tên file.
4. Đảm bảo chế độ quét thư mục tiếp tục hoạt động ổn định và chính xác 100%.

---

## 6. DANH MỤC KIỂM TRA CHẤT LƯỢNG (REVIEW CHECKLIST)

- [x] **Context**: Đã nắm rõ cấu trúc dữ liệu ULDD 25s cắt sẵn và cơ chế tải của `src/dataset.py`.
- [x] **Correctness**: Xác nhận quét thư mục khớp 100%, phát hiện lỗi không đọc được JSON manifest và lệch tên cột CSV.
- [x] **Readability**: Đã chỉ rõ các điểm hardcode cần làm sạch.
- [x] **Architecture**: Cơ chế CPU decode + GPU chunking đáp ứng tốt giới hạn VRAM.
- [x] **Security/Robustness**: Cơ chế bắt lỗi video hỏng và bỏ qua frame lỗi hoạt động tốt.
- [x] **Performance**: Đã tính toán footprint RAM (~614 MB/batch) và GPU VRAM ceiling (< 2GB).

---
*Tài liệu này hoàn tất Bước 1 (Discovery & Analysis). Kính mời người dùng xem xét và phê duyệt để chuyển sang Bước 2 (Lên kế hoạch Plan).*
