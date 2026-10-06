# Kế hoạch Kỹ thuật: Khắc phục lỗi FFmpeg swscaler Interlaced frames trên Kaggle

## 1. Mục tiêu
- Loại bỏ hoàn toàn lỗi crash và mất mẫu khi giải mã video định dạng `yuv411p` / `interlaced` trong notebook `notebooks/02_train_convgru_kaggle.ipynb`.
- Đảm bảo tính tương thích 100% về kích thước khung hình, thứ tự kênh RGB, padding letterbox giữa OpenCV và FFmpeg fallback.
- Ngăn chặn triệt để hiện tượng log spam từ thư viện C `swscaler` trên Kaggle.

---

## 2. Các bước triển khai chi tiết

### Bước 1: Bổ sung thư viện hỗ trợ
- Cập nhật Cell 1 và Cell 13 để nạp `subprocess`, `shutil`, `contextlib`.

### Bước 2: Thiết kế bộ chặn Log C-level (`_suppress_c_stderr`)
- Sử dụng cơ chế `os.dup2(devnull_fd, 2)` trong context manager để bọc vòng lặp `cap.read()`.
- Đảm bảo khôi phục nguyên trạng file descriptor `2` trong khối `finally` nhằm tránh ảnh hưởng tới các thông báo ngoại lệ Python khác.

### Bước 3: Xây dựng hàm trích xuất qua FFmpeg CLI (`_sample_video_frames_ffmpeg`)
- Kiểm tra sự tồn tại của `ffmpeg` thông qua `shutil.which("ffmpeg")` hoặc fallback `/usr/bin/ffmpeg`.
- Thiết lập bộ lọc (filtergraph) chuẩn xác:
  ```text
  yadif=0:-1:0,fps={fps_sample},scale={img_size}:{img_size}:force_original_aspect_ratio=decrease,format=rgb24,pad={img_size}:{img_size}:(ow-iw)/2:(oh-ih)/2:color=0x727272
  ```
- Nhận luồng raw RGB24 trực tiếp từ `stdout` subprocess và reshape thành `(N, H, W, 3)`.

### Bước 4: Tích hợp vào `_sample_video_frames` của `RawVideoFramesDataset`
- Ưu tiên đọc nhanh qua `cv2.VideoCapture`.
- Nếu số lượng khung hình trích xuất < `self.min_frames` (do lỗi giải mã swscaler hoặc tập tin có cờ interlaced), tự động kích hoạt fallback sang `_sample_video_frames_ffmpeg`.

### Bước 5: Kiểm thử và xác thực
- Kiểm thử AST cú pháp toàn bộ Notebook để đảm bảo không có lỗi biên dịch.
- Kiểm thử tính đúng đắn của logic trích xuất với dữ liệu video mẫu.
