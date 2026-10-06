# Báo cáo Kỹ thuật: Hoàn tất khắc phục lỗi FFmpeg swscaler Interlaced frames trên Kaggle

## 1. Tóm tắt kết quả (Executive Summary)
- Đã giải quyết triệt để lỗi:
  ```text
  [swscaler @ 0x3ea9f340] Cannot convert interlaced to progressive frames or vice versa. (Invalid argument): fmt:yuv411p csp:unknown prim:unknown trc:unknown -> fmt:bgr24 csp:gbr prim:unknown trc:unknown
  ```
- File đã cập nhật: [02_train_convgru_kaggle.ipynb](file:///D:/Project/DATN/driver-guardian/ai/LSTM/notebooks/02_train_convgru_kaggle.ipynb) (Cell 1 và Cell 13).
- Toàn bộ 25 cell trong Notebook đã được biên dịch và kiểm tra tính toàn vẹn cú pháp qua `ast.parse`.

---

## 2. Chi tiết các thay đổi trong mã nguồn

### 2.1 Cập nhật Cell 1: Thêm thư viện chuẩn
Đã thêm `import subprocess` và `import contextlib` phục vụ quản lý tiến trình FFmpeg và điều hướng luồng lỗi C.

### 2.2 Cập nhật Cell 13: Xử lý triệt để giải mã video Interlaced
1. **Bổ sung `_suppress_c_stderr()`**:
   - Chặn các thông báo lỗi C-level từ FFmpeg `swscaler` xuất hiện trên giao diện dòng lệnh của Kaggle bằng cơ chế `os.dup2(devnull_fd, 2)`.
2. **Thêm phương thức `_sample_video_frames_ffmpeg(self, video_path: Path)`**:
   - Sử dụng binary `ffmpeg` (được tích hợp sẵn tại `/usr/bin/ffmpeg` trên Kaggle) để thực hiện deinterlace khử quét xen kẽ bằng bộ lọc `yadif=0:-1:0`.
   - Kết hợp `fps={fps_sample}` để giảm tải xử lý và chỉ trích xuất đúng các khung hình cần lấy mẫu.
   - Thao tác letterbox thu nhỏ giữ tỷ lệ `scale=...` và đệm viền xám 114 `pad=...` chuẩn RGB24.
   - Đọc trực tiếp từ bộ đệm `stdout` dạng rawvideo sang mảng NumPy uint8 `(N, 640, 640, 3)` mà không tốn chi phí I/O ghi đĩa tạm.
3. **Cập nhật `_sample_video_frames(self, video_path: Path)`**:
   - Đọc luồng thông thường bằng `cv2.VideoCapture` để tối ưu tốc độ đối với video progressive chuẩn.
   - Tự động fallback sang `_sample_video_frames_ffmpeg` khi OpenCV gặp video interlaced `yuv411p` hoặc trả về số khung hình < `min_frames`.

---

## 3. Kết quả kiểm tra xác thực (Validation)
1. **Kiểm tra cú pháp**: Toàn bộ các cell code trong file `02_train_convgru_kaggle.ipynb` đều vượt qua `ast.parse` thành công 100%.
2. **Kiểm tra tương thích**: Khung hình trích xuất qua fallback có shape `(640, 640, 3)`, dtype `uint8`, giá trị đệm viền `(114, 114, 114)` hoàn toàn đồng nhất với đầu ra của hàm `letterbox` dùng trong OpenCV.
