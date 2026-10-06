# Phân tích Kỹ thuật: Khắc phục lỗi FFmpeg swscaler Interlaced frames trên Kaggle

## 1. Hiện tượng & Lỗi phát sinh (Problem Statement)

Khi chạy huấn luyện mô hình ConvGRU trên Kaggle Notebook (`notebooks/02_train_convgru_kaggle.ipynb`), trong quá trình DataLoader nạp các mẫu video từ dataset, thư viện OpenCV gặp lỗi từ thư viện FFmpeg C-level:

```text
[swscaler @ 0x3ea9f340] Cannot convert interlaced to progressive frames or vice versa. (Invalid argument): fmt:yuv411p csp:unknown prim:unknown trc:unknown -> fmt:bgr24 csp:gbr prim:unknown trc:unknown
```

Hệ quả:
1. `cap.read()` của OpenCV trả về `ret = False, frame = None` ngay từ khung hình đầu tiên.
2. `_sample_video_frames()` không thể trích xuất khung hình (`len(frames_rgb) == 0 < min_frames`), khiến mẫu video bị đánh dấu là hỏng (`status: "corrupted"`).
3. Hàng loạt mẫu video từ các camera/định dạng truyền hình NTSC/PAL cũ (như UTA-RLDD hoặc SUST sử dụng DV `yuv411p` quét xen kẽ interlaced) bị loại bỏ khỏi quá trình huấn luyện.
4. Lỗi swscaler được in trực tiếp từ C-level ra `stderr` (file descriptor 2), gây tràn log đầu ra (log flooding).

---

## 2. Nguyên nhân gốc rễ (Root Cause Analysis)

1. **Cơ chế chuyển đổi không gian màu của FFmpeg `swscale`**:
   - `yuv411p` là định dạng nén video 4:1:1 planar thường gặp trong chuẩn DV (Digital Video).
   - Video trong tập dữ liệu chứa cờ quét xen kẽ (`interlaced_frame = 1` hoặc field order top/bottom).
   - Trong các phiên bản FFmpeg hiện đại (được tích hợp sẵn trong OpenCV trên Kaggle Linux), hàm `sws_scale` thực hiện kiểm tra nghiêm ngặt: nếu nguồn là `interlaced` và đích là `progressive` (`bgr24`), `swscale` từ chối thực hiện phép scale/convert trực tiếp vì sẽ gây hiện tượng răng cưa và xé hình (comb artifacts / field distortion).
2. **Hạn chế cố hữu của `cv2.VideoCapture`**:
   - OpenCV chỉ gọi trực tiếp hàm `sws_scale()` mà **không** dựng một Filter Graph (bộ lọc khử xen kẽ như `yadif`, `bwdif`).
   - Cấu hình qua `OPENCV_FFMPEG_CAPTURE_OPTIONS` chỉ ảnh hưởng tới demuxer/decoder, không thể kích hoạt filter deinterlace trong OpenCV backend.
3. **Môi trường Kaggle**:
   - Trên Kaggle Linux, công cụ dòng lệnh `/usr/bin/ffmpeg` luôn được cài đặt sẵn đầy đủ các bộ lọc (filtergraph) mạnh mẽ.

---

## 3. Giải pháp kỹ thuật (Technical Solution)

1. **Cơ chế Fallback thông minh qua FFmpeg CLI (`_sample_video_frames_ffmpeg`)**:
   - Khi OpenCV gặp video interlaced và trả về ít hơn `min_frames`, pipeline tự động chuyển sang gọi subprocess `ffmpeg`.
   - Chuỗi filtergraph FFmpeg tối ưu:
     - `yadif=0:-1:0`: Khử quét xen kẽ (deinterlace) bảo toàn độ mượt thời gian.
     - `fps={fps_sample}`: Lấy mẫu tốc độ khung hình mục tiêu trước khi scale (giảm 80% tải tính toán).
     - `scale={img_size}:{img_size}:force_original_aspect_ratio=decrease`: Thu nhỏ giữ nguyên tỷ lệ khung hình.
     - `format=rgb24`: Chuẩn hóa sang RGB24 không subsampling.
     - `pad={img_size}:{img_size}:(ow-iw)/2:(oh-ih)/2:color=0x727272`: Đệm viền xám 114 chuẩn letterbox YOLO/ConvGRU.
   - Đầu ra được truyền trực tiếp qua pipe `-f rawvideo -pix_fmt rgb24 pipe:1` và ánh xạ thành NumPy mảng uint8 mà không ghi file tạm ra đĩa.
2. **Chặn rò rỉ log C-level (`_suppress_c_stderr`)**:
   - Dùng context manager điều hướng tạm thời file descriptor `2` về `/dev/null` khi gọi `cap.read()`.
   - Ngăn chặn hoàn toàn việc log swscaler gây rối giao diện notebook khi quét qua các video interlaced.
