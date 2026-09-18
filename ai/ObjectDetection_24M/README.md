# ObjectDetection_24M

Mô hình phát hiện đối tượng NMS-Free bằng PyTorch, gồm backbone, neck PAFPN và detection head. Cấu hình mặc định nhận diện 80 lớp với ảnh đầu vào 480 × 480.

## Cấu trúc

- `src/`: kiến trúc mô hình, cấu hình và bộ đọc dữ liệu Object365.
- `src/runtime/`: nhận diện trên ảnh và webcam.
- `delete/`: mã huấn luyện cũ, hiện chưa đầy đủ module phụ thuộc.
- `checkpoints_ftCOCO/`: trọng số và metadata mô hình, không đưa lên Git.

## Chạy nhận diện

Cần Python cùng các thư viện PyTorch, torchvision, NumPy, OpenCV, Matplotlib và Albumentations. Chuẩn bị checkpoint và metadata tương ứng; sửa đường dẫn trong `src/runtime/infer.py` hoặc `runcamera.py` theo máy đang dùng. Chọn `device="cpu"` nếu không dùng CUDA.

Chạy từ thư mục `ai/ObjectDetection_24M`:

```bash
python -m src.runtime.infer       # Nhận diện trên ảnh
python -m src.runtime.runcamera   # Webcam, nhấn q để thoát
```

Dữ liệu huấn luyện, checkpoint và ảnh kết quả được bỏ qua bởi `.gitignore` của dự án.
