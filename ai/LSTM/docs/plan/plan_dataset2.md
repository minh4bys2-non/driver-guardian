# Kế Hoạch Khắc Phục Lỗ Hổng Dữ Liệu (`src/dataset2.py`) - Tối ưu 4GB VRAM

Dựa trên cấu hình phần cứng giới hạn (RTX 3050 4GB VRAM, 16GB RAM), thiết kế pipeline cần phải cực kỳ khắt khe về mặt cấp phát bộ nhớ. Dưới đây là phương án tối ưu nhất.

---

## 1. Mục tiêu tái cấu trúc
- **Giữ RAM hệ thống an toàn (< 16GB):** Dataset chỉ giữ ảnh raw dưới dạng uint8 hoặc chuyển đổi nhanh lên Tensor, số worker cấu hình vừa phải.
- **VRAM chống tràn tuyệt đối (Cho 4GB GPU):** Trích xuất đặc trưng on-the-fly trong Main Loop, nhưng **áp dụng cơ chế chia nhỏ Mini-Chunk** khi đưa qua BackboneNeck. Không bao giờ đẩy nguyên một batch lớn (vd 32 khung hình) qua Backbone cùng lúc.
- **Tối ưu Băng thông PCIe:** Dữ liệu đẩy lên GPU 1 lần và nằm tại đó.

---

## 2. Kế hoạch triển khai chi tiết

### Giai đoạn 1: Sửa đổi Dataset thành `RawVideoFramesDataset` (`src/dataset2.py`)
- Dataset sử dụng OpenCV đọc video, lấy mẫu thời gian, chạy `letterbox` và chuẩn hóa thành Tensor.
- Trả về Tensor ảnh dạng `[T, 3, 640, 640]` và **không chứa PyTorch Model**. Các DataLoader workers (chạy trên CPU) tốn bộ nhớ rất ít.
- Xử lý video lỗi: Bỏ qua video hỏng ngay trong Dataset.

### Giai đoạn 2: Cập nhật hàm `collate_fn`
- Gom batch các khung hình thành tensor `[B, T_max, 3, 640, 640]` trên CPU.

### Giai đoạn 3: Trích xuất Đặc trưng theo Mini-Chunk trong Vòng lặp Huấn luyện (Main Loop)
Với GPU 4GB VRAM, ta không thể đẩy Tensor `[B, T_max, 3, 640, 640]` (VD: B=4, T=30 -> 120 ảnh) trực tiếp vào BackboneNeck, vì intermediate activations sẽ làm tràn 4GB VRAM ngay lập tức.
Giải pháp:
1. **Thiết kế wrapper `MiniChunkFeatureExtractor`** chạy trên GPU trong lúc train.
2. Đưa batch ảnh `[B, T, 3, 640, 640]` lên GPU. Flatten thành `[B*T, 3, 640, 640]`.
3. Cắt `B*T` thành các mini-chunk nhỏ (VD: `chunk_size = 4` hoặc `8` khung hình/lần).
4. Vòng lặp tính toán trên GPU với `torch.inference_mode()` và `torch.amp.autocast(dtype=torch.float16)`:
   - Qua mỗi mini-chunk, lấy được các tensor `p3_chunk, p4_chunk, p5_chunk`.
   - Nối (concat) kết quả lại ngay trên GPU (đặc trưng p3, p4, p5 rất nhẹ so với activation lúc qua Backbone).
5. Sau khi tính xong toàn bộ `B*T`, reshape trả về dạng `[B, T, C, H, W]` và đẩy vào mô hình LSTM/GRU để tính loss và Backward.

Cách này giữ nguyên tính on-the-fly, băng thông PCIe di chuyển 1 lần, nhưng **giữ trần (ceiling) bộ nhớ VRAM tĩnh**, an toàn 100% cho RTX 3050 4GB.

---

## 3. Tiêu chí nghiệm thu (Checklist)
- [ ] Xóa logic Model Extractor khỏi Dataset.
- [ ] Xây dựng thành phần `ChunkedBackboneNeck` tích hợp vào luồng train.
- [ ] Chạy thực nghiệm với `batch_size=4` và `seq_len=15` không vượt quá 3.5 GB VRAM trên GPU.

---
**Chờ phê duyệt:** 
Chiến lược chia nhỏ Mini-Chunk trên GPU này được thiết kế "đo ni đóng giày" cho phần cứng 4GB VRAM của bạn. Nếu ổn thỏa, tôi sẽ tiến hành **Bước 3: Bắt tay vào sửa code `src/dataset2.py`** và cập nhật mã liên quan.
