# Phân tích Kỹ thuật: Khắc phục lỗi Overflow c10::Half trong TemporalAttentionPooling khi dùng PyTorch AMP Float16

## 1. Khảo sát đầu ra từ Notebook thực thi (`temp/train-datn1.ipynb`)

Qua phân tích log thực thi thực tế trên GPU Tesla T4 (Kaggle Notebook) tại `temp/train-datn1.ipynb`:
- **Cell 1**: Khởi tạo môi trường GPU Tesla T4 (2 GPUs, 14.56 GB VRAM, PyTorch 2.11.0+cu128, CUDA sẵn sàng).
- **Cell 4**: Tự động nhận diện đường dẫn Dataset và Backbone checkpoint trong `/kaggle/input` thành công.
- **Cell 17**: Khởi tạo Trainer thành công:
  - Tập Train: 1956 videos, Tập Val: 484 videos.
  - PAFPN BackboneNeck: nạp 444 matched keys, kiến trúc `w=(16, 32, 64, 128, 256), neck_n=1`.
  - ConvGRUClassifier: 1,602,884 tham số huấn luyện.
- **Cell 19 (`trainer.train()`)**:
  Ngay tại batch đầu tiên của Epoch 01 (`Epoch 01/30 [Train]: 0%| | 0/123 [00:00<?, ?it/s]`), quá trình huấn luyện bị ngắt đột ngột với ngoại lệ:

```text
RuntimeError: value cannot be converted to type c10::Half without overflow
```

Traceback chi tiết:
```text
/tmp/ipykernel_50/118151701.py in train_epoch(self, epoch)
    317     with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
--> 318         logits = self.model((p3, p4, p5), seq_lens=seq_lens)
...
/tmp/ipykernel_50/567314362.py in forward(self, features, seq_lens)
     50     spatial_pooled = self.spatial_attn(gru_out)
---> 51     temporal_pooled = self.temporal_attn(spatial_pooled, seq_lens=seq_lens)
...
/tmp/ipykernel_50/1611635205.py in forward(self, x, seq_lens)
     66     mask = t_idx >= seq_lens.unsqueeze(1)
---> 67     scores = scores.masked_fill(mask, -1e9)
     68 
     69 weights = F.softmax(scores, dim=-1).unsqueeze(1)

RuntimeError: value cannot be converted to type c10::Half without overflow
```

---

## 2. Nguyên nhân gốc rễ (Root Cause Analysis)

1. **Giới hạn số học của chuẩn IEEE 754 Half-Precision (`torch.float16` / `c10::Half`)**:
   - Khi kích hoạt `torch.amp.autocast(device_type="cuda", dtype=torch.float16)`, các toán tử tuyến tính và mạng chú ý trong `ConvGRUClassifier` được tính toán với kiểu dữ liệu 16-bit float.
   - Giá trị hữu hạn nhỏ nhất (và lớn nhất) có thể biểu diễn được trong `torch.float16` là:
     $$\text{finfo}(\text{float16}).\text{min} = -65,504.0, \quad \text{finfo}(\text{float16}).\text{max} = 65,504.0$$
2. **Xung đột kiểu dữ liệu khi gọi `masked_fill`**:
   - Hằng số `-1e9` ($-1,000,000,000.0$) trong dòng lệnh:
     ```python
     scores = scores.masked_fill(mask, -1e9)
     ```
     vượt xa giới hạn dưới $-65,504.0$.
   - Trong PyTorch 2.x, trình biên dịch C++ kiểm tra tính tương thích và phát hiện tràn số kiểu Half (`c10::Half overflow`), lập tức ném ra ngoại lệ `RuntimeError`.
3. **Mục đích của việc mask điểm chú ý trước Softmax**:
   - Ta chỉ cần một giá trị âm đủ lớn để sau khi đi qua hàm hàm $e^x$ trong Softmax, trọng số chú ý tại các vị trí padding sẽ triệt tiêu về $0.0$.
   - Với kiểu `float16`, giá trị âm $x \le -100.0$ đã có $e^x = 0.0$. Do đó giá trị `-1e4` ($-10,000.0$) hoàn toàn triệt tiêu trọng số mà vẫn nằm an toàn trong dải đại diện của `torch.float16`.

---

## 3. Giải pháp kỹ thuật

Cập nhật hàm `forward()` của lớp `TemporalAttentionPooling` trong Cell 10 của `notebooks/02_train_convgru_kaggle.ipynb`:

```python
if seq_lens is not None:
    B, T = scores.shape
    t_idx = torch.arange(T, device=scores.device).unsqueeze(0)
    mask = t_idx >= seq_lens.unsqueeze(1)
    fill_value = -1e4 if scores.dtype == torch.float16 else -1e9
    scores = scores.masked_fill(mask, fill_value)
```

Giải pháp này hoàn toàn tương thích và đồng bộ với cách triển khai trong [`src/models.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py#L301), đảm bảo:
- Khi chạy Full Precision (`float32`): dùng `-1e9`.
- Khi chạy Mixed Precision (`float16`): dùng `-1e4`, không phát sinh lỗi tràn số và triệt tiêu hoàn toàn padding trong Softmax.
