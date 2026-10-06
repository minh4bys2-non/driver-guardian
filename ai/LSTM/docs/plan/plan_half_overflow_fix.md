# Kế hoạch Kỹ thuật: Khắc phục lỗi Overflow c10::Half trong TemporalAttentionPooling

## 1. Mục tiêu
- Sửa triệt để ngoại lệ `RuntimeError: value cannot be converted to type c10::Half without overflow` khi chạy `trainer.train()` trên GPU với AMP `float16`.
- Đảm bảo tính toán Softmax của cơ chế chú ý theo thời gian (Temporal Attention) hoàn toàn chính xác đối với các chuỗi có độ dài động.

---

## 2. Các bước triển khai

### Bước 1: Khảo sát mã nguồn Notebook
- Định vị chính xác vị trí phát sinh lỗi trong `notebooks/02_train_convgru_kaggle.ipynb` (Cell 10: `TemporalAttentionPooling`).
- Kiểm tra toàn bộ mã nguồn của Notebook và thư mục `src/` để đảm bảo không còn hằng số mask nào vượt quá giới hạn biểu diễn của `float16`.

### Bước 2: Chỉnh sửa mã nguồn Cell 10
- Thay thế việc gán cứng `-1e9` bằng giá trị `fill_value` phụ thuộc vào `scores.dtype`:
  ```python
  fill_value = -1e4 if scores.dtype == torch.float16 else -1e9
  scores = scores.masked_fill(mask, fill_value)
  ```

### Bước 3: Kiểm thử và xác thực
- Biên dịch cú pháp toàn bộ 25 cell trong Notebook bằng `ast.parse`.
- Thực thi unit test trên GPU với `torch.amp.autocast("cuda", dtype=torch.float16)` và `GradScaler`, kiểm tra cả chiều xuôi (forward) và chiều ngược (backward).
