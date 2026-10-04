# KẾ HOẠCH TRIỂN KHAI HOÀN THIỆN VÀ KHẮC PHỤC LỖI HAI KHỐI DEEPGRUCLASSIFIER & TEMPORALATTENTIONPOOLING

**Mã tài liệu:** `plan_deepgru_temporal_pooling.md`  
**Dự án:** Hệ Thống Nhận Diện Tài Xế Buồn Ngủ (Driver Guardian)  
**Tệp thực thi mục tiêu:** [`src/models.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py) & [`src/__init__.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/__init__.py)  
**Căn cứ phân tích:** [`docs/analsys_deepgru_temporal_pooling.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/analsys_deepgru_temporal_pooling.md)  
**Quy chuẩn áp dụng:** [`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md) — Bước 2: Lên kế hoạch thực hiện  
**Quyết định thiết kế người dùng:** **LOẠI BỎ HOÀN TOÀN `DeepLSTMClassifier` VÀ `build_model`**, tập trung tối đa vào pipeline chuẩn hóa duy nhất: `DeepGRUClassifier` kết hợp `TemporalAttentionPooling`.  
**Ngày cập nhật:** 02/10/2026  

---

## 1. MỤC TIÊU & NGUYÊN TẮC KỸ THUẬT

### 1.1. Mục tiêu cốt lõi
1. **Loại bỏ toàn bộ dấu tích của LSTM và `build_model`:** Tối giản kiến trúc dự án, loại bỏ hoàn toàn các lớp thừa không còn sử dụng (`DeepLSTMClassifier`, `build_model`, alias `@property lstm`), chuyển dịch trọn vẹn $100\%$ sang kiến trúc hiện đại **Deep GRU + Temporal Attention Pooling**.
2. **Khắc phục triệt để các lỗi P0 & P1:** Xử lý lỗi gãy gói `src/__init__.py`, lỗi tràn số FP16 trong `TemporalAttentionPooling`, lỗi nuốt tham số vị trí `*args` trong `forward`, và lỗi nạp checkpoint.
3. **Đảm bảo tính sẵn sàng triển khai (Production-ready):** Mô hình chạy ổn định trên CPU, GPU (FP32/FP16 AMP) và xuất khẩu thành công sang ONNX Opset 14 phục vụ runtime Jetson / Android.

### 1.2. Các yêu cầu & Ràng buộc kỹ thuật
1. **Tinh giản gói `src/__init__.py`:** Chỉ export các thành phần thực sự phục vụ pipeline GRU: `CNNAdapter`, `TemporalAttentionPooling`, `DeepGRUClassifier`, `DrowsinessLoss`, `DrowsinessBCELoss`, `build_loss`.
2. **Triệt tiêu lỗi tràn số FP16 trong Attention Masking:** Sử dụng `fill_value = -1e4 if scores.dtype == torch.float16 else -1e9` trong `TemporalAttentionPooling`.
3. **Chuẩn hóa chữ ký hàm `forward`:** Đặt `seq_lens` và `h_0` ở vị trí tường minh; hỗ trợ thông minh cả truyền 3 tensor không gian rời `(p3, p4, p5)` lẫn gọi vị trí `model(vector, seq_lens)`.
4. **Đồng bộ hóa tham số cấu hình:** Kết nối tham số `supervision_mode` từ `TrainConfig` vào luồng suy luận của `forward()`.

---

## 2. CHI TIẾT THIẾT KẾ KỸ THUẬT

### 2.1. Cải tiến khối `TemporalAttentionPooling`

```python
class TemporalAttentionPooling(nn.Module):
    """
    Gom tụ chuỗi thời gian có trọng số chú ý động (Temporal Attention MLP),
    triệt tiêu 100% gradient rác từ các khung hình zero-padding qua Attention Masking.
    Tương thích hoàn toàn với FP32, FP16 (AMP) và ONNX Runtime.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, gru_out: torch.Tensor, seq_lens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, h = gru_out.shape
        scores = self.attn(gru_out).squeeze(-1)  # [B, T]

        if seq_lens is not None:
            # Tạo boolean mask an toàn
            mask = torch.arange(t, device=gru_out.device).unsqueeze(0) < seq_lens.to(gru_out.device).unsqueeze(1)
            # Khắc phục lỗi tràn số FP16: -1e4 an toàn tuyệt đối với float16 (dải max 65504)
            fill_value = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(~mask, fill_value)

        weights = F.softmax(scores, dim=-1)  # [B, T]
        # Gom tụ có trọng số: [B, 1, T] x [B, T, H] -> [B, 1, H] -> [B, H]
        pooled = torch.bmm(weights.unsqueeze(1), gru_out).squeeze(1)
        return pooled, weights
```

### 2.2. Chuẩn hóa chữ ký hàm `forward` trong `DeepGRUClassifier`

```python
    def forward(
            self,
            features: Union[Tuple[torch.Tensor, ...], List[torch.Tensor], torch.Tensor],
            seq_lens: Optional[torch.Tensor] = None,
            h_0: Optional[torch.Tensor] = None,
            return_sequence: bool = False,
            return_weights: bool = False,
            return_state: bool = False,
            *args: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[Any, ...]]:
        # Hỗ trợ truyền rời 3 tensor không gian (ví dụ model(p3, p4, p5))
        if isinstance(seq_lens, torch.Tensor) and seq_lens.dim() >= 4:
            extra = [seq_lens]
            if isinstance(h_0, torch.Tensor) and h_0.dim() >= 4:
                extra.append(h_0)
                h_0 = None
            if len(args) > 0:
                extra.extend(args)
            features = (features, *extra)
            seq_lens = None
        elif len(args) > 0 and isinstance(features, torch.Tensor):
            features = (features, *args)
```

- **Tác động:**
  - Lời gọi theo vị trí `model(vector, seq_lens)` nhận diện đúng `seq_lens`, không còn bị ném lỗi `ValueError`.
  - Hỗ trợ xuất ONNX `torch.onnx.export(model, (dummy_vec, dummy_seq_lens), ...)` thành công $100\%$.
  - Tự động kích hoạt chế độ chuỗi khi `self.supervision_mode in ("sequence", "frame")`.

### 2.3. Khắc phục logic `from_checkpoint()` & Xóa tàn tích LSTM

- Loại bỏ hoàn toàn `@property def lstm`.
- Loại bỏ tham số legacy `hc` khỏi `forward()`.
- Xóa bỏ việc kiểm tra `lstm.weight_ih_l0` và logic `remapped_sd` trong `from_checkpoint()`.

---

## 3. LỘ TRÌNH THỰC HIỆN CHI TIẾT (WORKFLOW PHASES)

### Giai đoạn 1: Tinh chỉnh mã nguồn [`src/models.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py)
1. Cập nhật `TemporalAttentionPooling` với cơ chế tính `fill_value` an toàn theo `scores.dtype`.
2. Chuẩn hóa chữ ký hàm `forward()` của `DeepGRUClassifier`.
3. Xóa bỏ hoàn toàn tàn tích liên quan đến LSTM trong `DeepGRUClassifier`.
4. Không bổ sung `DeepLSTMClassifier` hay `build_model`.

### Giai đoạn 2: Cập nhật [`src/__init__.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/__init__.py)
1. Xuất bản danh sách module sạch:
   ```python
   __all__ = [
       "CNNAdapter",
       "TemporalAttentionPooling",
       "DeepGRUClassifier",
       "DrowsinessLoss",
       "DrowsinessBCELoss",
       "build_loss",
   ]
   ```
2. Đảm bảo lệnh `python -c "import src"` chạy thành công với exit code 0.

### Giai đoạn 3: Kiểm thử tự động đa chiều (Verification Test Suite)
Chạy script kiểm thử toàn diện:
1. **Test Case 1:** `import src` và kiểm tra export các symbol sạch sẽ.
2. **Test Case 2 (FP16 Half Precision):** Chạy `TemporalAttentionPooling` với `torch.float16`, xác nhận không bị overflow và gradient đệm bằng 0.0.
3. **Test Case 3 (Positional & 5D Spatial):** Gọi `model(vec, seq_lens)`, `model(p3, p4, p5)` và `model((p3_5d, p4_5d, p5_5d))`.
4. **Test Case 4 (ONNX Export):** Xuất mô hình `DeepGRUClassifier` kèm `seq_lens` ra ONNX buffer Opset 14 thành công.
5. **Test Case 5 (Backward Pass):** Kiểm tra tính liên tục của gradient qua chuỗi thời gian.

### Giai đoạn 4: Nghiệm thu và Báo cáo tổng kết
1. Tạo tệp [`docs/report_deepgru_temporal_pooling.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/report_deepgru_temporal_pooling.md) theo quy chuẩn Bước 3 của [`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md).
2. Trình bày kết quả hoàn thành với người dùng.

---

## 4. TIÊU CHÍ NGHIỆM THU (ACCEPTANCE CRITERIA)

- [x] Lệnh `python -c "import src"` thực thi thành công với exit code 0, không có `DeepLSTMClassifier` hay `build_model`.
- [x] Khối `TemporalAttentionPooling` chạy trơn tru trên tensor `torch.float16` mà không gặp lỗi overflow.
- [x] Lời gọi `DeepGRUClassifier(vector, seq_lens)` không bị crash `ValueError`.
- [x] Mô hình xuất khẩu ONNX Opset 14 thành công với `seq_lens`.
- [x] Không có hiện tượng rò rỉ gradient qua các khung hình padding.
- [ ] File báo cáo nghiệm thu [`docs/report_deepgru_temporal_pooling.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/report_deepgru_temporal_pooling.md) được tạo đầy đủ.
