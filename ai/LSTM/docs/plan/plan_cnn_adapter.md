# KẾ HOẠCH TRIỂN KHAI TÁI CẤU TRÚC KHỐI CNNADAPTER (SRC/MODELS.PY)
**Mã tài liệu:** `plan_cnn_adapter.md`  
**Dự án:** Hệ Thống Nhận Diện Tài Xế Buồn Ngủ (Driver Guardian - Deep LSTM / GRU Pipeline)  
**Tệp thực thi mục tiêu:** [`src/models.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py) & [`src/__init__.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/__init__.py)  
**Căn cứ phân tích:** [`docs/analsys_cnn_adapter.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/analsys_cnn_adapter.md)  
**Quy chuẩn áp dụng:** [`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md) — Bước 2: Lên kế hoạch thực hiện  
**Ngày cập nhật:** 01/10/2026  
**Phương án đã chọn:** **Phương án 1 (Phễu tích chập phân tầng - KHÔNG DÙNG GAP - Cài đặt trực tiếp dạng phương thức trong lớp `CNNAdapter`, KHÔNG TẠO LỚP MỚI)**  

---

## 1. MỤC TIÊU & NGUYÊN TẮC THIẾT KẾ

### 1.1. Mục tiêu cốt lõi
Tái cấu trúc khối [`CNNAdapter`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py#L9) theo **Phương án 1: Phễu tích chập phân tầng** nhằm nén đặc trưng không gian từ lưới $40 \times 40$ về vector biểu diễn $\text{out\_dim}$ ($256$ hoặc $512$) cho từng khung hình mà **tuyệt đối KHÔNG sử dụng Global Average Pooling (GAP)**.

### 1.2. Yêu cầu kỹ thuật & Ràng buộc kiến trúc
1. **Không tạo lớp `nn.Module` mới:** Toàn bộ các tầng tích chập của phễu phân tầng và logic xử lý được tích hợp trực tiếp bên trong lớp [`CNNAdapter`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py#L9) thông qua phương thức `hierarchical_conv_pyramid()`.
2. **Loại bỏ hoàn toàn Global Average Pooling (GAP):** Không dùng `AdaptiveAvgPool2d((1, 1))` hay bất kỳ phép trung bình số học nào làm triệt tiêu tọa độ và mối quan hệ tương hỗ giữa các cơ quan khuôn mặt.
3. **Bảo toàn tương quan không gian qua Receptive Field tăng dần:** Sử dụng chuỗi các lớp tích chập với bước trượt (Strided Convolutions $3\times 3, s=2$) để học tương quan cục bộ $\to$ tương quan vùng $\to$ tương quan toàn khuôn mặt, và tổng hợp về $1\times 1$ bằng kernel tích chập học được ($5 \times 5, s=1$).
4. **Khắc phục 100% các lỗi nghiêm trọng đã phát hiện:**
   - Triệt tiêu lỗi sai lệch kích thước ma trận `RuntimeError: mat1 and mat2 shapes cannot be multiplied`.
   - Triệt tiêu nguy cơ bùng nổ 419M tham số của lớp `Linear`.
   - Hỗ trợ đầy đủ tensor chuỗi video 5D $[B, T, C, H, W]$ qua cơ chế gập/trải chiều $(B, T) \leftrightarrow (B \times T)$.
   - Hỗ trợ linh hoạt kiểu tham số `in_channels`: cả `tuple(int)` như `(64, 128, 256)` lẫn `tuple(tuple(int))` như `((64, 80, 80), ...)`.
   - Hỗ trợ trọn vẹn 4 chế độ fusion: `'concat'`, `'attention'`, `'sum'`, `'mean'`.
   - Đồng bộ hóa hợp đồng giao diện `return_weights`: trả về `(output, weights)` khi `return_weights=True` và `output` khi `return_weights=False`.
5. **Khôi phục tính toàn vẹn của Source Package:** Bổ sung [`DeepLSTMClassifier`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py) và factory function `build_model` vào [`src/models.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py) để [`src/__init__.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/__init__.py) và [`model.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/model.py) hoạt động ổn định.

---

## 2. THIẾT KẾ CẤU TRÚC LỚP `CNNAdapter`

### 2.1. Cấu trúc nội bộ trong `CNNAdapter` (Không tạo lớp ngoài)

```python
class CNNAdapter(nn.Module):
    def __init__(
            self,
            in_channels: Union[Tuple[Tuple[int, ...], ...], Tuple[int, ...]] = (64, 128, 256),
            out_dim: int = 256,
            fusion: str = "concat",
            dropout: float = 0.1,
            hidden_dim: int = 256
    ):
        super().__init__()
        # 1. Chuẩn hóa tham số kênh đầu vào (hỗ trợ cả 1D tuple và 3D tuple)
        # 2. Các tầng căn chỉnh kích thước không gian về 40x40:
        #    - p3 downsampling: MaxPool2d(kernel_size=2, stride=2)
        #    - p5 upsampling  : Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        # 3. Các tầng tích chập của Phễu phân tầng (Hierarchical Conv Pyramid Layers):
        #    - self.pyramid_stage1: Conv2d(total_in_channels, hidden_dim, 3, stride=2, padding=1) + BatchNorm2d + SiLU
        #    - self.pyramid_stage2: Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1) + BatchNorm2d + SiLU
        #    - self.pyramid_stage3: Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1) + BatchNorm2d + SiLU
        #    - self.pyramid_stage4: Conv2d(hidden_dim, out_dim, 5, stride=1, padding=0) + BatchNorm2d + SiLU (KHÔNG DÙNG GAP)
        # 4. Cơ chế Fusion và Tương thích ngược:
        #    - self.linear_project (cho tensor vector 2D/3D đã pooling sẵn)
        #    - self.attention_mlp  (khi fusion="attention")
        #    - self.dropout
```

### 2.2. Phương thức `hierarchical_conv_pyramid(self, x: torch.Tensor) -> torch.Tensor`

```python
    def hierarchical_conv_pyramid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Phương thức phễu tích chập phân tầng nén không gian từ 40x40 về vector out_dim:
          - Stage 1: [N, 448, 40, 40] -> [N, hidden_dim, 20, 20] (Tương quan cục bộ mi mắt, môi)
          - Stage 2: [N, hidden_dim, 20, 20] -> [N, hidden_dim, 10, 10] (Tương quan khoảng cách hai mắt, mắt-miệng)
          - Stage 3: [N, hidden_dim, 10, 10] -> [N, hidden_dim, 5, 5] (Tương quan toàn khuôn mặt & góc nghiêng)
          - Stage 4: [N, hidden_dim, 5, 5] -> [N, out_dim, 1, 1] (Tổng hợp cấu trúc bằng kernel 5x5 học được)
          - Squeeze: [N, out_dim, 1, 1] -> [N, out_dim]
        Hoàn toàn KHÔNG dùng Global Average Pooling (GAP).
        """
        x = self.pyramid_stage1(x)
        x = self.pyramid_stage2(x)
        x = self.pyramid_stage3(x)
        x = self.pyramid_stage4(x)
        return x.squeeze(-1).squeeze(-1)
```

### 2.3. Luồng xử lý trong hàm `forward` của `CNNAdapter`

```text
features: (p3, p4, p5)
  │
  ├── 1. Tự động kiểm tra số chiều:
  │      - 5D Video [B, T, C, H, W] -> Gập (B * T, C, H, W)
  │      - 4D Frame [B, C, H, W]    -> Giữ nguyên
  │      - 2D/3D Vector             -> Đi qua nhánh Linear tương thích ngược
  │
  ├── 2. Căn chỉnh về lưới 40x40:
  │      - p3: [N, 64, 80, 80] -> p3_down -> [N, 64, 40, 40]
  │      - p4: [N, 128, 40, 40] giữ nguyên
  │      - p5: [N, 256, 20, 20] -> p5_up -> [N, 256, 40, 40]
  │
  ├── 3. Fusion:
  │      - 'concat'   : torch.cat([out_p3, out_p4, out_p5], dim=1) -> self.hierarchical_conv_pyramid(fused)
  │      - 'attention': Chiếu riêng từng tầng qua conv phân tầng, tính trọng số qua attention MLP
  │      - 'sum'/'mean': Chiếu riêng từng tầng qua conv phân tầng rồi cộng dồn / chia trung bình
  │
  ├── 4. Khôi phục chiều:
  │      - Nếu đầu vào là 5D: Reshape (B * T, out_dim) -> [B, T, out_dim]
  │      - Nếu đầu vào là 4D: [B, out_dim]
  │
  └── 5. Trả về:
         - (output, weights) nếu return_weights=True
         - output nếu return_weights=False
```

---

## 3. CÁC BƯỚC THỰC HIỆN CHI TIẾT (IMPLEMENTATION STEPS)

### Bước 2.1: Sửa đổi và chuẩn hóa [`src/models.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py)
1. **Lớp `CNNAdapter`:**
   - Khởi tạo đầy đủ các tầng `pyramid_stage1`, `pyramid_stage2`, `pyramid_stage3`, `pyramid_stage4`.
   - Viết phương thức `hierarchical_conv_pyramid(self, x: torch.Tensor) -> torch.Tensor`.
   - Cài đặt đầy đủ 4 chiến lược fusion: `concat`, `attention`, `sum`, `mean`.
   - Xử lý gập/trải chiều 5D video tensor.
   - Xử lý đúng cờ `return_weights`.
2. **Lớp `DeepGRUClassifier`:**
   - Tinh chỉnh forward để liên kết hoàn hảo với `CNNAdapter`.
3. **Lớp `DeepLSTMClassifier`:**
   - Cài đặt đầy đủ 3 lớp Deep LSTM, hỗ trợ streaming state `(h_0, c_0)`.
   - Cung cấp các phương thức `from_checkpoint()` và `from_config()`.
4. **Hàm factory `build_model(config, model_type="gru")`:**
   - Khởi tạo model theo cấu hình từ `TrainConfig`.

### Bước 2.2: Đồng bộ hóa [`src/__init__.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/__init__.py)
- Đảm bảo export đầy đủ `CNNAdapter`, `DeepLSTMClassifier`, `DeepGRUClassifier`, `build_model`.

### Bước 2.3: Thực hiện kiểm thử tự động toàn diện
Chạy test suite ngay trong `src/models.py`:
1. Test Tensor 4D đơn khung hình: $p_3 [4, 64, 80, 80], p_4 [4, 128, 40, 40], p_5 [4, 256, 20, 20] \implies [4, 256]$.
2. Test Tensor 5D chuỗi video: $p_3 [2, 120, 64, 80, 80], p_4 [2, 120, 128, 40, 40], p_5 [2, 120, 256, 20, 20] \implies [2, 120, 256]$.
3. Test Tensor 2D và 3D tương thích ngược.
4. Test 4 chế độ Fusion (`concat`, `attention`, `sum`, `mean`).
5. Test `DeepGRUClassifier` (sequence, clip, streaming $h_0$, return_weights).
6. Test `DeepLSTMClassifier` (sequence, clip, streaming state tuple `(h_0, c_0)`).
7. Test `build_model(config)` và kiểm tra import từ `src`.

### Bước 2.4: Tạo báo cáo tổng kết nghiệm thu
- Tạo tệp [`docs/report_cnn_adapter.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/report_cnn_adapter.md) tổng kết toàn bộ kết quả thực thi theo Bước 3 của [`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md).

---

## 4. TIÊU CHUẨN ĐÁNH GIÁ & HOÀN TẤT (ACCEPTANCE CHECKLIST)

- [ ] **Không tạo class mới:** Toàn bộ logic phễu tích chập nằm trong `hierarchical_conv_pyramid` của `CNNAdapter`.
- [ ] **Tuyệt đối không dùng GAP:** Không có `AdaptiveAvgPool2d((1, 1))` hay phép trung bình số học nào.
- [ ] **Không có lỗi Shape Mismatch:** Khắc phục triệt để lỗi kích thước ma trận `819200 vs 1536`.
- [ ] **Hỗ trợ Tensor 5D:** Chạy trơn tru với dữ liệu video $[B, T, C, H, W]$ từ DataLoader HDF5.
- [ ] **Đầy đủ 4 Fusion:** `concat`, `attention`, `sum`, `mean` đều hoạt động chuẩn xác.
- [ ] **Module Integrity:** `src/__init__.py`, `model.py`, `model1.py` import thành công $100\%$.
- [ ] **Kiểm thử tự động:** Toàn bộ test suite trong `src/models.py` chạy thành công không phát sinh lỗi.
- [ ] **Báo cáo nghiệm thu:** Tạo tệp [`docs/report_cnn_adapter.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/report_cnn_adapter.md).
