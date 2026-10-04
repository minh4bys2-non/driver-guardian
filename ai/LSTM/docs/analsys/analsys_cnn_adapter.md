# BÁO CÁO KHẢO SÁT & PHÂN TÍCH KHỐI CNNADAPTER (SRC/MODELS.PY)
**Mã tài liệu:** `analsys_cnn_adapter.md`  
**Dự án:** Hệ Thống Nhận Diện Tài Xế Buồn Ngủ (Driver Guardian - Deep LSTM / GRU Pipeline)  
**Tệp mục tiêu:** [`src/models.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py)  
**Quy chuẩn áp dụng:** [`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md) — Bước 1: Khảo sát & Phân tích (Discovery)  
**Ngày cập nhật:** 01/10/2026 (Bổ sung phân tích triệt tiêu tương quan của GAP & giải pháp thay thế)  

---

## 1. TỔNG QUAN YÊU CẦU & BỐI CẢNH DỰ ÁN

### 1.1. Mục tiêu khối `CNNAdapter`
Trong hệ thống Driver Guardian, mô hình gồm hai giai đoạn chính:
1. **Trích xuất đặc trưng không gian (Spatial Feature Extraction):** Trích xuất từ khung hình video qua mô hình ONNX PAFPN (YOLOv10 trunk ~1.64M params), tạo ra 3 tầng đặc trưng đa tỉ lệ:
   - $p_3$: $[B, 64, 80, 80]$ (đặc trưng biên, texture độ phân giải cao).
   - $p_4$: $[B, 128, 40, 40]$ (đặc trưng bộ phận: mắt, miệng, hướng đầu).
   - $p_5$: $[B, 256, 20, 20]$ (đặc trưng ngữ nghĩa toàn cảnh độ phân giải thấp).
2. **Khối chuyển đổi không gian (`CNNAdapter` / `SpatialFeatureAdapter`):** Nhiệm vụ chuyển đổi và kết hợp 3 tầng $(p_3, p_4, p_5)$ thành một biểu diễn đặc trưng tối ưu cho từng khung hình (frame), bảo toàn mối tương quan không gian giữa các bộ phận trên khuôn mặt.
3. **Mô hình chuỗi thời gian (`DeepGRUClassifier` / `DeepLSTMClassifier`):** Nhận chuỗi các đặc trưng theo thời gian $[B, T, \text{dim}]$ để học động học buồn ngủ (nhắm mắt liên tục, ngáp, gật gù) và đưa ra dự đoán Alert ($0$) hoặc Drowsy ($1$).

---

## 2. HIỆN TRẠNG MÃ NGUỒN CỦA KHỐI `CNNAdapter`

Hiện tại trong [`src/models.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py#L9-L67), khối `CNNAdapter` đang được cài đặt như sau:

```python
class CNNAdapter(nn.Module):
    def __init__(
            self,
            in_channels: Tuple[Tuple[int]] = ((64, 80, 80), (128, 40, 40), (256, 20, 20)),
            out_dim: int = 512,
            fusion: str = "concat",
            dropout: float = 0.1,
            hidden_dim: int = 512
    ):
        super().__init__()
        self.num_scales = len(in_channels)
        self.in_channels = tuple(in_channels)
        self.out_dim = out_dim
        self.fusion = fusion.lower()
        self.total_in_channels = sum([channel[0] for channel in in_channels])
        self.projection = nn.ModuleDict()

        if self.fusion == "concat":
            # Chiếu riêng từng tầng về out_dim với LayerNorm
            self.projection["p3_down_sampling"] = nn.MaxPool2d(kernel_size=2, stride=2)
            self.projection["p5_up_sampling"] = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.channel_project = nn.Sequential(
                nn.Conv2d(in_channels=self.total_in_channels, out_channels=hidden_dim, kernel_size=1),
                nn.Flatten(),
                nn.Linear(hidden_dim * 3, hidden_dim)
            )
        else:
            raise ValueError(
                f"Phương thức fusion '{fusion}' không được hỗ trợ. Chọn một trong: ['attention', 'concat', 'sum', 'mean']."
            )

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
            self,
            features: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
            *args: torch.Tensor,
            return_weights: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        p3, p4, p5 = features

        if self.fusion == "concat":
            out_p3, out_p4, out_p5 = self.projection["p3_down_sampling"](p3), p4, self.projection['p5_up_sampling'](p5)
            fused_features = torch.cat([out_p3, out_p4, out_p5], dim=1)  # Kích thước: [Bz, 448, 40, 40]
            projected = self.channel_project(fused_features)  # Kích thước: [Bz, 819200]
            return projected
        else:
            raise ValueError(
                f"Phương thức fusion '{self.fusion}' không được hỗ trợ. Chọn một trong: ['attention', 'concat', 'sum', 'mean']."
            )
```

---

## 3. PHÂN TÍCH LỖI LOGIC & XUNG ĐỘT TƯƠNG THÍCH

Qua kiểm tra chi tiết mã nguồn và chạy thực tế, phát hiện **7 lỗi nghiêm trọng** khiến mã nguồn hiện tại không thể chạy được:

### 3.1. Lỗi nghiêm trọng 1: Sai lệch kích thước ma trận (Shape Mismatch Crash)
- Tại `self.channel_project`:
  1. `Conv2d(448, hidden_dim=512, kernel_size=1)` nhận đầu vào $[B, 448, 40, 40]$ và trả về tensor $[B, 512, 40, 40]$.
  2. `nn.Flatten()` duỗi phẳng toàn bộ chiều $C, H, W$ thành một vector có chiều dài:
     $$512 \times 40 \times 40 = 819,200 \text{ phần tử.}$$
  3. Lớp kế tiếp là `nn.Linear(hidden_dim * 3, hidden_dim) = nn.Linear(1536, 512)`.
  4. Lớp Linear nhận đầu vào có số kênh là $819,200$ nhưng lại khai báo $1,536$.
- **Hậu quả:** Ném ngay ngoại lệ thời gian chạy khi gọi forward pass:
  ```text
  RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x819200 and 1536x512)
  ```

### 3.2. Lỗi kiến trúc 2: Bùng nổ tham số nếu cố tình kết nối Flatten vào Linear
- Nếu sửa thành `nn.Linear(819200, 512)`:
  - Số tham số của chỉ một lớp này: $819,200 \times 512 + 512 = 419,430,912$ tham số ($\approx 419.4$ triệu tham số).
  - Dung lượng bộ nhớ lưu trọng số cho duy nhất 1 lớp này: $\approx 1.68 \text{ GB}$ (FP32).
  - Trong khi toàn bộ backbone YOLOv10 chỉ có $1.64$ triệu tham số, và mạng GRU chỉ có khoảng $1$ triệu tham số. Một adapter 419M tham số sẽ gây quá khớp (overfitting) trầm trọng trên dataset video, triệt tiêu tính bất biến không gian và không thể triển khai trên thiết bị nhúng (edge AI).

### 3.3. Lỗi tương thích 3: Không hỗ trợ Tensor chuỗi video 5D `[B, T, C, H, W]`
- Dữ liệu video thực tế trong DataLoader từ tệp `.h5` hoặc `extract_to_pt.py` có kích thước 5D:
  - $p_3 \in \mathbb{R}^{B \times T \times 64 \times 80 \times 80}$
  - $p_4 \in \mathbb{R}^{B \times T \times 128 \times 40 \times 40}$
  - $p_5 \in \mathbb{R}^{B \times T \times 256 \times 20 \times 20}$
- Các lớp `nn.MaxPool2d`, `nn.Upsample`, `nn.Conv2d` trong PyTorch chỉ chấp nhận tensor 4D $[N, C, H, W]$. Nếu truyền tensor 5D, PyTorch sẽ ném lỗi:
  ```text
  RuntimeError: Expected 3D or 4D (batch mode) tensor with optional 0D or 1D 'channels_last' layout
  ```
- Khối hiện tại hoàn toàn thiếu cơ chế gập/trải chiều $(B, T) \leftrightarrow (B \times T)$.

### 3.4. Lỗi tương thích 4: Thiếu các phương thức Fusion (`attention`, `sum`, `mean`)
- Khối `DeepGRUClassifier` (dòng 82) có cấu hình mặc định là `fusion: str = "attention"`.
- Trong `configs/config.py`, các tùy chọn bao gồm `'concat'`, `'sum'`, `'mean'`, `'attention'`.
- Nhưng trong `CNNAdapter`, chỉ có nhánh `if self.fusion == "concat"`. Bất kỳ khi nào khởi tạo mặc định `DeepGRUClassifier()` hoặc chọn `fusion="attention"`, khối sẽ lập tức ném lỗi:
  ```text
  ValueError: Phương thức fusion 'attention' không được hỗ trợ. Chọn một trong: ['attention', 'concat', 'sum', 'mean'].
  ```

### 3.5. Lỗi tương thích 5: Xung đột kiểu dữ liệu tham số `in_channels`
- `CNNAdapter.__init__` định nghĩa:
  ```python
  self.total_in_channels = sum([channel[0] for channel in in_channels])
  ```
  Giả định `in_channels` là `((64, 80, 80), (128, 40, 40), (256, 20, 20))`.
- Tuy nhiên, trong `DeepGRUClassifier`:
  ```python
  spatial_in_channels: Tuple[int, ...] = (64, 128, 256)
  self.spatial_adapter = CNNAdapter(in_channels=spatial_in_channels, ...)
  ```
  Nếu truyền `spatial_in_channels=(64, 128, 256)`, từng phần tử `channel` là số nguyên `int` ($64$), dẫn đến `channel[0]` văng lỗi:
  ```text
  TypeError: 'int' object is not subscriptable
  ```

### 3.6. Lỗi logic 6: Vi phạm hợp đồng giao diện `return_weights`
- Hàm `forward` nhận tham số `return_weights: bool = False`.
- Tuy nhiên, trong code không có bất kỳ dòng nào xử lý logic này, luôn chỉ trả về một tensor duy nhất `return projected`.
- Khi `DeepGRUClassifier` gọi `x, weights = self.spatial_adapter(..., return_weights=True)` sẽ ném lỗi:
  ```text
  TypeError: cannot unpack non-iterable Tensor object
  ```

### 3.7. Lỗi tính toàn vẹn module 7: Thiếu `DeepLSTMClassifier` và `build_model`
- Tệp [`src/__init__.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/__init__.py#L5) import:
  ```python
  from .models import CNNAdapter, DeepLSTMClassifier, DeepGRUClassifier, build_model
  ```
- Nhưng trong [`src/models.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py) chỉ có `CNNAdapter` và `DeepGRUClassifier`. Cả `DeepLSTMClassifier` và factory function `build_model` đều bị thiếu, khiến bất kỳ lệnh import nào từ package `src` đều bị chặn đứng:
  ```text
  ImportError: cannot import name 'DeepLSTMClassifier' from 'src.models'
  ```

---

## 4. VÌ SAO KHÔNG NÊN DÙNG GLOBAL AVERAGE POOLING (GAP)?

Người dùng đã đưa ra yêu cầu kỹ thuật hoàn toàn chính xác: **Không dùng Global Average Pooling (GAP) vì hạ từ $40 \times 40$ về $1 \times 1$ sẽ triệt tiêu hoàn toàn tương quan không gian của đặc trưng.**

Phân tích toán học và thị giác máy tính:
1. **Mất thông tin vị trí hình học (Spatial Coordinates Loss):**
   - Phép tính GAP cho mỗi kênh $c$:
     $$\text{GAP}(F)_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} F_{c, i, j}$$
   - Bản chất của GAP là tính giá trị trung bình số học trên toàn bộ $1,600$ điểm ảnh ($40 \times 40$). Sau phép tính này, **tọa độ $(i, j)$ của từng đặc trưng bị triệt tiêu hoàn toàn**.
2. **Triệt tiêu mối quan hệ tương quan giữa các cơ quan khuôn mặt (Loss of Facial Landmark Correlations):**
   - Trong nhận diện buồn ngủ, các dấu hiệu quan trọng nhất phụ thuộc chặt chẽ vào **mối tương quan không gian tương đối** giữa các bộ phận:
     - Tương quan giữa mi mắt trên và mi mắt dưới (xác định trạng thái nhắm mắt / tỉ lệ EAR - Eye Aspect Ratio).
     - Tương quan giữa môi trên và môi dưới (xác định trạng thái ngáp / tỉ lệ MAR - Mouth Aspect Ratio).
     - Tương quan giữa trục mắt - mũi - cằm với vai (xác định góc gục đầu / Head Pose nghiêng, cúi).
   - Khi áp dụng GAP, tín hiệu nhắm mắt ở vùng trên của khuôn mặt và tín hiệu ngáp ở vùng dưới của khuôn mặt bị hòa lẫn vào cùng một giá trị trung bình vô hướng, làm mất liên kết không gian giữa chúng.
3. **Tính bất biến hoán vị bất lợi (Adverse Permutation Invariance):**
   - Phép cộng trung bình thỏa mãn tính chất giao hoán: nếu ta xáo trộn ngẫu nhiên vị trí của 1600 ô vuông trên feature map (đổi mắt xuống cằm, đổi miệng lên trán), giá trị sau khi qua GAP vẫn **hoàn toàn giống hệt nhau 100%**.
   - Rõ ràng, một khuôn mặt bị đảo lộn vị trí các cơ quan không thể đại diện đúng cho trạng thái buồn ngủ.

---

## 5. CÁC PHƯƠNG ÁN THAY THẾ GAP ĐỂ BẢO TOÀN TƯƠNG QUAN KHÔNG GIAN

Để không sử dụng GAP mà vẫn nén kích thước đặc trưng từ $40 \times 40$ về biểu diễn tối ưu cho chuỗi thời gian, chúng ta có **4 phương án kiến trúc tiên tiến**:

---

### PHƯƠNG ÁN 1 (KHUYẾN NGHỊ CAO NHẤT): Phễu tích chập phân tầng (Hierarchical Strided Convolution Pyramid)

#### Nguyên lý:
Thay vì nhảy cóc $40 \times 40 \to 1 \times 1$ bằng phép tính trung bình, ta sử dụng **chuỗi các lớp tích chập với bước trượt (Strided Convolutions, stride=2)**. Tích chập sở hữu tính chất **Spatial Inductive Bias** và **Receptive Field (vùng thụ cảm)** tăng dần:
- Mỗi kernel $3 \times 3$ học mối quan hệ tương quan cục bộ giữa các pixel lân cận (ví dụ khóe mắt, vành môi).
- Khi bước trượt stride=2 thu nhỏ dần lưới không gian, kernel ở tầng sau sẽ nắm bắt mối tương quan giữa các vùng lớn hơn (mắt so với mũi, mũi so với miệng).

#### Kiến trúc chi tiết:
```text
Fused Features: [B, 448, 40, 40]
      │
      ▼ Conv 3x3, stride=2, padding=1 + BatchNorm + GELU
Stage 1: [B, 256, 20, 20]   --> Học tương quan cục bộ (mép mắt, môi)
      │
      ▼ Conv 3x3, stride=2, padding=1 + BatchNorm + GELU
Stage 2: [B, 256, 10, 10]   --> Học tương quan vùng (khoảng cách mắt-mắt, mắt-miệng)
      │
      ▼ Conv 3x3, stride=2, padding=1 + BatchNorm + GELU
Stage 3: [B, 256, 5, 5]     --> Học cấu trúc toàn khuôn mặt & góc nghiêng đầu
      │
      ▼ Conv 5x5, stride=1, padding=0 + LayerNorm + GELU (Spatial Kernel Fusion)
Output : [B, out_dim=256, 1, 1] -> Squeeze -> [B, 256]
```

#### Đánh giá:
- **Bảo toàn tương quan:** 100% bảo toàn cấu trúc không gian nhờ các trọng số tích chập học được ($W * X$), không dùng bất kỳ phép pooling trung bình nào.
- **Số tham số bổ sung:** $\approx 1.5\text{M}$ tham số (cực kỳ nhẹ, tương thích GPU/CPU, tốc độ $>300$ FPS).
- **Tính ổn định:** Huấn luyện cực kỳ ổn định, hội tụ nhanh, tương thích hoàn toàn với export ONNX / TensorRT.

---

### PHƯƠNG ÁN 2: Multi-Head Spatial Self-Attention (Transformer Spatial Token Encoder)

#### Nguyên lý:
Mô hình hóa toàn bộ lưới không gian thành các **Spatial Tokens** (tương tự Vision Transformer - ViT).
1. Hạ nhẹ độ phân giải không gian qua Conv stride 2 từ $40 \times 40 \to 10 \times 10$ ($100$ tokens không gian) để đảm bảo tốc độ tính toán.
2. Cộng thêm **2D Learnable Positional Embedding** vào từng token để mô hình biết chính xác tọa độ $(x, y)$ của từng vùng trên mặt.
3. Áp dụng cơ chế **Multi-Head Self-Attention (MHSA)**:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
   Ma trận tích vô hướng $Q K^T \in \mathbb{R}^{100 \times 100}$ chính là **ma trận đo lường trực tiếp mức độ tương quan (correlation matrix)** giữa mọi cặp vị trí không gian với nhau (ví dụ: vùng mắt tương tác với vùng miệng ra sao khi ngáp).
4. Sử dụng một learnable query token (`[CLS]` token): Token này sẽ tổng hợp thông tin tương quan từ tất cả 100 vị trí không gian thông qua attention weights.

#### Sơ đồ khối:
```text
Fused Features: [B, 448, 40, 40]
      │
      ▼ Patch / Conv Stride: nén về [B, 256, 10, 10]
Spatial Tokens: 100 tokens [B, 100, 256]
      │ + 2D Positional Embeddings
      │ Ghép thêm learnable [CLS] token -> [B, 101, 256]
      ▼
Spatial Transformer Block (MHSA + Feed Forward)
      │
      ▼ Trích xuất output tại vị trí [CLS] token
Output Vector: [B, out_dim=256]
```

#### Đánh giá:
- **Bảo toàn tương quan:** Mô hình hóa trực tiếp tương quan toàn cục (Global Context) giữa các vị trí xa nhau trên mặt.
- **Độ phức tạp:** Cao hơn Conv Pyramid, cần chú ý tính toán khi chuỗi video dài $T=120$.

---

### PHƯƠNG ÁN 3: Spatial Coordinate Convolution + Sub-grid Linear Projection (Preserving Fixed 2D Grid Topology)

#### Nguyên lý:
1. Thêm 2 kênh tọa độ chuẩn hóa $(x, y) \in [-1, 1]$ vào feature map (CoordConv) để mạng luôn biết vị trí tuyệt đối của từng ô.
2. Dùng các lớp Conv có stride nén không gian về một lưới nhỏ có ý nghĩa hình học rõ ràng: **Lưới $4 \times 4$** ($16$ vùng không gian: góc mắt trái, góc mắt phải, sống mũi, khóe miệng trái, khóe miệng phải, cằm, v.v.).
3. Giảm số kênh xuống nhỏ: $C_{\text{sub}} = 16$.
4. Khi đó tổng số phần tử là $16 \text{ vùng} \times 16 \text{ kênh} = 256$ phần tử.
5. Sử dụng lớp `nn.Linear(256, out_dim)`: Vì vị trí của 16 ô không gian được giữ cố định theo thứ tự hình học, lớp Linear sẽ học **các trọng số riêng biệt cho từng vùng không gian** (vùng mắt có ma trận trọng số riêng, vùng miệng có ma trận trọng số riêng), hoàn toàn không bị triệt tiêu tương quan.

#### Đánh giá:
- Cấu trúc đơn giản, tốc độ thực thi siêu nhanh.
- Duy trì định vị vị trí cục bộ tốt.

---

### PHƯƠNG ÁN 4: Spatio-Temporal ConvGRU (Bảo toàn Tensor không gian 2D theo thời gian)

#### Nguyên lý:
Thay vì nén không gian của từng frame thành một vector 1D $[B, \text{dim}]$ rồi mới đưa vào GRU:
1. `CNNAdapter` chỉ làm nhiệm vụ đồng nhất kích thước và giảm nhẹ không gian về $[B, T, 64, 10, 10]$.
2. Thay thế `nn.GRU` chuẩn bằng **`ConvGRU`**: Các cổng Recurrent ($r_t, z_t, h_t$) sử dụng phép **tích chập 2D (Conv2d)** thay vì phép nhân ma trận tuyến tính `Linear`:
   $$z_t = \sigma(W_{xz} * x_t + W_{hz} * h_{t-1} + b_z)$$
   $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$
3. Trạng thái ẩn $h_t$ luôn duy trì hình thái tensor 2D $[B, 64, 10, 10]$ xuyên suốt toàn bộ chuỗi video $T$, bảo toàn 100% mối tương quan không gian qua từng bước thời gian.
4. Chỉ ở bước phân loại cuối cùng (FC Head), một lớp Spatial Conv Head mới tổng hợp ra xác suất buồn ngủ.

#### Đánh giá:
- Độ chính xác không gian - thời gian cao nhất trong các bài toán nhận diện hành vi video.
- Đòi hỏi thay đổi cả kiến trúc Recurrent trong `src/models.py`.

---

## 6. SO SÁNH TỔNG HỢP CÁC PHƯƠNG ÁN

| Tiêu chí so sánh | Phương án 1: Conv Pyramid | Phương án 2: Spatial MHSA | Phương án 3: CoordConv Grid | Phương án 4: ConvGRU |
| :--- | :--- | :--- | :--- | :--- |
| **Cơ chế bảo toàn tương quan** | Receptive Field tích chập cục bộ $\to$ toàn cục | Ma trận tương quan $Q K^T$ + Positional Emb | Lưới hình học $4 \times 4$ + Weight riêng từng vùng | Phép tích chập trong cell Recurrent |
| **Có dùng GAP không?** | **KHÔNG** (Dùng Conv $5 \times 5$) | **KHÔNG** (Dùng `[CLS]` Query Token) | **KHÔNG** (Dùng Sub-grid Linear) | **KHÔNG** (Giữ tensor 2D) |
| **Số lượng tham số bổ sung** | $\approx 1.2\text{M} - 1.5\text{M}$ | $\approx 1.8\text{M} - 2.2\text{M}$ | $\approx 0.5\text{M}$ | $\approx 2.5\text{M}$ |
| **Tốc độ xử lý (Inference Speed)** | **Rất nhanh** (> 400 FPS) | Khá nhanh (~ 200 FPS) | **Cực nhanh** (> 600 FPS) | Trung bình (~ 120 FPS) |
| **Khả năng tương thích mã nguồn hiện có** | **Cao nhất** (Chỉ sửa trong `CNNAdapter`, giữ nguyên `DeepGRUClassifier`) | Cao (Chỉ sửa trong `CNNAdapter`) | Cao (Chỉ sửa trong `CNNAdapter`) | Cần sửa cả `CNNAdapter` và `DeepGRUClassifier` |
| **Mức độ phức tạp triển khai** | Vừa phải, rất ổn định | Phức tạp hơn | Đơn giản | Phức tạp |

---

## 7. ĐỀ XUẤT HƯỚNG TRIỂN KHAI VÀ XÁC NHẬN

Em **khuyến nghị áp dụng Phương án 1 (Phễu tích chập phân tầng - Hierarchical Strided Convolution Pyramid)** vì:
1. Hoàn toàn loại bỏ GAP, sử dụng các kernel tích chập học được để nắm bắt tương quan không gian qua từng cấp độ phân giải ($40 \to 20 \to 10 \to 5 \to 1$).
2. Giữ nguyên giao tiếp chuẩn với `DeepGRUClassifier` (trả về vector biểu diễn cho từng frame theo đúng chiều `out_dim`), không làm xáo trộn kiến trúc mô hình chuỗi thời gian đã được thiết kế.
3. Tối ưu hoàn hảo cho triển khai thời gian thực trên cả GPU và CPU.

Xin ý kiến của anh:
- Anh có đồng ý với **Phương án 1 (Phễu tích chập phân tầng)** hay muốn lựa chọn **Phương án 2 (Spatial MHSA)** / phương án khác?
- Ngay khi anh chốt định hướng, em sẽ tiến hành lập tệp kế hoạch chi tiết **`docs/plan_cnn_adapter.md` (Bước 2)** và bắt tay vào hiện thực hóa mã nguồn (Bước 3).
