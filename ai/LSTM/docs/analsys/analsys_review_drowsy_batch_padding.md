# TÀI LIỆU PHÂN TÍCH & ĐÁNH GIÁ MÃ NGUỒN MODEL DROWSYCLASSIFIER (SRC/MODELS1.PY)
## KIỂM TRA LỖI ẨN TRONG QUÁ TRÌNH HUẤN LUYỆN KHI CÓ PADDING Ở BATCH

- **Mã yêu cầu**: `REVIEW_DROWSY_BATCH_PADDING`
- **Tệp phân tích**: `docs/analsys/analsys_review_drowsy_batch_padding.md`
- **Tệp đối tượng đánh giá chính**: `src/models1.py` (kết hợp `src/models.py`, `src/convgru.py`, `configs/config.py`, `src/train1.py`)
- **Kỹ năng áp dụng**: `agent-skills:code-review-and-quality` (Đánh giá đa chiều 5 trục: Correctness, Readability, Architecture, Security, Performance)
- **Mục tiêu**: Kiểm tra toàn diện kiến trúc mô hình `DrowsyClassifier`, phát hiện tất cả các lỗi ẩn (hidden bugs), lỗi logic toán học, rò rỉ gradient và nguy cơ sụt giảm hiệu năng trong quá trình huấn luyện khi batch chứa padding (do video có độ dài thời gian không đồng nhất).
- **Trạng thái**: Đã hoàn thành phân tích — Chờ người dùng xem xét và phê duyệt (Bước 1 theo quy trình chuẩn `AGENTS.md`).

---

## 1. TỔNG QUAN YÊU CẦU & KẾT LUẬN SƠ BỘ

### 1.1. Bối cảnh bài toán
Trong bài toán nhận diện tài xế buồn ngủ qua video (`driver-guardian`), các video clip có độ dài thời gian không đồng đều ($T_i$ thay đổi từ $10$ đến hơn $100$ khung hình). Khi gom nhóm vào mini-batch qua hàm `collate_video_frames` (`src/dataset.py`):
1. Tensor video 5D đầu vào được đệm số 0 (Zero-Padding) theo trục thời gian đến độ dài tối đa $T_{\max} = \max(T_i)$: `[B, T_max, 3, H, W]`.
2. Vector `seq_lens` kích thước `[B]` lưu trữ độ dài thực tế của từng clip.
3. Mô hình End-to-End `DrowsyClassifier` (`src/models1.py`) chịu trách nhiệm nhận tensor video 5D này, trích xuất đặc trưng không gian qua Spatial Backbone & PAFPN theo cơ chế Mini-Chunking, nén kênh qua `SpatialReductionNeck`, mô hình hóa chuỗi thời gian qua `ConvGRU`, gom tụ bằng cơ chế Chú ý Kép (`SpatialAttentionPooling` + `TemporalAttentionPooling`), và đưa ra logit dự đoán.

### 1.2. Kết luận đánh giá sơ bộ
Qua quá trình rà soát mã nguồn chi tiết kết hợp chạy kiểm thử thực nghiệm (empirical verification), chúng tôi xác nhận **CÓ TỒN TẠI 5 LỖI ẨN NGHIÊM TRỌNG VÀ 3 RỦI RO THIẾT KẾ** trong mô hình `DrowsyClassifier` và các module liên kết khi thực hiện huấn luyện với batch có padding:
- **01 Lỗi Fatal Runtime Crash**: Vỡ chiều không gian (Dimension Mismatch) trong `_align_features` do xung đột giữa `target_size=(14, 14)` mặc định của `DrowsyClassifier` và độ phân giải ảnh $640 \times 640$ của hệ thống.
- **01 Lỗi Toán học Gradient Checkpointing & BatchNorm**: Thống kê `running_mean`/`running_var` của BatchNorm bị cập nhật **2 lần mỗi step** (Double Running Stats Update) khi `gradient_checkpointing=True` kết hợp `freeze_bn=False` (cấu hình mặc định trong `configs/config.py` và `src/train1.py`).
- **01 Lỗi Rò rỉ Logic Attention tại Frame Padding**: `SpatialAttentionPooling` gán điểm `-1e9` cho frame padding nhưng `F.softmax` trên lưới không gian biến nó thành trọng số phân phối đều $\frac{1}{H \times W}$, kết hợp với trạng thái bị đóng băng của `ConvGRU` tạo ra **vector đặc trưng phi số 0** tại các vị trí padding.
- **01 Lỗi Nghẽn cổ chai Đồng bộ GPU-CPU (Host-Device Sync)**: Lệnh `bool(valid_mask.all())` ép CUDA stream phải đồng bộ về CPU ở mỗi forward pass, triệt tiêu tính bất đồng bộ của GPU pipeline.
- **01 Lỗi Cấp phát bộ nhớ không cần thiết**: Gọi in-place `chunk.requires_grad_(True)` cho input tensor khi dùng `use_reentrant=False` gây tính toán và lưu trữ gradient thừa của pixel ảnh đầu vào (`chunk.grad`).

---

## 2. BẢNG PHÂN LOẠI CHI TIẾT CÁC LỖI PHÁT HIỆN ĐƯỢC

Theo chuẩn phân loại mức độ nghiêm trọng của `agent-skills:code-review-and-quality`:

| STT | Mức độ nghiêm trọng | Vị trí mã nguồn | Tên lỗi / Vấn đề | Tác động trong quá trình huấn luyện |
| :---: | :---: | :---: | :--- | :--- |
| **01** | **Critical** | `src/models1.py` (L41, L88) & `src/models.py` (L102-108, L146) | **Lỗi vỡ kích thước không gian `_align_features` khi ảnh $\ne 224\times 224$** | **CRASH RUNTIME NGAY LẬP TỨC**: Khi huấn luyện với ảnh $640\times 640$ (mặc định của dự án), `torch.cat` báo lỗi kích thước chiều không gian $40 \ne 14$. |
| **02** | **Critical** | `src/models1.py` (L44, L173-177) & `configs/config.py` (L114-116) | **Lỗi cập nhật kép BatchNorm Stats khi Checkpoint + `freeze_bn=False`** | **SAI LỆCH THỐNG KÊ MÔ HÌNH**: `running_mean`/`running_var` của Backbone/Neck bị cập nhật 2 lần/step; thống kê bị méo mó nghiêm trọng khi kích thước chunk cuối bị lẻ do padding. |
| **03** | **Required** | `src/models.py` (L280-292) & `src/convgru.py` (L431-435) | **Rò rỉ vector đặc trưng & Attention rác tại khung hình Padding** | `SpatialAttentionPooling` gán trọng số đều $\frac{1}{HW}$ thay vì $0.0$; `ConvGRU` giữ nguyên trạng thái $h_{seq\_len-1}$, sinh vector đặc trưng giả tại các bước padding. |
| **04** | **Required** | `src/models1.py` (L133-139, L150-159) | **Hardcoded kích thước không gian tại edge-case batch rỗng / padding 100%** | Hardcode `(28, 28), (14, 14), (7, 7)` gây crash nếu ảnh khác $224\times 224$; ngắt Autograd graph gây lỗi DDP synchronization. |
| **05** | **Required** | `src/models1.py` (L142) | **CUDA Host-Device Synchronization do `bool(valid_mask.all())`** | Ép GPU dừng chờ CPU ở mọi forward pass, gây sụt giảm FPS và nghẽn pipeline huấn luyện. |
| **06** | **Optional** | `src/models1.py` (L174-175) | **Cấp phát gradient ảnh đầu vào thừa do `chunk.requires_grad_(True)`** | Lãng phí VRAM để lưu `chunk.grad` khi sử dụng PyTorch $\ge 2.0$ `use_reentrant=False`. |
| **07** | **Optional** | `src/models1.py` (L252-255) | **Thiếu ép kiểu an toàn khi `seq_lens` là Python list/tuple** | Gây `AttributeError` nếu người dùng truyền `seq_lens=[8, 5]` thay vì `torch.Tensor`. |
| **08** | **Optional** | `src/train1.py` (L160-171) & `src/models1.py` (L32-45) | **Bỏ quên `chunk_size` và `num_layers` trong khởi tạo** | `train1.py` không truyền `cfg.chunk_size` vào `DrowsyClassifier`; `DrowsyClassifier` không expose `num_layers`. |

---

## 3. PHÂN TÍCH KỸ THUẬT CHI TIẾT TỪNG VẤN ĐỀ

### 3.1. Vấn đề 1 (Critical): Xung đột kích thước không gian giữa `target_size` và độ phân giải đầu vào
- **Đoạn mã gây lỗi trong `src/models1.py`**:
  ```python
  # models1.py L41
  target_size: Tuple[int, int] = (14, 14),
  ```
- **Đoạn mã gây lỗi trong `src/models.py` (`SpatialReductionNeck`)**:
  ```python
  # models.py L102-108
  def _align_features(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
      th, tw = self.target_size
      out_p3 = self.p3_down(p3) if p3.shape[-2:] != (th, tw) else p3
      out_p4 = p4 if p4.shape[-2:] == (th, tw) else F.interpolate(p4, size=(th, tw), mode="bilinear", align_corners=False)
      out_p5 = self.p5_up(p5) if p5.shape[-2:] != (th, tw) else p5
      return out_p3, out_p4, out_p5
  ```
- **Bản chất lỗi**:
  - Khi input video là $640 \times 640$ (cấu hình chuẩn của `configs/config.py` dòng 69: `image_size: (640, 640)`):
    - Backbone PAFPN tạo ra:
      - $p3$: $80 \times 80$ (stride 8)
      - $p4$: $40 \times 40$ (stride 16)
      - $p5$: $20 \times 20$ (stride 32)
    - Khi `target_size = (14, 14)`:
      - `p3` ($80 \times 80$) đi qua `p3_down` (`MaxPool2d(stride=2)`): giảm 2 lần thành **$40 \times 40$** (chứ không về $14 \times 14$).
      - `p4` ($40 \times 40$) được nội suy `F.interpolate` về **$14 \times 14$**.
      - `p5` ($20 \times 20$) đi qua `p5_up` (`Upsample(scale_factor=2)`): tăng 2 lần thành **$40 \times 40$** (chứ không về $14 \times 14$).
    - Tại dòng 146 (`torch.cat([out_p3, out_p4, out_p5], dim=1)`):
      Hệ thống ghép nối tensor kích thước: `out_p3 [40, 40]`, `out_p4 [14, 14]`, `out_p5 [40, 40]`.
    - **Kết quả thực nghiệm**: PyTorch ném ngoại lệ:
      ```text
      RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 40 but got size 14 for tensor number 1 in the list.
      ```
- **Giải pháp**:
  1. Trong `DrowsyClassifier`, `target_size` phải được tự động tính toán thích ứng từ kích thước đầu vào hoặc truyền tương thích (ví dụ: với ảnh 640x640, target_size bắt buộc là (40, 40); với ảnh 224x224, target_size là (14, 14)).
  2. Trong `SpatialReductionNeck._align_features`, cần thay đổi `p3_down` và `p5_up` hoặc `adaptive_pool2d` / `interpolate` để đảm bảo 100% cả 3 mức $p3, p4, p5$ luôn khớp chính xác với `(th, tw)`.

---

### 3.2. Vấn đề 2 (Critical): Cập nhật kép thống kê BatchNorm và nhiễu loạn do Mini-Chunking
- **Đoạn mã liên quan trong `src/models1.py`**:
  ```python
  # models1.py L173-177
  if self.gradient_checkpointing and self.training:
      if not chunk.requires_grad:
          chunk.requires_grad_(True)
      o3, o4, o5 = torch.utils.checkpoint.checkpoint(_stage_fwd, chunk, use_reentrant=False)
  else:
      o3, o4, o5 = _stage_fwd(chunk)
  ```
- **Bản chất lỗi**:
  1. **Lỗi Double Update trong PyTorch Checkpoint**:
     Khi `gradient_checkpointing=True` và `freeze_bn=False` (giá trị mặc định trong `TrainConfig`):
     - Trong forward pass: `_stage_fwd(chunk)` chạy qua Backbone/PAFPN. Các lớp `BatchNorm2d` tính toán mean/var của chunk và cập nhật vào `running_mean` và `running_var`.
     - Trong backward pass: PyTorch `checkpoint` gọi lại `_stage_fwd(chunk)` để tính lại activation. Do `BatchNorm2d` vẫn ở chế độ `training`, nó **lại cập nhật một lần nữa** vào `running_mean` và `running_var`.
     - Kết quả thực nghiệm bằng script độc lập:
       ```text
       diff running_mean sau forward : 0.007195
       diff running_mean sau backward: 0.006475 (Được cập nhật lần 2!)
       ```
  2. **Nhiễu loạn thống kê do Padding làm lẻ kích thước chunk**:
     - Giả sử batch gồm 2 video: Video A có 32 frame, Video B có 17 frame thực tế (15 frame padding).
     - Tổng frame hợp lệ: $N_{\text{active}} = 49$.
     - Với `chunk_size = 32`:
       - Chunk 1: 32 frame. BatchNorm tính thống kê trên 32 frame.
       - Chunk 2: $49 - 32 = 17$ frame. BatchNorm tính thống kê trên 17 frame.
     - Nếu $N_{\text{active}} = 33$, Chunk 2 chỉ có **đúng 1 frame**. Khi đó BatchNorm tính mean/var trên 1 frame đơn lẻ (mất tính chất thống kê đại diện nhóm), làm lệch lạc phân phối đặc trưng.
- **Giải pháp**:
  - Đối với video backbone xử lý theo chunk: Bắt buộc cấu hình mặc định `freeze_bn = True`.
  - Khi `freeze_bn = True`, toàn bộ `BatchNorm2d` trong Backbone/Neck được giữ ở chế độ `eval()`, hoạt động như phép biến đổi affine tuyến tính cố định ($y = \gamma \hat{x} + \beta$), loại bỏ hoàn toàn lỗi cập nhật kép và triệt tiêu ảnh hưởng của padding.

---

### 3.3. Vấn đề 3 (Required): Rò rỉ vector đặc trưng & Attention rác tại khung hình Padding
- **Cơ chế lan truyền hiện tại qua các tầng**:
  1. Tại `DrowsyClassifier`:
     Khung hình padding được lọc ra trước Backbone, sau đó được đệm lại bằng số $0$ vào tensor $(p3, p4, p5)$.
  2. Tại `ConvGRU` (`src/convgru.py` L431-435):
     ```python
     if masks is not None:
         mask_t = masks[:, t].view(batch_size, 1, 1, 1).to(dtype=h_next.dtype)
         h_t = mask_t * h_next + (1.0 - mask_t) * h_t.to(dtype=h_next.dtype)
     ```
     Tại bước thời gian padding ($t \ge seq\_len$), `mask_t = 0.0`. Trạng thái ẩn $h_t$ không cập nhật mà **giữ nguyên giá trị của frame hợp lệ cuối cùng**: $h_t = h_{seq\_len - 1}$.
     Do đó, tensor đầu ra $st\_out[:, t]$ tại các frame padding **không phải là 0 mà lặp lại đặc trưng của frame cuối**.
  3. Tại `SpatialAttentionPooling` (`src/models.py` L280-292):
     ```python
     scores = torch.full((orig_b * orig_t, 1, h, w), fill_value=-1e9, ...)
     scores[mask] = valid_scores
     # Softmax trên không gian:
     weights = F.softmax(scores.view(-1, 1, h * w), dim=-1).view(-1, 1, h, w)
     pooled = (x_reshaped * weights).sum(dim=(-2, -1))
     ```
     - Tại các frame padding, toàn bộ điểm số không gian là vector hằng số $[-1e9, -1e9, \dots, -1e9]$.
     - Theo định nghĩa hàm Softmax: $\text{softmax}([-c, -c, \dots, -c])_i = \frac{e^{-c}}{K \cdot e^{-c}} = \frac{1}{K} = \frac{1}{H \times W}$.
     - Trọng số chú ý tại frame padding không bị triệt tiêu về $0$ mà trở thành **trọng số phân phối đều** ($\sum weights = 1.0$).
     - Kết quả gom tụ: $pooled$ tại frame padding bằng trung bình cộng không gian của frame cuối:
       ```text
       sample 1 frame 2 (padded) norm: 0.584791  (Phi số 0!)
       weight sum at padded frame    : 1.0       (Không phải 0.0!)
       ```
- **Tác động**:
  - Ở chế độ `attention_pooling` (clip-level): `TemporalAttentionPooling` dùng masking thời gian nhân với $0.0$, tạm thời che đi lỗi này.
  - Nhưng ở chế độ `sequence` (frame-level): Logits của frame padding bị gán $0.0$, tương ứng xác suất $50\%/50\%$ cho bài toán 2 lớp. Nếu hàm loss không mask bỏ các token padding qua `ignore_index`, loss sẽ bị cộng dồn rác ở mỗi frame padding.
  - Khi trực quan hóa (Attention Map Visualization): Bản đồ không gian của các frame padding bị hiển thị sai lệch thành vùng kích hoạt đều $1/(HW)$ thay vì đen hoàn toàn (zero).
- **Giải pháp**:
  Trong `SpatialAttentionPooling`, sau khi tính `pooled`, cần nhân trực tiếp với mặt nạ hợp lệ:
  `pooled = pooled * mask.view(orig_b * orig_t, 1)` và `weights = weights * mask.view(orig_b * orig_t, 1, 1, 1)`.

---

### 3.4. Vấn đề 4 (Required): Hardcoded spatial dimensions tại edge-case batch rỗng / padding 100%
- **Đoạn mã gây lỗi trong `src/models1.py`**:
  ```python
  # models1.py L136-138
  if total_frames == 0:
      return (
          torch.empty((0, c3, 28, 28), device=flat_frames.device, dtype=flat_frames.dtype),
          torch.empty((0, c4, 14, 14), device=flat_frames.device, dtype=flat_frames.dtype),
          torch.empty((0, c5, 7, 7), device=flat_frames.device, dtype=flat_frames.dtype),
      )

  # models1.py L152-158
  if num_active == 0:
      h3, w3 = self.target_size[0] * 2, self.target_size[1] * 2
      h4, w4 = self.target_size[0], self.target_size[1]
      h5, w5 = max(1, self.target_size[0] // 2), max(1, self.target_size[1] // 2)
  ```
- **Bản chất lỗi**:
  - Các kích thước không gian $(28, 28), (14, 14), (7, 7)$ bị hardcode cố định dựa trên độ phân giải $224 \times 224$.
  - Nếu đầu vào là $640 \times 640$, kích thước đúng phải là $80 \times 80, 40 \times 40, 20 \times 20$ (hoặc tính toán động theo stride: $H//8, H//16, H//32$).
  - Khi hardcode sai kích thước, tensor trả về sẽ làm sập (crash) ngay các tầng tích chập hoặc reshape phía sau.
- **Giải pháp**:
  Tính toán spatial shape động từ kích thước của `flat_frames`:
  `H_in, W_in = flat_frames.shape[-2:]`
  `h3, w3 = H_in // 8, W_in // 8`
  `h4, w4 = H_in // 16, W_in // 16`
  `h5, w5 = H_in // 32, W_in // 32`

---

### 3.5. Vấn đề 5 (Required): Nghẽn cổ chai Host-Device Sync do `bool(valid_mask.all())`
- **Đoạn mã liên quan trong `src/models1.py`**:
  ```python
  # models1.py L142
  if valid_mask is not None and not bool(valid_mask.all()):
  ```
- **Bản chất lỗi**:
  - `valid_mask` là một CUDA tensor.
  - Phép gọi `valid_mask.all()` trả về CUDA bool tensor.
  - Phép ép kiểu Python `bool(...)` buộc PyTorch phải gọi `cudaStreamSynchronize()` để kéo giá trị boolean 1-bit từ GPU về CPU.
  - Việc này diễn ra **ở mọi forward pass** của quá trình train, chặn đứng cơ chế non-blocking async execution của PyTorch CUDA, làm giảm đáng kể thông lượng huấn luyện (FPS).
- **Giải pháp**:
  Xử lý phân nhánh logic hoàn toàn trên GPU hoặc chỉ cần kiểm tra `if valid_mask is not None:`.

---

### 3.6. Vấn đề 6 (Optional): Lãng phí tài nguyên khi gọi in-place `chunk.requires_grad_(True)`
- **Đoạn mã liên quan trong `src/models1.py`**:
  ```python
  # models1.py L174-175
  if not chunk.requires_grad:
      chunk.requires_grad_(True)
  o3, o4, o5 = torch.utils.checkpoint.checkpoint(_stage_fwd, chunk, use_reentrant=False)
  ```
- **Bản chất lỗi**:
  - Với `use_reentrant=False` trong PyTorch 2.x, `checkpoint` tự động kích hoạt tính toán gradient cho các tham số bên trong `_stage_fwd` ngay cả khi input tensor không có `requires_grad=True`.
  - Việc bật `chunk.requires_grad_(True)` khiến PyTorch phải cấp phát tensor bộ nhớ chứa gradient của pixel ảnh đầu vào $\frac{\partial L}{\partial \text{chunk}}$ trong quá trình backward pass, làm tăng đỉnh bộ nhớ VRAM không cần thiết.
- **Giải pháp**:
  Có thể loại bỏ dòng `chunk.requires_grad_(True)` khi `use_reentrant=False`, hoặc dùng dummy tensor để kích hoạt nếu cần tương thích PyTorch cũ.

---

### 3.7. Vấn đề 7 (Optional): Thiếu ép kiểu an toàn khi `seq_lens` không phải là `torch.Tensor`
- **Đoạn mã trong `src/models1.py`**:
  ```python
  # models1.py L254
  lens_expanded = seq_lens.to(x.device).unsqueeze(1)
  ```
- **Bản chất lỗi**:
  Nếu người dùng truyền `seq_lens` dạng Python list `[8, 5]` hoặc numpy array, lệnh `.to(x.device)` sẽ gây `AttributeError`.
- **Giải pháp**:
  Chuẩn hóa: `seq_lens = torch.as_tensor(seq_lens, device=x.device, dtype=torch.long)`.

---

## 4. BẢNG TỔNG KẾT ĐÁNH GIÁ THEO 5 TRỤC (FIVE-AXIS REVIEW)

| Trục đánh giá | Đánh giá hiện trạng | Điểm mấu chốt cần cải tiến |
| :--- | :--- | :--- |
| **1. Correctness (Tính đúng đắn)** | ❌ **Không đạt chuẩn (Failed)** | - Lỗi vỡ kích thước trong `SpatialReductionNeck` với ảnh $640\times 640$.<br>- Lỗi cập nhật kép thống kê BatchNorm khi dùng Gradient Checkpoint.<br>- Lỗi rò rỉ attention trọng số phân phối đều tại khung hình padding. |
| **2. Readability & Simplicity** | ⚠️ **Cần đơn giản hóa** | - Lặp lại cơ chế mask và scatter tensor 3 lần độc lập giữa các module.<br>- Thiếu đồng bộ về cách đặt tên và truyền tham số (`target_size`, `chunk_size`). |
| **3. Architecture** | ⚠️ **Cần chuẩn hóa interface** | - `DrowsyClassifier.__init__` thiếu tham số `num_layers` dẫn đến hardcode số tầng ConvGRU.<br>- `train1.py` không truyền `cfg.chunk_size` vào `DrowsyClassifier`. |
| **4. Security & Robustness** | ⚠️ **Cần bổ sung guard clauses** | - Edge cases: `total_frames == 0`, `num_active == 0`, `seq_lens` dạng list.<br>- Cần bảo vệ an toàn để không ngắt Autograd graph trong DDP. |
| **5. Performance** | ⚠️ **Bị nghẽn cổ chai** | - Host-Device sync do `bool(valid_mask.all())`.<br>- Lãng phí VRAM do tính gradient ảnh đầu vào không cần thiết. |

---

## 5. ĐỀ XUẤT HƯỚNG GIẢI QUYẾT & KẾ HOẠCH BƯỚC TIẾP THEO

Theo quy trình làm việc chuẩn trong `AGENTS.md`:
1. **Bước 1 (Hiện tại)**: Hoàn thành file khảo sát & phân tích `docs/analsys/analsys_review_drowsy_batch_padding.md` trình bày chi tiết toàn bộ các lỗi ẩn và nguyên nhân gốc rễ.
2. **Bước 2**: Sau khi người dùng xác nhận và đồng ý với các phân tích trên, tiến hành lập kế hoạch chi tiết trong tệp `docs/plan/plan_review_drowsy_batch_padding.md`.
3. **Bước 3**: Triển khai sửa lỗi, tối ưu mã nguồn trong `src/models1.py`, `src/models.py`, `configs/config.py` và chạy bộ kiểm thử tự động toàn diện để nghiệm thu kết quả trong `docs/report/report_review_drowsy_batch_padding.md`.

---
*Tài liệu phân tích được lập bởi Antigravity Pair Programmer.*
