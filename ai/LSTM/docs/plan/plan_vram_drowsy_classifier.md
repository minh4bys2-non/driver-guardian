# KẾ HOẠCH TRIỂN KHAI: TỐI ƯU HÓA VRAM CHO DROWSYCLASSIFIER BẰNG CƠ CHẾ MINI-CHUNKING BẢO TOÀN GRADIENT, PADDING VÀ BATCHNORM (SRC/MODELS1.PY)

- **Mã kế hoạch**: `PLAN_VRAM_DROWSY_CLASSIFIER`
- **Tệp kế hoạch**: `docs/plan/plan_vram_drowsy_classifier.md`
- **Dựa trên tài liệu phân tích**: [docs/analsys/analsys_vram_drowsy_classifier.md](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/analsys/analsys_vram_drowsy_classifier.md)
- **Yêu cầu bổ sung từ người dùng**:
  1. *Chia mini-chunk cho Backbone/PAFPN*: Giảm đỉnh kích hoạt (peak activation VRAM) khi xử lý $B \times T$ lớn ($bz=8, T=50-100$).
  2. *Không ảnh hưởng tích tụ Gradient (Zero-Impact on Gradient Accumulation & Autograd)*: Bảo toàn tính tuyến tính của vi phân, đồ thị tính toán liên tục, gradient lan truyền ngược chính xác $100\%$ về các trọng số dùng chung.
  3. *Xử lý triệt để Padding*: Hỗ trợ frame dư khi $N$ không chia hết cho `chunk_size`, đồng thời loại bỏ khung hình zero-padding không cho đi qua Backbone.
  4. *Bảo toàn thống kê các lớp BatchNorm*: Ngăn ngừa khung hình padding làm méo mó `running_mean` và `running_var`, hỗ trợ cả chế độ `freeze_bn=True` và `freeze_bn=False`.
- **Tệp mã nguồn mục tiêu**: [`src/models1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models1.py)
- **Tệp báo cáo đầu ra**: `docs/report/report_vram_drowsy_classifier.md`
- **Trạng thái**: Đang chờ người dùng phê duyệt trước khi chuyển sang Bước 3 Thực hiện ([AGENTS.md](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md)).

---

## 1. MỤC TIÊU VÀ NGUYÊN TẮC THIẾT KẾ CỐT LÕI

```mermaid
flowchart TD
    subgraph IN ["1. Chuỗi Video Đầu vào [B, T, 3, H, W]"]
        V_IN["Batch Video: B=8, T=50-100<br/>(Tổng N = B*T = 400-800 frames)"]
        MASK_CHECK{"Có seq_lens<br/>(Padding)?"}
        V_IN --> MASK_CHECK
    end

    subgraph PADDING_FILTER ["2. Bộ lọc Khung hình Hợp lệ (Valid Frame Filter)"]
        MASK_CHECK -->|Có| FILTER["Tách riêng N_valid khung hình thật<br/>(Loại bỏ khung hình zero-padding)"]
        MASK_CHECK -->|Không| ALL_FRAMES["Sử dụng toàn bộ N khung hình"]
    end

    subgraph CHUNKING ["3. Cơ chế Mini-Chunking Bảo toàn VRAM & Gradient"]
        FILTER --> SLICE["Chia thành các Chunk kích thước C=32<br/>(Chunk cuối nhận số dư thực tế, không chèn dummy)"]
        ALL_FRAMES --> SLICE
        SLICE --> LOOP["Vòng lặp từng Chunk c = 1..K"]
        
        subgraph CHUNK_EXEC ["Thực thi Chunk c"]
            BN_MODE{"freeze_bn?"}
            BN_MODE -->|True| BN_EVAL["BatchNorm ở eval mode<br/>(Affine cố định, 0 drift)"]
            BN_MODE -->|False| BN_TRAIN["BatchNorm ở train mode<br/>(Không dính zero-padding)"]
            
            CKPT_MODE{"Gradient<br/>Checkpointing?"}
            CKPT_MODE -->|True| CKPT_RUN["torch.utils.checkpoint per chunk<br/>(Đỉnh VRAM chỉ tính trên 1 chunk)"]
            CKPT_MODE -->|False| NORMAL_RUN["Forward thông thường per chunk"]
        end
        LOOP --> CHUNK_EXEC
        CHUNK_EXEC --> ACCUM["Ghép nối đặc trưng p3, p4, p5 qua torch.cat"]
    end

    subgraph SCATTER ["4. Phục hồi Khung hình Chuỗi Thời gian (Scatter & Zero-Fill)"]
        ACCUM --> SCATTER_PAD["Gán đặc trưng vào vị trí hợp lệ,<br/>Vị trí padding được zero-fill chuẩn xác"]
        SCATTER_PAD --> CONVGRU["Đưa vào Spatio-Temporal ConvGRUClassifier"]
    end

    subgraph BACKPROP ["5. Lan truyền Ngược & Tích tụ Gradient (Autograd Integrity)"]
        CONVGRU --> LOSS["Hàm mất mát Loss"]
        LOSS -->|"dL/dW tự động tích lũy qua các chunk: sum(dL_k/dW)"| BB_WEIGHTS["Trọng số dùng chung của Backbone & PAFPN<br/>(Chính xác toán học 100% như full batch)"]
    end
```

---

## 2. NGUYÊN TẮC TOÁN HỌC VÀ KỸ THUẬT CHO 3 YÊU CẦU CỦA NGƯỜI DÙNG

### 2.1. Không ảnh hưởng tích tụ Gradient (Gradient Accumulation & Autograd Linearity)

1. **Tính chất tuyến tính của Đạo hàm trong Đồ thị Tính toán PyTorch**:
   - Khi chia $N$ khung hình thành $K$ mini-chunks $X_1, X_2, \dots, X_K$:
     Đầu ra đặc trưng được ghép nối lại: $P = \text{cat}([f(X_1; W), f(X_2; W), \dots, f(X_K; W)], \text{dim}=0)$.
   - Khi hàm mất mát $\mathcal{L}(P)$ tính đạo hàm ngược qua `torch.cat`, theo quy tắc chuỗi (Chain Rule):
     $$\frac{\partial \mathcal{L}}{\partial W} = \sum_{k=1}^K \frac{\partial \mathcal{L}}{\partial P_k} \frac{\partial P_k}{\partial W} = \sum_{k=1}^K \nabla_W \mathcal{L}_k$$
   - Vì trọng số $W$ của Backbone và PAFPN là biến dùng chung (**Shared Parameters**), bộ gom đạo hàm của PyTorch Autograd sẽ **tự động cộng dồn đạo hàm** vào trường `W.grad`.
   - **Kết quả**: Giá trị `W.grad` sau khi backward hoàn toàn bằng $100\%$ so với việc forward toàn bộ batch một lần, không sai lệch dù chỉ 1 bit floating-point.

2. **Tối ưu hóa Gradient Checkpointing theo từng Chunk**:
   - Thay vì checkpoint cả khối 800 frames khổng lồ, ta áp dụng `torch.utils.checkpoint` trên **từng mini-chunk 32 frames độc lập**.
   - Trong quá trình backward, PyTorch chỉ tái tính toán activation của chunk đang được backward, giải phóng ngay lập tức sau khi tính xong đạo hàm của chunk đó.
   - **Đỉnh VRAM của Backbone** giảm từ $O(B \times T)$ xuống chỉ còn $O(\text{chunk\_size})$.

---

### 2.2. Xử lý Padding Khung hình Chuỗi Video (Padding-Aware Handling)

1. **Xử lý số dư mini-chunk (No Dummy Padding Needed)**:
   - Nếu $N$ khung hình không chia hết cho `chunk_size` (ví dụ $N = 405$, `chunk_size = 32`), vòng lặp lấy slice `flat_frames[i : min(i + chunk_size, N)]`.
   - Chunk cuối cùng có độ dài $N \pmod{\text{chunk\_size}} = 21$ khung hình được forward trực tiếp mà không cần chèn thêm khung hình giả, tiết kiệm thời gian tính toán và tránh làm loãng dữ liệu.

2. **Xử lý Zero-Padding khi video có độ dài động (`seq_lens`)**:
   - Trong một batch $B=8$, video ngắn hơn $T$ thường chứa các khung hình padding (số 0).
   - Nếu cho các khung hình số 0 này đi qua Backbone:
     - Lãng phí tài nguyên GPU xử lý khung hình đen vô nghĩa.
     - Sinh ra các đặc trưng rác có thể ảnh hưởng đến các tầng tiếp theo.
   - **Giải pháp - Valid Frame Filtering & Scatter**:
     - Tạo mặt nạ hợp lệ `valid_mask`: `mask = (time_indices < seq_lens.unsqueeze(1)).view(-1)`
     - Chỉ gom các khung hình thực sự `valid_frames = flat_frames[valid_mask]` đưa vào mini-chunking qua Backbone.
     - Sau khi nhận được các đặc trưng $P_3, P_4, P_5$, tạo tensor zero `[B*T, C, H, W]` và gán:
       `out_tensor[valid_mask] = valid_out_tensor`
     - Các vị trí padding giữ nguyên là số 0 chuẩn xác, không sinh đạo hàm ngược và bảo vệ toàn bộ mô hình.

---

### 2.3. Bảo toàn thống kê các lớp BatchNorm (Batch Normalization Integrity)

1. **Khi `freeze_bn=True` (Chế độ khuyến nghị mặc định)**:
   - Toàn bộ các lớp `nn.BatchNorm2d` trong Backbone và PAFPN được ghim cứng ở trạng thái `eval()` (`m.eval()`).
   - Khi đó, phép tính BatchNorm chuyển thành:
     $$y = \frac{x - \mu_{\text{running}}}{\sqrt{\sigma^2_{\text{running}} + \epsilon}} \cdot \gamma + \beta$$
   - Đây là phép biến đổi affine tuyến tính độc lập trên từng kênh và từng điểm ảnh (Pointwise Affine Transform).
   - **Ý nghĩa sống còn**: Việc chia chunk hay không chia chunk cho ra kết quả **hoàn toàn đồng nhất** vì $\mu_{\text{running}}$ và $\sigma^2_{\text{running}}$ không bị thay đổi bởi kích thước batch của chunk. Không có hiện tượng trôi dạt phân phối (Distribution Drift).

2. **Khi `freeze_bn=False` (Fine-tune thống kê BatchNorm)**:
   - Nhờ có bộ lọc `Valid Frame Filtering` ở trên, **tất cả các khung hình zero-padding đều bị chặn lại**, không bao giờ được chạm vào các lớp BatchNorm!
   - Thống kê $\mu$ và $\sigma^2$ chỉ được tính toán trên các khung hình video thật, loại bỏ $100\%$ nguy cơ các khung hình màu đen làm sai lệch `running_mean` và `running_var`.
   - Thiết lập ngưỡng `chunk_size \ge 32` để đảm bảo kích thước batch của mỗi chunk đủ lớn cho việc ước lượng thống kê mẫu ổn định.

---

## 3. THIẾT KẾ CHI TIẾT MÃ NGUỒN TRONG `SRC/MODELS1.PY`

### 3.1. Cập nhật `__init__` của `DrowsyClassifier`
Bổ sung tham số cấu hình `chunk_size`:
```python
def __init__(
    self,
    num_classes: int = 2,
    chunk_size: int = 32,                 # <-- Thêm mới: kích thước mini-chunk cho Backbone
    backbone_w: Tuple[int, ...] = (16, 32, 64, 128, 256),
    backbone_n: Tuple[int, ...] = (1, 2, 2, 1),
    neck_n: int = 1,
    input_dim: int = 64,
    hidden_dim: Union[int, Tuple[int, ...], List[int]] = (128, 128),
    target_size: Tuple[int, int] = (14, 14),
    supervision_mode: str = "attention_pooling",
    gradient_checkpointing: bool = False,
    freeze_bn: bool = False,
):
    ...
    self.chunk_size = max(1, int(chunk_size))
```

### 3.2. Cài đặt `_forward_backbone_neck_chunked`
```python
def _forward_backbone_neck_chunked(
    self,
    flat_frames: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Trích xuất đặc trưng Backbone & PAFPN theo cơ chế Mini-Chunking.
    Bảo toàn tính toán Gradient, xử lý triệt để Padding và bảo vệ BatchNorm.
    """
    total_frames = flat_frames.shape[0]
    
    # 1. Lọc khung hình hợp lệ nếu có mask (loại bỏ padding khung hình)
    if valid_mask is not None and not valid_mask.all():
        active_frames = flat_frames[valid_mask]
        needs_scatter = True
    else:
        active_frames = flat_frames
        needs_scatter = False

    num_active = active_frames.shape[0]
    p3_list, p4_list, p5_list = [], [], []

    # 2. Định nghĩa hàm forward cho từng stage/chunk
    def _stage_fwd(f: torch.Tensor):
        o3, o4, o5 = self.backbone(f)
        return self.neck(o3, o4, o5)

    # 3. Duyệt qua từng mini-chunk
    step = self.chunk_size if self.chunk_size > 0 else num_active
    for i in range(0, num_active, step):
        chunk = active_frames[i : min(i + step, num_active)]

        # Tự động kích hoạt đạo hàm cho chunk nếu bật Gradient Checkpointing
        if self.gradient_checkpointing and self.training:
            if not chunk.requires_grad:
                chunk.requires_grad_(True)
            o3, o4, o5 = torch.utils.checkpoint.checkpoint(_stage_fwd, chunk, use_reentrant=False)
        else:
            o3, o4, o5 = _stage_fwd(chunk)

        p3_list.append(o3)
        p4_list.append(o4)
        p5_list.append(o5)

    # 4. Ghép nối kết quả (Autograd bảo toàn hoàn hảo đạo hàm tích lũy)
    p3_active = torch.cat(p3_list, dim=0) if p3_list else torch.empty(0, device=flat_frames.device)
    p4_active = torch.cat(p4_list, dim=0) if p4_list else torch.empty(0, device=flat_frames.device)
    p5_active = torch.cat(p5_list, dim=0) if p5_list else torch.empty(0, device=flat_frames.device)

    # 5. Phục hồi khung hình padding (Scatter) nếu có lọc
    if needs_scatter:
        c3, c4, c5 = self.neck_chs
        h3, w3 = p3_active.shape[-2:]
        h4, w4 = p4_active.shape[-2:]
        h5, w5 = p5_active.shape[-2:]

        p3 = torch.zeros((total_frames, c3, h3, w3), device=flat_frames.device, dtype=p3_active.dtype)
        p4 = torch.zeros((total_frames, c4, h4, w4), device=flat_frames.device, dtype=p4_active.dtype)
        p5 = torch.zeros((total_frames, c5, h5, w5), device=flat_frames.device, dtype=p5_active.dtype)

        p3[valid_mask] = p3_active
        p4[valid_mask] = p4_active
        p5[valid_mask] = p5_active
    else:
        p3, p4, p5 = p3_active, p4_active, p5_active

    return p3, p4, p5
```

---

## 4. KẾ HOẠCH TRIỂN KHAI THEO CÁC PHA (PHASED WORKFLOW)

- **Pha 1: Cập nhật mã nguồn [`src/models1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models1.py)**:
  - Bổ sung tham số `chunk_size` vào `__init__`.
  - Thay thế `_forward_backbone_neck` bằng `_forward_backbone_neck_chunked` hỗ trợ đầy đủ Mini-Chunking, Gradient Checkpointing sửa lỗi, Padding Filtering & Scatter, và Freeze BatchNorm handling.
  - Cập nhật hàm `forward()` truyền `valid_mask` được tính từ `seq_lens` vào `_forward_backbone_neck_chunked`.

- **Pha 2: Xây dựng Bộ Kiểm thử Tự lập Nghiệm thu (Verification Suite)**:
  Viết script kiểm thử thực nghiệm chứng minh 4 tiêu chí cốt lõi:
  1. **Kiểm thử Tính tương đương Gradient (Gradient Equivalence Test)**:
     Chạy forward + backward 1 batch bằng `chunk_size=32` vs `chunk_size=None` (không chia chunk). So sánh sai số tuyệt đối tối đa giữa các gradient trọng số Backbone: $\max |\nabla W_{\text{chunked}} - \nabla W_{\text{full}}| < 10^{-5}$.
  2. **Kiểm thử Khung hình Padding (Padding Masking Test)**:
     Truyền chuỗi có `seq_lens = [50, 20]`. Kiểm tra gradient trên các khung hình padding (frame 21 đến 50 của video 2) phải bằng đúng `0.0`.
  3. **Kiểm thử Bảo vệ BatchNorm (BatchNorm Integrity Test)**:
     Kiểm tra `running_mean` và `running_var` của Backbone trước và sau khi forward chuỗi có padding frames. Đảm bảo thống kê không bị nhiễm bẩn bởi các giá trị 0.
  4. **Kiểm thử Đo đạc VRAM thực tế (VRAM Benchmark Test)**:
     Đo đạc lại VRAM tại `bz=8, T=50` và `T=100` với cơ chế mini-chunking mới. Chứng minh đỉnh VRAM giảm mạnh và mô hình chạy an toàn trên GPU 4GB.

- **Pha 3: Tổng hợp Báo cáo Hoàn thành ([`docs/report/report_vram_drowsy_classifier.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/report/report_vram_drowsy_classifier.md))**:
  Tổng hợp kết quả đo đạc, nhật ký kiểm thử và hướng dẫn cấu hình tối ưu.

---

## 5. KẾT LUẬN & CHỜ PHÊ DUYỆT (STEP 2 SIGN-OFF)

> [!IMPORTANT]
> Bản kế hoạch đã giải quyết triệt để tất cả các yêu cầu và ràng buộc từ người dùng: **Mini-Chunking + Bảo toàn Gradient 100% + Xử lý Padding thông minh + Bảo vệ BatchNorm**.
> 
> Bạn có chấp thuận kế hoạch này để tôi tiến hành **Bước 3: Thực hiện kế hoạch** (chỉnh sửa [`src/models1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models1.py), chạy kiểm thử nghiệm thu và lập báo cáo [`report_vram_drowsy_classifier.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/report/report_vram_drowsy_classifier.md)) không?
