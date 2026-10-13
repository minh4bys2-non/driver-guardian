# TÀI LIỆU ĐÁNH GIÁ MÃ NGUỒN VÀ PHÂN TÍCH TÍNH TƯƠNG THÍCH EXPORT .PTE (VULKAN BACKEND) CHO FILE EXPORT_VULKAN.PY

- **Mã yêu cầu**: `REVIEW_EXPORT_VULKAN_PTE`
- **Tệp phân tích**: `docs/analsys/analsys_export_vulkan.md`
- **Tệp đối tượng đánh giá**: `export_vulkan.py`
- **Kỹ năng áp dụng**: `agent-skills:code-review-and-quality` (Đánh giá đa chiều 5 trục: Correctness, Readability, Architecture, Security, Performance kết hợp kiểm thử thực tế ExecuTorch Vulkan lowering)
- **Mục tiêu**: Kiểm tra và đánh giá chi tiết xem hai mô hình `ExportNMSFreeDetector` và `ExportConvGRUClassifier` trong tệp `export_vulkan.py` đã phù hợp để export sang định dạng ExecuTorch `.pte` (đặc biệt nhắm tới backend Vulkan) hay chưa; chỉ ra toàn bộ lỗi chặn (blockers), rủi ro kiến trúc và đề xuất giải pháp chuẩn hóa.
- **Trạng thái**: Đang chờ người dùng phê duyệt (Bước 1 theo quy trình chuẩn `AGENTS.md`).

---

## 1. TỔNG QUAN YÊU CẦU & KẾT LUẬN SƠ BỘ (VERDICT)

### 1.1. Yêu cầu từ người dùng
> */agent-skills:code-review-and-quality kiểm tra các model ExportNMSFreeDetector và ExportConvGRUClassifier trong file @[export_vulkan.py] đã phù hợp cho quá trình export .pte chưa*

### 1.2. Kết luận tổng quan (Review Verdict)
🔴 **REQUEST CHANGES (CHƯA PHÙ HỢP - CẦN SỬA ĐỔI TOÀN DIỆN TRƯỚC KHI EXPORT)**

Cả hai lớp `ExportNMSFreeDetector` và `ExportConvGRUClassifier` trong hiện trạng **KHÔNG THỂ** export thành công ra file `.pte` chạy ổn định trên Vulkan runtime của ExecuTorch vì gặp phải các lỗi nghiêm trọng (Critical Blockers) về mặt đồ thị tính toán tĩnh (Static Graph), toán tử phụ thuộc dữ liệu (Data-dependent SymNode), định dạng đầu ra không hợp lệ (trả về kiểu nguyên thủy Python thay vì Tensor) và nhúng logic phân mảnh (chunking loop) sai tầng kiến trúc.

---

## 2. KIỂM THỬ THỰC TẾ & BẰNG CHỨNG THỰC NGHIỆM (EMPIRICAL VERIFICATION)

Trong quá trình khảo sát trực tiếp trên môi trường thực tế của dự án (`torch 2.x` + `executorch` + `VulkanPartitioner`), chúng tôi đã thực hiện chạy thử nghiệm quy trình export và ghi nhận kết quả như sau:

| Đối tượng mô hình | `torch.export.export` | `to_edge` / `to_backend(VulkanPartitioner)` | Hiện trạng |
| :--- | :---: | :---: | :--- |
| **`ExportNMSFreeDetector`** | ⚠️ Cảnh báo | ❌ Thất bại (nếu thiếu `compile_config`) | Gặp lỗi SpecViolationError do BatchNorm chưa được cấu hình exception list, đồng thời unroll vòng lặp chunking sai lệch kích thước batch. |
| **`ExportConvGRUClassifier`** | ❌ **CRASH NGAY LẬP TỨC** | ⛔ Không thể đến bước lowering | Dính lỗi `GuardOnDataDependentSymNode` do `seq_lens` kích hoạt dynamic boolean masking trong `SpatialReductionNeck`, và lỗi trả về scalar `.item()`. |

### Bằng chứng lỗi cụ thể khi chạy thực nghiệm:
1. **Lỗi `ExportConvGRUClassifier` khi gọi `torch.export.export`**:
   ```text
   torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode: 
   Could not guard on data-dependent expression 716800*u0 > 0 (unhinted: 716800*u0 > 0). (Size-like symbols: u0)
   Caused by: (ai/LSTM/src/models.py:154 in forward: if valid_fused.numel() > 0:)
   ```
2. **Lỗi trả về kiểu dữ liệu không phải Tensor**:
   ```python
   score = probs[0, 1].item()  # Trả về Python float primitive!
   ```
   ExecuTorch runtime chỉ chấp nhận đầu ra là `executorch::aten::Tensor`.
3. **Lỗi `to_edge` của `ExportNMSFreeDetector`**:
   ```text
   torch._export.verifier.SpecViolationError: 
   Operator torch._ops.aten._native_batch_norm_legit_no_training.default is not in Core ATen opset.
   ```

---

## 3. ĐÁNH GIÁ CHI TIẾT THEO 5 TRỤC CHẤT LƯỢNG (FIVE-AXIS REVIEW)

### Trục 1: Tính đúng đắn (Correctness)

#### 🔴 Critical 1.1: Trả về scalar Python (`.item()`) trong `ExportConvGRUClassifier`
- **Mã nguồn hiện tại**:
  ```python
  # Line 61-62
  score = probs[0, 1].item()
  return score
  ```
- **Vấn đề kỹ thuật**: Hàm `.item()` trích xuất giá trị số thực từ Tensor thành một biến số thực nguyên thủy của Python (`float`). 
  - Trong `torch.export` (AOTAutograd / FX Graph), việc gọi `.item()` tạo ra graph-break hoặc data-dependent guard.
  - Runtime ExecuTorch (C++ Core) chỉ quản lý bộ đệm bộ nhớ cho các `Tensor`. ExecuTorch **hoàn toàn không hỗ trợ** graph trả về Python primitive `float`.
  - Type annotation khai báo `-> torch.Tensor` nhưng thực tế lại trả về `float` (mâu thuẫn kiểu dữ liệu).
- **Khắc phục**: Trả về trực tiếp `torch.Tensor` (ví dụ `probs[:, 1]` hoặc toàn bộ `probs`).

#### 🔴 Critical 1.2: Truyền `seq_lens` gây xung đột Data-Dependent SymNode trong `ExportConvGRUClassifier`
- **Mã nguồn hiện tại**:
  ```python
  # Line 57-59
  T = x[0].shape[1]
  seq_lens = torch.tensor([T], dtype=torch.long)
  logits = self.conv_gru(x, seq_lens=seq_lens)
  ```
- **Vấn đề kỹ thuật**: 
  - Khi truyền `seq_lens`, lớp `SpatialReductionNeck` trong `src/models.py` (dòng 148-163) sẽ kích hoạt nhánh che mặt nạ động:
    ```python
    mask = (time_indices < lens_expanded).view(-1)
    valid_fused = fused[mask]  # Boolean Tensor Indexing!
    if valid_fused.numel() > 0:  # Data-dependent condition!
    ```
  - Phép đánh chỉ mục bằng mặt nạ boolean `fused[mask]` tạo ra tensor có kích thước động phụ thuộc giá trị phần tử (`u0`). Câu lệnh `if valid_fused.numel() > 0` cố gắng rẽ nhánh đồ thị theo giá trị này trong quá trình symbolic tracing, dẫn đến exception `GuardOnDataDependentSymNode` làm gãy hoàn toàn quá trình export.
  - Trên thực tế suy luận (Inference), toàn bộ chuỗi $T$ frame trong cửa sổ trượt (sliding window) đều là frame hợp lệ (không có padding). Khi đặt `seq_lens=None`, mô hình sẽ đi qua nhánh thuần túy: `reduced = self.reduction(fused)` — hoàn toàn là đồ thị tĩnh và tương thích 100% với ExecuTorch.

#### 🔴 Critical 1.3: Dùng `with torch.inference_mode():` bên trong `forward()` của `ExportNMSFreeDetector`
- **Mã nguồn hiện tại**:
  ```python
  # Line 34
  with torch.inference_mode():
      for i in range(0, T, self.chunk_size):
          ...
  ```
- **Vấn đề kỹ thuật**: Context manager (`inference_mode` / `no_grad`) không được phép nhúng bên trong hàm `forward()` của `nn.Module` khi export bằng `torch.export`. Quá trình export cần được kiểm soát trạng thái gradient từ bên ngoài bằng `with torch.no_grad():` tại điểm gọi `torch.export.export()`.

---

### Trục 2: Kiến trúc & Thiết kế phân rã (Architecture)

#### 🔴 Critical 2.1: Nhúng vòng lặp chia Chunk (`chunk_size`) vào bên trong đồ thị mô hình CNN
- **Mã nguồn hiện tại**:
  ```python
  # Line 35-47
  for i in range(0, T, self.chunk_size):
      chunk = x[i: i + self.chunk_size]
      out3, out4, out5 = self.backbone(chunk)
      out3, out4, out5 = self.neck(out3, out4, out5)
      ...
  p3 = torch.cat(p3_list, dim=0).unsqueeze(0)
  ```
- **Vấn đề kiến trúc**:
  1. **Nhầm lẫn trách nhiệm giữa Network Architecture và Inference Pipeline**:
     - Mô hình mạng thần kinh nơ-ron CNN Backbone + Neck có nhiệm vụ biến đổi hình ảnh `[B, 3, H, W]` thành các tầng đặc trưng `p3, p4, p5`.
     - Vòng lặp chia chunk, quản lý bộ nhớ đệm sliding window là trách nhiệm của **Inference Engine / Runtime Wrapper** (phía C++ hoặc Python runtime), **không phải của đồ thị mô hình**.
  2. **Gây phình to đồ thị (Graph Bloat) & Nhân bản phép tính**:
     - Nếu export với `img = (50, 3, 640, 640)` và `chunk_size = 32`, FX Graph sẽ **unroll** vòng lặp thành 2 lần gọi liên tiếp mạng Backbone và PAFPN:
       - Lần 1: batch = 32
       - Lần 2: batch = 18
     - Toàn bộ đồ thị tính toán bị nhân đôi kích thước, các tham số trọng số và cấu trúc shader Vulkan phải biên dịch 2 biến thể kích thước batch khác nhau (32 và 18).
  3. **Mất hoàn toàn tính linh hoạt khi triển khai**:
     - Nếu mô hình detector export với cấu trúc loop này, nó bị **khóa cứng (hardcoded)** với kích thước $T = 50$. Khi client muốn suy luận 1 frame đơn lẻ, hoặc chuỗi 10 frame, file `.pte` sẽ bị crash kích thước tensor.
- **Khắc phục kiến trúc**:
  - `ExportNMSFreeDetector` chỉ nên nhận vào một tensor ảnh chuẩn `x` có kích thước cố định:
    - Lựa chọn A (Khuyên dùng cho Mobile/Edge Streaming): `[1, 3, 640, 640]` (suy luận từng frame khi camera đưa vào).
    - Lựa chọn B (Khuyên dùng cho Batch): `[K, 3, 640, 640]` (với $K$ là chunk cố định, ví dụ $K=1$ hoặc $K=8$ hoặc $K=32$).

#### 🟡 Required 2.2: Chữ ký hàm đầu vào của `ExportConvGRUClassifier` (`Tuple[Tensor]` vs Multiple Args)
- **Mã nguồn hiện tại**:
  ```python
  def forward(self, x: Tuple[torch.Tensor]) -> torch.Tensor:
  ```
- **Vấn đề**: Khi export sang ExecuTorch để nạp vào C++ API (ví dụ trên Android qua JNI hoặc Linux nhúng), việc truyền một Tuple chứa 3 Tensor lồng nhau thường gây khó khăn cho việc binding dữ liệu và quản lý memory buffer.
- **Khắc phục**: Đổi thành 3 tham số tường minh:
  ```python
  def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> torch.Tensor:
  ```
  Điều này tạo sự tương thích 1:1 hoàn hảo giữa 3 đầu ra của mô hình CNN Detector và 3 đầu vào của mô hình ConvGRU Classifier.

---

### Trục 3: Khả năng đọc & Tính đơn giản (Readability & Simplicity)

- **Type Annotations sai lệch**:
  - `Tuple[torch.Tensor]` trong Python typing đại diện cho một tuple có đúng 1 phần tử là Tensor, trong khi thực tế hàm trả về 3 Tensor `(p3, p4, p5)`. Cần sửa thành `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`.
  - `-> torch.Tensor` ở `ExportConvGRUClassifier` nhưng return `float`.
- **Thiếu Docstrings & Mô tả tham số**: Cả 2 class không có docstrings giải thích mục đích, định dạng kích thước shape đầu vào/đầu ra, chuẩn hóa dải giá trị $[0, 1]$ hay $[0, 255]$.
- **Không có kiểm tra `__name__ == '__main__'`**: Toàn bộ mã tải model và tạo tensor ngẫu nhiên chạy ở global scope, vi phạm nguyên tắc đóng gói script.

---

### Trục 4: An toàn & Bảo mật (Security)

- **Đường dẫn Checkpoint tuyệt đối và Hardcoded**:
  ```python
  gru_path = r"D:\Project\DATN\driver-guardian\ai\checkpoints\model_convgru\best.pt"
  cnn_path = r"D:\Project\DATN\driver-guardian\ai\checkpoints\model_cnn\best.pt"
  ```
  - Không tương thích môi trường khác (Linux, CI/CD, máy dev khác).
  - Không có cơ chế kiểm tra `Path(p).exists()` dẫn đến lỗi `FileNotFoundError` khi checkpoint chưa sẵn sàng.
  - Cần chuyển thành tham số dòng lệnh (`argparse`) hoặc file cấu hình YAML/JSON.

---

### Trục 5: Hiệu năng & Khả năng tương thích Vulkan Backend (Performance & Lowering)

- **Tương thích toán tử với Vulkan Backend**:
  - Qua kiểm thử thực nghiệm với Vulkan backend (`VulkanPartitioner`), các toán tử cốt lõi trong `PAFPN`, `Backbone` và `ConvGRU` (như `aten.convolution`, `aten.cat`, `aten.tanh`, `aten.sigmoid`, `aten._softmax`, `aten.bmm`) **đều được Vulkan Delegate hỗ trợ thành công** nếu đồ thị được cấu hình đúng.
- **Xử lý BatchNorm khi Lowering**:
  - Khi hạ cấp xuống Edge IR (`to_edge`), `BatchNorm2d` sinh ra toán tử `aten._native_batch_norm_legit_no_training`. Toán tử này cần được khai báo trong `EdgeCompileConfig(_core_aten_ops_exception_list=[...])` để vượt qua bộ kiểm tra Core ATen IR của ExecuTorch.
- **Đóng gói file `.pte` chưa hoàn thiện**:
  - Tệp `export_vulkan.py` hiện dừng lại ở dòng khai báo class mà chưa hề chứa pipeline export (`torch.export.export` -> `to_edge` -> `to_backend` -> `to_executorch` -> `save(.pte)`).

---

## 4. BẢNG TỔNG HỢP CÁC ĐIỂM CẦN CẢI THIỆN (ACTIONABLE REMEDIES)

| Mã | Mức độ | Vấn đề | Đề xuất giải pháp (Structural Remedy) |
| :--- | :--- | :--- | :--- |
| **ISSUE-01** | 🔴 **Critical** | Trả về `score = probs[0, 1].item()` (scalar) | Trả về Tensor `probs[:, 1]` hoặc `probs`, không dùng `.item()`. |
| **ISSUE-02** | 🔴 **Critical** | Truyền `seq_lens` gây Data-Dependent SymNode crash | Bỏ `seq_lens` (đặt `seq_lens=None`) trong chế độ export inference tĩnh. |
| **ISSUE-03** | 🔴 **Critical** | `with torch.inference_mode():` bên trong forward | Loại bỏ context manager khỏi `forward()`, đặt `with torch.no_grad():` tại caller. |
| **ISSUE-04** | 🔴 **Critical** | Nhúng chunking loop `for i in range(0, T, chunk)` vào CNN | Chuẩn hóa `ExportNMSFreeDetector` nhận trực tiếp tensor ảnh `[B, 3, H, W]`. |
| **ISSUE-05** | 🟡 **Required** | Import `ai.LSTM...` gây `ModuleNotFoundError` khi chạy trực tiếp | Thêm `sys.path.insert(0, ...)` đảm bảo tìm thấy root `driver-guardian`. |
| **ISSUE-06** | 🟡 **Required** | Thiếu Exception list cho BatchNorm trong `EdgeCompileConfig` | Thêm `EdgeCompileConfig(_core_aten_ops_exception_list=[torch.ops.aten._native_batch_norm_legit_no_training.default])`. |
| **ISSUE-07** | 🟡 **Required** | Đầu vào của GRU là `Tuple[Tensor]` | Tách thành 3 tham số tường minh `(p3, p4, p5)` giúp liên kết C++ ExecuTorch dễ dàng. |
| **ISSUE-08** | 🟢 **Consider** | Hardcoded path và thiếu CLI pipeline hoàn chỉnh | Bổ sung CLI `argparse`, kiểm tra file checkpoint tồn tại, tự động lưu file `.pte`. |

---

## 5. THIẾT KẾ ĐỀ XUẤT CHO MÔ HÌNH VÀ PIPELINE EXPORT CHUẨN MỰC

### 5.1. Thiết kế lớp `ExportNMSFreeDetector` chuẩn
```python
class ExportNMSFreeDetector(nn.Module):
    """
    Wrapper xuất mô hình trích xuất đặc trưng CNN Backbone + PAFPN sang ExecuTorch.
    Đầu vào: Tensor hình ảnh [B, 3, 640, 640] (với B=1 hoặc B=chunk_size cố định).
    Đầu ra: Tuple gồm 3 bản đồ đặc trưng (P3, P4, P5).
    """
    def __init__(self, backbone: Backbone, neck: PAFPN):
        super().__init__()
        self.backbone = backbone
        self.neck = neck

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out3, out4, out5 = self.backbone(x)
        p3, p4, p5 = self.neck(out3, out4, out5)
        return p3.float(), p4.float(), p5.float()
```

### 5.2. Thiết kế lớp `ExportConvGRUClassifier` chuẩn
```python
class ExportConvGRUClassifier(nn.Module):
    """
    Wrapper xuất mô hình phân loại chuỗi không gian - thời gian ConvGRU sang ExecuTorch.
    Đầu vào: 3 tensor đặc trưng độc lập (p3, p4, p5) với shape:
      - p3: [1, T, 64, 80, 80]
      - p4: [1, T, 128, 40, 40]
      - p5: [1, T, 256, 20, 20]
    Đầu ra: Tensor xác suất [1, 2] hoặc [1] (không dùng .item()).
    """
    def __init__(self, model: ConvGRUClassifier):
        super().__init__()
        self.conv_gru = model

    def forward(
        self, 
        p3: torch.Tensor, 
        p4: torch.Tensor, 
        p5: torch.Tensor
    ) -> torch.Tensor:
        # Không truyền seq_lens để loại bỏ nhánh che mặt nạ động (Dynamic SymNode)
        logits = self.conv_gru((p3, p4, p5), seq_lens=None)
        probs = F.softmax(logits, dim=-1)
        return probs[:, 1]  # Trả về Tensor xác suất buồn ngủ [1]
```

### 5.3. Quy trình Lowering hoàn chỉnh sang `.pte` (Vulkan Backend)
```python
# 1. Cấu hình compile config bỏ qua BatchNorm Core ATen violation
compile_config = EdgeCompileConfig(_core_aten_ops_exception_list=[
    torch.ops.aten._native_batch_norm_legit_no_training.default,
])

# 2. Export & Lowering mô hình CNN
with torch.no_grad():
    ep_cnn = torch.export.export(export_cnn, (sample_img,))
edge_cnn = to_edge(ep_cnn, compile_config=compile_config)
edge_cnn = edge_cnn.to_backend(VulkanPartitioner())
pte_cnn = edge_cnn.to_executorch()
with open("detector_backbone_neck_vulkan.pte", "wb") as f:
    f.write(pte_cnn.buffer)

# 3. Export & Lowering mô hình ConvGRU
with torch.no_grad():
    ep_gru = torch.export.export(export_gru, (sample_p3, sample_p4, sample_p5))
edge_gru = to_edge(ep_gru, compile_config=compile_config)
edge_gru = edge_gru.to_backend(VulkanPartitioner())
pte_gru = edge_gru.to_executorch()
with open("convgru_classifier_vulkan.pte", "wb") as f:
    f.write(pte_gru.buffer)
```

---

## 6. KẾ HOẠCH HÀNH ĐỘNG DỰ KIẾN (KHI ĐƯỢC PHÊ DUYỆT BƯỚC 1)

Sau khi tài liệu phân tích này được người dùng chấp thuận, Agent sẽ tiếp tục:
1. **Bước 2**: Soạn thảo tài liệu kế hoạch chi tiết `docs/plan/plan_export_vulkan.md` phân rã từng đầu việc tái cấu trúc tệp `export_vulkan.py`, thiết lập các kịch bản kiểm thử tĩnh và kiểm thử biên dịch `.pte`.
2. **Bước 3**: Tiến hành cập nhật mã nguồn tệp `export_vulkan.py`, chạy thử nghiệm sinh các file `.pte` thật trên máy, kiểm tra kích thước model và tính toàn vẹn của buffer ExecuTorch, sau đó lập báo cáo kết quả `docs/report/report_export_vulkan.md`.

---
*Tài liệu được hoàn thiện và sẵn sàng để người dùng xem xét, phản hồi hoặc phê duyệt.*
