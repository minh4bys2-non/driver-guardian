# TÀI LIỆU PHÂN TÍCH ĐỐI SOÁT & ĐỒNG BỘ MÃ NGUỒN VÀO KAGGLE NOTEBOOK
## ĐỒNG BỘ THAY ĐỔI TỪ `SRC/MODELS1.PY` VÀ `CONFIGS/CONFIG.PY` VÀO `NOTEBOOKS/03_TRAIN_DROWSY_CLASSIFIER_KAGGLE.IPYNB`

- **Mã yêu cầu**: `SYNC_KAGGLE_MODELS1`
- **Tệp phân tích**: `docs/analsys/analsys_sync_kaggle_models1.md`
- **Tệp nguồn tham chiếu**: 
  - `src/models1.py` (Kiến trúc mô hình End-to-End `DrowsyClassifier` với Mini-Chunking & Padding Isolation)
  - `configs/config.py` (Cấu hình hệ thống chuẩn hóa, cập nhật `freeze_bn = True`, `chunk_size = 32`)
  - `src/train1.py` (Pipeline huấn luyện tham chiếu đã tích hợp đầy đủ `chunk_size` và `freeze_bn`)
- **Tệp đối tượng cần cập nhật**: `notebooks/03_train_drowsy_classifier_kaggle.ipynb`
- **Trạng thái**: Đã hoàn thành phân tích — Chờ người dùng xem xét và phê duyệt (Bước 1 theo quy trình chuẩn `AGENTS.MD`).

---

## 1. TỔNG QUAN YÊU CẦU & BỐI CẢNH KỸ THUẬT

### 1.1. Bối cảnh bài toán
Trong giai đoạn tối ưu hóa bộ nhớ VRAM và bảo vệ tính toàn vẹn toán học khi huấn luyện mô hình End-to-End `DrowsyClassifier` với chuỗi video có độ dài biến thiên (batch có zero-padding):
1. **Tại `src/models1.py`**:
   - Đã được tái cấu trúc triệt để nhằm tích hợp cơ chế **Mini-Chunking** (`chunk_size = 32`), giới hạn đỉnh VRAM tại $O(\text{chunk\_size})$ thay vì bùng nổ theo toàn bộ batch $\times$ sequence $O(B \times T)$.
   - Đã tích hợp cơ chế **Padding-Aware Masking & Scattering**: lọc bỏ các khung hình đệm (zero-padding) trước khi đưa vào Backbone & Neck, triệt tiêu gradient trên vùng padding và phục hồi không gian đặc trưng bằng phép gán scatter.
   - Đã kích hoạt mặc định **`freeze_bn = True`**: Khóa running statistics (`running_mean`, `running_var`) của Batch Normalization trong Backbone & Neck ngay từ lúc khởi tạo, triệt tiêu 100% lỗi Double-Update khi kết hợp với Gradient Checkpointing (`use_reentrant=False`), đồng thời phòng vệ hiện tượng phân phối suy biến khi chunk cuối chỉ có 1 khung hình.
2. **Tại `configs/config.py`**:
   - Tham số `freeze_bn` trong lớp `TrainConfig` đã được chuyển đổi từ `False` sang `True` (dòng 116) để đồng bộ với cơ chế huấn luyện ổn định mới.
   - Tham số `chunk_size = 32` (dòng 82) hiện được dùng chung để kiểm soát bộ nhớ cho cả pipeline trích xuất Backbone tách rời (`train.py`) và mô hình hợp nhất End-to-End (`models1.py` / `train1.py`).
3. **Tại `src/train1.py`**:
   - Lớp `Trainer` đã bổ sung việc đọc `chunk_size = getattr(self.cfg, "chunk_size", 32)` và `freeze_bn = getattr(self.cfg, "freeze_bn", True)`, đồng thời truyền trực tiếp vào hàm khởi tạo của `DrowsyClassifier`.

### 1.2. Hiện trạng của `notebooks/03_train_drowsy_classifier_kaggle.ipynb`
Tệp Jupyter Notebook `notebooks/03_train_drowsy_classifier_kaggle.ipynb` đóng vai trò là môi trường thực thi huấn luyện độc lập trên nền tảng Kaggle GPU (Tesla T4 / P100). Hiện tại:
- **Cell 6 (`TrainConfig`)**: Vẫn lưu giữ giá trị cũ `freeze_bn: bool = False`, và ghi chú của `chunk_size` chưa nêu rõ vai trò kiểm soát đỉnh VRAM cho `DrowsyClassifier`.
- **Cell 11 (`DrowsyClassifier`)**: Vẫn là phiên bản mã nguồn cũ:
  - Khuyết thiếu tham số `chunk_size` trong hàm `__init__`.
  - Giá trị mặc định `freeze_bn` vẫn là `False`.
  - Chưa có cơ chế tự động gọi `_freeze_backbone_neck_bn()` ngay khi khởi tạo.
  - Hàm `_freeze_backbone_neck_bn` chỉ lọc `(nn.BatchNorm2d, nn.SyncBatchNorm)` thay vì lớp cơ sở `_BatchNorm`; thiếu hàm `_unfreeze_backbone_neck_bn()`.
  - Phương thức trích xuất đặc trưng `_forward_backbone_neck` nén toàn bộ $B \times T$ frames qua Backbone cùng lúc, chưa có mini-chunking, chưa có lọc padding frames, dễ gây tràn VRAM (CUDA OOM) trên Kaggle GPU khi sequence dài.
  - Hàm `forward()` nhận `seq_lens` nhưng không sinh `valid_mask` cho tầng trích xuất đặc trưng không gian.
- **Cell 16 (`Trainer`)**:
  - Khi khởi tạo `self.model = DrowsyClassifier(...)`, hoàn toàn bỏ quên tham số `chunk_size`.
  - Giá trị dự phòng `getattr(self.cfg, "freeze_bn", False)` vẫn là `False`.

---

## 2. BẢNG PHÂN TÍCH ĐỐI SOÁT CHI TIẾT (GAP ANALYSIS)

Dưới đây là bảng đối soát chi tiết giữa các tệp nguồn (`src/models1.py`, `configs/config.py`, `src/train1.py`) và tệp notebook Kaggle (`notebooks/03_train_drowsy_classifier_kaggle.ipynb`):

| Thành phần | Vị trí trong Notebook | Hiện trạng trong Notebook | Mã nguồn chuẩn (`src/models1.py` / `configs/config.py`) | Hành động cần cập nhật |
| :--- | :--- | :--- | :--- | :--- |
| **Cấu hình `TrainConfig`** | **Cell 6** (Dòng 78) | `freeze_bn: bool = False` | `freeze_bn: bool = True` (dòng 116 `configs/config.py`) | Đổi giá trị mặc định thành `True` để triệt tiêu lỗi Double-Update BatchNorm và bảo vệ thống kê mô hình. |
| **Ghi chú `chunk_size`** | **Cell 6** (Dòng 44-45) | Chỉ ghi chú `[Tham chiếu train.py]` | Dùng chung cho cả `DrowsyClassifier` mini-chunking | Bổ sung ghi chú giải thích rõ `chunk_size: int = 32` kiểm soát kích thước mini-chunk Backbone/Neck cho `DrowsyClassifier`. |
| **Signature `DrowsyClassifier.__init__`** | **Cell 11** (Dòng 19-33) | `num_classes: int = 2, backbone_w: ... freeze_bn: bool = False` (Thiếu `chunk_size`) | `num_classes: int = 2, chunk_size: int = 32, ... freeze_bn: bool = True` | Bổ sung tham số `chunk_size: int = 32`, đổi mặc định `freeze_bn: bool = True`. |
| **Khởi tạo thuộc tính `chunk_size`** | **Cell 11** (Dòng 34-45) | Không lưu thuộc tính `self.chunk_size` | `self.chunk_size = max(0, int(chunk_size)) if chunk_size is not None else 0` | Lưu thuộc tính `self.chunk_size` có kiểm tra giá trị hợp lệ $\ge 0$. |
| **Tự động khóa BatchNorm** | **Cell 11** (Dòng 45-60) | Chỉ khóa BN khi người dùng gọi `model.train()` sau này | `if self.freeze_bn: self._freeze_backbone_neck_bn()` ngay trong `__init__` | Gọi tự động ngay sau khi khởi tạo xong `self.backbone` và `self.neck`. |
| **Chuẩn hóa khóa/mở BatchNorm** | **Cell 11** (Dòng 68-75) | Kiểm tra `(nn.BatchNorm2d, nn.SyncBatchNorm)`, thiếu hàm mở khóa | Dùng `from torch.nn.modules.batchnorm import _BatchNorm` và có thêm `_unfreeze_backbone_neck_bn()` | Cập nhật kiểm tra bao quát `_BatchNorm` và bổ sung phương thức `_unfreeze_backbone_neck_bn()`. |
| **Thuật toán trích xuất đặc trưng** | **Cell 11** (Dòng 77-87) | `_forward_backbone_neck(self, flat_frames)` cũ: chạy toàn bộ frames cùng lúc | `_forward_backbone_neck_chunked(self, flat_frames, valid_mask)`: Chia mini-chunk + Lọc padding + Scatter phục hồi | Thay thế hoàn toàn bằng thuật toán Mini-Chunking tiên tiến có lọc mặt nạ padding và gom nhóm tensor. |
| **Hàm bọc tương thích ngược** | **Cell 11** | Không có | `_forward_backbone_neck` bọc gọi sang `_forward_backbone_neck_chunked` | Bổ sung hàm bọc để duy trì 100% tính tương thích ngược với code cũ nếu có. |
| **Mặt nạ Padding trong `forward`** | **Cell 11** (Dòng 89-130) | Chỉ reshape `B * T` và gọi thẳng `_forward_backbone_neck` | Tự động sinh `valid_mask = (time_indices < lens_expanded).view(-1)` từ `seq_lens` và truyền vào Backbone/Neck | Tích hợp logic sinh vector mặt nạ thời gian `valid_mask` boolean từ `seq_lens` trước khi trích xuất. |
| **Khởi tạo mô hình trong `Trainer`** | **Cell 16** (Dòng 101-115) | `freeze_bn = getattr(self.cfg, "freeze_bn", False)` và không truyền `chunk_size` | Đọc `chunk_size = getattr(self.cfg, "chunk_size", 32)`, `freeze_bn = getattr(..., True)` và truyền vào `DrowsyClassifier` | Đồng bộ khởi tạo mô hình trong `Trainer` khớp với `src/train1.py`. |
| **Kiểm thử nghiệm thu tại chỗ (Sanity Check)** | **Cell 11 / Cell 18** | Chưa có cell kiểm tra trực tiếp khả năng chịu tải VRAM và tính đúng đắn của Mini-Chunking | Bộ self-verification suite phong phú trong `src/models1.py` | Bổ sung đoạn code kiểm tra nhanh sau Cell 11 để người dùng verify ngay trên Kaggle trước khi train. |

---

## 3. PHÂN TÍCH KỸ THUẬT CHI TIẾT CÁC THAY ĐỔI CỐT LÕI

### 3.1. Cơ chế Mini-Chunking kết hợp Padding-Aware Filtering
Trong notebook cũ, khi batch có kích thước $B = 2$ và video dài $T = 80$ frames, toàn bộ $160$ khung hình được đẩy đồng thời qua 5 tầng của Backbone và 4 tầng của PAFPN:
$$\text{Peak Activation Memory} \propto B \times T_{\max} \times \sum_{l=1}^L (C_l \times H_l \times W_l)$$
Điều này khiến đỉnh bộ nhớ VRAM vọt lên hơn 10 GB ngay tại forward pass của Backbone, dễ gây lỗi `CUDA out of memory` trên GPU Tesla T4 (16GB) khi bật gradient checkpointing và optimizer AdamW.

Với thuật toán mới trong `src/models1.py`:
1. **Lọc khung hình đệm (Valid Masking)**:
   ```python
   time_indices = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
   lens_expanded = seq_lens.to(x.device).unsqueeze(1)            # [B, 1]
   valid_mask = (time_indices < lens_expanded).view(-1)          # [B * T]
   active_frames = flat_frames[valid_mask]
   ```
   Nếu video 1 dài 80 frames, video 2 dài 30 frames ($T_{\max} = 80$), thay vì tính toán 160 frames, mô hình chỉ tính toán đúng $80 + 30 = 110$ frames hợp lệ. Khung hình padding được bỏ qua hoàn toàn.
2. **Chia Mini-Chunk (`chunk_size = 32`)**:
   Số lượng frames hợp lệ được chia thành các lát cắt nhỏ có kích thước tối đa 32 frames. Tại mỗi thời điểm, bộ nhớ activation của Backbone/PAFPN chỉ lưu trữ cho 32 frames.
3. **Phục hồi Scatter**:
   Các feature maps hợp lệ được gán ngược lại vào tensor kích thước `[total_frames, C, H, W]` tại đúng các vị trí `valid_mask`, các vị trí padding được đảm bảo giữ nguyên giá trị 0.0 tuyệt đối.

### 3.2. Triệt tiêu lỗi Double-Update của BatchNorm khi Checkpoint
Khi `gradient_checkpointing = True` trong PyTorch:
- **Ở Forward pass**: PyTorch chạy module và tính toán output bình thường (nếu BatchNorm ở `train()`, `running_mean` và `running_var` được cập nhật lần 1).
- **Ở Backward pass**: PyTorch tái thực thi forward pass (Recomputation) từ tensor đầu vào đã lưu để lấy activation tính đạo hàm (nếu BatchNorm vẫn ở `train()`, `running_mean` và `running_var` lại được cập nhật lần 2!).
- Đồng thời, khi chia mini-chunk, chunk cuối cùng có thể chỉ có 1 hoặc 2 frames. Nếu BatchNorm ở `train()`, thống kê trên 1 frame sẽ làm lệch hoàn toàn phân phối của mô hình.

Việc thiết lập mặc định `freeze_bn = True` (và tự động khóa `m.eval()` cho toàn bộ các khối BatchNorm trong Backbone và PAFPN) đảm bảo:
- Triệt tiêu 100% lỗi Double-Update thống kê.
- BatchNorm hoạt động như một phép biến đổi tuyến tính affine cố định ($y = \gamma \hat{x} + \beta$), các trọng số $\gamma, \beta$ vẫn nhận gradient và được học tối ưu bình thường thông qua AdamW.

---

## 4. KẾ HOẠCH CẬP NHẬT DỰ KIẾN (BƯỚC TIẾP THEO)

Sau khi tài liệu phân tích này được người dùng xem xét và phê duyệt:
1. **Bước 2: Lập kế hoạch thực hiện (`docs/plan/plan_sync_kaggle_models1.md`)**:
   - Xác định danh sách các cell chính xác cần chỉnh sửa trong `notebooks/03_train_drowsy_classifier_kaggle.ipynb`.
   - Thiết kế mã nguồn chi tiết cho Cell 6, Cell 11, Cell 16.
   - Thiết kế kịch bản kiểm thử (Verification test script) để nghiệm thu notebook sau khi cập nhật (chạy không lỗi cú pháp, nạp cấu hình đúng, khởi tạo mô hình thành công, chuyển đổi tensor khớp).
2. **Bước 3: Thực hiện cập nhật & Nghiệm thu (`docs/report/report_sync_kaggle_models1.md`)**:
   - Sử dụng script an toàn chỉnh sửa nội dung JSON của file notebook `.ipynb`.
   - Chạy test kiểm tra toàn bộ notebook.
   - Xuất tài liệu báo cáo nghiệm thu hoàn tất nhiệm vụ.

---

## 5. KẾT LUẬN & ĐỀ XUẤT

Phân tích cho thấy sự cần thiết cấp bách của việc đồng bộ các cải tiến từ `src/models1.py` và `configs/config.py` vào `notebooks/03_train_drowsy_classifier_kaggle.ipynb`. Việc này sẽ giúp người dùng chạy huấn luyện trên Kaggle một cách trơn tru, loại bỏ hoàn toàn các lỗi tiềm ẩn về VRAM OOM và sai lệch phân phối BatchNorm.

> [!IMPORTANT]
> **HÀNH ĐỘNG TIẾP THEO**: Theo quy chuẩn `AGENTS.MD`, tài liệu phân tích này đã sẵn sàng. Kính mời người dùng xem xét và phê duyệt để Agent tiến hành lập kế hoạch chi tiết (Bước 2: `plan_sync_kaggle_models1.md`).
