# TÀI LIỆU PHÂN TÍCH YÊU CẦU: TRÍCH XUẤT CHECKPOINT VÀ TÍCH HỢP PHƯƠNG THỨC `FROM_CHECKPOINT` CHO MODEL BACKBONENECK

- **Mã yêu cầu**: `EXTRACT_BACKBONE_NECK`
- **Tệp phân tích**: `docs/analsys/analsys_extract_backbone_neck.md`
- **Tệp nguồn tham chiếu**:
  - `src/model_backbone_neck.py` (Mã nguồn hiện tại của lớp `BackboneNeck` và script trích xuất thô sơ đang bị chú thích)
  - `ai/NewVersionObDetect/checkpoints/.../landmark_finetune/best.pt` (Checkpoint gốc chứa đầy đủ Backbone, Neck và Landmark Head)
  - `ai/ObjectDetection_2p6M/src/backbone_neck.py` (Định nghĩa module cốt lõi `Backbone` và `PAFPN`)
  - `src/models1.py` & `src/models.py` (Tham chiếu triển khai phương thức `from_checkpoint` chuẩn hóa)
- **Trạng thái**: Đã hoàn thành khảo sát & phân tích — Chờ phê duyệt của người dùng (Bước 1 theo quy trình `AGENTS.md`).

---

## 1. TỔNG QUAN YÊU CẦU & BỐI CẢNH KỸ THUẬT

### 1.1. Bối cảnh bài toán
Trong pipeline cảnh báo buồn ngủ tài xế (`driver-guardian/ai/LSTM`), tầng trích xuất đặc trưng không gian (Spatial Feature Extractor) chỉ cần hai khối cốt lõi:
1. **`Backbone`**: Mạng tích chập trích xuất đặc trưng đa tầng từ ảnh khuôn mặt đầu vào $[B, 3, H, W]$.
2. **`Neck` (`PAFPN`)**: Khối tổng hợp đặc trưng kim tự tháp đa tỷ lệ (Path Aggregation Feature Pyramid Network) sinh ra 3 bản đồ đặc trưng:
   - $P_3$: $[B, 64, H/8, W/8]$ (stride 8)
   - $P_4$: $[B, 128, H/16, W/16]$ (stride 16)
   - $P_5$: $[B, 256, H/32, W/32]$ (stride 32)

Toàn bộ các tham số của tầng phát hiện mốc khuôn mặt (**`Head`** gồm các nhánh Anchor, Bounding Box, Landmark Cls/Reg) cùng với trạng thái tối ưu hóa (`optimizer`, `scaler`, `training_plan`) **hoàn toàn không được sử dụng** trong quá trình huấn luyện downstream hoặc inference của bộ phân loại `ConvGRU` / `DrowsyClassifier`.

### 1.2. Hiện trạng tệp `src/model_backbone_neck.py`
Khảo sát tệp `src/model_backbone_neck.py` cho thấy:
1. **Lớp `BackboneNeck`**:
   ```python
   class BackboneNeck(nn.Module):
       def __init__(self, backbone: nn.Module, neck: nn.Module):
           super().__init__()
           self.backbone, self.neck = backbone, neck

       def forward(self, x):
           return self.neck(*self.backbone(x))
   ```
   - **Hạn chế 1**: Bắt buộc người dùng phải tự khởi tạo `backbone` và `neck` từ bên ngoài rồi truyền vào. Không có khả năng tự khởi tạo độc lập từ các cấu hình kiến trúc mặc định (`backbone_w`, `backbone_n`, `neck_n`).
   - **Hạn chế 2**: **Chưa có phương thức `from_checkpoint`**. Khi cần nạp mô hình trong các module khác hoặc khi làm inference/export, mã nguồn phải phụ thuộc vào việc dựng toàn bộ detector phức tạp hoặc phải viết thủ công logic nạp weights.
2. **Hàm `main()` và quy trình trích xuất**:
   ```python
   def main():
       weights_path = r"D:\Project\DATN\driver-guardian\ai\NewVersionObDetect\checkpoints\231e35bb4061257f9bcb7cc5e3a0d0064e5e1beb6351a5e0adfc8d0c91ba4e11\da92589d1b67339be4491beaf147575e81a2699bbde06d654c05f852d7f0e9c9\landmark_finetune\best.pt"
       cnn_filepath = r"D:\Project\DATN\driver-guardian\ai\checkpoints\model_cnn\best1.pt"
       train_cfg = config_from_checkpoint(read_checkpoint(weights_path))
       checkpoint = read_checkpoint(weights_path)
       print(checkpoint["metadata"])
       # model_lmk = FaceLmkDetector(train_cfg)
       # checkpoint = read_checkpoint(weights_path, train_cfg)
       # model_lmk.load_state_dict(checkpoint.get('ema') or checkpoint['model'], strict=True)
       # model_backbone_neck = BackboneNeck(model_lmk.backbone, model_lmk.neck)
       # checkpoint = {
       #     "model": model_backbone_neck.state_dict(),
       # }
       # torch.save(checkpoint, cnn_filepath)
   ```
   - Các dòng code thực hiện trích xuất và lưu trọng số đang bị comment.
   - Code cũ phụ thuộc vào `FaceLmkDetector`, `transfer.checkpoint.read_checkpoint` (yêu cầu metadata strict SHA256 và các tệp JSON phụ trợ đi kèm).

---

## 2. PHÂN TÍCH ĐỐI SOÁT CHECKPOINT & THÔNG SỐ KỸ THUẬT

### 2.1. Khảo sát checkpoint nguồn (`weights_path`)
Kiểm tra thực tế checkpoint tại `weights_path`:
- **Đường dẫn**:
  `D:\Project\DATN\driver-guardian\ai\NewVersionObDetect\checkpoints\231e35bb4061257f9bcb7cc5e3a0d0064e5e1beb6351a5e0adfc8d0c91ba4e11\da92589d1b67339be4491beaf147575e81a2699bbde06d654c05f852d7f0e9c9\landmark_finetune\best.pt`
- **Dung lượng tệp hiện tại**: **65.87 MB** (65,867,513 bytes).
- **Cấu trúc dữ liệu trong checkpoint**:
  - `model`: 805 tensors chứa toàn bộ trọng số của Backbone, Neck, Head.
  - `ema`: 805 tensors chứa trọng số Exponential Moving Average tối ưu nhất.
  - `optimizer`, `scaler`: trạng thái bộ nhớ đạo hàm bậc một, bậc hai của AdamW.
  - `cfg`, `metadata`, `training_plan`, `rng`, `epoch`, `global_step`, `best_val`.

### 2.2. Khảo sát tham số của các phân vùng
| Phân vùng | Tiền tố khóa (Key prefix) | Số lượng Tensor | Ý nghĩa chức năng | Cần trích xuất? |
| :--- | :--- | :--- | :--- | :--- |
| **Backbone** | `backbone.*` | **294** tensors | Trích xuất đặc trưng không gian từ ảnh thô | **CÓ** (Bắt buộc) |
| **Neck (PAFPN)** | `neck.*` | **150** tensors | Tổng hợp đặc trưng kim tự tháp đa tầng P3, P4, P5 | **CÓ** (Bắt buộc) |
| **Head (Detection & Lmk)** | `head.*` | **361** tensors | Dự đoán bounding box và 478 điểm mốc khuôn mặt | **KHÔNG** (Dư thừa) |
| **Optimizer & Scaler** | `optimizer`, `scaler` | Hàng trăm states | Bộ nhớ Gradient cho huấn luyện | **KHÔNG** (Dư thừa) |

### 2.3. Hiệu quả tối ưu hóa khi trích xuất
- **Tổng số tensor của BackboneNeck**: $294 + 150 = \mathbf{444}$ tensors.
- **Tổng số lượng tham số (Parameters)**: **1,654,841** params.
- **Kích thước trọng số thô (Raw FP32 size)**: **~6.31 MB**.
- **Kích thước file lưu trữ dự kiến (.pt)**: **~6.5 MB** (so với 65.87 MB ban đầu).
- **Mức tiết kiệm dung lượng đĩa & bộ nhớ nạp**: Giảm hơn **83%** kích thước checkpoint, nạp cực nhanh và không kéo theo các tensor không dùng.

---

## 3. THIẾT KẾ GIẢI PHÁP KỸ THUẬT CHI TIẾT

### 3.1. Thiết kế lại lớp `BackboneNeck` trong `src/model_backbone_neck.py`

#### A. Hàm khởi tạo `__init__` linh hoạt
Hỗ trợ cả 2 chế độ:
1. **Chế độ truyền module có sẵn** (tương thích ngược):
   `BackboneNeck(backbone=existing_backbone, neck=existing_neck)`
2. **Chế độ tự dựng từ thông số kiến trúc** (độc lập, thuận tiện cho `from_checkpoint`):
   `BackboneNeck(backbone_w=(16, 32, 64, 128, 256), backbone_n=(1, 2, 2, 1), neck_n=1)`
   - Tự động nhập `Backbone` và `PAFPN` từ `ai.ObjectDetection_2p6M.src.backbone_neck`.
   - Lưu trữ thuộc tính cấu hình `backbone_w`, `backbone_n`, `neck_n`, và `neck_chs = (w[2], w[3], w[4]) = (64, 128, 256)`.

#### B. Phương thức `forward`
```python
def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Nhận: x [B, 3, H, W]
    Trả về: (p3, p4, p5)
      - p3: [B, 64, H/8, W/8]
      - p4: [B, 128, H/16, W/16]
      - p5: [B, 256, H/32, W/32]
    """
    return self.neck(*self.backbone(x))
```

#### C. Phương thức lớp `from_checkpoint`
Được thiết kế theo tiêu chuẩn công nghiệp (khớp với `models.py` và `NMSFreeDetector`):
```python
@classmethod
def from_checkpoint(
    cls,
    checkpoint_path: Union[str, Path, Dict[str, Any]],
    map_location: Optional[Union[str, torch.device]] = "cpu",
    eval_mode: bool = True,
    strict: bool = True,
    **kwargs,
) -> "BackboneNeck":
```
**Cơ chế hoạt động thông minh**:
1. **Nhận diện nguồn checkpoint**:
   - Nếu là đường dẫn (`str` hoặc `Path`): kiểm tra tồn tại và gọi `torch.load(path, map_location=map_location, weights_only=False)`.
   - Nếu là `dict` có sẵn trong bộ nhớ: sử dụng trực tiếp.
2. **Tự động nhận diện cấu hình kiến trúc**:
   - Ưu tiên 1: Lấy từ `kwargs` nếu người dùng truyền chỉ định.
   - Ưu tiên 2: Đọc từ trường `backbone_w`, `backbone_n`, `neck_n` lưu trong checkpoint trích xuất.
   - Ưu tiên 3: Đọc từ `cfg` / `metadata.architecture` trong checkpoint đầy đủ của detector.
   - Fallback: Dùng cấu hình chuẩn của hệ thống `backbone_w=(16, 32, 64, 128, 256)`, `backbone_n=(1, 2, 2, 1)`, `neck_n=1`.
3. **Trích xuất state_dict tương thích kép**:
   - Tìm kiếm theo thứ tự ưu tiên: `checkpoint.get("ema")` -> `checkpoint.get("model")` -> `checkpoint.get("model_state_dict")` -> `checkpoint`.
   - Lọc chỉ lấy các keys thuộc về Backbone và Neck (`backbone.*` và `neck.*`). Nhờ đó, phương thức có thể nạp thành công từ **cả checkpoint trích xuất riêng** lẫn **checkpoint gốc full detector**!
4. **Tải trọng số & Chuyển đổi trạng thái**:
   - `model.load_state_dict(filtered_state_dict, strict=strict)`.
   - Chuyển `model.to(map_location)`.
   - Kích hoạt `model.eval()` nếu `eval_mode=True`.

### 3.2. Thiết kế hàm trích xuất Checkpoint `extract_backbone_neck_checkpoint`
Hàm trích xuất độc lập, an toàn và tối ưu:
```python
def extract_backbone_neck_checkpoint(
    source_ckpt_path: Union[str, Path],
    output_ckpt_path: Union[str, Path],
    use_ema: bool = True,
) -> Path:
```
- **Quy trình**:
  1. Đọc checkpoint nguồn bằng `torch.load(..., map_location="cpu")`.
  2. Lấy trọng số từ `ema` (nếu `use_ema=True` và có `ema`) hoặc `model`.
  3. Lọc chính xác 444 tensors có tiền tố `backbone.` hoặc `neck.`.
  4. Đóng gói checkpoint mới có cấu trúc chuẩn:
     ```python
     extracted_checkpoint = {
         "model": backbone_neck_state_dict,
         "backbone_w": backbone_w,
         "backbone_n": backbone_n,
         "neck_n": neck_n,
         "metadata": {
             "task": "backbone_neck_feature_extraction",
             "num_tensors": len(backbone_neck_state_dict),
             "total_params": sum(p.numel() for p in backbone_neck_state_dict.values()),
             "source_checkpoint": str(source_ckpt_path),
             "output_features": ["p3", "p4", "p5"],
             "neck_channels": [64, 128, 256],
         }
     }
     ```
  5. Tạo thư mục cha nếu chưa có (`output_path.parent.mkdir(parents=True, exist_ok=True)`).
  6. Lưu bằng `torch.save(extracted_checkpoint, output_ckpt_path)`.
  7. Tự động kiểm tra tính hợp lệ bằng cách gọi `BackboneNeck.from_checkpoint(output_ckpt_path)` và chạy thử qua một batch ảnh dummy $[1, 3, 640, 640]$.

### 3.3. Đường dẫn Checkpoint đích
Trong `src/model_backbone_neck.py`, đường dẫn đích được chỉ định sẵn:
`cnn_filepath = r"D:\Project\DATN\driver-guardian\ai\checkpoints\model_cnn\best1.pt"`
- Chúng ta sẽ thực hiện trích xuất lưu vào đường dẫn `best1.pt` theo đúng biến cấu hình của tệp.
- Đồng thời cung cấp CLI / cờ để người dùng có thể lưu vào bất kỳ đường dẫn nào khác nếu muốn (ví dụ `best.pt`).

---

## 4. KẾ HOẠCH BƯỚC TIẾP THEO

Sau khi tài liệu phân tích này được người dùng xem xét và phê duyệt:
1. **Bước 2**: Lập tệp kế hoạch chi tiết `docs/plan/plan_extract_backbone_neck.md`.
2. **Bước 3**: Cập nhật mã nguồn `src/model_backbone_neck.py`, thực thi trích xuất checkpoint, kiểm thử nghiệm thu toàn diện và tạo tệp báo cáo `docs/report/report_extract_backbone_neck.md`.
