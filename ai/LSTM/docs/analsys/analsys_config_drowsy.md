# TÀI LIỆU PHÂN TÍCH YÊU CẦU: CẬP NHẬT THAM SỐ CẤU HÌNH CHO MODEL DROWSYCLASSIFIER (CONFIG.PY & CONFIG.YAML)

- **Mã yêu cầu**: `CONFIG_DROWSY_CLASSIFIER`
- **Tệp phân tích**: `docs/analsys/analsys_config_drowsy.md`
- **Mã nguồn tham chiếu**:
  - [`configs/config.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.py) (Cấu hình lớp Dataclass `TrainConfig` và hàm nạp `load_config`)
  - [`configs/config.yaml`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.yaml) (Tệp cấu hình YAML khai báo tham số thực tế)
  - [`src/models1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models1.py) (Định nghĩa kiến trúc mô hình End-to-End `DrowsyClassifier`)
  - [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py) (Pipeline huấn luyện End-to-End `DrowsyClassifier`)
  - [`src/models.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py) (Mô hình cốt lõi `ConvGRUClassifier`)
  - [`src/train.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train.py) (Pipeline huấn luyện trích xuất 2 giai đoạn với pretrained BackboneNeck)
- **Quy chuẩn áp dụng**: [`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md) — **Bước 1: Khảo sát & Phân tích (Discovery)**
- **Trạng thái**: Đang chờ người dùng phê duyệt để chuyển sang Bước 2 (Lên kế hoạch).

---

## 1. TỔNG QUAN YÊU CẦU & MỤC TIÊU CỐT LÕI

### 1.1. Yêu cầu từ người dùng
> *"cập nhật @[configs/config.py] và @[configs/config.yaml] thêm các tham số ở model DrowsyClassifier @[src/models1.py] @[src/train1.py] tham số nào dùng chung thì nào commet là dùng chung"*

### 1.2. Mục tiêu kỹ thuật
1. **Bổ sung đầy đủ các tham số cấu hình của `DrowsyClassifier`** từ [`src/models1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models1.py) và [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py) vào cả hai tệp:
   - [`configs/config.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.py) (Dataclass `TrainConfig`, phương thức xác thực `__post_init__`, bộ giải mã `load_yaml` / `load_json`).
   - [`configs/config.yaml`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.yaml) (Cấu hình phân nhóm rõ ràng).
2. **Gắn chú thích (comment) minh bạch**:
   - Mọi tham số được sử dụng đồng thời bởi cả hai mô hình/luồng huấn luyện (`ConvGRUClassifier` / `train.py` và `DrowsyClassifier` / `train1.py`) phải được chú thích rõ ràng là **`[DÙNG CHUNG]`**.
   - Các tham số đặc thù chỉ dùng riêng cho từng mô hình/luồng huấn luyện phải được phân tách và ghi chú rõ ràng (`[DÀNH RIÊNG DrowsyClassifier]` hoặc `[DÀNH RIÊNG ConvGRUClassifier]`).
3. **Đảm bảo tính tương thích ngược và an toàn kiểu dữ liệu (Type Safety)**:
   - Xử lý chuyển đổi Tuple cho các trường dạng danh sách (`backbone_w`, `backbone_n`, `target_size`) khi nạp từ YAML hoặc JSON.
   - Bổ sung ràng buộc xác thực dữ liệu (Assertions / Type Guard) trong `TrainConfig.__post_init__`.

---

## 2. KHẢO SÁT & ĐỐI CHIẾU THAM SỐ GIỮA CÁC MÔ HÌNH VÀ PIPELINE

### 2.1. Tham số trong khởi tạo `DrowsyClassifier` ([`src/models1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models1.py))
```python
class DrowsyClassifier(nn.Module):
    def __init__(
            self,
            num_classes: int = 2,
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
```

### 2.2. Tham số được đọc trong `Trainer` ([`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py))
```python
target_size = tuple(self.cfg.target_size) if hasattr(self.cfg, "target_size") and self.cfg.target_size else (max(1, h_in // 16), max(1, w_in // 16))
num_classes = getattr(self.cfg, "num_classes", 2)
input_dim = getattr(self.cfg, "convgru_in_dim", getattr(self.cfg, "input_dim", 64))
hidden_dim = getattr(self.cfg, "convgru_hidden_dim", getattr(self.cfg, "hidden_dim", 64))
gradient_checkpointing = getattr(self.cfg, "gradient_checkpointing", True)
freeze_bn = getattr(self.cfg, "freeze_bn", False)
```
*Lưu ý cải tiến đi kèm*: Để `src/train1.py` tiếp nhận trọn vẹn cấu hình từ file config, ta sẽ cập nhật `train1.py` truyền thêm `backbone_w`, `backbone_n`, `neck_n` được đọc từ `self.cfg`.

### 2.3. Bảng phân loại chi tiết các tham số cấu hình

| Nhóm tham số | Tên tham số | Kiểu dữ liệu | Giá trị mặc định | Phân loại | Mục đích & Giải thích kỹ thuật |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model Architecture** | `num_classes` | `int` | `2` | **`[DÙNG CHUNG]`** | Số lượng lớp đầu ra phân loại (0: Tỉnh táo, 1: Buồn ngủ). Cả 2 mô hình đều dùng. |
| | `input_dim` / `convgru_in_dim` | `int` | `128` | **`[DÙNG CHUNG]`** | Số kênh nén không gian đầu vào đưa vào ConvGRU sau `SpatialReductionNeck`. |
| | `hidden_dim` / `convgru_hidden_dim` | `Union[int, Tuple[int, ...]]` | `128` | **`[DÙNG CHUNG]`** | Số kênh ẩn trong các tầng ConvGRU. |
| | `num_layers` / `convgru_num_layers` | `int` | `2` | **`[DÙNG CHUNG]`** | Số tầng ConvGRU xếp chồng. |
| | `dropout` | `float` | `0.35` | **`[DÙNG CHUNG]`** | Tỷ lệ Dropout cho đầu phân loại FC Head. |
| | `supervision_mode` | `str` | `"attention_pooling"` | **`[DÙNG CHUNG]`** | Cơ chế gom tụ và giám sát chuỗi thời gian (`'attention_pooling'` hoặc `'sequence'`). |
| | `target_size` | `Tuple[int, int]` | `(14, 14)` | **`[DÙNG CHUNG]`** | Kích thước lưới không gian hội tụ `(H_feat, W_feat)` sau `SpatialReductionNeck`. Dùng chung cho cả `ConvGRUClassifier` và `DrowsyClassifier`. |
| | `backbone_w` | `Tuple[int, ...]` | `(16, 32, 64, 128, 256)` | **Dành riêng DrowsyClassifier** | Độ rộng kênh các tầng Backbone (c1..c5) khi khởi tạo Backbone từ đầu. |
| | `backbone_n` | `Tuple[int, ...]` | `(1, 2, 2, 1)` | **Dành riêng DrowsyClassifier** | Số lượng khối lặp lại trong từng tầng Backbone. |
| | `neck_n` | `int` | `1` | **Dành riêng DrowsyClassifier** | Số lượng khối lặp lại trong PAFPN của DrowsyClassifier. |
| | `gradient_checkpointing` | `bool` | `True` | **Dành riêng DrowsyClassifier** | Bật Gradient Checkpointing cho Backbone/PAFPN để tiết kiệm VRAM khi huấn luyện End-to-End. |
| | `freeze_bn` | `bool` | `False` | **Dành riêng DrowsyClassifier** | Khóa thống kê running mean/var của BatchNorm trong Backbone/Neck khi fine-tune. |
| | `cnn_neck_channels` | `Tuple[int, ...]` | `(64, 128, 256)` | **Dành riêng ConvGRUClassifier** | Kênh đặc trưng trích xuất từ Backbone tách rời đưa vào `SpatialReductionNeck` trong `src/models.py`. |
| | `adapter_dropout` | `float` | `0.25` | **Dành riêng ConvGRUClassifier** | Tỷ lệ Dropout sau cổ giảm kênh `SpatialReductionNeck` trong `src/models.py`. |
| **Dataset** | `dataset_dir`, `manifest_file`, `sample_interval`, `seq_len`, `image_size`, `video_exts`, `min_frames`, `use_augmentation` | Đa dạng | Theo file config | **`[DÙNG CHUNG]`** | Cấu hình nạp video thô, lấy mẫu khung hình, tăng cường dữ liệu và lọc clip hợp lệ. |
| | `backbone_neck_checkpoint` | `str` | Đường dẫn `.pt` | **Dành riêng train.py** | Checkpoint Backbone/PAFPN 2.6M được trích xuất offline trong `train.py`. |
| **DataLoader** | `batch_size`, `val_batch_size`, `num_workers`, `val_num_workers`, `pin_memory`, `shuffle`, `seed` | Đa dạng | Theo file config | **`[DÙNG CHUNG]`** | Cấu hình DataLoader cho tập huấn luyện và kiểm định. |
| | `chunk_size` | `int` | `32` | **Dành riêng train.py** | Kích thước chia nhỏ batch khi forward qua Backbone trích xuất rời chống tràn VRAM. |
| **Loss & Criterion** | `loss_type`, `bce_eps`, `bce_reduction`, `pos_weight` | Đa dạng | Theo file config | **`[DÙNG CHUNG]`** | Cấu hình hàm mất mát (CrossEntropyLoss hoặc BCEWithLogitsLoss). |
| **Optimizer & Scheduler** | `epochs`, `lr0`, `lr_min_factor`, `weight_decay`, `optimizer`, `betas`, `momentum`, `grad_clip_norm`, `gradient_accumulation_steps`, `empty_cache_interval`, `use_scheduler` | Đa dạng | Theo file config | **`[DÙNG CHUNG]`** | Cấu hình tối ưu hóa, tích lũy gradient, và lịch trình học (CosineAnnealingLR). |
| **Logging & Checkpoint** | `tb_log_dir`, `experiment_name`, `checkpoint_dir`, `save_ckpt_interval_epochs`, `save_all_epochs`, `ckpt_keep_last`, `enable_resume`, `resume`, `enable_step_logging`, `log_step_interval`, `early_stopping`, `patience`, `monitor_metric`, `monitor_mode`, `min_delta`, `history_csv_path` | Đa dạng | Theo file config | **`[DÙNG CHUNG]`** | Cấu hình ghi log TensorBoard, lưu trữ checkpoint nguyên tử (Atomic Save) và Early Stopping. |
| **Runtime & Hardware** | `device`, `amp`, `val_interval_epochs`, `use_tqdm` | Đa dạng | Theo file config | **`[DÙNG CHUNG]`** | Cấu hình thiết bị phần cứng GPU/CPU, tự động ép kiểu hỗn hợp FP16 (AMP). |

---

## 3. CHI TIẾT CÁC THAY ĐỔI CẦN THỰC HIỆN

### 3.1. Cập nhật `configs/config.py`
1. **Bổ sung các trường mới vào Dataclass `TrainConfig`**:
   - `backbone_w: Tuple[int, ...] = (16, 32, 64, 128, 256)`
   - `backbone_n: Tuple[int, ...] = (1, 2, 2, 1)`
   - `neck_n: int = 1`
   - `target_size: Tuple[int, int] = (14, 14)`
   - `gradient_checkpointing: bool = True`
   - `freeze_bn: bool = False`
2. **Cập nhật comment đồng bộ**: Gắn nhãn `[DÙNG CHUNG]` cho tất cả các tham số dùng chung; ghi rõ `[DÀNH RIÊNG DrowsyClassifier]` và `[DÀNH RIÊNG ConvGRUClassifier]`.
3. **Cập nhật `__post_init__`**:
   - Ràng buộc kích thước `target_size`: phải là tuple 2 phần tử dương `> 0`.
   - Ràng buộc cấu hình Backbone: `backbone_w` (tối thiểu 5 phần tử `> 0`), `backbone_n` (tối thiểu 4 phần tử `> 0`), `neck_n >= 1`.
   - Ràng buộc `hidden_dim`: hỗ trợ cả `int` và `tuple`/`list`.
4. **Cập nhật `load_yaml` và `load_json`**:
   - Mở rộng danh sách `tuple_fields`:
     ```python
     tuple_fields = ["image_size", "video_exts", "cnn_neck_channels", "betas", "backbone_w", "backbone_n", "target_size"]
     ```
   - Xử lý ép kiểu `tuple` an toàn khi đọc dữ liệu từ YAML/JSON.

### 3.2. Cập nhật `configs/config.yaml`
1. **Tổ chức lại Section `model:` thành các nhóm phân định rõ ràng**:
   - **Nhóm 1**: Các tham số `[DÙNG CHUNG]` (chung cho cả `ConvGRUClassifier` và `DrowsyClassifier`):
     - `num_classes: 2`
     - `input_dim: 128`
     - `hidden_dim: 128`
     - `convgru_hidden_dim: 128`
     - `num_layers: 2`
     - `dropout: 0.35`
     - `supervision_mode: "attention_pooling"`
     - `target_size: [14, 14]`
   - **Nhóm 2**: Tham số `[DÀNH RIÊNG DrowsyClassifier]` (`src/models1.py` & `src/train1.py`):
     - `backbone_w: [16, 32, 64, 128, 256]`
     - `backbone_n: [1, 2, 2, 1]`
     - `neck_n: 1`
     - `gradient_checkpointing: true`
     - `freeze_bn: false`
   - **Nhóm 3**: Tham số `[DÀNH RIÊNG ConvGRUClassifier]` (`src/models.py` & `src/train.py`):
     - `cnn_neck_channels: [64, 128, 256]`
     - `adapter_dropout: 0.25`
2. **Gắn nhãn comment `[DÙNG CHUNG]` cho các Section còn lại**:
   - `dataset`: Gắn `[DÙNG CHUNG]` cho các trường dữ liệu; chú thích `backbone_neck_checkpoint` là dành riêng cho `train.py`.
   - `dataloader`: Gắn `[DÙNG CHUNG]`; chú thích `chunk_size` là dành riêng cho `train.py`.
   - `loss`: Gắn `[DÙNG CHUNG]`.
   - `optimizer`: Gắn `[DÙNG CHUNG]`.
   - `logging`: Gắn `[DÙNG CHUNG]`.
   - `runtime`: Gắn `[DÙNG CHUNG]`.

### 3.3. Đề xuất hoàn thiện `src/train1.py` (Khuyến nghị đồng bộ)
Trong [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py), khi khởi tạo `DrowsyClassifier`, hiện tại đang sử dụng giá trị mặc định cho `backbone_w`, `backbone_n`, `neck_n`. Ta sẽ đồng bộ để đọc các giá trị này từ `self.cfg`:
```python
backbone_w = getattr(self.cfg, "backbone_w", (16, 32, 64, 128, 256))
backbone_n = getattr(self.cfg, "backbone_n", (1, 2, 2, 1))
neck_n = getattr(self.cfg, "neck_n", 1)

self.model = DrowsyClassifier(
    num_classes=num_classes,
    backbone_w=backbone_w,
    backbone_n=backbone_n,
    neck_n=neck_n,
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    target_size=target_size,
    supervision_mode=self.cfg.supervision_mode,
    gradient_checkpointing=gradient_checkpointing,
    freeze_bn=freeze_bn,
).to(self.device)
```

---

## 4. DỰ THẢO THAY ĐỔI MÃ NGUỒN CỤ THỂ

### 4.1. Dự thảo cấu trúc Section Model trong `configs/config.py`
```python
    # ---- 3. MODEL ARCHITECTURE CONFIGURATION ----
    # [DÙNG CHUNG] Số lượng lớp đầu ra phân loại (2: Tỉnh táo vs Buồn ngủ)
    num_classes: int = 2
    # [DÙNG CHUNG] Kích thước kênh sau SpatialReductionNeck đưa vào ConvGRU
    input_dim: int = 128
    # [DÙNG CHUNG] Bí danh đồng bộ cho input_dim được đọc bởi models.py
    convgru_in_dim: int = 128
    # [DÙNG CHUNG] Số kênh ẩn trong từng tầng ConvGRU
    hidden_dim: Union[int, Tuple[int, ...]] = 128
    # [DÙNG CHUNG] Bí danh đồng bộ cho hidden_dim được đọc bởi models.py
    convgru_hidden_dim: Union[int, Tuple[int, ...]] = 128
    # [DÙNG CHUNG] Số tầng ConvGRU xếp chồng
    num_layers: int = 2
    # [DÙNG CHUNG] Bí danh đồng bộ cho num_layers được đọc bởi models.py
    convgru_num_layers: int = 2
    # [DÙNG CHUNG] Tỷ lệ Dropout cho đầu phân loại FC Head
    dropout: float = 0.35
    # [DÙNG CHUNG] Cơ chế giám sát chuỗi thời gian ('attention_pooling' | 'sequence')
    supervision_mode: str = "attention_pooling"
    # [DÙNG CHUNG] Kích thước lưới không gian hội tụ (Height, Width) sau SpatialReductionNeck
    target_size: Tuple[int, int] = (14, 14)

    # ---- THAM SỐ DÀNH RIÊNG CHO DROWSYCLASSIFIER (src/models1.py & src/train1.py) ----
    # [DÀNH RIÊNG DrowsyClassifier] Độ rộng kênh các tầng Backbone (c1, c2, c3, c4, c5)
    backbone_w: Tuple[int, ...] = (16, 32, 64, 128, 256)
    # [DÀNH RIÊNG DrowsyClassifier] Số lượng khối lặp lại trong từng tầng Backbone
    backbone_n: Tuple[int, ...] = (1, 2, 2, 1)
    # [DÀNH RIÊNG DrowsyClassifier] Số lượng khối lặp lại trong PAFPN
    neck_n: int = 1
    # [DÀNH RIÊNG DrowsyClassifier/train1.py] Bật gradient checkpointing cho Backbone/Neck tiết kiệm VRAM
    gradient_checkpointing: bool = True
    # [DÀNH RIÊNG DrowsyClassifier/train1.py] Khóa running mean/var của BatchNorm trong Backbone/Neck
    freeze_bn: bool = False

    # ---- THAM SỐ DÀNH RIÊNG CHO CONVGRU PHÂN TẦNG (src/models.py & src/train.py) ----
    # [DÀNH RIÊNG ConvGRUClassifier] Kênh thực tế của (p3, p4, p5) từ PAFPN Backbone ngoài
    cnn_neck_channels: Tuple[int, int, int] = (64, 128, 256)
    # [DÀNH RIÊNG ConvGRUClassifier] Tỷ lệ Dropout sau cổ giảm kênh SpatialReductionNeck
    adapter_dropout: float = 0.25
```

### 4.2. Dự thảo cấu trúc Section Model trong `configs/config.yaml`
```yaml
# ---- 3. MODEL ARCHITECTURE CONFIGURATION ----
model:
  # === CÁC THAM SỐ DÙNG CHUNG (Cả ConvGRUClassifier và DrowsyClassifier) ===
  # [DÙNG CHUNG] Số lượng lớp phân loại đầu ra (2: Tỉnh táo / Buồn ngủ)
  num_classes: 2
  # [DÙNG CHUNG] Số kênh đặc trưng sau cổ giảm kênh SpatialReductionNeck (nén 448 -> 128)
  input_dim: 128
  # [DÙNG CHUNG] Số kênh ẩn của từng tầng ConvGRU
  hidden_dim: 128
  convgru_hidden_dim: 128
  # [DÙNG CHUNG] Số tầng ConvGRU xếp chồng
  num_layers: 2
  # [DÙNG CHUNG] Tỷ lệ Dropout cho đầu phân loại FC Head
  dropout: 0.35
  # [DÙNG CHUNG] Cơ chế giám sát chuỗi thời gian ('attention_pooling' | 'sequence')
  supervision_mode: "attention_pooling"
  # [DÙNG CHUNG] Kích thước lưới không gian hội tụ (Height, Width) sau SpatialReductionNeck
  target_size: [14, 14]

  # === THAM SỐ DÀNH RIÊNG CHO DROWSYCLASSIFIER (src/models1.py & src/train1.py) ===
  # [DÀNH RIÊNG DrowsyClassifier] Độ rộng kênh các tầng Backbone (c1, c2, c3, c4, c5)
  backbone_w: [16, 32, 64, 128, 256]
  # [DÀNH RIÊNG DrowsyClassifier] Số lượng khối lặp lại trong từng tầng Backbone
  backbone_n: [1, 2, 2, 1]
  # [DÀNH RIÊNG DrowsyClassifier] Số lượng khối lặp lại trong PAFPN
  neck_n: 1
  # [DÀNH RIÊNG DrowsyClassifier/train1.py] Bật gradient checkpointing cho Backbone/Neck tiết kiệm VRAM
  gradient_checkpointing: true
  # [DÀNH RIÊNG DrowsyClassifier/train1.py] Khóa running mean/var của BatchNorm trong Backbone/Neck
  freeze_bn: false

  # === THAM SỐ DÀNH RIÊNG CHO CONVGRU PHÂN TẦNG (src/models.py & src/train.py) ===
  # [DÀNH RIÊNG ConvGRUClassifier] Kênh đặc trưng của các tầng (p3, p4, p5) từ Backbone ngoài
  cnn_neck_channels: [64, 128, 256]
  # [DÀNH RIÊNG ConvGRUClassifier] Tỷ lệ Dropout sau cổ giảm kênh SpatialReductionNeck
  adapter_dropout: 0.25
```

---

## 5. KẾ HOẠCH NGHIỆM THU & TIÊU CHÍ CHẤT LƯỢNG (VERIFICATION CHECKLIST)

1. **Khả năng nạp và parse cấu hình**:
   - `load_config("configs/config.yaml")` trả về đối tượng `TrainConfig` hợp lệ, không ném ngoại lệ.
   - Các trường `backbone_w`, `backbone_n`, `target_size` được chuyển đổi chuẩn xác từ `list` sang kiểu `tuple`.
2. **Khởi tạo mô hình thành công từ cấu hình**:
   - `DrowsyClassifier` trong `src/models1.py` khởi tạo thành công với các tham số từ `TrainConfig`.
   - `ConvGRUClassifier.from_config(cfg)` trong `src/models.py` tiếp tục hoạt động bình thường, không suy giảm tương thích ngược.
3. **Chạy thử nghiệm (Dry-run)**:
   - Chạy lệnh kiểm thử đơn vị hoặc dry-run với cả `configs/config.py` và `configs/config.yaml` để xác nhận tính nhất quán.

---

## 6. KẾT LUẬN & TRẠNG THÁI HIỆN TẠI

Báo cáo phân tích đã hoàn thành theo đúng nguyên tắc **Bước 1: Khảo sát & Phân tích (Discovery)** của [`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md).
Kính mời người dùng xem xét và phản hồi để Agent tiến hành **Bước 2: Lên kế hoạch thực hiện (`docs/plan/plan_config_drowsy.md`)**.
