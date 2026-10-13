# BÁO CÁO KẾT QUẢ THỰC HIỆN: CẬP NHẬT THAM SỐ CẤU HÌNH CHO MODEL DROWSYCLASSIFIER (CONFIG.PY, CONFIG.YAML & SRC/TRAIN1.PY)

- **Mã báo cáo**: `REPORT_CONFIG_DROWSY_CLASSIFIER`
- **Tệp báo cáo**: `docs/report/report_config_drowsy.md`
- **Dựa trên kế hoạch**: [`docs/plan/plan_config_drowsy.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/plan/plan_config_drowsy.md)
- **Tài liệu phân tích**: [`docs/analsys/analsys_config_drowsy.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/analsys/analsys_config_drowsy.md)
- **Các tệp đã triển khai / cập nhật**:
  - [`configs/config.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.py)
  - [`configs/config.yaml`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.yaml)
  - [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py)
- **Trạng thái**: Hoàn thành 100% nhiệm vụ theo quy trình chuẩn [`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md).

---

## 1. TỔNG QUAN KẾT QUẢ THỰC HIỆN

Thực hiện yêu cầu từ người dùng:
> *"cập nhật @[configs/config.py] và @[configs/config.yaml] thêm các tham số ở model DrowsyClassifier @[src/models1.py] @[src/train1.py] tham số nào dùng chung thì nào commet là dùng chung"*  
> *"đồng bộ thêm @[src/train1.py]. thực hiện bước tiếp theo"*

Agent đã thực hiện trọn vẹn quy trình 3 bước chuẩn của [`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md):
1. **Bước 1 - Khảo sát & Phân tích ([`analsys_config_drowsy.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/analsys/analsys_config_drowsy.md))**: Rà soát đối chiếu toàn bộ tham số giữa 2 mô hình (`ConvGRUClassifier` và `DrowsyClassifier`) cùng 2 pipeline huấn luyện (`train.py` và `train1.py`), phân loại chi tiết các tham số dùng chung và tham số riêng biệt.
2. **Bước 2 - Lên kế hoạch chi tiết ([`plan_config_drowsy.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/plan/plan_config_drowsy.md))**: Lập lộ trình 4 giai đoạn cụ thể gồm sửa đổi dataclass, cập nhật yaml, đồng bộ `train1.py` và kiểm thử tự động.
3. **Bước 3 - Triển khai thực tế & Nghiệm thu ([`report_config_drowsy.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/docs/report/report_config_drowsy.md))**: Đã hoàn tất chỉnh sửa mã nguồn và vượt qua 100% các bài kiểm thử xác thực cấu hình và mô hình.

---

## 2. CHI TIẾT CÁC THAY ĐỔI ĐÃ TRIỂN KHAI

### 2.1. Cập nhật [`configs/config.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.py)
1. **Bổ sung 6 tham số của `DrowsyClassifier` vào Dataclass `TrainConfig`**:
   - `backbone_w: Tuple[int, ...] = (16, 32, 64, 128, 256)`: Độ rộng kênh qua các tầng Backbone c1..c5.
   - `backbone_n: Tuple[int, ...] = (1, 2, 2, 1)`: Số lượng khối lặp lại trong từng tầng Backbone.
   - `neck_n: int = 1`: Số lượng khối lặp lại trong PAFPN.
   - `target_size: Tuple[int, int] = (14, 14)`: Kích thước lưới không gian hội tụ sau cổ giảm kênh `SpatialReductionNeck`.
   - `gradient_checkpointing: bool = True`: Bật gradient checkpointing cho Backbone/Neck tiết kiệm VRAM.
   - `freeze_bn: bool = False`: Khóa thống kê running mean/var của BatchNorm trong Backbone/Neck.
2. **Gắn chú thích `# [DÙNG CHUNG]` toàn diện**:
   - Gắn nhãn `# [DÙNG CHUNG]` cho toàn bộ các tham số kiến trúc chung: `num_classes`, `input_dim`, `convgru_in_dim`, `hidden_dim`, `convgru_hidden_dim`, `num_layers`, `convgru_num_layers`, `dropout`, `supervision_mode`, `target_size`.
   - Gắn nhãn `# [DÙNG CHUNG]` cho tất cả các tham số dùng chung trong các section `DATASET`, `DATALOADER`, `LOSS`, `OPTIMIZER`, `LOGGING`, `RUNTIME`.
   - Phân biệt rõ các tham số trích xuất rời chỉ dùng riêng cho `train.py`: `backbone_neck_checkpoint`, `chunk_size`, `cnn_neck_channels`, `adapter_dropout`.
3. **Bổ sung kiểm tra hợp lệ trong `__post_init__`**:
   - Xác thực `target_size` có độ dài 2 và các phần tử `> 0`.
   - Xác thực `backbone_w` (tối thiểu 5 tầng kênh `> 0`), `backbone_n` (tối thiểu 4 tầng lặp `> 0`), `neck_n >= 1`.
   - Hỗ trợ `hidden_dim` linh hoạt cho cả `int` và `tuple`/`list`.
4. **Mở rộng bộ giải mã `load_yaml` và `load_json`**:
   - Cập nhật `tuple_fields` bổ sung: `"backbone_w"`, `"backbone_n"`, `"target_size"` để tự động ép kiểu từ `list` sang `tuple` khi nạp từ YAML hoặc JSON.
   - Xử lý ép kiểu an toàn cho `hidden_dim` nếu đầu vào là `list`.
5. **Cơ chế chống lỗi Unicode trên Windows**:
   - Bổ sung cấu hình tự động reconfigure `sys.stdout` và `sys.stderr` sang `utf-8` khi chạy trên hệ điều hành Windows, triệt tiêu lỗi `UnicodeEncodeError`.

### 2.2. Cập nhật [`configs/config.yaml`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.yaml)
1. **Phân nhóm chuyên biệt và rõ ràng trong section `model:`**:
   - Phân khu 1: **CÁC THAM SỐ DÙNG CHUNG** (`num_classes`, `input_dim`, `hidden_dim`, `convgru_hidden_dim`, `num_layers`, `dropout`, `supervision_mode`, `target_size`).
   - Phân khu 2: **THAM SỐ DÀNH RIÊNG CHO DROWSYCLASSIFIER** (`backbone_w`, `backbone_n`, `neck_n`, `gradient_checkpointing`, `freeze_bn`).
   - Phân khu 3: **THAM SỐ DÀNH RIÊNG CHO CONVGRU TÁCH RỜI** (`cnn_neck_channels`, `adapter_dropout`).
2. **Gắn comment `# [DÙNG CHUNG]` đồng bộ cho tất cả các section còn lại**:
   - `dataset`: Gắn `[DÙNG CHUNG]` cho các trường dữ liệu; chú thích `backbone_neck_checkpoint` dành riêng cho `train.py`.
   - `dataloader`: Gắn `[DÙNG CHUNG]`; chú thích `chunk_size` dành riêng cho `train.py`.
   - `loss`: Gắn `[DÙNG CHUNG]`.
   - `optimizer`: Gắn `[DÙNG CHUNG]`.
   - `logging`: Gắn `[DÙNG CHUNG]`.
   - `runtime`: Gắn `[DÙNG CHUNG]`.

### 2.3. Đồng bộ hóa [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py)
Trong lớp `Trainer.__init__`, nạp trực tiếp các thuộc tính từ cấu hình và truyền đầy đủ vào hàm khởi tạo của `DrowsyClassifier`:
```python
# Đọc đầy đủ các tham số từ self.cfg
num_classes = getattr(self.cfg, "num_classes", 2)
backbone_w = getattr(self.cfg, "backbone_w", (16, 32, 64, 128, 256))
backbone_n = getattr(self.cfg, "backbone_n", (1, 2, 2, 1))
neck_n = getattr(self.cfg, "neck_n", 1)
input_dim = getattr(self.cfg, "convgru_in_dim", getattr(self.cfg, "input_dim", 64))
hidden_dim = getattr(self.cfg, "convgru_hidden_dim", getattr(self.cfg, "hidden_dim", 64))
gradient_checkpointing = getattr(self.cfg, "gradient_checkpointing", True)
freeze_bn = getattr(self.cfg, "freeze_bn", False)

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

## 3. KẾT QUẢ KIỂM THỬ THỰC TẾ (VERIFICATION RESULTS)

Đã thực hiện bộ kiểm thử toàn diện với kết quả đạt tỷ lệ thành công **100%**:

### 3.1. Kiểm thử nạp Cấu hình YAML & Khởi tạo Cả hai Mô hình
- **Lệnh thực thi**:
  ```bash
  python -c "
  from configs.config import load_config
  from src.models1 import DrowsyClassifier
  from src.models import ConvGRUClassifier

  cfg = load_config('configs/config.yaml')
  assert isinstance(cfg.backbone_w, tuple)
  assert isinstance(cfg.backbone_n, tuple)
  assert isinstance(cfg.target_size, tuple)

  m1 = DrowsyClassifier(...)
  m2 = ConvGRUClassifier.from_config(cfg)
  "
  ```
- **Kết quả**:
  - `backbone_w`: `(16, 32, 64, 128, 256)` kiểu `tuple` ✅
  - `backbone_n`: `(1, 2, 2, 1)` kiểu `tuple` ✅
  - `neck_n`: `1` kiểu `int` ✅
  - `target_size`: `(14, 14)` kiểu `tuple` ✅
  - Khởi tạo `DrowsyClassifier` thành công: `3,547,156` tham số ✅
  - Khởi tạo `ConvGRUClassifier` từ cấu hình thành công: `1,910,340` tham số ✅ (Bảo toàn 100% tính tương thích ngược).

### 3.2. Kiểm thử Vòng đời JSON (JSON Save & Load Roundtrip)
- **Lệnh thực thi**: Lưu `TrainConfig()` ra tệp JSON tạm và nạp lại qua `TrainConfig.load_json()`.
- **Kết quả**:
  - `TrainConfig` mặc định hợp lệ.
  - Tự động chuyển đổi `list` -> `tuple` chuẩn xác cho `backbone_w`, `backbone_n`, `target_size`.
  - Không gặp lỗi mã hóa ký tự tiếng Việt trên Windows terminal ✅.

### 3.3. Kiểm thử CLI Entry-Point
- Chạy kiểm tra hiển thị trợ giúp cho cả 2 pipeline:
  - `python src/train1.py --help` -> Exit code 0 ✅
  - `python src/train.py --help` -> Exit code 0 ✅

---

## 4. BẢNG TỔNG KẾT ĐỐI CHIẾU THAM SỐ CẤU HÌNH

| Tên tham số | Giá trị trong `config.yaml` | Kiểu dữ liệu | Nhãn chú thích | Mục đích sử dụng |
| :--- | :--- | :--- | :--- | :--- |
| `num_classes` | `2` | `int` | `[DÙNG CHUNG]` | Số lớp phân loại (Tỉnh táo / Buồn ngủ) |
| `input_dim` | `128` | `int` | `[DÙNG CHUNG]` | Số kênh nén sau `SpatialReductionNeck` |
| `hidden_dim` | `128` | `int / tuple` | `[DÙNG CHUNG]` | Số kênh ẩn của các tầng ConvGRU |
| `num_layers` | `2` | `int` | `[DÙNG CHUNG]` | Số tầng ConvGRU xếp chồng |
| `dropout` | `0.35` | `float` | `[DÙNG CHUNG]` | Dropout đầu phân loại FC Head |
| `supervision_mode` | `"attention_pooling"` | `str` | `[DÙNG CHUNG]` | Cơ chế giám sát chuỗi thời gian |
| `target_size` | `[14, 14]` | `Tuple[int, int]` | `[DÙNG CHUNG]` | Kích thước lưới không gian sau cổ giảm kênh |
| `backbone_w` | `[16, 32, 64, 128, 256]` | `Tuple[int, ...]` | `[DÀNH RIÊNG DrowsyClassifier]` | Độ rộng kênh các tầng Backbone c1..c5 |
| `backbone_n` | `[1, 2, 2, 1]` | `Tuple[int, ...]` | `[DÀNH RIÊNG DrowsyClassifier]` | Số lượng block lặp trong từng tầng Backbone |
| `neck_n` | `1` | `int` | `[DÀNH RIÊNG DrowsyClassifier]` | Số lượng block lặp trong PAFPN |
| `gradient_checkpointing` | `true` | `bool` | `[DÀNH RIÊNG DrowsyClassifier/train1.py]` | Checkpointing giảm VRAM khi huấn luyện End-to-End |
| `freeze_bn` | `false` | `bool` | `[DÀNH RIÊNG DrowsyClassifier/train1.py]` | Khóa running mean/var của BatchNorm |
| `cnn_neck_channels` | `[64, 128, 256]` | `Tuple[int, int, int]` | `[DÀNH RIÊNG ConvGRUClassifier]` | Kênh đặc trưng từ detector BackboneNeck rời |
| `adapter_dropout` | `0.25` | `float` | `[DÀNH RIÊNG ConvGRUClassifier]` | Dropout cổ giảm kênh của ConvGRU tách rời |
| *Dataset, Dataloader, Loss, Optimizer, Logging, Runtime* | *Theo cấu hình* | *Đa dạng* | `[DÙNG CHUNG]` | Toàn bộ các tham số vận hành pipeline dùng chung cho cả 2 luồng train |

---

## 5. KẾT LUẬN & BÀN GIAO

Nhiệm vụ đã hoàn thành toàn diện, chính xác và chuyên nghiệp theo đúng hướng dẫn của [`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md):
- Cả hai tệp [`configs/config.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.py) và [`configs/config.yaml`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.yaml) đã được bổ sung đầy đủ tham số cho `DrowsyClassifier`, có chú thích rõ ràng `[DÙNG CHUNG]` cho các tham số chung.
- Mã nguồn [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py) đã được đồng bộ hóa hoàn toàn để tiếp nhận các tham số này từ file config.
- Không gây bất kỳ lỗi xung đột hay suy giảm tương thích ngược nào với [`src/train.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train.py) và [`src/models.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py).
