# TÀI LIỆU PHÂN TÍCH YÊU CẦU: XÂY DỰNG SCRIPT HUẤN LUYỆN END-TO-END DROWSYCLASSIFIER (SRC/TRAIN1.PY)

- **Mã yêu cầu**: `TRAIN_DROWSY_CLASSIFIER_END_TO_END`
- **Tệp phân tích**: `docs/analsys/analsys_train1.md`
- **Mã nguồn tham chiếu**:
  - [`src/train.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train.py) (Chuẩn mực về logging, checkpoint, AMP, OOM recovery, EarlyStopping, metrics)
  - [`src/models1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models1.py) (Kiến trúc mô hình End-to-End `DrowsyClassifier`)
  - [`src/models.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py) (`ConvGRUClassifier` cốt lõi)
  - [`src/dataset.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/dataset.py) (`build_raw_video_dataloaders`, `collate_video_frames`)
  - [`configs/config.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.py) & [`configs/config.yaml`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/configs/config.yaml)
- **Tệp mã nguồn dự kiến tạo**: [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py)
- **Quy chuẩn áp dụng**: [`AGENTS.md`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/AGENTS.md) — **Bước 1: Khảo sát & Phân tích (Discovery)**
- **Trạng thái**: Đã cập nhật theo phản hồi của người dùng (Bỏ nạp pretrained Backbone/Neck từ Detector 2.6M); đang chờ người dùng phê duyệt để chuyển sang Bước 2 (Lên kế hoạch).

---

## 1. TỔNG QUAN YÊU CẦU & MỤC TIÊU CỐT LÕI

### 1.1. Yêu cầu từ người dùng
> *"tạo file @src\train1.py để train model DrowsyClassifier với các thành phần và cách ghi log, checkpoint và các thành phần liên quan giống file @[src/train.py]"*  
> **Cập nhật chỉ đạo**: *"bỏ Hỗ trợ cơ chế nạp trọng số Pretrained Backbone/Neck từ checkpoint của bộ phát hiện (Detector 2.6M)"*

### 1.2. Mục tiêu kỹ thuật
1. **Kế thừa toàn diện các tiêu chuẩn chất lượng cao từ `src/train.py`**:
   - Ghi log trực quan qua **TensorBoard** (`SummaryWriter`) theo cả 2 cấp độ: Step (Loss, Acc, Recall, F1) và Epoch (Loss, Acc, Recall, F1, Learning Rate).
   - Thanh tiến trình **tqdm** thời gian thực hiển thị loss trung bình của batch.
   - Tự động ghi nhận lịch sử huấn luyện ra tệp **`training_history.csv`**.
   - Cơ chế lưu Checkpoint nguyên tử (**Atomic Save** via `.tmp` -> `rename`), chống hỏng file khi bị ngắt đột ngột; phân loại `last.pt`, `best.pt`, `epoch_{N}.pt`, cùng chính sách xoay vòng `ckpt_keep_last`.
   - Cơ chế khôi phục huấn luyện hoàn chỉnh (**Resumable Training**) bao gồm: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict` (Resumable AMP), `epoch`, `global_step`, `best_val_f1`.
   - Hỗ trợ phân giải bí danh checkpoint linh hoạt (`'last'`, `'best'`, hoặc đường dẫn cụ thể).
   - Cơ chế giám sát dừng sớm **EarlyStopping** theo metric cấu hình (`val_f1`).
   - Tự động bắt và phục hồi ngoại lệ **CUDA Out-Of-Memory (OOM)** cho cả vòng lặp Train và Validation, dọn dẹp tham chiếu và `torch.cuda.empty_cache()` để tiếp tục huấn luyện mà không làm sập chương trình.
   - Tích lũy Gradient (**Gradient Accumulation**) xử lý chuẩn xác tại ranh giới batch cuối epoch.
   - Cắt chuẩn Gradient (**Gradient Clipping**) qua `clip_grad_norm_`.
   - Cố định Seed toàn diện (`seed_everything`) cho Python random, NumPy, PyTorch và cuDNN deterministic.
   - Hỗ trợ đa dạng hàm mất mát (Loss): `CrossEntropyLoss` (chuẩn) và `BCEWithLogitsLoss`.

2. **Chuyên biệt hóa cho kiến trúc End-to-End thuần túy `DrowsyClassifier` (`src/models1.py`)**:
   - Khởi tạo toàn bộ mô hình `DrowsyClassifier` (Backbone + PAFPN + ConvGRU) và huấn luyện từ đầu (from scratch) theo kiến trúc End-to-End thống nhất.
   - **LOẠI BỎ HOÀN TOÀN** việc nạp trọng số Backbone/Neck từ detector 2.6M (`backbone_neck_checkpoint`). Checkpoint chỉ được nạp khi người dùng yêu cầu resume (`--resume`) một checkpoint trước đó của chính `DrowsyClassifier`.
   - Đầu vào trực tiếp là tensor video raw `[B, T, 3, H, W]` từ `DataLoader`.
   - Tối ưu hóa bộ nhớ VRAM cho bài toán huấn luyện Video End-to-End: Tận dụng cờ `gradient_checkpointing=True` và `freeze_bn=True` đã được tích hợp trong `DrowsyClassifier`.
   - Loại bỏ hoàn toàn sự phụ thuộc vào lớp bọc trung gian `ChunkedBackboneNeckExtractor`.

---

## 2. BẢNG PHÂN TÍCH ĐỐI CHIẾU: `SRC/TRAIN.PY` VS `SRC/TRAIN1.PY`

Nhằm đảm bảo `src/train1.py` vừa tương đồng 100% về cơ chế log, checkpoint, metrics, vừa tinh gọn, loại bỏ các thành phần nạp pretrained không cần thiết:

| Thành phần kỹ thuật | Hiện trạng trong `src/train.py` | Thiết kế tương ứng trong `src/train1.py` | Ghi chú & Rationale kỹ thuật |
| :--- | :--- | :--- | :--- |
| **Kiến trúc Mô hình chính** | `ConvGRUClassifier` ([src/models.py](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models.py)) | `DrowsyClassifier` ([src/models1.py](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/models1.py)) | `DrowsyClassifier` là mô hình End-to-End tích hợp Backbone + Neck + ConvGRU. |
| **Luồng xử lý Đặc trưng** | Giai đoạn 2: Gọi `ChunkedBackboneNeckExtractor` trích xuất `(p3, p4, p5)` không tính gradient, rồi mới đưa vào `ConvGRUClassifier`. | End-to-End: Đưa trực tiếp tensor video `frames_t` `[B, T, 3, H, W]` vào `DrowsyClassifier(frames_t, seq_lens=seq_lens)`. Gradient lan truyền xuyên suốt từ ConvGRU về Backbone. | Loại bỏ sự phụ thuộc vào lớp bọc trung gian `ChunkedBackboneNeckExtractor`. |
| **Khởi tạo Trọng số** | Nạp checkpoint `backbone_neck_checkpoint` vào `ChunkedBackboneNeckExtractor`. | **Khởi tạo ngẫu nhiên toàn bộ mạng (Train from scratch)**. Tuyệt đối không nạp pretrained detector 2.6M. | Tuân thủ chỉ đạo của người dùng: Bỏ nạp pretrained backbone/neck từ detector. |
| **Kiểm soát VRAM & Ổn định** | Không cần gradient checkpointing vì Backbone đã bị đóng băng `requires_grad=False`. | Kích hoạt `gradient_checkpointing=True` và `freeze_bn=False` (hoặc cấu hình) trong `DrowsyClassifier`. | Gradient Checkpointing giúp giảm 50-60% VRAM khi lan truyền ngược qua toàn bộ mạng video. |
| **Hàm mất mát (Loss Function)** | `compute_loss_and_preds`: Hỗ trợ BCE (1-logit) và CE (multi-class), xử lý an toàn không mất gradient. | Kế thừa 100% logic của `compute_loss_and_preds` từ `src/train.py`. | Đảm bảo tính nhất quán và triệt tiêu lỗi mất gradient. |
| **Bộ chỉ số (Metrics)** | `calculate_metrics`: Accuracy, Recall (macro), F1-Score (macro) qua `sklearn.metrics`. | Kế thừa 100% hàm `calculate_metrics` từ `src/train.py`. | Đảm bảo tính nhất quán đánh giá. |
| **Ghi nhận Lịch sử Huấn luyện (CSV)** | `_init_history_csv` & `_log_history_csv`: Ghi file `training_history.csv` gồm 11 cột chuẩn. | Kế thừa 100% cấu trúc tệp CSV và định dạng ghi log của `src/train.py`. | Tương thích hoàn toàn với các script vẽ biểu đồ và phân tích benchmark. |
| **TensorBoard Logging** | Ghi đầy đủ `Train_Step/*`, `Train_Epoch/*`, `Val_Epoch/*`. | Kế thừa 100% cấu trúc các tag và thời điểm ghi log của `src/train.py`. | Đồng bộ tuyệt đối trực quan hóa trên TensorBoard. |
| **Cơ chế Lưu Checkpoint** | Atomic save (`.tmp` -> rename), lưu `last.pt`, `best.pt`, `epoch_{N}.pt`, xoay vòng `ckpt_keep_last`. | Kế thừa 100% cơ chế Atomic Save và logic quản lý tệp checkpoint của `src/train.py`. | Ngăn ngừa hỏng tệp khi mất điện hoặc bị ngắt đột ngột. |
| **Cơ chế Phục hồi (Resume)** | Khôi phục model, optimizer, scheduler, scaler (Resumable AMP), epoch, best_val_f1. Hỗ trợ alias `'last'`, `'best'`. | Kế thừa 100% logic nạp và phân giải checkpoint của `src/train.py` cho `DrowsyClassifier`. | Phục hồi trạng thái đầy đủ khi tiếp tục train từ checkpoint của chính `DrowsyClassifier`. |
| **Xử lý Ngoại lệ CUDA OOM** | Bắt lỗi `"out of memory"`, dọn dẹp biến, `empty_cache()`, `zero_grad()`, bỏ qua batch. | Kế thừa 100% cơ chế bắt lỗi OOM cho cả Train và Val loops. | Giúp tiến trình huấn luyện không bị sập giữa chừng khi gặp video dài đột biến. |
| **CLI & Nạp Cấu hình** | `argparse` hỗ trợ `--config` và `--resume`. | Kế thừa 100% giao diện CLI của `src/train.py`. | Đồng bộ cách thức thực thi từ terminal. |

---

## 3. THIẾT KẾ KIẾN TRÚC CHI TIẾT CHO `SRC/TRAIN1.PY`

### 3.1. Cấu trúc Lớp `Trainer` trong `src/train1.py`

```text
src/train1.py
│
├── seed_everything(seed: int) -> None
├── calculate_metrics(preds, targets) -> Dict[str, float]
├── EarlyStopping
│   ├── __init__(patience, min_delta, mode, monitor)
│   └── step(current_val, epoch) -> bool
│
└── Trainer (hoển luyện DrowsyClassifier End-to-End)
    ├── __init__(config: TrainConfig)
    │   ├── Thiết lập thiết bị (CUDA/CPU)
    │   ├── Khởi tạo DataLoader (build_raw_video_dataloaders)
    │   ├── Khởi tạo mô hình DrowsyClassifier (từ models1.py) - KHÔNG nạp detector backbone
    │   ├── Cấu hình Loss (CrossEntropy / BCEWithLogits)
    │   ├── Cấu hình Optimizer & Scheduler (AdamW / SGD + CosineAnnealingLR)
    │   ├── Cấu hình TensorBoard SummaryWriter & Thư mục Checkpoint
    │   ├── Cấu hình AMP GradScaler
    │   ├── Cấu hình EarlyStopping
    │   └── Khởi tạo tệp CSV training_history.csv
    │
    ├── _init_history_csv() -> None
    ├── _log_history_csv(epoch, train_m, val_m, elapsed_sec) -> None
    ├── compute_loss_and_preds(logits, labels) -> (loss, preds)
    ├── train_epoch(epoch: int) -> Dict[str, float]
    ├── validate_epoch(epoch: int) -> Dict[str, float]
    ├── _resolve_checkpoint_path(checkpoint_path: str) -> Path
    ├── load_checkpoint(checkpoint_path: str) -> None (chỉ nạp checkpoint DrowsyClassifier)
    ├── _atomic_save(state: Dict[str, Any], target_path: Path) -> None
    ├── save_checkpoint(epoch: int, is_best: bool, val_f1: Optional[float]) -> None
    └── train() -> None
```

### 3.2. Khởi tạo Mô hình `DrowsyClassifier` trong `Trainer.__init__`
```python
# Xác định target_size tương thích với kích thước ảnh đầu vào
img_size = getattr(self.cfg, "image_size", (640, 640))
h_in, w_in = (img_size, img_size) if isinstance(img_size, int) else img_size
target_size = (max(1, h_in // 16), max(1, w_in // 16))

# Khởi tạo mô hình DrowsyClassifier End-to-End từ đầu (Scratch)
self.model = DrowsyClassifier(
    num_classes=getattr(self.cfg, "num_classes", 2),
    input_dim=getattr(self.cfg, "input_dim", 128),
    hidden_dim=getattr(self.cfg, "hidden_dim", 64),
    target_size=target_size,
    supervision_mode=getattr(self.cfg, "supervision_mode", "attention_pooling"),
    gradient_checkpointing=getattr(self.cfg, "gradient_checkpointing", True),
    freeze_bn=getattr(self.cfg, "freeze_bn", False),
).to(self.device)
```

---

## 4. TIÊU CHÍ NGHIỆM THU CHẤT LƯỢNG (QUALITY CHECKLIST)

Sau khi triển khai tại Bước 3, tệp [`src/train1.py`](file:///D:/Project/DATN/driver-guardian/ai/LSTM/src/train1.py) phải thỏa mãn đầy đủ các tiêu chí:
- [ ] Tuân thủ quy chuẩn Clean Code, Type Hints, Docstrings chuẩn Google/NumPy style.
- [ ] Tất cả đường dẫn dùng `pathlib.Path` tương thích đa nền tảng (Windows/Linux).
- [ ] Khởi tạo mô hình `DrowsyClassifier` thành công từ cấu hình mà không phụ thuộc/không nạp checkpoint detector Backbone.
- [ ] Vòng lặp `train_epoch` và `validate_epoch` truyền trực tiếp tensor 5D `[B, T, 3, H, W]` vào `DrowsyClassifier`.
- [ ] Cơ chế ghi log TensorBoard, CSV `training_history.csv` và thanh `tqdm` hoạt động chính xác 100% như `src/train.py`.
- [ ] Cơ chế lưu Checkpoint nguyên tử (`last.pt`, `best.pt`, `epoch_{N}.pt`) và Resume hoạt động hoàn hảo cho `DrowsyClassifier`.
- [ ] Cơ chế bắt lỗi và phục hồi CUDA OOM hoạt động an toàn.
- [ ] Chạy khối kiểm thử độc lập (`if __name__ == "__main__":`) hoặc smoke test xác nhận không phát sinh lỗi cú pháp hay runtime.

---

## 5. KẾT LUẬN & ĐỀ XUẤT BƯỚC TIẾP THEO

Tài liệu phân tích đã được cập nhật chính xác theo yêu cầu: **Bỏ cơ chế nạp trọng số Pretrained Backbone/Neck từ checkpoint Detector 2.6M**. Mô hình `DrowsyClassifier` sẽ được khởi tạo và huấn luyện thuần túy End-to-End.

Theo đúng quy định nghiêm ngặt tại **Mục 5 (`AGENTS.md`)**:
> *"Khi user chấp nhận `analsys_<viết tắt yêu cầu>.md` ở bước 1 thì mới thực hiện bước 2."*

**Đề xuất chuyển bước**: Kính đề nghị người dùng xem xét nội dung phân tích đã cập nhật. Khi người dùng xác nhận phê duyệt, Agent sẽ tiến hành ngay **Bước 2: Lập kế hoạch thực hiện (`docs/plan/plan_train1.md`)**.
