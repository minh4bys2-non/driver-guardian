# TÀI LIỆU PHÂN TÍCH YÊU CẦU: XÂY DỰNG MODULE ĐÁNH GIÁ VÀ TRỰC QUAN HÓA (SRC/EVALUATE.PY)

- **Mã yêu cầu**: `EVAL_CONVGRU`
- **Tệp phân tích**: `docs/analsys/analsys_eval_convgru.md`
- **Mục tiêu**: Xây dựng module `src/evaluate.py` nạp checkpoint mô hình `ConvGRUClassifier`, chạy kiểm định trên Validation DataLoader, tính toán và vẽ các biểu đồ trực quan (Ma trận nhầm lẫn Confusion Matrix, biểu đồ so sánh Loss, Accuracy, Recall qua mỗi giai đoạn huấn luyện).
- **Trạng thái**: Đang chờ người dùng phê duyệt (Bước 1 theo chuẩn AGENTS.md).

---

## 1. YÊU CẦU CỐT LÕI TỪ NGƯỜI DÙNG

Người dùng yêu cầu:
> *"Tạo file @src\evaluate.py thực hiện load 1 checkpoint của model ConvGRUClassifier và val dataloader để vẽ thông số như ma trận nhầm lẫn, biểu đồ so sánh loss, acc, recal, qua mỗi giai đoạn"*

### Các thành phần chính cần đáp ứng:
1. **Nạp Checkpoint mô hình `ConvGRUClassifier`**:
   - Đọc checkpoint định dạng `.pt` (ví dụ `best.pt`, `last.pt`, hoặc `epoch_*.pt`).
   - Tự động khôi phục cấu trúc mạng và các siêu tham số từ checkpoint/config mà không làm lệch kích thước tensor.
   - Tương thích thiết bị phần cứng linh hoạt (`cuda` khi có GPU, tự động fallback về `cpu` an toàn).

2. **Khởi tạo và chạy trên Validation DataLoader**:
   - Khởi tạo DataLoader độc lập cho tập kiểm định (Validation Set) dựa trên cấu hình video thô hoặc tập dữ liệu đã chuẩn hóa.
   - Tích hợp bộ trích xuất đặc trưng `ChunkedBackboneNeckExtractor` (từ `NMSFreeDetector`) để chuyển đổi khung hình video thô thành đặc trưng đa tỷ lệ `(p3, p4, p5)`.
   - Chạy suy luận tối ưu với `torch.no_grad()`, kiểm soát tràn bộ nhớ VRAM với chunking.

3. **Tính toán hệ thống chỉ số đánh giá toàn diện**:
   - **Loss**: Mất mát trung bình (CrossEntropy hoặc BCE tùy cấu hình).
   - **Accuracy**: Độ chính xác tổng thể.
   - **Recall**: Độ nhạy (Macro Recall & Per-class Recall), đặc biệt quan trọng đối với lớp Buồn ngủ (Drowsy) nhằm chống bỏ sót tai nạn.
   - **Precision & F1-Score**: Đảm bảo đánh giá cân bằng giữa báo động nhầm và bỏ sót.
   - **Confusion Matrix**: Ma trận nhầm lẫn phản ánh chi tiết True Positive, True Negative, False Positive, False Negative.

4. **Trực quan hóa đồ thị chuyên nghiệp**:
   - **Ma trận nhầm lẫn (Confusion Matrix)**: Biểu đồ Heatmap hiển thị số lượng mẫu tuyệt đối kèm tỷ lệ phần trăm chuẩn hóa (Normalized percentages) cho từng lớp (`Alert` vs `Drowsy`).
   - **Biểu đồ so sánh qua mỗi giai đoạn (Stages/Epochs Curve)**: Vẽ đồ thị so sánh Loss (Train vs Val), Accuracy (Train vs Val), Recall (Train vs Val), F1-Score qua các epoch từ nhật ký huấn luyện (`training_history.csv` hoặc quét qua các checkpoint epoch).
   - **Đồ thị ROC Curve & Precision-Recall Curve (Tính năng giá trị gia tăng)**: Đánh giá độ phân tách xác suất của mô hình tại các ngưỡng threshold khác nhau.

---

## 2. KHẢO SÁT HIỆN TRẠNG MÃ NGUỒN VÀ DỮ LIỆU

### 2.1. Kiến trúc mô hình (`src/models.py`)
- Lớp `ConvGRUClassifier`:
  - Kế thừa `nn.Module`.
  - Thành phần: `SpatialReductionNeck` (nén 448 kênh -> 64 kênh tại lưới 40x40) + `ConvGRU` (2 tầng) + `SpatialAttentionPooling` + `TemporalAttentionPooling` + `FC Head`.
  - Đã tích hợp sẵn 2 class method tiện ích:
    - `ConvGRUClassifier.from_config(config)`
    - `ConvGRUClassifier.from_checkpoint(checkpoint_path, map_location)`
  - Forward nhận `features` dạng tuple `(p3, p4, p5)` và `seq_lens`.

### 2.2. Pipeline dữ liệu và trích xuất đặc trưng (`src/dataset.py`)
- Lớp `ChunkedBackboneNeckExtractor`:
  - Nạp trọng số Backbone PAFPN của `NMSFreeDetector`.
  - Nhận tensor batch video thô `[B, T, 3, H, W]`, cắt nhỏ thành các chunk khung hình (mặc định 16/32 frames) để đẩy qua Backbone, tránh tràn VRAM.
  - Trả về tuple đặc trưng: `(p3, p4, p5)`.
- Hàm `build_raw_video_dataloaders`:
  - Tạo `train_loader` và `val_loader`.
  - Cần cung cấp hàm tạo `build_val_dataloader_only` hoặc tái sử dụng `build_raw_video_dataloaders` chỉ lấy `val_loader` để không tốn thời gian quét và khởi tạo tập huấn luyện khi chỉ cần đánh giá.

### 2.3. Dữ liệu Checkpoint và Logs hiện có
- Thư mục Checkpoint: `checkpoints/experiments/deepgru_raw_nmsfree/` chứa:
  - `best.pt` (Epoch 5, Val F1 = 0.7417, lưu đầy đủ `model_state_dict`, `config`, `epoch`).
  - `last.pt` (Epoch 6).
  - `epoch_1.pt` đến `epoch_6.pt`.
- Thư mục Logs: `logs/` chứa:
  - `training_history.csv` chứa dữ liệu chi tiết của 6 epoch đã huấn luyện:
    - Cột: `epoch, train_loss, train_acc, train_recall, train_f1, val_loss, val_acc, val_recall, val_f1, lr, time_sec`.
  - Log TensorBoard tại `logs/tensorboard/deepgru_raw_nmsfree/`.

### 2.4. Phát hiện vấn đề phụ thuộc mã nguồn cần xử lý
- **Lỗi trong `src/__init__.py`**:
  - `src/__init__.py` hiện đang import các lớp cũ không còn tồn tại: `CNNAdapter`, `DeepGRUClassifier`, `dataset1`, `dataset2`.
  - Điều này khiến mọi lệnh `from src... import ...` bị chặn với `ImportError`.
  - **Giải pháp**: Cần chuẩn hóa `src/__init__.py` để xuất đúng các module hiện tại (`ConvGRUClassifier`, `ChunkedBackboneNeckExtractor`, `build_raw_video_dataloaders`), giúp `src/evaluate.py` và toàn bộ package hoạt động đồng bộ.
- **Lỗi đường dẫn import trong `src/train.py`**:
  - `src/train.py` đang import `from src.models1 import ConvGRUClassifier` và `from src.dataset2 import ...` (tên file cũ trước khi đổi tên). Cần đồng bộ trỏ về `src.models` và `src.dataset`.

---

## 3. PHÂN TÍCH THIẾT KẾ CHI TIẾT MODULE `src/evaluate.py`

### 3.1. Cấu trúc module và phân định trách nhiệm (Clean Code & Modularity)

Module `src/evaluate.py` sẽ được tổ chức theo các khối chức năng độc lập:

```text
src/evaluate.py
│
├── 1. Imports, Thiết lập môi trường & Hạt giống (seed_everything)
├── 2. Bộ nạp mô hình & Checkpoint (ModelLoader)
│   ├── load_eval_model(checkpoint_path, backbone_path, config, device)
│   └── resolve_checkpoint_path(path_or_alias, ckpt_dir)
├── 3. Bộ nạp dữ liệu Validation (DataLoaderFactory)
│   └── build_eval_dataloader(config, override_batch_size, num_workers)
├── 4. Động cơ đánh giá (EvaluationEngine)
│   ├── run_evaluation(model, extractor, val_loader, criterion, device)
│   └── compute_classification_metrics(targets, preds, probs)
├── 5. Động cơ vẽ đồ thị và trực quan hóa (VisualizerEngine)
│   ├── plot_confusion_matrix(cm, class_names, save_path, title)
│   ├── plot_stage_metrics(history_csv_or_data, save_path, best_epoch)
│   └── plot_roc_pr_curves(targets, probs, save_path)
├── 6. Bộ xuất báo cáo tổng hợp (ReportExporter)
│   ├── save_evaluation_report(metrics, save_path_json)
│   └── print_evaluation_summary(metrics)
└── 7. Giao diện CLI & Hàm chính (CLI Entrypoint)
    └── main() / evaluate_pipeline()
```

### 3.2. Chi tiết các hàm chức năng cốt lõi

#### A. Hàm `load_eval_model`
- Nhận diện định dạng checkpoint `.pt`.
- Nạp cấu hình từ `ckpt.get("config")` nếu có; nếu không, đọc từ file `configs/config.yaml`.
- Khởi tạo `ConvGRUClassifier` và nạp `state_dict`.
- Khởi tạo `ChunkedBackboneNeckExtractor` nếu đánh giá trên video thô.
- Chuyển toàn bộ mô hình về chế độ `eval()` và đóng băng gradient (`requires_grad = False`).

#### B. Hàm `run_evaluation`
- Lặp qua `val_loader` với thanh tiến trình `tqdm`.
- Sử dụng `ChunkedBackboneNeckExtractor` để trích xuất đặc trưng an toàn theo từng mini-chunk.
- Chạy qua `ConvGRUClassifier`, thu thập:
  - `all_losses`: Loss của từng batch.
  - `all_preds`: Mảng nhãn dự đoán `[N]`.
  - `all_targets`: Mảng nhãn thực tế `[N]`.
  - `all_probs`: Mảng xác suất dự đoán `[N, C]`.
  - Đo thời gian suy luận trung bình trên mỗi video clip (Latency ms/clip).

#### C. Hàm `plot_confusion_matrix`
- Vẽ ma trận nhầm lẫn kích thước 2x2 cho 2 lớp:
  - `0`: Tỉnh táo (Alert)
  - `1`: Buồn ngủ (Drowsy)
- Hiển thị cả 2 thông số trong từng ô: Số lượng mẫu thực tế và Tỷ lệ phần trăm trên hàng (% Recall).
- Bổ sung bảng thông số tóm tắt: Accuracy, Macro F1, Drowsy Recall ngay bên cạnh đồ thị.
- Lưu trữ độ phân giải cao (DPI = 300) vào thư mục chỉ định.

#### D. Hàm `plot_stage_metrics` (So sánh qua mỗi giai đoạn huấn luyện)
- Đọc tệp nhật ký `training_history.csv` (chứa dữ liệu qua các epoch).
- Vẽ lưới biểu đồ 4 ô (2x2 Subplots):
  1. **Loss Curve**: Train Loss vs Val Loss qua từng epoch.
  2. **Accuracy Curve**: Train Acc vs Val Acc qua từng epoch.
  3. **Recall Curve**: Train Recall vs Val Recall qua từng epoch (điểm nhấn quan trọng theo yêu cầu).
  4. **F1-Score & Learning Rate**: Train F1 vs Val F1 và đường suy giảm Learning Rate.
- Đánh dấu rõ vị trí **Best Epoch** (Epoch đạt Val F1 / Recall cao nhất) bằng đường gióng đứt nét và điểm sao nổi bật.

#### E. Chế độ mở rộng: `evaluate_stages` (Đánh giá chuỗi Checkpoints)
- Nếu người dùng cung cấp thư mục chứa nhiều checkpoint (`epoch_1.pt`, `epoch_2.pt`, ...), hệ thống có tùy chọn tự động duyệt qua từng checkpoint để chạy validation thực tế và tổng hợp biểu đồ so sánh Loss, Acc, Recall của từng giai đoạn.

### 3.3. Thiết kế Cấu Hình Tập Trung Hằng Số & Tham Số (`EvalConfig`)

Thay vì sử dụng giao diện dòng lệnh CLI (`argparse`), toàn bộ tham số được quy hoạch tập trung vào một lớp `@dataclass class EvalConfig` đặt ngay ở đầu module `src/evaluate.py`. Người dùng chỉ cần mở tệp và chỉnh sửa trực tiếp:

```python
@dataclass
class EvalConfig:
    """Nơi tập trung toàn bộ hằng số và tham số cấu hình đánh giá mô hình."""
    # Đường dẫn tệp
    checkpoint_path: str = "checkpoints/experiments/deepgru_raw_nmsfree/best.pt"
    config_yaml_path: str = "configs/config.yaml"
    history_csv_path: str = "logs/training_history.csv"
    output_dir: str = "logs/evaluation"

    # Tham số phần cứng & loader
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 16
    num_workers: int = 0
    chunk_size: int = 32

    # Tùy chọn trực quan hóa
    plot_confusion_matrix: bool = True
    plot_stage_metrics: bool = True
    plot_roc_pr: bool = True
    save_json_summary: bool = True
    eval_all_checkpoints: bool = False
    class_names: Tuple[str, str] = ("Alert", "Drowsy")
```
Khi chạy, người dùng chỉ cần thực thi:
```bash
python src/evaluate.py
```

---

## 4. KẾ HOẠCH TRIỂN KHAI VÀ XÁC THỰC

1. **Bước 1 (Hiện tại)**: Gửi tài liệu phân tích `analsys_eval_convgru.md` cho người dùng xem xét và phê duyệt.
2. **Bước 2**: Lập kế hoạch chi tiết `plan_eval_convgru.md` sau khi người dùng chấp thuận Bước 1.
3. **Bước 3**: Triển khai thực hiện:
   - Chuẩn hóa các import lỗi thời trong `src/__init__.py` và `src/train.py` để đảm bảo tính toàn vẹn hệ thống.
   - Viết hoàn chỉnh `src/evaluate.py` với đầy đủ tính năng theo đặc tả.
   - Chạy thử nghiệm thực tế (Dry-run và Full-run trên `best.pt` và `training_history.csv`).
   - Kiểm tra các tệp ảnh kết quả sinh ra (`confusion_matrix.png`, `metrics_comparison_stages.png`, `evaluation_report.json`).
   - Viết báo cáo tổng hợp `report_eval_convgru.md`.

---

## 5. CÂU HỎI & XÁC NHẬN VỚI NGƯỜI DÙNG

Kính gửi người dùng, vui lòng xác nhận các điểm sau để tiếp tục sang Bước 2:
1. Bạn có đồng ý với thiết kế và cấu trúc của `src/evaluate.py` như đã phân tích ở trên không?
2. Biểu đồ so sánh qua mỗi giai đoạn: Chúng tôi sẽ ưu tiên đọc dữ liệu epoch từ `training_history.csv` để vẽ so sánh Train vs Val (Loss, Acc, Recall), đồng thời hỗ trợ flag chạy quét kiểm định qua danh sách các file `epoch_*.pt` nếu bạn muốn. Bạn có yêu cầu bổ sung nào cho phần này không?
3. Bạn có đồng ý cho phép cập nhật đồng bộ các dòng import bị lỗi thời trong `src/__init__.py` và `src/train.py` để script hoạt động trơn tru không?
