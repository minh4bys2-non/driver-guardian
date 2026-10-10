# TÀI LIỆU PHÂN TÍCH YÊU CẦU: BENCHMARK MÔ HÌNH MODELINFERENCE (SRC/EVALUATE.PY)

- **Mã yêu cầu**: `BENCHMARK_MODEL_INFERENCE`
- **Tệp phân tích**: `docs/analsys/analsys_benchmark_model.md`
- **Mục tiêu**: Tái cấu trúc và nâng cấp `src/evaluate.py` để benchmark toàn diện mô hình `ModelInference` (từ `model_inference.py`), đo lường hệ thống chỉ số phân loại (Accuracy, Recall, F1, F2), đánh giá tốc độ/thời gian suy luận (Latency, FPS) với các độ dài chuỗi khung hình khác nhau, đồng thời hỗ trợ lựa chọn linh hoạt nhiều bộ dữ liệu và tập chia khác nhau.
- **Trạng thái**: Đang chờ người dùng phê duyệt (Bước 1 theo quy trình chuẩn `AGENTS.md`).

---

## 1. YÊU CẦU CỐT LÕI TỪ NGƯỜI DÙNG

Người dùng yêu cầu:
> *"thực hiện sửa đổi file @src\evaluate.py để thực hiện benmark model ModelInference của file @[model_inference.py] về các chỉ số như acc, recal, F1, F2, tốc độ chạy của model với các mẫu có độ dài ngắn khác nhau, có thể tự do chọn các bộ dữ liệu khác nhau"*

### Phân rã các mục tiêu kỹ thuật cụ thể:
1. **Benchmark trực tiếp mô hình `ModelInference` (`model_inference.py`)**:
   - Thay thế việc đánh giá tách rời trước đây bằng việc đánh giá trực tiếp pipeline suy luận đầu-cuối `ModelInference` (kết hợp `NMSFreeDetector` Backbone/Neck và `ConvGRUClassifier`).
   - Hỗ trợ nạp checkpoints linh hoạt: `cnn_path`, `conv_gru_path`, tham số chia nhỏ khung hình `chunk_size` và thiết bị tính toán `device` (`cuda` / `cpu`).
   - Xử lý mượt mà đầu vào video `[T, 3, H, W]` (cả dạng `torch.uint8` và `torch.float32`).

2. **Hệ thống chỉ số đánh giá toàn diện (Acc, Recall, F1, F2)**:
   - **Độ chính xác (Accuracy)**: Tỷ lệ dự đoán đúng trên toàn bộ tập mẫu kiểm định.
   - **Độ nhạy (Recall)**: Macro Recall và Per-class Recall (đặc biệt quan trọng với lớp **Drowsy - Buồn ngủ** để tránh bỏ sót rủi ro tai nạn).
   - **F1-Score**: Trung bình điều hòa cân bằng giữa Precision và Recall.
   - **F2-Score ($\beta = 2$)**: 
     $$F_2 = (1 + 2^2) \frac{\text{Precision} \cdot \text{Recall}}{2^2 \cdot \text{Precision} + \text{Recall}} = 5 \cdot \frac{\text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$$
     Trong bài toán phát hiện buồn ngủ của tài xế (Driver Guardian), việc bỏ sót tài xế buồn ngủ (**False Negative**) mang tính chất nguy hiểm tính mạng cao hơn rất nhiều so với việc cảnh báo nhầm (**False Positive**). Do đó, chỉ số **F2-Score** coi trọng Recall gấp đôi Precision, là chỉ số then chốt để nghiệm thu hệ thống thực tế.
   - **Precision, Ma trận nhầm lẫn (Confusion Matrix), ROC-AUC và PR-AUC**: Đảm bảo phân tích chi tiết toàn diện.
   - **Phân tích ngưỡng quyết định (Decision Threshold Analysis)**: Khảo sát sự biến thiên của F1 và F2 theo các ngưỡng từ 0.05 đến 0.95 để tìm ra ngưỡng tối ưu nhất cho bài toán an toàn.

3. **Đo lường và khảo sát tốc độ với các mẫu có độ dài chuỗi khác nhau**:
   - Đo lường Latency trên mỗi mẫu video (ms/clip), Latency trên mỗi khung hình (ms/frame) và Throughput (FPS - Khung hình/giây).
   - **Khảo sát độ dài trên dữ liệu thực tế (Dataset Sequence Length Grouping)**:
     - Phân loại các video trong dataset theo các nhóm độ dài (ví dụ: $T \le 16$, $16 < T \le 32$, $32 < T \le 64$, $T > 64$, hoặc theo từng độ dài tự nhiên của clip).
     - Thống kê chi tiết Latency, FPS, Acc, Recall, F1, F2 riêng biệt cho từng nhóm độ dài để phân tích độ trễ và độ chính xác biến thiên theo thời lượng quan sát.
   - **Khảo sát tốc độ có kiểm soát (Controlled / Synthetic Sequence Length Sweep)**:
     - Quét qua dải độ dài chuỗi định sẵn: ví dụ $T \in [8, 16, 24, 32, 48, 64, 96, 128]$.
     - Thực hiện chạy khởi động GPU (warm-up) và đo lặp lại nhiều lần với đồng bộ hóa phần cứng `torch.cuda.synchronize()` để tính giá trị trung bình (mean), độ lệch chuẩn (std), tốc độ xử lý FPS và dung lượng VRAM tiêu thụ đỉnh (Peak VRAM in MB).

4. **Tự do lựa chọn linh hoạt các bộ dữ liệu khác nhau**:
   - Cho phép người dùng dễ dàng chuyển đổi cấu hình bộ dữ liệu:
     - `dataset_dir`: Chỉ định thư mục bộ dữ liệu bất kỳ (ví dụ: `E:/LSTM/UL-DD_Processed`, thư mục dataset Kaggle, hoặc thư mục video nội bộ).
     - `manifest_file`: Chỉ định file manifest CSV/JSON bất kỳ (ví dụ: `dataset_merged_split.csv`).
     - `split`: Tự do chọn tập đánh giá (`"val"`, `"test"`, `"train"`, `"all"`).
     - `dataset_source`: Bộ lọc tập dữ liệu con (`"all"`, `"ul-dd"`, `"uta-rldd"`, `"sust"`).
     - `seq_len`: Cho phép giữ nguyên độ dài tự nhiên của video (`None`) hoặc cố định số khung hình ($T$).
     - `sample_interval`: Khoảng cách lấy mẫu khung hình theo thời gian (giây).
     - `max_samples`: Giới hạn số mẫu đánh giá nhanh (dry-run) hoặc chạy toàn bộ tập dữ liệu.
     - **Chế độ kiểm thử độc lập (Synthetic Speed Benchmark Mode)**: Có khả năng chạy đo đạc tốc độ thuần túy ngay cả khi máy tính hiện tại chưa gắn ổ cứng chứa video, giúp kiểm tra hiệu năng phần cứng tức thì.

5. **Trực quan hóa và xuất báo cáo chuyên nghiệp**:
   - Sinh ảnh ma trận nhầm lẫn `confusion_matrix.png` hiển thị đầy đủ số lượng mẫu, tỷ lệ chuẩn hóa (%) và hộp tóm tắt các chỉ số Acc, Recall, F1, F2.
   - Sinh biểu đồ đa trục khảo sát tốc độ theo độ dài `benchmark_speed_vs_length.png` (Latency vs Length, FPS vs Length, ms/frame vs Length, VRAM vs Length).
   - Sinh đồ thị đường cong ROC và PR `roc_pr_curves.png`.
   - Sinh biểu đồ phân tích ngưỡng tối ưu `threshold_analysis.png` cho F1 và F2.
   - Xuất tệp báo cáo tổng hợp chi tiết `benchmark_summary.json` và bảng hiển thị tổng kết trên giao diện dòng lệnh.

---

## 2. KHẢO SÁT HIỆN TRẠNG MÃ NGUỒN VÀ MÔI TRƯỜNG THỰC THI

### 2.1. Lớp `ModelInference` (`model_inference.py`)
- Định nghĩa:
  ```python
  class ModelInference(nn.Module):
      def __init__(
          self,
          cnn_model: Optional[NMSFreeDetector] = None,
          conv_gru_model: Optional[ConvGRUClassifier] = None,
          chunk_size: int = 32,
          cnn_path: Optional[Union[str, Path]] = None,
          conv_gru_path: Optional[Union[str, Path]] = None,
          device: Optional[Union[str, torch.device]] = None,
      ):
          ...
      def forward(self, x: Union[np.ndarray, torch.Tensor]) -> float:
          ...
  ```
- **Đặc điểm hoạt động**:
  - Nhận đầu vào là chuỗi khung hình video `x` có kích thước 4 chiều: `[T, 3, H, W]`.
  - Hỗ trợ cả định dạng `uint8` [0, 255] và `float` [0.0, 1.0]. Tự động resize/letterbox về `640x640` nếu cần.
  - Sử dụng cơ chế chia nhỏ `chunk_size` (mặc định 32) khi đưa qua Backbone & Neck của `NMSFreeDetector` để chống tràn bộ nhớ VRAM.
  - Ghép các đặc trưng không gian `(p3, p4, p5)` và đẩy qua `ConvGRUClassifier`.
  - Trả về trực tiếp một giá trị `float` là xác suất buồn ngủ trong khoảng $[0.0, 1.0]$.
- **Checkpoints mặc định đã xác thực tồn tại trên hệ thống**:
  - CNN: `D:\Project\DATN\driver-guardian\ai\checkpoints\model_cnn\best.pt` (Tồn tại)
  - ConvGRU: `D:\Project\DATN\driver-guardian\ai\checkpoints\model_convgru\best.pt` (Tồn tại)
  - Thư mục phụ trợ: `checkpoints/experiments/deepgru_raw_nmsfree/best.pt` (Tồn tại)

### 2.2. Vấn đề môi trường và `sys.path`
- Khi nạp `NMSFreeDetector`, module này thực hiện `from utils.artifacts import validate_metadata`.
- Do đó, để import trơn tru mà không phát sinh `ModuleNotFoundError: No module named 'utils'`, file thực thi `src/evaluate.py` cần chủ động thêm các đường dẫn sau vào `sys.path`:
  - Thư mục gốc dự án: `driver-guardian`
  - Thư mục module Object Detection: `driver-guardian/ai/ObjectDetection_2p6M`
  - Thư mục module LSTM: `driver-guardian/ai/LSTM`

### 2.3. Khảo sát phần cứng
- GPU phát hiện: **NVIDIA GeForce RTX 3050 Laptop GPU** (hỗ trợ CUDA).
- Chức năng đo lường cần sử dụng `torch.cuda.synchronize()` trước và sau mỗi lượt đo thời gian để kết quả Latency và FPS phản ánh chính xác hiệu năng GPU thực tế.

### 2.4. Khảo sát dữ liệu thực tế
- Thư mục dữ liệu: `E:\LSTM\UL-DD_Processed` đã sẵn sàng với:
  - File manifest: `dataset_merged_split.csv` (2,319 clip video).
  - Tập `val`: 352 video clips.
  - Tập `test`: 352 video clips.
  - Tập `train`: 1,615 video clips.
  - Cân bằng nhãn: 1,166 Alert (Nhãn 0) và 1,153 Drowsy (Nhãn 1).
- Module `src/dataset.py` đã cung cấp sẵn lớp `RawVideoFramesDataset` đọc trực tiếp video thô thành tensor `[T, 3, H, W]` dạng `uint8`, hoàn toàn khớp với đầu vào của `ModelInference`.

---

## 3. THIẾT KẾ CHI TIẾT MODULE `SRC/EVALUATE.PY`

### 3.1. Cấu trúc module đề xuất

```text
src/evaluate.py
│
├── 1. Môi trường, Path Setup & Hạt giống (seed_everything, sys.path)
├── 2. Cấu hình tập trung nâng cao (EvalBenchmarkConfig)
├── 3. Bộ nạp mô hình ModelInference (load_benchmark_model)
├── 4. Bộ nạp dữ liệu linh hoạt (Dataset & DataLoader Factory)
├── 5. Động cơ đánh giá chỉ số phân loại (Classification Metrics Engine: Acc, Recall, F1, F2)
├── 6. Động cơ đo lường tốc độ theo độ dài (Sequence Length Speed Profiler)
├── 7. Động cơ phân tích theo nhóm độ dài thực tế (Dataset Binning Engine)
├── 8. Động cơ trực quan hóa đồ thị (Visualization Engine)
│   ├── plot_confusion_matrix (Heatmap + Acc/Rec/F1/F2 stats)
│   ├── plot_speed_vs_length (Multi-panel Latency, FPS, ms/frame, VRAM)
│   ├── plot_roc_pr_curves (ROC-AUC & Precision-Recall)
│   └── plot_threshold_analysis (F1 vs F2 curve theo decision threshold)
├── 9. Bộ xuất báo cáo (JSON Summary Exporter & Rich Console Formatter)
└── 10. Điểm thực thi chính (evaluate_benchmark() & main())
```

### 3.2. Thiết kế Lớp Cấu Hình Tập Trung `EvalBenchmarkConfig`

Toàn bộ tham số được tập trung trong `@dataclass class EvalBenchmarkConfig` đặt ngay đầu file, cho phép người dùng cấu hình nhanh chóng hoặc truyền tham số khi import:

```python
@dataclass
class EvalBenchmarkConfig:
    # ---- 1. CHECKPOINT & MÔ HÌNH ----
    cnn_checkpoint_path: str = r"D:\Project\DATN\driver-guardian\ai\checkpoints\model_cnn\best.pt"
    conv_gru_checkpoint_path: str = r"D:\Project\DATN\driver-guardian\ai\checkpoints\model_convgru\best.pt"
    chunk_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 2. CẤU HÌNH BỘ DỮ LIỆU TÙY CHỌN ----
    # Người dùng có thể tự do trỏ đến bất kỳ thư mục dữ liệu nào
    dataset_dir: str = r"E:\LSTM\UL-DD_Processed"
    manifest_file: Optional[str] = "dataset_merged_split.csv"
    # Lựa chọn split: 'val', 'test', 'train', hoặc 'all'
    split: str = "val"
    # Lọc tập nguồn: 'all', 'ul-dd', 'uta-rldd', 'sust'
    dataset_source: str = "all"
    # Khoảng cách lấy mẫu khung hình theo giây (0.2s tương đương 5 FPS)
    sample_interval: float = 0.2
    # Cố định độ dài khung hình (None = giữ nguyên độ dài clip thực tế)
    seq_len: Optional[int] = None
    # Giới hạn số mẫu đánh giá (None = đánh giá toàn bộ tập; int > 0 = kiểm tra nhanh)
    max_eval_samples: Optional[int] = None

    # ---- 3. BENCHMARK TỐC ĐỘ THEO ĐỘ DÀI MẪU (SPEED PROFILING) ----
    # Bật/tắt benchmark tốc độ có kiểm soát qua các độ dài chuỗi
    run_synthetic_speed_benchmark: bool = True
    # Danh sách các độ dài chuỗi cần benchmark tốc độ
    benchmark_seq_lengths: Tuple[int, ...] = (8, 16, 24, 32, 48, 64, 96, 128)
    # Kích thước khung hình benchmark chuẩn
    benchmark_img_size: int = 640
    # Số lượt chạy làm ấm GPU
    warmup_runs: int = 3
    # Số lượt lặp đo thời gian cho mỗi độ dài
    repeat_runs: int = 10

    # ---- 4. THRESHOLD & CHỈ SỐ ----
    # Ngưỡng phân loại mặc định (xác suất >= threshold -> Drowsy)
    classification_threshold: float = 0.5
    # Tên hiển thị các nhãn lớp
    class_names: Tuple[str, str] = ("Alert (Tỉnh táo)", "Drowsy (Buồn ngủ)")
    # Hạt giống ngẫu nhiên để tái lập kết quả
    seed: int = 42

    # ---- 5. XUẤT KẾT QUẢ & ĐỒ THỊ ----
    output_dir: str = "logs/benchmark"
    plot_confusion_matrix: bool = True
    plot_speed_curves: bool = True
    plot_roc_pr: bool = True
    plot_threshold_curves: bool = True
    save_json_summary: bool = True
```

### 3.3. Chi tiết thuật toán và công thức tính toán chỉ số

#### A. Công thức F2-Score
Với độ nhạy được ưu tiên gấp đôi:
$$F_2 = (1 + 2^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{2^2 \cdot \text{Precision} + \text{Recall}} = 5 \cdot \frac{\text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$$
- Hàm tính toán `compute_benchmark_metrics` sẽ xuất cả:
  - Macro F2 & Per-class F2 (lớp Alert và lớp Drowsy).
  - Tương tự với F1, Precision, Recall và Accuracy.

#### B. Phân tích tối ưu ngưỡng quyết định (Threshold Tuning)
- `ModelInference` trả về xác suất liên tục $P(\text{Drowsy}) \in [0.0, 1.0]$.
- Hệ thống sẽ quét qua 100 ngưỡng từ 0.01 đến 0.99, tính toán F1 và F2 tại mỗi ngưỡng, tự động tìm ra:
  - $\text{Threshold}^*_{F1}$: Ngưỡng đạt F1-Score cao nhất.
  - $\text{Threshold}^*_{F2}$: Ngưỡng đạt F2-Score cao nhất (ngưỡng an toàn thực tế).
- Xuất biểu đồ `threshold_analysis.png` trực quan hóa đường cong biến thiên của F1 và F2.

#### C. Đo lường tốc độ theo độ dài (Speed vs Sequence Length)
- Với mỗi độ dài $T \in \text{benchmark\_seq\_lengths}$:
  - Tạo dummy input: `torch.zeros((T, 3, 640, 640), dtype=torch.uint8)` hoặc tensor ngẫu nhiên.
  - Chạy `warmup_runs` lần để nạp cache GPU.
  - Khởi tạo CUDA Events hoặc `time.perf_counter()` kết hợp `torch.cuda.synchronize()`.
  - Đo `repeat_runs` lần để tính:
    - `mean_latency_ms`: Thời gian trung bình xử lý 1 clip dài $T$ frames.
    - `std_latency_ms`: Độ lệch chuẩn thời gian.
    - `latency_per_frame_ms`: `mean_latency_ms / T`.
    - `throughput_fps`: Tốc độ xử lý khung hình trên giây ($T \times 1000.0 / \text{mean\_latency\_ms}$).
    - `peak_vram_mb`: Dung lượng bộ nhớ GPU tiêu thụ tối đa (`torch.cuda.max_memory_allocated() / (1024 * 1024)`).

#### D. Phân tích phân nhóm theo độ dài thực tế trong Dataset (Dataset Length Binning)
- Khi chạy trên tập dữ liệu video thực tế (`val` hoặc `test`), mỗi video có độ dài khung hình $T_i$ thực tế.
- Hệ thống gom nhóm mẫu theo các khoảng độ dài:
  - Nhóm 1: $T \le 16$
  - Nhóm 2: $16 < T \le 32$
  - Nhóm 3: $32 < T \le 64$
  - Nhóm 4: $T > 64$
- Với mỗi nhóm, xuất ra bảng thống kê: Số mẫu, Accuracy, Recall Drowsy, F1, F2, Latency trung bình và FPS.

---

## 4. KẾ HOẠCH BÀN GIAO VÀ KIỂM THỬ XÁC THỰC

1. **Kiểm thử đơn vị nạp mô hình & dữ liệu**:
   - Khởi tạo thành công `ModelInference` từ cả CPU và CUDA GPU.
   - Thử nghiệm tải mẫu từ `RawVideoFramesDataset` với các split (`val`, `test`).
2. **Kiểm thử đo lường tốc độ theo độ dài**:
   - Chạy synthetic speed sweep với $T \in [8, 16, 24, 32, 48, 64, 96, 128]$.
   - Kiểm tra sinh ảnh biểu đồ `benchmark_speed_vs_length.png`.
3. **Kiểm thử đánh giá độ chính xác trên tập dữ liệu thực tế**:
   - Chạy trên tập `val` (352 clips) hoặc tập con để tính Acc, Recall, F1, F2, Latency.
   - Kiểm tra sinh các biểu đồ `confusion_matrix.png`, `roc_pr_curves.png`, `threshold_analysis.png`.
   - Kiểm tra tính đầy đủ của file `benchmark_summary.json`.
4. **Kiểm thử tương thích CLI & Python API**:
   - Thực thi trực tiếp `python src/evaluate.py`.
   - Hỗ trợ import `from src.evaluate import EvalBenchmarkConfig, evaluate_benchmark` vào Jupyter Notebook (`test.ipynb`) một cách thuận tiện.

---

## 5. CÂU HỎI & XÁC NHẬN VỚI NGƯỜI DÙNG

Kính gửi người dùng, vui lòng xác nhận các điểm sau để hoàn tất **Bước 1** và chuyển sang **Bước 2 (Lên kế hoạch thực hiện)** theo quy trình `AGENTS.md`:

1. **Về cấu trúc chỉ số**: Bạn có đồng ý với cách tính toán và hiển thị các chỉ số **Acc, Recall, F1, F2** (đặc biệt là F2 nhấn mạnh Recall cho lớp Buồn ngủ) và phân tích ngưỡng tối ưu như đã mô tả không?
2. **Về kiểm thử tốc độ theo độ dài**: Bạn có đồng ý kết hợp cả 2 hình thức:
   - (a) Đo đạc tốc độ có kiểm soát qua các độ dài cố định $[8, 16, 24, 32, 48, 64, 96, 128]$ kèm vẽ biểu đồ đa trục.
   - (b) Gom nhóm và thống kê độ chính xác/tốc độ theo từng khoảng độ dài thực tế của video trong dataset?
3. **Về việc lựa chọn bộ dữ liệu**: Bạn có đồng ý với cơ chế cấu hình tập trung `EvalBenchmarkConfig` cho phép tùy biến trực tiếp `dataset_dir`, `manifest_file`, `split` (`val`/`test`/`train`), và `dataset_source` không?
