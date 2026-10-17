# Driver Guardian — ConvGRU

Video RGB → backbone/PAFPN P4 → ConvGRU → spatial attention → temporal mean → hai logits (0: alert, 1: drowsy).

## Cấu hình và chạy

Toàn bộ giá trị mặc định nằm trong `src/config.py`. Sửa `DEFAULT_CONFIG` hoặc các giá trị mặc định của `TrainConfig` tại đó. Không dùng preset YAML cũ.

- `backbone_mode="scratch"`, `backbone_checkpoint=None`: train backbone/neck từ đầu và ConvGRU; BatchNorm cập nhật thống kê.
- `backbone_mode="frozen"`, `backbone_checkpoint="/path/to/detector_or_landmark.pt"`: nạp backbone/neck (ưu tiên EMA), giữ nguyên weights và BatchNorm ở eval; chỉ train adapter, ConvGRU, spatial attention và classifier.

`backbone_mode` chỉ nhận `scratch` hoặc `frozen`; `frozenv1` là tên thư mục run trong `save_dir`, không phải mode. Mặc định hiện tại là `frozen`; khi chuyển sang `scratch`, cần đặt cả `backbone_checkpoint=None`.

Mặc định dùng checkpoint `landmark_finetune2/best.pt` đã có trong repository và lưu run mới tại `ai/LSTM/checkpoints/head_landmarkv1`. Backbone/neck luôn ở `eval()`, kể cả khi gọi `model.train()`, và chạy không có gradient. Phần phía sau được khởi tạo mới để học trên cùng cách trích xuất đặc trưng lúc inference. Không resume run `scratchv1` để chuyển chế độ; resume luôn giữ cấu hình của run cũ.

Có thể truyền tham số trực tiếp trong Python hoặc notebook:

```python
from ai.LSTM.src.config import load_config
from ai.LSTM.src.training import train

cfg = load_config()
# Sửa cfg.backbone_checkpoint nếu checkpoint nằm ở đường dẫn khác.
cfg.save_dir = "ai/LSTM/checkpoints/head_landmarkv1"
model, metrics = train(cfg)
```

Mỗi thử nghiệm cần `save_dir` riêng. Thư mục có sẵn dữ liệu sẽ bị từ chối trừ khi resume. Cấu hình kiến trúc backbone của chế độ frozen được đọc từ checkpoint pretrained.

```bash
python -m ai.LSTM.src.training
python -m ai.LSTM.src.training --config path/to/config.json
python -m ai.LSTM.src.training --resume ai/LSTM/checkpoints/frozenv1/last.pt
```

Resume khôi phục weights, optimizer, scheduler, AMP scaler, best weights và RNG tại cuối epoch. Giữ nguyên tổng số epochs và các tham số huấn luyện để không đổi lịch LR. CLI resume lấy cấu hình từ checkpoint, không cần file pretrained gốc. Không hỗ trợ resume giữa epoch; epoch chưa lưu hoàn chỉnh sẽ được chạy lại. Muốn tái lập chính xác cần giữ nguyên môi trường, thiết bị và số worker; kernel GPU có thể không xác định hoàn toàn.

Dataset dùng `dataset_merged_split.csv` với các cột `processed_path`, `binary_label`, `clip_id`, `split`, `subject_id`, hoặc thư mục `train|val|test/0_alert|1_drowsy/*.mp4`. Manifest trên máy chủ cũng được hỗ trợ với các cột `rel_path`, `label`, `video_id`, `split` và tùy chọn `subject_id`, `orig_path`. `rel_path` trỏ tới clip đã xử lý, tương đối với `dataset_dir`; `orig_path` dùng làm định danh video gốc để kiểm tra leakage, không dùng để mở clip. Đường dẫn Windows được đổi dấu phân cách khi đọc trên Linux. Train/val/test phải khác subject. Video quá ngắn hoặc FPS thấp bị loại trước khi train; lỗi decode khi đọc sẽ dừng run và ghi traceback vào log.

Mặc định lấy cửa sổ 10 giây ở 5 FPS (50 frame), RGB uint8, letterbox 224 px rồi chia 255 trong model. Train chọn cửa sổ ngẫu nhiên thiên về cuối video; val/test lấy cửa sổ giữa cố định. Augmentation nhất quán giữa các frame, chỉ dùng cho train.

Gradient accumulation chuẩn hóa theo số mẫu thực tế, kể cả batch cuối nhỏ hơn. `gradient_checkpointing` không dùng khi scratch vì recompute sẽ cập nhật BatchNorm lần thứ hai; frozen không cần giữ đồ thị backbone. `chunk_size` giới hạn số frame đi qua backbone trong mỗi lượt forward.

## Artifacts của một run

```text
save_dir/
├── config.json       # Cấu hình đã dùng
├── train.log         # Cấu hình, kích thước tập, metrics 10 lần/epoch, lỗi và kết quả test
├── last.pt           # Đầy đủ trạng thái để resume, bao gồm best weights
├── best.pt           # Model tốt nhất theo val F1, hòa F1 thì chọn val loss thấp hơn
├── tensorboard/      # Event files cho dashboard training
├── test_metrics.json
└── evaluation_test/  # Khi chạy script evaluation
```

Checkpoint chứa phiên bản định dạng, kiến trúc thực tế, trọng số và cấu hình preprocessing. `last.pt` giữ thêm best weights để có thể tái tạo `best.pt` khi bị gián đoạn giữa hai lần ghi. Các file `.pt` được ghi tạm rồi thay thế nguyên tử; không ghi đè trực tiếp file đang dùng. `train.log` được append khi resume; có thể chứa thông báo của epoch bị gián đoạn, epoch trong `last.pt` mới là mốc khôi phục.

Checkpoint cũ thiếu metadata bị từ chối rõ ràng; cần chuyển đổi với kiến trúc/preprocessing gốc, không tự đoán cấu hình.

```python
from ai.LSTM.src.checkpoint import load_model

model, checkpoint = load_model("ai/LSTM/checkpoints/frozenv1/best.pt", device="cpu")
```

## Đánh giá và export

Evaluation mặc định đọc **test**, dùng kiến trúc và preprocessing đã lưu:

```bash
python -m ai.LSTM.src.evaluation --checkpoint ai/LSTM/checkpoints/frozenv1/best.pt
```

Có thể dùng `--dataset-dir`, `--device`, `--output-dir`; `--split val` chỉ dùng khi muốn đánh giá validation rõ ràng. Output gồm `evaluation.log`, `metrics.json` (có tên split) và `predictions.csv`.

```bash
python -m ai.LSTM.export_vulkan --checkpoint ai/LSTM/checkpoints/frozenv1/best.pt
```

Export dùng cả backbone và ConvGRU trong cùng checkpoint; kích thước ảnh/cửa sổ được đọc từ cấu hình đã lưu. ExecuTorch/Vulkan là phụ thuộc tùy chọn; `--cpu` bỏ delegate Vulkan.

Notebook 02/03/04 dùng chung `src/config.py` và `training.train()`. Trên Kaggle, mount toàn bộ repository (bao gồm `ai/ObjectDetection_2p6M`), đặt `DRIVER_GUARDIAN_ROOT` nếu cần và sửa đường dẫn dữ liệu/output trong config.

## Kiểm tra

```bash
python -m pytest ai/LSTM/tests -q
```

Test bao gồm hai chế độ training, checkpoint tự chứa, resume sau gián đoạn, ghi checkpoint lỗi, gradient accumulation với batch cuối nhỏ và evaluation trên đúng split. Test nhỏ không thay thế việc đánh giá hội tụ trên dữ liệu thật.

## TensorBoard

Training ghi metric tích lũy vào `train.log` và flush TensorBoard 10 lần mỗi epoch, chia đều theo số batch (dưới 10 batch thì ghi mỗi batch). Validation và biểu đồ tổng kết ghi cuối epoch. Trục step TensorBoard dùng 10 bước/epoch; step 10 là cuối epoch 1.

Training tự động ghi vào `<save_dir>/tensorboard`, dùng epoch làm trục bước:

```bash
tensorboard --logdir ai/LSTM/checkpoints --port 6006
```

Mở `http://localhost:6006`. Nếu đổi `save_dir`, trỏ `--logdir` vào thư mục run hoặc thư mục cha để so sánh nhiều run.

- Scalars: loss, accuracy (`acc`), precision, recall, F1, số mẫu, ROC-AUC, average precision cho train/val/test; learning rate, AMP scale, gradient norm trước clip, thời gian và tốc độ mỗi lượt.
- Images và PR Curves: confusion matrix, ROC, precision–recall theo ngưỡng. ROC/PR và AUC chỉ ghi khi split có đủ hai lớp.
- Histograms: xác suất drowsy theo nhãn thật, trọng số trainable cuối epoch, gradient sau clip ở bước cập nhật cuối epoch. Gradient norm là trung bình các norm hữu hạn trước clip trong epoch.
- Text: cấu hình run. Test chạy trên best checkpoint, được ghi ở bước `epochs + 1`; `test/checkpoint_epoch` chỉ epoch của checkpoint đó.

Writer flush mỗi epoch và đóng kể cả khi training lỗi. Resume giữ cùng thư mục log, ẩn các bước sau epoch đã lưu bằng `purge_step` của [PyTorch SummaryWriter](https://docs.pytorch.org/docs/stable/tensorboard.html). Thu thập dự đoán chỉ giữ nhãn, xác suất và lớp dự đoán trên CPU; không giữ video hoặc đồ thị gradient.
