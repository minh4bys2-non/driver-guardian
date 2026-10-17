# Face landmark transfer learning

Chỉnh trực tiếp `config_lmk.py`, không có CLI hoặc tham số dòng lệnh.

Cấu hình train hiện tại cho máy cá nhân: dữ liệu
`/run/media/tranmanhduy/Data/DataTransferOriginal/{train,val}`, nhãn
`annotations.jsonl`, batch size 2, 2 worker, prefetch 1, ảnh 480×480 và AMP.
Trunk khớp checkpoint object detection `train/best.pt` sẵn trong repository.
Thiết bị ưu tiên CUDA; nếu CUDA không khả dụng, Trainer chuyển sang CPU.

## Các file

| File | Vai trò |
| --- | --- |
| `config_lmk.py` | `TrainConfig`, `FineTuneConfig`, kiến trúc, dữ liệu, Wing Loss, log và checkpoint |
| `train_lmk.py` | Trainer dùng chung: train hai stage / fine-tune một stage, LR, AMP, EMA, resume |
| `finetune.py` | Chạy Trainer với FineTuneConfig |
| `checkpoint.py` | Hash, metadata, config snapshot, lưu nguyên tử, khôi phục trạng thái |
| `model_lmk.py` | Backbone/neck pretrained từ object detection và head landmark |
| `loss_lmk.py` | Detection loss + Wing Loss |
| `dataloader_lmk.py` | JSONL, augmentation, split, lấy batch có thể resume |
| `mediapipe_478.py` | Thứ tự hoán vị 478 landmark khi lật ngang |
| `inference.py` | Đọc kiến trúc từ checkpoint, suy luận ảnh/camera |
| `tests/test_pipeline.py` | Loss, train, resume giữa epoch, fine-tune và xác minh metadata |

## Train

Điền trong `TrainConfig`:

```python
trunk_ckpt = '/path/to/object_detection/best.pt'
train_root_dir = '/data/faces/train'
val_root_dir = '/data/faces/val'
jsonl_name = 'annotations.jsonl'
run_name = 'landmark_train'
resume = ''
```

Checkpoint nguồn phải đi kèm `architecture.json` và `categories.json` do pipeline
object detection tạo. Backbone/neck/strides trong config phải khớp nguồn.
Chỉ nạp backbone + neck, ưu tiên trọng số EMA; head landmark được khởi tạo mới.

Chạy `train_lmk.py` bằng chức năng Run module của IDE, hoặc từ gốc repository:

```bash
python -m transfer.train_lmk
```

Mặc định: 5 epoch đóng băng trunk, 45 epoch mở toàn bộ model. Sửa `stage1`,
`stage2` để đổi số epoch, LR và warmup. LR được tính theo epoch/batch bằng warmup
và cosine decay, không cần scheduler state riêng.

## Checkpoint và log theo hash

Dùng cùng SHA-256 canonical JSON và cách chia thư mục với object detection:

```text
checkpoints/<architecture_sha256>/<categories_sha256>/<run_name>/
├── architecture.json
├── categories.json
├── config.json
├── best.pt
├── last.pt
├── ckpt_step00005000.pt
├── stage1_head_only_final.pt
└── stage2_finetune_final.pt

logs/<architecture_sha256>/<categories_sha256>/<run_name>/<timestamp>.log
runs/<architecture_sha256>/<categories_sha256>/<run_name>/events.out.tfevents...
```

Hash kiến trúc bao gồm trunk, head, số landmark và cách mã hóa tọa độ; categories
là class `face`. Đổi learning rate không đổi hash kiến trúc. Dùng `run_name` khác
cho mỗi thí nghiệm; run mới không ghi đè thư mục đã tồn tại.

`config.json` giữ cấu hình ban đầu; mỗi checkpoint chứa cấu hình tại lúc lưu.
Checkpoint được ghi qua file tạm rồi thay thế nguyên tử. `save_ckpt_interval_steps`
quyết định chu kỳ lưu giữa epoch (0 để tắt); cuối mỗi epoch luôn lưu `last.pt`.
`ckpt_keep_last` giữ số checkpoint định kỳ gần nhất, không xóa best/cuối stage.
`best.pt` được chọn theo validation loss; nếu không có validation sẽ không tạo best.

Tần suất mặc định cho dataset lớn (batch size 2):

| Hoạt động | Chu kỳ |
| --- | --- |
| Loss/LR vào file và TensorBoard | 500 step |
| Thành phần loss, gradient norm, bộ nhớ GPU | 2.500 step |
| Checkpoint định kỳ + last.pt | 5.000 step, giữ 2 bản định kỳ gần nhất |
| Validation toàn bộ val, cập nhật best.pt | 20.000 step và cuối mỗi epoch |
| Thanh tiến trình terminal | Khoảng 1 giây/lần, không in INFO |

Validation và lưu định kỳ không lặp lại ở batch cuối epoch. Stage 1 giữ nguyên
5 epoch đóng băng backbone/neck; stage 2 có 45 epoch với LR head `3e-4`,
trunk `3e-5` trước khi áp dụng warmup/cosine. Đổi tần suất log/save/validation
trong step được phép khi resume; không thay đổi training plan hoặc dataset.
Tiến trình đang chạy chỉ nhận cấu hình mới khi khởi động lại từ checkpoint.

## Resume

Trong **đúng config đã dùng**, điền:

```python
resume = '/path/to/<architecture_sha256>/<categories_sha256>/<run_name>/last.pt'
```

Giữ `run_name`, `ckpt_dir`, dataset, batch size và training plan, rồi chạy lại
module train hoặc fine-tune tương ứng. Không cần giữ file pretrained gốc.

Khôi phục model, optimizer, GradScaler, EMA/update count, RNG Python/NumPy/Torch,
epoch, batch kế tiếp, global step và best validation loss. Checkpoint giữa epoch
được tiếp tục từ batch kế tiếp; checkpoint cuối epoch bắt đầu epoch mới.

Thứ tự shuffle và augmentation được xác định theo seed/epoch/sample, kể cả khi
worker nạp trước dữ liệu. Checkpoint kiểm tra hash annotation JSONL và chỉ số split
để từ chối resume khi nhãn hoặc cách chia dữ liệu đổi. Nội dung file ảnh không được
băm; cần giữ nguyên ảnh trong suốt run. Độ tái lập số học giữa các loại GPU/phần mềm
khác nhau vẫn phụ thuộc PyTorch.

Có thể đổi thiết bị, số worker, nơi ghi log và tần suất ghi/lưu. Mỗi phiên có file
log riêng; TensorBoard dùng lại thư mục run và loại event thừa sau checkpoint.

## Fine-tune

Điền trong `FineTuneConfig` ở cùng `config_lmk.py`:

```python
pretrained_ckpt = '/path/to/landmark_train/best.pt'
train_root_dir = '/data/new_faces'
val_root_dir = None
val_ratio = 0.15
run_name = 'landmark_finetune'
resume = ''
```

```bash
python -m transfer.finetune
```

Nạp full model landmark, ưu tiên EMA, tạo optimizer/scaler/EMA counter mới và bắt
đầu từ epoch/step 0. Fine-tune chỉ có **một giai đoạn**, mở băng toàn bộ backbone,
neck và head ngay từ epoch đầu. Mặc định 50 epoch, LR head `5e-5`, LR trunk `5e-6`,
warmup 1 epoch rồi cosine decay. `stage1.epochs=0` để bỏ qua head-only; cấu hình
stage duy nhất nằm trong `FineTuneConfig.stage2`. Checkpoint cuối là
`finetune_final.pt`, không tạo checkpoint head-only. Khi không có `val_root_dir`,
chia theo record gốc để cặp ảnh gốc/lật không lọt sang hai split. Cần dataset có ít
nhất hai record và batch size phù hợp phần train sau chia.

Để tiếp tục fine-tune bị dừng, đặt `FineTuneConfig.resume`; không nạp lại pretrained
và không reset optimizer. Để đổi dataset hoặc training plan, bắt đầu run fine-tune mới.

## Wing Loss

```text
e = (prediction - target) / sqrt(bbox_width * bbox_height)
C = w - w * log(1 + w / epsilon)
wing(e) = w * log(1 + abs(e) / epsilon) nếu abs(e) < w
          abs(e) - C                   nếu abs(e) >= w
```

Mặc định `lmk_wing_w=0.1`, `lmk_wing_epsilon=0.02`, theo tọa độ đã chuẩn hóa.
Giữ mask hợp lệ theo face, trọng số mắt/iris/miệng/mũi và assignment score.
Không có landmark hợp lệ thì landmark loss bằng 0. Loss được tính bằng FP32.
Checkpoint chứa loại loss và tham số; resume từ loss/config khác bị từ chối.

Định dạng mới yêu cầu metadata SHA-256 đi kèm checkpoint; checkpoint landmark cũ
chỉ có `model_signature` không được nạp bằng pipeline này.

## Inference và kiểm tra

Sửa `weights_path`, `image_input`, `mode`, `camera_index` ở cuối `inference.py`.
Kiến trúc và số landmark tự đọc từ checkpoint, không cần khai báo lại.

```bash
python -m transfer.inference
python -m unittest discover -s transfer/tests -v
```

Test dùng trunk nhỏ và dữ liệu synthetic trên CPU. Dữ liệu thực theo
[datastructure.md](datastructure.md). Môi trường cần PyTorch, torchvision, NumPy,
Pillow, OpenCV, matplotlib, TensorBoard và tqdm.
