# Training và fine-tune object detection

Cập nhật: 2026-09-10. Cả hai luồng dùng `NMSFreeDetector` hiện tại. Không thay
đổi kiến trúc trong `src/model.py`, `src/head.py`, `src/backbone_neck.py` hoặc
`src/blocks.py`. Fine-tune nạp nguyên model; không tự thay/reset head.

## Các lỗi đã xử lý

| Vấn đề | Ảnh hưởng | Khắc phục |
| --- | --- | --- |
| Thuộc tính `log_gradients` che khuất method cùng tên | Dừng ở batch đầu với `bool object is not callable` | Đổi thuộc tính thành `log_gradients_enabled` |
| Khởi tạo tổng thể ghi bias head về 0, helper gọi sai tên method | Mất prior classification và bias regression | Gọi đúng `ScaleHead.init_bias()` sau khởi tạo trọng số |
| Entry point gọi `setup_logging` bằng API cũ | Training/fine-tune không khởi động được | Gọi `setup_logging(cfg)` qua engine chung |
| Loss AMP trộn FP16 của prediction với FP32 của GT | CUDA báo lỗi `Index put ... Half ... Float` | Tính TAL, IoU và loss với prediction FP32; forward model vẫn dùng AMP |
| Checkpoint thiếu scaler/RNG/cursor/EMA updates | Resume sai thứ tự và trạng thái training | Lưu và khôi phục đầy đủ trạng thái, dùng sampler theo epoch |
| Validation chỉ log batch cuối, không dùng mAP | TensorBoard không phản ánh cả lượt validation | Log trung bình loss các batch và mAP trong cùng lượt forward |
| Model config thiếu annotation dataclass cho các tham số trunk/stride | `asdict` không lưu đủ cấu hình kiến trúc | Thêm type annotation, giữ nguyên mọi giá trị mặc định |
| Checkpoint không gắn cấu hình/mapping lớp | Có thể nạp nhầm stride hoặc nhãn dù shape tensor khớp | SHA-256 của kiến trúc, mapping lớp và các file metadata đi kèm |
| Fine-tune tự khởi tạo head khi shape không khớp | Làm khác kiến trúc pretrained | Bỏ nhánh thay head; nạp strict và từ chối mismatch |

Các trường cấu hình landmark đã được loại bỏ. Đây chỉ là bài toán object detection.

## Chữ ký kiến trúc và danh mục lớp

`utils/artifacts.py` tạo SHA-256 từ JSON canonical UTF-8: sort key, không có
whitespace thừa, tuple chuyển thành JSON array. Nội dung kiến trúc lấy trực tiếp
từ config: `nc`, `reg_max`, `backbone_w`, `backbone_n`, `neck_n`, `strides`.
Thay learning rate/epochs không tạo phiên bản kiến trúc mới.

Chữ ký kiến trúc mặc định hiện tại:

```text
2bfb6cc36ef5ab822a944006c746e5d348d67e387501c9027df5b957cf124031
```

`categories.json` là JSON array. Mỗi phần tử có `index` (output index từ 0),
`id` (ID trong dataset) và `name`. Thứ tự tăng theo `id`, khớp chính xác mapping
của dataloader. Ví dụ:

```json
[
  {"index": 0, "id": 3, "name": "person"},
  {"index": 1, "id": 8, "name": "car"}
]
```

Danh mục lớp có SHA-256 riêng. `nc` phải bằng số categories; train và val phải
khớp `id/name/index`. Engine không tự suy luận hoặc thay `nc` để chạy qua lỗi.
Fine-tune vẫn có thể dùng ảnh mới/subset các lớp, nhưng file categories phải giữ
đầy đủ mapping của checkpoint nếu muốn giữ nguyên model/head.

Đây là chữ ký định danh cấu hình bằng hash, không phải chữ ký dùng khóa riêng
và không phải checksum toàn bộ tensor trọng số.

## Cấu trúc checkpoint

Cả `TrainConfig` và `FineTuneConfig` mặc định dùng `ckpt_dir="./checkpoints"`.
Engine tạo thư mục theo phiên bản kiến trúc và danh mục lớp:

```text
checkpoints/
  <architecture_sha256>/
    <categories_sha256>/
      train/                         # hoặc finetune / cfg.run_name
        architecture.json
        categories.json
        best.pt
        last.pt
        ckpt_step00001000.pt
        ckpt_step00002000.pt
```

Mỗi run dùng chung một bộ `architecture.json` và `categories.json`, không tạo
thư mục metadata theo từng checkpoint hoặc bản sao ở thư mục cấp trên.
Hai file chỉ được tạo khi chưa tồn tại; nếu đã có thì kiểm tra nội dung và
không ghi đè. Tất cả checkpoint trong run phải khớp bộ metadata này.

`architecture.json` chứa `model` (6 trường config) và `sha256`. Metadata nhúng
trong `.pt` vẫn được giữ để đối chiếu chữ ký và phát hiện nạp nhầm bộ JSON.
Khi chia sẻ checkpoint, sao chép `.pt` cùng hai file JSON dùng chung vào một
thư mục. Pretrained và resume đều đọc JSON ngay cạnh file `.pt`.

Khi nạp pretrained hoặc resume, engine kiểm tra chữ ký, metadata đi kèm,
kiến trúc model và mapping dataset trước khi nạp trọng số. Sai hoặc thiếu metadata
sẽ báo lỗi, không tự đoán kiến trúc/categories. Checkpoint cũ chưa có metadata
không được tự động gắn chữ ký hay nạp vào luồng này.

Checkpoint được ghi qua file tạm rồi `os.replace` để không để lại file `.pt`
dở dang. Rotation chỉ xóa file checkpoint định kỳ cũ; giữ
`best.pt`, `last.pt` và hai file metadata dùng chung. `last.pt` luôn được lưu khi
hoàn tất, kể cả `save_best_only=True`.

Đã xuất metadata cho config mặc định và 80 categories từ dataset train hiện tại
vào `checkpoints/`. Đây là metadata kiến trúc, không phải trọng số đã train.

## Resume

Checkpoint training lưu model, optimizer, scheduler, EMA weights/updates,
GradScaler, Python/NumPy/PyTorch CPU/CUDA RNG, epoch, batch tiếp theo, global step,
best validation loss và toàn bộ config dataclass.

- `epoch`/`next_batch` là vị trí tiếp tục, đếm từ 0. Đã hết batch của epoch thì
  chuyển epoch tiếp theo; checkpoint cuối lưu `epoch=cfg.epochs`, `next_batch=0`.
- `EpochBatchSampler` tái tạo shuffle bằng seed và epoch, bỏ batch đã xử lý
  trước khi đọc ảnh. Seed augmentation gắn với epoch/chỉ số mẫu; retry dùng RNG
  riêng, nên không lệ thuộc tiến độ prefetch hoặc worker. Albumentations dùng
  `Compose.set_random_seed()`; đã kiểm tra với phiên bản 2.0.8.
- DataLoader có generator riêng, không tiêu thụ RNG dùng bởi model.
- Engine từ chối thay đổi các tham số ảnh hưởng resume như batch size, seed,
  augmentation, optimizer, loss, lịch LR/epochs và các cờ freeze/EMA.
- Giữ nguyên dataset và các file index/cache khi resume. Chữ ký categories
  không chứng minh ảnh/annotation trong dataset chưa bị sửa.

Để bắt đầu fine-tune mới, dùng `tfl_pretrained_pth`; để tiếp tục một run đã dừng,
dùng `resume`. Resume không yêu cầu file pretrained ban đầu còn tồn tại.

## Validation

Validation tính loss và mAP trên cùng prediction, dùng EMA nếu bật. TensorBoard
nhận trung bình từng thành phần loss của tất cả batch, thống nhất với giá trị
trả về. Đây là trung bình batch vì DetectionLoss chuẩn hóa theo target scores,
không coi loss là trung bình loss độc lập từng ảnh. Loader rỗng báo lỗi.

Các tag: `val/map_50`, `val/map_50_95`, `val/precision`, `val/recall`.
Metric dùng nhánh o2o, score threshold 0.001, tối đa 300 detections/ảnh;
precision/recall dùng score 0.25 và IoU 0.5. Matching xét score giảm dần và GT
chưa match. Đây là metric nội bộ, không phải toàn bộ giao thức COCOeval.
`best.pt` vẫn được chọn theo validation loss thấp nhất.

## Cách chạy

Chạy từ thư mục gốc dự án, sau khi cấu hình đường dẫn dữ liệu:

```bash
python -m train_.training
python -m finetune_.finetune_engine
```

`FineTuneConfig` kế thừa nguyên các tham số model từ `TrainConfig`, giảm LR và
số epochs, dùng thư mục dữ liệu/log riêng. Đặt `tfl_pretrained_pth` tới checkpoint
đã ký ở đường dẫn mới bên trên. Chiến lược hai giai đoạn được đặt trong `finetune_`:

- `epochs=30`: tổng số epoch; `head_only_epochs=5`: số epoch giai đoạn 1.
- Giai đoạn 1 đóng băng toàn bộ backbone/neck, kể cả BatchNorm; chỉ head học.
- Từ epoch 6, backbone/neck được mở băng và head tiếp tục học.
- `lr0=1e-4` là LR head trước hệ số warmup/cosine;
  `trunk_lr_factor=0.1` cho LR backbone/neck bằng 1/10 LR head.
- Scheduler chạy liên tục qua hai giai đoạn, không reset optimizer của head.
  Các nhóm trunk có trong optimizer từ đầu nhưng không có gradient khi đóng băng.
- `0 < head_only_epochs < epochs`, `0 < trunk_lr_factor < 1`, `lr0 > 0`.

Config đã bỏ `enable_transfer_learning`, `tfl_freeze_backbone`,
`tfl_freeze_neck` và `val_class_sampling`. `tfl_pretrained_pth` chỉ nằm trong
`FineTuneConfig`; việc đóng/mở băng dùng lịch hai giai đoạn.
Resume khôi phục giai đoạn theo epoch và từ chối thay đổi lịch hai giai đoạn.
Khi số lớp hoặc mapping categories đổi, head được khởi tạo lại; backbone/neck
được nạp từ pretrained. Train from scratch vẫn dùng `TrainConfig` và train toàn model.

Trong workspace hiện tại, `data/finetune/labels` và `data/finetune/images` chưa
tồn tại; `tfl_pretrained_pth` còn trống. Cần điền dữ liệu/pretrained thực tế trước
khi chạy fine-tune trên dataset của bạn. Khi bật sampling nhưng `class_sampling_path` không tồn tại, mọi class dùng
hệ số 100%; đặt `train_class_sampling=False` nếu muốn mỗi ảnh xuất hiện
một lần/epoch.

## Kiểm chứng

```bash
NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

Kiểm tra gồm SHA-256/mapping/metadata bị sửa hoặc bị thiếu, từ chối checkpoint
không ký, rotation, resume, logger, bias, pretrained strict và freeze BatchNorm.
Luồng đầu-cuối sử dụng ảnh PNG và JSONL thực trong fixture, model đúng cấu hình
mặc định 80 lớp, 2 worker/persistent workers, TensorBoard, validation, checkpoint
và resume cho cả training/fine-tune trên CPU và CUDA/AMP.

Kiểm tra GPU chạy đủ bước qua giai đoạn GradScaler giảm scale ban đầu và yêu
cầu có optimizer updates thực, trọng số head thay đổi, không chỉ forward thành
công. Kiểm tra resume đối chiếu trọng số với run liên tục. Đây là xác minh luồng
bằng dataset nhỏ, không phải một lần train toàn bộ Object365 hay xác nhận hội tụ.

Bộ test hai giai đoạn kiểm tra trunk bất biến ở giai đoạn 1, cập nhật ở
giai đoạn 2, tỷ lệ LR và resume tại cả hai giai đoạn lẫn mốc chuyển tiếp.

Kết quả kiểm chứng sau thay đổi: 23 tests đạt (21 tests còn lại đạt trong
lần chạy toàn bộ; 2 tests pipeline CPU/CUDA đạt khi chạy lại fixture AMP
đã tăng thời lượng). Bao gồm resume khớp chính xác model/EMA ở trước,
ngay tại và sau mốc mở băng.


## Tăng mẫu chứa class hiếm (`train_`)

Training engine dùng `dataloaderStage1.py`. `dataloader.py` là loader thông thường.
Cấu hình trong `src/config.py`, được `FineTuneConfig` kế thừa:

```python
train_class_sampling = True
class_sampling_path = "data/class_sampling.jsonl"
```

File JSONL chứa hệ số do bạn chuẩn bị, mỗi dòng dùng **category id gốc**,
không dùng index output của model:

```jsonl
{"id": 0, "probability": 103}
{"id": 1, "probability": 140}
{"id": 2, "probability": 100}
```

`probability=100` nghĩa là mỗi ảnh xuất hiện một lần, `140` là một lần chắc
chắn và 40% khả năng thêm một bản lặp; `103` tương ứng thêm 3%.
`300` cho ba lần xuất hiện. Giá trị 0–100 và class không được liệt kê dùng hệ số 1.
Đường dẫn rỗng hoặc file không tồn tại: mọi class mặc định 100%, không tăng mẫu.
File có tồn tại nhưng rỗng, trùng id, id ngoài categories, số âm/NaN/Inf vẫn báo lỗi.

Ảnh nhiều class lấy **max** hệ số của các class có mặt. Một class xuất hiện nhiều
bbox trong ảnh chỉ được tính một lần. Giữ mọi ảnh hợp lệ ít nhất một lần; các
nhãn crowd/fake bị loại theo config, class ngoài mapping và bbox rỗng/không hữu
hạn/nằm ngoài ảnh không làm tăng hệ số. Nếu tắt `include_images_without_annotations`,
ảnh không còn nhãn hợp lệ sau lọc được bỏ. Bbox được cắt vào biên ảnh trước letterbox.

Phần lẻ của hệ số được làm tròn ngẫu nhiên theo `seed` **khi tạo dataset**.
Danh sách lặp giữ cố định giữa các epoch để số step/scheduler/resume ổn định;
shuffle đổi theo epoch khi bật `shuffle`. Mỗi bản lặp có seed augmentation riêng
và seed đổi theo epoch, không nhân bản pixel ảnh trong RAM. Log hiển thị số ảnh
trước/sau cho từng class được tăng mẫu. Phân phối cuối có thể tăng cả class phổ
biến cùng xuất hiện trong ảnh; không bảo đảm các class có số mẫu bằng nhau.

Validation luôn giữ phân phối dữ liệu gốc, không có tùy chọn tăng mẫu.
Pretrain và fine-tune đều mặc định bật `train_class_sampling=True`, dùng chung
`train_` qua training engine. Pretrain đọc `data/class_sampling.jsonl`;
fine-tune đọc `data/finetune/class_sampling.jsonl` được cấu hình riêng trong
`finetune_`. File JSONL phải khớp categories của từng dataset.
Loader kiểm tra file sampling trước khi xây index dữ liệu.

Checkpoint ghi chữ ký hệ số và danh sách mẫu. Resume từ chối nếu nội dung
sampling thay đổi, kể cả sửa file tại cùng đường dẫn. Checkpoint cũ có bật
sampling nhưng thiếu chữ ký này không thể resume chính xác bằng loader mới;
vẫn có thể dùng làm pretrained cho một lần fine-tune mới.

Kiểm chứng sampling: toàn bộ 29 tests đã đạt, gồm CPU/CUDA và DataLoader
nhiều worker; sau khi chốt nguồn hệ số chỉ từ JSONL, chạy lại 6 tests sampling
và đều đạt. Test định dạng dùng đủ 18 dòng probability người dùng cung cấp.

Kiểm tra đồng nhất pretrain/fine-tune: 29/29 tests đạt trong 103,7 giây,
gồm CPU, CUDA/AMP, sampling bật theo config mặc định, validation, chuyển
giai đoạn đóng/mở băng và resume khớp chính xác model/EMA. Fixture của cả
hai luồng xác nhận train 3 ảnh thành 4 mẫu, validation vẫn giữ 3 ảnh.
