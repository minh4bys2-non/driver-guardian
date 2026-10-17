# Phân tích mạng phát hiện buồn ngủ: BatchNorm, training và khả năng học thời gian

Ngày tổng hợp: 16/10/2026.

Tài liệu tổng hợp toàn bộ các chủ đề, bằng chứng, quyết định và kiến thức rút ra trong cuộc trao đổi về pipeline LSTM/ConvGRU. Nội dung được sắp xếp lại theo vấn đề để dùng làm tài liệu nghiên cứu và đối chiếu các lần train, không phải bản chép nguyên văn hội thoại.

**Trạng thái tại thời điểm tổng hợp:** đã chuyển cấu hình mặc định sang train phần phía sau backbone/neck pretrained được đóng băng. Đã kiểm tra tính đúng của cơ chế này, nhưng chưa có kết quả training đầy đủ của run frozen trong workspace để kết luận chất lượng. Các số liệu đánh giá bên dưới thuộc run scratch cũ, trừ khi ghi rõ khác.

## Mục lục

1. [Bối cảnh và diễn biến trao đổi](#1-bối-cảnh-và-diễn-biến-trao-đổi)
2. [Đánh giá phải đi cùng cấu hình checkpoint](#2-đánh-giá-phải-đi-cùng-cấu-hình-checkpoint)
3. [Bằng chứng mô hình nghiêng về alert](#3-bằng-chứng-mô-hình-nghiêng-về-alert)
4. [BatchNorm nằm ở đâu và vì sao gây vấn đề](#4-batchnorm-nằm-ở-đâu-và-vì-sao-gây-vấn-đề)
5. [Thử nghiệm đối chiếu BatchNorm](#5-thử-nghiệm-đối-chiếu-batchnorm)
6. [Phân bố dữ liệu và khả năng khái quát hóa](#6-phân-bố-dữ-liệu-và-khả-năng-khái-quát-hóa)
7. [Các phương án đã cân nhắc và quyết định thực hiện](#7-các-phương-án-đã-cân-nhắc-và-quyết-định-thực-hiện)
8. [Phân tích chi tiết mạng phía sau backbone/neck](#8-phân-tích-chi-tiết-mạng-phía-sau-backboneneck)
9. [Phân tích luồng training và checkpoint](#9-phân-tích-luồng-training-và-checkpoint)
10. [Cách chứng minh model học quan hệ thời gian](#10-cách-chứng-minh-model-học-quan-hệ-thời-gian)
11. [Các thay đổi và kiểm tra đã thực hiện](#11-các-thay-đổi-và-kiểm-tra-đã-thực-hiện)
12. [Những kết luận được phép và chưa được phép rút ra](#12-những-kết-luận-được-phép-và-chưa-được-phép-rút-ra)
13. [Theo dõi run tiếp theo](#13-theo-dõi-run-tiếp-theo)
14. [Kiến thức rút ra](#14-kiến-thức-rút-ra)
15. [Nguồn đối chiếu](#15-nguồn-đối-chiếu)

## 1. Bối cảnh và diễn biến trao đổi

| Giai đoạn | Yêu cầu hoặc câu hỏi | Kết quả |
|---|---|---|
| Evaluation | Chỉnh evaluation cho hệ thống hiện tại; lấy config cùng checkpoint; truyền tham số trong code, không dùng CLI | Đã chỉnh entry point và lưu config đánh giá |
| Triệu chứng | Model dự đoán “full alert”; phân tích, chưa sửa code | Kiểm tra log, checkpoint, predictions và dữ liệu |
| Chẩn đoán | Tìm nguyên nhân từ model và training | Phát hiện chênh lệch BatchNorm và dấu hiệu khái quát hóa kém |
| Giữ backbone/neck | Đề xuất giải pháp nếu không thay backbone/neck | Phân biệt giữ kiến trúc với giữ cả weights và running statistics |
| Vị trí BatchNorm | BatchNorm nằm ở đâu? | Trong các block Conv của backbone và neck, dùng lại từ ObjectDetection |
| Thay đổi training | Train phần mạng phía sau để tránh chênh lệch này | Chọn pretrained detector/landmark, đóng băng backbone/neck |
| Kiểm tra thay đổi | Những file nào đã sửa? | Config, log training, tests và README; không thay kiến trúc |
| Lo ngại về thời gian | Frozen backbone có làm mất khả năng học quan hệ thời gian không? | Không cắt gradient của ConvGRU; khả năng học chưa đồng nghĩa đã học hữu ích |
| Phân tích sâu | Phân tích adapter, ConvGRU, attention, pooling, loss | Xác định những giới hạn và phép thử cần làm |
| Chờ kết quả | Tạm dừng thay đổi để đợi training frozen | Chưa đổi loss, pooling hay kiến trúc |
| Kiến thức BatchNorm | Chuẩn hóa có thể biểu hiện như overfit vào phân phối không? | Phân biệt overfitting, sai lệch thống kê chuẩn hóa và domain shift |

## 2. Đánh giá phải đi cùng cấu hình checkpoint

### 2.1. Vì sao cần tự chứa cấu hình?

Chỉ nạp weights là chưa đủ. Model cần được dựng đúng kiến trúc, và video cần được xử lý giống khi training.

Checkpoint hiện tại có các trường quan trọng:

| Trường | Vai trò |
|---|---|
| `format_version` | Kiểm tra định dạng checkpoint |
| `model` | Trọng số và buffers của model, gồm running statistics BatchNorm |
| `model_config` | Kiến trúc thực tế, kích thước hidden/input, chế độ frozen… |
| `config` | Cấu hình training và preprocessing |
| `epoch` | Epoch tương ứng với weights được lưu |

`load_model()` dựng `DrowsyClassifier` từ `model_config`, sau đó load state dict với `strict=True`. Không dựng model theo các giá trị mặc định mới rồi cố nạp weights cũ.

Evaluation lấy từ `config` các giá trị `image_size`, `window_sec`, `target_fps`, `amp`, cùng batch size và số worker mặc định. Có thể đổi vị trí dataset và các tham số vận hành mà không tự ý đổi preprocessing.

### 2.2. Giao diện đánh giá đã thống nhất

Truyền trực tiếp các đối số trong lời gọi `main(...)` của `src/evaluation.py`:

```python
main(
    checkpoint_path="path/to/best.pt",
    dataset_dir=None,
    split="test",
    output_dir=None,
    device="auto",
    batch_size=None,
    num_workers=None,
)
```

`None` ở dataset, batch size và số worker nghĩa là dùng giá trị trong checkpoint. `output_dir=None` lưu vào thư mục `evaluation_<split>` cạnh checkpoint. Đây là giao diện evaluation; training vẫn còn CLI tùy chọn, đồng thời hỗ trợ `train(cfg)` trực tiếp trong Python/notebook.

Artifacts đánh giá:

- `metrics.json`: các chỉ số và thông tin epoch/split/preprocessing.
- `predictions.csv`: nhãn thật, dự đoán, xác suất drowsy và thông tin clip.
- `evaluation_config.json`: cấu hình checkpoint cùng các giá trị runtime thực tế.
- `evaluation.log`: log đánh giá.

**Lưu ý:** đổi mặc định trong `config.py` không thay đổi checkpoint cũ. Khi run frozen có checkpoint mới, phải chọn đúng checkpoint đó trong evaluation. File metrics hiện có không tự trở thành kết quả của run frozen.

## 3. Bằng chứng mô hình nghiêng về alert

### 3.1. Checkpoint được phân tích

Run: `ai/LSTM/checkpoints/scratchv1`.

- `best.pt`: epoch 2.
- `last.pt`: epoch 4.
- Log có một phần epoch 5.
- Cấu hình đặt 100 epochs, nhưng artifacts hiện có không chứng minh run đã hoàn thành 100 epochs.

Config thực tế của run scratch:

| Tham số | Giá trị |
|---|---:|
| Backbone mode | scratch |
| Backbone checkpoint | không có |
| Input/hidden dimension | 128 / 128 |
| Batch size / accumulation | 24 / 1 |
| Chunk size | 8 frame |
| Image size | 224 |
| Window / FPS | 10 giây / 5 FPS |
| Dropout | 0,35 |
| Learning rate | 0,0003 |
| Weight decay | 0,01 |
| Warmup | 5 epochs |
| Freeze BatchNorm | false |
| Tham số trainable theo log | 2.538.963 |

### 3.2. Kết quả test đã lưu

Nguồn: `runs/metrics.json` và `runs/predictions.csv`.

| Chỉ số | Kết quả |
|---|---:|
| Tổng clip | 719 |
| Alert thực tế | 361 |
| Drowsy thực tế | 358 |
| Loss | 1,4007 |
| Accuracy | 39,92% |
| Precision drowsy | 23,57% |
| Recall drowsy | 9,22% |
| F1 drowsy | 0,1325 |
| Specificity | 70,36% |
| ROC-AUC | 0,3098 |
| Average precision | 0,3825 |

Confusion matrix, hàng là nhãn thật và cột là dự đoán:

| | Dự đoán alert | Dự đoán drowsy |
|---|---:|---:|
| Thực tế alert | 254 | 107 |
| Thực tế drowsy | 325 | 33 |

Toàn bộ test không phải 100% alert: có 579 dự đoán alert và 140 dự đoán drowsy. Tuy nhiên, model bỏ sót 325/358 clip drowsy.

Theo nhóm:

- Subject N: 227/227 clip drowsy bị dự đoán alert.
- Subject O: phát hiện 33/115 clip drowsy.
- VBDDD: cả 55 clip test đều bị dự đoán alert, bao gồm 16 clip drowsy.

### 3.3. Không chỉ là vấn đề ngưỡng

| Nhãn thật | P(drowsy) trung bình | P(drowsy) trung vị |
|---|---:|---:|
| Alert | 0,3195 | 0,2514 |
| Drowsy | 0,1758 | 0,0550 |

Điểm drowsy trung bình của mẫu thật sự drowsy còn thấp hơn mẫu alert. AUC 0,31 cho thấy thứ hạng điểm trên test đang bất lợi, không chỉ là ngưỡng 0,5 quá cao.

Điều này không tự chứng minh nhãn bị đảo. Hạ threshold có thể tăng recall và false positives cùng lúc, nhưng không sửa được thứ hạng điểm. Không dùng kết quả này làm lý do tự động đảo nhãn hoặc chọn threshold theo test.

### 3.4. Diễn biến training

| Epoch | Train loss | Train F1 | Val loss | Val F1 |
|---|---:|---:|---:|---:|
| 1 | 0,6808 | 0,4694 | 1,0807 | 0,0000 |
| 2 | 0,5946 | 0,6458 | 0,9737 | 0,4814 |
| 3 | 0,4410 | 0,7698 | 1,6040 | 0,0347 |
| 4 | 0,3914 | 0,7940 | 1,3736 | 0,4583 |

Train cải thiện nhưng validation dao động mạnh. Epoch 1 có val recall bằng 0; epoch 3 recall chỉ khoảng 1,82%.

Train metrics được thu trong quá trình model đang cập nhật trọng số, có dropout/augmentation và chế độ train. Validation dùng một snapshot trọng số ở chế độ eval. Khoảng cách hai đường metric không chỉ phản ánh overfitting; cần tách ảnh hưởng của chế độ chuẩn hóa và cách đo.

## 4. BatchNorm nằm ở đâu và vì sao gây vấn đề

### 4.1. Vị trí trong code

BatchNorm được khai báo trong `ai/ObjectDetection_2p6M/src/blocks.py`, bên trong block `Conv`:

```python
self.bn = nn.BatchNorm2d(c2)

def forward(self, x):
    return self.act(self.bn(self.conv(x)))
```

Mỗi block có thứ tự `Conv2d → BatchNorm2d → SiLU`. Backbone và PAFPN dùng block này trực tiếp hoặc thông qua C2f, C2fCIB, SCDown và các block liên quan.

LSTM import `Backbone` và `PAFPN` từ module ObjectDetection trong `src/model_backbone_neck.py`. Model checkpoint được kiểm tra có 73 lớp BatchNorm. Hàm khởi tạo đặt `eps=1e-3`, `momentum=0.03`.

### 4.2. Chuẩn hóa activation không phải làm sạch dữ liệu

BatchNorm chuẩn hóa activation bên trong mạng. Nó không tự loại bỏ frame lỗi, làm sạch nhãn sai, xác định video chất lượng thấp hay bảo đảm loại bỏ nhiễu.

Nó có thể giúp tối ưu ổn định hơn, nhưng vẫn phụ thuộc vào cách lấy batch và phân phối activation. Tên “normalization” không nên được hiểu thành “dữ liệu xấu đã được xử lý”.

### 4.3. Train và eval dùng hai nguồn thống kê

Ở chế độ train, BatchNorm dùng thống kê của batch hiện tại để chuẩn hóa và cập nhật running statistics. Ở eval, nó dùng running mean/variance đã lưu.

Công thức cập nhật với momentum `m`:

```text
running_new = (1 − m) × running_old + m × batch_stat
```

Nếu batch đủ đại diện, hai cơ chế có thể phù hợp với nhau. Nếu batch gồm các frame tương quan mạnh của cùng một video, thống kê của batch đó có thể rất khác thống kê tích lũy.

### 4.4. Batch optimizer khác batch mà BatchNorm nhìn thấy

Trong `model.py`, dữ liệu `[B,T,C,H,W]` được trải phẳng theo thứ tự video rồi chia chunk:

```text
A_frame1, A_frame2, ..., A_frame50,
B_frame1, B_frame2, ..., B_frame50, ...
```

Batch train 24 video tạo ra 1.200 frame. Nhưng `chunk_size=8` khiến mỗi lần backbone chạy chỉ nhận 8 frame, chủ yếu là các frame gần nhau của một video.

Do đó, BatchNorm không nhìn 24 người/video độc lập trong một lượt forward. Nó thường nhìn cùng một người, nền và ánh sáng. Một số chunk cắt qua ranh giới video, nhưng không làm thay đổi bản chất đa số chunk.

Gradient accumulation chỉ cộng gradient trước optimizer step. Nó không gộp các lần forward thành một batch thống kê BatchNorm lớn hơn.

### 4.5. Nhiều lần cập nhật running statistics trong một optimizer batch

Một batch đầy đủ có `24 × 50 / 8 = 150` chunk, tức mỗi lớp BatchNorm cập nhật 150 lần.

Với momentum 0,03:

```text
Phần hệ số còn lại của thống kê trước batch = 0,97^150 ≈ 1%
```

Running statistics sau batch vì vậy chịu ảnh hưởng rất lớn từ các chunk vừa chạy, đặc biệt những chunk gần cuối. Đây là tính chất của phép cập nhật, không phải phép đo “99% thông tin model chỉ đến từ video cuối”.

Checkpoint epoch 2 có `num_batches_tracked=72.564` ở lớp BatchNorm đầu được kiểm tra, phù hợp với số lượng chunk qua hai epoch.

### 4.6. Phân biệt ba vấn đề

| Vấn đề | Cơ chế | Biểu hiện có thể gặp |
|---|---|---|
| Overfitting | Weights học quá nhiều đặc điểm riêng của tập train | Train tốt, người/video mới kém |
| Chênh lệch chuẩn hóa | Train dùng batch statistics, eval dùng running statistics không phù hợp | Cùng clip, kể cả clip train, đổi mode làm dự đoán đổi mạnh |
| Domain shift | Dữ liệu mới khác về người, camera, ánh sáng, cách gán nhãn… | Kém trên một dataset/người cụ thể |

Ba vấn đề có thể xảy ra đồng thời. Không nên thấy train tốt/val kém rồi kết luận ngay là overfitting; cũng không nên quy toàn bộ lỗi cho BatchNorm.

## 5. Thử nghiệm đối chiếu BatchNorm

### 5.1. Cách thực hiện

Thử nghiệm chẩn đoán dùng `scratchv1/best.pt` trên 14 clip chọn theo người và nhãn, không phải mẫu ngẫu nhiên đại diện toàn dataset.

- Cùng weights, cùng clip và cùng cửa sổ giữa video.
- Không augmentation; dropout tắt.
- Lần một: model eval bình thường.
- Lần hai: chỉ các lớp BatchNorm chuyển sang dùng batch statistics; đặt momentum bằng 0 để không cập nhật running mean/variance.
- Chạy trong bộ nhớ, không optimizer step, không ghi lại checkpoint.

Momentum 0 không biến BatchNorm train thành eval: forward vẫn dùng batch statistics. Bộ đếm batch trong đối tượng tạm vẫn có thể tăng. Không có thay đổi nào được lưu vào checkpoint.

### 5.2. Kết quả

| Split | Người | Nhãn thật | P(drowsy), eval | P(drowsy), batch statistics |
|---|---|---:|---:|---:|
| Train | E | 0 | 0,0376 | 0,0200 |
| Train | E | 1 | 0,0456 | 0,8688 |
| Train | J | 0 | 0,0267 | 0,0742 |
| Train | J | 1 | 0,0572 | 0,8819 |
| Val | D | 0 | 0,0880 | 0,4204 |
| Val | D | 1 | 0,4972 | 0,4366 |
| Val | Q | 0 | 0,0825 | 0,2366 |
| Val | Q | 1 | 0,0429 | 0,7889 |
| Test | N | 0 | 0,1206 | 0,9295 |
| Test | N | 1 | 0,0444 | 0,2870 |
| Test | O | 0 | 0,7895 | 0,3119 |
| Test | O | 1 | 0,6528 | 0,9210 |
| Test | VBDDD subject4 | 0 | 0,0480 | 0,3516 |
| Test | VBDDD subject4 | 1 | 0,3244 | 0,0541 |

Các clip drowsy của E, J và Q đổi rất mạnh khi chỉ thay cách chuẩn hóa. Ngay cả clip train cũng bị sai ở eval. Đây là bằng chứng mạnh về chênh lệch train/eval.

Tuy nhiên, clip alert của N trở thành false positive khi dùng batch statistics; clip drowsy của VBDDD subject4 còn có điểm thấp hơn. Kết quả không chứng minh chỉ cần đổi mode là model sẽ tốt.

### 5.3. Giới hạn kết luận

- Không phải đánh giá lại toàn bộ dataset.
- Không xác định được bao nhiêu phần trăm lỗi do BatchNorm.
- Không chứng minh BatchNorm là nguyên nhân duy nhất.
- Không dùng kết quả này để bật `model.train()` khi inference: dropout và hành vi phụ thuộc batch sẽ gây vấn đề khác.
- Các phép thử 14 clip là chẩn đoán ad hoc trong phiên làm việc; chưa được lưu thành một benchmark tái lập riêng trong repository.

## 6. Phân bố dữ liệu và khả năng khái quát hóa

### 6.1. Dataset có nhãn khá cân bằng nhưng nguồn dữ liệu khác nhau

Đếm từ manifest `DataLSTM/dataset_merged_split.csv`:

| Dataset | Train alert | Train drowsy | Val alert | Val drowsy | Test alert | Test drowsy |
|---|---:|---:|---:|---:|---:|---:|
| UL-DD | 1.748 | 1.724 | 368 | 368 | 322 | 342 |
| SUST | 1.099 | 975 | 0 | 0 | 0 | 0 |
| VBDDD | 183 | 76 | 38 | 16 | 39 | 16 |
| Tổng | 3.030 | 2.775 | 406 | 384 | 361 | 358 |

SUST chiếm khoảng 35,7% train nhưng vắng mặt ở val/test. Tổng nhãn train khá cân bằng; mất cân bằng alert/drowsy toàn tập chưa phải nghi phạm chính.

Không có giao nhau của định danh `orig_path` giữa train–test và val–test trong lần kiểm tra. Điều này không thay thế kiểm tra duplicate bằng nội dung hoặc xác minh định danh người, đặc biệt khi một nguồn dữ liệu thiếu subject ID.

### 6.2. Dấu hiệu dự đoán phụ thuộc người

UL-DD validation gồm D và Q; UL-DD test gồm N và O.

| Người test | P(drowsy) trung bình, nhãn alert | P(drowsy) trung bình, nhãn drowsy |
|---|---:|---:|
| N | 0,1613 | 0,0555 |
| O | 0,4170 | 0,4141 |

Điểm của N gần như luôn thấp, còn O có hai nhãn gần như không tách được theo trung bình. Điều này phù hợp với giả thuyết model dùng đặc điểm người/bối cảnh, nhưng chưa xác định được nó đang dựa vào vùng ảnh nào.

### 6.3. Những việc cần kiểm tra

- Chất lượng theo từng dataset/người, thay vì chỉ metric tổng.
- Cửa sổ 10 giây thực tế có thể hiện nhãn video hay không.
- Khác biệt camera, ánh sáng, crop và cách gán nhãn giữa các nguồn.
- Các clip từ cùng nguồn có quá giống nhau trong train không.
- Ảnh hưởng của SUST bằng thí nghiệm có/không có SUST với thiết lập còn lại cố định.

Các hướng này mới được đề xuất; chưa thay đổi sampling, nhãn hoặc thành phần dataset.

## 7. Các phương án đã cân nhắc và quyết định thực hiện

### 7.1. Nếu giữ kiến trúc nhưng vẫn train backbone/neck

Phương án ưu tiên được đề xuất là xen kẽ frame của nhiều video trước khi chia chunk:

```text
A_frame1, B_frame1, C_frame1, ...,
A_frame2, B_frame2, C_frame2, ...
```

Sau backbone phải khôi phục đúng thứ tự `[B,T,C,H,W]` trước ConvGRU. Có thể thử chunk 8 trước, rồi tăng 16–32 nếu đủ VRAM. Batch video cần đủ đa dạng; batch chỉ có một video không tạo được đa dạng người bằng cách đổi thứ tự.

Mục tiêu là để BatchNorm nhìn nhiều video trong một lượt forward, không phải thay các lớp backbone/neck. Chỉ tăng chunk nhưng vẫn dùng frame liên tiếp của một video chưa chắc giải quyết vấn đề.

**Chưa triển khai phương án này.**

### 7.2. Hiệu chỉnh lại running statistics

Một phép thử khác là giữ weights của checkpoint cũ, chỉ ước lượng lại running statistics trên tập train, dùng preprocessing giống inference và dữ liệu đủ đại diện.

Không dùng val/test để hiệu chỉnh; chúng dùng để đánh giá. Không bật dropout trong lượt hiệu chỉnh. Phải kiểm tra cách chia chunk, không gọi máy móc một utility rồi giả định thống kê đã đúng.

Hiệu chỉnh thay đổi trạng thái backbone/neck dù không thay weights học được. Nó có thể cải thiện hoặc không, vì weights đã học theo cách chuẩn hóa trước đó.

**Chưa thực hiện hiệu chỉnh hoặc ghi checkpoint đã hiệu chỉnh.**

### 7.3. Quyết định được chọn: pretrained frozen trunk

Người dùng chọn checkpoint pretrained detector/landmark đang có, cụ thể `landmark_finetune2/best.pt`, thay vì lấy trunk từ `scratchv1/best.pt`.

Thiết lập mới:

- Nạp backbone/neck pretrained; loader ưu tiên EMA nếu checkpoint có EMA phù hợp.
- Giữ nguyên weights và buffers của backbone/neck.
- Luôn đặt backbone/neck ở eval, kể cả khi gọi `model.train()`.
- Forward backbone/neck không lưu đồ thị gradient.
- Khởi tạo mới adapter, ConvGRU, spatial attention và classifier.
- Optimizer chỉ nhận tham số có `requires_grad=True`.
- Run mới lưu ở `ai/LSTM/checkpoints/head_landmarkv1`.

Cơ chế frozen đã có trong code; thay đổi chính là chọn nó làm mặc định, chọn checkpoint và xác minh hành vi. Không thay kiến trúc backbone, neck hoặc ConvGRU.

Đóng băng giúp head học trên cùng cơ chế trích xuất đặc trưng được dùng lúc inference. Nó không bảo đảm running statistics pretrained phù hợp hoàn toàn với video mới, và không loại bỏ overfitting ở head.

### 7.4. Những cách không được xem là giải pháp đủ

- Tăng gradient accumulation để “tăng batch BatchNorm”.
- Bật train mode khi inference để lấy batch statistics.
- Chỉ đóng băng gradient nhưng để BatchNorm tiếp tục train.
- Freeze BatchNorm chưa học ngay từ đầu khi train scratch rồi kỳ vọng thống kê phù hợp.
- Thêm class weight hoặc hạ threshold trước khi kiểm tra chất lượng đặc trưng và thứ hạng điểm.
- Giả định train đủ 100 epochs tự khắc hết lỗi.

## 8. Phân tích chi tiết mạng phía sau backbone/neck

### 8.1. Kích thước và năng lực hiện tại

Theo config frozen mới:

```text
Video RGB uint8               [B, 50, 3, 224, 224]
Backbone/neck frozen → P4     [B, 50, 128, 14, 14]
Adapter                      [B, 50, 96, 14, 14]
ConvGRU hidden mỗi bước      [B, 96, 14, 14]
Spatial pooling              [B, 50, 96]
Temporal mean                [B, 96]
Dropout + Linear             [B, 2]
```

| Module | Tham số |
|---|---:|
| Adapter | 12.480 |
| ConvGRU | 497.952 |
| Spatial attention | 97 |
| Linear classifier | 194 |
| Tổng | 510.723 |

Khoảng 97,5% tham số head nằm trong ConvGRU. Đây là phân bổ năng lực kiến trúc, không phải bằng chứng model đã học thời gian hiệu quả.

### 8.2. Adapter

Adapter là `Conv 1×1, 128→96 → GroupNorm(1,96) → SiLU`.

Conv 1×1 học phối hợp các kênh của từng vị trí, giúp đặc trưng pretrained thích nghi với bài toán. Nó không tự nhìn frame trước/sau và không trực tiếp trộn các vị trí lân cận.

GroupNorm chuẩn hóa từng frame sau khi gộp B và T, không dùng running statistics theo batch. Head hiện có một GroupNorm và không có BatchNorm.

Nếu thông tin mắt/miệng còn được mã hóa trong P4, adapter có thể học cách sử dụng. Nếu tín hiệu đã bị loại bỏ ở backbone, adapter không khôi phục được thông tin chưa có trong đầu vào.

P4 14×14 khá thô so với ảnh 224×224, nhưng không thể từ kích thước này kết luận chi tiết mắt đã mất: chi tiết còn có thể nằm trong các kênh. Cần kiểm tra đặc trưng và hiệu năng thực tế.

### 8.3. ConvGRU và ý nghĩa các cổng

Theo đúng quy ước của code:

```text
r[t], u[t] = sigmoid(Conv([x[t], h[t−1]]))
candidate[t] = tanh(Conv([x[t], r[t] × h[t−1]]))
h[t] = (1 − u[t]) × h[t−1] + u[t] × candidate[t]
```

- Reset gate `r` điều khiển phần lịch sử đi vào candidate.
- Update gate `u` lớn: ghi nhiều candidate mới.
- Update gate `u` nhỏ: giữ nhiều trạng thái cũ trên đường nối trực tiếp.

Một số tài liệu dùng update gate với quy ước ngược lại; cần đọc công thức thay vì suy diễn từ tên biến. Công thức trong code không tự trở thành sai chỉ vì khác quy ước đặt tên.

Hidden state giữ cấu trúc không gian 96×14×14. Conv 3×3 có thể phối hợp lịch sử và đặc trưng hiện tại ở các vị trí lân cận. Model chưa ép mỗi frame thành một vector trước bước recurrent.

Khả năng học thời gian có thật, nhưng model vẫn có thể học hành vi gần như theo từng frame nếu các cổng và candidate giảm ảnh hưởng lịch sử.

### 8.4. Bias update gate và trí nhớ

Bias update gate được khởi tạo bằng −1. Nếu tạm bỏ qua đóng góp convolution:

```text
u ≈ sigmoid(−1) ≈ 0,269
h mới ≈ 0,731 × h cũ + 0,269 × candidate
```

Chỉ xét đường giữ trực tiếp, sau 10 bước hệ số còn khoảng `0,731^10 ≈ 4,4%`. Ở 5 FPS, 10 bước tương ứng 2 giây.

Đây không phải kết luận model chỉ nhớ được 2 giây. Candidate cũng phụ thuộc lịch sử, cổng thay đổi theo input và được học. Phép tính chỉ minh họa rằng bias −1 không tự bảo đảm bộ nhớ 50 frame.

Để đo khả năng giữ lịch sử, cần phân tích gate saturation, độ nhạy theo vị trí frame và chất lượng khi thay đổi lịch sử sau training.

### 8.5. Spatial attention

Mỗi bước tạo trọng số trên 196 vị trí:

```text
a[t] = softmax(Conv1×1(h[t]))
v[t] = tổng theo không gian của a[t] × h[t]
```

Attention có thể chọn vùng mắt, miệng, tư thế đầu hoặc nhiều vùng cùng lúc. Vì đầu vào là hidden state, việc chọn vùng có thể phụ thuộc lịch sử.

Nhưng loss không có nhãn vùng mắt/miệng. Mạng có thể chọn nền hoặc đặc điểm nhận dạng người nếu chúng hữu ích trên train. Một bản đồ attention được dùng chung để tổng hợp các kênh; không có các nhánh riêng buộc phải theo dõi từng bộ phận khuôn mặt.

Heatmap đẹp hoặc tập trung vào mắt không đủ chứng minh model học chuyển động mắt, cũng không phải bằng chứng nhân quả hoàn chỉnh về quyết định.

### 8.6. Temporal pooling thực tế là trung bình đều

Tên chế độ là `attention_pooling`, nhưng temporal weights được tính từ mask hợp lệ rồi chuẩn hóa tổng bằng 1. Với 50 bước đầy đủ, mỗi bước có trọng số 1/50.

```text
clip_vector = mean(v[1], ..., v[50])
```

Điểm tốt: mỗi `v[t]` đã chứa lịch sử qua ConvGRU. Vì vậy mean này không xóa hoàn toàn thứ tự và không tương đương trung bình ảnh thô.

Giới hạn: nếu sự kiện chỉ xuất hiện ở 5/50 bước và hidden state không giữ nó ở các bước sau, phần đóng góp trực tiếp chỉ chiếm 10% phép tổng hợp. Đây là ví dụ minh họa, không phải công thức cố định cho tác động của một sự kiện thực tế.

### 8.7. Classifier tuyến tính và trung bình logits

Ở eval, dropout tắt và classifier cuối là tuyến tính:

```text
Linear(mean(v[t])) = mean(Linear(v[t]))
```

Tính chất này đã được kiểm tra bằng forward trên model hiện tại. Clip logits tương đương trung bình logits của các trạng thái theo thời gian. Không được nhầm với trung bình xác suất sau softmax, vì softmax là phi tuyến.

ConvGRU vẫn có thể lưu một sự kiện quan trọng trong nhiều trạng thái. Nhưng việc nhận biết, ghi nhớ và khuếch đại sự kiện được giao chủ yếu cho ConvGRU; pooling cuối không tự học chọn thời điểm.

### 8.8. Gradient và phạm vi lịch sử

- Không có `detach` hidden state giữa các bước ConvGRU.
- Gradient từ loss clip đi vào các trạng thái và tham số recurrent.
- `no_grad()` chỉ bao quanh backbone frozen, không bao quanh adapter/head.
- Chunk size 8 chỉ chia extraction backbone; không cắt trí nhớ ConvGRU thành đoạn 8 frame.
- Training không truyền hidden state từ clip trước sang clip sau; mỗi clip khởi tạo h bằng 0.

Vì vậy model được train trên lịch sử tối đa trong cửa sổ 10 giây. API có thể nhận/trả hidden state, nhưng điều đó không chứng minh nó đã được train cho inference liên tục xuyên nhiều cửa sổ.

### 8.9. Sampling và tín hiệu chuyển động

Ở 5 FPS, mỗi mẫu cách nhau 0,2 giây. Những biến đổi ngắn có thể bị bỏ qua hoặc chỉ xuất hiện ở ít frame. Tăng hidden dimension không tái tạo được chuyển động chưa được lấy mẫu.

Train lấy cửa sổ ngẫu nhiên thiên về cuối video; val/test lấy cửa sổ giữa. Nhãn video có thể không mô tả chính xác mọi cửa sổ 10 giây. Cần xem những cửa sổ thực tế để đánh giá, không mặc định mọi sai số là do mạng.

## 9. Phân tích luồng training và checkpoint

### 9.1. Loss hiện tại học nhãn clip

`CrossEntropyLoss` nhận logits `[B,2]` và một nhãn cho mỗi clip. Không có loss ở từng frame, loss dự đoán chuyển động hoặc mục tiêu buộc model phân biệt các thứ tự thời gian.

Nếu một ảnh mắt nhắm, tư thế đầu hoặc bối cảnh đủ giúp giảm loss, model có thể chọn dấu hiệu tĩnh. Việc recurrent weights có gradient không chứng minh model đã học diễn biến hữu ích; chúng cũng có thể học biến đổi đặc trưng tĩnh hoặc làm mượt.

Đây là rủi ro của mục tiêu học, chưa phải bằng chứng cần thay loss ngay. Gán cùng nhãn clip cho mọi frame cũng có thể tạo supervision sai nếu dấu hiệu chỉ xuất hiện trong một đoạn.

### 9.2. Cơ chế cập nhật

Training hiện dùng AdamW, warmup rồi cosine decay theo epoch, AMP khi chạy CUDA và gradient clipping với `max_grad_norm=1.0`.

Gradient accumulation nhân mean loss với số mẫu của batch, cộng gradient, sau đó chia cho tổng số mẫu trong nhóm trước optimizer step. Cách này xử lý cả batch cuối nhỏ. Sau cập nhật, gradient được reset.

Dropout 0,35 nằm sau temporal pooling. Nó tạo khác biệt train/eval có chủ đích ở classifier, nhưng không có running statistics như BatchNorm.

### 9.3. Config mới không giống hoàn toàn run cũ

| Thành phần | Run scratch đã đo | Mặc định frozen mới |
|---|---:|---:|
| Backbone initialization | ngẫu nhiên | pretrained landmark |
| Backbone trainable | có | không |
| Input/hidden dimension | 128 / 128 | 96 / 96 |
| Batch size | 24 | 1 |
| Accumulation | 1 | 4 |
| Số clip mỗi update đầy đủ | 24 | 4 |
| Learning rate gốc | 0,0003 | 0,0003 |
| Window/FPS/image size | 10 / 5 / 224 | 10 / 5 / 224 |

Với 5.805 mẫu, run cũ có khoảng 242 optimizer steps/epoch; mặc định mới có khoảng 1.452 steps/epoch. Cùng số epoch hoặc cùng learning rate không có nghĩa là cùng quá trình tối ưu.

Vì có nhiều khác biệt, cải thiện ở run mới không phải thí nghiệm cô lập hoàn toàn tác động BatchNorm. Nó đánh giá phương án frozen pretrained tổng thể.

### 9.4. Chọn best và resume

Best checkpoint được chọn theo val F1, hòa F1 thì ưu tiên val loss thấp hơn. `last.pt` lưu optimizer, scheduler, scaler, RNG và best weights để resume hoặc tái tạo best checkpoint.

Resume giữ cấu hình của run gốc và kiểm tra các sai khác. Không resume `scratchv1/last.pt` để chuyển sang frozen; cần bắt đầu run mới.

Checkpoint của run frozen tự chứa backbone và head. Sau khi đã lưu, load/resume không cần lại file pretrained gốc.

### 9.5. Log hiện có và điều chưa đo

Log/TensorBoard có loss, accuracy, precision, recall, F1, AUC/AP khi đủ lớp, confusion matrix, score histogram, learning rate, gradient norm và histogram weights/gradients.

Những log này chưa trực tiếp đo trí nhớ thời gian. Chưa có benchmark xáo trộn frame, lặp frame, baseline không recurrent hoặc thống kê cổng ConvGRU được tích hợp trong training.

## 10. Cách chứng minh model học quan hệ thời gian

### 10.1. Phân biệt các mức bằng chứng

| Mức | Câu hỏi | Bằng chứng cần có |
|---|---|---|
| Cấu trúc | Có đường kết nối qua thời gian không? | Hidden state recurrent, không detach |
| Tối ưu | Các tham số recurrent có được cập nhật không? | Gradient và optimizer step |
| Phụ thuộc | Đầu ra có dùng lịch sử/thứ tự không? | Can thiệp vào chuỗi làm thay đổi đầu ra |
| Hữu ích | Dùng thời gian có tăng chất lượng trên người mới không? | So sánh có kiểm soát và metric theo nhóm |

Đã có bằng chứng ở hai mức đầu. Hai mức sau chưa được xác nhận trên head frozen đã train đầy đủ.

### 10.2. Các phép thử đề xuất

| Phép thử | Cách làm | Cách đọc kết quả |
|---|---|---|
| Giữ thứ tự | Chuỗi gốc | Mốc so sánh |
| Xáo trộn | Trộn cùng bộ frame bằng nhiều permutation | Đo nhạy với thứ tự; giảm chất lượng chưa tự chứng minh cơ chế đúng |
| Lặp một frame | Chọn frame rồi lặp đủ chiều dài; thử nhiều vị trí frame | Mất chuyển động có làm chất lượng giảm không? |
| Đảo ngược | Chạy chuỗi theo chiều ngược | Đo nhạy với chiều diễn biến; một số dấu hiệu không phụ thuộc chiều |
| Thay độ dài lịch sử | Giới hạn số bước đầu vào có kiểm soát | Model cần lịch sử ngắn hay dài? |
| Baseline không recurrent | Cùng frozen features, pooling/classifier không ConvGRU | ConvGRU có lợi ích vượt qua dấu hiệu tĩnh không? |
| Phân tích gradient theo thời gian | Đo độ nhạy đầu ra với đặc trưng ở từng bước | Gợi ý ảnh hưởng của lịch sử; không thay thế đánh giá hiệu năng |
| Quan sát gates/attention | Phân bố reset/update, attention theo frame | Tìm saturation hoặc hành vi bất thường; không suy diễn nhân quả chỉ từ heatmap |

Các phép thử trên mới là kế hoạch, chưa được thực hiện đầy đủ. Cần dùng nhiều clip/người, cố định preprocessing và báo cáo biến thiên giữa các phép thử. Dùng validation để chọn thiết kế; hạn chế lặp tối ưu trực tiếp theo test.

### 10.3. Những cách diễn giải cần tránh

- Đảo ngược không giảm accuracy → chưa đủ kết luận model không dùng thời gian.
- Xáo trộn làm đầu ra đổi → chỉ chứng minh nhạy với thứ tự, chưa chứng minh nhạy hữu ích.
- ConvGRU có nhiều tham số → không bảo đảm học quan hệ dài hạn.
- Gradient không bằng 0 → không bảo đảm trí nhớ đủ dài hoặc tổng quát hóa tốt.
- Attention nhìn vào mắt → không bảo đảm đang phân biệt chuyển động mắt.
- Thêm temporal attention → không tự buộc model bỏ dấu hiệu tĩnh.

## 11. Các thay đổi và kiểm tra đã thực hiện

### 11.1. Giai đoạn chỉnh evaluation

| File | Nội dung |
|---|---|
| `src/evaluation.py` | Bỏ argparse; nhận tham số qua main; dùng checkpoint config; lưu evaluation config; trả metrics/predictions; hỗ trợ chạy file trực tiếp |
| `tests/test_evaluation.py` | Kiểm tra cấu hình checkpoint, runtime overrides và artifacts |

Kết quả ghi nhận: 9 tests evaluation pass. Chưa chạy đánh giá dataset thực tế trong chính lượt chỉnh evaluation. Các artifacts thực tế được đọc ở các lượt phân tích sau.

### 11.2. Giai đoạn chuyển sang train head frozen

| File | Nội dung |
|---|---|
| `src/config.py` | Frozen làm mặc định, chọn landmark checkpoint, đổi save_dir |
| `src/training.py` | Log số tham số trainable của trunk/head và trạng thái eval của trunk |
| `tests/test_model.py` | Kiểm tra gradient các thành phần head và tính ổn định của frozen features/buffers |
| `tests/test_pipeline.py` | Cập nhật lời gọi evaluation theo giao diện Python mới |
| `README.md` | Mô tả cấu hình frozen và cách chạy trực tiếp bằng code |

Không sửa kiến trúc `model.py`, `models_convgru.py`, backbone hoặc neck trong lượt chuyển chế độ.

### 11.3. Những kiểm tra đã pass

- 64 tests thuộc model, pipeline, backbone checkpoint và evaluation.
- Một optimizer step với pretrained landmark checkpoint thật và input tổng hợp: head đổi, toàn bộ weights và BatchNorm buffers của backbone/neck không đổi.
- Tham số trainable: trunk 0, head 510.723.
- Kiểm tra frozen features không phụ thuộc chế độ train/eval của head và ổn định khi chia batch, trong dung sai số học của test.
- Các test pipeline có kiểm tra checkpoint tự chứa và resume sau gián đoạn.
- Kiểm tra tĩnh và forward xác nhận P4 128×14×14, head không có BatchNorm, temporal pooling đều và đẳng thức trung bình logits ở eval.

Các kiểm tra này xác minh tính đúng của triển khai. Không phải bằng chứng hội tụ, chất lượng trên dữ liệu thật hoặc khả năng học quan hệ thời gian hữu ích.

## 12. Những kết luận được phép và chưa được phép rút ra

| Nhận định | Mức độ bằng chứng |
|---|---|
| Scratch best checkpoint nghiêng mạnh về alert | Đã đo trên 719 clip test |
| Có chênh lệch lớn khi thay cách BatchNorm chuẩn hóa | Đã quan sát trên 14 clip, kể cả train |
| BatchNorm là nguyên nhân duy nhất | Chưa chứng minh |
| Mất cân bằng nhãn toàn tập là nguyên nhân chính | Dữ liệu hiện có không ủng hộ mạnh |
| Có rủi ro domain/subject shift | Có bằng chứng về phân bố và metric theo nhóm |
| Model chắc chắn đang nhìn nền | Chưa chứng minh |
| Frozen trunk giữ nguyên weights và BN buffers | Đã kiểm tra |
| Frozen pretrained chắc chắn tăng recall | Chưa có kết quả run đầy đủ |
| ConvGRU có đường học thời gian và gradient | Đã xác nhận từ code/test |
| ConvGRU thực sự dùng lịch sử hữu ích trên người mới | Chưa chứng minh |
| Temporal pooling hiện là mean, không learned temporal attention | Đã xác nhận |
| Cần thay ConvGRU bằng kiến trúc lớn hơn | Chưa có cơ sở |

## 13. Theo dõi run tiếp theo

Quyết định hiện tại là chờ run frozen, chưa tiếp tục đổi kiến trúc/loss/pooling.

### Trước khi train

- Bắt đầu run mới tại `head_landmarkv1` hoặc thư mục trống khác.
- Kiểm tra `mode=frozen`.
- Kiểm tra log `backbone/neck=0` và `backbone/neck eval=True`.
- Kiểm tra pretrained path và config thực tế được lưu.
- Không resume scratch để thử chế độ frozen.

### Trong training

- Theo dõi val recall drowsy, confusion matrix, F1, AUC/AP và loss.
- Phân biệt model đoán một lớp với model có phân biệt nhưng threshold chưa phù hợp.
- Theo dõi train–val gap, nhưng nhớ train metrics có augmentation/dropout và weights đang thay đổi.
- Nếu cần chẩn đoán sâu, đánh giá thêm tập con train cố định trong eval.
- Chưa kết luận chỉ từ một epoch hoặc một metric.

### Khi có checkpoint đủ để đánh giá

- Chọn checkpoint theo validation, không theo test.
- Trỏ evaluation đến checkpoint frozen mới.
- Tách kết quả theo dataset/người.
- Kiểm tra clip sai và score distributions trước khi chỉnh threshold.
- Thực hiện temporal ablation để đo lợi ích của ConvGRU.

### Cách đọc những tình huống thường gặp

| Hiện tượng | Hướng kiểm tra |
|---|---|
| Train và val đều kém | Khả năng biểu diễn của frozen features, nhãn cửa sổ, tối ưu head |
| Train tốt, val kém | Domain/subject shift, overfitting, khác biệt preprocessing/augmentation |
| Val AUC khá nhưng recall ở ngưỡng hiện tại thấp | Phân bố điểm và lựa chọn threshold bằng validation |
| Val AUC gần ngẫu nhiên hoặc thấp hơn nhiều | Không chỉ là threshold; kiểm tra biểu diễn, nhãn và phân phối |
| Clip gốc và lặp một frame gần tương đương | Nghi ngờ phụ thuộc dấu hiệu tĩnh; kiểm tra trên nhiều frame/clip |
| ConvGRU không hơn baseline tĩnh | Chưa có bằng chứng lợi ích của recurrent head với dữ liệu/thiết lập đó |

## 14. Kiến thức rút ra

1. BatchNorm chuẩn hóa activation, không làm sạch dataset.
2. Batch optimizer lớn chưa chắc batch thống kê BatchNorm lớn; chunking thay đổi điều đó.
3. Gradient accumulation không sửa batch statistics của từng forward.
4. Train tốt và eval kém có thể do overfitting, chênh lệch chuẩn hóa hoặc cả hai.
5. Dùng cùng clip và chỉ đổi cơ chế chuẩn hóa là phép chẩn đoán hữu ích để tách nguyên nhân.
6. Đóng băng weights khác với đóng băng running statistics; cần quản lý cả gradient lẫn mode.
7. Frozen pretrained giúp train/inference nhất quán hơn, nhưng không bảo đảm phù hợp domain.
8. Checkpoint phải chứa kiến trúc, preprocessing, weights và buffers để đánh giá có ý nghĩa.
9. Chỉ số tổng có thể che lỗi nghiêm trọng ở một người hoặc dataset.
10. AUC và score distributions giúp phân biệt vấn đề threshold với vấn đề xếp hạng.
11. Có recurrent layer chưa chứng minh model cần thời gian để phân loại.
12. Có gradient qua chuỗi chưa chứng minh model nhớ dài hoặc nhớ hữu ích.
13. Spatial attention khác temporal attention; phải đọc phép tính thực tế thay vì tên chế độ.
14. Mean pooling trên hidden states vẫn có lịch sử, nhưng có thể làm yếu sự kiện ngắn.
15. Linear sau mean tương đương mean logits ở eval, không tương đương mean xác suất.
16. Sampling rate giới hạn tín hiệu có sẵn; model lớn không khôi phục được thông tin chưa lấy mẫu.
17. Số clip không đồng nghĩa số quan sát độc lập khi nhiều clip đến từ cùng người/video.
18. So sánh run cần tính cả số optimizer steps, batch hiệu dụng, initialization và hidden size.
19. Test triển khai pass không thay thế đánh giá chất lượng hoặc temporal ablation.
20. Giữ rõ ranh giới giữa dữ kiện, giả thuyết, đề xuất và thay đổi đã thực hiện giúp tránh sửa theo suy đoán.

## 15. Nguồn đối chiếu

Các đường dẫn tương đối dưới đây tính từ vị trí tài liệu này:

- [Config hiện tại](../../src/config.py).
- [Model wrapper và cơ chế frozen/chunking](../../src/model.py).
- [Adapter, ConvGRU và pooling](../../src/models_convgru.py).
- [BackboneNeck loader](../../src/model_backbone_neck.py).
- [Training loop](../../src/training.py).
- [Checkpoint loader/saver](../../src/checkpoint.py).
- [Dataloader và sampling](../../src/dataloader.py).
- [Augmentation](../../src/augment.py).
- [Evaluation](../../src/evaluation.py).
- [TensorBoard metrics](../../src/tensorboard.py).
- [Block Conv chứa BatchNorm](../../../ObjectDetection_2p6M/src/blocks.py).
- [Backbone và PAFPN](../../../ObjectDetection_2p6M/src/backbone_neck.py).
- [Khởi tạo BatchNorm](../../../ObjectDetection_2p6M/utils/init_weights.py).
- [Config run scratch](../../checkpoints/scratchv1/config.json).
- [Log run scratch](../../checkpoints/scratchv1/train.log).
- [Metrics test đã phân tích](../../../../runs/metrics.json).
- [Predictions đã phân tích](../../../../runs/predictions.csv).
- [Config evaluation đã phân tích](../../../../runs/evaluation_config.json).
- Manifest dữ liệu cục bộ: `/home/tranmanhduy/Workspace/dataE/DataLSTM/dataset_merged_split.csv`.
- [PyTorch BatchNorm2d: train/eval và running statistics](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.batchnorm.BatchNorm2d.html).
- [PyTorch: cập nhật thống kê BatchNorm với update_bn](https://docs.pytorch.org/docs/main/optim.html).
- [Cho et al., 2014: Learning Phrase Representations using RNN Encoder–Decoder](https://aclanthology.org/D14-1179/).

Artifacts cục bộ có thể không được đưa vào Git và có thể bị thay thế bởi các lần chạy sau. Các bảng số liệu trong tài liệu là bản ghi của lần phân tích nêu trên, không tự cập nhật khi artifacts thay đổi.
