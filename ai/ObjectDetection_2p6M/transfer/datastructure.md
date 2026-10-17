# Cấu trúc dữ liệu face landmark

Tài liệu mô tả dữ liệu được đọc bởi `dataloader_lmk.py`. Các giá trị mặc định
được đối chiếu với `config_lmk.py`.

## 1. Tổ chức thư mục

Mỗi tập dữ liệu (split) có một thư mục ảnh và một file nhãn JSONL:

```text
DataTransferSplit/
├── train/
│   ├── images/
│   │   ├── face_001.jpg
│   │   └── group_a/
│   │       └── face_002.png
│   └── annotations.jsonl
└── val/
    ├── images/
    │   └── face_003.jpg
    └── annotations.jsonl
```

`train/` và `val/` là cách tổ chức đề xuất. Khi không có `val_root_dir` và
`val_ratio > 0`, `build_loaders` chia nội bộ theo record gốc với seed cố định.
Mỗi `FaceLandmarkDataset` đọc một `root_dir` riêng, ví dụ
`DataTransferSplit/train`.

| Cấu hình | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `root_dir` | Do người dùng cung cấp | Thư mục gốc của split |
| `images_dir_name` | `images` | Thư mục ảnh bên trong root |
| `jsonl_name` | `annotations.jsonl` | File annotation bên trong root |
| `index_cache_dir` | `./cache_face_lmk_indices` | Thư mục cache chỉ mục, tương đối với thư mục chạy chương trình |

Đường dẫn được ghép như sau:

```text
annotation = root_dir / jsonl_name
image      = root_dir / images_dir_name / record["file_name"]
```

Ví dụ `file_name = "group_a/face_002.png"` trỏ tới
`train/images/group_a/face_002.png`. Nên lưu `file_name` tương đối với thư mục
ảnh để có thể di chuyển toàn bộ dataset mà không sửa nhãn.

Không cần file nhãn riêng cho từng ảnh. Dataloader lấy danh sách mẫu từ JSONL,
không quét toàn bộ thư mục ảnh; ảnh không được tham chiếu trong JSONL sẽ không
được đọc. Ảnh được mở bằng Pillow và chuyển sang RGB.

## 2. Định dạng annotation JSONL

File dùng UTF-8, mỗi dòng không rỗng là một JSON object hoàn chỉnh mô tả một
ảnh. Không bọc toàn bộ file trong mảng JSON và không thêm dấu phẩy giữa các
dòng. Dòng trống được bỏ qua khi lập chỉ mục.

### Schema của một record

| Trường | Kiểu dữ liệu | Nội dung |
| --- | --- | --- |
| `file_name` | string | Tên hoặc đường dẫn ảnh tương đối với thư mục ảnh |
| `image_width` | number | Chiều rộng ảnh gốc, tính bằng pixel, lớn hơn 0 |
| `image_height` | number | Chiều cao ảnh gốc, tính bằng pixel, lớn hơn 0 |
| `faces` | array | Danh sách khuôn mặt trong ảnh |
| `faces[i].bounding_box_normalized` | object | Bbox gồm `xmin`, `ymin`, `xmax`, `ymax` |
| `faces[i].landmarks_normalized` | array | Danh sách K điểm, mỗi điểm có `x`, `y` |

Một ảnh có thể chứa nhiều khuôn mặt. Bbox và landmark trong cùng phần tử
`faces[i]` phải thuộc cùng một khuôn mặt. Các trường bổ sung như `z`, confidence
hoặc class trong annotation không được dataloader này sử dụng; class đầu ra
luôn là `0` (face).

Ví dụ dưới đây được trình bày nhiều dòng cho dễ đọc và rút gọn còn 3 landmark.
Khi lưu JSONL thực tế, ghi toàn bộ record trên **một dòng** và cung cấp đủ
**478 điểm theo đúng thứ tự MediaPipe** để dùng cấu hình mặc định:

```json
{
  "file_name": "group_a/face_002.png",
  "image_width": 640,
  "image_height": 480,
  "faces": [
    {
      "bounding_box_normalized": {
        "xmin": 0.25,
        "ymin": 0.20,
        "xmax": 0.75,
        "ymax": 0.85
      },
      "landmarks_normalized": [
        {"x": 0.40, "y": 0.35},
        {"x": 0.60, "y": 0.35},
        {"x": 0.50, "y": 0.65}
      ]
    }
  ]
}
```

### Quy ước tọa độ

Tất cả tọa độ lưu trên đĩa được chuẩn hóa theo **toàn bộ ảnh gốc**, không theo
vùng crop khuôn mặt:

```text
x_normalized = x_pixel / image_width
y_normalized = y_pixel / image_height
```

Gốc tọa độ nằm ở góc trên trái; x tăng sang phải, y tăng xuống dưới. Bbox dùng
hai góc `xyxy`, với `xmin < xmax` và `ymin < ymax`. Tọa độ mong đợi nằm trong
`[0, 1]`; bộ kiểm tra cho phép dung sai `normalized_coordinate_tolerance`
(mặc định `0.0001`). Không dùng NaN hoặc Inf.

K phải nhất quán giữa tất cả khuôn mặt trong cùng dataset. Dataloader suy ra
K từ annotation; nếu bật horizontal flip (`random` hoặc `paired`) thì bắt buộc
K = 478. Thứ tự landmark phải khớp MediaPipe vì phép lật dùng bảng hoán vị
`MEDIAPIPE_478_FLIP_PERMUTATION`.

### Điều kiện dữ liệu hợp lệ

- Mỗi record phải tham chiếu tới ảnh tồn tại và đọc được.
- `image_width`, `image_height` phải khớp kích thước ảnh thực tế. Bộ kiểm tra
  dùng kích thước trong JSONL, còn `__getitem__` dùng kích thước ảnh đọc từ đĩa;
  hiện không có bước đối chiếu hai nguồn kích thước này.
- Chiều rộng và chiều cao bbox sau resize theo letterbox phải đạt
  `min_box_size_px` (mặc định 48 pixel), không phải 48 pixel trên ảnh gốc.
- Mặc định `allow_empty_targets=False`: mỗi ảnh phải có ít nhất một khuôn mặt.
  Khi bật tùy chọn này, `faces: []` hoặc thiếu `faces` được xem là mẫu rỗng,
  nhưng toàn dataset vẫn phải có landmark để suy ra K.
- Mặc định `strict_dataset_schema=True`: phát hiện lỗi schema sẽ dừng khởi tạo
  và báo tối đa 20 lỗi chi tiết. Tắt chế độ này chỉ chuyển các lỗi đã thu thập
  thành cảnh báo, không tự sửa hoặc bỏ qua record lỗi.

## 3. Cache chỉ mục JSONL

Dataloader tự tạo mảng NumPy `int64` chứa vị trí byte bắt đầu của từng dòng
không rỗng để truy cập ngẫu nhiên mà không phải nạp toàn bộ annotation vào RAM.

```text
cache_face_lmk_indices/
└── annotations.jsonl.<hash_16_ký_tự>.idx.npy
```

Hash là 16 ký tự đầu của SHA-256 trên đường dẫn thực (`realpath`) của JSONL,
giúp phân biệt các split có cùng tên file nhãn. Cache được dùng lại khi thời
gian sửa đổi của cache không cũ hơn JSONL; nếu không, chỉ mục được xây lại.
Cache chỉ chứa offset, không chứa ảnh hoặc nhãn đã biến đổi và không cần chuẩn
bị thủ công. Kiểm tra schema vẫn chạy khi khởi tạo dataset, kể cả khi có cache.

## 4. Dữ liệu sau khi đọc

Ảnh và nhãn trên đĩa giữ nguyên. Khi lấy mẫu, dataloader thực hiện letterbox,
lật ngang nếu được cấu hình, augmentation hình học, ColorJitter rồi chuyển
ảnh thành tensor. Các biến thể này được tạo trong bộ nhớ, không lưu thêm ảnh.

Với ảnh gốc W × H và kích thước đích S × S:

```text
scale = min(S / W, S / H)
new_w = round(W * scale)
new_h = round(H * scale)
pad_x = (S - new_w) // 2
pad_y = (S - new_h) // 2
x_output = x_normalized * W * scale + pad_x
y_output = y_normalized * H * scale + pad_y
```

Cấu hình `TrainConfig` hiện dùng S = 480, padding RGB `(114, 114, 114)`.
Bbox và landmark đầu ra là tọa độ **pixel trên ảnh đã biến đổi**.

### Một mẫu từ `dataset[index]`

N là số khuôn mặt còn lại sau xử lý, K là số landmark:

| Khóa | Kiểu / shape | Ý nghĩa |
| --- | --- | --- |
| `image` | float32 `[3, S, S]` | Ảnh RGB, giá trị trong `[0, 1]` |
| `boxes` | float32 `[N, 4]` | Bbox pixel theo thứ tự `xmin, ymin, xmax, ymax` |
| `labels` | int64 `[N]` | Toàn bộ bằng `0` |
| `landmarks` | float32 `[N, K, 2]` | Tọa độ pixel `(x, y)` |
| `landmarks_valid` | bool `[N]` | Mặt nạ hợp lệ cho landmark của từng khuôn mặt, không phải từng điểm |
| `file_name` | string | Tên ảnh lấy từ record |
| `orig_size` | int64 `[2]` | `[width, height]` của ảnh thực tế trước xử lý |
| `was_flipped` | bool | Mẫu có được lật ngang hay không |
| `geometric_aug` | string | Phép biến đổi đã áp dụng, như `none`, `affine`, `affine+perspective+radial`, `fallback_original` |

Augmentation hình học có thể loại bbox không đạt điều kiện hoặc đánh dấu
`landmarks_valid=False`. Nếu không còn bbox và không cho phép target rỗng,
dataloader quay lại ảnh/nhãn trước augmentation hình học (`fallback_original`).

### Một batch từ `face_landmark_collate`

| Khóa | Kiểu / shape |
| --- | --- |
| `image` | float32 `[B, 3, S, S]` |
| `targets` | List dài B; mỗi phần tử chứa `boxes`, `labels`, `landmarks`, `landmarks_valid` |
| `file_name` | List B string |
| `orig_size` | int64 `[B, 2]`, thứ tự `[width, height]` |
| `was_flipped` | bool `[B]` |
| `geometric_aug` | List B string |

Số khuôn mặt có thể khác nhau giữa các ảnh nên `targets` được giữ dưới dạng
list, không stack hoặc padding thành một tensor chung.

Chế độ `paired` mặc định khi train tạo hai chỉ số cho mỗi record: `2*i` là
mẫu không lật, `2*i+1` là mẫu lật. Dataset có độ dài gấp đôi số record nhưng
không cần nhân đôi dữ liệu trên đĩa; các augmentation ngẫu nhiên khác vẫn có
thể khác nhau giữa hai mẫu.

## 5. Resume và chia split

`build_loaders` tạo batch sampler xác định thứ tự theo seed/epoch. Mỗi sample
nhận seed riêng, giúp augmentation không phụ thuộc số worker hoặc prefetch.
Resume bỏ qua trực tiếp các batch đã hoàn thành. Validation không shuffle,
không augmentation và giữ batch cuối dù không đủ batch size; train drop-last.
Nếu không có batch train sau chia, chương trình báo lỗi để điều chỉnh batch size.
