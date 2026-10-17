# Báo cáo cleanup LSTM

## Đã thực hiện

- Gom seed, metrics, early stopping, train/validation, checkpoint và CLI vào `src/training.py`; hai trainer chỉ giữ phần dựng model và forward riêng.
- Tách `extractor.py` khỏi dataset; extractor dùng wrapper backbone nội bộ, không import công cụ chuyển đổi ONNX. Đường dẫn checkpoint của extractor được truyền tường minh.
- Gom letterbox vào `preprocess.py`; giữ default kích thước cũ tại API augmentation.
- Chuẩn hóa namespace `ai.LSTM`, bỏ chỉnh `sys.path`, cấu hình stdout/OpenMP và khởi tạo augmenter khi import module. Bỏ các export trỏ đến thành phần không tồn tại trong `src/__init__.py`.
- Gom chuẩn hóa JSON/YAML, rút gọn comment/docstring và import thừa. Đổi đường dẫn Windows trong cấu hình mặc định sang đường dẫn repository; chuyển thông báo runtime sang logging.
- Notebook gọi module chung; chuyển cấu hình thí nghiệm vào ba preset YAML. Đã đối chiếu giá trị từng trường với bản notebook trước khi sửa.
- Thay demo low-light vốn tham chiếu module bị thiếu bằng demo letterbox/augmentation ảnh và video.
- Thêm CLI cho trích xuất checkpoint và export; chỉ import ExecuTorch khi thực sự export, không tự chọn checkpoint fallback khi người dùng truyền đường dẫn sai.
- Thu gọn requirements, thêm requirements cho test, cập nhật README và ignore dữ liệu/artifacts cục bộ. Giữ tài liệu nghiên cứu và media.

## Sửa lỗi đi kèm

- Manifest tương đối được tìm từ `dataset_dir`, tránh bỏ qua manifest khi chạy từ repository root.
- Ghi checkpoint dùng `Path.replace()` trực tiếp, tránh khoảng trống do xóa file đích trước khi rename.
- `save_checkpoint(is_best=True, val_f1=...)` cập nhật best score nhất quán giữa hai pipeline.

## Kiểm chứng

- `python -m pytest ai/LSTM/tests -q`: **11 passed** trên CPU.
- Các test bao gồm đọc video tạo tạm và manifest, padding/video lỗi, augmentation nhất quán theo frame, JSON/YAML, metrics/early stopping, mask gradient, BatchNorm và gradient checkpointing, train/validation một epoch và resume của cả hai pipeline, cùng wrapper xác suất export.
- Đối chiếu bản nguồn trước refactor với bản mới cho `ConvGRUClassifier` và `DrowsyClassifier`: state dict, đầu ra và gradient bằng nhau chính xác trên các ca tensor nhỏ CPU với seed cố định.
- Compile Python, kiểm tra cú pháp mọi code cell và CLI `--help` của hai trainer, trích xuất backbone, export đều qua.
- `git diff --check` không báo lỗi whitespace.
- Python trong `src/` và `configs/`: **4.368 → 2.861 dòng**. Code cell trong notebook: **6.833 → 228 dòng**; cấu hình notebook được chuyển sang YAML, không còn chứa bản sao model/trainer.

## Giới hạn và cách chuyển sang cấu trúc mới

- Chạy CLI bằng `python -m ai.LSTM...` từ repository root; import trực tiếp module thay cho các re-export cũ ở `src`.
- Notebook cần toàn bộ repository trên Kaggle. Giá trị preset được giữ, nhưng các bản kiến trúc nhúng trước đây nay dùng implementation chung trong scripts. Chưa xác nhận checkpoint của mọi biến thể notebook lịch sử tương thích.
- Đã chạy các cell demo ảnh. Cell đọc video thật không chạy được vì OpenCV hiện tại không giải mã AV1 của `docs/video/video.mp4`; cần decoder phù hợp hoặc đổi video trong notebook.
- Chưa chạy training notebook toàn bộ với dataset/checkpoint riêng, benchmark CUDA hoặc biên dịch ExecuTorch/Vulkan. Test train dùng dữ liệu nhỏ; backbone của test frozen pipeline được khởi tạo nhỏ qua fixture thay vì nạp checkpoint bên ngoài.
- Giữ hành vi mô hình và thuật toán tối ưu hiện có; đây không phải đợt đánh giá lại chất lượng mô hình hay mọi edge case thuật toán.
