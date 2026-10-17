# Phân tích tổ chức lại LSTM

## Mục tiêu

Tổ chức lại toàn bộ `ai/LSTM`, giảm mã lặp và comment dài, giữ code ngắn và dễ đọc. Giữ kiến trúc mô hình, định dạng checkpoint và hành vi huấn luyện hiện có; các lỗi hành vi phát hiện thêm cần được nêu riêng.

## Hiện trạng đã kiểm tra

- Khoảng 4.781 dòng trong các Python module và cấu hình YAML; có bốn notebook.
- `train_convgru.py` và `train_end2end.py` lặp seed, metrics, early stopping và nhiều phần điều phối huấn luyện. Hai luồng khác nhau ở cách xử lý backbone nên không nên gộp máy móc.
- `src/__init__.py` import `.evaluate` không tồn tại, xuất tên preprocessing chưa được định nghĩa và import cả trainer khi nạp package.
- Import đang trộn `src.*`, `configs.*`, `ai.LSTM.*` cùng nhiều đoạn chỉnh `sys.path`; có nguy cơ nhầm package với các thư mục AI khác.
- `dataloader.py` chứa cả đọc video, padding, dataloader và trích xuất backbone. `augment.py` cũng có hàm letterbox; cần đối chiếu hành vi trước khi dùng chung.
- `config.py` lặp chuyển đổi tuple và lọc trường giữa JSON/YAML; có alias cấu hình và tác dụng phụ lên môi trường khi khởi tạo.
- Cấu hình chứa đường dẫn ổ đĩa Windows cụ thể. Comment trong Python/YAML dài, nhắc nhiều tên module cũ.
- README mô tả model, notebook và module không khớp cây thư mục hiện tại.
- Notebook huấn luyện chứa nhiều mã triển khai riêng, dễ lệch khỏi scripts.
- `export_vulkan.py` phụ thuộc ExecuTorch, nhưng dependency này chưa được phân biệt trong hướng dẫn cài đặt.
- Chưa thấy thư mục test trong LSTM. Các thay đổi Git hiện có thuộc những phần khác của dự án.

## Hướng tổ chức đề xuất

- Giữ `src/`, `configs/`, `notebooks/`, `docs/`, `checkpoints/`, `logs/`; chỉ tách module khi trách nhiệm rõ ràng.
- Giữ hai entry point huấn luyện, gom phần dùng chung vào một module nhỏ. Tránh thêm hệ thống class hoặc abstraction không cần thiết.
- Tách trích xuất backbone khỏi xử lý dataset; chuẩn hóa import và thu gọn `__init__.py`.
- Gom logic đọc cấu hình; rút gọn docstring/comment, giữ giải thích về shape, mask, checkpoint và các quyết định khó hiểu.
- Đồng bộ notebook với module dùng lại, đồng thời ghi rõ cách chạy trên Kaggle và vị trí repository/dữ liệu cần thiết.
- Viết lại README theo cấu trúc thực tế; phân biệt dependency huấn luyện và export. Giữ nguyên tài liệu nghiên cứu, ảnh, video và artifacts.

## Kiểm chứng cần có khi triển khai

- Kiểm tra cú pháp, import package và entry point từ thư mục repository.
- Test cấu hình JSON/YAML, padding/mask chuỗi, metrics và early stopping.
- Smoke test forward/backward với tensor nhỏ và kiểm tra khả năng nạp state dict khi có đủ dependency.
- Kiểm tra notebook hợp lệ và tham chiếu module sau khi tổ chức lại.
- Chỉ xác nhận chạy notebook toàn bộ, huấn luyện dữ liệu thật hoặc export Vulkan nếu có đủ dữ liệu, checkpoint và runtime; báo rõ phần chưa chạy.

## Trạng thái

Đã khảo sát, chưa sửa mã nguồn. Theo mục 5 của `AGENTS.md`, cần duyệt phân tích này trước khi lập kế hoạch; kế hoạch cũng cần được duyệt trước khi triển khai.
