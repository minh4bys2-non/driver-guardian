# Kế hoạch cleanup

Đã được người dùng cho phép triển khai toàn bộ sau khi duyệt phân tích.

1. Chuẩn hóa import package; tách extractor khỏi dataset.
2. Gom vòng huấn luyện, metrics, seed và early stopping; giữ hai entry point.
3. Gom đọc cấu hình; rút gọn comment/docstring và bỏ demo hardcode trong module.
4. Chuyển notebook sang gọi module, giữ cấu hình thí nghiệm và bước đánh giá.
5. Đồng bộ README, thêm test hồi quy và ghi báo cáo kiểm chứng.
