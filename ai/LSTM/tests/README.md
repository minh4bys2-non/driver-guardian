# Kiểm tra pipeline training

Chạy từ repository root:

```bash
NO_ALBUMENTATIONS_UPDATE=1 python -m pytest ai/LSTM/tests -q
```

Các test chạy CPU, dùng thư mục tạm và không tải model/dataset bên ngoài. Test video cần OpenCV hỗ trợ ghi/đọc MP4 (`mp4v`).

- `test_model.py`: forward, padding, gradient và cập nhật đúng phần trainable trong hai chế độ scratch/frozen.
- `test_convgru.py`: truyền hidden state qua từng đoạn, attention/padding, đầu vào multi-scale/adapted và từ chối độ dài thập phân hoặc tensor rỗng.
- `test_augment.py`: letterbox giữ RGB/tỷ lệ/căn giữa, replay/reset video seed và giữ nguyên nguyên nhân lỗi augmentation.
- `test_backbone_checkpoint.py`: kiến trúc từ metadata, ưu tiên EMA và từ chối thiếu trọng số.
- `test_pipeline.py`: config, checkpoint nguyên tử, khôi phục RNG, resume giữa epoch đã hoàn tất, best F1/loss, khôi phục best.pt, accumulation, logging và export wrapper.
- `test_data_validation.py`: MP4/manifest thật, RGB, augmentation chỉ ở train, lỗi dữ liệu, trùng subject/source/path giữa các split và chạy train qua dataloader thật.
- `test_evaluation.py`: confusion matrix, F1, AUC/AP, loss theo số mẫu, tập một lớp và tập rỗng.
- `test_tensorboard.py`: ghi và đọc lại các sự kiện TensorBoard.

Resume được so sánh với chạy liên tục cho cả scratch/frozen trên CPU; frozen resume không còn file pretrained gốc. Test integration dùng video nhỏ tự tạo, không chứng minh khả năng hội tụ trên UL-DD. Chưa kiểm chứng CUDA AMP, dataloader nhiều worker hoặc biên dịch ExecuTorch/Vulkan.
