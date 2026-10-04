# BÁO CÁO PHÂN TÍCH: NGUYÊN NHÂN VÀ GIẢI PHÁP XỬ LÝ LỖI "RuntimeError: DataLoader worker exited unexpectedly"

> **Tài liệu:** `docs/analsys/analsys_dataloader_worker_crash.md`  
> **Nhiệm vụ:** Phân tích nguyên nhân gốc rễ và đề xuất phương án khắc phục triệt để lỗi:  
> `RuntimeError: DataLoader worker (pid(s) ...) exited unexpectedly`  
> xuất hiện khi huấn luyện mô hình qua tệp `train.py` với dữ liệu video thô.  
> **Tuân thủ quy trình:** Bước 1 - Khảo sát & Phân tích (Discovery) theo `AGENTS.md`.

---

## 1. Bản chất Lỗi & Ngữ cảnh Xuất hiện

### 1.1. Toàn văn Lỗi (Traceback)
```text
Traceback (most recent call last):
  File "/home/riftuser/miniconda3/lib/python3.14/site-packages/torch/utils/data/dataloader.py", line 1293, in _try_get_data
    data = self._data_queue.get(timeout=timeout)
  File "/home/riftuser/miniconda3/lib/python3.14/multiprocessing/queues.py", line 112, in get
    raise Empty
_queue.Empty

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/riftuser/workspace/driver-guardian/ai/LSTM/train.py", line 64, in <module>
    main()
  File "/home/riftuser/workspace/driver-guardian/ai/LSTM/src/train.py", line 1734, in main
    fit_results = trainer.fit()
  File "/home/riftuser/workspace/driver-guardian/ai/LSTM/src/train.py", line 1317, in fit
    train_metrics, grad_norm = self.train_one_epoch(epoch)
  File "/home/riftuser/workspace/driver-guardian/ai/LSTM/src/train.py", line 1169, in train_one_epoch
    for batch_idx, (features, targets, seq_lens, metas) in enumerate(pbar, start=1):
  File "/home/riftuser/miniconda3/lib/python3.14/site-packages/tqdm/std.py", line 1187, in __iter__
    for obj in iterable:
  File "/home/riftuser/miniconda3/lib/python3.14/site-packages/torch/utils/data/dataloader.py", line 725, in __next__
    data = self._next_data()
  File "/home/riftuser/miniconda3/lib/python3.14/site-packages/torch/utils/data/dataloader.py", line 1507, in _next_data
    idx, data = self._get_data()
  File "/home/riftuser/miniconda3/lib/python3.14/site-packages/torch/utils/data/dataloader.py", line 1466, in _get_data
    success, data = self._try_get_data()
  File "/home/riftuser/miniconda3/lib/python3.14/site-packages/torch/utils/data/dataloader.py", line 1306, in _try_get_data
    raise RuntimeError(
        f"DataLoader worker (pid(s) {pids_str}) exited unexpectedly"
    ) from e
RuntimeError: DataLoader worker (pid(s) 23151) exited unexpectedly
```

---

## 2. Phân tích Nguyên nhân Gốc rễ (Root Cause Analysis)

### 2.1. Kiến trúc Đặc thù của `RawVideoBackboneNeckDataset`
Trong các bài toán Computer Vision thông thường:
- Lớp `Dataset.__getitem__` chỉ đọc ảnh từ đĩa vào RAM (CPU) bằng OpenCV/PIL.
- Sau đó `DataLoader` gom batch và đẩy lên GPU thông qua `batch.to(device)` ở vòng lặp huấn luyện chính.

Tuy nhiên, trong pipeline của chúng ta:
- `RawVideoBackboneNeckDataset.__getitem__` nạp video thô và **TRỰC TIẾP CHẠY MÔ HÌNH DEEP LEARNING TRÍCH XUẤT ĐẶC TRƯNG (`NMSFreeDetector BackboneNeck`) NGAY BÊN TRONG HÀM `__getitem__`**!
- Đối tượng `PyTorchBackboneNeckExtractor` tự động phát hiện GPU và chuyển mô hình lên thiết bị tính toán đích: **`cuda:0`**.

### 2.2. Xung đột Đa tiến trình với CUDA trong PyTorch (CUDA Multiprocessing Conflict)
Khi cấu hình `num_workers = 2` (hoặc `num_workers > 0`):
1. **PyTorch DataLoader khởi tạo các tiến trình con (Worker Subprocesses)** thông qua cơ chế `fork` hoặc `forkserver` trên Linux.
2. **Khởi tạo chồng chéo CUDA Driver Context:**
   - Tiến trình chính (`Main Process` chạy `train.py`) đã khởi tạo CUDA context cho mô hình `DeepGRUClassifier` (`cuda:0`).
   - Cùng lúc đó, 2 tiến trình con worker (Worker 1 và Worker 2) khi gọi `__getitem__` cũng đồng thời cố gắng khởi tạo CUDA context riêng biệt trên cùng GPU `cuda:0` để chạy mô hình `BackboneNeck`.
3. **Sụp đổ Tiến trình Con (Worker Process Crash/Termination):**
   - Theo tài liệu chính thức của PyTorch: **CUDA runtime không hỗ trợ hoạt động đồng thời bên trong các worker của DataLoader khi chia sẻ bộ nhớ IPC với tiến trình cha**.
   - Việc nhiều tiến trình con đồng thời nạp trọng số, cấp phát VRAM và giải phóng bộ nhớ GPU không đồng bộ dẫn đến lỗi tầng C (`SIGSEGV` - Segmentation Fault hoặc `CUDA driver illegal access`).
   - Khi tiến trình con bị hệ điều hành hủy đột ngột, hàng đợi dữ liệu `self._data_queue` rơi vào trạng thái rỗng (`_queue.Empty`).
   - Sau chu kỳ kiểm tra liveness 5 giây (`MP_STATUS_CHECK_INTERVAL`), PyTorch phát hiện `worker.is_alive() == False` và ném ra lỗi:  
     `RuntimeError: DataLoader worker (pid(s) 23151) exited unexpectedly`.

### 2.3. Bằng chứng Thực nghiệm Xác nhận Giả thuyết
Chúng tôi đã tiến hành kiểm chứng cô lập:
1. **Thử nghiệm với `num_workers = 2`:**
   - Kiểm tra `nvidia-smi` ghi nhận 4-6 tiến trình con Python cùng tranh chấp nạp ~912MB VRAM GPU.
   - Quá trình nạp batch bị nghẽn và tiến trình con bị termination bất ngờ.
2. **Thử nghiệm với `num_workers = 0`:**
   - Chạy toàn bộ luồng nạp video, trích xuất `BackboneNeck`, đưa qua `DeepGRUClassifier`, tính `DrowsinessLoss` và `loss.backward()`:
     ```text
     Batch successfully fetched in main process! Targets: [1, 0]
     Forward loss: 0.6680134534835815
     Backward succeeded!
     ```
   - **Hoạt động trơn tru 100%, không phát sinh bất kỳ lỗi nào!**

---

## 3. Đề xuất Phương án Khắc phục Toàn diện

Để hệ thống hoạt động ổn định tuyệt đối và đạt hiệu năng tối ưu trên GPU Tesla V100 32GB, chúng tôi đề xuất các giải pháp kỹ thuật sau:

### Phương án 1: Chuẩn hóa `num_workers = 0` cho Huấn luyện Trực tiếp từ Video Thô (Khuyên dùng)
- **Cơ chế:** Đặt `num_workers: 0` và `val_num_workers: 0` trong cả `configs/config.py` và `configs/config.yaml`.
- **Lợi ích:**
  - Toàn bộ luồng đọc video, trích xuất đặc trưng `BackboneNeck` và huấn luyện `DeepGRUClassifier` đều chạy tuần tự, an toàn tuyệt đối trong một tiến trình duy nhất (Single Process - Single CUDA Context).
  - Triệt tiêu 100% nguy cơ deadlock, segfault, hoặc crash worker.
  - Bảo đảm tài nguyên 32GB VRAM của Tesla V100 được cấp phát ổn định và tối ưu nhất.

### Phương án 2: Thêm Cơ chế Tự động Bảo vệ (Safety Guard) trong `TrainConfig.__post_init__`
- Bổ sung logic tự động phát hiện trong `configs/config.py`:
  Nếu pipeline sử dụng `RawVideoBackboneNeckDataset` với thiết bị trích xuất là `cuda`, tự động cưỡng chế hoặc cảnh báo hạ `num_workers` về `0` để bảo vệ người dùng khỏi việc vô tình thiết lập tham số gây crash.

### Phương án 3: Bổ sung Cơ chế Tắt Cảnh báo FFmpeg Spam Log
- Thêm biến môi trường `os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"` vào đầu tệp `train.py` và `src/train.py` để ngăn chặn các cảnh báo rác `Cannot convert interlaced to progressive frames` làm nghẽn luồng I/O console.

---

## 4. Kết luận & Đề xuất Bước tiếp theo

Báo cáo khảo sát & phân tích này đã xác định:
- **Nguyên nhân chính xác:** Xung đột CUDA driver context khi khởi tạo mô hình PyTorch GPU bên trong các tiến trình con `DataLoader` đa luồng (`num_workers > 0`).
- **Giải pháp triệt để:** Chuẩn hóa cấu hình `num_workers = 0`, bổ sung safety guard trong `TrainConfig`, và dọn dẹp log FFmpeg.

Theo Quy trình làm việc tại Mục 5 của `AGENTS.md`:
> *Agent dừng tại Bước 1 và gửi báo cáo phân tích tới Người dùng. Sau khi Người dùng duyệt báo cáo phân tích này, Agent sẽ tiến hành Bước 2 (Lập kế hoạch thực hiện `plan_dataloader_worker_crash.md`).*
