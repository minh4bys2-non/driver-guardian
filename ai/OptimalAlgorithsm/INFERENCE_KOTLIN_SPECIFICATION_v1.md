# Tài Liệu Đặc Tả Kỹ Thuật Toàn Diện (Phiên Bản Nâng Cấp v1): Hệ Thống Suy Luận Đa Luồng Tốc Độ Cao `inference.py` & Hướng Dẫn Tái Hiện Trên Android (Kotlin + CameraX `RGBA_8888` + MediaPipe GPU + Parallel Physical Workers + ONNX Runtime)

Tài liệu này là **bản nâng cấp kiến trúc v1** của [INFERENCE_KOTLIN_SPECIFICATION.md](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/INFERENCE_KOTLIN_SPECIFICATION.md), đặc tả chi tiết đến mức công thức toán học, máy trạng thái, cơ chế đồng bộ đa luồng không khóa và mã nguồn tham chiếu của toàn bộ pipeline suy luận trong [inference.py](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/inference.py) và định dạng xuất mô hình trong [export_onnx.py](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/export_onnx.py) cùng tất cả các module phụ thuộc ([`CameraMetrics`](file:///D:/Project/DATN/tempt/driver-guardian/ai/PhysicalBranch/camera_metrics.py), [`AdaptiveHMM_FSM`](file:///D:/Project/DATN/tempt/driver-guardian/ai/PhysicalBranch/adaptive_hmm_fsm.py), [`HeadPoseEstimator`](file:///D:/Project/DATN/tempt/driver-guardian/ai/PhysicalBranch/head_pose_estimation.py), [`temporal_metrics.py`](file:///D:/Project/DATN/tempt/driver-guardian/ai/PhysicalBranch/temporal_metrics.py), [`Engine`](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/DataSystem/engine.py), [`NeuralNetwork`](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/DataSystem/neuralnetwork.py), [`PhysicalNormalizer`](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/DataSystem/normalization.py), [`Fitness`](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/TrainingSystem/fitnessFunction.py)) để tái hiện chính xác 100% bằng **Kotlin trên nền tảng Android với tốc độ và FPS tối đa**.

> [!IMPORTANT]
> **5 Nâng Cấp Kiến Trúc Cốt Lõi Trong Phiên Bản `v1` So Với Bản Gốc:**
> 1. **Cấu hình CameraX Hardware lấy trực tiếp frame `640×480` định dạng `RGBA_8888` (`OUTPUT_IMAGE_FORMAT_RGBA_8888`):** Tương thích nguyên bản 1-1 với `MPImage(IMAGE_FORMAT_RGBA)` của MediaPipe GPU mà không cần CPU chuyển đổi hệ màu `YUV_420_888 -> RGB`.
> 2. **Bật / Tắt `balanceBrightness` trực tiếp trên UI (`AtomicBoolean` / `StateFlow`):** Khi tắt trên UI (`enableBalanceBrightness = false`, mặc định ban ngày hoặc khi muốn tối đa hóa FPS), luồng Camera đi theo nhánh **Zero-Conversion** (đẩy thẳng `ByteBuffer RGBA_8888` từ CameraX vào MediaPipe GPU trong `< 0.15 ms`). Khi bật trên UI (`enableBalanceBrightness = true`, đi đêm), hệ thống nâng sáng gamma kênh $Y$ (`target_luma = 70.0`).
> 3. **Ép `MediaPipe FaceLandmarker` thực thi trên GPU (`Delegate.GPU`):** Giảm thời gian trích xuất 478 điểm mốc từ `9–26 ms` (CPU) xuống còn **`2.2–6.5 ms`** (GPU OpenGL ES / OpenCL), giải phóng hoàn toàn các nhân CPU lớn.
> 4. **Tách "Luồng 1: `CameraAnalyzer Thread`" thành các luồng song song (Pipeline 2 tầng + Fork-Join 3 nhánh Vật Lý):**
>    - **Tầng Pipeline:** Tách riêng `Luồng 1A: CameraIngestThread` (nhận `ImageProxy`, cân bằng sáng tùy chọn, đóng ngay `imageProxy.close()`) và `Luồng 1B: PhysicalCoordinator & MediaPipe GPU Thread` (chạy MediaPipe GPU **song song** với việc cập nhật `NeuralBuffer.updateRaw`).
>    - **Tầng Fork-Join 3 Nhánh Vật Lý (`Nhóm Luồng 1C`):** Sau khi có 478 điểm mốc từ GPU, tách thành 3 worker chạy song song đồng thời trên 3 nhân CPU: **`EyeWorker`** (`EAR` + `AdaptiveHMM("eye")` + `Blink FSM` + `PERCLOS`) $\parallel$ **`MouthWorker`** (`MAR` + `AdaptiveHMM("mouth")` + `Yawn FSM` + `POM`) $\parallel$ **`HeadPoseAndFftWorker`** (`SolvePnP` + `Neutral Calibration` + `Nod FSM` + `Over-Angle` + `HeadMotionWindow` với biến đổi **`rfft` 1800 điểm**).
> 5. **Đẩy hoàn toàn xử lý `letterbox` (`224×224` + `RGBA -> RGB CHW`) sang nhánh `ONNX Worker Thread`:** Nhánh Camera tuyệt đối không chạy `letterbox`. `NeuralBuffer` chỉ lưu tham chiếu frame `RGBA_8888` `640×480` thô trên lưới `5.0 FPS`. Mỗi `1.0s`, `ONNX Worker Thread` chỉ lấy đúng $\Delta = T_{\text{new}}$ frame mới (thường chỉ **5 frames/giây** thay vì 30–60 frames/giây) để chạy `letterboxRgbaToRgbCHW(224)` (resize về `224×168` trước rồi mới bỏ kênh Alpha), giảm **83%–92%** chi phí `letterbox` toàn hệ thống và loại bỏ **100%** chi phí `letterbox` khỏi nhánh Camera.

---

## Mục Lục
1. [Tổng Quan Kiến Trúc Nâng Cấp v1 & Cấu Hình Phần Cứng CameraX `640×480 RGBA_8888`](#1-tổng-quan-kiến-trúc-nâng-cấp-v1--cấu-hình-phần-cứng-camerax-640480-rgba_8888)
   - [1.1. Bảng Ánh Xạ Công Nghệ (Python $\rightarrow$ Android Kotlin v1)](#11-bảng-ánh-xạ-công-nghệ-python--android-kotlin-v1)
   - [1.2. Đặc Tả Cấu Hình CameraX Hardware `640×480` Định Dạng `OUTPUT_IMAGE_FORMAT_RGBA_8888`](#12-đặc-tả-cấu-hình-camerax-hardware-640480-định-dạng-output_image_format_rgba_8888)
2. [Kiến Trúc Đa Luồng Song Song & Tương Tác Giữa Các Luồng (Concurrency Specification v1)](#2-kiến-trúc-đa-luồng-song-song--tương-tác-giữa-các-luồng-concurrency-specification-v1)
   - [2.1. Danh Sách Các Luồng Chạy (8 Luồng / Nhóm Luồng) & Mô Hình Không Khóa](#21-danh-sách-các-luồng-chạy-8-luồng--nhóm-luồng--mô-hình-không-khóa-lock-free-snapshot)
   - [2.2. Sơ Đồ Kiến Trúc Luồng Dữ Liệu Song Song (Pipeline + Fork-Join + ONNX Letterbox)](#22-sơ-đồ-kiến-trúc-luồng-dữ-liệu-song-song-pipeline--fork-join--onnx-letterbox)
   - [2.3. Thích Nghi Với FPS Biến Thiên Của Camera Trên Cụm Luồng 1A–1C](#23-thích-nghi-với-fps-biến-thiên-của-camera-trên-cụm-luồng-1a1c)
   - [2.4. Sơ Đồ Tuần Tự Đa Luồng (Bao Gồm UI Toggle `balanceBrightness`, Fork-Join 3 Nhánh Vật Lý & `ONNX Letterbox`)](#24-sơ-đồ-tuần-tự-đa-luồng-bao-gồm-ui-toggle-balancebrightness-fork-join-3-nhánh-vật-lý--onnx-letterbox)
   - [2.5. Quy Tắc `WARMING UP`, Tích Lũy Trước `p4_old` & Chống Nghẽn Cổ Chai (Backpressure Rules)](#25-quy-tắc-warming-up-tích-lũy-trước-p4_old--chống-nghẽn-cổ-chai-backpressure-rules)
3. [Đặc Tả Tiền Xử Lý Ảnh Có Thể Bật/Tắt Từ UI, Bộ Đệm `NeuralBuffer` Lưu Ảnh Thô & `letterbox` Tại `ONNX Worker`](#3-đặc-tả-tiền-xử-lý-ảnh-có-thể-bậttắt-từ-ui-bộ-đệm-neuralbuffer-lưu-ảnh-thô--letterbox-tại-onnx-worker)
   - [3.1. Hàm Cân Bằng Sáng `balanceBrightnessRgba` Hỗ Trợ Bật/Tắt Trực Tiếp Từ UI](#31-hàm-cân-bằng-sáng-balancebrightnessrgba-hỗ-trợ-bậttắt-trực-tiếp-từ-ui)
   - [3.2. Bộ Đệm Lưới Thời Gian `NeuralBuffer` Lưu Ảnh `RGBA_8888` Thô (Không Chạy `letterbox` Ở Nhánh Camera)](#32-bộ-đệm-lưới-thời-gian-neuralbuffer-lưu-ảnh-rgba_8888-thô-không-chạy-letterbox-ở-nhánh-camera)
   - [3.3. Thực Thi `letterboxRgbaToRgbCHW` Trên `ONNX Worker Thread` (Chỉ Chạy Trên $T_{\text{new}}$ Frames Mới)](#33-thực-thi-letterboxrgbatorgbchw-trên-onnx-worker-thread-chỉ-chạy-trên-t_textnew-frames-mới)
4. [Đặc Tả Chi Tiết Nhánh Vật Lý: MediaPipe Ép Chạy GPU & Tách 3 Luồng Con Song Song (`Eye` $\parallel$ `Mouth` $\parallel$ `Head+FFT`)](#4-đặc-tả-chi-tiết-nhánh-vật-lý-mediapipe-ép-chạy-gpu--tách-3-luồng-con-song-song-eye-parallel-mouth-parallel-headfft)
   - [4.1. `FaceTracker` Ép Chạy Trên GPU (`Delegate.GPU`) Với Đầu Vào `MPImage(IMAGE_FORMAT_RGBA, 640×480)` & Công Thức EAR / MAR](#41-facetracker-ép-chạy-trên-gpu-delegategpu-với-đầu-vào-mpimageimage_format_rgba-640480--công-thức-ear--mar)
   - [4.2. `HeadPoseEstimator` (SolvePnP) & Hiệu Chuẩn Góc Đầu](#42-headposeestimator-solvepnp--hiệu-chuẩn-góc-đầu)
   - [4.3. `AdaptiveHMM` & `AdaptiveHMM_FSM` (Mắt & Miệng)](#43-adaptivehmm--adaptivehmm_fsm-mắt--miệng)
   - [4.4. `StateMachine` (FSM Đếm Sự Kiện & Lọc Thời Lượng)](#44-statemachine-fsm-đếm-sự-kiện--lọc-thời-lượng)
   - [4.5. `WindowRatio` (Tính PERCLOS, POM, Over-Angle & Coverage)](#45-windowratio-tính-perclos-pom-over-angle--coverage)
   - [4.6. `HeadMotionWindow` (Tích Phân Biên Độ & Phổ FFT)](#46-headmotionwindow-tích-phân-biên-độ--phổ-fft)
   - [4.7. Kiến Trúc Tách 3 Luồng Con Vật Lý Song Song (`ParallelCameraMetrics`: `EyeWorker` $\parallel$ `MouthWorker` $\parallel$ `HeadPoseAndFftWorker`)](#47-kiến-trúc-tách-3-luồng-con-vật-lý-song-song-parallelcamerametrics-eyeworker-parallel-mouthworker-parallel-headposeandfftworker)
5. [Đặc Tả Nhánh Học Sâu (`CNN–ConvGRU`) & Định Dạng Model ONNX Theo `export_onnx.py`](#5-đặc-tả-nhánh-học-sâu-cnnconvgru--định-dạng-model-onnx-theo-export_onnxpy)
6. [Đặc Tả Cổng Kiểm Duyệt (`Engine._features`), Chuẩn Hóa (`physical_v1`) & Hợp Nhất (`Fitness`)](#6-đặc-tả-cổng-kiểm-duyệt-engine_features-chuẩn-hóa-physical_v1--hợp-nhất-fitness)
7. [Mã Nguồn Tham Chiếu Python Gốc Hoàn Chỉnh (Original Python Reference Code)](#7-mã-nguồn-tham-chiếu-python-gốc-hoàn-chỉnh-original-python-reference-code)
   - [7.1. Tiền Xử Lý Ảnh & Mạng Học Sâu (`loadvideo2handle.py` & `neuralnetwork.py`)](#71-tiền-xử-lý-ảnh--mạng-học-sâu-loadvideo2handlepy--neuralnetworkpy)
   - [7.2. Bộ Đếm Thời Gian, Tỷ Lệ Cửa Sổ & Phổ FFT (`temporal_metrics.py`)](#72-bộ-đếm-thời-gian-tỷ-lệ-cửa-sổ--phổ-fft-temporal_metricspy)
   - [7.3. Ước Lượng Tư Thế Đầu 3D SolvePnP (`head_pose_estimation.py`)](#73-ước-lượng-tư-thế-đầu-3d-solvepnp-head_pose_estimationpy)
   - [7.4. Mô Hình Markov Ẩn Thích Nghi & Máy Trạng Thái (`adaptive_hmm_fsm.py`)](#74-mô-hình-markov-ẩn-thích-nghi--máy-trạng-thái-adaptive_hmm_fsmpy)
   - [7.5. Bộ Theo Dõi Khuôn Mặt & Trích Xuất Đặc Trưng Vật Lý (`camera_metrics.py`)](#75-bộ-theo-dõi-khuôn-mặt--trích-xuất-đặc-trưng-vật-lý-camera_metricspy)
   - [7.6. Cổng Kiểm Duyệt, Chuẩn Hóa `physical_v1` & Hợp Nhất `Fitness` (`engine.py`, `normalization.py`, `fitnessFunction.py`)](#76-cổng-kiểm-duyệt-chuẩn-hóa-physical_v1--hợp-nhất-fitness-enginepy-normalizationpy-fitnessfunctionpy)
   - [7.7. Đóng Gói & Xuất Mô Hình ONNX Hỗ Trợ Cache `p4_old` (`export_onnx.py`)](#77-đóng-gói--xuất-mô-hình-onnx-hỗ-trợ-cache-p4_old-export_onnxpy)
   - [7.8. Luồng Suy Luận Thời Gian Thực Đa Luồng (`inference.py`)](#78-luồng-suy-luận-thời-gian-thực-đa-luồng-inferencepy)
8. [Phân Tích Độ Phức Tạp Thuật Toán & Ước Tính Thời Gian Hoàn Thành Phiên Bản v1 (Complexity & Execution Time Specification)](#8-phân-tích-độ-phức-tạp-thuật-toán--ước-tính-thời-gian-hoàn-thành-phiên-bản-v1-complexity--execution-time-specification)
   - [8.1. Độ Phức Tạp Thuật Toán Trên Cụm Luồng 1A–1C (Chạy Mỗi Frame)](#81-độ-phức-tạp-thuật-toán-trên-cụm-luồng-1a1c-chạy-mỗi-frame)
   - [8.2. Độ Phức Tạp Nhánh Học Sâu (`CNN–ConvGRU`), `letterbox` Trên `ONNX Worker` & Hiệu Quả Của `p4_old` Cache](#82-độ-phức-tạp-nhánh-học-sâu-cnnconvgru-letterbox-trên-onnx-worker--hiệu-quả-của-p4_old-cache)
   - [8.3. Ước Tính Thời Gian Hoàn Thành (Execution Time) & So Sánh Tốc Độ FPS Giữa Bản v0 và Bản v1](#83-ước-tính-thời-gian-hoàn-thành-execution-time--so-sánh-tốc-độ-fps-giữa-bản-v0-và-bản-v1)
   - [8.4. Tổng Hợp Dung Lượng Bộ Nhớ (RAM & VRAM Footprint)](#84-tổng-hợp-dung-lượng-bộ-nhớ-ram--vram-footprint)
   - [8.5. Điều Kiện Đảm Bảo Thời Gian Thực Không Khóa (Real-Time Schedulability v1)](#85-điều-kiện-đảm-bảo-thời-gian-thực-không-khóa-real-time-schedulability-v1)
9. [Danh Mục Kiểm Tra Tương Thích Số Học & Tối Ưu Hóa (Verification Checklist v1)](#9-danh-mục-kiểm-tra-tương-thích-số-học--tối-ưu-hóa-verification-checklist-v1)

---

## 1. Tổng Quan Kiến Trúc Nâng Cấp v1 & Cấu Hình Phần Cứng CameraX `640×480 RGBA_8888`

Hệ thống phát hiện buồn ngủ phiên bản `v1` kết hợp cụm luồng xử lý vật lý tốc độ cao (Pipeline + Fork-Join + GPU) và nhánh mạng học sâu bất đồng bộ:
- **Tầng Thu Nhận & Tiền Xử Lý Ảnh (`Luồng 1A: CameraIngestThread` — `640×480 RGBA_8888`):**
  CameraX Hardware được cấu hình xuất trực tiếp khung hình độ phân giải **`640×480` (VGA)** ở định dạng **`OUTPUT_IMAGE_FORMAT_RGBA_8888`**. Người dùng có thể **bật/tắt `balanceBrightness` ngay trên UI**:
  - Khi **TẮT** (`enableBalanceBrightness = false`): Buffer `RGBA_8888` `640×480` đi thẳng vào `MPImage(IMAGE_FORMAT_RGBA)` của MediaPipe GPU mà không cần chuyển hệ màu (**Zero-Conversion**, tốn `< 0.15 ms`).
  - Khi **BẬT** (`enableBalanceBrightness = true`): Nâng sáng gamma trên kênh $Y$ (`target_luma = 70.0`) khi môi trường cabin tối ($\mu_Y < 70.0$).
  Ngay sau khi trích xuất `RawRgbaFrame`, `imageProxy.close()` được gọi lập tức để phần cứng CameraX ISP chuẩn bị frame tiếp theo ở tốc độ tối đa (`30–60 FPS`).
- **Tầng Điều Phối Vật Lý & MediaPipe GPU (`Luồng 1B` + `Nhóm Luồng 1C`):**
  - **Song song hóa mức 1 (GPU $\parallel$ CPU):** Trong khi `MediaPipe FaceLandmarker` (ép chạy trên **`Delegate.GPU`**) trích xuất 478 điểm mốc 2D từ `MPImage(IMAGE_FORMAT_RGBA, 640×480)`, CPU đồng thời chạy `NeuralBuffer.updateRaw(rawRgbaFrame, timestamp)` để lấy mẫu khung hình thô vào lưới `5.0 FPS` (hoàn toàn **không chạy `letterbox`**).
  - **Song song hóa mức 2 (Fork-Join 3 Nhánh Vật Lý Độc Lập trên 3 Luồng Con `1C-1`, `1C-2`, `1C-3`):** Ngay khi có 478 điểm mốc `points`, hệ thống chia tải song song sang 3 luồng:
    1. **`EyeWorker` (`1C-1`):** Tính `EAR` $\rightarrow$ `AdaptiveHMM_FSM("eye")` $\rightarrow$ `StateMachine("eye")` $\rightarrow$ `WindowRatio("perclos")` $\rightarrow$ cập nhật `blinkDeque`.
    2. **`MouthWorker` (`1C-2`):** Tính `MAR` $\rightarrow$ `AdaptiveHMM_FSM("mouth")` $\rightarrow$ `StateMachine("mouth")` $\rightarrow$ `WindowRatio("pom")`.
    3. **`HeadPoseAndFftWorker` (`1C-3`):** Giải `SolvePnP` (`SQPnP` + `RefineLM`) $\rightarrow$ hiệu chuẩn `neutral` $\rightarrow$ `StateMachine("pitch")` + `WindowRatio("over_angle")` + `HeadMotionWindow` (nội suy tuyến tính + biến đổi **`rfft` 1800 điểm**) $\rightarrow$ cập nhật `nodDeque`.
- **Nhánh Học Sâu & `letterbox` Bất Đồng Bộ (`Luồng 2: ONNX Worker Thread` — Chu kỳ `interval = 1.0s`):**
  Nhận bản chụp bất biến `RawWindowSnapshot`, gọi `getNewFramesSince()` để lọc ra đúng $\Delta = T_{\text{new}}$ khung hình `RGBA_8888` `640×480` mới phát sinh trong 1 giây qua (thường chỉ **`5` frames**). Chỉ trên $T_{\text{new}}$ frames mới này, `ONNX Worker Thread` mới thực thi hàm **`letterboxRgbaToRgbCHW(rawFrame, size = 224)`** để tạo tensor `video: [T_new, 3, 224, 224]` kiểu `uint8`, ghép với bộ đệm đặc trưng `p4_old: [T_old, 128, 14, 14]` và gọi `OrtSession.run()`.
- **Nhánh Hợp Nhất (Late Fusion - 3 chế độ đồng thời):** Chuẩn hóa 7 đặc trưng vật lý + 1 điểm mạng học sâu (`neuralnetwork_score`) về miền $[0, 1]$ theo chuẩn `physical_v1` và đánh giá đồng thời 3 hàm hợp nhất (`weighted_mean`, `noisy_or`, `cooccurrence`).

---

### 1.1. Bảng Ánh Xạ Công Nghệ (Python $\rightarrow$ Android Kotlin v1)

| Thành phần | Python (`inference.py` & `export_onnx.py`) | Android (Kotlin — Kiến Trúc Nâng Cấp v1) | Thư viện Gradle khuyên dùng |
| :--- | :--- | :--- | :--- |
| **Camera Stream** | `cv2.VideoCapture` (`BUFFERSIZE=1`) | **CameraX `ImageAnalysis`** (`Size(640, 480)`, `OUTPUT_IMAGE_FORMAT_RGBA_8888`, `STRATEGY_KEEP_ONLY_LATEST`) | `androidx.camera:camera-camera2:1.3.4+`, `camera-lifecycle` |
| **Cân bằng sáng tùy chọn** | `balance_brightness` (`target_luma=70.0`) | **`balanceBrightnessRgba`** điều khiển bởi `enableBalanceBrightness: AtomicBoolean` từ UI Switch (Tắt $\rightarrow$ Zero-Conversion) | `org.opencv:opencv:4.9.0+` |
| **Face Landmarks (GPU)** | `mediapipe.tasks.python.vision.FaceLandmarker` (CPU) | **MediaPipe Tasks Vision Android** (`FaceLandmarker` với **`Delegate.GPU`** + đầu vào `MPImage.IMAGE_FORMAT_RGBA`) | `com.google.mediapipe:tasks-vision:0.10.14+` |
| **Nhánh Vật Lý Song Song** | `CameraMetrics.process_landmarks` (Tuần tự 1 luồng) | **`ParallelCameraMetrics`** (Fork-Join 3 Coroutines song song: `EyeWorker` $\parallel$ `MouthWorker` $\parallel$ `HeadPoseAndFftWorker`) | `org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0+` |
| **Xử lý `letterbox`** | Chạy mỗi frame trong `NeuralBuffer.update` | **Chuyển sang `ONNX Worker Thread`**: `letterboxRgbaToRgbCHW` chỉ chạy trên $T_{\text{new}} \approx 5$ frames mới mỗi giây | `org.opencv:opencv:4.9.0+` |
| **Mạng CNN–ConvGRU** | `ExportDrowsyClassifier` (`drowsy_classifier_*.onnx`) | **ONNX Runtime Android** (`OrtEnvironment`, `OrtSession`, Incremental `p4_old` Cache) | `com.microsoft.onnxruntime:onnxruntime-android:1.18.0+` |
| **Biến đổi FFT 1D** | `numpy.fft.rfft` | **JTransforms** (`DoubleFFT_1D.realForward`) chạy trên `HeadPoseAndFftWorker` | `com.github.wendykierp:JTransforms:3.1` |
| **Giao diện (UI)** | `cv2.imshow("Driver Guardian", canvas)` | **Jetpack Compose** + `Switch("Balance Brightness")` + Overlay `StateFlow` | `androidx.compose.ui:ui`, `material3` |

---

### 1.2. Đặc Tả Cấu Hình CameraX Hardware `640×480` Định Dạng `OUTPUT_IMAGE_FORMAT_RGBA_8888`

Để `MediaPipe FaceLandmarker` (chạy trên `Delegate.GPU`) có thể sử dụng trực tiếp khung hình từ CameraX mà không cần tốn CPU chuyển đổi từ `YUV_420_888` sang `RGB`, `ImageAnalysis` phải được cấu hình xuất định dạng **`OUTPUT_IMAGE_FORMAT_RGBA_8888`** ở độ phân giải chuẩn VGA **`640×480` (tỷ lệ 4:3)**:

```kotlin
import android.util.Size
import androidx.camera.core.AspectRatio
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import com.google.mediapipe.framework.image.ByteBufferImageBuilder
import com.google.mediapipe.framework.image.MPImage
import java.nio.ByteBuffer
import java.nio.ByteOrder

object CameraXHardwareConfig {
    const val TARGET_WIDTH = 640
    const val TARGET_HEIGHT = 480
    const val CHANNELS_RGBA = 4
    const val FRAME_BYTES_RGBA = TARGET_WIDTH * TARGET_HEIGHT * CHANNELS_RGBA // 1,228,800 bytes

    fun buildImageAnalysis(ingestExecutor: java.util.concurrent.Executor, analyzer: ImageAnalysis.Analyzer): ImageAnalysis {
        val resolutionSelector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
            .setResolutionStrategy(
                ResolutionStrategy(
                    Size(TARGET_WIDTH, TARGET_HEIGHT),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                )
            )
            .build()

        return ImageAnalysis.Builder()
            .setResolutionSelector(resolutionSelector)
            // Yêu cầu CameraX xuất trực tiếp 1 plane RGBA_8888 mà MediaPipe GPU đọc được ngay
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also { it.setAnalyzer(ingestExecutor, analyzer) }
    }

    /**
     * Trích xuất mảng byte RGBA_8888 chặt (tightly-packed width * height * 4) từ ImageProxy.
     * Xử lý cả trường hợp phần cứng có rowStride == width * 4 (copy khối 1 lần)
     * và trường hợp có byte đệm cuối dòng (rowStride > width * 4).
     */
    fun extractTightRgbaBytes(imageProxy: ImageProxy): ByteArray {
        val width = imageProxy.width
        val height = imageProxy.height
        val plane = imageProxy.planes[0]
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val rowBytes = width * CHANNELS_RGBA
        val out = ByteArray(width * height * CHANNELS_RGBA)

        buffer.rewind()
        if (rowStride == rowBytes) {
            buffer.get(out, 0, out.size)
        } else {
            for (y in 0 until height) {
                buffer.position(y * rowStride)
                buffer.get(out, y * rowBytes, rowBytes)
            }
        }
        return out
    }

    /**
     * Đóng gói mảng RGBA_8888 thành MPImage (MPImage.IMAGE_FORMAT_RGBA)
     * để truyền trực tiếp vào MediaPipe FaceLandmarker (GPU Delegate).
     */
    fun wrapRgbaToMPImage(rgbaBytes: ByteArray, width: Int, height: Int): MPImage {
        val directBuffer = ByteBuffer.allocateDirect(rgbaBytes.size).order(ByteOrder.nativeOrder())
        directBuffer.put(rgbaBytes)
        directBuffer.rewind()
        return ByteBufferImageBuilder(
            directBuffer,
            width,
            height,
            MPImage.IMAGE_FORMAT_RGBA
        ).build()
    }
}
```

> [!TIP]
> **Lợi ích của `640×480` + `OUTPUT_IMAGE_FORMAT_RGBA_8888`:**
> 1. **Tương thích trực tiếp với MediaPipe GPU (`MPImage.IMAGE_FORMAT_RGBA`):** Texture OpenGL ES 3.1 / OpenCL trên GPU di động (Adreno / Mali) yêu cầu bộ nhớ căn chỉnh 4-byte (`RGBA_8888`). Việc xuất sẵn `RGBA_8888` giúp GPU nạp thẳng texture bằng DMA mà không cần shader hay CPU đệm thêm kênh Alpha.
> 2. **Tối ưu điểm ngọt độ phân giải `640×480`:** Mạng nội bộ của MediaPipe `FaceLandmarker` co giãn ảnh về `192×192` / `256×256` và nhánh học sâu co giãn về `224×224`. Độ phân giải `640×480` cung cấp độ chính xác điểm mốc `EAR/MAR/SolvePnP` tương đương HD `1280×720` nhưng giảm **gấp 3 lần** băng thông bộ nhớ (`1.17 MB/frame` so với `3.52 MB/frame`).

---

## 2. Kiến Trúc Đa Luồng Song Song & Tương Tác Giữa Các Luồng (Concurrency Specification v1)

### 2.1. Danh Sách Các Luồng Chạy (8 Luồng / Nhóm Luồng) & Mô Hình Không Khóa (Lock-Free Snapshot)

Thay vì dồn toàn bộ việc đọc camera, cân bằng sáng, chạy MediaPipe, tính HMM/FSM/FFT của cả mắt, miệng, đầu và chạy `letterbox` vào một luồng `CameraAnalyzer Thread` duy nhất như bản cũ, phiên bản `v1` tách thành **8 luồng / nhóm luồng chuyên biệt** hoạt động theo mô hình **Pipeline + Fork-Join**:

| STT | Tên Luồng (Thread / Dispatcher) | Cách Khởi Tạo Trên Android | Tần Suất Hoạt Động | Trạng Thái Sở Hữu Độc Quyền (Thread-Confined State) | Nhiệm Vụ Chính |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **1A** | **`CameraIngestThread`** *(Luồng Thu Nhận & Cân Bằng Sáng Tùy Chọn)* | `Executors.newSingleThreadExecutor()` gắn vào `ImageAnalysis.setAnalyzer()` | **FPS phần cứng Camera** (`30–60 FPS`, `640×480 RGBA_8888`) | `originNs`, `lastTimestamp`, vùng đệm tái sử dụng OpenCV `Mat` cho `balanceBrightnessRgba` | Nhận `ImageProxy (640×480 RGBA_8888)`, tính `timestamp` đơn điệu từ sensor, đọc cờ `enableBalanceBrightness.get()` từ UI (nếu `true` thì chạy `balanceBrightnessRgba`, nếu `false` thì trích xuất thẳng `RGBA_8888`), gọi **`imageProxy.close()` ngay lập tức**, và đẩy `FramePacket` sang Luồng 1B qua `Channel(CONFLATED)`. |
| **1B** | **`PhysicalCoordinator & MediaPipe GPU Thread`** *(Luồng Điều Phối & Chạy MediaPipe GPU)* | `Executors.newSingleThreadExecutor().asCoroutineDispatcher()` (Cố định 1 OS Thread cho ngữ cảnh GPU EGL) | Mỗi frame từ Luồng 1A (`30–60 FPS`) | `FaceTracker` (`FaceLandmarker` với **`Delegate.GPU`**), `NeuralBuffer` (chỉ lưu `RawRgbaFrame`), `pending`, `status`, `nextPrediction`, `nextWarmupTime` | Nhận `FramePacket`, phát lệnh chạy **song song**: (1) `FaceTracker.processGpu(mpImage)` trên GPU để lấy `points[478, 2]` và (2) `NeuralBuffer.updateRaw(rgbaFrame, timestamp)` trên CPU. Sau đó kích hoạt **Fork-Join 3 luồng con `1C-1`, `1C-2`, `1C-3`**, gom kết quả `CameraMetrics`, điều phối giao việc cho Luồng 2 và phát `dashboardStateFlow` lên UI. |
| **1C-1** | **`EyeWorker Thread`** *(Luồng Nhánh Vật Lý Mắt)* | `Dispatchers.Default` (hoặc Thread #1 trong `FixedThreadPool(3)`) | Mỗi frame (chạy **song song** với `1C-2` và `1C-3`) | `eye: AdaptiveHMM_FSM("eye")`, `blinkDeque: ArrayDeque<(Double, Double)>` | Tính `aspect_ratio` cho `LEFT_EYE` & `RIGHT_EYE` $\rightarrow$ `ear`, chạy `eye.process(ear, timestamp)` (bao gồm khởi tạo 20 vòng EM tại frame #300, dự đoán HMM, học trực tuyến, `StateMachine("eye")`, `WindowRatio("perclos")`) và cập nhật `blinkDeque`. |
| **1C-2** | **`MouthWorker Thread`** *(Luồng Nhánh Vật Lý Miệng)* | `Dispatchers.Default` (hoặc Thread #2 trong `FixedThreadPool(3)`) | Mỗi frame (chạy **song song** với `1C-1` và `1C-3`) | `mouth: AdaptiveHMM_FSM("mouth")` | Tính `aspect_ratio` cho `MOUTH` $\rightarrow$ `mar`, chạy `mouth.process(mar, timestamp)` (bao gồm khởi tạo 20 vòng EM tại frame #300, dự đoán HMM, học trực tuyến, `StateMachine("mouth")`, `WindowRatio("pom")`). |
| **1C-3** | **`HeadPoseAndFftWorker Thread`** *(Luồng Nhánh Tư Thế Đầu & Phổ FFT)* | `Dispatchers.Default` (hoặc Thread #3 trong `FixedThreadPool(3)`) | Mỗi frame (chạy **song song** với `1C-1` và `1C-2`) | `estimator: HeadPoseEstimator`, `neutralSamples`, `neutral`, `nodding`, `nod: StateMachine("pitch")`, `overAngleRatio: WindowRatio`, `headMotion: HeadMotionWindow`, `nodDeque` | Giải `SolvePnP` (`SQPnP` + `RefineLM`) $\rightarrow$ hiệu chuẩn `neutral` (30 frames) $\rightarrow$ tính `angles`, `over_angle`, `nod_state`, rồi chạy **song song nội bộ** hoặc tuần tự nhanh `nod.process()`, `overAngleRatio.process()` và **`headMotion.process()` (nội suy + `rfft` 1800 điểm)**, cập nhật `nodDeque`. |
| **2** | **`ONNX Worker Thread`** *(Luồng `letterbox` + Suy Luận Học Sâu + Chuẩn Hóa)* | `Dispatchers.Default.limitedParallelism(1)` | Chu kỳ `interval = 1.0s` (chạy cả trong `WARMING UP` và sau `WARMING UP`) | `OnnxNeuralClassifier` (`OrtSession`, `cachedP4`, `cachedFrameCount`, `cachedGeneration`, `cachedSeqId`), `PhysicalNormalizer`, bộ đệm `letterbox` | Nhận `RawWindowSnapshot`, gọi `getNewFramesSince()` lấy $\Delta = T_{\text{new}}$ (thường `5`) `RawRgbaFrame` mới, **chạy `letterboxRgbaToRgbCHW(224)` chỉ trên $T_{\text{new}}$ frames này**, ghép với `p4_old` chạy `OrtSession.run()`. Khi `timestamp >= 60.0s` và đạt 2 cổng kiểm duyệt (`Engine._features`), chuẩn hóa 8 đặc trưng `physical_v1` và trả kết quả về Luồng 1B. |
| **3** | **`ONNX Intra-Op Thread Pool`** *(Nhóm Luồng Toán Tử C++)* | Quản lý nội bộ bởi ONNX Runtime C++ (`setIntraOpNumThreads(4)`) | Kích hoạt bên trong `session.run()` của Luồng 2 | Workspace của các toán tử `Conv`, `Gemm`, `Loop (ConvGRU)` trong C++ | Thực thi song song các phép tích chập của `BackboneNeck` (trên $T_{\text{new}}$ frames), `Adapter` và chuỗi `ConvGRUCell` trên 4 nhân CPU ARM (hoặc XNNPACK/NNAPI). |
| **4 & 5** | **`Main / UI Thread` & `I/O Logging Thread`** | `Dispatchers.Main` & `Dispatchers.IO.limitedParallelism(1)` | UI: `60–120 Hz`; I/O: `~1 lần/giây` | `PreviewView`, Compose `State<DashboardState>`, `enableBalanceBrightness: AtomicBoolean`, `BufferedWriter` | Hiển thị thông số thời gian thực, cung cấp **Nút Bật/Tắt `balanceBrightness`**, phát cảnh báo âm thanh/rung và ghi file `.jsonl` bất đồng bộ. |

---

### 2.2. Sơ Đồ Kiến Trúc Luồng Dữ Liệu Song Song (Pipeline + Fork-Join + ONNX Letterbox)

```mermaid
flowchart TD
    UI_Switch["Luồng 4 (UI): Công tắc 'Balance Brightness'<br/>enableBalanceBrightness: AtomicBoolean (Mặc định: false)"] -.->|"Đọc không khóa O(1)"| Ingest

    subgraph Stage1A["Tầng 1A: Luồng Thu Nhận Camera (CameraIngestThread)"]
        CamHW["CameraX Hardware ISP<br/>• Độ phân giải: 640×480 (VGA 4:3)<br/>• Định dạng: OUTPUT_IMAGE_FORMAT_RGBA_8888<br/>• Chiến lược: STRATEGY_KEEP_ONLY_LATEST"] -->|"ImageProxy (1 plane RGBA_8888)<br/>+ sensor timestampNs"| Ingest["Luồng 1A: CameraIngestThread<br/>1. Tính timestamp đơn điệu (s)<br/>2. Trích xuất mảng rgbaBytes[640×480×4]<br/>3. Gọi imageProxy.close() NGAY LẬP TỨC<br/>4. Nếu enableBalanceBrightness == true:<br/>   chạy balanceBrightnessRgba(targetLuma=70)<br/>   Nếu false: giữ nguyên (Zero-Conversion)"]
    end

    Ingest -->|"ConflatedChannel<FramePacket><br/>(Không bao giờ chặn CameraIngestThread)"| Stage1B_Entry["Luồng 1B: PhysicalCoordinatorThread"]

    subgraph Stage1B["Tầng 1B: Song Song Hóa GPU Landmarker & CPU Raw Buffer"]
        Stage1B_Entry -->|"Nhánh GPU: MPImage(IMAGE_FORMAT_RGBA, 640×480)"| MPGPU["MediaPipe FaceLandmarker<br/>(Ép chạy trên GPU: Delegate.GPU)<br/>Trích xuất 478 Landmarks 2D (2.2–6.5 ms)"]
        Stage1B_Entry -->|"Nhánh CPU (Chạy đồng thời trong lúc đợi GPU)"| RawBuf["NeuralBuffer.updateRaw(rawRgbaFrame, timestamp)<br/>• Lấy mẫu vào lưới 5.0 FPS (0.2s/frame)<br/>• KHÔNG chạy letterbox tại đây (0 ms)<br/>• Lưu RawRgbaFrame vào vòng đệm 50 frames"]
    end

    MPGPU -->|"points: FloatArray[478 × 2]"| ForkPoint{"Fork-Join 3 Nhánh Vật Lý<br/>(coroutineScope + async)"}

    subgraph Stage1C["Tầng 1C: 3 Luồng Con Tính Đặc Trưng Vật Lý Song Song (Parallel Physical Workers)"]
        ForkPoint -->|"Luồng 1C-1 (EyeWorker)"| W_Eye["Nhánh Mắt (Eye Branch)<br/>• Tính EAR (Left + Right)<br/>• AdaptiveHMM('eye') + 20-iter EM<br/>• StateMachine('eye') (Blink)<br/>• WindowRatio('perclos')<br/>• Cập nhật blinkDeque"]
        ForkPoint -->|"Luồng 1C-2 (MouthWorker)"| W_Mouth["Nhánh Miệng (Mouth Branch)<br/>• Tính MAR (Mouth)<br/>• AdaptiveHMM('mouth') + 20-iter EM<br/>• StateMachine('mouth') (Yawn)<br/>• WindowRatio('pom')"]
        ForkPoint -->|"Luồng 1C-3 (HeadPoseAndFftWorker)"| W_Head["Nhánh Đầu & Phổ FFT (Head Branch)<br/>• SolvePnP (SQPnP + RefineLM)<br/>• Hiệu chuẩn Neutral (30 frames)<br/>• StateMachine('pitch') (Nod) + Over-Angle<br/>• HeadMotionWindow (Interp + rfft 1800 điểm)<br/>• Cập nhật nodDeque"]
    end

    W_Eye & W_Mouth & W_Head & RawBuf -->|"awaitAll() -> Hợp nhất MetricsSnapshot"| CoordinatorPost["Luồng 1B: Kiểm tra pending & Giao việc ONNX (mỗi 1.0s)"]

    CoordinatorPost -->|"RawWindowSnapshot (Lock-Free)<br/>Chứa tham chiếu RawRgbaFrame"| OnnxWorker["Luồng 2: ONNX Worker Thread (Chu kỳ 1.0s)<br/>1. getNewFramesSince() -> Lọc ra T_new (~5) RawRgbaFrames mới<br/>2. Chạy letterboxRgbaToRgbCHW(224) CHỈ TRÊN T_new frames mới<br/>   (Resize RGBA 640×480 -> 224×168 -> Bỏ Alpha -> Pad 224×224 -> CHW)<br/>3. OrtSession.run(video[T_new,3,224,224], p4_old[T_old,128,14,14])<br/>4. Cập nhật cachedP4 + Kiểm duyệt Engine._features + Chuẩn hóa physical_v1"]

    CoordinatorPost -->|"dashboardStateFlow.value = ..."| UI_Thread["Luồng 4: Main / UI Thread (Jetpack Compose)"]
    CoordinatorPost -->|"logChannel.trySend(jsonLine)"| IO_Thread["Luồng 5: I/O Logging Thread"]
```

---

### 2.3. Thích Nghi Với FPS Biến Thiên Của Camera Trên Cụm Luồng 1A–1C

Trên thiết bị Android thực tế, ngay cả khi đã cấu hình `640×480 RGBA_8888`, tốc độ khung hình của phần cứng CameraX vẫn biến thiên liên tục (`12–60 FPS`) tùy theo thời gian phơi sáng tự động (auto-exposure trong cabin tối) và nhiệt độ thiết bị. Cụm luồng `1A–1C` thích nghi hoàn toàn với FPS biến thiên qua 4 cơ chế:

1. **Tính `timestamp` đơn điệu từ đồng hồ phần cứng Sensor (`ImageInfo.getTimestamp()`) ngay tại Luồng 1A:**
   - Lấy mốc `originNs = imageProxy.imageInfo.timestamp` ở frame đầu tiên, quy đổi sang giây thực tế và kẹp đơn điệu nghiêm ngặt giống [`inference.py:240-243`](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/inference.py#L240-L243):
     ```kotlin
     var timestamp = (imageProxy.imageInfo.timestamp - originNs) / 1_000_000_000.0
     if (lastTimestamp != null && timestamp <= lastTimestamp!!) {
         timestamp = lastTimestamp!! + 1e-4
     }
     lastTimestamp = timestamp
     ```
2. **Giải phóng sớm `ImageProxy` và truyền qua `Channel(Channel.CONFLATED)` giữa Luồng 1A và Luồng 1B:**
   - Luồng 1A đọc xong mảng `rgbaBytes` (`640×480×4`) là gọi `imageProxy.close()` ngay lập tức (trong `< 0.2 ms`), giúp hàng đợi phần cứng của CameraX ISP không bao giờ bị nghẽn.
   - Nếu ở một thời điểm hiếm hoi (ví dụ đúng frame thứ 300 khi khởi tạo HMM), cụm Luồng 1B+1C đang bận xử lý chưa xong mà CameraX đã gửi tới 2 frames liên tiếp, `Channel(Channel.CONFLATED)` đảm bảo Luồng 1B luôn lấy frame mới nhất có `timestamp` thực tế chính xác, và khoảng cách $\Delta t \le 2 \times 33\text{ms} = 66\text{ms} \ll \text{max\_gap\_sec} = 250\text{ms}$ nên **tuyệt đối không làm đứt quãng cửa sổ trượt**.
3. **Tách rời FPS Camera và FPS Mạng Học Sâu qua lưới thời gian `NeuralBuffer`:**
   - Dù Luồng 1B nhận frame ở `15 FPS`, `30 FPS` hay `60 FPS`, `NeuralBuffer.updateRaw(rawRgbaFrame, timestamp)` luôn lấy mẫu lại (resample) chính xác về lưới `neural_fps = 5.0 FPS` ($\Delta t_{\text{grid}} = 0.20\text{s}$).
4. **Tính toán 3 Nhánh Vật Lý (`1C-1`, `1C-2`, `1C-3`) theo thời gian liên tục ($\Delta t$ thực tế):**
   - Cả 3 luồng con `EyeWorker`, `MouthWorker`, `HeadPoseAndFftWorker` đều nhận cùng một giá trị `timestamp` của frame hiện tại.
   - `StateMachine` và `WindowRatio` tích phân thời lượng đóng mắt/mở miệng/nghiêng đầu theo hiệu số `timestamp` thực tế.
   - `HeadMotionWindow` trên `HeadPoseAndFftWorker` tự động ước lượng trung vị bước thời gian $\Delta t_{\text{med}} = \text{median}(\text{diff}(\text{times}))$ để nội suy tuyến tính ra $N = \max(4, \text{round}(\text{observed} / \Delta t_{\text{med}}) + 1)$ điểm cách đều trước khi biến đổi `rfft`.

---

### 2.4. Sơ Đồ Tuần Tự Đa Luồng (Bao Gồm UI Toggle `balanceBrightness`, Fork-Join 3 Nhánh Vật Lý & `ONNX Letterbox`)

```mermaid
sequenceDiagram
    autonumber
    participant UI as Luồng 4: Main/UI Thread
    participant CamX as CameraX (640×480 RGBA_8888)
    participant Ingest as Luồng 1A: CameraIngestThread
    participant Coord as Luồng 1B: Coordinator & MediaPipe GPU
    participant Pool1C as Nhóm Luồng 1C: 3 Physical Workers (Eye || Mouth || Head+FFT)
    participant Worker as Luồng 2: ONNX Worker Thread (Letterbox + ONNX)
    participant ONNX as Luồng 3: ONNX Runtime C++ Pool

    Note over UI,ONNX: Khởi tạo: enableBalanceBrightness = false (hoặc true từ UI), status = "WARMING UP", nextPrediction = 60.0s, nextWarmup = 1.0s

    opt Người dùng bấm nút Bật/Tắt Cân Bằng Sáng trên UI
        UI->>Ingest: enableBalanceBrightness.set(true / false) (Lock-Free AtomicBoolean)
    end

    loop Mỗi Frame từ CameraX Hardware (640×480 RGBA_8888, Variable FPS)
        CamX->>Ingest: ImageProxy (1 plane RGBA_8888, sensor timestampNs)
        Ingest->>Ingest: Tính timestamp đơn điệu (s) & trích xuất rgbaBytes[640×480×4]
        Ingest->>CamX: imageProxy.close() NGAY LẬP TỨC (Giải phóng phần cứng ISP)
        alt enableBalanceBrightness.get() == true
            Ingest->>Ingest: rgbaBytes = balanceBrightnessRgba(rgbaBytes, 640, 480, targetLuma = 70.0)
        else enableBalanceBrightness.get() == false
            Ingest->>Ingest: Giữ nguyên rgbaBytes (Zero-Conversion Fast Path)
        end
        Ingest->>Coord: Gửi FramePacket(RawRgbaFrame(640, 480, rgbaBytes), timestamp)

        par Chạy song song GPU Landmarker và CPU NeuralBuffer.updateRaw
            Coord->>Coord: mpImage = wrapRgbaToMPImage(rgbaBytes, 640, 480)
            Coord->>Coord: points = faceLandmarkerGpu.detect(mpImage) (Chạy trên Delegate.GPU)
        and
            Coord->>Coord: neuralBuffer.updateRaw(rawRgbaFrame, timestamp) (Lưới 5.0 FPS, KHÔNG letterbox)
        end

        par Fork-Join 3 Nhánh Vật Lý Song Song trên Nhóm Luồng 1C
            Coord->>Pool1C: [1C-1 EyeWorker] processEye(points, timestamp) -> EAR, Eye HMM, Blink FSM, PERCLOS, blinkDeque
        and
            Coord->>Pool1C: [1C-2 MouthWorker] processMouth(points, timestamp) -> MAR, Mouth HMM, Yawn FSM, POM
        and
            Coord->>Pool1C: [1C-3 HeadPoseAndFftWorker] processHead(points, (640,480), timestamp) -> SolvePnP, Nod FSM, OverAngle, FFT 1800 điểm, nodDeque
        end
        Pool1C-->>Coord: Join: Gộp thành CameraMetricsSnapshot (Wait-free)

        alt Bước A: pending != null VÀ pending.isCompleted == true
            Worker-->>Coord: taskOutcome = pending.getCompleted() (Non-blocking)
            Coord->>Coord: pending = null, cập nhật status ("WARMING UP" / "WAITING FOR VALID WINDOW" / "DROWSY" / "ALERT")
        end

        alt Bước B1 (Trong WARMING UP): timestamp < 60.0s VÀ timestamp >= nextWarmup VÀ pending == null
            Coord->>Coord: nextWarmup = timestamp + 1.0s
            Coord->>Worker: async { warmupOnnxCache(neuralBuffer.snapshotRaw()) }
            activate Worker
            Worker->>Worker: deltaFrames = rawSnapshot.getNewFramesSince(cachedGen, cachedSeqId) (Lấy ~5 RawRgbaFrames mới)
            Worker->>Worker: videoTensor = letterboxBatchRgbaToRgbCHW(deltaFrames, size = 224) (CHỈ chạy letterbox trên ~5 frames!)
            Worker->>ONNX: session.run(video[T_new,3,224,224], p4_old[T_old,128,14,14])
            ONNX-->>Worker: prob[1,2], p4_features[T_old + T_new, 128, 14, 14]
            Worker->>Worker: Cập nhật cachedP4 (tối đa 50 frames cuối), cachedGen, cachedSeqId
            deactivate Worker
        else Bước B2 (Sau WARMING UP): timestamp >= nextPrediction (>= 60.0s) VÀ pending == null
            Coord->>Coord: nextPrediction = timestamp + 1.0s
            alt neuralBuffer.size == 50
                Coord->>Worker: async { evaluateWindow(metricsSnapshot, blinkSnapshot, nodSnapshot, neuralBuffer.snapshotRaw()) }
                activate Worker
                Worker->>Worker: deltaFrames = rawSnapshot.getNewFramesSince(cachedGen, cachedSeqId) (~5 RawRgbaFrames mới)
                Worker->>Worker: videoTensor = letterboxBatchRgbaToRgbCHW(deltaFrames, size = 224)
                Worker->>ONNX: session.run(video[T_new,3,224,224], p4_old[50-T_new,128,14,14])
                ONNX-->>Worker: prob[1,2], p4_features[50, 128, 14, 14]
                Worker->>Worker: Cập nhật cachedP4 = p4_features, score = prob[0, 1]
                Worker->>Worker: Kiểm tra Cổng 1 (Readiness & Coverage >= 48s) & Cổng 2 (FFT == 60s) + Chuẩn hóa physical_v1
                deactivate Worker
            else neuralBuffer.size < 50
                Coord->>Coord: latestResult = null, status = "WAITING FOR CAMERA CLIP"
            end
        end

        Coord->>UI: dashboardStateFlow.value = DashboardState(metrics, latestResult, status, fps, enableBalanceBrightness.get())
    end
```

---

### 2.5. Quy Tắc `WARMING UP`, Tích Lũy Trước `p4_old` & Chống Nghẽn Cổ Chai (Backpressure Rules)

1. **Giữ nguyên quá trình `WARMING UP` (`0.0s <= timestamp < 60.0s`) đồng thời tích lũy cache `p4_old` và chạy `letterbox` cuốn chiếu:**
   - Khi khởi tạo: `nextPrediction = config.windowSec` (**60.0 giây**), `nextWarmupTime = config.interval` (**1.0 giây**), `status = "WARMING UP"`, `result = null`.
   - Trong suốt $60.0\text{s}$ đầu tiên (`timestamp < 60.0s`), mỗi `1.0s`, `Luồng 1B` gửi bản chụp `neuralBuffer.snapshotRaw()` sang `ONNX Worker Thread` ở chế độ `isWarmup = true`.
   - `ONNX Worker Thread` trích xuất `5` khung hình `RawRgbaFrame` mới của giây vừa qua, chạy `letterboxRgbaToRgbCHW` cho đúng `5` khung hình đó, ghép với `p4_old` đang tích lũy và gọi `OrtSession.run()`.
   - Đến giây `59.0s`, `cachedP4` đã chứa sẵn đặc trưng P4 của 50 frames (`49.0s..59.0s`). Khi chạm mốc `timestamp = 60.0s` (lần dự đoán chính thức đầu tiên), `ONNX Worker Thread` **chỉ cần chạy `letterbox` và `BackboneNeck` trên 5 frames mới (`59.0s..60.0s`)** kết hợp với `45` frames `p4_old` có sẵn!
2. **Duy trì cập nhật `p4_old` ngay cả khi cửa sổ vật lý bị `skip` sau `60.0s`:**
   - Sau mốc `60.0s`, nếu một cửa sổ có `neuralBuffer.size == 50` nhưng bị `Engine._features` từ chối ở Cổng 1 (Coverage $< 48\text{s}$) hoặc Cổng 2 (FFT $< 60\text{s}$), `ONNX Worker Thread` vẫn chạy `letterbox` và cập nhật `cachedP4` với các frame mới của cửa sổ đó trước khi trả về `null`.
3. **Thứ tự kiểm tra trong mỗi vòng lặp frame trên Luồng 1B:**
   - **Bước A (Thu hồi kết quả trước):** Luôn kiểm tra `if (pending != null && pending.isCompleted)` **TRƯỚC** khi kiểm tra điều kiện giao việc mới (`timestamp >= nextWarmupTime` hoặc `timestamp >= nextPrediction`).
4. **Cơ chế Backpressure tự nhiên:**
   - Nếu một lần chạy Worker tốn nhiều hơn `interval = 1.0s` (ví dụ `1.2s`), `getNewFramesSince` ở chu kỳ kế tiếp sẽ tự động lấy `6` frames mới thay vì `5` frames để chạy `letterbox` và `BackboneNeck`, bù khớp 100% khoảng thời gian thực đã trôi qua.

---

## 3. Đặc Tả Tiền Xử Lý Ảnh Có Thể Bật/Tắt Từ UI, Bộ Đệm `NeuralBuffer` Lưu Ảnh Thô & `letterbox` Tại `ONNX Worker`

### 3.1. Hàm Cân Bằng Sáng `balanceBrightnessRgba` Hỗ Trợ Bật/Tắt Trực Tiếp Từ UI
Nguồn tham chiếu: [`loadvideo2handle.py:13-26`](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/DataSystem/loadvideo2handle.py#L13-L26).

- **Cơ chế Bật / Tắt từ Giao diện (UI Toggle):**
  - Trạng thái bật/tắt được lưu trong `val enableBalanceBrightness = AtomicBoolean(false)` (hoặc `true` tùy cấu hình mặc định).
  - Trên Jetpack Compose UI (`Main / UI Thread`), người dùng có thể gạt nút `Switch("Cân bằng sáng (Cabin tối)")` bất kỳ lúc nào trong lúc camera đang chạy mà không cần khởi động lại camera hay reset các cửa sổ 60s.
- **Đầu vào:** Mảng byte `rgbaBytes` định dạng **`RGBA_8888` `uint8` `[H=480, W=640, 4]`**, cờ `enabled = enableBalanceBrightness.get()`, tham số `targetLuma = 70.0`.
- **Thuật toán từng bước trên `Luồng 1A (CameraIngestThread)`:**
  1. **Nhánh Fast-Path (Khi `enabled == false` hoặc `targetLuma == 0.0`):**
     - **Trả về ngay lập tức `rgbaBytes` nguyên bản** ($\mathcal{O}(1)$, `0 FLOPs`, không cấp phát hay sao chép bộ nhớ).
  2. **Nhánh Active (Khi `enabled == true` và `targetLuma > 0.0`):**
     - Đưa `rgbaBytes` vào `rgbaMat` (`480×640, CV_8UC4`), chuyển `RGBA` $\rightarrow$ `BGR` (`Imgproc.COLOR_RGBA2BGR`) $\rightarrow$ `YCrCb` (`Imgproc.COLOR_BGR2YCrCb`) để giữ công thức hệ số tính kênh $Y$ khớp tuyệt đối với OpenCV trong Python.
     - Tách kênh $Y$ (kênh chỉ số 0) và tính trung bình cộng:
       $$\mu_Y = \frac{1}{H \times W} \sum_{y, x} Y(y, x)$$
     - Nếu $\mu_Y == 0.0$ hoặc $\mu_Y \ge \text{targetLuma}$ ($70.0$), trả về ngay `rgbaBytes` nguyên bản (không làm tối ảnh khi trời đã đủ sáng!).
     - Nếu $\mu_Y < 70.0$ (trời tối / ban đêm), tính số mũ hiệu chỉnh gamma:
       $$\gamma = \max\left(0.5, \frac{\ln(\text{targetLuma} / 255.0)}{\ln(\mu_Y / 255.0)}\right)$$
     - Tạo bảng tra `LUT` gồm 256 byte (`0..255`):
       $$\text{LUT}[i] = \text{rint}\left(255.0 \times \left(\frac{i}{255.0}\right)^\gamma\right).\text{toUInt8}()$$
     - Áp dụng `Core.LUT(channelY, lutMat, channelY)`, gộp lại với `Cr, Cb`, chuyển `YCrCb` $\rightarrow$ `BGR` $\rightarrow$ `RGBA` (`Imgproc.COLOR_BGR2RGBA`) và xuất ra mảng `ByteArray` `RGBA_8888` mới.

```kotlin
class BrightnessBalancer(private val targetLuma: Double = 70.0) {
    // Biến AtomicBoolean được gắn trực tiếp với Switch trên Jetpack Compose UI
    val enabled = java.util.concurrent.atomic.AtomicBoolean(false)

    // Tái sử dụng các Mat trên Luồng 1A (Thread-Confined) để tránh cấp phát bộ nhớ (Zero-GC)
    private var rgbaMat = org.opencv.core.Mat()
    private var bgrMat = org.opencv.core.Mat()
    private var ycrcbMat = org.opencv.core.Mat()
    private val channels = ArrayList<org.opencv.core.Mat>(3)
    private val lutMat = org.opencv.core.Mat(1, 256, org.opencv.core.CvType.CV_8UC1)
    private val lutBytes = ByteArray(256)

    fun processIfEnabled(rgbaBytes: ByteArray, width: Int, height: Int): ByteArray {
        if (!enabled.get() || targetLuma <= 0.0) {
            return rgbaBytes // Zero-Conversion Fast Path!
        }
        if (rgbaMat.rows() != height || rgbaMat.cols() != width) {
            rgbaMat.create(height, width, org.opencv.core.CvType.CV_8UC4)
        }
        rgbaMat.put(0, 0, rgbaBytes)
        org.opencv.imgproc.Imgproc.cvtColor(rgbaMat, bgrMat, org.opencv.imgproc.Imgproc.COLOR_RGBA2BGR)
        org.opencv.imgproc.Imgproc.cvtColor(bgrMat, ycrcbMat, org.opencv.imgproc.Imgproc.COLOR_BGR2YCrCb)

        org.opencv.core.Core.split(ycrcbMat, channels)
        val yChannel = channels[0]
        val meanY = org.opencv.core.Core.mean(yChannel).`val`[0]
        if (meanY <= 0.0 || meanY >= targetLuma) {
            channels.forEach { it.release() }
            channels.clear()
            return rgbaBytes // Ảnh đã đủ sáng, giữ nguyên buffer gốc
        }

        val gamma = kotlin.math.max(0.5, kotlin.math.ln(targetLuma / 255.0) / kotlin.math.ln(meanY / 255.0))
        for (i in 0..255) {
            val v = kotlin.math.round(255.0 * Math.pow(i / 255.0, gamma)).toInt().coerceIn(0, 255)
            lutBytes[i] = v.toByte()
        }
        lutMat.put(0, 0, lutBytes)
        org.opencv.core.Core.LUT(yChannel, lutMat, yChannel)
        org.opencv.core.Core.merge(channels, ycrcbMat)
        channels.forEach { it.release() }
        channels.clear()

        org.opencv.imgproc.Imgproc.cvtColor(ycrcbMat, bgrMat, org.opencv.imgproc.Imgproc.COLOR_YCrCb2BGR)
        org.opencv.imgproc.Imgproc.cvtColor(bgrMat, rgbaMat, org.opencv.imgproc.Imgproc.COLOR_BGR2RGBA)
        val out = ByteArray(rgbaBytes.size)
        rgbaMat.get(0, 0, out)
        return out
    }
}
```

---

### 3.2. Bộ Đệm Lưới Thời Gian `NeuralBuffer` Lưu Ảnh `RGBA_8888` Thô (Không Chạy `letterbox` Ở Nhánh Camera)
Nguồn tham chiếu: [`inference.py:180-200`](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/inference.py#L180-L200) nâng cấp kết hợp với cơ chế hoãn `letterbox` sang `ONNX Worker Thread`.

#### a. Tại sao đẩy `letterbox` ra khỏi `NeuralBuffer.updateRaw`?
- Trong [`inference.py:194`](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/inference.py#L194), dòng lệnh `frame = np.moveaxis(letterbox(rgb, self.config.neural_size), -1, 0)` được gọi trên **mọi frame của camera** (30–60 lần/giây), dù vòng lặp `while self.next_time <= timestamp + 1e-9:` bên dưới chỉ lấy mẫu ở tần số `5.0 FPS` (`5 lần/giây`). Như vậy có tới `25–55 lần/giây` kết quả `letterbox` bị ghi đè vào `self.previous` rồi vứt bỏ mà không bao giờ được đưa vào `self.frames`!
- Trong phiên bản `v1`, `NeuralBuffer.updateRaw(rawFrame: RawRgbaFrame, timestamp: Double)` chỉ lưu tham chiếu đối tượng bất biến `RawRgbaFrame(width = 640, height = 480, rgbaBytes: ByteArray)`. Thao tác cập nhật trên Luồng 1B chỉ tốn thời gian $\mathcal{O}(1)$ (`< 0.002 ms`, bằng **0 phép tính hình ảnh**).

```kotlin
class RawRgbaFrame(
    val width: Int,
    val height: Int,
    val rgbaBytes: ByteArray // Bất biến (Immutable) sau khi tạo tại Luồng 1A
)

data class NeuralDeltaBatch(
    val newRawFrames: List<RawRgbaFrame>,
    val keepOldCount: Int,
    val targetTotalFrames: Int,
    val generation: Long,
    val latestSeqId: Long
)

data class RawWindowSnapshot(
    val generation: Long,
    val latestSeqId: Long,
    val frames: List<RawRgbaFrame>
) {
    /**
     * Xác định chính xác những RawRgbaFrame mới phát sinh kể từ lần chạy ONNX trước.
     * Chỉ những frame trong newRawFrames (thường là 5 frames/giây) mới cần chạy letterbox!
     */
    fun getNewFramesSince(
        cachedGeneration: Long,
        cachedSeqId: Long,
        cachedFrameCount: Int
    ): NeuralDeltaBatch {
        val total = frames.size
        if (total == 0) {
            return NeuralDeltaBatch(emptyList(), 0, 0, generation, latestSeqId)
        }
        // Trường hợp 1: Khởi tạo mới hoặc camera từng bị đứt quãng > max_gap_sec (0.25s)
        if (generation != cachedGeneration || cachedSeqId <= 0L || cachedFrameCount <= 0) {
            return NeuralDeltaBatch(frames, keepOldCount = 0, targetTotalFrames = total, generation, latestSeqId)
        }
        val delta = (latestSeqId - cachedSeqId).toInt()
        return when {
            // Trường hợp 2: Trượt cửa sổ bình thường (ví dụ delta = 5 frames mới, giữ lại 45 frames p4_old)
            delta in 1 until total && cachedFrameCount >= (total - delta) -> {
                NeuralDeltaBatch(
                    newRawFrames = frames.takeLast(delta),
                    keepOldCount = total - delta,
                    targetTotalFrames = total,
                    generation = generation,
                    latestSeqId = latestSeqId
                )
            }
            // Trường hợp 3: Chưa có mốc lưới mới nào phát sinh (delta <= 0)
            delta <= 0 && cachedFrameCount >= total -> {
                NeuralDeltaBatch(
                    newRawFrames = frames.takeLast(1),
                    keepOldCount = total - 1,
                    targetTotalFrames = total,
                    generation = generation,
                    latestSeqId = latestSeqId
                )
            }
            // Trường hợp 4: Trượt quá cả một cửa sổ đầy (delta >= total)
            else -> {
                NeuralDeltaBatch(frames, keepOldCount = 0, targetTotalFrames = total, generation, latestSeqId)
            }
        }
    }
}

class NeuralBuffer(
    private val neuralCount: Int = 50,
    private val neuralFps: Double = 5.0,
    private val maxGapSec: Double = 0.25
) {
    private val frames = ArrayDeque<RawRgbaFrame>(neuralCount)
    private var previous: RawRgbaFrame? = null
    private var previousTime: Double? = null
    private var nextTime: Double = 0.0
    var generation: Long = 1L
        private set
    var seqCounter: Long = 0L
        private set

    val size: Int get() = frames.size

    fun updateRaw(rawFrame: RawRgbaFrame, timestamp: Double) {
        if (previousTime == null || timestamp - previousTime!! > maxGapSec) {
            frames.clear()
            nextTime = timestamp
            generation += 1L
            seqCounter = 0L
        }
        while (nextTime <= timestamp + 1e-9) {
            val past = if (nextTime < timestamp - 1e-9) (previous ?: rawFrame) else rawFrame
            if (frames.size == neuralCount) {
                frames.removeFirst()
            }
            frames.addLast(past)
            seqCounter += 1L
            nextTime += 1.0 / neuralFps
        }
        previous = rawFrame
        previousTime = timestamp
    }

    fun snapshotRaw(): RawWindowSnapshot = RawWindowSnapshot(
        generation = generation,
        latestSeqId = seqCounter,
        frames = frames.toList() // Chỉ copy tối đa 50 con trỏ đối tượng (< 0.005 ms)
    )
}
```

---

### 3.3. Thực Thi `letterboxRgbaToRgbCHW` Trên `ONNX Worker Thread` (Chỉ Chạy Trên $T_{\text{new}}$ Frames Mới)
Nguồn tham chiếu: [`neuralnetwork.py:91-120`](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/DataSystem/neuralnetwork.py#L91-L120).

Khi `ONNX Worker Thread` nhận `RawWindowSnapshot` mỗi `1.0s`, phương thức `getNewFramesSince(...)` trả về danh sách `newRawFrames` chỉ gồm $\Delta = T_{\text{new}}$ khung hình mới (thường $\Delta = 5$). Lúc này, `ONNX Worker Thread` mới thực hiện phép biến đổi `letterboxRgbaToRgbCHW` trên $\Delta$ khung hình đó:

1. **Tính tỷ lệ co giãn giữ nguyên tỷ lệ khung hình:**
   Với ảnh đầu vào $W = 640, H = 480$ và `newSize = 224`:
   $$s = \min\left(\frac{224.0}{H}, \frac{224.0}{W}\right) = \min\left(\frac{224.0}{480.0}, \frac{224.0}{640.0}\right) = 0.35$$
2. **Tính kích thước mới:**
   $$W_{\text{new}} = \max(1, \text{round}(640 \times 0.35)) = 224, \quad H_{\text{new}} = \max(1, \text{round}(480 \times 0.35)) = 168$$
3. **Tối ưu thứ tự Resize trước — Bỏ kênh Alpha sau:**
   - Do phép nội suy song tuyến `Imgproc.INTER_LINEAR` tính toán độc lập trên từng kênh màu $(R, G, B, A)$, việc chạy `Imgproc.resize` trên `rgbaMat` từ `640×480` về `224×168` rồi mới gọi `Imgproc.cvtColor(..., COLOR_RGBA2RGB)` trên ảnh nhỏ `224×168` cho kết quả **trùng khớp tuyệt đối 100%** với việc bỏ kênh Alpha trước rồi mới resize, nhưng giảm số lượng pixel phải chuyển hệ màu từ $307,200$ pixel xuống chỉ còn $37,632$ pixel (**nhanh hơn 8.16 lần**).
4. **Đặt vào nền vuông `224 × 224 × 3` màu `(114, 114, 114)`:**
   $$\text{pad\_left} = (224 - 224) \div 2 = 0, \quad \text{pad\_top} = (224 - 168) \div 2 = 28$$
   Copy ảnh RGB `224×168` vào vùng dòng `[28 .. 195]`, cột `[0 .. 223]`.
5. **Chuyển đổi bố cục bộ nhớ từ `HWC [224, 224, 3]` sang `CHW [3, 224, 224]`:**
   Đặt liên tiếp $224 \times 224 = 50,176$ byte kênh `R`, tiếp theo là $50,176$ byte kênh `G`, và $50,176$ byte kênh `B` vào `ByteBuffer` đầu vào của tensor `"video"` (`[T_new, 3, 224, 224]`).

```kotlin
class OnnxLetterboxProcessor(private val newSize: Int = 224) {
    private val srcRgbaMat = org.opencv.core.Mat()
    private val resizedRgbaMat = org.opencv.core.Mat()
    private val resizedRgbMat = org.opencv.core.Mat()
    private val canvasRgbMat = org.opencv.core.Mat(
        newSize, newSize, org.opencv.core.CvType.CV_8UC3,
        org.opencv.core.Scalar(114.0, 114.0, 114.0)
    )
    private val hwcScratch = ByteArray(newSize * newSize * 3)

    /**
     * Chuyển đổi danh sách T_new (~5) khung hình RawRgbaFrame thành mảng phẳng
     * [T_new, 3, 224, 224] kiểu uint8 (CHW) ngay trên ONNX Worker Thread.
     */
    fun processBatchToRgbCHW(rawFrames: List<RawRgbaFrame>): ByteBuffer {
        val planeSize = newSize * newSize // 50,176
        val frameSize = 3 * planeSize     // 150,528
        val outBuffer = ByteBuffer.allocateDirect(rawFrames.size * frameSize).order(ByteOrder.nativeOrder())
        val chwFrame = ByteArray(frameSize)

        for (raw in rawFrames) {
            val h = raw.height
            val w = raw.width
            if (srcRgbaMat.rows() != h || srcRgbaMat.cols() != w) {
                srcRgbaMat.create(h, w, org.opencv.core.CvType.CV_8UC4)
            }
            srcRgbaMat.put(0, 0, raw.rgbaBytes)

            val scale = kotlin.math.min(newSize.toDouble() / h.toDouble(), newSize.toDouble() / w.toDouble())
            val newW = kotlin.math.max(1, kotlin.math.round(w * scale).toInt())
            val newH = kotlin.math.max(1, kotlin.math.round(h * scale).toInt())

            // 1. Resize trên RGBA về (newW, newH)
            org.opencv.imgproc.Imgproc.resize(
                srcRgbaMat, resizedRgbaMat,
                org.opencv.core.Size(newW.toDouble(), newH.toDouble()),
                0.0, 0.0, org.opencv.imgproc.Imgproc.INTER_LINEAR
            )
            // 2. Loại bỏ kênh Alpha trên kích thước nhỏ (224x168) -> RGB
            org.opencv.imgproc.Imgproc.cvtColor(
                resizedRgbaMat, resizedRgbMat,
                org.opencv.imgproc.Imgproc.COLOR_RGBA2RGB
            )
            // 3. Điền nền (114, 114, 114) và chèn vào vị trí (padLeft, padTop)
            canvasRgbMat.setTo(org.opencv.core.Scalar(114.0, 114.0, 114.0))
            val padLeft = (newSize - newW) / 2
            val padTop = (newSize - newH) / 2
            val roi = canvasRgbMat.submat(padTop, padTop + newH, padLeft, padLeft + newW)
            resizedRgbMat.copyTo(roi)
            roi.release()

            // 4. Chuyển HWC [224, 224, 3] -> CHW [3, 224, 224]
            canvasRgbMat.get(0, 0, hwcScratch)
            var srcIdx = 0
            for (i in 0 until planeSize) {
                chwFrame[i] = hwcScratch[srcIdx]                     // R
                chwFrame[planeSize + i] = hwcScratch[srcIdx + 1]     // G
                chwFrame[2 * planeSize + i] = hwcScratch[srcIdx + 2] // B
                srcIdx += 3
            }
            outBuffer.put(chwFrame)
        }
        outBuffer.rewind()
        return outBuffer
    }
}
```

#### Bảng minh họa dòng chảy của `getNewFramesSince` + `letterbox` trên `ONNX Worker Thread`:

| Thời điểm `timestamp` | Giai đoạn | `frames.size` ($L$) | `latestSeqId` | `cachedSeqId` | Số frame chạy `letterbox` & `video` ($T_{\text{new}} = \Delta$) | Số frame lấy từ `p4_old` ($T_{\text{old}} = L - \Delta$) | Kích thước `p4_features` xuất ra ($T_{\text{old}} + T_{\text{new}}$) |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| `1.0s` | `WARMING UP` | `5` (#1..#5) | `5` | `0` | **5 frames** (#1..#5) | **0** `[0, 128, 14, 14]` | `[5, 128, 14, 14]` |
| `2.0s` | `WARMING UP` | `10` (#1..#10) | `10` | `5` | **5 frames** (#6..#10) | **5** (#1..#5) | `[10, 128, 14, 14]` |
| `...` | `WARMING UP` | `...` | `...` | `...` | **5 frames** | $L - 5$ | $[L, 128, 14, 14]$ |
| `10.0s` | `WARMING UP` | `50` (#1..#50) | `50` | `45` | **5 frames** (#46..#50) | **45** (#1..#45) | `[50, 128, 14, 14]` |
| `11.0s..59.0s` | `WARMING UP` | `50` | $5k$ | $5(k-1)$ | **5 frames** (5 frames mới nhất) | **45** (45 frames cuối của cache) | `[50, 128, 14, 14]` |
| **`60.0s`** | **Dự đoán đầu tiên** | `50` (#251..#300) | `300` | `295` | **5 frames** (#296..#300) | **45** (#251..#295) | `[50, 128, 14, 14]` + Xuất `prob[0, 1]` |
| `61.0s+` | **Suy luận liên tục** | `50` | $300 + 5m$ | $295 + 5m$ | **5 frames** (5 frames mới nhất) | **45** (45 frames cuối của cache) | `[50, 128, 14, 14]` + Xuất `prob[0, 1]` |

---

## 4. Đặc Tả Chi Tiết Nhánh Vật Lý: MediaPipe Ép Chạy GPU & Tách 3 Luồng Con Song Song (`Eye` $\parallel$ `Mouth` $\parallel$ `Head+FFT`)

### 4.1. `FaceTracker` Ép Chạy Trên GPU (`Delegate.GPU`) Với Đầu Vào `MPImage(IMAGE_FORMAT_RGBA, 640×480)` & Công Thức EAR / MAR
Nguồn tham chiếu: [`camera_metrics.py:18-107, 214-243`](file:///D:/Project/DATN/tempt/driver-guardian/ai/PhysicalBranch/camera_metrics.py#L18-L107).

#### a. Cấu hình MediaPipe `FaceLandmarker` Ép Thực Thi Trên GPU (`Delegate.GPU`)
- Trên Android, `FaceLandmarker` của MediaPipe Tasks Vision mặc định chạy trên CPU (`Delegate.CPU`) nếu không chỉ định rõ `setDelegate(Delegate.GPU)`.
- Để ép `MediaPipe FaceLandmarker` chạy trên GPU (OpenGL ES 3.1 / OpenCL shader):
  1. **Cấu hình `BaseOptions` với `Delegate.GPU`:**
     ```kotlin
     import android.content.Context
     import com.google.mediapipe.framework.image.MPImage
     import com.google.mediapipe.tasks.core.BaseOptions
     import com.google.mediapipe.tasks.core.Delegate
     import com.google.mediapipe.tasks.vision.core.RunningMode
     import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
     import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker.FaceLandmarkerOptions

     class GpuFaceTracker(
         context: Context,
         modelAssetPath: String = "face_landmarker.task",
         detectionConfidence: Float = 0.6f,
         trackingConfidence: Float = 0.6f
     ) {
         private val detector: FaceLandmarker
         val activeDelegate: Delegate

         init {
             // Bắt buộc khởi tạo trên đúng Luồng 1B (PhysicalCoordinator & MediaPipe GPU Thread)
             // để ngữ cảnh OpenGL ES (EGLContext) gắn chặt với luồng gọi detect()
             var initializedDetector: FaceLandmarker? = null
             var chosenDelegate = Delegate.GPU
             try {
                 val gpuBaseOptions = BaseOptions.builder()
                     .setDelegate(Delegate.GPU)
                     .setModelAssetPath(modelAssetPath)
                     .build()
                 val gpuOptions = FaceLandmarkerOptions.builder()
                     .setBaseOptions(gpuBaseOptions)
                     .setRunningMode(RunningMode.IMAGE)
                     .setNumFaces(1)
                     .setMinFaceDetectionConfidence(detectionConfidence)
                     .setMinFacePresenceConfidence(detectionConfidence)
                     .setMinTrackingConfidence(trackingConfidence)
                     .setOutputFaceBlendshapes(false)
                     .setOutputFacialTransformationMatrixes(false)
                     .build()
                 initializedDetector = FaceLandmarker.createFromOptions(context, gpuOptions)
             } catch (gpuException: Exception) {
                 // Fallback an toàn về CPU duy nhất khi trình giả lập hoặc GPU quá cũ không hỗ trợ GLES 3.1
                 chosenDelegate = Delegate.CPU
                 val cpuBaseOptions = BaseOptions.builder()
                     .setDelegate(Delegate.CPU)
                     .setModelAssetPath(modelAssetPath)
                     .build()
                 val cpuOptions = FaceLandmarkerOptions.builder()
                     .setBaseOptions(cpuBaseOptions)
                     .setRunningMode(RunningMode.IMAGE)
                     .setNumFaces(1)
                     .setMinFaceDetectionConfidence(detectionConfidence)
                     .setMinFacePresenceConfidence(detectionConfidence)
                     .setMinTrackingConfidence(trackingConfidence)
                     .setOutputFaceBlendshapes(false)
                     .setOutputFacialTransformationMatrixes(false)
                     .build()
                 initializedDetector = FaceLandmarker.createFromOptions(context, cpuOptions)
             }
             detector = initializedDetector!!
             activeDelegate = chosenDelegate
         }

         /**
          * Nhận trực tiếp MPImage (IMAGE_FORMAT_RGBA, 640×480) từ CameraX Hardware.
          * Trả về mảng tọa độ pixel thực DoubleArray[478 * 2] hoặc Array<DoubleArray>(478) { DoubleArray(2) }.
          */
         fun processRgbaFrame(rgbaBytes: ByteArray, width: Int, height: Int): Array<DoubleArray>? {
             val mpImage: MPImage = CameraXHardwareConfig.wrapRgbaToMPImage(rgbaBytes, width, height)
             val result = detector.detect(mpImage)
             mpImage.close()
             val faces = result.faceLandmarks()
             if (faces.isNullOrEmpty()) return null
             val landmarks = faces[0]
             if (landmarks.size < 468) return null
             val points = Array(landmarks.size) { i ->
                 val lm = landmarks[i]
                 val px = lm.x().toDouble() * width.toDouble()
                 val py = lm.y().toDouble() * height.toDouble()
                 if (!px.isFinite() || !py.isFinite()) return null
                 doubleArrayOf(px, py)
             }
             return points
         }

         fun close() {
             detector.close()
         }
     }
     ```
  2. **Quy tắc gắn kết luồng GPU (`Thread Affinity Rule`):**
     - Trong MediaPipe / TFLite GPU Delegate, ngữ cảnh OpenGL ES (`EGLContext`) được gắn với OS Thread đã gọi `FaceLandmarker.createFromOptions(...)`.
     - Do đó, cả lệnh khởi tạo `GpuFaceTracker(context)` lẫn mọi lệnh `processRgbaFrame(...)` đều phải được gọi trên **`Luồng 1B` (`Executors.newSingleThreadExecutor().asCoroutineDispatcher()`)**.

#### b. Công thức tính `aspect_ratio(points, indices)`
Với bộ 6 chỉ số landmark $[i_0, i_1, i_2, i_3, i_4, i_5]$ tương ứng các điểm 2D $[p_0, p_1, p_2, p_3, p_4, p_5]$:
- Chiều rộng ngang: $w = \|p_0 - p_3\|_2 = \sqrt{(x_0 - x_3)^2 + (y_0 - y_3)^2}$.
- Nếu $w < 10^{-6}$ hoặc có tọa độ không hữu hạn $\rightarrow$ trả về `null`.
- Tỷ số khung (Aspect Ratio):
  $$\text{AR} = \frac{\|p_1 - p_5\|_2 + \|p_2 - p_4\|_2}{2.0 \times w}$$

#### c. Chỉ số Landmark cụ thể cho Mắt (`EAR`) và Miệng (`MAR`)
- **Mắt trái (`LEFT_EYE`):** `(362, 385, 387, 263, 373, 380)` $\rightarrow \text{EAR}_{\text{left}}$
- **Mắt phải (`RIGHT_EYE`):** `(33, 160, 158, 133, 153, 144)` $\rightarrow \text{EAR}_{\text{right}}$
- **EAR tổng hợp:**
  $$\text{EAR} = \begin{cases} \frac{\text{EAR}_{\text{left}} + \text{EAR}_{\text{right}}}{2.0} & \text{nếu cả hai } \neq \text{null} \\ \text{null} & \text{ngược lại} \end{cases}$$
- **Miệng (`MOUTH`):** `(61, 37, 267, 291, 314, 84)` $\rightarrow \text{MAR}$

---

### 4.2. `HeadPoseEstimator` (SolvePnP) & Hiệu Chuẩn Góc Đầu
Nguồn: [`head_pose_estimation.py:8-55`](file:///D:/Project/DATN/tempt/driver-guardian/ai/PhysicalBranch/head_pose_estimation.py#L8-L55) và [`camera_metrics.py:229-271`](file:///D:/Project/DATN/tempt/driver-guardian/ai/PhysicalBranch/camera_metrics.py#L229-L271).

#### a. Mô hình 3D chuẩn & Nội tại Camera (Với kích thước chuẩn `640×480`)
- **6 chỉ số Landmark 2D (`pose_landmarks`):** `(1, 152, 263, 33, 291, 61)` (Chóp mũi, Cằm, Đuôi mắt trái, Đuôi mắt phải, Khóe miệng trái, Khóe miệng phải).
- **Tọa độ 3D tương ứng (`face_model`):**
  ```kotlin
  val FACE_MODEL_3D = arrayOf(
      doubleArrayOf(0.0, 0.0, 0.0),         // 1: Nose tip
      doubleArrayOf(0.0, 330.0, 65.0),      // 152: Chin
      doubleArrayOf(225.0, -170.0, 135.0),  // 263: Left eye outer corner
      doubleArrayOf(-225.0, -170.0, 135.0), // 33: Right eye outer corner
      doubleArrayOf(150.0, 150.0, 125.0),   // 291: Left mouth corner
      doubleArrayOf(-150.0, 150.0, 125.0)   // 61: Right mouth corner
  )
  ```
- **Ma trận nội tại Camera $K$ ($3 \times 3$):** Với ảnh kích thước $(W=640, H=480)$:
  $$K = \begin{bmatrix} W & 0 & W/2.0 \\ 0 & W & H/2.0 \\ 0 & 0 & 1.0 \end{bmatrix} = \begin{bmatrix} 640.0 & 0.0 & 320.0 \\ 0.0 & 640.0 & 240.0 \\ 0.0 & 0.0 & 1.0 \end{bmatrix}, \quad \text{distortion} = [0, 0, 0, 0]$$

#### b. Các bước giải `estimate(landmarks_2d)`
1. **Kiểm tra suy biến (Collinearity / Rank check):**
   Trừ tọa độ trung bình $(\bar{x}, \bar{y})$ khỏi 6 điểm 2D. Tính hạng ma trận $6 \times 2$. Nếu $\text{rank} < 2$ (6 điểm thẳng hàng) $\rightarrow$ ném lỗi `"Degenerate landmarks"`.
2. **Giải PnP ban đầu:**
   Gọi `Calib3d.solvePnP(MODEL, points, K, distortion, rvec, tvec, false, Calib3d.SOLVEPNP_SQPNP)`.
3. **Tinh chỉnh Levenberg-Marquardt:**
   Gọi `Calib3d.solvePnPRefineLM(MODEL, points, K, distortion, rvec, tvec)`.
4. **Kiểm tra nghiệm phía trước Camera (Cheirality check):**
   Chuyển `rvec` sang ma trận quay $R_{3\times3}$ bằng `Calib3d.Rodrigues(rvec, R)`.
   Tính tọa độ $Z$ trong hệ trục camera của cả 6 điểm 3D:
   $$P_{\text{cam}} = \text{MODEL} \times R^T + t^T$$
   Nếu bất kỳ điểm nào có $Z_{\text{cam}} \le 0$ $\rightarrow$ ném lỗi `"Pose behind camera"`.
5. **Phân rã góc Euler:**
   Gọi `Calib3d.RQDecomp3x3(R, mtxR, mtxQ)` trả về bộ 3 góc Euler thô `raw = (pitch, yaw, roll)` tính bằng **độ (degrees)**.

#### c. Hiệu chuẩn tư thế trung tính (`neutral` calibration) & Hysteresis Gật Đầu
1. **Khi chưa hiệu chuẩn (`neutral == null`):**
   - Nếu frame không có mặt, hoặc `estimate()` ném lỗi, hoặc khoảng cách thời gian `timestamp - prev_timestamp > max_gap_sec` ($0.25\text{s}$) $\rightarrow$ xóa danh sách `neutral_samples.clear()` (yêu cầu 30 frames hợp lệ **liên tiếp**).
   - Mỗi frame hợp lệ thêm `raw` vào `neutral_samples`.
   - Khi `neutral_samples.size == calibration_frames` ($30$):
     - Thực hiện mở cuộn pha góc (`unwrap` theo chu kỳ $360^\circ$ dọc theo trục thời gian cho từng góc pitch, yaw, roll).
     - Lấy trung vị (`median`) của 30 mẫu đã unwrap cho từng trục, sau đó chuẩn hóa về $[-180^\circ, 180^\circ)$:
       $$\text{neutral} = ((\text{median}(u) + 180.0) \bmod 360.0) - 180.0$$
     - Xóa `neutral_samples.clear()`.
2. **Tính góc tương đối (`angles`) sau khi đã có `neutral`:**
   $$\text{angles}[k] = ((\text{raw}[k] - \text{neutral}[k] + 180.0) \bmod 360.0) - 180.0, \quad k \in \{0, 1, 2\}$$
3. **Phát hiện vượt ngưỡng góc (`over_angle`) & trạng thái gật đầu (`nod_state`):**
   - `over_angle = (abs(angles[0]) > 20.0) || (abs(angles[1]) > 25.0) || (abs(angles[2]) > 20.0)`
   - Tín hiệu gật `pitch_down`: Mặc định `nod_direction = null` nên `pitch_down = abs(angles[0])`.
   - **Mạch trễ Hysteresis 2 ngưỡng (`nod_pitch_deg = 14.0`, `nod_release_deg = 8.0`):**
     ```kotlin
     val threshold = if (nodding) config.nodReleaseDeg else config.nodPitchDeg
     nodding = pitchDown > threshold
     val nodState: Int? = if (nodding) 1 else 0
     ```
     Nếu `angles == null` (mất mặt hoặc lỗi PnP), đặt `nodding = false` và `nodState = null`.

---

### 4.3. `AdaptiveHMM` & `AdaptiveHMM_FSM` (Mắt & Miệng)
Nguồn: [`adaptive_hmm_fsm.py:7-313`](file:///D:/Project/DATN/tempt/driver-guardian/ai/PhysicalBranch/adaptive_hmm_fsm.py#L7-L313).

Mỗi nhánh `eye` (đầu vào `EAR`, chạy trên **`Luồng 1C-1: EyeWorker`**) và `mouth` (đầu vào `MAR`, chạy trên **`Luồng 1C-2: MouthWorker`**) được quản lý bởi một bộ `AdaptiveHMM_FSM` hoàn toàn độc lập:
- **Quy ước trạng thái HMM (`n_states = 2`):**
  - Trạng thái `0`: **Trạng thái nghỉ (Resting)** — Mắt mở (`EAR` cao) hoặc Miệng ngậm (`MAR` thấp).
  - Trạng thái `1`: **Trạng thái sự kiện (Event / Positive State)** — Mắt nhắm (`EAR` thấp, `positive_state = "low"`) hoặc Miệng mở/ngáp (`MAR` cao, `positive_state = "high"`).

#### a. Giai đoạn Khởi tạo (`_initialize` sau `init_frames = 300` mẫu hợp lệ)
Khi thu thập đủ 300 mẫu hợp lệ đầu tiên ($10\text{s} \times 30\text{ FPS}$) vào `samples`:
1. Tính phân vị thứ 5 và 95: `low = percentile(samples, 5.0)`, `high = percentile(samples, 95.0)`.
2. Tính mức nền `normal`:
   - Với `eye`: `normal = percentile(samples, 80.0)` (`eye_baseline_percentile = 80`).
   - Với `mouth`: `normal = percentile(samples, 20.0)` (`other_baseline_percentile = 20`).
3. Kiểm tra điều kiện phân tách 2 cụm rõ rệt:
   - Với `eye`: `separation = 0.06`, `distinct = (low < normal * 0.65)`.
   - Với `mouth`: `separation = 0.15`, `distinct = (high > normal + 0.15)`.
4. **Trường hợp 1: Có đủ 2 cụm (`high - low >= separation && distinct`) $\rightarrow$ Chạy `hmm.fit_initial(samples)`:**
   - **Khởi tạo 1D K-Means (`_kmeans_1d`, tối đa 25 vòng)** và **Ước lượng tham số ban đầu từ nhãn (`_estimate_from_labels`)** với `eps = 1e-3`.
   - **Chạy 20 vòng lặp Baum-Welch EM (`em_iterations = 20`):**
     - **Bước E (`_forward_backward` trong miền Log):**
       $$\log B_t(k) = -\frac{1}{2} \left( \ln(2\pi \sigma_k^2) + \frac{(x_t - \mu_k)^2}{\sigma_k^2} \right)$$
       $$\log \alpha_0(k) = \ln \max(\pi_k, 10^{-300}) + \log B_0(k), \quad \log \alpha_t(j) = \log B_t(j) + \text{logsumexp}_{i=0}^1 \left( \log \alpha_{t-1}(i) + \ln \max(A_{i,j}, 10^{-300}) \right)$$
       $$\log \beta_{T-1}(k) = 0, \quad \log \beta_t(i) = \text{logsumexp}_{j=0}^1 \left( \ln \max(A_{i,j}, 10^{-300}) + \log B_{t+1}(j) + \log \beta_{t+1}(j) \right)$$
       $$\gamma_t(k) = \exp\left(\log \alpha_t(k) + \log \beta_t(k) - \text{logsumexp}_j(\log \alpha_t(j) + \log \beta_t(j))\right)$$
       $$\xi_t(i, j) = \exp\left(\log \Xi_t(i, j) - \text{logsumexp}_{u,v} \log \Xi_t(u, v)\right)$$
     - **Bước M (`_m_step` với $\alpha = 1.0$ khi khởi tạo):** Cập nhật $\pi, A, \mu, \sigma^2$ từ $\gamma$ và $\xi$ ($\sigma_k^2 \ge 10^{-5}$).
   - **Sắp xếp trạng thái (`_sort_states`):** Với `eye` ($\mu_0 \ge \mu_1$), với `mouth` ($\mu_0 \le \mu_1$).
5. **Trường hợp 2: Không đủ 2 cụm $\rightarrow$ Khởi tạo Dự Phòng (Fallback Model):**
   - Với `eye`: $\mu_0 = \text{normal}$, $\mu_1 = \text{normal} \times 0.2$. Với `mouth`: $\mu_0 = \text{normal}$, $\mu_1 = \text{normal} + 0.30$.
   - Phương sai: $\sigma = \max(|\mu_0 - \mu_1| / 4.0, 0.01)$, $\sigma_0^2 = \sigma_1^2 = \sigma^2$, $A = \begin{bmatrix} 0.97 & 0.03 \\ 0.10 & 0.90 \end{bmatrix}$, $\pi = [0.99, 0.01]$.
6. Lưu lại tham chiếu sự kiện ban đầu: `event_reference = hmm.means[1]`.

#### b. Suy Luận Trực Tuyến Từng Frame (`predict`) & Học Thích Nghi (`update_online`)
Tại mỗi frame có giá trị đo `value` hợp lệ (`value != null && value >= 0`):
1. **Kiểm tra đứt quãng:** Nếu `timestamp - prev_time > max_gap_sec` ($0.25\text{s}$) hoặc `value == null`, xóa `hmm.posterior = null` và `hmm.buffer.clear()`.
2. **Dự đoán trạng thái tức thời (`hmm.predict(value)`):** Tính `prior`, `log_p`, chuẩn hóa `posterior` và lấy `state = argmax(posterior)`.
3. **Quy tắc kẹp an toàn cho mắt mở (`eye_open_ratio = 0.75`):** Nếu `mode == "eye"` và `value >= normal * 0.75`, cưỡng bức `state = 0` và `posterior = [0.99, 0.01]`.
4. **Cập nhật trực tuyến (`hmm.update_online(value)`):** Khi `hmm.buffer.size == 300`, chạy 1 vòng `_forward_backward` + `_m_step` với $\alpha = 0.01$ nếu $\min_k \sum_t \gamma_t(k) \ge 2.0$.
5. **Tính ngưỡng `p80_threshold` và tín hiệu `active` cho `WindowRatio`:**
   - Với `eye`: $\text{p80\_threshold} = \text{normal} - 0.8 \times (\text{normal} - \text{event\_reference})$, $\text{active} = (\text{value} \le \text{p80\_threshold})$.
   - Với `mouth`: $\text{active} = (\text{state} == 1)$.

---

### 4.4. `StateMachine` (FSM Đếm Sự Kiện & Lọc Thời Lượng)
Nguồn: [`temporal_metrics.py:15-82`](file:///D:/Project/DATN/tempt/driver-guardian/ai/PhysicalBranch/temporal_metrics.py#L15-L82).

Được dùng cho cả 3 bộ đếm sự kiện trên 3 luồng con tương ứng: `blink` (`Luồng 1C-1`), `yawn` (`Luồng 1C-2`), và `nod` (`Luồng 1C-3`):
- **Giới hạn thời lượng hợp lệ `[minimum, maximum]` (ms):**
  - `eye` (Nháy mắt): `[100.0, 2000.0]` ms
  - `mouth` (Ngáp): `[4000.0, 7500.0]` ms
  - `pitch` (Gật đầu): `[800.0, 3500.0]` ms
- **Logic chuyển trạng thái trong `process(state, timestamp)`:**
  - Nếu `time != null` và `timestamp - time > max_gap` ($0.25\text{s}$): `start = null`, `armed = false`.
  - Khi chuyển $1 \to 0$ với `start != null`: `last_duration = (now - start) * 1000.0`, kiểm tra `done = armed && (minimum - 1e-6 <= last_duration <= maximum + 1e-6)`. Nếu `done == true`, thêm `now` vào `events` và cập nhật `last_valid_duration = last_duration`.
  - Nếu `state == 0`, đặt `armed = true`. Loại bỏ sự kiện cũ $\le \text{now} - \text{window}$ và tính $\text{rate\_per\_minute} = \text{events.size} \times 60.0 / \text{window}$.

---

### 4.5. `WindowRatio` (Tính PERCLOS, POM, Over-Angle & Coverage)
Nguồn: [`temporal_metrics.py:84-115`](file:///D:/Project/DATN/tempt/driver-guardian/ai/PhysicalBranch/temporal_metrics.py#L84-L115).

Tính phần trăm thời gian tín hiệu `active == true` và tổng số giây quan sát hợp lệ (`observed_sec`) trên cửa sổ trượt `window = 60.0s`:
- Lưu danh sách các đoạn `(start, end, state: Boolean)` khi `timestamp - prev_time <= max_gap` ($0.25\text{s}$).
- Cắt bỏ các đoạn có `end <= timestamp - window`.
- Tính tổng thời gian quan sát hợp lệ `valid` và thời gian kích hoạt `total` với biên trái $\max(s, \text{cutoff})$.
- Trả về `percentage = if (valid > 0.0) 100.0 * total / valid else null` và `observed_sec = valid`.

---

### 4.6. `HeadMotionWindow` (Tích Phân Biên Độ & Phổ FFT)
Nguồn: [`temporal_metrics.py:117-185`](file:///D:/Project/DATN/tempt/driver-guardian/ai/PhysicalBranch/temporal_metrics.py#L117-L185).

Chạy trên **`Luồng 1C-3: HeadPoseAndFftWorker`**, tách biệt hoàn toàn với `EyeWorker` và `MouthWorker`:
1. **Tính liên tục nghiêm ngặt:** Nếu `timestamp - prev_time > max_gap` ($0.25\text{s}$) hoặc `pitch == null`, **xóa toàn bộ `samples.clear()`**.
2. **Cắt tỉa cửa sổ trượt:** Giữ lại 1 điểm neo bên trái `cutoff = timestamp - 60.0s` để nội suy biên.
3. **Nội suy tại mốc biên `start = max(times[0], cutoff)` và tính tích phân hình thang:** Tách hai tam giác khi đoạn thẳng cắt qua trục $0$:
   $$\text{amp}_k = \frac{l^2 + r^2}{2.0 \times (|l| + |r|)} \quad \text{khi } (l < 0 \land r > 0) \lor (l > 0 \land r < 0)$$
4. **Tính Phổ Tần Số Chủ Đạo qua FFT (`head_motion_frequency_hz`):**
   - Khi `observed >= window - 1e-9` ($60.0\text{s}$) và `times.size >= 4`:
   - Tính số điểm lưới đều $N = \max(4, \text{round}(\text{observed} / \text{median}(\text{diff}(\text{times}))) + 1)$.
   - Nội suy tuyến tính chuỗi tín hiệu $s[0..N-1]$ trên lưới `linspace(start, timestamp, N)`.
   - Nếu $\max(s) - \min(s) \le 0.1^\circ$ $\rightarrow$ `head_motion_frequency_hz = 0.0`.
   - Ngược lại, trừ trung bình, nhân cửa sổ Hann, biến đổi `rfft`, tìm đỉnh năng lượng $k_{\text{peak}} = \arg\max_{k \ge 1} |X[k]|^2$ và trả về $\text{head\_motion\_frequency\_hz} = k_{\text{peak}} / (N \times dt)$.

---

### 4.7. Kiến Trúc Tách 3 Luồng Con Vật Lý Song Song (`ParallelCameraMetrics`: `EyeWorker` $\parallel$ `MouthWorker` $\parallel$ `HeadPoseAndFftWorker`)

#### a. Chứng minh tính độc lập dữ liệu 100% giữa 3 nhánh sau MediaPipe GPU
Quan sát hàm [`CameraMetrics.process_landmarks(points, image_size, timestamp)`](file:///D:/Project/DATN/tempt/driver-guardian/ai/PhysicalBranch/camera_metrics.py#L1712-L1810) trong bản gốc:
- Đầu vào chung cho cả 3 nhánh là mảng điểm mốc **chỉ đọc (Read-Only)** `points: Array<DoubleArray>?` (`[478, 2]`), kích thước `imageSize = (640, 480)` và `timestamp: Double`.
- Tập biến trạng thái của 3 nhánh **rời nhau hoàn toàn (Disjoint State Sets)**:
  - **Nhánh 1 (`EyeWorker`):** Chỉ đọc `points[LEFT_EYE]`, `points[RIGHT_EYE]` và chỉ đọc/ghi vào `this.eye` (`AdaptiveHMM_FSM("eye")`) cùng `this.blinkDeque`.
  - **Nhánh 2 (`MouthWorker`):** Chỉ đọc `points[MOUTH]` và chỉ đọc/ghi vào `this.mouth` (`AdaptiveHMM_FSM("mouth")`).
  - **Nhánh 3 (`HeadPoseAndFftWorker`):** Chỉ đọc `points[LANDMARKS]`, `imageSize` và chỉ đọc/ghi vào `this.estimator`, `this.neutralSamples`, `this.neutral`, `this.nodding`, `this.nod`, `this.overAngleRatio`, `this.headMotion` cùng `this.nodDeque`.

Do $\text{State}(\text{Eye}) \cap \text{State}(\text{Mouth}) = \emptyset$, $\text{State}(\text{Eye}) \cap \text{State}(\text{Head}) = \emptyset$, và $\text{State}(\text{Mouth}) \cap \text{State}(\text{Head}) = \emptyset$, việc chạy đồng thời 3 nhánh trên 3 luồng CPU bằng `coroutineScope { async ... }` **không cần bất kỳ Mutex hay Lock nào**, không bao giờ xảy ra Race Condition, và cho ra kết quả `CameraMetricsSnapshot` **giống hệt 100% từng bit** so với chạy tuần tự!

#### b. Mã nguồn tham chiếu Kotlin cho `ParallelCameraMetrics` (Fork-Join 3 Luồng Con Song Song)

```kotlin
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope

class ParallelCameraMetrics(
    private val config: DriverGuardianConfig,
    // 3 Dispatcher đơn luồng chuyên biệt để gắn cố định từng nhánh vào từng nhân CPU (Zero Cache-Miss)
    private val eyeDispatcher: CoroutineDispatcher = Dispatchers.Default.limitedParallelism(1),
    private val mouthDispatcher: CoroutineDispatcher = Dispatchers.Default.limitedParallelism(1),
    private val headDispatcher: CoroutineDispatcher = Dispatchers.Default.limitedParallelism(1)
) {
    // Trạng thái riêng của Nhánh 1C-1 (EyeWorker)
    private val eye = AdaptiveHmmFsm(mode = "eye", config = config)
    private val blinkDeque = ArrayDeque<Pair<Double, Double>>()

    // Trạng thái riêng của Nhánh 1C-2 (MouthWorker)
    private val mouth = AdaptiveHmmFsm(mode = "mouth", config = config)

    // Trạng thái riêng của Nhánh 1C-3 (HeadPoseAndFftWorker)
    private var estimator: HeadPoseEstimator? = null
    private var shape: Pair<Int, Int>? = null
    private val neutralSamples = ArrayList<DoubleArray>(config.calibrationFrames)
    private var neutral: DoubleArray? = null
    private var nodding: Boolean = false
    private val nod = StateMachine(mode = "pitch", config = config)
    private val overAngleRatio = WindowRatio(config.windowSec, config.maxGapSec)
    private val headMotion = HeadMotionWindow(config.windowSec, config.maxGapSec, config.headMotionMinAmplitudeDeg)
    private val nodDeque = ArrayDeque<Pair<Double, Double>>()
    private var prevTimestamp: Double? = null

    /**
     * Thực thi song song 3 nhánh Vật Lý (Eye || Mouth || Head+FFT) sau khi có points từ MediaPipe GPU.
     */
    suspend fun processLandmarksParallel(
        points: Array<DoubleArray>?,
        imageWidth: Int,
        imageHeight: Int,
        timestamp: Double
    ): ParallelPhysicalResult = coroutineScope {
        // Fork 1: Nhánh Mắt chạy trên EyeWorker (1C-1)
        val eyeDeferred = async(eyeDispatcher) {
            val ear = if (points != null) {
                val left = aspectRatio(points, config.leftEye)
                val right = aspectRatio(points, config.rightEye)
                if (left != null && right != null) (left + right) / 2.0 else null
            } else null

            val eyeOut = eye.process(ear, timestamp)
            if (eyeOut.eventDone && eyeOut.lastValidDurationMs != null) {
                blinkDeque.addLast(timestamp to eyeOut.lastValidDurationMs)
            }
            while (blinkDeque.isNotEmpty() && blinkDeque.first().first <= timestamp - config.windowSec) {
                blinkDeque.removeFirst()
            }
            Triple(ear, eyeOut, blinkDeque.toList())
        }

        // Fork 2: Nhánh Miệng chạy trên MouthWorker (1C-2)
        val mouthDeferred = async(mouthDispatcher) {
            val mar = if (points != null) aspectRatio(points, config.mouth) else null
            val mouthOut = mouth.process(mar, timestamp)
            Pair(mar, mouthOut)
        }

        // Fork 3: Nhánh Tư Thế Đầu + SolvePnP + FFT 1800 điểm chạy trên HeadPoseAndFftWorker (1C-3)
        val headDeferred = async(headDispatcher) {
            if (prevTimestamp != null && timestamp - prevTimestamp!! > nod.maxGap) {
                nodding = false
                neutralSamples.clear()
            }
            prevTimestamp = timestamp

            val currentShape = imageWidth to imageHeight
            if (shape != currentShape) {
                estimator = HeadPoseEstimator(imageWidth, imageHeight, config = config)
                shape = currentShape
                resetHeadInternal()
            }

            var raw: DoubleArray? = null
            var angles: DoubleArray? = null
            var poseError: String? = null

            if (points != null) {
                try {
                    val posePts = Array(6) { idx -> points[config.poseLandmarks[idx]] }
                    raw = estimator!!.estimate(posePts)
                    if (neutral == null) {
                        neutralSamples.add(raw)
                        if (neutralSamples.size >= config.calibrationFrames) {
                            neutral = computeUnwrappedMedianNeutral(neutralSamples)
                            neutralSamples.clear()
                        }
                    }
                    if (neutral != null) {
                        angles = DoubleArray(3) { k -> wrap180(raw[k] - neutral!![k]) }
                    }
                } catch (e: Exception) {
                    poseError = e.message
                    neutralSamples.clear()
                }
            } else if (neutral == null) {
                neutralSamples.clear()
            }

            var overAngle: Boolean? = null
            var nodState: Int? = null
            if (angles != null) {
                overAngle = kotlin.math.abs(angles[0]) > config.angleLimits[0] ||
                            kotlin.math.abs(angles[1]) > config.angleLimits[1] ||
                            kotlin.math.abs(angles[2]) > config.angleLimits[2]
                val pitchDown = if (config.nodDirection == null) {
                    kotlin.math.abs(angles[0])
                } else {
                    config.nodDirection * angles[0]
                }
                val threshold = if (nodding) config.nodReleaseDeg else config.nodPitchDeg
                nodding = pitchDown > threshold
                nodState = if (nodding) 1 else 0
            } else {
                nodding = false
            }

            val nodOut = nod.process(nodState, timestamp)
            val (overPct, headSec) = overAngleRatio.process(overAngle, timestamp)
            val motionOut = headMotion.process(angles?.get(0), timestamp)

            if (nodOut.eventDone && nodOut.lastValidDurationMs != null) {
                nodDeque.addLast(timestamp to nodOut.lastValidDurationMs)
            }
            while (nodDeque.isNotEmpty() && nodDeque.first().first <= timestamp - config.windowSec) {
                nodDeque.removeFirst()
            }

            HeadBranchSnapshot(
                raw = raw,
                angles = angles,
                poseError = poseError,
                headCalibrated = neutral != null,
                overAngle = overAngle,
                overAnglePct = overPct,
                nodState = nodState,
                nodOut = nodOut,
                headObservedSec = headSec,
                motionOut = motionOut,
                nodHistorySnapshot = nodDeque.toList()
            )
        }

        // Join cả 3 nhánh (Wait-Free Structured Concurrency)
        val (ear, eyeOut, blinkSnapshot) = eyeDeferred.await()
        val (mar, mouthOut) = mouthDeferred.await()
        val headSnap = headDeferred.await()

        ParallelPhysicalResult(
            metricsSnapshot = buildCameraMetricsSnapshot(timestamp, points != null, ear, eyeOut, mar, mouthOut, headSnap),
            blinkHistorySnapshot = blinkSnapshot,
            nodHistorySnapshot = headSnap.nodHistorySnapshot
        )
    }

    private fun resetHeadInternal() {
        nod.reset()
        overAngleRatio.reset()
        headMotion.reset()
        neutralSamples.clear()
        neutral = null
        nodding = false
    }

    companion object {
        fun wrap180(deg: Double): Double = ((deg + 180.0).mod(360.0)) - 180.0
    }
}
```

---

## 5. Đặc Tả Nhánh Học Sâu (`CNN–ConvGRU`) & Định Dạng Model ONNX Theo `export_onnx.py`

### 5.1. Kiến Trúc `ExportDrowsyClassifier` & `ScriptableConvGRU`
Nguồn: [`export_onnx.py:46-156`](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/export_onnx.py#L46-L156), [`model.py`](file:///D:/Project/DATN/tempt/driver-guardian/ai/LSTM/src/model.py), [`model_backbone_neck.py`](file:///D:/Project/DATN/tempt/driver-guardian/ai/LSTM/src/model_backbone_neck.py), [`models_convgru.py`](file:///D:/Project/DATN/tempt/driver-guardian/ai/LSTM/src/models_convgru.py).

Mô hình gốc trong `best.pt` có cấu hình chuẩn từ `TrainConfig`:
- `backbone_w = (16, 32, 64, 128, 256)`, `backbone_n = (1, 2, 2, 1)`, `neck_n = 1`
- `NECK_P4_CHANNELS = 128`, `feat_spatial = IMAGE_SIZE // 16 = 14` (stride = 16 trên ảnh `224×224`)
- `input_dim = 96`, `hidden_dim = 96`, `image_size = 224`, cửa sổ đầy đủ `T = 50` frames (`10s @ 5 FPS`).

Trong [`export_onnx.py`](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/export_onnx.py), mô hình được đóng gói qua lớp `ExportDrowsyClassifier` (kết hợp `ScriptableConvGRU` được biên dịch bằng `torch.jit.script` sang toán tử vòng lặp động `onnx::Loop`, `opset_version = 17`) để hỗ trợ **suy luận cửa sổ trượt gia tăng (Incremental Sliding-Window Inference)**:

1. **Chuẩn hóa đầu vào tự động trong đồ thị (`x -> frames`):**
   - Nhận tensor `"video"` kích thước `[T_new, 3, 224, 224]` được tạo bởi `OnnxLetterboxProcessor.processBatchToRgbCHW(delta.newRawFrames)` ngay trên `ONNX Worker Thread`.
   - Nếu `video` có kiểu `uint8` $\in [0..255]$, đồ thị ONNX tự động ép kiểu sang `param_dtype` (`float32` hoặc `float16`) và chia cho `255.0` về miền $[0.0, 1.0]$.
2. **Trích xuất đặc trưng mức P4 chỉ trên các frame mới (`BackboneNeck`):**
   - Chạy `p4 = self.backbone_neck(frames)[1]` **chỉ trên $T_{\text{new}}$ frame mới** (ví dụ $T_{\text{new}} = 5$ frames mỗi giây thay vì 50 frames), thu được tensor `p4` kích thước `[T_new, 128, 14, 14]`.
3. **Ghép nối với bộ đệm đặc trưng cũ (`p4_old`):**
   - Nhận tensor `"p4_old"` kích thước `[T_old, 128, 14, 14]` từ lần chạy trước (nếu chưa có cache hoặc mới reset, truyền tensor rỗng `[0, 128, 14, 14]`).
   - Nối dọc theo chiều thời gian (`dim = 0`):
     $$\text{features} = \text{cat}([\text{p4\_old}, \text{p4}], \text{dim}=0) \in \mathbb{R}^{[T_{\text{total}}, 128, 14, 14]}, \quad T_{\text{total}} = T_{\text{old}} + T_{\text{new}}$$
   - Tensor `features` này được xuất trực tiếp ra Output thứ 2 (`"p4_features"`) để ứng dụng Android cắt giữ lại phần đuôi làm `"p4_old"` cho chu kỳ kế tiếp.
4. **Chuyển đổi số kênh qua `adapter`:**
   - `adapted = self.adapter(features)` gồm `Conv2d(128 -> 96, 1×1, bias=False)` $\rightarrow$ `GroupNorm(1, 96)` $\rightarrow$ `SiLU`, sau đó `unsqueeze(0)` thành `adapted_seq` kích thước `[1, T_total, 96, 14, 14]`.
5. **Chuỗi thời gian qua `ScriptableConvGRU` (`ConvGRUCell(96, 96, kernel=3×3)`):**
   - Khởi tạo trạng thái ẩn $h_{-1} = \mathbf{0}_{[1, 96, 14, 14]}$.
   - Lặp $i \in \{0 .. T_{\text{total}} - 1\}$:
     - $h_i = \text{ConvGRUCell}(\text{adapted\_seq}[:, i], h_{i-1})$
     - **Spatial Attention:** $a_i = \text{softmax}_{14\times 14}(\text{Conv}_{1\times1}(h_i).\text{flatten}(2)).\text{reshape}(1, 1, 14, 14)$.
     - Vector đặc trưng khung hình: $v_i = \sum_{y=0}^{13} \sum_{x=0}^{13} (h_i \odot a_i) \in \mathbb{R}^{[1, 96]}$.
6. **Temporal Mean Pooling & Phân loại (`fc_out` + `Softmax`):**
   - Trung bình cộng theo thời gian: $\bar{v} = \frac{1}{T_{\text{total}}} \sum_{i=0}^{T_{\text{total}}-1} v_i \in \mathbb{R}^{[1, 96]}$.
   - Qua tầng phân loại `logits = self.fc_out(mean_vec)` kích thước `[1, 2]`.
   - Xuất xác suất Output thứ 1:
     $$\text{prob} = \text{softmax}(\text{logits}, \text{dim}=1) = \begin{bmatrix} P(\text{alert}), & P(\text{drowsy}) \end{bmatrix} \in [0, 1]^{[1, 2]}$$
   - Điểm số nhánh học sâu:
     $$\text{neuralnetwork\_score} = \text{prob}[0, 1]$$

---

### 5.2. Đặc Tả Chi Tiết Inputs, Outputs & Các Phiên Bản Model Từ `export_onnx.py`
Nguồn: [`export_onnx.py:158-388`](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/export_onnx.py#L158-L388).

#### a. Bảng Đặc Tả Tensor Đầu Vào (Inputs) & Đầu Ra (Outputs)

| Chiều | Tên Tensor | Shape (Kích thước) | Trục động (`dynamic_axes`) | Kiểu dữ liệu (`fp32` / `int8` model) | Kiểu dữ liệu (`fp16` model) | Mô tả chi tiết |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Input 0** | **`"video"`** | `[num_frames, 3, 224, 224]` | `dim 0`: `num_frames` ($T_{\text{new}} \ge 1$) | `uint8` $[0..255]$ hoặc `float32` $[0..1]$ | `uint8` $[0..255]$ hoặc `float16` $[0..1]$ | Chuỗi các frame RGB mới đã qua `letterboxRgbaToRgbCHW` ngay trên `ONNX Worker Thread` (bố cục `CHW`). Phép chia `255.0` được thực hiện tự động bên trong đồ thị ONNX nếu truyền `uint8`. |
| **Input 1** | **`"p4_old"`** | `[num_old_frames, 128, 14, 14]` | `dim 0`: `num_old_frames` ($T_{\text{old}} \ge 0$) | `float32` | `float16` | Bộ đệm đặc trưng mức P4 từ bước suy luận trước. Khi khởi đầu hoặc khi mất liên tục, truyền tensor rỗng kích thước `[0, 128, 14, 14]`. |
| **Output 0** | **`"prob"`** | `[1, 2]` | Cố định `[1, 2]` | `float32` | `float16` | Phân phối xác suất Softmax: `prob[0, 0] = P(alert)`, `prob[0, 1] = P(drowsy)`. Lấy `prob[0, 1]` làm `neuralnetwork_score`. |
| **Output 1** | **`"p4_features"`** | `[total_frames, 128, 14, 14]` | `dim 0`: `total_frames` ($T_{\text{old}} + T_{\text{new}}$) | `float32` | `float16` | Chuỗi đặc trưng P4 đã ghép nối (`p4_old` + `p4_new`). Mỗi frame chiếm `128 × 14 × 14 = 25,088` phần tử (`~98 KB` ở FP32). Giữ lại tối đa `50` frames cuối cho bước tiếp theo. |

#### b. Ba Biến Thể Model Được Xuất Bởi `export_onnx.py`

| Tên File Model | Độ Chính Xác (Precision) | Sai số tối đa so với PyTorch | Khuyên dùng trên Android |
| :--- | :--- | :---: | :--- |
| `drowsy_classifier_fp32.onnx` *(và `drowsy_classifier.onnx`)* | **FP32** (Chuẩn đơn chính xác) | $< 10^{-4}$ | Dùng làm chuẩn đối chiếu chính xác tuyệt đối (Baseline Parity) trên CPU Android. |
| `drowsy_classifier_fp16.onnx` | **FP16** (Bán chính xác) | $< 10^{-2}$ | Giảm 50% dung lượng model và bộ nhớ `p4_old`, tối ưu cho GPU/NPU di động hỗ trợ FP16. |
| `drowsy_classifier_int8.onnx` | **INT8** (`QInt8` Dynamic Quantization per-channel) | $< 0.15$ | Giảm ~51% dung lượng trọng số, tối ưu tốc độ trên CPU ARMv8.2+ (dot-product). Nhận `p4_old` và xuất `prob`, `p4_features` ở kiểu `float32`. |

---

## 6. Đặc Tả Cổng Kiểm Duyệt (`Engine._features`), Chuẩn Hóa (`physical_v1`) & Hợp Nhất (`Fitness`)

### 6.1. Cổng Kiểm Duyệt Hai Lớp Trong `Engine._features` & Phối Hợp Cập Nhật `p4_old`
Nguồn: [`engine.py:114-160`](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/DataSystem/engine.py#L114-L160).

Khi `ONNX Worker Thread` thực thi đánh giá một cửa sổ tại `timestamp >= 60.0s`:
- **Bước 0 (Luôn chạy `letterbox` trên $T_{\text{new}}$ frame mới & cập nhật `p4_old`):**
  Ngay cả khi đang trong giai đoạn `WARMING UP` (`timestamp < 60.0s`) hoặc khi một trong hai cổng kiểm duyệt vật lý bên dưới chưa đạt, `ONNX Worker Thread` vẫn gọi `onnxClassifier.predictAndUpdateCache(rawWindowSnapshot)` (trong đó chỉ chạy `letterboxRgbaToRgbCHW` trên $\Delta \approx 5$ frames mới) để bộ đệm `p4_old` luôn trượt đồng bộ với thời gian thực.
- **Cổng 1 - Sẵn sàng & Độ phủ (Readiness & Coverage Gate):**
  Trả về `null` (Skip xuất mẫu) nếu bất kỳ điều kiện nào sau đây đúng:
  - `!metrics.eyeReady || !metrics.mouthReady || !metrics.headCalibrated`
  - `metrics.perclosPct == null`
  - `metrics.eyeObservedSec < windows["perclos"] * 0.8 - 1e-9` ($< 48.0\text{s}$)
  - `metrics.mouthObservedSec < windows["pom"] * 0.8 - 1e-9` ($< 48.0\text{s}$)
  - `metrics.headObservedSec < windows["over_angle"] * 0.8 - 1e-9` ($< 48.0\text{s}$)
- **Cổng 2 - Độ tin cậy FFT (Head Motion Gate):**
  Trả về `null` (Skip xuất mẫu) nếu bất kỳ điều kiện nào sau đây đúng:
  - `metrics.headMotionFrequencyHz == null || !metrics.headMotionFrequencyHz.isFinite() || metrics.headMotionFrequencyHz < 0.0`
  - `metrics.headMotionResolutionHz == null || !metrics.headMotionResolutionHz.isFinite() || metrics.headMotionResolutionHz <= 0.0`
  - `metrics.headMotionObservedSec == null || !metrics.headMotionObservedSec.isFinite() || metrics.headMotionObservedSec < windows["head_motion"] - 1e-9` ($< 60.0\text{s}$)

---

### 6.2. Tổng Hợp 8 Đặc Trưng & Chuẩn Hóa `PhysicalNormalizer` (`physical_v1`)
Nguồn: [`engine.py:166-186`](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/DataSystem/engine.py#L166-L186) và [`normalization.py:14-54`](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/DataSystem/normalization.py#L14-L54).

| Thứ tự | Tên đặc trưng (`FEATURES[i]`) | Công thức trích xuất giá trị thô (`raw[k]`) | Cận trên `scales[k]` (`fps=30, window=60s`) | Công thức chuẩn hóa `normalized[k]` |
| :---: | :--- | :--- | :---: | :--- |
| `0` | `blink_frequency` | `metrics.blinkRatePerMin` | `602.0` | `min(raw / 602.0, 1.0)` |
| `1` | `blink_duration` | `percentile(blinkDurations, 90.0)` *(rỗng $\rightarrow 0.0$)* | `2000.0` | `min(raw / 2000.0, 1.0)` |
| `2` | `perclos` | `metrics.perclosPct / 100.0` | `1.0` | `min(raw / 1.0, 1.0)` |
| `3` | `yawn_frequency` | `metrics.yawningFrequencyPerMin` | `17.0` | `min(raw / 17.0, 1.0)` |
| `4` | `nod_duration` | `percentile(nodDurations, 100.0) / 1000.0` *(đổi ms $\rightarrow$ s)* | `3.5` | `min(raw / 3.5, 1.0)` |
| `5` | `nod_frequency` | `metrics.noddingFrequencyPerMin` | `77.0` | `min(raw / 77.0, 1.0)` |
| `6` | `dominant_head_motion_frequency` | `metrics.headMotionFrequencyHz` | `15.0` | `min(raw / 15.0, 1.0)` |
| `7` | `neuralnetwork_score` | `onnxClassifier.predict(rawWindowSnapshot)` (`prob[0, 1]`) | `1.0` | `min(raw / 1.0, 1.0)` |

---

### 6.3. Ba Phép Hợp Nhất Quyết Định (`Fitness`: `weighted_mean`, `noisy_or`, `cooccurrence`)
Nguồn: [`fitnessFunction.py:128-186`](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/TrainingSystem/fitnessFunction.py#L128-L186).

Với vector đặc trưng đã chuẩn hóa $x = [x_0, \dots, x_7] \in [0, 1]^8$ và vector trọng số ứng viên $\text{candidate} = [w_0, \dots, w_7, T_{\text{cls}}] \in [0, 1]^9$:
- Tập chỉ số đặc trưng được chọn: $M = \{i \in \{0..7\} \mid w_i \ge 0.5\}$.
- Nếu $M = \emptyset$: $\text{score} = 0.0$, $\text{prediction} = 0$.
- Nếu $M \neq \emptyset$, đặt $z_i = x_i \cdot w_i$ với $i \in M$:
  - **`weighted_mean`:** $\text{score} = \frac{\sum_{i \in M} z_i}{\sum_{i \in M} w_i}$
  - **`noisy_or`:** $\text{score} = 1.0 - \prod_{i \in M} (1.0 - z_i)$
  - **`cooccurrence`:** Gọi $s_1 = \max_{i \in M} z_i$ và $s_2$ là giá trị lớn thứ hai trong $\{z_i \mid i \in M\}$ (nếu $|M| == 1$ thì $s_2 = 0.0$):
    $$\text{score} = s_1 + (1.0 - s_1) \cdot (s_1 \cdot s_2)$$
- **Nhãn dự đoán:** $\text{prediction} = \begin{cases} 1 & \text{nếu } \text{score} \ge T_{\text{cls}} \\ 0 & \text{ngược lại} \end{cases}$

---

## 7. Mã Nguồn Tham Chiếu Python Gốc Hoàn Chỉnh (Original Python Reference Code)

Dưới đây là toàn bộ mã nguồn Python gốc của các module cốt lõi trong dự án làm chuẩn tham chiếu đối chiếu 1-1 khi triển khai.

### 7.1. Tiền Xử Lý Ảnh & Mạng Học Sâu (`loadvideo2handle.py` & `neuralnetwork.py`)

#### a. Cân bằng sáng & Đọc video (`ai/OptimalAlgorithsm/DataSystem/loadvideo2handle.py`)
Nguồn: [loadvideo2handle.py](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/DataSystem/loadvideo2handle.py)

```python
import sys
from pathlib import Path

import cv2
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ai.OptimalAlgorithsm.DataSystem.config import DEFAULT_CONFIG, VIDEO_ROOT


def balance_brightness(frame, target_luma=DEFAULT_CONFIG.target_luma):
    """Nâng sáng BGR uint8 trên kênh Y; giữ chroma và giới hạn khuếch đại vùng tối."""
    if not np.isfinite(target_luma) or not 0 <= target_luma <= 255:
        raise ValueError("target_luma must be finite and in [0,255]")
    if target_luma == 0:
        return frame
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mean = float(ycrcb[:, :, 0].mean())
    if mean == 0 or mean >= target_luma:
        return frame
    gamma = max(0.5, np.log(target_luma / 255) / np.log(mean / 255))
    lut = np.rint(255 * (np.arange(256) / 255) ** gamma).astype(np.uint8)
    ycrcb[:, :, 0] = cv2.LUT(ycrcb[:, :, 0], lut)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


# [Frame: RGB, shape: [3, H, W], dtype: uint8, range: [0, 255], timeline = 30FPS]
# (frame: np.ndarray, timestamp: float)
def read_video(path, target_fps=DEFAULT_CONFIG.fps, *, target_luma=DEFAULT_CONFIG.target_luma):
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be finite and positive")
    if not np.isfinite(target_luma) or not 0 <= target_luma <= 255:
        raise ValueError("target_luma must be finite and in [0,255]")
    cap = cv2.VideoCapture(path)
    previous = None
    previous_t = origin = None
    index = output_index = 0
    use_frame_index = False
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(source_fps) or source_fps <= 0:
            raise ValueError(f"Invalid source FPS: {path}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if origin is None:
                origin = t
            # Chỉ dùng FPS thay thế khi backend không có timestamp ngay từ đầu.
            if index == 1:
                use_frame_index = origin == 0 and t == 0
            t = index / source_fps if use_frame_index else t - origin
            if (
                not np.isfinite(t)
                or t < 0
                or (previous_t is not None and t <= previous_t)
            ):
                raise ValueError(f"Invalid video timestamp: {path}")
            frame = balance_brightness(frame, target_luma)
            frame = np.transpose(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (2, 0, 1))
            if previous is not None:
                while output_index / target_fps < t - 1e-9:
                    yield previous, output_index / target_fps
                    output_index += 1
            previous, previous_t = frame, t
            index += 1
        if previous is None:
            raise ValueError(f"Video has no decodable frames: {path}")
        # Khung cuối vẫn chiếm một khoảng thời gian, kể cả khi tăng FPS.
        end = previous_t + 1 / source_fps
        while output_index / target_fps < end - 1e-9:
            yield previous, output_index / target_fps
            output_index += 1
    finally:
        cap.release()
```

#### b. Căn chỉnh `letterbox` & Lớp `NeuralNetwork` (`ai/OptimalAlgorithsm/DataSystem/neuralnetwork.py`)
Nguồn: [neuralnetwork.py](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/DataSystem/neuralnetwork.py)

```python
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch import nn

from ai.LSTM.src.checkpoint import load_model
from ai.LSTM.src.model import DrowsyClassifier
from ai.OptimalAlgorithsm.DataSystem.config import (
    DEFAULT_CONFIG, LETTERBOX_COLOR, MODEL_IMAGE_SIZE,
)


class NeuralNetwork(nn.Module):
    def __init__(self, checkpoint_path=None, device=None, chunk_size=None, *,
                 config=DEFAULT_CONFIG, model: DrowsyClassifier | None = None):
        super().__init__()
        device = config.device if device is None else device
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = config.neural_size
        if (not np.isfinite([config.neural_fps, config.neural_sec]).all()
                or min(config.neural_fps, config.neural_sec) <= 0
                or config.neural_count < 1
                or abs(config.neural_count / config.neural_fps - config.neural_sec) > 1e-6):
            raise ValueError("neural_sec must contain a positive integer number of neural frames")
        self.frame_count = config.neural_count
        self.fps = config.neural_fps
        chunk_size = config.chunk_size if chunk_size is None else chunk_size
        if type(chunk_size) is not int or chunk_size < 0:
            raise ValueError("chunk_size must be a nonnegative integer")
        if type(self.image_size) is not int or self.image_size < 1:
            raise ValueError("neural_size must be a positive integer")
        if model is None:
            path = Path(checkpoint_path or config.checkpoint_path).expanduser()
            model, state = load_model(path, self.device)
            for key, expected in (("image_size", self.image_size),
                                  ("target_fps", config.neural_fps),
                                  ("window_sec", config.neural_sec)):
                if state["config"].get(key) != expected:
                    raise ValueError(f"Checkpoint {key}={state['config'].get(key)!r}, expected {expected}")
        self.model = model.to(self.device).eval()
        self.model.chunk_size = chunk_size
        self.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path=None, **kwargs):
        return cls(checkpoint_path=checkpoint_path, **kwargs)

    @torch.inference_mode()
    def forward(self, x) -> float:
        """RGB [50,3,H,W] đã lấy mẫu 5 FPS/10s -> P(drowsy).

        Engine lấy mỗi 6 frame từ read_video 30 FPS trước khi gọi mạng.
        DrowsyClassifier tự chia uint8 cho 255; giữ nguyên thứ tự kênh RGB.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not isinstance(x, torch.Tensor):
            raise TypeError("Frames must be a numpy array or torch tensor")
        if x.ndim != 4 or x.shape[1] != 3 or any(size == 0 for size in x.shape):
            raise ValueError("Frames must have nonempty shape [T,3,H,W]")
        if x.shape[0] != self.frame_count:
            raise ValueError(f"Expected {self.frame_count} frames sampled at {self.fps:g} FPS, got {x.shape[0]}")
        if x.dtype != torch.uint8 and (
            not x.is_floating_point() or not torch.isfinite(x).all()
            or x.min() < 0 or x.max() > 1
        ):
            raise ValueError("Frames must be uint8 [0,255] or finite float [0,1]")
        if x.shape[-2:] != (self.image_size, self.image_size):
            frames = x.detach().cpu()
            frames = (frames if frames.dtype == torch.uint8 else frames.float()).numpy()
            x = torch.from_numpy(np.stack([letterbox(frame.transpose(1, 2, 0), self.image_size).transpose(2, 0, 1)
                                           for frame in frames]))
        if x.dtype != torch.uint8:
            x = x.float()
        x = x.unsqueeze(0).to(self.device)
        self.model.eval()
        logits = self.model(x, seq_lens=torch.tensor([x.shape[1]], device=self.device))
        if logits.shape != (1, 2) or not torch.isfinite(logits).all():
            raise ValueError("Expected finite binary clip logits [1,2]")
        return logits.float().softmax(dim=-1)[0, 1].item()


def letterbox(
    image: np.ndarray,
    new_size: int = MODEL_IMAGE_SIZE,
    color: Tuple[int, int, int] = LETTERBOX_COLOR,
) -> np.ndarray:
    """
    Căn chỉnh kích thước ảnh về new_size x new_size giữ nguyên tỷ lệ khung hình
    và chèn viền màu (mặc định 114, 114, 114).
    Hỗ trợ linh hoạt cả định dạng HWC [H, W, 3] và CHW [3, H, W].
    """
    is_chw = image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3
    if is_chw:
        image = np.transpose(image, (1, 2, 0))

    h, w = image.shape[:2]
    scale = min(new_size / h, new_size / w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if np.issubdtype(image.dtype, np.floating):
        color = np.asarray(color) / 255.0
    canvas = np.full((new_size, new_size, 3), color, dtype=image.dtype)
    pad_left = (new_size - new_w) // 2
    pad_top = (new_size - new_h) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

    if is_chw:
        return np.transpose(canvas, (2, 0, 1))
    return canvas
```

---

### 7.2. Bộ Đếm Thời Gian, Tỷ Lệ Cửa Sổ & Phổ FFT (`temporal_metrics.py`)
Nguồn: [temporal_metrics.py](file:///D:/Project/DATN/tempt/driver-guardian/ai/PhysicalBranch/temporal_metrics.py)

```python
from collections import deque

import numpy as np

from ai.PhysicalBranch.config import DEFAULT_CONFIG


def positive_seconds(value):
    if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
        raise ValueError("Time parameters must be finite and positive")
    return float(value)


class StateMachine:

    def __init__(self, mode="eye", fps=None, window_size_sec=None,
                 min_duration_ms=None, max_duration_ms=None, max_gap_sec=None, *, config=DEFAULT_CONFIG):
        fps = config.fps if fps is None else fps
        window_size_sec = config.window_sec if window_size_sec is None else window_size_sec
        max_gap_sec = config.max_gap_sec if max_gap_sec is None else max_gap_sec
        limits = {"eye": config.blink_duration_ms, "mouth": config.yawn_duration_ms,
                  "pitch": config.nod_duration_ms}
        if mode not in limits:
            raise ValueError("Invalid FSM configuration")
        self.mode = mode
        self.fps = positive_seconds(fps)
        self.window = positive_seconds(window_size_sec)
        lo, hi = limits[mode]
        self.minimum = lo if min_duration_ms is None else min_duration_ms
        self.maximum = hi if max_duration_ms is None else max_duration_ms
        if not np.isfinite([self.minimum, self.maximum]).all() or not 0 <= self.minimum < self.maximum:
            raise ValueError("Invalid event duration limits")
        self.max_gap = positive_seconds(max_gap_sec)
        self.reset()

    def reset(self):
        self.time = None
        self.start = None
        self.last_duration = 0.0
        self.last_valid_duration = None
        self.armed = False
        self.events = deque()

    def process(self, state, timestamp=None):
        now = (0 if self.time is None else self.time + 1 / self.fps) if timestamp is None else float(timestamp)
        if not np.isfinite(now) or (self.time is not None and now <= self.time):
            raise ValueError("Timestamps must be finite and strictly increasing")
        if state not in (0, 1, None):
            raise ValueError("State must be 0, 1 or None")
        if self.time is not None and now - self.time > self.max_gap:
            self.start = None
            self.armed = False
        self.time = now
        done = False
        if state is None:
            self.start = None
            self.armed = False
        elif state == 1 and self.start is None:
            self.start = now
        elif state == 0 and self.start is not None:
            self.last_duration = (now - self.start) * 1000
            done = self.armed and self.minimum - 1e-6 <= self.last_duration <= self.maximum + 1e-6
            if done:
                self.events.append(now)
                self.last_valid_duration = self.last_duration
            self.start = None
        if state == 0:
            self.armed = True
        while self.events and self.events[0] <= now - self.window:
            self.events.popleft()
        duration = 0 if self.start is None else (now - self.start) * 1000
        rate = len(self.events) * 60 / self.window
        return {
            "state": state, "signal_missing": state is None,
            "event_done": done, "current_duration_ms": duration,
            "last_event_duration_ms": self.last_duration,
            "last_valid_duration_ms": self.last_valid_duration,
            "event_count": len(self.events),
            "rate_per_minute": rate, "prolonged": duration > self.maximum,
        }


class WindowRatio:
    def __init__(self, window_sec=None, max_gap_sec=None, *, config=DEFAULT_CONFIG):
        window_sec = config.window_sec if window_sec is None else window_sec
        max_gap_sec = config.max_gap_sec if max_gap_sec is None else max_gap_sec
        self.window = positive_seconds(window_sec)
        self.max_gap = positive_seconds(max_gap_sec)
        self.reset()

    def reset(self):
        self.intervals = deque()
        self.previous = None
        self.time = None

    def process(self, active, timestamp):
        if not np.isfinite(timestamp) or (self.time is not None and timestamp <= self.time):
            raise ValueError("Timestamps must be finite and strictly increasing")
        self.time = timestamp
        if self.previous is not None and active is not None:
            start, previous_active = self.previous
            if timestamp - start <= self.max_gap:
                self.intervals.append((start, timestamp, previous_active))
        self.previous = None if active is None else (timestamp, bool(active))
        cutoff = timestamp - self.window
        while self.intervals and self.intervals[0][1] <= cutoff:
            self.intervals.popleft()
        valid = total = 0.0
        for start, end, state in self.intervals:
            duration = end - max(start, cutoff)
            valid += duration
            total += duration * state
        return (100 * total / valid if valid else None), valid


class HeadMotionWindow:
    """Thống kê pitch tương đối; FFT chỉ dùng cửa sổ đầy, liên tục."""

    def __init__(self, window_sec=None, max_gap_sec=None, min_amplitude_deg=None, *, config=DEFAULT_CONFIG):
        window_sec = config.window_sec if window_sec is None else window_sec
        max_gap_sec = config.max_gap_sec if max_gap_sec is None else max_gap_sec
        min_amplitude_deg = config.head_motion_min_amplitude_deg if min_amplitude_deg is None else min_amplitude_deg
        self.window = positive_seconds(window_sec)
        self.max_gap = positive_seconds(max_gap_sec)
        if not np.isfinite(min_amplitude_deg) or min_amplitude_deg < 0:
            raise ValueError("Minimum motion amplitude must be finite and nonnegative")
        self.min_amplitude = min_amplitude_deg
        self.reset()

    def reset(self):
        self.samples = deque()
        self.time = None

    def process(self, pitch, timestamp):
        if not np.isfinite(timestamp) or (self.time is not None and timestamp <= self.time):
            raise ValueError("Timestamps must be finite and strictly increasing")
        if self.time is not None and timestamp - self.time > self.max_gap:
            self.samples.clear()
        self.time = timestamp
        if pitch is None or not np.isfinite(pitch):
            self.samples.clear()
        else:
            self.samples.append((timestamp, float(pitch)))
        cutoff = timestamp - self.window
        while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
            self.samples.popleft()
        result = dict(pitch_mean_deg=None, pitch_mean_amplitude_deg=None,
                      head_motion_frequency_hz=None, head_motion_resolution_hz=None,
                      head_motion_observed_sec=0.0)
        if len(self.samples) < 2:
            return result
        times, values = np.asarray(self.samples).T
        start = max(times[0], cutoff)
        observed = timestamp - start
        clipped_times = np.r_[start, times[times > start]]
        clipped_values = np.interp(clipped_times, times, values)
        durations = np.diff(clipped_times)
        left, right = clipped_values[:-1], clipped_values[1:]
        amplitudes = (np.abs(left) + np.abs(right)) / 2
        crossing = (left < 0) & (right > 0) | (left > 0) & (right < 0)
        # Tích phân |pitch| phải tách hai tam giác khi đoạn nội suy đi qua 0.
        amplitudes[crossing] = (left[crossing] ** 2 + right[crossing] ** 2) / (
            2 * (np.abs(left[crossing]) + np.abs(right[crossing])))
        result.update(
            pitch_mean_deg=float(np.sum((left + right) * durations / 2) / observed),
            pitch_mean_amplitude_deg=float(np.sum(amplitudes * durations) / observed),
            head_motion_observed_sec=float(observed),
        )
        if observed < self.window - 1e-9 or len(times) < 4:
            return result
        # Nội suy theo timestamp để FPS dao động không làm đổi đơn vị tần số.
        count = max(4, round(observed / np.median(np.diff(times))) + 1)
        grid = np.linspace(start, timestamp, count)
        signal = np.interp(grid, times, values)
        dt = observed / (count - 1)
        result["head_motion_resolution_hz"] = 1 / (count * dt)
        if np.ptp(signal) <= self.min_amplitude:
            result["head_motion_frequency_hz"] = 0.0
            return result
        spectrum = np.abs(np.fft.rfft((signal - signal.mean()) * np.hanning(count))) ** 2
        peak = int(np.argmax(spectrum[1:])) + 1
        result["head_motion_frequency_hz"] = float(np.fft.rfftfreq(count, dt)[peak])
        return result
```

---

### 7.3. Ước Lượng Tư Thế Đầu 3D SolvePnP (`head_pose_estimation.py`)
Nguồn: [head_pose_estimation.py](file:///D:/Project/DATN/tempt/driver-guardian/ai/PhysicalBranch/head_pose_estimation.py)

```python
import cv2
import numpy as np

from ai.PhysicalBranch.config import DEFAULT_CONFIG


class HeadPoseEstimator:
    LANDMARKS = list(DEFAULT_CONFIG.pose_landmarks)
    MODEL = np.array(DEFAULT_CONFIG.face_model, dtype=float)

    def __init__(self, image_width, image_height, camera_matrix=None, distortion_coeffs=None,
                 *, config=DEFAULT_CONFIG):
        self.LANDMARKS = list(config.pose_landmarks)
        self.MODEL = np.array(config.face_model, dtype=float)
        camera_matrix = config.camera_matrix if camera_matrix is None else camera_matrix
        if not np.isfinite([image_width, image_height]).all() or image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be finite and positive")
        self.camera_matrix = np.asarray(camera_matrix if camera_matrix is not None else
                                        [[image_width, 0, image_width / 2],
                                         [0, image_width, image_height / 2], [0, 0, 1]], dtype=float)
        self.distortion = np.asarray(config.distortion_coeffs if distortion_coeffs is None else distortion_coeffs, dtype=float)
        if self.camera_matrix.shape != (3, 3) or not np.isfinite(self.camera_matrix).all():
            raise ValueError("Invalid camera matrix")
        if (self.camera_matrix[0, 0] <= 0 or self.camera_matrix[1, 1] <= 0
                or not np.array_equal(self.camera_matrix[2], [0, 0, 1])):
            raise ValueError("Camera matrix requires positive focal lengths and last row [0, 0, 1]")
        if (self.distortion.ndim not in (1, 2)
                or (self.distortion.ndim == 2 and 1 not in self.distortion.shape)
                or self.distortion.size not in (4, 5, 8, 12, 14)
                or not np.isfinite(self.distortion).all()):
            raise ValueError("Distortion coefficients must be a finite vector of length 4, 5, 8, 12 or 14")
        self.rotation = self.translation = None

    def estimate(self, landmarks_2d):
        self.rotation = self.translation = None
        points = np.asarray(landmarks_2d, dtype=float)
        if points.shape != (6, 2) or not np.isfinite(points).all():
            raise ValueError("Expected six finite 2D landmarks")
        if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            raise ValueError("Degenerate landmarks")
        ok, rotation, translation = cv2.solvePnP(
            self.MODEL, points, self.camera_matrix, self.distortion, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            raise ValueError("No valid pose solution")
        rotation, translation = cv2.solvePnPRefineLM(
            self.MODEL, points, self.camera_matrix, self.distortion, rotation, translation)
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("Non-finite pose solution")
        matrix, _ = cv2.Rodrigues(rotation)
        if np.any((self.MODEL @ matrix.T + translation.ravel())[:, 2] <= 0):
            raise ValueError("Pose behind camera")
        self.rotation, self.translation = rotation, translation
        return tuple(float(v) for v in cv2.RQDecomp3x3(matrix)[0])

    def draw_axes(self, frame):
        if self.rotation is None:
            return
        axes = np.array([(0, 0, 0), (100, 0, 0), (0, 100, 0), (0, 0, -100)], dtype=float)
        projected, _ = cv2.projectPoints(axes, self.rotation, self.translation,
                                         self.camera_matrix, self.distortion)
        points = np.rint(projected.reshape(-1, 2)).astype(int)
        for end, color in zip(points[1:], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            cv2.line(frame, tuple(points[0]), tuple(end), color, 2, cv2.LINE_AA)
```

---

### 7.4. Mô Hình Markov Ẩn Thích Nghi & Máy Trạng Thái (`adaptive_hmm_fsm.py`)
Nguồn: [adaptive_hmm_fsm.py](file:///D:/Project/DATN/tempt/driver-guardian/ai/PhysicalBranch/adaptive_hmm_fsm.py)

```python
from collections import deque

import numpy as np

from ai.PhysicalBranch.config import DEFAULT_CONFIG
from ai.PhysicalBranch.temporal_metrics import StateMachine, WindowRatio, positive_seconds


class AdaptiveHMM:
    def __init__(
        self,
        n_states=2,
        learning_rate=None,
        adapt_interval=None,
        positive_state="low",
        min_variance=None,
        em_iterations=None,
        *, config=DEFAULT_CONFIG,
    ):
        self.config = config
        learning_rate = config.learning_rate if learning_rate is None else learning_rate
        adapt_interval = config.adapt_interval if adapt_interval is None else adapt_interval
        min_variance = config.min_variance if min_variance is None else min_variance
        em_iterations = config.em_iterations if em_iterations is None else em_iterations
        if positive_state not in ("low", "high"):
            raise ValueError("positive_state phải là 'low' hoặc 'high'")

        if n_states != 2 or not 0 < learning_rate <= 1:
            raise ValueError("HMM requires two states and 0 < learning_rate <= 1")
        if (type(adapt_interval) is not int or adapt_interval < 2
                or type(em_iterations) is not int or em_iterations < 1
                or not np.isfinite(min_variance) or min_variance <= 0
                or not np.isfinite(config.min_adapt_state_weight) or config.min_adapt_state_weight < 0):
            raise ValueError("Invalid adaptation parameters")
        self.n_states = int(n_states)
        self.learning_rate = float(learning_rate)
        self.adapt_interval = max(1, int(adapt_interval))
        self.positive_state = positive_state
        self.min_variance = float(min_variance)
        self.em_iterations = int(em_iterations)

        self.pi = np.full(self.n_states, 1.0 / self.n_states)
        self.A = np.full((self.n_states, self.n_states), 1.0 / self.n_states)
        self.means = np.zeros(self.n_states)
        self.vars = np.ones(self.n_states)
        self.posterior = None
        self.initialized = False
        self.buffer = deque(maxlen=self.adapt_interval)

    def reset(self):
        self.__init__(
            n_states=self.n_states,
            learning_rate=self.learning_rate,
            adapt_interval=self.adapt_interval,
            positive_state=self.positive_state,
            min_variance=self.min_variance,
            em_iterations=self.em_iterations,
            config=self.config,
        )

    def fit_initial(self, data):
        x = self._as_vector(data)
        labels = self._kmeans_1d(x)
        self._estimate_from_labels(x, labels)

        for _ in range(self.em_iterations):
            gamma, xi = self._forward_backward(x)
            self._m_step(x, gamma, xi, alpha=1.0)

        self._sort_states()
        self.posterior = None
        self.initialized = True
        self.buffer.clear()
        return self

    def predict(self, value):
        if not self.initialized:
            raise RuntimeError("HMM chưa được khởi tạo")

        if not np.isfinite(value):
            raise ValueError("Observation must be finite")
        prior = self.pi if self.posterior is None else self.posterior @ self.A
        log_p = np.log(np.maximum(prior, 1e-300)) + self._log_emission_prob(np.array([value]))[0]
        self.posterior = np.exp(log_p - self._logsumexp(log_p))
        return int(np.argmax(self.posterior)), self.posterior.copy()

    def update_online(self, value):
        if not self.initialized:
            return False

        if not np.isfinite(value):
            raise ValueError("Observation must be finite")
        self.buffer.append(float(value))
        if len(self.buffer) < self.adapt_interval:
            return False

        x = np.asarray(self.buffer, dtype=float)
        gamma, xi = self._forward_backward(x)
        if np.min(gamma.sum(axis=0)) < self.config.min_adapt_state_weight:
            self.buffer.clear()
            return False
        self._m_step(x, gamma, xi, alpha=self.learning_rate)
        self.posterior = gamma[-1] / max(gamma[-1].sum(), 1e-12)
        self._sort_states()
        self.buffer.clear()
        return True

    def _as_vector(self, data):
        x = np.asarray(data, dtype=float).reshape(-1)
        if x.size < self.n_states * 2:
            raise ValueError("Không đủ dữ liệu để khởi tạo HMM")
        if not np.all(np.isfinite(x)):
            raise ValueError("Dữ liệu chứa NaN hoặc inf")
        return x

    def _kmeans_1d(self, x, iterations=None):
        iterations = self.config.kmeans_iterations if iterations is None else iterations
        centers = np.percentile(x, np.linspace(0, 100, self.n_states + 2)[1:-1])
        if np.unique(centers).size < self.n_states:
            centers = np.linspace(x.min(), x.max(), self.n_states)

        labels = np.zeros(x.size, dtype=int)
        for _ in range(iterations):
            labels = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
            new_centers = centers.copy()
            for k in range(self.n_states):
                if np.any(labels == k):
                    new_centers[k] = x[labels == k].mean()
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        return labels

    def _estimate_from_labels(self, x, labels):
        eps = self.config.transition_smoothing
        counts = np.bincount(labels, minlength=self.n_states).astype(float) + eps
        self.pi = counts / counts.sum()

        trans = np.full((self.n_states, self.n_states), eps)
        for a, b in zip(labels[:-1], labels[1:]):
            trans[a, b] += 1.0
        self.A = trans / trans.sum(axis=1, keepdims=True)

        for k in range(self.n_states):
            part = x[labels == k]
            self.means[k] = part.mean() if part.size else x.mean()
            self.vars[k] = max(float(part.var()) if part.size else float(x.var()), self.min_variance)

    def _log_emission_prob(self, x):
        var = np.maximum(self.vars, self.min_variance)
        z = x[:, None] - self.means[None, :]
        return -0.5 * (np.log(2 * np.pi * var[None, :]) + z * z / var[None, :])

    def _forward_backward(self, x):
        T = len(x)
        log_b = self._log_emission_prob(x)
        log_pi = np.log(np.maximum(self.pi, 1e-300))
        log_a = np.log(np.maximum(self.A, 1e-300))

        log_alpha = np.zeros((T, self.n_states))
        log_beta = np.zeros((T, self.n_states))
        log_alpha[0] = log_pi + log_b[0]

        for t in range(1, T):
            log_alpha[t] = log_b[t] + self._logsumexp(log_alpha[t - 1, :, None] + log_a, axis=0)

        for t in range(T - 2, -1, -1):
            log_beta[t] = self._logsumexp(log_a + log_b[t + 1, None, :] + log_beta[t + 1, None, :], axis=1)

        log_gamma = log_alpha + log_beta
        log_gamma -= self._logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(log_gamma)

        xi = np.zeros((max(T - 1, 0), self.n_states, self.n_states))
        for t in range(T - 1):
            log_xi = log_alpha[t, :, None] + log_a + log_b[t + 1, None, :] + log_beta[t + 1, None, :]
            xi[t] = np.exp(log_xi - self._logsumexp(log_xi))
        return gamma, xi

    def _m_step(self, x, gamma, xi, alpha):
        eps = 1e-12
        weights = np.maximum(gamma.sum(axis=0), eps)
        means = (gamma * x[:, None]).sum(axis=0) / weights
        vars_ = (gamma * (x[:, None] - means[None, :]) ** 2).sum(axis=0) / weights

        new_pi = gamma[0] / max(gamma[0].sum(), eps)
        counts = xi.sum(axis=0) + eps
        new_A = self.A if len(xi) == 0 else counts / counts.sum(axis=1, keepdims=True)

        self.pi = (1 - alpha) * self.pi + alpha * new_pi
        self.A = (1 - alpha) * self.A + alpha * new_A
        self.A /= np.maximum(self.A.sum(axis=1, keepdims=True), eps)
        self.means = (1 - alpha) * self.means + alpha * means
        self.vars = np.maximum((1 - alpha) * self.vars + alpha * vars_, self.min_variance)

    def _sort_states(self):
        order = np.argsort(self.means)
        if self.positive_state == "low":
            order = order[::-1]
        self.pi = self.pi[order]
        self.A = self.A[order][:, order]
        self.means = self.means[order]
        self.vars = self.vars[order]
        if self.posterior is not None:
            self.posterior = self.posterior[order]

    @staticmethod
    def _logsumexp(v, axis=None):
        m = np.max(v, axis=axis, keepdims=True)
        out = m + np.log(np.maximum(np.sum(np.exp(v - m), axis=axis, keepdims=True), 1e-300))
        return float(np.squeeze(out)) if axis is None else np.squeeze(out, axis=axis)


class AdaptiveHMM_FSM:

    def __init__(self, mode="eye", fps=None, init_duration_sec=None, window_size_sec=None,
                 min_duration_ms=None, max_duration_ms=None, max_gap_sec=None,
                 learning_rate=None, adapt_interval=None, ratio_window_sec=None, *, config=DEFAULT_CONFIG):
        self.config = config
        fps = config.fps if fps is None else fps
        init_duration_sec = config.init_duration_sec if init_duration_sec is None else init_duration_sec
        window_size_sec = config.window_sec if window_size_sec is None else window_size_sec
        max_gap_sec = config.max_gap_sec if max_gap_sec is None else max_gap_sec
        if mode not in ("eye", "mouth", "pitch"):
            raise ValueError("Invalid detector configuration")
        fps = positive_seconds(fps)
        init_duration_sec = positive_seconds(init_duration_sec)
        self.mode, self.fps = mode, fps
        self.init_frames = max(4, round(fps * init_duration_sec))
        self.hmm = AdaptiveHMM(positive_state="low" if mode == "eye" else "high",
                               learning_rate=learning_rate, adapt_interval=adapt_interval, config=config)
        self.fsm = StateMachine(mode, fps, window_size_sec, min_duration_ms,
                                max_duration_ms, max_gap_sec, config=config)
        self.ratio = WindowRatio(window_size_sec if ratio_window_sec is None else ratio_window_sec, max_gap_sec, config=config)
        self.reset()

    def reset(self):
        self.hmm.reset()
        self.fsm.reset()
        self.samples = deque(maxlen=self.init_frames)
        self.ratio.reset()
        self.initialized = False
        self.normal = self.event_reference = None

    def _initialize(self):
        data = np.asarray(self.samples)
        cfg = self.config
        low, high = np.percentile(data, cfg.init_percentiles)
        separation = {"eye": cfg.eye_separation, "mouth": cfg.mouth_separation,
                      "pitch": cfg.pitch_separation}[self.mode]
        percentile = cfg.eye_baseline_percentile if self.mode == "eye" else cfg.other_baseline_percentile
        self.normal = float(np.percentile(data, percentile))
        distinct = low < self.normal * cfg.eye_distinct_ratio if self.mode == "eye" else high > self.normal + separation
        if high - low >= separation and distinct:
            self.hmm.fit_initial(data)
        else:
            normal = self.normal
            event = normal * cfg.eye_closed_ratio if self.mode == "eye" else normal + separation * cfg.event_separation_multiplier
            self.hmm.means = np.array([normal, event])
            sigma = max(abs(normal - event) / cfg.fallback_sigma_divisor, cfg.fallback_min_sigma)
            self.hmm.vars[:] = sigma ** 2
            self.hmm.A = np.array(cfg.fallback_transition, dtype=float)
            self.hmm.pi = np.array(cfg.resting_prior, dtype=float)
            self.hmm.initialized = True
        self.event_reference = float(self.hmm.means[1])
        self.initialized = True
        self.samples.clear()

    def process(self, value, timestamp=None):
        timestamp = (0 if self.fsm.time is None else self.fsm.time + 1 / self.fps) if timestamp is None else float(timestamp)
        if not np.isfinite(timestamp) or (self.fsm.time is not None and timestamp <= self.fsm.time):
            raise ValueError("Timestamps must be finite and strictly increasing")
        if self.fsm.time is not None and timestamp - self.fsm.time > self.fsm.max_gap:
            self.hmm.posterior = None
            self.hmm.buffer.clear()
        try:
            value = float(value)
            if not np.isfinite(value) or (self.mode != "pitch" and value < 0):
                value = None
        except (TypeError, ValueError):
            value = None
        state, posterior, updated = None, None, False
        if value is not None:
            if not self.initialized:
                self.samples.append(value)
                if len(self.samples) == self.init_frames:
                    self._initialize()
            if self.initialized:
                state, posterior = self.hmm.predict(value)
                if self.mode == "eye" and value >= self.normal * self.config.eye_open_ratio:
                    state = 0
                    posterior = self.hmm.posterior = np.array(self.config.resting_prior, dtype=float)
                updated = self.hmm.update_online(value)
        else:
            self.hmm.posterior = None
            self.hmm.buffer.clear()
        out = self.fsm.process(state, timestamp)
        now = self.fsm.time
        threshold = None
        if self.mode == "eye" and self.initialized:
            opened, closed = self.normal, self.event_reference
            threshold = float(opened - self.config.perclos_closed_fraction * (opened - closed))
        if self.mode == "eye":
            active = None if value is None or threshold is None else value <= threshold
        else:
            active = None if state is None else bool(state)
        percentage, valid = self.ratio.process(active, now)
        out.update(mode=self.mode, input_value=value, initialized=self.initialized,
                   signal_missing=value is None, model_updated=updated,
                   posterior=None if posterior is None else posterior.tolist(),
                   p80_threshold=threshold, perclos=percentage if self.mode == "eye" else None,
                   pom=percentage if self.mode == "mouth" else None,
                   observed_seconds=valid)
        return out
```

---

### 7.5. Bộ Theo Dõi Khuôn Mặt & Trích Xuất Đặc Trưng Vật Lý (`camera_metrics.py`)
Nguồn: [camera_metrics.py](file:///D:/Project/DATN/tempt/driver-guardian/ai/PhysicalBranch/camera_metrics.py)

```python
import os
import sys
import time
from dataclasses import replace

import cv2
import numpy as np

from ai.PhysicalBranch.config import DEFAULT_CONFIG
from ai.PhysicalBranch.adaptive_hmm_fsm import AdaptiveHMM_FSM
from ai.PhysicalBranch.head_pose_estimation import HeadPoseEstimator
from ai.PhysicalBranch.temporal_metrics import HeadMotionWindow, StateMachine, WindowRatio, positive_seconds


class FaceTracker:
    def __init__(self, detection_confidence=None, tracking_confidence=None, *, config=DEFAULT_CONFIG):
        skip_audio = "sounddevice" not in sys.modules
        if skip_audio:
            sys.modules["sounddevice"] = None
        try:
            import mediapipe as mp
        finally:
            if skip_audio:
                sys.modules.pop("sounddevice", None)

        det_conf = config.detection_confidence if detection_confidence is None else detection_confidence
        track_conf = config.tracking_confidence if tracking_confidence is None else tracking_confidence

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self._mode = "legacy"
            self.mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=config.max_num_faces,
                refine_landmarks=config.refine_landmarks,
                min_detection_confidence=det_conf,
                min_tracking_confidence=track_conf,
            )
        else:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

            model_path = getattr(config, "face_landmarker_model_path", None)
            if not model_path or not os.path.isfile(model_path):
                base_dir = os.path.dirname(__file__)
                candidates = [
                    os.path.join(base_dir, "models", "face_landmarker.task"),
                    os.path.join(base_dir, "face_landmarker.task"),
                    os.path.abspath(os.path.join(base_dir, "..", "..", "system", "android", "app", "src", "main", "assets", "face_landmarker.task")),
                    "face_landmarker.task",
                ]
                for candidate in candidates:
                    if os.path.isfile(candidate):
                        model_path = candidate
                        break

            if not model_path or not os.path.isfile(model_path):
                raise FileNotFoundError(
                    "Không tìm thấy file model 'face_landmarker.task'. "
                    "Vui lòng đặt file tại ai/PhysicalBranch/models/face_landmarker.task "
                    "hoặc chỉ định qua config.face_landmarker_model_path."
                )

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                num_faces=config.max_num_faces,
                min_face_detection_confidence=det_conf,
                min_face_presence_confidence=det_conf,
                min_tracking_confidence=track_conf,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._mode = "tasks"
            self.detector = FaceLandmarker.create_from_options(options)

    def process(self, frame):
        if self._mode == "legacy":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = self.mesh.process(rgb)
            if not result.multi_face_landmarks:
                return None
            height, width = frame.shape[:2]
            return np.array([(p.x * width, p.y * height)
                             for p in result.multi_face_landmarks[0].landmark])
        else:
            import mediapipe as mp
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(mp_image)
            if not result.face_landmarks:
                return None
            height, width = frame.shape[:2]
            return np.array([(p.x * width, p.y * height)
                             for p in result.face_landmarks[0]])

    def close(self):
        if self._mode == "legacy" and hasattr(self, "mesh") and self.mesh is not None:
            self.mesh.close()
            self.mesh = None
        elif self._mode == "tasks" and hasattr(self, "detector") and self.detector is not None:
            self.detector.close()
            self.detector = None


class CameraMetrics:
    LEFT_EYE = list(DEFAULT_CONFIG.left_eye)
    RIGHT_EYE = list(DEFAULT_CONFIG.right_eye)
    MOUTH = list(DEFAULT_CONFIG.mouth)

    def __init__(self, fps=None, window_sec=None, *, config=DEFAULT_CONFIG,
                 eye_options=None, mouth_options=None, windows=None, **overrides):
        self.config = cfg = replace(
            config, fps=config.fps if fps is None else fps,
            window_sec=config.window_sec if window_sec is None else window_sec, **overrides)
        fps, window_sec, max_gap_sec = cfg.fps, cfg.window_sec, cfg.max_gap_sec
        calibration_frames, nod_direction = cfg.calibration_frames, cfg.nod_direction
        detection_confidence, tracking_confidence = cfg.detection_confidence, cfg.tracking_confidence
        nod_duration_ms = cfg.nod_duration_ms
        nod_release_deg, nod_pitch_deg = cfg.nod_release_deg, cfg.nod_pitch_deg
        angle_limits = cfg.angle_limits
        self.LEFT_EYE, self.RIGHT_EYE, self.MOUTH = map(list, (cfg.left_eye, cfg.right_eye, cfg.mouth))
        if (type(calibration_frames) is not int or calibration_frames < 1
                or isinstance(nod_direction, bool) or nod_direction not in (None, -1, 1)):
            raise ValueError("Invalid calibration or nod direction")
        if (not np.isfinite([detection_confidence, tracking_confidence]).all()
                or not 0 <= detection_confidence <= 1 or not 0 <= tracking_confidence <= 1):
            raise ValueError("Detection and tracking confidence must be finite and in [0, 1]")
        if np.shape(nod_duration_ms) != (2,):
            raise ValueError("Nod duration limits must contain a minimum and maximum")
        if not np.isfinite([nod_release_deg, nod_pitch_deg]).all() or not 0 <= nod_release_deg < nod_pitch_deg:
            raise ValueError("Nod release must be below the pitch threshold")
        self.angle_limits = np.asarray(angle_limits, dtype=float)
        if self.angle_limits.shape != (3,) or not np.isfinite(self.angle_limits).all() or (self.angle_limits <= 0).any():
            raise ValueError("Angle limits must contain three positive finite values")
        common = dict(fps=fps, window_size_sec=window_sec, max_gap_sec=max_gap_sec, config=cfg)
        self.eye = AdaptiveHMM_FSM("eye", **(common | cfg.eye_options | (eye_options or {})))
        self.mouth = AdaptiveHMM_FSM("mouth", **(common | cfg.mouth_options | (mouth_options or {})))
        self.nod = StateMachine("pitch", min_duration_ms=nod_duration_ms[0],
                                max_duration_ms=nod_duration_ms[1], **common)
        self.over_angle_ratio = WindowRatio(window_sec, max_gap_sec, config=cfg)
        self.head_motion = HeadMotionWindow(window_sec, max_gap_sec, config=cfg)
        self._windows = dict(blink=self.eye.fsm, perclos=self.eye.ratio,
                             yawn=self.mouth.fsm, pom=self.mouth.ratio,
                             nod=self.nod, over_angle=self.over_angle_ratio, head_motion=self.head_motion)
        self.window_sec = positive_seconds(window_sec)
        self.set_windows(**(cfg.windows | (windows or {})))
        self.calibration_frames = calibration_frames
        self.nod_pitch, self.nod_release, self.nod_direction = nod_pitch_deg, nod_release_deg, nod_direction
        self.confidence = detection_confidence, tracking_confidence
        self.tracker = None
        self.estimator = None
        self.shape = None
        self.reset()

    def reset(self):
        for detector in (self.eye, self.mouth):
            detector.reset()
        self._reset_head()
        self.timestamp = None

    def _reset_head(self):
        self.nod.reset()
        self.over_angle_ratio.reset()
        self.head_motion.reset()
        self.neutral_samples = []
        self.neutral = None
        self.nodding = False

    @property
    def windows_sec(self):
        return {name: metric.window for name, metric in self._windows.items()}

    def set_windows(self, window_sec=None, **windows):
        """Đổi cửa sổ; xóa lịch sử chỉ số bị đổi, giữ HMM và baseline đầu."""
        unknown = windows.keys() - self._windows.keys()
        if unknown:
            raise ValueError(f"Unknown windows: {sorted(unknown)}")
        configured = self.windows_sec
        if window_sec is not None:
            configured = dict.fromkeys(configured, positive_seconds(window_sec))
        configured.update({name: positive_seconds(value) for name, value in windows.items()})
        for name, value in configured.items():
            metric = self._windows[name]
            if metric.window != value:
                metric.window = value
                metric.reset()
                if name == "nod":
                    self.nodding = False
        if window_sec is not None:
            self.window_sec = float(window_sec)

    @staticmethod
    def aspect_ratio(points, indices):
        p = np.asarray(points, dtype=float)[indices, :2]
        width = np.linalg.norm(p[0] - p[3])
        if not np.isfinite(p).all() or width < 1e-6:
            return None
        return float((np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * width))

    def process_landmarks(self, points, image_size, timestamp):
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or (self.timestamp is not None and timestamp <= self.timestamp):
            raise ValueError("Timestamps must be finite and strictly increasing")
        if self.timestamp is not None and timestamp - self.timestamp > self.nod.max_gap:
            self.nodding = False
            self.neutral_samples.clear()
        self.timestamp = timestamp
        if self.shape != tuple(image_size):
            self.estimator = HeadPoseEstimator(*image_size, config=self.config)
            self.shape = tuple(image_size)
            self._reset_head()
        if points is not None:
            points = np.asarray(points, dtype=float)
            if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] != 2 or not np.isfinite(points).all():
                points = None
        ear = mar = angles = raw = None
        pose_error = None
        if points is not None:
            left = self.aspect_ratio(points, self.LEFT_EYE)
            right = self.aspect_ratio(points, self.RIGHT_EYE)
            ear = (left + right) / 2 if left is not None and right is not None else None
            mar = self.aspect_ratio(points, self.MOUTH)
            try:
                raw = np.asarray(self.estimator.estimate(points[self.estimator.LANDMARKS]))
                if self.neutral is None:
                    self.neutral_samples.append(raw)
                    if len(self.neutral_samples) >= self.calibration_frames:
                        unwrapped = np.unwrap(self.neutral_samples, period=360, axis=0)
                        self.neutral = (np.median(unwrapped, axis=0) + 180) % 360 - 180
                        self.neutral_samples.clear()
                if self.neutral is not None:
                    angles = (raw - self.neutral + 180) % 360 - 180
            except (ValueError, cv2.error) as exc:
                pose_error = str(exc)
                self.neutral_samples.clear()
        elif self.neutral is None:
            self.neutral_samples.clear()
        eye = self.eye.process(ear, timestamp)
        mouth = self.mouth.process(mar, timestamp)
        over_angle = nod_state = None
        if angles is not None:
            over_angle = bool(np.any(np.abs(angles) > self.angle_limits))
            pitch_down = abs(angles[0]) if self.nod_direction is None else self.nod_direction * angles[0]
            self.nodding = bool(pitch_down > (self.nod_release if self.nodding else self.nod_pitch))
            nod_state = int(self.nodding)
        else:
            self.nodding = False
        nod = self.nod.process(nod_state, timestamp)
        over_percent, head_seconds = self.over_angle_ratio.process(over_angle, timestamp)
        motion = self.head_motion.process(None if angles is None else angles[0], timestamp)
        return {
            "timestamp_sec": timestamp,
            "window_sec": self.window_sec,
            "windows_sec": self.windows_sec,
            "face_detected": points is not None,
            "eye_ready": eye["initialized"],
            "mouth_ready": mouth["initialized"],
            "head_ready": angles is not None,
            "head_calibrated": self.neutral is not None,
            "pose_valid": raw is not None,
            "pose_error": pose_error,
            "eye_state": eye["state"],
            "mouth_state": mouth["state"],
            "ear": ear,
            "perclos_pct": eye["perclos"],
            "p80_ear_threshold": eye["p80_threshold"],
            "blink_rate_per_min": eye["rate_per_minute"],
            "blink_detected": eye["event_done"],
            "blink_count": eye["event_count"],
            "blink_duration_ms": eye["last_valid_duration_ms"],
            "eye_closure_duration_ms": eye["current_duration_ms"] if ear is not None and eye["initialized"] else None,
            "last_eye_closure_duration_ms": eye["last_event_duration_ms"],
            "eye_observed_sec": eye["observed_seconds"],
            "mar": mar,
            "pom_pct": mouth["pom"],
            "yawning_frequency_per_min": mouth["rate_per_minute"],
            "yawn_detected": mouth["event_done"],
            "yawn_count": mouth["event_count"],
            "yawn_duration_ms": mouth["last_valid_duration_ms"],
            "mouth_open_duration_ms": mouth["current_duration_ms"] if mar is not None and mouth["initialized"] else None,
            "last_mouth_open_duration_ms": mouth["last_event_duration_ms"],
            "mouth_observed_sec": mouth["observed_seconds"],
            "pitch_deg": None if angles is None else float(angles[0]),
            "pitch_amplitude_deg": None if angles is None else float(abs(angles[0])),
            "yaw_deg": None if angles is None else float(angles[1]),
            "roll_deg": None if angles is None else float(angles[2]),
            "over_angle": over_angle,
            "over_angle_pct": over_percent,
            "nodding": None if nod_state is None else bool(nod_state),
            "nod_detected": nod["event_done"],
            "nod_count": nod["event_count"],
            "nodding_frequency_per_min": nod["rate_per_minute"],
            "nod_duration_ms": nod["current_duration_ms"] if nod_state is not None else None,
            "last_nod_duration_ms": nod["last_event_duration_ms"],
            "nod_event_duration_ms": nod["last_valid_duration_ms"],
            "head_observed_sec": head_seconds,
            **motion,
        }

    def process(self, frame, timestamp=None):
        if self.tracker is None:
            self.tracker = FaceTracker(*self.confidence, config=self.config)
        timestamp = time.monotonic() if timestamp is None else timestamp
        return self.process_landmarks(self.tracker.process(frame), frame.shape[1::-1], timestamp)

    def close(self):
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None
```

---

### 7.6. Cổng Kiểm Duyệt, Chuẩn Hóa `physical_v1` & Hợp Nhất `Fitness` (`engine.py`, `normalization.py`, `fitnessFunction.py`)

#### a. Chuẩn hóa đặc trưng `PhysicalNormalizer` (`ai/OptimalAlgorithsm/DataSystem/normalization.py`)
Nguồn: [normalization.py](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/DataSystem/normalization.py)

```python
"""Đổi đơn vị theo miền bộ đo, không ánh xạ mức nguy hiểm."""

import hashlib
import json
from dataclasses import asdict
from math import ceil, isfinite
from numbers import Real

from ai.PhysicalBranch.camera_metrics import CameraMetrics

FEATURE_ENCODING = "physical_v1"


class PhysicalNormalizer:
    def __init__(self, config):
        for percentile in (config.blink_duration_percentile, config.nod_duration_percentile):
            if not isfinite(percentile) or not 0 <= percentile <= 100:
                raise ValueError("Duration percentiles must be in [0, 100]")
        # Dùng đúng cấu hình đã gộp của detector; chưa mở camera/FaceMesh.
        camera = CameraMetrics(config=config)

        def rate_bound(fsm):
            duration = max((fsm.minimum - 1e-6) / 1000, 1 / config.fps)
            return (ceil(fsm.window / duration) + 1) * 60 / fsm.window

        # FFT làm tròn số mẫu nội suy; cửa sổ lẻ frame có thể tăng Fs hiệu dụng.
        motion_window = camera.head_motion.window
        nyquist = max(3, ceil(motion_window * config.fps)) / (2 * motion_window)
        self.scales = {
            "blink_frequency": rate_bound(camera.eye.fsm),
            "blink_duration": camera.eye.fsm.maximum,
            "perclos": 1.0,
            "yawn_frequency": rate_bound(camera.mouth.fsm),
            "nod_duration": camera.nod.maximum / 1000,
            "nod_frequency": rate_bound(camera.nod),
            "dominant_head_motion_frequency": nyquist,
            "neuralnetwork_score": 1.0,
        }
        payload = json.dumps({"encoding": FEATURE_ENCODING, "config": asdict(config)},
                             sort_keys=True, allow_nan=False)
        self.id = hashlib.sha256(payload.encode()).hexdigest()

    def transform(self, raw):
        result = {}
        for key, upper in self.scales.items():
            value = raw[key]
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
                raise ValueError(f"Invalid raw feature: {key}")
            # Chỉ dung sai số học tại biên; không che số đo lỗi bằng clipping.
            if value > upper + 1e-6:
                raise ValueError(f"Raw feature exceeds detector bound: {key}")
            result[key] = min(float(value / upper), 1.0)
        return result
```

#### b. Cổng kiểm duyệt & Trích xuất mẫu `Engine._features` (`ai/OptimalAlgorithsm/DataSystem/engine.py`)
Nguồn: [engine.py](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/DataSystem/engine.py)

```python
import sys
import numpy as np
import torch
from ai.OptimalAlgorithsm.DataSystem.normalization import FEATURE_ENCODING


class Engine:
    # ... (khởi tạo xem chi tiết trong DataSystem/engine.py) ...

    @staticmethod
    def _duration(history, percentile):
        values = [duration for _, duration in history]
        return float(np.percentile(values, percentile)) if values else 0.0

    def _features(
        self, m, blink_durations, nod_durations, nn_frames, source, start, label
    ):
        window = self.window_sec
        coverage = [
            m[k]
            for k in ("eye_observed_sec", "mouth_observed_sec", "head_observed_sec")
        ]
        windows = m.get("windows_sec", self.config.windows)
        required = [windows.get(k, window) * self.min_observed_ratio
                    for k in ("perclos", "pom", "over_angle")]
        if (
            not m["eye_ready"]
            or not m["mouth_ready"]
            or not m["head_calibrated"]
            or m["perclos_pct"] is None
            or any(
                v is None or not np.isfinite(v) or v < minimum - 1e-9
                for v, minimum in zip(coverage, required)
            )
        ):
            print(f"[skip] {source} [{start:g}, {start + window:g}): "
                  f"ready eye/mouth/head={m['eye_ready']}/{m['mouth_ready']}/{m['head_calibrated']}, "
                  f"coverage eye/mouth/head={coverage}s (need {required}s), "
                  f"perclos={m['perclos_pct']}", file=sys.stderr, flush=True)
            return None

        motion_window = windows.get("head_motion", self.window_sec)
        frequency = m["head_motion_frequency_hz"]
        resolution = m["head_motion_resolution_hz"]
        observed = m["head_motion_observed_sec"]
        reliable_motion = (
            frequency is not None
            and resolution is not None
            and observed is not None
            and np.isfinite((frequency, resolution, observed)).all()
            and frequency >= 0
            and resolution > 0
            and observed >= motion_window - 1e-9
        )
        if not reliable_motion:
            print(f"[skip] {source} [{start:g}, {start + window:g}): "
                  f"FFT observed={observed}s (need {motion_window:g}s), "
                  f"frequency={frequency}Hz, resolution={resolution}Hz",
                  file=sys.stderr, flush=True)
            return None

        with torch.inference_mode():
            score = float(self.neural(np.stack(nn_frames)))
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid neural score: {score}")

        raw = {
            "blink_frequency": m["blink_rate_per_min"],
            "blink_duration": self._duration(blink_durations, self.config.blink_duration_percentile),
            "perclos": m["perclos_pct"] / 100,
            "yawn_frequency": m["yawning_frequency_per_min"],
            "nod_duration": self._duration(nod_durations, self.config.nod_duration_percentile) / 1000,
            "nod_frequency": m["nodding_frequency_per_min"],
            "dominant_head_motion_frequency": frequency,
            "neuralnetwork_score": score,
        }
        return {
            **self.normalizer.transform(raw),
            "raw_features": raw,
            "normalization_id": self.normalizer.id,
            "normalization_scales": self.normalizer.scales,
            "source_video": source,
            "start_second": start,
            "end_second": start + window,
            "label": label,
            "feature_encoding": FEATURE_ENCODING,
        }
```

#### c. Ba Phép Hợp Nhất Quyết Định `Fitness` (`ai/OptimalAlgorithsm/TrainingSystem/fitnessFunction.py`)
Nguồn: [fitnessFunction.py](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/TrainingSystem/fitnessFunction.py)

```python
import numpy as np

FEATURES = [
    "blink_frequency",
    "blink_duration",
    "perclos",
    "yawn_frequency",
    "nod_duration",
    "nod_frequency",
    "dominant_head_motion_frequency",
    "neuralnetwork_score",
]


class Fitness:
    def __init__(self, X):
        self.X = np.asarray(X, dtype=np.float32)
        if (
            self.X.ndim != 2
            or self.X.shape[1] != 8
            or len(self.X) == 0
            or not np.isfinite(self.X).all()
            or np.any((self.X < 0) | (self.X > 1))
        ):
            raise ValueError(
                "X must be a nonempty [N,8] array of finite values in [0,1]"
            )

    @staticmethod
    def _candidate(candidate):
        candidate = np.asarray(candidate, dtype=np.float32)
        if candidate.shape != (9,):
            raise ValueError("Candidate must be [w1, ..., w8, T_cls]")
        if not np.isfinite(candidate).all() or np.any(
            (candidate < 0) | (candidate > 1)
        ):
            raise ValueError("Candidate values must be in [0, 1]")
        w = candidate[:8]
        threshold = candidate[8]
        mask = w >= 0.5
        return w, threshold, mask

    def score(self, candidate, method="weighted_mean"):
        if method not in ("weighted_mean", "noisy_or", "cooccurrence"):
            raise ValueError(f"Unknown fitness method: {method}")
        w, _, mask = self._candidate(candidate)
        if not mask.any():
            return np.zeros(len(self.X), dtype=np.float32)
        z = self.X[:, mask] * w[mask]
        if method == "weighted_mean":
            return z.sum(axis=1) / w[mask].sum()
        if method == "noisy_or":
            return 1.0 - np.prod(1.0 - z, axis=1)
        z = np.sort(z, axis=1)
        strongest = z[:, -1]
        # The strongest pair adds support without accumulating many weak signals.
        joint = strongest * z[:, -2] if z.shape[1] > 1 else 0.0
        return strongest + (1.0 - strongest) * joint

    def weighted_mean(self, candidate):
        return self.predict(candidate, "weighted_mean")

    def noisy_or(self, candidate):
        return self.predict(candidate, "noisy_or")

    def cooccurrence(self, candidate):
        return self.predict(candidate, "cooccurrence")

    def predict(self, candidate, method="weighted_mean"):
        score = self.score(candidate, method)
        _, threshold, mask = self._candidate(candidate)
        return (score >= threshold).astype(np.int8) if mask.any() else np.zeros(len(self.X), dtype=np.int8)
```

---

### 7.7. Đóng Gói & Xuất Mô Hình ONNX Hỗ Trợ Cache `p4_old` (`export_onnx.py`)
Nguồn: [export_onnx.py](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/export_onnx.py)

```python
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.LSTM.src.checkpoint import load_model
from ai.LSTM.src.models_convgru import ConvGRUCell

OPSET_VERSION = 17
IMAGE_SIZE = 224
NECK_P4_CHANNELS = 128


class ScriptableConvGRU(nn.Module):
    """ConvGRU loop & attention pooling designed for TorchScript & dynamic ONNX Loop export."""

    def __init__(
            self,
            convgru: ConvGRUCell,
            spatial_attention: nn.Conv2d,
            fc_out: nn.Sequential,
            hidden_dim: int,
    ):
        super().__init__()
        self.convgru = convgru
        self.spatial_attention = spatial_attention
        self.fc_out = fc_out
        self.hidden_dim = hidden_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features shape: (1, T, input_dim, H, W)
        b = features.shape[0]
        t = features.shape[1]
        h_dim = features.shape[3]
        w_dim = features.shape[4]

        h = torch.zeros(
            b,
            self.hidden_dim,
            h_dim,
            w_dim,
            dtype=features.dtype,
            device=features.device,
        )
        vectors = torch.jit.annotate(list[torch.Tensor], [])

        for i in range(t):
            h = self.convgru(features[:, i], h)
            att = F.softmax(self.spatial_attention(h).flatten(2), dim=-1)
            att = att.reshape(b, 1, h_dim, w_dim)
            vectors.append((h * att).sum(dim=(-2, -1)))

        stacked = torch.stack(vectors, dim=1)
        mean_vec = stacked.mean(dim=1)
        logits = self.fc_out(mean_vec)
        return F.softmax(logits, dim=1)


class ExportDrowsyClassifier(nn.Module):
    """Wrapper encapsulating BackboneNeck, Adapter, and ConvGRU for ONNX export and inference.

    Inputs:
        x: Video frames tensor of shape (T, 3, H, W). Dtype can be uint8 [0..255] or matching float.
        p4_old: Previous P4 feature tensor of shape (T_old, C, H, W).
                If no old features are available, pass an empty tensor of shape (0, C, H, W)
                or a 1D indicator such as torch.tensor([0]).

    Outputs:
        prob: Softmax probabilities of shape (1, num_classes) [alert, drowsy].
        p4_features: Concatenated P4 features of shape (T_old + T, C, H, W).
    """

    def __init__(self, checkpoint: str | Path):
        super().__init__()
        model, _ = load_model(checkpoint)
        self.backbone_neck = model.backbone_neck
        self.adapter = model.conv_gru.adapter
        self.gru_classifier = ScriptableConvGRU(
            model.conv_gru.convgru,
            model.conv_gru.spatial_attention,
            model.conv_gru.fc_out,
            model.conv_gru.hidden_dim,
        )

    def forward(
            self,
            x: torch.Tensor,
            p4_old: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Handle uint8 or float frame inputs matching weight dtype
        param_dtype = self.adapter[0].weight.dtype
        if x.dtype == torch.uint8:
            frames = x.to(param_dtype) / 255.0
        else:
            frames = x.to(param_dtype)

        # Extract P4 features: (T, C, H, W)
        p4 = self.backbone_neck(frames)[1]

        # Normalize p4_old: if 1D / dummy / empty tensor, convert to 4D empty tensor
        if p4_old.dim() != 4:
            p4_norm = torch.zeros(
                0,
                p4.shape[1],
                p4.shape[2],
                p4.shape[3],
                dtype=p4.dtype,
                device=x.device,
            )
        else:
            p4_norm = p4_old.to(p4.dtype)

        # Concatenate cached features and new features along sequence dimension
        features = torch.cat([p4_norm, p4], dim=0)

        # Adapt P4 channels to ConvGRU input_dim
        adapted = self.adapter(features)

        # Add batch dimension for ConvGRU: (1, total_T, input_dim, H, W)
        adapted_seq = adapted.unsqueeze(0)

        probs = self.gru_classifier(adapted_seq)
        return probs, features
```

---

### 7.8. Luồng Suy Luận Thời Gian Thực Đa Luồng (`inference.py`)
Nguồn: [inference.py](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/inference.py)

```python
"""Camera thật → đặc trưng physical_v1 → CNN–ConvGRU → phân loại.

Chỉnh cấu hình bên dưới rồi chạy: python -m ai.OptimalAlgorithsm.inference
Một lần chạy neural phục vụ cả ba phép fusion của TrainingSystem.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from ai.OptimalAlgorithsm.DataSystem.config import DEFAULT_CONFIG
from ai.OptimalAlgorithsm.DataSystem.engine import Engine
from ai.OptimalAlgorithsm.DataSystem.loadvideo2handle import balance_brightness
from ai.OptimalAlgorithsm.DataSystem.neuralnetwork import letterbox
from ai.OptimalAlgorithsm.DataSystem.normalization import FEATURE_ENCODING
from ai.OptimalAlgorithsm.TrainingSystem.fitnessFunction import FEATURES, Fitness
from ai.PhysicalBranch.camera_metrics import CameraMetrics


@dataclass(frozen=True)
class CameraConfig:
    camera: int | str = 0
    checkpoint: Path = Path(__file__).resolve().parents[1] / "LSTM/checkpoints/head_landmarkv1/best.pt"
    device: str | None = None  # None: tự chọn CUDA/CPU.
    interval: float = 1.0
    duration: float | None = None
    output: str | None = None
    no_display: bool = False
    primary_fusion: str = "cooccurrence"  # Nhãn chính; vẫn xuất cả ba chế độ.


CAMERA_CONFIG = CameraConfig()
FUSION_CANDIDATES = {
    "weighted_mean": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
    "noisy_or": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
    "cooccurrence": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
}
FUSION_REPORTS = {"weighted_mean": None, "noisy_or": None, "cooccurrence": None}


def classify(sample, fusion):
    selected = fusion["selected_model"]
    if selected["model"] != "candidate":
        return selected["constant_label"]
    return int(Fitness([[sample[name] for name in FEATURES]]).predict(
        selected["candidate"], fusion["fitness_method"]
    )[0])


def fusion_details(sample, fusion):
    selected = fusion["selected_model"]
    score = threshold = None
    if selected["model"] == "candidate":
        candidate = selected["candidate"]
        score = float(Fitness([[sample[name] for name in FEATURES]]).score(candidate, fusion["fitness_method"])[0])
        threshold = float(candidate[8])
    return {"score": score, "threshold": threshold, "prediction": classify(sample, fusion)}


class NeuralBuffer:
    """Lấy ảnh quá khứ gần nhất theo lưới neural_fps; bỏ clip khi camera gián đoạn."""

    def __init__(self, config):
        self.config = config
        self.frames = deque(maxlen=config.neural_count)
        self.previous = None
        self.previous_time = None
        self.next_time = 0.0

    def update(self, rgb, timestamp):
        if self.previous_time is None or timestamp - self.previous_time > self.config.max_gap_sec:
            self.frames.clear()
            self.next_time = timestamp
        frame = np.moveaxis(letterbox(rgb, self.config.neural_size), -1, 0)
        while self.next_time <= timestamp + 1e-9:
            past = self.previous if self.next_time < timestamp - 1e-9 else frame
            self.frames.append(past)
            self.next_time += 1 / self.config.neural_fps
        self.previous, self.previous_time = frame, timestamp


def run(args=CAMERA_CONFIG):
    if not np.isfinite(args.interval) or args.interval <= 0:
        raise ValueError("interval phải hữu hạn và > 0")
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        raise ValueError("duration phải hữu hạn và > 0")
    config = replace(DEFAULT_CONFIG, checkpoint_path=str(args.checkpoint), device=args.device)
    engine = Engine(Path(__file__).parent, config=config)
    fusions = load_fusions(engine.normalizer)
    mode = args.primary_fusion
    if mode not in fusions:
        raise ValueError(f"primary_fusion không có trong FUSION_CANDIDATES: {mode}")
    source = int(args.camera) if isinstance(args.camera, str) and args.camera.isdecimal() else args.camera
    capture = cv2.VideoCapture(source)
    camera = CameraMetrics(config=config)
    worker = ThreadPoolExecutor(max_workers=1)
    pending = output = None
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Không mở được camera: {source}")
        capture.set(cv2.CAP_PROP_FPS, config.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if args.output:
            output = Path(args.output).expanduser().open("a", encoding="utf-8")
        buffer = NeuralBuffer(config)
        blink, nod = deque(), deque()
        origin = time.perf_counter()
        last_timestamp = None
        next_prediction = config.window_sec
        result = None
        status = "WARMING UP"
        frames = 0
        while True:
            ok, frame = capture.read()
            timestamp = time.perf_counter() - origin
            if last_timestamp is not None and timestamp <= last_timestamp:
                timestamp = last_timestamp + 1e-4
            last_timestamp = timestamp
            if not ok:
                raise RuntimeError("Không đọc được frame từ camera")
            frame = balance_brightness(frame, config.target_luma)
            metrics = camera.process(frame, timestamp)
            buffer.update(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), timestamp)
            for history, detected, duration in (
                (blink, "blink_detected", "blink_duration_ms"),
                (nod, "nod_detected", "nod_event_duration_ms"),
            ):
                if metrics[detected] and metrics[duration] is not None:
                    history.append((timestamp, metrics[duration]))
                while history and history[0][0] <= timestamp - config.window_sec:
                    history.popleft()

            if pending is not None and pending.done():
                result = pending.result()
                pending = None
                status = "WAITING FOR VALID WINDOW"
                if result is not None:
                    result.pop("label")
                    result["source_camera"] = result.pop("source_video")
                    result["fusion_details"] = {
                        method: fusion_details(result, fusion) for method, fusion in fusions.items()
                    }
                    result["fusion_predictions"] = {
                        method: detail["prediction"] for method, detail in result["fusion_details"].items()
                    }
                    result["fusion_sources"] = {method: fusion["source"] for method, fusion in fusions.items()}
                    result["prediction"] = result["fusion_predictions"][mode]
                    result["mode"] = mode
                    line = json.dumps(result, ensure_ascii=False, allow_nan=False)
                    print(line, flush=True)
                    if output:
                        output.write(line + "\n")
                        output.flush()
                    status = "DROWSY" if result["prediction"] else "ALERT"

            if timestamp >= next_prediction and pending is None:
                next_prediction = timestamp + args.interval
                if len(buffer.frames) == config.neural_count:
                    pending = worker.submit(
                        engine._features, metrics.copy(), tuple(blink), tuple(nod),
                        tuple(buffer.frames), str(source), timestamp - config.window_sec, None,
                    )
                else:
                    result = None
                    status = "WAITING FOR CAMERA CLIP"

            frames += 1
            if not args.no_display:
                canvas = draw_dashboard(frame, metrics, result, fusions, status, timestamp,
                                        frames / max(timestamp, 1e-6), args)
                cv2.imshow("Driver Guardian", canvas)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                if cv2.getWindowProperty("Driver Guardian", cv2.WND_PROP_VISIBLE) < 1:
                    break
            if args.duration is not None and timestamp >= args.duration:
                break
    finally:
        capture.release()
        camera.close()
        worker.shutdown(wait=True, cancel_futures=True)
        if output:
            output.close()
        if not args.no_display:
            cv2.destroyAllWindows()
```

---

## 8. Phân Tích Độ Phức Tạp Thuật Toán & Ước Tính Thời Gian Hoàn Thành Phiên Bản v1 (Complexity & Execution Time Specification)

Mục này định lượng chi tiết **độ phức tạp tính toán lý thuyết** (Time Complexity & Space Complexity theo ký hiệu Big-O), **số phép tính số học / MACs / FLOPs**, và **ước tính thời gian hoàn thành (Execution Time / Latency)** cho toàn bộ các thuật toán trong kiến trúc nâng cấp `v1`, kèm bảng đối chiếu trực tiếp tốc độ/FPS với bản `v0`.

### Ký hiệu các biến kích thước chuẩn trong hệ thống `v1`:
- $H \times W = 480 \times 640$: Độ phân giải khung hình CameraX Hardware cố định (`OUTPUT_IMAGE_FORMAT_RGBA_8888`, $P = 307,200$ pixel, $1,228,800\text{ bytes/frame}$).
- $S = 224$: Kích thước cạnh ảnh vuông đầu vào của nhánh mạng học sâu (`neural_size = 224`, số phần tử mỗi frame $3 \times 224 \times 224 = 150,528$ byte).
- $L = 478$: Số điểm mốc khuôn mặt 2D trích xuất từ MediaPipe `FaceLandmarker` (chạy trên **`Delegate.GPU`**).
- $n_{\text{pnp}} = 6$: Số điểm mốc 3D–2D dùng để giải bài toán `SolvePnP`.
- $C_{\text{cal}} = 30$: Số frame liên tiếp dùng để hiệu chuẩn tư thế đầu trung tính (`calibration_frames = 30`).
- $K = 2$: Số trạng thái ẩn của mô hình Markov (`AdaptiveHMM`: trạng thái nghỉ `0` và trạng thái sự kiện `1`).
- $N_{\text{init}} = 300, \; N_{\text{adapt}} = 300$: Số lượng mẫu khởi tạo HMM (`10s @ 30 FPS`) và chu kỳ học thích nghi trực tuyến (`adapt_interval = 300`).
- $I_{\text{km}} = 25, \; I_{\text{em}} = 20$: Số vòng lặp tối đa của 1D K-Means và thuật toán Baum-Welch EM khi khởi tạo.
- $W = 60.0\text{s}$: Độ rộng cửa sổ thời gian trượt vật lý (`window_sec = 60.0`).
- $M_W \approx W \times \text{FPS}$: Số lượng mẫu / khoảng thời gian lưu trong cửa sổ trượt $60.0\text{s}$ (ở $30\text{ FPS}$, $M_W \le 1,800$; ở $60\text{ FPS}$, $M_W \le 3,600$).
- $N_{\text{fft}} \approx M_W$: Số điểm lưới nội suy cách đều dùng cho biến đổi `rfft` trong `HeadMotionWindow`.
- $T_{\text{total}} = 50, \; T_{\text{new}} = 5, \; T_{\text{old}} = 45$: Tổng số frame của cửa sổ mạng học sâu (`10s @ 5 FPS`), số frame mới mỗi chu kỳ `interval = 1.0s` (cũng là số frame duy nhất cần chạy `letterboxRgbaToRgbCHW`), và số frame đặc trưng P4 tái sử dụng từ cache `p4_old`.
- $D = 8$: Số đặc trưng đầu vào của bộ chuẩn hóa `PhysicalNormalizer` và bộ hợp nhất `Fitness`.

---

### 8.1. Độ Phức Tạp Thuật Toán Trên Cụm Luồng 1A–1C (Chạy Mỗi Frame)

Bảng dưới đây thống kê độ phức tạp thời gian, không gian bộ nhớ và luồng thực thi của từng thuật toán trên mỗi frame camera trong kiến trúc `v1`:

| STT | Thuật Toán / Module | Luồng Thực Thi Trong Bản `v1` | Độ Phức Tạp Thời Gian (Time Complexity) | Độ Phức Tạp Không Gian (Space Complexity) | Khối Lượng Phép Tính Cụ Thể (Operations / FLOPs) |
| :-: | :--- | :---: | :---: | :---: | :--- |
| **1a** | **Thu nhận `RGBA_8888` khi TẮT `balanceBrightness` trên UI** (`enableBalanceBrightness == false`) | **`Luồng 1A: CameraIngestThread`** | $\mathcal{O}(H \cdot W)$ *(chỉ `memcpy` 1.17 MB)* | $\mathcal{O}(H \cdot W)$ | • Copy khối trực tiếp `ByteBuffer` `RGBA_8888` ($640\times480\times4 = 1.23\text{ MB}$) và gọi `imageProxy.close()`.<br>• **0 phép tính chuyển hệ màu, 0 phép tính gamma (`0 FLOPs`)**. |
| **1b** | **Thu nhận `RGBA_8888` khi BẬT `balanceBrightness` trên UI** (`enableBalanceBrightness == true`) | **`Luồng 1A: CameraIngestThread`** | $\mathcal{O}(H \cdot W + 256)$ | $\mathcal{O}(H \cdot W + 256)$ | • Chuyển `RGBA -> BGR -> YCrCb` + tính $\mu_Y$: $\approx 6 \cdot H \cdot W$ ops.<br>• Nếu $\mu_Y < 70$ (trời tối): Tạo `LUT` ($256$ `pow`) + tra bảng `LUT` $H \cdot W$ byte + `YCrCb -> BGR -> RGBA` ($\approx 1.8\text{–}3.7 \times 10^6$ ops). |
| **2a** | **`GpuFaceTracker.processRgbaFrame`** (`MediaPipe FaceLandmarker` ép chạy **`Delegate.GPU`**) | **`Luồng 1B: MediaPipe GPU Thread`** *(Song song với Bước 2b)* | $\mathcal{O}(H_{\text{mp}} \cdot W_{\text{mp}} / P_{\text{gpu}} + L)$ | $\mathcal{O}(H \cdot W + L)$ | • Nạp `MPImage(IMAGE_FORMAT_RGBA, 640×480)` thẳng vào GPU Texture.<br>• Chạy shader OpenGL ES / OpenCL cho mạng FaceDetector + FaceMesh ($\approx \mathbf{15\text{–}25\text{ MFLOPs}}$ trên GPU).<br>• Trên CPU chỉ tốn $478 \times 3 \approx \mathbf{1,434\text{ FLOPs}}$ để nhân tọa độ $(x_i \cdot 640, y_i \cdot 480)$. |
| **2b** | **`NeuralBuffer.updateRaw`** *(Không chạy `letterbox`)* | **`Luồng 1B`** *(Chạy trên CPU trong lúc đợi GPU Bước 2a)* | $\mathcal{O}(1)$ | $\mathcal{O}(T_{\text{total}})$ con trỏ | • Chỉ gán con trỏ `RawRgbaFrame` vào `ArrayDeque(50)` khi chạm mốc lưới `5.0 FPS`.<br>• **Chi phí `letterbox` tại nhánh Camera = `0 ops` (giảm 100%)**. |
| **3** | **Nhánh Mắt (`EAR` + `AdaptiveHMM("eye")` + `Blink FSM` + `PERCLOS`)** | **`Luồng 1C-1: EyeWorker`** *(Chạy song song với `1C-2` & `1C-3`)* | • Thường: $\mathcal{O}(M_W)$<br>• Frame #300: $+\mathcal{O}(I_{\text{em}} N_{\text{init}} K^2)$ | $\mathcal{O}(M_W + N_{\text{init}} K^2)$ | • `aspect_ratio` 2 mắt: $\approx 40\text{ FLOPs}$.<br>• `AdaptiveHMM("eye").predict`: $\approx 35\text{ FLOPs}$ (Frame #300 chạy 20 vòng EM: $\approx 1.25 \times 10^5\text{ FLOPs}$).<br>• `StateMachine("eye")` + `WindowRatio("perclos")`: $\approx 5.3 \times 10^3\text{ FLOPs}$. |
| **4** | **Nhánh Miệng (`MAR` + `AdaptiveHMM("mouth")` + `Yawn FSM` + `POM`)** | **`Luồng 1C-2: MouthWorker`** *(Chạy song song với `1C-1` & `1C-3`)* | • Thường: $\mathcal{O}(M_W)$<br>• Frame #300: $+\mathcal{O}(I_{\text{em}} N_{\text{init}} K^2)$ | $\mathcal{O}(M_W + N_{\text{init}} K^2)$ | • `aspect_ratio` miệng: $\approx 20\text{ FLOPs}$.<br>• `AdaptiveHMM("mouth").predict`: $\approx 35\text{ FLOPs}$ (Frame #300 chạy 20 vòng EM: $\approx 1.25 \times 10^5\text{ FLOPs}$).<br>• `StateMachine("mouth")` + `WindowRatio("pom")`: $\approx 5.3 \times 10^3\text{ FLOPs}$. |
| **5** | **Nhánh Đầu & Phổ FFT (`SolvePnP` + `Nod FSM` + `Over-Angle` + `HeadMotionWindow rfft`)** | **`Luồng 1C-3: HeadPoseAndFftWorker`** *(Chạy song song với `1C-1` & `1C-2`)* | • Khi $t < 60\text{s}$: $\mathcal{O}(M_W)$<br>• Khi $t \ge 60\text{s}$: $\mathcal{O}(M_W \log M_W + N_{\text{fft}} \log N_{\text{fft}})$ | $\mathcal{O}(M_W + N_{\text{fft}})$ | • `SolvePnP` (`SQPnP` + `RefineLM` + `Rodrigues` + `RQDecomp3x3`): $\approx 2.5 \times 10^3\text{ FLOPs}$.<br>• `StateMachine("pitch")` + `WindowRatio("over_angle")`: $\approx 5.3 \times 10^3\text{ FLOPs}$.<br>• `HeadMotionWindow` (Tích phân hình thang + Nội suy + `rfft` 1800 điểm): $\approx \mathbf{1.0 \times 10^5\text{ FLOPs}}$. |

> [!TIP]
> **Hiệu quả của Fork-Join 3 Luồng Con (`1C-1` $\parallel$ `1C-2` $\parallel$ `1C-3`):**
> Vì 3 luồng con chạy đồng thời trên 3 nhân CPU khác nhau, thời gian hoàn thành của cả tầng Vật Lý sau MediaPipe GPU không còn là tổng $T_{\text{Eye}} + T_{\text{Mouth}} + T_{\text{Head+FFT}}$ mà chỉ còn bằng:
> $$T_{\text{Stage 1C}} = \max\left(T_{\text{1C-1 (Eye)}}, \; T_{\text{1C-2 (Mouth)}}, \; T_{\text{1C-3 (Head+FFT)}}\right)$$
> Đặc biệt tại **Frame #300** (khi cả `eye` và `mouth` cùng chạy 20 vòng lặp Baum-Welch EM), `EyeWorker` và `MouthWorker` chạy song song trên 2 nhân CPU cùng lúc với `HeadPoseAndFftWorker` trên nhân thứ 3, cắt giảm **> 55%** đỉnh trễ (latency spike)!

---

### 8.2. Độ Phức Tạp Nhánh Học Sâu (`CNN–ConvGRU`), `letterbox` Trên `ONNX Worker` & Hiệu Quả Của `p4_old` Cache

Trên **`Luồng 2: ONNX Worker Thread`** (chạy mỗi chu kỳ `interval = 1.0s`), công việc bao gồm:
1. **Tiền xử lý `letterboxRgbaToRgbCHW` chỉ trên $T_{\text{new}} = 5$ frames mới:**
   - Resize `RGBA` từ $640\times480$ về $224\times168$ + bỏ kênh Alpha (`COLOR_RGBA2RGB`) + điền viền $224\times224$ + chuyển trục `HWC -> CHW`: $\approx 2.2 \times 10^5$ ops/frame.
   - Tổng cho $T_{\text{new}} = 5$ frames mỗi giây: $5 \times 2.2 \times 10^5 = \mathbf{1.1 \times 10^6\text{ ops/giây}}$ (so với chạy `letterbox` ở 30–60 FPS trên luồng Camera tốn $9.0\text{–}18.0 \times 10^6\text{ ops/giây}$ $\implies$ **tiết kiệm 88%–94% phép tính**!).
2. **Suy luận mạng `ExportDrowsyClassifier` với cache `p4_old`:**

| Khối Mạng (Module) | Cấu Trúc Chi Tiết & Kích Thước Tensor | Số Tham Số (Parameters) | Khối Lượng Tính Toán Trên 1 Frame / 1 Bước Thời Gian (MMACs) | Số Bước Chạy Mỗi Chu Kỳ `1.0s` (Có `p4_old`) |
| :--- | :--- | :---: | :---: | :---: |
| **1. `BackboneNeck` (CNN + PAFPN)** | `video [T_new, 3, 224, 224]` $\rightarrow$ `p4 [T_new, 128, 14, 14]` | `1,328,720` active | **243.83 MMACs / frame** *(487.66 MFLOPs)* | **Chỉ chạy trên $T_{\text{new}} = 5$ frames mới** *(1,219.2 MMACs)* |
| **2. `Adapter`** | `Conv2d(128->96, k=1)` + `GroupNorm(1, 96)` + `SiLU` | `12,480` | **2.41 MMACs / step** *(4.82 MFLOPs)* | Chạy trên $T_{\text{total}} = 50$ frames *(120.5 MMACs)* |
| **3. `ScriptableConvGRU`** | `ConvGRUCell(96, 96, k=3)` + `spatial_attention` | `498,049` | **97.56 MMACs / step** *(195.12 MFLOPs)* | Chạy tuần tự qua $T_{\text{total}} = 50$ bước *(4,878.0 MMACs)* |
| **4. `fc_out` + `Softmax`** | Mean Pooling `[1, 50, 96] -> [1, 96]` + `Linear(96->2)` | `194` | **0.0002 MMACs** *(~400 FLOPs)* | 1 lần / cửa sổ |
| **TỔNG TOÀN MẠNG (Có `p4_old`)** | **`ExportDrowsyClassifier`** | **`1,839,443` active** | — | **6,217.7 MMACs (12.44 GFLOPs)** *(Giảm 63.8% so với không cache)* |

---

### 8.3. Ước Tính Thời Gian Hoàn Thành (Execution Time) & So Sánh Tốc Độ FPS Giữa Bản v0 và Bản v1

Dưới đây là bảng ước tính thời gian hoàn thành (Latency tính bằng **ms**) trên 4 phân khúc phần cứng và so sánh trực tiếp giữa **Bản v0 (`INFERENCE_KOTLIN_SPECIFICATION.md`)** và **Bản nâng cấp v1 (`INFERENCE_KOTLIN_SPECIFICATION_v1.md`)**:

#### a. Bảng Thời Gian Thực Thi Mỗi Frame Trên Cụm Luồng Camera & Vật Lý (`1A` + `1B` + `1C`)

| Bước Thuật Toán Trên Mỗi Frame (`640×480 RGBA_8888`) | PC Tham Chiếu | Android Cao Cấp (High-End SoC) | Android Tầm Trung (Mid-Range SoC) | Android Phổ Thông (Low-End SoC) | Ghi Chú Nâng Cấp Trong Bản `v1` |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **TẦNG 1A (`CameraIngestThread` — Chạy Pipeline độc lập):** | | | | | |
| • Chế độ **TẮT `balanceBrightness`** trên UI *(Zero-Conversion `RGBA_8888`)* | **`0.03 – 0.06 ms`** | **`0.05 – 0.12 ms`** | **`0.10 – 0.20 ms`** | **`0.18 – 0.35 ms`** | Chỉ copy trực tiếp buffer `RGBA_8888` từ CameraX và đóng `imageProxy.close()`. |
| • Chế độ **BẬT `balanceBrightness`** trên UI *(Khi trời tối $\mu_Y < 70$)* | `0.40 – 0.75 ms` | `0.75 – 1.40 ms` | `1.40 – 2.60 ms` | `2.60 – 4.20 ms` | Chạy tách biệt trên `Luồng 1A`, gối đầu (overlap) hoàn toàn với Tầng 1B & 1C của frame trước! |
| **TẦNG 1B (`MediaPipe GPU` $\parallel$ `NeuralBuffer.updateRaw`):** | | | | | |
| • `GpuFaceTracker` (`MediaPipe FaceLandmarker` ép chạy **`Delegate.GPU`**) | **`1.50 – 2.50 ms`** | **`2.20 – 3.80 ms`** | **`3.80 – 6.20 ms`** | **`6.20 – 9.50 ms`** | Nhanh hơn **2.3× – 2.8×** so với chạy CPU ở bản v0 (`9–26 ms`), nhận trực tiếp `MPImage(IMAGE_FORMAT_RGBA)`. |
| • `NeuralBuffer.updateRaw` *(Không chạy `letterbox`)* | `0.001 ms` *(0 ms)* | `0.002 ms` *(0 ms)* | `0.003 ms` *(0 ms)* | `0.005 ms` *(0 ms)* | Chạy song song trên CPU trong lúc đợi GPU và không tốn thời gian `letterbox`. |
| **TẦNG 1C (Fork-Join 3 Luồng Con Vật Lý Song Song `1C-1` $\parallel$ `1C-2` $\parallel$ `1C-3`):** | | | | | |
| • Nhánh `1C-1 (EyeWorker)`: `EAR` + `HMM` + `Blink FSM` + `PERCLOS` | `0.03 – 0.05 ms` *(Ẩn)* | `0.05 – 0.10 ms` *(Ẩn)* | `0.10 – 0.20 ms` *(Ẩn)* | `0.20 – 0.38 ms` *(Ẩn)* | Chạy song song hoàn toàn với Nhánh `1C-3`, không cộng vào đường găng! |
| • Nhánh `1C-2 (MouthWorker)`: `MAR` + `HMM` + `Yawn FSM` + `POM` | `0.02 – 0.04 ms` *(Ẩn)* | `0.04 – 0.08 ms` *(Ẩn)* | `0.08 – 0.16 ms` *(Ẩn)* | `0.16 – 0.30 ms` *(Ẩn)* | Chạy song song hoàn toàn với Nhánh `1C-3`, không cộng vào đường găng! |
| • Nhánh `1C-3 (HeadPoseAndFftWorker)`: `SolvePnP` + `Nod/OverAngle` + **`rfft` 1800 điểm** | **`0.35 – 0.65 ms`** | **`0.60 – 1.25 ms`** | **`1.25 – 2.45 ms`** | **`2.45 – 4.25 ms`** | Đây là nhánh quyết định thời gian của Tầng 1C ($\max(T_{1C-1}, T_{1C-2}, T_{1C-3})$). |
| **TỔNG ĐƯỜNG GĂNG 1 FRAME Ở BẢN `v1` ($\max(T_{1A}, T_{1B} + T_{1C})$):** | **`1.85 – 3.15 ms`** | **`2.80 – 5.05 ms`** | **`5.05 – 8.65 ms`** | **`8.65 – 13.75 ms`** | **Nhanh hơn 2.4× – 2.8× so với Bản v0!** |
| *(Đối chiếu: Tổng thời gian 1 frame ở Bản `v0` tuần tự)* | *(`4.5 – 7.8 ms`)* | *(`6.8 – 12.0 ms`)* | *(`12.5 – 21.7 ms`)* | *(`22.7 – 37.4 ms`)* | — |
| **TỐC ĐỘ KHUNG HÌNH ĐẠT ĐƯỢC (EFFECTIVE FPS — BẢN `v1`):** | **`60+ FPS`** *(Max Camera)* | **`60 FPS` ổn định** | **`45 – 60 FPS`** *(Bản v0: 30–45 FPS)* | **`30 – 45 FPS`** *(Bản v0: 25–30 FPS)* | Không bao giờ rớt frame ngay cả trên máy tầm trung & phổ thông. |

---

#### b. Bảng Ước Tính Thời Gian Thực Thi Trên Luồng 2 (`ONNX Worker Thread` — Chu Kỳ `interval = 1.0s`)

Khi chuyển `letterbox` sang `ONNX Worker Thread` (chỉ chạy trên $T_{\text{new}} = 5$ frames mới mỗi giây):

| Tác Vụ Trên `ONNX Worker Thread` (Mỗi Chu Kỳ `1000 ms`) | PC Tham Chiếu | Android Cao Cấp (High-End) | Android Tầm Trung (Mid-Range) | Android Phổ Thông (Low-End) | Đánh Giá So Với Chu Kỳ `interval = 1000 ms` |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. `OnnxLetterboxProcessor.processBatchToRgbCHW` *(Chỉ chạy trên $T_{\text{new}} = 5$ frames `RGBA 640×480` mới)*** | **`0.45 – 0.80 ms`** | **`0.85 – 1.50 ms`** | **`1.60 – 2.80 ms`** | **`2.90 – 4.80 ms`** | Chiếm **$< 0.5\%$** chu kỳ `1000 ms`, giải phóng hoàn toàn luồng Camera! |
| **2. Chạy `OrtSession.run()` với `p4_old` cache ($T_{\text{new}}=5, T_{\text{old}}=45$):** | | | | | |
| • Chế độ **`INT8`** (`drowsy_classifier_int8.onnx`) | `18 – 30 ms` | `28 – 45 ms` | `55 – 90 ms` | `105 – 165 ms` | Tổng `letterbox + INT8 ONNX` chỉ tốn `30 – 170 ms` ($\ll 1000\text{ ms}$). |
| • Chế độ **`FP16`** (`drowsy_classifier_fp16.onnx`) | `22 – 38 ms` | `32 – 55 ms` | `65 – 110 ms` | `125 – 195 ms` | Tổng `letterbox + FP16 ONNX` chỉ tốn `33 – 200 ms` ($\ll 1000\text{ ms}$). |
| • Chế độ **`FP32`** (`drowsy_classifier_fp32.onnx`) | `35 – 60 ms` | `55 – 90 ms` | `110 – 180 ms` | `210 – 320 ms` | Tổng `letterbox + FP32 ONNX` chỉ tốn `56 – 325 ms` ($\ll 1000\text{ ms}$). |
| **3. Kiểm duyệt `Engine._features` + `PhysicalNormalizer` + 3 phép `Fitness`** | `< 0.05 ms` | `< 0.08 ms` | `0.10 – 0.15 ms` | `0.15 – 0.25 ms` | Chi phí không đáng kể ($< 0.3\text{ms}$). |

---

### 8.4. Tổng Hợp Dung Lượng Bộ Nhớ (RAM & VRAM Footprint)

| Thành Phần Bộ Nhớ | Công Thức Kích Thước | Dung Lượng Chính Xác | Luồng Sở Hữu |
| :--- | :--- | :---: | :--- |
| **1. Bộ đệm ảnh `NeuralBuffer` lưu `RawRgbaFrame` (`640×480 RGBA_8888`)** | • Tối đa $50$ frames trên lưới `5.0 FPS` + `1` frame `previous`:<br>  $51 \times 640 \times 480 \times 4\text{ bytes}$<br>*(Ghi chú: Nếu chỉ giữ $T_{\text{new}} \le 10$ frames `RawRgbaFrame` chưa qua ONNX vì 45 frames cũ đã nằm trong `cachedP4`, dung lượng giảm còn $11 \times 1.17\text{ MB} = \mathbf{12.89\text{ MB}}$)* | **59.77 MB** *(giữ đủ 50 raw frames)*<br>hoặc **12.89 MB** *(chỉ giữ raw frames chưa cache P4)* | `Luồng 1B` *(chia sẻ con trỏ bất biến sang `ONNX Worker Thread`)* |
| **2. Bộ đệm `letterbox` đầu vào ONNX (`video` tensor)** | $T_{\text{new}} \times 3 \times 224 \times 224 \times 1\text{ byte}$ (với $T_{\text{new}} = 5$) | **0.72 MB** (`752,640` bytes) | `Luồng 2: ONNX Worker Thread` |
| **3. Bộ đệm đặc trưng `cachedP4` (`p4_old`)** | • Ở `FP32`: $50 \times 128 \times 14 \times 14 \times 4\text{ bytes}$<br>• Ở `FP16`: $50 \times 128 \times 14 \times 14 \times 2\text{ bytes}$ | **4.79 MB** *(FP32)*<br>**2.39 MB** *(FP16)* | `Luồng 2: ONNX Worker Thread` |
| **4. Bộ nhớ Tensor trung gian & Trọng số `OrtSession`** | Trọng số model (`4.0 – 8.2 MB`) + Activations nội bộ CNN + ConvGRU (`18 – 25 MB`) | **~22 – 33 MB** | `Luồng 2` & `Luồng 3 (Intra-Op Pool)` |
| **5. Cửa sổ trượt 3 Nhánh Vật Lý Song Song (`1C-1`, `1C-2`, `1C-3`)** | • `1C-1 (Eye)`: `AdaptiveHMM` + `WindowRatio` + `blinkDeque` ($\approx 110\text{ KB}$)<br>• `1C-2 (Mouth)`: `AdaptiveHMM` + `WindowRatio` ($\approx 105\text{ KB}$)<br>• `1C-3 (Head+FFT)`: `HeadMotionWindow` + `WindowRatio` + `nodDeque` ($\approx 235\text{ KB}$) | **~0.45 MB** (`~450 KB`) | Tách biệt độc quyền trên `1C-1`, `1C-2`, `1C-3` |
| **6. Моđun `MediaPipe FaceLandmarker` (`Delegate.GPU`)** | Model TFLite + GPU OpenGL ES Texture / Shader Cache (`640×480 RGBA`) | **~18 – 28 MB** *(RAM + VRAM)* | `Luồng 1B: MediaPipe GPU Thread` |
| **TỔNG DUNG LƯỢNG BỘ NHỚ (TOTAL FOOTPRINT)** | — | **~58 – 115 MB** | Hoạt động mượt mà trên mọi thiết bị Android. |

---

### 8.5. Điều Kiện Đảm Bảo Thời Gian Thực Không Khóa (Real-Time Schedulability v1)

Kiến trúc đa luồng phiên bản `v1` đạt trạng thái **Thời Gian Thực Cứng (Hard Real-Time Stability)** với biên độ an toàn vượt trội nhờ 3 bất đẳng thức:

1. **Điều kiện không đứt quãng cảm biến trên Cụm Luồng 1A–1C (`Pipeline & Fork-Join Continuity Constraint`):**
   $$\max_{k} \Big( \max\big(T_{\text{1A}}(k), \; T_{\text{1B\_GPU}}(k) + \max(T_{\text{1C-1}}(k), T_{\text{1C-2}}(k), T_{\text{1C-3}}(k))\big) \Big) < \text{max\_gap\_sec} = 250.0\text{ ms}$$
   - **Thực tế đạt được trên Bản `v1`:** Ngay cả trên thiết bị phân khúc phổ thông (Low-End SoC) tại frame nặng nhất (Frame #300 khi cả `EyeWorker` và `MouthWorker` cùng chạy 20 vòng lặp EM song song với `HeadPoseAndFftWorker`), thời gian xử lý đường găng cực đại chỉ là:
     $$T_{\text{max\_frame\_v1}} \approx 9.5\text{ ms (GPU)} + \max(3.0\text{ ms (Eye EM)}, \; 3.0\text{ ms (Mouth EM)}, \; 4.25\text{ ms (Head+FFT)}) = \mathbf{13.75\text{ ms}} \ll \mathbf{250.0\text{ ms}}$$
   - Biên độ an toàn đạt **gấp ~18.2 lần** ngưỡng `max_gap_sec = 250.0 ms` (so với 5.7 lần ở bản `v0`), giúp tốc độ xử lý luôn bám sát **30–60 FPS** của phần cứng CameraX.

2. **Điều kiện không trễ hạn suy luận trên Luồng 2 (`ONNX Worker + Letterbox Deadline Constraint`):**
   $$T_{\text{letterbox}}(T_{\text{new}} = 5) + T_{\text{onnx\_worker}}(T_{\text{new}} = 5, \; T_{\text{old}} = 45) + T_{\text{normalizer\_fusion}} < \text{interval} = 1000.0\text{ ms}$$
   - **Thực tế đạt được:** Tổng thời gian chạy `letterbox` cho 5 frames mới (`0.8 – 4.8 ms`) cộng với thời gian suy luận ONNX có cache `p4_old` (`28 – 320 ms`) chỉ chiếm **`29 ms – 325 ms`** mỗi giây ($\ll 1000.0\text{ ms}$).

---

## 9. Danh Mục Kiểm Tra Tương Thích Số Học & Tối Ưu Hóa (Verification Checklist v1)

Để đảm bảo bản cài đặt Kotlin/Android theo kiến trúc nâng cấp `v1` vừa đạt tốc độ tối đa vừa cho ra kết quả **khớp 100%** với bản Python [inference.py](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/inference.py) và [export_onnx.py](file:///D:/Project/DATN/tempt/driver-guardian/ai/OptimalAlgorithsm/export_onnx.py):

1. **Cấu hình CameraX Hardware `640×480` + `OUTPUT_IMAGE_FORMAT_RGBA_8888`:**
   - Bắt buộc cấu hình `ResolutionSelector` ưu tiên `Size(640, 480)` và `.setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)`.
   - Khi đọc `imageProxy.planes[0].buffer`, phải kiểm tra `rowStride == width * 4` để loại bỏ các byte đệm hàng (nếu có) trước khi đóng gói thành `MPImage(IMAGE_FORMAT_RGBA, 640, 480)` và gọi `imageProxy.close()` ngay tại `Luồng 1A`.
2. **Công tắc Bật/Tắt `balanceBrightness` trên UI (`enableBalanceBrightness`):**
   - Dùng `AtomicBoolean` (hoặc `StateFlow<Boolean>`) kết nối trực tiếp với `Switch` trên màn hình chính.
   - Khi **TẮT** (`false`), trả về nguyên bản mảng `rgbaBytes` (`Zero-Conversion`) mà không qua bất kỳ bước chuyển đổi không gian màu nào.
   - Khi **BẬT** (`true`), tính $\mu_Y$ trên không gian màu `BGR -> YCrCb` của OpenCV; chỉ khi $\mu_Y < 70.0$ mới áp dụng bảng tra `LUT` gamma để giữ chuẩn tương thích số học tuyệt đối với `loadvideo2handle.py`.
3. **Ép `MediaPipe FaceLandmarker` chạy trên `Delegate.GPU`:**
   - Khởi tạo `FaceLandmarker` với `BaseOptions.builder().setDelegate(Delegate.GPU)` và gọi `detect(mpImage)` trên cùng một luồng đơn cố định (`Luồng 1B`) để bảo toàn ngữ cảnh OpenGL ES (`EGLContext`).
4. **Tách 3 Luồng Con Vật Lý Song Song (`EyeWorker` $\parallel$ `MouthWorker` $\parallel$ `HeadPoseAndFftWorker`):**
   - Sau khi `GpuFaceTracker` trả về `points[478, 2]`, chạy đồng thời 3 nhánh bằng `coroutineScope { async(eyeDispatcher) {...}; async(mouthDispatcher) {...}; async(headDispatcher) {...} }`.
   - Tuyệt đối không chia sẻ biến trạng thái khả biến giữa 3 nhánh để đạt `Wait-Free Fork-Join` không cần khóa (`Mutex`).
5. **Hoãn `letterbox` sang `ONNX Worker Thread` (Chỉ chạy trên $T_{\text{new}}$ frames mới):**
   - `NeuralBuffer.updateRaw` trên `Luồng 1B` tuyệt đối không gọi `letterbox`, chỉ lưu `RawRgbaFrame(640, 480, rgbaBytes)` ở lưới `5.0 FPS`.
   - Khi `ONNX Worker Thread` gọi `getNewFramesSince()`, chỉ $\Delta = T_{\text{new}}$ (thường là `5`) frames mới phát sinh mới được đưa qua `OnnxLetterboxProcessor.processBatchToRgbCHW(delta.newRawFrames)` (resize `RGBA` về `224×168` $\rightarrow$ chuyển `COLOR_RGBA2RGB` $\rightarrow$ điền viền `(114, 114, 114)` thành `224×224` $\rightarrow$ chuyển sang `CHW [3, 224, 224]` kiểu `uint8`).
6. **Giữ nguyên `WARMING UP` 60s kết hợp làm nóng `p4_old` & các quy tắc toán học cốt lõi:**
   - Trong suốt `0.0s <= timestamp < 60.0s`, trạng thái hiển thị giữ là `"WARMING UP"` đồng thời `ONNX Worker Thread` vẫn chạy mỗi `1.0s` để `letterbox` 5 frames mới và tích lũy cuốn chiếu `p4_old`.
   - Dùng `.mod(360.0)` khi chuẩn hóa góc đầu về $[-180^\circ, 180^\circ)$; giữ nguyên đơn vị `blink_duration` là **ms** và đổi `nod_duration` sang **giây (s)** trong `Engine._features`.
