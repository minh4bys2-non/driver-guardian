# PhysicalBranch Kotlin Module

Module nhận diện và giám sát các chỉ số thể chất (EAR, MAR, Head Pose, PERCLOS, Blinking, Yawning, Nodding) của tài xế được chuyển đổi từ Python sang **Kotlin**.

Tương thích hoàn toàn với hệ thống **Android Automotive**, **Android OS** và các ứng dụng chạy trên nền **Kotlin/JVM**.

---

## 1. Bảng đối chiếu Python vs Kotlin

| Python Source File | Kotlin Source File | Mô tả chức năng |
| :--- | :--- | :--- |
| `ai/interface.py` | [`AIResult.kt`](src/main/kotlin/com/driverguardian/ai/AIResult.kt) | Data Transfer Object chuẩn hóa kết quả AI, tuần tự hóa JSON (`toJson`) |
| `ai/PhysicalBranch/adaptive_hmm_fsm.py` | [`AdaptiveHMM.kt`](src/main/kotlin/com/driverguardian/ai/physical/AdaptiveHMM.kt) | Mô hình HMM thích ứng (1D K-Means, EM Baum-Welch, Online Adaptation) |
| `ai/PhysicalBranch/adaptive_hmm_fsm.py` | [`StateMachine.kt`](src/main/kotlin/com/driverguardian/ai/physical/StateMachine.kt) | Máy trạng thái hữu hạn FSM đếm tần suất và đo thời lượng sự kiện trong cửa sổ trượt |
| `ai/PhysicalBranch/adaptive_hmm_fsm.py` | [`WindowRatio.kt`](src/main/kotlin/com/driverguardian/ai/physical/WindowRatio.kt) | Tính toán tỷ lệ thời gian tích lũy theo cửa sổ trượt (PERCLOS, POM, OverAngle) |
| `ai/PhysicalBranch/adaptive_hmm_fsm.py` | [`AdaptiveHmmFsm.kt`](src/main/kotlin/com/driverguardian/ai/physical/AdaptiveHmmFsm.kt) | Bộ nhận diện phức hợp tự hiệu chuẩn kết hợp HMM + FSM + Ratio (cho mắt/miệng/đầu) |
| `ai/PhysicalBranch/head_pose_estimation.py` | [`HeadPoseEstimator.kt`](src/main/kotlin/com/driverguardian/ai/physical/HeadPoseEstimator.kt) | Ước lượng tư thế đầu 3D (Pitch, Yaw, Roll) từ các điểm mốc khuôn mặt |
| `ai/PhysicalBranch/PitchFSM.py` | [`PitchFsm.kt`](src/main/kotlin/com/driverguardian/ai/physical/PitchFsm.kt) | FSM động học góc Pitch, phát hiện gật gù, ngủ gục (Micro-sleep) |
| `ai/PhysicalBranch/camera_metrics.py` | [`CameraMetrics.kt`](src/main/kotlin/com/driverguardian/ai/physical/CameraMetrics.kt) | Điều phối toàn bộ các chỉ số vật lý, hiệu chuẩn tư thế trung hòa, xuất `AIResult` |
| *(Tiện ích toán học)* | [`MathUtils.kt`](src/main/kotlin/com/driverguardian/ai/physical/MathUtils.kt) | Phân vị (percentile), trung vị (median), phương sai, LogSumExp, Rodrigues, Euler |
| *(Mô hình dữ liệu)* | [`Models.kt`](src/main/kotlin/com/driverguardian/ai/physical/Models.kt) | `Point2D`, `Point3D`, `HeadPoseAngles`, `FsmMode`, `PositiveState`, v.v. |

---

## 2. Cấu trúc thư mục

```text
PhysicalBranch_kotlin/
├── build.gradle.kts           # Cấu hình Gradle module độc lập
├── settings.gradle.kts        # Tên project module
├── README.md                  # Hướng dẫn chi tiết
└── src/
    ├── main/kotlin/com/driverguardian/ai/
    │   ├── AIResult.kt        # Interface DTO chuẩn
    │   └── physical/
    │       ├── Models.kt
    │       ├── MathUtils.kt
    │       ├── AdaptiveHMM.kt
    │       ├── StateMachine.kt
    │       ├── WindowRatio.kt
    │       ├── AdaptiveHmmFsm.kt
    │       ├── HeadPoseEstimator.kt
    │       ├── PitchFsm.kt
    │       └── CameraMetrics.kt
    └── test/kotlin/com/driverguardian/ai/physical/
        └── PhysicalBranchTest.kt # Test suite kiểm chứng thuật toán
```

---

## 3. Điểm cải tiến và đặc tính nổi bật của phiên bản Kotlin

1. **Thuần Kotlin (Pure Kotlin Math)**:
   - Các thuật toán chính (K-Means 1D, HMM Baum-Welch EM, State Machine, Window Ratio, Direct 3D Geometry Head Pose) được viết hoàn toàn bằng toán học thuần túy trên Kotlin.
   - **Không bắt buộc** phải nạp các thư viện C++ native cồng kềnh (`.so` của OpenCV) nếu chạy trên môi trường Android nhẹ hoặc xe hơi bị hạn chế tài nguyên.
   - Vẫn hỗ trợ mở rộng kết nối OpenCV Java/Android khi có sẵn.

2. **Xử lý triệt để khác biệt toán học Python vs JVM**:
   - Toán tử chia lấy dư `%` trong Python là mathematical modulo (luôn dương với số chia dương), trong khi Java/Kotlin là remainder (giữ dấu của số bị chia). Đã được chuẩn hóa qua hàm `MathUtils.wrapAngle180` và `MathUtils.mod`, loại bỏ lỗi sai lệch góc khi quay đầu.
   - Hàm `MathUtils.percentile` khớp 100% kết quả nội suy tuyến tính của `numpy.percentile(x, p, method='linear')`.
   - Hàm `logSumExp` ổn định số học, loại bỏ hiện tượng Underflow/Overflow.

3. **Tương thích hoàn toàn với `AIResult`**:
   - `CameraMetricsResult.toAIResult()` chuyển đổi trực tiếp sang đối tượng `AIResult`.
   - `AIResult.toJson()` xuất JSON có các trường `snake_case` chính xác theo chuẩn của backend và các mô-đun AI khác.

---

## 4. Hướng dẫn tích hợp vào Android Automotive

### Cách 1: Tích hợp dưới dạng Subproject / Module trong Gradle

1. Trong file `settings.gradle` hoặc `settings.gradle.kts` của project Android:
   ```kotlin
   include(":PhysicalBranch_kotlin")
   project(":PhysicalBranch_kotlin").projectDir = file("ai/PhysicalBranch_kotlin")
   ```

2. Trong file `app/build.gradle.kts`:
   ```kotlin
   dependencies {
       implementation(project(":PhysicalBranch_kotlin"))
   }
   ```

### Cách 2: Sao chép mã nguồn trực tiếp
Có thể sao chép thư mục `src/main/kotlin/com/driverguardian/ai` vào trực tiếp thư mục mã nguồn của ứng dụng Android (`app/src/main/java` hoặc `app/src/main/kotlin`).

---

## 5. Ví dụ sử dụng trong mã nguồn

### Ví dụ 1: Xử lý Landmark và lấy kết quả AIResult

```kotlin
import com.driverguardian.ai.physical.CameraMetrics
import com.driverguardian.ai.physical.Point2D

// Khởi tạo CameraMetrics
val metricsCoordinator = CameraMetrics(
    fps = 30.0,
    windowSec = 60.0,
    calibrationFrames = 30
)

// Khi nhận được frame và MediaPipe FaceMesh landmarks (mảng Point2D):
fun onFrameAnalyzed(landmarks: Array<Point2D>, width: Double, height: Double, timestampSec: Double) {
    val result = metricsCoordinator.processLandmarks(
        points = landmarks,
        imageWidth = width,
        imageHeight = height,
        timestampSec = timestampSec
    )

    // Chuyển đổi thành AIResult chuẩn
    val aiResult = result.toAIResult(
        windowSec = 60.0,
        lstmDrowsinessProbability = null, // Điền nếu có mô hình LSTM
        warningScore = null
    )

    // Xuất chuỗi JSON gửi qua WebSocket/REST/MessageQueue
    val jsonString = aiResult.toJson()
    println("AI Output: $jsonString")
    
    // Đọc trực tiếp các chỉ số
    println("EAR: ${result.ear}, PERCLOS: ${result.perclosPct}%")
    println("Blink rate: ${result.blinkRatePerMin} blinks/min")
    println("Pitch: ${result.pitchDeg} deg, Nodding: ${result.nodding}")
}
```

### Ví dụ 2: Dùng trực tiếp DirectGeometryHeadPoseEstimator

```kotlin
import com.driverguardian.ai.physical.DirectGeometryHeadPoseEstimator
import com.driverguardian.ai.physical.Point3D

val estimator = DirectGeometryHeadPoseEstimator(imageWidth = 640.0, imageHeight = 480.0)

// 5 điểm mốc 3D từ MediaPipe: 1 (mũi), 152 (cằm), 10 (trán), 263 (mắt trái), 33 (mắt phải)
val keyPoints = mapOf(
    1 to Point3D(0.5, 0.5, 0.0),
    152 to Point3D(0.5, 0.7, 0.02),
    10 to Point3D(0.5, 0.3, 0.02),
    263 to Point3D(0.6, 0.4, 0.01),
    33 to Point3D(0.4, 0.4, 0.01)
)

val angles = estimator.estimate(keyPoints)
println("Pitch: ${angles.pitch}°, Yaw: ${angles.yaw}°, Roll: ${angles.roll}°")
```

### Ví dụ 3: Giám sát động học gật đầu với PitchFsm

```kotlin
import com.driverguardian.ai.physical.PitchFsm

val pitchFsm = PitchFsm(
    pitchDownThreshold = 14.0, // Ngưỡng cúi đầu
    nodMinDurationMs = 800.0,  // Tối thiểu 800ms tính là gật ngủ
    nodMaxDurationMs = 3500.0  // Quá 3.5s tính là ngủ gục sâu
)

fun onHeadPoseUpdated(pitch: Double, yaw: Double, roll: Double, timestampSec: Double) {
    val fsmResult = pitchFsm.update(pitch, yaw, roll, timestampSec)
    
    if (fsmResult.isCritical) {
        // Cảnh báo âm thanh khẩn cấp cho tài xế!
        println("NGUY HIỂM: ${fsmResult.eventLog}")
    } else if (fsmResult.isWarning) {
        println("CẢNH BÁO: ${fsmResult.eventLog}")
    }
}
```
