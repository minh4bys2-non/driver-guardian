# 📱 PROJECT CONTEXT: DRIVER GUARDIAN (ANDROID / ANDROID AUTOMOTIVE)

## 🎯 Tổng quan dự án

Ứng dụng Android **"Driver Guardian"** — Hệ thống giám sát và cảnh báo tình trạng buồn ngủ của tài
xế theo thời gian thực (Driver Monitoring System - DMS), thuộc Đồ án Tốt nghiệp tại Học viện Công
nghệ Bưu chính Viễn thông – Cơ sở tại TP.HCM (PTITHCM).

- **Mục tiêu:** Giám sát người lái trên thiết bị di động và môi trường ô tô thông minh (Android
  Automotive, chế độ màn hình ngang `landscape`), phát hiện sớm các dấu hiệu mệt mỏi, vi ngủ
  (*microsleep*), chớp mắt bất thường, ngáp và gật đầu bằng On-device Edge AI, đồng thời đồng bộ
  lịch sử chuyến đi và sự kiện cảnh báo với hệ thống Backend trung tâm.
- **Pipeline giám sát thời gian thực (Production):**
  `CameraX` → `MediaPipe Face Landmarker` (`face_landmarker.task` - 478 điểm mốc 3D) →
  `PhysicalBranchProcessor` → `PhysicalMetrics` (EAR, MAR, PERCLOS, POM, Head Pose 3D, Nodding,
  Adaptive HMM + Temporal FSM).
- **Ranh giới AI & Demo (Strict AI Boundary):**
    - `face_landmarker.task` chỉ trích xuất tọa độ mốc khuôn mặt, **không phải** model phân loại
      buồn ngủ.
    - Suy luận mạng học sâu ONNX (`OnnxRuntimeEngine`, `EndToEndDemoInferenceRunner`) hiện được tách
      biệt trong màn hình `OnnxDemoScreen`, không tự động kích hoạt cảnh báo giả trong phiên lái
      thực tế.
    - Luồng cảnh báo nguy hiểm hiện tại được kích hoạt tường minh qua nút `DEMO: Cảnh báo nguy hiểm`
      (`DangerAlertScreen`) để kiểm thử trọn vẹn chu trình sự kiện mà không bịa đặt điểm số AI giả.

---

## 🛠 Tech Stack & Kiến trúc

- **Ngôn ngữ:** Kotlin (JVM Target 17).
- **SDK:** `minSdk = 29`, `targetSdk = 36`, `compileSdk = 36`.
- **Giao diện (UI):** Jetpack Compose (Material 3, `Navigation Compose`, thiết kế tối ưu cho màn
  hình ngang `landscape` và bảng điều khiển xe hơi với `AppSidebar` / `BottomAppNavigation`).
- **Camera & Thị giác máy tính:**
    - `CameraX` (`camera-core`, `camera-camera2`, `camera-lifecycle`, `camera-view` v1.5.3).
    - `Google MediaPipe Tasks Vision` (`tasks-vision:1.0.0`) chạy chế độ `LIVE_STREAM`.
- **Suy luận AI trên thiết bị (On-device AI):**
    - Nhánh vật lý (`ai/physical`): Toán học hình học 3D (EAR, MAR, SolvePnP Head Pose),
      `AdaptiveHmm`, `TemporalStateMachine`, `WindowRatio` (khớp số học với nhánh Python gốc).
    - Nhánh học sâu (`ai/runtime`, `ai/demo`): `ONNX Runtime Android`
      (`onnxruntime-android:1.30.0`).
- **Mạng, Xác thực & Bảo mật:**
    - `Retrofit 2` (`2.11.0`) + `Gson Converter` kết nối với **FastAPI Backend** và **Oracle
      Database**.
    - `Android Credential Manager` (`androidx.credentials:1.3.0`) + `Google ID` (`googleid:1.1.1`)
      xác thực Google OAuth2 để đổi lấy JWT Access Token & Refresh Token từ Backend.
    - `EncryptedSharedPreferences` (`security-crypto`, chuẩn mã hóa AES-256-GCM trên Android
      Keystore) để lưu trữ an toàn Refresh Token và thông tin phiên.
    - `AuthInterceptor` tự động đính kèm `Authorization: Bearer <token>`, xử lý lỗi HTTP 401, đồng
      bộ hóa việc làm mới token (`/auth/refresh`) và tự động thử lại request.
- **Kiến trúc tổng thể:** **MVVM + Clean Architecture phân tầng** (`ui` ↔ `domain` ↔ `data` + `ai`),
  quản lý trạng thái bất đồng bộ bằng Kotlin Coroutines và `StateFlow`.

---

## 🖼 Quy trình xử lý UI (BẮT BUỘC)

Khi tạo mới hoặc chỉnh sửa một màn hình / component giao diện trong dự án, bạn phải tuân thủ các
bước sau:

1. **Jetpack Compose Thuần Túy:**
    - Xây dựng UI hoàn toàn bằng các hàm `@Composable` sử dụng Material 3
      (`androidx.compose.material3`). **Không** tạo các file XML Layout truyền thống trừ khi cấu
      hình hệ thống bắt buộc.
2. **Tách biệt Trạng thái (State Hoisting):**
    - Màn hình (`*Screen.kt`) nhận trạng thái từ `UiState` (thông qua
      `collectAsStateWithLifecycle()`) và truyền các sự kiện người dùng lên `ViewModel` qua các hàm
      callback (`onAction: () -> Unit`).
3. **Tái sử dụng Component & Hệ thống Theme:**
    - Ưu tiên tái sử dụng các component có sẵn trong `ui/components/` (`AppSidebar`,
      `DashboardCard`, `MetricCard`, `PrimaryActionButton`, `StatusIndicator`, `AiAnalyticsCharts`).
    - Sử dụng bảng màu và kiểu chữ từ `DriverGuardianTheme` (`ui/theme/Color.kt`, `Theme.kt`,
      `Type.kt` như `SafeGreen`, `WarningYellow`, `DangerRed`, `AccentBlue`). Không tự ý hardcode
      các mã màu rời rạc bên ngoài hệ thống theme.
4. **Hỗ trợ Đa kích thước & Màn hình Ngang (Automotive / Landscape Responsive):**
    - Giao diện phải tương thích tốt với bố cục ngang (`landscape`) trên màn hình xe hơi/máy tính
      bảng (`maxWidth >= 720.dp` hiển thị `AppSidebar`) cũng như màn hình nhỏ hơn
      (`BottomAppNavigation`).

---

## 🏗 Tổ chức thư mục (Package Structure)

Mã nguồn đặt tại `app/src/main/java/com/example/driverguardian` và được phân chia rõ ràng theo từng
package:

- `com.example.driverguardian.ai`: Toàn bộ lõi xử lý AI và giám sát thời gian thực trên thiết bị
    - `ai.contract`: Hợp đồng đầu vào/đầu ra của model (`ModelContract`, `DrowsinessModelContract`,
      `EndToEndDemoModelContract`).
    - `ai.demo`: Pipeline tiền xử lý, bộ đệm chuỗi `BoundedSequenceBuffer` và bộ chạy suy luận cho
      bản demo ONNX.
    - `ai.monitoring`: Điều phối luồng giám sát thời gian thực (`DriverMonitoringEngine`,
      `DriverMonitoringCoordinator`), thu nhận khung hình CameraX (`camera/`), cảm biến chuyển động
      (`device/`), đồng bộ thời gian (`time/`, `sync/`) và trạng thái chẩn đoán (`model/`).
    - `ai.parity`: Bộ công cụ kiểm thử đối sánh độ chính xác số học (Golden Vectors) giữa Kotlin và
      Python.
    - `ai.physical`: Nhánh phân tích đặc trưng vật lý (`AspectRatioCalculator`, `HeadPoseEstimator`,
      `AdaptiveHmm`, `TemporalStateMachine`, `WindowRatio`, `PhysicalBranchProcessor`).
    - `ai.runtime`: Lớp bọc quản lý phiên chạy `OnnxRuntimeEngine` và `OnnxModelSession`.
    - `ai.tensor`: Cấu trúc dữ liệu Tensor, shape và tiện ích tạo dữ liệu kiểm thử.
- `com.example.driverguardian.data`: Tầng dữ liệu, xác thực và kết nối mạng
    - `data.auth`: Quản lý đăng nhập Google (`GoogleAuthManager`), lưu trữ token bảo mật
      (`TokenStore`), quản lý phiên (`SessionManager`) và `AuthInterceptor`.
    - `data.remote`: Cấu hình Retrofit (`ApiClient`), định nghĩa endpoint (`DriverGuardianApi`) và
      các đối tượng truyền tải dữ liệu (`dto/ApiDtos.kt`).
    - `data.repository`: Triển khai logic truy xuất dữ liệu và ánh xạ từ DTO sang Domain
      (`AuthRepository`, `DriverGuardianRepository`).
- `com.example.driverguardian.domain.model`: Chứa các model nghiệp vụ thuần Kotlin (`Driver`,
  `Vehicle`, `DrivingSession`, `DrowsinessEvent`, `UserProfile`, v.v.).
- `com.example.driverguardian.ui`: Tầng giao diện người dùng (Jetpack Compose) và ViewModel
    - `ui.auth`: `AuthViewModel` quản lý trạng thái xác thực toàn cục.
    - `ui.components`: Các thành phần giao diện dùng chung (`AppSidebar`, `DashboardCard`,
      `MetricCard`, `PrimaryActionButton`, `StatusIndicator`, `AiAnalyticsCharts`).
    - `ui.history`: `TripHistoryViewModel` và định dạng hiển thị lịch sử chuyến đi.
    - `ui.monitoring`: `MonitoringViewModel`, `CameraPreview` (tích hợp CameraX `PreviewView` vào
      Compose) và bảng debug thông số vật lý.
    - `ui.navigation`: Cấu hình điều hướng `AppNavigation` và định nghĩa các route trong
      `Screen.kt`.
    - `ui.screens`: Chứa toàn bộ các màn hình chức năng (`auth/`, `home/`, `selection/`, `pretrip/`,
      `driving/`, `summary/`, `history/`, `analytics/`, `alerts/`, `settings/`, `system/`,
      `onnxdemo/`).
    - `ui.session`: `DrivingSessionViewModel` và `DrivingSessionUiState` quản lý vòng đời phiên lái
      xe.
    - `ui.theme`: Cấu hình màu sắc, kiểu chữ và Material 3 Theme (`Color.kt`, `Theme.kt`,
      `Type.kt`).

---

## 🧠 Nguyên tắc Coding (AI Instructions)

- **Đơn giản, Rõ ràng & Chuẩn Kotlin:** Viết code idiomatic Kotlin (sử dụng `data class`,
  `sealed interface`/`sealed class`, `StateFlow`, Coroutines, null-safety). Tránh lạm dụng `!!` hoặc
  các kỹ thuật phức tạp không cần thiết.
- **Trung thực về Dữ liệu & AI (Strict Honesty):**
    - Không tự gán các giá trị mặc định giả (fake defaults) cho các chỉ số trong `PhysicalMetrics`
      khi chưa phát hiện được khuôn mặt hoặc đang trong giai đoạn hiệu chuẩn (`null` phải được giữ
      nguyên để UI hiển thị trạng thái chờ chính xác).
    - Tuyệt đối không tạo điểm số phân loại AI giả mạo để tự động kích hoạt cảnh báo khi chưa có
      model phân loại được nạp.
- **Đảm bảo Hiệu năng Thời gian thực (Real-time Performance):**
    - Mọi tác vụ nặng (phân tích ảnh CameraX, MediaPipe Face Landmarker, tính toán nhánh vật lý, suy
      luận ONNX, gọi API) phải chạy trên background thread/dispatcher phù hợp, không gây nghẽn
      Main/UI Thread.
    - Quản lý chặt chẽ vòng đời tài nguyên (`CameraX`, `FaceLandmarker`, `OrtSession`) để tránh rò
      rỉ bộ nhớ (memory leak).
- **Bảo mật & Phân quyền (`USER` ≠ `DRIVER`):**
    - Không lưu trữ thông tin nhạy cảm (mật khẩu, khóa bí mật, token thô) trực tiếp trong mã nguồn
      hoặc commit file `.env` / URL cục bộ của máy cá nhân.
    - Tuân thủ ranh giới `USER` ≠ `DRIVER`: Tài khoản đăng nhập cần được liên kết với hồ sơ tài xế
      (`DRIVER_ID`) thông qua `/auth/me` hoặc `/drivers/me` trước khi cho phép bắt đầu phiên lái xe.
- **Kiểm thử & Khớp số học (Unit Test & Parity):**
    - Khi thay đổi logic trong `ai/physical`, `ai/monitoring`, `data` hoặc `ui`, cần đảm bảo bộ kiểm
      thử đơn vị (`.\gradlew.bat :app:testDebugUnitTest`) và các bài test parity với golden vectors
      của Python (`physical_fixtures.json`, `hmm_golden.json`, `head_pose_golden.json`) luôn vượt
      qua.
- **Tuyệt đối không hardcode:** những config liên quan tới xử lý liên quan tới AI của từng nhánh,
  package thì tập trung config ở một file ở nhánh, package đó
- **Không dùng lệnh cmd tùy tiện** chỉ dùng lệnh cmd khi cần thiết, không được dùng cmd để thêm và
  sửa file code và cấu hình

---

## 📋 Nguyên tắc khi làm việc

- Sau mỗi bước thực hiện phải giải thích những gì chuẩn bị làm.
- Chỉ thực hiện đúng yêu cầu được đưa ra, không tự ý thêm bất kỳ tính năng ngoài phạm vi nào khác.
- Mỗi lần trả lời phải nói rõ những gì chuẩn bị làm trước khi thực hiện.
- Trả lời mặc định bằng **tiếng Việt**.
- Nếu có yêu cầu về code trong module Android thì phải trả lời bằng code **Kotlin** (và Jetpack
  Compose / Gradle KTS khi cần), không trả lời bằng pseudo-code.
- Tham khảo các tài liệu kỹ thuật trong dự án để nắm rõ ngữ cảnh khi cần:
    - Tổng quan toàn hệ thống & AI: `../../README.md`
    - Hướng dẫn cài đặt & chạy Android / Backend / Oracle DB: `README.md`
    - Phân tích chi tiết kiến trúc Android: `architecture_analysis.md`
    - Đặc tả tối ưu hóa đa luồng trên phần cứng Snapdragon 680: `snapdragon_680_v2.md`
    - Kiến trúc xác thực & Ranh giới AI: `../docs/AUTHENTICATION_AND_AI_BOUNDARY.md`
    - Tài liệu API Backend: `../backend/README.md` và
      `app/src/main/java/com/example/driverguardian/data/remote/DriverGuardianApi.kt`
- Dự án đã khai báo quyền `INTERNET` và `CAMERA` trong `AndroidManifest.xml`, sử dụng
  `BuildConfig.API_BASE_URL` (mặc định `http://127.0.0.1:8000/` kết hợp
  `adb reverse tcp:8000 tcp:8000`) và `BuildConfig.GOOGLE_SERVER_CLIENT_ID`.
- Không sử dụng `git add -f` để commit các file mô hình nhị phân `.onnx` bị loại trừ trong
  `.gitignore`.
