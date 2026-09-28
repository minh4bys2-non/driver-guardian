# Driver Guardian — Kiến Trúc Hệ Thống, Tổ Chức File, Class & Phụ Thuộc Module

Tài liệu này mô tả chi tiết toàn bộ kiến trúc hệ thống **Driver Guardian (Android System)**, cấu tạo từng module, tổ chức file/class, cách thức import và luồng dữ liệu thông qua các biểu đồ **Mermaid**.

---

## 1. Tổng Quan Kiến Trúc Toàn Hệ Thống

Hệ thống Driver Guardian được thiết kế theo mô hình **Edge AI kết hợp Giám sát Thời Gian Thực (Real-time Driver Monitoring System - DMS)**. Ứng dụng chạy trên thiết bị di động và môi trường ô tô thông minh (**Android Automotive / Tablet HUD**), trực tiếp thực thi các mô hình học máy (ONNX Runtime) trên phần cứng biên (On-device Edge AI) để bảo vệ quyền riêng tư và tối ưu độ trễ cảnh báo.

```mermaid
flowchart TB
    subgraph EdgeDevice ["📱 Thiết Bị Biên (Android / Android Automotive)"]
        direction TB
        Camera["📷 Camera Cabin (RGB Stream)"] --> Preprocess["⚙️ Tiền Xử Lý Khung Hình (Pre-processing)"]
        
        subgraph AIPipeline ["🧠 Phân Hệ AI & Suy Luận Biên (com.example.driverguardian.ai)"]
            Preprocess --> BNeck["BackboneNeckPipeline<br/>(backbone_neck.onnx)"]
            BNeck --> PFeatures["Trích Xuất Đặc Trưng<br/>(P3, P4, P5 Feature Maps)"]
            PFeatures --> FaceAnalysis["Phân Tích Khuôn Mặt & Động Thái<br/>(EAR, MAR, Head Pose)"]
            FaceAnalysis --> DrowsinessContract["DrowsinessModelContract<br/>(Kiểm tra Hợp đồng Tensor)"]
            DrowsinessContract --> OnnxEngine["OnnxRuntimeEngine & OnnxModelSession<br/>(onnxruntime-android 1.30.0)"]
        end

        subgraph UILayer ["🖥️ Phân Hệ Giao Diện (com.example.driverguardian.ui)"]
            OnnxEngine --> StateEngine["Quản Lý Trạng Thái & Chẩn Đoán<br/>(StateFlow / UiState)"]
            StateEngine --> ComposeNav["AppNavigation & NavHost<br/>(Jetpack Compose Navigation)"]
            ComposeNav --> Screens["13 Màn Hình Nghiệp Vụ<br/>(Driving, Danger Alert, Analytics, Demo...)"]
        end
    end

    subgraph BackendSystem ["☁️ Hệ Thống Máy Chủ Trung Tâm"]
        Screens -. "Telemetry & Cảnh Báo (Đang phát triển)" .-> BackendAPI["🚀 REST / WebSocket API Gateway"]
        BackendAPI --> CentralDB[("🗄️ Cơ Sở Dữ Liệu Oracle")]
    end
```

---

## 2. Cấu Trúc Tổ Chức Thư Mục & Tập Tin (File Organization)

Hệ thống mã nguồn Android được tổ chức trong module gốc `:app` với gói package chuẩn `com.example.driverguardian`, phân chia ranh giới rõ rệt giữa tầng giao diện (`ui`), tầng suy luận trí tuệ nhân tạo (`ai`), tài nguyên mẫu (`assets`), công cụ phát triển (`scripts`) và kiểm thử (`test`).

### 2.1. Sơ Đồ Cây Thư Mục Trực Quan

```mermaid
flowchart TB
    Root["📂 system/android"]
    Root --> GradleConfigs["📄 build.gradle.kts / settings.gradle.kts"]
    Root --> ScriptsDir["📂 scripts/"]
    Root --> AppDir["📂 app/"]

    ScriptsDir --> GenRef["🐍 generate_backbone_neck_reference.py"]

    AppDir --> AppGradle["📄 build.gradle.kts (SDK 36, ONNX, Compose)"]
    AppDir --> SrcDir["📂 src/"]

    SrcDir --> MainDir["📂 main/"]
    SrcDir --> TestDir["📂 test/"]

    TestDir --> TestAI["📂 com/example/driverguardian/ai/"]
    TestAI --> BBTest["📄 BackboneNeckContractTest.kt"]
    TestAI --> OnnxTest["📄 OnnxFoundationTest.kt"]

    MainDir --> Manifest["📄 AndroidManifest.xml"]
    MainDir --> AssetsDir["📂 assets/"]
    MainDir --> JavaDir["📂 java/com/example/driverguardian/"]

    AssetsDir --> ModelsDir["📂 models/ (backbone_neck.onnx)"]
    AssetsDir --> VectorsDir["📂 onnx_test_vectors/ (manifest.json)"]

    JavaDir --> MainActivity["📄 MainActivity.kt (Điểm vào Activity)"]
    JavaDir --> UIPackage["📂 ui/ (Tầng Giao Diện)"]
    JavaDir --> AIPackage["📂 ai/ (Tầng Trí Tuệ Nhân Tạo)"]

    UIPackage --> UINav["📂 navigation/ (AppNavigation.kt, Screen.kt)"]
    UIPackage --> UIScreens["📂 screens/ (13 Màn hình chức năng)"]
    UIPackage --> UIComp["📂 components/ (8 Component dùng chung)"]
    UIPackage --> UIMock["📂 mock/ (MockModels.kt, MockData.kt)"]
    UIPackage --> UITheme["📂 theme/ (Color.kt, Theme.kt, Type.kt)"]

    AIPackage --> AIRuntime["📂 runtime/ (Engine, Session, Diagnostics)"]
    AIPackage --> AITensor["📂 tensor/ (FloatTensor, Shape, Preview)"]
    AIPackage --> AIContract["📂 contract/drowsiness/ (Contract & Mapping)"]
    AIPackage --> AIDetection["📂 detection/ (BackboneNeckPipeline.kt)"]
    AIPackage --> AIParity["📂 parity/ (GoldenVector Runner & Loader)"]
```

### 2.2. Bảng Kê Chi Tiết Toàn Bộ File Trong Hệ Thống

| Nhóm / Package | Đường Dẫn File | Vai Trò & Trách Nhiệm Cốt Lõi |
| :--- | :--- | :--- |
| **Cấu hình & Điểm vào** | `system/android/build.gradle.kts` | Cấu hình Gradle root, plugin Kotlin & Android. |
| | `system/android/settings.gradle.kts` | Định nghĩa module `:app` và kho lưu trữ maven (Google, MavenCentral). |
| | `system/android/app/build.gradle.kts` | Định nghĩa compileSdk 36, Java 17, Compose BOM, `onnxruntime-android:1.30.0`. |
| | `app/src/main/AndroidManifest.xml` | Khai báo Single Activity (`MainActivity`), khóa hướng màn hình ngang (landscape). |
| | `MainActivity.kt` | Điểm vào duy nhất của app, thiết lập theme và khởi chạy `DriverGuardianApp`. |
| **UI - Navigation** | `ui/navigation/Screen.kt` | Sealed class định nghĩa các Route điều hướng an toàn (Home, Driving, OnnxDemo...). |
| | `ui/navigation/AppNavigation.kt` | Quản lý `NavHostController`, responsive layout (Sidebar >= 720dp, BottomBar < 720dp). |
| **UI - Screens** | `ui/screens/home/HomeScreen.kt` | Màn hình chính: tổng quan trạng thái hệ thống, nút bắt đầu chuyến đi. |
| | `ui/screens/selection/SelectionScreen.kt` | Chọn tài xế và phương tiện trước khi khởi hành. |
| | `ui/screens/pretrip/PreTripCheckScreen.kt` | Danh sách kiểm tra điều kiện an toàn (Camera, AI Model, Cảm biến, Mạng). |
| | `ui/screens/driving/ActiveDrivingScreen.kt` | Màn hình HUD lái xe: hiển thị chỉ số EAR, MAR, Head Pose, mức cảnh báo. |
| | `ui/screens/driving/DangerAlertScreen.kt` | Màn hình cảnh báo nguy hiểm mức cao nhất (toàn màn hình, ẩn sidebar). |
| | `ui/screens/summary/TripSummaryScreen.kt` | Báo cáo tóm tắt khi kết thúc chuyến đi (thời gian lái, số cảnh báo). |
| | `ui/screens/history/TripHistoryScreen.kt` | Danh sách lịch sử các chuyến đi đã hoàn thành. |
| | `ui/screens/history/TripDetailScreen.kt` | Chi tiết từng chuyến đi theo Route `trip_detail/{id}`. |
| | `ui/screens/alerts/AlertHistoryScreen.kt` | Lịch sử nhật ký các lần phát hiện buồn ngủ và cảnh báo. |
| | `ui/screens/analytics/AnalyticsScreen.kt` | Báo cáo thống kê xu hướng tập trung của tài xế theo tuần/tháng. |
| | `ui/screens/settings/SettingsScreen.kt` | Thiết lập ngưỡng cảnh báo âm thanh, độ nhạy phát hiện, cấu hình HUD. |
| | `ui/screens/system/SystemInfoScreen.kt` | Thông tin phiên bản phần mềm, phần cứng và lối tắt mở màn hình ONNX Demo. |
| | `ui/screens/onnxdemo/OnnxDemoScreen.kt` | Giao diện kiểm thử, chẩn đoán mô hình ONNX, chạy dummy input & parity test. |
| | `ui/screens/onnxdemo/OnnxDemoViewModel.kt` | ViewModel quản lý nạp mô hình, chạy suy luận ONNX, kiểm tra contract và parity. |
| **UI - Components** | `ui/components/AppSidebar.kt` | Thanh điều hướng bên sườn dành cho tablet / màn hình ô tô kích thước lớn. |
| | `ui/components/DashboardCard.kt` | Thẻ hiển thị thông tin dashboard với hiệu ứng nền bo góc. |
| | `ui/components/PrimaryActionButton.kt` | Nút bấm thao tác chính với trạng thái loading và icon. |
| | `ui/components/StatusIndicator.kt` | Chỉ báo trạng thái hoạt động (Xanh: Tốt, Vàng: Cảnh báo, Đỏ: Lỗi). |
| | `ui/components/AiAnalyticsCharts.kt` | Biểu đồ trực quan hóa dữ liệu AI (tần suất chớp mắt, tỷ lệ nhắm mắt P80). |
| | `ui/components/MockBarChart.kt` | Biểu đồ cột Canvas Compose phục vụ trực quan hóa dữ liệu mô phỏng. |
| | `ui/components/EmptyState.kt` | Giao diện hiển thị khi danh sách trống hoặc chưa có dữ liệu. |
| | `ui/components/MetricCard.kt` | Thẻ đo lường thông số vận hành (vận tốc, thời gian lái liên tục). |
| **UI - Mock & Theme**| `ui/mock/MockModels.kt` | Các cấu trúc dữ liệu UI: `DrivingUiState`, `AlertLevel`, `DriverUiModel`, v.v. |
| | `ui/mock/MockData.kt` | Dữ liệu mẫu phục vụ phát triển giao diện độc lập với backend. |
| | `ui/mock/AiAnalyticsMockData.kt` | Dữ liệu mẫu chuỗi thời gian cho đồ thị phân tích mệt mỏi. |
| | `ui/theme/Color.kt` | Bảng màu Material 3 chuyên dụng cho ô tô (SafeGreen, WarningYellow, DangerRed). |
| | `ui/theme/Theme.kt` | Cấu hình Dark/Light ColorScheme và Theme Wrapper `DriverGuardianTheme`. |
| | `ui/theme/Type.kt` | Cấu hình Typography chữ dễ đọc trong môi trường rung lắc trên ô tô. |
| **AI - Runtime** | `ai/runtime/OnnxRuntimeEngine.kt` | Lớp bọc cấp cao quản lý vòng đời một phiên ONNX (`loadModel`, `run`, `close`). |
| | `ai/runtime/OnnxModelSession.kt` | Lớp nội bộ đóng gói `OrtSession`, chuyển đổi FloatTensor sang bộ nhớ native. |
| | `ai/runtime/OrtTypeMapper.kt` | Ánh xạ kiểu dữ liệu giữa ONNX native `OnnxJavaType` và `RuntimeTensorType`. |
| | `ai/runtime/RuntimeMetadata.kt` | Cấu trúc dữ liệu metadata: `TensorMetadata`, `ModelMetadata`, `RuntimeDiagnostics`. |
| | `ai/runtime/RuntimeState.kt` | Sealed class biểu diễn trạng thái: NotLoaded, Loading, Ready, Running, Failed, Closed. |
| | `ai/runtime/RuntimeError.kt` | Sealed class định nghĩa hệ thống lỗi chi tiết và Result wrapper `RuntimeResult<T>`. |
| **AI - Tensor** | `ai/tensor/TensorData.kt` | Sealed interface `TensorData` và cài đặt `FloatTensor` lưu mảng số thực. |
| | `ai/tensor/TensorShape.kt` | Kiểm tra tính hợp lệ của Shape, chống tràn số học khi tính tổng phần tử. |
| | `ai/tensor/DummyTensorFactory.kt` | Tạo dữ liệu đầu vào giả lập (zero-filled tensor) để kiểm tra mô hình. |
| | `ai/tensor/TensorPreviewFactory.kt` | Trích xuất chuỗi xem trước (head/tail) dữ liệu tensor để hiển thị trên UI. |
| **AI - Contract** | `ai/contract/drowsiness/ModelContract.kt` | Interface thẩm định mô hình `ModelContract` và `ContractValidationResult`. |
| | `ai/contract/drowsiness/DrowsinessModelContract.kt` | Xác thực cấu trúc 4 tensor đầu vào (mắt, miệng, hình học) và 1 đầu ra. |
| | `ai/contract/drowsiness/DrowsinessInputMapping.kt` | Ánh xạ ngữ nghĩa tên node (Left Eye, Right Eye, Mouth, Geometry). |
| **AI - Detection** | `ai/detection/BackboneNeckPipeline.kt` | Pipeline nạp `backbone_neck.onnx`, trích xuất 3 mức đặc trưng không gian P3, P4, P5. |
| **AI - Parity** | `ai/parity/GoldenTestCase.kt` | Cấu trúc test case so sánh kết quả số học từ Python với Android. |
| | `ai/parity/GoldenVectorAssetLoader.kt` | Nạp file `manifest.json` và đọc tensor nhị phân little-endian từ assets. |
| | `ai/parity/GoldenVectorRunner.kt` | Bộ so sánh sai số số học tuyệt đối (`atol`) và tương đối (`rtol`). |
| | `ai/parity/NumericalTolerance.kt` | Cấu hình ngưỡng sai số chấp nhận được (`NumericalTolerance.Configured`). |
| **Scripts & Test** | `scripts/generate_backbone_neck_reference.py` | Kịch bản Python chạy PyTorch/ONNX sinh vector chuẩn tham chiếu (`.f32`). |
| | `app/src/test/ai/BackboneNeckContractTest.kt` | Unit test kiểm tra tính đúng đắn của hợp đồng Backbone+Neck. |
| | `app/src/test/ai/OnnxFoundationTest.kt` | Unit test kiểm tra chống tràn số, dummy tensor và kiểm tra hợp đồng. |

---

## 3. Cấu Tạo Chi Tiết Các Module & Subsystem

### 3.1. Phân Hệ Giao Diện Người Dùng (`ui`)
Tầng `ui` được xây dựng hoàn toàn bằng **Jetpack Compose**, tuân thủ nguyên tắc thiết kế Declarative UI:
- **Routing & Responsive**: [Screen.kt](file:///home/tranmanhduy/Workspace/ptithcm/driver-guardian/system/android/app/src/main/java/com/example/driverguardian/ui/navigation/Screen.kt) sử dụng sealed class giúp định tuyến kiểu an toàn (type-safe). [AppNavigation.kt](file:///home/tranmanhduy/Workspace/ptithcm/driver-guardian/system/android/app/src/main/java/com/example/driverguardian/ui/navigation/AppNavigation.kt) áp dụng `BoxWithConstraints` để tự động chuyển đổi giữa `AppSidebar` (màn hình rộng >= 720dp) và `NavigationBar` (màn hình nhỏ).
- **MVI / MVVM Trong OnnxDemo**: [OnnxDemoViewModel.kt](file:///home/tranmanhduy/Workspace/ptithcm/driver-guardian/system/android/app/src/main/java/com/example/driverguardian/ui/screens/onnxdemo/OnnxDemoViewModel.kt) kế thừa `AndroidViewModel`, quản lý luồng bất đồng bộ:
  - Đọc file mô hình từ Android Assets trên `Dispatchers.IO`.
  - Thực thi nạp session ONNX và tính toán suy luận trên `Dispatchers.Default`.
  - Phát trạng thái bất biến thông qua `StateFlow<OnnxDemoUiState>` tới giao diện Compose.

### 3.2. Phân Hệ Suy Luận AI Biên (`ai`)
Tầng `ai` hoàn toàn độc lập với Android UI và framework giao diện, có thể chạy trong môi trường Unit Test chuẩn JVM:
1. **Quản Lý Vòng Đời ONNX (`ai.runtime`)**:
   - [OnnxRuntimeEngine](file:///home/tranmanhduy/Workspace/ptithcm/driver-guardian/system/android/app/src/main/java/com/example/driverguardian/ai/runtime/OnnxRuntimeEngine.kt) áp dụng mẫu thiết kế Thread-Safe Wrapper với đối tượng khóa `synchronized(lock)`, đảm bảo không bao giờ xảy ra Race Condition khi vừa nạp mô hình vừa thực hiện suy luận.
   - [OnnxModelSession](file:///home/tranmanhduy/Workspace/ptithcm/driver-guardian/system/android/app/src/main/java/com/example/driverguardian/ai/runtime/OnnxModelSession.kt) chịu trách nhiệm cấp phát bộ nhớ Direct Buffer (`ByteBuffer.allocateDirect`), sắp xếp thứ tự byte gốc (`ByteOrder.nativeOrder()`) để chuyển dữ liệu vào C++ native code mà không bị chi phí sao chép bộ nhớ dư thừa. Tự động đóng tensor native trong khối `finally`.
2. **Trừu Tượng Hóa Dữ Liệu Tensor (`ai.tensor`)**:
   - [FloatTensor](file:///home/tranmanhduy/Workspace/ptithcm/driver-guardian/system/android/app/src/main/java/com/example/driverguardian/ai/tensor/TensorData.kt) chứa mảng `FloatArray` phẳng kèm theo danh sách chiều `shape: List<Long>`.
   - [TensorShape](file:///home/tranmanhduy/Workspace/ptithcm/driver-guardian/system/android/app/src/main/java/com/example/driverguardian/ai/tensor/TensorShape.kt) ngăn chặn triệt để lỗi tràn số học (Integer/Long Overflow) khi nhân các chiều ma trận bằng hàm `Math.multiplyExact()`.
3. **Thẩm Định Hợp Đồng Mô Hình (`ai.contract`)**:
   - [DrowsinessModelContract](file:///home/tranmanhduy/Workspace/ptithcm/driver-guardian/system/android/app/src/main/java/com/example/driverguardian/ai/contract/drowsiness/DrowsinessModelContract.kt) giải quyết bài toán chống suy đoán sai node name: Vì 3 input chuỗi ảnh (mắt trái, mắt phải, miệng) đều có chung shape `[1, 30, 3, 64, 64]`, contract từ chối tự động ghép nối nếu không có ánh xạ tường minh (`Explicit Mapping`).
4. **Pipeline Trích Xuất Đặc Trưng (`ai.detection`)**:
   - [BackboneNeckPipeline](file:///home/tranmanhduy/Workspace/ptithcm/driver-guardian/system/android/app/src/main/java/com/example/driverguardian/ai/detection/BackboneNeckPipeline.kt) nạp mô hình `backbone_neck.onnx`, nhận đầu vào ảnh RGB chuẩn hóa `[B, 3, 640, 640]` và trích xuất đồng thời 3 bản đồ đặc trưng đa tỷ lệ:
     - `p3`: Shape `[B, 64, 80, 80]` (Đặc trưng tỷ lệ lớn, độ phân giải cao)
     - `p4`: Shape `[B, 128, 40, 40]` (Đặc trưng tỷ lệ trung bình)
     - `p5`: Shape `[B, 256, 20, 20]` (Đặc trưng tỷ lệ nhỏ, ngữ nghĩa cao)
5. **Kiểm Chứng Số Học Chuẩn PyTorch (`ai.parity`)**:
   - [GoldenVectorRunner](file:///home/tranmanhduy/Workspace/ptithcm/driver-guardian/system/android/app/src/main/java/com/example/driverguardian/ai/parity/GoldenVectorRunner.kt) tính toán sai số tuyệt đối $max|a - e|$ và sai số tương đối $max\frac{|a - e|}{|e|}$, so sánh với ngưỡng sai số cho phép ($atol + rtol \times |e|$) để khẳng định tính tương thích số học hoàn hảo giữa mô hình huấn luyện bằng Python và mô hình chạy trên Android.

---

## 4. Sơ Đồ Lớp (Class Diagrams)

### 4.1. Sơ Đồ Lớp Tầng Trí Tuệ Nhân Tạo & Runtime (`com.example.driverguardian.ai.*`)

```mermaid
classDiagram
    direction TB

    class Closeable {
        <<interface>>
        +close() void
    }

    class AutoCloseable {
        <<interface>>
        +close() void
    }

    class OnnxRuntimeEngine {
        -lock: Any
        -environment: OrtEnvironment
        -modelSession: OnnxModelSession
        -closed: Boolean
        +state: RuntimeState
        +diagnostics: RuntimeDiagnostics
        +modelMetadata(): ModelMetadata
        +loadModel(modelName: String, bytes: ByteArray): RuntimeResult
        +run(inputs: Map, captureFullOutputs: Boolean): RuntimeResult
        +recordFailure(error: RuntimeError): void
        +close(): void
    }

    class OnnxModelSession {
        -environment: OrtEnvironment
        -options: OrtSessionOptions
        -session: OrtSession
        +metadata: ModelMetadata
        +run(inputs: Map, captureFullOutputs: Boolean): RuntimeResult
        -validateInputs(inputs: Map): RuntimeError
        -captureFloats(tensor: OnnxTensor, captureFull: Boolean): Pair
        +close(): void
    }

    class TensorData {
        <<interface>>
        +shape: List_Long
        +type: RuntimeTensorType
    }

    class FloatTensor {
        +shape: List_Long
        +values: FloatArray
        +type: RuntimeTensorType
    }

    class TensorShape {
        <<object>>
        +checkedElementCount(shape: List_Long): Long
        +validateConcreteShape(shape: List_Long): Boolean
    }

    class ModelMetadata {
        +modelName: String
        +inputs: List_TensorMetadata
        +outputs: List_TensorMetadata
        +loadDurationNanos: Long
        +inputCount: Int
        +outputCount: Int
    }

    class RuntimeDiagnostics {
        +runtimeVersion: String
        +modelName: String
        +loadDurationNanos: Long
        +lastInferenceDurationNanos: Long
        +runCount: Long
        +lastError: RuntimeError
    }

    class ModelContract {
        <<interface>>
        +validate(metadata: ModelMetadata): ContractValidationResult
    }

    class DrowsinessModelContract {
        -mapping: DrowsinessInputMapping
        +validate(metadata: ModelMetadata): ContractValidationResult
    }

    class BackboneNeckPipeline {
        -engine: OnnxRuntimeEngine
        +metadata: ModelMetadata
        +run(images: FloatTensor): RuntimeResult
        +close(): void
        +load(assets: AssetManager)$ RuntimeResult
    }

    class BackboneNeckFeatures {
        +p3: FloatTensor
        +p4: FloatTensor
        +p5: FloatTensor
        +inferenceDurationNanos: Long
    }

    class GoldenVectorRunner {
        <<object>>
        +compare(testCase: GoldenTestCase, actual: RuntimeInferenceResult, tolerance: NumericalTolerance): ParityResult
    }

    Closeable <|.. OnnxRuntimeEngine
    Closeable <|.. BackboneNeckPipeline
    AutoCloseable <|.. OnnxModelSession
    TensorData <|.. FloatTensor
    ModelContract <|.. DrowsinessModelContract

    OnnxRuntimeEngine *-- OnnxModelSession
    OnnxRuntimeEngine --> RuntimeDiagnostics
    OnnxModelSession --> ModelMetadata
    BackboneNeckPipeline *-- OnnxRuntimeEngine
    BackboneNeckPipeline ..> BackboneNeckFeatures
    BackboneNeckPipeline ..> FloatTensor
    GoldenVectorRunner ..> FloatTensor
```

### 4.2. Sơ Đồ Lớp Tầng Giao Diện & Quản Lý Trạng Thái (`ui.*`)

```mermaid
classDiagram
    direction TB

    class MainActivity {
        +onCreate(savedInstanceState: Bundle): void
    }

    class Screen {
        <<sealed class>>
        +route: String
    }
    class Home { data object }
    class ActiveDriving { data object }
    class DangerAlert { data object }
    class TripDetail {
        +createRoute(id: String): String
    }
    class OnnxDemo { data object }

    Screen <|-- Home
    Screen <|-- ActiveDriving
    Screen <|-- DangerAlert
    Screen <|-- TripDetail
    Screen <|-- OnnxDemo

    class AndroidViewModel {
        <<framework>>
    }

    class OnnxDemoViewModel {
        -engine: OnnxRuntimeEngine
        -contract: DrowsinessModelContract
        -goldenSuite: LoadedGoldenSuite
        -mutableState: MutableStateFlow_OnnxDemoUiState
        +uiState: StateFlow_OnnxDemoUiState
        +loadModel(): void
        +runDummyInference(): void
        +runGoldenParity(): void
        +onCleared(): void
    }

    class OnnxDemoUiState {
        +runtimeState: RuntimeState
        +diagnostics: RuntimeDiagnostics
        +metadata: ModelMetadata
        +contract: ContractValidationResult
        +lastInference: RuntimeInferenceResult
        +dummyAvailable: Boolean
        +goldenAvailable: Boolean
        +goldenResults: List_Pair
    }

    class DrivingUiState {
        +driverName: String
        +vehiclePlate: String
        +statusLabel: String
        +level: AlertLevel
        +prediction: PredictionUiModel
        +drivingTime: String
        +alertCount: Int
    }

    class AlertLevel {
        <<enumeration>>
        Safe
        Warning
        Danger
    }

    AndroidViewModel <|-- OnnxDemoViewModel
    MainActivity ..> Screen : uses navigation
    OnnxDemoViewModel --> OnnxDemoUiState
    DrivingUiState --> AlertLevel
```

---

## 5. Quy Tắc Import & Luồng Phụ Thuộc Module (Import Architecture)

### 5.1. Nguyên Tắc Phân Tầng Phụ Thuộc (Clean Architecture Principles)

Hệ thống tuân thủ nghiêm ngặt mô hình phụ thuộc một chiều (**Unidirectional Dependency Flow**):
1. **Độc lập Tầng AI (`ai` layer)**:
   - Các package `ai.runtime`, `ai.tensor`, `ai.contract`, `ai.detection`, `ai.parity` **tuyệt đối không import** bất kỳ class nào từ tầng `ui` hoặc framework Android UI (`androidx.compose.*`, `android.view.*`).
   - Phân hệ `ai` chỉ phụ thuộc vào `ai.onnxruntime.*`, Java/Kotlin Standard Library và một số ít tiện ích Android cơ bản như `android.content.res.AssetManager`.
2. **Tầng Giao Diện (`ui` layer)**:
   - Tầng `ui` quan sát và kích hoạt chức năng của tầng `ai` gián tiếp thông qua **ViewModel** (`OnnxDemoViewModel`) hoặc các Pipeline chuyên biệt.
   - Các màn hình UI nghiệp vụ chỉ làm việc với UI State (`DrivingUiState`, `OnnxDemoUiState`) và Data Model mô phỏng (`MockModels`).
3. **Không Phụ Thuộc Vòng Tròn (No Circular Dependencies)**:
   - Luồng dữ liệu và tham chiếu import luôn chảy từ tầng ngoài (UI) vào tầng trong (Domain/AI Runtime).

### 5.2. Biểu Đồ Ma Trận Import Giữa Các Gói Mã Nguồn

```mermaid
flowchart LR
    subgraph UI ["Tầng UI (com.example.driverguardian.ui)"]
        direction TB
        Main["MainActivity"]
        Navigation["navigation (AppNavigation, Screen)"]
        Screens["screens (Home, Driving, SystemInfo...)"]
        OnnxDemoVM["screens.onnxdemo (OnnxDemoViewModel)"]
        Components["components (DashboardCard, Sidebar...)"]
        Mock["mock (MockData, MockModels)"]
        Theme["theme (Color, Theme, Type)"]
    end

    subgraph AI ["Tầng AI (com.example.driverguardian.ai)"]
        direction TB
        Detection["detection (BackboneNeckPipeline)"]
        Contract["contract.drowsiness (DrowsinessModelContract)"]
        Parity["parity (GoldenVectorRunner, Loader)"]
        Runtime["runtime (OnnxRuntimeEngine, Session)"]
        Tensor["tensor (FloatTensor, TensorShape)"]
    end

    subgraph Vendor ["Thư Viện Ngoài & Native"]
        ORT["ai.onnxruntime (ONNX Runtime Native C++)"]
        Compose["androidx.compose (Material3, Runtime)"]
    end

    %% Quan hệ trong tầng UI
    Main --> Theme
    Main --> Navigation
    Navigation --> Screens
    Navigation --> Components
    Screens --> Components
    Screens --> Mock
    Screens --> Theme
    Screens --> OnnxDemoVM
    Components --> Theme

    %% Cầu nối giữa UI và AI
    OnnxDemoVM ==> Contract
    OnnxDemoVM ==> Parity
    OnnxDemoVM ==> Runtime
    OnnxDemoVM ==> Tensor

    %% Quan hệ nội bộ tầng AI
    Detection --> Runtime
    Detection --> Tensor
    Parity --> Runtime
    Parity --> Tensor
    Contract --> Runtime
    Runtime --> Tensor
    Runtime ==> ORT

    %% Phụ thuộc Vendor
    Main --> Compose
    Screens --> Compose
    Components --> Compose
```

### 5.3. Bảng Chi Tiết Quan Hệ Import Cụ Thể Từng Package

| Package Nguồn | Các Package Được Phép Import | Mục Đích Sử Dụng |
| :--- | :--- | :--- |
| `ui.navigation` | `ui.screens.*`, `ui.components.*`, `androidx.navigation.*` | Điều phối chuyển màn hình, gắn layout sidebar/bottom bar. |
| `ui.screens.onnxdemo` | `ai.contract.drowsiness.*`, `ai.parity.*`, `ai.runtime.*`, `ai.tensor.*` | Quản lý vòng đời mô hình, gửi input, nhận output suy luận và kết quả parity. |
| `ui.screens.*` | `ui.components.*`, `ui.mock.*`, `ui.theme.*`, `androidx.compose.*` | Hiển thị giao diện người dùng, đọc dữ liệu trạng thái. |
| `ai.detection` | `ai.runtime.*`, `ai.tensor.*`, `android.content.res.AssetManager` | Đóng gói pipeline trích xuất đặc trưng Backbone+Neck. |
| `ai.contract.drowsiness` | `ai.runtime.ModelMetadata`, `ai.runtime.RuntimeTensorType` | Kiểm tra tính tương thích giữa metadata mô hình ONNX và hợp đồng tensor. |
| `ai.parity` | `ai.runtime.RuntimeInferenceResult`, `ai.tensor.*`, `org.json.*` | Đọc manifest dữ liệu chuẩn và so sánh số học với đầu ra runtime. |
| `ai.runtime` | `ai.tensor.*`, `ai.onnxruntime.*` | Điều khiển C++ native ONNX engine, ánh xạ bộ nhớ trực tiếp. |
| `ai.tensor` | `ai.runtime.RuntimeTensorType` | Định nghĩa cấu trúc mảng FloatTensor và kích thước ma trận. |

---

## 6. Luồng Tương Tác & Dữ Liệu Thời Gian Thực (Execution & Data Flows)

### 6.1. Luồng Nạp Mô Hình & Suy Luận Trong ONNX Demo

Biểu đồ tuần tự dưới đây mô tả quá trình từ khi người dùng bấm nút trên giao diện cho đến khi ONNX Runtime xử lý dữ liệu native và phản hồi kết quả về UI State:

```mermaid
sequenceDiagram
    autonumber
    actor User as 👤 Người Dùng
    participant Screen as 🖥️ OnnxDemoScreen
    participant VM as ⚙️ OnnxDemoViewModel
    participant IO as 🧵 Dispatchers.IO
    participant Default as 🧵 Dispatchers.Default
    participant Engine as 🛡️ OnnxRuntimeEngine
    participant Session as ⚡ OnnxModelSession
    participant Native as 🔌 ONNX C++ Native (ORT)

    User->>Screen: Bấm "Load Model"
    Screen->>VM: loadModel()
    VM->>VM: Cập nhật state = Loading
    VM->>IO: withContext(Dispatchers.IO) đọc assets/models/backbone_neck.onnx
    IO-->>VM: Trả về ByteArray (modelBytes)
    
    VM->>Default: withContext(Dispatchers.Default) engine.loadModel(...)
    Default->>Engine: loadModel(modelName, modelBytes)
    Engine->>Session: OnnxModelSession.create(env, name, bytes)
    Session->>Native: ortEnvironment.createSession(bytes, options)
    Native-->>Session: OrtSession khởi tạo thành công
    Session-->>Engine: ModelMetadata (inputs, outputs, latency)
    Engine-->>VM: RuntimeResult.Success(metadata)
    VM->>VM: mutableState.update { Ready, metadata }
    VM-->>Screen: StateFlow phát OnnxDemoUiState mới
    Screen-->>User: Hiển thị Metadata, Ready, kích hoạt nút "Run Dummy"

    User->>Screen: Bấm "Run Dummy Inference"
    Screen->>VM: runDummyInference()
    VM->>Default: DummyTensorFactory.create(metadata)
    Default-->>VM: FloatTensor (zero-filled inputs)
    VM->>Default: engine.run(inputs, captureFullOutputs=true)
    Default->>Engine: run(inputs)
    Engine->>Session: run(inputs)
    Session->>Session: allocateDirect ByteBuffer & put(floatArray)
    Session->>Native: session.run(nativeInputs)
    Native-->>Session: OrtSession.Result
    Session->>Session: Trích xuất floatValues & TensorPreview
    Session-->>Engine: RuntimeResult.Success(RuntimeInferenceResult)
    Engine-->>VM: Trả về kết quả suy luận & thời gian thực thi (ns)
    VM->>VM: mutableState.update { lastInference, diagnostics }
    VM-->>Screen: StateFlow cập nhật UI
    Screen-->>User: Hiển thị thời gian suy luận (ms) và dữ liệu preview tensor
```

### 6.2. Luồng Trích Xuất Đặc Trưng Không Gian Trong Pipeline Thị Giác Máy Tính

Khi tích hợp camera thực tế, `BackboneNeckPipeline` sẽ hoạt động theo luồng:

```mermaid
sequenceDiagram
    autonumber
    participant Camera as 📷 CameraX / NDK Camera
    participant Preprocess as ⚙️ Image Preprocessor
    participant Pipeline as 🧠 BackboneNeckPipeline
    participant Contract as 📋 BackboneNeckContract
    participant Engine as ⚡ OnnxRuntimeEngine
    participant Consumer as 🎯 Drowsiness / Head Pose Detector

    Camera->>Preprocess: Cung cấp khung hình NV21/YUV_420_888
    Preprocess->>Preprocess: Resize 640x640, BGR->RGB, chuẩn hóa [0..1]
    Preprocess->>Pipeline: run(images: FloatTensor [1, 3, 640, 640])
    
    Pipeline->>Contract: validateInput(images)
    Contract-->>Pipeline: Hợp lệ (Shape = [1, 3, 640, 640], FloatArray size khớp)
    
    Pipeline->>Engine: run({"images": images}, captureFullOutputs = true)
    Engine-->>Pipeline: RuntimeResult.Success (Outputs: p3, p4, p5)
    
    Pipeline->>Contract: features(result, batchSize = 1)
    Contract-->>Pipeline: Đóng gói BackboneNeckFeatures (p3: 80x80, p4: 40x40, p5: 20x20)
    
    Pipeline-->>Consumer: BackboneNeckFeatures
    Consumer->>Consumer: Trích xuất đặc trưng vùng mắt/miệng & suy luận trạng thái mệt mỏi
```

### 6.3. Vòng Đời Điều Hướng Người Dùng (User Navigation Lifecycle)

```mermaid
flowchart TD
    Start([🚀 Khởi Chạy Ứng Dụng]) --> Home["Trang Chủ (HomeScreen)"]

    Home --> Selection["Chọn Tài Xế & Xe (SelectionScreen)"]
    Selection --> PreTrip["Kiểm Tra An Toàn (PreTripCheckScreen)"]
    
    PreTrip --> Driving["Giám Sát Chuyến Đi (ActiveDrivingScreen)"]
    
    Driving -- Phát hiện buồn ngủ cao độ --> Danger["🚨 Cảnh Báo Nguy Hiểm (DangerAlertScreen)"]
    Danger -- Tài xế xác nhận đã tỉnh táo --> Driving
    
    Driving -- Kết thúc chuyến đi --> Summary["Tổng Kết Chuyến Đi (TripSummaryScreen)"]
    Summary --> Detail["Chi Tiết Chuyến Đi (TripDetailScreen)"]
    Summary --> Home

    Home -. Menu điều hướng .-> History["Lịch Sử Chuyến Đi (TripHistoryScreen)"]
    Home -. Menu điều hướng .-> Alerts["Nhật Ký Cảnh Báo (AlertHistoryScreen)"]
    Home -. Menu điều hướng .-> Analytics["Phân Tích Thống Kê (AnalyticsScreen)"]
    Home -. Menu điều hướng .-> Settings["Cài Đặt Hệ Thống (SettingsScreen)"]
    Home -. Menu điều hướng .-> System["Thông Tin Hệ Thống (SystemInfoScreen)"]
    
    System -. Nút kiểm thử kỹ thuật .-> OnnxDemo["Chẩn Đoán ONNX Demo (OnnxDemoScreen)"]
```

---

## 7. Tổng Kết

Kiến trúc mã nguồn Android của **Driver Guardian** đạt được sự phân tách trách nhiệm tối ưu:
1. **Module hóa chặt chẽ**: Tách bạch 100% giữa Tầng Trí tuệ Nhân tạo (`ai`) và Tầng Giao diện người dùng (`ui`).
2. **An toàn kiểu & bộ nhớ**: Đóng gói C++ ONNX native code với cơ chế Direct ByteBuffer, đồng bộ đa luồng (`synchronized lock`), giải phóng bộ nhớ tự động, kiểm tra shape chống tràn số học.
3. **Khả năng mở rộng**: Dễ dàng tích hợp thêm các head phân loại (Drowsiness Head, Distraction Head, Head Pose SQPnP) vào `BackboneNeckPipeline` mà không làm thay đổi kiến trúc giao diện hiện tại.
