# Driver Guardian Android

Android Automotive demo dùng Jetpack Compose. Phần integration foundation kết nối luồng chuẩn bị chuyến đi với FastAPI; nhánh physical monitoring dùng CameraX, MediaPipe Face Landmarker và cảm biến định hướng thiết bị. Mô hình ONNX buồn ngủ và fusion vẫn là phần tương lai.

## MediaPipe Face Landmarker asset

Ứng dụng chỉ tải asset đã commit tại `app/src/main/assets/face_landmarker.task`; không tải model lúc runtime và không có mock fallback.

- Nguồn chính thức: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`
- Model bundle: MediaPipe Face Landmarker, biến thể `float16`; endpoint upstream dùng revision alias `latest` (không công bố revision ID riêng trong URL)
- Ngày tải (UTC): 2026-09-30
- Kích thước: 3,758,596 bytes
- SHA-256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`

Bytes được giữ nguyên theo bản tải từ nguồn chính thức. Runtime dùng `RunningMode.LIVE_STREAM`, `numFaces = 1`; thiếu hoặc hỏng asset là lỗi khởi tạo rõ ràng.

## Kiến trúc integration

- `data/remote`: Retrofit API và DTO bám đúng JSON contract của FastAPI.
- `data/repository`: ánh xạ DTO sang domain model và chuyển lỗi mạng/HTTP thành thông báo an toàn cho UI.
- `domain/model`: `Driver`, `Vehicle`, `ModelVersion`, `DrivingSession`, `TripSession`, `DrowsinessEvent` với ID backend thật và nullable data được giữ nguyên.
- `ui/session`: shared `DrivingSessionViewModel` + `StateFlow` giữ active session, completion và acknowledgement state.
- `ui/history`: `TripHistoryViewModel` riêng tải history/detail/events và presentation mapper hiển thị giá trị thiếu bằng `—`.
- `ui/screens`: Selection tải dữ liệu thật; PreTrip tạo session; ActiveDriving lưu demo danger event trước khi mở màn hình cảnh báo.

Không có SQLite, fake backend, offline queue hoặc DI framework trong phase này.

## FastAPI base URL

Debug mặc định dùng địa chỉ host nhìn từ Android Emulator:

```text
http://10.0.2.2:8000/
```

Override bằng Gradle property, không sửa source:

```powershell
.\gradlew.bat :app:assembleDebug -PDRIVER_GUARDIAN_API_BASE_URL=http://192.168.1.10:8000/
```

Build tự thêm dấu `/` cuối nếu thiếu. Quyền `INTERNET` nằm trong main manifest; HTTP cleartext chỉ được bật trong debug manifest để phục vụ local development.

## Backend endpoints đang dùng

- `GET /drivers`
- `GET /vehicles`
- `GET /model-versions/active`
- `POST /sessions`
- `POST /events`
- `POST /sessions/{id}/complete`
- `GET /sessions`
- `GET /sessions/{id}`
- `GET /sessions/{id}/events`
- `POST /events/{id}/acknowledge`

Datetime được giữ dưới dạng `String` theo response hiện tại. `DrivingSession.modelVersionId`, event `confidence` và `durationMs` giữ nullable đúng backend contract. Mapping cảnh báo là `WARNING -> 1`, `DANGER -> 2`.

## Luồng hiện tại

1. Shared ViewModel tải drivers, vehicles và active model. Loading, empty và error được hiển thị; không fallback sang `MockData`.
2. Người dùng chọn driver và vehicle thật. Chỉ có thể tiếp tục khi đủ hai lựa chọn và active model.
3. PreTrip vẫn hiển thị các kiểm tra thiết bị mock nhưng hiển thị riêng trạng thái backend/model. Nút bắt đầu gọi `POST /sessions` và chỉ điều hướng khi thành công.
4. ActiveDriving hiển thị driver, vehicle và session backend thật. Camera/physical/device-motion là dữ liệu runtime; không ánh xạ thành kết luận WARNING/DANGER tự động.
5. Demo danger gọi `POST /events` với `DANGER`, level `2`; DangerAlert chỉ đóng sau khi acknowledgement được backend xác nhận.
6. Kết thúc chuyến gọi completion API; summary chỉ mở sau khi completed session và persisted events đã tải thành công.
7. Summary, history và detail dùng session/event thật; null score/metric/end time hiển thị `—`, không tạo số mặc định.

## Xử lý lỗi

Repository trả kết quả lỗi rõ ràng cho connection refused, timeout, HTTP 4xx/5xx và response parsing. UI chỉ hiển thị thông báo tổng quát, không hiển thị stack trace hoặc nội dung lỗi database. Empty driver/vehicle list và active-model 404 chặn tạo session thay vì dùng dữ liệu giả.

## Giới hạn hiện tại

- Chưa có offline persistence/retry queue; mất mạng sẽ trả lỗi và người dùng thử lại thủ công.
- Drowsiness ONNX runtime chưa được nối vào camera pipeline; `MonitoringSnapshot` không phải ONNX tensor contract và chưa có fusion/classifier.
- Analytics, alert-history, home dashboard và các tiện ích map/music/report vẫn là mock hoặc chưa hỗ trợ; trip summary/history/detail không còn dùng `MockData`.
- Backend hiện chưa tính `safety_score`; UI giữ và hiển thị `NULL` thành `—`.
- Camera trước trên emulator phụ thuộc virtual-camera support; trạng thái thiếu camera/cảm biến được hiển thị rõ, không giả lập metric.

## Build và test

```powershell
.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug --console=plain --no-daemon
```

Contract tests dùng MockWebServer, không cần FastAPI hay Oracle đang chạy.
