# Driver Guardian Android

Android Automotive application built with Jetpack Compose. The system integrates real-time multi-sensor monitoring (CameraX + MediaPipe Face Landmarker + Rotation Vector sensor) with a FastAPI backend and Oracle database.

## Authentication & Session Management

- **Architecture**: Modern Android Credential Manager (`GetGoogleIdOption`) exchanges Google ID Token with backend (`POST /auth/google`) for Driver Guardian JWT access token and refresh token.
- **Secure Persistence**: Refresh token and user profile are persisted via `EncryptedSharedPreferences` backed by the Android Keystore (AES-256-GCM).
- **Retrofit AuthInterceptor**: Automatically attaches `Authorization: Bearer <access_token>`, intercepts HTTP 401, coordinates synchronized token refresh, retries failed requests once, and redirects to `LoginScreen` on unrecoverable authentication errors.
- **Driver Profile Resolution**: Authenticated driver identity is automatically resolved from `/auth/me`. For users with `ROLE=DRIVER`, manual driver picker is bypassed and replaced with the authenticated driver's credentials.
- **Unlinked Driver Handling**: Users whose Google account is not yet bound to a `DRIVER_ID` are displayed a safe blocked state, preventing invalid driving sessions until linked by an administrator.

## Strict AI Boundary & Model Status

- **face_landmarker.task**: Official MediaPipe face landmark model (478 3D facial landmarks). It is **NOT** a drowsiness classification model.
- **Drowsiness ONNX Model**: **Future Integration**. There is currently NO trained drowsiness ONNX artifact in the system. Existing ONNX-related code is architectural framework for future deployment.
- **Current Real-time Monitoring Pipeline**:
  - CameraX → MediaPipe Face Landmarker → `PhysicalBranchProcessor` → `PhysicalMetrics` (EAR, MAR, PERCLOS, POM, Head Pose, Nodding).
  - Rotation Vector Sensor → `DeviceMotionProcessor` → `DeviceMotionMetrics` (Pitch, Roll, Yaw, Tilt).
  - `MonitoringSnapshot` provides unified multi-sensor diagnostics.
- **Danger Demo Workflow**: Explicitly labeled `DEMO: Cảnh báo nguy hiểm`. Physical/sensor metrics do **not** automatically trigger warning or danger states without trained neural classification.

## Build and Tests

Run unit tests and debug assemble:

```powershell
.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug --console=plain
```

- Total Android Unit Tests: 145 passing (124 monitoring/sensor tests + 21 authentication & repository tests).
- Build status: assembleDebug PASS.
