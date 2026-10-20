# Driver Guardian Android

Android Automotive application built with Jetpack Compose. The production monitoring path uses CameraX, MediaPipe Face Landmarker, and the Physical Branch, with a FastAPI backend and Oracle database.

## Authentication & Session Management

- **Architecture**: Modern Android Credential Manager (`GetGoogleIdOption`) exchanges Google ID Token with backend (`POST /auth/google`) for Driver Guardian JWT access token and refresh token.
- **Secure Persistence**: Refresh token and user profile are persisted via `EncryptedSharedPreferences` backed by the Android Keystore (AES-256-GCM).
- **Retrofit AuthInterceptor**: Automatically attaches `Authorization: Bearer <access_token>`, intercepts HTTP 401, coordinates synchronized token refresh, retries failed requests once, and redirects to `LoginScreen` on unrecoverable authentication errors.
- **Driver Profile Resolution**: Authenticated driver identity is automatically resolved from `/auth/me`. For users with `ROLE=DRIVER`, manual driver picker is bypassed and replaced with the authenticated driver's credentials.
- **Unlinked Driver Handling**: Users whose Google account is not yet bound to a `DRIVER_ID` are displayed a safe blocked state, preventing invalid driving sessions until linked by an administrator.

## Strict AI Boundary & Model Status

- **face_landmarker.task**: Official MediaPipe face landmark model (478 3D facial landmarks). It is **NOT** a drowsiness classification model.
- **Drowsiness ONNX Model**: the ONNX Demo accepts an optional local demo binary. The binary is not tracked by Git and the demo is isolated from Active Driving alerts.
- **Current Real-time Monitoring Pipeline**:
  - CameraX → MediaPipe Face Landmarker → `PhysicalBranchProcessor` → `PhysicalMetrics` (EAR, MAR, PERCLOS, POM, Head Pose, Nodding).
  - Device Motion source code is retained for future evaluation but is not registered by the production monitoring engine.
  - `MonitoringSnapshot` provides physical and camera diagnostics.
- **Danger Demo Workflow**: Explicitly labeled `DEMO: Cảnh báo nguy hiểm`. Physical/sensor metrics do **not** automatically trigger warning or danger states without trained neural classification.

## Build and Tests

Run unit tests and debug assemble:

```powershell
.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug --console=plain
```

Model-independent tests always run. Tests that require the optional ONNX binary are reported as skipped when the binary is absent; when it is present, its size, SHA-256, contract, and inference parity must all pass.

## Local System Setup

### Prerequisites

- Podman
- Python 3.10 or newer
- JDK 17 and Android SDK 36
- Android SDK Platform Tools (`adb`)

Commands below use PowerShell and assume the repository root is the current directory unless another directory is shown.

### 1. Run Oracle Database with Podman

Create a persistent volume and a Podman secret. Enter the database password only at the interactive prompt; do not write it into this repository:

```powershell
podman volume create driver-guardian-oracle-data
podman secret create driver-guardian-oracle-password -
```

Start Oracle Free on port 1521:

```powershell
podman run -d --name driver-guardian-oracle `
  -p 1521:1521 `
  --secret driver-guardian-oracle-password,type=env,target=ORACLE_PASSWORD `
  -v driver-guardian-oracle-data:/opt/oracle/oradata `
  docker.io/gvenzl/oracle-free:23-slim
podman logs -f driver-guardian-oracle
```

Wait until the database reports that it is ready. The default pluggable database service is `FREEPDB1`. Provision an application user locally, then review and apply the scripts in `system/database/schema` in numeric order. Do not apply `system/database/recovered/driver_data_tablespace.sql`; it is machine-specific reference material.

### 2. Configure and run FastAPI on port 8000

Create a local virtual environment and install dependencies:

```powershell
Set-Location system/backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill the local `.env` with the Oracle application user's `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, and `DB_SERVICE`. The `.env` file is ignored by Git; never commit it or paste credentials into documentation.

Start the API:

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Verify process and database health from the laptop:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/health/database
```

`/health` confirms the FastAPI process. `/health/database` must also succeed before testing trip persistence.

### 3. Configure the Android API URL

The committed default is:

```text
http://127.0.0.1:8000/
```

On Android, `127.0.0.1` refers to the Android device or emulator itself. If FastAPI runs on the laptop, configure ADB reverse before launching the app:

```powershell
adb reverse tcp:8000 tcp:8000
adb reverse --list
```

The URL can be overridden without editing tracked source:

```powershell
Set-Location system/android
.\gradlew.bat -PDRIVER_GUARDIAN_API_BASE_URL=http://10.0.2.2:8000/ :app:assembleDebug
```

Use `10.0.2.2` for the standard Android Emulator host gateway when not using ADB reverse. For a physical device connected over the same network without ADB reverse, use the laptop's reachable LAN address. Do not commit machine-specific URLs.

### 4. Run with Android Emulator

1. Start an API 29-or-newer AVD from Android Studio Device Manager.
2. Confirm it appears with `adb devices`.
3. Because the default API URL is `127.0.0.1`, run `adb -e reverse tcp:8000 tcp:8000`.
4. Run the `app` configuration from Android Studio, or build and install the debug APK with Gradle and `adb`.
5. Open `/health` and `/health/database` on the laptop first if the app cannot reach the backend.

### 5. Run on a physical Android device over USB

1. Enable Developer options and USB debugging on the device.
2. Connect it by USB and accept the debugging authorization prompt.
3. Confirm it appears as `device` with `adb devices`.
4. Run `adb -d reverse tcp:8000 tcp:8000` before launching the app.
5. If more than one device is connected, use `adb -s <serial> reverse tcp:8000 tcp:8000`.
6. Run the app from Android Studio and verify CameraX, MediaPipe, and trip lifecycle behavior.

ADB reverse rules may need to be recreated after reconnecting or restarting the device/emulator.

### 6. Provide the optional ONNX Demo model locally

Obtain the approved demo artifact from the team's private artifact storage and place it at:

```text
system/android/app/src/main/assets/models/driver_guardian_end2end_demo.onnx
```

Expected artifact metadata:

```text
Size:   13,414,830 bytes
SHA-256: 3b0189550f7ed1ae53a620d143a8b1ed715f5e970b0977e0d44cfd6cb9680057
```

The repository ignores `*.onnx`. Confirm the model remains ignored:

```powershell
git check-ignore -v system/android/app/src/main/assets/models/driver_guardian_end2end_demo.onnx
git status --short
```

Do not use `git add -f` for the model. Without the binary, the app still builds and ONNX Demo reports that the model is unavailable; model-backed unit tests are skipped with an explicit reason. With the binary present, integrity, model contract, and ONNX Runtime parity tests run normally and any mismatch fails the test suite.
