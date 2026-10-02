# Realtime Multi-Sensor Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real CameraX/MediaPipe facial-landmark monitoring and independently calibrated device-orientation monitoring, synchronized into nullable `MonitoringSnapshot` values without drowsiness classification or AI fusion.

**Architecture:** CameraX supplies rotation-corrected, unmirrored frames to asset-backed MediaPipe LIVE_STREAM; a neutral adapter converts landmarks to canonical pixels for the existing `PhysicalBranchProcessor`. SensorManager supplies rotation-vector or explicit fallbacks to a pure `DeviceMotionProcessor`; `DriverMonitoringEngine` verifies/normalizes camera time, performs latest-not-future synchronization, and exposes snapshots, orthogonal runtime state, and diagnostics.

**Tech Stack:** Kotlin 2.2.21, Android API 29+, CameraX 1.5.3, MediaPipe Tasks Vision 1.0.0, Compose, StateFlow, JUnit 4.

**Spec:** `system/android/docs/superpowers/specs/2026-09-30-multi-sensor-monitoring-design.md`

## Global Constraints

- Change only `system/android/**` on `feature/onnx-system-framework`; never merge/rebase/checkout/push main, create a PR, or force push.
- Keep `PhysicalMetrics` and `DeviceMotionMetrics` separate; do not add tensor mapping, fusion, warning score, classifier, or automatic WARNING/DANGER event.
- Algorithmic camera input and landmarks remain unmirrored; PreviewView/overlay mirroring is UI-only.
- Use monotonic clocks only; camera/sensor timestamps are not compared until their relationship is verified or normalized.
- Vendor unmodified official `app/src/main/assets/face_landmarker.task`; assets-only loading, no runtime download, no mock fallback.
- Use `RunningMode.LIVE_STREAM`, `numFaces = 1`, and `ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST`; close each `ImageProxy` exactly once.
- Device neutral calibration defaults to 750 ms and at least 10 valid samples, resetting per trip; unavailable values remain null.
- `MonitoringSyncConfig.SENSOR_FRESHNESS_THRESHOLD_MS = 150.0` is configurable operational policy, not a model requirement.
- Preserve the existing Android → FastAPI → Oracle trip and DEMO danger flow.
- Create exactly one commit only after all verification passes: `feat(android): integrate camera and device motion monitoring`; push only the feature branch.

## Review Focus

- Camera timestamps marked UNKNOWN with unstable receipt offsets must suppress motion association, covered by `MonitoringTimebaseNormalizerTest.unstableUnknownClockNeverVerifies` in Task 3.
- Orientation samples crossing +180/-180 during calibration must produce a neutral near the wrap boundary, covered by `DeviceMotionProcessorTest.calibrationUsesWrappedMedian` in Task 4.
- A MediaPipe callback arriving after stop/reset must not update a new trip, covered by `DriverMonitoringCoordinatorTest.ignoresResultFromPreviousGeneration` in Task 6.
- Permission denial followed by granting in the same active trip must bind camera once without duplicating the sensor listener, covered by `MonitoringViewModelTest.permissionRecoveryDoesNotRestartSensor` in Task 9.
- Image analysis failure must still close the proxy once and only once, covered by `CameraXFrameSourceTest.closesProxyExactlyOnceWhenConversionFails` in Task 8.

---

### Task 1: Pin dependencies and verify the official model asset

**Files:**
- Modify: `system/android/app/build.gradle.kts`
- Create: `system/android/app/src/main/assets/face_landmarker.task`
- Modify/Create: `system/android/README.md`
- Create: `system/android/app/src/test/java/com/example/driverguardian/ai/monitoring/FaceLandmarkerAssetTest.kt`

**Interfaces:**
- Consumes: official model URL and exact downloaded bytes.
- Produces: readable `face_landmarker.task`, pinned CameraX/MediaPipe dependencies, documented byte size/SHA-256/provenance.

- [ ] Write `FaceLandmarkerAssetTest.assetExistsIsReadableAndMatchesDocumentedSha256` using classpath/project asset resolution; assert non-empty bytes and the exact documented digest.
- [ ] Run `gradlew.bat :app:testDebugUnitTest --tests "*FaceLandmarkerAssetTest"` and verify RED because the asset/documented digest is absent.
- [ ] Download the official model once, print byte size and SHA-256, confirm it is below GitHub's normal file limit, and stop before Git LFS if it is not.
- [ ] Add the unmodified asset; document official URL, available revision/version, UTC download date, byte size, and SHA-256 in `system/android/README.md`.
- [ ] Pin CameraX `1.5.3` and MediaPipe Tasks Vision `1.0.0`; never use `latest.release`.
- [ ] Re-run the focused asset test and verify GREEN.

### Task 2: Canonical camera geometry, landmark adapter, and timestamp gate

**Files:**
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/camera/CameraFrameGeometry.kt`
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/camera/CameraTimestampGate.kt`
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/camera/MediaPipeLandmarkAdapter.kt`
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/camera/LandmarkOverlayMapper.kt`
- Create tests under `system/android/app/src/test/java/com/example/driverguardian/ai/monitoring/camera/` for all four classes.

**Interfaces:**
- Produces: `CameraFrameGeometry.rotatedSize(width, height, rotationDegrees)`, `CameraTimestampGate.accept(timestampNs)`, `MediaPipeLandmarkAdapter.toPixelLandmarks(List<NormalizedLandmark>, width, height)`, and UI-only `LandmarkOverlayMapper.frontMirrorCopy(...)`.
- Guarantees: algorithmic landmarks are unmirrored and Python-equivalent; overlay mirror returns copies.

- [ ] Write RED tests for 0/90/180/270 dimensions, invalid rotation, normalized-to-pixel conversion, left/right preservation, yaw-sensitive landmark ordering, non-mutating overlay copy, nanoseconds-to-seconds conversion, and duplicate/out-of-order rejection.
- [ ] Run the four focused test classes and confirm expected missing-symbol failures.
- [ ] Implement the minimal pure geometry/gate/adapter APIs; adapter uses only `x * width`, `y * height` and copies z without UI mirroring.
- [ ] Re-run focused tests and all existing physical parity tests; verify GREEN and unchanged yaw/left-right fixture semantics.

### Task 3: Verify and normalize the camera/sensor time relationship

**Files:**
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/time/CameraTimestampSource.kt`
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/time/MonitoringTimebaseNormalizer.kt`
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/time/MonitoringTimebaseConfig.kt`
- Create: `system/android/app/src/test/java/com/example/driverguardian/ai/monitoring/time/MonitoringTimebaseNormalizerTest.kt`

**Interfaces:**
- Produces: `observe(cameraTimestampNs, receiptElapsedNs)`, `normalizeCameraTimestampNs(cameraTimestampNs): Long?`, `relation: TimebaseRelation`, `offsetNs: Long?`.
- `CameraTimestampSource` values: `REALTIME`, `UNKNOWN`; both require at least 8 runtime receipt pairs. REALTIME still needs causal direct-comparison evidence; UNKNOWN is never labeled direct from receipt pairs and may only use a conservative stable offset.

- [ ] Write RED tests for runtime-verified REALTIME, stable UNKNOWN conservative offset, insufficient samples, unstable offsets, reset, monotonic normalized output, and `unstableUnknownClockNeverVerifies`.
- [ ] Run the focused test and confirm RED.
- [ ] Implement verification with named sample-count, spread, and maximum capture-to-receipt latency tolerances and no wall clock; return null until verified.
- [ ] Re-run focused tests and verify GREEN.

### Task 4: Device orientation model and robust per-trip calibration

**Files:**
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/device/DeviceOrientation.kt`
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/device/DeviceMotionMetrics.kt`
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/device/DeviceMotionConfig.kt`
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/device/OrientationMath.kt`
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/device/DeviceMotionProcessor.kt`
- Create: `system/android/app/src/test/java/com/example/driverguardian/ai/monitoring/device/DeviceMotionProcessorTest.kt`
- Create: `system/android/app/src/test/java/com/example/driverguardian/ai/monitoring/device/OrientationMathTest.kt`

**Interfaces:**
- `DeviceOrientation(timestampSec, pitchDeg?, rollDeg?, yawDeg?, accuracy?, source)`.
- `DeviceMotionProcessor.process(DeviceOrientation): DeviceMotionMetrics?`, `reset()`, `markUnavailable(timestampSec)`.
- `DeviceMotionMetrics` contains timestamp, source, availability, calibration, raw/delta axes, null sample age, and accuracy.

- [ ] Write RED tests for quaternion/rotation-vector conversion, display rotations 0/90/180/270, angle normalization, strictly increasing timestamps, 750 ms plus 10-sample calibration, null deltas during calibration, wrapped-median neutral, optional yaw, reset/session recalibration, and unavailable sensor.
- [ ] Run focused tests and confirm RED.
- [ ] Implement pure quaternion-to-matrix/display-remap/euler math and `DeviceMotionProcessor` with reference-unwrapped per-axis median; no Android/Compose dependencies.
- [ ] Re-run focused tests and verify GREEN.

### Task 5: Monitoring models and latest-not-future synchronization

**Files:**
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/model/MonitoringSnapshot.kt`
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/model/MonitoringRuntimeState.kt`
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/model/MonitoringDiagnostics.kt`
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/sync/MonitoringSyncConfig.kt`
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/sync/DeviceMotionHistory.kt`
- Create: `system/android/app/src/test/java/com/example/driverguardian/ai/monitoring/sync/DeviceMotionHistoryTest.kt`

**Interfaces:**
- `MonitoringSnapshot(physical: PhysicalMetrics?, deviceMotion: DeviceMotionMetrics?, timestampSec: Double)`.
- `DeviceMotionHistory.add(metrics)`, `latestAtOrBefore(frameTimestampSec)`, `associate(frameTimestampSec, config)`.
- Named default: `SENSOR_FRESHNESS_THRESHOLD_MS = 150.0`.

- [ ] Write RED tests for samples before/at/after frame time, future exclusion, exact and stale ages, named custom threshold, bounded history, unavailable motion, and independent nullable physical/device values.
- [ ] Run focused tests and confirm RED.
- [ ] Implement immutable models and bounded history; association copies `sampleAgeMs` and returns null when stale.
- [ ] Re-run focused tests and verify GREEN.

### Task 6: Platform-neutral monitoring coordinator

**Files:**
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/DriverMonitoringCoordinator.kt`
- Create: `system/android/app/src/test/java/com/example/driverguardian/ai/monitoring/DriverMonitoringCoordinatorTest.kt`

**Interfaces:**
- Consumes: `PhysicalBranch`, `DeviceMotionProcessor`, `MonitoringTimebaseNormalizer`, `DeviceMotionHistory`.
- Produces: `onSensorOrientation`, `onCameraResult(landmarks?, width, height, cameraTimestampNs, receiptElapsedNs, generation)`, `startSession(generation)`, `stopSession()`, StateFlows for snapshot/state/diagnostics.

- [ ] Write RED tests for reset/start, no-face physical output, verified synchronization, unverified suppression, stale motion, sensor-valid/no-face independence, camera-valid/sensor-unavailable independence, duplicate timestamps, and `ignoresResultFromPreviousGeneration`.
- [ ] Run focused tests and confirm RED.
- [ ] Implement coordinator only; it contains no CameraX, MediaPipe, SensorManager, Compose, network, or classification code.
- [ ] Re-run focused tests plus physical processor tests and verify GREEN.

### Task 7: Android device-orientation source and fallbacks

**Files:**
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/device/AndroidDeviceOrientationSource.kt`
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/device/SensorSelection.kt`
- Create: `system/android/app/src/test/java/com/example/driverguardian/ai/monitoring/device/SensorSelectionTest.kt`

**Interfaces:**
- Selection order: ROTATION_VECTOR, GAME_ROTATION_VECTOR, accelerometer+magnetometer, UNAVAILABLE.
- Runtime API: `start(displayRotationProvider, callback)`, `stop()`, `selectedSource`, with idempotent registration cleanup.

- [ ] Write RED pure selection tests for every inventory combination and explicit unavailable state.
- [ ] Run focused tests and confirm RED.
- [ ] Implement selection plus a lightweight SensorEvent adapter using monotonic event time, current display remap, paired accel/magnetometer fallback, accuracy propagation, and diagnostic rate/processing timing.
- [ ] Re-run focused tests and verify GREEN.

### Task 8: MediaPipe LIVE_STREAM and CameraX adapters

**Files:**
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/camera/FaceLandmarkerClient.kt`
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/camera/MediaPipeFaceLandmarkerClient.kt`
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/camera/CameraXFrameSource.kt`
- Create: `system/android/app/src/test/java/com/example/driverguardian/ai/monitoring/camera/CameraXFrameSourceTest.kt`
- Create: `system/android/app/src/test/java/com/example/driverguardian/ai/monitoring/camera/FaceLandmarkerConfigTest.kt`

**Interfaces:**
- Face client initializes `face_landmarker.task`, `LIVE_STREAM`, `numFaces=1`; result includes canonical unmirrored landmarks, dimensions, camera/receipt timestamps, and inference ms.
- Camera source binds Preview + RGBA ImageAnalysis with KEEP_ONLY_LATEST on a dedicated executor, prefers front lens, reports camera timestamp-source characteristic, and unbinds idempotently.

- [ ] Write RED tests around extracted configuration/close-guard helpers for assets-only options, LIVE_STREAM, one face, monotonic task milliseconds, proxy success/failure close-once, and metadata cleanup on late/error callbacks.
- [ ] Run focused tests and confirm RED.
- [ ] Implement MediaPipe initialization with explicit missing/corrupt error and no fallback; rotate frames according to CameraX metadata without algorithmic mirror.
- [ ] Implement CameraX lifecycle binding/analyzer; close proxy in one `finally`, clear analyzer/unbind/close executors on stop.
- [ ] Re-run focused tests and verify GREEN.

### Task 9: Runtime engine, permission state, ViewModel, and session lifecycle

**Files:**
- Create: `system/android/app/src/main/java/com/example/driverguardian/ai/monitoring/DriverMonitoringEngine.kt`
- Create: `system/android/app/src/main/java/com/example/driverguardian/ui/monitoring/MonitoringViewModel.kt`
- Create: `system/android/app/src/main/java/com/example/driverguardian/ui/monitoring/MonitoringViewModelFactory.kt`
- Create: `system/android/app/src/test/java/com/example/driverguardian/ui/monitoring/MonitoringViewModelTest.kt`
- Modify: `system/android/app/src/main/java/com/example/driverguardian/ui/navigation/AppNavigation.kt`
- Modify: `system/android/app/src/main/java/com/example/driverguardian/ui/session/DrivingSessionViewModel.kt` only if a minimal lifecycle hook is required.

**Interfaces:**
- Engine owns Android adapters and coordinator; `startSession(sessionId, lifecycleOwner, surfaceProvider, permission)`, `updatePermission`, `stopSession`, `close`.
- ViewModel exposes engine flows and prevents duplicate start/stop/register operations for one session.

- [ ] Write RED ViewModel tests for per-session reset, explicit GRANTED/DENIED/UNAVAILABLE, permission recovery without duplicate sensor registration, session change recalibration, stop at completion, and idempotent cleanup.
- [ ] Run focused tests and confirm RED.
- [ ] Implement engine/ViewModel/factory and wire active-session start plus trip-end/screen-disposal stop in navigation; camera denial must not stop independent sensor processing.
- [ ] Re-run focused tests plus existing session ViewModel tests and verify GREEN.

### Task 10: Camera permission, Preview, and collapsible real debug UI

**Files:**
- Modify: `system/android/app/src/main/AndroidManifest.xml`
- Create: `system/android/app/src/main/java/com/example/driverguardian/ui/monitoring/CameraPreview.kt`
- Create: `system/android/app/src/main/java/com/example/driverguardian/ui/monitoring/MonitoringDebugPanel.kt`
- Create: `system/android/app/src/main/java/com/example/driverguardian/ui/monitoring/MonitoringPresentation.kt`
- Create: `system/android/app/src/test/java/com/example/driverguardian/ui/monitoring/MonitoringPresentationTest.kt`
- Modify: `system/android/app/src/main/java/com/example/driverguardian/ui/screens/driving/ActiveDrivingScreen.kt`

**Interfaces:**
- Presentation maps every nullable metric to `—`, labels HEAD POSE separately from DEVICE / VEHICLE ORIENTATION, and exposes independent camera/face/sensor statuses.
- Active screen hosts `PreviewView`, permission launcher, collapse state, real metrics, diagnostics, and existing backend actions.

- [ ] Write RED presentation tests for null placeholders, independent no-face/sensor-unavailable labels, calibration/stale labels, and DEMO danger label.
- [ ] Run focused tests and confirm RED.
- [ ] Add CAMERA permission and Compose runtime request; denial/error/unavailable states render without crash.
- [ ] Replace synthetic AI metric panel with real monitoring panel while retaining session controls and relabeling danger button `DEMO: cảnh báo nguy hiểm`.
- [ ] Re-run focused tests and existing presentation/session/history tests; verify GREEN.

### Task 11: Full verification, emulator evidence, single commit, and feature push

**Files:**
- Modify only if a failing runtime check exposes a real bug; any fix starts with a reproducing RED test.

**Interfaces:**
- Produces evidence for the requested final-report fields and no out-of-scope Git changes.

- [ ] Run `gradlew.bat :app:testDebugUnitTest --console=plain --no-daemon`; record old/new/total counts and zero failures.
- [ ] Run `gradlew.bat :app:assembleDebug --console=plain --no-daemon`; require PASS.
- [ ] Install current debug APK on `emulator-5554`; validate permission flow, CameraX preview/input, MediaPipe initialization/landmarks or explicit emulator camera limitation, runtime metric updates, and cleanup.
- [ ] Inventory emulator sensors; change virtual rotation/tilt where supported and record source, raw/delta values, calibration behavior, update rate, processing time, and any unsupported limitation without claiming hardware PASS.
- [ ] Record camera timestamp source, sensor timestamp source, verified relation, nullable offset, FPS, inference ms, physical ms, sensor Hz, and sensor processing ms.
- [ ] Re-run the existing real Selection → session → DEMO danger → acknowledge → complete → history/detail flow against FastAPI/Oracle, or report the exact external blocker without modifying backend/database.
- [ ] Verify `git diff --check`, branch, expected starting ancestry, and that every changed path is under `system/android/**`; print model size/SHA-256 again.
- [ ] Only when all required verification passes, create the single commit `feat(android): integrate camera and device motion monitoring`.
- [ ] Push only `git push origin feature/onnx-system-framework`; verify main refs were not pushed or modified.
