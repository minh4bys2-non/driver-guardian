# Realtime Multi-Sensor Monitoring Design

## Intent and scope

Driver Guardian will acquire real driver-face landmarks and real phone/device orientation during an active driving session. The two signal domains remain independent: MediaPipe only converts camera frames to landmarks, the existing `PhysicalBranchProcessor` converts canonical pixel landmarks to `PhysicalMetrics`, and a new device-motion pipeline converts Android orientation sensors to `DeviceMotionMetrics`. A coordinator associates the latest eligible device sample with each physical result and publishes `MonitoringSnapshot` without classifying the driver as NORMAL, WARNING, or DANGER.

Only `system/android/**` changes. Backend, database, AI training, shared, paper, and main-branch history remain untouched.

## Dependency and model policy

- Pin CameraX artifacts to stable `1.5.3`: `camera-core`, `camera-camera2`, `camera-lifecycle`, `camera-view`.
- Pin MediaPipe Tasks Vision to stable `1.0.0`.
- Commit the official unmodified model at `app/src/main/assets/face_landmarker.task`.
- Download only during implementation from the official Google MediaPipe model URL: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`.
- Runtime initialization loads the model only through `BaseOptions.setModelAssetPath("face_landmarker.task")`; there is no network model download and no mock fallback.
- `system/android/README.md` records the official source URL, available revision/version metadata, UTC download date, byte size, and SHA-256. The implementation verifies the downloaded size before commit and stops before Git LFS if it approaches GitHub's file limit.
- A JVM asset test verifies that the committed asset exists, is readable, is non-empty, and matches the documented SHA-256. Runtime initialization reports an explicit model initialization error if MediaPipe rejects a missing or corrupt asset.

## Architecture

### Driver-vision pipeline

`CameraX Preview + ImageAnalysis` uses the front camera when present, binds to the Active Driving lifecycle, and uses `STRATEGY_KEEP_ONLY_LATEST`. A single background analyzer converts each accepted `ImageProxy` to an oriented ARGB image, submits it to MediaPipe Face Landmarker in `RunningMode.LIVE_STREAM` with `numFaces = 1`, and closes the `ImageProxy` exactly once in `finally` after the frame has been copied. MediaPipe results are processed on a non-UI executor.

The live-stream adapter owns MediaPipe-specific types. It publishes a library-neutral result containing canonical landmarks, original monotonic frame time, canonical image dimensions, inference duration, and face-presence state. MediaPipe types never enter `ai.physical`.

`PhysicalBranchProcessor` remains the source of EAR, MAR, PERCLOS, POM, blink/yawn rates, head pose, nod, over-angle, and neutral head calibration. It receives unmirrored algorithmic landmark coordinates to preserve Python parity and is reset at every monitoring start. Duplicate or out-of-order camera timestamps are rejected before calling it so its strict monotonic contract is preserved without throwing through the camera callback. Yaw sign and left/right landmark semantics are verified against the unmirrored Python reference fixtures before runtime integration.

### Device-motion pipeline

`AndroidDeviceOrientationSource` selects `TYPE_ROTATION_VECTOR`; if absent it selects `TYPE_GAME_ROTATION_VECTOR`; if both are absent it uses accelerometer plus magnetic-field samples where both sensors exist. If no viable source exists, it publishes explicit `UNAVAILABLE` state and nullable values.

The Android adapter converts the chosen sensor data to a rotation matrix, remaps axes for the current display rotation using `SensorManager.remapCoordinateSystem`, then sends a platform-neutral orientation sample to `DeviceMotionProcessor`. Heavy physical/HMM work never runs on the sensor callback.

`DeviceMotionProcessor`:

- accepts finite, strictly increasing monotonic timestamps;
- retains raw pitch, roll, and yaw as nullable degrees;
- collects valid orientation samples for a configurable startup calibration window of `750 ms` (within the approved 0.5–1.0 second range);
- derives neutral pitch and roll from the per-axis median and derives optional yaw with a circular mean/median-safe angular method;
- remains `CALIBRATING` until the window and minimum valid-sample requirement are satisfied; insufficient samples never fabricate a zero baseline;
- computes normalized deltas in `[-180, 180)`;
- preserves sensor type and accuracy;
- resets calibration and history on every new session or explicit monitoring reset;
- emits immutable `DeviceMotionMetrics` without Compose, Retrofit, FastAPI, CameraX, MediaPipe, or physical-branch dependencies.

### Coordinator

`DriverMonitoringEngine` coordinates camera binding, Face Landmarker, `PhysicalBranchProcessor`, sensor registration, and device-motion history. It exposes:

- `StateFlow<MonitoringSnapshot?>` for future AI input;
- `StateFlow<MonitoringRuntimeState>` for independent phase, face, camera-permission, and sensor status;
- `StateFlow<MonitoringDiagnostics>` for approximate FPS and processing timing.

For a physical result at normalized synchronization time `t`, the coordinator selects the newest device sample whose normalized timestamp is `<= t`. It never blocks a frame waiting for a sensor event and never uses a future sample. `MonitoringSyncConfig.SENSOR_FRESHNESS_THRESHOLD_MS` defaults to `150.0`; it is an operational starting value, not a trained-model requirement. An older sample is stale and produces `deviceMotion = null` for that snapshot while runtime sensor status becomes `STALE`. Physical output remains valid when motion is unavailable; motion state remains valid when no face is present.

`MonitoringSnapshot` is deliberately not an ONNX tensor contract:

```kotlin
data class MonitoringSnapshot(
    val physical: PhysicalMetrics?,
    val deviceMotion: DeviceMotionMetrics?,
    val timestampSec: Double
)
```

No weighted score, threshold decision, fusion rule, tensor ordering, or automatic backend event is added.

## Coordinate systems

### Camera and landmarks

The canonical algorithmic camera space is the rotation-corrected, unmirrored camera image used by MediaPipe and `PhysicalBranchProcessor`:

- origin: top-left;
- positive x: right in the unmirrored camera image;
- positive y: down in the unmirrored camera image;
- width/height: dimensions after applying `ImageInfo.rotationDegrees`;
- front camera: no horizontal mirror in the algorithmic path.

Required rotation is applied before constructing the MediaPipe `MPImage`; the image is not mirrored. MediaPipe normalized landmarks therefore describe the same unmirrored logical space expected by the Python reference. `MediaPipeLandmarkAdapter` performs only the Python-equivalent conversion `xPixel = xNormalized * imageWidth`, `yPixel = yNormalized * imageHeight` and returns `LandmarkPoint`; it does not perform physical calculations.

Preview mirroring is a UI-only concern handled by `PreviewView`. If an overlay is added, it transforms a copy of canonical landmarks for display and never mutates the landmarks passed to `PhysicalBranchProcessor`. Pure transformation metadata and landmark conversion are independently unit-tested for 0/90/180/270-degree rotations; a separate overlay-copy test covers front-camera mirror mapping. The implementation follows CameraX rotation metadata and the official MediaPipe Android live-stream preparation pattern rather than inferring orientation from device posture.

### Device orientation

After display-axis remapping, the canonical device convention is:

- pitch: forward/back tilt about the display-horizontal axis;
- roll: left/right tilt about the display-vertical axis;
- yaw: heading/rotation about the axis normal to the display.

Raw and delta values use degrees and `[-180, 180)`. Pitch/roll are primary future context. Yaw remains available but optional because its reference differs between rotation-vector variants and magnetic fallback. Null is used whenever a value is unavailable; zero is never fabricated.

## Timebase and synchronization

- Camera source time preserves `ImageProxy.imageInfo.timestamp` nanoseconds. Frame receipt also records `SystemClock.elapsedRealtimeNanos()` so the camera clock can be related to Android's elapsed-realtime monotonic domain at runtime.
- MediaPipe receives a strictly increasing millisecond form required by `detectAsync`; a bounded frame-metadata map preserves the exact original nanosecond-derived seconds for the callback.
- Sensor time originates from `SensorEvent.timestamp` nanoseconds and is converted to seconds.
- `MonitoringTimebaseNormalizer` must verify the camera/elapsed-realtime relationship from multiple receipt pairs before camera/sensor synchronization is enabled. It never assumes the two raw clock origins are identical across devices.
- Neither the camera timestamp-source characteristic nor an assumed platform clock is trusted by itself. A source reported as REALTIME must still provide eight runtime camera/receipt pairs demonstrating causal direct comparability (receipt minus capture is non-negative and within the named maximum delivery latency) before its offset is recorded as zero/direct. UNKNOWN is never promoted to direct solely from receipt timing: when its observed offset is stable, normalization uses the smallest observed receipt offset minus the configured maximum delivery latency. This deliberately conservative bound prevents a normalized camera timestamp from silently moving into the future relative to sensor samples. An unstable relationship remains unverified and suppresses device association.
- Runtime diagnostics expose `timebaseRelationVerified`, nullable `timebaseOffsetNs`, and source labels for the final report.
- Wall-clock time and `System.currentTimeMillis()` are forbidden.
- Duplicate/out-of-order frames or sensor samples are skipped safely and counted diagnostically.

## Runtime state and lifecycle

`MonitoringRuntimeState` separates orthogonal concerns:

- phase: `IDLE`, `WAITING_PERMISSION`, `STARTING`, `CALIBRATING`, `RUNNING`, `ERROR`;
- camera permission: `GRANTED`, `DENIED`, `UNAVAILABLE`;
- face: `UNKNOWN`, `DETECTED`, `NO_FACE`;
- sensor: `CALIBRATING`, `AVAILABLE`, `STALE`, `UNAVAILABLE`;
- nullable user-safe initialization/runtime error.

At start monitoring, the engine resets physical and device processors, clears synchronization/timebase history, initializes the asset-backed landmarker, registers the selected sensor source, collects the `750 ms` neutral calibration window, and binds preview/analysis to the lifecycle. Sensor state remains `CALIBRATING` until a stable baseline exists. At trip end or screen disposal it unbinds CameraX, clears the analyzer, closes MediaPipe, unregisters all sensor listeners, shuts down owned executors, and clears transient monitoring state. Re-entry creates a fresh per-trip calibration.

Camera permission is declared in the manifest and requested from Compose. Denial exposes `DENIED` without crashing. Missing front camera may fall back to another available camera while reporting the selected lens; no camera exposes `UNAVAILABLE`.

## Active Driving UI

The Active Driving screen receives monitoring state separately from session/backend state. It contains:

- a CameraX `PreviewView` surface;
- permission/unavailable/error presentation;
- a compact collapsible debug panel;
- PHYSICAL values: face detected, EAR, MAR, PERCLOS, POM, blink rate, yawn rate, head pitch/yaw/roll, nodding, over-angle;
- DEVICE / VEHICLE ORIENTATION values: sensor source/availability, calibration, device pitch/roll/yaw deltas, sample age;
- approximate camera FPS, MediaPipe inference ms, physical processing ms, sensor Hz, and sensor processing ms.

All nullable values display `—`. HEAD POSE and DEVICE / VEHICLE ORIENTATION are labeled separately. The current synthetic AI metric card is replaced by monitoring/debug data; no classification is inferred. The existing persisted danger workflow remains, with its button relabeled `DEMO: cảnh báo nguy hiểm`.

## Error handling

- Missing/corrupt model: phase `ERROR` with an explicit Face Landmarker initialization message; no mock output.
- Camera denied/unavailable/bind failure: explicit camera state; sensor pipeline may continue independently.
- No face: submit `null` landmarks at the accepted camera timestamp so temporal physical state advances consistently and publish `NO_FACE`.
- Sensor unavailable: `DeviceMotionMetrics` remains unavailable/null; camera/physical processing continues.
- MediaPipe async error: publish a user-safe error, release associated frame metadata/image, and keep shutdown idempotent.
- Lifecycle cleanup is idempotent so navigation and trip completion cannot double-close resources.

## Testing strategy

TDD order:

1. Pure camera timestamp gate and frame transform tests.
2. MediaPipe landmark adapter and asset integrity tests.
3. Pure device orientation/remapping, multi-sample calibration window, robust neutral, normalization, timestamp, reset, and unavailable-state tests.
4. Camera timebase verification/offset tests and coordinator synchronization tests for latest-not-future sample, named 150 ms staleness configuration, unverified timebase suppression, independent nullability, no-face, and session restart.
5. Camera/MediaPipe and sensor Android adapters with thin boundaries; unit tests cover their platform-neutral logic.
6. Runtime state/ViewModel and debug presentation tests.
7. Manifest, dependencies, Preview/permission/lifecycle integration.
8. Full unit regression and debug assembly.
9. Emulator validation for permission, preview, Face Landmarker result/error, sensor inventory/changes, lifecycle cleanup, and backend trip regression where available.

Hardware claims remain evidence-based: camera or rotation-vector behavior is not reported PASS when the emulator does not expose usable input. Performance measurements are approximate runtime diagnostics, not benchmarks.

## Git and verification

Implementation remains on `feature/onnx-system-framework`; no main merge/rebase/checkout/push, PR, or force push. After all old and new unit tests and `:app:assembleDebug` pass, create one commit:

`feat(android): integrate camera and device motion monitoring`

Push only `origin feature/onnx-system-framework`.
