package com.example.driverguardian.ai.monitoring

import com.example.driverguardian.ai.monitoring.device.*
import com.example.driverguardian.ai.monitoring.model.*
import com.example.driverguardian.ai.monitoring.sync.*
import com.example.driverguardian.ai.monitoring.time.*
import com.example.driverguardian.ai.physical.model.LandmarkPoint
import com.example.driverguardian.ai.physical.pipeline.PhysicalBranch
import com.example.driverguardian.ai.physical.pipeline.PhysicalBranchProcessor
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

class DriverMonitoringCoordinator(
    private val physicalBranch: PhysicalBranch = PhysicalBranchProcessor(),
    private val deviceProcessor: DeviceMotionProcessor = DeviceMotionProcessor(),
    cameraTimestampSource: CameraTimestampSource,
    private val syncConfig: MonitoringSyncConfig = MonitoringSyncConfig(),
) {
    private val timebase = MonitoringTimebaseNormalizer(cameraTimestampSource)
    private val deviceHistory = DeviceMotionHistory()
    private val _snapshot = MutableStateFlow<MonitoringSnapshot?>(null)
    private val _runtimeState = MutableStateFlow(MonitoringRuntimeState())
    private val _diagnostics = MutableStateFlow(MonitoringDiagnostics(cameraTimestampSource = cameraTimestampSource))
    private var activeGeneration: Long? = null
    private var lastCameraTimestampSec: Double? = null
    private var lastSensorTimestampSec: Double? = null

    val snapshot: StateFlow<MonitoringSnapshot?> = _snapshot.asStateFlow()
    val runtimeState: StateFlow<MonitoringRuntimeState> = _runtimeState.asStateFlow()
    val diagnostics: StateFlow<MonitoringDiagnostics> = _diagnostics.asStateFlow()

    @Synchronized fun startSession(generation: Long) {
        activeGeneration = generation
        physicalBranch.reset()
        deviceProcessor.reset()
        deviceHistory.clear()
        timebase.reset()
        lastCameraTimestampSec = null
        lastSensorTimestampSec = null
        _snapshot.value = null
        _diagnostics.value = MonitoringDiagnostics(cameraTimestampSource = _diagnostics.value.cameraTimestampSource)
        _runtimeState.value = MonitoringRuntimeState(
            camera = CameraRuntimeState.STARTING,
            face = FaceRuntimeState.INITIALIZING,
            sensor = SensorRuntimeState.CALIBRATING,
        )
        updateTimebaseDiagnostics()
    }

    @Synchronized fun stopSession() {
        activeGeneration = null
        _runtimeState.value = MonitoringRuntimeState()
    }

    @Synchronized fun updateCameraPermission(granted: Boolean) {
        _runtimeState.value = _runtimeState.value.copy(
            camera = if (granted) CameraRuntimeState.STARTING else CameraRuntimeState.PERMISSION_REQUIRED,
            message = if (granted) null else "Camera permission is required for facial monitoring",
        )
    }

    @Synchronized fun reportCameraError(generation: Long, message: String) {
        if (generation != activeGeneration) return
        _runtimeState.value = _runtimeState.value.copy(
            camera = CameraRuntimeState.ERROR,
            face = FaceRuntimeState.ERROR,
            message = message,
        )
    }

    @Synchronized fun reportSensorUnavailable(message: String) {
        _runtimeState.value = _runtimeState.value.copy(sensor = SensorRuntimeState.UNAVAILABLE, message = message)
    }

    @Synchronized fun reportCameraTimestampSource(source: CameraTimestampSource) {
        timebase.updateCameraTimestampSource(source)
        _diagnostics.value = _diagnostics.value.copy(cameraTimestampSource = source)
        updateTimebaseDiagnostics()
    }

    @Synchronized fun onSensorOrientation(orientation: DeviceOrientation) {
        if (activeGeneration == null) return
        val started = System.nanoTime()
        val metrics = deviceProcessor.process(orientation) ?: return
        val sensorDeltaSec = lastSensorTimestampSec?.let { orientation.timestampSec - it }
        lastSensorTimestampSec = orientation.timestampSec
        deviceHistory.add(metrics)
        _runtimeState.value = _runtimeState.value.copy(
            sensor = when (metrics.calibrationState) {
                DeviceCalibrationState.CALIBRATING -> SensorRuntimeState.CALIBRATING
                DeviceCalibrationState.AVAILABLE -> SensorRuntimeState.AVAILABLE
                DeviceCalibrationState.UNAVAILABLE -> SensorRuntimeState.UNAVAILABLE
            },
        )
        _diagnostics.value = _diagnostics.value.copy(
            sensorHz = sensorDeltaSec?.takeIf { it > 0.0 }?.let { 1.0 / it },
            sensorProcessingMs = (System.nanoTime() - started) / 1_000_000.0,
        )
    }

    @Synchronized fun onCameraResult(
        landmarks: List<LandmarkPoint>?,
        width: Int,
        height: Int,
        cameraTimestampNs: Long,
        receiptElapsedNs: Long,
        generation: Long,
        inferenceMs: Double? = null,
    ) {
        if (generation != activeGeneration) return
        timebase.observe(cameraTimestampNs, receiptElapsedNs)
        val normalizedNs = timebase.normalizeCameraTimestampNs(cameraTimestampNs) ?: run {
            updateTimebaseDiagnostics()
            return
        }
        val timestampSec = normalizedNs / 1_000_000_000.0
        val cameraDeltaSec = lastCameraTimestampSec?.let { timestampSec - it }
        lastCameraTimestampSec = timestampSec
        val started = System.nanoTime()
        val physical = runCatching { physicalBranch.processLandmarks(landmarks, width, height, timestampSec) }.getOrNull() ?: return
        val processingMs = (System.nanoTime() - started) / 1_000_000.0
        _snapshot.value = MonitoringSnapshot(physical, deviceHistory.associate(timestampSec, syncConfig), timestampSec)
        _runtimeState.value = _runtimeState.value.copy(
            camera = CameraRuntimeState.AVAILABLE,
            face = if (physical.faceDetected) FaceRuntimeState.FACE_DETECTED else FaceRuntimeState.NO_FACE,
        )
        _diagnostics.value = _diagnostics.value.copy(
            cameraFps = cameraDeltaSec?.takeIf { it > 0.0 }?.let { 1.0 / it },
            inferenceMs = inferenceMs,
            physicalProcessingMs = processingMs,
        )
        updateTimebaseDiagnostics()
    }

    private fun updateTimebaseDiagnostics() {
        _diagnostics.value = _diagnostics.value.copy(
            timebaseRelation = timebase.relation,
            timebaseOffsetNs = timebase.offsetNs,
        )
    }
}
