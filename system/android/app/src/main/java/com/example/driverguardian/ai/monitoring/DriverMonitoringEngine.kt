package com.example.driverguardian.ai.monitoring

import android.content.Context
import android.hardware.display.DisplayManager
import android.view.Display
import androidx.camera.core.Preview
import androidx.lifecycle.LifecycleOwner
import com.example.driverguardian.ai.monitoring.camera.CameraXFrameSource
import com.example.driverguardian.ai.monitoring.camera.MediaPipeFaceLandmarkerClient
import com.example.driverguardian.ai.monitoring.device.AndroidDeviceOrientationSource
import com.example.driverguardian.ai.monitoring.device.SafeDisplayRotationProvider
import com.example.driverguardian.ai.monitoring.model.MonitoringDiagnostics
import com.example.driverguardian.ai.monitoring.model.MonitoringRuntimeState
import com.example.driverguardian.ai.monitoring.model.MonitoringSnapshot
import com.example.driverguardian.ai.monitoring.time.CameraTimestampSource
import kotlinx.coroutines.flow.StateFlow

enum class CameraPermissionState { UNKNOWN, GRANTED, DENIED, UNAVAILABLE }

interface MonitoringEngineControl : AutoCloseable {
    val snapshot: StateFlow<MonitoringSnapshot?>
    val runtimeState: StateFlow<MonitoringRuntimeState>
    val diagnostics: StateFlow<MonitoringDiagnostics>
    fun startSession(sessionId: Int)
    fun updatePermission(permission: CameraPermissionState)
    fun attachCamera(lifecycleOwner: LifecycleOwner, surfaceProvider: Preview.SurfaceProvider) = Unit
    fun stopSession()
}

class DriverMonitoringEngine(private val context: Context) : MonitoringEngineControl {
    private val coordinator = DriverMonitoringCoordinator(cameraTimestampSource = CameraTimestampSource.UNKNOWN)
    private val sensorSource = AndroidDeviceOrientationSource(context)
    private val displayRotation = SafeDisplayRotationProvider {
        (context.getSystemService(Context.DISPLAY_SERVICE) as DisplayManager)
            .getDisplay(Display.DEFAULT_DISPLAY)?.rotation
    }
    private var cameraSource: CameraXFrameSource? = null
    private var lifecycleOwner: LifecycleOwner? = null
    private var surfaceProvider: Preview.SurfaceProvider? = null
    private var permission = CameraPermissionState.UNKNOWN
    private var activeSessionId: Int? = null

    override val snapshot = coordinator.snapshot
    override val runtimeState = coordinator.runtimeState
    override val diagnostics = coordinator.diagnostics

    override fun startSession(sessionId: Int) {
        if (activeSessionId == sessionId) return
        if (activeSessionId != null) stopSession()
        activeSessionId = sessionId
        coordinator.startSession(sessionId.toLong())
        val sensorStarted = sensorSource.start(
            displayRotationProvider = displayRotation::rotation,
            callback = coordinator::onSensorOrientation,
        )
        if (!sensorStarted) coordinator.reportSensorUnavailable("No usable device-orientation sensor")
        coordinator.updateCameraPermission(permission == CameraPermissionState.GRANTED)
        bindCameraIfReady()
    }

    override fun updatePermission(permission: CameraPermissionState) {
        if (this.permission == permission) return
        this.permission = permission
        coordinator.updateCameraPermission(permission == CameraPermissionState.GRANTED)
        if (permission == CameraPermissionState.GRANTED) bindCameraIfReady() else cameraSource?.stop()
    }

    override fun attachCamera(lifecycleOwner: LifecycleOwner, surfaceProvider: Preview.SurfaceProvider) {
        this.lifecycleOwner = lifecycleOwner
        this.surfaceProvider = surfaceProvider
        bindCameraIfReady()
    }

    private fun bindCameraIfReady() {
        val sessionId = activeSessionId ?: return
        val owner = lifecycleOwner ?: return
        val surface = surfaceProvider ?: return
        if (permission != CameraPermissionState.GRANTED) return
        try {
            if (cameraSource == null) {
                val face = MediaPipeFaceLandmarkerClient(
                    context = context,
                    onResult = { result -> coordinator.onCameraResult(
                        result.landmarks, result.width, result.height, result.cameraTimestampNs,
                        result.receiptElapsedNs, result.generation, result.inferenceMs,
                    ) },
                    onError = coordinator::reportCameraError,
                )
                cameraSource = CameraXFrameSource(
                    context,
                    face,
                    coordinator::reportCameraError,
                    coordinator::reportCameraTimestampSource,
                )
            }
            cameraSource!!.start(owner, surface, sessionId.toLong())
        } catch (error: Exception) {
            coordinator.reportCameraError(sessionId.toLong(), error.message ?: "Monitoring camera initialization failed")
        }
    }

    override fun stopSession() {
        if (activeSessionId == null) return
        cameraSource?.stop()
        sensorSource.stop()
        coordinator.stopSession()
        activeSessionId = null
    }

    override fun close() {
        stopSession()
        cameraSource?.close()
        cameraSource = null
        sensorSource.close()
    }
}
