package com.example.driverguardian.ui.monitoring

import androidx.camera.core.Preview
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.ViewModel
import com.example.driverguardian.ai.monitoring.CameraPermissionState
import com.example.driverguardian.ai.monitoring.MonitoringEngineControl
import com.example.driverguardian.ai.monitoring.ThreadDebugLogger

class MonitoringViewModel(private val engine: MonitoringEngineControl) : ViewModel() {
    val snapshot = engine.snapshot
    val runtimeState = engine.runtimeState
    val diagnostics = engine.diagnostics
    val balanceBrightnessEnabled = engine.balanceBrightnessEnabled

    private var activeSessionId: Int? = null
    private var permission = CameraPermissionState.UNKNOWN

    fun startSession(sessionId: Int) {
        if (activeSessionId == sessionId) return
        if (activeSessionId != null) engine.stopSession()
        activeSessionId = sessionId
        permission = CameraPermissionState.UNKNOWN
        ThreadDebugLogger.infoUi("MonitoringViewModel.startSession(sessionId=$sessionId)")
        engine.startSession(sessionId)
    }

    fun updatePermission(newPermission: CameraPermissionState) {
        if (permission == newPermission) return
        permission = newPermission
        ThreadDebugLogger.logUi("MonitoringViewModel.updatePermission(permission=$newPermission)")
        engine.updatePermission(newPermission)
    }

    fun attachCamera(lifecycleOwner: LifecycleOwner, surfaceProvider: Preview.SurfaceProvider) {
        engine.attachCamera(lifecycleOwner, surfaceProvider)
    }

    fun setBalanceBrightness(enabled: Boolean) {
        ThreadDebugLogger.infoUi("MonitoringViewModel.setBalanceBrightness(enabled=$enabled)")
        engine.setBalanceBrightness(enabled)
    }

    fun stopSession() {
        if (activeSessionId == null) return
        ThreadDebugLogger.infoUi("MonitoringViewModel.stopSession(sessionId=$activeSessionId)")
        engine.stopSession()
        activeSessionId = null
    }

    override fun onCleared() {
        engine.close()
        activeSessionId = null
    }
}
