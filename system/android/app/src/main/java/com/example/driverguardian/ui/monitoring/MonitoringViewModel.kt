package com.example.driverguardian.ui.monitoring

import androidx.camera.core.Preview
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.ViewModel
import com.example.driverguardian.ai.monitoring.CameraPermissionState
import com.example.driverguardian.ai.monitoring.MonitoringEngineControl

class MonitoringViewModel(private val engine: MonitoringEngineControl) : ViewModel() {
    val snapshot = engine.snapshot
    val runtimeState = engine.runtimeState
    val diagnostics = engine.diagnostics
    private var activeSessionId: Int? = null
    private var permission = CameraPermissionState.UNKNOWN

    fun startSession(sessionId: Int) {
        if (activeSessionId == sessionId) return
        if (activeSessionId != null) engine.stopSession()
        activeSessionId = sessionId
        permission = CameraPermissionState.UNKNOWN
        engine.startSession(sessionId)
    }

    fun updatePermission(newPermission: CameraPermissionState) {
        if (permission == newPermission) return
        permission = newPermission
        engine.updatePermission(newPermission)
    }

    fun attachCamera(lifecycleOwner: LifecycleOwner, surfaceProvider: Preview.SurfaceProvider) {
        engine.attachCamera(lifecycleOwner, surfaceProvider)
    }

    fun stopSession() {
        if (activeSessionId == null) return
        engine.stopSession()
        activeSessionId = null
    }

    override fun onCleared() {
        engine.close()
        activeSessionId = null
    }
}
