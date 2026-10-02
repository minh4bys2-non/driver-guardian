package com.example.driverguardian.ui.monitoring

import com.example.driverguardian.ai.monitoring.MonitoringEngineControl
import com.example.driverguardian.ai.monitoring.CameraPermissionState
import com.example.driverguardian.ai.monitoring.model.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import org.junit.Assert.*
import org.junit.Test

class MonitoringViewModelTest {
    private class FakeEngine : MonitoringEngineControl {
        override val snapshot = MutableStateFlow<MonitoringSnapshot?>(null)
        override val runtimeState = MutableStateFlow(MonitoringRuntimeState())
        override val diagnostics = MutableStateFlow(MonitoringDiagnostics())
        var starts = mutableListOf<Int>()
        var permissions = mutableListOf<CameraPermissionState>()
        var stops = 0
        override fun startSession(sessionId: Int) { starts += sessionId }
        override fun updatePermission(permission: CameraPermissionState) { permissions += permission }
        override fun stopSession() { stops++ }
        override fun close() = Unit
    }

    @Test fun oneSessionStartsOnceAndPermissionRecoveryDoesNotRestartSensor() {
        val engine = FakeEngine(); val vm = MonitoringViewModel(engine)
        vm.startSession(10)
        vm.startSession(10)
        vm.updatePermission(CameraPermissionState.DENIED)
        vm.updatePermission(CameraPermissionState.GRANTED)
        vm.updatePermission(CameraPermissionState.GRANTED)
        assertEquals(listOf(10), engine.starts)
        assertEquals(listOf(CameraPermissionState.DENIED, CameraPermissionState.GRANTED), engine.permissions)
        assertEquals(0, engine.stops)
    }

    @Test fun sessionChangeStopsOldAndStartsFreshCalibration() {
        val engine = FakeEngine(); val vm = MonitoringViewModel(engine)
        vm.startSession(1); vm.startSession(2)
        assertEquals(listOf(1, 2), engine.starts)
        assertEquals(1, engine.stops)
    }

    @Test fun stopIsIdempotent() {
        val engine = FakeEngine(); val vm = MonitoringViewModel(engine)
        vm.startSession(1); vm.stopSession(); vm.stopSession()
        assertEquals(1, engine.stops)
    }
}
