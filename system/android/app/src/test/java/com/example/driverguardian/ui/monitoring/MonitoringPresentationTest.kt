package com.example.driverguardian.ui.monitoring

import com.example.driverguardian.ai.monitoring.model.*
import org.junit.Assert.*
import org.junit.Test

class MonitoringPresentationTest {
    @Test fun nullMetricsRemainPlaceholdersAndDomainsStaySeparate() {
        val p = MonitoringPresentation.from(null, MonitoringRuntimeState(
            camera = CameraRuntimeState.AVAILABLE,
            face = FaceRuntimeState.NO_FACE,
            sensor = SensorRuntimeState.UNAVAILABLE,
        ), MonitoringDiagnostics())
        assertEquals("—", p.ear)
        assertEquals("—", p.perclos)
        assertEquals("—", p.headPose)
        assertEquals("Không thấy khuôn mặt", p.faceStatus)
        assertEquals("DEMO: cảnh báo nguy hiểm", p.dangerActionLabel)
    }

    @Test fun missingPhysicalDataAndTimebaseAreExplicitNotFabricated() {
        val p = MonitoringPresentation.from(null, MonitoringRuntimeState(), MonitoringDiagnostics())
        assertTrue(p.timebaseStatus.contains("UNVERIFIED"))
    }

    @Test fun performanceExposesCameraPipelineButNotOptionalDeviceMotion() {
        val p = MonitoringPresentation.from(
            MonitoringSnapshot(null, null, 1.0),
            MonitoringRuntimeState(),
            MonitoringDiagnostics(cameraFps = 12.5, inferenceMs = 7.0, sensorHz = 50.0),
        )
        assertTrue(p.performance.contains("Camera 12.5 FPS"))
        assertTrue(p.performance.contains("MediaPipe 7.0 ms"))
        assertFalse(p.performance.contains("Sensor"))
    }

}
