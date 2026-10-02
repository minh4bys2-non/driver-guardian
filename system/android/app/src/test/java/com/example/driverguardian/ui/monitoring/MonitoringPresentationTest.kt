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
        assertEquals("—", p.headPose)
        assertEquals("—", p.deviceOrientation)
        assertEquals("—", p.rawDeviceOrientation)
        assertEquals("Không thấy khuôn mặt", p.faceStatus)
        assertEquals("Không có cảm biến", p.sensorStatus)
        assertEquals("DEMO: cảnh báo nguy hiểm", p.dangerActionLabel)
    }

    @Test fun calibrationAndTimebaseAreExplicitNotFabricated() {
        val p = MonitoringPresentation.from(null, MonitoringRuntimeState(sensor = SensorRuntimeState.CALIBRATING), MonitoringDiagnostics())
        assertEquals("Đang hiệu chuẩn", p.sensorStatus)
        assertTrue(p.timebaseStatus.contains("UNVERIFIED"))
    }


    @Test fun performanceAndRawDeviceOrientationAreReportedWithoutChangingDomains() {
        val motion = com.example.driverguardian.ai.monitoring.device.DeviceMotionMetrics(
            timestampSec = 1.0,
            source = com.example.driverguardian.ai.monitoring.device.DeviceSensorSource.ROTATION_VECTOR,
            calibrationState = com.example.driverguardian.ai.monitoring.device.DeviceCalibrationState.AVAILABLE,
            rawPitchDeg = 4.0, rawRollDeg = 5.0, rawYawDeg = null,
            deltaPitchDeg = 1.0, deltaRollDeg = 2.0, deltaYawDeg = null,
            accuracy = 3,
        )
        val p = MonitoringPresentation.from(
            MonitoringSnapshot(null, motion, 1.0),
            MonitoringRuntimeState(),
            MonitoringDiagnostics(cameraFps = 12.5, sensorHz = 50.0),
        )
        assertTrue(p.rawDeviceOrientation.contains("P 4.0"))
        assertTrue(p.deviceOrientation.contains("P 1.0"))
        assertTrue(p.performance.contains("Camera 12.5 FPS"))
        assertTrue(p.performance.contains("Sensor 50.0 Hz"))
    }
}
