package com.example.driverguardian.ai.monitoring

import com.example.driverguardian.ai.monitoring.device.*
import com.example.driverguardian.ai.monitoring.model.CameraRuntimeState
import com.example.driverguardian.ai.monitoring.time.*
import com.example.driverguardian.ai.physical.model.*
import com.example.driverguardian.ai.physical.pipeline.PhysicalBranch
import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread

class DriverMonitoringCoordinatorTest {
    private fun verifyRealtime(c: DriverMonitoringCoordinator, generation: Long, startNs: Long = 1_000_000_000L) {
        repeat(8) { i ->
            val timestamp = startNs + i * 10_000_000L
            c.onCameraResult(null, 640, 480, timestamp, timestamp + 2_000_000L, generation)
        }
    }

    private open class FakePhysical : PhysicalBranch {
        var calls = 0
        var resets = 0
        override fun reset() { resets++ }
        override fun processLandmarks(landmarks: List<LandmarkPoint>?, imageWidth: Int, imageHeight: Int, timestampSec: Double): PhysicalMetrics {
            calls++
            return PhysicalMetrics(timestampSec, landmarks != null, false, false, false, false,
                ear=null, eyeState=null, perclosPct=null, p80EarThreshold=null, blinkRatePerMin=null,
                blinkDetected=false, eyeClosureDurationMs=null, mar=null, mouthState=null, pomPct=null,
                yawnDetected=false, yawningFrequencyPerMin=null, mouthOpenDurationMs=null,
                pitchDeg=null, yawDeg=null, rollDeg=null, overAngle=null, overAnglePct=null,
                nodding=null, nodDetected=false, nodDurationMs=null, noddingFrequencyPerMin=null)
        }
    }

    @Test fun verifiedRealtimeSynchronizesLatestNotFutureSensor() {
        val physical = FakePhysical()
        val c = DriverMonitoringCoordinator(physical, cameraTimestampSource = CameraTimestampSource.REALTIME)
        c.startSession(7)
        c.onSensorOrientation(DeviceOrientation(1.0, 1.0, 2.0, null, 3, DeviceSensorSource.ROTATION_VECTOR))
        verifyRealtime(c, 7, 1_030_000_000L)
        assertEquals(1, physical.calls)
        assertEquals(100.0, c.snapshot.value!!.deviceMotion!!.sampleAgeMs!!, 1e-6)
        assertFalse(c.snapshot.value!!.physical!!.faceDetected)
    }

    @Test fun unverifiedClockSuppressesFrameUntilRelationshipKnown() {
        val physical = FakePhysical()
        val c = DriverMonitoringCoordinator(physical, cameraTimestampSource = CameraTimestampSource.UNKNOWN)
        c.startSession(1)
        c.onCameraResult(null, 1, 1, 10L, 100L, 1)
        assertNull(c.snapshot.value)
        assertEquals(0, physical.calls)
    }

    @Test fun cameraRuntimeCanPromoteUnknownTimestampSourceToRealtime() {
        val physical = FakePhysical()
        val c = DriverMonitoringCoordinator(physical, cameraTimestampSource = CameraTimestampSource.UNKNOWN)
        c.startSession(1)
        c.reportCameraTimestampSource(CameraTimestampSource.REALTIME)
        verifyRealtime(c, 1)
        assertNotNull(c.snapshot.value)
        assertEquals(1, physical.calls)
        assertEquals(TimebaseRelation.DIRECT_MONOTONIC, c.diagnostics.value.timebaseRelation)
        assertEquals(0L, c.diagnostics.value.timebaseOffsetNs)
    }

    @Test fun staleSensorDoesNotErasePhysicalResult() {
        val c = DriverMonitoringCoordinator(FakePhysical(), cameraTimestampSource = CameraTimestampSource.REALTIME)
        c.startSession(1)
        c.onSensorOrientation(DeviceOrientation(1.0, 1.0, 2.0, null, 3, DeviceSensorSource.ROTATION_VECTOR))
        verifyRealtime(c, 1, 1_130_000_000L)
        assertNotNull(c.snapshot.value!!.physical)
        assertNull(c.snapshot.value!!.deviceMotion)
    }

    @Test fun ignoresResultFromPreviousGeneration() {
        val physical = FakePhysical()
        val c = DriverMonitoringCoordinator(physical, cameraTimestampSource = CameraTimestampSource.REALTIME)
        c.startSession(1)
        c.startSession(2)
        c.onCameraResult(null, 640, 480, 1_000_000_000L, 1_000_000_000L, 1)
        assertNull(c.snapshot.value)
        assertEquals(0, physical.calls)
        assertEquals(2, physical.resets)
    }

    @Test fun ignoresCameraErrorFromPreviousGeneration() {
        val c = DriverMonitoringCoordinator(FakePhysical(), cameraTimestampSource = CameraTimestampSource.REALTIME)
        c.startSession(1)
        c.startSession(2)
        c.reportCameraError(1, "late failure")
        assertEquals(CameraRuntimeState.STARTING, c.runtimeState.value.camera)
        assertNull(c.runtimeState.value.message)

        c.reportCameraError(2, "current failure")
        assertEquals(CameraRuntimeState.ERROR, c.runtimeState.value.camera)
        assertEquals("current failure", c.runtimeState.value.message)
    }

    @Test fun stopRejectsLateCallbacksAndNewSessionResetsTimeGate() {
        val physical = FakePhysical()
        val c = DriverMonitoringCoordinator(physical, cameraTimestampSource = CameraTimestampSource.REALTIME)
        c.startSession(1); c.stopSession()
        c.onCameraResult(null, 1, 1, 2_000_000_000L, 2_000_000_000L, 1)
        assertNull(c.snapshot.value)
        c.startSession(2)
        verifyRealtime(c, 2)
        assertNotNull(c.snapshot.value)
    }

    @Test fun diagnosticsReportCameraAndSensorRatesAndResetForNewSession() {
        val c = DriverMonitoringCoordinator(FakePhysical(), cameraTimestampSource = CameraTimestampSource.REALTIME)
        c.startSession(1)
        c.onSensorOrientation(DeviceOrientation(1.0, 1.0, 2.0, null, 3, DeviceSensorSource.ROTATION_VECTOR))
        c.onSensorOrientation(DeviceOrientation(1.1, 1.0, 2.0, null, 3, DeviceSensorSource.ROTATION_VECTOR))
        verifyRealtime(c, 1)
        c.onCameraResult(null, 1, 1, 1_570_000_000L, 1_572_000_000L, 1)
        assertEquals(10.0, c.diagnostics.value.sensorHz!!, 1e-6)
        assertEquals(2.0, c.diagnostics.value.cameraFps!!, 1e-6)

        c.startSession(2)
        assertNull(c.diagnostics.value.sensorHz)
        assertNull(c.diagnostics.value.cameraFps)
    }

    @Test fun sessionRestartWaitsForInFlightCameraCallbackAndCannotPublishOldGeneration() {
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        val physical = object : FakePhysical() {
            override fun processLandmarks(
                landmarks: List<LandmarkPoint>?, imageWidth: Int, imageHeight: Int, timestampSec: Double,
            ): PhysicalMetrics {
                entered.countDown()
                assertTrue(release.await(2, TimeUnit.SECONDS))
                return super.processLandmarks(landmarks, imageWidth, imageHeight, timestampSec)
            }
        }
        val c = DriverMonitoringCoordinator(physical, cameraTimestampSource = CameraTimestampSource.REALTIME)
        c.startSession(1)
        repeat(7) { i ->
            val timestamp = 1_000_000_000L + i * 10_000_000L
            c.onCameraResult(null, 1, 1, timestamp, timestamp + 1_000_000L, 1)
        }
        val callback = thread { c.onCameraResult(null, 1, 1, 1_070_000_000L, 1_071_000_000L, 1) }
        assertTrue(entered.await(1, TimeUnit.SECONDS))
        val restart = thread { c.startSession(2) }
        Thread.sleep(50)
        assertTrue("restart must serialize behind the in-flight callback", restart.isAlive)
        release.countDown()
        callback.join(1_000)
        restart.join(1_000)
        assertFalse(callback.isAlive)
        assertFalse(restart.isAlive)
        assertNull(c.snapshot.value)
    }
}
