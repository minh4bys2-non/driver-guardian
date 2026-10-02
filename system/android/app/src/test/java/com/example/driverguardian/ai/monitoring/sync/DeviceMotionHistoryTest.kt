package com.example.driverguardian.ai.monitoring.sync

import com.example.driverguardian.ai.monitoring.device.*
import org.junit.Assert.*
import org.junit.Test
import java.lang.reflect.Modifier
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicReference

class DeviceMotionHistoryTest {
    private fun metric(t: Double, state: DeviceCalibrationState = DeviceCalibrationState.AVAILABLE) =
        DeviceMotionMetrics(t, DeviceSensorSource.ROTATION_VECTOR, state, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 3)

    @Test fun choosesLatestNotFutureAndRecordsExactAge() {
        val h = DeviceMotionHistory()
        h.add(metric(1.0)); h.add(metric(1.1)); h.add(metric(1.3))
        assertEquals(1.1, h.latestAtOrBefore(1.2)!!.timestampSec, 0.0)
        assertEquals(100.0, h.associate(1.2)!!.sampleAgeMs!!, 1e-9)
        assertNull(h.latestAtOrBefore(0.9))
    }

    @Test fun namedDefaultThresholdAndCustomThresholdControlStaleness() {
        val h = DeviceMotionHistory(); h.add(metric(1.0))
        assertEquals(150.0, MonitoringSyncConfig.SENSOR_FRESHNESS_THRESHOLD_MS, 0.0)
        assertNotNull(h.associate(1.15))
        assertNull(h.associate(1.151))
        assertNotNull(h.associate(1.2, MonitoringSyncConfig(sensorFreshnessThresholdMs = 250.0)))
    }

    @Test fun unavailableSamplesAreNeverAssociatedAndHistoryIsBounded() {
        val h = DeviceMotionHistory(maxEntries = 2)
        h.add(metric(1.0)); h.add(metric(2.0)); h.add(metric(3.0, DeviceCalibrationState.UNAVAILABLE))
        assertNull(h.latestAtOrBefore(1.5))
        assertNull(h.associate(3.0))
        assertEquals(2.0, h.latestAtOrBefore(2.5)!!.timestampSec, 0.0)
    }

    @Test fun sensorWritesAndCameraReadsShareOneSynchronizationBoundary() {
        listOf("add", "latestAtOrBefore", "associate", "clear").forEach { methodName ->
            val method = DeviceMotionHistory::class.java.declaredMethods.first { it.name == methodName }
            assertTrue("$methodName must be synchronized", Modifier.isSynchronized(method.modifiers))
        }

        val history = DeviceMotionHistory(maxEntries = 120)
        val start = CountDownLatch(1)
        val failure = AtomicReference<Throwable?>()
        val writer = Thread {
            start.await()
            runCatching { repeat(5_000) { history.add(metric(it / 1_000.0)) } }
                .exceptionOrNull()?.let(failure::set)
        }
        val reader = Thread {
            start.await()
            runCatching { repeat(5_000) { history.associate(it / 1_000.0) } }
                .exceptionOrNull()?.let(failure::set)
        }
        writer.start(); reader.start(); start.countDown(); writer.join(); reader.join()
        assertNull(failure.get())
    }
}
