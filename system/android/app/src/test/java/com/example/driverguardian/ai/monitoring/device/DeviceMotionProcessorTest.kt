package com.example.driverguardian.ai.monitoring.device

import org.junit.Assert.*
import org.junit.Test

class DeviceMotionProcessorTest {
    private fun sample(t: Double, pitch: Double, roll: Double, yaw: Double? = null) =
        DeviceOrientation(t, pitch, roll, yaw, 3, DeviceSensorSource.ROTATION_VECTOR)

    @Test fun calibratesOnlyAfterWindowAndMinimumSamples() {
        val p = DeviceMotionProcessor(DeviceMotionConfig(calibrationDurationSec = 0.75, minimumCalibrationSamples = 10))
        repeat(9) { i -> assertEquals(DeviceCalibrationState.CALIBRATING, p.process(sample(i * 0.1, 10.0, -5.0))!!.calibrationState) }
        val ready = p.process(sample(0.9, 12.0, -3.0))!!
        assertEquals(DeviceCalibrationState.AVAILABLE, ready.calibrationState)
        assertEquals(2.0, ready.deltaPitchDeg!!, 1e-9)
        assertEquals(2.0, ready.deltaRollDeg!!, 1e-9)
    }

    @Test fun calibrationUsesWrappedMedian() {
        val p = DeviceMotionProcessor(DeviceMotionConfig(0.5, 6))
        val values = listOf(179.0, -179.0, 178.0, -178.0, 179.5, -179.5)
        values.forEachIndexed { i, v -> p.process(sample(i * 0.1, 0.0, 0.0, v)) }
        val result = p.process(sample(0.6, 0.0, 0.0, 180.0))!!
        assertEquals(DeviceCalibrationState.AVAILABLE, result.calibrationState)
        assertTrue(kotlin.math.abs(result.deltaYawDeg!!) < 2.0)
    }

    @Test fun optionalYawStaysNullAndResetRequiresRecalibration() {
        val p = DeviceMotionProcessor(DeviceMotionConfig(0.5, 6))
        repeat(7) { i -> p.process(sample(i * 0.1, 2.0, 3.0, null)) }
        assertNull(p.process(sample(0.8, 3.0, 4.0, null))!!.deltaYawDeg)
        p.reset()
        assertEquals(DeviceCalibrationState.CALIBRATING, p.process(sample(1.0, 3.0, 4.0))!!.calibrationState)
    }

    @Test fun rejectsNonIncreasingSamplesAndExposesUnavailableWithoutZeros() {
        val p = DeviceMotionProcessor()
        assertNotNull(p.process(sample(1.0, 0.0, 0.0)))
        assertNull(p.process(sample(1.0, 1.0, 1.0)))
        val unavailable = p.markUnavailable(2.0)
        assertEquals(DeviceCalibrationState.UNAVAILABLE, unavailable.calibrationState)
        assertNull(unavailable.rawPitchDeg)
        assertNull(unavailable.deltaPitchDeg)
    }
}
