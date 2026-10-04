package com.example.driverguardian.ai.physical

import com.example.driverguardian.ai.physical.model.LandmarkPoint
import com.example.driverguardian.ai.physical.pipeline.PhysicalBranchConfig
import com.example.driverguardian.ai.physical.pipeline.PhysicalBranchProcessor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PhysicalBranchProcessorTest {

    private fun createFaceLandmarks(dx: Float = 0f, dy: Float = 0f): List<LandmarkPoint> {
        val list = MutableList(500) { LandmarkPoint(200f, 200f) }
        // Left eye: 362, 385, 387, 263, 373, 380
        list[362] = LandmarkPoint(400f + dx, 150f + dy)
        list[385] = LandmarkPoint(420f + dx, 140f + dy)
        list[387] = LandmarkPoint(440f + dx, 140f + dy)
        list[263] = LandmarkPoint(460f + dx, 150f + dy)
        list[373] = LandmarkPoint(440f + dx, 160f + dy)
        list[380] = LandmarkPoint(420f + dx, 160f + dy)

        // Right eye: 33, 160, 158, 133, 153, 144
        list[33] = LandmarkPoint(180f + dx, 150f + dy)
        list[160] = LandmarkPoint(200f + dx, 140f + dy)
        list[158] = LandmarkPoint(220f + dx, 140f + dy)
        list[133] = LandmarkPoint(240f + dx, 150f + dy)
        list[153] = LandmarkPoint(220f + dx, 160f + dy)
        list[144] = LandmarkPoint(200f + dx, 160f + dy)

        // Mouth: 61, 37, 267, 291, 314, 84
        list[61] = LandmarkPoint(270f + dx, 350f + dy)
        list[37] = LandmarkPoint(300f + dx, 335f + dy)
        list[267] = LandmarkPoint(340f + dx, 335f + dy)
        list[291] = LandmarkPoint(370f + dx, 350f + dy)
        list[314] = LandmarkPoint(340f + dx, 365f + dy)
        list[84] = LandmarkPoint(300f + dx, 365f + dy)

        // Head pose points: 1 (nose), 152 (chin)
        list[1] = LandmarkPoint(320f + dx, 240f + dy)
        list[152] = LandmarkPoint(320f + dx, 450f + dy)

        return list
    }

    @Test
    fun testNullLandmarksReturnsNullMetricsWithoutDefaults() {
        val processor = PhysicalBranchProcessor()
        val metrics = processor.processLandmarks(null, 640, 480, 1.0)

        assertFalse(metrics.faceDetected)
        assertFalse(metrics.eyeReady)
        assertFalse(metrics.mouthReady)
        assertFalse(metrics.headReady)
        assertFalse(metrics.headCalibrated)
        assertFalse(metrics.poseValid)

        assertNull("EAR must be null when face is missing", metrics.ear)
        assertNull("MAR must be null when face is missing", metrics.mar)
        assertNull("Eye state must be null when face is missing", metrics.eyeState)
        assertNull("Pitch must be null when face is missing", metrics.pitchDeg)
        assertNull("Nodding must be null when face is missing", metrics.nodding)
        assertNull("OverAngle must be null when face is missing", metrics.overAngle)
    }

    @Test
    fun testNeutralCalibrationAndNormalizedAngles() {
        // Calibration requires 5 frames in this config
        val config = PhysicalBranchConfig(calibrationFrames = 5)
        val processor = PhysicalBranchProcessor(config)

        var t = 0.0
        val landmarks = createFaceLandmarks()

        // Supply 4 frames -> not calibrated yet
        for (i in 0 until 4) {
            t += 0.033
            val m = processor.processLandmarks(landmarks, 640, 480, t)
            assertFalse(m.headCalibrated)
            assertNull(m.pitchDeg)
        }

        // 5th frame -> calibrated!
        t += 0.033
        val m5 = processor.processLandmarks(landmarks, 640, 480, t)
        assertTrue("Processor must be calibrated on 5th frame", m5.headCalibrated)
        assertTrue(m5.headReady)
        assertNotNull(m5.pitchDeg)
        assertNotNull(m5.yawDeg)
        assertNotNull(m5.rollDeg)

        // Because raw pose was identical for all 5 frames, normalized angles must be ~0.0
        assertEquals(0.0, m5.pitchDeg!!, 1e-2)
        assertEquals(0.0, m5.yawDeg!!, 1e-2)
        assertEquals(0.0, m5.rollDeg!!, 1e-2)
    }

    @Test
    fun testResetClearsAllTemporalState() {
        val config = PhysicalBranchConfig(calibrationFrames = 5)
        val processor = PhysicalBranchProcessor(config)
        val landmarks = createFaceLandmarks()

        var t = 0.0
        for (i in 0 until 5) {
            t += 0.033
            processor.processLandmarks(landmarks, 640, 480, t)
        }
        assertTrue(processor.neutralPose != null)

        processor.reset()
        assertNull(processor.neutralPose)
        assertNull(processor.lastTimestampSec)
        assertFalse(processor.nodding)
    }
}
