package com.example.driverguardian.ai.physical

import com.example.driverguardian.ai.physical.temporal.AdaptiveHmmFsm
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AdaptiveHmmFsmTest {

    @Test
    fun testEyeCalibrationAndPerclosCalculation() {
        val detector = AdaptiveHmmFsm(mode = "eye", fps = 30, initDurationSec = 0.5) // 15 frames init
        var t = 0.0

        // Supply 14 frames of EAR ~ 0.32 -> not initialized yet
        for (i in 0 until 14) {
            t += 1.0 / 30.0
            val res = detector.process(0.32, t)
            assertFalse(res.initialized)
            assertNull(res.p80Threshold)
        }

        // 15th frame -> initializes!
        t += 1.0 / 30.0
        val res15 = detector.process(0.32, t)
        assertTrue("Detector must be initialized on 15th frame", res15.initialized)
        assertNotNull(res15.p80Threshold)
        assertNotNull(detector.normal)
        assertTrue("Normal EAR must be ~0.32", detector.normal!! > 0.28)

        // Safety guard: if value >= normal * 0.75, state must be 0
        t += 1.0 / 30.0
        val resGuard = detector.process(0.30, t)
        assertEquals("Safety guard forces state 0 for open eye", 0, resGuard.state)
    }

    @Test
    fun testMouthModePomCalculation() {
        val detector = AdaptiveHmmFsm(mode = "mouth", fps = 30, initDurationSec = 0.5)
        var t = 0.0
        for (i in 0 until 15) {
            t += 1.0 / 30.0
            detector.process(0.18, t)
        }
        assertTrue(detector.initialized)

        // Yawning value 0.65 -> state 1
        t += 1.0 / 30.0
        val res = detector.process(0.65, t)
        assertEquals(1, res.state)
    }

    @Test
    fun testResetClearsState() {
        val detector = AdaptiveHmmFsm(mode = "eye", fps = 30, initDurationSec = 0.5)
        var t = 0.0
        for (i in 0 until 15) {
            t += 1.0 / 30.0
            detector.process(0.32, t)
        }
        assertTrue(detector.initialized)

        detector.reset()
        assertFalse(detector.initialized)
        assertNull(detector.normal)
        assertNull(detector.eventReference)
    }
}
