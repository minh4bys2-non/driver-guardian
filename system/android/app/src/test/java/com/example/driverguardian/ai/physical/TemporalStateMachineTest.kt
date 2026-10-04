package com.example.driverguardian.ai.physical

import com.example.driverguardian.ai.physical.temporal.TemporalStateMachine
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class TemporalStateMachineTest {

    @Test
    fun testValidBlinkDetection() {
        val fsm = TemporalStateMachine(mode = "eye", fps = 30) // min 100ms, max 2000ms

        // Start open (0) at t=0
        var res = fsm.process(0, 0.0)
        assertFalse(res.eventDone)
        assertEquals(0.0, res.currentDurationMs, 1e-4)

        // Eye closes (1) at t=0.05
        res = fsm.process(1, 0.05)
        assertFalse(res.eventDone)
        assertEquals(0.0, res.currentDurationMs, 1e-4)

        // Eye closed (1) at t=0.25 (duration = 200ms)
        res = fsm.process(1, 0.25)
        assertFalse(res.eventDone)
        assertEquals(200.0, res.currentDurationMs, 1e-4)
        assertFalse(res.prolonged)

        // Eye opens (0) at t=0.30 -> blink event done! (duration = 250ms in [100, 2000])
        res = fsm.process(0, 0.30)
        assertTrue("Blink event must be done", res.eventDone)
        assertEquals(250.0, res.lastEventDurationMs, 1e-4)
        assertEquals(0.0, res.currentDurationMs, 1e-4)
        assertTrue(res.ratePerMinute > 0.0)
    }

    @Test
    fun testShortGlitchIgnored() {
        val fsm = TemporalStateMachine(mode = "eye", fps = 30) // min 100ms
        fsm.process(0, 0.0)
        fsm.process(1, 0.05)
        val res = fsm.process(0, 0.10) // 50ms < 100ms
        assertFalse("Glitch < 100ms must not be counted as blink", res.eventDone)
        assertEquals(50.0, res.lastEventDurationMs, 1e-4)
        assertEquals(0.0, res.ratePerMinute, 1e-4)
    }

    @Test
    fun testProlongedEyeClosure() {
        val fsm = TemporalStateMachine(mode = "eye", fps = 30) // max 2000ms
        var t = 0.0
        var res = fsm.process(1, t)
        for (i in 1..25) {
            t += 0.1
            res = fsm.process(1, t)
        }
        assertTrue("Prolonged flag must be set for closure > 2000ms", res.prolonged)
        assertEquals(2500.0, res.currentDurationMs, 1e-4)
    }

    @Test
    fun testGapExceedingMaxGapResetsStart() {
        val fsm = TemporalStateMachine(mode = "eye", maxGapSec = 0.25)
        fsm.process(1, 0.0)
        // Next frame is 0.5s later (> 0.25s gap)
        val res = fsm.process(1, 0.5)
        // start should have been reset to 0.5
        assertEquals(0.0, res.currentDurationMs, 1e-4)
    }

    @Test
    fun testMissingSignalResetsStart() {
        val fsm = TemporalStateMachine(mode = "eye")
        fsm.process(1, 0.0)
        fsm.process(null, 0.1) // missing face
        val res = fsm.process(0, 0.2)
        assertFalse("Missing signal between open and close must reset event", res.eventDone)
        assertEquals(0.0, res.currentDurationMs, 1e-4)
    }
}
