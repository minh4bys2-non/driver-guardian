package com.example.driverguardian.ai.physical

import com.example.driverguardian.ai.physical.temporal.WindowRatio
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class WindowRatioTest {

    @Test
    fun testElapsedRatioCalculation() {
        val wr = WindowRatio(windowSec = 60.0, maxGapSec = 0.25)

        // First point establishes previous, no interval yet
        val (p0, v0) = wr.process(true, 0.0)
        assertNull(p0)
        assertEquals(0.0, v0, 1e-6)

        // Step by 0.1s with active=true up to t=1.0 (10 intervals of 0.1s)
        var t = 0.0
        for (i in 1..10) {
            t += 0.1
            wr.process(true, t)
        }
        t += 0.1
        val (p1, v1) = wr.process(true, t)
        assertNotNull(p1)
        assertEquals(100.0, p1!!, 1e-4)
        assertEquals(1.1, v1, 1e-4)

        // Now step by 0.1s with active=false up to t=2.1 (10 intervals of 0.1s)
        for (i in 1..10) {
            t += 0.1
            wr.process(false, t)
        }
        t += 0.1
        val (p2, v2) = wr.process(false, t)
        assertNotNull(p2)
        // 12 active intervals of 0.1s out of 22 total intervals -> (1.2 / 2.2) * 100 = 54.545%
        assertEquals((1.2 / 2.2) * 100.0, p2!!, 1e-3)
        assertEquals(2.2, v2, 1e-3)
    }

    @Test
    fun testGapExceedingMaxGapDiscardsInterval() {
        val wr = WindowRatio(windowSec = 60.0, maxGapSec = 0.25)
        wr.process(true, 0.0)
        // gap of 1.0s > 0.25s
        val (p, v) = wr.process(true, 1.0)
        assertNull("Gap > 0.25s must discard interval", p)
        assertEquals(0.0, v, 1e-6)
    }

    @Test
    fun testNullActiveResetsPrevious() {
        val wr = WindowRatio(windowSec = 60.0, maxGapSec = 0.25)
        wr.process(true, 0.0)
        wr.process(null, 0.1) // missing
        val (p, v) = wr.process(true, 0.2)
        assertNull(p)
        assertEquals(0.0, v, 1e-6)
    }

    @Test
    fun testResetClearsHistory() {
        val wr = WindowRatio(windowSec = 60.0)
        wr.process(true, 0.0)
        wr.process(true, 0.1)
        wr.reset()
        assertNull(wr.time)
        val (p, v) = wr.process(true, 10.0)
        assertNull(p)
        assertEquals(0.0, v, 1e-6)
    }
}
