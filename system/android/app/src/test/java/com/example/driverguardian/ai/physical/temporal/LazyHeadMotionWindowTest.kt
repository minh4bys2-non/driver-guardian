package com.example.driverguardian.ai.physical.temporal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin

class LazyHeadMotionWindowTest {

    @Test
    fun `recordFrame updates trapezoidal integrals and evaluateExactFft recovers dominant sine frequency`() {
        val window = LazyHeadMotionWindow(window = 60.0, maxGap = 0.25, minAmplitude = 2.0)
        val fps = 30.0
        val targetFreqHz = 1.0
        val amplitudeDeg = 5.0
        val totalFrames = 1801 // 60.0 seconds at 30 FPS

        var lastMetrics: HeadMotionMetrics? = null
        for (i in 0 until totalFrames) {
            val t = i / fps
            val pitch = amplitudeDeg * sin(2.0 * PI * targetFreqHz * t)
            lastMetrics = window.recordFrame(pitch, t)
        }

        assertNotNull(lastMetrics)
        assertEquals(60.0, lastMetrics!!.headMotionObservedSec, 1e-3)
        assertNotNull(lastMetrics.pitchMeanDeg)
        assertEquals(0.0, lastMetrics.pitchMeanDeg!!, 0.1)
        assertNotNull(lastMetrics.pitchMeanAmplitudeDeg)
        assertTrue(lastMetrics.pitchMeanAmplitudeDeg!! > 2.0)

        val snapshot = window.snapshotForFft()
        assertNotNull(snapshot)

        val fftMetrics = LazyHeadMotionWindow.evaluateExactFft(snapshot!!)
        assertNotNull(fftMetrics.frequency)
        assertNotNull(fftMetrics.resolution)
        assertEquals(targetFreqHz, fftMetrics.frequency!!, 0.05)
        assertEquals(1.0 / 60.0, fftMetrics.resolution!!, 1e-3)
    }

    @Test
    fun `evaluateExactFft returns 0 Hz when mean amplitude is below minAmplitude`() {
        val window = LazyHeadMotionWindow(window = 10.0, maxGap = 0.25, minAmplitude = 2.0)
        val fps = 30.0
        for (i in 0..300) {
            val t = i / fps
            val pitch = 0.5 * sin(2.0 * PI * 1.5 * t) // peak-to-peak 1.0 deg < minAmplitude 2.0 deg
            window.recordFrame(pitch, t)
        }
        val snapshot = window.snapshotForFft()
        assertNotNull(snapshot)
        val fftMetrics = LazyHeadMotionWindow.evaluateExactFft(snapshot!!)
        assertEquals(0.0, fftMetrics.frequency!!, 1e-9)
        assertNotNull(fftMetrics.resolution)
    }

    @Test
    fun `gap larger than maxGap resets contiguous run`() {
        val window = LazyHeadMotionWindow(window = 60.0, maxGap = 0.25, minAmplitude = 2.0)
        window.recordFrame(4.0, 0.0)
        window.recordFrame(4.0, 0.1)
        val afterGap = window.recordFrame(4.0, 1.0) // gap 0.9s > 0.25s
        assertEquals(0.0, afterGap.headMotionObservedSec, 1e-9)
        assertNull(afterGap.pitchMeanDeg)
    }
}
