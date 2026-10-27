package com.example.driverguardian.ai.neural

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NeuralBufferTest {

    @Test
    fun `lazy letterbox executes only on 5 FPS grid and caps buffer at 50 frames`() {
        val buffer = NeuralBuffer()
        val width = 64
        val height = 48
        val dummyRgba = ByteArray(width * height * 4) { 120.toByte() }
        val fps = 30.0

        // Feed 12 seconds at 30 FPS (361 frames: t = 0.0 .. 12.0s)
        for (i in 0..360) {
            val t = i / fps
            buffer.updateFromRgba(dummyRgba, width, height, t)
        }

        // Buffer is capped at 50 frames (10s at 5 FPS)
        assertEquals(50, buffer.size)
        // Over 12.0s at 5.0 FPS grid (0.0, 0.2, ..., 12.0), exactly 61 grid ticks occur (not 361!)
        assertEquals(61L, buffer.totalLetterboxConversions)
    }

    @Test
    fun `getNewFramesSince extracts only 5 delta frames after 1 second step`() {
        val buffer = NeuralBuffer()
        val width = 32
        val height = 24
        val dummyRgba = ByteArray(width * height * 4) { 80.toByte() }
        val fps = 30.0

        // Feed first 10.0 seconds (0..300 frames) -> fills 50 grid frames
        for (i in 0..300) {
            buffer.updateFromRgba(dummyRgba, width, height, i / fps)
        }
        val snap1 = buffer.snapshot()
        assertEquals(50, snap1.size)

        // First evaluation has no cache -> processes all 50 frames
        val delta1 = snap1.getNewFramesSince(cachedGeneration = -1L, cachedSeqId = -1L, cachedFrameCount = 0)
        assertEquals(50, delta1.newChwFrames.size)
        assertEquals(0, delta1.keepOldCount)

        // Feed 1.0 more second (frames 301..330, t = 10.033..11.0s) -> adds 5 new grid frames
        for (i in 301..330) {
            buffer.updateFromRgba(dummyRgba, width, height, i / fps)
        }
        val snap2 = buffer.snapshot()
        assertEquals(50, snap2.size)

        val delta2 = snap2.getNewFramesSince(
            cachedGeneration = delta1.generation,
            cachedSeqId = delta1.latestSeqId,
            cachedFrameCount = 50,
        )
        assertEquals(5, delta2.newChwFrames.size)
        assertEquals(45, delta2.keepOldCount)
        assertTrue(delta2.latestSeqId > delta1.latestSeqId)
    }
}
