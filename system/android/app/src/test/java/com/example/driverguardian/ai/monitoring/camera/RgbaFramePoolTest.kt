package com.example.driverguardian.ai.monitoring.camera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertSame
import org.junit.Test

class RgbaFramePoolTest {

    @Test
    fun testPoolInitializationAndCapacity() {
        val pool = RgbaFramePool(width = 640, height = 480, poolSize = 3)
        assertEquals(3, pool.capacity)
        assertEquals(640 * 480 * 4, pool.frameBytesSize)
        for (i in 0 until 3) {
            val buf = pool.getBufferAt(i)
            assertNotNull(buf)
            assertEquals(pool.frameBytesSize, buf.size)
        }
    }

    @Test
    fun testAcquireNextBufferCyclesTripleBuffers() {
        val pool = RgbaFramePool(width = 10, height = 10, poolSize = 3)
        val buf0 = pool.acquireNextBuffer()
        val buf1 = pool.acquireNextBuffer()
        val buf2 = pool.acquireNextBuffer()
        val buf3 = pool.acquireNextBuffer()

        assertSame("Slot 3 must wrap around to Slot 0", buf0, buf3)
        assertSame(pool.getBufferAt(0), buf0)
        assertSame(pool.getBufferAt(1), buf1)
        assertSame(pool.getBufferAt(2), buf2)
    }
}
