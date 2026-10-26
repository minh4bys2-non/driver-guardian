package com.example.driverguardian.ai.monitoring.camera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CameraXFrameSourceTest {
    @Test fun closesProxyExactlyOnceWhenConversionFails() {
        var closes = 0
        val guard = CloseOnce { closes++ }
        runCatching {
            try { error("conversion failed") } finally { guard.close() }
        }
        guard.close()
        assertEquals(1, closes)
    }

    @Test fun closeGuardClosesExactlyOnceOnSuccess() {
        var closes = 0
        CloseOnce { closes++ }.use { }
        assertEquals(1, closes)
    }

    @Test fun invalidatedStartTokenCannotBindAfterStopOrRestart() {
        val gate = StartGenerationGate()
        val first = gate.newStart()
        assertTrue(gate.isCurrent(first))
        gate.invalidate()
        assertFalse(gate.isCurrent(first))
        val second = gate.newStart()
        assertTrue(gate.isCurrent(second))
        assertFalse(gate.isCurrent(first))
    }

    @Test fun testFramePoolIntegrationAndControlFlags() {
        val pool = RgbaFramePool(width = 640, height = 480, poolSize = 3)
        val balancer = BrightnessBalancer(targetLuma = 70.0)

        assertFalse(balancer.enabled.get())
        balancer.enabled.set(true)
        assertTrue(balancer.enabled.get())
        balancer.enabled.set(false)
        assertFalse(balancer.enabled.get())

        assertEquals(3, pool.capacity)
    }
}
