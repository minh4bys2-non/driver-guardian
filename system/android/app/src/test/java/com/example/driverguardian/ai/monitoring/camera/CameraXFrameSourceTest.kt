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
}
