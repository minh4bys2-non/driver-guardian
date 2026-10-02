package com.example.driverguardian.ai.monitoring.camera

import org.junit.Assert.*
import org.junit.Test

class CameraTimestampGateTest {
    @Test fun acceptsStrictlyIncreasingNanosecondsAndConvertsToSeconds() {
        val gate = CameraTimestampGate()
        assertEquals(1.5, gate.accept(1_500_000_000L)!!, 0.0)
        assertNull(gate.accept(1_500_000_000L))
        assertNull(gate.accept(1_499_999_999L))
        assertEquals(2.0, gate.accept(2_000_000_000L)!!, 0.0)
        gate.reset()
        assertEquals(1.0, gate.accept(1_000_000_000L)!!, 0.0)
    }
}
