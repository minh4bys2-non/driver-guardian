package com.example.driverguardian.ai.monitoring.device

import org.junit.Assert.*
import org.junit.Test

class OrientationMathTest {
    @Test fun normalizesAcrossWrapBoundary() {
        assertEquals(-179.0, OrientationMath.normalizeDegrees(181.0), 1e-9)
        assertEquals(179.0, OrientationMath.normalizeDegrees(-181.0), 1e-9)
        assertEquals(2.0, OrientationMath.shortestDeltaDegrees(179.0, -179.0), 1e-9)
    }

    @Test fun identityQuaternionProducesZeroAnglesForEveryDisplayRotation() {
        for (rotation in listOf(0, 90, 180, 270)) {
            val o = OrientationMath.fromQuaternion(0.0, 0.0, 0.0, 1.0, rotation)
            assertTrue(o.pitchDeg.isFinite())
            assertTrue(o.rollDeg.isFinite())
            assertTrue(o.yawDeg.isFinite())
        }
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsUnsupportedDisplayRotation() {
        OrientationMath.fromQuaternion(0.0, 0.0, 0.0, 1.0, 45)
    }
}
