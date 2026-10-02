package com.example.driverguardian.ai.monitoring.device

import android.view.Surface
import org.junit.Assert.assertEquals
import org.junit.Test

class SafeDisplayRotationProviderTest {
    @Test fun throwingNonVisualContextLookupFallsBackWithoutCrashingSensorCallback() {
        val provider = SafeDisplayRotationProvider { throw UnsupportedOperationException("Context has no display") }
        assertEquals(Surface.ROTATION_0, provider.rotation())
    }

    @Test fun validDisplayRotationIsPreserved() {
        val provider = SafeDisplayRotationProvider { Surface.ROTATION_270 }
        assertEquals(Surface.ROTATION_270, provider.rotation())
    }
}
