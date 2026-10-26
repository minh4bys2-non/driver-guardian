package com.example.driverguardian.ai.monitoring.camera

import com.google.mediapipe.tasks.vision.core.RunningMode
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FaceLandmarkerConfigTest {
    @Test fun configurationIsAssetsOnlyVideoModeOneFaceAndOptimizedThresholds() {
        assertEquals("face_landmarker.task", FaceLandmarkerConfiguration.MODEL_ASSET_PATH)
        assertEquals(RunningMode.VIDEO, FaceLandmarkerConfiguration.RUNNING_MODE)
        assertEquals(1, FaceLandmarkerConfiguration.NUM_FACES)
        assertEquals(0.6f, FaceLandmarkerConfiguration.MIN_FACE_DETECTION_CONFIDENCE, 1e-4f)
        assertEquals(0.6f, FaceLandmarkerConfiguration.MIN_FACE_PRESENCE_CONFIDENCE, 1e-4f)
        assertEquals(0.6f, FaceLandmarkerConfiguration.MIN_TRACKING_CONFIDENCE, 1e-4f)
        assertFalse(FaceLandmarkerConfiguration.OUTPUT_FACE_BLENDSHAPES)
        assertFalse(FaceLandmarkerConfiguration.OUTPUT_FACIAL_TRANSFORMATION_MATRIXES)
    }

    @Test fun taskMillisecondsAreStrictlyIncreasingWithoutChangingCameraNanoseconds() {
        val clock = MonotonicTaskTimestamp()
        assertEquals(1000L, clock.fromCameraNanoseconds(1_000_000_000L))
        assertEquals(1001L, clock.fromCameraNanoseconds(1_000_100_000L))
        assertEquals(1002L, clock.fromCameraNanoseconds(999_000_000L))
        clock.reset()
        assertEquals(500L, clock.fromCameraNanoseconds(500_000_000L))
    }

    @Test fun nativeResourceCloseIsDeferredUntilInFlightCallbackFinishes() {
        var closes = 0
        val lifecycle = DeferredResourceClose { closes++ }
        assertTrue(lifecycle.tryBeginWork())
        lifecycle.requestClose()
        assertEquals(0, closes)
        assertFalse(lifecycle.tryBeginWork())
        lifecycle.finishWork()
        assertEquals(1, closes)
        lifecycle.requestClose()
        assertEquals(1, closes)
    }

    @Test fun nativeResourceClosesImmediatelyWhenIdle() {
        var closes = 0
        val lifecycle = DeferredResourceClose { closes++ }
        lifecycle.requestClose()
        assertEquals(1, closes)
        assertFalse(lifecycle.tryBeginWork())
    }
}
