package com.example.driverguardian.ai.monitoring.camera

import com.example.driverguardian.ai.physical.model.LandmarkPoint
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import org.junit.Assert.*
import org.junit.Test

class CameraGeometryTest {
    @Test fun rotatedSizeHonorsRightAngles() {
        assertEquals(640 to 480, CameraFrameGeometry.rotatedSize(640, 480, 0))
        assertEquals(480 to 640, CameraFrameGeometry.rotatedSize(640, 480, 90))
        assertEquals(640 to 480, CameraFrameGeometry.rotatedSize(640, 480, 180))
        assertEquals(480 to 640, CameraFrameGeometry.rotatedSize(640, 480, 270))
    }

    @Test(expected = IllegalArgumentException::class)
    fun rotatedSizeRejectsNonRightAngle() { CameraFrameGeometry.rotatedSize(640, 480, 45) }

    @Test fun adapterPreservesAlgorithmicLeftRightWithoutMirroring() {
        val input = listOf(
            NormalizedLandmark.create(0.20f, 0.25f, -0.1f),
            NormalizedLandmark.create(0.80f, 0.75f, 0.1f),
        )
        val output = MediaPipeLandmarkAdapter.toPixelLandmarks(input, 100, 200)
        assertEquals(LandmarkPoint(20f, 50f, -0.1f), output[0])
        assertEquals(LandmarkPoint(80f, 150f, 0.1f), output[1])
        assertTrue(output[0].x < output[1].x)
    }

    @Test fun previewMirrorReturnsCopyAndDoesNotMutateAlgorithmicCoordinates() {
        val original = listOf(LandmarkPoint(20f, 10f), LandmarkPoint(80f, 20f))
        val mirrored = LandmarkOverlayMapper.frontMirrorCopy(original, 100)
        assertEquals(listOf(LandmarkPoint(80f, 10f), LandmarkPoint(20f, 20f)), mirrored)
        assertEquals(listOf(LandmarkPoint(20f, 10f), LandmarkPoint(80f, 20f)), original)
        assertNotSame(original[0], mirrored[0])
    }
}
