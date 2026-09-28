package com.example.driverguardian.ai.physical

import com.example.driverguardian.ai.physical.geometry.HeadPoseEstimator
import com.example.driverguardian.ai.physical.model.LandmarkPoint
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class HeadPoseEstimatorTest {

    @Test
    fun testHeadPoseEstimationOnProjectedFace() {
        val estimator = HeadPoseEstimator(640, 480)

        // Ground-truth 2D projection of MODEL with known rvec=[0.1, -0.15, 0.05], tvec=[10, 20, 700]
        // Corresponding to OpenCV RQDecomp angles: Pitch=5.5548, Yaw=-8.7208, Roll=2.4487
        val pts2d = listOf(
            LandmarkPoint(329.14285714f, 258.28571429f), // 1: nose
            LandmarkPoint(329.58988775f, 520.19830571f), // 152: chin
            LandmarkPoint(531.02598379f, 96.96919001f),  // 263: left eye
            LandmarkPoint(135.21557997f, 137.93608149f), // 33: right eye
            LandmarkPoint(461.12946959f, 372.48398418f), // 291: left mouth
            LandmarkPoint(197.80164808f, 400.09110419f)  // 61: right mouth
        )

        val pose = estimator.estimate(pts2d)
        assertNotNull(pose)

        // Parity with OpenCV on these projected landmarks within 0.05 degrees
        assertEquals(6.26, pose.pitchDeg, 0.05)
        assertEquals(2.83, pose.yawDeg, 0.05)
        assertEquals(-2.32, pose.rollDeg, 0.05)
    }

    @Test(expected = IllegalArgumentException::class)
    fun testDegenerateCollinearPointsThrows() {
        val estimator = HeadPoseEstimator(640, 480)
        // All points on a straight horizontal line
        val collinear = List(6) { i -> LandmarkPoint(100f + i * 20f, 200f) }
        estimator.estimate(collinear)
    }

    @Test(expected = IllegalArgumentException::class)
    fun testInsufficientLandmarksThrows() {
        val estimator = HeadPoseEstimator(640, 480)
        estimator.estimate(List(5) { LandmarkPoint(100f, 100f) })
    }

    @Test(expected = IllegalArgumentException::class)
    fun testNonFiniteLandmarkThrows() {
        val estimator = HeadPoseEstimator(640, 480)
        val pts = List(6) { i ->
            if (i == 0) LandmarkPoint(Float.NaN, 200f) else LandmarkPoint(100f + i * 20f, 150f + i * 10f)
        }
        estimator.estimate(pts)
    }
}
