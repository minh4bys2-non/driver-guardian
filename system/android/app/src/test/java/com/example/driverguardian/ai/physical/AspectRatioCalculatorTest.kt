package com.example.driverguardian.ai.physical

import com.example.driverguardian.ai.physical.geometry.AspectRatioCalculator
import com.example.driverguardian.ai.physical.model.LandmarkPoint
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class AspectRatioCalculatorTest {

    private fun createBoxPoints(x: Float, y: Float, w: Float, h: Float): List<LandmarkPoint> {
        val points = MutableList(500) { LandmarkPoint(0f, 0f) }
        // Left eye: 362 (p0), 385 (p1), 387 (p2), 263 (p3), 373 (p4), 380 (p5)
        points[362] = LandmarkPoint(x, y + h / 2f)           // p0 (left corner)
        points[385] = LandmarkPoint(x + w / 3f, y)          // p1 (top left)
        points[387] = LandmarkPoint(x + 2f * w / 3f, y)     // p2 (top right)
        points[263] = LandmarkPoint(x + w, y + h / 2f)      // p3 (right corner)
        points[373] = LandmarkPoint(x + 2f * w / 3f, y + h) // p4 (bottom right)
        points[380] = LandmarkPoint(x + w / 3f, y + h)      // p5 (bottom left)

        // Right eye: 33 (p0), 160 (p1), 158 (p2), 133 (p3), 153 (p4), 144 (p5)
        points[33] = LandmarkPoint(x + 200f, y + h / 2f)
        points[160] = LandmarkPoint(x + 200f + w / 3f, y)
        points[158] = LandmarkPoint(x + 200f + 2f * w / 3f, y)
        points[133] = LandmarkPoint(x + 200f + w, y + h / 2f)
        points[153] = LandmarkPoint(x + 200f + 2f * w / 3f, y + h)
        points[144] = LandmarkPoint(x + 200f + w / 3f, y + h)

        // Mouth: 61, 37, 267, 291, 314, 84
        points[61] = LandmarkPoint(x + 100f, y + 100f + h / 2f)
        points[37] = LandmarkPoint(x + 100f + w / 3f, y + 100f)
        points[267] = LandmarkPoint(x + 100f + 2f * w / 3f, y + 100f)
        points[291] = LandmarkPoint(x + 100f + w, y + 100f + h / 2f)
        points[314] = LandmarkPoint(x + 100f + 2f * w / 3f, y + 100f + h)
        points[84] = LandmarkPoint(x + 100f + w / 3f, y + 100f + h)

        return points
    }

    @Test
    fun testNormalEarAndMarCalculation() {
        val points = createBoxPoints(100f, 100f, 60f, 20f)
        val ear = AspectRatioCalculator.calculateEar(points)
        val mar = AspectRatioCalculator.calculateMar(points)

        assertNotNull(ear)
        assertNotNull(mar)
        // With width = 60, height = 20, formula (h + h) / (2 * w) = 20 / 60 = 0.3333...
        assertEquals(20.0 / 60.0, ear!!, 1e-4)
        assertEquals(20.0 / 60.0, mar!!, 1e-4)
    }

    @Test
    fun testDegenerateZeroWidthReturnsNull() {
        // Zero width
        val points = MutableList(500) { LandmarkPoint(100f, 100f) }
        val ear = AspectRatioCalculator.calculateEar(points)
        val mar = AspectRatioCalculator.calculateMar(points)

        assertNull("Zero width must yield null EAR", ear)
        assertNull("Zero width must yield null MAR", mar)
    }

    @Test
    fun testInsufficientLandmarksReturnsNull() {
        val shortList = List(100) { LandmarkPoint(10f, 10f) }
        val ear = AspectRatioCalculator.calculateEar(shortList)
        assertNull("Short landmark list must yield null EAR", ear)
    }

    @Test
    fun testNonFiniteLandmarksReturnsNull() {
        val points = createBoxPoints(100f, 100f, 60f, 20f).toMutableList()
        points[362] = LandmarkPoint(Float.NaN, 100f)
        val ear = AspectRatioCalculator.calculateEar(points)
        assertNull("NaN coordinate must yield null EAR", ear)
    }
}
