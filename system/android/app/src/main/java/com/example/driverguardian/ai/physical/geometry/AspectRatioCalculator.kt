package com.example.driverguardian.ai.physical.geometry

import com.example.driverguardian.ai.physical.model.LandmarkPoint
import kotlin.math.hypot

/**
 * Geometric aspect ratio calculator for Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR)
 * using the exact 6-point formulation from Soukupová & Čech (2016) and Python CameraMetrics.
 */
object AspectRatioCalculator {

    val LEFT_EYE_INDICES = intArrayOf(362, 385, 387, 263, 373, 380)
    val RIGHT_EYE_INDICES = intArrayOf(33, 160, 158, 133, 153, 144)
    val MOUTH_INDICES = intArrayOf(61, 37, 267, 291, 314, 84)

    /**
     * Calculates the aspect ratio of 6 ordered landmark points:
     * ratio = ( ||p1 - p5|| + ||p2 - p4|| ) / ( 2 * ||p0 - p3|| )
     *
     * Returns null if landmarks are degenerate (width < 1e-6), missing, or non-finite.
     */
    fun calculateAspectRatio(points: List<LandmarkPoint>, indices: IntArray): Double? {
        if (indices.size != 6) return null
        if (points.size <= indices.maxOrNull()!!) return null

        val p = Array(6) { i ->
            val pt = points[indices[i]]
            val x = pt.x.toDouble()
            val y = pt.y.toDouble()
            if (!x.isFinite() || !y.isFinite()) return null
            x to y
        }

        val width = hypot(p[0].first - p[3].first, p[0].second - p[3].second)
        if (width < 1e-6) return null

        val h1 = hypot(p[1].first - p[5].first, p[1].second - p[5].second)
        val h2 = hypot(p[2].first - p[4].first, p[2].second - p[4].second)

        return (h1 + h2) / (2.0 * width)
    }

    /**
     * Computes the combined Eye Aspect Ratio (EAR) as the average of left and right eye AR.
     */
    fun calculateEar(points: List<LandmarkPoint>): Double? {
        val left = calculateAspectRatio(points, LEFT_EYE_INDICES) ?: return null
        val right = calculateAspectRatio(points, RIGHT_EYE_INDICES) ?: return null
        return (left + right) / 2.0
    }

    /**
     * Computes the Mouth Aspect Ratio (MAR).
     */
    fun calculateMar(points: List<LandmarkPoint>): Double? {
        return calculateAspectRatio(points, MOUTH_INDICES)
    }
}
