package com.example.driverguardian.ai.physical.model

/**
 * Platform-independent normalized or pixel landmark point.
 */
data class LandmarkPoint(
    val x: Float,
    val y: Float,
    val z: Float? = null
)
