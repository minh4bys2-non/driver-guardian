package com.example.driverguardian.ai.physical.model

/**
 * Standardized immutable DTO exposing physiological, ocular, oral, and head kinematics metrics.
 * All metrics preserve exact nullability without synthetic fallback defaults (e.g. fake 0 / 0.0).
 */
data class PhysicalMetrics(
    val timestampSec: Double,
    val faceDetected: Boolean,

    val eyeReady: Boolean,
    val mouthReady: Boolean,
    val headReady: Boolean,
    val headCalibrated: Boolean,
    val poseValid: Boolean = false,
    val poseError: String? = null,

    // Ocular metrics
    val ear: Double?,
    val eyeState: Int?,
    val perclosPct: Double?,
    val p80EarThreshold: Double?,
    val blinkRatePerMin: Double?,
    val blinkDetected: Boolean,
    val eyeClosureDurationMs: Double?,
    val lastEyeClosureDurationMs: Double? = null,
    val eyeObservedSec: Double? = null,

    // Oral metrics
    val mar: Double?,
    val mouthState: Int?,
    val pomPct: Double?,
    val yawnDetected: Boolean,
    val yawningFrequencyPerMin: Double?,
    val mouthOpenDurationMs: Double?,
    val lastMouthOpenDurationMs: Double? = null,
    val mouthObservedSec: Double? = null,

    // Head pose metrics
    val pitchDeg: Double?,
    val yawDeg: Double?,
    val rollDeg: Double?,
    val overAngle: Boolean?,
    val overAnglePct: Double?,

    // Nodding metrics
    val nodding: Boolean?,
    val nodDetected: Boolean,
    val nodDurationMs: Double?,
    val lastNodDurationMs: Double? = null,
    val noddingFrequencyPerMin: Double?,
    val headObservedSec: Double? = null
)
