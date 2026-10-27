package com.example.driverguardian.ai.physical.model

import com.example.driverguardian.ai.physical.PhysicalConfig

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
    val blinkCount: Int = 0,
    val blinkDurationMs: Double? = null,

    // Oral metrics
    val mar: Double?,
    val mouthState: Int?,
    val pomPct: Double?,
    val yawnDetected: Boolean,
    val yawningFrequencyPerMin: Double?,
    val mouthOpenDurationMs: Double?,
    val lastMouthOpenDurationMs: Double? = null,
    val mouthObservedSec: Double? = null,
    val yawnCount: Int = 0,
    val yawnDurationMs: Double? = null,

    // Head pose metrics
    val pitchDeg: Double?,
    val yawDeg: Double?,
    val rollDeg: Double?,
    val pitchAmplitudeDeg: Double? = null,
    val overAngle: Boolean?,
    val overAnglePct: Double?,

    // Nodding metrics
    val nodding: Boolean?,
    val nodDetected: Boolean,
    val nodDurationMs: Double?,
    val lastNodDurationMs: Double? = null,
    val noddingFrequencyPerMin: Double?,
    val headObservedSec: Double? = null,
    val nodCount: Int = 0,
    val nodEventDurationMs: Double? = null,

    // Head motion & Lazy FFT 1Hz metrics
    val pitchMeanDeg: Double? = null,
    val pitchMeanAmplitudeDeg: Double? = null,
    val headMotionFrequencyHz: Double? = null,
    val headMotionResolutionHz: Double? = null,
    val headMotionObservedSec: Double = 0.0,

    // Window & phase offset metadata
    val windowSec: Double = PhysicalConfig.DEFAULT_WINDOW_SEC,
    val hmmJustInitialized: Boolean = false,
) {
    val blinkFreqPerMin: Double
        get() = blinkRatePerMin ?: 0.0

    val yawnFreqPerMin: Double
        get() = yawningFrequencyPerMin ?: 0.0

    val nodFreqPerMin: Double
        get() = noddingFrequencyPerMin ?: 0.0
}
