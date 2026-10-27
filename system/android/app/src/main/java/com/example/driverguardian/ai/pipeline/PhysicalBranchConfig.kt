package com.example.driverguardian.ai.pipeline

import com.example.driverguardian.ai.physical.PhysicalConfig

/**
 * Configuration for the Physical Branch pipeline (`ai.pipeline`) matching Python CameraMetrics defaults.
 * All defaults reference [PhysicalConfig] so nothing is hardcoded in the processor.
 */
data class PhysicalBranchConfig(
    val fps: Int = PhysicalConfig.DEFAULT_FPS,
    val windowSec: Double = PhysicalConfig.DEFAULT_WINDOW_SEC,
    val calibrationFrames: Int = PhysicalConfig.DEFAULT_CALIBRATION_FRAMES,
    val angleLimits: DoubleArray = PhysicalConfig.DEFAULT_ANGLE_LIMITS_DEG.clone(), // Pitch, Yaw, Roll limits
    val nodPitchDeg: Double = PhysicalConfig.DEFAULT_NOD_PITCH_DEG,
    val nodReleaseDeg: Double = PhysicalConfig.DEFAULT_NOD_RELEASE_DEG,
    val nodDirection: Int? = PhysicalConfig.DEFAULT_NOD_DIRECTION,
    val nodDurationMinMs: Double = PhysicalConfig.DEFAULT_NOD_DURATION_MIN_MS,
    val nodDurationMaxMs: Double = PhysicalConfig.DEFAULT_NOD_DURATION_MAX_MS,
    val maxGapSec: Double = PhysicalConfig.DEFAULT_MAX_GAP_SEC,
    val eyeInitDurationSec: Double = PhysicalConfig.DEFAULT_INIT_DURATION_SEC,
    val mouthInitDurationSec: Double = PhysicalConfig.DEFAULT_INIT_DURATION_SEC,
    val learningRate: Double = PhysicalConfig.DEFAULT_HMM_LEARNING_RATE,
    val adaptInterval: Int = PhysicalConfig.DEFAULT_HMM_ADAPT_INTERVAL,
    val headMotionMinAmplitudeDeg: Double = PhysicalConfig.DEFAULT_HEAD_MOTION_MIN_AMPLITUDE_DEG,
    val physicalConfig: PhysicalConfig = PhysicalConfig.DEFAULT
) {
    init {
        require(calibrationFrames >= 1) { "Calibration frames must be >= 1" }
        require(nodDirection == null || nodDirection == 1 || nodDirection == -1) { "Nod direction must be null, 1 or -1" }
        require(nodReleaseDeg in 0.0..<nodPitchDeg) { "Nod release must be below the pitch threshold" }
        require(angleLimits.size == 3 && angleLimits.all { it > 0.0 && it.isFinite() }) {
            "Angle limits must contain three positive finite values"
        }
        require(fps > 0 && windowSec > 0.0 && maxGapSec > 0.0) { "Invalid temporal parameters" }
        require(nodDurationMinMs in 0.0..<nodDurationMaxMs) { "Invalid nod duration bounds" }
        require(headMotionMinAmplitudeDeg >= 0.0 && headMotionMinAmplitudeDeg.isFinite()) {
            "Invalid headMotionMinAmplitudeDeg"
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is PhysicalBranchConfig) return false
        if (fps != other.fps) return false
        if (windowSec != other.windowSec) return false
        if (calibrationFrames != other.calibrationFrames) return false
        if (!angleLimits.contentEquals(other.angleLimits)) return false
        if (nodPitchDeg != other.nodPitchDeg) return false
        if (nodReleaseDeg != other.nodReleaseDeg) return false
        if (nodDirection != other.nodDirection) return false
        if (nodDurationMinMs != other.nodDurationMinMs) return false
        if (nodDurationMaxMs != other.nodDurationMaxMs) return false
        if (maxGapSec != other.maxGapSec) return false
        if (eyeInitDurationSec != other.eyeInitDurationSec) return false
        if (mouthInitDurationSec != other.mouthInitDurationSec) return false
        if (learningRate != other.learningRate) return false
        if (adaptInterval != other.adaptInterval) return false
        if (headMotionMinAmplitudeDeg != other.headMotionMinAmplitudeDeg) return false
        return true
    }

    override fun hashCode(): Int {
        var result = fps
        result = 31 * result + windowSec.hashCode()
        result = 31 * result + calibrationFrames
        result = 31 * result + angleLimits.contentHashCode()
        result = 31 * result + nodPitchDeg.hashCode()
        result = 31 * result + nodReleaseDeg.hashCode()
        result = 31 * result + (nodDirection ?: 0)
        result = 31 * result + nodDurationMinMs.hashCode()
        result = 31 * result + nodDurationMaxMs.hashCode()
        result = 31 * result + maxGapSec.hashCode()
        result = 31 * result + eyeInitDurationSec.hashCode()
        result = 31 * result + mouthInitDurationSec.hashCode()
        result = 31 * result + learningRate.hashCode()
        result = 31 * result + adaptInterval
        result = 31 * result + headMotionMinAmplitudeDeg.hashCode()
        return result
    }
}
