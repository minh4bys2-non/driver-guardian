package com.example.driverguardian.ai.physical.pipeline

/**
 * Configuration for the Physical Branch pipeline matching Python CameraMetrics defaults.
 */
data class PhysicalBranchConfig(
    val fps: Int = 30,
    val windowSec: Double = 60.0,
    val calibrationFrames: Int = 30,
    val angleLimits: DoubleArray = doubleArrayOf(20.0, 25.0, 20.0), // Pitch, Yaw, Roll limits
    val nodPitchDeg: Double = 14.0,
    val nodReleaseDeg: Double = 8.0,
    val nodDirection: Int = 1,
    val nodDurationMinMs: Double = 800.0,
    val nodDurationMaxMs: Double = 3500.0,
    val maxGapSec: Double = 0.25,
    val eyeInitDurationSec: Double = 5.0,
    val mouthInitDurationSec: Double = 5.0,
    val learningRate: Double = 0.01,
    val adaptInterval: Int = 300
) {
    init {
        require(calibrationFrames >= 1) { "Calibration frames must be >= 1" }
        require(nodDirection == 1 || nodDirection == -1) { "Nod direction must be 1 or -1" }
        require(nodReleaseDeg in 0.0..<nodPitchDeg) { "Nod release must be below the pitch threshold" }
        require(angleLimits.size == 3 && angleLimits.all { it > 0.0 && it.isFinite() }) {
            "Angle limits must contain three positive finite values"
        }
        require(fps > 0 && windowSec > 0.0 && maxGapSec > 0.0) { "Invalid temporal parameters" }
        require(nodDurationMinMs in 0.0..<nodDurationMaxMs) { "Invalid nod duration bounds" }
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
        return true
    }

    override fun hashCode(): Int {
        var result = fps
        result = 31 * result + windowSec.hashCode()
        result = 31 * result + calibrationFrames
        result = 31 * result + angleLimits.contentHashCode()
        result = 31 * result + nodPitchDeg.hashCode()
        result = 31 * result + nodReleaseDeg.hashCode()
        result = 31 * result + nodDirection
        result = 31 * result + nodDurationMinMs.hashCode()
        result = 31 * result + nodDurationMaxMs.hashCode()
        result = 31 * result + maxGapSec.hashCode()
        result = 31 * result + eyeInitDurationSec.hashCode()
        result = 31 * result + mouthInitDurationSec.hashCode()
        result = 31 * result + learningRate.hashCode()
        result = 31 * result + adaptInterval
        return result
    }
}
