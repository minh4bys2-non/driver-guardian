package com.driverguardian.ai.physical

import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * 2D point representation for image plane coordinates.
 */
data class Point2D(val x: Double, val y: Double) {
    fun distanceTo(other: Point2D): Double = hypot(x - other.x, y - other.y)

    companion object {
        fun of(x: Number, y: Number) = Point2D(x.toDouble(), y.toDouble())
    }
}

/**
 * 3D point / vector representation for 3D landmark points and geometric calculations.
 */
data class Point3D(val x: Double, val y: Double, val z: Double) {
    operator fun plus(other: Point3D): Point3D = Point3D(x + other.x, y + other.y, z + other.z)
    operator fun minus(other: Point3D): Point3D = Point3D(x - other.x, y - other.y, z - other.z)
    operator fun times(scalar: Double): Point3D = Point3D(x * scalar, y * scalar, z * scalar)
    operator fun div(scalar: Double): Point3D = Point3D(x / scalar, y / scalar, z / scalar)

    fun norm(): Double = sqrt(x * x + y * y + z * z)

    fun normalized(): Point3D {
        val n = norm()
        if (n < 1e-12) return Point3D(0.0, 0.0, 0.0)
        return Point3D(x / n, y / n, z / n)
    }

    fun cross(other: Point3D): Point3D = Point3D(
        x = y * other.z - z * other.y,
        y = z * other.x - x * other.z,
        z = x * other.y - y * other.x
    )

    fun dot(other: Point3D): Double = x * other.x + y * other.y + z * other.z

    fun toDoubleArray(): DoubleArray = doubleArrayOf(x, y, z)

    companion object {
        fun of(x: Number, y: Number, z: Number) = Point3D(x.toDouble(), y.toDouble(), z.toDouble())
    }
}

/**
 * Head pose Euler angles in degrees.
 * - pitch: Up (+) / Down (-) or vice versa based on calibration.
 * - yaw: Left / Right rotation.
 * - roll: Tilt left / right.
 */
data class HeadPoseAngles(
    val pitch: Double,
    val yaw: Double,
    val roll: Double
) {
    fun toDoubleArray(): DoubleArray = doubleArrayOf(pitch, yaw, roll)
}

/**
 * Operating mode for temporal FSM and detectors.
 */
enum class FsmMode(val id: String) {
    EYE("eye"),
    MOUTH("mouth"),
    PITCH("pitch");

    companion object {
        fun fromString(str: String): FsmMode = when (str.lowercase()) {
            "eye" -> EYE
            "mouth" -> MOUTH
            "pitch" -> PITCH
            else -> throw IllegalArgumentException("Unknown FsmMode: $str")
        }
    }
}

/**
 * Target state configuration for Hidden Markov Models.
 * - LOW: Event corresponds to low emission mean (e.g. eye closure has lower EAR).
 * - HIGH: Event corresponds to high emission mean (e.g. mouth opening has higher MAR).
 */
enum class PositiveState(val id: String) {
    LOW("low"),
    HIGH("high");

    companion object {
        fun fromString(str: String): PositiveState = when (str.lowercase()) {
            "low" -> LOW
            "high" -> HIGH
            else -> throw IllegalArgumentException("Unknown PositiveState: $str")
        }
    }
}

/**
 * Output of StateMachine.process().
 */
data class StateMachineResult(
    val state: Int?,
    val signalMissing: Boolean,
    val eventDone: Boolean,
    val currentDurationMs: Double,
    val lastEventDurationMs: Double,
    val ratePerMinute: Double,
    val prolonged: Boolean
)

/**
 * Output of WindowRatio.process().
 */
data class WindowRatioResult(
    val percentage: Double?,
    val observedSeconds: Double
)

/**
 * Combined output of AdaptiveHMM_FSM.process().
 */
data class AdaptiveHmmFsmResult(
    val mode: FsmMode,
    val inputValue: Double?,
    val initialized: Boolean,
    val signalMissing: Boolean,
    val state: Int?,
    val eventDone: Boolean,
    val currentDurationMs: Double,
    val lastEventDurationMs: Double,
    val ratePerMinute: Double,
    val prolonged: Boolean,
    val modelUpdated: Boolean,
    val posterior: DoubleArray?,
    val p80Threshold: Double?,
    val perclos: Double?,
    val pom: Double?,
    val observedSeconds: Double
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is AdaptiveHmmFsmResult) return false
        return mode == other.mode &&
                inputValue == other.inputValue &&
                initialized == other.initialized &&
                signalMissing == other.signalMissing &&
                state == other.state &&
                eventDone == other.eventDone &&
                currentDurationMs == other.currentDurationMs &&
                lastEventDurationMs == other.lastEventDurationMs &&
                ratePerMinute == other.ratePerMinute &&
                prolonged == other.prolonged &&
                modelUpdated == other.modelUpdated &&
                p80Threshold == other.p80Threshold &&
                perclos == other.perclos &&
                pom == other.pom &&
                observedSeconds == other.observedSeconds &&
                (posterior contentEquals other.posterior)
    }

    override fun hashCode(): Int {
        var result = mode.hashCode()
        result = 31 * result + (inputValue?.hashCode() ?: 0)
        result = 31 * result + initialized.hashCode()
        result = 31 * result + signalMissing.hashCode()
        result = 31 * result + (state ?: 0)
        result = 31 * result + eventDone.hashCode()
        result = 31 * result + currentDurationMs.hashCode()
        result = 31 * result + lastEventDurationMs.hashCode()
        result = 31 * result + ratePerMinute.hashCode()
        result = 31 * result + prolonged.hashCode()
        result = 31 * result + modelUpdated.hashCode()
        result = 31 * result + (posterior?.contentHashCode() ?: 0)
        result = 31 * result + (p80Threshold?.hashCode() ?: 0)
        result = 31 * result + (perclos?.hashCode() ?: 0)
        result = 31 * result + (pom?.hashCode() ?: 0)
        result = 31 * result + observedSeconds.hashCode()
        return result
    }
}
