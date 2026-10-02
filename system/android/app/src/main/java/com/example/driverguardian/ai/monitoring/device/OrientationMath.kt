package com.example.driverguardian.ai.monitoring.device

import kotlin.math.asin
import kotlin.math.atan2

data class EulerOrientation(val pitchDeg: Double, val rollDeg: Double, val yawDeg: Double)

object OrientationMath {
    fun normalizeDegrees(value: Double): Double = ((value + 180.0) % 360.0 + 360.0) % 360.0 - 180.0
    fun shortestDeltaDegrees(neutral: Double, current: Double): Double = normalizeDegrees(current - neutral)

    fun fromQuaternion(x: Double, y: Double, z: Double, w: Double, displayRotationDegrees: Int): EulerOrientation {
        require(displayRotationDegrees in setOf(0, 90, 180, 270))
        val norm = kotlin.math.sqrt(x * x + y * y + z * z + w * w)
        require(norm > 0.0 && norm.isFinite())
        val qx = x / norm; val qy = y / norm; val qz = z / norm; val qw = w / norm
        val roll = atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        val pitch = asin((2.0 * (qw * qy - qz * qx)).coerceIn(-1.0, 1.0))
        val yaw = atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        val degrees = 180.0 / Math.PI
        return EulerOrientation(
            pitchDeg = normalizeDegrees(pitch * degrees),
            rollDeg = normalizeDegrees(roll * degrees),
            yawDeg = normalizeDegrees(yaw * degrees - displayRotationDegrees),
        )
    }
}
