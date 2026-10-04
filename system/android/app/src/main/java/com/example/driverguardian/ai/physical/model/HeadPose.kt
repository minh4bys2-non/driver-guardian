package com.example.driverguardian.ai.physical.model

/**
 * 3D head rotation angles in degrees.
 *
 * @param pitchDeg Rotation around lateral axis (positive = pitch down/nod, negative = pitch up).
 * @param yawDeg Rotation around vertical axis (positive = turn left, negative = turn right).
 * @param rollDeg Rotation around longitudinal axis (positive = tilt left, negative = tilt right).
 */
data class HeadPose(
    val pitchDeg: Double,
    val yawDeg: Double,
    val rollDeg: Double
)
