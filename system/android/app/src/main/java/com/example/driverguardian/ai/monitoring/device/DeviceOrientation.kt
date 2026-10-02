package com.example.driverguardian.ai.monitoring.device

enum class DeviceSensorSource { ROTATION_VECTOR, GAME_ROTATION_VECTOR, ACCELEROMETER_MAGNETOMETER, UNAVAILABLE }

data class DeviceOrientation(
    val timestampSec: Double,
    val pitchDeg: Double?,
    val rollDeg: Double?,
    val yawDeg: Double?,
    val accuracy: Int?,
    val source: DeviceSensorSource,
)
