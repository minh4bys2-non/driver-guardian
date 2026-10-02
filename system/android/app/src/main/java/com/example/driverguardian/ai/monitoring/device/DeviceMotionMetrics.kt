package com.example.driverguardian.ai.monitoring.device

enum class DeviceCalibrationState { CALIBRATING, AVAILABLE, UNAVAILABLE }

data class DeviceMotionMetrics(
    val timestampSec: Double,
    val source: DeviceSensorSource,
    val calibrationState: DeviceCalibrationState,
    val rawPitchDeg: Double?,
    val rawRollDeg: Double?,
    val rawYawDeg: Double?,
    val deltaPitchDeg: Double?,
    val deltaRollDeg: Double?,
    val deltaYawDeg: Double?,
    val accuracy: Int?,
    val sampleAgeMs: Double? = null,
)
