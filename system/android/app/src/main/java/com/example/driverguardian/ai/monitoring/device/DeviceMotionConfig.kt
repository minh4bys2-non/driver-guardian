package com.example.driverguardian.ai.monitoring.device

data class DeviceMotionConfig(
    val calibrationDurationSec: Double = 0.75,
    val minimumCalibrationSamples: Int = 10,
) {
    init {
        require(calibrationDurationSec in 0.5..1.0)
        require(minimumCalibrationSamples >= 2)
    }
}
