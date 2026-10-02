package com.example.driverguardian.ai.monitoring.device

object SensorSelection {
    fun choose(
        hasRotationVector: Boolean,
        hasGameRotationVector: Boolean,
        hasAccelerometer: Boolean,
        hasMagnetometer: Boolean,
    ): DeviceSensorSource = when {
        hasRotationVector -> DeviceSensorSource.ROTATION_VECTOR
        hasGameRotationVector -> DeviceSensorSource.GAME_ROTATION_VECTOR
        hasAccelerometer && hasMagnetometer -> DeviceSensorSource.ACCELEROMETER_MAGNETOMETER
        else -> DeviceSensorSource.UNAVAILABLE
    }
}
