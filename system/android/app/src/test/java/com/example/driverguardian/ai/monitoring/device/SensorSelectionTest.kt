package com.example.driverguardian.ai.monitoring.device

import org.junit.Assert.assertEquals
import org.junit.Test

class SensorSelectionTest {
    @Test fun selectsExplicitPriorityAndFallbacks() {
        assertEquals(DeviceSensorSource.ROTATION_VECTOR, SensorSelection.choose(true, true, true, true))
        assertEquals(DeviceSensorSource.GAME_ROTATION_VECTOR, SensorSelection.choose(false, true, true, true))
        assertEquals(DeviceSensorSource.ACCELEROMETER_MAGNETOMETER, SensorSelection.choose(false, false, true, true))
        assertEquals(DeviceSensorSource.UNAVAILABLE, SensorSelection.choose(false, false, true, false))
        assertEquals(DeviceSensorSource.UNAVAILABLE, SensorSelection.choose(false, false, false, true))
    }
}
