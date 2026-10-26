package com.example.driverguardian.ai.monitoring.camera

import org.junit.Assert.assertEquals
import org.junit.Test

class CameraXHardwareConfigTest {

    @Test
    fun testHardwareResolutionConstants() {
        assertEquals(640, CameraXHardwareConfig.TARGET_WIDTH)
        assertEquals(480, CameraXHardwareConfig.TARGET_HEIGHT)
        assertEquals(4, CameraXHardwareConfig.CHANNELS_RGBA)
        assertEquals(640 * 480 * 4, CameraXHardwareConfig.FRAME_BYTES_RGBA)
    }
}
