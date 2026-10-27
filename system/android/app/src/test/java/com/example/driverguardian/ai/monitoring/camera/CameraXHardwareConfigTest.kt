package com.example.driverguardian.ai.monitoring.camera

import android.view.Surface
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

    @Test
    fun testResolveLandscapeTargetRotation() {
        // Hỗ trợ đầy đủ cả 2 chiều ngang chuẩn
        assertEquals(
            Surface.ROTATION_90,
            CameraXHardwareConfig.resolveLandscapeTargetRotation(Surface.ROTATION_90)
        )
        assertEquals(
            Surface.ROTATION_270,
            CameraXHardwareConfig.resolveLandscapeTargetRotation(Surface.ROTATION_270)
        )

        // Thiết bị tự nhiên dọc: tránh ghim nhầm ROTATION_0 khi chuyển từ màn hình khóa
        assertEquals(
            Surface.ROTATION_90,
            CameraXHardwareConfig.resolveLandscapeTargetRotation(Surface.ROTATION_0, isNaturalLandscapeDevice = false)
        )
        assertEquals(
            Surface.ROTATION_90,
            CameraXHardwareConfig.resolveLandscapeTargetRotation(Surface.ROTATION_180, isNaturalLandscapeDevice = false)
        )

        // Thiết bị ô tô tự nhiên ngang (Android Automotive head unit)
        assertEquals(
            Surface.ROTATION_0,
            CameraXHardwareConfig.resolveLandscapeTargetRotation(Surface.ROTATION_0, isNaturalLandscapeDevice = true)
        )
        assertEquals(
            Surface.ROTATION_180,
            CameraXHardwareConfig.resolveLandscapeTargetRotation(Surface.ROTATION_180, isNaturalLandscapeDevice = true)
        )
    }
}
