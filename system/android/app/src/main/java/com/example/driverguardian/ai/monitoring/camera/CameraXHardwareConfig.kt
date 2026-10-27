package com.example.driverguardian.ai.monitoring.camera

import android.util.Size
import android.view.Surface
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import com.example.driverguardian.ai.monitoring.MonitoringConfig
import java.util.concurrent.Executor

object CameraXHardwareConfig {
    const val TARGET_WIDTH = MonitoringConfig.TARGET_WIDTH
    const val TARGET_HEIGHT = MonitoringConfig.TARGET_HEIGHT
    const val CHANNELS_RGBA = MonitoringConfig.CHANNELS_RGBA
    const val FRAME_BYTES_RGBA = MonitoringConfig.FRAME_BYTES_RGBA // 1,228,800 bytes

    fun buildImageAnalysis(
        ingestExecutor: Executor,
        targetRotation: Int = Surface.ROTATION_90,
        analyzer: ImageAnalysis.Analyzer
    ): ImageAnalysis {
        val resolutionSelector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
            .setResolutionStrategy(
                ResolutionStrategy(
                    Size(TARGET_WIDTH, TARGET_HEIGHT),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                )
            )
            .build()

        return ImageAnalysis.Builder()
            .setResolutionSelector(resolutionSelector)
            .setTargetRotation(targetRotation)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also { it.setAnalyzer(ingestExecutor, analyzer) }
    }

    /**
     * Chuẩn hóa hướng màn hình cho ứng dụng chạy ở chế độ sensorLandscape.
     * Luôn giữ ROTATION_90 hoặc ROTATION_270 khi người dùng xoay ngang điện thoại/máy tính bảng.
     * Chống hiện tượng ghim nhầm ROTATION_0 (Portrait) khi vừa chuyển từ màn hình khóa.
     */
    fun resolveLandscapeTargetRotation(
        displayRotation: Int,
        isNaturalLandscapeDevice: Boolean = false,
    ): Int = when (displayRotation) {
        Surface.ROTATION_90, Surface.ROTATION_270 -> displayRotation
        Surface.ROTATION_0, Surface.ROTATION_180 -> {
            if (isNaturalLandscapeDevice) displayRotation else Surface.ROTATION_90
        }
        else -> Surface.ROTATION_90
    }

    /**
     * Sao chép trực tiếp từ ImageProxy vào mảng đích dstBuffer lấy từ RgbaFramePool.
     * Xử lý chính xác trường hợp rowStride == rowBytes và rowStride > rowBytes.
     * Tuyệt đối KHÔNG cấp phát mảng ByteArray mới trên mỗi frame (Zero-GC).
     */
    fun extractTightRgbaBytesInto(imageProxy: ImageProxy, dstBuffer: ByteArray) {
        val width = imageProxy.width
        val height = imageProxy.height
        val plane = imageProxy.planes[0]
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val rowBytes = width * CHANNELS_RGBA

        buffer.rewind()
        if (rowStride == rowBytes) {
            buffer.get(dstBuffer, 0, width * height * CHANNELS_RGBA)
        } else {
            for (y in 0 until height) {
                buffer.position(y * rowStride)
                buffer.get(dstBuffer, y * rowBytes, rowBytes)
            }
        }
    }
}
