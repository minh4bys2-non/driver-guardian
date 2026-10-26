package com.example.driverguardian.ai.monitoring.camera

import android.util.Size
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
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also { it.setAnalyzer(ingestExecutor, analyzer) }
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
