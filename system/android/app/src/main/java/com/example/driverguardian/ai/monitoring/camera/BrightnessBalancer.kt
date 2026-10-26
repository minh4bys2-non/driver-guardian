package com.example.driverguardian.ai.monitoring.camera

import com.example.driverguardian.ai.monitoring.MonitoringConfig
import com.example.driverguardian.ai.monitoring.ThreadDebugLogger
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.Closeable
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.roundToInt

class BrightnessBalancer(private val targetLuma: Double = MonitoringConfig.TARGET_LUMA) : Closeable {
    val enabled = AtomicBoolean(MonitoringConfig.DEFAULT_BALANCE_BRIGHTNESS_ENABLED)

    // Tái sử dụng các Mat cố định trên Luồng 1A (Zero-GC, khởi tạo lazy khi thực sự cần dùng OpenCV)
    private var rgbaMat: Mat? = null
    private var rgbMat: Mat? = null
    private var ycrcbMat: Mat? = null
    private val channels = ArrayList<Mat>(3)
    private var lutMat: Mat? = null
    private val lutBytes = ByteArray(MonitoringConfig.LUT_SIZE)

    private fun ensureMatsAllocated(width: Int, height: Int): Boolean {
        if (!OpenCvInitializer.ensureInitialized()) return false
        if (rgbaMat == null) rgbaMat = Mat()
        if (rgbMat == null) rgbMat = Mat()
        if (ycrcbMat == null) ycrcbMat = Mat()
        if (lutMat == null) lutMat = Mat(1, MonitoringConfig.LUT_SIZE, CvType.CV_8UC1)
        val rgba = rgbaMat!!
        if (rgba.rows() != height || rgba.cols() != width) {
            rgba.create(height, width, CvType.CV_8UC4)
        }
        return true
    }

    /**
     * Sửa đổi trực tiếp tại chỗ (in-place) trên rgbaBytes lấy từ RgbaFramePool.
     * Không cấp phát mảng ByteArray mới.
     */
    fun processInPlaceIfDark(rgbaBytes: ByteArray, width: Int, height: Int) {
        if (!enabled.get() || targetLuma <= 0.0) {
            return // Nhánh Zero-Conversion (< 0.001 ms)
        }

        // Bước 1: Kiểm tra nhanh Luma trung bình (Fast Luma Check) trên lưới lấy mẫu bước nhảy 4 pixel
        // Tránh gọi cvtColor & Core.split khi môi trường đã đủ sáng (meanY >= 70.0)
        var sumY = 0L
        var sampleCount = 0
        val totalBytes = width * height * MonitoringConfig.CHANNELS_RGBA
        var idx = 0
        while (idx < totalBytes) {
            val r = rgbaBytes[idx].toInt() and 0xFF
            val g = rgbaBytes[idx + 1].toInt() and 0xFF
            val b = rgbaBytes[idx + 2].toInt() and 0xFF
            sumY += (
                MonitoringConfig.LUMA_COEFF_R * r +
                    MonitoringConfig.LUMA_COEFF_G * g +
                    MonitoringConfig.LUMA_COEFF_B * b
                ) shr MonitoringConfig.LUMA_SHIFT_BITS
            sampleCount++
            idx += MonitoringConfig.FAST_LUMA_SAMPLE_STEP_BYTES // Lấy mẫu cách mỗi 4 pixel
        }
        val approxMeanY = sumY.toDouble() / sampleCount.toDouble()
        if (approxMeanY <= 0.0 || approxMeanY >= targetLuma + MonitoringConfig.FAST_LUMA_SKIP_MARGIN) {
            return // Môi trường đủ sáng, thoát ngay không tốn bất kỳ lệnh cvtColor nào
        }

        // Bước 2: Khi trời tối, thực hiện tính chính xác trên OpenCV YCrCb để khớp 100% với Python
        try {
            if (!ensureMatsAllocated(width, height)) {
                return
            }
            val rgba = rgbaMat!!
            val rgb = rgbMat!!
            val ycrcb = ycrcbMat!!
            val lut = lutMat!!

            rgba.put(0, 0, rgbaBytes)
            Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
            Imgproc.cvtColor(rgb, ycrcb, Imgproc.COLOR_RGB2YCrCb)

            Core.split(ycrcb, channels)
            val yChannel = channels[0]
            val meanY = Core.mean(yChannel).`val`[0]
            if (meanY <= 0.0 || meanY >= targetLuma) {
                channels.forEach { it.release() }
                channels.clear()
                return
            }

            val gamma = max(
                MonitoringConfig.GAMMA_MIN,
                ln(targetLuma / MonitoringConfig.MAX_PIXEL_VALUE) / ln(meanY / MonitoringConfig.MAX_PIXEL_VALUE),
            )
            for (i in 0 until MonitoringConfig.LUT_SIZE) {
                val v = (MonitoringConfig.MAX_PIXEL_VALUE * (i / MonitoringConfig.MAX_PIXEL_VALUE).pow(gamma))
                    .roundToInt()
                    .coerceIn(0, MonitoringConfig.LUT_SIZE - 1)
                lutBytes[i] = v.toByte()
            }
            lut.put(0, 0, lutBytes)
            Core.LUT(yChannel, lut, yChannel)
            Core.merge(channels, ycrcb)
            channels.forEach { it.release() }
            channels.clear()

            Imgproc.cvtColor(ycrcb, rgb, Imgproc.COLOR_YCrCb2RGB)
            Imgproc.cvtColor(rgb, rgba, Imgproc.COLOR_RGB2RGBA)
            rgba.get(0, 0, rgbaBytes)
            ThreadDebugLogger.logIngest(
                String.format(
                    java.util.Locale.US,
                    "BrightnessBalancer applied LUT: meanY=%.1f -> targetLuma=%.1f (gamma=%.3f)",
                    meanY,
                    targetLuma,
                    gamma,
                ),
            )
        } catch (error: Throwable) {
            // Đảm bảo channels được giải phóng trong trường hợp có exception
            channels.forEach { it.release() }
            channels.clear()
        }
    }

    override fun close() {
        runCatching { rgbaMat?.release() }
        runCatching { rgbMat?.release() }
        runCatching { ycrcbMat?.release() }
        runCatching { lutMat?.release() }
        rgbaMat = null
        rgbMat = null
        ycrcbMat = null
        lutMat = null
        channels.forEach { runCatching { it.release() } }
        channels.clear()
    }
}
