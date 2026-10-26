package com.example.driverguardian.ai.monitoring.camera

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

class BrightnessBalancer(private val targetLuma: Double = 70.0) : Closeable {
    val enabled = AtomicBoolean(false)

    // Tái sử dụng các Mat cố định trên Luồng 1A (Zero-GC)
    private var rgbaMat = Mat()
    private var rgbMat = Mat()
    private var ycrcbMat = Mat()
    private val channels = ArrayList<Mat>(3)
    private val lutMat = Mat(1, 256, CvType.CV_8UC1)
    private val lutBytes = ByteArray(256)

    init {
        OpenCvInitializer.ensureInitialized()
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
        val totalBytes = width * height * 4
        var idx = 0
        while (idx < totalBytes) {
            val r = rgbaBytes[idx].toInt() and 0xFF
            val g = rgbaBytes[idx + 1].toInt() and 0xFF
            val b = rgbaBytes[idx + 2].toInt() and 0xFF
            sumY += (77 * r + 150 * g + 29 * b) shr 8
            sampleCount++
            idx += 16 // Lấy mẫu cách mỗi 4 pixel
        }
        val approxMeanY = sumY.toDouble() / sampleCount.toDouble()
        if (approxMeanY <= 0.0 || approxMeanY >= targetLuma + 1.5) {
            return // Môi trường đủ sáng, thoát ngay không tốn bất kỳ lệnh cvtColor nào
        }

        // Bước 2: Khi trời tối, thực hiện tính chính xác trên OpenCV YCrCb để khớp 100% với Python
        try {
            if (rgbaMat.rows() != height || rgbaMat.cols() != width) {
                rgbaMat.create(height, width, CvType.CV_8UC4)
            }
            rgbaMat.put(0, 0, rgbaBytes)
            Imgproc.cvtColor(rgbaMat, rgbMat, Imgproc.COLOR_RGBA2RGB)
            Imgproc.cvtColor(rgbMat, ycrcbMat, Imgproc.COLOR_RGB2YCrCb)

            Core.split(ycrcbMat, channels)
            val yChannel = channels[0]
            val meanY = Core.mean(yChannel).`val`[0]
            if (meanY <= 0.0 || meanY >= targetLuma) {
                channels.forEach { it.release() }
                channels.clear()
                return
            }

            val gamma = max(0.5, ln(targetLuma / 255.0) / ln(meanY / 255.0))
            for (i in 0..255) {
                val v = (255.0 * (i / 255.0).pow(gamma)).roundToInt().coerceIn(0, 255)
                lutBytes[i] = v.toByte()
            }
            lutMat.put(0, 0, lutBytes)
            Core.LUT(yChannel, lutMat, yChannel)
            Core.merge(channels, ycrcbMat)
            channels.forEach { it.release() }
            channels.clear()

            Imgproc.cvtColor(ycrcbMat, rgbMat, Imgproc.COLOR_YCrCb2RGB)
            Imgproc.cvtColor(rgbMat, rgbaMat, Imgproc.COLOR_RGB2RGBA)
            rgbaMat.get(0, 0, rgbaBytes)
        } catch (error: Throwable) {
            // Đảm bảo channels được giải phóng trong trường hợp có exception
            channels.forEach { it.release() }
            channels.clear()
        }
    }

    override fun close() {
        rgbaMat.release()
        rgbMat.release()
        ycrcbMat.release()
        lutMat.release()
        channels.forEach { it.release() }
        channels.clear()
    }
}
