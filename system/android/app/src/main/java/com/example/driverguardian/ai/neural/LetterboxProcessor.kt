package com.example.driverguardian.ai.neural

import com.example.driverguardian.ai.monitoring.camera.OpenCvInitializer
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.Closeable
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Bộ xử lý `letterboxRgbaToRgbCHW(224)` (Đặc tả Snapdragon 680 v2 Mục 3.2).
 *
 * Chỉ được kích hoạt đúng tại thời điểm lọt lưới `5.0 FPS` (5 lần/giây) trên Luồng 1B.
 * Biến đổi 1 frame RGBA (640x480) sang mảng phẳng RGB CHW `[3, 224, 224]` `uint8` (`150,528 bytes`).
 */
class LetterboxProcessor(
    val config: NeuralBranchConfig = NeuralBranchConfig.DEFAULT
) : Closeable {

    private val newSize: Int = config.imageSize
    private val padByte: Byte = config.letterboxPadValue.toByte()
    private val padScalar = Scalar(
        config.letterboxPadValue.toDouble(),
        config.letterboxPadValue.toDouble(),
        config.letterboxPadValue.toDouble()
    )

    private var openCvReady: Boolean = false
    private var srcRgbaMat: Mat? = null
    private var resizedRgbaMat: Mat? = null
    private var resizedRgbMat: Mat? = null
    private var canvasRgbMat: Mat? = null
    private val hwcScratch = ByteArray(newSize * newSize * config.channelsRgb)

    init {
        try {
            openCvReady = OpenCvInitializer.ensureInitialized()
            if (openCvReady) {
                srcRgbaMat = Mat()
                resizedRgbaMat = Mat()
                resizedRgbMat = Mat()
                canvasRgbMat = Mat(newSize, newSize, CvType.CV_8UC3, padScalar)
            }
        } catch (_: Throwable) {
            openCvReady = false
        }
    }

    /**
     * Biến đổi 1 frame RGBA (`width x height`) sang mảng phẳng RGB CHW `[3, newSize, newSize]` `uint8`.
     * Tối ưu thứ tự: Resize RGBA (`640x480 -> 224x168`) trước rồi mới bỏ kênh Alpha!
     */
    fun convertRgbaToRgbCHW(rgbaBytes: ByteArray, width: Int, height: Int): ByteArray {
        val planeSize = config.planeSize
        val chwOut = ByteArray(config.chwBytesPerFrame)

        val scale = min(newSize.toDouble() / height.toDouble(), newSize.toDouble() / width.toDouble())
        val newW = max(1, (width * scale).roundToInt())
        val newH = max(1, (height * scale).roundToInt())
        val padLeft = (newSize - newW) / 2
        val padTop = (newSize - newH) / 2

        if (openCvReady && srcRgbaMat != null && resizedRgbaMat != null && resizedRgbMat != null && canvasRgbMat != null) {
            try {
                val src = srcRgbaMat!!
                val resizedRgba = resizedRgbaMat!!
                val resizedRgb = resizedRgbMat!!
                val canvas = canvasRgbMat!!

                if (src.rows() != height || src.cols() != width) {
                    src.create(height, width, CvType.CV_8UC4)
                }
                src.put(0, 0, rgbaBytes)

                Imgproc.resize(
                    src,
                    resizedRgba,
                    Size(newW.toDouble(), newH.toDouble()),
                    0.0,
                    0.0,
                    Imgproc.INTER_LINEAR
                )
                Imgproc.cvtColor(resizedRgba, resizedRgb, Imgproc.COLOR_RGBA2RGB)

                canvas.setTo(padScalar)
                val roi = canvas.submat(padTop, padTop + newH, padLeft, padLeft + newW)
                resizedRgb.copyTo(roi)
                roi.release()

                canvas.get(0, 0, hwcScratch)
                var srcIdx = 0
                for (i in 0 until planeSize) {
                    chwOut[i] = hwcScratch[srcIdx]                     // Kênh R
                    chwOut[planeSize + i] = hwcScratch[srcIdx + 1]     // Kênh G
                    chwOut[2 * planeSize + i] = hwcScratch[srcIdx + 2] // Kênh B
                    srcIdx += 3
                }
                return chwOut
            } catch (_: Throwable) {
                // Fallback sang nội suy song tuyến thuần Kotlin nếu OpenCV native không khả dụng (ví dụ JVM Unit Test)
            }
        }

        return convertPureKotlinBilinear(rgbaBytes, width, height, newW, newH, padLeft, padTop, chwOut)
    }

    private fun convertPureKotlinBilinear(
        rgbaBytes: ByteArray,
        width: Int,
        height: Int,
        newW: Int,
        newH: Int,
        padLeft: Int,
        padTop: Int,
        chwOut: ByteArray
    ): ByteArray {
        val planeSize = config.planeSize
        chwOut.fill(padByte)

        val scaleX = width.toDouble() / newW.toDouble()
        val scaleY = height.toDouble() / newH.toDouble()

        for (dy in 0 until newH) {
            val fy = (dy + 0.5) * scaleY - 0.5
            val y0 = fy.toInt().coerceIn(0, height - 1)
            val y1 = (y0 + 1).coerceIn(0, height - 1)
            val wy = (fy - y0).coerceIn(0.0, 1.0)
            val wy0 = 1.0 - wy

            val dstRow = (padTop + dy) * newSize
            for (dx in 0 until newW) {
                val fx = (dx + 0.5) * scaleX - 0.5
                val x0 = fx.toInt().coerceIn(0, width - 1)
                val x1 = (x0 + 1).coerceIn(0, width - 1)
                val wx = (fx - x0).coerceIn(0.0, 1.0)
                val wx0 = 1.0 - wx

                val i00 = (y0 * width + x0) * 4
                val i01 = (y0 * width + x1) * 4
                val i10 = (y1 * width + x0) * 4
                val i11 = (y1 * width + x1) * 4

                val dstIdx = dstRow + padLeft + dx
                for (c in 0 until 3) {
                    val v00 = rgbaBytes[i00 + c].toInt() and 0xFF
                    val v01 = rgbaBytes[i01 + c].toInt() and 0xFF
                    val v10 = rgbaBytes[i10 + c].toInt() and 0xFF
                    val v11 = rgbaBytes[i11 + c].toInt() and 0xFF

                    val top = v00 * wx0 + v01 * wx
                    val bot = v10 * wx0 + v11 * wx
                    val interp = (top * wy0 + bot * wy).roundToInt().coerceIn(0, 255)
                    chwOut[c * planeSize + dstIdx] = interp.toByte()
                }
            }
        }
        return chwOut
    }

    override fun close() {
        try {
            srcRgbaMat?.release()
            resizedRgbaMat?.release()
            resizedRgbMat?.release()
            canvasRgbMat?.release()
        } catch (_: Throwable) {
        }
    }
}
