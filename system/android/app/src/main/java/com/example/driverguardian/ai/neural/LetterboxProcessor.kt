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
    private var rotatedRgbaMat: Mat? = null
    private var resizedRgbaMat: Mat? = null
    private var resizedRgbMat: Mat? = null
    private var canvasRgbMat: Mat? = null
    private val hwcScratch = ByteArray(newSize * newSize * config.channelsRgb)

    init {
        try {
            openCvReady = OpenCvInitializer.ensureInitialized()
            if (openCvReady) {
                srcRgbaMat = Mat()
                rotatedRgbaMat = Mat()
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
     * Tối ưu thứ tự: Xoay (nếu rotationDegrees != 0) -> Resize RGBA -> bỏ kênh Alpha -> Pad ROI.
     */
    fun convertRgbaToRgbCHW(
        rgbaBytes: ByteArray,
        width: Int,
        height: Int,
        rotationDegrees: Int = 0
    ): ByteArray {
        val planeSize = config.planeSize
        val chwOut = ByteArray(config.chwBytesPerFrame)

        val effWidth = if (rotationDegrees == 90 || rotationDegrees == 270) height else width
        val effHeight = if (rotationDegrees == 90 || rotationDegrees == 270) width else height

        val scale = min(newSize.toDouble() / effHeight.toDouble(), newSize.toDouble() / effWidth.toDouble())
        val newW = max(1, (effWidth * scale).roundToInt())
        val newH = max(1, (effHeight * scale).roundToInt())
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

                val effectiveSrc = when (rotationDegrees) {
                    90 -> {
                        val rotated = rotatedRgbaMat ?: Mat().also { rotatedRgbaMat = it }
                        org.opencv.core.Core.rotate(src, rotated, org.opencv.core.Core.ROTATE_90_CLOCKWISE)
                        rotated
                    }
                    180 -> {
                        val rotated = rotatedRgbaMat ?: Mat().also { rotatedRgbaMat = it }
                        org.opencv.core.Core.rotate(src, rotated, org.opencv.core.Core.ROTATE_180)
                        rotated
                    }
                    270 -> {
                        val rotated = rotatedRgbaMat ?: Mat().also { rotatedRgbaMat = it }
                        org.opencv.core.Core.rotate(src, rotated, org.opencv.core.Core.ROTATE_90_COUNTERCLOCKWISE)
                        rotated
                    }
                    else -> src
                }

                Imgproc.resize(
                    effectiveSrc,
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

        return convertPureKotlinBilinear(rgbaBytes, width, height, newW, newH, padLeft, padTop, chwOut, rotationDegrees)
    }

    private fun convertPureKotlinBilinear(
        rgbaBytes: ByteArray,
        width: Int,
        height: Int,
        newW: Int,
        newH: Int,
        padLeft: Int,
        padTop: Int,
        chwOut: ByteArray,
        rotationDegrees: Int = 0
    ): ByteArray {
        val planeSize = config.planeSize
        chwOut.fill(padByte)

        val effWidth = if (rotationDegrees == 90 || rotationDegrees == 270) height else width
        val effHeight = if (rotationDegrees == 90 || rotationDegrees == 270) width else height

        val scaleX = effWidth.toDouble() / newW.toDouble()
        val scaleY = effHeight.toDouble() / newH.toDouble()

        for (dy in 0 until newH) {
            val fy = (dy + 0.5) * scaleY - 0.5
            val ey0 = fy.toInt().coerceIn(0, effHeight - 1)
            val ey1 = (ey0 + 1).coerceIn(0, effHeight - 1)
            val wy = (fy - ey0).coerceIn(0.0, 1.0)
            val wy0 = 1.0 - wy

            val dstRow = (padTop + dy) * newSize
            for (dx in 0 until newW) {
                val fx = (dx + 0.5) * scaleX - 0.5
                val ex0 = fx.toInt().coerceIn(0, effWidth - 1)
                val ex1 = (ex0 + 1).coerceIn(0, effWidth - 1)
                val wx = (fx - ex0).coerceIn(0.0, 1.0)
                val wx0 = 1.0 - wx

                fun mapCoord(ex: Int, ey: Int): Pair<Int, Int> = when (rotationDegrees) {
                    90 -> (width - 1 - ey) to ex
                    180 -> (width - 1 - ex) to (height - 1 - ey)
                    270 -> ey to (height - 1 - ex)
                    else -> ex to ey
                }

                val (x00, y00) = mapCoord(ex0, ey0)
                val (x01, y01) = mapCoord(ex1, ey0)
                val (x10, y10) = mapCoord(ex0, ey1)
                val (x11, y11) = mapCoord(ex1, ey1)

                val i00 = (y00 * width + x00) * 4
                val i01 = (y01 * width + x01) * 4
                val i10 = (y10 * width + x10) * 4
                val i11 = (y11 * width + x11) * 4

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
            rotatedRgbaMat?.release()
            resizedRgbaMat?.release()
            resizedRgbMat?.release()
            canvasRgbMat?.release()
        } catch (_: Throwable) {
        }
    }
}
