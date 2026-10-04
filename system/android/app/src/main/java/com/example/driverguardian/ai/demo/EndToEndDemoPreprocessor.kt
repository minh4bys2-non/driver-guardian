package com.example.driverguardian.ai.demo

import android.graphics.Bitmap

/**
 * Preprocessor for the demo end-to-end ONNX model.
 *
 * Pipeline:
 * Frame (RGB / Bitmap)
 * -> Aspect-preserving letterbox to 640x640 with pad value 114
 * -> Normalization: pixel / 255.0f
 * -> Format: planar NCHW (FloatArray of shape [3, 640, 640], size 1,228,800)
 *
 * Designed with a pure-Kotlin pixel engine so that unit tests can verify letterboxing,
 * padding, scaling, and NCHW channel layout without requiring Android native graphics mocks.
 */
object EndToEndDemoPreprocessor {

    const val TARGET_WIDTH = 640
    const val TARGET_HEIGHT = 640
    const val TARGET_CHANNELS = 3
    const val PAD_VALUE = 114
    const val NORMALIZATION_SCALE = 255.0f
    const val FRAME_ELEMENT_COUNT = TARGET_CHANNELS * TARGET_HEIGHT * TARGET_WIDTH // 1,228,800

    val NORMALIZED_PAD_VALUE: Float = PAD_VALUE / NORMALIZATION_SCALE // ~0.4470588f

    /**
     * Geometry metadata resulting from letterbox calculation.
     */
    data class LetterboxGeometry(
        val scale: Float,
        val scaledWidth: Int,
        val scaledHeight: Int,
        val padLeft: Int,
        val padTop: Int,
        val padRight: Int,
        val padBottom: Int,
    )

    /**
     * Computes letterbox geometry preserving aspect ratio within [targetWidth] x [targetHeight].
     */
    fun calculateGeometry(
        srcWidth: Int,
        srcHeight: Int,
        targetWidth: Int = TARGET_WIDTH,
        targetHeight: Int = TARGET_HEIGHT,
    ): LetterboxGeometry {
        require(srcWidth > 0 && srcHeight > 0) { "Source dimensions must be positive: ${srcWidth}x$srcHeight" }
        require(targetWidth > 0 && targetHeight > 0) { "Target dimensions must be positive: ${targetWidth}x$targetHeight" }

        val scale = minOf(targetWidth.toFloat() / srcWidth, targetHeight.toFloat() / srcHeight)
        val scaledWidth = (srcWidth * scale).toInt().coerceAtMost(targetWidth)
        val scaledHeight = (srcHeight * scale).toInt().coerceAtMost(targetHeight)

        val padX = targetWidth - scaledWidth
        val padY = targetHeight - scaledHeight

        val padLeft = padX / 2
        val padRight = padX - padLeft
        val padTop = padY / 2
        val padBottom = padY - padTop

        return LetterboxGeometry(
            scale = scale,
            scaledWidth = scaledWidth,
            scaledHeight = scaledHeight,
            padLeft = padLeft,
            padTop = padTop,
            padRight = padRight,
            padBottom = padBottom,
        )
    }

    /**
     * Preprocesses an Android [Bitmap] into an NCHW [3, 640, 640] float buffer.
     */
    fun preprocessBitmap(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        return preprocessRgbPixels(pixels, width, height)
    }

    /**
     * Preprocesses an array of ARGB/RGB 32-bit pixel integers (e.g. from Bitmap or synthetic test frame)
     * into planar NCHW [3, 640, 640] floats scaled to [0.0, 1.0].
     *
     * Channel ordering in output:
     * - Channel 0 (Red): index 0 until (640*640)
     * - Channel 1 (Green): index (640*640) until (2*640*640)
     * - Channel 2 (Blue): index (2*640*640) until (3*640*640)
     */
    fun preprocessRgbPixels(
        pixels: IntArray,
        srcWidth: Int,
        srcHeight: Int,
        targetWidth: Int = TARGET_WIDTH,
        targetHeight: Int = TARGET_HEIGHT,
        padValue: Int = PAD_VALUE,
    ): FloatArray {
        require(pixels.size == srcWidth * srcHeight) {
            "Pixel array size (${pixels.size}) does not match dimensions (${srcWidth}x$srcHeight)"
        }

        val geom = calculateGeometry(srcWidth, srcHeight, targetWidth, targetHeight)
        val normalizedPad = padValue / NORMALIZATION_SCALE
        val spatialSize = targetWidth * targetHeight
        val output = FloatArray(TARGET_CHANNELS * spatialSize)

        val channelR = 0
        val channelG = spatialSize
        val channelB = 2 * spatialSize

        // Fill background with pad value
        output.fill(normalizedPad)

        // Resample source pixels into letterboxed area (nearest-neighbor sampling for deterministic parity)
        for (dstY in 0 until geom.scaledHeight) {
            val srcY = ((dstY + 0.5f) / geom.scale).toInt().coerceIn(0, srcHeight - 1)
            val outY = geom.padTop + dstY
            val srcRowOffset = srcY * srcWidth
            val outRowOffset = outY * targetWidth

            for (dstX in 0 until geom.scaledWidth) {
                val srcX = ((dstX + 0.5f) / geom.scale).toInt().coerceIn(0, srcWidth - 1)
                val outX = geom.padLeft + dstX
                val pixel = pixels[srcRowOffset + srcX]

                val r = ((pixel shr 16) and 0xFF) / NORMALIZATION_SCALE
                val g = ((pixel shr 8) and 0xFF) / NORMALIZATION_SCALE
                val b = (pixel and 0xFF) / NORMALIZATION_SCALE

                val targetIdx = outRowOffset + outX
                output[channelR + targetIdx] = r
                output[channelG + targetIdx] = g
                output[channelB + targetIdx] = b
            }
        }

        return output
    }
}
