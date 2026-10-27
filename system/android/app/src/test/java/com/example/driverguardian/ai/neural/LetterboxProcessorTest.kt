package com.example.driverguardian.ai.neural

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LetterboxProcessorTest {

    @Test
    fun testLetterboxRotationMapping() {
        val processor = LetterboxProcessor(NeuralBranchConfig(imageSize = 32, letterboxPadValue = 0))
        val width = 4
        val height = 2
        // Ảnh 4x2 RGBA: chỉ có pixel góc trên-trái (x=0, y=0) là màu đỏ rực (255, 0, 0)
        val rgba = ByteArray(width * height * 4) { 0 }
        rgba[0] = 255.toByte() // R
        rgba[1] = 0.toByte()   // G
        rgba[2] = 0.toByte()   // B
        rgba[3] = 255.toByte() // A

        // 1. Góc xoay 0°: điểm sáng ở góc trên
        val chw0 = processor.convertRgbaToRgbCHW(rgba, width, height, rotationDegrees = 0)
        assertEquals(3 * 32 * 32, chw0.size)
        // Kênh R (plane đầu tiên) phải có ít nhất một giá trị dương ở nửa trên ảnh
        var hasPositiveR0 = false
        for (i in 0 until 32 * 16) {
            if ((chw0[i].toInt() and 0xFF) > 100) {
                hasPositiveR0 = true
                break
            }
        }
        assertTrue("Xoay 0°: điểm sáng phải nằm ở nửa trên của ảnh", hasPositiveR0)

        // 2. Góc xoay 90° (theo chiều kim đồng hồ): điểm sáng (0, 0) chuyển sang góc trên-phải
        val chw90 = processor.convertRgbaToRgbCHW(rgba, width, height, rotationDegrees = 90)
        assertEquals(3 * 32 * 32, chw90.size)

        // 3. Góc xoay 180°: điểm sáng (0, 0) chuyển sang góc dưới-phải
        val chw180 = processor.convertRgbaToRgbCHW(rgba, width, height, rotationDegrees = 180)
        assertEquals(3 * 32 * 32, chw180.size)

        // 4. Góc xoay 270°: điểm sáng (0, 0) chuyển sang góc dưới-trái
        val chw270 = processor.convertRgbaToRgbCHW(rgba, width, height, rotationDegrees = 270)
        assertEquals(3 * 32 * 32, chw270.size)
    }
}
