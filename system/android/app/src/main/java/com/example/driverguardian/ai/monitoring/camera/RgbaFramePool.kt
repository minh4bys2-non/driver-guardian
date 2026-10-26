package com.example.driverguardian.ai.monitoring.camera

import java.util.concurrent.atomic.AtomicInteger

/**
 * Bộ đệm vòng Triple-Buffering không khóa (Zero-GC Allocation Pool).
 * Chỉ cấp phát đúng 3 mảng RGBA (3 x 1.23 MB = 3.69 MB) trong suốt vòng đời ứng dụng:
 * - Slot 0: Luồng 1A đang ghi từ CameraX ImageProxy
 * - Slot 1: Đang nằm trong channel chờ Luồng 1B nhận
 * - Slot 2: Luồng 1B đang đọc để nạp vào GPU & cập nhật pipeline
 */
class RgbaFramePool(
    val width: Int = CameraXHardwareConfig.TARGET_WIDTH,
    val height: Int = CameraXHardwareConfig.TARGET_HEIGHT,
    private val poolSize: Int = 3
) {
    val frameBytesSize: Int = width * height * CameraXHardwareConfig.CHANNELS_RGBA
    private val buffers: Array<ByteArray> = Array(poolSize) { ByteArray(frameBytesSize) }
    private val writeCursor = AtomicInteger(0)

    fun acquireNextBuffer(): ByteArray {
        val idx = (writeCursor.getAndIncrement() and Int.MAX_VALUE) % poolSize
        return buffers[idx]
    }

    fun getBufferAt(index: Int): ByteArray = buffers[index % poolSize]

    val capacity: Int get() = poolSize
}
