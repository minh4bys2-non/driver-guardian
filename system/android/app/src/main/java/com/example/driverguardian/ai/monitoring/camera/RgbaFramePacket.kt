package com.example.driverguardian.ai.monitoring.camera

/**
 * Gói dữ liệu khung hình RGBA chuẩn chuyển giao từ Luồng 1A (CameraIngestThread)
 * sang Luồng 1B (MediaPipe & UnifiedPhysicalThread) hoặc các bộ điều phối.
 *
 * Mảng [buffer] trỏ trực tiếp vào một trong 3 slot tĩnh của [RgbaFramePool] (Zero-GC).
 */
data class RgbaFramePacket(
    val buffer: ByteArray,
    val width: Int,
    val height: Int,
    val rotationDegrees: Int,
    val timestampSec: Double,
    val sensorTimestampNs: Long,
    val receiptElapsedNs: Long,
    val generation: Long,
    val ingestLatencyMs: Double,
)
