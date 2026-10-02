package com.example.driverguardian.ai.monitoring.camera

object CameraFrameGeometry {
    fun rotatedSize(width: Int, height: Int, rotationDegrees: Int): Pair<Int, Int> {
        require(width > 0 && height > 0) { "Frame dimensions must be positive" }
        return when (rotationDegrees) {
            0, 180 -> width to height
            90, 270 -> height to width
            else -> throw IllegalArgumentException("Rotation must be 0, 90, 180, or 270")
        }
    }
}
