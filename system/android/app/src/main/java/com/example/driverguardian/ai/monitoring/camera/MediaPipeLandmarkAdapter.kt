package com.example.driverguardian.ai.monitoring.camera

import com.example.driverguardian.ai.physical.model.LandmarkPoint
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark

object MediaPipeLandmarkAdapter {
    /** Converts canonical, rotation-corrected MediaPipe coordinates. Never mirrors. */
    fun toPixelLandmarks(
        landmarks: List<NormalizedLandmark>,
        width: Int,
        height: Int,
    ): List<LandmarkPoint> {
        require(width > 0 && height > 0) { "Frame dimensions must be positive" }
        return landmarks.map { LandmarkPoint(it.x() * width, it.y() * height, it.z()) }
    }
}
