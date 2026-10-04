package com.example.driverguardian.ai.monitoring.camera

import com.example.driverguardian.ai.physical.model.LandmarkPoint

object LandmarkOverlayMapper {
    /** UI-only transformation. Algorithmic landmark instances are never reused or mutated. */
    fun frontMirrorCopy(landmarks: List<LandmarkPoint>, displayWidth: Int): List<LandmarkPoint> {
        require(displayWidth > 0) { "Display width must be positive" }
        return landmarks.map { LandmarkPoint(displayWidth - it.x, it.y, it.z) }
    }
}
