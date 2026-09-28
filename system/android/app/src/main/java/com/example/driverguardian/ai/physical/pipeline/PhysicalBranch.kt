package com.example.driverguardian.ai.physical.pipeline

import com.example.driverguardian.ai.physical.model.LandmarkPoint
import com.example.driverguardian.ai.physical.model.PhysicalMetrics

/**
 * Public Physical Branch interface for processing facial landmark frames into driver physical metrics.
 */
interface PhysicalBranch {

    /**
     * Processes a single video frame's facial landmarks and returns full physiological & kinematic metrics.
     *
     * @param landmarks List of 468+ facial landmarks, or null if no face was detected.
     * @param imageWidth Source frame pixel width (> 0).
     * @param imageHeight Source frame pixel height (> 0).
     * @param timestampSec Monotonically increasing frame capture timestamp in seconds.
     * @return PhysicalMetrics containing ocular, oral, and head pose metrics with exact nullability.
     */
    fun processLandmarks(
        landmarks: List<LandmarkPoint>?,
        imageWidth: Int,
        imageHeight: Int,
        timestampSec: Double
    ): PhysicalMetrics

    /**
     * Resets all internal temporal states, sliding windows, calibration samples, and HMM buffers.
     */
    fun reset()
}
