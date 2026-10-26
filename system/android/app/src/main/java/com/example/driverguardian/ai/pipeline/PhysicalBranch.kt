package com.example.driverguardian.ai.pipeline

import com.example.driverguardian.ai.physical.model.LandmarkPoint
import com.example.driverguardian.ai.physical.model.PhysicalMetrics
import com.example.driverguardian.ai.physical.temporal.PitchSamplesSnapshot

/**
 * Public Physical Branch interface (`UnifiedCameraMetrics` on Thread 1B) for processing facial landmark frames
 * into driver physical metrics, maintaining event history deques and lazy FFT snapshots.
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
     * Captures an immutable snapshot of pitch samples for 1Hz Lazy FFT evaluation on Thread 2.
     */
    fun snapshotPitchSamplesForFft(): PitchSamplesSnapshot? = null

    /**
     * Captures an immutable snapshot of `(timestampSec, durationMs)` blink events within the current 60s window.
     */
    fun snapshotBlinkDurations(): List<Pair<Double, Double>> = emptyList()

    /**
     * Captures an immutable snapshot of `(timestampSec, durationMs)` nod events within the current 60s window.
     */
    fun snapshotNodDurations(): List<Pair<Double, Double>> = emptyList()

    /**
     * Updates the cached FFT dominant frequency and resolution produced by Thread 2.
     */
    fun updateCachedFftResult(frequencyHz: Double?, resolutionHz: Double?) = Unit

    /**
     * Resets all internal temporal states, sliding windows, calibration samples, and HMM buffers.
     */
    fun reset()
}
