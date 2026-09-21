package com.driverguardian.ai.physical

/**
 * States of the Pitch Finite State Machine.
 */
enum class PitchFsmState {
    NORMAL,
    PITCH_DOWN
}

/**
 * Dynamic classification result of head pitch temporal tracking.
 */
data class PitchFsmResult(
    val state: PitchFsmState,
    val nodFrequencyPerMin: Int,
    val currentDurationMs: Double,
    val lastNodDurationMs: Double,
    val eventLog: String,
    val poseStatus: String,
    val isCritical: Boolean,
    val isWarning: Boolean
)

/**
 * Temporal accumulator and state machine for head pitch and drowsiness nod detection.
 *
 * Corresponds to the Temporal FSM logic in ai/PhysicalBranch/PitchFSM.py.
 */
class PitchFsm(
    val pitchDownThreshold: Double = 14.0,
    val nodMinDurationMs: Double = 800.0,
    val nodMaxDurationMs: Double = 3500.0,
    val windowSizeSec: Double = 60.0,
    val maxMissingFrames: Int = 12
) {
    var state: PitchFsmState = PitchFsmState.NORMAL
        private set

    var pitchDownStartTime: Double? = null
        private set

    val nodTimestamps: ArrayDeque<Double> = ArrayDeque()
    var lastNodDuration: Double = 0.0
        private set

    var eventLog: String = "System Stable"
        private set

    var nodFrequency: Int = 0
        private set

    var missingFaceCounter: Int = 0
        private set

    fun reset() {
        state = PitchFsmState.NORMAL
        pitchDownStartTime = null
        nodTimestamps.clear()
        lastNodDuration = 0.0
        eventLog = "System Stable"
        nodFrequency = 0
        missingFaceCounter = 0
    }

    /**
     * Handles missing face / lost tracking frames.
     */
    fun onMissingFace() {
        missingFaceCounter++
        if (missingFaceCounter > maxMissingFrames) {
            state = PitchFsmState.NORMAL
            pitchDownStartTime = null
        }
    }

    /**
     * Updates the pitch FSM with calibrated head angles at the given time.
     *
     * @param pitch Calibrated pitch angle in degrees (positive = nodding downward).
     * @param yaw Calibrated yaw angle in degrees (positive = right, negative = left).
     * @param roll Calibrated roll angle in degrees.
     * @param currentTimeSec Current timestamp in seconds.
     * @return PitchFsmResult
     */
    fun update(
        pitch: Double,
        yaw: Double,
        roll: Double,
        currentTimeSec: Double
    ): PitchFsmResult {
        missingFaceCounter = 0

        var currentDuration = 0.0

        if (pitch > pitchDownThreshold) {
            if (state == PitchFsmState.NORMAL) {
                state = PitchFsmState.PITCH_DOWN
                pitchDownStartTime = currentTimeSec
            } else {
                currentDuration = (currentTimeSec - (pitchDownStartTime ?: currentTimeSec)) * 1000.0
                lastNodDuration = currentDuration

                if (currentDuration > nodMaxDurationMs) {
                    eventLog = "CRITICAL: PROLONGED MICRO-SLEEP / DISTRACTION"
                } else if (currentDuration >= nodMinDurationMs) {
                    eventLog = "WARNING: DROWSY HEAD DROPPING DETECTED"
                }
            }
        } else {
            if (state == PitchFsmState.PITCH_DOWN) {
                val durationMs = (currentTimeSec - (pitchDownStartTime ?: currentTimeSec)) * 1000.0
                lastNodDuration = durationMs
                state = PitchFsmState.NORMAL

                if (durationMs in nodMinDurationMs..nodMaxDurationMs) {
                    eventLog = "F_nod: DROWSY HEAD NOD DETECTED (${durationMs.toInt()}ms)"
                    nodTimestamps.addLast(currentTimeSec)
                } else if (durationMs < nodMinDurationMs) {
                    eventLog = "INTENTIONAL GLANCE / TAPLO LOOK (${durationMs.toInt()}ms)"
                }
                pitchDownStartTime = null
            }
        }

        // Maintain 60-second sliding window for nod timestamps
        while (nodTimestamps.isNotEmpty() && (currentTimeSec - nodTimestamps.first() > windowSizeSec)) {
            nodTimestamps.removeFirst()
        }
        nodFrequency = nodTimestamps.size

        val isCrit = eventLog.contains("CRITICAL") || nodFrequency >= 3
        val isWarn = eventLog.contains("WARNING") || nodFrequency >= 1

        val poseStatus = when {
            isCrit -> "CRITICAL: FATIGUE / SLEEP GUCK!"
            isWarn -> "WARNING: DROWSY SIGNALS"
            yaw > 15.0 -> "Looking Right"
            yaw < -15.0 -> "Looking Left"
            else -> "Looking Center"
        }

        return PitchFsmResult(
            state = state,
            nodFrequencyPerMin = nodFrequency,
            currentDurationMs = currentDuration,
            lastNodDurationMs = lastNodDuration,
            eventLog = eventLog,
            poseStatus = poseStatus,
            isCritical = isCrit,
            isWarning = isWarn
        )
    }
}
