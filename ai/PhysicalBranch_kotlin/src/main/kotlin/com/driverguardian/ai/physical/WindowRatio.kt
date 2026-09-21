package com.driverguardian.ai.physical

/**
 * An interval record tracking start time, end time, and active boolean flag.
 */
data class RatioInterval(
    val start: Double,
    val end: Double,
    val active: Boolean
)

/**
 * Calculates continuous time-weighted ratios in a sliding window (e.g. PERCLOS, POM, OverAngle time ratio).
 *
 * Corresponds to WindowRatio in ai/PhysicalBranch/adaptive_hmm_fsm.py.
 */
class WindowRatio(
    val windowSec: Double = 60.0,
    val maxGapSec: Double = 0.25
) {
    val intervals: ArrayDeque<RatioInterval> = ArrayDeque()
    var previous: Pair<Double, Boolean>? = null
    var time: Double? = null

    init {
        require(windowSec > 0.0) { "window_sec must be > 0, got $windowSec" }
        require(maxGapSec > 0.0) { "max_gap_sec must be > 0, got $maxGapSec" }
        reset()
    }

    fun reset() {
        intervals.clear()
        previous = null
        time = null
    }

    /**
     * Updates with an active boolean flag (or null if missing signal) at the specified timestamp.
     *
     * @param active True if condition is active (e.g. eye closed, mouth open), False otherwise, null if signal lost.
     * @param timestamp Measurement timestamp in seconds.
     * @return WindowRatioResult containing percentage (0.0 - 100.0 or null) and valid observed seconds.
     */
    fun process(active: Boolean?, timestamp: Double): WindowRatioResult {
        require(timestamp.isFinite()) { "Timestamps must be finite, got $timestamp" }
        if (time != null) {
            require(timestamp > time!!) { "Timestamps must be strictly increasing: previous=$time, current=$timestamp" }
        }
        time = timestamp

        if (previous != null && active != null) {
            val (start, previousActive) = previous!!
            if (timestamp - start <= maxGapSec) {
                intervals.addLast(RatioInterval(start, timestamp, previousActive))
            }
        }

        previous = if (active == null) null else Pair(timestamp, active)

        val cutoff = timestamp - windowSec
        while (intervals.isNotEmpty() && intervals.first().end <= cutoff) {
            intervals.removeFirst()
        }

        var valid = 0.0
        var total = 0.0
        for (interval in intervals) {
            val duration = interval.end - maxOf(interval.start, cutoff)
            valid += duration
            if (interval.active) {
                total += duration
            }
        }

        val percentage = if (valid > 0.0) 100.0 * total / valid else null
        return WindowRatioResult(percentage, valid)
    }
}
