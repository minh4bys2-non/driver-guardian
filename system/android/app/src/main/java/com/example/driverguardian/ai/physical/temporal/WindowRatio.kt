package com.example.driverguardian.ai.physical.temporal

import kotlin.math.max

/**
 * Sliding window ratio accumulator calculating percentage of active time over a time window.
 *
 * Used for PERCLOS (P80), POM (Percentage of Open Mouth), and Over-angle percentage.
 * Calculated strictly by elapsed time, NOT frame count.
 */
class WindowRatio(
    val windowSec: Double = 60.0,
    val maxGapSec: Double = 0.25
) {
    init {
        require(windowSec > 0.0 && maxGapSec > 0.0) { "Window and gap must be positive" }
    }

    private data class Interval(val start: Double, val end: Double, val active: Boolean)

    private val intervals = ArrayDeque<Interval>()
    private var previous: Pair<Double, Boolean>? = null
    var time: Double? = null
        private set

    fun reset() {
        intervals.clear()
        previous = null
        time = null
    }

    /**
     * Integrates an observation over the elapsed window.
     *
     * @param active Whether the monitored state is active (true/false) or missing (null).
     * @param timestamp Monotonically increasing timestamp in seconds.
     * @return Pair of (percentage 0.0-100.0 or null if no valid time, validObservedSeconds).
     */
    fun process(active: Boolean?, timestamp: Double): Pair<Double?, Double> {
        require(timestamp.isFinite() && (time == null || timestamp > time!!)) {
            "Timestamps must be finite and strictly increasing: current=$time, next=$timestamp"
        }
        time = timestamp

        if (previous != null && active != null) {
            val (start, prevActive) = previous!!
            if (timestamp - start <= maxGapSec) {
                intervals.addLast(Interval(start, timestamp, prevActive))
            }
        }
        previous = if (active == null) null else (timestamp to active)

        val cutoff = timestamp - windowSec
        while (intervals.isNotEmpty() && intervals.first().end <= cutoff) {
            intervals.removeFirst()
        }

        var valid = 0.0
        var total = 0.0
        for (interval in intervals) {
            val duration = interval.end - max(interval.start, cutoff)
            valid += duration
            if (interval.active) {
                total += duration
            }
        }

        val percentage = if (valid > 0.0) (100.0 * total / valid) else null
        return percentage to valid
    }
}
