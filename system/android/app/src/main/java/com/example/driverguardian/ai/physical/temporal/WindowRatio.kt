package com.example.driverguardian.ai.physical.temporal

import com.example.driverguardian.ai.physical.PhysicalConfig

/**
 * Sliding window ratio accumulator calculating percentage of active time over a time window
 * using O(1) Running Sum (`RunningWindowRatio` in Snapdragon 680 v2 Spec Section 4.5).
 *
 * Used for PERCLOS (P80), POM (Percentage of Open Mouth), and Over-angle percentage.
 * Calculated strictly by elapsed time, NOT frame count.
 */
class WindowRatio(
    val windowSec: Double = PhysicalConfig.DEFAULT_WINDOW_SEC,
    val maxGapSec: Double = PhysicalConfig.DEFAULT_MAX_GAP_SEC,
    val config: PhysicalConfig = PhysicalConfig.DEFAULT
) {
    init {
        require(windowSec > 0.0 && maxGapSec > 0.0) { "Window and gap must be positive" }
    }

    var window: Double = windowSec

    private data class Interval(val start: Double, val end: Double, val active: Boolean)

    private val intervals = ArrayDeque<Interval>(config.windowRatioInitialCapacity)
    private var previousTime: Double? = null
    private var previousActive: Boolean = false
    var time: Double? = null
        private set

    // Hai biến tổng trượt cộng dồn O(1) (< 20 FLOPs/frame)
    private var runningValid: Double = 0.0
    private var runningTotal: Double = 0.0
    private var stepsSinceRecalc: Int = 0

    fun reset() {
        intervals.clear()
        previousTime = null
        previousActive = false
        time = null
        runningValid = 0.0
        runningTotal = 0.0
        stepsSinceRecalc = 0
    }

    /**
     * Integrates an observation over the elapsed window in O(1) time.
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

        if (previousTime != null && active != null) {
            val start = previousTime!!
            val dur = timestamp - start
            if (dur <= maxGapSec) {
                intervals.addLast(Interval(start, timestamp, previousActive))
                runningValid += dur
                if (previousActive) {
                    runningTotal += dur
                }
            }
        }

        if (active == null) {
            previousTime = null
        } else {
            previousTime = timestamp
            previousActive = active
        }

        val cutoff = timestamp - window
        while (intervals.isNotEmpty() && intervals.first().end <= cutoff) {
            val removed = intervals.removeFirst()
            val dur = removed.end - removed.start
            runningValid -= dur
            if (removed.active) {
                runningTotal -= dur
            }
        }

        // Khử sai số dấu chấm động tích lũy định kỳ mỗi recalcIntervalSteps bước
        stepsSinceRecalc++
        if (stepsSinceRecalc >= config.windowRatioRecalcIntervalSteps) {
            stepsSinceRecalc = 0
            var v = 0.0
            var t = 0.0
            for (iv in intervals) {
                val d = iv.end - iv.start
                v += d
                if (iv.active) t += d
            }
            runningValid = v
            runningTotal = t
        }

        // Hiệu chỉnh phần đầu của đoạn đầu tiên nếu nó bị cắt một phần bởi biên cutoff
        var effectiveValid = runningValid
        var effectiveTotal = runningTotal
        if (intervals.isNotEmpty()) {
            val first = intervals.first()
            if (first.start < cutoff) {
                val clippedOut = cutoff - first.start
                effectiveValid -= clippedOut
                if (first.active) {
                    effectiveTotal -= clippedOut
                }
            }
        }

        if (effectiveValid <= config.windowRatioMinValidEpsilon) {
            return null to 0.0
        }
        val pct = config.percentageScale * effectiveTotal / effectiveValid
        return pct to effectiveValid
    }
}
