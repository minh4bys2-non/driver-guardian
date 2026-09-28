package com.example.driverguardian.ai.physical.temporal

/**
 * Result produced by TemporalStateMachine after processing a state observation.
 */
data class FsmResult(
    val state: Int?,
    val signalMissing: Boolean,
    val eventDone: Boolean,
    val currentDurationMs: Double,
    val lastEventDurationMs: Double,
    val ratePerMinute: Double,
    val prolonged: Boolean
)

/**
 * Temporal Finite State Machine accumulating event durations and frequencies.
 *
 * Modes:
 * - "eye": 100 - 2000 ms (blinks vs prolonged closure)
 * - "mouth": 3500 - 7500 ms (yawns)
 * - "pitch": 800 - 3500 ms (drowsy head nods)
 */
class TemporalStateMachine(
    val mode: String = "eye",
    val fps: Int = 30,
    val windowSizeSec: Double = 60.0,
    minDurationMs: Double? = null,
    maxDurationMs: Double? = null,
    val maxGapSec: Double = 0.25
) {
    val minimum: Double
    val maximum: Double

    init {
        val limits = mapOf(
            "eye" to (100.0 to 2000.0),
            "mouth" to (3500.0 to 7500.0),
            "pitch" to (800.0 to 3500.0)
        )
        val defaultLimits = limits[mode] ?: throw IllegalArgumentException("Invalid mode: $mode")
        require(fps > 0 && windowSizeSec > 0.0 && maxGapSec > 0.0) { "Invalid FSM configuration" }

        minimum = minDurationMs ?: defaultLimits.first
        maximum = maxDurationMs ?: defaultLimits.second
        require(minimum in 0.0..<maximum) { "Invalid event duration limits: [$minimum, $maximum]" }
    }

    var time: Double? = null
        private set
    var start: Double? = null
        private set
    var lastDuration: Double = 0.0
        private set
    val events = ArrayDeque<Double>()

    fun reset() {
        time = null
        start = null
        lastDuration = 0.0
        events.clear()
    }

    fun process(state: Int?, timestamp: Double? = null): FsmResult {
        val now = timestamp ?: (if (time == null) 0.0 else time!! + 1.0 / fps)
        require(now.isFinite() && (time == null || now > time!!)) {
            "Timestamps must be finite and strictly increasing: current=$time, next=$now"
        }
        require(state == null || state == 0 || state == 1) { "State must be 0, 1, or null" }

        if (time != null && now - time!! > maxGapSec) {
            start = null
        }
        time = now

        var done = false
        if (state == null) {
            start = null
        } else if (state == 1 && start == null) {
            start = now
        } else if (state == 0 && start != null) {
            lastDuration = (now - start!!) * 1000.0
            done = lastDuration in minimum..maximum
            if (done) {
                events.addLast(now)
            }
            start = null
        }

        while (events.isNotEmpty() && events.first() <= now - windowSizeSec) {
            events.removeFirst()
        }

        val duration = if (start == null) 0.0 else (now - start!!) * 1000.0
        val rate = events.size * 60.0 / windowSizeSec
        val prolonged = duration > maximum

        return FsmResult(
            state = state,
            signalMissing = state == null,
            eventDone = done,
            currentDurationMs = duration,
            lastEventDurationMs = lastDuration,
            ratePerMinute = rate,
            prolonged = prolonged
        )
    }
}
