package com.example.driverguardian.ai.physical.temporal

import com.example.driverguardian.ai.physical.PhysicalConfig

/**
 * Result produced by TemporalStateMachine after processing a state observation.
 */
data class FsmResult(
    val state: Int?,
    val signalMissing: Boolean,
    val eventDone: Boolean,
    val currentDurationMs: Double,
    val lastEventDurationMs: Double,
    val lastValidDurationMs: Double? = null,
    val eventCount: Int = 0,
    val ratePerMinute: Double,
    val prolonged: Boolean
)

/**
 * Temporal Finite State Machine accumulating event durations and frequencies.
 */
class TemporalStateMachine(
    val mode: String = "eye",
    val fps: Int = PhysicalConfig.DEFAULT_FPS,
    val windowSizeSec: Double = PhysicalConfig.DEFAULT_WINDOW_SEC,
    minDurationMs: Double? = null,
    maxDurationMs: Double? = null,
    val maxGapSec: Double = PhysicalConfig.DEFAULT_MAX_GAP_SEC,
    val config: PhysicalConfig = PhysicalConfig.DEFAULT
) {
    val minimum: Double
    val maximum: Double
    var window: Double = windowSizeSec

    init {
        val limits = mapOf(
            "eye" to (config.blinkDurationMinMs to config.blinkDurationMaxMs),
            "mouth" to (config.yawnDurationMinMs to config.yawnDurationMaxMs),
            "pitch" to (config.nodDurationMinMs to config.nodDurationMaxMs)
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
    var lastValidDuration: Double? = null
        private set
    var armed: Boolean = false
        private set
    val events = ArrayDeque<Double>()

    fun reset() {
        time = null
        start = null
        lastDuration = 0.0
        lastValidDuration = null
        armed = false
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
            armed = false
        }
        time = now

        var done = false
        if (state == null) {
            start = null
            armed = false
        } else if (state == 1 && start == null) {
            start = now
        } else if (state == 0 && start != null) {
            lastDuration = (now - start!!) * config.millisecondsPerSecond
            val eps = config.durationBoundaryEpsilonMs
            done = armed && lastDuration >= (minimum - eps) && lastDuration <= (maximum + eps)
            if (done) {
                events.addLast(now)
                lastValidDuration = lastDuration
            }
            start = null
        }
        if (state == 0) {
            armed = true
        }

        while (events.isNotEmpty() && events.first() <= now - window) {
            events.removeFirst()
        }

        val duration = if (start == null) 0.0 else (now - start!!) * config.millisecondsPerSecond
        val rate = events.size * config.secondsPerMinute / window
        val prolonged = duration > maximum

        return FsmResult(
            state = state,
            signalMissing = state == null,
            eventDone = done,
            currentDurationMs = duration,
            lastEventDurationMs = lastDuration,
            lastValidDurationMs = lastValidDuration,
            eventCount = events.size,
            ratePerMinute = rate,
            prolonged = prolonged
        )
    }
}
