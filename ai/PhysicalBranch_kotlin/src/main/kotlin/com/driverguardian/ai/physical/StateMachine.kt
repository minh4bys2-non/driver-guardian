package com.driverguardian.ai.physical

/**
 * Finite State Machine tracking event durations, event completions, and event frequency
 * in a rolling time window (e.g. blinks, yawns, head nods).
 *
 * Corresponds to StateMachine in ai/PhysicalBranch/adaptive_hmm_fsm.py.
 */
class StateMachine(
    val mode: FsmMode = FsmMode.EYE,
    val fps: Double = 30.0,
    val windowSizeSec: Double = 60.0,
    minDurationMs: Double? = null,
    maxDurationMs: Double? = null,
    val maxGapSec: Double = 0.25
) {
    val minimum: Double
    val maximum: Double

    var time: Double? = null
    var start: Double? = null
    var lastDuration: Double = 0.0
    val events: ArrayDeque<Double> = ArrayDeque()

    init {
        require(fps > 0.0) { "fps must be > 0, got $fps" }
        require(windowSizeSec > 0.0) { "window_size_sec must be > 0, got $windowSizeSec" }
        require(maxGapSec > 0.0) { "max_gap_sec must be > 0, got $maxGapSec" }

        val (lo, hi) = when (mode) {
            FsmMode.EYE -> Pair(100.0, 2000.0)
            FsmMode.MOUTH -> Pair(3500.0, 7500.0)
            FsmMode.PITCH -> Pair(800.0, 3500.0)
        }
        minimum = minDurationMs ?: lo
        maximum = maxDurationMs ?: hi
        require(minimum >= 0.0 && minimum < maximum) { "Invalid event duration limits: min=$minimum, max=$maximum" }
        reset()
    }

    constructor(
        modeStr: String,
        fps: Double = 30.0,
        windowSizeSec: Double = 60.0,
        minDurationMs: Double? = null,
        maxDurationMs: Double? = null,
        maxGapSec: Double = 0.25
    ) : this(
        mode = FsmMode.fromString(modeStr),
        fps = fps,
        windowSizeSec = windowSizeSec,
        minDurationMs = minDurationMs,
        maxDurationMs = maxDurationMs,
        maxGapSec = maxGapSec
    )

    fun reset() {
        time = null
        start = null
        lastDuration = 0.0
        events.clear()
    }

    /**
     * Updates the FSM with a new binary state (0 = inactive/normal, 1 = active/event, null = missing).
     *
     * @param state Current binary state.
     * @param timestamp Optional measurement timestamp in seconds.
     */
    fun process(state: Int?, timestamp: Double? = null): StateMachineResult {
        val now = if (timestamp == null) {
            if (time == null) 0.0 else time!! + 1.0 / fps
        } else {
            timestamp
        }

        require(now.isFinite()) { "Timestamp must be finite, got $now" }
        if (time != null) {
            require(now > time!!) { "Timestamps must be strictly increasing: previous=$time, current=$now" }
            if (now - time!! > maxGapSec) {
                start = null
            }
        }
        time = now

        if (state != null) {
            require(state == 0 || state == 1) { "State must be 0, 1 or null, got $state" }
        }

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

        // Evict expired events outside the sliding window
        while (events.isNotEmpty() && events.first() <= now - windowSizeSec) {
            events.removeFirst()
        }

        val currentDuration = if (start == null) 0.0 else (now - start!!) * 1000.0
        val rate = events.size * 60.0 / windowSizeSec

        return StateMachineResult(
            state = state,
            signalMissing = state == null,
            eventDone = done,
            currentDurationMs = currentDuration,
            lastEventDurationMs = lastDuration,
            ratePerMinute = rate,
            prolonged = currentDuration > maximum
        )
    }
}
