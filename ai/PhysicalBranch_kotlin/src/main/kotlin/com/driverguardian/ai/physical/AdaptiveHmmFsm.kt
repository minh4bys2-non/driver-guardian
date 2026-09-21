package com.driverguardian.ai.physical

import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Composite detector integrating AdaptiveHMM, StateMachine, and WindowRatio for a given mode.
 *
 * Corresponds to AdaptiveHMM_FSM in ai/PhysicalBranch/adaptive_hmm_fsm.py.
 */
class AdaptiveHmmFsm(
    val mode: FsmMode = FsmMode.EYE,
    val fps: Double = 30.0,
    val initDurationSec: Double = 5.0,
    val windowSizeSec: Double = 60.0,
    minDurationMs: Double? = null,
    maxDurationMs: Double? = null,
    val maxGapSec: Double = 0.25,
    val learningRate: Double = 0.01,
    val adaptInterval: Int = 300
) {
    val initFrames: Int = maxOf(4, (fps * initDurationSec).roundToInt())
    val hmm: AdaptiveHMM = AdaptiveHMM(
        positiveState = if (mode == FsmMode.EYE) PositiveState.LOW else PositiveState.HIGH,
        learningRate = learningRate,
        adaptInterval = adaptInterval
    )
    val fsm: StateMachine = StateMachine(
        mode = mode,
        fps = fps,
        windowSizeSec = windowSizeSec,
        minDurationMs = minDurationMs,
        maxDurationMs = maxDurationMs,
        maxGapSec = maxGapSec
    )
    val ratio: WindowRatio = WindowRatio(
        windowSec = windowSizeSec,
        maxGapSec = maxGapSec
    )

    val samples: ArrayDeque<Double> = ArrayDeque()
    var initialized: Boolean = false
    var normal: Double? = null
    var eventReference: Double? = null

    init {
        require(fps > 0.0) { "fps must be > 0, got $fps" }
        require(initDurationSec > 0.0) { "init_duration_sec must be > 0, got $initDurationSec" }
        reset()
    }

    constructor(
        modeStr: String,
        fps: Double = 30.0,
        initDurationSec: Double = 5.0,
        windowSizeSec: Double = 60.0,
        minDurationMs: Double? = null,
        maxDurationMs: Double? = null,
        maxGapSec: Double = 0.25,
        learningRate: Double = 0.01,
        adaptInterval: Int = 300
    ) : this(
        mode = FsmMode.fromString(modeStr),
        fps = fps,
        initDurationSec = initDurationSec,
        windowSizeSec = windowSizeSec,
        minDurationMs = minDurationMs,
        maxDurationMs = maxDurationMs,
        maxGapSec = maxGapSec,
        learningRate = learningRate,
        adaptInterval = adaptInterval
    )

    fun reset() {
        hmm.reset()
        fsm.reset()
        samples.clear()
        ratio.reset()
        initialized = false
        normal = null
        eventReference = null
    }

    private fun initialize() {
        val data = samples.toDoubleArray()
        val low = MathUtils.percentile(data, 5.0)
        val high = MathUtils.percentile(data, 95.0)
        val separation = when (mode) {
            FsmMode.EYE -> 0.06
            FsmMode.MOUTH -> 0.15
            FsmMode.PITCH -> 10.0
        }
        val normVal = MathUtils.percentile(data, if (mode == FsmMode.EYE) 80.0 else 20.0)
        this.normal = normVal

        val distinct = if (mode == FsmMode.EYE) {
            low < normVal * 0.65
        } else {
            high > normVal + separation
        }

        if (high - low >= separation && distinct) {
            hmm.fitInitial(data)
        } else {
            val normalState = normVal
            val eventState = if (mode == FsmMode.EYE) normalState * 0.2 else normalState + separation * 2.0
            hmm.means = doubleArrayOf(normalState, eventState)
            val sigma = maxOf(abs(normalState - eventState) / 4.0, 0.01)
            hmm.vars = doubleArrayOf(sigma * sigma, sigma * sigma)
            hmm.A = arrayOf(doubleArrayOf(0.97, 0.03), doubleArrayOf(0.1, 0.9))
            hmm.pi = doubleArrayOf(0.99, 0.01)
            hmm.initialized = true
        }

        this.eventReference = hmm.means[1]
        this.initialized = true
        samples.clear()
    }

    /**
     * Processes an incoming scalar observation (EAR, MAR, or pitch) at the given timestamp.
     *
     * @param value Observation value or null if face/landmarks lost.
     * @param timestamp Observation timestamp in seconds.
     * @return Full AdaptiveHmmFsmResult data class.
     */
    fun process(value: Double?, timestamp: Double? = null): AdaptiveHmmFsmResult {
        if (timestamp != null) {
            require(timestamp.isFinite()) { "Timestamp must be finite, got $timestamp" }
            if (fsm.time != null) {
                require(timestamp > fsm.time!!) { "Timestamps must be strictly increasing: previous=${fsm.time}, current=$timestamp" }
                if (timestamp - fsm.time!! > fsm.maxGapSec) {
                    hmm.posterior = null
                    hmm.buffer.clear()
                }
            }
        }

        val validVal = if (value != null && value.isFinite()) value else null

        var state: Int? = null
        var posterior: DoubleArray? = null
        var updated = false

        if (validVal != null) {
            if (!initialized) {
                samples.addLast(validVal)
                if (samples.size == initFrames) {
                    initialize()
                }
            }
            if (initialized) {
                val (predState, predPost) = hmm.predict(validVal)
                state = predState
                posterior = predPost

                if (mode == FsmMode.EYE && normal != null && validVal >= normal!! * 0.75) {
                    state = 0
                    posterior = doubleArrayOf(0.99, 0.01)
                    hmm.posterior = posterior
                }
                updated = hmm.updateOnline(validVal)
            }
        } else {
            hmm.posterior = null
            hmm.buffer.clear()
        }

        val outFsm = fsm.process(state, timestamp)
        val now = fsm.time ?: 0.0

        var threshold: Double? = null
        if (mode == FsmMode.EYE && initialized && normal != null && eventReference != null) {
            val opened = normal!!
            val closed = eventReference!!
            threshold = opened - 0.8 * (opened - closed)
        }

        val active: Boolean? = when (mode) {
            FsmMode.EYE -> {
                if (validVal == null || threshold == null) null else validVal < threshold
            }
            else -> {
                if (state == null) null else state == 1
            }
        }

        val ratioResult = ratio.process(active, now)

        return AdaptiveHmmFsmResult(
            mode = mode,
            inputValue = validVal,
            initialized = initialized,
            signalMissing = validVal == null,
            state = outFsm.state,
            eventDone = outFsm.eventDone,
            currentDurationMs = outFsm.currentDurationMs,
            lastEventDurationMs = outFsm.lastEventDurationMs,
            ratePerMinute = outFsm.ratePerMinute,
            prolonged = outFsm.prolonged,
            modelUpdated = updated,
            posterior = posterior,
            p80Threshold = threshold,
            perclos = if (mode == FsmMode.EYE) ratioResult.percentage else null,
            pom = if (mode == FsmMode.MOUTH) ratioResult.percentage else null,
            observedSeconds = ratioResult.observedSeconds
        )
    }
}
