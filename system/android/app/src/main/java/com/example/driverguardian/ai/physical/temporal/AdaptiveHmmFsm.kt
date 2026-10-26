package com.example.driverguardian.ai.physical.temporal

import com.example.driverguardian.ai.physical.PhysicalConfig
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Output data class produced by AdaptiveHmmFsm for eye or mouth detectors.
 */
data class AdaptiveHmmFsmResult(
    val mode: String,
    val inputValue: Double?,
    val state: Int?,
    val signalMissing: Boolean,
    val initialized: Boolean,
    val modelUpdated: Boolean,
    val posterior: DoubleArray?,
    val p80Threshold: Double?,
    val perclos: Double?,
    val pom: Double?,
    val eventDone: Boolean,
    val currentDurationMs: Double,
    val lastEventDurationMs: Double,
    val lastValidDurationMs: Double? = null,
    val eventCount: Int = 0,
    val ratePerMinute: Double,
    val prolonged: Boolean,
    val observedSeconds: Double
)

/**
 * Combined pipeline coordinating AdaptiveHmm, TemporalStateMachine, and O(1) WindowRatio.
 */
class AdaptiveHmmFsm(
    val mode: String = "eye",
    val fps: Int = PhysicalConfig.DEFAULT_FPS,
    initDurationSec: Double = PhysicalConfig.DEFAULT_INIT_DURATION_SEC,
    windowSizeSec: Double = PhysicalConfig.DEFAULT_WINDOW_SEC,
    minDurationMs: Double? = null,
    maxDurationMs: Double? = null,
    val maxGapSec: Double = PhysicalConfig.DEFAULT_MAX_GAP_SEC,
    learningRate: Double = PhysicalConfig.DEFAULT_HMM_LEARNING_RATE,
    adaptInterval: Int = PhysicalConfig.DEFAULT_HMM_ADAPT_INTERVAL,
    val config: PhysicalConfig = PhysicalConfig.DEFAULT
) {
    init {
        require(mode == "eye" || mode == "mouth" || mode == "pitch") { "Invalid mode: $mode" }
        require(fps > 0 && initDurationSec > 0.0) { "Invalid detector configuration" }
    }

    val initFrames = max(config.minInitFrames, (fps * initDurationSec).roundToInt())
    val hmm = AdaptiveHmm(
        nStates = config.hmmStates,
        positiveState = if (mode == "eye") "low" else "high",
        learningRate = learningRate,
        adaptInterval = adaptInterval,
        minVariance = config.hmmMinVariance,
        emIterations = config.hmmEmIterations,
        config = config
    )
    val fsm = TemporalStateMachine(
        mode = mode,
        fps = fps,
        windowSizeSec = windowSizeSec,
        minDurationMs = minDurationMs,
        maxDurationMs = maxDurationMs,
        maxGapSec = maxGapSec,
        config = config
    )
    val ratio = WindowRatio(
        windowSec = windowSizeSec,
        maxGapSec = maxGapSec,
        config = config
    )

    private val samples = ArrayDeque<Double>(initFrames)
    var initialized: Boolean = false
        private set
    var normal: Double? = null
        private set
    var eventReference: Double? = null
        private set

    fun reset() {
        hmm.reset()
        fsm.reset()
        samples.clear()
        ratio.reset()
        initialized = false
        normal = null
        eventReference = null
    }

    private fun initializeHMM() {
        val data = samples.toDoubleArray()
        val sorted = data.clone().apply { sort() }
        val low = percentile(sorted, config.initLowPercentile)
        val high = percentile(sorted, config.initHighPercentile)
        val separation = when (mode) {
            "eye" -> config.eyeSeparation
            "mouth" -> config.mouthSeparation
            else -> config.pitchSeparation
        }
        val norm = percentile(
            sorted,
            if (mode == "eye") config.eyeBaselinePercentile else config.otherBaselinePercentile
        )
        this.normal = norm

        val distinct = if (mode == "eye") {
            low < norm * config.eyeDistinctRatio
        } else {
            high > norm + separation
        }

        if (high - low >= separation && distinct) {
            hmm.fitInitial(data)
        } else {
            val event = if (mode == "eye") norm * config.eyeClosedRatio else norm + separation * config.eventSeparationMultiplier
            hmm.means = doubleArrayOf(norm, event)
            val sigma = max(abs(norm - event) / config.fallbackSigmaDivisor, config.fallbackMinSigma)
            hmm.vars = doubleArrayOf(sigma * sigma, sigma * sigma)
            hmm.a = Array(config.fallbackTransition.size) { r -> config.fallbackTransition[r].clone() }
            hmm.pi = config.restingPrior.clone()
            hmm.initialized = true
        }

        this.eventReference = hmm.means[1]
        this.initialized = true
        samples.clear()
    }

    fun process(value: Double?, timestamp: Double? = null): AdaptiveHmmFsmResult {
        if (timestamp != null) {
            require(timestamp.isFinite() && (fsm.time == null || timestamp > fsm.time!!)) {
                "Timestamps must be finite and strictly increasing"
            }
            if (fsm.time != null && timestamp - fsm.time!! > fsm.maxGapSec) {
                hmm.posterior = null
                hmm.buffer.clear()
            }
        }

        val validValue = if (value != null && value.isFinite() && (mode == "pitch" || value >= 0.0)) value else null

        var state: Int? = null
        var posterior: DoubleArray? = null
        var updated = false

        if (validValue != null) {
            if (!initialized) {
                if (samples.size >= initFrames) {
                    samples.removeFirst()
                }
                samples.addLast(validValue)
                if (samples.size == initFrames) {
                    initializeHMM()
                }
            }
            if (initialized) {
                val pred = hmm.predict(validValue)
                state = pred.first
                posterior = pred.second

                if (mode == "eye" && normal != null && validValue >= normal!! * config.eyeOpenRatio) {
                    state = 0
                    val safePost = config.restingPrior.clone()
                    hmm.posterior = safePost
                    posterior = safePost
                }
                updated = hmm.updateOnline(validValue)
            }
        } else {
            hmm.posterior = null
            hmm.buffer.clear()
        }

        val fsmRes = fsm.process(state, timestamp)
        val now = fsm.time!!

        var threshold: Double? = null
        if (mode == "eye" && initialized && normal != null && eventReference != null) {
            val opened = normal!!
            val closed = eventReference!!
            threshold = opened - config.perclosClosedFraction * (opened - closed)
        }

        val active: Boolean? = if (mode == "eye") {
            if (validValue == null || threshold == null) null else validValue < threshold
        } else {
            if (state == null) null else state == 1
        }

        val (percentage, validSec) = ratio.process(active, now)

        return AdaptiveHmmFsmResult(
            mode = mode,
            inputValue = validValue,
            state = state,
            signalMissing = validValue == null,
            initialized = initialized,
            modelUpdated = updated,
            posterior = posterior?.clone(),
            p80Threshold = threshold,
            perclos = if (mode == "eye") percentage else null,
            pom = if (mode == "mouth") percentage else null,
            eventDone = fsmRes.eventDone,
            currentDurationMs = fsmRes.currentDurationMs,
            lastEventDurationMs = fsmRes.lastEventDurationMs,
            lastValidDurationMs = fsmRes.lastValidDurationMs,
            eventCount = fsmRes.eventCount,
            ratePerMinute = fsmRes.ratePerMinute,
            prolonged = fsmRes.prolonged,
            observedSeconds = validSec
        )
    }

    private fun percentile(sorted: DoubleArray, pct: Double): Double {
        val n = sorted.size
        if (n == 0) return 0.0
        val rank = (pct / 100.0) * (n - 1)
        val lower = rank.toInt()
        val upper = if (lower + 1 < n) lower + 1 else lower
        val weight = rank - lower
        return sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}
