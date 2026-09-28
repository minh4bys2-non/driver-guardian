package com.example.driverguardian.ai.physical.temporal

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
    val ratePerMinute: Double,
    val prolonged: Boolean,
    val observedSeconds: Double
)

/**
 * Combined pipeline coordinating AdaptiveHmm, TemporalStateMachine, and WindowRatio.
 *
 * Implements the 5-second initial calibration window, P80 dynamic threshold for PERCLOS,
 * and online adaptation.
 */
class AdaptiveHmmFsm(
    val mode: String = "eye",
    val fps: Int = 30,
    initDurationSec: Double = 5.0,
    windowSizeSec: Double = 60.0,
    minDurationMs: Double? = null,
    maxDurationMs: Double? = null,
    val maxGapSec: Double = 0.25,
    learningRate: Double = 0.01,
    adaptInterval: Int = 300
) {
    init {
        require(mode == "eye" || mode == "mouth" || mode == "pitch") { "Invalid mode: $mode" }
        require(fps > 0 && initDurationSec > 0.0) { "Invalid detector configuration" }
    }

    val initFrames = max(4, (fps * initDurationSec).roundToInt())
    val hmm = AdaptiveHmm(
        positiveState = if (mode == "eye") "low" else "high",
        learningRate = learningRate,
        adaptInterval = adaptInterval
    )
    val fsm = TemporalStateMachine(
        mode = mode,
        fps = fps,
        windowSizeSec = windowSizeSec,
        minDurationMs = minDurationMs,
        maxDurationMs = maxDurationMs,
        maxGapSec = maxGapSec
    )
    val ratio = WindowRatio(
        windowSec = windowSizeSec,
        maxGapSec = maxGapSec
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
        val low = percentile(sorted, 5.0)
        val high = percentile(sorted, 95.0)
        val separation = when (mode) {
            "eye" -> 0.06
            "mouth" -> 0.15
            else -> 10.0
        }
        val norm = percentile(sorted, if (mode == "eye") 80.0 else 20.0)
        this.normal = norm

        val distinct = if (mode == "eye") {
            low < norm * 0.65
        } else {
            high > norm + separation
        }

        if (high - low >= separation && distinct) {
            hmm.fitInitial(data)
        } else {
            val event = if (mode == "eye") norm * 0.2 else norm + separation * 2.0
            hmm.means = doubleArrayOf(norm, event)
            val sigma = max(abs(norm - event) / 4.0, 0.01)
            hmm.vars = doubleArrayOf(sigma * sigma, sigma * sigma)
            hmm.a = arrayOf(
                doubleArrayOf(0.97, 0.03),
                doubleArrayOf(0.1, 0.9)
            )
            hmm.pi = doubleArrayOf(0.99, 0.01)
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

        val validValue = if (value != null && value.isFinite()) value else null

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

                if (mode == "eye" && normal != null && validValue >= normal!! * 0.75) {
                    state = 0
                    val safePost = doubleArrayOf(0.99, 0.01)
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
            threshold = opened - 0.8 * (opened - closed)
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
