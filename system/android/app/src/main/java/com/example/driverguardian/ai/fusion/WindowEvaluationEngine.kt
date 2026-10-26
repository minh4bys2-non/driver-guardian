package com.example.driverguardian.ai.fusion

import com.example.driverguardian.ai.neural.NeuralClassifier
import com.example.driverguardian.ai.neural.NeuralWindowSnapshot
import com.example.driverguardian.ai.neural.OnnxPredictionResult
import com.example.driverguardian.ai.physical.model.PhysicalMetrics
import com.example.driverguardian.ai.physical.temporal.HeadMotionMetrics
import com.example.driverguardian.ai.physical.temporal.LazyHeadMotionWindow
import com.example.driverguardian.ai.physical.temporal.PitchSamplesSnapshot
import kotlin.math.floor

/**
 * Pipeline operating state matching `snapdragon_680_v2.md` Section 7.
 */
enum class PipelineStatus(val displayLabel: String) {
    WARMING_UP("WARMING UP"),
    WAITING_FOR_CAMERA_CLIP("WAITING FOR CAMERA CLIP"),
    WAITING_FOR_VALID_WINDOW("WAITING FOR VALID WINDOW"),
    ALERT("ALERT"),
    DROWSY("DROWSY"),
}

/**
 * Snapshot of the 1Hz window evaluation produced by Thread 2 (`HeavyComputeWorker`).
 */
data class WindowEvaluationSnapshot(
    val evaluatedAtSec: Double,
    val elapsedSec: Double,
    val windowSec: Double,
    val status: PipelineStatus,
    val neuralBufferSize: Int,
    val cachedP4Count: Int,
    val latestDeltaFramesCount: Int,
    val latestNeuralScore: Double?,
    val onnxInferenceMs: Double,
    val lazyFftMs: Double,
    val workerTotalMs: Double,
    val onnxModelVariant: String,
    val headMotionMetrics: HeadMotionMetrics?,
    val rawFeatures: Map<String, Double>?,
    val normalizedFeatures: Map<String, Double>?,
    val fusionSummary: FusionEvaluationSummary?,
    val lastSkipReason: String?,
)

/**
 * Executes the 1Hz window evaluation cycle on Thread 2 (`HeavyComputeWorker`):
 * 1. Incremental `p4_old` ONNX warmup/evaluation via [NeuralClassifier].
 * 2. Gate 1: Readiness & Coverage check (>= 48.0s observed for Eye, Mouth, Head).
 * 3. Gate 2: Lazy 1Hz FFT via [LazyHeadMotionWindow.evaluateExactFft] (>= 60.0s observed).
 * 4. Raw 8D feature extraction (`physical_v1`), [PhysicalNormalizer] scaling, and 3-method [FitnessEvaluator].
 */
class WindowEvaluationEngine(
    private val windowSec: Double = FusionConfig.DEFAULT_WINDOW_SEC,
    private val minObservedRatio: Double = FusionConfig.DEFAULT_MIN_OBSERVED_RATIO,
    private val headMotionMinObservedSec: Double = FusionConfig.DEFAULT_HEAD_MOTION_MIN_OBSERVED_SEC,
    private val blinkDurationPercentile: Double = FusionConfig.DEFAULT_BLINK_DURATION_PERCENTILE,
    private val nodDurationPercentile: Double = FusionConfig.DEFAULT_NOD_DURATION_PERCENTILE,
    private val requiredNeuralFrames: Int = FusionConfig.DEFAULT_REQUIRED_NEURAL_FRAMES,
    private val normalizer: PhysicalNormalizer = PhysicalNormalizer(),
    private val fitnessEvaluator: FitnessEvaluator = FitnessEvaluator(),
) {
    val minObservedSec: Double
        get() = windowSec * minObservedRatio

    /**
     * Evaluates a 1Hz tick given the latest physical metrics, snapshots of blink/nod deques,
     * pitch samples snapshot for Lazy FFT, and neural buffer snapshot.
     */
    fun evaluateWindow(
        nowSec: Double,
        elapsedSec: Double,
        physicalMetrics: PhysicalMetrics?,
        blinkDurationsMs: DoubleArray,
        nodDurationsMs: DoubleArray,
        pitchSnapshot: PitchSamplesSnapshot?,
        neuralSnapshot: NeuralWindowSnapshot,
        neuralClassifier: NeuralClassifier?,
    ): WindowEvaluationSnapshot {
        val workerStartNs = System.nanoTime()

        // Always update incremental p4_old cache if neural buffer has frames (both WARMING UP & post-60s)
        var onnxResult: OnnxPredictionResult? = null
        if (neuralClassifier != null && neuralSnapshot.size > 0) {
            onnxResult = neuralClassifier.predictAndUpdateCache(neuralSnapshot)
        }

        val cachedP4Count = onnxResult?.cachedP4Count ?: neuralClassifier?.cachedFrameCount ?: 0
        val deltaFramesCount = onnxResult?.deltaFramesProcessed ?: 0
        val onnxMs = onnxResult?.inferenceMs ?: 0.0
        val modelVariant = onnxResult?.modelVariant ?: neuralClassifier?.activeModelVariant ?: "UNLOADED"

        // Pre-60s WARMING UP phase (Section 2.2)
        if (elapsedSec < windowSec) {
            val workerTotalMs = (System.nanoTime() - workerStartNs) / 1_000_000.0
            return WindowEvaluationSnapshot(
                evaluatedAtSec = nowSec,
                elapsedSec = elapsedSec,
                windowSec = windowSec,
                status = PipelineStatus.WARMING_UP,
                neuralBufferSize = neuralSnapshot.size,
                cachedP4Count = cachedP4Count,
                latestDeltaFramesCount = deltaFramesCount,
                latestNeuralScore = onnxResult?.drowsyScore,
                onnxInferenceMs = onnxMs,
                lazyFftMs = 0.0,
                workerTotalMs = workerTotalMs,
                onnxModelVariant = modelVariant,
                headMotionMetrics = null,
                rawFeatures = null,
                normalizedFeatures = null,
                fusionSummary = null,
                lastSkipReason = String.format(
                    java.util.Locale.US,
                    "warming up (%.1fs / %.1fs)",
                    elapsedSec,
                    windowSec,
                ),
            )
        }

        if (neuralSnapshot.size < requiredNeuralFrames) {
            val workerTotalMs = (System.nanoTime() - workerStartNs) / 1_000_000.0
            return WindowEvaluationSnapshot(
                evaluatedAtSec = nowSec,
                elapsedSec = elapsedSec,
                windowSec = windowSec,
                status = PipelineStatus.WAITING_FOR_CAMERA_CLIP,
                neuralBufferSize = neuralSnapshot.size,
                cachedP4Count = cachedP4Count,
                latestDeltaFramesCount = deltaFramesCount,
                latestNeuralScore = null,
                onnxInferenceMs = onnxMs,
                lazyFftMs = 0.0,
                workerTotalMs = workerTotalMs,
                onnxModelVariant = modelVariant,
                headMotionMetrics = null,
                rawFeatures = null,
                normalizedFeatures = null,
                fusionSummary = null,
                lastSkipReason = String.format(
                    java.util.Locale.US,
                    "neural buffer frames (%d) < required (%d)",
                    neuralSnapshot.size,
                    requiredNeuralFrames,
                ),
            )
        }

        val neuralScore = onnxResult?.drowsyScore
        if (neuralScore == null || !neuralScore.isFinite()) {
            val workerTotalMs = (System.nanoTime() - workerStartNs) / 1_000_000.0
            return WindowEvaluationSnapshot(
                evaluatedAtSec = nowSec,
                elapsedSec = elapsedSec,
                windowSec = windowSec,
                status = PipelineStatus.WAITING_FOR_VALID_WINDOW,
                neuralBufferSize = neuralSnapshot.size,
                cachedP4Count = cachedP4Count,
                latestDeltaFramesCount = deltaFramesCount,
                latestNeuralScore = null,
                onnxInferenceMs = onnxMs,
                lazyFftMs = 0.0,
                workerTotalMs = workerTotalMs,
                onnxModelVariant = modelVariant,
                headMotionMetrics = null,
                rawFeatures = null,
                normalizedFeatures = null,
                fusionSummary = null,
                lastSkipReason = "neural score unavailable (${onnxResult?.errorMessage ?: "model not ready"})",
            )
        }

        if (physicalMetrics == null) {
            val workerTotalMs = (System.nanoTime() - workerStartNs) / 1_000_000.0
            return WindowEvaluationSnapshot(
                evaluatedAtSec = nowSec,
                elapsedSec = elapsedSec,
                windowSec = windowSec,
                status = PipelineStatus.WAITING_FOR_VALID_WINDOW,
                neuralBufferSize = neuralSnapshot.size,
                cachedP4Count = cachedP4Count,
                latestDeltaFramesCount = deltaFramesCount,
                latestNeuralScore = neuralScore,
                onnxInferenceMs = onnxMs,
                lazyFftMs = 0.0,
                workerTotalMs = workerTotalMs,
                onnxModelVariant = modelVariant,
                headMotionMetrics = null,
                rawFeatures = null,
                normalizedFeatures = null,
                fusionSummary = null,
                lastSkipReason = "physical metrics unavailable",
            )
        }

        // Gate 1: Readiness & Coverage Gate (Section 6.1)
        val gate1SkipReason = checkGate1ReadinessAndCoverage(physicalMetrics)
        if (gate1SkipReason != null) {
            val workerTotalMs = (System.nanoTime() - workerStartNs) / 1_000_000.0
            return WindowEvaluationSnapshot(
                evaluatedAtSec = nowSec,
                elapsedSec = elapsedSec,
                windowSec = windowSec,
                status = PipelineStatus.WAITING_FOR_VALID_WINDOW,
                neuralBufferSize = neuralSnapshot.size,
                cachedP4Count = cachedP4Count,
                latestDeltaFramesCount = deltaFramesCount,
                latestNeuralScore = neuralScore,
                onnxInferenceMs = onnxMs,
                lazyFftMs = 0.0,
                workerTotalMs = workerTotalMs,
                onnxModelVariant = modelVariant,
                headMotionMetrics = null,
                rawFeatures = null,
                normalizedFeatures = null,
                fusionSummary = null,
                lastSkipReason = gate1SkipReason,
            )
        }

        // Gate 2: Lazy FFT 1Hz & Head Motion Gate (Section 4.3 & 6.1)
        val fftStartNs = System.nanoTime()
        val headMotionMetrics = if (pitchSnapshot != null) {
            LazyHeadMotionWindow.evaluateExactFft(pitchSnapshot)
        } else {
            null
        }
        val lazyFftMs = (System.nanoTime() - fftStartNs) / 1_000_000.0

        val gate2SkipReason = checkGate2HeadMotion(headMotionMetrics)
        if (gate2SkipReason != null) {
            val workerTotalMs = (System.nanoTime() - workerStartNs) / 1_000_000.0
            return WindowEvaluationSnapshot(
                evaluatedAtSec = nowSec,
                elapsedSec = elapsedSec,
                windowSec = windowSec,
                status = PipelineStatus.WAITING_FOR_VALID_WINDOW,
                neuralBufferSize = neuralSnapshot.size,
                cachedP4Count = cachedP4Count,
                latestDeltaFramesCount = deltaFramesCount,
                latestNeuralScore = neuralScore,
                onnxInferenceMs = onnxMs,
                lazyFftMs = lazyFftMs,
                workerTotalMs = workerTotalMs,
                onnxModelVariant = modelVariant,
                headMotionMetrics = headMotionMetrics,
                rawFeatures = null,
                normalizedFeatures = null,
                fusionSummary = null,
                lastSkipReason = gate2SkipReason,
            )
        }

        // Extract 8 raw features (`physical_v1`)
        val blinkDurationP90Ms = computePercentile(blinkDurationsMs, blinkDurationPercentile)
        val nodDurationP100Sec = computePercentile(nodDurationsMs, nodDurationPercentile) / FusionConfig.MS_PER_SECOND
        val perclosRatio = (physicalMetrics.perclosPct ?: 0.0) / 100.0
        val dominantFrequencyHz = headMotionMetrics?.frequency ?: 0.0

        val rawFeatures = linkedMapOf(
            FusionConfig.FEATURE_BLINK_FREQUENCY to physicalMetrics.blinkFreqPerMin,
            FusionConfig.FEATURE_BLINK_DURATION to blinkDurationP90Ms,
            FusionConfig.FEATURE_PERCLOS to perclosRatio,
            FusionConfig.FEATURE_YAWN_FREQUENCY to physicalMetrics.yawnFreqPerMin,
            FusionConfig.FEATURE_NOD_DURATION to nodDurationP100Sec,
            FusionConfig.FEATURE_NOD_FREQUENCY to physicalMetrics.nodFreqPerMin,
            FusionConfig.FEATURE_DOMINANT_HEAD_MOTION_FREQUENCY to dominantFrequencyHz,
            FusionConfig.FEATURE_NEURALNETWORK_SCORE to neuralScore.coerceIn(0.0, 1.0),
        )

        val normalizedFeatures = normalizer.transform(rawFeatures)
        val fusionSummary = fitnessEvaluator.evaluateAll(normalizedFeatures)
        val status = if (fusionSummary.isDrowsy) {
            PipelineStatus.DROWSY
        } else {
            PipelineStatus.ALERT
        }
        val workerTotalMs = (System.nanoTime() - workerStartNs) / 1_000_000.0

        return WindowEvaluationSnapshot(
            evaluatedAtSec = nowSec,
            elapsedSec = elapsedSec,
            windowSec = windowSec,
            status = status,
            neuralBufferSize = neuralSnapshot.size,
            cachedP4Count = cachedP4Count,
            latestDeltaFramesCount = deltaFramesCount,
            latestNeuralScore = neuralScore,
            onnxInferenceMs = onnxMs,
            lazyFftMs = lazyFftMs,
            workerTotalMs = workerTotalMs,
            onnxModelVariant = modelVariant,
            headMotionMetrics = headMotionMetrics,
            rawFeatures = rawFeatures,
            normalizedFeatures = normalizedFeatures,
            fusionSummary = fusionSummary,
            lastSkipReason = null,
        )
    }

    private fun checkGate1ReadinessAndCoverage(metrics: PhysicalMetrics): String? {
        if (!metrics.eyeReady) {
            return "eye HMM not ready"
        }
        if (!metrics.mouthReady) {
            return "mouth HMM not ready"
        }
        if (!metrics.headCalibrated) {
            return "head pose not calibrated"
        }
        if (metrics.perclosPct == null) {
            return "PERCLOS is null"
        }
        val eps = FusionConfig.GATE_COVERAGE_EPSILON_SEC
        val eyeObs = metrics.eyeObservedSec ?: 0.0
        if (!eyeObs.isFinite() || eyeObs < minObservedSec - eps) {
            return String.format(
                java.util.Locale.US,
                "eye coverage %.1fs < %.1fs",
                eyeObs,
                minObservedSec,
            )
        }
        val mouthObs = metrics.mouthObservedSec ?: 0.0
        if (!mouthObs.isFinite() || mouthObs < minObservedSec - eps) {
            return String.format(
                java.util.Locale.US,
                "mouth coverage %.1fs < %.1fs",
                mouthObs,
                minObservedSec,
            )
        }
        val headObs = metrics.headObservedSec ?: 0.0
        if (!headObs.isFinite() || headObs < minObservedSec - eps) {
            return String.format(
                java.util.Locale.US,
                "head coverage %.1fs < %.1fs",
                headObs,
                minObservedSec,
            )
        }
        return null
    }

    private fun checkGate2HeadMotion(headMotion: HeadMotionMetrics?): String? {
        if (headMotion == null) {
            return "head motion FFT snapshot unavailable"
        }
        val eps = FusionConfig.GATE_COVERAGE_EPSILON_SEC
        if (!headMotion.observed.isFinite() || headMotion.observed < headMotionMinObservedSec - eps) {
            return String.format(
                java.util.Locale.US,
                "head_motion.observed %.1fs < %.1fs",
                headMotion.observed,
                headMotionMinObservedSec,
            )
        }
        val freq = headMotion.frequency
        if (freq == null || !freq.isFinite() || freq < 0.0) {
            return "head_motion.frequency is null, non-finite, or negative"
        }
        val res = headMotion.resolution
        if (res == null || !res.isFinite() || res <= 0.0) {
            return "head_motion.resolution is null, non-finite, or non-positive"
        }
        return null
    }

    companion object {
        /**
         * Computes linear-interpolated percentile matching `numpy.percentile(..., method="linear")`.
         * Returns `0.0` if [values] is empty.
         */
        fun computePercentile(values: DoubleArray, percentile: Double): Double {
            if (values.isEmpty()) return 0.0
            val sorted = values.sortedArray()
            if (sorted.size == 1) return sorted[0]
            val p = percentile.coerceIn(0.0, 100.0) / 100.0
            val virtualIndex = p * (sorted.size - 1)
            val lowerIndex = floor(virtualIndex).toInt().coerceIn(0, sorted.lastIndex)
            val upperIndex = (lowerIndex + 1).coerceAtMost(sorted.lastIndex)
            val fraction = virtualIndex - lowerIndex
            return sorted[lowerIndex] + fraction * (sorted[upperIndex] - sorted[lowerIndex])
        }
    }
}
