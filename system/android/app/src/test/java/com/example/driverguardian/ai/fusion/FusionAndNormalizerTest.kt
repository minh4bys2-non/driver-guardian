package com.example.driverguardian.ai.fusion

import com.example.driverguardian.ai.neural.NeuralClassifier
import com.example.driverguardian.ai.neural.NeuralWindowSnapshot
import com.example.driverguardian.ai.neural.OnnxPredictionResult
import com.example.driverguardian.ai.physical.model.PhysicalMetrics
import com.example.driverguardian.ai.physical.temporal.PitchSamplesSnapshot
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin

class FusionAndNormalizerTest {

    @Test
    fun `PhysicalNormalizer scales and clips 8 features into 0 to 1`() {
        val normalizer = PhysicalNormalizer()
        val raw = mapOf(
            FusionConfig.FEATURE_BLINK_FREQUENCY to 301.0, // 301 / 602 = 0.5
            FusionConfig.FEATURE_BLINK_DURATION to 1000.0, // 1000 / 2000 = 0.5
            FusionConfig.FEATURE_PERCLOS to 0.25, // 0.25 / 1.0 = 0.25
            FusionConfig.FEATURE_YAWN_FREQUENCY to 34.0, // 34 / 17 = 2.0 -> clipped to 1.0
            FusionConfig.FEATURE_NOD_DURATION to 1.75, // 1.75 / 3.5 = 0.5
            FusionConfig.FEATURE_NOD_FREQUENCY to 77.0, // 77 / 77 = 1.0
            FusionConfig.FEATURE_DOMINANT_HEAD_MOTION_FREQUENCY to 3.0, // 3 / 15 = 0.2
            FusionConfig.FEATURE_NEURALNETWORK_SCORE to 0.8, // 0.8 / 1.0 = 0.8
        )

        val norm = normalizer.transform(raw)
        assertEquals(0.5, norm.getValue(FusionConfig.FEATURE_BLINK_FREQUENCY), 1e-9)
        assertEquals(0.5, norm.getValue(FusionConfig.FEATURE_BLINK_DURATION), 1e-9)
        assertEquals(0.25, norm.getValue(FusionConfig.FEATURE_PERCLOS), 1e-9)
        assertEquals(1.0, norm.getValue(FusionConfig.FEATURE_YAWN_FREQUENCY), 1e-9)
        assertEquals(0.5, norm.getValue(FusionConfig.FEATURE_NOD_DURATION), 1e-9)
        assertEquals(1.0, norm.getValue(FusionConfig.FEATURE_NOD_FREQUENCY), 1e-9)
        assertEquals(0.2, norm.getValue(FusionConfig.FEATURE_DOMINANT_HEAD_MOTION_FREQUENCY), 1e-9)
        assertEquals(0.8, norm.getValue(FusionConfig.FEATURE_NEURALNETWORK_SCORE), 1e-9)
    }

    @Test
    fun `FitnessEvaluator computes exact weighted_mean, noisy_or, and cooccurrence`() {
        val evaluator = FitnessEvaluator()
        val x = doubleArrayOf(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        val candidate = doubleArrayOf(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5)

        val wm = evaluator.score(x, candidate, FusionConfig.FUSION_WEIGHTED_MEAN)
        val no = evaluator.score(x, candidate, FusionConfig.FUSION_NOISY_OR)
        val co = evaluator.score(x, candidate, FusionConfig.FUSION_COOCCURRENCE)

        // All x_i = 0.5, w_i = 1.0:
        // weighted_mean = 0.5
        // noisy_or = 1 - (1 - 0.5)^8 = 1 - 1/256 = 0.99609375
        // cooccurrence = each pair x_i * x_j = 0.25 -> weighted average of pairs = 0.25
        assertEquals(0.5, wm, 1e-9)
        assertEquals(1.0 - (1.0 / 256.0), no, 1e-9)
        assertEquals(0.25, co, 1e-9)
        assertEquals(1, evaluator.predict(x, candidate, FusionConfig.FUSION_WEIGHTED_MEAN))
        assertEquals(1, evaluator.predict(x, candidate, FusionConfig.FUSION_NOISY_OR))
        assertEquals(0, evaluator.predict(x, candidate, FusionConfig.FUSION_COOCCURRENCE))
    }

    @Test
    fun `WindowEvaluationEngine updates p4_old during warmup and evaluates gates post 60s`() {
        var classifierCalls = 0
        val fakeClassifier = object : NeuralClassifier {
            override val modelVariant: String = "drowsy_classifier_int8.onnx"
            override var cachedFrameCount: Int = 0
            override fun predictAndUpdateCache(snapshot: NeuralWindowSnapshot): OnnxPredictionResult {
                classifierCalls++
                cachedFrameCount = snapshot.size
                return OnnxPredictionResult(
                    probDrowsy = 0.9,
                    tNewFrames = 5,
                    tOldFrames = (snapshot.size - 5).coerceAtLeast(0),
                    cachedFrameCount = snapshot.size,
                    latencyMs = 12.5,
                    modelVariant = modelVariant,
                )
            }
            override fun resetCache() {
                cachedFrameCount = 0
            }
            override fun close() = Unit
        }

        val engine = WindowEvaluationEngine()
        val dummyNeuralSnap = NeuralWindowSnapshot(
            generation = 1L,
            latestSeqId = 50L,
            frames = List(50) { ByteArray(10) },
        )

        // 1. During WARMING_UP (elapsed = 30.0s < 60.0s), p4_old is STILL warmed up!
        val warmupEval = engine.evaluateWindow(
            nowSec = 30.0,
            elapsedSec = 30.0,
            physicalMetrics = null,
            blinkDurationsMs = doubleArrayOf(),
            nodDurationsMs = doubleArrayOf(),
            pitchSnapshot = null,
            neuralSnapshot = dummyNeuralSnap,
            neuralClassifier = fakeClassifier,
        )
        assertEquals(PipelineStatus.WARMING_UP, warmupEval.status)
        assertEquals(1, classifierCalls)
        assertEquals(50, warmupEval.cachedP4Count)
        assertNull(warmupEval.fusionSummary)

        // 2. Post-60s with full coverage (>= 48s) and valid 60s pitch FFT
        val fps = 30.0
        val nSamples = 1801
        val ts = DoubleArray(nSamples) { it / fps }
        val vs = DoubleArray(nSamples) { 4.0 * sin(2.0 * PI * 1.0 * ts[it]) }
        val pitchSnap = PitchSamplesSnapshot(
            times = ts,
            values = vs,
            timestamp = 60.0,
            windowSec = 60.0,
            minAmplitudeDeg = 2.0,
        )
        val readyPhysical = PhysicalMetrics(
            timestampSec = 61.0,
            faceDetected = true,
            eyeReady = true,
            mouthReady = true,
            headReady = true,
            headCalibrated = true,
            ear = 0.22,
            eyeState = 0,
            perclosPct = 20.0,
            p80EarThreshold = 0.25,
            blinkRatePerMin = 15.0,
            blinkDetected = false,
            eyeClosureDurationMs = 0.0,
            eyeObservedSec = 55.0,
            blinkCount = 15,
            blinkDurationMs = 200.0,
            mar = 0.15,
            mouthState = 0,
            pomPct = 5.0,
            yawnDetected = false,
            yawningFrequencyPerMin = 2.0,
            mouthOpenDurationMs = 0.0,
            mouthObservedSec = 55.0,
            yawnCount = 2,
            yawnDurationMs = 4500.0,
            pitchDeg = 1.0,
            yawDeg = 0.5,
            rollDeg = -0.5,
            overAngle = false,
            overAnglePct = 2.0,
            nodding = false,
            nodDetected = false,
            nodDurationMs = 0.0,
            noddingFrequencyPerMin = 1.0,
            headObservedSec = 55.0,
            nodCount = 1,
            nodEventDurationMs = 1200.0,
            headMotionObservedSec = 60.0,
        )

        val activeEval = engine.evaluateWindow(
            nowSec = 61.0,
            elapsedSec = 61.0,
            physicalMetrics = readyPhysical,
            blinkDurationsMs = doubleArrayOf(150.0, 200.0, 300.0),
            nodDurationsMs = doubleArrayOf(1200.0),
            pitchSnapshot = pitchSnap,
            neuralSnapshot = dummyNeuralSnap,
            neuralClassifier = fakeClassifier,
        )
        assertTrue(activeEval.status == PipelineStatus.ALERT || activeEval.status == PipelineStatus.DROWSY)
        assertNotNull(activeEval.rawFeatures)
        assertNotNull(activeEval.normalizedFeatures)
        assertNotNull(activeEval.fusionSummary)
        assertEquals(3, activeEval.fusionSummary!!.fusionDetails.size)
    }
}
