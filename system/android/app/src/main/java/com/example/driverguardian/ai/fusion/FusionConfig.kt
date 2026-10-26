package com.example.driverguardian.ai.fusion

import kotlin.math.ceil
import kotlin.math.max

/**
 * Concentrated configuration for the Fusion & Normalization branch (`ai.fusion`).
 * Strictly follows `snapdragon_680_v2.md` Section 5 & Section 6 without hardcoding in processors.
 */
object FusionConfig {

    const val FEATURE_ENCODING = "physical_v1"

    const val FEATURE_BLINK_FREQUENCY = "blink_frequency"
    const val FEATURE_BLINK_DURATION = "blink_duration"
    const val FEATURE_PERCLOS = "perclos"
    const val FEATURE_YAWN_FREQUENCY = "yawn_frequency"
    const val FEATURE_NOD_DURATION = "nod_duration"
    const val FEATURE_NOD_FREQUENCY = "nod_frequency"
    const val FEATURE_DOMINANT_HEAD_MOTION_FREQUENCY = "dominant_head_motion_frequency"
    const val FEATURE_NEURALNETWORK_SCORE = "neuralnetwork_score"

    val FEATURES: List<String> = listOf(
        FEATURE_BLINK_FREQUENCY,
        FEATURE_BLINK_DURATION,
        FEATURE_PERCLOS,
        FEATURE_YAWN_FREQUENCY,
        FEATURE_NOD_DURATION,
        FEATURE_NOD_FREQUENCY,
        FEATURE_DOMINANT_HEAD_MOTION_FREQUENCY,
        FEATURE_NEURALNETWORK_SCORE,
    )

    const val FUSION_WEIGHTED_MEAN = "weighted_mean"
    const val FUSION_NOISY_OR = "noisy_or"
    const val FUSION_COOCCURRENCE = "cooccurrence"

    val FUSION_METHODS: List<String> = listOf(
        FUSION_WEIGHTED_MEAN,
        FUSION_NOISY_OR,
        FUSION_COOCCURRENCE,
    )

    // Window & Gate parameters (Section 5.1)
    const val DEFAULT_WINDOW_SEC = 60.0
    const val DEFAULT_MIN_OBSERVED_RATIO = 0.8
    const val DEFAULT_HEAD_MOTION_MIN_OBSERVED_SEC = 60.0
    const val DEFAULT_BLINK_DURATION_PERCENTILE = 90.0
    const val DEFAULT_NOD_DURATION_PERCENTILE = 100.0
    const val DEFAULT_PRIMARY_FUSION = FUSION_COOCCURRENCE
    const val DEFAULT_CANDIDATE_SELECTION_THRESHOLD = 0.5
    const val MS_PER_SECOND = 1000.0

    // Numerical constants for Normalization & Fitness (Section 6.2 & 6.3)
    const val NORMALIZER_UPPER_CLIP = 1.0
    const val NORMALIZER_LOWER_BOUND = 0.0
    const val RATE_BOUND_DURATION_EPSILON_MS = 1e-6
    const val RATE_BOUND_SECONDS_PER_MINUTE = 60.0
    const val DEFAULT_NORMALIZER_FPS = 30.0
    const val FITNESS_WEIGHT_EPSILON = 1e-12
    const val COOCCURRENCE_PAIR_WEIGHT_FACTOR = 0.5

    // Default Upper Bounds (`physical_v1` at 30 FPS, 60s window - Section 6.2)
    const val SCALE_BLINK_FREQUENCY = 602.0
    const val SCALE_BLINK_DURATION_MS = 2000.0
    const val SCALE_PERCLOS = 1.0
    const val SCALE_YAWN_FREQUENCY = 17.0
    const val SCALE_NOD_DURATION_SEC = 3.5
    const val SCALE_NOD_FREQUENCY = 77.0
    const val SCALE_DOMINANT_HEAD_MOTION_FREQUENCY_HZ = 15.0
    const val SCALE_NEURALNETWORK_SCORE = 1.0

    val DEFAULT_SCALES: Map<String, Double> = mapOf(
        FEATURE_BLINK_FREQUENCY to SCALE_BLINK_FREQUENCY,
        FEATURE_BLINK_DURATION to SCALE_BLINK_DURATION_MS,
        FEATURE_PERCLOS to SCALE_PERCLOS,
        FEATURE_YAWN_FREQUENCY to SCALE_YAWN_FREQUENCY,
        FEATURE_NOD_DURATION to SCALE_NOD_DURATION_SEC,
        FEATURE_NOD_FREQUENCY to SCALE_NOD_FREQUENCY,
        FEATURE_DOMINANT_HEAD_MOTION_FREQUENCY to SCALE_DOMINANT_HEAD_MOTION_FREQUENCY_HZ,
        FEATURE_NEURALNETWORK_SCORE to SCALE_NEURALNETWORK_SCORE,
    )

    // Default 9D candidate vectors: 8 weights + 1 decision threshold (Section 5.1 & 6.3)
    val DEFAULT_CANDIDATE_VECTOR: DoubleArray = doubleArrayOf(
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        DEFAULT_CANDIDATE_SELECTION_THRESHOLD,
    )

    val DEFAULT_FUSION_CANDIDATES: Map<String, DoubleArray> = mapOf(
        FUSION_WEIGHTED_MEAN to DEFAULT_CANDIDATE_VECTOR.copyOf(),
        FUSION_NOISY_OR to DEFAULT_CANDIDATE_VECTOR.copyOf(),
        FUSION_COOCCURRENCE to DEFAULT_CANDIDATE_VECTOR.copyOf(),
    )

    /**
     * Exact parity with `rate_bound(fsm)` in `drowsy_detection/fusion/normalization.py`.
     */
    fun computeRateBound(
        minimumDurationMs: Double,
        windowSec: Double = DEFAULT_WINDOW_SEC,
        fps: Double = DEFAULT_NORMALIZER_FPS,
    ): Double {
        val durationSec = max(
            (minimumDurationMs - RATE_BOUND_DURATION_EPSILON_MS) / MS_PER_SECOND,
            1.0 / fps,
        )
        return (ceil(windowSec / durationSec) + 1.0) * RATE_BOUND_SECONDS_PER_MINUTE / windowSec
    }
}
