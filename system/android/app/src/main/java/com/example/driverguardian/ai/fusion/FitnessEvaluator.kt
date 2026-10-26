package com.example.driverguardian.ai.fusion

import kotlin.math.max

/**
 * Result of a single fusion method (`weighted_mean`, `noisy_or`, or `cooccurrence`).
 */
data class FusionMethodDetail(
    val method: String,
    val score: Double,
    val threshold: Double,
    val prediction: Int,
) {
    val isDrowsy: Boolean
        get() = prediction == 1
}

/**
 * Aggregated fusion evaluation across all 3 methods plus the primary fusion method selection.
 */
data class FusionEvaluationSummary(
    val primaryFusion: String,
    val primaryScore: Double,
    val primaryThreshold: Double,
    val primaryPrediction: Int,
    val fusionDetails: Map<String, FusionMethodDetail>,
) {
    val isDrowsy: Boolean
        get() = primaryPrediction == 1
}

/**
 * Implements the 3 decision fusion methods (`weighted_mean`, `noisy_or`, `cooccurrence`)
 * with exact arithmetic parity to `Fitness` in `drowsy_detection/fusion/fitness.py`.
 */
class FitnessEvaluator(
    private val fusionCandidates: Map<String, DoubleArray> = FusionConfig.DEFAULT_FUSION_CANDIDATES,
    private val primaryFusion: String = FusionConfig.DEFAULT_PRIMARY_FUSION,
    private val featureOrder: List<String> = FusionConfig.FEATURES,
) {
    /**
     * Computes the fusion score in [0.0, 1.0] for a single normalized feature vector [x] (length F)
     * and candidate vector [candidate] (length F + 1, where last element is the decision threshold).
     */
    fun score(
        x: DoubleArray,
        candidate: DoubleArray,
        method: String = primaryFusion,
    ): Double {
        val f = x.size
        require(candidate.size == f + 1) {
            "Candidate vector length (${candidate.size}) must equal feature length + 1 (${f + 1})"
        }
        for (i in 0 until f) {
            val value = x[i]
            require(
                value.isFinite() &&
                    value >= FusionConfig.NORMALIZER_LOWER_BOUND &&
                    value <= FusionConfig.NORMALIZER_UPPER_CLIP,
            ) {
                "Normalized feature at index $i must be in [0.0, 1.0], got $value"
            }
        }
        for (i in candidate.indices) {
            val weightOrThreshold = candidate[i]
            require(
                weightOrThreshold.isFinite() &&
                    weightOrThreshold >= FusionConfig.NORMALIZER_LOWER_BOUND &&
                    weightOrThreshold <= FusionConfig.NORMALIZER_UPPER_CLIP,
            ) {
                "Candidate parameter at index $i must be in [0.0, 1.0], got $weightOrThreshold"
            }
        }

        // Selection mask M = { i in 0 until f | w[i] >= 0.5 } matching snapdragon_680_v2.md Section 6.3 & fitnessFunction.py
        val selectedIndices = IntArray(f)
        var selectedCount = 0
        for (i in 0 until f) {
            if (candidate[i] >= FusionConfig.DEFAULT_CANDIDATE_SELECTION_THRESHOLD) {
                selectedIndices[selectedCount++] = i
            }
        }
        if (selectedCount == 0) {
            return 0.0
        }

        return when (method) {
            FusionConfig.FUSION_WEIGHTED_MEAN -> {
                var zSum = 0.0
                var wSum = 0.0
                for (k in 0 until selectedCount) {
                    val idx = selectedIndices[k]
                    val w = candidate[idx]
                    zSum += x[idx] * w
                    wSum += w
                }
                if (wSum <= FusionConfig.FITNESS_WEIGHT_EPSILON) 0.0 else zSum / max(wSum, FusionConfig.FITNESS_WEIGHT_EPSILON)
            }

            FusionConfig.FUSION_NOISY_OR -> {
                var survivalProduct = 1.0
                for (k in 0 until selectedCount) {
                    val idx = selectedIndices[k]
                    val z = (x[idx] * candidate[idx]).coerceIn(
                        FusionConfig.NORMALIZER_LOWER_BOUND,
                        FusionConfig.NORMALIZER_UPPER_CLIP,
                    )
                    survivalProduct *= (1.0 - z)
                }
                1.0 - survivalProduct
            }

            FusionConfig.FUSION_COOCCURRENCE -> {
                // z_i = x_i * w_i for i in M.
                // s1 = max(z), s2 = second_max(z) (or 0.0 if |M| == 1).
                // score = s1 + (1.0 - s1) * (s1 * s2)
                var strongest = 0.0
                var secondStrongest = 0.0
                for (k in 0 until selectedCount) {
                    val idx = selectedIndices[k]
                    val z = x[idx] * candidate[idx]
                    if (z >= strongest) {
                        secondStrongest = strongest
                        strongest = z
                    } else if (z > secondStrongest) {
                        secondStrongest = z
                    }
                }
                val joint = if (selectedCount > 1) strongest * secondStrongest else 0.0
                strongest + (1.0 - strongest) * joint
            }

            else -> throw IllegalArgumentException("Unsupported fusion method: '$method'")
        }
    }

    /**
     * Evaluates binary prediction (`1` = DROWSY, `0` = ALERT) using `score >= threshold`.
     */
    fun predict(
        x: DoubleArray,
        candidate: DoubleArray,
        method: String = primaryFusion,
    ): Int {
        val threshold = candidate.last()
        val s = score(x = x, candidate = candidate, method = method)
        return if (s >= threshold) 1 else 0
    }

    /**
     * Evaluates all configured fusion methods (`weighted_mean`, `noisy_or`, `cooccurrence`)
     * on the normalized feature map and returns a [FusionEvaluationSummary].
     */
    fun evaluateAll(normalizedFeatures: Map<String, Double>): FusionEvaluationSummary {
        val x = DoubleArray(featureOrder.size) { idx ->
            val key = featureOrder[idx]
            normalizedFeatures[key]
                ?: throw IllegalArgumentException("Missing normalized feature '$key'")
        }
        val details = LinkedHashMap<String, FusionMethodDetail>(FusionConfig.FUSION_METHODS.size)
        for (method in FusionConfig.FUSION_METHODS) {
            val candidate = fusionCandidates[method] ?: FusionConfig.DEFAULT_CANDIDATE_VECTOR
            val threshold = candidate.last()
            val methodScore = score(x = x, candidate = candidate, method = method)
            val methodPrediction = if (methodScore >= threshold) 1 else 0
            details[method] = FusionMethodDetail(
                method = method,
                score = methodScore,
                threshold = threshold,
                prediction = methodPrediction,
            )
        }
        val primaryDetail = details[primaryFusion]
            ?: details.values.first()
        return FusionEvaluationSummary(
            primaryFusion = primaryDetail.method,
            primaryScore = primaryDetail.score,
            primaryThreshold = primaryDetail.threshold,
            primaryPrediction = primaryDetail.prediction,
            fusionDetails = details,
        )
    }
}
