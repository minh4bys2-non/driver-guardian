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

        var weightSum = 0.0
        for (i in 0 until f) {
            weightSum += candidate[i]
        }
        if (weightSum <= FusionConfig.FITNESS_WEIGHT_EPSILON) {
            return 0.0
        }

        return when (method) {
            FusionConfig.FUSION_WEIGHTED_MEAN -> {
                var weightedSum = 0.0
                for (i in 0 until f) {
                    weightedSum += x[i] * candidate[i]
                }
                weightedSum / max(weightSum, FusionConfig.FITNESS_WEIGHT_EPSILON)
            }

            FusionConfig.FUSION_NOISY_OR -> {
                var survivalProduct = 1.0
                for (i in 0 until f) {
                    val clampedTerm = (1.0 - candidate[i] * x[i]).coerceIn(
                        FusionConfig.NORMALIZER_LOWER_BOUND,
                        FusionConfig.NORMALIZER_UPPER_CLIP,
                    )
                    survivalProduct *= clampedTerm
                }
                1.0 - survivalProduct
            }

            FusionConfig.FUSION_COOCCURRENCE -> {
                var pairInteractionSum = 0.0
                var pairWeightSum = 0.0
                for (i in 0 until f) {
                    for (j in (i + 1) until f) {
                        val pairWeight = FusionConfig.COOCCURRENCE_PAIR_WEIGHT_FACTOR * (candidate[i] + candidate[j])
                        pairInteractionSum += (x[i] * x[j]) * pairWeight
                        pairWeightSum += pairWeight
                    }
                }
                if (pairWeightSum <= FusionConfig.FITNESS_WEIGHT_EPSILON) {
                    0.0
                } else {
                    pairInteractionSum / max(pairWeightSum, FusionConfig.FITNESS_WEIGHT_EPSILON)
                }
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
