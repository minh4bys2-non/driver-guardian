package com.example.driverguardian.ai.fusion

import kotlin.math.min

/**
 * Normalizes the 8 raw window features (`physical_v1`) into [0.0, 1.0] using upper-bound clipping.
 * Exact mathematical parity with `PhysicalNormalizer` in `drowsy_detection/fusion/normalization.py`.
 */
class PhysicalNormalizer(
    val scales: Map<String, Double> = FusionConfig.DEFAULT_SCALES,
    val featureOrder: List<String> = FusionConfig.FEATURES,
) {
    init {
        for (name in featureOrder) {
            val scale = scales[name]
            require(scale != null && scale.isFinite() && scale > FusionConfig.NORMALIZER_LOWER_BOUND) {
                "Scale for feature '$name' must be finite and positive, got $scale"
            }
        }
    }

    /**
     * Transforms raw feature map into normalized [0.0, 1.0] feature map.
     * Throws [IllegalArgumentException] if any required feature is missing, NaN/Infinite, or negative.
     */
    fun transform(raw: Map<String, Double>): Map<String, Double> {
        val normalized = LinkedHashMap<String, Double>(featureOrder.size)
        for (name in featureOrder) {
            val value = raw[name]
                ?: throw IllegalArgumentException("Missing required raw feature '$name'")
            require(value.isFinite() && value >= FusionConfig.NORMALIZER_LOWER_BOUND) {
                "Feature '$name' must be finite and non-negative, got $value"
            }
            val upper = scales.getValue(name)
            normalized[name] = min(value / upper, FusionConfig.NORMALIZER_UPPER_CLIP)
        }
        return normalized
    }

    /**
     * Converts normalized map into an ordered 8D feature vector matching [featureOrder].
     */
    fun toOrderedVector(normalized: Map<String, Double>): DoubleArray {
        return DoubleArray(featureOrder.size) { index ->
            val name = featureOrder[index]
            normalized[name]
                ?: throw IllegalArgumentException("Missing normalized feature '$name'")
        }
    }
}
