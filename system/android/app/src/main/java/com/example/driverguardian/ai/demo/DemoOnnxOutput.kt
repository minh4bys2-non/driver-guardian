package com.example.driverguardian.ai.demo

/**
 * Encapsulates the output of the demo end-to-end ONNX model.
 *
 * Explicitly labeled: DEMO ONNX OUTPUT.
 * This is NOT certified driver state and MUST NOT be connected to safety alerts or backend events.
 */
data class DemoOnnxOutput(
    val alertProbability: Float,
    val drowsyProbability: Float,
    val inferenceMs: Double,
    val sequenceFrames: Int,
    val modelLoaded: Boolean,
    val rawAlertLogit: Float,
    val rawDrowsyLogit: Float,
    val label: String = "DEMO ONNX OUTPUT",
    val timestampMs: Long = System.currentTimeMillis(),
) {
    companion object {
        /**
         * Numerically stable 2-class softmax.
         */
        fun computeSoftmax(alertLogit: Float, drowsyLogit: Float): Pair<Float, Float> {
            val maxLogit = maxOf(alertLogit, drowsyLogit)
            val exp0 = kotlin.math.exp(alertLogit - maxLogit)
            val exp1 = kotlin.math.exp(drowsyLogit - maxLogit)
            val sum = exp0 + exp1
            return (exp0 / sum) to (exp1 / sum)
        }
    }
}
