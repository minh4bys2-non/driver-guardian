package com.example.driverguardian.ai.parity

sealed interface NumericalTolerance {
    data object Unconfigured : NumericalTolerance
    data class Configured(val absolute: Double, val relative: Double) : NumericalTolerance {
        init { require(absolute.isFinite() && relative.isFinite() && absolute >= 0.0 && relative >= 0.0) }
    }
}

enum class ParityStatus { PASS, FAIL, NOT_RUN, ERROR }
data class OutputComparison(
    val outputName: String,
    val allClose: Boolean,
    val maximumAbsoluteError: Double,
    val maximumRelativeError: Double,
    val mismatchCount: Int,
)
data class ParityResult(val status: ParityStatus, val comparisons: List<OutputComparison> = emptyList(), val message: String? = null)
