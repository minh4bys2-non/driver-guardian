package com.example.driverguardian.ai.parity

import com.example.driverguardian.ai.runtime.RuntimeInferenceResult

/** Compares copied float outputs with explicit absolute and relative tolerances. */
object GoldenVectorRunner {
    fun compare(testCase: GoldenTestCase, actual: RuntimeInferenceResult, tolerance: NumericalTolerance): ParityResult {
        if (tolerance !is NumericalTolerance.Configured) return ParityResult(ParityStatus.NOT_RUN, message = "Numerical tolerance is unconfigured.")
        if (testCase.expectedOutputs.isEmpty()) return ParityResult(ParityStatus.ERROR, message = "No expected outputs are configured.")
        val outputs = actual.outputs.associateBy { it.name }
        if (outputs.size != actual.outputs.size) return ParityResult(ParityStatus.ERROR, message = "Actual output names are duplicated.")
        val comparisons = mutableListOf<OutputComparison>()
        for ((name, expected) in testCase.expectedOutputs) {
            val values = outputs[name]?.floatValues ?: return ParityResult(ParityStatus.ERROR, comparisons, "Float output '$name' is unavailable.")
            if (values.size != expected.size) return ParityResult(ParityStatus.FAIL, comparisons, "Output '$name' size differs: expected ${expected.size}, actual ${values.size}.")
            var maxAbs = 0.0
            var maxRel = 0.0
            var mismatch = 0
            expected.indices.forEach { index ->
                val e = expected[index].toDouble()
                val a = values[index].toDouble()
                val absolute = kotlin.math.abs(a - e)
                val relative = if (e == 0.0) if (absolute == 0.0) 0.0 else Double.POSITIVE_INFINITY else absolute / kotlin.math.abs(e)
                maxAbs = maxOf(maxAbs, absolute)
                maxRel = maxOf(maxRel, relative)
                if (!a.isFinite() || !e.isFinite() || absolute > tolerance.absolute + tolerance.relative * kotlin.math.abs(e)) mismatch++
            }
            comparisons += OutputComparison(name, mismatch == 0, maxAbs, maxRel, mismatch)
        }
        return ParityResult(if (comparisons.all { it.allClose }) ParityStatus.PASS else ParityStatus.FAIL, comparisons)
    }
}
