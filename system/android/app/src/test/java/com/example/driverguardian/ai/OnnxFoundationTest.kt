package com.example.driverguardian.ai

import com.example.driverguardian.ai.contract.drowsiness.*
import com.example.driverguardian.ai.parity.*
import com.example.driverguardian.ai.runtime.*
import com.example.driverguardian.ai.tensor.DummyTensorFactory
import com.example.driverguardian.ai.tensor.TensorPreviewFactory
import com.example.driverguardian.ai.tensor.TensorShape
import org.junit.Assert.*
import org.junit.Test

class OnnxFoundationTest {
    @Test fun elementCountChecksOverflowAndInvalidDimensions() {
        assertEquals(24L, TensorShape.checkedElementCount(listOf(2, 3, 4)))
        assertNull(TensorShape.checkedElementCount(listOf(1, -1, 4)))
        assertNull(TensorShape.checkedElementCount(listOf(Long.MAX_VALUE, 2)))
    }

    @Test fun dynamicInputIsRejectedWithoutConcreteOverride() {
        val result = DummyTensorFactory.create(model(listOf(tensor("x", listOf(1, -1), RuntimeTensorType.FLOAT))))
        assertTrue((result as RuntimeResult.Failure).error is RuntimeError.DynamicShapeRequiresConcreteDimensions)
    }

    @Test fun unsupportedDummyTypeIsRejected() {
        val result = DummyTensorFactory.create(model(listOf(tensor("x", listOf(1, 2), RuntimeTensorType.INT64))))
        assertTrue((result as RuntimeResult.Failure).error is RuntimeError.UnsupportedTensorType)
    }

    @Test fun contractKeepsSimilarImageInputsSemanticallyUnresolved() {
        val inputs = listOf(
            tensor("a", listOf(1, 30, 3, 64, 64)), tensor("b", listOf(1, 30, 3, 64, 64)),
            tensor("c", listOf(1, 30, 3, 64, 64)), tensor("d", listOf(1, 30, 10)),
        )
        val result = DrowsinessModelContract().validate(model(inputs, listOf(tensor("score", listOf(1, 1)))))
        assertEquals(MetadataStatus.VALID, result.metadataStatus)
        assertEquals(SemanticStatus.UNRESOLVED, result.semanticStatus)
    }

    @Test fun numericalToleranceProducesPassAndFail() {
        val case = GoldenTestCase("case", emptyMap(), mapOf("score" to floatArrayOf(1f, 2f)))
        val close = RuntimeInferenceResult(1, listOf(output(floatArrayOf(1.0001f, 1.9999f))))
        val far = RuntimeInferenceResult(1, listOf(output(floatArrayOf(1.1f, 2f))))
        val tolerance = NumericalTolerance.Configured(0.001, 0.0)
        assertEquals(ParityStatus.PASS, GoldenVectorRunner.compare(case, close, tolerance).status)
        assertEquals(ParityStatus.FAIL, GoldenVectorRunner.compare(case, far, tolerance).status)
        assertEquals(ParityStatus.NOT_RUN, GoldenVectorRunner.compare(case, close, NumericalTolerance.Unconfigured).status)
    }

    @Test fun previewIsTruncatedAtRequestedLimit() {
        val preview = TensorPreviewFactory.fromFloats(floatArrayOf(1f, 2f, 3f), 2)
        assertEquals(listOf("1.0", "2.0"), preview.values)
        assertEquals(3L, preview.totalElementCount)
        assertTrue(preview.truncated)
    }

    private fun tensor(name: String, shape: List<Long>, type: RuntimeTensorType = RuntimeTensorType.FLOAT) =
        TensorMetadata(name, type, "ONNX_TENSOR_ELEMENT_DATA_TYPE_${type.name}", shape)

    private fun model(inputs: List<TensorMetadata>, outputs: List<TensorMetadata> = listOf(tensor("out", listOf(1)))) =
        ModelMetadata("test.onnx", inputs, outputs, 0)

    private fun output(values: FloatArray) = RuntimeOutput(
        "score", RuntimeTensorType.FLOAT, listOf(values.size.toLong()), TensorPreviewFactory.fromFloats(values), values,
    )
}
