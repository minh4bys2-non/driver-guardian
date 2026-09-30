package com.example.driverguardian.ai.runtime

sealed class RuntimeError(open val detail: String, open val cause: Throwable? = null) {
    data class ModelNotFound(val assetPath: String) : RuntimeError("Model not found\nExpected path: assets/$assetPath")
    data class ModelLoadFailure(override val detail: String, override val cause: Throwable? = null) : RuntimeError(detail, cause)
    data class InvalidModel(override val detail: String, override val cause: Throwable? = null) : RuntimeError(detail, cause)
    data class SessionCreationFailure(override val detail: String, override val cause: Throwable? = null) : RuntimeError(detail, cause)
    data class UnsupportedTensorType(val inputName: String, val tensorType: RuntimeTensorType) : RuntimeError("Input '$inputName' uses unsupported type $tensorType")
    data class DynamicShapeRequiresConcreteDimensions(val inputName: String, val shape: List<Long>) : RuntimeError("Input '$inputName' has dynamic shape $shape and needs concrete dimensions")
    data class MissingInput(val inputName: String) : RuntimeError("Missing input '$inputName'")
    data class UnexpectedInput(val inputName: String) : RuntimeError("Unexpected input '$inputName'")
    data class TensorShapeMismatch(val inputName: String, val expected: List<Long>, val actual: List<Long>) : RuntimeError("Input '$inputName' expected shape $expected but received $actual")
    data class ContractMismatch(override val detail: String) : RuntimeError(detail)
    data class InferenceFailure(override val detail: String, override val cause: Throwable? = null) : RuntimeError(detail, cause)
    data object ClosedRuntime : RuntimeError("The ONNX runtime is closed")
}

sealed interface RuntimeResult<out T> {
    data class Success<T>(val value: T) : RuntimeResult<T>
    data class Failure(val error: RuntimeError) : RuntimeResult<Nothing>
}
