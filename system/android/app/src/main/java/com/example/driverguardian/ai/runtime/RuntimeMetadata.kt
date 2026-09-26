package com.example.driverguardian.ai.runtime

enum class RuntimeTensorType { FLOAT, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, STRING, FLOAT16, BFLOAT16, UNKNOWN }

data class TensorMetadata(
    val name: String,
    val type: RuntimeTensorType,
    val onnxType: String,
    val shape: List<Long>,
    val dimensionNames: List<String?> = emptyList(),
) {
    val rank: Int get() = shape.size
    val dynamicDimensions: List<Int> get() = shape.indices.filter { shape[it] <= 0L || dimensionNames.getOrNull(it) != null }
    val hasDynamicDimensions: Boolean get() = dynamicDimensions.isNotEmpty()
}

data class ModelMetadata(val modelName: String, val inputs: List<TensorMetadata>, val outputs: List<TensorMetadata>, val loadDurationNanos: Long) {
    val inputCount: Int get() = inputs.size
    val outputCount: Int get() = outputs.size
}
data class TensorPreview(val values: List<String>, val totalElementCount: Long?, val truncated: Boolean)
data class RuntimeOutput(val name: String, val type: RuntimeTensorType, val shape: List<Long>, val preview: TensorPreview, val floatValues: FloatArray? = null)
data class RuntimeInferenceResult(val durationNanos: Long, val outputs: List<RuntimeOutput>)
data class RuntimeDiagnostics(
    val runtimeVersion: String,
    val modelName: String? = null,
    val loadDurationNanos: Long? = null,
    val lastInferenceDurationNanos: Long? = null,
    val runCount: Long = 0,
    val lastError: RuntimeError? = null,
)
