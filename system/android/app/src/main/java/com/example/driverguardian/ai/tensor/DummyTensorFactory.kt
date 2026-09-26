package com.example.driverguardian.ai.tensor

import com.example.driverguardian.ai.runtime.*

object DummyTensorFactory {
    const val MAX_ELEMENTS_PER_INPUT = 16_777_216L
    const val MAX_TOTAL_ELEMENTS = 33_554_432L

    fun create(metadata: ModelMetadata, overrides: ConcreteShapeOverrides = ConcreteShapeOverrides()): RuntimeResult<Map<String, TensorData>> {
        val result = linkedMapOf<String, TensorData>()
        var total = 0L
        for (input in metadata.inputs) {
            if (input.type != RuntimeTensorType.FLOAT) return RuntimeResult.Failure(RuntimeError.UnsupportedTensorType(input.name, input.type))
            val override = overrides.dimensionsByInput[input.name]
            val shape = when {
                input.hasDynamicDimensions && override == null -> return RuntimeResult.Failure(RuntimeError.DynamicShapeRequiresConcreteDimensions(input.name, input.shape))
                override != null -> override
                else -> input.shape
            }
            if (!TensorShape.isConcrete(shape)) return RuntimeResult.Failure(RuntimeError.DynamicShapeRequiresConcreteDimensions(input.name, shape))
            if (override != null && input.shape.size != override.size) return RuntimeResult.Failure(RuntimeError.TensorShapeMismatch(input.name, input.shape, override))
            input.shape.zip(shape).forEach { (declared, concrete) ->
                if (declared > 0L && declared != concrete) return RuntimeResult.Failure(RuntimeError.TensorShapeMismatch(input.name, input.shape, shape))
            }
            val count = TensorShape.checkedElementCount(shape) ?: return RuntimeResult.Failure(RuntimeError.TensorShapeMismatch(input.name, input.shape, shape))
            if (count > MAX_ELEMENTS_PER_INPUT || total > MAX_TOTAL_ELEMENTS - count || count > Int.MAX_VALUE) {
                return RuntimeResult.Failure(RuntimeError.InferenceFailure("Dummy input '${input.name}' exceeds the safety limit"))
            }
            total += count
            result[input.name] = FloatTensor(shape, FloatArray(count.toInt()))
        }
        return RuntimeResult.Success(result)
    }
}
