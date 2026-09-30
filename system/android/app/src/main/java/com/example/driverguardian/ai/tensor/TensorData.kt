package com.example.driverguardian.ai.tensor

import com.example.driverguardian.ai.runtime.RuntimeTensorType

sealed interface TensorData { val shape: List<Long>; val type: RuntimeTensorType }
data class FloatTensor(override val shape: List<Long>, val values: FloatArray) : TensorData {
    override val type = RuntimeTensorType.FLOAT
}
data class ConcreteShapeOverrides(val dimensionsByInput: Map<String, List<Long>> = emptyMap())
