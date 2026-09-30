package com.example.driverguardian.ai.tensor

import com.example.driverguardian.ai.runtime.TensorPreview
import java.lang.reflect.Array

object TensorPreviewFactory {
    const val DEFAULT_LIMIT = 32
    fun fromValue(value: Any?, limit: Int = DEFAULT_LIMIT): TensorPreview {
        require(limit > 0)
        val values = mutableListOf<String>()
        var total = 0L
        fun visit(current: Any?) {
            if (current != null && current.javaClass.isArray) {
                for (index in 0 until Array.getLength(current)) visit(Array.get(current, index))
            } else {
                total++
                if (values.size < limit) values += current?.toString() ?: "null"
            }
        }
        visit(value)
        return TensorPreview(values, total, total > limit)
    }
    fun fromFloats(values: FloatArray, limit: Int = DEFAULT_LIMIT) = TensorPreview(values.take(limit).map(Float::toString), values.size.toLong(), values.size > limit)
}
