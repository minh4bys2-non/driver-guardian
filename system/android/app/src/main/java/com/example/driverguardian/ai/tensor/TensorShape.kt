package com.example.driverguardian.ai.tensor

object TensorShape {
    fun checkedElementCount(shape: List<Long>): Long? {
        if (shape.isEmpty() || shape.any { it <= 0L }) return null
        var count = 1L
        for (dimension in shape) {
            if (count > Long.MAX_VALUE / dimension) return null
            count *= dimension
        }
        return count
    }
    fun isConcrete(shape: List<Long>) = shape.isNotEmpty() && shape.all { it > 0L }
}
