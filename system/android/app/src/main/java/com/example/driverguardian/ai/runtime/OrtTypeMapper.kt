package com.example.driverguardian.ai.runtime

import ai.onnxruntime.NodeInfo
import ai.onnxruntime.TensorInfo

internal object OrtTypeMapper {
    fun metadata(name: String, nodeInfo: NodeInfo): TensorMetadata {
        val info = nodeInfo.info
        if (info !is TensorInfo) {
            return TensorMetadata(name, RuntimeTensorType.UNKNOWN, info.javaClass.simpleName, emptyList())
        }
        return TensorMetadata(
            name = name,
            type = when (info.type.name) {
                "FLOAT" -> RuntimeTensorType.FLOAT
                "DOUBLE" -> RuntimeTensorType.DOUBLE
                "INT8" -> RuntimeTensorType.INT8
                "INT16" -> RuntimeTensorType.INT16
                "INT32" -> RuntimeTensorType.INT32
                "INT64" -> RuntimeTensorType.INT64
                "UINT8" -> RuntimeTensorType.UINT8
                "BOOL" -> RuntimeTensorType.BOOL
                "STRING" -> RuntimeTensorType.STRING
                "FLOAT16" -> RuntimeTensorType.FLOAT16
                "BFLOAT16" -> RuntimeTensorType.BFLOAT16
                else -> RuntimeTensorType.UNKNOWN
            },
            onnxType = info.onnxType.toString(),
            shape = info.shape.toList(),
            dimensionNames = info.dimensionNames.map { it?.takeIf(String::isNotBlank) },
        )
    }
}
