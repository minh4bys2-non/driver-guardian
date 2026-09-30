package com.example.driverguardian.ai.parity

import com.example.driverguardian.ai.tensor.TensorData

data class GoldenTestCase(
    val name: String,
    val inputs: Map<String, TensorData>,
    val expectedOutputs: Map<String, FloatArray>,
)

data class GoldenVectorConvention(
    val manifestAsset: String = "onnx_test_vectors/manifest.json",
    val binaryEncoding: String = "raw little-endian float32",
)
