package com.example.driverguardian.ai.runtime

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import com.example.driverguardian.ai.tensor.FloatTensor
import com.example.driverguardian.ai.tensor.TensorData
import com.example.driverguardian.ai.tensor.TensorPreviewFactory
import com.example.driverguardian.ai.tensor.TensorShape
import java.nio.ByteBuffer
import java.nio.ByteOrder

internal class OnnxModelSession private constructor(
    private val environment: OrtEnvironment,
    private val options: OrtSession.SessionOptions,
    private val session: OrtSession,
    val metadata: ModelMetadata,
) : AutoCloseable {

    fun run(inputs: Map<String, TensorData>): RuntimeResult<RuntimeInferenceResult> {
        validateInputs(inputs)?.let { return RuntimeResult.Failure(it) }
        val nativeInputs = linkedMapOf<String, OnnxTensor>()
        return try {
            inputs.forEach { (name, data) ->
                val tensor = data as FloatTensor
                val buffer = ByteBuffer.allocateDirect(tensor.values.size * Float.SIZE_BYTES)
                    .order(ByteOrder.nativeOrder()).asFloatBuffer()
                buffer.put(tensor.values).flip()
                nativeInputs[name] = OnnxTensor.createTensor(environment, buffer, tensor.shape.toLongArray())
            }
            val start = System.nanoTime()
            session.run(nativeInputs).use { result ->
                val duration = System.nanoTime() - start
                val outputs = result.map { entry ->
                    val declared = metadata.outputs.firstOrNull { it.name == entry.key }
                    val actualShape = (entry.value.info as? TensorInfo)?.shape?.toList() ?: declared?.shape.orEmpty()
                    val floatCapture = (entry.value as? OnnxTensor)
                        ?.takeIf { declared?.type == RuntimeTensorType.FLOAT }
                        ?.let(::captureFloats)
                    val value = if (floatCapture == null) entry.value.value else null
                    RuntimeOutput(
                        name = entry.key,
                        type = declared?.type ?: RuntimeTensorType.UNKNOWN,
                        shape = actualShape,
                        preview = floatCapture?.second ?: TensorPreviewFactory.fromValue(value),
                        floatValues = floatCapture?.first,
                    )
                }
                RuntimeResult.Success(RuntimeInferenceResult(duration, outputs))
            }
        } catch (throwable: Throwable) {
            RuntimeResult.Failure(RuntimeError.InferenceFailure(throwable.message ?: "ONNX inference failed", throwable))
        } finally {
            nativeInputs.values.forEach { runCatching { it.close() } }
        }
    }

    private fun validateInputs(inputs: Map<String, TensorData>): RuntimeError? {
        val expected = metadata.inputs.associateBy { it.name }
        expected.keys.firstOrNull { it !in inputs }?.let { return RuntimeError.MissingInput(it) }
        inputs.keys.firstOrNull { it !in expected }?.let { return RuntimeError.UnexpectedInput(it) }
        inputs.forEach { (name, data) ->
            val input = expected.getValue(name)
            if (data.type != input.type) return RuntimeError.UnsupportedTensorType(name, data.type)
            if (data.shape.size != input.shape.size || input.shape.zip(data.shape).any { (declared, actual) -> declared > 0L && declared != actual }) {
                return RuntimeError.TensorShapeMismatch(name, input.shape, data.shape)
            }
            val floatData = data as? FloatTensor ?: return RuntimeError.UnsupportedTensorType(name, data.type)
            val count = TensorShape.checkedElementCount(data.shape)
                ?: return RuntimeError.TensorShapeMismatch(name, input.shape, data.shape)
            if (count != floatData.values.size.toLong()) return RuntimeError.TensorShapeMismatch(name, input.shape, data.shape)
        }
        return null
    }

    override fun close() {
        runCatching { session.close() }
        runCatching { options.close() }
    }

    companion object {
        private const val MAX_CAPTURED_FLOATS = 1_000_000

        fun create(environment: OrtEnvironment, modelName: String, modelBytes: ByteArray): RuntimeResult<OnnxModelSession> {
            if (modelBytes.isEmpty()) return RuntimeResult.Failure(RuntimeError.InvalidModel("Model bytes are empty"))
            val start = System.nanoTime()
            val options = OrtSession.SessionOptions()
            return try {
                val session = environment.createSession(modelBytes, options)
                val metadata = ModelMetadata(
                    modelName = modelName,
                    inputs = session.inputInfo.map { OrtTypeMapper.metadata(it.key, it.value) },
                    outputs = session.outputInfo.map { OrtTypeMapper.metadata(it.key, it.value) },
                    loadDurationNanos = System.nanoTime() - start,
                )
                RuntimeResult.Success(OnnxModelSession(environment, options, session, metadata))
            } catch (throwable: Throwable) {
                runCatching { options.close() }
                RuntimeResult.Failure(RuntimeError.SessionCreationFailure(throwable.message ?: "Could not create ONNX session", throwable))
            }
        }

        private fun captureFloats(tensor: OnnxTensor): Pair<FloatArray?, TensorPreview> {
            val buffer = tensor.floatBuffer
            val total = buffer.remaining()
            val previewCount = minOf(total, TensorPreviewFactory.DEFAULT_LIMIT)
            val previewValues = FloatArray(previewCount)
            buffer.duplicate().get(previewValues)
            val preview = TensorPreview(previewValues.map(Float::toString), total.toLong(), total > previewCount)
            val full = if (total <= MAX_CAPTURED_FLOATS) FloatArray(total).also { buffer.duplicate().get(it) } else null
            return full to preview
        }
    }
}
