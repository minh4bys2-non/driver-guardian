package com.example.driverguardian.ai.detection

import java.io.Closeable
import android.content.res.AssetManager
import com.example.driverguardian.ai.runtime.*
import com.example.driverguardian.ai.tensor.FloatTensor
import com.example.driverguardian.ai.tensor.TensorShape
import org.jetbrains.annotations.TestOnly
import java.io.FileNotFoundException

/** Owned FLOAT feature maps in NCHW order; valid after the pipeline is closed. */
data class BackboneNeckFeatures(
    val p3: FloatTensor,
    val p4: FloatTensor,
    val p5: FloatTensor,
    val inferenceDurationNanos: Long,
)
/** Reuse one instance and call load/run off the main thread. Input is already preprocessed RGB / 255. */
class BackboneNeckPipeline private constructor(
    private val engine: OnnxRuntimeEngine,
    val metadata: ModelMetadata,
) : Closeable {
    fun run(images: FloatTensor): RuntimeResult<BackboneNeckFeatures> {
        BackboneNeckContract.validateInput(images)?.let { return RuntimeResult.Failure(it) }
        return when (val result = engine.run(mapOf("images" to images), captureFullOutputs = true)) {
            is RuntimeResult.Failure -> result
            is RuntimeResult.Success -> BackboneNeckContract.features(result.value, images.shape[0])
        }
    }

    override fun close() = engine.close()

    companion object {
        const val MODEL_ASSET = "models/backbone_neck.onnx"

        fun load(assets: AssetManager): RuntimeResult<BackboneNeckPipeline> {
            val bytes = try {
                assets.open(MODEL_ASSET).use { it.readBytes() }
            } catch (_: FileNotFoundException) {
                return RuntimeResult.Failure(RuntimeError.ModelNotFound(MODEL_ASSET))
            } catch (exception: Exception) {
                return RuntimeResult.Failure(RuntimeError.ModelLoadFailure("Could not read $MODEL_ASSET", exception))
            }
            return load(bytes)
        }

        fun load(modelBytes: ByteArray): RuntimeResult<BackboneNeckPipeline> {
            val engine = OnnxRuntimeEngine()
            when (val result = engine.loadModel(MODEL_ASSET, modelBytes)) {
                is RuntimeResult.Failure -> {
                    engine.close()
                    return result
                }
                is RuntimeResult.Success -> {
                    val error = BackboneNeckContract.validateModel(result.value)
                    if (error != null) {
                        engine.close()
                        return RuntimeResult.Failure(error)
                    }
                    return RuntimeResult.Success(BackboneNeckPipeline(engine, result.value))
                }
            }
        }
    }
}

internal object BackboneNeckContract {
    private val imageShape = listOf(3L, 640L, 640L)
    private val featureShapes = linkedMapOf(
        "p3" to listOf(64L, 80L, 80L),
        "p4" to listOf(128L, 40L, 40L),
        "p5" to listOf(256L, 20L, 20L),
    )

    fun validateModel(metadata: ModelMetadata): RuntimeError? {
        val input = metadata.inputs.singleOrNull()
        if (input?.name != "images" || !matches(input, imageShape)) {
            return RuntimeError.ContractMismatch("Expected images FLOAT [batch_size, 3, 640, 640] with dynamic batch")
        }
        if (metadata.outputs.size != featureShapes.size || featureShapes.any { (name, shape) ->
                metadata.outputs.singleOrNull { it.name == name }?.let { matches(it, shape) } != true
            }) {
            return RuntimeError.ContractMismatch("Expected dynamic-batch FLOAT outputs p3 [B,64,80,80], p4 [B,128,40,40], p5 [B,256,20,20]")
        }
        return null
    }

    fun validateInput(images: FloatTensor): RuntimeError? {
        if (images.shape.size != 4 || images.shape[0] <= 0 || images.shape.drop(1) != imageShape) {
            return RuntimeError.TensorShapeMismatch("images", listOf(-1L) + imageShape, images.shape)
        }
        val count = TensorShape.checkedElementCount(images.shape)
        if (count != images.values.size.toLong()) {
            return RuntimeError.ContractMismatch("images shape ${images.shape} does not match ${images.values.size} values")
        }
        return null
    }

    fun features(result: RuntimeInferenceResult, batchSize: Long): RuntimeResult<BackboneNeckFeatures> {
        val tensors = featureShapes.map { (name, shape) ->
            val output = result.outputs.singleOrNull { it.name == name }
            val expected = listOf(batchSize) + shape
            val values = output?.floatValues
            if (output?.type != RuntimeTensorType.FLOAT || output.shape != expected ||
                values == null || values.size.toLong() != TensorShape.checkedElementCount(expected)) {
                return RuntimeResult.Failure(RuntimeError.ContractMismatch("Expected complete FLOAT output '$name' with shape $expected"))
            }
            FloatTensor(output.shape, values)
        }
        return RuntimeResult.Success(BackboneNeckFeatures(tensors[0], tensors[1], tensors[2], result.durationNanos))
    }

    private fun matches(tensor: TensorMetadata, tail: List<Long>): Boolean =
        tensor.type == RuntimeTensorType.FLOAT && tensor.shape.size == 4 &&
            tensor.shape[0] < 0 && tensor.shape.drop(1) == tail
}
