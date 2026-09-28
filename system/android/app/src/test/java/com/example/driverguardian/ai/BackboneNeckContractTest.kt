package com.example.driverguardian.ai

import com.example.driverguardian.ai.detection.BackboneNeckContract
import com.example.driverguardian.ai.detection.BackboneNeckPipeline
import com.example.driverguardian.ai.runtime.*
import com.example.driverguardian.ai.tensor.FloatTensor
import org.junit.Assert.*
import org.junit.Test
import java.io.File

class BackboneNeckContractTest {
    @Test fun measuresRealModelInferenceTime() {
        val loaded = BackboneNeckPipeline.load(File("src/main/assets/${BackboneNeckPipeline.MODEL_ASSET}").readBytes())
        assertTrue("Model load failed: $loaded", loaded is RuntimeResult.Success)
        (loaded as RuntimeResult.Success).value.use { pipeline ->
            val images = FloatTensor(listOf(1, 3, 640, 640), FloatArray(3 * 640 * 640) { (it % 256) / 255f })
            repeat(3) {
                val result = pipeline.run(images)
                if (result is RuntimeResult.Failure) error("Warm-up failed: ${result.error}")
            }
            val inferenceMs = mutableListOf<Double>()
            val pipelineMs = mutableListOf<Double>()
            repeat(10) {
                val start = System.nanoTime()
                val result = pipeline.run(images)
                val elapsed = System.nanoTime() - start
                if (result is RuntimeResult.Failure) error("Inference failed: ${result.error}")
                val features = (result as RuntimeResult.Success).value
                assertTrue(features.inferenceDurationNanos > 0)
                inferenceMs += features.inferenceDurationNanos / 1_000_000.0
                pipelineMs += elapsed / 1_000_000.0
            }
            println("BackboneNeck CPU, batch=1, 640x640, 3 warm-ups, 10 measured runs")
            println("Inference (ms): avg=${inferenceMs.average()}, min=${inferenceMs.minOrNull()}, max=${inferenceMs.maxOrNull()}")
            println("Pipeline (ms): avg=${pipelineMs.average()}, min=${pipelineMs.minOrNull()}, max=${pipelineMs.maxOrNull()}")
        }
    }

    private val input = tensor("images", listOf(-1, 3, 640, 640))
    private val outputs = listOf(
        tensor("p3", listOf(-1, 64, 80, 80)),
        tensor("p4", listOf(-1, 128, 40, 40)),
        tensor("p5", listOf(-1, 256, 20, 20)),
    )

    @Test fun validatesDynamicModelAndRejectsWrongContract() {
        assertNull(BackboneNeckContract.validateModel(model()))
        assertNotNull(BackboneNeckContract.validateModel(model(input.copy(name = "wrong"))))
        assertNotNull(BackboneNeckContract.validateModel(model(input.copy(shape = listOf(1, 3, 640, 640)))))
        assertNotNull(BackboneNeckContract.validateModel(model(outputs = outputs.dropLast(1))))
        assertNotNull(BackboneNeckContract.validateModel(model(outputs = outputs.map { it.copy(type = RuntimeTensorType.INT64) })))
    }

    @Test fun rejectsInvalidBatchLayoutAndStorage() {
        for (shape in listOf(emptyList(), listOf(0L, 3, 640, 640), listOf(1L, 640, 640, 3),
            listOf(1L, 3, 320, 320), listOf(Long.MAX_VALUE, 3, 640, 640), listOf(1L, 3, 640, 640))) {
            assertNotNull(BackboneNeckContract.validateInput(FloatTensor(shape, floatArrayOf())))
        }
        assertNull(BackboneNeckContract.validateInput(FloatTensor(listOf(2, 3, 640, 640), FloatArray(2 * 3 * 640 * 640))))
    }

    @Test fun mapsOutputsByNameAndPreservesLargeBatchFeatures() {
        val result = inference(3)
        val features = (BackboneNeckContract.features(result.copy(outputs = result.outputs.reversed()), 3)
            as RuntimeResult.Success).value
        assertEquals(1_228_800, features.p3.values.size)
        assertSame(result.outputs[0].floatValues, features.p3.values)
        assertEquals(listOf(3L, 128, 40, 40), features.p4.shape)
        assertEquals(listOf(3L, 256, 20, 20), features.p5.shape)
        assertEquals(42L, features.inferenceDurationNanos)
    }

    @Test fun rejectsMissingTruncatedAndWrongBatchOutputs() {
        val result = inference(1)
        assertTrue(BackboneNeckContract.features(result.copy(outputs = result.outputs.dropLast(1)), 1) is RuntimeResult.Failure)
        assertTrue(BackboneNeckContract.features(result, 2) is RuntimeResult.Failure)
        for (values in listOf(null, floatArrayOf(1f))) {
            val damaged = result.outputs.toMutableList()
            damaged[0] = damaged[0].copy(floatValues = values)
            assertTrue(BackboneNeckContract.features(result.copy(outputs = damaged), 1) is RuntimeResult.Failure)
        }
    }

    private fun tensor(name: String, shape: List<Long>) = TensorMetadata(name, RuntimeTensorType.FLOAT, "FLOAT", shape)
    private fun model(input: TensorMetadata = this.input, outputs: List<TensorMetadata> = this.outputs) =
        ModelMetadata("backbone_neck.onnx", listOf(input), outputs, 0)

    private fun inference(batch: Long) = RuntimeInferenceResult(42, outputs.map {
        val shape = listOf(batch) + it.shape.drop(1)
        val values = FloatArray(shape.reduce(Long::times).toInt())
        RuntimeOutput(it.name, it.type, shape, TensorPreview(emptyList(), values.size.toLong(), true), values)
    })
}
