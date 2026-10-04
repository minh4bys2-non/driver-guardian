package com.example.driverguardian.ai.demo

import com.example.driverguardian.ai.contract.demo.EndToEndDemoModelContract
import com.example.driverguardian.ai.contract.drowsiness.MetadataStatus
import com.example.driverguardian.ai.contract.drowsiness.SemanticStatus
import com.example.driverguardian.ai.runtime.ModelMetadata
import com.example.driverguardian.ai.runtime.OnnxRuntimeEngine
import com.example.driverguardian.ai.runtime.RuntimeResult
import com.example.driverguardian.ai.runtime.RuntimeTensorType
import com.example.driverguardian.ai.runtime.TensorMetadata
import com.example.driverguardian.ai.tensor.FloatTensor
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.security.MessageDigest

class EndToEndDemoModelTest {

    private val demoModelFile = File("src/main/assets/models/driver_guardian_end2end_demo.onnx")

    @Test
    fun demoModelAssetExistsAndSha256Matches() {
        assertTrue("Model asset file must exist", demoModelFile.exists())
        assertEquals("Model size must be 13,414,830 bytes", 13414830L, demoModelFile.length())

        val digest = MessageDigest.getInstance("SHA-256")
        val bytes = demoModelFile.readBytes()
        val calculatedSha256 = digest.digest(bytes).joinToString("") { "%02x".format(it) }

        val expectedSha256 = "3b0189550f7ed1ae53a620d143a8b1ed715f5e970b0977e0d44cfd6cb9680057"
        assertEquals("SHA-256 must match exactly", expectedSha256, calculatedSha256)
    }

    @Test
    fun contractValidatesExpectedDemoModelMetadata() {
        val contract = EndToEndDemoModelContract()
        val validMetadata = ModelMetadata(
            modelName = "driver_guardian_end2end_demo.onnx",
            inputs = listOf(
                TensorMetadata(
                    name = "video",
                    type = RuntimeTensorType.FLOAT,
                    onnxType = "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT",
                    shape = listOf(-1L, -1L, 3L, 640L, 640L),
                    dimensionNames = listOf("batch_size", "seq_len", null, null, null),
                )
            ),
            outputs = listOf(
                TensorMetadata(
                    name = "logits",
                    type = RuntimeTensorType.FLOAT,
                    onnxType = "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT",
                    shape = listOf(-1L, -1L, 2L),
                    dimensionNames = listOf("batch_size", "seq_len", null),
                )
            ),
            loadDurationNanos = 10_000_000L,
        )

        val result = contract.validate(validMetadata)
        assertEquals(MetadataStatus.VALID, result.metadataStatus)
        assertEquals(SemanticStatus.RESOLVED, result.semanticStatus)
        assertTrue(result.isRunnable)
    }

    @Test
    fun contractRejectsMismatchedInputsOrOutputs() {
        val contract = EndToEndDemoModelContract()

        // Wrong input name
        val wrongInputName = ModelMetadata(
            modelName = "mismatch.onnx",
            inputs = listOf(
                TensorMetadata("wrong_input", RuntimeTensorType.FLOAT, "FLOAT", listOf(1, 1, 3, 640, 640))
            ),
            outputs = listOf(
                TensorMetadata("logits", RuntimeTensorType.FLOAT, "FLOAT", listOf(1, 1, 2))
            ),
            loadDurationNanos = 0,
        )
        val res1 = contract.validate(wrongInputName)
        assertEquals(MetadataStatus.INVALID, res1.metadataStatus)
        assertFalse(res1.isRunnable)

        // Wrong spatial dimensions
        val wrongSpatial = ModelMetadata(
            modelName = "mismatch2.onnx",
            inputs = listOf(
                TensorMetadata("video", RuntimeTensorType.FLOAT, "FLOAT", listOf(1, 1, 3, 224, 224))
            ),
            outputs = listOf(
                TensorMetadata("logits", RuntimeTensorType.FLOAT, "FLOAT", listOf(1, 1, 2))
            ),
            loadDurationNanos = 0,
        )
        val res2 = contract.validate(wrongSpatial)
        assertEquals(MetadataStatus.INVALID, res2.metadataStatus)
        assertFalse(res2.isRunnable)

        // Wrong class count (e.g. 5 classes instead of 2)
        val wrongClasses = ModelMetadata(
            modelName = "mismatch3.onnx",
            inputs = listOf(
                TensorMetadata("video", RuntimeTensorType.FLOAT, "FLOAT", listOf(1, 1, 3, 640, 640))
            ),
            outputs = listOf(
                TensorMetadata("logits", RuntimeTensorType.FLOAT, "FLOAT", listOf(1, 1, 5))
            ),
            loadDurationNanos = 0,
        )
        val res3 = contract.validate(wrongClasses)
        assertEquals(MetadataStatus.INVALID, res3.metadataStatus)
        assertFalse(res3.isRunnable)
    }

    @Test
    fun letterboxGeometryCalculations() {
        // Landscape frame 1280x720 into 640x640:
        // Scale = 640 / 1280 = 0.5. Scaled: 640x360. Pad top/bottom = (640-360)/2 = 140.
        val landscapeGeom = EndToEndDemoPreprocessor.calculateGeometry(1280, 720, 640, 640)
        assertEquals(0.5f, landscapeGeom.scale, 0.001f)
        assertEquals(640, landscapeGeom.scaledWidth)
        assertEquals(360, landscapeGeom.scaledHeight)
        assertEquals(0, landscapeGeom.padLeft)
        assertEquals(0, landscapeGeom.padRight)
        assertEquals(140, landscapeGeom.padTop)
        assertEquals(140, landscapeGeom.padBottom)

        // Portrait frame 720x1280 into 640x640:
        // Scale = 640 / 1280 = 0.5. Scaled: 360x640. Pad left/right = (640-360)/2 = 140.
        val portraitGeom = EndToEndDemoPreprocessor.calculateGeometry(720, 1280, 640, 640)
        assertEquals(0.5f, portraitGeom.scale, 0.001f)
        assertEquals(360, portraitGeom.scaledWidth)
        assertEquals(640, portraitGeom.scaledHeight)
        assertEquals(140, portraitGeom.padLeft)
        assertEquals(140, portraitGeom.padRight)
        assertEquals(0, portraitGeom.padTop)
        assertEquals(0, portraitGeom.padBottom)

        // Square frame 640x640:
        val squareGeom = EndToEndDemoPreprocessor.calculateGeometry(640, 640, 640, 640)
        assertEquals(1.0f, squareGeom.scale, 0.001f)
        assertEquals(0, squareGeom.padLeft)
        assertEquals(0, squareGeom.padTop)
    }

    @Test
    fun rgbPreprocessingLetterboxPadAndScaling() {
        // Construct a small 2x2 test image with known RGB values
        // Top-left: Pure Red (0xFFFF0000)
        // Top-right: Pure Green (0xFF00FF00)
        // Bottom-left: Pure Blue (0xFF0000FF)
        // Bottom-right: White (0xFFFFFFFF)
        val pixels = intArrayOf(
            0xFFFF0000.toInt(), 0xFF00FF00.toInt(),
            0xFF0000FF.toInt(), 0xFFFFFFFF.toInt(),
        )

        val nchw = EndToEndDemoPreprocessor.preprocessRgbPixels(
            pixels = pixels,
            srcWidth = 2,
            srcHeight = 2,
            targetWidth = 640,
            targetHeight = 640,
            padValue = 114,
        )

        assertEquals("Output must be exactly 3 * 640 * 640 floats", 3 * 640 * 640, nchw.size)

        val channelR = 0
        val channelG = 640 * 640
        val channelB = 2 * 640 * 640

        // Padded boundary pixel: e.g. at (0, 0), since 2x2 scales to 640x640 with pad 0 for square,
        // let's test pad with an aspect-mismatched image (e.g. 2x1)
        val nonSquarePixels = intArrayOf(0xFFFF0000.toInt(), 0xFF00FF00.toInt())
        val letterboxed = EndToEndDemoPreprocessor.preprocessRgbPixels(
            pixels = nonSquarePixels,
            srcWidth = 2,
            srcHeight = 1,
            targetWidth = 640,
            targetHeight = 640,
            padValue = 114,
        )

        // In 2x1 -> scaled to 640x320. Pad top is 160.
        // Row 0 is within top padding:
        val padRowIdx = 10 * 640 + 10
        val expectedPadFloat = 114f / 255.0f
        assertEquals("Padding pixel R must be pad/255", expectedPadFloat, letterboxed[channelR + padRowIdx], 1e-5f)
        assertEquals("Padding pixel G must be pad/255", expectedPadFloat, letterboxed[channelG + padRowIdx], 1e-5f)
        assertEquals("Padding pixel B must be pad/255", expectedPadFloat, letterboxed[channelB + padRowIdx], 1e-5f)

        // Center row (y = 320) is within scaled content:
        val contentIdx = 320 * 640 + 100 // left side (Red)
        assertTrue("Content pixel R must be > 0.9", letterboxed[channelR + contentIdx] > 0.9f)
    }

    @Test
    fun boundedSequenceBufferCapacityAndFifo() {
        val buffer = BoundedSequenceBuffer(maxSequenceLength = 3, sampleIntervalMs = 0L)
        assertEquals(0, buffer.currentSize)
        assertFalse(buffer.isFull)

        val dummyFrame = FloatArray(EndToEndDemoPreprocessor.FRAME_ELEMENT_COUNT) { 0.1f }

        // Offer 1st frame
        assertTrue(buffer.offer(dummyFrame, 1000L))
        assertEquals(1, buffer.currentSize)

        // Offer 2nd frame
        assertTrue(buffer.offer(dummyFrame, 1500L))
        assertEquals(2, buffer.currentSize)

        // Offer 3rd frame
        assertTrue(buffer.offer(dummyFrame, 2000L))
        assertEquals(3, buffer.currentSize)
        assertTrue(buffer.isFull)

        // Offer 4th frame -> should evict 1st and maintain size 3
        val fourthFrame = FloatArray(EndToEndDemoPreprocessor.FRAME_ELEMENT_COUNT) { 0.9f }
        assertTrue(buffer.offer(fourthFrame, 2500L))
        assertEquals(3, buffer.currentSize)

        // Check tensor conversion
        val tensor = buffer.toFloatTensor()
        assertNotNull(tensor)
        assertEquals(listOf(1L, 3L, 3L, 640L, 640L), tensor!!.shape)
        assertEquals(3 * EndToEndDemoPreprocessor.FRAME_ELEMENT_COUNT, tensor.values.size)

        buffer.clear()
        assertEquals(0, buffer.currentSize)
        assertNull(buffer.toFloatTensor())
    }

    @Test
    fun boundedSequenceBufferSamplingRateLimit() {
        val buffer = BoundedSequenceBuffer(maxSequenceLength = 10, sampleIntervalMs = 500L) // 2 FPS
        val dummyFrame = FloatArray(EndToEndDemoPreprocessor.FRAME_ELEMENT_COUNT) { 0.5f }

        // 1st frame at 1000ms: accepted
        assertTrue(buffer.offer(dummyFrame, 1000L))
        assertEquals(1, buffer.currentSize)

        // Frame at 1200ms (only 200ms later): dropped by 2 FPS limit
        assertFalse(buffer.offer(dummyFrame, 1200L))
        assertEquals(1, buffer.currentSize)

        // Frame at 1450ms (450ms later): dropped
        assertFalse(buffer.offer(dummyFrame, 1450L))
        assertEquals(1, buffer.currentSize)

        // Frame at 1501ms (501ms later): accepted!
        assertTrue(buffer.offer(dummyFrame, 1501L))
        assertEquals(2, buffer.currentSize)

        // Forced frame at 1550ms: accepted due to force=true
        assertTrue(buffer.offer(dummyFrame, 1550L, force = true))
        assertEquals(3, buffer.currentSize)
    }

    @Test
    fun softmaxComputationProperties() {
        // Equal logits
        val (p0, p1) = DemoOnnxOutput.computeSoftmax(0.0f, 0.0f)
        assertEquals(0.5f, p0, 1e-6f)
        assertEquals(0.5f, p1, 1e-6f)

        // Known logits from Python test: alert=0.046868, drowsy=-0.008286
        val (alertP, drowsyP) = DemoOnnxOutput.computeSoftmax(0.046868f, -0.008286f)
        assertTrue("Alert prob must be > Drowsy prob", alertP > drowsyP)
        assertEquals("Probabilities must sum to 1.0", 1.0f, alertP + drowsyP, 1e-5f)

        // Numerical stability check with large logits
        val (safeP0, safeP1) = DemoOnnxOutput.computeSoftmax(1000.0f, 1000.0f)
        assertFalse(safeP0.isNaN())
        assertFalse(safeP1.isNaN())
        assertEquals(0.5f, safeP0, 1e-6f)
        assertEquals(0.5f, safeP1, 1e-6f)
    }

    @Test
    fun onnxRuntimeEngineCompatibilityVectorParity() {
        val engine = OnnxRuntimeEngine()
        val bytes = demoModelFile.readBytes()

        // 1. Model load
        val loadResult = engine.loadModel("driver_guardian_end2end_demo.onnx", bytes)
        assertTrue("Model load must succeed: ${(loadResult as? RuntimeResult.Failure)?.error}", loadResult is RuntimeResult.Success)

        val metadata = (loadResult as RuntimeResult.Success).value
        val contractResult = EndToEndDemoModelContract().validate(metadata)
        assertTrue("Contract must be runnable", contractResult.isRunnable)

        // 2. Test T=1 constant 0.5 (Parity with Python ORT test)
        // Python result: shape [1, 1, 2], raw logits: [[[0.04686835, -0.00828595]]]
        val t1Size = 1 * 3 * 640 * 640
        val t1Tensor = FloatTensor(listOf(1L, 1L, 3L, 640L, 640L), FloatArray(t1Size) { 0.5f })
        val t1Run = engine.run(mapOf("video" to t1Tensor))
        assertTrue("T=1 inference must succeed", t1Run is RuntimeResult.Success)

        val t1Outputs = (t1Run as RuntimeResult.Success).value.outputs
        val t1Logits = t1Outputs.firstOrNull { it.name == "logits" }
        assertNotNull(t1Logits)
        assertEquals(listOf(1L, 1L, 2L), t1Logits!!.shape)

        val t1Values = t1Logits.floatValues
        assertNotNull(t1Values)
        assertEquals(2, t1Values!!.size)

        val expectedT1Alert = 0.04686835f
        val expectedT1Drowsy = -0.00828595f
        assertEquals("T=1 Alert logit parity", expectedT1Alert, t1Values[0], 1e-4f)
        assertEquals("T=1 Drowsy logit parity", expectedT1Drowsy, t1Values[1], 1e-4f)
        assertFalse("Output must not be NaN", t1Values[0].isNaN() || t1Values[1].isNaN())
        assertFalse("Output must not be Infinite", t1Values[0].isInfinite() || t1Values[1].isInfinite())

        // 3. Test T=2 constant 0.0 (Parity with Python ORT test)
        // Python result: shape [1, 2, 2], raw logits:
        // [[[0.04824053, -0.00908979], [0.05069056, -0.01034536]]]
        val t2Size = 2 * 3 * 640 * 640
        val t2Tensor = FloatTensor(listOf(1L, 2L, 3L, 640L, 640L), FloatArray(t2Size) { 0.0f })
        val t2Run = engine.run(mapOf("video" to t2Tensor))
        assertTrue("T=2 inference must succeed", t2Run is RuntimeResult.Success)

        val t2Logits = (t2Run as RuntimeResult.Success).value.outputs.first { it.name == "logits" }
        assertEquals(listOf(1L, 2L, 2L), t2Logits.shape)

        val t2Values = t2Logits.floatValues
        assertNotNull(t2Values)
        assertEquals(4, t2Values!!.size)

        // Step 0
        assertEquals("T=2 Step 0 Alert parity", 0.04824053f, t2Values[0], 1e-4f)
        assertEquals("T=2 Step 0 Drowsy parity", -0.00908979f, t2Values[1], 1e-4f)
        // Step 1
        assertEquals("T=2 Step 1 Alert parity", 0.05069056f, t2Values[2], 1e-4f)
        assertEquals("T=2 Step 1 Drowsy parity", -0.01034536f, t2Values[3], 1e-4f)

        engine.close()
    }
}
