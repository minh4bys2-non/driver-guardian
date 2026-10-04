package com.example.driverguardian.ai.contract.demo

import com.example.driverguardian.ai.contract.drowsiness.ContractValidationResult
import com.example.driverguardian.ai.contract.drowsiness.MetadataStatus
import com.example.driverguardian.ai.contract.drowsiness.ModelContract
import com.example.driverguardian.ai.contract.drowsiness.SemanticStatus
import com.example.driverguardian.ai.runtime.ModelMetadata
import com.example.driverguardian.ai.runtime.RuntimeTensorType

/**
 * DEMO / TEMPORARY ONNX CONTRACT
 *
 * Validates the demo end-to-end ONNX model artifact (driver_guardian_end2end_demo.onnx).
 *
 * IMPORTANT:
 * This contract is strictly for technical pipeline validation:
 * Camera / preprocessing -> ONNX Runtime Android -> Tensor inference -> Logits / probabilities.
 *
 * It is NOT:
 * - official production model
 * - final trained model
 * - validated drowsiness model
 *
 * Expected Tensor Contract:
 * - Input:
 *     name = "video"
 *     type = FLOAT32
 *     shape = [batch_size, seq_len, 3, 640, 640] (dynamic B, dynamic T, 3 channels, 640x640 letterbox)
 * - Output:
 *     name = "logits"
 *     type = FLOAT32
 *     shape = [batch_size, seq_len, 2] (dynamic B, dynamic T, binary logits: index 0 = Alert, index 1 = Drowsy)
 */
class EndToEndDemoModelContract : ModelContract {

    companion object {
        const val EXPECTED_INPUT_NAME = "video"
        const val EXPECTED_OUTPUT_NAME = "logits"
        const val EXPECTED_CHANNELS = 3L
        const val EXPECTED_HEIGHT = 640L
        const val EXPECTED_WIDTH = 640L
        const val EXPECTED_NUM_CLASSES = 2L
    }

    override fun validate(metadata: ModelMetadata): ContractValidationResult {
        val messages = mutableListOf<String>()

        // 1. Structure check: exactly 1 input and 1 output
        if (metadata.inputs.size != 1) {
            messages += "Expected exactly 1 input tensor ('$EXPECTED_INPUT_NAME'), found ${metadata.inputs.size}."
        }
        if (metadata.outputs.size != 1) {
            messages += "Expected exactly 1 output tensor ('$EXPECTED_OUTPUT_NAME'), found ${metadata.outputs.size}."
        }

        // 2. Validate input 'video'
        val videoInput = metadata.inputs.firstOrNull { it.name == EXPECTED_INPUT_NAME }
        if (videoInput == null) {
            messages += "Missing required input tensor '$EXPECTED_INPUT_NAME'."
        } else {
            if (videoInput.type != RuntimeTensorType.FLOAT) {
                messages += "Input '$EXPECTED_INPUT_NAME' must be FLOAT, found ${videoInput.type}."
            }
            if (videoInput.rank != 5) {
                messages += "Input '$EXPECTED_INPUT_NAME' rank must be 5 [B, T, 3, 640, 640], found rank ${videoInput.rank} (${videoInput.shape})."
            } else {
                val c = videoInput.shape[2]
                val h = videoInput.shape[3]
                val w = videoInput.shape[4]
                if (c != EXPECTED_CHANNELS || h != EXPECTED_HEIGHT || w != EXPECTED_WIDTH) {
                    messages += "Input '$EXPECTED_INPUT_NAME' spatial/channel shape must be [*, *, 3, 640, 640], found [*, *, $c, $h, $w]."
                }
            }
        }

        // 3. Validate output 'logits'
        val logitsOutput = metadata.outputs.firstOrNull { it.name == EXPECTED_OUTPUT_NAME }
        if (logitsOutput == null) {
            messages += "Missing required output tensor '$EXPECTED_OUTPUT_NAME'."
        } else {
            if (logitsOutput.type != RuntimeTensorType.FLOAT) {
                messages += "Output '$EXPECTED_OUTPUT_NAME' must be FLOAT, found ${logitsOutput.type}."
            }
            if (logitsOutput.rank != 3) {
                messages += "Output '$EXPECTED_OUTPUT_NAME' rank must be 3 [B, T, 2], found rank ${logitsOutput.rank} (${logitsOutput.shape})."
            } else {
                val numClasses = logitsOutput.shape[2]
                if (numClasses != EXPECTED_NUM_CLASSES) {
                    messages += "Output '$EXPECTED_OUTPUT_NAME' class dimension must be $EXPECTED_NUM_CLASSES, found $numClasses."
                }
            }
        }

        val isValid = messages.isEmpty()
        return if (isValid) {
            ContractValidationResult(
                metadataStatus = MetadataStatus.VALID,
                semanticStatus = SemanticStatus.RESOLVED,
                messages = listOf("DEMO / TEMPORARY ONNX CONTRACT: Validated end-to-end video input [B, T, 3, 640, 640] -> logits [B, T, 2]."),
            )
        } else {
            ContractValidationResult(
                metadataStatus = MetadataStatus.INVALID,
                semanticStatus = SemanticStatus.INVALID,
                messages = messages,
            )
        }
    }
}
