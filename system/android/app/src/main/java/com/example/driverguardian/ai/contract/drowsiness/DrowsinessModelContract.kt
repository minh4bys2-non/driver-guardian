package com.example.driverguardian.ai.contract.drowsiness

import com.example.driverguardian.ai.runtime.ModelMetadata
import com.example.driverguardian.ai.runtime.RuntimeTensorType

/** Validates the known semantic tensor contract without guessing ONNX node names. */
class DrowsinessModelContract(private val mapping: DrowsinessInputMapping = DrowsinessInputMapping.Unresolved) : ModelContract {
    override fun validate(metadata: ModelMetadata): ContractValidationResult {
        val messages = mutableListOf<String>()
        val structuralValid = metadata.inputs.size == 4 && metadata.outputs.isNotEmpty()
        if (!structuralValid) messages += "Expected 4 tensor inputs and at least 1 output; found ${metadata.inputs.size} and ${metadata.outputs.size}."
        if (metadata.inputs.any { it.type == RuntimeTensorType.UNKNOWN } || metadata.outputs.any { it.type == RuntimeTensorType.UNKNOWN }) {
            messages += "The model contains a non-tensor or unknown tensor type."
        }
        val metadataStatus = if (structuralValid && messages.isEmpty()) MetadataStatus.VALID else MetadataStatus.INVALID
        if (mapping is DrowsinessInputMapping.Unresolved) {
            messages += "Semantic input mapping is unresolved. Similar image tensor shapes must not be mapped by guesswork."
            return ContractValidationResult(metadataStatus, SemanticStatus.UNRESOLVED, messages)
        }
        val explicitMapping = mapping as DrowsinessInputMapping.Explicit
        val expected = mapOf(
            DrowsinessSemanticInput.LEFT_EYE_SEQUENCE to listOf(1L, 30L, 3L, 64L, 64L),
            DrowsinessSemanticInput.RIGHT_EYE_SEQUENCE to listOf(1L, 30L, 3L, 64L, 64L),
            DrowsinessSemanticInput.MOUTH_SEQUENCE to listOf(1L, 30L, 3L, 64L, 64L),
            DrowsinessSemanticInput.GEOMETRY_SEQUENCE to listOf(1L, 30L, 10L),
        )
        val names = explicitMapping.nodeNames
        if (names.keys != DrowsinessSemanticInput.entries.toSet()) messages += "Explicit mapping must name every semantic input."
        if (names.values.size != names.values.toSet().size) messages += "Explicit mapping contains duplicate node names."
        expected.forEach { (semantic, shape) ->
            val nodeName = names[semantic]
            val tensor = metadata.inputs.firstOrNull { it.name == nodeName }
            when {
                nodeName == null -> Unit
                tensor == null -> messages += "$semantic maps to missing node '$nodeName'."
                tensor.type != RuntimeTensorType.FLOAT -> messages += "$semantic must be FLOAT, found ${tensor.type}."
                tensor.shape != shape -> messages += "$semantic expected $shape, found ${tensor.shape}."
            }
        }
        val output = explicitMapping.probabilityOutputName?.let { name -> metadata.outputs.firstOrNull { it.name == name } }
        when {
            explicitMapping.probabilityOutputName == null -> messages += "The drowsiness probability output mapping is unresolved."
            output == null -> messages += "Mapped probability output '${explicitMapping.probabilityOutputName}' is missing."
            output.type != RuntimeTensorType.FLOAT || output.shape != listOf(1L, 1L) -> messages += "Probability output must be FLOAT [1, 1]."
        }
        val semanticValid = messages.isEmpty() && metadataStatus == MetadataStatus.VALID
        return ContractValidationResult(metadataStatus, if (semanticValid) SemanticStatus.RESOLVED else SemanticStatus.INVALID, messages)
    }
}
