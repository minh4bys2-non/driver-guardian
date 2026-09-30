package com.example.driverguardian.ai.contract.drowsiness

import com.example.driverguardian.ai.runtime.ModelMetadata

interface ModelContract { fun validate(metadata: ModelMetadata): ContractValidationResult }

enum class MetadataStatus { VALID, INVALID }
enum class SemanticStatus { RESOLVED, UNRESOLVED, INVALID }
data class ContractValidationResult(
    val metadataStatus: MetadataStatus,
    val semanticStatus: SemanticStatus,
    val messages: List<String>,
) { val isRunnable: Boolean get() = metadataStatus == MetadataStatus.VALID && semanticStatus == SemanticStatus.RESOLVED }
