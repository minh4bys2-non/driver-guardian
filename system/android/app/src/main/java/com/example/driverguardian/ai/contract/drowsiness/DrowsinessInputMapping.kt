package com.example.driverguardian.ai.contract.drowsiness

enum class DrowsinessSemanticInput { LEFT_EYE_SEQUENCE, RIGHT_EYE_SEQUENCE, MOUTH_SEQUENCE, GEOMETRY_SEQUENCE }

sealed interface DrowsinessInputMapping {
    data object Unresolved : DrowsinessInputMapping
    data class Explicit(val nodeNames: Map<DrowsinessSemanticInput, String>, val probabilityOutputName: String? = null) : DrowsinessInputMapping
}
