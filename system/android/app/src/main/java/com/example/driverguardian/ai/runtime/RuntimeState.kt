package com.example.driverguardian.ai.runtime

sealed interface RuntimeState {
    data object NotLoaded : RuntimeState
    data object Loading : RuntimeState
    data class Ready(val metadata: ModelMetadata) : RuntimeState
    data class Running(val metadata: ModelMetadata) : RuntimeState
    data class Failed(val error: RuntimeError, val metadata: ModelMetadata? = null) : RuntimeState
    data object Closed : RuntimeState
}
