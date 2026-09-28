package com.example.driverguardian.ui.session

import com.example.driverguardian.domain.model.Driver
import com.example.driverguardian.domain.model.DrowsinessEvent
import com.example.driverguardian.domain.model.DrivingSession
import com.example.driverguardian.domain.model.ModelVersion
import com.example.driverguardian.domain.model.Vehicle

sealed interface LoadState {
    data object Loading : LoadState
    data object Success : LoadState
    data class Empty(val message: String) : LoadState
    data class Error(val message: String) : LoadState
}

sealed interface SessionSubmissionState {
    data object Idle : SessionSubmissionState
    data object Submitting : SessionSubmissionState
    data object Success : SessionSubmissionState
    data class Error(val message: String) : SessionSubmissionState
}

sealed interface EventSubmissionState {
    data object Idle : EventSubmissionState
    data object Submitting : EventSubmissionState
    data class Success(val event: DrowsinessEvent) : EventSubmissionState
    data class Error(val message: String) : EventSubmissionState
}

enum class DriverState(val wireValue: String, val alertLevel: Int) {
    Warning("WARNING", 1),
    Danger("DANGER", 2)
}

data class DrivingSessionUiState(
    val loadState: LoadState = LoadState.Loading,
    val drivers: List<Driver> = emptyList(),
    val vehicles: List<Vehicle> = emptyList(),
    val activeModel: ModelVersion? = null,
    val selectedDriverId: Int? = null,
    val selectedVehicleId: Int? = null,
    val activeSession: DrivingSession? = null,
    val lastEvent: DrowsinessEvent? = null,
    val sessionSubmissionState: SessionSubmissionState = SessionSubmissionState.Idle,
    val eventSubmissionState: EventSubmissionState = EventSubmissionState.Idle
) {
    val selectedDriver: Driver? get() = drivers.firstOrNull { it.id == selectedDriverId }
    val selectedVehicle: Vehicle? get() = vehicles.firstOrNull { it.id == selectedVehicleId }
    val activeDriver: Driver? get() = activeSession?.let { session -> drivers.firstOrNull { it.id == session.driverId } }
    val activeVehicle: Vehicle? get() = activeSession?.let { session -> vehicles.firstOrNull { it.id == session.vehicleId } }
    val canContinue: Boolean
        get() = loadState == LoadState.Success && selectedDriver != null && selectedVehicle != null && activeModel != null
}
