package com.example.driverguardian.ui.session

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.example.driverguardian.data.repository.DriverGuardianRepository
import com.example.driverguardian.data.repository.RepositoryResult
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class DrivingSessionViewModel(
    private val repository: DriverGuardianRepository
) : ViewModel() {
    private val mutableUiState = MutableStateFlow(DrivingSessionUiState())
    val uiState: StateFlow<DrivingSessionUiState> = mutableUiState.asStateFlow()

    init {
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            mutableUiState.value = DrivingSessionUiState(loadState = LoadState.Loading)
            val driversResult = repository.getDrivers()
            val vehiclesResult = repository.getVehicles()
            val modelResult = repository.getActiveModelVersion()
            val error = listOf(driversResult, vehiclesResult, modelResult)
                .filterIsInstance<RepositoryResult.Error>()
                .firstOrNull()
            if (error != null) {
                mutableUiState.update { it.copy(loadState = LoadState.Error(error.message)) }
                return@launch
            }

            val drivers = (driversResult as RepositoryResult.Success).value
            val vehicles = (vehiclesResult as RepositoryResult.Success).value
            val model = (modelResult as RepositoryResult.Success).value
            val loadState = if (drivers.isEmpty() || vehicles.isEmpty()) {
                LoadState.Empty("Chưa có đủ tài xế hoặc phương tiện để bắt đầu chuyến đi.")
            } else {
                LoadState.Success
            }
            mutableUiState.value = DrivingSessionUiState(
                loadState = loadState,
                drivers = drivers,
                vehicles = vehicles,
                activeModel = model
            )
        }
    }

    fun selectDriver(driverId: Int) {
        mutableUiState.update { state ->
            if (state.drivers.any { it.id == driverId }) state.copy(selectedDriverId = driverId) else state
        }
    }

    fun selectVehicle(vehicleId: Int) {
        mutableUiState.update { state ->
            if (state.vehicles.any { it.id == vehicleId }) state.copy(selectedVehicleId = vehicleId) else state
        }
    }

    fun createSession() {
        val state = mutableUiState.value
        if (!state.canContinue || state.sessionSubmissionState == SessionSubmissionState.Submitting) return
        val driverId = requireNotNull(state.selectedDriverId)
        val vehicleId = requireNotNull(state.selectedVehicleId)
        val modelId = requireNotNull(state.activeModel).id
        mutableUiState.update { it.copy(sessionSubmissionState = SessionSubmissionState.Submitting) }
        viewModelScope.launch {
            when (val result = repository.createSession(driverId, vehicleId, modelId)) {
                is RepositoryResult.Success -> mutableUiState.update {
                    it.copy(
                        activeSession = result.value,
                        lastEvent = null,
                        completedSession = null,
                        completedSessionEvents = emptyList(),
                        eventSubmissionState = EventSubmissionState.Idle,
                        completionState = CompletionState.Idle,
                        acknowledgementState = AcknowledgementState.Idle,
                        sessionSubmissionState = SessionSubmissionState.Success
                    )
                }
                is RepositoryResult.Error -> mutableUiState.update {
                    it.copy(sessionSubmissionState = SessionSubmissionState.Error(result.message))
                }
            }
        }
    }

    fun submitDangerEvent(confidence: Double?, durationMs: Int?) {
        submitEvent(DriverState.Danger, confidence, durationMs)
    }

    fun finishSession() {
        val currentState = mutableUiState.value
        val session = currentState.activeSession
        if (session == null) {
            mutableUiState.update { it.copy(completionState = CompletionState.Error("Chưa có phiên lái đang hoạt động.")) }
            return
        }
        if (currentState.completionState == CompletionState.Submitting) return
        val alreadyCompleted = currentState.completedSession?.takeIf { it.id == session.id }
        mutableUiState.update { it.copy(completionState = CompletionState.Submitting) }
        viewModelScope.launch {
            if (alreadyCompleted != null) {
                loadCompletedSessionEvents(alreadyCompleted)
                return@launch
            }
            when (val completion = repository.completeSession(session.id)) {
                is RepositoryResult.Error -> updateCompletionIfCurrent(session.id) {
                    it.copy(completionState = CompletionState.Error(completion.message))
                }
                is RepositoryResult.Success -> {
                    if (mutableUiState.value.activeSession?.id != session.id) return@launch
                    mutableUiState.update {
                        it.copy(
                            activeSession = completion.value,
                            completedSession = completion.value,
                            completedSessionEvents = emptyList()
                        )
                    }
                    loadCompletedSessionEvents(completion.value)
                }
            }
        }
    }

    private suspend fun loadCompletedSessionEvents(completedSession: com.example.driverguardian.domain.model.DrivingSession) {
        when (val events = repository.getSessionEvents(completedSession.id)) {
            is RepositoryResult.Error -> updateCompletionIfCurrent(completedSession.id) {
                it.copy(completionState = CompletionState.Error(events.message))
            }
            is RepositoryResult.Success -> updateCompletionIfCurrent(completedSession.id) {
                it.copy(
                    completedSessionEvents = events.value,
                    completionState = CompletionState.Success
                )
            }
        }
    }

    fun acknowledgeLastEvent() {
        val event = mutableUiState.value.lastEvent
        if (event == null) {
            mutableUiState.update { it.copy(acknowledgementState = AcknowledgementState.Error("Không có cảnh báo cần xác nhận.")) }
            return
        }
        if (mutableUiState.value.acknowledgementState == AcknowledgementState.Submitting) return
        mutableUiState.update { it.copy(acknowledgementState = AcknowledgementState.Submitting) }
        viewModelScope.launch {
            when (val result = repository.acknowledgeEvent(event.id)) {
                is RepositoryResult.Success -> mutableUiState.update {
                    if (it.lastEvent?.id != event.id) it else it.copy(
                        lastEvent = result.value,
                        acknowledgementState = AcknowledgementState.Success
                    )
                }
                is RepositoryResult.Error -> mutableUiState.update {
                    if (it.lastEvent?.id != event.id) it else {
                        it.copy(acknowledgementState = AcknowledgementState.Error(result.message))
                    }
                }
            }
        }
    }

    private inline fun updateCompletionIfCurrent(
        sessionId: Int,
        transform: (DrivingSessionUiState) -> DrivingSessionUiState
    ) {
        mutableUiState.update { if (it.activeSession?.id == sessionId) transform(it) else it }
    }

    fun submitEvent(driverState: DriverState, confidence: Double?, durationMs: Int?) {
        val session = mutableUiState.value.activeSession
        if (session == null) {
            mutableUiState.update {
                it.copy(eventSubmissionState = EventSubmissionState.Error("Chưa có phiên lái đang hoạt động."))
            }
            return
        }
        if (mutableUiState.value.eventSubmissionState == EventSubmissionState.Submitting) return
        mutableUiState.update { it.copy(eventSubmissionState = EventSubmissionState.Submitting) }
        viewModelScope.launch {
            when (val result = repository.createEvent(
                session.id,
                driverState.wireValue,
                driverState.alertLevel,
                confidence,
                durationMs
            )) {
                is RepositoryResult.Success -> mutableUiState.update {
                    if (it.activeSession?.id != session.id) {
                        it
                    } else {
                        val active = requireNotNull(it.activeSession)
                        it.copy(
                            activeSession = active.copy(totalAlerts = active.totalAlerts + 1),
                            lastEvent = result.value,
                            eventSubmissionState = EventSubmissionState.Success(result.value)
                        )
                    }
                }
                is RepositoryResult.Error -> mutableUiState.update {
                    if (it.activeSession?.id != session.id) it else {
                        it.copy(eventSubmissionState = EventSubmissionState.Error(result.message))
                    }
                }
            }
        }
    }

    fun consumeSessionSubmission() {
        mutableUiState.update { it.copy(sessionSubmissionState = SessionSubmissionState.Idle) }
    }

    fun consumeEventSubmission() {
        mutableUiState.update { it.copy(eventSubmissionState = EventSubmissionState.Idle) }
    }

    fun consumeAcknowledgement() {
        mutableUiState.update { it.copy(acknowledgementState = AcknowledgementState.Idle) }
    }

    fun consumeCompletion() {
        mutableUiState.update { it.copy(completionState = CompletionState.Idle) }
    }
}

class DrivingSessionViewModelFactory(
    private val repository: DriverGuardianRepository
) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        require(modelClass.isAssignableFrom(DrivingSessionViewModel::class.java))
        return DrivingSessionViewModel(repository) as T
    }
}
