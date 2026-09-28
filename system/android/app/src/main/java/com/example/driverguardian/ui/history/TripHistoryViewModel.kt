package com.example.driverguardian.ui.history

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.example.driverguardian.data.repository.DriverGuardianRepository
import com.example.driverguardian.data.repository.RepositoryResult
import com.example.driverguardian.domain.model.DrowsinessEvent
import com.example.driverguardian.domain.model.TripSession
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

sealed interface TripLoadState {
    data object Idle : TripLoadState
    data object Loading : TripLoadState
    data object Success : TripLoadState
    data class Empty(val message: String) : TripLoadState
    data class Error(val message: String) : TripLoadState
}

data class TripHistoryUiState(
    val historyState: TripLoadState = TripLoadState.Idle,
    val trips: List<TripSession> = emptyList(),
    val detailState: TripLoadState = TripLoadState.Idle,
    val selectedTrip: TripSession? = null,
    val selectedEvents: List<DrowsinessEvent> = emptyList(),
    val requestedSessionId: Int? = null
)

class TripHistoryViewModel(private val repository: DriverGuardianRepository) : ViewModel() {
    private val mutableUiState = MutableStateFlow(TripHistoryUiState())
    val uiState: StateFlow<TripHistoryUiState> = mutableUiState.asStateFlow()

    fun loadHistory(driverId: Int? = null) {
        mutableUiState.update { it.copy(historyState = TripLoadState.Loading, trips = emptyList()) }
        viewModelScope.launch {
            when (val result = repository.getSessions(driverId = driverId, status = "COMPLETED", limit = 100)) {
                is RepositoryResult.Error -> mutableUiState.update {
                    it.copy(historyState = TripLoadState.Error(result.message))
                }
                is RepositoryResult.Success -> mutableUiState.update {
                    if (result.value.isEmpty()) {
                        it.copy(historyState = TripLoadState.Empty("Chưa có chuyến đi đã hoàn thành."), trips = emptyList())
                    } else {
                        it.copy(historyState = TripLoadState.Success, trips = result.value)
                    }
                }
            }
        }
    }

    fun loadDetail(sessionId: Int) {
        mutableUiState.update {
            it.copy(
                detailState = TripLoadState.Loading,
                selectedTrip = null,
                selectedEvents = emptyList(),
                requestedSessionId = sessionId
            )
        }
        viewModelScope.launch {
            when (val detail = repository.getSession(sessionId)) {
                is RepositoryResult.Error -> updateDetailIfCurrent(sessionId) {
                    it.copy(detailState = TripLoadState.Error(detail.message))
                }
                is RepositoryResult.Success -> when (val events = repository.getSessionEvents(sessionId)) {
                    is RepositoryResult.Error -> updateDetailIfCurrent(sessionId) {
                        it.copy(detailState = TripLoadState.Error(events.message))
                    }
                    is RepositoryResult.Success -> updateDetailIfCurrent(sessionId) {
                        it.copy(
                            detailState = TripLoadState.Success,
                            selectedTrip = detail.value,
                            selectedEvents = events.value
                        )
                    }
                }
            }
        }
    }

    private inline fun updateDetailIfCurrent(
        sessionId: Int,
        transform: (TripHistoryUiState) -> TripHistoryUiState
    ) {
        mutableUiState.update { if (it.requestedSessionId == sessionId) transform(it) else it }
    }
}

class TripHistoryViewModelFactory(
    private val repository: DriverGuardianRepository
) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        require(modelClass.isAssignableFrom(TripHistoryViewModel::class.java))
        return TripHistoryViewModel(repository) as T
    }
}
