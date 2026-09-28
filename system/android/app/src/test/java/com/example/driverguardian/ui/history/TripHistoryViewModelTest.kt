package com.example.driverguardian.ui.history

import com.example.driverguardian.data.repository.DriverGuardianRepository
import com.example.driverguardian.data.repository.RepositoryResult
import com.example.driverguardian.domain.model.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class TripHistoryViewModelTest {
    private val dispatcher = StandardTestDispatcher()
    @Before fun setUp() = Dispatchers.setMain(dispatcher)
    @After fun tearDown() = Dispatchers.resetMain()

    @Test fun `history exposes real success empty and error states`() = runTest(dispatcher) {
        val trip = trip(4)
        val success = TripHistoryViewModel(HistoryFake(history = RepositoryResult.Success(listOf(trip))))
        success.loadHistory(1)
        advanceUntilIdle()
        assertEquals(listOf(trip), success.uiState.value.trips)
        assertEquals(TripLoadState.Success, success.uiState.value.historyState)

        val empty = TripHistoryViewModel(HistoryFake(history = RepositoryResult.Success(emptyList())))
        empty.loadHistory(1)
        advanceUntilIdle()
        assertTrue(empty.uiState.value.historyState is TripLoadState.Empty)

        val failed = TripHistoryViewModel(HistoryFake(history = RepositoryResult.Error("offline")))
        failed.loadHistory(1)
        advanceUntilIdle()
        assertEquals(TripLoadState.Error("offline"), failed.uiState.value.historyState)
    }

    @Test fun `detail is ready only with server detail and events`() = runTest(dispatcher) {
        val trip = trip(4)
        val event = DrowsinessEvent(7, 4, "2026-09-28T00:01:00", "DANGER", 2, null, null, "N", "SYNCED", null, null, null, null)
        val viewModel = TripHistoryViewModel(HistoryFake(detail = RepositoryResult.Success(trip), events = RepositoryResult.Success(listOf(event))))
        viewModel.loadDetail(4)
        advanceUntilIdle()
        assertEquals(TripLoadState.Success, viewModel.uiState.value.detailState)
        assertEquals(trip, viewModel.uiState.value.selectedTrip)
        assertEquals(listOf(event), viewModel.uiState.value.selectedEvents)
        assertEquals(null, viewModel.uiState.value.selectedTrip?.safetyScore)
        assertEquals(null, viewModel.uiState.value.selectedEvents.single().confidence)
    }

    private fun trip(id: Int) = TripSession(id, 1, "Lan", 2, null, "51A", 3, null, "2026-09-28T00:00:00", null, 0, 0, null, "ACTIVE", "SYNCED")
}

private class HistoryFake(
    private val history: RepositoryResult<List<TripSession>> = RepositoryResult.Success(emptyList()),
    private val detail: RepositoryResult<TripSession> = RepositoryResult.Error("not configured"),
    private val events: RepositoryResult<List<DrowsinessEvent>> = RepositoryResult.Success(emptyList())
) : DriverGuardianRepository {
    override suspend fun getSessions(driverId: Int?, status: String?, limit: Int?) = history
    override suspend fun getSession(sessionId: Int) = detail
    override suspend fun getSessionEvents(sessionId: Int) = events
    override suspend fun getDrivers() = RepositoryResult.Success(emptyList<Driver>())
    override suspend fun getVehicles() = RepositoryResult.Success(emptyList<Vehicle>())
    override suspend fun getActiveModelVersion() = RepositoryResult.Error("unused")
    override suspend fun createSession(driverId: Int, vehicleId: Int, modelVersionId: Int) = RepositoryResult.Error("unused")
    override suspend fun completeSession(sessionId: Int) = RepositoryResult.Error("unused")
    override suspend fun acknowledgeEvent(eventId: Int) = RepositoryResult.Error("unused")
    override suspend fun createEvent(sessionId: Int, driverState: String, alertLevel: Int, confidence: Double?, durationMs: Int?) = RepositoryResult.Error("unused")
}
