package com.example.driverguardian.ui.session

import com.example.driverguardian.data.repository.DriverGuardianRepository
import com.example.driverguardian.data.repository.RepositoryResult
import com.example.driverguardian.domain.model.Driver
import com.example.driverguardian.domain.model.DrowsinessEvent
import com.example.driverguardian.domain.model.DrivingSession
import com.example.driverguardian.domain.model.ModelVersion
import com.example.driverguardian.domain.model.TripSession
import com.example.driverguardian.domain.model.Vehicle
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class DrivingSessionViewModelTest {
    private val dispatcher = StandardTestDispatcher()

    @Before fun setUp() = Dispatchers.setMain(dispatcher)
    @After fun tearDown() = Dispatchers.resetMain()

    @Test
    fun `load succeeds and continue requires both selections`() = runTest(dispatcher) {
        val viewModel = DrivingSessionViewModel(FakeRepository())
        advanceUntilIdle()
        assertEquals(LoadState.Success, viewModel.uiState.value.loadState)
        assertFalse(viewModel.uiState.value.canContinue)
        viewModel.selectDriver(1)
        viewModel.selectVehicle(2)
        assertTrue(viewModel.uiState.value.canContinue)
    }

    @Test
    fun `empty data has no mock fallback`() = runTest(dispatcher) {
        val viewModel = DrivingSessionViewModel(FakeRepository(drivers = emptyList()))
        advanceUntilIdle()
        assertTrue(viewModel.uiState.value.loadState is LoadState.Empty)
        assertTrue(viewModel.uiState.value.drivers.isEmpty())
        assertFalse(viewModel.uiState.value.canContinue)
    }

    @Test
    fun `load failure exposes error and no reference data`() = runTest(dispatcher) {
        val viewModel = DrivingSessionViewModel(
            FakeRepository(driverResult = RepositoryResult.Error("Không thể kết nối tới máy chủ Driver Guardian."))
        )
        advanceUntilIdle()
        assertEquals(
            LoadState.Error("Không thể kết nối tới máy chủ Driver Guardian."),
            viewModel.uiState.value.loadState
        )
        assertTrue(viewModel.uiState.value.drivers.isEmpty())
    }

    @Test
    fun `session success stores session while failure stays explicit`() = runTest(dispatcher) {
        val successful = readyViewModel(FakeRepository())
        successful.createSession()
        advanceUntilIdle()
        assertNotNull(successful.uiState.value.activeSession)
        assertEquals(SessionSubmissionState.Success, successful.uiState.value.sessionSubmissionState)

        val failed = readyViewModel(FakeRepository(sessionResult = RepositoryResult.Error("Không tạo được phiên.")))
        failed.createSession()
        advanceUntilIdle()
        assertEquals(SessionSubmissionState.Error("Không tạo được phiên."), failed.uiState.value.sessionSubmissionState)
        assertEquals(null, failed.uiState.value.activeSession)
    }

    @Test
    fun `duplicate session taps make one request while submission is in flight`() = runTest(dispatcher) {
        val gate = CompletableDeferred<Unit>()
        val repository = FakeRepository(sessionGate = gate)
        val viewModel = readyViewModel(repository)
        viewModel.createSession()
        viewModel.createSession()
        runCurrent()
        assertEquals(1, repository.sessionCalls)
        assertEquals(SessionSubmissionState.Submitting, viewModel.uiState.value.sessionSubmissionState)
        gate.complete(Unit)
        advanceUntilIdle()
        assertEquals(SessionSubmissionState.Success, viewModel.uiState.value.sessionSubmissionState)
    }

    @Test
    fun `event without active session is rejected before repository`() = runTest(dispatcher) {
        val repository = FakeRepository()
        val viewModel = DrivingSessionViewModel(repository)
        advanceUntilIdle()
        viewModel.submitDangerEvent(null, null)
        assertEquals(EventSubmissionState.Error("Chưa có phiên lái đang hoạt động."), viewModel.uiState.value.eventSubmissionState)
        assertEquals(0, repository.eventCalls)
    }

    @Test
    fun `duplicate event taps make one request and one count increment`() = runTest(dispatcher) {
        val gate = CompletableDeferred<Unit>()
        val repository = FakeRepository(eventGate = gate)
        val viewModel = readyViewModel(repository)
        viewModel.createSession()
        advanceUntilIdle()
        viewModel.submitDangerEvent(null, null)
        viewModel.submitDangerEvent(null, null)
        runCurrent()
        assertEquals(1, repository.eventCalls)
        gate.complete(Unit)
        advanceUntilIdle()
        assertEquals(1, viewModel.uiState.value.activeSession?.totalAlerts)
    }

    @Test
    fun `stale event completion cannot update a replacement session`() = runTest(dispatcher) {
        val gate = CompletableDeferred<Unit>()
        val repository = FakeRepository(eventGate = gate)
        val viewModel = readyViewModel(repository)
        viewModel.createSession()
        advanceUntilIdle()
        viewModel.submitDangerEvent(null, null)
        runCurrent()

        repository.sessionResult = RepositoryResult.Success(
            DrivingSession(8, 1, 2, 3, "2026-09-28T00:02:00", 0, 0, "ACTIVE", "SYNCED")
        )
        viewModel.createSession()
        runCurrent()
        assertEquals(8, viewModel.uiState.value.activeSession?.id)
        assertEquals(EventSubmissionState.Idle, viewModel.uiState.value.eventSubmissionState)

        gate.complete(Unit)
        advanceUntilIdle()
        assertEquals(0, viewModel.uiState.value.activeSession?.totalAlerts)
        assertEquals(null, viewModel.uiState.value.lastEvent)
        assertEquals(EventSubmissionState.Idle, viewModel.uiState.value.eventSubmissionState)
    }

    @Test
    fun `danger event uses level two and exposes failure`() = runTest(dispatcher) {
        val repository = FakeRepository(eventResult = RepositoryResult.Error("Không lưu được cảnh báo."))
        val viewModel = readyViewModel(repository)
        viewModel.createSession()
        advanceUntilIdle()
        viewModel.submitDangerEvent(null, null)
        advanceUntilIdle()
        assertEquals("DANGER", repository.lastDriverState)
        assertEquals(2, repository.lastAlertLevel)
        assertEquals(EventSubmissionState.Error("Không lưu được cảnh báo."), viewModel.uiState.value.eventSubmissionState)
    }

    @Test
    fun `warning maps to level one`() = runTest(dispatcher) {
        val repository = FakeRepository()
        val viewModel = readyViewModel(repository)
        viewModel.createSession()
        advanceUntilIdle()
        viewModel.submitEvent(DriverState.Warning, 0.75, 800)
        advanceUntilIdle()
        assertEquals("WARNING", repository.lastDriverState)
        assertEquals(1, repository.lastAlertLevel)
        assertTrue(viewModel.uiState.value.eventSubmissionState is EventSubmissionState.Success)
        assertNotNull(viewModel.uiState.value.lastEvent)
        assertEquals(1, viewModel.uiState.value.activeSession?.totalAlerts)
        viewModel.consumeEventSubmission()
        assertEquals(EventSubmissionState.Idle, viewModel.uiState.value.eventSubmissionState)
        assertNotNull(viewModel.uiState.value.lastEvent)
    }

    @Test
    fun `active identity remains bound to session after selection changes`() = runTest(dispatcher) {
        val drivers = listOf(
            Driver(1, "DRV-1", "Lan", null, null, "ACTIVE", "2026-09-28T00:00:00"),
            Driver(6, "DRV-6", "Minh", null, null, "ACTIVE", "2026-09-28T00:00:00")
        )
        val vehicles = listOf(
            Vehicle(2, "51A", "Bus", "BUS", null, "ACTIVE", "2026-09-28T00:00:00"),
            Vehicle(7, "29B", "Van", "VAN", null, "ACTIVE", "2026-09-28T00:00:00")
        )
        val viewModel = readyViewModel(FakeRepository(drivers = drivers, vehicles = vehicles))
        viewModel.createSession()
        advanceUntilIdle()
        viewModel.selectDriver(6)
        viewModel.selectVehicle(7)
        assertEquals(1, viewModel.uiState.value.activeDriver?.id)
        assertEquals(2, viewModel.uiState.value.activeVehicle?.id)
    }

    @Test
    fun `finish loads persisted events before exposing success`() = runTest(dispatcher) {
        val completed = DrivingSession(4, 1, 2, 3, "2026-09-28T00:00:00", 90, 1, "COMPLETED", "SYNCED", "2026-09-28T00:01:30", null)
        val event = DrowsinessEvent(5, 4, "2026-09-28T00:01:00", "DANGER", 2, null, null, "N", "SYNCED")
        val repository = FakeRepository(
            completionResult = RepositoryResult.Success(completed),
            sessionEventsResult = RepositoryResult.Success(listOf(event))
        )
        val viewModel = readyViewModel(repository)
        viewModel.createSession()
        advanceUntilIdle()
        viewModel.finishSession()
        advanceUntilIdle()
        assertEquals(CompletionState.Success, viewModel.uiState.value.completionState)
        assertEquals(completed, viewModel.uiState.value.completedSession)
        assertEquals(listOf(event), viewModel.uiState.value.completedSessionEvents)
    }

    @Test
    fun `duplicate finish taps make one request and failure stays explicit`() = runTest(dispatcher) {
        val gate = CompletableDeferred<Unit>()
        val repository = FakeRepository(
            completionResult = RepositoryResult.Error("Không kết thúc được chuyến đi."),
            completionGate = gate
        )
        val viewModel = readyViewModel(repository)
        viewModel.createSession()
        advanceUntilIdle()
        viewModel.finishSession()
        viewModel.finishSession()
        runCurrent()
        assertEquals(1, repository.completionCalls)
        gate.complete(Unit)
        advanceUntilIdle()
        assertEquals(CompletionState.Error("Không kết thúc được chuyến đi."), viewModel.uiState.value.completionState)
    }

    @Test
    fun `event load failure keeps completed session and retry does not complete twice`() = runTest(dispatcher) {
        val completed = DrivingSession(4, 1, 2, 3, "2026-09-28T00:00:00", 90, 1, "COMPLETED", "SYNCED", "2026-09-28T00:01:30", null)
        val event = DrowsinessEvent(5, 4, "2026-09-28T00:01:00", "DANGER", 2, null, null, "N", "SYNCED")
        val repository = FakeRepository(
            completionResult = RepositoryResult.Success(completed),
            sessionEventsResult = RepositoryResult.Error("Không tải được sự kiện.")
        )
        val viewModel = readyViewModel(repository)
        viewModel.createSession()
        advanceUntilIdle()
        viewModel.finishSession()
        advanceUntilIdle()
        assertEquals(completed, viewModel.uiState.value.completedSession)
        assertEquals(CompletionState.Error("Không tải được sự kiện."), viewModel.uiState.value.completionState)

        repository.sessionEventsResult = RepositoryResult.Success(listOf(event))
        viewModel.finishSession()
        advanceUntilIdle()
        assertEquals(1, repository.completionCalls)
        assertEquals(listOf(event), viewModel.uiState.value.completedSessionEvents)
        assertEquals(CompletionState.Success, viewModel.uiState.value.completionState)
    }

    @Test
    fun `stale completion cannot replace a newer active session`() = runTest(dispatcher) {
        val gate = CompletableDeferred<Unit>()
        val completed = DrivingSession(4, 1, 2, 3, "2026-09-28T00:00:00", 90, 0, "COMPLETED", "SYNCED", "2026-09-28T00:01:30", null)
        val repository = FakeRepository(completionResult = RepositoryResult.Success(completed), completionGate = gate)
        val viewModel = readyViewModel(repository)
        viewModel.createSession()
        advanceUntilIdle()
        viewModel.finishSession()
        runCurrent()
        repository.sessionResult = RepositoryResult.Success(DrivingSession(8, 1, 2, 3, "2026-09-28T00:02:00", 0, 0, "ACTIVE", "SYNCED"))
        viewModel.createSession()
        advanceUntilIdle()
        gate.complete(Unit)
        advanceUntilIdle()
        assertEquals(8, viewModel.uiState.value.activeSession?.id)
        assertEquals(null, viewModel.uiState.value.completedSession)
    }

    @Test
    fun `acknowledge is guarded and updates the last event only after success`() = runTest(dispatcher) {
        val gate = CompletableDeferred<Unit>()
        val acknowledged = DrowsinessEvent(5, 4, "2026-09-28T00:01:00", "DANGER", 2, null, null, "Y", "SYNCED")
        val repository = FakeRepository(ackResult = RepositoryResult.Success(acknowledged), ackGate = gate)
        val viewModel = readyViewModel(repository)
        viewModel.createSession()
        advanceUntilIdle()
        viewModel.submitDangerEvent(null, null)
        advanceUntilIdle()
        viewModel.acknowledgeLastEvent()
        viewModel.acknowledgeLastEvent()
        runCurrent()
        assertEquals(1, repository.ackCalls)
        gate.complete(Unit)
        advanceUntilIdle()
        assertEquals(AcknowledgementState.Success, viewModel.uiState.value.acknowledgementState)
        assertEquals("Y", viewModel.uiState.value.lastEvent?.acknowledged)
    }

    private suspend fun TestScope.readyViewModel(repository: FakeRepository): DrivingSessionViewModel {
        val viewModel = DrivingSessionViewModel(repository)
        advanceUntilIdle()
        viewModel.selectDriver(1)
        viewModel.selectVehicle(2)
        return viewModel
    }
}

private class FakeRepository(
    private val drivers: List<Driver> = listOf(Driver(1, "DRV-1", "Lan", null, null, "ACTIVE", "2026-09-28T00:00:00")),
    private val vehicles: List<Vehicle> = listOf(Vehicle(2, "51A", "Bus", "BUS", null, "ACTIVE", "2026-09-28T00:00:00")),
    private val model: ModelVersion = ModelVersion(3, "v1", null, null, null, "Y", "2026-09-28T00:00:00"),
    sessionResult: RepositoryResult<DrivingSession> = RepositoryResult.Success(DrivingSession(4, 1, 2, 3, "2026-09-28T00:00:00", 0, 0, "ACTIVE", "SYNCED")),
    private val eventResult: RepositoryResult<DrowsinessEvent> = RepositoryResult.Success(DrowsinessEvent(5, 4, "2026-09-28T00:01:00", "DANGER", 2, null, null, "N", "SYNCED")),
    private val driverResult: RepositoryResult<List<Driver>>? = null,
    private val sessionGate: CompletableDeferred<Unit>? = null,
    private val eventGate: CompletableDeferred<Unit>? = null,
    private val completionResult: RepositoryResult<DrivingSession> = RepositoryResult.Error("Not configured"),
    sessionEventsResult: RepositoryResult<List<DrowsinessEvent>> = RepositoryResult.Success(emptyList()),
    private val ackResult: RepositoryResult<DrowsinessEvent> = RepositoryResult.Error("Not configured"),
    private val completionGate: CompletableDeferred<Unit>? = null,
    private val ackGate: CompletableDeferred<Unit>? = null
) : DriverGuardianRepository {
    var sessionResult = sessionResult
    var lastDriverState: String? = null
    var lastAlertLevel: Int? = null
    var sessionCalls = 0
    var eventCalls = 0
    var completionCalls = 0
    var ackCalls = 0
    var sessionEventsResult = sessionEventsResult

    override suspend fun getDrivers() = driverResult ?: RepositoryResult.Success(drivers)
    override suspend fun getVehicles() = RepositoryResult.Success(vehicles)
    override suspend fun getActiveModelVersion() = RepositoryResult.Success(model)
    override suspend fun createSession(driverId: Int, vehicleId: Int, modelVersionId: Int): RepositoryResult<DrivingSession> {
        sessionCalls += 1
        sessionGate?.await()
        return sessionResult
    }
    override suspend fun completeSession(sessionId: Int): RepositoryResult<DrivingSession> {
        completionCalls += 1
        completionGate?.await()
        return completionResult
    }
    override suspend fun getSessions(driverId: Int?, status: String?, limit: Int?): RepositoryResult<List<TripSession>> =
        RepositoryResult.Error("Not configured")
    override suspend fun getSession(sessionId: Int): RepositoryResult<TripSession> =
        RepositoryResult.Error("Not configured")
    override suspend fun getSessionEvents(sessionId: Int): RepositoryResult<List<DrowsinessEvent>> = sessionEventsResult
    override suspend fun acknowledgeEvent(eventId: Int): RepositoryResult<DrowsinessEvent> {
        ackCalls += 1
        ackGate?.await()
        return ackResult
    }
    override suspend fun createEvent(sessionId: Int, driverState: String, alertLevel: Int, confidence: Double?, durationMs: Int?): RepositoryResult<DrowsinessEvent> {
        eventCalls += 1
        lastDriverState = driverState
        lastAlertLevel = alertLevel
        eventGate?.await()
        return eventResult
    }
}
