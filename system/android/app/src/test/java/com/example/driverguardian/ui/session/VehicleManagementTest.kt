package com.example.driverguardian.ui.session

import com.example.driverguardian.data.repository.DriverGuardianRepository
import com.example.driverguardian.data.repository.RepositoryResult
import com.example.driverguardian.domain.model.Driver
import com.example.driverguardian.domain.model.DriverSummary
import com.example.driverguardian.domain.model.DrowsinessEvent
import com.example.driverguardian.domain.model.DrivingSession
import com.example.driverguardian.domain.model.ModelVersion
import com.example.driverguardian.domain.model.TripSession
import com.example.driverguardian.domain.model.Vehicle
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class VehicleManagementTest {

    private val testDispatcher = StandardTestDispatcher()

    private class VehicleFakeRepository(
        var vehiclesList: MutableList<Vehicle> = mutableListOf(),
        var driversList: List<Driver> = listOf(Driver(1, "DRV0005", "User Driver", null, null, "ACTIVE", "")),
        var activeModel: ModelVersion = ModelVersion(1, "v1", null, null, null, "Y", "")
    ) : DriverGuardianRepository {
        var createVehicleCalls = 0
        var updateVehicleCalls = 0

        override suspend fun getDrivers() = RepositoryResult.Success(driversList)
        override suspend fun getVehicles() = RepositoryResult.Success(vehiclesList.toList())
        override suspend fun getActiveModelVersion() = RepositoryResult.Success(activeModel)

        override suspend fun createVehicle(
            plateNumber: String,
            vehicleName: String,
            vehicleType: String,
            deviceCode: String?
        ): RepositoryResult<Vehicle> {
            createVehicleCalls++
            val v = Vehicle(
                id = vehiclesList.size + 100,
                plateNumber = plateNumber,
                name = vehicleName,
                type = vehicleType,
                deviceCode = deviceCode,
                status = "ACTIVE",
                createdAt = "2026-10-04T00:00:00",
                driverId = 1
            )
            vehiclesList.add(v)
            return RepositoryResult.Success(v)
        }

        override suspend fun updateVehicle(
            vehicleId: Int,
            plateNumber: String?,
            vehicleName: String?,
            vehicleType: String?,
            deviceCode: String?
        ): RepositoryResult<Vehicle> {
            updateVehicleCalls++
            val existingIndex = vehiclesList.indexOfFirst { it.id == vehicleId }
            if (existingIndex < 0) return RepositoryResult.Error("Vehicle not found", 404)
            val old = vehiclesList[existingIndex]
            val updated = old.copy(
                plateNumber = plateNumber ?: old.plateNumber,
                name = vehicleName ?: old.name,
                type = vehicleType ?: old.type,
                deviceCode = deviceCode ?: old.deviceCode
            )
            vehiclesList[existingIndex] = updated
            return RepositoryResult.Success(updated)
        }

        override suspend fun createSession(driverId: Int, vehicleId: Int, modelVersionId: Int) = RepositoryResult.Error("unused")
        override suspend fun completeSession(sessionId: Int) = RepositoryResult.Error("unused")
        override suspend fun getSessions(driverId: Int?, status: String?, limit: Int?) = RepositoryResult.Error("unused")
        override suspend fun getSession(sessionId: Int) = RepositoryResult.Error("unused")
        override suspend fun getSessionEvents(sessionId: Int) = RepositoryResult.Error("unused")
        override suspend fun acknowledgeEvent(eventId: Int) = RepositoryResult.Error("unused")
        override suspend fun createEvent(sessionId: Int, driverState: String, alertLevel: Int, confidence: Double?, durationMs: Int?) = RepositoryResult.Error("unused")
    }

    private lateinit var fakeRepo: VehicleFakeRepository
    private lateinit var viewModel: DrivingSessionViewModel

    @Before
    fun setUp() {
        Dispatchers.setMain(testDispatcher)
        fakeRepo = VehicleFakeRepository()
        viewModel = DrivingSessionViewModel(fakeRepo)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun `zero vehicles allows loadState Success with empty vehicle list`() = runTest {
        viewModel.setAuthenticatedDriver(DriverSummary(5, "DRV0005", "Nguyen Test User"))
        testDispatcher.scheduler.advanceUntilIdle()

        val state = viewModel.uiState.value
        assertEquals(LoadState.Success, state.loadState)
        assertTrue(state.vehicles.isEmpty())
        assertNull(state.selectedVehicleId)
        assertFalse(state.canContinue) // Cannot continue without selected vehicle
    }

    @Test
    fun `adding vehicle appends to list and auto-selects newly created vehicle`() = runTest {
        viewModel.setAuthenticatedDriver(DriverSummary(5, "DRV0005", "Nguyen Test User"))
        testDispatcher.scheduler.advanceUntilIdle()

        var callbackSuccess = false
        viewModel.addVehicle("29A-88888", "Mazda CX-5", "SUV", "DEV01") { success, _ ->
            callbackSuccess = success
        }
        testDispatcher.scheduler.advanceUntilIdle()

        assertTrue(callbackSuccess)
        val state = viewModel.uiState.value
        assertEquals(1, state.vehicles.size)
        assertEquals("29A-88888", state.vehicles.first().plateNumber)
        assertEquals(state.vehicles.first().id, state.selectedVehicleId)
        assertTrue(state.canContinue)
    }

    @Test
    fun `single vehicle is auto-selected on refresh`() = runTest {
        fakeRepo.vehiclesList.add(
            Vehicle(10, "51A-99999", "Honda CR-V", "SUV", null, "ACTIVE", "", 5)
        )
        viewModel.setAuthenticatedDriver(DriverSummary(5, "DRV0005", "Nguyen Test User"))
        viewModel.refresh()
        testDispatcher.scheduler.advanceUntilIdle()

        val state = viewModel.uiState.value
        assertEquals(1, state.vehicles.size)
        assertEquals(10, state.selectedVehicleId)
        assertTrue(state.canContinue)
    }

    @Test
    fun `multiple vehicles allow selecting specific vehicle`() = runTest {
        fakeRepo.vehiclesList.add(Vehicle(10, "51A-11111", "Car 1", "Sedan", null, "ACTIVE", "", 5))
        fakeRepo.vehiclesList.add(Vehicle(20, "51A-22222", "Car 2", "SUV", null, "ACTIVE", "", 5))
        viewModel.setAuthenticatedDriver(DriverSummary(5, "DRV0005", "Nguyen Test User"))
        viewModel.refresh()
        testDispatcher.scheduler.advanceUntilIdle()

        val state = viewModel.uiState.value
        assertEquals(2, state.vehicles.size)

        viewModel.selectVehicle(20)
        assertEquals(20, viewModel.uiState.value.selectedVehicleId)

        viewModel.selectVehicle(10)
        assertEquals(10, viewModel.uiState.value.selectedVehicleId)
    }

    @Test
    fun `updating vehicle updates name and plate in UI state`() = runTest {
        fakeRepo.vehiclesList.add(Vehicle(10, "51A-11111", "Old Name", "Sedan", null, "ACTIVE", "", 5))
        viewModel.refresh()
        testDispatcher.scheduler.advanceUntilIdle()

        var callbackSuccess = false
        viewModel.updateVehicle(10, "51A-11111", "New Name", "Sedan", null) { success, _ ->
            callbackSuccess = success
        }
        testDispatcher.scheduler.advanceUntilIdle()

        assertTrue(callbackSuccess)
        val vehicle = viewModel.uiState.value.vehicles.first { it.id == 10 }
        assertEquals("New Name", vehicle.name)
    }

    @Test
    fun `authenticated driver card uses actual logged in driver profile`() = runTest {
        val customDriver = DriverSummary(77, "DRV0077", "Pham Hoang Long")
        viewModel.setAuthenticatedDriver(customDriver)

        val state = viewModel.uiState.value
        assertNotNull(state.authenticatedDriver)
        assertEquals("Pham Hoang Long", state.authenticatedDriver?.fullName)
        assertEquals("DRV0077", state.authenticatedDriver?.driverCode)
        assertEquals(77, state.selectedDriverId)
    }
}
