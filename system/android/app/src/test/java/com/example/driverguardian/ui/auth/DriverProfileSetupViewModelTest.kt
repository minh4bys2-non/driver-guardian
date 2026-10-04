package com.example.driverguardian.ui.auth

import com.example.driverguardian.data.auth.AuthState
import com.example.driverguardian.data.repository.AuthRepository
import com.example.driverguardian.data.repository.RepositoryResult
import com.example.driverguardian.domain.model.DriverSummary
import com.example.driverguardian.domain.model.UserProfile
import com.example.driverguardian.ui.screens.auth.DriverProfileSetupViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
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
class DriverProfileSetupViewModelTest {

    private val testDispatcher = StandardTestDispatcher()
    private val fakeAuthState = MutableStateFlow<AuthState>(AuthState.Unauthenticated)

    private class FakeAuthRepo(
        override val authState: StateFlow<AuthState>
    ) : AuthRepository {
        var createProfileResult: RepositoryResult<UserProfile> = RepositoryResult.Success(
            UserProfile(1, "user@test.com", "Test Driver", null, "DRIVER", DriverSummary(10, "DRV0010", "Test Driver"))
        )
        var lastFullName: String? = null
        var lastPhone: String? = null
        var lastLicense: String? = null
        var userProfile: UserProfile? = UserProfile(1, "user@test.com", "Google Account Name", null, "DRIVER", null)
        var logoutCalled = false

        override suspend fun loginWithGoogle(idToken: String) = RepositoryResult.Error("unused")
        override suspend fun restoreSession() = RepositoryResult.Success(userProfile)
        override suspend fun logout(): RepositoryResult<Unit> {
            logoutCalled = true
            return RepositoryResult.Success(Unit)
        }
        override fun getCurrentUser() = userProfile

        override suspend fun createDriverProfile(
            fullName: String,
            phoneNumber: String?,
            licenseNumber: String
        ): RepositoryResult<UserProfile> {
            lastFullName = fullName
            lastPhone = phoneNumber
            lastLicense = licenseNumber
            return createProfileResult
        }
    }

    private lateinit var fakeRepo: FakeAuthRepo
    private lateinit var viewModel: DriverProfileSetupViewModel

    @Before
    fun setUp() {
        Dispatchers.setMain(testDispatcher)
        fakeRepo = FakeAuthRepo(fakeAuthState)
        viewModel = DriverProfileSetupViewModel(fakeRepo)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun `initial state pre-fills display name from user profile`() {
        assertEquals("Google Account Name", viewModel.uiState.value.fullName)
        assertEquals("", viewModel.uiState.value.phoneNumber)
        assertEquals("", viewModel.uiState.value.licenseNumber)
        assertFalse(viewModel.uiState.value.isLoading)
        assertNull(viewModel.uiState.value.errorMessage)
    }

    @Test
    fun `validation fails on empty full name`() {
        viewModel.onFullNameChange("   ")
        viewModel.onLicenseNumberChange("GPLX-12345")
        val valid = viewModel.validate()

        assertFalse(valid)
        assertNotNull(viewModel.uiState.value.fullNameError)
    }

    @Test
    fun `validation fails on short license number`() {
        viewModel.onFullNameChange("Tran Van B")
        viewModel.onLicenseNumberChange("12")
        val valid = viewModel.validate()

        assertFalse(valid)
        assertNotNull(viewModel.uiState.value.licenseError)
    }

    @Test
    fun `validation fails on invalid phone length`() {
        viewModel.onFullNameChange("Tran Van B")
        viewModel.onPhoneNumberChange("123") // too short
        viewModel.onLicenseNumberChange("GPLX-12345")
        val valid = viewModel.validate()

        assertFalse(valid)
        assertNotNull(viewModel.uiState.value.phoneError)
    }

    @Test
    fun `validation succeeds with valid inputs`() {
        viewModel.onFullNameChange("Tran Van B")
        viewModel.onPhoneNumberChange("0987654321")
        viewModel.onLicenseNumberChange("GPLX-123456")
        val valid = viewModel.validate()

        assertTrue(valid)
        assertNull(viewModel.uiState.value.fullNameError)
        assertNull(viewModel.uiState.value.phoneError)
        assertNull(viewModel.uiState.value.licenseError)
    }

    @Test
    fun `saveProfile success updates state and invokes callback`() = runTest {
        viewModel.onFullNameChange("Nguyen Van User")
        viewModel.onPhoneNumberChange("0901112233")
        viewModel.onLicenseNumberChange("GPLX-999888")

        var callbackInvoked = false
        viewModel.saveProfile { callbackInvoked = true }
        testDispatcher.scheduler.advanceUntilIdle()

        assertTrue(callbackInvoked)
        assertFalse(viewModel.uiState.value.isLoading)
        assertTrue(viewModel.uiState.value.isSuccess)
        assertNull(viewModel.uiState.value.errorMessage)
        assertEquals("Nguyen Van User", fakeRepo.lastFullName)
        assertEquals("0901112233", fakeRepo.lastPhone)
        assertEquals("GPLX-999888", fakeRepo.lastLicense)
    }

    @Test
    fun `saveProfile error sets errorMessage`() = runTest {
        fakeRepo.createProfileResult = RepositoryResult.Error("Số GPLX đã tồn tại trong hệ thống.", 409)

        viewModel.onFullNameChange("Nguyen Van User")
        viewModel.onLicenseNumberChange("GPLX-DUP")

        var callbackInvoked = false
        viewModel.saveProfile { callbackInvoked = true }
        testDispatcher.scheduler.advanceUntilIdle()

        assertFalse(callbackInvoked)
        assertFalse(viewModel.uiState.value.isLoading)
        assertFalse(viewModel.uiState.value.isSuccess)
        assertEquals("Số GPLX đã tồn tại trong hệ thống.", viewModel.uiState.value.errorMessage)
    }

    @Test
    fun `logout delegates to repository`() = runTest {
        viewModel.logout()
        testDispatcher.scheduler.advanceUntilIdle()
        assertTrue(fakeRepo.logoutCalled)
    }
}
