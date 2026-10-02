package com.example.driverguardian.ui.auth

import com.example.driverguardian.data.auth.AuthState
import com.example.driverguardian.data.repository.AuthRepository
import com.example.driverguardian.data.repository.RepositoryResult
import com.example.driverguardian.domain.model.DriverSummary
import com.example.driverguardian.domain.model.UserProfile
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
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
class AuthViewModelTest {
    private val testDispatcher = StandardTestDispatcher()
    private val fakeAuthState = MutableStateFlow<AuthState>(AuthState.Unauthenticated)

    private val sampleDriver = DriverSummary(1, "DRV001", "Nguyễn Văn An")
    private val linkedUser = UserProfile(1, "driver.an@driverguardian.com", "Nguyễn Văn An", null, "DRIVER", sampleDriver)
    private val unlinkedUser = UserProfile(2, "new@driverguardian.com", "New Driver", null, "DRIVER", null)

    private lateinit var fakeRepository: FakeAuthRepository
    private lateinit var viewModel: AuthViewModel

    private class FakeAuthRepository(
        override val authState: StateFlow<AuthState>
    ) : AuthRepository {
        var loginResult: RepositoryResult<UserProfile> = RepositoryResult.Success(
            UserProfile(1, "test@test.com", "Test", null, "DRIVER", null)
        )
        var restoreResult: RepositoryResult<UserProfile?> = RepositoryResult.Success(null)
        var logoutCallCount = 0

        override suspend fun loginWithGoogle(idToken: String): RepositoryResult<UserProfile> = loginResult

        override suspend fun restoreSession(): RepositoryResult<UserProfile?> = restoreResult

        override suspend fun logout(): RepositoryResult<Unit> {
            logoutCallCount++
            return RepositoryResult.Success(Unit)
        }

        override fun getCurrentUser(): UserProfile? = null
    }

    @Before
    fun setUp() {
        Dispatchers.setMain(testDispatcher)
        fakeRepository = FakeAuthRepository(fakeAuthState)
        viewModel = AuthViewModel(fakeRepository)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun `initial state reflects repository unauthenticated state`() = runTest {
        advanceUntilIdle()
        assertEquals(AuthUiState.Unauthenticated, viewModel.uiState.value)
    }

    @Test
    fun `loginWithGoogle success updates uiState to Authenticated`() = runTest {
        fakeRepository.loginResult = RepositoryResult.Success(linkedUser)

        viewModel.loginWithGoogle("valid_id_token")
        fakeAuthState.value = AuthState.Authenticated(linkedUser)
        advanceUntilIdle()

        assertTrue(viewModel.uiState.value is AuthUiState.Authenticated)
        assertEquals(linkedUser, (viewModel.uiState.value as AuthUiState.Authenticated).user)
    }

    @Test
    fun `unlinked driver profile updates uiState to UnlinkedDriver`() = runTest {
        fakeRepository.loginResult = RepositoryResult.Success(unlinkedUser)

        viewModel.loginWithGoogle("valid_id_token")
        fakeAuthState.value = AuthState.UnlinkedDriver(unlinkedUser)
        advanceUntilIdle()

        assertTrue(viewModel.uiState.value is AuthUiState.UnlinkedDriver)
        assertEquals(unlinkedUser, (viewModel.uiState.value as AuthUiState.UnlinkedDriver).user)
    }

    @Test
    fun `login failure sets Error state with message`() = runTest {
        fakeRepository.loginResult = RepositoryResult.Error("Token không hợp lệ", 401)

        viewModel.loginWithGoogle("invalid_token")
        advanceUntilIdle()

        assertTrue(viewModel.uiState.value is AuthUiState.Error)
        assertEquals("Token không hợp lệ", (viewModel.uiState.value as AuthUiState.Error).message)
    }

    @Test
    fun `logout invokes repository logout and sets Unauthenticated`() = runTest {
        fakeAuthState.value = AuthState.Authenticated(linkedUser)
        advanceUntilIdle()

        viewModel.logout()
        fakeAuthState.value = AuthState.Unauthenticated
        advanceUntilIdle()

        assertEquals(1, fakeRepository.logoutCallCount)
        assertEquals(AuthUiState.Unauthenticated, viewModel.uiState.value)
    }

    @Test
    fun `clearError resets uiState to Unauthenticated`() = runTest {
        fakeRepository.loginResult = RepositoryResult.Error("Lỗi kết nối", cause = null)
        viewModel.loginWithGoogle("bad_token")
        advanceUntilIdle()

        assertTrue(viewModel.uiState.value is AuthUiState.Error)

        viewModel.clearError()
        assertEquals(AuthUiState.Unauthenticated, viewModel.uiState.value)
    }
}
