package com.example.driverguardian.data.auth

import android.content.Context
import android.content.ContextWrapper
import android.os.CancellationSignal
import androidx.credentials.ClearCredentialStateRequest
import androidx.credentials.CreateCredentialRequest
import androidx.credentials.CreateCredentialResponse
import androidx.credentials.CredentialManager
import androidx.credentials.CredentialManagerCallback
import androidx.credentials.GetCredentialRequest
import androidx.credentials.GetCredentialResponse
import androidx.credentials.PrepareGetCredentialResponse
import androidx.credentials.exceptions.ClearCredentialException
import androidx.credentials.exceptions.CreateCredentialException
import androidx.credentials.exceptions.GetCredentialCancellationException
import androidx.credentials.exceptions.GetCredentialException
import androidx.credentials.exceptions.NoCredentialException
import com.example.driverguardian.data.repository.AuthRepository
import com.example.driverguardian.data.repository.RepositoryResult
import com.example.driverguardian.domain.model.UserProfile
import com.example.driverguardian.ui.auth.AuthUiState
import com.example.driverguardian.ui.auth.AuthViewModel
import com.google.android.libraries.identity.googleid.GetGoogleIdOption
import com.google.android.libraries.identity.googleid.GetSignInWithGoogleOption
import com.google.android.libraries.identity.googleid.GoogleIdTokenCredential
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
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.util.concurrent.Executor

@OptIn(ExperimentalCoroutinesApi::class)
class GoogleAuthManagerTest {
    private val testDispatcher = StandardTestDispatcher()
    private val testClientId = "767234214974-test.apps.googleusercontent.com"
    private lateinit var dummyContext: Context
    private lateinit var fakeCredentialManager: FakeCredentialManager
    private lateinit var googleAuthManager: GoogleAuthManager

    private class FakeCredentialManager : CredentialManager {
        val recordedRequests = mutableListOf<GetCredentialRequest>()
        var getCredentialHandler: suspend (GetCredentialRequest) -> GetCredentialResponse = {
            throw NoCredentialException("No credentials configured in fake")
        }

        override suspend fun getCredential(
            context: Context,
            request: GetCredentialRequest
        ): GetCredentialResponse {
            recordedRequests.add(request)
            return getCredentialHandler(request)
        }

        override suspend fun getCredential(
            context: Context,
            pendingGetCredentialHandle: PrepareGetCredentialResponse.PendingGetCredentialHandle
        ): GetCredentialResponse = throw NotImplementedError()

        override suspend fun prepareGetCredential(
            request: GetCredentialRequest
        ): PrepareGetCredentialResponse = throw NotImplementedError()

        override suspend fun createCredential(
            context: Context,
            request: CreateCredentialRequest
        ): CreateCredentialResponse = throw NotImplementedError()

        override suspend fun clearCredentialState(
            request: ClearCredentialStateRequest
        ): Unit = throw NotImplementedError()

        override fun getCredentialAsync(
            context: Context,
            request: GetCredentialRequest,
            cancellationSignal: CancellationSignal?,
            executor: Executor,
            callback: CredentialManagerCallback<GetCredentialResponse, GetCredentialException>
        ) = throw NotImplementedError()

        override fun getCredentialAsync(
            context: Context,
            pendingGetCredentialHandle: PrepareGetCredentialResponse.PendingGetCredentialHandle,
            cancellationSignal: CancellationSignal?,
            executor: Executor,
            callback: CredentialManagerCallback<GetCredentialResponse, GetCredentialException>
        ) = throw NotImplementedError()

        override fun prepareGetCredentialAsync(
            request: GetCredentialRequest,
            cancellationSignal: CancellationSignal?,
            executor: Executor,
            callback: CredentialManagerCallback<PrepareGetCredentialResponse, GetCredentialException>
        ) = throw NotImplementedError()

        override fun createCredentialAsync(
            context: Context,
            request: CreateCredentialRequest,
            cancellationSignal: CancellationSignal?,
            executor: Executor,
            callback: CredentialManagerCallback<CreateCredentialResponse, CreateCredentialException>
        ) = throw NotImplementedError()

        override fun clearCredentialStateAsync(
            request: ClearCredentialStateRequest,
            cancellationSignal: CancellationSignal?,
            executor: Executor,
            callback: CredentialManagerCallback<Void?, ClearCredentialException>
        ) = throw NotImplementedError()

        override fun createSettingsPendingIntent() = throw NotImplementedError()
    }

    private class FakeAuthRepository(
        override val authState: StateFlow<AuthState>
    ) : AuthRepository {
        var lastSubmittedToken: String? = null
        var loginResult: RepositoryResult<UserProfile> = RepositoryResult.Success(
            UserProfile(1, "test@test.com", "Test", null, "DRIVER", null)
        )

        override suspend fun loginWithGoogle(idToken: String): RepositoryResult<UserProfile> {
            lastSubmittedToken = idToken
            return loginResult
        }

        override suspend fun restoreSession(): RepositoryResult<UserProfile?> = RepositoryResult.Success(null)
        override suspend fun logout(): RepositoryResult<Unit> = RepositoryResult.Success(Unit)
        override fun getCurrentUser(): UserProfile? = null
    }

    @Before
    fun setUp() {
        Dispatchers.setMain(testDispatcher)
        dummyContext = ContextWrapper(null)
        fakeCredentialManager = FakeCredentialManager()
        googleAuthManager = GoogleAuthManager(
            context = dummyContext,
            serverClientId = testClientId,
            credentialManager = fakeCredentialManager
        )
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    @Test
    fun `authorized-account bottom sheet succeeds on first try without fallback`() = runTest(testDispatcher) {
        val expectedToken = "token_authorized_account_123"
        fakeCredentialManager.getCredentialHandler = { request ->
            val option = request.credentialOptions.first() as GetGoogleIdOption
            assertTrue("Expected authorized accounts filter to be true on first try", option.filterByAuthorizedAccounts)
            assertEquals(testClientId, option.serverClientId)

            val credential = GoogleIdTokenCredential.Builder()
                .setId("user@gmail.com")
                .setIdToken(expectedToken)
                .build()
            GetCredentialResponse(credential)
        }

        val result = googleAuthManager.getGoogleIdTokenFromBottomSheet()

        assertTrue(result.isSuccess)
        assertEquals(expectedToken, result.getOrNull())
        assertEquals("Should not retry if authorized account exists", 1, fakeCredentialManager.recordedRequests.size)
    }

    @Test
    fun `no authorized account triggers retry with all accounts`() = runTest(testDispatcher) {
        val expectedToken = "token_from_all_accounts_456"
        var callCount = 0

        fakeCredentialManager.getCredentialHandler = { request ->
            callCount++
            val option = request.credentialOptions.first() as GetGoogleIdOption
            if (callCount == 1) {
                assertTrue("First attempt should check authorized accounts", option.filterByAuthorizedAccounts)
                throw NoCredentialException("No authorized account found")
            } else {
                assertTrue("Second attempt should fall back to all accounts", !option.filterByAuthorizedAccounts)
                val credential = GoogleIdTokenCredential.Builder()
                    .setId("user_other@gmail.com")
                    .setIdToken(expectedToken)
                    .build()
                GetCredentialResponse(credential)
            }
        }

        val result = googleAuthManager.getGoogleIdTokenFromBottomSheet()

        assertTrue(result.isSuccess)
        assertEquals(expectedToken, result.getOrNull())
        assertEquals("Should have made exactly 2 attempts", 2, fakeCredentialManager.recordedRequests.size)
    }

    @Test
    fun `button uses GetSignInWithGoogleOption with configured serverClientId`() = runTest(testDispatcher) {
        val expectedToken = "token_from_button_siwg"
        fakeCredentialManager.getCredentialHandler = { request ->
            val option = request.credentialOptions.first()
            assertTrue("Button must use GetSignInWithGoogleOption", option is GetSignInWithGoogleOption)
            val siwgOption = option as GetSignInWithGoogleOption
            assertEquals(testClientId, siwgOption.serverClientId)

            val credential = GoogleIdTokenCredential.Builder()
                .setId("driver@gmail.com")
                .setIdToken(expectedToken)
                .build()
            GetCredentialResponse(credential)
        }

        val result = googleAuthManager.getGoogleIdTokenFromButton()

        assertTrue(result.isSuccess)
        assertEquals(expectedToken, result.getOrNull())
        assertEquals(1, fakeCredentialManager.recordedRequests.size)
    }

    @Test
    fun `persistent Google button defaults to GetSignInWithGoogleOption`() = runTest(testDispatcher) {
        val expectedToken = "token_default_delegate"
        fakeCredentialManager.getCredentialHandler = { request ->
            val option = request.credentialOptions.first()
            assertTrue("Default getGoogleIdToken must delegate to button flow", option is GetSignInWithGoogleOption)
            val credential = GoogleIdTokenCredential.Builder()
                .setId("driver@gmail.com")
                .setIdToken(expectedToken)
                .build()
            GetCredentialResponse(credential)
        }

        val result = googleAuthManager.getGoogleIdToken()

        assertTrue(result.isSuccess)
        assertEquals(expectedToken, result.getOrNull())
    }

    @Test
    fun `no credential error handled without crash`() = runTest(testDispatcher) {
        fakeCredentialManager.getCredentialHandler = {
            throw NoCredentialException("No credentials available on device")
        }

        val result = googleAuthManager.getGoogleIdTokenFromButton()

        assertTrue("Failure must be returned instead of uncaught crash", result.isFailure)
        assertTrue(result.exceptionOrNull() is NoCredentialException)
    }

    @Test
    fun `cancellation handled gracefully in button and bottom-sheet flows`() = runTest(testDispatcher) {
        fakeCredentialManager.getCredentialHandler = {
            throw GetCredentialCancellationException("User dismissed credential dialog")
        }

        val buttonResult = googleAuthManager.getGoogleIdTokenFromButton()
        assertTrue(buttonResult.isFailure)
        assertTrue(buttonResult.exceptionOrNull() is GetCredentialCancellationException)

        fakeCredentialManager.recordedRequests.clear()
        val bottomSheetResult = googleAuthManager.getGoogleIdTokenFromBottomSheet()
        assertTrue(bottomSheetResult.isFailure)
        assertTrue(bottomSheetResult.exceptionOrNull() is GetCredentialCancellationException)
        assertEquals("Bottom sheet should not retry when user cancels", 1, fakeCredentialManager.recordedRequests.size)
    }

    @Test
    fun `successful ID token forwarded to backend via AuthViewModel`() = runTest(testDispatcher) {
        val fakeAuthState = MutableStateFlow<AuthState>(AuthState.Unauthenticated)
        val fakeRepository = FakeAuthRepository(fakeAuthState)
        val viewModel = AuthViewModel(fakeRepository)

        val targetToken = "verified_google_id_token_live"
        fakeCredentialManager.getCredentialHandler = {
            val credential = GoogleIdTokenCredential.Builder()
                .setId("driver@gmail.com")
                .setIdToken(targetToken)
                .build()
            GetCredentialResponse(credential)
        }

        val authResult = googleAuthManager.getGoogleIdTokenFromButton()
        assertTrue(authResult.isSuccess)

        // Forward token to backend
        viewModel.loginWithGoogle(authResult.getOrThrow())
        advanceUntilIdle()

        assertEquals("Token from Google must be forwarded to repository/backend", targetToken, fakeRepository.lastSubmittedToken)
    }
}
