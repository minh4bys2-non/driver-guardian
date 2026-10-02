package com.example.driverguardian.data.repository

import com.example.driverguardian.data.auth.AuthState
import com.example.driverguardian.data.auth.InMemoryTokenStore
import com.example.driverguardian.data.auth.SessionManager
import com.example.driverguardian.data.remote.ApiClient
import com.example.driverguardian.data.remote.DriverGuardianApi
import com.example.driverguardian.domain.model.AuthTokens
import com.example.driverguardian.domain.model.DriverSummary
import com.example.driverguardian.domain.model.UserProfile
import kotlinx.coroutines.test.runTest
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class AuthRepositoryTest {
    private lateinit var server: MockWebServer
    private lateinit var api: DriverGuardianApi
    private lateinit var tokenStore: InMemoryTokenStore
    private lateinit var sessionManager: SessionManager
    private lateinit var repository: NetworkAuthRepository

    @Before
    fun setUp() {
        server = MockWebServer()
        server.start()

        tokenStore = InMemoryTokenStore()
        sessionManager = SessionManager(tokenStore)
        api = ApiClient.create(server.url("/").toString())
        repository = NetworkAuthRepository(api, sessionManager)
    }

    @After
    fun tearDown() {
        server.shutdown()
    }

    @Test
    fun `loginWithGoogle success stores tokens, user, and emits Authenticated`() = runTest {
        val authResponseJson = """
            {
                "access_token": "acc_123",
                "refresh_token": "ref_456",
                "token_type": "bearer",
                "expires_in": 1800,
                "user": {
                    "user_id": 1,
                    "email": "driver.an@driverguardian.com",
                    "display_name": "Nguyễn Văn An",
                    "avatar_url": null,
                    "role": "DRIVER",
                    "driver": {
                        "driver_id": 1,
                        "driver_code": "DRV001",
                        "full_name": "Nguyễn Văn An"
                    }
                }
            }
        """.trimIndent()

        server.enqueue(MockResponse().setResponseCode(200).setBody(authResponseJson))

        val result = repository.loginWithGoogle("valid_google_token")
        assertTrue(result is RepositoryResult.Success)

        val user = (result as RepositoryResult.Success).value
        assertEquals("driver.an@driverguardian.com", user.email)
        assertEquals(1, user.driver?.driverId)

        assertEquals("acc_123", sessionManager.getAccessToken())
        assertEquals("ref_456", sessionManager.getRefreshToken())
        assertTrue(sessionManager.authState.value is AuthState.Authenticated)
    }

    @Test
    fun `loginWithGoogle 401 returns Error result and sets Error state`() = runTest {
        server.enqueue(
            MockResponse()
                .setResponseCode(401)
                .setBody("""{"detail":"Token expired or invalid"}""")
        )

        val result = repository.loginWithGoogle("expired_google_token")
        assertTrue(result is RepositoryResult.Error)
        assertEquals(401, (result as RepositoryResult.Error).statusCode)
        assertTrue(sessionManager.authState.value is AuthState.Error)
    }

    @Test
    fun `restoreSession fetches getMe and updates user when tokens exist`() = runTest {
        sessionManager.setSession(
            AuthTokens("acc_123", "ref_456"),
            UserProfile(1, "old@email.com", "Old Name", null, "DRIVER", null)
        )

        val meResponseJson = """
            {
                "user_id": 1,
                "email": "driver.an@driverguardian.com",
                "display_name": "Nguyễn Văn An",
                "avatar_url": null,
                "role": "DRIVER",
                "driver": {
                    "driver_id": 1,
                    "driver_code": "DRV001",
                    "full_name": "Nguyễn Văn An"
                }
            }
        """.trimIndent()

        server.enqueue(MockResponse().setResponseCode(200).setBody(meResponseJson))

        val result = repository.restoreSession()
        assertTrue(result is RepositoryResult.Success)

        val user = (result as RepositoryResult.Success).value
        assertEquals("Nguyễn Văn An", user?.displayName)
        assertEquals(1, user?.driver?.driverId)
        assertTrue(sessionManager.authState.value is AuthState.Authenticated)
    }

    @Test
    fun `restoreSession returns null and clears session when no tokens exist`() = runTest {
        val result = repository.restoreSession()
        assertTrue(result is RepositoryResult.Success)
        assertNull((result as RepositoryResult.Success).value)
        assertEquals(AuthState.Unauthenticated, sessionManager.authState.value)
    }

    @Test
    fun `logout calls server revocation and clears local session`() = runTest {
        sessionManager.setSession(
            AuthTokens("acc_123", "ref_456"),
            UserProfile(1, "user@test.com", "User", null, "DRIVER", DriverSummary(1, "DRV001", "User"))
        )

        server.enqueue(MockResponse().setResponseCode(200).setBody("""{"detail":"Logged out successfully"}"""))

        val result = repository.logout()
        assertTrue(result is RepositoryResult.Success)

        assertNull(sessionManager.getAccessToken())
        assertNull(sessionManager.getRefreshToken())
        assertEquals(AuthState.Unauthenticated, sessionManager.authState.value)

        val recordedRequest = server.takeRequest()
        assertEquals("/auth/logout", recordedRequest.path)
    }
}
