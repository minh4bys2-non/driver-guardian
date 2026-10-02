package com.example.driverguardian.data.auth

import com.example.driverguardian.domain.model.AuthTokens
import com.example.driverguardian.domain.model.DriverSummary
import com.example.driverguardian.domain.model.UserProfile
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test

class AuthInterceptorTest {
    private lateinit var server: MockWebServer
    private lateinit var tokenStore: InMemoryTokenStore
    private lateinit var sessionManager: SessionManager
    private lateinit var okHttpClient: OkHttpClient

    private var mockRefreshTokenResult: String? = "refreshed_access_token_999"
    private var refreshCallCount = 0

    private val testUser = UserProfile(
        userId = 1,
        email = "test@driverguardian.com",
        displayName = "Test Driver",
        avatarUrl = null,
        role = "DRIVER",
        driver = DriverSummary(1, "DRV001", "Test Driver")
    )

    @Before
    fun setUp() {
        server = MockWebServer()
        server.start()

        tokenStore = InMemoryTokenStore()
        sessionManager = SessionManager(tokenStore)
        sessionManager.setSession(
            AuthTokens("initial_token_111", "refresh_token_222"),
            testUser
        )

        refreshCallCount = 0
        mockRefreshTokenResult = "refreshed_access_token_999"

        val fakeRefreshProvider = object : TokenRefreshProvider {
            override fun refreshAccessToken(): String? {
                refreshCallCount++
                val newAccessToken = mockRefreshTokenResult
                if (newAccessToken != null) {
                    sessionManager.updateTokens(newAccessToken, null)
                }
                return newAccessToken
            }
        }

        val interceptor = AuthInterceptor(sessionManager, fakeRefreshProvider)
        okHttpClient = OkHttpClient.Builder()
            .addInterceptor(interceptor)
            .build()
    }

    @After
    fun tearDown() {
        server.shutdown()
    }

    @Test
    fun `attaches Bearer token to protected requests`() {
        server.enqueue(MockResponse().setResponseCode(200).setBody("""{"status":"ok"}"""))

        val request = Request.Builder()
            .url(server.url("/sessions"))
            .build()

        val response = okHttpClient.newCall(request).execute()
        assertEquals(200, response.code)

        val recordedRequest = server.takeRequest()
        assertEquals("Bearer initial_token_111", recordedRequest.getHeader("Authorization"))
    }

    @Test
    fun `does not attach Bearer token to public endpoints`() {
        server.enqueue(MockResponse().setResponseCode(200).setBody("""{"status":"ok"}"""))

        val request = Request.Builder()
            .url(server.url("/auth/google"))
            .build()

        val response = okHttpClient.newCall(request).execute()
        assertEquals(200, response.code)

        val recordedRequest = server.takeRequest()
        assertNull(recordedRequest.getHeader("Authorization"))
    }

    @Test
    fun `on 401 response refreshes token once and retries original request with new token`() {
        // First request returns 401
        server.enqueue(MockResponse().setResponseCode(401).setBody("""{"detail":"Token expired"}"""))
        // Retried request returns 200
        server.enqueue(MockResponse().setResponseCode(200).setBody("""{"status":"success"}"""))

        val request = Request.Builder()
            .url(server.url("/sessions"))
            .build()

        val response = okHttpClient.newCall(request).execute()
        assertEquals(200, response.code)
        assertEquals(1, refreshCallCount)

        // Verify first request used expired token
        val firstRequest = server.takeRequest()
        assertEquals("Bearer initial_token_111", firstRequest.getHeader("Authorization"))

        // Verify second request used refreshed token
        val secondRequest = server.takeRequest()
        assertEquals("Bearer refreshed_access_token_999", secondRequest.getHeader("Authorization"))
    }

    @Test
    fun `on 401 response when refresh fails clears session and does not infinite loop`() {
        mockRefreshTokenResult = null
        server.enqueue(MockResponse().setResponseCode(401).setBody("""{"detail":"Invalid token"}"""))

        val request = Request.Builder()
            .url(server.url("/sessions"))
            .build()

        val response = okHttpClient.newCall(request).execute()
        assertEquals(401, response.code)
        assertEquals(1, refreshCallCount)

        // Session must be cleared on refresh failure
        assertNull(sessionManager.getAccessToken())
        assertEquals(AuthState.Unauthenticated, sessionManager.authState.value)
    }
}
