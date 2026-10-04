package com.example.driverguardian.data.auth

import com.example.driverguardian.domain.model.AuthTokens
import com.example.driverguardian.domain.model.DriverSummary
import com.example.driverguardian.domain.model.UserProfile
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class SessionManagerTest {
    private lateinit var tokenStore: InMemoryTokenStore
    private lateinit var sessionManager: SessionManager

    private val sampleDriver = DriverSummary(
        driverId = 1,
        driverCode = "DRV001",
        fullName = "Nguyễn Văn An"
    )

    private val linkedUser = UserProfile(
        userId = 1,
        email = "driver.an@driverguardian.com",
        displayName = "Nguyễn Văn An",
        avatarUrl = null,
        role = "DRIVER",
        driver = sampleDriver
    )

    private val unlinkedUser = UserProfile(
        userId = 2,
        email = "new.user@gmail.com",
        displayName = "New User",
        avatarUrl = null,
        role = "DRIVER",
        driver = null
    )

    private val tokens = AuthTokens(
        accessToken = "access_token_123",
        refreshToken = "refresh_token_456"
    )

    @Before
    fun setUp() {
        tokenStore = InMemoryTokenStore()
        sessionManager = SessionManager(tokenStore)
    }

    @Test
    fun `initial state is Unauthenticated when store is empty`() {
        assertEquals(AuthState.Unauthenticated, sessionManager.authState.value)
        assertFalse(sessionManager.isAuthenticated())
        assertNull(sessionManager.getAccessToken())
        assertNull(sessionManager.getCurrentUser())
    }

    @Test
    fun `initial state resolves to Authenticated when linked user and token exist`() {
        tokenStore.saveTokens(tokens)
        tokenStore.saveUser(linkedUser)

        val restoredManager = SessionManager(tokenStore)
        assertTrue(restoredManager.authState.value is AuthState.Authenticated)
        assertEquals(linkedUser, (restoredManager.authState.value as AuthState.Authenticated).user)
        assertTrue(restoredManager.isAuthenticated())
    }

    @Test
    fun `initial state resolves to UnlinkedDriver when role is DRIVER but driver profile is null`() {
        tokenStore.saveTokens(tokens)
        tokenStore.saveUser(unlinkedUser)

        val restoredManager = SessionManager(tokenStore)
        assertTrue(restoredManager.authState.value is AuthState.UnlinkedDriver)
        assertEquals(unlinkedUser, (restoredManager.authState.value as AuthState.UnlinkedDriver).user)
    }

    @Test
    fun `setSession stores tokens and user and emits Authenticated`() {
        sessionManager.setSession(tokens, linkedUser)

        assertEquals("access_token_123", sessionManager.getAccessToken())
        assertEquals("refresh_token_456", sessionManager.getRefreshToken())
        assertEquals(linkedUser, sessionManager.getCurrentUser())
        assertTrue(sessionManager.authState.value is AuthState.Authenticated)
    }

    @Test
    fun `clearSession clears tokens and emits Unauthenticated`() {
        sessionManager.setSession(tokens, linkedUser)
        sessionManager.clearSession()

        assertNull(sessionManager.getAccessToken())
        assertNull(sessionManager.getRefreshToken())
        assertNull(sessionManager.getCurrentUser())
        assertEquals(AuthState.Unauthenticated, sessionManager.authState.value)
    }

    @Test
    fun `updateTokens updates accessToken and refreshToken in store`() {
        sessionManager.setSession(tokens, linkedUser)
        sessionManager.updateTokens("new_access_token", "new_refresh_token")

        assertEquals("new_access_token", sessionManager.getAccessToken())
        assertEquals("new_refresh_token", sessionManager.getRefreshToken())
    }
}
