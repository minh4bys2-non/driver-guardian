package com.example.driverguardian.data.auth

import com.example.driverguardian.domain.model.AuthTokens
import com.example.driverguardian.domain.model.UserProfile
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

sealed interface AuthState {
    data object Unauthenticated : AuthState
    data object Authenticating : AuthState
    data class Authenticated(val user: UserProfile) : AuthState
    data class UnlinkedDriver(val user: UserProfile) : AuthState
    data class Error(val message: String) : AuthState
}

class SessionManager(private val tokenStore: TokenStore) {
    private val _authState = MutableStateFlow<AuthState>(resolveInitialState())
    val authState: StateFlow<AuthState> = _authState.asStateFlow()

    private fun resolveInitialState(): AuthState {
        val user = tokenStore.getUser()
        val token = tokenStore.getAccessToken()
        return if (user != null && token != null) {
            mapUserToState(user)
        } else {
            AuthState.Unauthenticated
        }
    }

    fun mapUserToState(user: UserProfile): AuthState {
        return if (user.role == "DRIVER" && user.driver == null) {
            AuthState.UnlinkedDriver(user)
        } else {
            AuthState.Authenticated(user)
        }
    }

    fun setAuthenticating() {
        _authState.value = AuthState.Authenticating
    }

    fun setSession(tokens: AuthTokens, user: UserProfile) {
        tokenStore.saveTokens(tokens)
        tokenStore.saveUser(user)
        _authState.value = mapUserToState(user)
    }

    fun updateUser(user: UserProfile) {
        tokenStore.saveUser(user)
        _authState.value = mapUserToState(user)
    }

    fun updateTokens(accessToken: String, refreshToken: String?) {
        tokenStore.updateAccessToken(accessToken, refreshToken)
    }

    fun setError(message: String) {
        _authState.value = AuthState.Error(message)
    }

    fun clearSession() {
        tokenStore.clear()
        _authState.value = AuthState.Unauthenticated
    }

    fun getAccessToken(): String? = tokenStore.getAccessToken()
    fun getRefreshToken(): String? = tokenStore.getRefreshToken()
    fun getCurrentUser(): UserProfile? = tokenStore.getUser()
    fun isAuthenticated(): Boolean = tokenStore.getAccessToken() != null
}
