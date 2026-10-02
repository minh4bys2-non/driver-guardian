package com.example.driverguardian.data.auth

import com.example.driverguardian.data.remote.DriverGuardianApi
import com.example.driverguardian.data.remote.dto.RefreshTokenRequestDto
import com.example.driverguardian.data.remote.dto.toDomain
import kotlinx.coroutines.runBlocking
import okhttp3.Interceptor
import okhttp3.Response
import java.net.HttpURLConnection

interface TokenRefreshProvider {
    /**
     * Refreshes the access token using the stored refresh token.
     * Returns the new access token on success, or null if refresh failed/expired.
     */
    fun refreshAccessToken(): String?
}

class NetworkTokenRefreshProvider(
    private val refreshApi: DriverGuardianApi,
    private val sessionManager: SessionManager
) : TokenRefreshProvider {
    override fun refreshAccessToken(): String? {
        val refreshToken = sessionManager.getRefreshToken() ?: return null
        return try {
            val response = runBlocking {
                refreshApi.refreshToken(RefreshTokenRequestDto(refreshToken))
            }
            sessionManager.updateTokens(response.accessToken, response.refreshToken)
            sessionManager.updateUser(response.user.toDomain())
            response.accessToken
        } catch (_: Exception) {
            null
        }
    }
}

class AuthInterceptor(
    private val sessionManager: SessionManager,
    private val refreshProvider: TokenRefreshProvider
) : Interceptor {

    private val lock = Any()

    override fun intercept(chain: Interceptor.Chain): Response {
        val request = chain.request()
        val path = request.url().encodedPath()

        // Never attach or intercept for public auth/health endpoints
        if (isPublicEndpoint(path)) {
            return chain.proceed(request)
        }

        val token = sessionManager.getAccessToken()
        val authenticatedRequest = if (!token.isNullOrBlank()) {
            request.newBuilder()
                .header("Authorization", "Bearer $token")
                .build()
        } else {
            request
        }

        val response = chain.proceed(authenticatedRequest)

        if (response.code() != HttpURLConnection.HTTP_UNAUTHORIZED) {
            return response
        }

        // Handle 401: Refresh once and retry
        val newToken: String? = synchronized(lock) {
            val tokenAfterWait = sessionManager.getAccessToken()
            if (tokenAfterWait != null && tokenAfterWait != token) {
                // Token was already refreshed by another thread in the pool
                tokenAfterWait
            } else {
                refreshProvider.refreshAccessToken()
            }
        }

        return if (newToken != null && newToken != token) {
            response.close()
            val retriedRequest = request.newBuilder()
                .header("Authorization", "Bearer $newToken")
                .build()
            chain.proceed(retriedRequest)
        } else {
            // Refresh failed or revoked: clear session
            sessionManager.clearSession()
            response
        }
    }

    private fun isPublicEndpoint(path: String): Boolean {
        return path.contains("/auth/google") ||
                path.contains("/auth/refresh") ||
                path.contains("/health") ||
                path == "/"
    }
}
