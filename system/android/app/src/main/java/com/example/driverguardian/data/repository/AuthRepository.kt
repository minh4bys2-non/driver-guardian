package com.example.driverguardian.data.repository

import com.example.driverguardian.data.auth.AuthState
import com.example.driverguardian.data.auth.SessionManager
import com.example.driverguardian.data.remote.DriverGuardianApi
import com.example.driverguardian.data.remote.dto.GoogleAuthRequestDto
import com.example.driverguardian.data.remote.dto.LogoutRequestDto
import com.example.driverguardian.data.remote.dto.toDomain
import com.example.driverguardian.data.remote.dto.toTokensDomain
import com.example.driverguardian.domain.model.UserProfile
import kotlinx.coroutines.flow.StateFlow
import retrofit2.HttpException
import java.io.IOException

interface AuthRepository {
    val authState: StateFlow<AuthState>
    suspend fun loginWithGoogle(idToken: String): RepositoryResult<UserProfile>
    suspend fun restoreSession(): RepositoryResult<UserProfile?>
    suspend fun logout(): RepositoryResult<Unit>
    fun getCurrentUser(): UserProfile?
}

class NetworkAuthRepository(
    private val api: DriverGuardianApi,
    private val sessionManager: SessionManager
) : AuthRepository {

    override val authState: StateFlow<AuthState> = sessionManager.authState

    override suspend fun loginWithGoogle(idToken: String): RepositoryResult<UserProfile> {
        sessionManager.setAuthenticating()
        return try {
            val response = api.authenticateGoogle(GoogleAuthRequestDto(idToken))
            val tokens = response.toTokensDomain()
            val user = response.user.toDomain()
            sessionManager.setSession(tokens, user)
            RepositoryResult.Success(user)
        } catch (e: HttpException) {
            val message = when (e.code()) {
                401 -> "Token Google không hợp lệ hoặc đã hết hạn."
                403 -> "Tài khoản của bạn đã bị vô hiệu hóa."
                else -> "Đăng nhập Google thất bại (${e.code()})."
            }
            sessionManager.setError(message)
            RepositoryResult.Error(message, e.code(), e)
        } catch (e: IOException) {
            val message = "Không thể kết nối đến máy chủ Driver Guardian."
            sessionManager.setError(message)
            RepositoryResult.Error(message, cause = e)
        } catch (e: Exception) {
            val message = "Lỗi xác thực: ${e.localizedMessage ?: "Không xác định"}"
            sessionManager.setError(message)
            RepositoryResult.Error(message, cause = e)
        }
    }

    override suspend fun restoreSession(): RepositoryResult<UserProfile?> {
        val cachedUser = sessionManager.getCurrentUser()
        val accessToken = sessionManager.getAccessToken()

        if (cachedUser == null || accessToken.isNullOrBlank()) {
            sessionManager.clearSession()
            return RepositoryResult.Success(null)
        }

        return try {
            val meResponse = api.getMe()
            val updatedUser = meResponse.toDomain()
            sessionManager.updateUser(updatedUser)
            RepositoryResult.Success(updatedUser)
        } catch (e: HttpException) {
            if (e.code() == 401 || e.code() == 403) {
                sessionManager.clearSession()
                RepositoryResult.Success(null)
            } else {
                RepositoryResult.Success(cachedUser)
            }
        } catch (_: Exception) {
            RepositoryResult.Success(cachedUser)
        }
    }

    override suspend fun logout(): RepositoryResult<Unit> {
        val refreshToken = sessionManager.getRefreshToken()
        try {
            if (!refreshToken.isNullOrBlank()) {
                api.logout(LogoutRequestDto(refreshToken))
            }
        } catch (_: Exception) {
            // Best effort server-side revocation
        } finally {
            sessionManager.clearSession()
        }
        return RepositoryResult.Success(Unit)
    }

    override fun getCurrentUser(): UserProfile? = sessionManager.getCurrentUser()
}
