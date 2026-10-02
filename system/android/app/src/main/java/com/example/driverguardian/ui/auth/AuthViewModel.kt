package com.example.driverguardian.ui.auth

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.example.driverguardian.data.auth.AuthState
import com.example.driverguardian.data.repository.AuthRepository
import com.example.driverguardian.data.repository.RepositoryResult
import com.example.driverguardian.domain.model.UserProfile
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed interface AuthUiState {
    data object Initial : AuthUiState
    data object Loading : AuthUiState
    data class Authenticated(val user: UserProfile) : AuthUiState
    data class UnlinkedDriver(val user: UserProfile) : AuthUiState
    data object Unauthenticated : AuthUiState
    data class Error(val message: String) : AuthUiState
}

class AuthViewModel(
    private val authRepository: AuthRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow<AuthUiState>(AuthUiState.Initial)
    val uiState: StateFlow<AuthUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            authRepository.authState.collect { state ->
                _uiState.value = when (state) {
                    is AuthState.Authenticated -> AuthUiState.Authenticated(state.user)
                    is AuthState.UnlinkedDriver -> AuthUiState.UnlinkedDriver(state.user)
                    is AuthState.Authenticating -> AuthUiState.Loading
                    is AuthState.Error -> AuthUiState.Error(state.message)
                    is AuthState.Unauthenticated -> AuthUiState.Unauthenticated
                }
            }
        }
    }

    fun restoreSession() {
        viewModelScope.launch {
            _uiState.value = AuthUiState.Loading
            when (val result = authRepository.restoreSession()) {
                is RepositoryResult.Success -> {
                    if (result.value == null) {
                        _uiState.value = AuthUiState.Unauthenticated
                    }
                }
                is RepositoryResult.Error -> {
                    _uiState.value = AuthUiState.Unauthenticated
                }
            }
        }
    }

    fun loginWithGoogle(idToken: String) {
        viewModelScope.launch {
            _uiState.value = AuthUiState.Loading
            when (val result = authRepository.loginWithGoogle(idToken)) {
                is RepositoryResult.Success -> {
                    // Handled by state flow
                }
                is RepositoryResult.Error -> {
                    _uiState.value = AuthUiState.Error(result.message)
                }
            }
        }
    }

    fun logout() {
        viewModelScope.launch {
            _uiState.value = AuthUiState.Loading
            authRepository.logout()
            _uiState.value = AuthUiState.Unauthenticated
        }
    }

    fun clearError() {
        _uiState.value = AuthUiState.Unauthenticated
    }
}

class AuthViewModelFactory(
    private val authRepository: AuthRepository
) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        return AuthViewModel(authRepository) as T
    }
}
