package com.example.driverguardian.ui.screens.auth

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.example.driverguardian.data.repository.AuthRepository
import com.example.driverguardian.data.repository.RepositoryResult
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class DriverProfileSetupUiState(
    val fullName: String = "",
    val phoneNumber: String = "",
    val licenseNumber: String = "",
    val fullNameError: String? = null,
    val phoneError: String? = null,
    val licenseError: String? = null,
    val isLoading: Boolean = false,
    val errorMessage: String? = null,
    val isSuccess: Boolean = false
)

class DriverProfileSetupViewModel(
    private val authRepository: AuthRepository
) : ViewModel() {

    private val _uiState = MutableStateFlow(DriverProfileSetupUiState())
    val uiState: StateFlow<DriverProfileSetupUiState> = _uiState.asStateFlow()

    init {
        val user = authRepository.getCurrentUser()
        if (user?.displayName != null) {
            _uiState.update { it.copy(fullName = user.displayName) }
        }
    }

    fun onFullNameChange(value: String) {
        _uiState.update { it.copy(fullName = value, fullNameError = null, errorMessage = null) }
    }

    fun onPhoneNumberChange(value: String) {
        _uiState.update { it.copy(phoneNumber = value, phoneError = null, errorMessage = null) }
    }

    fun onLicenseNumberChange(value: String) {
        _uiState.update { it.copy(licenseNumber = value, licenseError = null, errorMessage = null) }
    }

    fun validate(): Boolean {
        val state = _uiState.value
        val nameTrimmed = state.fullName.trim()
        val phoneTrimmed = state.phoneNumber.trim()
        val licenseTrimmed = state.licenseNumber.trim()

        var hasError = false
        var nameError: String? = null
        var phoneError: String? = null
        var licenseError: String? = null

        if (nameTrimmed.length < 2 || nameTrimmed.length > 100) {
            nameError = "Họ và tên phải có từ 2 đến 100 ký tự."
            hasError = true
        }

        if (phoneTrimmed.isNotEmpty() && (phoneTrimmed.length < 7 || phoneTrimmed.length > 20)) {
            phoneError = "Số điện thoại phải từ 7 đến 20 ký tự."
            hasError = true
        }

        if (licenseTrimmed.length < 3 || licenseTrimmed.length > 50) {
            licenseError = "Số GPLX phải có từ 3 đến 50 ký tự."
            hasError = true
        }

        _uiState.update {
            it.copy(
                fullNameError = nameError,
                phoneError = phoneError,
                licenseError = licenseError
            )
        }
        return !hasError
    }

    fun saveProfile(onSuccess: () -> Unit) {
        if (!validate()) return
        val state = _uiState.value
        if (state.isLoading) return

        _uiState.update { it.copy(isLoading = true, errorMessage = null) }

        viewModelScope.launch {
            val phone = state.phoneNumber.trim().ifEmpty { null }
            when (val result = authRepository.createDriverProfile(
                fullName = state.fullName.trim(),
                phoneNumber = phone,
                licenseNumber = state.licenseNumber.trim()
            )) {
                is RepositoryResult.Success -> {
                    _uiState.update { it.copy(isLoading = false, isSuccess = true) }
                    onSuccess()
                }
                is RepositoryResult.Error -> {
                    _uiState.update {
                        it.copy(
                            isLoading = false,
                            errorMessage = result.message
                        )
                    }
                }
            }
        }
    }

    fun logout() {
        viewModelScope.launch {
            authRepository.logout()
        }
    }
}

class DriverProfileSetupViewModelFactory(
    private val authRepository: AuthRepository
) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        return DriverProfileSetupViewModel(authRepository) as T
    }
}
