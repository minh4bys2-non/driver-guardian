package com.example.driverguardian.data.auth

import android.content.Context
import androidx.credentials.Credential
import androidx.credentials.CredentialManager
import androidx.credentials.CustomCredential
import androidx.credentials.GetCredentialRequest
import androidx.credentials.GetCredentialResponse
import androidx.credentials.exceptions.GetCredentialCancellationException
import com.google.android.libraries.identity.googleid.GetGoogleIdOption
import com.google.android.libraries.identity.googleid.GetSignInWithGoogleOption
import com.google.android.libraries.identity.googleid.GoogleIdTokenCredential

class GoogleAuthManager(
    private val context: Context,
    val serverClientId: String,
    private val credentialManager: CredentialManager = CredentialManager.create(context)
) {
    /**
     * Explicit Sign in with Google button flow.
     * Google officially recommends this button flow when:
     * - user dismissed the bottom sheet
     * - no Google Account exists on device
     * - existing account requires reauthentication
     *
     * MUST use GetSignInWithGoogleOption.
     */
    suspend fun getGoogleIdTokenFromButton(): Result<String> {
        if (serverClientId.isBlank()) {
            return Result.failure(
                IllegalStateException("GOOGLE_SERVER_CLIENT_ID chưa được cấu hình trên ứng dụng.")
            )
        }

        val signInWithGoogleOption = GetSignInWithGoogleOption.Builder(serverClientId)
            .build()

        val request = GetCredentialRequest.Builder()
            .addCredentialOption(signInWithGoogleOption)
            .build()

        return try {
            val response: GetCredentialResponse = credentialManager.getCredential(
                request = request,
                context = context
            )
            Result.success(extractIdToken(response.credential))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Bottom-sheet flow.
     * First try: GetGoogleIdOption with filterByAuthorizedAccounts = true.
     * If no authorized credential exists: retry with filterByAuthorizedAccounts = false
     * to allow users to select other Google accounts present on the device.
     */
    suspend fun getGoogleIdTokenFromBottomSheet(): Result<String> {
        if (serverClientId.isBlank()) {
            return Result.failure(
                IllegalStateException("GOOGLE_SERVER_CLIENT_ID chưa được cấu hình trên ứng dụng.")
            )
        }

        // 1. Try with authorized accounts first
        val authorizedOption = GetGoogleIdOption.Builder()
            .setFilterByAuthorizedAccounts(true)
            .setServerClientId(serverClientId)
            .setAutoSelectEnabled(false)
            .build()

        val authorizedRequest = GetCredentialRequest.Builder()
            .addCredentialOption(authorizedOption)
            .build()

        try {
            val response: GetCredentialResponse = credentialManager.getCredential(
                request = authorizedRequest,
                context = context
            )
            return Result.success(extractIdToken(response.credential))
        } catch (e: GetCredentialCancellationException) {
            // User cancelled; do not force second prompt
            return Result.failure(e)
        } catch (e: Exception) {
            // No authorized credential or other error; retry with filterByAuthorizedAccounts = false
        }

        // 2. Retry with all accounts on device
        val allAccountsOption = GetGoogleIdOption.Builder()
            .setFilterByAuthorizedAccounts(false)
            .setServerClientId(serverClientId)
            .setAutoSelectEnabled(false)
            .build()

        val allAccountsRequest = GetCredentialRequest.Builder()
            .addCredentialOption(allAccountsOption)
            .build()

        return try {
            val response: GetCredentialResponse = credentialManager.getCredential(
                request = allAccountsRequest,
                context = context
            )
            Result.success(extractIdToken(response.credential))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Default sign-in method: delegates to the explicit button flow.
     */
    suspend fun getGoogleIdToken(): Result<String> = getGoogleIdTokenFromButton()

    private fun extractIdToken(credential: Credential): String {
        if (credential is GoogleIdTokenCredential) {
            return credential.idToken
        }
        if (credential is CustomCredential &&
            (credential.type == GoogleIdTokenCredential.TYPE_GOOGLE_ID_TOKEN_CREDENTIAL ||
             credential.type == GoogleIdTokenCredential.TYPE_GOOGLE_ID_TOKEN_SIWG_CREDENTIAL)
        ) {
            val googleIdTokenCredential = GoogleIdTokenCredential.createFrom(credential.data)
            return googleIdTokenCredential.idToken
        }
        throw IllegalStateException("Định dạng thông tin Google Credential không hợp lệ: ${credential.type}")
    }
}

