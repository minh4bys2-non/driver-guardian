package com.example.driverguardian.data.auth

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import com.example.driverguardian.domain.model.AuthTokens
import com.example.driverguardian.domain.model.UserProfile
import com.google.gson.Gson

interface TokenStore {
    fun saveTokens(tokens: AuthTokens)
    fun saveUser(user: UserProfile)
    fun getAccessToken(): String?
    fun getRefreshToken(): String?
    fun getUser(): UserProfile?
    fun updateAccessToken(accessToken: String, refreshToken: String?)
    fun clear()
}

class EncryptedTokenStore(context: Context) : TokenStore {
    private val prefs: SharedPreferences = try {
        val masterKey = MasterKey.Builder(context)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
        EncryptedSharedPreferences.create(
            context,
            "driver_guardian_secure_prefs",
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
        )
    } catch (_: Exception) {
        // Fallback for testing environments or legacy Keystore issues
        context.getSharedPreferences("driver_guardian_auth_prefs", Context.MODE_PRIVATE)
    }

    private val gson = Gson()

    override fun saveTokens(tokens: AuthTokens) {
        prefs.edit()
            .putString(KEY_ACCESS_TOKEN, tokens.accessToken)
            .putString(KEY_REFRESH_TOKEN, tokens.refreshToken)
            .apply()
    }

    override fun saveUser(user: UserProfile) {
        prefs.edit()
            .putString(KEY_USER_PROFILE, gson.toJson(user))
            .apply()
    }

    override fun getAccessToken(): String? = prefs.getString(KEY_ACCESS_TOKEN, null)

    override fun getRefreshToken(): String? = prefs.getString(KEY_REFRESH_TOKEN, null)

    override fun getUser(): UserProfile? {
        val json = prefs.getString(KEY_USER_PROFILE, null) ?: return null
        return try {
            gson.fromJson(json, UserProfile::class.java)
        } catch (_: Exception) {
            null
        }
    }

    override fun updateAccessToken(accessToken: String, refreshToken: String?) {
        val editor = prefs.edit().putString(KEY_ACCESS_TOKEN, accessToken)
        if (refreshToken != null) {
            editor.putString(KEY_REFRESH_TOKEN, refreshToken)
        }
        editor.apply()
    }

    override fun clear() {
        prefs.edit().clear().apply()
    }

    companion object {
        private const val KEY_ACCESS_TOKEN = "access_token"
        private const val KEY_REFRESH_TOKEN = "refresh_token"
        private const val KEY_USER_PROFILE = "user_profile"
    }
}

class InMemoryTokenStore : TokenStore {
    private var accessToken: String? = null
    private var refreshToken: String? = null
    private var user: UserProfile? = null

    override fun saveTokens(tokens: AuthTokens) {
        accessToken = tokens.accessToken
        refreshToken = tokens.refreshToken
    }

    override fun saveUser(user: UserProfile) {
        this.user = user
    }

    override fun getAccessToken(): String? = accessToken

    override fun getRefreshToken(): String? = refreshToken

    override fun getUser(): UserProfile? = user

    override fun updateAccessToken(accessToken: String, refreshToken: String?) {
        this.accessToken = accessToken
        if (refreshToken != null) {
            this.refreshToken = refreshToken
        }
    }

    override fun clear() {
        accessToken = null
        refreshToken = null
        user = null
    }
}
