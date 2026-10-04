package com.example.driverguardian.data.remote

import com.example.driverguardian.data.auth.AuthInterceptor
import com.example.driverguardian.data.auth.NetworkTokenRefreshProvider
import com.example.driverguardian.data.auth.SessionManager
import com.google.gson.GsonBuilder
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

object ApiClient {
    fun create(baseUrl: String, sessionManager: SessionManager? = null): DriverGuardianApi {
        val normalizedBaseUrl = if (baseUrl.endsWith("/")) baseUrl else "$baseUrl/"
        val gson = GsonBuilder().serializeNulls().create()

        val okHttpClientBuilder = OkHttpClient.Builder()

        if (sessionManager != null) {
            val unauthenticatedRetrofit = Retrofit.Builder()
                .baseUrl(normalizedBaseUrl)
                .addConverterFactory(GsonConverterFactory.create(gson))
                .build()
            val refreshApi = unauthenticatedRetrofit.create(DriverGuardianApi::class.java)
            val refreshProvider = NetworkTokenRefreshProvider(refreshApi, sessionManager)
            val authInterceptor = AuthInterceptor(sessionManager, refreshProvider)
            okHttpClientBuilder.addInterceptor(authInterceptor)
        }

        return Retrofit.Builder()
            .baseUrl(normalizedBaseUrl)
            .client(okHttpClientBuilder.build())
            .addConverterFactory(GsonConverterFactory.create(gson))
            .build()
            .create(DriverGuardianApi::class.java)
    }
}
