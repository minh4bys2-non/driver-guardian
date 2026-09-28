package com.example.driverguardian.data.remote

import com.example.driverguardian.data.remote.dto.DriverResponseDto
import com.example.driverguardian.data.remote.dto.DrowsinessEventCreateDto
import com.example.driverguardian.data.remote.dto.DrowsinessEventResponseDto
import com.example.driverguardian.data.remote.dto.DrivingSessionCreateDto
import com.example.driverguardian.data.remote.dto.DrivingSessionResponseDto
import com.example.driverguardian.data.remote.dto.ModelVersionResponseDto
import com.example.driverguardian.data.remote.dto.TripSessionResponseDto
import com.example.driverguardian.data.remote.dto.VehicleResponseDto
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Path
import retrofit2.http.Query

interface DriverGuardianApi {
    @GET("drivers") suspend fun getDrivers(): List<DriverResponseDto>
    @GET("vehicles") suspend fun getVehicles(): List<VehicleResponseDto>
    @GET("model-versions/active") suspend fun getActiveModelVersion(): ModelVersionResponseDto
    @POST("sessions") suspend fun createSession(@Body request: DrivingSessionCreateDto): DrivingSessionResponseDto
    @POST("events") suspend fun createEvent(@Body request: DrowsinessEventCreateDto): DrowsinessEventResponseDto
    @POST("sessions/{sessionId}/complete") suspend fun completeSession(@Path("sessionId") sessionId: Int): DrivingSessionResponseDto
    @GET("sessions") suspend fun getSessions(
        @Query("driver_id") driverId: Int? = null,
        @Query("status") status: String? = null,
        @Query("limit") limit: Int? = null
    ): List<TripSessionResponseDto>
    @GET("sessions/{sessionId}") suspend fun getSession(@Path("sessionId") sessionId: Int): TripSessionResponseDto
    @GET("sessions/{sessionId}/events") suspend fun getSessionEvents(@Path("sessionId") sessionId: Int): List<DrowsinessEventResponseDto>
    @POST("events/{eventId}/acknowledge") suspend fun acknowledgeEvent(@Path("eventId") eventId: Int): DrowsinessEventResponseDto
}
