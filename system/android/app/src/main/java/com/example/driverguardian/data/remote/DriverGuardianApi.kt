package com.example.driverguardian.data.remote

import com.example.driverguardian.data.remote.dto.AuthTokensResponseDto
import com.example.driverguardian.data.remote.dto.DriverProfileCreateDto
import com.example.driverguardian.data.remote.dto.DriverProfileUpdateDto
import com.example.driverguardian.data.remote.dto.DriverResponseDto
import com.example.driverguardian.data.remote.dto.DrowsinessEventCreateDto
import com.example.driverguardian.data.remote.dto.DrowsinessEventResponseDto
import com.example.driverguardian.data.remote.dto.DrivingSessionCreateDto
import com.example.driverguardian.data.remote.dto.DrivingSessionResponseDto
import com.example.driverguardian.data.remote.dto.GoogleAuthRequestDto
import com.example.driverguardian.data.remote.dto.LogoutRequestDto
import com.example.driverguardian.data.remote.dto.ModelVersionResponseDto
import com.example.driverguardian.data.remote.dto.RefreshTokenRequestDto
import com.example.driverguardian.data.remote.dto.TripSessionResponseDto
import com.example.driverguardian.data.remote.dto.UserProfileResponseDto
import com.example.driverguardian.data.remote.dto.VehicleCreateDto
import com.example.driverguardian.data.remote.dto.VehicleResponseDto
import com.example.driverguardian.data.remote.dto.VehicleUpdateDto
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.PATCH
import retrofit2.http.POST
import retrofit2.http.Path
import retrofit2.http.Query

interface DriverGuardianApi {
    @POST("auth/google") suspend fun authenticateGoogle(@Body request: GoogleAuthRequestDto): AuthTokensResponseDto
    @POST("auth/refresh") suspend fun refreshToken(@Body request: RefreshTokenRequestDto): AuthTokensResponseDto
    @POST("auth/logout") suspend fun logout(@Body request: LogoutRequestDto): Map<String, Any>
    @GET("auth/me") suspend fun getMe(): UserProfileResponseDto

    @GET("drivers") suspend fun getDrivers(): List<DriverResponseDto>
    @POST("drivers/me") suspend fun createDriverProfile(@Body request: DriverProfileCreateDto): UserProfileResponseDto
    @GET("drivers/me") suspend fun getMyDriverProfile(): DriverResponseDto
    @PATCH("drivers/me") suspend fun updateMyDriverProfile(@Body request: DriverProfileUpdateDto): DriverResponseDto

    @GET("vehicles") suspend fun getVehicles(): List<VehicleResponseDto>
    @POST("vehicles") suspend fun createVehicle(@Body request: VehicleCreateDto): VehicleResponseDto
    @PATCH("vehicles/{vehicleId}") suspend fun updateVehicle(
        @Path("vehicleId") vehicleId: Int,
        @Body request: VehicleUpdateDto
    ): VehicleResponseDto

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
