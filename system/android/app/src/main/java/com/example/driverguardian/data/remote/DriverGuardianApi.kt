package com.example.driverguardian.data.remote

import com.example.driverguardian.data.remote.dto.DriverResponseDto
import com.example.driverguardian.data.remote.dto.DrowsinessEventCreateDto
import com.example.driverguardian.data.remote.dto.DrowsinessEventResponseDto
import com.example.driverguardian.data.remote.dto.DrivingSessionCreateDto
import com.example.driverguardian.data.remote.dto.DrivingSessionResponseDto
import com.example.driverguardian.data.remote.dto.ModelVersionResponseDto
import com.example.driverguardian.data.remote.dto.VehicleResponseDto
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST

interface DriverGuardianApi {
    @GET("drivers") suspend fun getDrivers(): List<DriverResponseDto>
    @GET("vehicles") suspend fun getVehicles(): List<VehicleResponseDto>
    @GET("model-versions/active") suspend fun getActiveModelVersion(): ModelVersionResponseDto
    @POST("sessions") suspend fun createSession(@Body request: DrivingSessionCreateDto): DrivingSessionResponseDto
    @POST("events") suspend fun createEvent(@Body request: DrowsinessEventCreateDto): DrowsinessEventResponseDto
}
