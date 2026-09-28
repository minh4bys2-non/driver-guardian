package com.example.driverguardian.data.remote.dto

import com.example.driverguardian.domain.model.Driver
import com.example.driverguardian.domain.model.DrowsinessEvent
import com.example.driverguardian.domain.model.DrivingSession
import com.example.driverguardian.domain.model.ModelVersion
import com.example.driverguardian.domain.model.Vehicle
import com.google.gson.annotations.SerializedName

data class DriverResponseDto(
    @SerializedName("driver_id") val driverId: Int,
    @SerializedName("driver_code") val driverCode: String,
    @SerializedName("full_name") val fullName: String,
    @SerializedName("phone_number") val phoneNumber: String?,
    @SerializedName("license_number") val licenseNumber: String?,
    @SerializedName("status") val status: String,
    @SerializedName("created_at") val createdAt: String
)

data class VehicleResponseDto(
    @SerializedName("vehicle_id") val vehicleId: Int,
    @SerializedName("plate_number") val plateNumber: String,
    @SerializedName("vehicle_name") val vehicleName: String?,
    @SerializedName("vehicle_type") val vehicleType: String?,
    @SerializedName("device_code") val deviceCode: String?,
    @SerializedName("status") val status: String,
    @SerializedName("created_at") val createdAt: String
)

data class ModelVersionResponseDto(
    @SerializedName("model_version_id") val modelVersionId: Int,
    @SerializedName("version_name") val versionName: String,
    @SerializedName("model_type") val modelType: String?,
    @SerializedName("file_name") val fileName: String?,
    @SerializedName("description") val description: String?,
    @SerializedName("is_active") val isActive: String,
    @SerializedName("deployed_at") val deployedAt: String
)

data class DrivingSessionCreateDto(
    @SerializedName("driver_id") val driverId: Int,
    @SerializedName("vehicle_id") val vehicleId: Int,
    @SerializedName("model_version_id") val modelVersionId: Int
)

data class DrivingSessionResponseDto(
    @SerializedName("session_id") val sessionId: Int,
    @SerializedName("driver_id") val driverId: Int,
    @SerializedName("vehicle_id") val vehicleId: Int,
    @SerializedName("model_version_id") val modelVersionId: Int?,
    @SerializedName("start_time") val startTime: String,
    @SerializedName("duration_seconds") val durationSeconds: Int,
    @SerializedName("total_alerts") val totalAlerts: Int,
    @SerializedName("status") val status: String,
    @SerializedName("sync_status") val syncStatus: String
)

data class DrowsinessEventCreateDto(
    @SerializedName("session_id") val sessionId: Int,
    @SerializedName("driver_state") val driverState: String,
    @SerializedName("alert_level") val alertLevel: Int,
    @SerializedName("confidence") val confidence: Double?,
    @SerializedName("duration_ms") val durationMs: Int?
)

data class DrowsinessEventResponseDto(
    @SerializedName("event_id") val eventId: Int,
    @SerializedName("session_id") val sessionId: Int,
    @SerializedName("event_time") val eventTime: String,
    @SerializedName("driver_state") val driverState: String,
    @SerializedName("alert_level") val alertLevel: Int,
    @SerializedName("confidence") val confidence: Double?,
    @SerializedName("duration_ms") val durationMs: Int?,
    @SerializedName("acknowledged") val acknowledged: String,
    @SerializedName("sync_status") val syncStatus: String
)

fun DriverResponseDto.toDomain() = Driver(driverId, driverCode, fullName, phoneNumber, licenseNumber, status, createdAt)
fun VehicleResponseDto.toDomain() = Vehicle(vehicleId, plateNumber, vehicleName, vehicleType, deviceCode, status, createdAt)
fun ModelVersionResponseDto.toDomain() = ModelVersion(modelVersionId, versionName, modelType, fileName, description, isActive, deployedAt)
fun DrivingSessionResponseDto.toDomain() = DrivingSession(sessionId, driverId, vehicleId, modelVersionId, startTime, durationSeconds, totalAlerts, status, syncStatus)
fun DrowsinessEventResponseDto.toDomain() = DrowsinessEvent(eventId, sessionId, eventTime, driverState, alertLevel, confidence, durationMs, acknowledged, syncStatus)
