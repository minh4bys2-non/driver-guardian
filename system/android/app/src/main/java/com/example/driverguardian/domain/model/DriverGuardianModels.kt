package com.example.driverguardian.domain.model

data class Driver(
    val id: Int,
    val code: String,
    val fullName: String,
    val phoneNumber: String?,
    val licenseNumber: String?,
    val status: String,
    val createdAt: String
)

data class Vehicle(
    val id: Int,
    val plateNumber: String,
    val name: String?,
    val type: String?,
    val deviceCode: String?,
    val status: String,
    val createdAt: String
)

data class ModelVersion(
    val id: Int,
    val versionName: String,
    val modelType: String?,
    val fileName: String?,
    val description: String?,
    val isActive: String,
    val deployedAt: String
)

data class DrivingSession(
    val id: Int,
    val driverId: Int,
    val vehicleId: Int,
    val modelVersionId: Int?,
    val startTime: String,
    val durationSeconds: Int,
    val totalAlerts: Int,
    val status: String,
    val syncStatus: String
)

data class DrowsinessEvent(
    val id: Int,
    val sessionId: Int,
    val eventTime: String,
    val driverState: String,
    val alertLevel: Int,
    val confidence: Double?,
    val durationMs: Int?,
    val acknowledged: String,
    val syncStatus: String
)
