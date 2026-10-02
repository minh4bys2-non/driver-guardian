package com.example.driverguardian.ai.monitoring.model

enum class CameraRuntimeState { IDLE, PERMISSION_REQUIRED, STARTING, AVAILABLE, UNAVAILABLE, ERROR }
enum class FaceRuntimeState { IDLE, INITIALIZING, NO_FACE, FACE_DETECTED, ERROR }
enum class SensorRuntimeState { IDLE, CALIBRATING, AVAILABLE, UNAVAILABLE, ERROR }

data class MonitoringRuntimeState(
    val camera: CameraRuntimeState = CameraRuntimeState.IDLE,
    val face: FaceRuntimeState = FaceRuntimeState.IDLE,
    val sensor: SensorRuntimeState = SensorRuntimeState.IDLE,
    val message: String? = null,
)
