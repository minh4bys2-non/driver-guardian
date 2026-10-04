package com.example.driverguardian.ai.monitoring.model

import com.example.driverguardian.ai.monitoring.time.CameraTimestampSource
import com.example.driverguardian.ai.monitoring.time.TimebaseRelation

data class MonitoringDiagnostics(
    val cameraTimestampSource: CameraTimestampSource? = null,
    val sensorTimestampSource: String = "SensorEvent.timestamp / elapsedRealtimeNanos",
    val timebaseRelation: TimebaseRelation = TimebaseRelation.UNVERIFIED,
    val timebaseOffsetNs: Long? = null,
    val cameraFps: Double? = null,
    val inferenceMs: Double? = null,
    val physicalProcessingMs: Double? = null,
    val sensorHz: Double? = null,
    val sensorProcessingMs: Double? = null,
)
