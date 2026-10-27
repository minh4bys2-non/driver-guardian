package com.example.driverguardian.ai.monitoring.model

import com.example.driverguardian.ai.monitoring.MonitoringConfig
import com.example.driverguardian.ai.monitoring.time.CameraTimestampSource
import com.example.driverguardian.ai.monitoring.time.TimebaseRelation

data class MonitoringDiagnostics(
    val cameraTimestampSource: CameraTimestampSource? = null,
    val sensorTimestampSource: String = MonitoringConfig.SENSOR_TIMESTAMP_SOURCE_LABEL,
    val timebaseRelation: TimebaseRelation = TimebaseRelation.UNVERIFIED,
    val timebaseOffsetNs: Long? = null,
    val cameraFps: Double? = null,
    val inferenceMs: Double? = null,
    val physicalProcessingMs: Double? = null,
    val sensorHz: Double? = null,
    val sensorProcessingMs: Double? = null,
    val ingestMs: Double? = null,
    val letterboxMs: Double? = null,
    val lazyFftMs: Double? = null,
    val onnxInferenceMs: Double? = null,
    val workerTotalMs: Double? = null,
    val activeDelegate: String? = null,
    val onnxModelVariant: String? = null,
    val balanceBrightnessEnabled: Boolean = MonitoringConfig.DEFAULT_BALANCE_BRIGHTNESS_ENABLED,
)
