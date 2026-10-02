package com.example.driverguardian.ai.monitoring.model

import com.example.driverguardian.ai.monitoring.device.DeviceMotionMetrics
import com.example.driverguardian.ai.physical.model.PhysicalMetrics

/** Runtime observation only; deliberately not an ONNX tensor contract. */
data class MonitoringSnapshot(
    val physical: PhysicalMetrics?,
    val deviceMotion: DeviceMotionMetrics?,
    val timestampSec: Double,
)
