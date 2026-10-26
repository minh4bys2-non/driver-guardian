package com.example.driverguardian.ai.monitoring.model

import com.example.driverguardian.ai.fusion.WindowEvaluationSnapshot
import com.example.driverguardian.ai.monitoring.device.DeviceMotionMetrics
import com.example.driverguardian.ai.physical.model.PhysicalMetrics

/** Real-time synchronized observation snapshot containing physical metrics, device motion, and 1Hz window evaluation. */
data class MonitoringSnapshot(
    val physical: PhysicalMetrics?,
    val deviceMotion: DeviceMotionMetrics?,
    val timestampSec: Double,
    val windowEvaluation: WindowEvaluationSnapshot? = null,
)
