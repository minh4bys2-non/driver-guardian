package com.example.driverguardian.ai.monitoring.sync

data class MonitoringSyncConfig(
    val sensorFreshnessThresholdMs: Double = SENSOR_FRESHNESS_THRESHOLD_MS,
) {
    init { require(sensorFreshnessThresholdMs >= 0.0 && sensorFreshnessThresholdMs.isFinite()) }

    companion object {
        /** Operational synchronization policy, not a trained-model requirement. */
        const val SENSOR_FRESHNESS_THRESHOLD_MS = 150.0
    }
}
