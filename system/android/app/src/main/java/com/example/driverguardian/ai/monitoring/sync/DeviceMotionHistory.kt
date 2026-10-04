package com.example.driverguardian.ai.monitoring.sync

import com.example.driverguardian.ai.monitoring.device.DeviceCalibrationState
import com.example.driverguardian.ai.monitoring.device.DeviceMotionMetrics

class DeviceMotionHistory(private val maxEntries: Int = 120) {
    private val entries = ArrayDeque<DeviceMotionMetrics>()
    init { require(maxEntries > 0) }

    @Synchronized
    fun add(metrics: DeviceMotionMetrics) {
        if (entries.lastOrNull()?.timestampSec?.let { metrics.timestampSec <= it } == true) return
        entries.addLast(metrics)
        while (entries.size > maxEntries) entries.removeFirst()
    }

    @Synchronized
    fun latestAtOrBefore(frameTimestampSec: Double): DeviceMotionMetrics? =
        entries.lastOrNull { it.timestampSec <= frameTimestampSec && it.calibrationState != DeviceCalibrationState.UNAVAILABLE }

    @Synchronized
    fun associate(
        frameTimestampSec: Double,
        config: MonitoringSyncConfig = MonitoringSyncConfig(),
    ): DeviceMotionMetrics? {
        val sample = latestAtOrBefore(frameTimestampSec) ?: return null
        val ageMs = (frameTimestampSec - sample.timestampSec) * 1_000.0
        if (ageMs < -1e-6 || ageMs > config.sensorFreshnessThresholdMs + 1e-6) return null
        return sample.copy(sampleAgeMs = ageMs.coerceAtLeast(0.0))
    }

    @Synchronized
    fun clear() = entries.clear()
}
