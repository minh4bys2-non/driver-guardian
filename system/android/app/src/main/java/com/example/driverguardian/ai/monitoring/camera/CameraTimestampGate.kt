package com.example.driverguardian.ai.monitoring.camera

class CameraTimestampGate {
    private var lastTimestampNs: Long? = null

    fun accept(timestampNs: Long): Double? {
        if (timestampNs < 0L || lastTimestampNs?.let { timestampNs <= it } == true) return null
        lastTimestampNs = timestampNs
        return timestampNs / NANOS_PER_SECOND
    }

    fun reset() { lastTimestampNs = null }

    private companion object { const val NANOS_PER_SECOND = 1_000_000_000.0 }
}
