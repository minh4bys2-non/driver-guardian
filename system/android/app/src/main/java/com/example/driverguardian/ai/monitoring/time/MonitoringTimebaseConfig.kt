package com.example.driverguardian.ai.monitoring.time

data class MonitoringTimebaseConfig(
    val verificationSampleCount: Int = 8,
    val maximumOffsetSpreadNs: Long = 5_000_000L,
    val maximumCaptureToReceiptLatencyNs: Long = 1_000_000_000L,
) {
    init {
        require(verificationSampleCount >= 2)
        require(maximumOffsetSpreadNs >= 0L)
        require(maximumCaptureToReceiptLatencyNs >= 0L)
    }
}
