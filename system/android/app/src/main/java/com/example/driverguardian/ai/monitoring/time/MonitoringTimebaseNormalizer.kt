package com.example.driverguardian.ai.monitoring.time

class MonitoringTimebaseNormalizer(
    source: CameraTimestampSource,
    private val config: MonitoringTimebaseConfig = MonitoringTimebaseConfig(),
) {
    private var source: CameraTimestampSource = source
    private val offsets = ArrayDeque<Long>()
    private var verifiedOffset: Long? = null
    private var verifiedRelation: TimebaseRelation = TimebaseRelation.UNVERIFIED
    private var lastNormalizedNs: Long? = null

    val relation: TimebaseRelation
        get() = verifiedRelation

    val offsetNs: Long? get() = verifiedOffset

    fun updateCameraTimestampSource(source: CameraTimestampSource) {
        if (this.source == source) return
        this.source = source
        reset()
    }

    fun observe(cameraTimestampNs: Long, receiptElapsedNs: Long) {
        if (verifiedOffset != null || cameraTimestampNs < 0L || receiptElapsedNs < 0L) return
        offsets.addLast(receiptElapsedNs - cameraTimestampNs)
        while (offsets.size > config.verificationSampleCount) offsets.removeFirst()
        if (offsets.size == config.verificationSampleCount) {
            val sorted = offsets.sorted()
            val directlyComparable = source == CameraTimestampSource.REALTIME &&
                sorted.first() >= 0L &&
                sorted.last() <= config.maximumCaptureToReceiptLatencyNs
            when {
                directlyComparable -> {
                    verifiedOffset = 0L
                    verifiedRelation = TimebaseRelation.DIRECT_MONOTONIC
                }
                source == CameraTimestampSource.UNKNOWN &&
                    sorted.last() - sorted.first() <= config.maximumOffsetSpreadNs -> {
                    // A receipt timestamp is an upper bound on capture time. Subtract the
                    // configured maximum delivery latency so normalized camera time cannot
                    // silently move into the future relative to sensor samples.
                    verifiedOffset = sorted.first() - config.maximumCaptureToReceiptLatencyNs
                    verifiedRelation = TimebaseRelation.OFFSET_NORMALIZED
                }
            }
        }
    }

    fun normalizeCameraTimestampNs(cameraTimestampNs: Long): Long? {
        val offset = verifiedOffset ?: return null
        if (cameraTimestampNs < 0L) return null
        val normalized = cameraTimestampNs + offset
        if (lastNormalizedNs?.let { normalized <= it } == true) return null
        lastNormalizedNs = normalized
        return normalized
    }

    fun reset() {
        offsets.clear()
        verifiedOffset = null
        verifiedRelation = TimebaseRelation.UNVERIFIED
        lastNormalizedNs = null
    }
}
