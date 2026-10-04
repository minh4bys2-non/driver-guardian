package com.example.driverguardian.ai.monitoring.time

import org.junit.Assert.*
import org.junit.Test

class MonitoringTimebaseNormalizerTest {
    @Test fun realtimeSourceRequiresRuntimeCausalityEvidenceBeforeDirectComparison() {
        val normalizer = MonitoringTimebaseNormalizer(CameraTimestampSource.REALTIME)
        assertEquals(TimebaseRelation.UNVERIFIED, normalizer.relation)
        repeat(7) { i -> normalizer.observe(i * 10_000_000L, i * 10_000_000L + 20_000_000L) }
        assertNull(normalizer.normalizeCameraTimestampNs(70_000_000L))
        normalizer.observe(70_000_000L, 90_000_000L)
        assertEquals(TimebaseRelation.DIRECT_MONOTONIC, normalizer.relation)
        assertEquals(80_000_000L, normalizer.normalizeCameraTimestampNs(80_000_000L))
        assertEquals(0L, normalizer.offsetNs)
    }

    @Test fun realtimeSourceRejectsImpossibleFutureOrExcessivelyOldCameraTimestamps() {
        val n = MonitoringTimebaseNormalizer(
            CameraTimestampSource.REALTIME,
            MonitoringTimebaseConfig(verificationSampleCount = 4, maximumCaptureToReceiptLatencyNs = 100L),
        )
        n.observe(100L, 90L)
        n.observe(100L, 250L)
        n.observe(100L, 110L)
        n.observe(100L, 120L)
        assertEquals(TimebaseRelation.UNVERIFIED, n.relation)
        assertNull(n.offsetNs)
    }

    @Test fun unknownClockUsesConservativeStableOffsetThatCannotSelectFutureSensorData() {
        val n = MonitoringTimebaseNormalizer(
            CameraTimestampSource.UNKNOWN,
            MonitoringTimebaseConfig(
                verificationSampleCount = 8,
                maximumOffsetSpreadNs = 5L,
                maximumCaptureToReceiptLatencyNs = 100L,
            ),
        )
        repeat(7) { i -> n.observe(i * 1_000_000L, 1_000_000_000L + i * 1_000_000L) }
        assertEquals(TimebaseRelation.UNVERIFIED, n.relation)
        assertNull(n.normalizeCameraTimestampNs(8_000_000L))
        n.observe(7_000_000L, 1_007_000_000L)
        assertEquals(TimebaseRelation.OFFSET_NORMALIZED, n.relation)
        assertEquals(1_007_999_900L, n.normalizeCameraTimestampNs(8_000_000L))
        assertEquals(999_999_900L, n.offsetNs)
        assertTrue(n.normalizeCameraTimestampNs(9_000_000L)!! <= 1_009_000_000L)
    }

    @Test fun unknownClockNeverClaimsDirectAndUsesConservativeOffsetEvenWhenDifferenceIsSmall() {
        val n = MonitoringTimebaseNormalizer(CameraTimestampSource.UNKNOWN)
        repeat(8) { i -> n.observe(i * 10_000_000L, i * 10_000_000L + 20_000_000L) }
        assertEquals(TimebaseRelation.OFFSET_NORMALIZED, n.relation)
        assertEquals(-980_000_000L, n.offsetNs)
    }

    @Test fun unstableUnknownClockNeverVerifies() {
        val n = MonitoringTimebaseNormalizer(CameraTimestampSource.UNKNOWN)
        repeat(12) { i ->
            n.observe(
                i * 1_000_000L,
                i * 1_000_000L + if (i % 2 == 0) 2_000_000_000L else 2_020_000_000L,
            )
        }
        assertEquals(TimebaseRelation.UNVERIFIED, n.relation)
        assertNull(n.normalizeCameraTimestampNs(13_000_000L))
    }

    @Test fun normalizedOutputMustRemainMonotonicAndResetClearsVerification() {
        val n = MonitoringTimebaseNormalizer(
            CameraTimestampSource.UNKNOWN,
            MonitoringTimebaseConfig(maximumCaptureToReceiptLatencyNs = 10L),
        )
        repeat(8) { i -> n.observe(i.toLong(), 1_000L + i) }
        assertEquals(1_000L, n.normalizeCameraTimestampNs(10L))
        assertNull(n.normalizeCameraTimestampNs(9L))
        n.reset()
        assertEquals(TimebaseRelation.UNVERIFIED, n.relation)
        assertNull(n.offsetNs)
    }
}
