package com.example.driverguardian.ui.monitoring

import com.example.driverguardian.ai.fusion.PipelineStatus
import com.example.driverguardian.ai.fusion.WindowEvaluationSnapshot
import com.example.driverguardian.ai.monitoring.model.*
import org.junit.Assert.*
import org.junit.Test

class MonitoringPresentationTest {
    @Test fun nullMetricsRemainPlaceholdersAndDomainsStaySeparate() {
        val p = MonitoringPresentation.from(null, MonitoringRuntimeState(
            camera = CameraRuntimeState.AVAILABLE,
            face = FaceRuntimeState.NO_FACE,
            sensor = SensorRuntimeState.UNAVAILABLE,
        ), MonitoringDiagnostics())
        assertEquals("—", p.ear)
        assertEquals("—", p.perclos)
        assertEquals("—", p.headPose)
        assertEquals("Không thấy khuôn mặt", p.faceStatus)
        assertEquals("DEMO: cảnh báo nguy hiểm", p.dangerActionLabel)
        assertEquals("WARMUP", p.compactWindowStatus)
    }

    @Test fun compactWindowStatusFormatsAccuratelyAcrossPhases() {
        fun createEval(status: PipelineStatus, elapsed: Double, window: Double = 60.0): WindowEvaluationSnapshot =
            WindowEvaluationSnapshot(
                evaluatedAtSec = elapsed,
                elapsedSec = elapsed,
                windowSec = window,
                status = status,
                neuralBufferSize = 0,
                cachedP4Count = 0,
                latestDeltaFramesCount = 0,
                latestNeuralScore = null,
                onnxInferenceMs = 0.0,
                lazyFftMs = 0.0,
                workerTotalMs = 0.0,
                onnxModelVariant = "test",
                headMotionMetrics = null,
                rawFeatures = null,
                normalizedFeatures = null,
                fusionSummary = null,
                lastSkipReason = null,
            )

        val warmupP = MonitoringPresentation.from(
            MonitoringSnapshot(null, null, 15.0, createEval(PipelineStatus.WARMING_UP, 15.2, 60.0)),
            MonitoringRuntimeState(),
            MonitoringDiagnostics(),
        )
        assertEquals("WARMUP 15s/60s", warmupP.compactWindowStatus)

        val waitValidP = MonitoringPresentation.from(
            MonitoringSnapshot(null, null, 48.0, createEval(PipelineStatus.WAITING_FOR_VALID_WINDOW, 48.0, 60.0)),
            MonitoringRuntimeState(),
            MonitoringDiagnostics(),
        )
        assertEquals("WAIT VALID 48s/60s", waitValidP.compactWindowStatus)

        val alertP = MonitoringPresentation.from(
            MonitoringSnapshot(null, null, 60.0, createEval(PipelineStatus.ALERT, 60.0, 60.0)),
            MonitoringRuntimeState(),
            MonitoringDiagnostics(),
        )
        assertEquals("ALERT", alertP.compactWindowStatus)

        val drowsyP = MonitoringPresentation.from(
            MonitoringSnapshot(null, null, 61.0, createEval(PipelineStatus.DROWSY, 61.0, 60.0)),
            MonitoringRuntimeState(),
            MonitoringDiagnostics(),
        )
        assertEquals("DROWSY", drowsyP.compactWindowStatus)
    }

    @Test fun missingPhysicalDataAndTimebaseAreExplicitNotFabricated() {
        val p = MonitoringPresentation.from(null, MonitoringRuntimeState(), MonitoringDiagnostics())
        assertTrue(p.timebaseStatus.contains("UNVERIFIED"))
    }

    @Test fun performanceExposesCameraPipelineButNotOptionalDeviceMotion() {
        val p = MonitoringPresentation.from(
            MonitoringSnapshot(null, null, 1.0),
            MonitoringRuntimeState(),
            MonitoringDiagnostics(cameraFps = 12.5, inferenceMs = 7.0, sensorHz = 50.0),
        )
        assertTrue(p.performance.contains("Camera 12.5 FPS"))
        assertTrue(p.performance.contains("MediaPipe 7.0 ms"))
        assertFalse(p.performance.contains("Sensor"))
    }
}
