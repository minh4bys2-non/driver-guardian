package com.example.driverguardian.ui.monitoring

import com.example.driverguardian.ai.fusion.FusionConfig
import com.example.driverguardian.ai.fusion.PipelineStatus
import com.example.driverguardian.ai.monitoring.MonitoringConfig
import com.example.driverguardian.ai.monitoring.model.CameraRuntimeState
import com.example.driverguardian.ai.monitoring.model.FaceRuntimeState
import com.example.driverguardian.ai.monitoring.model.MonitoringDiagnostics
import com.example.driverguardian.ai.monitoring.model.MonitoringRuntimeState
import com.example.driverguardian.ai.monitoring.model.MonitoringSnapshot
import java.util.Locale

data class MonitoringPresentation(
    val cameraStatus: String,
    val faceStatus: String,
    val ear: String,
    val mar: String,
    val perclos: String,
    val headPose: String,
    val timebaseStatus: String,
    val performance: String,
    val error: String?,
    val dangerActionLabel: String = "DEMO: cảnh báo nguy hiểm",
    val pipelineStatus: PipelineStatus = PipelineStatus.WARMING_UP,
    val pipelineStatusLabel: String = PipelineStatus.WARMING_UP.displayLabel,
    val pipelineProgressText: String = "—",
    val compactWindowStatus: String = "WARMUP",
    val readinessSummary: String = "—",
    val coverageSummary: String = "—",
    val eyeSummary: String = "—",
    val mouthSummary: String = "—",
    val headPoseDetail: String = "—",
    val nodSummary: String = "—",
    val headMotionSummary: String = "—",
    val neuralSummary: String = "—",
    val normalizedFeaturesSummary: String = "—",
    val primaryFusionSummary: String = "—",
    val weightedMeanSummary: String = "—",
    val noisyOrSummary: String = "—",
    val cooccurrenceSummary: String = "—",
    val threadTelemetrySummary: String = "—",
    val lastSkipReason: String? = null,
    val balanceBrightnessEnabled: Boolean = MonitoringConfig.DEFAULT_BALANCE_BRIGHTNESS_ENABLED,
    val isDrowsyPrediction: Boolean = false,
) {
    companion object {
        fun from(
            snapshot: MonitoringSnapshot?,
            state: MonitoringRuntimeState,
            diagnostics: MonitoringDiagnostics,
        ): MonitoringPresentation {
            val physical = snapshot?.physical
            val eval = snapshot?.windowEvaluation
            val status = eval?.status ?: PipelineStatus.WARMING_UP
            val fusion = eval?.fusionSummary

            val eyeStateText = when (physical?.eyeState) {
                1 -> "CLOSED (1)"
                0 -> "OPEN (0)"
                else -> "—"
            }
            val mouthStateText = when (physical?.mouthState) {
                1 -> "YAWNING (1)"
                0 -> "NORMAL (0)"
                else -> "—"
            }
            val nodStateText = when (physical?.nodding) {
                true -> "NODDING (1)"
                false -> "NORMAL (0)"
                null -> "—"
            }

            val eyeSummary = if (physical == null) {
                "—"
            } else {
                "State: $eyeStateText • P80 EAR: ${number(physical.p80EarThreshold, 3)} • " +
                    "Blinks: ${physical.blinkCount} (${number(physical.blinkFreqPerMin, 1)}/min) • " +
                    "Last: ${physical.blinkDurationMs?.let { "${number(it, 0)} ms" } ?: "—"}"
            }

            val mouthSummary = if (physical == null) {
                "—"
            } else {
                "State: $mouthStateText • POM: ${physical.pomPct?.let { "${number(it, 1)}%" } ?: "—"} • " +
                    "Yawns: ${physical.yawnCount} (${number(physical.yawnFreqPerMin, 1)}/min) • " +
                    "Last: ${physical.yawnDurationMs?.let { "${number(it, 0)} ms" } ?: "—"}"
            }

            val headPoseDetail = if (physical == null) {
                "—"
            } else {
                "Pitch |Amp|: ${physical.pitchAmplitudeDeg?.let { "${number(it, 1)}°" } ?: "—"} • " +
                    "Over-Angle: ${physical.overAngle?.let { if (it) "YES" else "NO" } ?: "—"} " +
                    "(${physical.overAnglePct?.let { "${number(it, 1)}%" } ?: "—"})"
            }

            val nodSummary = if (physical == null) {
                "—"
            } else {
                "State: $nodStateText • Nods: ${physical.nodCount} (${number(physical.nodFreqPerMin, 1)}/min) • " +
                    "Cur: ${physical.nodDurationMs?.let { "${number(it, 0)} ms" } ?: "—"} • " +
                    "Last: ${physical.nodEventDurationMs?.let { "${number(it, 0)} ms" } ?: "—"}"
            }

            val readinessSummary = if (physical == null) {
                "Eye HMM: WAIT • Mouth HMM: WAIT • Head Calib: WAIT"
            } else {
                "Eye HMM: ${if (physical.eyeReady) "READY" else "INIT"} • " +
                    "Mouth HMM: ${if (physical.mouthReady) "READY" else "INIT"} • " +
                    "Head Calib: ${if (physical.headCalibrated) "READY" else "CALIBRATING"}"
            }

            val coverageSummary = if (physical == null) {
                "—"
            } else {
                "Eye ${number(physical.eyeObservedSec, 1)}s • " +
                    "Mouth ${number(physical.mouthObservedSec, 1)}s • " +
                    "Head ${number(physical.headObservedSec, 1)}s • " +
                    "FFT ${number(physical.headMotionObservedSec, 1)}s / ${number(physical.windowSec, 0)}s"
            }

            val freqHz = eval?.headMotionMetrics?.frequency ?: physical?.headMotionFrequencyHz
            val resHz = eval?.headMotionMetrics?.resolution ?: physical?.headMotionResolutionHz
            val headMotionObs = eval?.headMotionMetrics?.observed ?: physical?.headMotionObservedSec
            val headMotionSummary = if (physical == null && eval?.headMotionMetrics == null) {
                "—"
            } else {
                "Mean Pitch: ${physical?.pitchMeanDeg?.let { "${number(it, 2)}°" } ?: "—"} • " +
                    "Mean |Pitch|: ${physical?.pitchMeanAmplitudeDeg?.let { "${number(it, 2)}°" } ?: "—"} • " +
                    "Dom Freq (1Hz FFT): ${freqHz?.let { "${number(it, 2)} Hz" } ?: "—"} " +
                    "(Δf=${resHz?.let { "${number(it, 3)} Hz" } ?: "—"}, obs=${number(headMotionObs, 1)}s)"
            }

            val neuralSummary = if (eval == null) {
                "Model: ${diagnostics.onnxModelVariant ?: "drowsy_classifier_int8.onnx"} • Chờ chu kỳ 1Hz"
            } else {
                "Grid 5FPS: ${eval.neuralBufferSize} frames • " +
                    "p4_old Cache: ${eval.cachedP4Count}/50 (ΔT_new=${eval.latestDeltaFramesCount}) • " +
                    "CNN-ConvGRU Score: ${number(eval.latestNeuralScore, 3)} • " +
                    "Model: ${eval.onnxModelVariant}"
            }

            val normalizedFeaturesSummary = eval?.normalizedFeatures?.let { norm ->
                FusionConfig.FEATURES.joinToString(" • ") { key ->
                    val shortKey = when (key) {
                        FusionConfig.FEATURE_BLINK_FREQUENCY -> "blink_freq"
                        FusionConfig.FEATURE_BLINK_DURATION -> "blink_dur(P90)"
                        FusionConfig.FEATURE_PERCLOS -> "perclos"
                        FusionConfig.FEATURE_YAWN_FREQUENCY -> "yawn_freq"
                        FusionConfig.FEATURE_NOD_DURATION -> "nod_dur(P100)"
                        FusionConfig.FEATURE_NOD_FREQUENCY -> "nod_freq"
                        FusionConfig.FEATURE_DOMINANT_HEAD_MOTION_FREQUENCY -> "fft_freq"
                        FusionConfig.FEATURE_NEURALNETWORK_SCORE -> "neural"
                        else -> key
                    }
                    "$shortKey=${number(norm[key], 3)}"
                }
            } ?: "—"

            val primaryFusionSummary = if (fusion == null) {
                "—"
            } else {
                "${fusion.primaryFusion.uppercase(Locale.US)}: score=${number(fusion.primaryScore, 3)} / " +
                    "thr=${number(fusion.primaryThreshold, 2)} -> " +
                    if (fusion.isDrowsy) "DROWSY (1)" else "ALERT (0)"
            }

            fun formatMethod(methodKey: String): String {
                val detail = fusion?.fusionDetails?.get(methodKey) ?: return "—"
                val predLabel = if (detail.isDrowsy) "DROWSY (1)" else "ALERT (0)"
                return "score=${number(detail.score, 3)} (thr=${number(detail.threshold, 2)}) -> $predLabel"
            }

            val progressText = if (eval != null) {
                "Elapsed ${number(eval.elapsedSec, 1)}s / Window ${number(eval.windowSec, 0)}s"
            } else if (snapshot != null) {
                "t=${number(snapshot.timestampSec, 1)}s"
            } else {
                "—"
            }

            val shortStatusLabel = when (status) {
                PipelineStatus.WARMING_UP -> "WARMUP"
                PipelineStatus.WAITING_FOR_CAMERA_CLIP -> "WAIT CLIP"
                PipelineStatus.WAITING_FOR_VALID_WINDOW -> "WAIT VALID"
                PipelineStatus.ALERT -> "ALERT"
                PipelineStatus.DROWSY -> "DROWSY"
            }

            val shortProgress = if (eval != null) {
                "${eval.elapsedSec.toInt()}s/${eval.windowSec.toInt()}s"
            } else if (snapshot != null) {
                "${snapshot.timestampSec.toInt()}s"
            } else {
                ""
            }

            val compactWindowStatus = if (shortProgress.isNotEmpty() && (status == PipelineStatus.WARMING_UP || status == PipelineStatus.WAITING_FOR_VALID_WINDOW)) {
                "$shortStatusLabel $shortProgress"
            } else {
                shortStatusLabel
            }

            val threadTelemetrySummary = listOfNotNull(
                diagnostics.ingestMs?.let { "1A Ingest ${number(it, 2)} ms" },
                diagnostics.inferenceMs?.let { "1B MediaPipe(${diagnostics.activeDelegate ?: "GPU"}) ${number(it, 1)} ms" },
                diagnostics.physicalProcessingMs?.let { "1B Physical ${number(it, 2)} ms" },
                diagnostics.letterboxMs?.let { "1B Letterbox(5Hz) ${number(it, 2)} ms" },
                diagnostics.lazyFftMs?.let { "2 FFT(1Hz) ${number(it, 2)} ms" },
                diagnostics.onnxInferenceMs?.let { "2+3 ONNX(2T) ${number(it, 1)} ms" },
                diagnostics.workerTotalMs?.let { "2 WorkerTotal ${number(it, 1)} ms" },
            ).joinToString(" • ").ifBlank { "—" }

            return MonitoringPresentation(
                cameraStatus = when (state.camera) {
                    CameraRuntimeState.IDLE -> "Chưa khởi động"
                    CameraRuntimeState.PERMISSION_REQUIRED -> "Cần quyền camera"
                    CameraRuntimeState.STARTING -> "Đang khởi động"
                    CameraRuntimeState.AVAILABLE -> "Sẵn sàng"
                    CameraRuntimeState.UNAVAILABLE -> "Không có camera"
                    CameraRuntimeState.ERROR -> "Lỗi camera"
                },
                faceStatus = when (state.face) {
                    FaceRuntimeState.IDLE -> "Chưa khởi động"
                    FaceRuntimeState.INITIALIZING -> "Đang khởi tạo MediaPipe"
                    FaceRuntimeState.NO_FACE -> "Không thấy khuôn mặt"
                    FaceRuntimeState.FACE_DETECTED -> "Đã nhận diện khuôn mặt"
                    FaceRuntimeState.ERROR -> "Lỗi Face Landmarker"
                },
                ear = number(physical?.ear, 3),
                mar = number(physical?.mar, 3),
                perclos = physical?.perclosPct?.let { "${number(it, 1)}%" } ?: "—",
                headPose = triple(physical?.pitchDeg, physical?.yawDeg, physical?.rollDeg),
                timebaseStatus = "${diagnostics.timebaseRelation} / offset ${diagnostics.timebaseOffsetNs?.let { "$it ns" } ?: "—"}",
                performance = listOfNotNull(
                    diagnostics.cameraFps?.let { "Camera ${number(it, 1)} FPS" },
                    diagnostics.inferenceMs?.let { "MediaPipe ${number(it, 1)} ms" },
                    diagnostics.physicalProcessingMs?.let { "Physical ${number(it, 1)} ms" },
                ).joinToString(" • ").ifBlank { "—" },
                error = state.message,
                pipelineStatus = status,
                pipelineStatusLabel = status.displayLabel,
                pipelineProgressText = progressText,
                compactWindowStatus = compactWindowStatus,
                readinessSummary = readinessSummary,
                coverageSummary = coverageSummary,
                eyeSummary = eyeSummary,
                mouthSummary = mouthSummary,
                headPoseDetail = headPoseDetail,
                nodSummary = nodSummary,
                headMotionSummary = headMotionSummary,
                neuralSummary = neuralSummary,
                normalizedFeaturesSummary = normalizedFeaturesSummary,
                primaryFusionSummary = primaryFusionSummary,
                weightedMeanSummary = formatMethod(FusionConfig.FUSION_WEIGHTED_MEAN),
                noisyOrSummary = formatMethod(FusionConfig.FUSION_NOISY_OR),
                cooccurrenceSummary = formatMethod(FusionConfig.FUSION_COOCCURRENCE),
                threadTelemetrySummary = threadTelemetrySummary,
                lastSkipReason = eval?.lastSkipReason,
                balanceBrightnessEnabled = diagnostics.balanceBrightnessEnabled,
                isDrowsyPrediction = fusion?.isDrowsy == true,
            )
        }

        private fun number(value: Double?, decimals: Int): String =
            value?.takeIf { it.isFinite() }?.let { String.format(Locale.US, "%.${decimals}f", it) } ?: "—"

        private fun triple(pitch: Double?, yaw: Double?, roll: Double?): String =
            if (pitch == null && yaw == null && roll == null) "—"
            else "P ${number(pitch, 1)}° • Y ${number(yaw, 1)}° • R ${number(roll, 1)}°"
    }
}
