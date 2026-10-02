package com.example.driverguardian.ui.monitoring

import com.example.driverguardian.ai.monitoring.model.*
import java.util.Locale

data class MonitoringPresentation(
    val cameraStatus: String,
    val faceStatus: String,
    val sensorStatus: String,
    val ear: String,
    val mar: String,
    val headPose: String,
    val rawDeviceOrientation: String,
    val deviceOrientation: String,
    val sampleAge: String,
    val timebaseStatus: String,
    val performance: String,
    val error: String?,
    val dangerActionLabel: String = "DEMO: cảnh báo nguy hiểm",
) {
    companion object {
        fun from(snapshot: MonitoringSnapshot?, state: MonitoringRuntimeState, diagnostics: MonitoringDiagnostics): MonitoringPresentation {
            val physical = snapshot?.physical
            val motion = snapshot?.deviceMotion
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
                sensorStatus = when (state.sensor) {
                    SensorRuntimeState.IDLE -> "Chưa khởi động"
                    SensorRuntimeState.CALIBRATING -> "Đang hiệu chuẩn"
                    SensorRuntimeState.AVAILABLE -> "Sẵn sàng"
                    SensorRuntimeState.UNAVAILABLE -> "Không có cảm biến"
                    SensorRuntimeState.ERROR -> "Lỗi cảm biến"
                },
                ear = number(physical?.ear, 3),
                mar = number(physical?.mar, 3),
                headPose = triple(physical?.pitchDeg, physical?.yawDeg, physical?.rollDeg),
                rawDeviceOrientation = triple(motion?.rawPitchDeg, motion?.rawYawDeg, motion?.rawRollDeg),
                deviceOrientation = triple(motion?.deltaPitchDeg, motion?.deltaYawDeg, motion?.deltaRollDeg),
                sampleAge = motion?.sampleAgeMs?.let { "${number(it, 1)} ms" } ?: "—",
                timebaseStatus = "${diagnostics.timebaseRelation} / offset ${diagnostics.timebaseOffsetNs?.let { "$it ns" } ?: "—"}",
                performance = listOfNotNull(
                    diagnostics.cameraFps?.let { "Camera ${number(it, 1)} FPS" },
                    diagnostics.inferenceMs?.let { "MediaPipe ${number(it, 1)} ms" },
                    diagnostics.physicalProcessingMs?.let { "Physical ${number(it, 1)} ms" },
                    diagnostics.sensorHz?.let { "Sensor ${number(it, 1)} Hz" },
                    diagnostics.sensorProcessingMs?.let { "Sensor ${number(it, 2)} ms" },
                ).joinToString(" • ").ifBlank { "—" },
                error = state.message,
            )
        }

        private fun number(value: Double?, decimals: Int): String =
            value?.takeIf { it.isFinite() }?.let { String.format(Locale.US, "%.${decimals}f", it) } ?: "—"

        private fun triple(pitch: Double?, yaw: Double?, roll: Double?): String =
            if (pitch == null && yaw == null && roll == null) "—"
            else "P ${number(pitch, 1)}° • Y ${number(yaw, 1)}° • R ${number(roll, 1)}°"
    }
}
