package com.example.driverguardian.ui.mock

import androidx.compose.ui.graphics.Color
import com.example.driverguardian.ui.theme.DangerRed
import com.example.driverguardian.ui.theme.SafeGreen
import com.example.driverguardian.ui.theme.WarningYellow

enum class AlertLevel(val label: String, val color: Color) {
    Safe("An toàn", SafeGreen),
    Warning("Cảnh báo mức 1", WarningYellow),
    Danger("Nguy hiểm", DangerRed)
}

data class DriverUiModel(
    val id: String,
    val name: String,
    val code: String,
    val status: String
)

data class VehicleUiModel(
    val id: String,
    val name: String,
    val plate: String,
    val type: String,
    val deviceCode: String,
    val deviceStatus: String
)

data class SystemCheckUiModel(
    val name: String,
    val status: String,
    val description: String,
    val warning: Boolean = false
)

data class PredictionUiModel(
    val drowsiness: Int,
    val confidence: Int,
    val ear: String,
    val mar: String,
    val headPose: String
)

data class DrivingUiState(
    val driverName: String,
    val vehiclePlate: String,
    val statusLabel: String,
    val statusMessage: String,
    val level: AlertLevel,
    val prediction: PredictionUiModel,
    val drivingTime: String,
    val alertCount: Int,
    val lastAlertAgo: String,
    val syncStatus: String
)

data class TripSummaryUiModel(
    val startTime: String,
    val endTime: String,
    val duration: String,
    val totalAlerts: Int,
    val levelOneAlerts: Int,
    val levelTwoAlerts: Int,
    val riskyWindow: String,
    val safetyScore: Int,
    val rank: String,
    val timeline: List<String>,
    val recommendations: List<String>
)

data class TripHistoryUiModel(
    val id: String,
    val date: String,
    val driverName: String,
    val vehiclePlate: String,
    val duration: String,
    val alerts: Int,
    val score: Int,
    val syncStatus: String
)

data class AlertEventUiModel(
    val id: String,
    val time: String,
    val stateType: String,
    val level: AlertLevel,
    val confidence: Int,
    val ear: String,
    val mar: String,
    val acknowledged: Boolean,
    val syncStatus: String
)

data class HourlyStatisticUiModel(
    val hour: Int,
    val value: Int
)
