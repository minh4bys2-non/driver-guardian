package com.example.driverguardian.ui.screens.driving

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Map
import androidx.compose.material.icons.filled.MusicNote
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.driverguardian.ui.components.DashboardCard
import com.example.driverguardian.ui.components.MetricCard
import com.example.driverguardian.ui.mock.DrivingUiState
import com.example.driverguardian.ui.mock.MockData
import com.example.driverguardian.ui.session.DrivingSessionUiState
import com.example.driverguardian.ui.session.EventSubmissionState
import com.example.driverguardian.ui.session.CompletionState
import com.example.driverguardian.ui.theme.DriverGuardianTheme
import com.example.driverguardian.ui.theme.SafeGreen
import kotlinx.coroutines.launch

@Composable
fun ActiveDrivingScreen(
    sessionState: DrivingSessionUiState,
    snackbarHostState: SnackbarHostState,
    onDangerDemo: () -> Unit,
    onDangerPersisted: () -> Unit,
    onFinishTrip: () -> Unit,
    onTripCompleted: () -> Unit
) {
    val scope = rememberCoroutineScope()
    val drivingState = MockData.drivingNormal.copy(
        driverName = sessionState.activeDriver?.fullName ?: "Không xác định tài xế phiên",
        vehiclePlate = sessionState.activeVehicle?.plateNumber ?: "Không xác định phương tiện phiên",
        drivingTime = sessionState.activeSession?.let { "${it.durationSeconds}s (khởi tạo)" } ?: "Chưa có",
        alertCount = sessionState.activeSession?.totalAlerts ?: 0,
        lastAlertAgo = sessionState.lastEvent?.let { "Vừa lưu #${it.id}" } ?: "Chưa có",
        syncStatus = sessionState.activeSession?.syncStatus ?: "Chưa có phiên"
    )
    LaunchedEffect(sessionState.eventSubmissionState) {
        when (val submission = sessionState.eventSubmissionState) {
            is EventSubmissionState.Success -> onDangerPersisted()
            is EventSubmissionState.Error -> snackbarHostState.showSnackbar(submission.message)
            else -> Unit
        }
    }
    LaunchedEffect(sessionState.completionState) {
        when (val completion = sessionState.completionState) {
            CompletionState.Success -> onTripCompleted()
            is CompletionState.Error -> snackbarHostState.showSnackbar(completion.message)
            else -> Unit
        }
    }
    ActiveDrivingContent(
        state = drivingState,
        sessionLabel = sessionState.activeSession?.let { "Phiên #${it.id} • ${it.status} • ${it.syncStatus}" }
            ?: "Chưa có phiên backend",
        dangerEnabled = sessionState.activeSession != null && sessionState.eventSubmissionState != EventSubmissionState.Submitting,
        dangerSubmitting = sessionState.eventSubmissionState == EventSubmissionState.Submitting,
        finishEnabled = sessionState.activeSession != null &&
            (sessionState.activeSession.status == "ACTIVE" || sessionState.completedSession?.id == sessionState.activeSession.id) &&
            sessionState.completionState != CompletionState.Submitting,
        finishSubmitting = sessionState.completionState == CompletionState.Submitting,
        onPause = { scope.launch { snackbarHostState.showSnackbar("Đã tạm dừng trạng thái mô phỏng") } },
        onFinishTrip = onFinishTrip,
        onMockFeature = { scope.launch { snackbarHostState.showSnackbar("Chức năng đang được mô phỏng") } },
        onDangerDemo = onDangerDemo
    )
}

@Composable
fun ActiveDrivingContent(
    state: DrivingUiState,
    sessionLabel: String,
    dangerEnabled: Boolean,
    dangerSubmitting: Boolean,
    finishEnabled: Boolean,
    finishSubmitting: Boolean,
    onPause: () -> Unit,
    onFinishTrip: () -> Unit,
    onMockFeature: () -> Unit,
    onDangerDemo: () -> Unit
) {
    Column(modifier = Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text("${state.driverName} • ${state.vehiclePlate}", style = MaterialTheme.typography.headlineMedium)
                Text(sessionLabel, color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.titleMedium)
            }
            Icon(Icons.Default.Videocam, contentDescription = null, tint = SafeGreen, modifier = Modifier.size(42.dp))
        }

        Row(modifier = Modifier.weight(1f), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            Card(
                modifier = Modifier.weight(1.05f).fillMaxSize(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                border = BorderStroke(2.dp, state.level.color)
            ) {
                Column(
                    modifier = Modifier.fillMaxSize().padding(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Box(
                        modifier = Modifier
                            .size(220.dp)
                            .background(state.level.color.copy(alpha = 0.18f), CircleShape),
                        contentAlignment = Alignment.Center
                    ) {
                        Box(
                            modifier = Modifier
                                .size(158.dp)
                                .background(state.level.color, CircleShape),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(state.statusLabel, color = MaterialTheme.colorScheme.onPrimary, style = MaterialTheme.typography.headlineMedium)
                        }
                    }
                    Spacer(Modifier.height(20.dp))
                    Text(state.statusMessage, style = MaterialTheme.typography.titleLarge)
                    Spacer(Modifier.height(14.dp))
                    OutlinedButton(onClick = onDangerDemo, enabled = dangerEnabled) {
                        Text(if (dangerSubmitting) "Đang lưu cảnh báo…" else "Demo cảnh báo nguy hiểm")
                    }
                }
            }

            Column(modifier = Modifier.weight(1.2f), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                DashboardCard("Thông số AI demo / mock") {
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        MetricCard("Buồn ngủ", "${state.prediction.drowsiness}%", Modifier.weight(1f), state.level.color)
                        MetricCard("Độ tin cậy", "${state.prediction.confidence}%", Modifier.weight(1f))
                        MetricCard("EAR", state.prediction.ear, Modifier.weight(1f))
                    }
                    Spacer(Modifier.height(10.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        MetricCard("MAR", state.prediction.mar, Modifier.weight(1f))
                        MetricCard("Tư thế đầu", state.prediction.headPose, Modifier.weight(2f))
                    }
                }
                DashboardCard("Phiên lái") {
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        MetricCard("Thời gian lái", state.drivingTime, Modifier.weight(1f))
                        MetricCard("Cảnh báo", state.alertCount.toString(), Modifier.weight(1f), state.level.color)
                    }
                    Spacer(Modifier.height(10.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        MetricCard("Cảnh báo gần nhất", state.lastAlertAgo, Modifier.weight(1f))
                        MetricCard("Đồng bộ", state.syncStatus, Modifier.weight(1f), SafeGreen)
                    }
                }
            }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            OutlinedButton(onClick = onPause, modifier = Modifier.weight(1f).height(56.dp)) {
                Icon(Icons.Default.Pause, contentDescription = null)
                Text("  Tạm dừng")
            }
            Button(onClick = onFinishTrip, enabled = finishEnabled, modifier = Modifier.weight(1f).height(56.dp)) {
                Icon(Icons.Default.Stop, contentDescription = null)
                Text(if (finishSubmitting) "  Đang kết thúc…" else "  Kết thúc chuyến")
            }
            OutlinedButton(onClick = onMockFeature, modifier = Modifier.weight(1f).height(56.dp)) {
                Icon(Icons.Default.MusicNote, contentDescription = null)
                Text("  Mở nhạc")
            }
            OutlinedButton(onClick = onMockFeature, modifier = Modifier.weight(1f).height(56.dp)) {
                Icon(Icons.Default.Map, contentDescription = null)
                Text("  Mở bản đồ")
            }
        }
    }
}

@Preview(widthDp = 1280, heightDp = 720, showBackground = true, name = "Normal")
@Composable
private fun ActiveDrivingNormalPreview() {
    DriverGuardianTheme {
        Box(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background).padding(20.dp)) {
            ActiveDrivingContent(MockData.drivingNormal, "Phiên demo", true, false, true, false, {}, {}, {}, {})
        }
    }
}

@Preview(widthDp = 1280, heightDp = 720, showBackground = true, name = "Warning")
@Composable
private fun ActiveDrivingWarningPreview() {
    DriverGuardianTheme {
        Box(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background).padding(20.dp)) {
            ActiveDrivingContent(MockData.drivingWarning, "Phiên demo", true, false, true, false, {}, {}, {}, {})
        }
    }
}

@Preview(widthDp = 1280, heightDp = 720, showBackground = true, name = "Danger")
@Composable
private fun ActiveDrivingDangerPreview() {
    DriverGuardianTheme {
        Box(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background).padding(20.dp)) {
            ActiveDrivingContent(MockData.drivingDanger, "Phiên demo", true, false, true, false, {}, {}, {}, {})
        }
    }
}
