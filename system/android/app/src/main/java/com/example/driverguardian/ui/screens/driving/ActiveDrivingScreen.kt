package com.example.driverguardian.ui.screens.driving

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.driverguardian.ai.monitoring.CameraPermissionState
import com.example.driverguardian.ai.monitoring.model.MonitoringDiagnostics
import com.example.driverguardian.ai.monitoring.model.MonitoringRuntimeState
import com.example.driverguardian.ui.mock.DrivingUiState
import com.example.driverguardian.ui.mock.MockData
import com.example.driverguardian.ui.monitoring.*
import com.example.driverguardian.ui.session.*
import com.example.driverguardian.ui.theme.DriverGuardianTheme
import com.example.driverguardian.ui.theme.SafeGreen
import kotlinx.coroutines.launch

@Composable
fun ActiveDrivingScreen(
    sessionState: DrivingSessionUiState,
    monitoringViewModel: MonitoringViewModel,
    snackbarHostState: SnackbarHostState,
    onDangerDemo: () -> Unit,
    onDangerPersisted: () -> Unit,
    onFinishTrip: () -> Unit,
    onTripCompleted: () -> Unit,
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val snapshot by monitoringViewModel.snapshot.collectAsStateWithLifecycle()
    val runtimeState by monitoringViewModel.runtimeState.collectAsStateWithLifecycle()
    val diagnostics by monitoringViewModel.diagnostics.collectAsStateWithLifecycle()
    val presentation = remember(snapshot, runtimeState, diagnostics) {
        MonitoringPresentation.from(snapshot, runtimeState, diagnostics)
    }
    var cameraGranted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        cameraGranted = granted
        monitoringViewModel.updatePermission(if (granted) CameraPermissionState.GRANTED else CameraPermissionState.DENIED)
    }

    LaunchedEffect(sessionState.activeSession?.id, sessionState.activeSession?.status) {
        val session = sessionState.activeSession
        if (session != null && session.status == "ACTIVE") {
            monitoringViewModel.startSession(session.id)
            if (cameraGranted) monitoringViewModel.updatePermission(CameraPermissionState.GRANTED)
            else permissionLauncher.launch(Manifest.permission.CAMERA)
        } else if (session != null) monitoringViewModel.stopSession()
    }
    LaunchedEffect(sessionState.eventSubmissionState) {
        when (val submission = sessionState.eventSubmissionState) {
            is EventSubmissionState.Success -> onDangerPersisted()
            is EventSubmissionState.Error -> snackbarHostState.showSnackbar(submission.message)
            else -> Unit
        }
    }
    LaunchedEffect(sessionState.completionState) {
        when (val completion = sessionState.completionState) {
            CompletionState.Success -> { monitoringViewModel.stopSession(); onTripCompleted() }
            is CompletionState.Error -> snackbarHostState.showSnackbar(completion.message)
            else -> Unit
        }
    }

    val drivingState = MockData.drivingNormal.copy(
        driverName = sessionState.activeDriver?.fullName ?: "Không xác định tài xế phiên",
        vehiclePlate = sessionState.activeVehicle?.plateNumber ?: "Không xác định phương tiện phiên",
        drivingTime = sessionState.activeSession?.let { "${it.durationSeconds}s (khởi tạo)" } ?: "Chưa có",
        alertCount = sessionState.activeSession?.totalAlerts ?: 0,
        lastAlertAgo = sessionState.lastEvent?.let { "Vừa lưu #${it.id}" } ?: "Chưa có",
        syncStatus = sessionState.activeSession?.syncStatus ?: "Chưa có phiên",
    )
    ActiveDrivingContent(
        state = drivingState,
        presentation = presentation,
        sessionLabel = sessionState.activeSession?.let { "Phiên #${it.id} • ${it.status} • ${it.syncStatus}" } ?: "Chưa có phiên backend",
        dangerEnabled = sessionState.activeSession != null && sessionState.eventSubmissionState != EventSubmissionState.Submitting,
        dangerSubmitting = sessionState.eventSubmissionState == EventSubmissionState.Submitting,
        finishEnabled = sessionState.activeSession?.status == "ACTIVE" && sessionState.completionState != CompletionState.Submitting,
        finishSubmitting = sessionState.completionState == CompletionState.Submitting,
        cameraContent = { CameraPreview(monitoringViewModel::attachCamera, Modifier.fillMaxSize()) },
        onPause = { scope.launch { snackbarHostState.showSnackbar("Monitoring tiếp tục; pause chưa được hỗ trợ") } },
        onFinishTrip = onFinishTrip,
        onMockFeature = { scope.launch { snackbarHostState.showSnackbar("Chức năng đang được mô phỏng") } },
        onDangerDemo = onDangerDemo,
    )
}

@Composable
private fun ActiveDrivingContent(
    state: DrivingUiState,
    presentation: MonitoringPresentation,
    sessionLabel: String,
    dangerEnabled: Boolean,
    dangerSubmitting: Boolean,
    finishEnabled: Boolean,
    finishSubmitting: Boolean,
    cameraContent: @Composable () -> Unit,
    onPause: () -> Unit,
    onFinishTrip: () -> Unit,
    onMockFeature: () -> Unit,
    onDangerDemo: () -> Unit,
) {
    Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
            Column {
                Text("${state.driverName} • ${state.vehiclePlate}", style = MaterialTheme.typography.headlineMedium)
                Text(sessionLabel, color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.titleMedium)
            }
            Icon(Icons.Default.Videocam, "Camera monitoring", tint = SafeGreen, modifier = Modifier.size(42.dp))
        }
        Row(Modifier.weight(1f), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Card(Modifier.weight(1f).fillMaxSize(), border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary)) {
                Box(Modifier.fillMaxSize()) {
                    cameraContent()
                    Text(
                        "Preview only • algorithmic landmarks unmirrored",
                        Modifier.align(Alignment.BottomStart).background(MaterialTheme.colorScheme.surface.copy(alpha = 0.8f)).padding(8.dp),
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
            Column(Modifier.weight(1.25f).fillMaxHeight(), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                MonitoringDebugPanel(presentation, Modifier.weight(1f).fillMaxWidth())
                Card(Modifier.fillMaxWidth()) {
                    Text(
                        "Backend: ${state.drivingTime} • ${state.alertCount} cảnh báo • ${state.syncStatus}",
                        Modifier.padding(12.dp),
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
                OutlinedButton(onClick = onDangerDemo, enabled = dangerEnabled, modifier = Modifier.fillMaxWidth()) {
                    Text(if (dangerSubmitting) "Đang lưu cảnh báo…" else presentation.dangerActionLabel)
                }
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            OutlinedButton(onClick = onPause, modifier = Modifier.weight(1f).height(52.dp)) { Icon(Icons.Default.Pause, null); Text("  Tạm dừng") }
            Button(onClick = onFinishTrip, enabled = finishEnabled, modifier = Modifier.weight(1f).height(52.dp)) {
                Icon(Icons.Default.Stop, null); Text(if (finishSubmitting) "  Đang kết thúc…" else "  Kết thúc chuyến")
            }
            OutlinedButton(onClick = onMockFeature, modifier = Modifier.weight(1f).height(52.dp)) { Icon(Icons.Default.MusicNote, null); Text("  Mở nhạc") }
            OutlinedButton(onClick = onMockFeature, modifier = Modifier.weight(1f).height(52.dp)) { Icon(Icons.Default.Map, null); Text("  Mở bản đồ") }
        }
    }
}

@Preview(widthDp = 1280, heightDp = 720, showBackground = true)
@Composable
private fun ActiveDrivingPreview() {
    DriverGuardianTheme {
        val p = MonitoringPresentation.from(null, MonitoringRuntimeState(), MonitoringDiagnostics())
        Box(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background).padding(20.dp)) {
            ActiveDrivingContent(MockData.drivingNormal, p, "Phiên demo", true, false, true, false, {}, {}, {}, {}, {})
        }
    }
}
