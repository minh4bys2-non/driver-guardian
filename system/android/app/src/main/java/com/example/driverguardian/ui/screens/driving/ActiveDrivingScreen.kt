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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.driverguardian.ai.fusion.PipelineStatus
import com.example.driverguardian.ai.monitoring.CameraPermissionState
import com.example.driverguardian.ai.monitoring.model.MonitoringDiagnostics
import com.example.driverguardian.ai.monitoring.model.MonitoringRuntimeState
import com.example.driverguardian.ui.components.KeepScreenOn
import com.example.driverguardian.ui.mock.DrivingUiState
import com.example.driverguardian.ui.mock.MockData
import com.example.driverguardian.ui.monitoring.*
import com.example.driverguardian.ui.session.*
import com.example.driverguardian.ui.theme.AccentBlue
import com.example.driverguardian.ui.theme.DangerRed
import com.example.driverguardian.ui.theme.DriverGuardianTheme
import com.example.driverguardian.ui.theme.SafeGreen
import com.example.driverguardian.ui.theme.WarningYellow
import kotlinx.coroutines.launch

@Composable
fun ActiveDrivingScreen(
    sessionState: DrivingSessionUiState,
    monitoringViewModel: MonitoringViewModel,
    snackbarHostState: SnackbarHostState,
    onDangerDemo: () -> Unit = {},
    onDangerPersisted: () -> Unit,
    onFinishTrip: () -> Unit,
    onTripCompleted: () -> Unit,
) {
    KeepScreenOn()
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val snapshot by monitoringViewModel.snapshot.collectAsStateWithLifecycle()
    val runtimeState by monitoringViewModel.runtimeState.collectAsStateWithLifecycle()
    val diagnostics by monitoringViewModel.diagnostics.collectAsStateWithLifecycle()
    val balanceBrightnessEnabled by monitoringViewModel.balanceBrightnessEnabled.collectAsStateWithLifecycle()
    val presentation = remember(snapshot, runtimeState, diagnostics) {
        MonitoringPresentation.from(snapshot, runtimeState, diagnostics)
    }
    var cameraGranted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED,
        )
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
            CompletionState.Success -> {
                monitoringViewModel.stopSession()
                onTripCompleted()
            }
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
        balanceBrightnessEnabled = balanceBrightnessEnabled,
        onToggleBalanceBrightness = monitoringViewModel::setBalanceBrightness,
        sessionLabel = sessionState.activeSession?.let { "Phiên #${it.id} • ${it.status} • ${it.syncStatus}" }
            ?: "Chưa có phiên backend",
        finishEnabled = sessionState.activeSession?.status == "ACTIVE" && sessionState.completionState != CompletionState.Submitting,
        finishSubmitting = sessionState.completionState == CompletionState.Submitting,
        cameraContent = { CameraPreview(monitoringViewModel::attachCamera, Modifier.fillMaxSize()) },
        onPause = { scope.launch { snackbarHostState.showSnackbar("Monitoring tiếp tục; pause chưa được hỗ trợ") } },
        onFinishTrip = onFinishTrip,
    )
}

@Composable
private fun ActiveDrivingContent(
    state: DrivingUiState,
    presentation: MonitoringPresentation,
    balanceBrightnessEnabled: Boolean,
    onToggleBalanceBrightness: (Boolean) -> Unit,
    sessionLabel: String,
    finishEnabled: Boolean,
    finishSubmitting: Boolean,
    cameraContent: @Composable () -> Unit,
    onPause: () -> Unit,
    onFinishTrip: () -> Unit,
) {
    val pipelineColor = when (presentation.pipelineStatus) {
        PipelineStatus.ALERT -> SafeGreen
        PipelineStatus.DROWSY -> DangerRed
        PipelineStatus.WARMING_UP -> AccentBlue
        PipelineStatus.WAITING_FOR_CAMERA_CLIP,
        PipelineStatus.WAITING_FOR_VALID_WINDOW -> WarningYellow
    }

    Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column {
                Text("${state.driverName} • ${state.vehiclePlate}", style = MaterialTheme.typography.headlineMedium)
                Text(
                    sessionLabel,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    style = MaterialTheme.typography.titleMedium,
                )
            }
            Icon(
                Icons.Default.Videocam,
                "Camera monitoring",
                tint = pipelineColor,
                modifier = Modifier.size(36.dp),
            )
        }
        Row(Modifier.weight(1f).fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Card(
                Modifier.weight(1f).fillMaxSize(),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary),
            ) {
                Box(Modifier.fillMaxSize()) {
                    cameraContent()

                    // Bottom overlay: Action buttons (Pause and Finish Trip)
                    Box(
                        modifier = Modifier
                            .align(Alignment.BottomCenter)
                            .fillMaxWidth()
                            .background(
                                Brush.verticalGradient(
                                    listOf(
                                        Color.Transparent,
                                        MaterialTheme.colorScheme.surface.copy(alpha = 0.85f),
                                    )
                                )
                            )
                            .padding(horizontal = 10.dp, vertical = 8.dp),
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            FilledTonalButton(
                                onClick = onPause,
                                modifier = Modifier.weight(1f).height(46.dp),
                                colors = ButtonDefaults.filledTonalButtonColors(
                                    containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.9f),
                                    contentColor = MaterialTheme.colorScheme.onSurface,
                                ),
                            ) {
                                Icon(Icons.Default.Pause, contentDescription = "Tạm dừng", modifier = Modifier.size(18.dp))
                                Spacer(Modifier.width(6.dp))
                                Text("Tạm dừng", fontWeight = FontWeight.SemiBold)
                            }
                            Button(
                                onClick = onFinishTrip,
                                enabled = finishEnabled,
                                modifier = Modifier.weight(1f).height(46.dp),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = DangerRed,
                                    contentColor = MaterialTheme.colorScheme.onError,
                                ),
                            ) {
                                Icon(Icons.Default.Stop, contentDescription = "Kết thúc", modifier = Modifier.size(18.dp))
                                Spacer(Modifier.width(6.dp))
                                Text(
                                    if (finishSubmitting) "Đang dừng…" else "Kết thúc chuyến",
                                    fontWeight = FontWeight.SemiBold,
                                )
                            }
                        }
                    }
                }
            }
            MonitoringDebugPanel(
                presentation = presentation,
                modifier = Modifier.weight(1.3f).fillMaxSize(),
                balanceBrightnessEnabled = balanceBrightnessEnabled,
                onToggleBalanceBrightness = onToggleBalanceBrightness,
            )
        }
    }
}

@Preview(widthDp = 1280, heightDp = 720, showBackground = true)
@Composable
private fun ActiveDrivingPreview() {
    DriverGuardianTheme {
        val p = MonitoringPresentation.from(null, MonitoringRuntimeState(), MonitoringDiagnostics())
        Box(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background).padding(20.dp)) {
            ActiveDrivingContent(
                state = MockData.drivingNormal,
                presentation = p,
                balanceBrightnessEnabled = false,
                onToggleBalanceBrightness = {},
                sessionLabel = "Phiên demo",
                finishEnabled = true,
                finishSubmitting = false,
                cameraContent = {},
                onPause = {},
                onFinishTrip = {},
            )
        }
    }
}
