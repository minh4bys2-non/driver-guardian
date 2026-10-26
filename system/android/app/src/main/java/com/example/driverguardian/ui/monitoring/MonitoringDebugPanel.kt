package com.example.driverguardian.ui.monitoring

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.driverguardian.ai.fusion.PipelineStatus
import com.example.driverguardian.ui.theme.AccentBlue
import com.example.driverguardian.ui.theme.DangerRed
import com.example.driverguardian.ui.theme.SafeGreen
import com.example.driverguardian.ui.theme.WarningYellow

@Composable
fun MonitoringDebugPanel(
    presentation: MonitoringPresentation,
    modifier: Modifier = Modifier,
    balanceBrightnessEnabled: Boolean = presentation.balanceBrightnessEnabled,
    onToggleBalanceBrightness: (Boolean) -> Unit = {},
) {
    var expanded by remember { mutableStateOf(false) }
    val statusColor = when (presentation.pipelineStatus) {
        PipelineStatus.ALERT -> SafeGreen
        PipelineStatus.DROWSY -> DangerRed
        PipelineStatus.WARMING_UP -> AccentBlue
        PipelineStatus.WAITING_FOR_CAMERA_CLIP,
        PipelineStatus.WAITING_FOR_VALID_WINDOW -> WarningYellow
    }

    Card(modifier = modifier) {
        Column(
            modifier = Modifier
                .padding(14.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            // Header: Camera & Face Status + Compact Window Status + Toggle Details
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = "Camera: ${presentation.cameraStatus} • Face: ${presentation.faceStatus}",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier.weight(1f),
                )
                Surface(
                    shape = MaterialTheme.shapes.small,
                    color = statusColor.copy(alpha = 0.16f),
                    border = BorderStroke(1.dp, statusColor),
                ) {
                    Text(
                        text = presentation.compactWindowStatus,
                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                        style = MaterialTheme.typography.labelMedium,
                        color = statusColor,
                        fontWeight = FontWeight.Bold,
                    )
                }
                TextButton(onClick = { expanded = !expanded }) {
                    Text(if (expanded) "Thu gọn" else "Chi tiết")
                }
            }

            // Quick summary row: Core physical metrics & Balance Brightness switch
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text(
                        text = "EAR ${presentation.ear} • MAR ${presentation.mar} • PERCLOS ${presentation.perclos}",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Medium,
                    )
                    Text(
                        text = "HEAD POSE: ${presentation.headPose}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Text(
                        text = "Cân bằng sáng",
                        style = MaterialTheme.typography.labelMedium,
                    )
                    Switch(
                        checked = balanceBrightnessEnabled,
                        onCheckedChange = onToggleBalanceBrightness,
                    )
                }
            }

            if (expanded) {
                HorizontalDivider()

                // 1. Đặc trưng vật lý & Cửa sổ
                Text(
                    text = "1. ĐẶC TRƯNG VẬT LÝ & CỬA SỔ",
                    style = MaterialTheme.typography.labelMedium,
                    color = AccentBlue,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    text = "Cửa sổ: ${presentation.pipelineProgressText} • Độ phủ: ${presentation.coverageSummary}",
                    style = MaterialTheme.typography.bodySmall,
                )
                Text(text = "Mắt: ${presentation.eyeSummary}", style = MaterialTheme.typography.bodySmall)
                Text(text = "Miệng: ${presentation.mouthSummary}", style = MaterialTheme.typography.bodySmall)
                Text(text = "Đầu & Gật: ${presentation.headPoseDetail} • ${presentation.nodSummary}", style = MaterialTheme.typography.bodySmall)
                presentation.lastSkipReason?.let { skip ->
                    Text(
                        text = "Gate Notice: $skip",
                        style = MaterialTheme.typography.bodySmall,
                        color = WarningYellow,
                    )
                }

                HorizontalDivider()

                // 2. Phân loại AI & Hợp nhất Fitness
                Text(
                    text = "2. PHÂN LOẠI AI & HỢP NHẤT FITNESS",
                    style = MaterialTheme.typography.labelMedium,
                    color = AccentBlue,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(text = "Mô hình: ${presentation.neuralSummary}", style = MaterialTheme.typography.bodySmall)
                Text(
                    text = "Quyết định: ${presentation.primaryFusionSummary}",
                    style = MaterialTheme.typography.bodySmall,
                    fontWeight = FontWeight.SemiBold,
                    color = if (presentation.isDrowsyPrediction) DangerRed else SafeGreen,
                )

                HorizontalDivider()

                // 3. Hiệu năng hệ thống (Telemetry)
                Text(
                    text = "3. HIỆU NĂNG HỆ THỐNG",
                    style = MaterialTheme.typography.labelMedium,
                    color = AccentBlue,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(text = "FPS & Độ trễ: ${presentation.performance}", style = MaterialTheme.typography.bodySmall)
                Text(text = "Đa luồng: ${presentation.threadTelemetrySummary}", style = MaterialTheme.typography.bodySmall)

                presentation.error?.let {
                    Text(
                        text = it,
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }
    }
}
