package com.example.driverguardian.ui.screens.summary

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.driverguardian.ui.components.DashboardCard
import com.example.driverguardian.ui.components.MetricCard
import com.example.driverguardian.ui.history.countEvents
import com.example.driverguardian.ui.history.formatDateTime
import com.example.driverguardian.ui.history.formatDuration
import com.example.driverguardian.ui.history.formatNullableNumber
import com.example.driverguardian.ui.session.DrivingSessionUiState
import com.example.driverguardian.ui.theme.SafeGreen
import com.example.driverguardian.ui.theme.WarningYellow
import kotlinx.coroutines.launch

@Composable
fun TripSummaryScreen(
    sessionState: DrivingSessionUiState,
    snackbarHostState: SnackbarHostState,
    onHome: () -> Unit,
    onDetail: (Int) -> Unit
) {
    val session = sessionState.completedSession
    val events = sessionState.completedSessionEvents
    val counts = countEvents(events)
    val scope = rememberCoroutineScope()
    if (session == null) {
        Box(Modifier.fillMaxSize()) { Text("Chưa có dữ liệu chuyến đi đã hoàn thành.") }
        return
    }
    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text("Tổng kết chuyến đi #${session.id}", style = MaterialTheme.typography.headlineMedium)
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            MetricCard("Bắt đầu", formatDateTime(session.startTime), Modifier.weight(1f))
            MetricCard("Kết thúc", formatDateTime(session.endTime), Modifier.weight(1f))
            MetricCard("Tổng thời gian", formatDuration(session.durationSeconds), Modifier.weight(1f), SafeGreen)
            MetricCard("Tổng cảnh báo", session.totalAlerts.toString(), Modifier.weight(1f), WarningYellow)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            DashboardCard("Dữ liệu chuyến đi", modifier = Modifier.weight(1f)) {
                Text("Điểm an toàn: ${formatNullableNumber(session.safetyScore)}", style = MaterialTheme.typography.headlineMedium, color = SafeGreen)
                Text("Cảnh báo mức 1: ${counts.warnings}")
                Text("Cảnh báo mức 2: ${counts.dangers}")
                Text("Trạng thái: ${session.status}")
                Text("Đồng bộ: ${session.syncStatus}", color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            DashboardCard("Timeline đã lưu", modifier = Modifier.weight(1f)) {
                Text("• Bắt đầu: ${formatDateTime(session.startTime)}")
                events.forEach { Text("• ${formatDateTime(it.eventTime)} • ${it.driverState} • mức ${it.alertLevel}") }
                Text("• Kết thúc: ${formatDateTime(session.endTime)}")
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(onClick = onHome, modifier = Modifier.weight(1f).height(56.dp)) { Text("Về trang chủ") }
            OutlinedButton(onClick = { onDetail(session.id) }, modifier = Modifier.weight(1f).height(56.dp)) { Text("Xem chi tiết") }
            OutlinedButton(
                onClick = { scope.launch { snackbarHostState.showSnackbar("Xuất báo cáo chưa được hỗ trợ trong phase này") } },
                modifier = Modifier.weight(1f).height(56.dp)
            ) { Text("Xuất báo cáo") }
        }
    }
}
