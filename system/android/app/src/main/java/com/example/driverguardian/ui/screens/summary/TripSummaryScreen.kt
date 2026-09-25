package com.example.driverguardian.ui.screens.summary

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.driverguardian.ui.components.DashboardCard
import com.example.driverguardian.ui.components.MetricCard
import com.example.driverguardian.ui.mock.MockData
import com.example.driverguardian.ui.theme.SafeGreen
import com.example.driverguardian.ui.theme.WarningYellow
import kotlinx.coroutines.launch

@Composable
fun TripSummaryScreen(
    snackbarHostState: SnackbarHostState,
    onHome: () -> Unit,
    onDetail: () -> Unit
) {
    val summary = MockData.tripSummary
    val scope = rememberCoroutineScope()
    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text("Tổng kết chuyến đi", style = MaterialTheme.typography.headlineMedium)
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            MetricCard("Bắt đầu", summary.startTime, Modifier.weight(1f))
            MetricCard("Kết thúc", summary.endTime, Modifier.weight(1f))
            MetricCard("Tổng thời gian", summary.duration, Modifier.weight(1f), SafeGreen)
            MetricCard("Tổng cảnh báo", summary.totalAlerts.toString(), Modifier.weight(1f), WarningYellow)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            DashboardCard("Đánh giá", modifier = Modifier.weight(1f)) {
                Text("Điểm an toàn: ${summary.safetyScore}/100", style = MaterialTheme.typography.headlineMedium, color = SafeGreen)
                Text("Xếp loại: ${summary.rank}", color = MaterialTheme.colorScheme.onSurfaceVariant)
                Spacer(Modifier.height(12.dp))
                LinearProgressIndicator(progress = { summary.safetyScore / 100f }, modifier = Modifier.fillMaxWidth(), color = SafeGreen)
                Spacer(Modifier.height(12.dp))
                Text("Cảnh báo mức 1: ${summary.levelOneAlerts}")
                Text("Cảnh báo mức 2: ${summary.levelTwoAlerts}")
                Text("Khung giờ nguy cơ cao nhất: ${summary.riskyWindow}")
            }
            DashboardCard("Timeline", modifier = Modifier.weight(1f)) {
                summary.timeline.forEach { Text("• $it", style = MaterialTheme.typography.bodyLarge) }
            }
            DashboardCard("Khuyến nghị", modifier = Modifier.weight(1f), borderColor = WarningYellow) {
                summary.recommendations.forEach { Text("• $it", style = MaterialTheme.typography.bodyLarge) }
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(onClick = onHome, modifier = Modifier.weight(1f).height(56.dp)) { Text("Về trang chủ") }
            OutlinedButton(onClick = onDetail, modifier = Modifier.weight(1f).height(56.dp)) { Text("Xem chi tiết") }
            OutlinedButton(
                onClick = { scope.launch { snackbarHostState.showSnackbar("Xuất báo cáo đang được mô phỏng") } },
                modifier = Modifier.weight(1f).height(56.dp)
            ) { Text("Xuất báo cáo") }
        }
    }
}
