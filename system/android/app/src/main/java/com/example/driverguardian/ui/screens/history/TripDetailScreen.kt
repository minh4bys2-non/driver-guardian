package com.example.driverguardian.ui.screens.history

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.driverguardian.ui.components.DashboardCard
import com.example.driverguardian.ui.components.MetricCard
import com.example.driverguardian.ui.mock.MockData
import com.example.driverguardian.ui.theme.SafeGreen
import com.example.driverguardian.ui.theme.WarningYellow

@Composable
fun TripDetailScreen(id: String) {
    val trip = MockData.trips.firstOrNull { it.id == id } ?: MockData.trips.first()
    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text("Chi tiết chuyến đi của tôi #${trip.id}", style = MaterialTheme.typography.headlineMedium)
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            MetricCard("Hồ sơ", MockData.currentDriver.name, Modifier.weight(1f))
            MetricCard("Xe", trip.vehiclePlate, Modifier.weight(1f))
            MetricCard("Tổng thời gian", trip.duration, Modifier.weight(1f), SafeGreen)
            MetricCard("Điểm an toàn", trip.score.toString(), Modifier.weight(1f), SafeGreen)
        }
        Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            DashboardCard("Thông tin chuyến", modifier = Modifier.weight(1f)) {
                Text("Bắt đầu: 22:40")
                Text("Kết thúc: 02:15")
                Text("Số cảnh báo: ${trip.alerts}", color = WarningYellow)
                Text("Tọa độ: Không có dữ liệu vị trí", color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text("Phiên bản module AI: v1.0.0", color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            DashboardCard("Timeline của tôi", modifier = Modifier.weight(1f)) {
                MockData.tripSummary.timeline.forEach { Text("• $it") }
            }
            DashboardCard("Sự kiện cá nhân", modifier = Modifier.weight(1f)) {
                MockData.alerts.take(4).forEach {
                    Text("${it.time} • ${it.stateType} • ${it.level.label} • ${it.syncStatus}")
                }
            }
        }
    }
}
