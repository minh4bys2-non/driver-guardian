package com.example.driverguardian.ui.screens.alerts

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.driverguardian.ui.components.DashboardCard
import com.example.driverguardian.ui.mock.AlertEventUiModel
import com.example.driverguardian.ui.mock.AlertLevel
import com.example.driverguardian.ui.mock.MockData
import com.example.driverguardian.ui.theme.SafeGreen

@Composable
fun AlertHistoryScreen() {
    var filter by remember { mutableStateOf("Tất cả") }
    val filtered = MockData.alerts.filter {
        when (filter) {
            "Cảnh báo nhẹ" -> it.level == AlertLevel.Warning
            "Nguy hiểm" -> it.level == AlertLevel.Danger
            "Đã xác nhận" -> it.acknowledged
            "Chưa xác nhận" -> !it.acknowledged
            else -> true
        }
    }
    Column(modifier = Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(14.dp)) {
        Text("Lịch sử cảnh báo của tôi", style = MaterialTheme.typography.headlineMedium)
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            listOf("Tất cả", "Cảnh báo nhẹ", "Nguy hiểm", "Đã xác nhận", "Chưa xác nhận").forEach {
                FilterChip(selected = filter == it, onClick = { filter = it }, label = { Text(it) })
            }
        }
        LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            items(filtered) { AlertRow(it) }
        }
    }
}

@Composable
private fun AlertRow(alert: AlertEventUiModel) {
    DashboardCard(alert.time, borderColor = alert.level.color) {
        Row(
            horizontalArrangement = Arrangement.spacedBy(14.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1.1f)) {
                Text(alert.stateType, style = MaterialTheme.typography.titleLarge, color = alert.level.color)
                Text(alert.level.label, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Text("Confidence: ${alert.confidence}%", modifier = Modifier.weight(1f))
            Text("EAR: ${alert.ear}", modifier = Modifier.weight(0.7f))
            Text("MAR: ${alert.mar}", modifier = Modifier.weight(0.7f))
            Text(if (alert.acknowledged) "Đã xác nhận" else "Chưa xác nhận", color = if (alert.acknowledged) SafeGreen else alert.level.color, modifier = Modifier.weight(1f))
            Text(alert.syncStatus, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.weight(1f))
        }
    }
}
