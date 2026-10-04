package com.example.driverguardian.ui.screens.history

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.driverguardian.ui.components.DashboardCard
import com.example.driverguardian.ui.components.MetricCard
import com.example.driverguardian.ui.history.*
import com.example.driverguardian.ui.theme.SafeGreen
import com.example.driverguardian.ui.theme.WarningYellow

@Composable
fun TripDetailScreen(
    id: Int?,
    state: TripHistoryUiState,
    onLoad: (Int) -> Unit
) {
    if (id == null) {
        Text("Mã chuyến đi không hợp lệ.", color = MaterialTheme.colorScheme.error)
        return
    }
    LaunchedEffect(id) { onLoad(id) }
    when (val loadState = state.detailState) {
        TripLoadState.Idle, TripLoadState.Loading -> CircularProgressIndicator()
        is TripLoadState.Empty -> Text(loadState.message)
        is TripLoadState.Error -> Column {
            Text(loadState.message, color = MaterialTheme.colorScheme.error)
            Button(onClick = { onLoad(id) }) { Text("Thử lại") }
        }
        TripLoadState.Success -> {
            val trip = state.selectedTrip?.takeIf { it.id == id }
            if (trip == null) {
                CircularProgressIndicator()
                return
            }
            val counts = countEvents(state.selectedEvents)
            Column(
                modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Text("Chi tiết chuyến đi #${trip.id}", style = MaterialTheme.typography.headlineMedium)
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    MetricCard("Tài xế", trip.driverName, Modifier.weight(1f))
                    MetricCard("Xe", trip.plateNumber, Modifier.weight(1f))
                    MetricCard("Tổng thời gian", formatDuration(trip.durationSeconds), Modifier.weight(1f), SafeGreen)
                    MetricCard("Điểm an toàn", formatNullableNumber(trip.safetyScore), Modifier.weight(1f), SafeGreen)
                }
                Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                    DashboardCard("Thông tin chuyến", modifier = Modifier.weight(1f)) {
                        Text("Bắt đầu: ${formatDateTime(trip.startTime)}")
                        Text("Kết thúc: ${formatDateTime(trip.endTime)}")
                        Text("Cảnh báo mức 1: ${counts.warnings}", color = WarningYellow)
                        Text("Cảnh báo mức 2: ${counts.dangers}", color = WarningYellow)
                        Text("Phiên bản AI: ${trip.versionName ?: "—"}")
                    }
                    DashboardCard("Timeline", modifier = Modifier.weight(1f)) {
                        Text("• ${formatDateTime(trip.startTime)} • Bắt đầu")
                        state.selectedEvents.forEach { Text("• ${formatDateTime(it.eventTime)} • ${it.driverState}") }
                        Text("• ${formatDateTime(trip.endTime)} • Kết thúc")
                    }
                    DashboardCard("Sự kiện đã lưu", modifier = Modifier.weight(1f)) {
                        if (state.selectedEvents.isEmpty()) Text("Không có cảnh báo.")
                        state.selectedEvents.forEach {
                            Text("${formatDateTime(it.eventTime)} • ${it.driverState} • confidence ${formatNullableNumber(it.confidence)} • EAR ${formatNullableNumber(it.earValue)}")
                        }
                    }
                }
            }
        }
    }
}
