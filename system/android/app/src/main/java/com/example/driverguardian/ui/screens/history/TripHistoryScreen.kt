package com.example.driverguardian.ui.screens.history

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.driverguardian.domain.model.TripSession
import com.example.driverguardian.ui.components.DashboardCard
import com.example.driverguardian.ui.history.*
import com.example.driverguardian.ui.theme.SafeGreen
import com.example.driverguardian.ui.theme.WarningYellow

@Composable
fun TripHistoryScreen(
    state: TripHistoryUiState,
    onLoad: (Int?) -> Unit,
    onDetail: (Int) -> Unit
) {
    var query by remember { mutableStateOf("") }
    LaunchedEffect(Unit) { onLoad(null) }
    val trips = state.trips.filter {
        it.plateNumber.contains(query, true) || formatDateTime(it.startTime).contains(query, true)
    }
    Column(modifier = Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(14.dp)) {
        Text("Lịch sử chuyến đi", style = MaterialTheme.typography.headlineMedium)
        OutlinedTextField(
            value = query,
            onValueChange = { query = it },
            leadingIcon = { Icon(Icons.Default.Search, contentDescription = null) },
            placeholder = { Text("Tìm theo ngày hoặc biển số") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth()
        )
        when (val loadState = state.historyState) {
            TripLoadState.Idle, TripLoadState.Loading -> CircularProgressIndicator()
            is TripLoadState.Empty -> Text(loadState.message)
            is TripLoadState.Error -> Column {
                Text(loadState.message, color = MaterialTheme.colorScheme.error)
                Button(onClick = { onLoad(null) }) { Text("Thử lại") }
            }
            TripLoadState.Success -> LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                items(trips, key = { it.id }) { TripRow(it, onDetail) }
            }
        }
    }
}

@Composable
private fun TripRow(trip: TripSession, onDetail: (Int) -> Unit) {
    val mutedColor = MaterialTheme.colorScheme.onSurfaceVariant
    DashboardCard(title = formatDateTime(trip.startTime)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1.2f)) {
                Text(trip.driverName, style = MaterialTheme.typography.titleMedium)
                Text(trip.plateNumber, color = mutedColor)
            }
            Text("Thời lượng: ${formatDuration(trip.durationSeconds)}", modifier = Modifier.weight(1f))
            Text("Cảnh báo: ${trip.totalAlerts}", color = if (trip.totalAlerts > 0) WarningYellow else mutedColor, modifier = Modifier.weight(1f))
            Text("Điểm: ${formatNullableNumber(trip.safetyScore)}", color = SafeGreen, modifier = Modifier.weight(0.8f))
            Text(trip.syncStatus, color = mutedColor, modifier = Modifier.weight(1f))
            Button(onClick = { onDetail(trip.id) }, modifier = Modifier.height(48.dp)) { Text("Xem chi tiết") }
        }
    }
}
