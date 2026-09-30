package com.example.driverguardian.ui.screens.history

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
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
import com.example.driverguardian.ui.mock.MockData
import com.example.driverguardian.ui.mock.TripHistoryUiModel
import com.example.driverguardian.ui.theme.SafeGreen
import com.example.driverguardian.ui.theme.WarningYellow

@Composable
fun TripHistoryScreen(onDetail: (String) -> Unit) {
    var query by remember { mutableStateOf("") }
    val trips = MockData.trips.filter {
        it.vehiclePlate.contains(query, true) || it.date.contains(query, true)
    }

    Column(modifier = Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(14.dp)) {
        Text("Lịch sử chuyến đi của tôi", style = MaterialTheme.typography.headlineMedium)
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
            listOf("7 ngày", "Xe của tôi", "Đã đồng bộ").forEach {
                FilterChip(selected = false, onClick = {}, label = { Text(it) })
            }
            OutlinedTextField(
                value = query,
                onValueChange = { query = it },
                leadingIcon = { Icon(Icons.Default.Search, contentDescription = null) },
                placeholder = { Text("Tìm theo ngày hoặc biển số") },
                singleLine = true,
                modifier = Modifier.weight(1f)
            )
        }
        LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            items(trips) {
                TripRow(it, onDetail)
            }
        }
    }
}

@Composable
private fun TripRow(trip: TripHistoryUiModel, onDetail: (String) -> Unit) {
    val mutedColor = MaterialTheme.colorScheme.onSurfaceVariant
    DashboardCard(title = trip.date) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1.2f)) {
                Text("Chuyến cá nhân", style = MaterialTheme.typography.titleMedium)
                Text(trip.vehiclePlate, color = mutedColor)
            }
            Text("Thời lượng: ${trip.duration}", modifier = Modifier.weight(1f))
            Text("Cảnh báo: ${trip.alerts}", color = if (trip.alerts > 4) WarningYellow else mutedColor, modifier = Modifier.weight(1f))
            Text("Điểm: ${trip.score}", color = SafeGreen, modifier = Modifier.weight(0.8f))
            Text(trip.syncStatus, color = mutedColor, modifier = Modifier.weight(1f))
            Button(onClick = { onDetail(trip.id) }, modifier = Modifier.height(48.dp)) { Text("Xem chi tiết") }
        }
    }
}
