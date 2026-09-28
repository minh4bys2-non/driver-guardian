package com.example.driverguardian.ui.screens.selection

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.DirectionsCar
import androidx.compose.material.icons.filled.Person
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.driverguardian.domain.model.Driver
import com.example.driverguardian.domain.model.Vehicle
import com.example.driverguardian.ui.components.DashboardCard
import com.example.driverguardian.ui.session.DrivingSessionUiState
import com.example.driverguardian.ui.session.LoadState
import com.example.driverguardian.ui.theme.SafeGreen

@Composable
fun SelectionScreen(
    state: DrivingSessionUiState,
    onSelectDriver: (Int) -> Unit,
    onSelectVehicle: (Int) -> Unit,
    onRetry: () -> Unit,
    onCancel: () -> Unit,
    onContinue: () -> Unit
) {
    Column(modifier = Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        when (val loadState = state.loadState) {
            LoadState.Loading -> Row(
                modifier = Modifier.weight(1f).fillMaxWidth(),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                CircularProgressIndicator()
                Text("  Đang tải dữ liệu từ FastAPI…")
            }
            is LoadState.Error -> StatusPanel(loadState.message, onRetry, Modifier.weight(1f))
            is LoadState.Empty -> StatusPanel(loadState.message, onRetry, Modifier.weight(1f))
            LoadState.Success -> SelectionLists(state, onSelectDriver, onSelectVehicle, Modifier.weight(1f))
        }

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            OutlinedButton(onClick = onCancel, modifier = Modifier.weight(1f).height(56.dp)) { Text("Hủy") }
            Button(onClick = onContinue, enabled = state.canContinue, modifier = Modifier.weight(1f).height(56.dp)) {
                Text("Tiếp tục")
            }
        }
    }
}

@Composable
private fun SelectionLists(
    state: DrivingSessionUiState,
    onSelectDriver: (Int) -> Unit,
    onSelectVehicle: (Int) -> Unit,
    modifier: Modifier = Modifier
) {
    Row(modifier = modifier, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
        DashboardCard("Chọn tài xế", modifier = Modifier.weight(1f)) {
            LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                items(state.drivers, key = { it.id }) { driver ->
                    DriverCard(driver, driver.id == state.selectedDriverId) { onSelectDriver(driver.id) }
                }
            }
        }
        DashboardCard("Chọn phương tiện", modifier = Modifier.weight(1f)) {
            LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                items(state.vehicles, key = { it.id }) { vehicle ->
                    VehicleCard(vehicle, vehicle.id == state.selectedVehicleId) { onSelectVehicle(vehicle.id) }
                }
            }
        }
    }
}

@Composable
private fun StatusPanel(message: String, onRetry: () -> Unit, modifier: Modifier = Modifier) {
    DashboardCard("Không thể chuẩn bị chuyến đi", modifier = modifier) {
        Text(message, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Button(onClick = onRetry) { Text("Thử lại") }
    }
}

@Composable
private fun DriverCard(driver: Driver, selected: Boolean, onClick: () -> Unit) {
    SelectableCard(selected, onClick) {
        Icon(Icons.Default.Person, contentDescription = null, tint = SafeGreen)
        Column(modifier = Modifier.weight(1f)) {
            Text(driver.fullName, style = MaterialTheme.typography.titleMedium)
            Text("${driver.code} • ${driver.status}", color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun VehicleCard(vehicle: Vehicle, selected: Boolean, onClick: () -> Unit) {
    SelectableCard(selected, onClick) {
        Icon(Icons.Default.DirectionsCar, contentDescription = null, tint = SafeGreen)
        Column(modifier = Modifier.weight(1f)) {
            Text(vehicle.name ?: vehicle.plateNumber, style = MaterialTheme.typography.titleMedium)
            Text("${vehicle.plateNumber} • ${vehicle.type ?: "Chưa phân loại"}", color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text("Thiết bị: ${vehicle.deviceCode ?: "Chưa gán"}", color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun SelectableCard(selected: Boolean, onClick: () -> Unit, content: @Composable RowScope.() -> Unit) {
    Card(
        onClick = onClick,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        border = BorderStroke(if (selected) 2.dp else 1.dp, if (selected) SafeGreen else MaterialTheme.colorScheme.outlineVariant)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().height(88.dp).padding(horizontal = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(14.dp),
            verticalAlignment = Alignment.CenterVertically,
            content = content
        )
    }
}
