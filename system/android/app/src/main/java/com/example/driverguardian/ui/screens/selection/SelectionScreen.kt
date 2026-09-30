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
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
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
import com.example.driverguardian.ui.mock.VehicleUiModel
import com.example.driverguardian.ui.theme.SafeGreen

@Composable
fun SelectionScreen(
    onCancel: () -> Unit,
    onContinue: () -> Unit
) {
    val currentDriver = MockData.currentDriver
    var selectedVehicleId by remember { mutableStateOf(MockData.vehicles.first().id) }

    Column(modifier = Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Row(modifier = Modifier.weight(1f), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            DashboardCard("Tài xế", modifier = Modifier.weight(0.75f)) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(116.dp),
                    horizontalArrangement = Arrangement.spacedBy(14.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(Icons.Default.Person, contentDescription = null, tint = SafeGreen)
                    Column(modifier = Modifier.weight(1f)) {
                        Text(currentDriver.name, style = MaterialTheme.typography.titleLarge)
                        Text(currentDriver.code, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Text(currentDriver.status, color = SafeGreen, style = MaterialTheme.typography.labelLarge)
                    }
                }
            }

            DashboardCard("Chọn phương tiện", modifier = Modifier.weight(1.25f)) {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    items(MockData.vehicles) {
                        VehicleCard(it, selected = it.id == selectedVehicleId, onClick = { selectedVehicleId = it.id })
                    }
                }
            }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            OutlinedButton(onClick = onCancel, modifier = Modifier.weight(1f).height(56.dp)) {
                Text("Hủy")
            }
            Button(onClick = onContinue, modifier = Modifier.weight(1f).height(56.dp)) {
                Text("Tiếp tục")
            }
        }
    }
}

@Composable
private fun VehicleCard(vehicle: VehicleUiModel, selected: Boolean, onClick: () -> Unit) {
    SelectableCard(selected = selected, onClick = onClick) {
        Icon(Icons.Default.DirectionsCar, contentDescription = null, tint = SafeGreen)
        Column(modifier = Modifier.weight(1f)) {
            Text(vehicle.name, style = MaterialTheme.typography.titleMedium)
            Text("${vehicle.plate} • ${vehicle.type}", color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text("Thiết bị: ${vehicle.deviceStatus}", color = MaterialTheme.colorScheme.onSurfaceVariant)
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
            modifier = Modifier
                .fillMaxWidth()
                .height(88.dp)
                .padding(horizontal = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(14.dp),
            verticalAlignment = Alignment.CenterVertically,
            content = content
        )
    }
}
