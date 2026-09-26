package com.example.driverguardian.ui.screens.pretrip

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.driverguardian.ui.components.DashboardCard
import com.example.driverguardian.ui.mock.MockData
import com.example.driverguardian.ui.mock.SystemCheckUiModel
import com.example.driverguardian.ui.theme.SafeGreen
import com.example.driverguardian.ui.theme.WarningYellow
import kotlinx.coroutines.launch

@Composable
fun PreTripCheckScreen(
    snackbarHostState: SnackbarHostState,
    onStartMonitoring: () -> Unit
) {
    val scope = rememberCoroutineScope()
    Column(modifier = Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Text("Kiểm tra hệ thống trước chuyến đi", style = MaterialTheme.typography.headlineMedium)
        Row(modifier = Modifier.weight(1f), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            DashboardCard("Danh sách kiểm tra", modifier = Modifier.weight(1.35f)) {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    items(MockData.checks) { CheckRow(it) }
                }
            }
            DashboardCard("Lưu ý", modifier = Modifier.weight(1f), borderColor = WarningYellow) {
                Text(
                    "Hệ thống vẫn có thể giám sát và cảnh báo khi mất Internet. Dữ liệu sẽ được đồng bộ sau.",
                    style = MaterialTheme.typography.titleMedium
                )
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            OutlinedButton(
                onClick = { scope.launch { snackbarHostState.showSnackbar("Đã chạy lại kiểm tra mô phỏng") } },
                modifier = Modifier.weight(1f).height(56.dp)
            ) {
                Icon(Icons.Default.Refresh, contentDescription = null)
                Text("  Kiểm tra lại")
            }
            Button(onClick = onStartMonitoring, modifier = Modifier.weight(1f).height(56.dp)) {
                Text("Bắt đầu giám sát")
            }
        }
    }
}

@Composable
private fun CheckRow(item: SystemCheckUiModel) {
    Row(
        modifier = Modifier.fillMaxWidth().height(66.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Icon(
                if (item.warning) Icons.Default.Warning else Icons.Default.CheckCircle,
                contentDescription = null,
                tint = if (item.warning) WarningYellow else SafeGreen
            )
            Column {
                Text(item.name, style = MaterialTheme.typography.titleMedium)
                Text(item.description, color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodyMedium)
            }
        }
        Text(item.status, color = if (item.warning) WarningYellow else SafeGreen, style = MaterialTheme.typography.labelLarge)
    }
}
