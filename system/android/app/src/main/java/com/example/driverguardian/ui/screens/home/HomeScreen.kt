package com.example.driverguardian.ui.screens.home

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.CloudDone
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.driverguardian.ui.components.DashboardCard
import com.example.driverguardian.ui.components.MetricCard
import com.example.driverguardian.ui.components.PrimaryActionButton
import com.example.driverguardian.ui.components.StatusIndicator
import com.example.driverguardian.ui.mock.MockData
import com.example.driverguardian.ui.theme.AccentBlue
import com.example.driverguardian.ui.theme.DriverGuardianTheme
import com.example.driverguardian.ui.theme.SafeGreen
import com.example.driverguardian.ui.theme.WarningYellow

@Composable
fun HomeScreen(
    onStartTrip: () -> Unit,
    onHistory: () -> Unit,
    onAnalytics: () -> Unit,
    onPreTrip: () -> Unit,
    onSettings: () -> Unit
) {
    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val compact = maxWidth < 900.dp

        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(18.dp)
        ) {
            if (compact) {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    HeaderText(MockData.currentDriver.name)
                    PrimaryActionButton(
                        text = "Bắt đầu chuyến đi",
                        onClick = onStartTrip,
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            } else {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    HeaderText(MockData.currentDriver.name, modifier = Modifier.weight(1f))
                    Spacer(modifier = Modifier.width(16.dp))
                    PrimaryActionButton(
                        text = "Bắt đầu chuyến đi",
                        onClick = onStartTrip,
                        modifier = Modifier.widthIn(min = 220.dp, max = 340.dp)
                    )
                }
            }

            if (compact) {
                Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    DriverCard(Modifier.fillMaxWidth())
                    VehicleCard(onStartTrip, Modifier.fillMaxWidth())
                }
            } else {
                Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                    DriverCard(Modifier.weight(1f))
                    VehicleCard(onStartTrip, Modifier.weight(1f))
                }
            }

            if (compact) {
                Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    SystemStatusCard(Modifier.fillMaxWidth())
                    QuickStatsCard(Modifier.fillMaxWidth())
                }
            } else {
                Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                    SystemStatusCard(Modifier.weight(1.2f))
                    QuickStatsCard(Modifier.weight(1f))
                }
            }

            if (compact) {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Button(onClick = onHistory, modifier = Modifier.fillMaxWidth().height(54.dp)) { Text("Xem lịch sử") }
                    Button(onClick = onAnalytics, modifier = Modifier.fillMaxWidth().height(54.dp)) { Text("Xem phân tích") }
                    Button(onClick = onPreTrip, modifier = Modifier.fillMaxWidth().height(54.dp)) { Text("Kiểm tra hệ thống") }
                    Button(onClick = onSettings, modifier = Modifier.fillMaxWidth().height(54.dp)) { Text("Cài đặt cảnh báo") }
                }
            } else {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Button(onClick = onHistory, modifier = Modifier.weight(1f).height(54.dp)) { Text("Xem lịch sử") }
                    Button(onClick = onAnalytics, modifier = Modifier.weight(1f).height(54.dp)) { Text("Xem phân tích") }
                    Button(onClick = onPreTrip, modifier = Modifier.weight(1f).height(54.dp)) { Text("Kiểm tra hệ thống") }
                    Button(onClick = onSettings, modifier = Modifier.weight(1f).height(54.dp)) { Text("Cài đặt cảnh báo") }
                }
            }
        }
    }
}

@Composable
private fun HeaderText(driverName: String, modifier: Modifier = Modifier) {
    Column(modifier = modifier) {
        Text("Xin chào, $driverName", style = MaterialTheme.typography.headlineLarge)
        Text("Sẵn sàng cho chuyến đi an toàn", color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.titleMedium)
    }
}

@Composable
private fun DriverCard(modifier: Modifier = Modifier) {
    val driver = MockData.currentDriver

    DashboardCard("Hồ sơ của tôi", modifier = modifier) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(72.dp)
                    .clip(CircleShape)
                    .background(MaterialTheme.colorScheme.surfaceVariant),
                contentAlignment = Alignment.Center
            ) {
                Icon(Icons.Default.Person, contentDescription = null, tint = SafeGreen, modifier = Modifier.size(38.dp))
            }
            Column(modifier = Modifier.padding(start = 16.dp).weight(1f)) {
                Text(driver.name, style = MaterialTheme.typography.titleLarge)
                Text(driver.code, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Spacer(Modifier.height(8.dp))
                StatusIndicator(driver.status, color = SafeGreen)
            }
            OutlinedButton(onClick = {}, enabled = false) { Text("Cá nhân") }
        }
    }
}

@Composable
private fun VehicleCard(onStartTrip: () -> Unit, modifier: Modifier = Modifier) {
    val vehicle = MockData.vehicles.first()

    DashboardCard("Phương tiện", modifier = modifier) {
        Text(vehicle.name, style = MaterialTheme.typography.titleLarge)
        Text("Biển số: ${vehicle.plate}", color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text("Mã thiết bị: ${vehicle.deviceCode}", color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.height(10.dp))
        OutlinedButton(onClick = onStartTrip) { Text("Đổi xe") }
    }
}

@Composable
private fun SystemStatusCard(modifier: Modifier = Modifier) {
    DashboardCard("Trạng thái hệ thống", modifier = modifier) {
        SystemRow("Camera", "Sẵn sàng", SafeGreen)
        SystemRow("Module AI", "Sẵn sàng", SafeGreen)
        SystemRow("Kết nối máy chủ", "Đã kết nối", SafeGreen)
        SystemRow("Dữ liệu chờ đồng bộ", "2 sự kiện", WarningYellow)
    }
}

@Composable
private fun QuickStatsCard(modifier: Modifier = Modifier) {
    DashboardCard("Thống kê nhanh", modifier = modifier) {
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            MetricCard("Chuyến trong tháng", "12", modifier = Modifier.weight(1f), valueColor = SafeGreen)
            MetricCard("Cảnh báo", "5", modifier = Modifier.weight(1f), valueColor = WarningYellow)
        }
        Spacer(Modifier.height(12.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            MetricCard("Thời gian lái", "8g 35p", modifier = Modifier.weight(1f), valueColor = AccentBlue)
            MetricCard("Giờ nguy cơ", "01:00-03:00", modifier = Modifier.weight(1f), valueColor = WarningYellow)
        }
    }
}

@Composable
private fun SystemRow(name: String, value: String, color: androidx.compose.ui.graphics.Color) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 7.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            val icon = if (name == "Camera") Icons.Default.Videocam else if (name.contains("máy chủ")) Icons.Default.CloudDone else Icons.Default.CheckCircle
            Icon(icon, contentDescription = null, tint = color)
            Text("  $name", style = MaterialTheme.typography.bodyLarge)
        }
        Text(value, color = color, style = MaterialTheme.typography.labelLarge)
    }
}

@Preview(widthDp = 1280, heightDp = 720, showBackground = true)
@Composable
private fun HomeScreenPreview() {
    DriverGuardianTheme {
        Box(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background).padding(20.dp)) {
            HomeScreen({}, {}, {}, {}, {})
        }
    }
}
