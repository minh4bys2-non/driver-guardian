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
    onSettings: () -> Unit,
    driverName: String = MockData.currentDriver.name,
    onLogout: (() -> Unit)? = null
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
                    HeaderText(driverName)
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
                    HeaderText(driverName, modifier = Modifier.weight(1f))
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
                    DriverCard(driverName, onLogout, Modifier.fillMaxWidth())
                    VehicleCard(onStartTrip, Modifier.fillMaxWidth())
                }
            } else {
                Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                    DriverCard(driverName, onLogout, Modifier.weight(1f))
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
                    Button(onClick = onSettings, modifier = Modifier.weight(1f).height(54.dp)) { Text("Cài đặt") }
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
private fun DriverCard(driverName: String, onLogout: (() -> Unit)?, modifier: Modifier = Modifier) {
    val driver = MockData.currentDriver

    DashboardCard("Hồ sơ tài xế", modifier = modifier) {
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
                Text(driverName, style = MaterialTheme.typography.titleLarge)
                Text(driver.code, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Spacer(Modifier.height(8.dp))
                StatusIndicator(driver.status, color = SafeGreen)
            }
            if (onLogout != null) {
                OutlinedButton(onClick = onLogout) { Text("Đăng xuất") }
            } else {
                OutlinedButton(onClick = {}, enabled = false) { Text("Cá nhân") }
            }
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
    DashboardCard("Chỉ số an toàn", modifier = modifier) {
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            MetricCard(label = "Điểm an toàn", value = "98", modifier = Modifier.weight(1f), valueColor = SafeGreen)
            MetricCard(label = "Cảnh báo tuần", value = "1", modifier = Modifier.weight(1f), valueColor = AccentBlue)
            MetricCard(label = "Giờ lái xe", value = "24h", modifier = Modifier.weight(1f), valueColor = MaterialTheme.colorScheme.primary)
        }
    }
}

@Composable
private fun SystemRow(label: String, status: String, color: androidx.compose.ui.graphics.Color) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(label, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Default.CheckCircle, contentDescription = null, tint = color, modifier = Modifier.size(16.dp))
            Spacer(Modifier.width(6.dp))
            Text(status, color = color)
        }
    }
}

@Preview(showBackground = true, widthDp = 1000)
@Composable
private fun HomeScreenPreview() {
    DriverGuardianTheme {
        HomeScreen({}, {}, {}, {}, {})
    }
}
