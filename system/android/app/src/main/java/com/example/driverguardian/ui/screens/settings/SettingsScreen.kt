package com.example.driverguardian.ui.screens.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.driverguardian.ui.components.DashboardCard

@Composable
fun SettingsScreen() {
    var sound by remember { mutableStateOf(true) }
    var volume by remember { mutableFloatStateOf(0.8f) }
    var vibration by remember { mutableStateOf(true) }
    var floatingAlert by remember { mutableStateOf(true) }
    var showEar by remember { mutableStateOf(true) }
    var showConfidence by remember { mutableStateOf(true) }
    var saver by remember { mutableStateOf(false) }
    var wifiOnly by remember { mutableStateOf(false) }
    var autoSync by remember { mutableStateOf(true) }
    var cleanSynced by remember { mutableStateOf(true) }
    var noPhoto by remember { mutableStateOf(true) }
    var noVideo by remember { mutableStateOf(true) }

    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text("Cài đặt", style = MaterialTheme.typography.headlineMedium)
        Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            DashboardCard("Cảnh báo", modifier = Modifier.weight(1f)) {
                SettingSwitch("Bật âm thanh", sound) { sound = it }
                Text("Âm lượng cảnh báo: ${(volume * 100).toInt()}%", color = MaterialTheme.colorScheme.onSurfaceVariant)
                Slider(value = volume, onValueChange = { volume = it })
                SettingSwitch("Bật rung", vibration) { vibration = it }
                SettingSwitch("Thông báo nổi", floatingAlert) { floatingAlert = it }
                Text("Kiểu âm báo: Chuông ngắn", color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            DashboardCard("Giám sát", modifier = Modifier.weight(1f)) {
                SettingSwitch("Hiển thị EAR/MAR", showEar) { showEar = it }
                SettingSwitch("Hiển thị confidence", showConfidence) { showConfidence = it }
                SettingSwitch("Chế độ tiết kiệm tài nguyên", saver) { saver = it }
                Text("Tần suất cập nhật UI: 1 giây", color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            DashboardCard("Đồng bộ", modifier = Modifier.weight(1f)) {
                SettingSwitch("Chỉ đồng bộ khi có Wi-Fi", wifiOnly) { wifiOnly = it }
                SettingSwitch("Tự động đồng bộ", autoSync) { autoSync = it }
                SettingSwitch("Xóa dữ liệu đã đồng bộ sau 7 ngày", cleanSynced) { cleanSynced = it }
            }
            DashboardCard("Quyền riêng tư", modifier = Modifier.weight(1f)) {
                SettingSwitch("Không lưu ảnh camera", noPhoto) { noPhoto = it }
                SettingSwitch("Không lưu video", noVideo) { noVideo = it }
                Spacer(Modifier.height(8.dp))
                Text(
                    "Demo chỉ hiển thị dữ liệu mẫu. Không tích hợp camera, không lưu ảnh/video và không gửi dữ liệu ra ngoài.",
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
private fun SettingSwitch(label: String, checked: Boolean, onCheckedChange: (Boolean) -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth().height(54.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(label, style = MaterialTheme.typography.bodyLarge)
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}
