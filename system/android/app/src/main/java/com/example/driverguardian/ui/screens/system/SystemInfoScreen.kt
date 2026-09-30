package com.example.driverguardian.ui.screens.system

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.driverguardian.ui.components.DashboardCard
import com.example.driverguardian.ui.components.MetricCard
import com.example.driverguardian.ui.theme.SafeGreen
import com.example.driverguardian.ui.theme.WarningYellow
import kotlinx.coroutines.launch

@Composable
fun SystemInfoScreen(
    snackbarHostState: SnackbarHostState,
    onOpenOnnxDemo: () -> Unit,
) {
    val scope = rememberCoroutineScope()
    Column(modifier = Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Text("Thông tin hệ thống", style = MaterialTheme.typography.headlineMedium)
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            MetricCard("Phiên bản ứng dụng", "1.0.0", Modifier.weight(1f), SafeGreen)
            MetricCard("Module AI", "v1.0.0", Modifier.weight(1f))
            MetricCard("Sự kiện chờ đồng bộ", "2", Modifier.weight(1f), WarningYellow)
            MetricCard("Inference TB", "42 ms", Modifier.weight(1f))
        }
        DashboardCard("Trạng thái") {
            listOf(
                "Camera: Sẵn sàng",
                "Model: Sẵn sàng",
                "Kết nối API: Đã kết nối",
                "Oracle: Đã kết nối",
                "Dung lượng Room Database giả: 18 MB"
            ).forEach {
                Text(it, style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(
                onClick = { scope.launch { snackbarHostState.showSnackbar("Đã kiểm tra lại hệ thống mock") } },
                modifier = Modifier.weight(1f).height(56.dp)
            ) { Text("Kiểm tra lại hệ thống") }
            OutlinedButton(
                onClick = { scope.launch { snackbarHostState.showSnackbar("Nhật ký mock: không có lỗi mới") } },
                modifier = Modifier.weight(1f).height(56.dp)
            ) { Text("Xem nhật ký") }
            OutlinedButton(
                onClick = { scope.launch { snackbarHostState.showSnackbar("Driver Guardian Demo 1.0.0") } },
                modifier = Modifier.weight(1f).height(56.dp)
            ) { Text("Thông tin ứng dụng") }
        }
        OutlinedButton(
            onClick = onOpenOnnxDemo,
            modifier = Modifier.fillMaxWidth().height(56.dp)
        ) { Text("Mở ONNX Runtime Demo") }
    }
}
