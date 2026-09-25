package com.example.driverguardian.ui.screens.driving

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.automirrored.filled.VolumeOff
import androidx.compose.material.icons.Icons
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
import com.example.driverguardian.ui.theme.DangerRed
import kotlinx.coroutines.launch

@Composable
fun DangerAlertScreen(
    snackbarHostState: SnackbarHostState,
    onDismiss: () -> Unit
) {
    val scope = rememberCoroutineScope()
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Box(
            modifier = Modifier
                .size(180.dp)
                .background(DangerRed.copy(alpha = 0.2f), CircleShape),
            contentAlignment = Alignment.Center
        ) {
            Icon(Icons.Default.Warning, contentDescription = null, tint = DangerRed, modifier = Modifier.size(112.dp))
        }
        Text("PHÁT HIỆN DẤU HIỆU BUỒN NGỦ", color = DangerRed, style = MaterialTheme.typography.headlineLarge)
        Text(
            "Vui lòng tập trung và tìm vị trí an toàn để nghỉ ngơi",
            color = MaterialTheme.colorScheme.onSurface,
            style = MaterialTheme.typography.titleLarge
        )
        Text("Mức cảnh báo: Nguy hiểm • Trạng thái kéo dài: 5 giây", color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.titleMedium)
        Row(
            modifier = Modifier.fillMaxWidth(0.82f),
            horizontalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Button(onClick = onDismiss, modifier = Modifier.weight(1f).height(64.dp)) { Text("Tôi đã tỉnh táo") }
            OutlinedButton(
                onClick = { scope.launch { snackbarHostState.showSnackbar("Tìm trạm nghỉ đang được mô phỏng") } },
                modifier = Modifier.weight(1f).height(64.dp)
            ) { Text("Tìm trạm nghỉ") }
            OutlinedButton(
                onClick = { scope.launch { snackbarHostState.showSnackbar("Âm báo đã tắt trong mock state") } },
                modifier = Modifier.weight(1f).height(64.dp)
            ) {
                Icon(Icons.AutoMirrored.Filled.VolumeOff, contentDescription = null)
                Text("  Tắt âm báo")
            }
        }
    }
}
