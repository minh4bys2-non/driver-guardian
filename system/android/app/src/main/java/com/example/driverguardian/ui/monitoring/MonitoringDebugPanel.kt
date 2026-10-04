package com.example.driverguardian.ui.monitoring

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun MonitoringDebugPanel(presentation: MonitoringPresentation, modifier: Modifier = Modifier) {
    var expanded by remember { mutableStateOf(true) }
    Card(modifier = modifier) {
        Column(
            Modifier.padding(14.dp).verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Realtime multi-sensor", style = MaterialTheme.typography.titleMedium)
                TextButton(onClick = { expanded = !expanded }) { Text(if (expanded) "Thu gọn" else "Mở") }
            }
            Text("Camera: ${presentation.cameraStatus} • Face: ${presentation.faceStatus}")
            Text("Sensor: ${presentation.sensorStatus}")
            if (expanded) {
                HorizontalDivider()
                Text("PHYSICAL / FACE", style = MaterialTheme.typography.labelLarge)
                Text("EAR ${presentation.ear} • MAR ${presentation.mar}")
                Text("HEAD POSE: ${presentation.headPose}")
                Text("DEVICE / VEHICLE ORIENTATION", style = MaterialTheme.typography.labelLarge)
                Text("Raw: ${presentation.rawDeviceOrientation}")
                Text("Delta: ${presentation.deviceOrientation}")
                Text("Sensor age: ${presentation.sampleAge}")
                Text("Timebase: ${presentation.timebaseStatus}")
                Text("Performance: ${presentation.performance}")
                presentation.error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
                Text("Không có fusion / warning score / tự động tạo backend event", style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}
