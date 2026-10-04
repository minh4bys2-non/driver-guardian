package com.example.driverguardian.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Logout
import androidx.compose.material.icons.filled.AccountCircle
import androidx.compose.material.icons.filled.Analytics
import androidx.compose.material.icons.filled.DirectionsCar
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Notifications
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationRailItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.driverguardian.domain.model.UserProfile
import com.example.driverguardian.ui.navigation.Screen
import com.example.driverguardian.ui.theme.DangerRed
import com.example.driverguardian.ui.theme.SafeGreen

import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll

data class SidebarItem(
    val label: String,
    val route: String,
    val icon: ImageVector
)

private val sidebarItems = listOf(
    SidebarItem("Trang chủ", Screen.Home.route, Icons.Default.Home),
    SidebarItem("Chuyến đi", Screen.Selection.route, Icons.Default.DirectionsCar),
    SidebarItem("Lịch sử", Screen.TripHistory.route, Icons.Default.History),
    SidebarItem("Phân tích", Screen.Analytics.route, Icons.Default.Analytics),
    SidebarItem("Cảnh báo", Screen.AlertHistory.route, Icons.Default.Notifications),
    SidebarItem("Cài đặt", Screen.Settings.route, Icons.Default.Settings),
    SidebarItem("Hệ thống", Screen.SystemInfo.route, Icons.Default.Info),
    SidebarItem("ONNX Demo", Screen.OnnxDemo.route, Icons.Default.Analytics),
)

@Composable
fun AppSidebar(
    currentRoute: String?,
    onNavigate: (String) -> Unit,
    userProfile: UserProfile? = null,
    onLogout: (() -> Unit)? = null,
    modifier: Modifier = Modifier
) {
    val scrollState = rememberScrollState()
    Column(
        modifier = modifier
            .fillMaxHeight()
            .width(200.dp)
            .background(MaterialTheme.colorScheme.surface)
            .verticalScroll(scrollState)
            .padding(vertical = 12.dp, horizontal = 12.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 8.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Safety Center",
                color = SafeGreen,
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold
            )
        }

        if (userProfile != null) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 8.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Default.AccountCircle,
                    contentDescription = null,
                    tint = SafeGreen,
                    modifier = Modifier.size(24.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = userProfile.displayName ?: userProfile.email,
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.SemiBold,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    userProfile.driver?.let {
                        Text(
                            text = "Mã TX: ${it.driverCode}",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            fontSize = 11.sp
                        )
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        sidebarItems.forEach { item ->
            NavigationRailItem(
                selected = currentRoute == item.route,
                onClick = { onNavigate(item.route) },
                icon = { Icon(item.icon, contentDescription = item.label) },
                label = { Text(item.label) },
                alwaysShowLabel = true
            )
        }

        Spacer(modifier = Modifier.height(12.dp))

        if (onLogout != null) {
            OutlinedButton(
                onClick = onLogout,
                colors = ButtonDefaults.outlinedButtonColors(contentColor = DangerRed),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(40.dp)
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.Logout,
                    contentDescription = "Đăng xuất",
                    modifier = Modifier.size(16.dp)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text("Đăng xuất", fontSize = 13.sp)
            }
            Spacer(modifier = Modifier.height(4.dp))
        }

        Text(
            text = "Driver Guardian • v1.0",
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.padding(horizontal = 8.dp)
        )
    }
}
