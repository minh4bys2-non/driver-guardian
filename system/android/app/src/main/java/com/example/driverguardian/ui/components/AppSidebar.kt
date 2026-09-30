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
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Analytics
import androidx.compose.material.icons.filled.DirectionsCar
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Notifications
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationRailItem
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import com.example.driverguardian.ui.navigation.Screen
import com.example.driverguardian.ui.theme.SafeGreen

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
    SidebarItem("Hệ thống", Screen.SystemInfo.route, Icons.Default.Info)
)

@Composable
fun AppSidebar(
    currentRoute: String?,
    onNavigate: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxHeight()
            .width(188.dp)
            .background(MaterialTheme.colorScheme.surface)
            .padding(vertical = 22.dp, horizontal = 12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 8.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Safety Center",
                color = SafeGreen,
                style = MaterialTheme.typography.titleLarge
            )
        }
        Spacer(modifier = Modifier.height(12.dp))
        sidebarItems.forEach { item ->
            NavigationRailItem(
                selected = currentRoute == item.route,
                onClick = { onNavigate(item.route) },
                icon = { Icon(item.icon, contentDescription = item.label) },
                label = { Text(item.label) },
                alwaysShowLabel = true
            )
        }
        Spacer(modifier = Modifier.weight(1f))
        Text(
            text = "UI demo • mock data",
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.padding(horizontal = 8.dp)
        )
    }
}
