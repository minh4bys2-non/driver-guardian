package com.example.driverguardian.ui.navigation

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Analytics
import androidx.compose.material.icons.filled.DirectionsCar
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Notifications
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.example.driverguardian.ui.components.AppSidebar
import com.example.driverguardian.ui.screens.alerts.AlertHistoryScreen
import com.example.driverguardian.ui.screens.analytics.AnalyticsScreen
import com.example.driverguardian.ui.screens.driving.ActiveDrivingScreen
import com.example.driverguardian.ui.screens.driving.DangerAlertScreen
import com.example.driverguardian.ui.screens.history.TripDetailScreen
import com.example.driverguardian.ui.screens.history.TripHistoryScreen
import com.example.driverguardian.ui.screens.home.HomeScreen
import com.example.driverguardian.ui.screens.onnxdemo.OnnxDemoScreen
import com.example.driverguardian.ui.screens.pretrip.PreTripCheckScreen
import com.example.driverguardian.ui.screens.selection.SelectionScreen
import com.example.driverguardian.ui.screens.settings.SettingsScreen
import com.example.driverguardian.ui.screens.summary.TripSummaryScreen
import com.example.driverguardian.ui.screens.system.SystemInfoScreen

@Composable
fun DriverGuardianApp() {
    val navController = rememberNavController()
    val snackbarHostState = remember { SnackbarHostState() }
    val backStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = backStackEntry?.destination?.route
    val showSidebar = currentRoute != Screen.DangerAlert.route

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = androidx.compose.material3.MaterialTheme.colorScheme.background,
        contentColor = androidx.compose.material3.MaterialTheme.colorScheme.onBackground
    ) {
        BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
            val useSidebar = showSidebar && maxWidth >= 720.dp

            Box(modifier = Modifier.fillMaxSize()) {
                Row(modifier = Modifier.fillMaxSize()) {
                    if (useSidebar) {
                        AppSidebar(
                            currentRoute = currentRoute,
                            onNavigate = { route -> navController.safeNavigate(route) }
                        )
                    }
                    AppNavHost(
                        navController = navController,
                        snackbarHostState = snackbarHostState,
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxSize()
                            .padding(
                                start = 16.dp,
                                top = 16.dp,
                                end = 16.dp,
                                bottom = if (showSidebar && !useSidebar) 96.dp else 16.dp
                            )
                    )
                }
                if (showSidebar && !useSidebar) {
                    BottomAppNavigation(
                        currentRoute = currentRoute,
                        onNavigate = { route -> navController.safeNavigate(route) },
                        modifier = Modifier.fillMaxSize()
                    )
                }
                SnackbarHost(hostState = snackbarHostState)
            }
        }
    }
}

private data class BottomNavItem(
    val label: String,
    val route: String,
    val icon: ImageVector
)

private val bottomNavItems = listOf(
    BottomNavItem("Trang chủ", Screen.Home.route, Icons.Default.Home),
    BottomNavItem("Chuyến đi", Screen.Selection.route, Icons.Default.DirectionsCar),
    BottomNavItem("Lịch sử", Screen.TripHistory.route, Icons.Default.History),
    BottomNavItem("Phân tích", Screen.Analytics.route, Icons.Default.Analytics),
    BottomNavItem("Cảnh báo", Screen.AlertHistory.route, Icons.Default.Notifications),
    BottomNavItem("Cài đặt", Screen.Settings.route, Icons.Default.Settings)
)

@Composable
private fun BottomAppNavigation(
    currentRoute: String?,
    onNavigate: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    Box(modifier = modifier, contentAlignment = Alignment.BottomCenter) {
        NavigationBar(modifier = Modifier.fillMaxWidth()) {
            bottomNavItems.forEach { item ->
                NavigationBarItem(
                    selected = currentRoute == item.route,
                    onClick = { onNavigate(item.route) },
                    icon = { Icon(item.icon, contentDescription = item.label) },
                    label = { Text(item.label, maxLines = 1) }
                )
            }
        }
    }
}

@Composable
private fun AppNavHost(
    navController: NavHostController,
    snackbarHostState: SnackbarHostState,
    modifier: Modifier = Modifier
) {
    NavHost(
        navController = navController,
        startDestination = Screen.Home.route,
        modifier = modifier
    ) {
        composable(Screen.Home.route) {
            HomeScreen(
                onStartTrip = { navController.safeNavigate(Screen.Selection.route) },
                onHistory = { navController.safeNavigate(Screen.TripHistory.route) },
                onAnalytics = { navController.safeNavigate(Screen.Analytics.route) },
                onPreTrip = { navController.safeNavigate(Screen.PreTripCheck.route) },
                onSettings = { navController.safeNavigate(Screen.Settings.route) }
            )
        }
        composable(Screen.Selection.route) {
            SelectionScreen(
                onCancel = { navController.safeNavigate(Screen.Home.route) },
                onContinue = { navController.safeNavigate(Screen.PreTripCheck.route) }
            )
        }
        composable(Screen.PreTripCheck.route) {
            PreTripCheckScreen(
                snackbarHostState = snackbarHostState,
                onStartMonitoring = { navController.safeNavigate(Screen.ActiveDriving.route) }
            )
        }
        composable(Screen.ActiveDriving.route) {
            ActiveDrivingScreen(
                snackbarHostState = snackbarHostState,
                onDangerDemo = { navController.safeNavigate(Screen.DangerAlert.route) },
                onFinishTrip = { navController.safeNavigate(Screen.TripSummary.route) }
            )
        }
        composable(Screen.DangerAlert.route) {
            DangerAlertScreen(
                snackbarHostState = snackbarHostState,
                onDismiss = { navController.safeNavigate(Screen.ActiveDriving.route) }
            )
        }
        composable(Screen.TripSummary.route) {
            TripSummaryScreen(
                snackbarHostState = snackbarHostState,
                onHome = { navController.safeNavigate(Screen.Home.route) },
                onDetail = { navController.safeNavigate(Screen.TripDetail.createRoute("1")) }
            )
        }
        composable(Screen.TripHistory.route) {
            TripHistoryScreen(onDetail = { navController.safeNavigate(Screen.TripDetail.createRoute(it)) })
        }
        composable(Screen.TripDetail.route) {
            TripDetailScreen(id = it.arguments?.getString("id") ?: "1")
        }
        composable(Screen.Analytics.route) {
            AnalyticsScreen()
        }
        composable(Screen.AlertHistory.route) {
            AlertHistoryScreen()
        }
        composable(Screen.Settings.route) {
            SettingsScreen()
        }
        composable(Screen.SystemInfo.route) {
            SystemInfoScreen(
                snackbarHostState = snackbarHostState,
                onOpenOnnxDemo = { navController.safeNavigate(Screen.OnnxDemo.route) }
            )
        }
        composable(Screen.OnnxDemo.route) {
            OnnxDemoScreen()
        }
    }
}

private fun NavHostController.safeNavigate(route: String) {
    navigate(route) {
        launchSingleTop = true
    }
}
