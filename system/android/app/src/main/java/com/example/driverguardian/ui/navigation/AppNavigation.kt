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
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.example.driverguardian.BuildConfig
import com.example.driverguardian.data.auth.EncryptedTokenStore
import com.example.driverguardian.data.auth.GoogleAuthManager
import com.example.driverguardian.data.auth.SessionManager
import com.example.driverguardian.data.remote.ApiClient
import com.example.driverguardian.data.repository.NetworkAuthRepository
import com.example.driverguardian.data.repository.NetworkDriverGuardianRepository
import com.example.driverguardian.domain.model.UserProfile
import com.example.driverguardian.ui.auth.AuthUiState
import com.example.driverguardian.ui.auth.AuthViewModel
import com.example.driverguardian.ui.auth.AuthViewModelFactory
import com.example.driverguardian.ui.components.AppSidebar
import com.example.driverguardian.ui.history.TripHistoryViewModel
import com.example.driverguardian.ui.history.TripHistoryViewModelFactory
import com.example.driverguardian.ui.monitoring.MonitoringViewModel
import com.example.driverguardian.ui.monitoring.MonitoringViewModelFactory
import com.example.driverguardian.ui.screens.alerts.AlertHistoryScreen
import com.example.driverguardian.ui.screens.analytics.AnalyticsScreen
import com.example.driverguardian.ui.screens.auth.LoginScreen
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
import com.example.driverguardian.ui.session.DrivingSessionUiState
import com.example.driverguardian.ui.session.DrivingSessionViewModel
import com.example.driverguardian.ui.session.DrivingSessionViewModelFactory
import kotlinx.coroutines.launch

@Composable
fun DriverGuardianApp() {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    val navController = rememberNavController()
    val snackbarHostState = remember { SnackbarHostState() }

    val tokenStore = remember { EncryptedTokenStore(context) }
    val sessionManager = remember { SessionManager(tokenStore) }
    val api = remember { ApiClient.create(BuildConfig.API_BASE_URL, sessionManager) }
    val authRepository = remember { NetworkAuthRepository(api, sessionManager) }
    val authViewModel: AuthViewModel = viewModel(
        factory = remember { AuthViewModelFactory(authRepository) }
    )
    val authUiState by authViewModel.uiState.collectAsStateWithLifecycle()

    val repository = remember { NetworkDriverGuardianRepository(api) }
    val sessionViewModel: DrivingSessionViewModel = viewModel(
        factory = remember { DrivingSessionViewModelFactory(repository) }
    )
    val historyViewModel: TripHistoryViewModel = viewModel(
        factory = remember { TripHistoryViewModelFactory(repository) }
    )
    val monitoringViewModel: MonitoringViewModel = viewModel(
        factory = remember { MonitoringViewModelFactory(context) }
    )

    val sessionUiState by sessionViewModel.uiState.collectAsStateWithLifecycle()
    val historyUiState by historyViewModel.uiState.collectAsStateWithLifecycle()
    val backStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = backStackEntry?.destination?.route

    val isAuthFlow = currentRoute == Screen.Login.route
    val showSidebar = currentRoute != Screen.DangerAlert.route && !isAuthFlow

    val currentUserProfile: UserProfile? = when (val state = authUiState) {
        is AuthUiState.Authenticated -> state.user
        is AuthUiState.UnlinkedDriver -> state.user
        else -> null
    }

    LaunchedEffect(Unit) {
        authViewModel.restoreSession()
    }

    LaunchedEffect(authUiState) {
        when (val state = authUiState) {
            is AuthUiState.Authenticated -> {
                sessionViewModel.setAuthenticatedDriver(state.user.driver)
                if (currentRoute == Screen.Login.route) {
                    navController.navigate(Screen.Home.route) {
                        popUpTo(Screen.Login.route) { inclusive = true }
                    }
                }
            }
            is AuthUiState.Unauthenticated -> {
                sessionViewModel.setAuthenticatedDriver(null)
                if (currentRoute != null && currentRoute != Screen.Login.route) {
                    navController.navigate(Screen.Login.route) {
                        popUpTo(0) { inclusive = true }
                    }
                }
            }
            is AuthUiState.UnlinkedDriver -> {
                sessionViewModel.setAuthenticatedDriver(null)
                if (currentRoute != null && currentRoute != Screen.Login.route) {
                    navController.navigate(Screen.Login.route) {
                        popUpTo(0) { inclusive = true }
                    }
                }
            }
            else -> {}
        }
    }

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
                            onNavigate = { route -> navController.safeNavigate(route) },
                            userProfile = currentUserProfile,
                            onLogout = { authViewModel.logout() }
                        )
                    }
                    AppNavHost(
                        navController = navController,
                        snackbarHostState = snackbarHostState,
                        sessionUiState = sessionUiState,
                        sessionViewModel = sessionViewModel,
                        historyViewModel = historyViewModel,
                        historyUiState = historyUiState,
                        monitoringViewModel = monitoringViewModel,
                        authUiState = authUiState,
                        authViewModel = authViewModel,
                        currentUserProfile = currentUserProfile,
                        startDestination = if (sessionManager.isAuthenticated()) Screen.Home.route else Screen.Login.route,
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxSize()
                            .padding(
                                start = if (isAuthFlow) 0.dp else 16.dp,
                                top = if (isAuthFlow) 0.dp else 16.dp,
                                end = if (isAuthFlow) 0.dp else 16.dp,
                                bottom = if (isAuthFlow) 0.dp else if (showSidebar && !useSidebar) 96.dp else 16.dp
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
    sessionUiState: DrivingSessionUiState,
    sessionViewModel: DrivingSessionViewModel,
    historyViewModel: TripHistoryViewModel,
    historyUiState: com.example.driverguardian.ui.history.TripHistoryUiState,
    monitoringViewModel: MonitoringViewModel,
    authUiState: AuthUiState,
    authViewModel: AuthViewModel,
    currentUserProfile: UserProfile?,
    startDestination: String,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    val googleAuthManager = remember {
        GoogleAuthManager(context, BuildConfig.GOOGLE_SERVER_CLIENT_ID)
    }

    NavHost(
        navController = navController,
        startDestination = startDestination,
        modifier = modifier
    ) {
        composable(Screen.Login.route) {
            // Auto bottom-sheet flow on screen launch (try authorized accounts -> fallback all accounts)
            androidx.compose.runtime.LaunchedEffect(Unit) {
                if (authUiState is AuthUiState.Initial || authUiState is AuthUiState.Unauthenticated) {
                    val result = googleAuthManager.getGoogleIdTokenFromBottomSheet()
                    result.onSuccess { idToken ->
                        authViewModel.loginWithGoogle(idToken)
                    }.onFailure {
                        // Dismissal or no accounts for bottom sheet is handled gracefully;
                        // user clicks the persistent "Tiếp tục với Google" button.
                    }
                }
            }

            LoginScreen(
                state = authUiState,
                onLoginWithGoogle = {
                    coroutineScope.launch {
                        val result = googleAuthManager.getGoogleIdTokenFromButton()
                        result.onSuccess { idToken ->
                            authViewModel.loginWithGoogle(idToken)
                        }.onFailure { error ->
                            val msg = error.localizedMessage ?: "Đăng nhập Google thất bại"
                            authViewModel.clearError()
                            snackbarHostState.showSnackbar(msg)
                        }
                    }
                },
                onLogout = { authViewModel.logout() },
                onRetry = { authViewModel.clearError() }
            )
        }
        composable(Screen.Home.route) {
            val driverDisplayName = currentUserProfile?.driver?.fullName
                ?: currentUserProfile?.displayName
                ?: "Tài xế"
            HomeScreen(
                onStartTrip = { navController.safeNavigate(Screen.Selection.route) },
                onHistory = { navController.safeNavigate(Screen.TripHistory.route) },
                onAnalytics = { navController.safeNavigate(Screen.Analytics.route) },
                onPreTrip = { navController.safeNavigate(Screen.PreTripCheck.route) },
                onSettings = { navController.safeNavigate(Screen.Settings.route) },
                driverName = driverDisplayName,
                onLogout = { authViewModel.logout() }
            )
        }
        composable(Screen.Selection.route) {
            SelectionScreen(
                state = sessionUiState,
                onSelectDriver = sessionViewModel::selectDriver,
                onSelectVehicle = sessionViewModel::selectVehicle,
                onRetry = sessionViewModel::refresh,
                onCancel = { navController.safeNavigate(Screen.Home.route) },
                onContinue = { navController.safeNavigate(Screen.PreTripCheck.route) }
            )
        }
        composable(Screen.PreTripCheck.route) {
            PreTripCheckScreen(
                state = sessionUiState,
                snackbarHostState = snackbarHostState,
                onStartMonitoring = sessionViewModel::createSession,
                onSessionCreated = {
                    sessionViewModel.consumeSessionSubmission()
                    navController.safeNavigate(Screen.ActiveDriving.route)
                }
            )
        }
        composable(Screen.ActiveDriving.route) {
            ActiveDrivingScreen(
                sessionState = sessionUiState,
                monitoringViewModel = monitoringViewModel,
                snackbarHostState = snackbarHostState,
                onDangerDemo = { sessionViewModel.submitDangerEvent(null, null) },
                onDangerPersisted = {
                    sessionViewModel.consumeEventSubmission()
                    navController.safeNavigate(Screen.DangerAlert.route)
                },
                onFinishTrip = sessionViewModel::finishSession,
                onTripCompleted = {
                    sessionViewModel.consumeCompletion()
                    navController.safeNavigate(Screen.TripSummary.route)
                }
            )
        }
        composable(Screen.DangerAlert.route) {
            DangerAlertScreen(
                sessionState = sessionUiState,
                snackbarHostState = snackbarHostState,
                onAcknowledge = sessionViewModel::acknowledgeLastEvent,
                onDismiss = {
                    sessionViewModel.consumeAcknowledgement()
                    navController.popBackStack()
                }
            )
        }
        composable(Screen.TripSummary.route) {
            TripSummaryScreen(
                sessionState = sessionUiState,
                snackbarHostState = snackbarHostState,
                onHome = { navController.safeNavigate(Screen.Home.route) },
                onDetail = { id -> navController.safeNavigate(Screen.TripDetail.createRoute(id.toString())) }
            )
        }
        composable(Screen.TripHistory.route) {
            TripHistoryScreen(
                state = historyUiState,
                onLoad = historyViewModel::loadHistory,
                onDetail = { navController.safeNavigate(Screen.TripDetail.createRoute(it.toString())) }
            )
        }
        composable(Screen.TripDetail.route) {
            TripDetailScreen(
                id = it.arguments?.getString("id")?.toIntOrNull(),
                state = historyUiState,
                onLoad = historyViewModel::loadDetail
            )
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
