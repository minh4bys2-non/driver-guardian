package com.example.driverguardian.ui.navigation

sealed class Screen(val route: String) {
    data object Home : Screen("home")
    data object Selection : Screen("selection")
    data object PreTripCheck : Screen("pre_trip_check")
    data object ActiveDriving : Screen("active_driving")
    data object DangerAlert : Screen("danger_alert")
    data object TripSummary : Screen("trip_summary")
    data object TripHistory : Screen("trip_history")
    data object TripDetail : Screen("trip_detail/{id}") {
        fun createRoute(id: String) = "trip_detail/$id"
    }
    data object Analytics : Screen("analytics")
    data object AlertHistory : Screen("alert_history")
    data object Settings : Screen("settings")
    data object SystemInfo : Screen("system_info")
    data object OnnxDemo : Screen("onnx_demo")
}
