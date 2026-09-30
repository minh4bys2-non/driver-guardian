package com.example.driverguardian.ui.mock

object MockData {
    val currentDriver = DriverUiModel("d1", "Nguyễn Văn An", "DRV-1024", "Đang hoạt động")

    val vehicles = listOf(
        VehicleUiModel("v1", "Hyundai Universe", "51B-268.45", "Xe khách", "CAM-AI-9041", "Sẵn sàng"),
        VehicleUiModel("v2", "Thaco Town", "60F-112.90", "Xe tải", "CAM-AI-7750", "Sẵn sàng"),
        VehicleUiModel("v3", "Ford Transit", "29B-430.11", "Xe dịch vụ", "CAM-AI-6623", "Cần đồng bộ"),
        VehicleUiModel("v4", "Isuzu QKR", "50H-998.20", "Xe tải nhẹ", "CAM-AI-5512", "Sẵn sàng")
    )

    val checks = listOf(
        SystemCheckUiModel("Camera tài xế", "Sẵn sàng", "Khung hình mô phỏng ổn định"),
        SystemCheckUiModel("Module dự đoán", "Sẵn sàng", "Phiên bản UI mock v1.0.0"),
        SystemCheckUiModel("Loa cảnh báo", "Sẵn sàng", "Âm lượng mô phỏng 80%"),
        SystemCheckUiModel("Dung lượng lưu trữ", "Còn 68%", "Đủ cho phiên lái hiện tại"),
        SystemCheckUiModel("Kết nối mạng", "Không ổn định", "Vẫn lưu cục bộ để đồng bộ sau", warning = true),
        SystemCheckUiModel("Kết nối máy chủ", "Đã kết nối", "Mock API online"),
        SystemCheckUiModel("GPS", "Sẵn sàng", "Chỉ hiển thị trạng thái giả lập")
    )

    val drivingNormal = DrivingUiState(
        driverName = "Nguyễn Văn An",
        vehiclePlate = "51B-268.45",
        statusLabel = "TỈNH TÁO",
        statusMessage = "Không phát hiện dấu hiệu nguy hiểm",
        level = AlertLevel.Safe,
        prediction = PredictionUiModel(18, 94, "0.28", "0.31", "Bình thường"),
        drivingTime = "01:25:32",
        alertCount = 2,
        lastAlertAgo = "34 phút",
        syncStatus = "Đã đồng bộ"
    )

    val drivingWarning = drivingNormal.copy(
        statusLabel = "MỆT MỎI",
        statusMessage = "Dấu hiệu mệt nhẹ, hãy nghỉ khi có thể",
        level = AlertLevel.Warning,
        prediction = PredictionUiModel(58, 88, "0.23", "0.39", "Nghiêng nhẹ")
    )

    val drivingDanger = drivingNormal.copy(
        statusLabel = "NGUY HIỂM",
        statusMessage = "Phát hiện dấu hiệu buồn ngủ kéo dài",
        level = AlertLevel.Danger,
        prediction = PredictionUiModel(86, 91, "0.16", "0.48", "Cúi đầu")
    )

    val tripSummary = TripSummaryUiModel(
        startTime = "22:40",
        endTime = "02:15",
        duration = "03:35:00",
        totalAlerts = 5,
        levelOneAlerts = 3,
        levelTwoAlerts = 2,
        riskyWindow = "01:00-03:00",
        safetyScore = 78,
        rank = "Khá",
        timeline = listOf(
            "22:40 - Bắt đầu chuyến đi",
            "23:45 - Cảnh báo mệt mỏi mức 1",
            "01:12 - Cảnh báo buồn ngủ mức 2",
            "01:20 - Tài xế xác nhận đã tỉnh táo",
            "02:15 - Kết thúc chuyến đi"
        ),
        recommendations = listOf(
            "Bạn đã lái xe liên tục hơn 2 giờ.",
            "Nên nghỉ từ 15 đến 20 phút trước chuyến tiếp theo.",
            "Khung giờ 01:00-03:00 có nguy cơ cao."
        )
    )

    val trips = listOf(
        TripHistoryUiModel("1", "27/07/2026", currentDriver.name, "51B-268.45", "03:35", 5, 78, "Đã đồng bộ"),
        TripHistoryUiModel("2", "26/07/2026", currentDriver.name, "51B-268.45", "02:15", 1, 91, "Đã đồng bộ"),
        TripHistoryUiModel("3", "25/07/2026", currentDriver.name, "60F-112.90", "04:10", 7, 70, "Chờ đồng bộ"),
        TripHistoryUiModel("4", "24/07/2026", currentDriver.name, "51B-268.45", "01:55", 0, 96, "Đã đồng bộ"),
        TripHistoryUiModel("5", "23/07/2026", currentDriver.name, "29B-430.11", "02:42", 3, 84, "Đã đồng bộ"),
        TripHistoryUiModel("6", "22/07/2026", currentDriver.name, "51B-268.45", "03:08", 2, 88, "Đã đồng bộ"),
        TripHistoryUiModel("7", "21/07/2026", currentDriver.name, "60F-112.90", "05:20", 8, 66, "Chờ đồng bộ"),
        TripHistoryUiModel("8", "20/07/2026", currentDriver.name, "51B-268.45", "02:30", 1, 93, "Đã đồng bộ")
    )

    val alerts = listOf(
        AlertEventUiModel("a1", "27/07 01:12", "Buồn ngủ", AlertLevel.Danger, 91, "0.16", "0.48", true, "Đã đồng bộ"),
        AlertEventUiModel("a2", "27/07 00:58", "Mệt mỏi", AlertLevel.Warning, 84, "0.22", "0.41", false, "Chờ đồng bộ"),
        AlertEventUiModel("a3", "25/07 02:14", "Buồn ngủ", AlertLevel.Danger, 93, "0.15", "0.50", true, "Đã đồng bộ"),
        AlertEventUiModel("a4", "25/07 23:45", "Mệt mỏi", AlertLevel.Warning, 78, "0.25", "0.36", true, "Đã đồng bộ"),
        AlertEventUiModel("a5", "23/07 01:30", "Buồn ngủ", AlertLevel.Danger, 89, "0.18", "0.46", false, "Chờ đồng bộ"),
        AlertEventUiModel("a6", "22/07 03:05", "Mệt mỏi", AlertLevel.Warning, 81, "0.24", "0.39", true, "Đã đồng bộ")
    )

    val hourlyStats = (0..23).map { hour ->
        val value = when (hour) {
            1 -> 9
            2 -> 11
            3 -> 8
            22, 23 -> 5
            else -> (hour % 4) + 1
        }
        HourlyStatisticUiModel(hour, value)
    }

    val trend7Days = listOf(4, 2, 5, 3, 7, 6, 4)
}
