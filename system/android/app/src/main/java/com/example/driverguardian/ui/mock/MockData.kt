package com.example.driverguardian.ui.mock

object MockData {
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

    val alerts = listOf(
        AlertEventUiModel("a1", "27/07 01:12", "Buồn ngủ", AlertLevel.Danger, 91, "0.16", "0.48", true, "Đã đồng bộ"),
        AlertEventUiModel("a2", "27/07 00:58", "Mệt mỏi", AlertLevel.Warning, 84, "0.22", "0.41", false, "Chờ đồng bộ"),
        AlertEventUiModel("a3", "25/07 02:14", "Buồn ngủ", AlertLevel.Danger, 93, "0.15", "0.50", true, "Đã đồng bộ"),
        AlertEventUiModel("a4", "25/07 23:45", "Mệt mỏi", AlertLevel.Warning, 78, "0.25", "0.36", true, "Đã đồng bộ"),
        AlertEventUiModel("a5", "23/07 01:30", "Buồn ngủ", AlertLevel.Danger, 89, "0.18", "0.46", false, "Chờ đồng bộ"),
        AlertEventUiModel("a6", "22/07 03:05", "Mệt mỏi", AlertLevel.Warning, 81, "0.24", "0.39", true, "Đã đồng bộ")
    )

}
