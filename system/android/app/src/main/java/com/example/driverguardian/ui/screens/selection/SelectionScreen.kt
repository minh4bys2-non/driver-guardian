package com.example.driverguardian.ui.screens.selection

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.DirectionsCar
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Person
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.driverguardian.domain.model.Driver
import com.example.driverguardian.domain.model.DriverSummary
import com.example.driverguardian.domain.model.Vehicle
import com.example.driverguardian.ui.components.DashboardCard
import com.example.driverguardian.ui.session.DrivingSessionUiState
import com.example.driverguardian.ui.session.LoadState
import com.example.driverguardian.ui.theme.DangerRed
import com.example.driverguardian.ui.theme.SafeGreen

@Composable
fun SelectionScreen(
    state: DrivingSessionUiState,
    onSelectDriver: (Int) -> Unit,
    onSelectVehicle: (Int) -> Unit,
    onAddVehicle: ((String, String, String, String?, (Boolean, String?) -> Unit) -> Unit)? = null,
    onEditVehicle: ((Int, String?, String?, String?, String?, (Boolean, String?) -> Unit) -> Unit)? = null,
    onRetry: () -> Unit,
    onCancel: () -> Unit,
    onContinue: () -> Unit
) {
    var showAddDialog by remember { mutableStateOf(false) }
    var vehicleToEdit by remember { mutableStateOf<Vehicle?>(null) }

    Column(modifier = Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        when (val loadState = state.loadState) {
            LoadState.Loading -> Row(
                modifier = Modifier.weight(1f).fillMaxWidth(),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                CircularProgressIndicator()
                Text("  Đang tải dữ liệu…")
            }
            is LoadState.Error -> StatusPanel(loadState.message, onRetry, Modifier.weight(1f))
            is LoadState.Empty -> StatusPanel(loadState.message, onRetry, Modifier.weight(1f))
            LoadState.Success -> SelectionLists(
                state = state,
                onSelectDriver = onSelectDriver,
                onSelectVehicle = onSelectVehicle,
                onOpenAddVehicle = { showAddDialog = true },
                onOpenEditVehicle = { vehicleToEdit = it },
                modifier = Modifier.weight(1f)
            )
        }

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            OutlinedButton(onClick = onCancel, modifier = Modifier.weight(1f).height(56.dp)) { Text("Hủy") }
            Button(onClick = onContinue, enabled = state.canContinue, modifier = Modifier.weight(1f).height(56.dp)) {
                Text("Tiếp tục")
            }
        }
    }

    if (showAddDialog && onAddVehicle != null) {
        VehicleFormDialog(
            title = "Thêm phương tiện mới",
            confirmButtonText = "Lưu phương tiện",
            onDismiss = { showAddDialog = false },
            onSubmit = { plate, name, type, device, onComplete ->
                onAddVehicle(plate, name, type, device, onComplete)
            }
        )
    }

    if (vehicleToEdit != null && onEditVehicle != null) {
        val v = vehicleToEdit!!
        VehicleFormDialog(
            title = "Chỉnh sửa phương tiện",
            confirmButtonText = "Lưu thay đổi",
            initialPlate = v.plateNumber,
            initialName = v.name ?: "",
            initialType = v.type ?: "",
            initialDevice = v.deviceCode ?: "",
            onDismiss = { vehicleToEdit = null },
            onSubmit = { plate, name, type, device, onComplete ->
                onEditVehicle(v.id, plate, name, type, device, onComplete)
            }
        )
    }
}

@Composable
private fun SelectionLists(
    state: DrivingSessionUiState,
    onSelectDriver: (Int) -> Unit,
    onSelectVehicle: (Int) -> Unit,
    onOpenAddVehicle: () -> Unit,
    onOpenEditVehicle: (Vehicle) -> Unit,
    modifier: Modifier = Modifier
) {
    Row(modifier = modifier, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
        if (state.authenticatedDriver != null) {
            DashboardCard("Tài xế đã đăng nhập", modifier = Modifier.weight(1f)) {
                AuthenticatedDriverCard(state.authenticatedDriver)
            }
        } else {
            DashboardCard("Chọn tài xế", modifier = Modifier.weight(1f)) {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    items(state.drivers, key = { it.id }) { driver ->
                        DriverCard(driver, driver.id == state.selectedDriverId) { onSelectDriver(driver.id) }
                    }
                }
            }
        }

        DashboardCard("Chọn phương tiện", modifier = Modifier.weight(1f)) {
            if (state.vehicles.isEmpty()) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.DirectionsCar,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.size(52.dp)
                    )
                    Spacer(modifier = Modifier.height(14.dp))
                    Text(
                        text = "Bạn chưa có phương tiện",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = "Vui lòng thêm phương tiện của bạn để bắt đầu chuyến đi.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(18.dp))
                    Button(
                        onClick = onOpenAddVehicle,
                        colors = ButtonDefaults.buttonColors(containerColor = SafeGreen)
                    ) {
                        Icon(Icons.Default.Add, contentDescription = null, tint = Color.Black, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Thêm phương tiện", color = Color.Black, fontWeight = FontWeight.SemiBold)
                    }
                }
            } else {
                Column(modifier = Modifier.fillMaxSize()) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 8.dp),
                        horizontalArrangement = Arrangement.End
                    ) {
                        OutlinedButton(
                            onClick = onOpenAddVehicle,
                            contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp)
                        ) {
                            Icon(Icons.Default.Add, contentDescription = null, modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(4.dp))
                            Text("Thêm phương tiện", style = MaterialTheme.typography.labelMedium)
                        }
                    }

                    LazyColumn(
                        modifier = Modifier.weight(1f),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        items(state.vehicles, key = { it.id }) { vehicle ->
                            VehicleCard(
                                vehicle = vehicle,
                                selected = vehicle.id == state.selectedVehicleId,
                                onClick = { onSelectVehicle(vehicle.id) },
                                onEdit = { onOpenEditVehicle(vehicle) }
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun StatusPanel(message: String, onRetry: () -> Unit, modifier: Modifier = Modifier) {
    DashboardCard("Không thể chuẩn bị chuyến đi", modifier = modifier) {
        Text(message, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(modifier = Modifier.height(12.dp))
        Button(onClick = onRetry) { Text("Thử lại") }
    }
}

@Composable
private fun AuthenticatedDriverCard(driver: DriverSummary) {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        border = BorderStroke(2.dp, SafeGreen)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().height(88.dp).padding(horizontal = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(14.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(Icons.Default.Person, contentDescription = null, tint = SafeGreen)
            Column(modifier = Modifier.weight(1f)) {
                Text(driver.fullName, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                Text("${driver.driverCode} • Đã xác thực Google", color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

@Composable
private fun DriverCard(driver: Driver, selected: Boolean, onClick: () -> Unit) {
    SelectableCard(selected, onClick) {
        Icon(Icons.Default.Person, contentDescription = null, tint = SafeGreen)
        Column(modifier = Modifier.weight(1f)) {
            Text(driver.fullName, style = MaterialTheme.typography.titleMedium)
            Text("${driver.code} • ${driver.status}", color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun VehicleCard(
    vehicle: Vehicle,
    selected: Boolean,
    onClick: () -> Unit,
    onEdit: (() -> Unit)? = null
) {
    SelectableCard(selected, onClick) {
        Icon(Icons.Default.DirectionsCar, contentDescription = null, tint = SafeGreen)
        Column(modifier = Modifier.weight(1f)) {
            Text(vehicle.name ?: vehicle.plateNumber, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Text("${vehicle.plateNumber} • ${vehicle.type ?: "Chưa phân loại"}", color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text("Thiết bị: ${vehicle.deviceCode ?: "Chưa gán"}", color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        if (onEdit != null) {
            IconButton(onClick = onEdit) {
                Icon(
                    imageVector = Icons.Default.Edit,
                    contentDescription = "Chỉnh sửa",
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.size(20.dp)
                )
            }
        }
    }
}

@Composable
private fun SelectableCard(selected: Boolean, onClick: () -> Unit, content: @Composable RowScope.() -> Unit) {
    Card(
        onClick = onClick,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        border = BorderStroke(if (selected) 2.dp else 1.dp, if (selected) SafeGreen else MaterialTheme.colorScheme.outlineVariant)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().height(88.dp).padding(horizontal = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(14.dp),
            verticalAlignment = Alignment.CenterVertically,
            content = content
        )
    }
}

@Composable
fun VehicleFormDialog(
    title: String,
    confirmButtonText: String,
    initialPlate: String = "",
    initialName: String = "",
    initialType: String = "",
    initialDevice: String = "",
    onDismiss: () -> Unit,
    onSubmit: (plate: String, name: String, type: String, device: String?, onComplete: (Boolean, String?) -> Unit) -> Unit
) {
    var plateNumber by remember { mutableStateOf(initialPlate) }
    var vehicleName by remember { mutableStateOf(initialName) }
    var vehicleType by remember { mutableStateOf(initialType) }
    var deviceCode by remember { mutableStateOf(initialDevice) }

    var plateError by remember { mutableStateOf<String?>(null) }
    var nameError by remember { mutableStateOf<String?>(null) }
    var typeError by remember { mutableStateOf<String?>(null) }
    var serverError by remember { mutableStateOf<String?>(null) }
    var isSubmitting by remember { mutableStateOf(false) }

    fun validate(): Boolean {
        var valid = true
        if (plateNumber.trim().length < 3) {
            plateError = "Biển số xe phải từ 3 ký tự trở lên."
            valid = false
        } else {
            plateError = null
        }
        if (vehicleName.trim().length < 2) {
            nameError = "Tên xe phải từ 2 ký tự trở lên."
            valid = false
        } else {
            nameError = null
        }
        if (vehicleType.trim().length < 2) {
            typeError = "Loại xe phải từ 2 ký tự trở lên."
            valid = false
        } else {
            typeError = null
        }
        return valid
    }

    AlertDialog(
        onDismissRequest = { if (!isSubmitting) onDismiss() },
        title = { Text(title, fontWeight = FontWeight.Bold) },
        text = {
            Column(
                modifier = Modifier.verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                if (serverError != null) {
                    Text(serverError!!, color = DangerRed, style = MaterialTheme.typography.bodySmall)
                }

                OutlinedTextField(
                    value = plateNumber,
                    onValueChange = { plateNumber = it.uppercase(); plateError = null; serverError = null },
                    label = { Text("Biển số xe *") },
                    placeholder = { Text("Ví dụ: 51A-12345") },
                    isError = plateError != null,
                    supportingText = plateError?.let { { Text(it, color = DangerRed) } },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )

                OutlinedTextField(
                    value = vehicleName,
                    onValueChange = { vehicleName = it; nameError = null; serverError = null },
                    label = { Text("Tên xe *") },
                    placeholder = { Text("Ví dụ: VinFast VF 8") },
                    isError = nameError != null,
                    supportingText = nameError?.let { { Text(it, color = DangerRed) } },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )

                OutlinedTextField(
                    value = vehicleType,
                    onValueChange = { vehicleType = it; typeError = null; serverError = null },
                    label = { Text("Loại xe *") },
                    placeholder = { Text("Ví dụ: SUV / Sedan / Xe tải") },
                    isError = typeError != null,
                    supportingText = typeError?.let { { Text(it, color = DangerRed) } },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )

                OutlinedTextField(
                    value = deviceCode,
                    onValueChange = { deviceCode = it; serverError = null },
                    label = { Text("Mã thiết bị (tùy chọn)") },
                    placeholder = { Text("Ví dụ: DEVICE001") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    if (validate()) {
                        isSubmitting = true
                        serverError = null
                        onSubmit(
                            plateNumber.trim(),
                            vehicleName.trim(),
                            vehicleType.trim(),
                            deviceCode.trim().ifEmpty { null }
                        ) { success, errorMsg ->
                            isSubmitting = false
                            if (success) {
                                onDismiss()
                            } else {
                                serverError = errorMsg ?: "Không thể lưu phương tiện"
                            }
                        }
                    }
                },
                enabled = !isSubmitting,
                colors = ButtonDefaults.buttonColors(containerColor = SafeGreen)
            ) {
                if (isSubmitting) {
                    CircularProgressIndicator(modifier = Modifier.size(18.dp), color = Color.Black)
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Đang lưu…", color = Color.Black)
                } else {
                    Text(confirmButtonText, color = Color.Black, fontWeight = FontWeight.Bold)
                }
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss, enabled = !isSubmitting) {
                Text("Hủy")
            }
        },
        shape = RoundedCornerShape(16.dp)
    )
}
