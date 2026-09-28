package com.example.driverguardian.ui.history

import com.example.driverguardian.domain.model.DrowsinessEvent
import java.time.LocalDateTime
import java.time.OffsetDateTime
import java.time.format.DateTimeFormatter

private const val MissingValue = "—"
private val displayDateTime = DateTimeFormatter.ofPattern("dd/MM/yyyy HH:mm:ss")

data class EventCounts(val warnings: Int, val dangers: Int)

fun formatDuration(seconds: Int): String {
    val safeSeconds = seconds.coerceAtLeast(0)
    val hours = safeSeconds / 3600
    val minutes = (safeSeconds % 3600) / 60
    val remainingSeconds = safeSeconds % 60
    return "%02d:%02d:%02d".format(hours, minutes, remainingSeconds)
}

fun formatNullableNumber(value: Number?): String = value?.toString() ?: MissingValue

fun formatDateTime(value: String?): String {
    if (value.isNullOrBlank()) return MissingValue
    return runCatching { OffsetDateTime.parse(value).format(displayDateTime) }
        .recoverCatching { LocalDateTime.parse(value).format(displayDateTime) }
        .getOrDefault(value)
}

fun countEvents(events: List<DrowsinessEvent>): EventCounts = EventCounts(
    warnings = events.count { it.driverState == "WARNING" || it.alertLevel == 1 },
    dangers = events.count { it.driverState == "DANGER" || it.alertLevel >= 2 }
)
