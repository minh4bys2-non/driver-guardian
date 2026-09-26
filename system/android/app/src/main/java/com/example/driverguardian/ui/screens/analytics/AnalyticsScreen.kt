package com.example.driverguardian.ui.screens.analytics

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.driverguardian.ui.components.ChartSeries
import com.example.driverguardian.ui.components.ChartThreshold
import com.example.driverguardian.ui.components.DashboardCard
import com.example.driverguardian.ui.components.MetricCard
import com.example.driverguardian.ui.components.MetricLineChart
import com.example.driverguardian.ui.components.ModelNmeChart
import com.example.driverguardian.ui.components.ProbabilityChart
import com.example.driverguardian.ui.components.RiskBarChart
import com.example.driverguardian.ui.mock.AiAnalyticsMockData
import com.example.driverguardian.ui.mock.ModelSummary
import com.example.driverguardian.ui.theme.AccentBlue
import com.example.driverguardian.ui.theme.DangerRed
import com.example.driverguardian.ui.theme.DriverGuardianTheme
import com.example.driverguardian.ui.theme.SafeGreen
import com.example.driverguardian.ui.theme.WarningYellow

@Composable
fun AnalyticsScreen() {
    val data = AiAnalyticsMockData
    val latest = data.latestMetrics
    val timeline = data.metricTimeline
    val timeLabels = timeline.map { it.time }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("Phân tích & Đánh giá AI", style = MaterialTheme.typography.headlineMedium)
            Text(
                "Dữ liệu minh họa phục vụ đánh giá giao diện",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                style = MaterialTheme.typography.bodyLarge
            )
        }

        SectionTitle("A. Giám sát kết quả AI")
        MetricGrid(
            items = listOf(
                MetricDisplay("EAR", "%.2f".format(latest.ear), valueColor = WarningYellow),
                MetricDisplay("MAR", "%.2f".format(latest.mar), valueColor = WarningYellow),
                MetricDisplay("PERCLOS", "%.2f".format(latest.perclos), valueColor = DangerRed),
                MetricDisplay("LSTM Probability", latest.lstmProbability.toPercent()),
                MetricDisplay("Fused Probability", latest.fusedProbability.toPercent(), valueColor = DangerRed),
                MetricDisplay("Alert Status", if (data.alertActive) "Cảnh báo" else "An toàn", valueColor = if (data.alertActive) DangerRed else SafeGreen)
            )
        )

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            RiskMetricCard("Eye Risk", data.riskSnapshot.eyeRisk, Modifier.weight(1f))
            RiskMetricCard("Mouth Risk", data.riskSnapshot.mouthRisk, Modifier.weight(1f))
            RiskMetricCard("Head Risk", data.riskSnapshot.headRisk, Modifier.weight(1f))
        }

        MetricTimelineCharts(timeLabels, timelineModifier = Modifier.fillMaxWidth())

        BoxWithConstraints {
            val compact = maxWidth < 980.dp
            if (compact) {
                Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    DashboardCard("Xác suất buồn ngủ theo thời gian") {
                        ProbabilityChart(timeline)
                    }
                    DashboardCard("Thành phần nguy cơ buồn ngủ") {
                        RiskBarChart(data.riskSnapshot)
                    }
                    SafetyOverrideCard()
                }
            } else {
                Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                    DashboardCard("Xác suất buồn ngủ theo thời gian", modifier = Modifier.weight(1.25f)) {
                        ProbabilityChart(timeline)
                    }
                    Column(modifier = Modifier.weight(0.75f), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                        DashboardCard("Thành phần nguy cơ buồn ngủ") {
                            RiskBarChart(data.riskSnapshot)
                        }
                        SafetyOverrideCard()
                    }
                }
            }
        }

        SectionTitle("B. Đánh giá mô hình")
        DashboardCard("Kết quả đánh giá mô hình") {
            Text("NME càng thấp càng tốt", color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodyLarge)
            Spacer(Modifier.height(12.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                data.modelSummaries.forEach { summary ->
                    ModelResultCard(summary, Modifier.weight(1f))
                }
            }
            Spacer(Modifier.height(12.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                MetricCard("Average Best NME", "%.5f".format(data.averageBestNme), Modifier.weight(1f), SafeGreen)
                MetricCard("Best Specialist", data.bestSpecialist, Modifier.weight(1f), SafeGreen)
            }
        }

        DashboardCard("NME theo quá trình huấn luyện") {
            ModelNmeChart(data.nmeTimeline)
        }

        DashboardCard("Bảng kết quả Testing") {
            EvaluationTable(data.modelSummaries, data.averageBestNme)
        }

        DashboardCard("Full Face Landmark Model") {
            Text(
                "Validation loss được trình bày riêng, không so sánh trên cùng trục với NME.",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                style = MaterialTheme.typography.bodyLarge
            )
            Spacer(Modifier.height(12.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                data.fullFaceStages.forEach { stage ->
                    Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Text(stage.label, style = MaterialTheme.typography.titleMedium)
                        MetricCard("Best Validation Loss", "%.6f".format(stage.bestValidationLoss), Modifier.fillMaxWidth(), AccentBlue)
                        MetricCard("Best Epoch", stage.bestEpoch.toString(), Modifier.fillMaxWidth(), SafeGreen)
                    }
                }
            }
        }
    }
}

@Composable
private fun MetricTimelineCharts(
    timeLabels: List<String>,
    timelineModifier: Modifier = Modifier
) {
    val timeline = AiAnalyticsMockData.metricTimeline
    DashboardCard("Biến động chỉ số theo thời gian", modifier = timelineModifier) {
        BoxWithConstraints {
            val compact = maxWidth < 900.dp
            if (compact) {
                Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
                    SmallMetricTimeline(
                        title = "EAR Timeline",
                        xLabels = timeLabels,
                        color = AccentBlue,
                        values = timeline.map { it.ear },
                        safe = 0.25f,
                        danger = 0.15f,
                        yRange = 0.10f..0.32f,
                        modifier = Modifier.fillMaxWidth()
                    )
                    SmallMetricTimeline(
                        title = "MAR Timeline",
                        xLabels = timeLabels,
                        color = WarningYellow,
                        values = timeline.map { it.mar },
                        safe = 0.50f,
                        danger = 0.73f,
                        yRange = 0.30f..0.80f,
                        modifier = Modifier.fillMaxWidth()
                    )
                    SmallMetricTimeline(
                        title = "PERCLOS Timeline",
                        xLabels = timeLabels,
                        color = DangerRed,
                        values = timeline.map { it.perclos },
                        safe = 0.10f,
                        danger = 0.15f,
                        yRange = 0f..0.22f,
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            } else {
                Row(horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                    SmallMetricTimeline(
                        title = "EAR Timeline",
                        xLabels = timeLabels,
                        color = AccentBlue,
                        values = timeline.map { it.ear },
                        safe = 0.25f,
                        danger = 0.15f,
                        yRange = 0.10f..0.32f,
                        modifier = Modifier.weight(1f)
                    )
                    SmallMetricTimeline(
                        title = "MAR Timeline",
                        xLabels = timeLabels,
                        color = WarningYellow,
                        values = timeline.map { it.mar },
                        safe = 0.50f,
                        danger = 0.73f,
                        yRange = 0.30f..0.80f,
                        modifier = Modifier.weight(1f)
                    )
                    SmallMetricTimeline(
                        title = "PERCLOS Timeline",
                        xLabels = timeLabels,
                        color = DangerRed,
                        values = timeline.map { it.perclos },
                        safe = 0.10f,
                        danger = 0.15f,
                        yRange = 0f..0.22f,
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }
    }
}

@Composable
private fun SmallMetricTimeline(
    title: String,
    xLabels: List<String>,
    color: Color,
    values: List<Float>,
    safe: Float,
    danger: Float,
    yRange: ClosedFloatingPointRange<Float>,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        Text(title, style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(8.dp))
        MetricLineChart(
            xLabels = xLabels,
            series = listOf(ChartSeries(title.removeSuffix(" Timeline"), color, values)),
            thresholds = listOf(
                ChartThreshold("Safe threshold", safe, SafeGreen),
                ChartThreshold("Danger threshold", danger, DangerRed)
            ),
            yRange = yRange
        )
    }
}

@Composable
private fun SafetyOverrideCard() {
    DashboardCard("Cảnh báo an toàn", borderColor = if (AiAnalyticsMockData.hardOverride) DangerRed else SafeGreen) {
        if (AiAnalyticsMockData.hardOverride) {
            Text("Hard override đang bật", color = DangerRed, style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(8.dp))
            AiAnalyticsMockData.overrideReasons.forEach { reason ->
                Text("• ${AiAnalyticsMockData.overrideReasonLabel(reason)}", style = MaterialTheme.typography.bodyLarge)
            }
        } else {
            Text("Không có cảnh báo an toàn cưỡng bức", color = SafeGreen, style = MaterialTheme.typography.bodyLarge)
        }
    }
}

@Composable
private fun EvaluationTable(
    summaries: List<ModelSummary>,
    averageBestNme: Float
) {
    TableRow(
        values = listOf("Model", "Best Epoch", "Initial NME", "Best NME", "Improvement"),
        strong = true,
        color = MaterialTheme.colorScheme.onSurfaceVariant
    )
    summaries.forEach { summary ->
        TableRow(
            values = listOf(
                summary.name,
                summary.bestEpoch.toString(),
                "%.5f".format(summary.initialNme),
                "%.5f".format(summary.bestNme),
                "%.2f%%".format(summary.improvementPercent)
            )
        )
    }
    Spacer(Modifier.height(10.dp))
    Text("Average Best NME: %.5f".format(averageBestNme), color = SafeGreen, style = MaterialTheme.typography.titleMedium)
}

@Composable
private fun TableRow(
    values: List<String>,
    strong: Boolean = false,
    color: Color = MaterialTheme.colorScheme.onSurface
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 7.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        values.forEachIndexed { index, value ->
            Text(
                text = value,
                modifier = Modifier.weight(if (index == 0) 1.25f else 1f),
                color = color,
                style = if (strong) MaterialTheme.typography.labelLarge else MaterialTheme.typography.bodyLarge
            )
        }
    }
}

@Composable
private fun ModelResultCard(
    summary: ModelSummary,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text(summary.name, style = MaterialTheme.typography.titleMedium)
        MetricCard("Best Epoch", summary.bestEpoch.toString(), Modifier.fillMaxWidth())
        MetricCard("Best NME", "%.5f".format(summary.bestNme), Modifier.fillMaxWidth(), SafeGreen)
        MetricCard("Improvement", "%.2f%%".format(summary.improvementPercent), Modifier.fillMaxWidth(), AccentBlue)
    }
}

@Composable
private fun MetricGrid(items: List<MetricDisplay>) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        items.chunked(3).forEach { rowItems ->
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                rowItems.forEach { item ->
                    MetricCard(
                        label = item.label,
                        value = item.value,
                        modifier = Modifier.weight(1f),
                        valueColor = item.valueColor ?: MaterialTheme.colorScheme.onSurface
                    )
                }
            }
        }
    }
}

@Composable
private fun RiskMetricCard(
    label: String,
    value: Float,
    modifier: Modifier = Modifier
) {
    DashboardCard(label, modifier = modifier) {
        Text(value.toPercent(), color = riskColor(value), style = MaterialTheme.typography.headlineMedium)
        Spacer(Modifier.height(10.dp))
        LinearProgressIndicator(
            progress = { value.coerceIn(0f, 1f) },
            modifier = Modifier.fillMaxWidth(),
            color = riskColor(value),
            trackColor = MaterialTheme.colorScheme.surfaceVariant
        )
    }
}

@Composable
private fun SectionTitle(text: String) {
    Text(text, color = SafeGreen, style = MaterialTheme.typography.titleLarge)
}

private data class MetricDisplay(
    val label: String,
    val value: String,
    val valueColor: Color? = null
)

private fun Float.toPercent(): String = "${(this * 100).toInt()}%"

private fun riskColor(value: Float): Color = when {
    value >= 0.75f -> DangerRed
    value >= 0.55f -> WarningYellow
    else -> SafeGreen
}

@Preview(widthDp = 1280, heightDp = 720, showBackground = true)
@Composable
private fun AnalyticsScreenPreview() {
    DriverGuardianTheme {
        Box(
            Modifier
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.background)
                .padding(20.dp)
        ) {
            AnalyticsScreen()
        }
    }
}
