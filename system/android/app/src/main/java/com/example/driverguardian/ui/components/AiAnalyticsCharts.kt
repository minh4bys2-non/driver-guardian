package com.example.driverguardian.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import com.example.driverguardian.ui.mock.AiMetricPoint
import com.example.driverguardian.ui.mock.ModelNmePoint
import com.example.driverguardian.ui.mock.RiskSnapshot
import com.example.driverguardian.ui.theme.AccentBlue
import com.example.driverguardian.ui.theme.DangerRed
import com.example.driverguardian.ui.theme.SafeGreen
import com.example.driverguardian.ui.theme.WarningYellow
import kotlin.math.max

data class ChartSeries(
    val label: String,
    val color: Color,
    val values: List<Float?>
)

data class ChartThreshold(
    val label: String,
    val value: Float,
    val color: Color
)

@Composable
fun MetricLineChart(
    xLabels: List<String>,
    series: List<ChartSeries>,
    thresholds: List<ChartThreshold>,
    yRange: ClosedFloatingPointRange<Float>,
    modifier: Modifier = Modifier,
    valueFormatter: (Float) -> String = { "%.2f".format(it) }
) {
    ChartLegend(series = series, thresholds = thresholds)
    Spacer(Modifier.height(10.dp))
    LineChartCanvas(
        xLabels = xLabels,
        series = series,
        thresholds = thresholds,
        yRange = yRange,
        valueFormatter = valueFormatter,
        modifier = modifier
            .fillMaxWidth()
            .heightIn(min = 220.dp)
            .height(260.dp)
    )
}

@Composable
fun ProbabilityChart(
    points: List<AiMetricPoint>,
    modifier: Modifier = Modifier
) {
    MetricLineChart(
        xLabels = points.map { it.time },
        series = listOf(
            ChartSeries("LSTM Probability", AccentBlue, points.map { it.lstmProbability }),
            ChartSeries("Fused Probability", SafeGreen, points.map { it.fusedProbability })
        ),
        thresholds = listOf(ChartThreshold("Alert Threshold = 55%", 0.55f, DangerRed)),
        yRange = 0f..1f,
        valueFormatter = { "${(it * 100).toInt()}%" },
        modifier = modifier
    )
    if ((points.lastOrNull()?.fusedProbability ?: 0f) >= 0.55f) {
        Spacer(Modifier.height(8.dp))
        Text("Trạng thái vùng cuối: Cảnh báo", color = DangerRed, style = MaterialTheme.typography.labelLarge)
    }
}

@Composable
fun RiskBarChart(
    risk: RiskSnapshot,
    modifier: Modifier = Modifier
) {
    val labelColor = MaterialTheme.colorScheme.onSurfaceVariant
    val trackColor = MaterialTheme.colorScheme.surfaceVariant
    val bars = listOf(
        "Eye" to risk.eyeRisk,
        "Mouth" to risk.mouthRisk,
        "Head" to risk.headRisk
    )

    Column(modifier = modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        bars.forEach { (label, value) ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(label, modifier = Modifier.width(64.dp), style = MaterialTheme.typography.bodyMedium, color = labelColor)
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .height(22.dp)
                        .background(trackColor, RoundedCornerShape(6.dp))
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth(value.coerceIn(0f, 1f))
                            .height(22.dp)
                            .background(riskColor(value), RoundedCornerShape(6.dp))
                    )
                }
                Text("${(value * 100).toInt()}%", modifier = Modifier.width(48.dp), color = riskColor(value), style = MaterialTheme.typography.labelLarge)
            }
        }
        Text("Trục Y quy đổi 0-100%, giúp so sánh nguồn rủi ro từ mắt, miệng và tư thế đầu.", color = labelColor, style = MaterialTheme.typography.bodyMedium)
    }
}

@Composable
fun ModelNmeChart(
    points: List<ModelNmePoint>,
    modifier: Modifier = Modifier
) {
    val xLabels = points.map { it.epoch.toString() }
    MetricLineChart(
        xLabels = xLabels,
        series = listOf(
            ChartSeries("Left Eye", AccentBlue, points.map { it.leftEye }),
            ChartSeries("Right Eye", SafeGreen, points.map { it.rightEye }),
            ChartSeries("Mouth", WarningYellow, points.map { it.mouth })
        ),
        thresholds = emptyList(),
        yRange = 0f..0.70f,
        valueFormatter = { "%.2f".format(it) },
        modifier = modifier
    )
    Text("Dot đánh dấu từng epoch; Left Eye giữ gap sau khi không còn dữ liệu NME.", color = MaterialTheme.colorScheme.onSurfaceVariant, style = MaterialTheme.typography.bodyMedium)
}

@Composable
private fun ChartLegend(
    series: List<ChartSeries>,
    thresholds: List<ChartThreshold>
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(horizontalArrangement = Arrangement.spacedBy(14.dp), verticalAlignment = Alignment.CenterVertically) {
            series.forEach { item ->
                LegendItem(label = item.label, color = item.color)
            }
        }
        if (thresholds.isNotEmpty()) {
            Row(horizontalArrangement = Arrangement.spacedBy(14.dp), verticalAlignment = Alignment.CenterVertically) {
                thresholds.forEach { item ->
                    LegendItem(label = item.label, color = item.color, dashed = true)
                }
            }
        }
    }
}

@Composable
private fun LegendItem(
    label: String,
    color: Color,
    dashed: Boolean = false
) {
    val labelColor = MaterialTheme.colorScheme.onSurfaceVariant
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        Canvas(modifier = Modifier.width(28.dp).height(10.dp)) {
            drawLine(
                color = color,
                start = Offset(0f, size.height / 2f),
                end = Offset(size.width, size.height / 2f),
                strokeWidth = 3.dp.toPx(),
                cap = StrokeCap.Round,
                pathEffect = if (dashed) PathEffect.dashPathEffect(floatArrayOf(8.dp.toPx(), 6.dp.toPx())) else null
            )
        }
        Text(label, color = labelColor, style = MaterialTheme.typography.bodyMedium)
    }
}

@Composable
private fun LineChartCanvas(
    xLabels: List<String>,
    series: List<ChartSeries>,
    thresholds: List<ChartThreshold>,
    yRange: ClosedFloatingPointRange<Float>,
    valueFormatter: (Float) -> String,
    modifier: Modifier = Modifier
) {
    val containerColor = MaterialTheme.colorScheme.surfaceVariant
    val labelColor = MaterialTheme.colorScheme.onSurfaceVariant
    val gridStrongColor = labelColor.copy(alpha = 0.18f)
    val gridSoftColor = labelColor.copy(alpha = 0.10f)
    val markerFillColor = MaterialTheme.colorScheme.surface

    Column(
        modifier = modifier
            .background(containerColor, RoundedCornerShape(8.dp))
            .padding(12.dp)
    ) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(valueFormatter(yRange.endInclusive), color = labelColor, style = MaterialTheme.typography.bodyMedium)
            Text(valueFormatter(yRange.start), color = labelColor, style = MaterialTheme.typography.bodyMedium)
        }
        Canvas(modifier = Modifier.fillMaxWidth().weight(1f)) {
            if (xLabels.size < 2) return@Canvas

            val leftPadding = 34.dp.toPx()
            val rightPadding = 10.dp.toPx()
            val topPadding = 8.dp.toPx()
            val bottomPadding = 12.dp.toPx()
            val plotLeft = leftPadding
            val plotRight = size.width - rightPadding
            val plotTop = topPadding
            val plotBottom = size.height - bottomPadding
            val plotWidth = max(1f, plotRight - plotLeft)
            val plotHeight = max(1f, plotBottom - plotTop)
            val minY = yRange.start
            val maxY = yRange.endInclusive
            val ySpan = max(0.001f, maxY - minY)

            fun offsetFor(index: Int, value: Float): Offset {
                val x = plotLeft + plotWidth * (index.toFloat() / (xLabels.lastIndex).coerceAtLeast(1))
                val normalized = ((value - minY) / ySpan).coerceIn(0f, 1f)
                val y = plotBottom - plotHeight * normalized
                return Offset(x, y)
            }

            repeat(5) { tick ->
                val y = plotTop + plotHeight * (tick / 4f)
                drawLine(
                    color = gridStrongColor,
                    start = Offset(plotLeft, y),
                    end = Offset(plotRight, y),
                    strokeWidth = 1.dp.toPx()
                )
            }

            repeat(xLabels.size) { index ->
                val x = plotLeft + plotWidth * (index.toFloat() / (xLabels.lastIndex).coerceAtLeast(1))
                drawLine(
                    color = gridSoftColor,
                    start = Offset(x, plotTop),
                    end = Offset(x, plotBottom),
                    strokeWidth = 1.dp.toPx()
                )
            }

            thresholds.forEach { threshold ->
                val y = offsetFor(0, threshold.value).y
                drawLine(
                    color = threshold.color,
                    start = Offset(plotLeft, y),
                    end = Offset(plotRight, y),
                    strokeWidth = 2.dp.toPx(),
                    pathEffect = PathEffect.dashPathEffect(floatArrayOf(10.dp.toPx(), 8.dp.toPx()))
                )
            }

            series.forEach { item ->
                var path = Path()
                var hasActivePath = false

                item.values.forEachIndexed { index, value ->
                    if (value == null) {
                        if (hasActivePath) {
                            drawPath(
                                path = path,
                                color = item.color,
                                style = Stroke(width = 3.dp.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round)
                            )
                        }
                        path = Path()
                        hasActivePath = false
                    } else {
                        val point = offsetFor(index, value)
                        if (!hasActivePath) {
                            path.moveTo(point.x, point.y)
                            hasActivePath = true
                        } else {
                            path.lineTo(point.x, point.y)
                        }
                    }
                }

                if (hasActivePath) {
                    drawPath(
                        path = path,
                        color = item.color,
                        style = Stroke(width = 3.dp.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round)
                    )
                }

                item.values.forEachIndexed { index, value ->
                    if (value != null) {
                        drawCircle(color = item.color, radius = 4.5.dp.toPx(), center = offsetFor(index, value))
                        drawCircle(color = markerFillColor, radius = 2.dp.toPx(), center = offsetFor(index, value))
                    }
                }
            }
        }
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            xLabels.forEachIndexed { index, label ->
                if (index == 0 || index == xLabels.lastIndex || index % 2 == 0) {
                    Text(label, color = labelColor, style = MaterialTheme.typography.bodyMedium)
                }
            }
        }
    }
}

private fun riskColor(value: Float): Color = when {
    value >= 0.75f -> DangerRed
    value >= 0.55f -> WarningYellow
    else -> SafeGreen
}
