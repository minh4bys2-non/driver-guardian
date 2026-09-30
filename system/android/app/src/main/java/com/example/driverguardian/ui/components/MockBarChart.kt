package com.example.driverguardian.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.example.driverguardian.ui.mock.HourlyStatisticUiModel
import com.example.driverguardian.ui.theme.AccentBlue
import com.example.driverguardian.ui.theme.WarningYellow

@Composable
fun MockBarChart(
    data: List<HourlyStatisticUiModel>,
    modifier: Modifier = Modifier
) {
    val max = data.maxOfOrNull { it.value }?.coerceAtLeast(1) ?: 1
    val labelColor = MaterialTheme.colorScheme.onSurfaceVariant
    Row(
        modifier = modifier
            .fillMaxWidth()
            .height(220.dp)
            .padding(top = 12.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalAlignment = Alignment.Bottom
    ) {
        data.forEach { item ->
            val barHeight = 150.dp * (item.value.toFloat() / max.toFloat())
            val color = if (item.hour in 1..3) WarningYellow else AccentBlue
            Column(
                modifier = Modifier.weight(1f),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Bottom
            ) {
                Box(
                    modifier = Modifier
                        .width(18.dp)
                        .height(barHeight)
                        .clip(RoundedCornerShape(topStart = 6.dp, topEnd = 6.dp))
                        .background(color)
                )
                Text(
                    text = item.hour.toString().padStart(2, '0'),
                    color = labelColor,
                    style = MaterialTheme.typography.bodyMedium
                )
            }
        }
    }
}

@Composable
fun MockLineChart(
    values: List<Int>,
    modifier: Modifier = Modifier,
    lineColor: Color = AccentBlue
) {
    val containerColor = MaterialTheme.colorScheme.surfaceVariant
    val markerColor = MaterialTheme.colorScheme.surface
    androidx.compose.foundation.Canvas(
        modifier = modifier
            .fillMaxWidth()
            .height(180.dp)
            .background(containerColor, RoundedCornerShape(8.dp))
            .padding(12.dp)
    ) {
        if (values.size < 2) return@Canvas
        val max = values.maxOrNull()?.coerceAtLeast(1) ?: 1
        val step = size.width / (values.lastIndex)
        val path = androidx.compose.ui.graphics.Path()
        values.forEachIndexed { index, value ->
            val x = index * step
            val y = size.height - (size.height * value / max)
            if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(
            path = path,
            color = lineColor,
            style = androidx.compose.ui.graphics.drawscope.Stroke(
                width = 6.dp.toPx(),
                cap = androidx.compose.ui.graphics.StrokeCap.Round,
                join = androidx.compose.ui.graphics.StrokeJoin.Round
            )
        )
        values.forEachIndexed { index, value ->
            drawCircle(
                color = markerColor,
                radius = 5.dp.toPx(),
                center = androidx.compose.ui.geometry.Offset(
                    x = index * step,
                    y = size.height - (size.height * value / max)
                )
            )
        }
    }
}
