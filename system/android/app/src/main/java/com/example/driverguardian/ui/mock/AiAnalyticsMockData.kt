package com.example.driverguardian.ui.mock

data class AiMetricPoint(
    val time: String,
    val ear: Float,
    val mar: Float,
    val perclos: Float,
    val lstmProbability: Float,
    val fusedProbability: Float
)

data class RiskSnapshot(
    val eyeRisk: Float,
    val mouthRisk: Float,
    val headRisk: Float
)

data class ModelNmePoint(
    val epoch: Int,
    val leftEye: Float?,
    val rightEye: Float?,
    val mouth: Float?
)

data class ModelSummary(
    val name: String,
    val bestEpoch: Int,
    val initialNme: Float,
    val bestNme: Float,
    val improvementPercent: Float
)

data class FullFaceTrainingStage(
    val label: String,
    val bestValidationLoss: Float,
    val bestEpoch: Int
)

object AiAnalyticsMockData {
    val latestMetrics = AiMetricPoint(
        time = "01:00",
        ear = 0.18f,
        mar = 0.61f,
        perclos = 0.17f,
        lstmProbability = 0.72f,
        fusedProbability = 0.81f
    )

    val alertActive = true

    val riskSnapshot = RiskSnapshot(
        eyeRisk = 0.82f,
        mouthRisk = 0.63f,
        headRisk = 0.41f
    )

    val metricTimeline = listOf(
        AiMetricPoint("00:00", 0.29f, 0.34f, 0.04f, 0.10f, 0.11f),
        AiMetricPoint("00:10", 0.27f, 0.38f, 0.05f, 0.15f, 0.17f),
        AiMetricPoint("00:20", 0.25f, 0.42f, 0.07f, 0.22f, 0.25f),
        AiMetricPoint("00:30", 0.22f, 0.49f, 0.09f, 0.35f, 0.40f),
        AiMetricPoint("00:40", 0.19f, 0.57f, 0.12f, 0.50f, 0.53f),
        AiMetricPoint("00:50", 0.17f, 0.66f, 0.16f, 0.63f, 0.68f),
        AiMetricPoint("01:00", 0.16f, 0.75f, 0.20f, 0.78f, 0.85f)
    )

    val hardOverride = true
    val overrideReasons = listOf("perclos_sustained")

    val modelSummaries = listOf(
        ModelSummary("Left Eye", bestEpoch = 12, initialNme = 0.66143f, bestNme = 0.10085f, improvementPercent = 84.75f),
        ModelSummary("Right Eye", bestEpoch = 25, initialNme = 0.63918f, bestNme = 0.08083f, improvementPercent = 87.35f),
        ModelSummary("Mouth", bestEpoch = 24, initialNme = 0.63098f, bestNme = 0.08302f, improvementPercent = 86.84f)
    )

    val averageBestNme = 0.08823f
    val bestSpecialist = "Right Eye"

    val nmeTimeline = listOf(
        ModelNmePoint(1, 0.66143f, 0.63918f, 0.63098f),
        ModelNmePoint(2, 0.48803f, 0.63620f, 0.33671f),
        ModelNmePoint(3, 0.30690f, 0.36851f, 0.39247f),
        ModelNmePoint(4, 0.24240f, 0.21324f, 0.28345f),
        ModelNmePoint(5, 0.19144f, 0.25657f, 0.17555f),
        ModelNmePoint(6, 0.16695f, 0.15783f, 0.15336f),
        ModelNmePoint(7, 0.16122f, 0.13882f, 0.12624f),
        ModelNmePoint(9, 0.14686f, 0.15110f, 0.13320f),
        ModelNmePoint(11, 0.13043f, 0.10396f, 0.12242f),
        ModelNmePoint(12, 0.10085f, 0.09549f, 0.14448f),
        ModelNmePoint(14, null, 0.09475f, 0.10702f),
        ModelNmePoint(18, null, 0.08780f, 0.10150f),
        ModelNmePoint(21, null, 0.08360f, 0.08710f),
        ModelNmePoint(24, null, 0.09962f, 0.08302f),
        ModelNmePoint(25, null, 0.08083f, 0.08908f)
    )

    val fullFaceStages = listOf(
        FullFaceTrainingStage("Transfer lần 1", bestValidationLoss = 2.1702f, bestEpoch = 42),
        FullFaceTrainingStage("Fine-tune", bestValidationLoss = 3.201954f, bestEpoch = 33)
    )

    fun overrideReasonLabel(reason: String): String = when (reason) {
        "perclos_sustained" -> "PERCLOS duy trì ở mức nguy hiểm"
        "confirmed_yawn" -> "Phát hiện ngáp kéo dài"
        "head_microsleep" -> "Phát hiện gật đầu/microsleep"
        else -> reason
    }
}
