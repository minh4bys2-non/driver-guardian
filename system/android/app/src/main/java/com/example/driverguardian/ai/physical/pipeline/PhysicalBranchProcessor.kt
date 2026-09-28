package com.example.driverguardian.ai.physical.pipeline

import com.example.driverguardian.ai.physical.geometry.AspectRatioCalculator
import com.example.driverguardian.ai.physical.geometry.HeadPoseEstimator
import com.example.driverguardian.ai.physical.model.HeadPose
import com.example.driverguardian.ai.physical.model.LandmarkPoint
import com.example.driverguardian.ai.physical.model.PhysicalMetrics
import com.example.driverguardian.ai.physical.temporal.AdaptiveHmmFsm
import com.example.driverguardian.ai.physical.temporal.TemporalStateMachine
import com.example.driverguardian.ai.physical.temporal.WindowRatio
import kotlin.math.abs

/**
 * Production implementation of PhysicalBranch coordinating EAR/MAR geometry,
 * adaptive HMM/FSM detectors, head pose estimation with neutral calibration,
 * nod hysteresis detection, and over-angle ratio tracking.
 */
class PhysicalBranchProcessor(
    val config: PhysicalBranchConfig = PhysicalBranchConfig()
) : PhysicalBranch {

    val eye = AdaptiveHmmFsm(
        mode = "eye",
        fps = config.fps,
        initDurationSec = config.eyeInitDurationSec,
        windowSizeSec = config.windowSec,
        maxGapSec = config.maxGapSec,
        learningRate = config.learningRate,
        adaptInterval = config.adaptInterval
    )

    val mouth = AdaptiveHmmFsm(
        mode = "mouth",
        fps = config.fps,
        initDurationSec = config.mouthInitDurationSec,
        windowSizeSec = config.windowSec,
        maxGapSec = config.maxGapSec,
        learningRate = config.learningRate,
        adaptInterval = config.adaptInterval
    )

    val nod = TemporalStateMachine(
        mode = "pitch",
        fps = config.fps,
        windowSizeSec = config.windowSec,
        minDurationMs = config.nodDurationMinMs,
        maxDurationMs = config.nodDurationMaxMs,
        maxGapSec = config.maxGapSec
    )

    val overAngleRatio = WindowRatio(
        windowSec = config.windowSec,
        maxGapSec = config.maxGapSec
    )

    private var currentShape: Pair<Int, Int>? = null
    private var estimator: HeadPoseEstimator? = null

    private val neutralSamples = mutableListOf<HeadPose>()
    var neutralPose: HeadPose? = null
        private set

    var nodding: Boolean = false
        private set

    var lastTimestampSec: Double? = null
        private set

    override fun reset() {
        eye.reset()
        mouth.reset()
        nod.reset()
        overAngleRatio.reset()
        neutralSamples.clear()
        neutralPose = null
        nodding = false
        lastTimestampSec = null
    }

    override fun processLandmarks(
        landmarks: List<LandmarkPoint>?,
        imageWidth: Int,
        imageHeight: Int,
        timestampSec: Double
    ): PhysicalMetrics {
        require(timestampSec.isFinite() && (lastTimestampSec == null || timestampSec > lastTimestampSec!!)) {
            "Timestamps must be finite and strictly increasing: last=$lastTimestampSec, next=$timestampSec"
        }

        if (lastTimestampSec != null && timestampSec - lastTimestampSec!! > config.maxGapSec) {
            nodding = false
        }
        lastTimestampSec = timestampSec

        val newShape = imageWidth to imageHeight
        if (currentShape != newShape) {
            estimator = HeadPoseEstimator(imageWidth, imageHeight)
            currentShape = newShape
            neutralSamples.clear()
            neutralPose = null
            nodding = false
            nod.reset()
            overAngleRatio.reset()
        }

        val validLandmarks = if (landmarks != null && landmarks.size >= 468 && landmarks.all { it.x.isFinite() && it.y.isFinite() }) {
            landmarks
        } else {
            null
        }

        var ear: Double? = null
        var mar: Double? = null
        var rawPose: HeadPose? = null
        var normalizedAngles: DoubleArray? = null
        var poseError: String? = null

        if (validLandmarks != null) {
            ear = AspectRatioCalculator.calculateEar(validLandmarks)
            mar = AspectRatioCalculator.calculateMar(validLandmarks)

            try {
                val pose6 = HeadPoseEstimator.LANDMARKS.map { idx -> validLandmarks[idx] }
                val estimated = estimator!!.estimate(pose6)
                rawPose = estimated

                if (neutralPose == null) {
                    neutralSamples.add(estimated)
                    if (neutralSamples.size >= config.calibrationFrames) {
                        val medPitch = median(neutralSamples.map { it.pitchDeg })
                        val medYaw = median(neutralSamples.map { it.yawDeg })
                        val medRoll = median(neutralSamples.map { it.rollDeg })
                        neutralPose = HeadPose(medPitch, medYaw, medRoll)
                        neutralSamples.clear()
                    }
                }

                if (neutralPose != null) {
                    val n = neutralPose!!
                    val normPitch = normalizeAngle(estimated.pitchDeg - n.pitchDeg)
                    val normYaw = normalizeAngle(estimated.yawDeg - n.yawDeg)
                    val normRoll = normalizeAngle(estimated.rollDeg - n.rollDeg)
                    normalizedAngles = doubleArrayOf(normPitch, normYaw, normRoll)
                }
            } catch (e: Exception) {
                poseError = e.message ?: e.javaClass.simpleName
                neutralSamples.clear()
            }
        } else if (neutralPose == null) {
            neutralSamples.clear()
        }

        val eyeResult = eye.process(ear, timestampSec)
        val mouthResult = mouth.process(mar, timestampSec)

        var overAngle: Boolean? = null
        var nodState: Int? = null

        if (normalizedAngles != null) {
            val isOver = abs(normalizedAngles[0]) > config.angleLimits[0] ||
                    abs(normalizedAngles[1]) > config.angleLimits[1] ||
                    abs(normalizedAngles[2]) > config.angleLimits[2]
            overAngle = isOver

            val pitchDown = config.nodDirection * normalizedAngles[0]
            val threshold = if (nodding) config.nodReleaseDeg else config.nodPitchDeg
            nodding = pitchDown > threshold
            nodState = if (nodding) 1 else 0
        } else {
            nodding = false
        }

        val nodResult = nod.process(nodState, timestampSec)
        val (overPercent, headSeconds) = overAngleRatio.process(overAngle, timestampSec)

        return PhysicalMetrics(
            timestampSec = timestampSec,
            faceDetected = validLandmarks != null,
            eyeReady = eyeResult.initialized,
            mouthReady = mouthResult.initialized,
            headReady = normalizedAngles != null,
            headCalibrated = neutralPose != null,
            poseValid = rawPose != null,
            poseError = poseError,
            ear = ear,
            eyeState = eyeResult.state,
            perclosPct = eyeResult.perclos,
            p80EarThreshold = eyeResult.p80Threshold,
            blinkRatePerMin = eyeResult.ratePerMinute,
            blinkDetected = eyeResult.eventDone,
            eyeClosureDurationMs = if (ear != null && eyeResult.initialized) eyeResult.currentDurationMs else null,
            lastEyeClosureDurationMs = eyeResult.lastEventDurationMs,
            eyeObservedSec = eyeResult.observedSeconds,
            mar = mar,
            mouthState = mouthResult.state,
            pomPct = mouthResult.pom,
            yawnDetected = mouthResult.eventDone,
            yawningFrequencyPerMin = mouthResult.ratePerMinute,
            mouthOpenDurationMs = if (mar != null && mouthResult.initialized) mouthResult.currentDurationMs else null,
            lastMouthOpenDurationMs = mouthResult.lastEventDurationMs,
            mouthObservedSec = mouthResult.observedSeconds,
            pitchDeg = normalizedAngles?.get(0),
            yawDeg = normalizedAngles?.get(1),
            rollDeg = normalizedAngles?.get(2),
            overAngle = overAngle,
            overAnglePct = overPercent,
            nodding = if (nodState == null) null else (nodState == 1),
            nodDetected = nodResult.eventDone,
            nodDurationMs = if (nodState != null) nodResult.currentDurationMs else null,
            lastNodDurationMs = nodResult.lastEventDurationMs,
            noddingFrequencyPerMin = nodResult.ratePerMinute,
            headObservedSec = headSeconds
        )
    }

    private fun normalizeAngle(diff: Double): Double {
        // (diff + 180) % 360 - 180 with Euclidean modulo
        val shifted = diff + 180.0
        val mod = ((shifted % 360.0) + 360.0) % 360.0
        return mod - 180.0
    }

    private fun median(values: List<Double>): Double {
        val sorted = values.sorted()
        val n = sorted.size
        return if (n % 2 == 1) {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        }
    }
}
