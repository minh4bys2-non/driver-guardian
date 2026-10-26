package com.example.driverguardian.ai.pipeline

import com.example.driverguardian.ai.physical.geometry.AspectRatioCalculator
import com.example.driverguardian.ai.physical.geometry.HeadPoseEstimator
import com.example.driverguardian.ai.physical.model.HeadPose
import com.example.driverguardian.ai.physical.model.LandmarkPoint
import com.example.driverguardian.ai.physical.model.PhysicalMetrics
import com.example.driverguardian.ai.physical.temporal.AdaptiveHmmFsm
import com.example.driverguardian.ai.physical.temporal.LazyHeadMotionWindow
import com.example.driverguardian.ai.physical.temporal.PitchSamplesSnapshot
import com.example.driverguardian.ai.physical.temporal.TemporalStateMachine
import com.example.driverguardian.ai.physical.temporal.WindowRatio
import kotlin.math.abs

/**
 * Production implementation of [PhysicalBranch] (`UnifiedCameraMetrics` in Snapdragon 680 v2 Spec Section 4.7)
 * coordinating EAR/MAR geometry, adaptive HMM/FSM detectors, head pose estimation with unwrapped neutral calibration,
 * nod hysteresis detection, O(1) over-angle ratio tracking, and 30Hz/1Hz LazyHeadMotionWindow.
 */
class PhysicalBranchProcessor(
    val config: PhysicalBranchConfig = PhysicalBranchConfig()
) : PhysicalBranch {

    private val physCfg = config.physicalConfig

    val eye = AdaptiveHmmFsm(
        mode = "eye",
        fps = config.fps,
        initDurationSec = config.eyeInitDurationSec,
        windowSizeSec = config.windowSec,
        maxGapSec = config.maxGapSec,
        learningRate = config.learningRate,
        adaptInterval = config.adaptInterval,
        config = physCfg
    )

    val mouth = AdaptiveHmmFsm(
        mode = "mouth",
        fps = config.fps,
        initDurationSec = config.mouthInitDurationSec,
        windowSizeSec = config.windowSec,
        maxGapSec = config.maxGapSec,
        learningRate = config.learningRate,
        adaptInterval = config.adaptInterval,
        config = physCfg
    )

    val nod = TemporalStateMachine(
        mode = "pitch",
        fps = config.fps,
        windowSizeSec = config.windowSec,
        minDurationMs = config.nodDurationMinMs,
        maxDurationMs = config.nodDurationMaxMs,
        maxGapSec = config.maxGapSec,
        config = physCfg
    )

    val overAngleRatio = WindowRatio(
        windowSec = config.windowSec,
        maxGapSec = config.maxGapSec,
        config = physCfg
    )

    val headMotion = LazyHeadMotionWindow(
        window = config.windowSec,
        maxGap = config.maxGapSec,
        minAmplitude = config.headMotionMinAmplitudeDeg,
        config = physCfg
    )

    val blinkDeque = ArrayDeque<Pair<Double, Double>>()
    val nodDeque = ArrayDeque<Pair<Double, Double>>()

    private var currentShape: Pair<Int, Int>? = null
    private var estimator: HeadPoseEstimator? = null

    private val neutralSamples = ArrayList<HeadPose>(config.calibrationFrames)
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
        headMotion.reset()
        blinkDeque.clear()
        nodDeque.clear()
        neutralSamples.clear()
        neutralPose = null
        nodding = false
        lastTimestampSec = null
    }

    override fun snapshotPitchSamplesForFft(): PitchSamplesSnapshot? = headMotion.snapshotForFft()

    override fun snapshotBlinkDurations(): List<Pair<Double, Double>> = blinkDeque.toList()

    override fun snapshotNodDurations(): List<Pair<Double, Double>> = nodDeque.toList()

    override fun updateCachedFftResult(frequencyHz: Double?, resolutionHz: Double?) {
        headMotion.cachedFreqHz = frequencyHz
        headMotion.cachedResHz = resolutionHz
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
            neutralSamples.clear()
        }
        lastTimestampSec = timestampSec

        val newShape = imageWidth to imageHeight
        if (currentShape != newShape) {
            estimator = HeadPoseEstimator(imageWidth, imageHeight, physCfg)
            currentShape = newShape
            neutralSamples.clear()
            neutralPose = null
            nodding = false
            nod.reset()
            overAngleRatio.reset()
            headMotion.reset()
        }

        val validLandmarks = if (
            landmarks != null &&
            landmarks.size >= physCfg.minLandmarkCount &&
            landmarks.all { it.x.isFinite() && it.y.isFinite() }
        ) {
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
            ear = AspectRatioCalculator.calculateEar(validLandmarks, physCfg)
            mar = AspectRatioCalculator.calculateMar(validLandmarks, physCfg)

            try {
                val pose6 = physCfg.poseLandmarkIndices.map { idx -> validLandmarks[idx] }
                val estimated = estimator!!.estimate(pose6)
                rawPose = estimated

                if (neutralPose == null) {
                    neutralSamples.add(estimated)
                    if (neutralSamples.size >= config.calibrationFrames) {
                        val medPitch = computeUnwrappedMedianNeutral(neutralSamples.map { it.pitchDeg })
                        val medYaw = computeUnwrappedMedianNeutral(neutralSamples.map { it.yawDeg })
                        val medRoll = computeUnwrappedMedianNeutral(neutralSamples.map { it.rollDeg })
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

        val wasEyeInit = eye.initialized
        val wasMouthInit = mouth.initialized

        val eyeResult = eye.process(ear, timestampSec)
        val mouthResult = mouth.process(mar, timestampSec)
        val hmmJustInitialized = (!wasEyeInit && eye.initialized) || (!wasMouthInit && mouth.initialized)

        var overAngle: Boolean? = null
        var nodState: Int? = null

        if (normalizedAngles != null) {
            val isOver = abs(normalizedAngles[0]) > config.angleLimits[0] ||
                abs(normalizedAngles[1]) > config.angleLimits[1] ||
                abs(normalizedAngles[2]) > config.angleLimits[2]
            overAngle = isOver

            val pitchDown = if (config.nodDirection == null) {
                abs(normalizedAngles[0])
            } else {
                config.nodDirection * normalizedAngles[0]
            }
            val threshold = if (nodding) config.nodReleaseDeg else config.nodPitchDeg
            nodding = pitchDown > threshold
            nodState = if (nodding) 1 else 0
        } else {
            nodding = false
        }

        val nodResult = nod.process(nodState, timestampSec)
        val (overPercent, headSeconds) = overAngleRatio.process(overAngle, timestampSec)
        val motionResult = headMotion.recordFrame(normalizedAngles?.get(0), timestampSec)

        // Cập nhật lịch sử thời lượng chớp mắt & gật đầu trong cửa sổ 60s
        if (eyeResult.eventDone && eyeResult.lastValidDurationMs != null) {
            blinkDeque.addLast(timestampSec to eyeResult.lastValidDurationMs)
        }
        while (blinkDeque.isNotEmpty() && blinkDeque.first().first <= timestampSec - config.windowSec) {
            blinkDeque.removeFirst()
        }

        if (nodResult.eventDone && nodResult.lastValidDurationMs != null) {
            nodDeque.addLast(timestampSec to nodResult.lastValidDurationMs)
        }
        while (nodDeque.isNotEmpty() && nodDeque.first().first <= timestampSec - config.windowSec) {
            nodDeque.removeFirst()
        }

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
            blinkCount = eyeResult.eventCount,
            blinkDurationMs = eyeResult.lastValidDurationMs,
            mar = mar,
            mouthState = mouthResult.state,
            pomPct = mouthResult.pom,
            yawnDetected = mouthResult.eventDone,
            yawningFrequencyPerMin = mouthResult.ratePerMinute,
            mouthOpenDurationMs = if (mar != null && mouthResult.initialized) mouthResult.currentDurationMs else null,
            lastMouthOpenDurationMs = mouthResult.lastEventDurationMs,
            mouthObservedSec = mouthResult.observedSeconds,
            yawnCount = mouthResult.eventCount,
            yawnDurationMs = mouthResult.lastValidDurationMs,
            pitchDeg = normalizedAngles?.get(0),
            yawDeg = normalizedAngles?.get(1),
            rollDeg = normalizedAngles?.get(2),
            pitchAmplitudeDeg = normalizedAngles?.get(0)?.let { abs(it) },
            overAngle = overAngle,
            overAnglePct = overPercent,
            nodding = if (nodState == null) null else (nodState == 1),
            nodDetected = nodResult.eventDone,
            nodDurationMs = if (nodState != null) nodResult.currentDurationMs else null,
            lastNodDurationMs = nodResult.lastEventDurationMs,
            noddingFrequencyPerMin = nodResult.ratePerMinute,
            headObservedSec = headSeconds,
            nodCount = nodResult.eventCount,
            nodEventDurationMs = nodResult.lastValidDurationMs,
            pitchMeanDeg = motionResult.pitchMeanDeg,
            pitchMeanAmplitudeDeg = motionResult.pitchMeanAmplitudeDeg,
            headMotionFrequencyHz = motionResult.headMotionFrequencyHz,
            headMotionResolutionHz = motionResult.headMotionResolutionHz,
            headMotionObservedSec = motionResult.headMotionObservedSec,
            windowSec = config.windowSec,
            hmmJustInitialized = hmmJustInitialized
        )
    }

    /**
     * Mở cuộn pha theo chu kỳ 360° (`np.unwrap(samples, period=360)`) trước khi lấy median
     * rồi chuẩn hóa về [-180°, 180°).
     */
    private fun computeUnwrappedMedianNeutral(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val unwrapped = DoubleArray(values.size)
        unwrapped[0] = values[0]
        for (i in 1 until values.size) {
            val diff = values[i] - values[i - 1]
            val wrappedDiff = normalizeAngle(diff)
            unwrapped[i] = unwrapped[i - 1] + wrappedDiff
        }
        val med = median(unwrapped.toList())
        return normalizeAngle(med)
    }

    private fun normalizeAngle(diff: Double): Double {
        val half = physCfg.angleWrapHalfPeriodDeg
        val period = physCfg.angleWrapPeriodDeg
        val shifted = diff + half
        val mod = ((shifted % period) + period) % period
        return mod - half
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
