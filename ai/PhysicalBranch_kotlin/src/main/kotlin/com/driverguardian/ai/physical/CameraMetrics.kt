package com.driverguardian.ai.physical

import com.driverguardian.ai.AIResult
import kotlin.math.abs

/**
 * Result data class produced by CameraMetrics on each processed frame.
 */
data class CameraMetricsResult(
    val timestampSec: Double,
    val faceDetected: Boolean,
    val eyeReady: Boolean,
    val mouthReady: Boolean,
    val headReady: Boolean,
    val headCalibrated: Boolean,
    val poseValid: Boolean,
    val poseError: String? = null,

    val eyeState: Int? = null,
    val mouthState: Int? = null,

    val ear: Double? = null,
    val perclosPct: Double? = null,
    val p80EarThreshold: Double? = null,
    val blinkRatePerMin: Double? = null,
    val blinkDetected: Boolean = false,
    val eyeClosureDurationMs: Double? = null,
    val lastEyeClosureDurationMs: Double = 0.0,
    val eyeObservedSec: Double = 0.0,

    val mar: Double? = null,
    val pomPct: Double? = null,
    val yawningFrequencyPerMin: Double? = null,
    val yawnDetected: Boolean = false,
    val mouthOpenDurationMs: Double? = null,
    val lastMouthOpenDurationMs: Double = 0.0,
    val mouthObservedSec: Double = 0.0,

    val pitchDeg: Double? = null,
    val yawDeg: Double? = null,
    val rollDeg: Double? = null,

    val overAngle: Boolean? = null,
    val overAnglePct: Double? = null,
    val nodding: Boolean? = null,
    val nodDetected: Boolean = false,
    val noddingFrequencyPerMin: Double = 0.0,
    val nodDurationMs: Double? = null,
    val lastNodDurationMs: Double = 0.0,
    val headObservedSec: Double = 0.0,

    val processingMs: Double? = null,
    val fps: Double? = null
) {
    /**
     * Converts CameraMetricsResult into the standardized AIResult interface model.
     */
    fun toAIResult(
        windowSec: Double = 60.0,
        lstmDrowsinessProbability: Double? = null,
        warningScore: Double? = null
    ): AIResult = AIResult(
        timestampSec = timestampSec,
        windowSec = windowSec,
        ear = ear,
        blinkRatePerMin = blinkRatePerMin,
        eyeClosureDurationMs = eyeClosureDurationMs,
        perclosPct = perclosPct,
        mar = mar,
        mouthOpenDurationMs = mouthOpenDurationMs,
        pitchDeg = pitchDeg,
        pitchDeviationDurationMs = nodDurationMs,
        pitchDeviationRatePerMin = noddingFrequencyPerMin,
        lstmDrowsinessProbability = lstmDrowsinessProbability,
        warningScore = warningScore
    )
}

/**
 * Camera metrics coordinator calculating EAR, MAR, head pose angles, PERCLOS, POM,
 * blink rate, yawn frequency, and nodding dynamics from facial landmarks.
 *
 * Corresponds to CameraMetrics in ai/PhysicalBranch/camera_metrics.py.
 */
class CameraMetrics(
    val fps: Double = 30.0,
    val windowSec: Double = 60.0,
    val calibrationFrames: Int = 30,
    val angleLimits: DoubleArray = doubleArrayOf(20.0, 25.0, 20.0),
    val nodPitchDeg: Double = 14.0,
    val nodReleaseDeg: Double = 8.0,
    val nodDirection: Int = 1,
    val nodDurationMs: Pair<Double, Double> = Pair(800.0, 3500.0),
    val maxGapSec: Double = 0.25,
    val detectionConfidence: Double = 0.6,
    val trackingConfidence: Double = 0.6,
    val eyeInitDurationSec: Double = 5.0,
    val eyeMinDurationMs: Double = 100.0,
    val eyeMaxDurationMs: Double = 2000.0,
    val mouthInitDurationSec: Double = 5.0,
    val mouthMinDurationMs: Double = 3500.0,
    val mouthMaxDurationMs: Double = 7500.0,
    val learningRate: Double = 0.01,
    val adaptInterval: Int = 300
) {
    companion object {
        val LEFT_EYE = intArrayOf(362, 385, 387, 263, 373, 380)
        val RIGHT_EYE = intArrayOf(33, 160, 158, 133, 153, 144)
        val MOUTH = intArrayOf(61, 37, 267, 291, 314, 84)

        /**
         * Computes the geometric aspect ratio from landmark points.
         * Used for EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio).
         *
         * ratio = (||p1 - p5|| + ||p2 - p4||) / (2 * ||p0 - p3||)
         */
        fun aspectRatio(points: Array<Point2D>, indices: IntArray): Double? {
            if (points.size <= indices.max()) return null
            val p0 = points[indices[0]]
            val p1 = points[indices[1]]
            val p2 = points[indices[2]]
            val p3 = points[indices[3]]
            val p4 = points[indices[4]]
            val p5 = points[indices[5]]

            val width = p0.distanceTo(p3)
            if (!p0.x.isFinite() || !p3.x.isFinite() || width < 1e-6) {
                return null
            }
            val h1 = p1.distanceTo(p5)
            val h2 = p2.distanceTo(p4)
            return (h1 + h2) / (2.0 * width)
        }
    }

    val eye: AdaptiveHmmFsm
    val mouth: AdaptiveHmmFsm
    val nod: StateMachine
    val overAngleRatio: WindowRatio

    val neutralSamples: MutableList<HeadPoseAngles> = mutableListOf()
    var neutral: HeadPoseAngles? = null
    var nodding: Boolean = false
    var timestamp: Double? = null

    var currentWidth: Double = 0.0
    var currentHeight: Double = 0.0
    var estimator: HeadPoseEstimator? = null

    init {
        require(calibrationFrames >= 1) { "calibration_frames must be >= 1" }
        require(nodDirection == -1 || nodDirection == 1) { "nod_direction must be -1 or 1" }
        require(nodReleaseDeg in 0.0..<nodPitchDeg) { "nod_release must be below nod_pitch" }
        require(angleLimits.size == 3 && angleLimits.all { it > 0 && it.isFinite() }) {
            "Angle limits must contain three positive finite values"
        }

        eye = AdaptiveHmmFsm(
            mode = FsmMode.EYE,
            fps = fps,
            initDurationSec = eyeInitDurationSec,
            windowSizeSec = windowSec,
            minDurationMs = eyeMinDurationMs,
            maxDurationMs = eyeMaxDurationMs,
            maxGapSec = maxGapSec,
            learningRate = learningRate,
            adaptInterval = adaptInterval
        )

        mouth = AdaptiveHmmFsm(
            mode = FsmMode.MOUTH,
            fps = fps,
            initDurationSec = mouthInitDurationSec,
            windowSizeSec = windowSec,
            minDurationMs = mouthMinDurationMs,
            maxDurationMs = mouthMaxDurationMs,
            maxGapSec = maxGapSec,
            learningRate = learningRate,
            adaptInterval = adaptInterval
        )

        nod = StateMachine(
            mode = FsmMode.PITCH,
            fps = fps,
            windowSizeSec = windowSec,
            minDurationMs = nodDurationMs.first,
            maxDurationMs = nodDurationMs.second,
            maxGapSec = maxGapSec
        )

        overAngleRatio = WindowRatio(
            windowSec = windowSec,
            maxGapSec = maxGapSec
        )

        reset()
    }

    fun reset() {
        eye.reset()
        mouth.reset()
        nod.reset()
        overAngleRatio.reset()
        neutralSamples.clear()
        neutral = null
        nodding = false
        timestamp = null
    }

    /**
     * Processes 2D facial landmarks (e.g. 468+ points from MediaPipe FaceMesh).
     *
     * @param points Array of 2D landmark points, or null if face not detected.
     * @param imageWidth Frame width.
     * @param imageHeight Frame height.
     * @param timestampSec Current frame timestamp in seconds.
     * @return CameraMetricsResult
     */
    fun processLandmarks(
        points: Array<Point2D>?,
        imageWidth: Double,
        imageHeight: Double,
        timestampSec: Double
    ): CameraMetricsResult {
        require(timestampSec.isFinite()) { "Timestamps must be finite, got $timestampSec" }
        if (timestamp != null) {
            require(timestampSec > timestamp!!) { "Timestamps must be strictly increasing: prev=$timestamp, curr=$timestampSec" }
            if (timestampSec - timestamp!! > nod.maxGapSec) {
                nodding = false
            }
        }
        timestamp = timestampSec

        if (currentWidth != imageWidth || currentHeight != imageHeight) {
            currentWidth = imageWidth
            currentHeight = imageHeight
            estimator = DefaultHeadPoseEstimator(imageWidth, imageHeight)
            neutralSamples.clear()
            neutral = null
            nodding = false
            nod.reset()
            overAngleRatio.reset()
        }

        val validPoints = if (points != null && points.size >= 468 && points.all { it.x.isFinite() && it.y.isFinite() }) {
            points
        } else {
            null
        }

        var ear: Double? = null
        var mar: Double? = null
        var rawAngles: HeadPoseAngles? = null
        var calibratedAngles: HeadPoseAngles? = null
        var poseError: String? = null

        if (validPoints != null) {
            val left = aspectRatio(validPoints, LEFT_EYE)
            val right = aspectRatio(validPoints, RIGHT_EYE)
            ear = if (left != null && right != null) (left + right) / 2.0 else null
            mar = aspectRatio(validPoints, MOUTH)

            try {
                val poseLandmarks = Array(6) { i -> validPoints[HeadPoseConstants.LANDMARKS[i]] }
                val raw = estimator!!.estimate(poseLandmarks)
                rawAngles = raw

                if (neutral == null) {
                    neutralSamples.add(raw)
                    if (neutralSamples.size >= calibrationFrames) {
                        val pitchArr = DoubleArray(neutralSamples.size) { neutralSamples[it].pitch }
                        val yawArr = DoubleArray(neutralSamples.size) { neutralSamples[it].yaw }
                        val rollArr = DoubleArray(neutralSamples.size) { neutralSamples[it].roll }
                        neutral = HeadPoseAngles(
                            pitch = MathUtils.median(pitchArr),
                            yaw = MathUtils.median(yawArr),
                            roll = MathUtils.median(rollArr)
                        )
                        neutralSamples.clear()
                    }
                }

                if (neutral != null) {
                    calibratedAngles = HeadPoseAngles(
                        pitch = MathUtils.wrapAngle180(raw.pitch - neutral!!.pitch),
                        yaw = MathUtils.wrapAngle180(raw.yaw - neutral!!.yaw),
                        roll = MathUtils.wrapAngle180(raw.roll - neutral!!.roll)
                    )
                }
            } catch (e: Exception) {
                poseError = e.message ?: e.toString()
                neutralSamples.clear()
            }
        } else if (neutral == null) {
            neutralSamples.clear()
        }

        val eyeResult = eye.process(ear, timestampSec)
        val mouthResult = mouth.process(mar, timestampSec)

        var overAngle: Boolean? = null
        var nodState: Int? = null

        if (calibratedAngles != null) {
            overAngle = abs(calibratedAngles.pitch) > angleLimits[0] ||
                    abs(calibratedAngles.yaw) > angleLimits[1] ||
                    abs(calibratedAngles.roll) > angleLimits[2]

            val pitchDown = nodDirection * calibratedAngles.pitch
            val threshold = if (nodding) nodReleaseDeg else nodPitchDeg
            nodding = pitchDown > threshold
            nodState = if (nodding) 1 else 0
        } else {
            nodding = false
        }

        val nodResult = nod.process(nodState, timestampSec)
        val overRatioResult = overAngleRatio.process(overAngle, timestampSec)

        return CameraMetricsResult(
            timestampSec = timestampSec,
            faceDetected = validPoints != null,
            eyeReady = eyeResult.initialized,
            mouthReady = mouthResult.initialized,
            headReady = calibratedAngles != null,
            headCalibrated = neutral != null,
            poseValid = rawAngles != null,
            poseError = poseError,
            eyeState = eyeResult.state,
            mouthState = mouthResult.state,
            ear = ear,
            perclosPct = eyeResult.perclos,
            p80EarThreshold = eyeResult.p80Threshold,
            blinkRatePerMin = eyeResult.ratePerMinute,
            blinkDetected = eyeResult.eventDone,
            eyeClosureDurationMs = if (ear != null && eyeResult.initialized) eyeResult.currentDurationMs else null,
            lastEyeClosureDurationMs = eyeResult.lastEventDurationMs,
            eyeObservedSec = eyeResult.observedSeconds,
            mar = mar,
            pomPct = mouthResult.pom,
            yawningFrequencyPerMin = mouthResult.ratePerMinute,
            yawnDetected = mouthResult.eventDone,
            mouthOpenDurationMs = if (mar != null && mouthResult.initialized) mouthResult.currentDurationMs else null,
            lastMouthOpenDurationMs = mouthResult.lastEventDurationMs,
            mouthObservedSec = mouthResult.observedSeconds,
            pitchDeg = calibratedAngles?.pitch,
            yawDeg = calibratedAngles?.yaw,
            rollDeg = calibratedAngles?.roll,
            overAngle = overAngle,
            overAnglePct = overRatioResult.percentage,
            nodding = if (nodState != null) nodState == 1 else null,
            nodDetected = nodResult.eventDone,
            noddingFrequencyPerMin = nodResult.ratePerMinute,
            nodDurationMs = if (nodState != null) nodResult.currentDurationMs else null,
            lastNodDurationMs = nodResult.lastEventDurationMs,
            headObservedSec = overRatioResult.observedSeconds
        )
    }
}
