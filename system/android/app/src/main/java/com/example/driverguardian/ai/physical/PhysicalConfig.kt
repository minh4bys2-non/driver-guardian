package com.example.driverguardian.ai.physical

/**
 * Cấu hình tập trung duy nhất cho toàn bộ nhánh vật lý (`ai.physical`).
 * Tuyệt đối không hardcode các hằng số hình học, HMM, FSM, WindowRatio hay FFT rải rác trong các class.
 */
data class PhysicalConfig(
    // 1. Cấu hình Hình học EAR / MAR (AspectRatioCalculator)
    val leftEyeIndices: IntArray = DEFAULT_LEFT_EYE_INDICES.clone(),
    val rightEyeIndices: IntArray = DEFAULT_RIGHT_EYE_INDICES.clone(),
    val mouthIndices: IntArray = DEFAULT_MOUTH_INDICES.clone(),
    val minAspectWidthEpsilon: Double = DEFAULT_MIN_ASPECT_WIDTH_EPSILON,
    val aspectHeightDivisor: Double = DEFAULT_ASPECT_HEIGHT_DIVISOR,
    val minLandmarkCount: Int = DEFAULT_MIN_LANDMARK_COUNT,

    // 2. Cấu hình Giải Tư thế Đầu 3D SolvePnP (HeadPoseEstimator)
    val poseLandmarkIndices: IntArray = DEFAULT_POSE_LANDMARK_INDICES.clone(),
    val faceModel3d: Array<DoubleArray> = DEFAULT_FACE_MODEL_3D.map { it.clone() }.toTypedArray(),
    val modelEyeOuterDistance: Double = DEFAULT_MODEL_EYE_OUTER_DISTANCE,
    val degeneracyTraceEpsilon: Double = DEFAULT_DEGENERACY_TRACE_EPSILON,
    val degeneracyDetRatioEpsilon: Double = DEFAULT_DEGENERACY_DET_RATIO_EPSILON,
    val degeneracyEyeDistEpsilon: Double = DEFAULT_DEGENERACY_EYE_DIST_EPSILON,
    val lmSeeds: Array<DoubleArray> = DEFAULT_LM_SEEDS.map { it.clone() }.toTypedArray(),
    val lmInitialLambda: Double = DEFAULT_LM_INITIAL_LAMBDA,
    val lmLambdaScaleFactor: Double = DEFAULT_LM_LAMBDA_SCALE_FACTOR,
    val lmMaxLambda: Double = DEFAULT_LM_MAX_LAMBDA,
    val lmMaxIterations: Int = DEFAULT_LM_MAX_ITERATIONS,
    val lmJacobianEps: Double = DEFAULT_LM_JACOBIAN_EPS,
    val lmConvergenceTol: Double = DEFAULT_LM_CONVERGENCE_TOL,
    val lmDiagonalDampingEps: Double = DEFAULT_LM_DIAGONAL_DAMPING_EPS,
    val rodriguesMinThetaEps: Double = DEFAULT_RODRIGUES_MIN_THETA_EPS,
    val pivotMinEps: Double = DEFAULT_PIVOT_MIN_EPS,
    val safeProjectionZEps: Double = DEFAULT_SAFE_PROJECTION_Z_EPS,

    // 3. Cấu hình AdaptiveHMM & AdaptiveHMM_FSM
    val hmmStates: Int = DEFAULT_HMM_STATES,
    val hmmLearningRate: Double = DEFAULT_HMM_LEARNING_RATE,
    val hmmAdaptInterval: Int = DEFAULT_HMM_ADAPT_INTERVAL,
    val hmmMinVariance: Double = DEFAULT_HMM_MIN_VARIANCE,
    val hmmEmIterations: Int = DEFAULT_HMM_EM_ITERATIONS,
    val hmmKMeansIterations: Int = DEFAULT_HMM_KMEANS_ITERATIONS,
    val hmmKMeansConvergenceTol: Double = DEFAULT_HMM_KMEANS_CONVERGENCE_TOL,
    val hmmTransitionSmoothing: Double = DEFAULT_HMM_TRANSITION_SMOOTHING,
    val hmmMinAdaptStateWeight: Double = DEFAULT_HMM_MIN_ADAPT_STATE_WEIGHT,
    val hmmLogFloor: Double = DEFAULT_HMM_LOG_FLOOR,
    val hmmWeightFloor: Double = DEFAULT_HMM_WEIGHT_FLOOR,
    val initDurationSec: Double = DEFAULT_INIT_DURATION_SEC,
    val minInitFrames: Int = DEFAULT_MIN_INIT_FRAMES,
    val initLowPercentile: Double = DEFAULT_INIT_LOW_PERCENTILE,
    val initHighPercentile: Double = DEFAULT_INIT_HIGH_PERCENTILE,
    val eyeSeparation: Double = DEFAULT_EYE_SEPARATION,
    val mouthSeparation: Double = DEFAULT_MOUTH_SEPARATION,
    val pitchSeparation: Double = DEFAULT_PITCH_SEPARATION,
    val eyeBaselinePercentile: Double = DEFAULT_EYE_BASELINE_PERCENTILE,
    val otherBaselinePercentile: Double = DEFAULT_OTHER_BASELINE_PERCENTILE,
    val eyeDistinctRatio: Double = DEFAULT_EYE_DISTINCT_RATIO,
    val eyeOpenRatio: Double = DEFAULT_EYE_OPEN_RATIO,
    val eyeClosedRatio: Double = DEFAULT_EYE_CLOSED_RATIO,
    val eventSeparationMultiplier: Double = DEFAULT_EVENT_SEPARATION_MULTIPLIER,
    val fallbackSigmaDivisor: Double = DEFAULT_FALLBACK_SIGMA_DIVISOR,
    val fallbackMinSigma: Double = DEFAULT_FALLBACK_MIN_SIGMA,
    val fallbackTransition: Array<DoubleArray> = DEFAULT_FALLBACK_TRANSITION.map { it.clone() }
        .toTypedArray(),
    val restingPrior: DoubleArray = DEFAULT_RESTING_PRIOR.clone(),
    val perclosClosedFraction: Double = DEFAULT_PERCLOS_CLOSED_FRACTION,

    // 4. Cấu hình TemporalStateMachine (FSM)
    val defaultFps: Int = DEFAULT_FPS,
    val windowSec: Double = DEFAULT_WINDOW_SEC,
    val maxGapSec: Double = DEFAULT_MAX_GAP_SEC,
    val blinkDurationMinMs: Double = DEFAULT_BLINK_DURATION_MIN_MS,
    val blinkDurationMaxMs: Double = DEFAULT_BLINK_DURATION_MAX_MS,
    val yawnDurationMinMs: Double = DEFAULT_YAWN_DURATION_MIN_MS,
    val yawnDurationMaxMs: Double = DEFAULT_YAWN_DURATION_MAX_MS,
    val nodDurationMinMs: Double = DEFAULT_NOD_DURATION_MIN_MS,
    val nodDurationMaxMs: Double = DEFAULT_NOD_DURATION_MAX_MS,
    val durationBoundaryEpsilonMs: Double = DEFAULT_DURATION_BOUNDARY_EPSILON_MS,
    val secondsPerMinute: Double = DEFAULT_SECONDS_PER_MINUTE,
    val millisecondsPerSecond: Double = DEFAULT_MILLISECONDS_PER_SECOND,

    // 5. Cấu hình RunningWindowRatio O(1) & LazyHeadMotionWindow (1Hz FFT)
    val windowRatioInitialCapacity: Int = DEFAULT_WINDOW_RATIO_INITIAL_CAPACITY,
    val windowRatioRecalcIntervalSteps: Int = DEFAULT_WINDOW_RATIO_RECALC_INTERVAL_STEPS,
    val windowRatioMinValidEpsilon: Double = DEFAULT_WINDOW_RATIO_MIN_VALID_EPSILON,
    val percentageScale: Double = DEFAULT_PERCENTAGE_SCALE,
    val headMotionInitialCapacity: Int = DEFAULT_HEAD_MOTION_INITIAL_CAPACITY,
    val headMotionMinAmplitudeDeg: Double = DEFAULT_HEAD_MOTION_MIN_AMPLITUDE_DEG,
    val headMotionMinSamplesForIntegral: Int = DEFAULT_HEAD_MOTION_MIN_SAMPLES_FOR_INTEGRAL,
    val headMotionMinSamplesForFft: Int = DEFAULT_HEAD_MOTION_MIN_SAMPLES_FOR_FFT,
    val headMotionWindowBoundaryEpsilon: Double = DEFAULT_HEAD_MOTION_WINDOW_BOUNDARY_EPSILON,

    // 6. Cấu hình Hiệu chuẩn Góc Đầu & Hysteresis Gật đầu
    val calibrationFrames: Int = DEFAULT_CALIBRATION_FRAMES,
    val angleLimitsDeg: DoubleArray = DEFAULT_ANGLE_LIMITS_DEG.clone(),
    val nodPitchDeg: Double = DEFAULT_NOD_PITCH_DEG,
    val nodReleaseDeg: Double = DEFAULT_NOD_RELEASE_DEG,
    val nodDirection: Int? = DEFAULT_NOD_DIRECTION,
    val angleWrapPeriodDeg: Double = DEFAULT_ANGLE_WRAP_PERIOD_DEG,
    val angleWrapHalfPeriodDeg: Double = DEFAULT_ANGLE_WRAP_HALF_PERIOD_DEG,
) {
    companion object {
        val DEFAULT_LEFT_EYE_INDICES = intArrayOf(362, 385, 387, 263, 373, 380)
        val DEFAULT_RIGHT_EYE_INDICES = intArrayOf(33, 160, 158, 133, 153, 144)
        val DEFAULT_MOUTH_INDICES = intArrayOf(61, 37, 267, 291, 314, 84)
        const val DEFAULT_MIN_ASPECT_WIDTH_EPSILON = 1e-6
        const val DEFAULT_ASPECT_HEIGHT_DIVISOR = 2.0
        const val DEFAULT_MIN_LANDMARK_COUNT = 468

        val DEFAULT_POSE_LANDMARK_INDICES = intArrayOf(1, 152, 263, 33, 291, 61)
        val DEFAULT_FACE_MODEL_3D = arrayOf(
            doubleArrayOf(0.0, 0.0, 0.0),         // 1: Nose tip
            doubleArrayOf(0.0, 330.0, 65.0),      // 152: Chin
            doubleArrayOf(225.0, -170.0, 135.0),  // 263: Left eye outer corner
            doubleArrayOf(-225.0, -170.0, 135.0), // 33: Right eye outer corner
            doubleArrayOf(150.0, 150.0, 125.0),   // 291: Left mouth corner
            doubleArrayOf(-150.0, 150.0, 125.0),  // 61: Right mouth corner
        )
        const val DEFAULT_MODEL_EYE_OUTER_DISTANCE = 450.0
        const val DEFAULT_DEGENERACY_TRACE_EPSILON = 1e-8
        const val DEFAULT_DEGENERACY_DET_RATIO_EPSILON = 1e-5
        const val DEFAULT_DEGENERACY_EYE_DIST_EPSILON = 1e-4
        val DEFAULT_LM_SEEDS = arrayOf(
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(0.2, 0.0, 0.0),
            doubleArrayOf(-0.2, 0.0, 0.0),
        )
        const val DEFAULT_LM_INITIAL_LAMBDA = 1e-3
        const val DEFAULT_LM_LAMBDA_SCALE_FACTOR = 10.0
        const val DEFAULT_LM_MAX_LAMBDA = 1e12
        const val DEFAULT_LM_MAX_ITERATIONS = 35
        const val DEFAULT_LM_JACOBIAN_EPS = 1e-6
        const val DEFAULT_LM_CONVERGENCE_TOL = 1e-8
        const val DEFAULT_LM_DIAGONAL_DAMPING_EPS = 1e-6
        const val DEFAULT_RODRIGUES_MIN_THETA_EPS = 1e-12
        const val DEFAULT_PIVOT_MIN_EPS = 1e-14
        const val DEFAULT_SAFE_PROJECTION_Z_EPS = 1e-6

        const val DEFAULT_HMM_STATES = 2
        const val DEFAULT_HMM_LEARNING_RATE = 0.01
        const val DEFAULT_HMM_ADAPT_INTERVAL = 300
        const val DEFAULT_HMM_MIN_VARIANCE = 1e-5
        const val DEFAULT_HMM_EM_ITERATIONS = 20
        const val DEFAULT_HMM_KMEANS_ITERATIONS = 25
        const val DEFAULT_HMM_KMEANS_CONVERGENCE_TOL = 1e-6
        const val DEFAULT_HMM_TRANSITION_SMOOTHING = 1e-3
        const val DEFAULT_HMM_MIN_ADAPT_STATE_WEIGHT = 2.0
        const val DEFAULT_HMM_LOG_FLOOR = 1e-300
        const val DEFAULT_HMM_WEIGHT_FLOOR = 1e-12
        const val DEFAULT_INIT_DURATION_SEC = 10.0
        const val DEFAULT_MIN_INIT_FRAMES = 4
        const val DEFAULT_INIT_LOW_PERCENTILE = 5.0
        const val DEFAULT_INIT_HIGH_PERCENTILE = 95.0
        const val DEFAULT_EYE_SEPARATION = 0.06
        const val DEFAULT_MOUTH_SEPARATION = 0.15
        const val DEFAULT_PITCH_SEPARATION = 10.0
        const val DEFAULT_EYE_BASELINE_PERCENTILE = 80.0
        const val DEFAULT_OTHER_BASELINE_PERCENTILE = 20.0
        const val DEFAULT_EYE_DISTINCT_RATIO = 0.65
        const val DEFAULT_EYE_OPEN_RATIO = 0.75
        const val DEFAULT_EYE_CLOSED_RATIO = 0.20
        const val DEFAULT_EVENT_SEPARATION_MULTIPLIER = 2.0
        const val DEFAULT_FALLBACK_SIGMA_DIVISOR = 4.0
        const val DEFAULT_FALLBACK_MIN_SIGMA = 0.01
        val DEFAULT_FALLBACK_TRANSITION = arrayOf(
            doubleArrayOf(0.97, 0.03),
            doubleArrayOf(0.10, 0.90),
        )
        val DEFAULT_RESTING_PRIOR = doubleArrayOf(0.99, 0.01)
        const val DEFAULT_PERCLOS_CLOSED_FRACTION = 0.80

        const val DEFAULT_FPS = 30
        const val DEFAULT_WINDOW_SEC = 60.0
        const val DEFAULT_MAX_GAP_SEC = 0.5
        const val DEFAULT_BLINK_DURATION_MIN_MS = 100.0
        const val DEFAULT_BLINK_DURATION_MAX_MS = 2000.0
        const val DEFAULT_YAWN_DURATION_MIN_MS = 4000.0
        const val DEFAULT_YAWN_DURATION_MAX_MS = 7500.0
        const val DEFAULT_NOD_DURATION_MIN_MS = 800.0
        const val DEFAULT_NOD_DURATION_MAX_MS = 3500.0
        const val DEFAULT_DURATION_BOUNDARY_EPSILON_MS = 1e-6
        const val DEFAULT_SECONDS_PER_MINUTE = 60.0
        const val DEFAULT_MILLISECONDS_PER_SECOND = 1000.0
        const val DEFAULT_WINDOW_RATIO_INITIAL_CAPACITY = 2048
        const val DEFAULT_WINDOW_RATIO_RECALC_INTERVAL_STEPS = 1800
        const val DEFAULT_WINDOW_RATIO_MIN_VALID_EPSILON = 1e-12
        const val DEFAULT_PERCENTAGE_SCALE = 100.0
        const val DEFAULT_HEAD_MOTION_INITIAL_CAPACITY = 2048
        const val DEFAULT_HEAD_MOTION_MIN_AMPLITUDE_DEG = 0.1
        const val DEFAULT_HEAD_MOTION_MIN_SAMPLES_FOR_INTEGRAL = 2
        const val DEFAULT_HEAD_MOTION_MIN_SAMPLES_FOR_FFT = 4
        const val DEFAULT_HEAD_MOTION_WINDOW_BOUNDARY_EPSILON = 1e-9

        const val DEFAULT_CALIBRATION_FRAMES = 30
        val DEFAULT_ANGLE_LIMITS_DEG = doubleArrayOf(20.0, 25.0, 20.0)
        const val DEFAULT_NOD_PITCH_DEG = 14.0
        const val DEFAULT_NOD_RELEASE_DEG = 8.0
        val DEFAULT_NOD_DIRECTION: Int? = 1
        const val DEFAULT_ANGLE_WRAP_PERIOD_DEG = 360.0
        const val DEFAULT_ANGLE_WRAP_HALF_PERIOD_DEG = 180.0

        val DEFAULT = PhysicalConfig()
    }
}
