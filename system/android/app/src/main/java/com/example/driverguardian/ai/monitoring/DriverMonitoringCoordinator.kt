package com.example.driverguardian.ai.monitoring

import com.example.driverguardian.ai.fusion.PipelineStatus
import com.example.driverguardian.ai.fusion.WindowEvaluationEngine
import com.example.driverguardian.ai.fusion.WindowEvaluationSnapshot
import com.example.driverguardian.ai.monitoring.device.DeviceCalibrationState
import com.example.driverguardian.ai.monitoring.device.DeviceMotionProcessor
import com.example.driverguardian.ai.monitoring.device.DeviceOrientation
import com.example.driverguardian.ai.monitoring.model.CameraRuntimeState
import com.example.driverguardian.ai.monitoring.model.FaceRuntimeState
import com.example.driverguardian.ai.monitoring.model.MonitoringDiagnostics
import com.example.driverguardian.ai.monitoring.model.MonitoringRuntimeState
import com.example.driverguardian.ai.monitoring.model.MonitoringSnapshot
import com.example.driverguardian.ai.monitoring.model.SensorRuntimeState
import com.example.driverguardian.ai.monitoring.sync.DeviceMotionHistory
import com.example.driverguardian.ai.monitoring.sync.MonitoringSyncConfig
import com.example.driverguardian.ai.monitoring.time.CameraTimestampSource
import com.example.driverguardian.ai.monitoring.time.MonitoringTimebaseNormalizer
import com.example.driverguardian.ai.neural.NeuralBuffer
import com.example.driverguardian.ai.neural.NeuralClassifier
import com.example.driverguardian.ai.neural.NeuralWindowSnapshot
import com.example.driverguardian.ai.physical.model.LandmarkPoint
import com.example.driverguardian.ai.physical.model.PhysicalMetrics
import com.example.driverguardian.ai.physical.temporal.PitchSamplesSnapshot
import com.example.driverguardian.ai.pipeline.PhysicalBranch
import com.example.driverguardian.ai.pipeline.PhysicalBranchProcessor
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Snapshot payload handed off lock-free from Thread 1B (`UnifiedPhysicalThread`)
 * to Thread 2 (`HeavyComputeWorker`) once every 1.0s.
 */
data class HeavyEvaluationRequest(
    val generation: Long,
    val nowSec: Double,
    val elapsedSec: Double,
    val physicalMetrics: PhysicalMetrics,
    val blinkDurationsMs: DoubleArray,
    val nodDurationsMs: DoubleArray,
    val pitchSnapshot: PitchSamplesSnapshot?,
    val neuralSnapshot: NeuralWindowSnapshot,
)

class DriverMonitoringCoordinator(
    private val physicalBranch: PhysicalBranch = PhysicalBranchProcessor(),
    private val deviceProcessor: DeviceMotionProcessor = DeviceMotionProcessor(),
    cameraTimestampSource: CameraTimestampSource,
    private val syncConfig: MonitoringSyncConfig = MonitoringSyncConfig(),
    private val neuralBuffer: NeuralBuffer = NeuralBuffer(),
    private val windowEvaluationEngine: WindowEvaluationEngine = WindowEvaluationEngine(),
) {
    private val timebase = MonitoringTimebaseNormalizer(cameraTimestampSource)
    private val deviceHistory = DeviceMotionHistory()
    private val _snapshot = MutableStateFlow<MonitoringSnapshot?>(null)
    private val _runtimeState = MutableStateFlow(MonitoringRuntimeState())
    private val _diagnostics = MutableStateFlow(MonitoringDiagnostics(cameraTimestampSource = cameraTimestampSource))

    private var activeGeneration: Long? = null
    private var sessionOriginSec: Double? = null
    private var lastCameraTimestampSec: Double? = null
    private var lastSensorTimestampSec: Double? = null
    private var nextEvaluationSec: Double = 0.0
    private var physicalFrameCounter: Long = 0L
    private var latestWindowEvaluation: WindowEvaluationSnapshot? = null

    val snapshot: StateFlow<MonitoringSnapshot?> = _snapshot.asStateFlow()
    val runtimeState: StateFlow<MonitoringRuntimeState> = _runtimeState.asStateFlow()
    val diagnostics: StateFlow<MonitoringDiagnostics> = _diagnostics.asStateFlow()

    @Synchronized fun startSession(generation: Long) {
        activeGeneration = generation
        physicalBranch.reset()
        deviceProcessor.reset()
        deviceHistory.clear()
        neuralBuffer.reset()
        timebase.reset()
        sessionOriginSec = null
        lastCameraTimestampSec = null
        lastSensorTimestampSec = null
        nextEvaluationSec = 0.0
        physicalFrameCounter = 0L
        latestWindowEvaluation = null
        _snapshot.value = null
        _diagnostics.value = MonitoringDiagnostics(
            cameraTimestampSource = _diagnostics.value.cameraTimestampSource,
            activeDelegate = _diagnostics.value.activeDelegate,
            onnxModelVariant = _diagnostics.value.onnxModelVariant,
            balanceBrightnessEnabled = _diagnostics.value.balanceBrightnessEnabled,
        )
        _runtimeState.value = MonitoringRuntimeState(
            camera = CameraRuntimeState.STARTING,
            face = FaceRuntimeState.INITIALIZING,
            // Device motion is optional and is not part of the production camera pipeline.
            // It transitions out of IDLE only if an optional source supplies samples.
            sensor = SensorRuntimeState.IDLE,
        )
        updateTimebaseDiagnostics()
        ThreadDebugLogger.infoPhysical("Session started (generation=$generation): physical, neuralBuffer, and timebase reset")
    }

    @Synchronized fun stopSession() {
        val stoppedGen = activeGeneration
        activeGeneration = null
        _runtimeState.value = MonitoringRuntimeState()
        ThreadDebugLogger.infoPhysical("Session stopped (generation=$stoppedGen)")
    }

    @Synchronized fun updateCameraPermission(granted: Boolean) {
        _runtimeState.value = _runtimeState.value.copy(
            camera = if (granted) CameraRuntimeState.STARTING else CameraRuntimeState.PERMISSION_REQUIRED,
            message = if (granted) null else "Camera permission is required for facial monitoring",
        )
    }

    @Synchronized fun reportCameraError(generation: Long, message: String) {
        if (generation != activeGeneration) return
        _runtimeState.value = _runtimeState.value.copy(
            camera = CameraRuntimeState.ERROR,
            face = FaceRuntimeState.ERROR,
            message = message,
        )
        ThreadDebugLogger.warnIngest("Camera error (generation=$generation): $message")
    }

    @Synchronized fun reportSensorUnavailable(message: String) {
        _runtimeState.value = _runtimeState.value.copy(sensor = SensorRuntimeState.UNAVAILABLE)
    }

    @Synchronized fun reportCameraTimestampSource(source: CameraTimestampSource) {
        timebase.updateCameraTimestampSource(source)
        _diagnostics.value = _diagnostics.value.copy(cameraTimestampSource = source)
        updateTimebaseDiagnostics()
    }

    @Synchronized fun reportActiveDelegate(delegateName: String) {
        _diagnostics.value = _diagnostics.value.copy(activeDelegate = delegateName)
    }

    @Synchronized fun reportOnnxModelVariant(modelVariant: String) {
        _diagnostics.value = _diagnostics.value.copy(onnxModelVariant = modelVariant)
    }

    @Synchronized fun reportBalanceBrightness(enabled: Boolean) {
        _diagnostics.value = _diagnostics.value.copy(balanceBrightnessEnabled = enabled)
    }

    @Synchronized fun onSensorOrientation(orientation: DeviceOrientation) {
        if (activeGeneration == null) return
        val started = System.nanoTime()
        val metrics = deviceProcessor.process(orientation) ?: return
        val sensorDeltaSec = lastSensorTimestampSec?.let { orientation.timestampSec - it }
        lastSensorTimestampSec = orientation.timestampSec
        deviceHistory.add(metrics)
        _runtimeState.value = _runtimeState.value.copy(
            sensor = when (metrics.calibrationState) {
                DeviceCalibrationState.CALIBRATING -> SensorRuntimeState.CALIBRATING
                DeviceCalibrationState.AVAILABLE -> SensorRuntimeState.AVAILABLE
                DeviceCalibrationState.UNAVAILABLE -> SensorRuntimeState.UNAVAILABLE
            },
        )
        _diagnostics.value = _diagnostics.value.copy(
            sensorHz = sensorDeltaSec?.takeIf { it > 0.0 }?.let { 1.0 / it },
            sensorProcessingMs = (System.nanoTime() - started) / 1_000_000.0,
        )
    }

    /**
     * Synchronous entry point used by existing unit tests and direct callbacks.
     */
    @Synchronized fun onCameraResult(
        landmarks: List<LandmarkPoint>?,
        width: Int,
        height: Int,
        cameraTimestampNs: Long,
        receiptElapsedNs: Long,
        generation: Long,
        inferenceMs: Double? = null,
    ) {
        onCameraResultWithRgba(
            landmarks = landmarks,
            rgbaBytes = null,
            rawWidth = width,
            rawHeight = height,
            rotationDegrees = 0,
            rotatedWidth = width,
            rotatedHeight = height,
            cameraTimestampNs = cameraTimestampNs,
            receiptElapsedNs = receiptElapsedNs,
            generation = generation,
            inferenceMs = inferenceMs,
            ingestLatencyMs = null,
            onHeavyEvaluationDue = null,
        )
    }

    /**
     * Full Thread 1B (`UnifiedPhysicalThread`) processing step:
     * 1. Normalizes monotonic hardware timestamp.
     * 2. Appends RGBA frame into [NeuralBuffer] (lazy 5.0 FPS letterbox on grid).
     * 3. Runs [PhysicalBranch.processLandmarks] (EAR, MAR, SolvePnP, HMM, FSM, O(1) WindowRatio, LazyHeadMotionWindow.recordFrame).
     * 4. Applies the 1-frame phase offset at HMM initialization frame #300 (`hmmJustInitialized`).
     * 5. Checks the 1.0s cadence and emits a [HeavyEvaluationRequest] for Thread 2 (`HeavyComputeWorker`).
     */
    @Synchronized fun onCameraResultWithRgba(
        landmarks: List<LandmarkPoint>?,
        rgbaBytes: ByteArray?,
        rawWidth: Int,
        rawHeight: Int,
        rotationDegrees: Int,
        rotatedWidth: Int,
        rotatedHeight: Int,
        cameraTimestampNs: Long,
        receiptElapsedNs: Long,
        generation: Long,
        inferenceMs: Double? = null,
        ingestLatencyMs: Double? = null,
        onHeavyEvaluationDue: ((HeavyEvaluationRequest) -> Unit)? = null,
    ) {
        if (generation != activeGeneration) return
        timebase.observe(cameraTimestampNs, receiptElapsedNs)
        val normalizedNs = timebase.normalizeCameraTimestampNs(cameraTimestampNs) ?: run {
            updateTimebaseDiagnostics()
            return
        }
        var timestampSec = normalizedNs / 1_000_000_000.0
        val lastCamSec = lastCameraTimestampSec
        if (lastCamSec != null && timestampSec <= lastCamSec) {
            timestampSec = lastCamSec + MonitoringConfig.MONOTONIC_TIMESTAMP_MIN_STEP_SEC
        }
        val cameraDeltaSec = lastCamSec?.let { timestampSec - it }
        lastCameraTimestampSec = timestampSec
        if (sessionOriginSec == null) {
            sessionOriginSec = timestampSec
            nextEvaluationSec = timestampSec + MonitoringConfig.EVALUATION_INTERVAL_SEC
        }
        val elapsedSec = (timestampSec - (sessionOriginSec ?: timestampSec)).coerceAtLeast(0.0)

        // Append into NeuralBuffer (only executes 224x224 letterbox when frame lands on 5.0 FPS grid)
        var letterboxMs: Double? = null
        if (rgbaBytes != null) {
            val lbStartNs = System.nanoTime()
            val addedToGrid = neuralBuffer.append(
                rgbaBytes = rgbaBytes,
                width = rawWidth,
                height = rawHeight,
                timestampSec = timestampSec,
                rotationDegrees = rotationDegrees,
            )
            if (addedToGrid) {
                letterboxMs = (System.nanoTime() - lbStartNs) / 1_000_000.0
            }
        }

        val started = System.nanoTime()
        val physical = runCatching {
            physicalBranch.processLandmarks(landmarks, rotatedWidth, rotatedHeight, timestampSec)
        }.getOrNull() ?: return
        val processingMs = (System.nanoTime() - started) / 1_000_000.0
        physicalFrameCounter++

        // Log critical physical milestones on Thread-1B-Physical
        if (physical.hmmJustInitialized) {
            // Section 2.2: 1-frame phase offset at HMM initialization frame #300
            if (cameraDeltaSec != null && cameraDeltaSec > 0.0) {
                nextEvaluationSec += cameraDeltaSec
            }
            ThreadDebugLogger.infoPhysical(
                String.format(
                    java.util.Locale.US,
                    "HMM initialized at frame #%d (t=%.3fs, elapsed=%.2fs): P80_EAR=%.4f, P20_MAR=%.4f, nextEvalSec=%.3fs",
                    physicalFrameCounter,
                    timestampSec,
                    elapsedSec,
                    physical.p80EarThreshold ?: Double.NaN,
                    (physicalBranch as? PhysicalBranchProcessor)?.mouth?.normal ?: Double.NaN,
                    nextEvaluationSec,
                ),
            )
        }
        if (physical.blinkDetected) {
            ThreadDebugLogger.infoPhysical(
                String.format(
                    java.util.Locale.US,
                    "Blink event detected at t=%.3fs: duration=%.1fms, blinkCount=%d, blinkFreq=%.2f/min, PERCLOS=%.2f%%",
                    timestampSec,
                    physical.blinkDurationMs ?: 0.0,
                    physical.blinkCount,
                    physical.blinkFreqPerMin,
                    physical.perclosPct ?: 0.0,
                ),
            )
        }
        if (physical.yawnDetected) {
            ThreadDebugLogger.infoPhysical(
                String.format(
                    java.util.Locale.US,
                    "Yawn event detected at t=%.3fs: duration=%.1fms, yawnCount=%d, yawnFreq=%.2f/min, POM=%.2f%%",
                    timestampSec,
                    physical.yawnDurationMs ?: 0.0,
                    physical.yawnCount,
                    physical.yawnFreqPerMin,
                    physical.pomPct ?: 0.0,
                ),
            )
        }
        if (physical.nodDetected) {
            ThreadDebugLogger.infoPhysical(
                String.format(
                    java.util.Locale.US,
                    "Nod event detected at t=%.3fs: duration=%.1fms, nodCount=%d, nodFreq=%.2f/min",
                    timestampSec,
                    physical.nodEventDurationMs ?: 0.0,
                    physical.nodCount,
                    physical.nodFreqPerMin,
                ),
            )
        }
        if (physicalFrameCounter == 1L || physicalFrameCounter % MonitoringConfig.LOG_PHYSICAL_EVERY_N_FRAMES == 0L) {
            ThreadDebugLogger.logPhysical(
                String.format(
                    java.util.Locale.US,
                    "Frame #%d: face=%s, EAR=%s, MAR=%s, PERCLOS=%s, pitch=%s, physMs=%.2fms, mpMs=%s",
                    physicalFrameCounter,
                    physical.faceDetected,
                    physical.ear?.let { "%.3f".format(java.util.Locale.US, it) } ?: "null",
                    physical.mar?.let { "%.3f".format(java.util.Locale.US, it) } ?: "null",
                    physical.perclosPct?.let { "%.1f%%".format(java.util.Locale.US, it) } ?: "null",
                    physical.pitchDeg?.let { "%.1f°".format(java.util.Locale.US, it) } ?: "null",
                    processingMs,
                    inferenceMs?.let { "%.2fms".format(java.util.Locale.US, it) } ?: "n/a",
                ),
            )
        }

        // Check if a 1Hz evaluation tick is due
        // Snapdragon 680 v2 Spec Section 2.4 & 2.5:
        // Do not trigger evaluation on the exact frame HMM EM initialization ran on Thread 1B during WARMING_UP
        val skipDueToHmmInit = physical.hmmJustInitialized && elapsedSec < MonitoringConfig.WINDOW_SEC
        if (!skipDueToHmmInit && onHeavyEvaluationDue != null && timestampSec + MonitoringConfig.EVALUATION_TICK_EPSILON_SEC >= nextEvaluationSec) {
            while (nextEvaluationSec <= timestampSec + MonitoringConfig.EVALUATION_TICK_EPSILON_SEC) {
                nextEvaluationSec += MonitoringConfig.EVALUATION_INTERVAL_SEC
            }
            val cutoffSec = timestampSec - MonitoringConfig.WINDOW_SEC
            val blinkDurations = physicalBranch.snapshotBlinkDurations()
                .asSequence()
                .filter { it.first > cutoffSec }
                .map { it.second }
                .toList()
                .toDoubleArray()
            val nodDurations = physicalBranch.snapshotNodDurations()
                .asSequence()
                .filter { it.first > cutoffSec }
                .map { it.second }
                .toList()
                .toDoubleArray()
            val pitchSnapshot = physicalBranch.snapshotPitchSamplesForFft()
            val neuralSnapshot = neuralBuffer.snapshot()

            onHeavyEvaluationDue(
                HeavyEvaluationRequest(
                    generation = generation,
                    nowSec = timestampSec,
                    elapsedSec = elapsedSec,
                    physicalMetrics = physical,
                    blinkDurationsMs = blinkDurations,
                    nodDurationsMs = nodDurations,
                    pitchSnapshot = pitchSnapshot,
                    neuralSnapshot = neuralSnapshot,
                ),
            )
        }

        val currentWindowEval = latestWindowEvaluation ?: if (onHeavyEvaluationDue != null) {
            WindowEvaluationSnapshot(
                evaluatedAtSec = timestampSec,
                elapsedSec = elapsedSec,
                windowSec = MonitoringConfig.WINDOW_SEC,
                status = if (elapsedSec < MonitoringConfig.WINDOW_SEC) {
                    PipelineStatus.WARMING_UP
                } else {
                    PipelineStatus.WAITING_FOR_VALID_WINDOW
                },
                neuralBufferSize = neuralBuffer.size,
                cachedP4Count = 0,
                latestDeltaFramesCount = 0,
                latestNeuralScore = null,
                onnxInferenceMs = 0.0,
                lazyFftMs = 0.0,
                workerTotalMs = 0.0,
                onnxModelVariant = _diagnostics.value.onnxModelVariant ?: "LOADING",
                headMotionMetrics = null,
                rawFeatures = null,
                normalizedFeatures = null,
                fusionSummary = null,
                lastSkipReason = if (elapsedSec < MonitoringConfig.WINDOW_SEC) {
                    String.format(
                        java.util.Locale.US,
                        "warming up (%.1fs / %.1fs)",
                        elapsedSec,
                        MonitoringConfig.WINDOW_SEC,
                    )
                } else {
                    "awaiting 1Hz evaluation"
                },
            )
        } else {
            null
        }

        _snapshot.value = MonitoringSnapshot(
            physical = physical,
            deviceMotion = deviceHistory.associate(timestampSec, syncConfig),
            timestampSec = timestampSec,
            windowEvaluation = currentWindowEval,
        )
        _runtimeState.value = _runtimeState.value.copy(
            camera = CameraRuntimeState.AVAILABLE,
            face = if (physical.faceDetected) FaceRuntimeState.FACE_DETECTED else FaceRuntimeState.NO_FACE,
        )
        _diagnostics.value = _diagnostics.value.copy(
            cameraFps = cameraDeltaSec?.takeIf { it > 0.0 }?.let { 1.0 / it },
            inferenceMs = inferenceMs,
            physicalProcessingMs = processingMs,
            ingestMs = ingestLatencyMs ?: _diagnostics.value.ingestMs,
            letterboxMs = letterboxMs ?: _diagnostics.value.letterboxMs,
        )
        updateTimebaseDiagnostics()
    }

    /**
     * Executed on Thread 2 (`HeavyComputeWorker`) to run the 1Hz window evaluation
     * (Lazy FFT 1Hz + incremental ONNX `p4_old` + `physical_v1` normalization + 3-method `Fitness`).
     */
    fun runHeavyEvaluation(
        request: HeavyEvaluationRequest,
        neuralClassifier: NeuralClassifier?,
    ): WindowEvaluationSnapshot? {
        synchronized(this) {
            if (request.generation != activeGeneration) return null
        }

        val evalSnapshot = windowEvaluationEngine.evaluateWindow(
            nowSec = request.nowSec,
            elapsedSec = request.elapsedSec,
            physicalMetrics = request.physicalMetrics,
            blinkDurationsMs = request.blinkDurationsMs,
            nodDurationsMs = request.nodDurationsMs,
            pitchSnapshot = request.pitchSnapshot,
            neuralSnapshot = request.neuralSnapshot,
            neuralClassifier = neuralClassifier,
        )

        val fusion = evalSnapshot.fusionSummary
        if (fusion != null) {
            ThreadDebugLogger.infoHeavyWorker(
                String.format(
                    java.util.Locale.US,
                    "1Hz Eval [status=%s, elapsed=%.1fs]: primary=%s (score=%.3f, pred=%d), " +
                        "weighted_mean=%.3f, noisy_or=%.3f, cooccurrence=%.3f, neural=%.3f, fftFreq=%.2fHz, " +
                        "fftMs=%.2fms, onnxMs=%.2fms, totalMs=%.2fms",
                    evalSnapshot.status.displayLabel,
                    evalSnapshot.elapsedSec,
                    fusion.primaryFusion,
                    fusion.primaryScore,
                    fusion.primaryPrediction,
                    fusion.fusionDetails["weighted_mean"]?.score ?: 0.0,
                    fusion.fusionDetails["noisy_or"]?.score ?: 0.0,
                    fusion.fusionDetails["cooccurrence"]?.score ?: 0.0,
                    evalSnapshot.latestNeuralScore ?: 0.0,
                    evalSnapshot.headMotionMetrics?.frequency ?: 0.0,
                    evalSnapshot.lazyFftMs,
                    evalSnapshot.onnxInferenceMs,
                    evalSnapshot.workerTotalMs,
                ),
            )
        } else {
            ThreadDebugLogger.logHeavyWorker(
                String.format(
                    java.util.Locale.US,
                    "1Hz Eval [status=%s, elapsed=%.1fs]: skipReason='%s', neuralBuf=%d, p4Cached=%d, " +
                        "deltaFrames=%d, neuralScore=%s, onnxMs=%.2fms, totalMs=%.2fms",
                    evalSnapshot.status.displayLabel,
                    evalSnapshot.elapsedSec,
                    evalSnapshot.lastSkipReason ?: "none",
                    evalSnapshot.neuralBufferSize,
                    evalSnapshot.cachedP4Count,
                    evalSnapshot.latestDeltaFramesCount,
                    evalSnapshot.latestNeuralScore?.let { "%.3f".format(java.util.Locale.US, it) } ?: "null",
                    evalSnapshot.onnxInferenceMs,
                    evalSnapshot.workerTotalMs,
                ),
            )
        }

        synchronized(this) {
            if (request.generation != activeGeneration) return null
            latestWindowEvaluation = evalSnapshot
            evalSnapshot.headMotionMetrics?.let { hm ->
                physicalBranch.updateCachedFftResult(hm.frequency, hm.resolution)
            }
            val currentSnap = _snapshot.value
            if (currentSnap != null) {
                val updatedPhysical = currentSnap.physical?.let { phys ->
                    val hm = evalSnapshot.headMotionMetrics
                    if (hm != null) {
                        phys.copy(
                            headMotionFrequencyHz = hm.frequency,
                            headMotionResolutionHz = hm.resolution,
                            headMotionObservedSec = hm.observed,
                        )
                    } else {
                        phys
                    }
                }
                _snapshot.value = currentSnap.copy(
                    physical = updatedPhysical,
                    windowEvaluation = evalSnapshot,
                )
            }
            _diagnostics.value = _diagnostics.value.copy(
                lazyFftMs = evalSnapshot.lazyFftMs,
                onnxInferenceMs = evalSnapshot.onnxInferenceMs,
                workerTotalMs = evalSnapshot.workerTotalMs,
                onnxModelVariant = evalSnapshot.onnxModelVariant,
            )
        }
        return evalSnapshot
    }

    private fun updateTimebaseDiagnostics() {
        _diagnostics.value = _diagnostics.value.copy(
            timebaseRelation = timebase.relation,
            timebaseOffsetNs = timebase.offsetNs,
        )
    }
}
