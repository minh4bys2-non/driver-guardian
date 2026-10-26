package com.example.driverguardian.ai.monitoring

import android.content.Context
import android.os.Process
import androidx.camera.core.Preview
import androidx.lifecycle.LifecycleOwner
import com.example.driverguardian.ai.monitoring.camera.CameraXFrameSource
import com.example.driverguardian.ai.monitoring.camera.FaceLandmarkerClient
import com.example.driverguardian.ai.monitoring.camera.FaceLandmarkerFrameResult
import com.example.driverguardian.ai.monitoring.camera.MediaPipeFaceLandmarkerClient
import com.example.driverguardian.ai.monitoring.camera.RgbaFramePacket
import com.example.driverguardian.ai.monitoring.model.MonitoringDiagnostics
import com.example.driverguardian.ai.monitoring.model.MonitoringRuntimeState
import com.example.driverguardian.ai.monitoring.model.MonitoringSnapshot
import com.example.driverguardian.ai.monitoring.time.CameraTimestampSource
import com.example.driverguardian.ai.neural.NeuralClassifier
import com.example.driverguardian.ai.neural.OnnxNeuralClassifier
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference

enum class CameraPermissionState { UNKNOWN, GRANTED, DENIED, UNAVAILABLE }

interface MonitoringEngineControl : AutoCloseable {
    val snapshot: StateFlow<MonitoringSnapshot?>
    val runtimeState: StateFlow<MonitoringRuntimeState>
    val diagnostics: StateFlow<MonitoringDiagnostics>
    val balanceBrightnessEnabled: StateFlow<Boolean>
        get() = MutableStateFlow(MonitoringConfig.DEFAULT_BALANCE_BRIGHTNESS_ENABLED)
    fun startSession(sessionId: Int)
    fun updatePermission(permission: CameraPermissionState)
    fun attachCamera(lifecycleOwner: LifecycleOwner, surfaceProvider: Preview.SurfaceProvider) = Unit
    fun setBalanceBrightness(enabled: Boolean) = Unit
    fun stopSession()
}

/**
 * Multi-threaded real-time monitoring engine implementing the Snapdragon 680 v2 architecture:
 * - Thread 1A (`CameraIngestThread`): CameraX RGBA_8888 ingest, Zero-GC pool, BrightnessBalancer.
 * - Thread 1B (`UnifiedPhysicalThread`): Conflated frame consumer, MediaPipe GPU FaceLandmarker,
 *   lazy 5.0 FPS `NeuralBuffer`, and 30-60 FPS `PhysicalBranchProcessor`.
 * - Thread 2 (`HeavyComputeWorker`): 1.0 Hz window evaluator (`LazyHeadMotionWindow.evaluateExactFft`,
 *   `PhysicalNormalizer`, `FitnessEvaluator`) & exclusive owner of `p4_old` cache.
 * - Thread 3 (`OnnxIntraOpPool`): 2 Intra-Op ONNX threads managed inside `OnnxNeuralClassifier`.
 */
class DriverMonitoringEngine(private val context: Context) : MonitoringEngineControl {
    private val coordinator = DriverMonitoringCoordinator(cameraTimestampSource = CameraTimestampSource.UNKNOWN)
    private val _balanceBrightnessEnabled = MutableStateFlow(MonitoringConfig.DEFAULT_BALANCE_BRIGHTNESS_ENABLED)

    private var cameraSource: CameraXFrameSource? = null
    private var faceLandmarker: FaceLandmarkerClient? = null
    private var neuralClassifier: NeuralClassifier? = null
    private var lifecycleOwner: LifecycleOwner? = null
    private var surfaceProvider: Preview.SurfaceProvider? = null
    private var permission = CameraPermissionState.UNKNOWN
    private var activeSessionId: Int? = null

    // Dedicated executors for Thread 1B and Thread 2
    private val physicalExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread({
            runCatching { Process.setThreadPriority(MonitoringConfig.THREAD_PRIORITY_1B_PHYSICAL) }
            runnable.run()
        }, MonitoringConfig.THREAD_NAME_1B_PHYSICAL)
    }
    private val heavyWorkerExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread({
            runCatching { Process.setThreadPriority(MonitoringConfig.THREAD_PRIORITY_2_HEAVY_WORKER) }
            runnable.run()
        }, MonitoringConfig.THREAD_NAME_2_HEAVY_WORKER)
    }

    private val physicalScope = CoroutineScope(SupervisorJob() + physicalExecutor.asCoroutineDispatcher())
    private val heavyWorkerScope = CoroutineScope(SupervisorJob() + heavyWorkerExecutor.asCoroutineDispatcher())

    private val frameChannel = Channel<RgbaFramePacket>(Channel.CONFLATED)
    private val heavyEvalChannel = Channel<HeavyEvaluationRequest>(Channel.CONFLATED)
    private val inFlightPacketRef = AtomicReference<RgbaFramePacket?>(null)
    private val isHeavyWorkerBusy = java.util.concurrent.atomic.AtomicBoolean(false)

    private var physicalConsumerJob: Job? = null
    private var heavyWorkerJob: Job? = null

    override val snapshot = coordinator.snapshot
    override val runtimeState = coordinator.runtimeState
    override val diagnostics = coordinator.diagnostics
    override val balanceBrightnessEnabled: StateFlow<Boolean> = _balanceBrightnessEnabled.asStateFlow()

    init {
        startBackgroundWorkers()
    }

    private fun startBackgroundWorkers() {
        // Thread 2 (`HeavyComputeWorker`): loads ONNX model once and consumes 1Hz evaluation requests
        heavyWorkerJob = heavyWorkerScope.launch {
            ensureNeuralClassifierLoaded()
            for (request in heavyEvalChannel) {
                isHeavyWorkerBusy.set(true)
                try {
                    val classifier = ensureNeuralClassifierLoaded()
                    coordinator.runHeavyEvaluation(request, classifier)
                } catch (error: Throwable) {
                    ThreadDebugLogger.logHeavyWorker(
                        "Heavy evaluation failed: ${error.message ?: error.javaClass.simpleName}",
                    )
                } finally {
                    isHeavyWorkerBusy.set(false)
                }
            }
        }

        // Thread 1B (`UnifiedPhysicalThread`): consumes latest frame from Thread 1A
        physicalConsumerJob = physicalScope.launch {
            for (packet in frameChannel) {
                val detector = faceLandmarker
                if (detector != null && detector.isReadyForFrame()) {
                    inFlightPacketRef.set(packet)
                    detector.detectRgba(
                        rgbaBytes = packet.buffer,
                        width = packet.width,
                        height = packet.height,
                        rotationDegrees = packet.rotationDegrees,
                        cameraTimestampNs = packet.sensorTimestampNs,
                        receiptElapsedNs = packet.receiptElapsedNs,
                        generation = packet.generation,
                    )
                }
            }
        }
    }

    private fun ensureNeuralClassifierLoaded(): NeuralClassifier? {
        neuralClassifier?.let { return it }
        return runCatching {
            OnnxNeuralClassifier.createFromAssets(context).also { loaded ->
                neuralClassifier = loaded
                coordinator.reportOnnxModelVariant(loaded.modelVariant)
                ThreadDebugLogger.infoHeavyWorker("ONNX classifier ready on Thread 2: variant=${loaded.modelVariant}")
            }
        }.onFailure { error ->
            ThreadDebugLogger.logHeavyWorker(
                "ONNX classifier initialization failed: ${error.message ?: error.javaClass.simpleName}",
            )
        }.getOrNull()
    }

    override fun setBalanceBrightness(enabled: Boolean) {
        _balanceBrightnessEnabled.value = enabled
        cameraSource?.setBalanceBrightness(enabled)
        coordinator.reportBalanceBrightness(enabled)
        ThreadDebugLogger.logUi("User toggled Balance Brightness -> enabled=$enabled")
    }

    override fun startSession(sessionId: Int) {
        if (activeSessionId == sessionId) return
        if (activeSessionId != null) stopSession()
        activeSessionId = sessionId
        heavyWorkerScope.launch {
            neuralClassifier?.resetCache()
        }
        coordinator.startSession(sessionId.toLong())
        coordinator.reportBalanceBrightness(_balanceBrightnessEnabled.value)
        coordinator.updateCameraPermission(permission == CameraPermissionState.GRANTED)
        bindCameraIfReady()
    }

    override fun updatePermission(permission: CameraPermissionState) {
        if (this.permission == permission) return
        this.permission = permission
        coordinator.updateCameraPermission(permission == CameraPermissionState.GRANTED)
        if (permission == CameraPermissionState.GRANTED) bindCameraIfReady() else cameraSource?.stop()
    }

    override fun attachCamera(lifecycleOwner: LifecycleOwner, surfaceProvider: Preview.SurfaceProvider) {
        this.lifecycleOwner = lifecycleOwner
        this.surfaceProvider = surfaceProvider
        bindCameraIfReady()
    }

    private fun bindCameraIfReady() {
        val sessionId = activeSessionId ?: return
        val owner = lifecycleOwner ?: return
        val surface = surfaceProvider ?: return
        if (permission != CameraPermissionState.GRANTED) return
        try {
            if (faceLandmarker == null) {
                val detector = MediaPipeFaceLandmarkerClient(
                    context = context,
                    onResult = ::handleFaceLandmarkerResultOnPhysicalThread,
                    onError = coordinator::reportCameraError,
                )
                faceLandmarker = detector
                coordinator.reportActiveDelegate(detector.activeDelegate.name)
            }
            if (cameraSource == null) {
                cameraSource = CameraXFrameSource(
                    context = context,
                    faceLandmarker = null, // Frame handoff goes via conflated `frameChannel` to Thread 1B
                    onFramePacket = { packet ->
                        frameChannel.trySend(packet)
                    },
                    onError = coordinator::reportCameraError,
                    onTimestampSource = coordinator::reportCameraTimestampSource,
                ).also { source ->
                    source.setBalanceBrightness(_balanceBrightnessEnabled.value)
                }
            }
            cameraSource!!.start(owner, surface, sessionId.toLong())
        } catch (error: Exception) {
            coordinator.reportCameraError(
                sessionId.toLong(),
                error.message ?: "Monitoring camera initialization failed",
            )
        }
    }

    private fun handleFaceLandmarkerResultOnPhysicalThread(result: FaceLandmarkerFrameResult) {
        val packet = inFlightPacketRef.getAndSet(null)
        coordinator.onCameraResultWithRgba(
            landmarks = result.landmarks,
            rgbaBytes = packet?.buffer,
            rawWidth = packet?.width ?: result.width,
            rawHeight = packet?.height ?: result.height,
            rotationDegrees = packet?.rotationDegrees ?: 0,
            rotatedWidth = result.width,
            rotatedHeight = result.height,
            cameraTimestampNs = result.cameraTimestampNs,
            receiptElapsedNs = result.receiptElapsedNs,
            generation = result.generation,
            inferenceMs = result.inferenceMs,
            ingestLatencyMs = packet?.ingestLatencyMs,
            onHeavyEvaluationDue = { request ->
                if (!isHeavyWorkerBusy.get()) {
                    heavyEvalChannel.trySend(request)
                }
            },
        )
    }

    override fun stopSession() {
        if (activeSessionId == null) return
        cameraSource?.stop()
        inFlightPacketRef.set(null)
        coordinator.stopSession()
        activeSessionId = null
    }

    override fun close() {
        stopSession()
        physicalConsumerJob?.cancel()
        heavyWorkerJob?.cancel()
        frameChannel.close()
        heavyEvalChannel.close()
        cameraSource?.close()
        cameraSource = null
        faceLandmarker?.close()
        faceLandmarker = null
        runCatching { neuralClassifier?.close() }
        neuralClassifier = null
        physicalScope.cancel()
        heavyWorkerScope.cancel()
        physicalExecutor.shutdownNow()
        heavyWorkerExecutor.shutdownNow()
    }
}
