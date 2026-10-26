package com.example.driverguardian.ai.monitoring.camera

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.os.Process
import android.os.SystemClock
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.example.driverguardian.ai.monitoring.MonitoringConfig
import com.example.driverguardian.ai.monitoring.ThreadDebugLogger
import com.example.driverguardian.ai.monitoring.time.CameraTimestampSource
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

internal class StartGenerationGate {
    private var generation = 0L

    @Synchronized fun newStart(): Long = ++generation
    @Synchronized fun invalidate() { generation++ }
    @Synchronized fun isCurrent(token: Long): Boolean = token == generation
}

/**
 * Luồng 1A: Camera Ingest Thread & Quản lý CameraX.
 *
 * Chịu trách nhiệm:
 * 1. Cấu hình CameraX với độ phân giải VGA 640x480 RGBA_8888, tỷ lệ 4:3, STRATEGY_KEEP_ONLY_LATEST.
 * 2. Thu nhận khung hình và sao chép trực tiếp vào bộ đệm tĩnh [RgbaFramePool] (Triple-Buffering Zero-GC).
 * 3. Giải phóng sớm [ImageProxy.close] ngay lập tức (< 0.2 ms) để ISP camera hardware không bao giờ bị nghẽn.
 * 4. Đồng hồ phần cứng đơn điệu nghiêm ngặt (Strict Monotonic Clamping) chống giật lùi thời gian.
 * 5. Cân bằng sáng tùy chọn [BrightnessBalancer] (OpenCV in-place, Fast Luma Check bỏ qua khi trời đủ sáng).
 * 6. Phát gói [RgbaFramePacket] tới downstream (Luồng 1B) hoặc chuyển tiếp an toàn tới [faceLandmarker].
 */
class CameraXFrameSource(
    private val context: Context,
    private val faceLandmarker: FaceLandmarkerClient? = null,
    private val onFramePacket: ((RgbaFramePacket) -> Unit)? = null,
    private val onError: (Long, String) -> Unit,
    private val onTimestampSource: (CameraTimestampSource) -> Unit = {},
    private val pool: RgbaFramePool = RgbaFramePool(),
    val brightnessBalancer: BrightnessBalancer = BrightnessBalancer(),
) : AutoCloseable {
    private var provider: ProcessCameraProvider? = null
    private var analysis: ImageAnalysis? = null
    private var executor: ExecutorService? = null
    private val startGate = StartGenerationGate()

    private var originNs: Long? = null
    private var lastTimestampSec: Double? = null
    private var ingestedFramesCount: Long = 0L

    var timestampSource: CameraTimestampSource = CameraTimestampSource.UNKNOWN
        private set
    var isFrontCamera: Boolean? = null
        private set

    val framePool: RgbaFramePool get() = pool
    val isBalanceBrightnessEnabled: Boolean get() = brightnessBalancer.enabled.get()

    fun setBalanceBrightness(enabled: Boolean) {
        brightnessBalancer.enabled.set(enabled)
        ThreadDebugLogger.logIngest("BalanceBrightness updated -> enabled=$enabled (targetLuma=${MonitoringConfig.TARGET_LUMA})")
    }

    @Synchronized fun start(lifecycleOwner: LifecycleOwner, surfaceProvider: Preview.SurfaceProvider, generation: Long) {
        stop()
        val startToken = startGate.newStart()
        originNs = null
        lastTimestampSec = null
        ingestedFramesCount = 0L

        val localExecutor = Executors.newSingleThreadExecutor { runnable ->
            Thread({
                runCatching { Process.setThreadPriority(MonitoringConfig.THREAD_PRIORITY_1A_INGEST) }
                runnable.run()
            }, MonitoringConfig.THREAD_NAME_1A_INGEST)
        }
        executor = localExecutor

        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            synchronized(this) {
                if (!startGate.isCurrent(startToken)) {
                    localExecutor.shutdownNow()
                    return@synchronized
                }
                try {
                    val cameraProvider = future.get()
                    provider = cameraProvider
                    val selector = when {
                        cameraProvider.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA) -> CameraSelector.DEFAULT_FRONT_CAMERA.also { isFrontCamera = true }
                        cameraProvider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA) -> CameraSelector.DEFAULT_BACK_CAMERA.also { isFrontCamera = false }
                        else -> throw IllegalStateException("No usable CameraX camera")
                    }
                    val preview = Preview.Builder().build().also { it.setSurfaceProvider(surfaceProvider) }
                    val imageAnalysis = CameraXHardwareConfig.buildImageAnalysis(localExecutor) { proxy ->
                        analyze(proxy, generation)
                    }
                    analysis = imageAnalysis
                    cameraProvider.unbindAll()
                    val camera = cameraProvider.bindToLifecycle(lifecycleOwner, selector, preview, imageAnalysis)
                    timestampSource = readTimestampSource(camera.cameraInfo)
                    onTimestampSource(timestampSource)
                    ThreadDebugLogger.logIngest(
                        "CameraX bound: target=${MonitoringConfig.TARGET_WIDTH}x${MonitoringConfig.TARGET_HEIGHT} " +
                            "RGBA_8888, front=$isFrontCamera, timestampSource=$timestampSource, generation=$generation"
                    )
                } catch (error: Exception) {
                    if (startGate.isCurrent(startToken)) {
                        onError(generation, "CameraX initialization failed: ${error.message ?: error.javaClass.simpleName}")
                        stop()
                    } else {
                        localExecutor.shutdownNow()
                    }
                }
            }
        }, ContextCompat.getMainExecutor(context))
    }

    private fun analyze(proxy: ImageProxy, generation: Long) {
        val receiptElapsedNs = SystemClock.elapsedRealtimeNanos()
        val width = proxy.width
        val height = proxy.height
        val rotationDegrees = proxy.imageInfo.rotationDegrees
        val cameraTimestampNs = proxy.imageInfo.timestamp

        // Lấy buffer tĩnh từ pool (Zero-GC Triple-Buffering)
        val dstBuffer = pool.acquireNextBuffer()
        try {
            CameraXHardwareConfig.extractTightRgbaBytesInto(proxy, dstBuffer)
        } catch (error: Exception) {
            proxy.close()
            onError(generation, "Camera frame extraction failed: ${error.message ?: error.javaClass.simpleName}")
            return
        }

        // Đóng ImageProxy NGAY LẬP TỨC (< 0.2 ms) để phần cứng CameraX ISP không bao giờ bị nghẽn
        proxy.close()

        try {
            // Đồng hồ phần cứng đơn điệu nghiêm ngặt (Strict Monotonic Clamping)
            if (originNs == null) {
                originNs = cameraTimestampNs
            }
            var timestampSec = (cameraTimestampNs - originNs!!) / 1_000_000_000.0
            val last = lastTimestampSec
            if (last != null && timestampSec <= last) {
                timestampSec = last + MonitoringConfig.MONOTONIC_TIMESTAMP_MIN_STEP_SEC
            }
            lastTimestampSec = timestampSec
            ingestedFramesCount++

            // Tiền xử lý cân bằng sáng tùy chọn trên Luồng 1A (Zero-Conversion khi trời đủ sáng)
            brightnessBalancer.processInPlaceIfDark(dstBuffer, width, height)

            val ingestLatencyMs = (SystemClock.elapsedRealtimeNanos() - receiptElapsedNs) / 1_000_000.0
            if (ingestLatencyMs >= MonitoringConfig.INGEST_SLOW_WARN_THRESHOLD_MS) {
                ThreadDebugLogger.warnIngest(
                    String.format(
                        java.util.Locale.US,
                        "Slow ingest frame #%d: t=%.3fs, ingestLatencyMs=%.2fms (balanceBrightness=%s)",
                        ingestedFramesCount,
                        timestampSec,
                        ingestLatencyMs,
                        brightnessBalancer.enabled.get(),
                    )
                )
            } else if (ingestedFramesCount == 1L || ingestedFramesCount % MonitoringConfig.LOG_INGEST_EVERY_N_FRAMES == 0L) {
                ThreadDebugLogger.logIngest(
                    String.format(
                        java.util.Locale.US,
                        "Frame #%d ingested: %dx%d rot=%d, t=%.3fs, ingestMs=%.2fms, balanceBrightness=%s",
                        ingestedFramesCount,
                        width,
                        height,
                        rotationDegrees,
                        timestampSec,
                        ingestLatencyMs,
                        brightnessBalancer.enabled.get(),
                    )
                )
            }
            val packet = RgbaFramePacket(
                buffer = dstBuffer,
                width = width,
                height = height,
                rotationDegrees = rotationDegrees,
                timestampSec = timestampSec,
                sensorTimestampNs = cameraTimestampNs,
                receiptElapsedNs = receiptElapsedNs,
                generation = generation,
                ingestLatencyMs = ingestLatencyMs,
            )

            // Đẩy gói frame sang downstream listener nếu có
            onFramePacket?.invoke(packet)

            // Chuyển tiếp mảng byte trực tiếp sang MediaPipe Face Landmarker (Zero-GC)
            faceLandmarker?.let { landmarker ->
                if (landmarker.isReadyForFrame()) {
                    landmarker.detectRgba(
                        rgbaBytes = dstBuffer,
                        width = width,
                        height = height,
                        rotationDegrees = rotationDegrees,
                        cameraTimestampNs = cameraTimestampNs,
                        receiptElapsedNs = receiptElapsedNs,
                        generation = generation,
                    )
                }
            }
        } catch (error: Exception) {
            onError(generation, "Camera ingest processing failed: ${error.message ?: error.javaClass.simpleName}")
        }
    }

    @Synchronized fun stop() {
        startGate.invalidate()
        analysis?.clearAnalyzer()
        analysis = null
        provider?.unbindAll()
        provider = null
        executor?.shutdownNow()
        executor = null
        isFrontCamera = null
    }

    override fun close() {
        stop()
        brightnessBalancer.close()
        faceLandmarker?.close()
    }

    @ExperimentalCamera2Interop
    private fun readTimestampSource(cameraInfo: androidx.camera.core.CameraInfo): CameraTimestampSource {
        val value = Camera2CameraInfo.from(cameraInfo)
            .getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)
        return if (value == CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME) {
            CameraTimestampSource.REALTIME
        } else CameraTimestampSource.UNKNOWN
    }
}
