package com.example.driverguardian.ai.monitoring.camera

import android.content.Context
import android.graphics.Bitmap
import android.hardware.camera2.CameraCharacteristics
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
import com.example.driverguardian.ai.monitoring.time.CameraTimestampSource
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

internal class StartGenerationGate {
    private var generation = 0L

    @Synchronized fun newStart(): Long = ++generation
    @Synchronized fun invalidate() { generation++ }
    @Synchronized fun isCurrent(token: Long): Boolean = token == generation
}

class CameraXFrameSource(
    private val context: Context,
    private val faceLandmarker: FaceLandmarkerClient,
    private val onError: (Long, String) -> Unit,
    private val onTimestampSource: (CameraTimestampSource) -> Unit = {},
) : AutoCloseable {
    private var provider: ProcessCameraProvider? = null
    private var analysis: ImageAnalysis? = null
    private var executor: ExecutorService? = null
    private val startGate = StartGenerationGate()

    var timestampSource: CameraTimestampSource = CameraTimestampSource.UNKNOWN
        private set
    var isFrontCamera: Boolean? = null
        private set

    @Synchronized fun start(lifecycleOwner: LifecycleOwner, surfaceProvider: Preview.SurfaceProvider, generation: Long) {
        stop()
        val startToken = startGate.newStart()
        val localExecutor = Executors.newSingleThreadExecutor { runnable -> Thread(runnable, "driver-guardian-camera") }
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
                    val imageAnalysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build()
                    imageAnalysis.setAnalyzer(localExecutor) { proxy -> analyze(proxy, generation) }
                    analysis = imageAnalysis
                    cameraProvider.unbindAll()
                    val camera = cameraProvider.bindToLifecycle(lifecycleOwner, selector, preview, imageAnalysis)
                    timestampSource = readTimestampSource(camera.cameraInfo)
                    onTimestampSource(timestampSource)
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
        CloseOnce(proxy::close).use {
            try {
                if (!faceLandmarker.isReadyForFrame()) return
                val receiptElapsedNs = SystemClock.elapsedRealtimeNanos()
                val bitmap = rgbaBitmap(proxy)
                faceLandmarker.detect(
                    bitmap = bitmap,
                    width = proxy.width,
                    height = proxy.height,
                    rotationDegrees = proxy.imageInfo.rotationDegrees,
                    cameraTimestampNs = proxy.imageInfo.timestamp,
                    receiptElapsedNs = receiptElapsedNs,
                    generation = generation,
                )
            } catch (error: Exception) {
                onError(generation, "Camera frame conversion failed: ${error.message ?: error.javaClass.simpleName}")
            }
        }
    }

    private fun rgbaBitmap(proxy: ImageProxy): Bitmap {
        val plane = proxy.planes.firstOrNull() ?: error("RGBA frame has no plane")
        require(plane.pixelStride == 4) { "Expected RGBA pixel stride 4, got ${plane.pixelStride}" }
        val rowBytes = proxy.width * plane.pixelStride
        val compact = ByteBuffer.allocate(rowBytes * proxy.height)
        val source = plane.buffer.duplicate()
        val row = ByteArray(rowBytes)
        repeat(proxy.height) { y ->
            source.position(y * plane.rowStride)
            source.get(row)
            compact.put(row)
        }
        compact.rewind()
        return Bitmap.createBitmap(proxy.width, proxy.height, Bitmap.Config.ARGB_8888).also { it.copyPixelsFromBuffer(compact) }
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

    override fun close() { stop(); faceLandmarker.close() }

    @ExperimentalCamera2Interop
    private fun readTimestampSource(cameraInfo: androidx.camera.core.CameraInfo): CameraTimestampSource {
        val value = Camera2CameraInfo.from(cameraInfo)
            .getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)
        return if (value == CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME) {
            CameraTimestampSource.REALTIME
        } else CameraTimestampSource.UNKNOWN
    }
}
