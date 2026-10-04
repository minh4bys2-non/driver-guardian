package com.example.driverguardian.ai.monitoring.camera

import android.content.Context
import android.graphics.Bitmap
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

class MediaPipeFaceLandmarkerClient(
    context: Context,
    private val onResult: (FaceLandmarkerFrameResult) -> Unit,
    private val onError: (Long, String) -> Unit,
) : FaceLandmarkerClient {
    private data class Pending(
        val width: Int,
        val height: Int,
        val rotation: Int,
        val cameraTimestampNs: Long,
        val receiptElapsedNs: Long,
        val generation: Long,
        val submittedElapsedNs: Long,
        val image: MPImage,
    )

    private val timestamps = MonotonicTaskTimestamp()
    private val pending = AtomicReference<Pending?>()
    private val firstResultLogged = AtomicBoolean(false)
    private lateinit var landmarker: FaceLandmarker
    private val mainHandler = Handler(Looper.getMainLooper())
    private val lifecycle = DeferredResourceClose {
        mainHandler.post { landmarker.close() }
    }

    init {
        val bytes = try {
            context.assets.open(FaceLandmarkerConfiguration.MODEL_ASSET_PATH).use { it.readBytes() }
        } catch (error: Exception) {
            throw IllegalStateException("Missing MediaPipe asset ${FaceLandmarkerConfiguration.MODEL_ASSET_PATH}", error)
        }
        val digest = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
        require(digest == FaceLandmarkerConfiguration.MODEL_SHA256) {
            "Corrupt MediaPipe asset: SHA-256 mismatch"
        }
        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath(FaceLandmarkerConfiguration.MODEL_ASSET_PATH).build())
            .setRunningMode(FaceLandmarkerConfiguration.RUNNING_MODE)
            .setNumFaces(FaceLandmarkerConfiguration.NUM_FACES)
            .setResultListener { result, outputImage ->
                val item = pending.getAndSet(null)
                if (firstResultLogged.compareAndSet(false, true)) {
                    Log.d(TAG, "First face result timestampMs=${result.timestampMs()} matched=${item != null} faces=${result.faceLandmarks().size}")
                }
                try {
                    if (item != null) {
                        val (rotatedWidth, rotatedHeight) = CameraFrameGeometry.rotatedSize(item.width, item.height, item.rotation)
                        val face = result.faceLandmarks().firstOrNull()
                        onResult(
                            FaceLandmarkerFrameResult(
                                landmarks = face?.let { MediaPipeLandmarkAdapter.toPixelLandmarks(it, rotatedWidth, rotatedHeight) },
                                width = rotatedWidth,
                                height = rotatedHeight,
                                cameraTimestampNs = item.cameraTimestampNs,
                                receiptElapsedNs = item.receiptElapsedNs,
                                generation = item.generation,
                                inferenceMs = (SystemClock.elapsedRealtimeNanos() - item.submittedElapsedNs) / 1_000_000.0,
                            ),
                        )
                    }
                } finally {
                    outputImage.close()
                    if (item?.image !== outputImage) item?.image?.close()
                    lifecycle.finishWork()
                }
            }
            .setErrorListener { error ->
                try {
                    val item = pending.getAndSet(null)
                    item?.image?.close()
                    Log.e(TAG, "MediaPipe Face Landmarker inference failed", error)
                    if (item != null) {
                        onError(item.generation, "MediaPipe inference failed: ${error.message ?: error.javaClass.simpleName}")
                    }
                } finally {
                    lifecycle.finishWork()
                }
            }
            .build()
        landmarker = try {
            FaceLandmarker.createFromOptions(context, options)
        } catch (error: Exception) {
            throw IllegalStateException("Unable to initialize MediaPipe Face Landmarker asset", error)
        }
    }

    override fun isReadyForFrame(): Boolean = lifecycle.isAvailable()

    override fun detect(bitmap: Bitmap, width: Int, height: Int, rotationDegrees: Int, cameraTimestampNs: Long, receiptElapsedNs: Long, generation: Long) {
        if (!lifecycle.tryBeginWork()) return
        val taskTimestampMs = timestamps.fromCameraNanoseconds(cameraTimestampNs)
        val image = BitmapImageBuilder(bitmap).build()
        val item = Pending(width, height, rotationDegrees, cameraTimestampNs, receiptElapsedNs, generation, SystemClock.elapsedRealtimeNanos(), image)
        pending.set(item)
        try {
            val processing = ImageProcessingOptions.builder().setRotationDegrees(rotationDegrees).build()
            landmarker.detectAsync(image, processing, taskTimestampMs)
        } catch (error: Exception) {
            pending.compareAndSet(item, null)
            item.image.close()
            lifecycle.finishWork()
            Log.e(TAG, "MediaPipe Face Landmarker frame submission failed", error)
            onError(item.generation, "MediaPipe frame submission failed: ${error.message ?: error.javaClass.simpleName}")
        }
    }

    override fun close() {
        timestamps.reset()
        lifecycle.requestClose()
    }

    private companion object {
        const val TAG = "DGFaceLandmarker"
    }
}
