package com.example.driverguardian.ai.monitoring.camera

import android.graphics.Bitmap
import com.example.driverguardian.ai.physical.model.LandmarkPoint
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import java.io.Closeable
import java.util.concurrent.atomic.AtomicBoolean

object FaceLandmarkerConfiguration {
    const val MODEL_ASSET_PATH = "face_landmarker.task"
    const val MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
    const val NUM_FACES = 1
    val RUNNING_MODE: RunningMode = RunningMode.VIDEO
    const val MIN_FACE_DETECTION_CONFIDENCE = 0.6f
    const val MIN_FACE_PRESENCE_CONFIDENCE = 0.6f
    const val MIN_TRACKING_CONFIDENCE = 0.6f
    const val OUTPUT_FACE_BLENDSHAPES = false
    const val OUTPUT_FACIAL_TRANSFORMATION_MATRIXES = false
}

data class FaceLandmarkerFrameResult(
    val landmarks: List<LandmarkPoint>?,
    val width: Int,
    val height: Int,
    val cameraTimestampNs: Long,
    val receiptElapsedNs: Long,
    val generation: Long,
    val inferenceMs: Double,
)

interface FaceLandmarkerClient : Closeable {
    val activeDelegate: Delegate get() = Delegate.CPU

    fun isReadyForFrame(): Boolean = true

    fun detect(
        bitmap: Bitmap,
        width: Int,
        height: Int,
        rotationDegrees: Int,
        cameraTimestampNs: Long,
        receiptElapsedNs: Long,
        generation: Long,
    )

    fun detectRgba(
        rgbaBytes: ByteArray,
        width: Int,
        height: Int,
        rotationDegrees: Int,
        cameraTimestampNs: Long,
        receiptElapsedNs: Long,
        generation: Long,
    ) {
        val compact = java.nio.ByteBuffer.wrap(rgbaBytes)
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { it.copyPixelsFromBuffer(compact) }
        detect(bitmap, width, height, rotationDegrees, cameraTimestampNs, receiptElapsedNs, generation)
    }
}

class MonotonicTaskTimestamp {
    private var lastMs: Long? = null
    @Synchronized fun fromCameraNanoseconds(timestampNs: Long): Long {
        val candidate = timestampNs / 1_000_000L
        val next = lastMs?.let { maxOf(candidate, it + 1L) } ?: candidate
        lastMs = next
        return next
    }
    @Synchronized fun reset() { lastMs = null }
}

class CloseOnce(private val action: () -> Unit) : Closeable {
    private val closed = AtomicBoolean(false)
    override fun close() { if (closed.compareAndSet(false, true)) action() }
}

class DeferredResourceClose(private val closeAction: () -> Unit) {
    private var inFlight = false
    private var closeRequested = false
    private var closed = false

    fun isAvailable(): Boolean = synchronized(this) { !inFlight && !closeRequested && !closed }

    fun tryBeginWork(): Boolean = synchronized(this) {
        if (inFlight || closeRequested || closed) false else {
            inFlight = true
            true
        }
    }

    fun finishWork() {
        val shouldClose = synchronized(this) {
            inFlight = false
            (closeRequested && !closed).also { if (it) closed = true }
        }
        if (shouldClose) closeAction()
    }

    fun requestClose() {
        val shouldClose = synchronized(this) {
            closeRequested = true
            (!inFlight && !closed).also { if (it) closed = true }
        }
        if (shouldClose) closeAction()
    }
}
