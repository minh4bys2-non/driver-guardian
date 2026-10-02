package com.example.driverguardian.ai.monitoring.camera

import android.graphics.Bitmap
import com.example.driverguardian.ai.physical.model.LandmarkPoint
import com.google.mediapipe.tasks.vision.core.RunningMode
import java.io.Closeable
import java.util.concurrent.atomic.AtomicBoolean

object FaceLandmarkerConfiguration {
    const val MODEL_ASSET_PATH = "face_landmarker.task"
    const val MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
    const val NUM_FACES = 1
    val RUNNING_MODE: RunningMode = RunningMode.LIVE_STREAM
    val runtimeDownloadUrl: String? = null
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

class SingleFrameAdmission {
    private val inFlight = AtomicBoolean(false)
    fun isAvailable(): Boolean = !inFlight.get()
    fun tryAcquire(): Boolean = inFlight.compareAndSet(false, true)
    fun release() { inFlight.set(false) }
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
