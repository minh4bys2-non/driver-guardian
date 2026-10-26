package com.example.driverguardian.ai.monitoring.camera

import android.content.Context
import android.graphics.Bitmap
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import com.example.driverguardian.ai.monitoring.MonitoringConfig
import com.example.driverguardian.ai.monitoring.ThreadDebugLogger
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.ByteBufferImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean

class MediaPipeFaceLandmarkerClient(
    context: Context,
    private val onResult: (FaceLandmarkerFrameResult) -> Unit,
    private val onError: (Long, String) -> Unit,
) : FaceLandmarkerClient {

    private val timestamps = MonotonicTaskTimestamp()
    private val firstResultLogged = AtomicBoolean(false)
    private val landmarker: FaceLandmarker
    override val activeDelegate: Delegate

    private val mainHandler = Handler(Looper.getMainLooper())
    private val lifecycle = DeferredResourceClose {
        mainHandler.post { landmarker.close() }
    }

    // DirectByteBuffer cố định duy nhất tái sử dụng cho mọi frame (Zero-GC)
    private var reusableDirectBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(CameraXHardwareConfig.FRAME_BYTES_RGBA).order(ByteOrder.nativeOrder())

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

        var initializedDetector: FaceLandmarker? = null
        var chosenDelegate = Delegate.GPU
        try {
            val gpuBaseOptions = BaseOptions.builder()
                .setDelegate(Delegate.GPU)
                .setModelAssetPath(FaceLandmarkerConfiguration.MODEL_ASSET_PATH)
                .build()
            val gpuOptions = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(gpuBaseOptions)
                .setRunningMode(FaceLandmarkerConfiguration.RUNNING_MODE)
                .setNumFaces(FaceLandmarkerConfiguration.NUM_FACES)
                .setMinFaceDetectionConfidence(FaceLandmarkerConfiguration.MIN_FACE_DETECTION_CONFIDENCE)
                .setMinFacePresenceConfidence(FaceLandmarkerConfiguration.MIN_FACE_PRESENCE_CONFIDENCE)
                .setMinTrackingConfidence(FaceLandmarkerConfiguration.MIN_TRACKING_CONFIDENCE)
                .setOutputFaceBlendshapes(FaceLandmarkerConfiguration.OUTPUT_FACE_BLENDSHAPES)
                .setOutputFacialTransformationMatrixes(FaceLandmarkerConfiguration.OUTPUT_FACIAL_TRANSFORMATION_MATRIXES)
                .build()
            initializedDetector = FaceLandmarker.createFromOptions(context, gpuOptions)
            ThreadDebugLogger.infoPhysical("MediaPipe FaceLandmarker initialized with Delegate.GPU (Adreno 610)")
        } catch (gpuException: Throwable) {
            Log.w(TAG, "MediaPipe GPU delegate initialization failed, falling back to Delegate.CPU", gpuException)
            chosenDelegate = Delegate.CPU
            val cpuBaseOptions = BaseOptions.builder()
                .setDelegate(Delegate.CPU)
                .setModelAssetPath(FaceLandmarkerConfiguration.MODEL_ASSET_PATH)
                .build()
            val cpuOptions = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(cpuBaseOptions)
                .setRunningMode(FaceLandmarkerConfiguration.RUNNING_MODE)
                .setNumFaces(FaceLandmarkerConfiguration.NUM_FACES)
                .setMinFaceDetectionConfidence(FaceLandmarkerConfiguration.MIN_FACE_DETECTION_CONFIDENCE)
                .setMinFacePresenceConfidence(FaceLandmarkerConfiguration.MIN_FACE_PRESENCE_CONFIDENCE)
                .setMinTrackingConfidence(FaceLandmarkerConfiguration.MIN_TRACKING_CONFIDENCE)
                .setOutputFaceBlendshapes(FaceLandmarkerConfiguration.OUTPUT_FACE_BLENDSHAPES)
                .setOutputFacialTransformationMatrixes(FaceLandmarkerConfiguration.OUTPUT_FACIAL_TRANSFORMATION_MATRIXES)
                .build()
            initializedDetector = FaceLandmarker.createFromOptions(context, cpuOptions)
            ThreadDebugLogger.infoPhysical("MediaPipe FaceLandmarker initialized with Delegate.CPU")
        }

        landmarker = initializedDetector!!
        activeDelegate = chosenDelegate
    }

    override fun isReadyForFrame(): Boolean = lifecycle.isAvailable()

    override fun detect(
        bitmap: Bitmap,
        width: Int,
        height: Int,
        rotationDegrees: Int,
        cameraTimestampNs: Long,
        receiptElapsedNs: Long,
        generation: Long,
    ) {
        val image = BitmapImageBuilder(bitmap).build()
        processVideoImage(image, width, height, rotationDegrees, cameraTimestampNs, receiptElapsedNs, generation)
    }

    override fun detectRgba(
        rgbaBytes: ByteArray,
        width: Int,
        height: Int,
        rotationDegrees: Int,
        cameraTimestampNs: Long,
        receiptElapsedNs: Long,
        generation: Long,
    ) {
        val requiredBytes = width * height * 4
        if (reusableDirectBuffer.capacity() != requiredBytes) {
            reusableDirectBuffer = ByteBuffer.allocateDirect(requiredBytes).order(ByteOrder.nativeOrder())
        }
        reusableDirectBuffer.clear()
        reusableDirectBuffer.put(rgbaBytes, 0, requiredBytes)
        reusableDirectBuffer.rewind()

        val image = ByteBufferImageBuilder(
            reusableDirectBuffer,
            width,
            height,
            MPImage.IMAGE_FORMAT_RGBA,
        ).build()
        processVideoImage(image, width, height, rotationDegrees, cameraTimestampNs, receiptElapsedNs, generation)
    }

    private fun processVideoImage(
        image: MPImage,
        width: Int,
        height: Int,
        rotationDegrees: Int,
        cameraTimestampNs: Long,
        receiptElapsedNs: Long,
        generation: Long,
    ) {
        if (!lifecycle.tryBeginWork()) {
            image.close()
            return
        }
        val taskTimestampMs = timestamps.fromCameraNanoseconds(cameraTimestampNs)
        val submittedNs = SystemClock.elapsedRealtimeNanos()
        try {
            val processing = ImageProcessingOptions.builder()
                .setRotationDegrees(rotationDegrees)
                .build()

            // Thực thi suy luận đồng bộ detectForVideo theo chuỗi frame thời gian (RunningMode.VIDEO)
            val result = landmarker.detectForVideo(image, processing, taskTimestampMs)
            val inferenceMs = (SystemClock.elapsedRealtimeNanos() - submittedNs) / 1_000_000.0

            if (firstResultLogged.compareAndSet(false, true)) {
                ThreadDebugLogger.infoPhysical(
                    "First video face result timestampMs=$taskTimestampMs faces=${result.faceLandmarks().size} " +
                        "inferenceMs=$inferenceMs delegate=$activeDelegate"
                )
            }

            val (rotatedWidth, rotatedHeight) = CameraFrameGeometry.rotatedSize(width, height, rotationDegrees)
            val face = result.faceLandmarks().firstOrNull()
            onResult(
                FaceLandmarkerFrameResult(
                    landmarks = face?.let { MediaPipeLandmarkAdapter.toPixelLandmarks(it, rotatedWidth, rotatedHeight) },
                    width = rotatedWidth,
                    height = rotatedHeight,
                    cameraTimestampNs = cameraTimestampNs,
                    receiptElapsedNs = receiptElapsedNs,
                    generation = generation,
                    inferenceMs = inferenceMs,
                )
            )
        } catch (error: Throwable) {
            Log.e(TAG, "MediaPipe Face Landmarker detectForVideo failed", error)
            onError(generation, "MediaPipe inference failed: ${error.message ?: error.javaClass.simpleName}")
        } finally {
            image.close()
            lifecycle.finishWork()
        }
    }

    override fun close() {
        timestamps.reset()
        lifecycle.requestClose()
    }

    private companion object {
        const val TAG = MonitoringConfig.TAG_THREAD_1B_PHYSICAL
    }
}
