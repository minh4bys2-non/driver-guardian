package com.example.driverguardian.ai.neural

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.SystemClock
import com.example.driverguardian.ai.monitoring.ThreadDebugLogger
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.min

/**
 * Kết quả một lần suy luận gia tăng của `OnnxNeuralClassifier` trên Luồng 2 & Luồng 3.
 */
data class OnnxPredictionResult(
    val probDrowsy: Double?,
    val tNewFrames: Int,
    val tOldFrames: Int,
    val cachedFrameCount: Int,
    val latencyMs: Double,
    val modelVariant: String,
    val errorMessage: String? = null,
) {
    val drowsyScore: Double? get() = probDrowsy
    val cachedP4Count: Int get() = cachedFrameCount
    val deltaFramesProcessed: Int get() = tNewFrames
    val inferenceMs: Double get() = latencyMs
}

/**
 * Giao diện trừu tượng cho bộ phân loại mạng học sâu để dễ dàng kiểm thử đơn vị (Unit Test)
 * mà không cần phụ thuộc phần cứng Android khi chạy mock.
 */
interface NeuralClassifier : Closeable {
    val modelVariant: String
    val activeModelVariant: String get() = modelVariant
    val cachedFrameCount: Int
    fun predictAndUpdateCache(snapshot: NeuralWindowSnapshot): OnnxPredictionResult
    fun resetCache()
}

/**
 * Bộ phân loại học sâu `ExportDrowsyClassifier` (`CNN–ConvGRU`) trên ONNX Runtime
 * được tối ưu hóa cho tập lệnh `ARMv8.0-A` của Qualcomm Snapdragon 680 (Đặc tả Mục 5.3):
 * - Giới hạn `setIntraOpNumThreads(2)` và `allow_spinning = "0"` để chừa 2 nhân lớn Kryo 265 Gold cho Camera & UI.
 * - Tích hợp XNNPACK.
 * - Duy trì bộ đệm đặc trưng `cachedP4` (`p4_old` kích thước `[T_old, 128, 14, 14]`, tối đa 50 frames = 4.79 MB)
 *   để giảm 63.8% khối lượng tính toán mỗi giây.
 */
class OnnxNeuralClassifier(
    modelBytes: ByteArray,
    override val modelVariant: String = NeuralBranchConfig.DEFAULT_PRIMARY_MODEL_ASSET_PATH,
    val config: NeuralBranchConfig = NeuralBranchConfig.DEFAULT
) : NeuralClassifier {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    // Bộ đệm đặc trưng P4 (tối đa 50 x 25,088 floats = 4.79 MB) thuộc quyền sở hữu độc quyền của Luồng 2
    private var cachedP4: FloatArray = FloatArray(config.neuralCount * config.p4ElementsPerFrame)
    override var cachedFrameCount: Int = 0
        private set
    var cachedGeneration: Long = -1L
        private set
    var cachedSeqId: Long = -1L
        private set
    var lastProbDrowsy: Double? = null
        private set

    init {
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(config.intraOpNumThreads)
            setInterOpNumThreads(config.interOpNumThreads)
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            try {
                addConfigEntry(config.allowSpinningConfigKey, config.allowSpinningConfigValue)
            } catch (_: Throwable) {
            }
            if (config.enableXnnpack) {
                try {
                    addXnnpack(mapOf(config.xnnpackThreadsKey to config.intraOpNumThreads.toString()))
                } catch (_: Throwable) {
                    // Fallback an toàn về CPU chuẩn nếu bản build ORT không kèm XNNPACK
                }
            }
        }
        session = env.createSession(modelBytes, options)
        ThreadDebugLogger.infoOnnxPool(
            "OrtSession initialized: model=$modelVariant, intraOpThreads=${config.intraOpNumThreads}, " +
                "interOpThreads=${config.interOpNumThreads}, allow_spinning=${config.allowSpinningConfigValue}, xnnpack=${config.enableXnnpack}"
        )
    }

    override fun resetCache() {
        cachedFrameCount = 0
        cachedGeneration = -1L
        cachedSeqId = -1L
        lastProbDrowsy = null
    }

    /**
     * Thực thi suy luận cửa sổ trượt gia tăng và cập nhật cache `p4_old`.
     * Chạy độc quyền trên Luồng 2 (`HeavyComputeWorker`) và sử dụng Luồng 3 (`2 Intra-Op Threads`).
     */
    override fun predictAndUpdateCache(snapshot: NeuralWindowSnapshot): OnnxPredictionResult {
        val startedNs = try {
            SystemClock.elapsedRealtimeNanos()
        } catch (_: Throwable) {
            System.nanoTime()
        }

        val delta = snapshot.getNewFramesSince(cachedGeneration, cachedSeqId, cachedFrameCount)
        if (delta.newChwFrames.isEmpty()) {
            return OnnxPredictionResult(
                probDrowsy = lastProbDrowsy,
                tNewFrames = 0,
                tOldFrames = cachedFrameCount,
                cachedFrameCount = cachedFrameCount,
                latencyMs = 0.0,
                modelVariant = modelVariant
            )
        }

        val tNew = delta.newChwFrames.size
        val tOld = delta.keepOldCount
        val chwBytes = config.chwBytesPerFrame
        val p4Elements = config.p4ElementsPerFrame

        // 1. Đóng gói tensor "video": uint8 [T_new, 3, 224, 224]
        val videoByteBuffer = ByteBuffer.allocateDirect(tNew * chwBytes).order(ByteOrder.nativeOrder())
        for (frameBytes in delta.newChwFrames) {
            videoByteBuffer.put(frameBytes, 0, chwBytes)
        }
        videoByteBuffer.rewind()
        val videoShape = longArrayOf(
            tNew.toLong(),
            config.channelsRgb.toLong(),
            config.imageSize.toLong(),
            config.imageSize.toLong()
        )

        // 2. Đóng gói tensor "p4_old": float32 [T_old, 128, 14, 14]
        val oldElements = tOld * p4Elements
        val p4OldBuffer = FloatBuffer.allocate(oldElements)
        if (tOld > 0) {
            val startOffset = (cachedFrameCount - tOld) * p4Elements
            p4OldBuffer.put(cachedP4, startOffset, oldElements)
        }
        p4OldBuffer.rewind()
        val p4OldShape = longArrayOf(
            tOld.toLong(),
            config.p4Channels.toLong(),
            config.p4Height.toLong(),
            config.p4Width.toLong()
        )

        OnnxTensor.createTensor(env, videoByteBuffer, videoShape, OnnxJavaType.UINT8).use { videoTensor ->
            OnnxTensor.createTensor(env, p4OldBuffer, p4OldShape).use { p4OldTensor ->
                val inputs = mapOf(
                    config.inputVideoTensorName to videoTensor,
                    config.inputP4OldTensorName to p4OldTensor
                )
                session.run(inputs).use { results ->
                    val probTensor = results[0] as OnnxTensor
                    val p4OutTensor = results[1] as OnnxTensor

                    val probFloatBuf = probTensor.floatBuffer
                    val probDrowsy = probFloatBuf.get(config.drowsyClassIndex).toDouble()

                    val p4OutBuf = p4OutTensor.floatBuffer
                    val totalOutFrames = tOld + tNew
                    val keepFrames = min(totalOutFrames, config.neuralCount)
                    val keepElements = keepFrames * p4Elements
                    val skipElements = (totalOutFrames - keepFrames) * p4Elements

                    if (cachedP4.size < keepElements) {
                        cachedP4 = FloatArray(config.neuralCount * p4Elements)
                    }
                    p4OutBuf.position(skipElements)
                    p4OutBuf.get(cachedP4, 0, keepElements)

                    cachedFrameCount = keepFrames
                    cachedGeneration = delta.generation
                    cachedSeqId = delta.latestSeqId
                    lastProbDrowsy = probDrowsy

                    val endedNs = try {
                        SystemClock.elapsedRealtimeNanos()
                    } catch (_: Throwable) {
                        System.nanoTime()
                    }
                    val latencyMs = (endedNs - startedNs) / 1_000_000.0
                    ThreadDebugLogger.logOnnxPool(
                        String.format(
                            java.util.Locale.US,
                            "ONNX incremental run: T_new=%d, T_old=%d -> p4_cached=%d/%d, probDrowsy=%.4f, latencyMs=%.2fms (%s)",
                            tNew,
                            tOld,
                            cachedFrameCount,
                            config.neuralCount,
                            probDrowsy,
                            latencyMs,
                            modelVariant,
                        )
                    )

                    return OnnxPredictionResult(
                        probDrowsy = probDrowsy,
                        tNewFrames = tNew,
                        tOldFrames = tOld,
                        cachedFrameCount = cachedFrameCount,
                        latencyMs = latencyMs,
                        modelVariant = modelVariant
                    )
                }
            }
        }
    }

    override fun close() {
        try {
            session.close()
        } catch (_: Throwable) {
        }
    }

    companion object {
        /**
         * Khởi tạo `OnnxNeuralClassifier` từ `assets`, ưu tiên `drowsy_classifier_int8.onnx`
         * và fallback sang `drowsy_classifier_fp32.onnx` theo cấu hình `NeuralBranchConfig`.
         */
        fun createFromAssets(
            context: Context,
            config: NeuralBranchConfig = NeuralBranchConfig.DEFAULT
        ): OnnxNeuralClassifier {
            val candidates = listOf(config.primaryModelAssetPath, config.fallbackModelAssetPath).distinct()
            var lastError: Throwable? = null
            for (assetPath in candidates) {
                try {
                    val bytes = context.assets.open(assetPath).use { it.readBytes() }
                    return OnnxNeuralClassifier(
                        modelBytes = bytes,
                        modelVariant = assetPath,
                        config = config
                    )
                } catch (e: Throwable) {
                    lastError = e
                }
            }
            throw IllegalStateException(
                "Failed to initialize OnnxNeuralClassifier from assets $candidates",
                lastError
            )
        }
    }
}
