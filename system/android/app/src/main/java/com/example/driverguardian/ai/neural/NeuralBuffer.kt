package com.example.driverguardian.ai.neural

import java.io.Closeable

/**
 * Lô các khung hình CHW mới phát sinh kể từ lần chạy ONNX trước, phục vụ suy luận gia tăng với `p4_old`.
 */
data class NeuralDeltaBatch(
    val newChwFrames: List<ByteArray>,
    val keepOldCount: Int,
    val targetTotalFrames: Int,
    val generation: Long,
    val latestSeqId: Long
)

/**
 * Bản chụp bất biến của `NeuralBuffer` gửi từ Luồng 1B sang Luồng 2 mỗi 1.0s.
 */
data class NeuralWindowSnapshot(
    val generation: Long,
    val latestSeqId: Long,
    val frames: List<ByteArray>
) {
    val size: Int get() = frames.size

    fun getNewFramesSince(
        cachedGeneration: Long,
        cachedSeqId: Long,
        cachedFrameCount: Int
    ): NeuralDeltaBatch {
        val total = frames.size
        if (total == 0) {
            return NeuralDeltaBatch(emptyList(), 0, 0, generation, latestSeqId)
        }
        if (generation != cachedGeneration || cachedSeqId <= 0L || cachedFrameCount <= 0) {
            return NeuralDeltaBatch(
                newChwFrames = frames,
                keepOldCount = 0,
                targetTotalFrames = total,
                generation = generation,
                latestSeqId = latestSeqId
            )
        }
        val delta = (latestSeqId - cachedSeqId).toInt()
        return when {
            delta in 1 until total && cachedFrameCount >= (total - delta) -> {
                NeuralDeltaBatch(
                    newChwFrames = frames.takeLast(delta),
                    keepOldCount = total - delta,
                    targetTotalFrames = total,
                    generation = generation,
                    latestSeqId = latestSeqId
                )
            }
            delta <= 0 && cachedFrameCount >= total -> {
                NeuralDeltaBatch(
                    newChwFrames = frames.takeLast(1),
                    keepOldCount = total - 1,
                    targetTotalFrames = total,
                    generation = generation,
                    latestSeqId = latestSeqId
                )
            }
            else -> {
                NeuralDeltaBatch(
                    newChwFrames = frames,
                    keepOldCount = 0,
                    targetTotalFrames = total,
                    generation = generation,
                    latestSeqId = latestSeqId
                )
            }
        }
    }
}

/**
 * Bộ đệm `NeuralBuffer` lấy mẫu trên lưới `5.0 FPS` (`7.32 MB` RAM - Đặc tả Snapdragon 680 v2 Mục 3.2).
 *
 * Chỉ kích hoạt `letterboxProcessor.convertRgbaToRgbCHW` khi chạm mốc lưới `nextTime <= timestamp + 1e-9`
 * (đúng 5 lần/giây), giảm 83%–92% chi phí tính toán và giảm 52 MB RAM so với lưu 50 frames RGBA thô.
 */
class NeuralBuffer(
    val config: NeuralBranchConfig = NeuralBranchConfig.DEFAULT,
    private val letterboxProcessor: LetterboxProcessor = LetterboxProcessor(config)
) : Closeable {

    private val frames = ArrayDeque<ByteArray>(config.neuralCount)

    // Bộ đệm tĩnh duy nhất giữ frame RGBA liền trước (Zero-GC)
    private var previousRawBytes = ByteArray(config.defaultSourceRgbaBytes)
    private var previousWidth = config.defaultSourceWidth
    private var previousHeight = config.defaultSourceHeight
    private var previousChwCache: ByteArray? = null
    private var hasPrevious = false
    private var previousTime: Double? = null
    private var nextTime: Double = 0.0

    var generation: Long = 1L
        private set
    var seqCounter: Long = 0L
        private set
    var totalLetterboxConversions: Long = 0L
        private set

    val size: Int get() = frames.size

    fun reset() {
        frames.clear()
        previousChwCache = null
        hasPrevious = false
        previousTime = null
        nextTime = 0.0
        generation += 1L
        seqCounter = 0L
    }

    /**
     * Cập nhật từ khung hình RGBA trên Luồng 1B.
     * Trả về số lần thực thi `letterbox` trong lời gọi này (0 nếu không chạm mốc lưới 5.0 FPS).
     */
    fun updateFromRgba(rgbaBytes: ByteArray, width: Int, height: Int, timestamp: Double): Int {
        if (previousTime == null || timestamp - previousTime!! > config.maxGapSec) {
            frames.clear()
            nextTime = timestamp
            generation += 1L
            seqCounter = 0L
            hasPrevious = false
            previousChwCache = null
        }

        val eps = config.gridEpsilonSec
        val stepSec = 1.0 / config.neuralFps
        var currentChwCache: ByteArray? = null
        var conversionsThisStep = 0

        while (nextTime <= timestamp + eps) {
            val chosenChw: ByteArray = if (nextTime < timestamp - eps && hasPrevious) {
                if (previousChwCache == null) {
                    previousChwCache = letterboxProcessor.convertRgbaToRgbCHW(
                        previousRawBytes,
                        previousWidth,
                        previousHeight
                    )
                    conversionsThisStep++
                    totalLetterboxConversions++
                }
                previousChwCache!!
            } else {
                if (currentChwCache == null) {
                    currentChwCache = letterboxProcessor.convertRgbaToRgbCHW(rgbaBytes, width, height)
                    conversionsThisStep++
                    totalLetterboxConversions++
                }
                currentChwCache
            }

            if (frames.size == config.neuralCount) {
                frames.removeFirst()
            }
            frames.addLast(chosenChw)
            seqCounter += 1L
            nextTime += stepSec
        }

        // Lưu lại mảng byte thô vào vùng nhớ tĩnh có sẵn (Zero-GC)
        val copyLen = width * height * config.channelsRgba
        if (previousRawBytes.size < copyLen) {
            previousRawBytes = ByteArray(copyLen)
        }
        System.arraycopy(rgbaBytes, 0, previousRawBytes, 0, copyLen)
        previousWidth = width
        previousHeight = height
        previousChwCache = currentChwCache
        hasPrevious = true
        previousTime = timestamp

        return conversionsThisStep
    }

    /**
     * Alias tiện lợi cho Luồng 1B trả về `true` nếu khung hình này nằm trên lưới 5.0 FPS và đã kích hoạt letterbox.
     */
    fun append(
        rgbaBytes: ByteArray,
        width: Int,
        height: Int,
        timestampSec: Double,
        rotationDegrees: Int = 0,
    ): Boolean = updateFromRgba(rgbaBytes, width, height, timestampSec) > 0

    fun snapshot(): NeuralWindowSnapshot = NeuralWindowSnapshot(
        generation = generation,
        latestSeqId = seqCounter,
        frames = frames.toList()
    )

    override fun close() {
        frames.clear()
        letterboxProcessor.close()
    }
}
