package com.example.driverguardian.ai.demo

import com.example.driverguardian.ai.tensor.FloatTensor
import java.util.ArrayDeque

/**
 * Bounded sequence buffer for the demo end-to-end ONNX model.
 *
 * Requirements:
 * - Rate-limits incoming frames to approximately 2 FPS (~500ms interval).
 * - Maintains a bounded FIFO buffer of size [maxSequenceLength] (default T = 10).
 * - Never accumulates an unbounded sequence.
 * - Produces [1, T, 3, 640, 640] [FloatTensor] inputs for the demo ONNX model.
 */
class BoundedSequenceBuffer(
    val maxSequenceLength: Int = DEFAULT_MAX_T,
    val sampleIntervalMs: Long = DEFAULT_SAMPLE_INTERVAL_MS,
) {
    companion object {
        const val DEFAULT_MAX_T = 10
        const val DEFAULT_SAMPLE_INTERVAL_MS = 500L // ~2 FPS
    }

    init {
        require(maxSequenceLength >= 1) { "maxSequenceLength must be >= 1, got $maxSequenceLength" }
        require(sampleIntervalMs >= 0) { "sampleIntervalMs must be >= 0, got $sampleIntervalMs" }
    }

    private val lock = Any()
    private val buffer = ArrayDeque<FloatArray>(maxSequenceLength)
    private var lastSampleTimestampMs: Long = -1L
    private var totalFramesSampled: Long = 0L

    val currentSize: Int
        get() = synchronized(lock) { buffer.size }

    val isFull: Boolean
        get() = synchronized(lock) { buffer.size == maxSequenceLength }

    /**
     * Attempts to add a preprocessed frame [FloatArray(3 * 640 * 640)].
     *
     * @param frame Preprocessed frame of length [EndToEndDemoPreprocessor.FRAME_ELEMENT_COUNT]
     * @param timestampMs Frame timestamp in milliseconds
     * @param force If true, bypasses the 2 FPS rate limit (useful for unit tests / synthetic bursts)
     * @return True if the frame was accepted and added to the buffer, false if skipped due to sampling interval
     */
    fun offer(
        frame: FloatArray,
        timestampMs: Long,
        force: Boolean = false,
    ): Boolean = synchronized(lock) {
        require(frame.size == EndToEndDemoPreprocessor.FRAME_ELEMENT_COUNT) {
            "Frame size ${frame.size} does not match expected element count ${EndToEndDemoPreprocessor.FRAME_ELEMENT_COUNT}"
        }

        if (!force && lastSampleTimestampMs >= 0L && (timestampMs - lastSampleTimestampMs) < sampleIntervalMs) {
            return false // Rate-limited
        }

        lastSampleTimestampMs = timestampMs
        totalFramesSampled++

        // Evict oldest if bounded capacity reached
        if (buffer.size >= maxSequenceLength) {
            buffer.removeFirst()
        }

        buffer.addLast(frame)
        return true
    }

    /**
     * Builds a [FloatTensor] formatted for the ONNX video input:
     * Shape: [1, T, 3, 640, 640]
     *
     * Returns null if buffer is empty.
     */
    fun toFloatTensor(): FloatTensor? = synchronized(lock) {
        val t = buffer.size
        if (t == 0) return null

        val frameElements = EndToEndDemoPreprocessor.FRAME_ELEMENT_COUNT
        val totalElements = t * frameElements
        val concatenated = FloatArray(totalElements)

        var offset = 0
        for (frame in buffer) {
            System.arraycopy(frame, 0, concatenated, offset, frameElements)
            offset += frameElements
        }

        val shape = listOf(1L, t.toLong(), 3L, 640L, 640L)
        return FloatTensor(shape, concatenated)
    }

    /**
     * Clears all buffered frames and resets the sample timer.
     */
    fun clear() = synchronized(lock) {
        buffer.clear()
        lastSampleTimestampMs = -1L
    }
}
