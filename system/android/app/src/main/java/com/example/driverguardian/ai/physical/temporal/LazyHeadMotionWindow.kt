package com.example.driverguardian.ai.physical.temporal

import com.example.driverguardian.ai.physical.PhysicalConfig
import org.jtransforms.fft.DoubleFFT_1D
import java.util.Arrays
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Thống kê chuyển động góc gật đầu (pitch) và kết quả phổ tần số FFT.
 */
data class HeadMotionMetrics(
    val pitchMeanDeg: Double?,
    val pitchMeanAmplitudeDeg: Double?,
    val headMotionFrequencyHz: Double?,
    val headMotionResolutionHz: Double?,
    val headMotionObservedSec: Double
) {
    val frequency: Double?
        get() = headMotionFrequencyHz

    val resolution: Double?
        get() = headMotionResolutionHz

    val observed: Double
        get() = headMotionObservedSec
}

/**
 * Bản chụp bất biến của chuỗi mẫu pitch gửi từ Luồng 1B sang Luồng 2 để tính FFT 1Hz.
 */
data class PitchSamplesSnapshot(
    val times: DoubleArray,
    val values: DoubleArray,
    val timestamp: Double,
    val windowSec: Double,
    val minAmplitudeDeg: Double,
    val minFftSamples: Int = PhysicalConfig.DEFAULT_HEAD_MOTION_MIN_SAMPLES_FOR_FFT,
    val windowBoundaryEpsilon: Double = PhysicalConfig.DEFAULT_HEAD_MOTION_WINDOW_BOUNDARY_EPSILON
)

/**
 * LazyHeadMotionWindow (Đặc tả Snapdragon 680 v2 Mục 4.6):
 * - Pha 1 (`recordFrame`): Chạy ở MỌI FRAME trên Luồng 1B (< 0.04 ms) để kiểm tra liên tục `maxGap`,
 *   cắt cửa sổ và tính tích phân hình thang tách 2 tam giác khi qua 0.
 * - Pha 2 (`evaluateExactFft`): Chỉ chạy đúng 1 lần/giây (1 Hz) trên Luồng 2 khi đánh giá cửa sổ.
 */
class LazyHeadMotionWindow(
    var window: Double = PhysicalConfig.DEFAULT_WINDOW_SEC,
    val maxGap: Double = PhysicalConfig.DEFAULT_MAX_GAP_SEC,
    val minAmplitude: Double = PhysicalConfig.DEFAULT_HEAD_MOTION_MIN_AMPLITUDE_DEG,
    val config: PhysicalConfig = PhysicalConfig.DEFAULT
) {
    init {
        require(window > 0.0 && maxGap > 0.0 && minAmplitude >= 0.0) {
            "Invalid LazyHeadMotionWindow parameters"
        }
    }

    data class Sample(val time: Double, val pitch: Double)

    private val samples = ArrayDeque<Sample>(config.headMotionInitialCapacity)
    var lastTime: Double? = null
        private set

    @Volatile
    var cachedFreqHz: Double? = null

    @Volatile
    var cachedResHz: Double? = null

    fun reset() {
        samples.clear()
        lastTime = null
        cachedFreqHz = null
        cachedResHz = null
    }

    /**
     * Chạy ở MỌI FRAME trên Luồng 1B (< 0.04 ms).
     * Bảo toàn 100% logic reset cửa sổ và tích phân hình thang của `HeadMotionWindow.process`.
     */
    fun recordFrame(pitch: Double?, timestamp: Double): HeadMotionMetrics {
        require(timestamp.isFinite() && (lastTime == null || timestamp > lastTime!!)) {
            "Timestamps must be finite and strictly increasing: last=$lastTime, next=$timestamp"
        }
        if (lastTime != null && timestamp - lastTime!! > maxGap) {
            samples.clear()
            cachedFreqHz = null
            cachedResHz = null
        }
        lastTime = timestamp

        if (pitch == null || !pitch.isFinite()) {
            samples.clear()
            cachedFreqHz = null
            cachedResHz = null
        } else {
            samples.addLast(Sample(timestamp, pitch))
        }

        val cutoff = timestamp - window
        while (samples.size > 1 && samples.elementAt(1).time <= cutoff) {
            samples.removeFirst()
        }

        if (samples.size < config.headMotionMinSamplesForIntegral) {
            return HeadMotionMetrics(null, null, null, null, 0.0)
        }

        val first = samples.first()
        val start = max(first.time, cutoff)
        val observed = timestamp - start
        if (observed <= config.windowRatioMinValidEpsilon) {
            return HeadMotionMetrics(null, null, null, null, 0.0)
        }

        if (observed < window - config.headMotionWindowBoundaryEpsilon ||
            samples.size < config.headMotionMinSamplesForFft
        ) {
            cachedFreqHz = null
            cachedResHz = null
        }

        // Tích phân hình thang trên các đoạn trong [start, timestamp]
        var sumMean = 0.0
        var sumAmp = 0.0
        val iter = samples.iterator()
        var prevSample = iter.next()

        while (iter.hasNext()) {
            val currSample = iter.next()
            if (currSample.time <= start) {
                prevSample = currSample
                continue
            }
            val segStart = if (prevSample.time < start) start else prevSample.time
            val leftVal = if (prevSample.time < start) {
                val ratio = (start - prevSample.time) / (currSample.time - prevSample.time)
                prevSample.pitch + ratio * (currSample.pitch - prevSample.pitch)
            } else {
                prevSample.pitch
            }
            val rightVal = currSample.pitch
            val dt = currSample.time - segStart

            sumMean += 0.5 * (leftVal + rightVal) * dt
            val absL = abs(leftVal)
            val absR = abs(rightVal)
            // Tích phân |pitch| phải tách hai tam giác khi đoạn nội suy đi qua 0
            val amp = if ((leftVal < 0.0 && rightVal > 0.0) || (leftVal > 0.0 && rightVal < 0.0)) {
                (leftVal * leftVal + rightVal * rightVal) / (2.0 * (absL + absR))
            } else {
                0.5 * (absL + absR)
            }
            sumAmp += amp * dt
            prevSample = currSample
        }

        return HeadMotionMetrics(
            pitchMeanDeg = sumMean / observed,
            pitchMeanAmplitudeDeg = sumAmp / observed,
            headMotionFrequencyHz = cachedFreqHz,
            headMotionResolutionHz = cachedResHz,
            headMotionObservedSec = observed
        )
    }

    /**
     * Đóng gói bản chụp chỉ khi chuẩn bị giao việc cho Luồng 2 (1 lần/giây).
     */
    fun snapshotForFft(): PitchSamplesSnapshot? {
        val now = lastTime ?: return null
        val n = samples.size
        if (n < config.headMotionMinSamplesForFft) return null
        val cutoff = now - window
        val start = max(samples.first().time, cutoff)
        if (now - start < window - config.headMotionWindowBoundaryEpsilon) return null

        val times = DoubleArray(n)
        val values = DoubleArray(n)
        var i = 0
        for (s in samples) {
            times[i] = s.time
            values[i] = s.pitch
            i++
        }
        return PitchSamplesSnapshot(
            times = times,
            values = values,
            timestamp = now,
            windowSec = window,
            minAmplitudeDeg = minAmplitude,
            minFftSamples = config.headMotionMinSamplesForFft,
            windowBoundaryEpsilon = config.headMotionWindowBoundaryEpsilon
        )
    }

    companion object {
        /**
         * Chạy trên Luồng 2 (HeavyComputeWorker) đúng 1 lần/giây (1 Hz).
         * Cho kết quả trùng khớp 100% với `HeadMotionWindow.process` trong `temporal_metrics.py`.
         */
        fun evaluateExactFft(snap: PitchSamplesSnapshot): HeadMotionMetrics {
            val times = snap.times
            val values = snap.values
            val n = times.size
            if (n < snap.minFftSamples) {
                return HeadMotionMetrics(null, null, null, null, 0.0)
            }
            val cutoff = snap.timestamp - snap.windowSec
            val start = max(times[0], cutoff)
            val observed = snap.timestamp - start
            if (observed < snap.windowSec - snap.windowBoundaryEpsilon) {
                return HeadMotionMetrics(null, null, null, null, observed.coerceAtLeast(0.0))
            }

            // 1. Tính median(diff(times))
            val diffs = DoubleArray(n - 1)
            for (i in 0 until n - 1) {
                diffs[i] = times[i + 1] - times[i]
            }
            Arrays.sort(diffs)
            val mid = diffs.size / 2
            val medianDt = if (diffs.size % 2 == 1) {
                diffs[mid]
            } else {
                0.5 * (diffs[mid - 1] + diffs[mid])
            }
            if (medianDt <= 0.0 || !medianDt.isFinite()) {
                return HeadMotionMetrics(null, null, null, null, observed)
            }

            val count = max(snap.minFftSamples, (observed / medianDt).roundToInt() + 1)
            val dt = observed / (count - 1).toDouble()
            val resolutionHz = 1.0 / (count.toDouble() * dt)

            // 2. Nội suy tuyến tính np.interp(grid, times, values)
            val signal = DoubleArray(count)
            var cursor = 0
            var minVal = Double.POSITIVE_INFINITY
            var maxVal = Double.NEGATIVE_INFINITY
            var sum = 0.0
            var absSum = 0.0

            for (i in 0 until count) {
                val g = if (i == count - 1) snap.timestamp else start + i * dt
                while (cursor < n - 2 && times[cursor + 1] < g) {
                    cursor++
                }
                val t0 = times[cursor]
                val t1 = times[cursor + 1]
                val v0 = values[cursor]
                val v1 = values[cursor + 1]
                val interpVal = when {
                    g <= t0 -> v0
                    g >= t1 -> v1
                    t1 - t0 <= 0.0 -> v0
                    else -> v0 + (g - t0) / (t1 - t0) * (v1 - v0)
                }
                signal[i] = interpVal
                if (interpVal < minVal) minVal = interpVal
                if (interpVal > maxVal) maxVal = interpVal
                sum += interpVal
                absSum += abs(interpVal)
            }

            val mean = sum / count.toDouble()
            val meanAmp = absSum / count.toDouble()
            if (maxVal - minVal <= snap.minAmplitudeDeg) {
                return HeadMotionMetrics(
                    pitchMeanDeg = mean,
                    pitchMeanAmplitudeDeg = meanAmp,
                    headMotionFrequencyHz = 0.0,
                    headMotionResolutionHz = resolutionHz,
                    headMotionObservedSec = observed,
                )
            }

            // 3. Trừ trung bình & nhân cửa sổ Hann (np.hanning)
            val fftData = DoubleArray(count)
            val twoPiOverNMinus1 = 2.0 * Math.PI / (count - 1).toDouble()
            for (i in 0 until count) {
                val hann = 0.5 - 0.5 * cos(twoPiOverNMinus1 * i)
                fftData[i] = (signal[i] - mean) * hann
            }

            // 4. Biến đổi rfft 1D bằng JTransforms
            val fft = DoubleFFT_1D(count.toLong())
            fft.realForward(fftData)

            // 5. Tìm đỉnh phổ năng lượng |X[k]|^2 từ k = 1 .. count / 2
            // Quy ước đóng gói của JTransforms DoubleFFT_1D.realForward(a):
            // - Với k < count / 2: Re[k] = a[2*k], Im[k] = a[2*k + 1]
            // - Với k == count / 2 và count chẵn: Re[n/2] = a[1], Im[n/2] = 0.0
            // - Với k == count / 2 và count lẻ: Re[(n-1)/2] = a[n-1], Im[(n-1)/2] = a[1]
            val half = count / 2
            var peakBin = 1
            var maxPower = -1.0
            for (k in 1..half) {
                val re: Double
                val im: Double
                if (k < half) {
                    re = fftData[2 * k]
                    im = fftData[2 * k + 1]
                } else if (count % 2 == 0) {
                    re = fftData[1]
                    im = 0.0
                } else {
                    re = fftData[count - 1]
                    im = fftData[1]
                }
                val power = re * re + im * im
                if (power > maxPower) {
                    maxPower = power
                    peakBin = k
                }
            }

            val dominantFreqHz = peakBin.toDouble() / (count.toDouble() * dt)
            return HeadMotionMetrics(
                pitchMeanDeg = mean,
                pitchMeanAmplitudeDeg = meanAmp,
                headMotionFrequencyHz = dominantFreqHz,
                headMotionResolutionHz = resolutionHz,
                headMotionObservedSec = observed,
            )
        }
    }
}
