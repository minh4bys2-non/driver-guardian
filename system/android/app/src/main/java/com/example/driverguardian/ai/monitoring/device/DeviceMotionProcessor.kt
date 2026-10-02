package com.example.driverguardian.ai.monitoring.device

class DeviceMotionProcessor(private val config: DeviceMotionConfig = DeviceMotionConfig()) {
    private val samples = mutableListOf<DeviceOrientation>()
    private var neutral: DeviceOrientation? = null
    private var lastTimestampSec: Double? = null

    fun process(orientation: DeviceOrientation): DeviceMotionMetrics? {
        if (!orientation.timestampSec.isFinite() || lastTimestampSec?.let { orientation.timestampSec <= it } == true) return null
        lastTimestampSec = orientation.timestampSec
        if (orientation.pitchDeg == null || orientation.rollDeg == null ||
            !orientation.pitchDeg.isFinite() || !orientation.rollDeg.isFinite() ||
            orientation.yawDeg?.isFinite() == false
        ) return markUnavailableInternal(orientation.timestampSec, orientation.source, orientation.accuracy)

        if (neutral == null) {
            samples += orientation
            val elapsed = samples.last().timestampSec - samples.first().timestampSec
            if (samples.size >= config.minimumCalibrationSamples && elapsed >= config.calibrationDurationSec) {
                neutral = DeviceOrientation(
                    timestampSec = orientation.timestampSec,
                    pitchDeg = wrappedMedian(samples.mapNotNull { it.pitchDeg }),
                    rollDeg = wrappedMedian(samples.mapNotNull { it.rollDeg }),
                    yawDeg = samples.mapNotNull { it.yawDeg }.takeIf { it.size == samples.size }?.let(::wrappedMedian),
                    accuracy = orientation.accuracy,
                    source = orientation.source,
                )
                samples.clear()
            }
        }
        val n = neutral
        return DeviceMotionMetrics(
            timestampSec = orientation.timestampSec,
            source = orientation.source,
            calibrationState = if (n == null) DeviceCalibrationState.CALIBRATING else DeviceCalibrationState.AVAILABLE,
            rawPitchDeg = orientation.pitchDeg,
            rawRollDeg = orientation.rollDeg,
            rawYawDeg = orientation.yawDeg,
            deltaPitchDeg = n?.pitchDeg?.let { OrientationMath.shortestDeltaDegrees(it, orientation.pitchDeg) },
            deltaRollDeg = n?.rollDeg?.let { OrientationMath.shortestDeltaDegrees(it, orientation.rollDeg) },
            deltaYawDeg = if (orientation.yawDeg != null) n?.yawDeg?.let { OrientationMath.shortestDeltaDegrees(it, orientation.yawDeg) } else null,
            accuracy = orientation.accuracy,
        )
    }

    fun markUnavailable(timestampSec: Double): DeviceMotionMetrics {
        if (lastTimestampSec == null || timestampSec > lastTimestampSec!!) lastTimestampSec = timestampSec
        return markUnavailableInternal(timestampSec, DeviceSensorSource.UNAVAILABLE, null)
    }

    fun reset() { samples.clear(); neutral = null; lastTimestampSec = null }

    private fun markUnavailableInternal(timestampSec: Double, source: DeviceSensorSource, accuracy: Int?) =
        DeviceMotionMetrics(timestampSec, source, DeviceCalibrationState.UNAVAILABLE, null, null, null, null, null, null, accuracy)

    private fun wrappedMedian(values: List<Double>): Double {
        val reference = values.first()
        val unwrapped = values.map { reference + OrientationMath.shortestDeltaDegrees(reference, it) }.sorted()
        val middle = unwrapped.size / 2
        val median = if (unwrapped.size % 2 == 1) unwrapped[middle] else (unwrapped[middle - 1] + unwrapped[middle]) / 2.0
        return OrientationMath.normalizeDegrees(median)
    }
}
