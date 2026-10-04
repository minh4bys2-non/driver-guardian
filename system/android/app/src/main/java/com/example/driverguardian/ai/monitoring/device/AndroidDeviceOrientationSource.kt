package com.example.driverguardian.ai.monitoring.device

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.SystemClock
import android.view.Surface
import kotlin.math.PI

class AndroidDeviceOrientationSource(context: Context) : SensorEventListener, AutoCloseable {
    private val manager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private var callback: ((DeviceOrientation) -> Unit)? = null
    private var displayRotationProvider: (() -> Int)? = null
    private var accuracy: Int? = null
    private var accelerometer: FloatArray? = null
    private var magnetometer: FloatArray? = null
    private var started = false

    var selectedSource: DeviceSensorSource = DeviceSensorSource.UNAVAILABLE
        private set

    fun start(displayRotationProvider: () -> Int, callback: (DeviceOrientation) -> Unit): Boolean {
        if (started) return selectedSource != DeviceSensorSource.UNAVAILABLE
        this.callback = callback
        this.displayRotationProvider = displayRotationProvider
        val rotation = manager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        val game = manager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
        val accel = manager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val magnetic = manager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
        selectedSource = SensorSelection.choose(rotation != null, game != null, accel != null, magnetic != null)
        started = true
        val registered = when (selectedSource) {
            DeviceSensorSource.ROTATION_VECTOR -> manager.registerListener(this, rotation, SensorManager.SENSOR_DELAY_GAME)
            DeviceSensorSource.GAME_ROTATION_VECTOR -> manager.registerListener(this, game, SensorManager.SENSOR_DELAY_GAME)
            DeviceSensorSource.ACCELEROMETER_MAGNETOMETER ->
                manager.registerListener(this, accel, SensorManager.SENSOR_DELAY_GAME) &&
                    manager.registerListener(this, magnetic, SensorManager.SENSOR_DELAY_GAME)
            DeviceSensorSource.UNAVAILABLE -> false
        }
        if (!registered) {
            manager.unregisterListener(this)
            selectedSource = DeviceSensorSource.UNAVAILABLE
            callback(DeviceOrientation(SystemClock.elapsedRealtimeNanos() / 1e9, null, null, null, null, DeviceSensorSource.UNAVAILABLE))
        }
        return registered
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (!started) return
        val matrix = FloatArray(9)
        when (event.sensor.type) {
            Sensor.TYPE_ROTATION_VECTOR, Sensor.TYPE_GAME_ROTATION_VECTOR ->
                SensorManager.getRotationMatrixFromVector(matrix, event.values)
            Sensor.TYPE_ACCELEROMETER -> {
                accelerometer = event.values.copyOf()
                if (!fallbackMatrix(matrix)) return
            }
            Sensor.TYPE_MAGNETIC_FIELD -> {
                magnetometer = event.values.copyOf()
                if (!fallbackMatrix(matrix)) return
            }
            else -> return
        }
        val remapped = FloatArray(9)
        val (axisX, axisY) = axesForRotation(displayRotationProvider?.invoke() ?: Surface.ROTATION_0)
        if (!SensorManager.remapCoordinateSystem(matrix, axisX, axisY, remapped)) return
        val angles = SensorManager.getOrientation(remapped, FloatArray(3))
        val degrees = 180.0 / PI
        callback?.invoke(
            DeviceOrientation(
                timestampSec = event.timestamp / 1e9,
                pitchDeg = OrientationMath.normalizeDegrees(angles[1] * degrees),
                rollDeg = OrientationMath.normalizeDegrees(angles[2] * degrees),
                yawDeg = OrientationMath.normalizeDegrees(angles[0] * degrees),
                accuracy = accuracy,
                source = selectedSource,
            ),
        )
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) { this.accuracy = accuracy }

    fun stop() {
        if (!started) return
        manager.unregisterListener(this)
        started = false
        callback = null
        displayRotationProvider = null
        accelerometer = null
        magnetometer = null
    }

    override fun close() = stop()

    private fun fallbackMatrix(output: FloatArray): Boolean {
        val gravity = accelerometer ?: return false
        val geomagnetic = magnetometer ?: return false
        return SensorManager.getRotationMatrix(output, null, gravity, geomagnetic)
    }

    private fun axesForRotation(rotation: Int): Pair<Int, Int> = when (rotation) {
        Surface.ROTATION_0, 0 -> SensorManager.AXIS_X to SensorManager.AXIS_Y
        Surface.ROTATION_90, 90 -> SensorManager.AXIS_Y to SensorManager.AXIS_MINUS_X
        Surface.ROTATION_180, 180 -> SensorManager.AXIS_MINUS_X to SensorManager.AXIS_MINUS_Y
        Surface.ROTATION_270, 270 -> SensorManager.AXIS_MINUS_Y to SensorManager.AXIS_X
        else -> SensorManager.AXIS_X to SensorManager.AXIS_Y
    }
}
