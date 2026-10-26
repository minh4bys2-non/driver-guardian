package com.example.driverguardian.ai.monitoring.camera

import android.util.Log
import org.opencv.android.OpenCVLoader
import java.util.concurrent.atomic.AtomicBoolean

object OpenCvInitializer {
    private const val TAG = "OpenCvInitializer"
    private val isLoaded = AtomicBoolean(false)

    fun ensureInitialized(): Boolean {
        if (isLoaded.get()) return true
        val success = try {
            OpenCVLoader.initLocal().also { ok ->
                if (ok) {
                    Log.i(TAG, "OpenCV native library loaded successfully")
                } else {
                    Log.e(TAG, "OpenCV native library initialization returned false")
                }
            }
        } catch (error: Throwable) {
            Log.e(TAG, "Failed to load OpenCV native library", error)
            false
        }
        if (success) {
            isLoaded.set(true)
        }
        return success
    }
}
