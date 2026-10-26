package com.example.driverguardian.ai.monitoring

import android.util.Log

/**
 * Structured multi-thread debug logger with `Thread-*` tags (`tag:Thread-`).
 * Safe to invoke on both Android runtime and JVM unit tests.
 */
object ThreadDebugLogger {

    fun logIngest(message: String) {
        safeDebug(MonitoringConfig.TAG_THREAD_1A_INGEST, message)
    }

    fun warnIngest(message: String) {
        safeWarn(MonitoringConfig.TAG_THREAD_1A_INGEST, message)
    }

    fun logPhysical(message: String) {
        safeDebug(MonitoringConfig.TAG_THREAD_1B_PHYSICAL, message)
    }

    fun infoPhysical(message: String) {
        safeInfo(MonitoringConfig.TAG_THREAD_1B_PHYSICAL, message)
    }

    fun logHeavyWorker(message: String) {
        safeDebug(MonitoringConfig.TAG_THREAD_2_HEAVY_WORKER, message)
    }

    fun infoHeavyWorker(message: String) {
        safeInfo(MonitoringConfig.TAG_THREAD_2_HEAVY_WORKER, message)
    }

    fun logOnnxPool(message: String) {
        safeDebug(MonitoringConfig.TAG_THREAD_3_ONNX_POOL, message)
    }

    fun infoOnnxPool(message: String) {
        safeInfo(MonitoringConfig.TAG_THREAD_3_ONNX_POOL, message)
    }

    fun logUi(message: String) {
        safeDebug(MonitoringConfig.TAG_THREAD_4_UI, message)
    }

    fun infoUi(message: String) {
        safeInfo(MonitoringConfig.TAG_THREAD_4_UI, message)
    }

    private fun safeDebug(tag: String, message: String) {
        runCatching {
            Log.d(tag, "[${Thread.currentThread().name}] $message")
        }
    }

    private fun safeInfo(tag: String, message: String) {
        runCatching {
            Log.i(tag, "[${Thread.currentThread().name}] $message")
        }
    }

    private fun safeWarn(tag: String, message: String) {
        runCatching {
            Log.w(tag, "[${Thread.currentThread().name}] $message")
        }
    }
}
