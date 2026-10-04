package com.example.driverguardian.ai.monitoring.device

import android.view.Surface

class SafeDisplayRotationProvider(private val lookup: () -> Int?) {
    fun rotation(): Int = runCatching { lookup() }.getOrNull()
        ?.takeIf { it in Surface.ROTATION_0..Surface.ROTATION_270 }
        ?: Surface.ROTATION_0
}
