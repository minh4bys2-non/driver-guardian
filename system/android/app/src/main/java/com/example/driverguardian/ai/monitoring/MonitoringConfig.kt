package com.example.driverguardian.ai.monitoring

import android.os.Process

/**
 * Concentrated configuration for the Monitoring & Concurrency branch (`ai.monitoring`).
 * Strictly follows `snapdragon_680_v2.md` Section 1, 2, 3 without hardcoding in classes.
 * Note: Thread 5 (`.jsonl` file I/O logger) is intentionally excluded per user specification.
 */
object MonitoringConfig {

    // Logcat tags with `Thread-` prefix (filterable via `tag:Thread-`)
    const val TAG_THREAD_1A_INGEST = "Thread-1A-Ingest"
    const val TAG_THREAD_1B_PHYSICAL = "Thread-1B-Physical"
    const val TAG_THREAD_2_HEAVY_WORKER = "Thread-2-HeavyWorker"
    const val TAG_THREAD_3_ONNX_POOL = "Thread-3-OnnxPool"
    const val TAG_THREAD_4_UI = "Thread-4-UI"

    // Thread names & Android OS priorities (Section 2.1)
    const val THREAD_NAME_1A_INGEST = "CameraIngestThread"
    const val THREAD_NAME_1B_PHYSICAL = "UnifiedPhysicalThread"
    const val THREAD_NAME_2_HEAVY_WORKER = "HeavyComputeWorker"

    const val THREAD_PRIORITY_1A_INGEST = Process.THREAD_PRIORITY_DISPLAY
    const val THREAD_PRIORITY_1B_PHYSICAL = Process.THREAD_PRIORITY_URGENT_DISPLAY
    const val THREAD_PRIORITY_2_HEAVY_WORKER = Process.THREAD_PRIORITY_DEFAULT

    // Periodic logging cadence so 30-60 FPS threads do not flood Logcat
    const val LOG_INGEST_EVERY_N_FRAMES = 60L
    const val LOG_PHYSICAL_EVERY_N_FRAMES = 30L
    const val INGEST_SLOW_WARN_THRESHOLD_MS = 8.0

    // CameraX Hardware & RgbaFramePool (Section 3.1)
    const val TARGET_WIDTH = 640
    const val TARGET_HEIGHT = 480
    const val CHANNELS_RGBA = 4
    const val FRAME_BYTES_RGBA = TARGET_WIDTH * TARGET_HEIGHT * CHANNELS_RGBA
    const val RGBA_FRAME_POOL_SIZE = 3
    const val MONOTONIC_TIMESTAMP_MIN_STEP_SEC = 1e-4

    // BrightnessBalancer (Section 3.2)
    const val DEFAULT_BALANCE_BRIGHTNESS_ENABLED = false
    const val TARGET_LUMA = 70.0
    const val FAST_LUMA_SAMPLE_STEP_BYTES = 16
    const val FAST_LUMA_SKIP_MARGIN = 1.5
    const val GAMMA_MIN = 0.5
    const val MAX_PIXEL_VALUE = 255.0
    const val LUT_SIZE = 256
    const val LUMA_COEFF_R = 77
    const val LUMA_COEFF_G = 150
    const val LUMA_COEFF_B = 29
    const val LUMA_SHIFT_BITS = 8

    // MediaPipe Face Landmarker (Section 3.3)
    const val FACE_LANDMARKER_MODEL_ASSET_PATH = "face_landmarker.task"
    const val FACE_LANDMARKER_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
    const val FACE_LANDMARKER_NUM_FACES = 1
    const val FACE_LANDMARKER_RUNNING_MODE_NAME = "VIDEO"
    const val FACE_LANDMARKER_MIN_DETECTION_CONFIDENCE = 0.6f
    const val FACE_LANDMARKER_MIN_PRESENCE_CONFIDENCE = 0.6f
    const val FACE_LANDMARKER_MIN_TRACKING_CONFIDENCE = 0.6f
    const val FACE_LANDMARKER_OUTPUT_BLENDSHAPES = false
    const val FACE_LANDMARKER_OUTPUT_TRANSFORMATION_MATRIXES = false

    // Window & 1Hz Evaluation Cadence (Section 2.2 & 5.1)
    const val WINDOW_SEC = 60.0
    const val EVALUATION_INTERVAL_SEC = 1.0
    const val EVALUATION_TICK_EPSILON_SEC = 1e-6

    // Sensor Diagnostics
    const val SENSOR_TIMESTAMP_SOURCE_LABEL = "SensorEvent.timestamp / elapsedRealtimeNanos"
}
