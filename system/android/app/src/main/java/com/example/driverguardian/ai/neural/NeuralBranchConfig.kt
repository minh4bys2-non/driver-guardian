package com.example.driverguardian.ai.neural

/**
 * Cấu hình tập trung duy nhất cho toàn bộ nhánh học sâu & tiền xử lý ảnh (`ai.neural`).
 * Tuyệt đối không hardcode các thông số letterbox, NeuralBuffer hay ONNX Runtime trong các class xử lý.
 */
data class NeuralBranchConfig(
    // 1. Cấu hình Model ONNX trên ARMv8.0-A (Snapdragon 680)
    // Ưu tiên INT8 hoặc FP32, tuyệt đối tránh FP16 trên CPU ARMv8.0-A
    val primaryModelAssetPath: String = DEFAULT_PRIMARY_MODEL_ASSET_PATH,
    val fallbackModelAssetPath: String = DEFAULT_FALLBACK_MODEL_ASSET_PATH,
    val inputVideoTensorName: String = DEFAULT_INPUT_VIDEO_TENSOR_NAME,
    val inputP4OldTensorName: String = DEFAULT_INPUT_P4_OLD_TENSOR_NAME,
    val outputProbTensorName: String = DEFAULT_OUTPUT_PROB_TENSOR_NAME,
    val outputP4FeaturesTensorName: String = DEFAULT_OUTPUT_P4_FEATURES_TENSOR_NAME,
    val drowsyClassIndex: Int = DEFAULT_DROWSY_CLASS_INDEX,

    // 2. Cấu hình Luồng C++ ONNX Runtime (Luồng 3: ONNX Intra-Op Pool)
    val intraOpNumThreads: Int = DEFAULT_INTRA_OP_NUM_THREADS,
    val interOpNumThreads: Int = DEFAULT_INTER_OP_NUM_THREADS,
    val allowSpinningConfigKey: String = DEFAULT_ALLOW_SPINNING_CONFIG_KEY,
    val allowSpinningConfigValue: String = DEFAULT_ALLOW_SPINNING_CONFIG_VALUE,
    val xnnpackThreadsKey: String = DEFAULT_XNNPACK_THREADS_KEY,
    val enableXnnpack: Boolean = DEFAULT_ENABLE_XNNPACK,

    // 3. Cấu hình Letterbox & Kích thước Tensor
    val imageSize: Int = DEFAULT_IMAGE_SIZE,
    val channelsRgb: Int = DEFAULT_CHANNELS_RGB,
    val channelsRgba: Int = DEFAULT_CHANNELS_RGBA,
    val letterboxPadValue: Int = DEFAULT_LETTERBOX_PAD_VALUE,
    val defaultSourceWidth: Int = DEFAULT_SOURCE_WIDTH,
    val defaultSourceHeight: Int = DEFAULT_SOURCE_HEIGHT,

    // 4. Cấu hình Lưới Thời Gian NeuralBuffer (5.0 FPS, 10s = 50 frames)
    val neuralCount: Int = DEFAULT_NEURAL_COUNT,
    val neuralFps: Double = DEFAULT_NEURAL_FPS,
    val neuralSec: Double = DEFAULT_NEURAL_SEC,
    val maxGapSec: Double = DEFAULT_MAX_GAP_SEC,
    val gridEpsilonSec: Double = DEFAULT_GRID_EPSILON_SEC,

    // 5. Cấu hình Bộ Đệm Đặc Trưng P4 (`p4_old` Cache)
    val p4Channels: Int = DEFAULT_P4_CHANNELS,
    val p4Height: Int = DEFAULT_P4_HEIGHT,
    val p4Width: Int = DEFAULT_P4_WIDTH,
) {
    val planeSize: Int get() = imageSize * imageSize
    val chwBytesPerFrame: Int get() = channelsRgb * planeSize
    val p4ElementsPerFrame: Int get() = p4Channels * p4Height * p4Width
    val defaultSourceRgbaBytes: Int get() = defaultSourceWidth * defaultSourceHeight * channelsRgba

    init {
        require(imageSize > 0 && channelsRgb == 3 && channelsRgba == 4) { "Invalid image or channel dimensions" }
        require(neuralCount >= 1 && neuralFps > 0.0 && neuralSec > 0.0 && maxGapSec > 0.0) {
            "Invalid NeuralBuffer temporal configuration"
        }
        require(intraOpNumThreads >= 1 && interOpNumThreads >= 1) { "Invalid ONNX thread configuration" }
        require(p4Channels > 0 && p4Height > 0 && p4Width > 0) { "Invalid P4 feature dimensions" }
    }

    companion object {
        const val DEFAULT_PRIMARY_MODEL_ASSET_PATH = "drowsy_classifier_fp32.onnx"
        const val DEFAULT_FALLBACK_MODEL_ASSET_PATH = "drowsy_classifier_int8.onnx"
        const val DEFAULT_INPUT_VIDEO_TENSOR_NAME = "video"
        const val DEFAULT_INPUT_P4_OLD_TENSOR_NAME = "p4_old"
        const val DEFAULT_OUTPUT_PROB_TENSOR_NAME = "prob"
        const val DEFAULT_OUTPUT_P4_FEATURES_TENSOR_NAME = "p4_features"
        const val DEFAULT_DROWSY_CLASS_INDEX = 1

        // Giới hạn đúng 2 luồng trên Snapdragon 680 để chừa 2 nhân Kryo 265 Gold cho Luồng 1B và Luồng 4 (UI)
        const val DEFAULT_INTRA_OP_NUM_THREADS = 2
        const val DEFAULT_INTER_OP_NUM_THREADS = 1
        const val DEFAULT_ALLOW_SPINNING_CONFIG_KEY = "session.intra_op.allow_spinning"
        const val DEFAULT_ALLOW_SPINNING_CONFIG_VALUE = "0"
        const val DEFAULT_XNNPACK_THREADS_KEY = "intra_op_num_threads"
        const val DEFAULT_ENABLE_XNNPACK = true

        const val DEFAULT_IMAGE_SIZE = 224
        const val DEFAULT_CHANNELS_RGB = 3
        const val DEFAULT_CHANNELS_RGBA = 4
        const val DEFAULT_LETTERBOX_PAD_VALUE = 114
        const val DEFAULT_SOURCE_WIDTH = 640
        const val DEFAULT_SOURCE_HEIGHT = 480

        const val DEFAULT_NEURAL_COUNT = 50
        const val DEFAULT_NEURAL_FPS = 5.0
        const val DEFAULT_NEURAL_SEC = 10.0
        const val DEFAULT_MAX_GAP_SEC = 2.5
        const val DEFAULT_GRID_EPSILON_SEC = 1e-9

        const val DEFAULT_P4_CHANNELS = 128
        const val DEFAULT_P4_HEIGHT = 14
        const val DEFAULT_P4_W = 14
        const val DEFAULT_P4_WIDTH = 14

        val DEFAULT = NeuralBranchConfig()
    }
}
