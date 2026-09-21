package com.driverguardian.ai

/**
 * AIResult is the standardized data transfer object capturing all facial,
 * temporal, and machine learning drowsiness metrics produced by the AI module.
 *
 * Corresponds to AIResult dataclass in ai/interface.py.
 */
data class AIResult(
    /** Thời điểm đo, tính bằng giây từ đầu phiên/video. */
    val timestampSec: Double,

    /** Cửa sổ trượt tính tần suất và PERCLOS, đơn vị giây. */
    val windowSec: Double = 60.0,

    /** Tỷ lệ hình học của mắt (EAR), phản ánh độ mở mắt. */
    val ear: Double? = null,

    /** Tần suất nháy mắt, lần/phút. */
    val blinkRatePerMin: Double? = null,

    /** Thời gian nhắm mắt liên tục hiện tại, ms. */
    val eyeClosureDurationMs: Double? = null,

    /** Phần trăm thời gian mắt đóng ít nhất 80% (P80), 0–100. */
    val perclosPct: Double? = null,

    /** Tỷ lệ hình học của miệng (MAR), phản ánh độ mở miệng. */
    val mar: Double? = null,

    /** Thời gian mở miệng liên tục hiện tại, ms; dùng nhận diện ngáp. */
    val mouthOpenDurationMs: Double? = null,

    /** Góc cúi/ngửa đầu so với tư thế nhìn thẳng đã hiệu chuẩn, độ. */
    val pitchDeg: Double? = null,

    /** Thời gian pitch liên tục vượt dải bình thường hiện tại, ms. */
    val pitchDeviationDurationMs: Double? = null,

    /** Tần suất pitch vượt dải bình thường, lần/phút. */
    val pitchDeviationRatePerMin: Double? = null,

    /** Xác suất buồn ngủ từ LSTM sau softmax, 0–1. */
    val lstmDrowsinessProbability: Double? = null,

    /** Mức cần cảnh báo tổng hợp từ các thang đo và LSTM: 0 = không cần, 1 = cao nhất. */
    val warningScore: Double? = null
) {
    /**
     * Converts the AIResult instance to a Map with snake_case keys matching Python's asdict(self).
     */
    fun toMap(): Map<String, Any?> = linkedMapOf(
        "timestamp_sec" to timestampSec,
        "window_sec" to windowSec,
        "ear" to ear,
        "blink_rate_per_min" to blinkRatePerMin,
        "eye_closure_duration_ms" to eyeClosureDurationMs,
        "perclos_pct" to perclosPct,
        "mar" to mar,
        "mouth_open_duration_ms" to mouthOpenDurationMs,
        "pitch_deg" to pitchDeg,
        "pitch_deviation_duration_ms" to pitchDeviationDurationMs,
        "pitch_deviation_rate_per_min" to pitchDeviationRatePerMin,
        "lstm_drowsiness_probability" to lstmDrowsinessProbability,
        "warning_score" to warningScore
    )

    /**
     * Serializes to JSON string with snake_case keys, disallowing NaN/Infinity
     * equivalent to Python's json.dumps(asdict(self), allow_nan=False).
     */
    fun toJson(): String {
        fun formatVal(name: String, v: Double?): String {
            if (v == null) return "null"
            if (!v.isFinite()) throw IllegalArgumentException("Out of range float values are not JSON compliant: $name=$v")
            return if (v.compareTo(v.toLong().toDouble()) == 0) {
                "${v.toLong()}.0"
            } else {
                v.toString()
            }
        }

        if (!timestampSec.isFinite()) throw IllegalArgumentException("Out of range float values are not JSON compliant: timestampSec=$timestampSec")
        if (!windowSec.isFinite()) throw IllegalArgumentException("Out of range float values are not JSON compliant: windowSec=$windowSec")

        val sb = StringBuilder()
        sb.append("{")
        sb.append("\"timestamp_sec\": ").append(formatVal("timestamp_sec", timestampSec)).append(", ")
        sb.append("\"window_sec\": ").append(formatVal("window_sec", windowSec)).append(", ")
        sb.append("\"ear\": ").append(formatVal("ear", ear)).append(", ")
        sb.append("\"blink_rate_per_min\": ").append(formatVal("blink_rate_per_min", blinkRatePerMin)).append(", ")
        sb.append("\"eye_closure_duration_ms\": ").append(formatVal("eye_closure_duration_ms", eyeClosureDurationMs)).append(", ")
        sb.append("\"perclos_pct\": ").append(formatVal("perclos_pct", perclosPct)).append(", ")
        sb.append("\"mar\": ").append(formatVal("mar", mar)).append(", ")
        sb.append("\"mouth_open_duration_ms\": ").append(formatVal("mouth_open_duration_ms", mouthOpenDurationMs)).append(", ")
        sb.append("\"pitch_deg\": ").append(formatVal("pitch_deg", pitchDeg)).append(", ")
        sb.append("\"pitch_deviation_duration_ms\": ").append(formatVal("pitch_deviation_duration_ms", pitchDeviationDurationMs)).append(", ")
        sb.append("\"pitch_deviation_rate_per_min\": ").append(formatVal("pitch_deviation_rate_per_min", pitchDeviationRatePerMin)).append(", ")
        sb.append("\"lstm_drowsiness_probability\": ").append(formatVal("lstm_drowsiness_probability", lstmDrowsinessProbability)).append(", ")
        sb.append("\"warning_score\": ").append(formatVal("warning_score", warningScore))
        sb.append("}")
        return sb.toString()
    }

    companion object {
        /**
         * Parses an AIResult from a map with either camelCase or snake_case keys.
         */
        fun fromMap(map: Map<String, Any?>): AIResult {
            fun num(vararg keys: String): Double? {
                for (k in keys) {
                    if (map.containsKey(k)) {
                        val v = map[k] ?: return null
                        return when (v) {
                            is Number -> v.toDouble()
                            is String -> v.toDoubleOrNull()
                            else -> null
                        }
                    }
                }
                return null
            }

            val ts = num("timestamp_sec", "timestampSec")
                ?: throw IllegalArgumentException("Missing required field 'timestamp_sec'")
            val win = num("window_sec", "windowSec") ?: 60.0

            return AIResult(
                timestampSec = ts,
                windowSec = win,
                ear = num("ear"),
                blinkRatePerMin = num("blink_rate_per_min", "blinkRatePerMin"),
                eyeClosureDurationMs = num("eye_closure_duration_ms", "eyeClosureDurationMs"),
                perclosPct = num("perclos_pct", "perclosPct"),
                mar = num("mar"),
                mouthOpenDurationMs = num("mouth_open_duration_ms", "mouthOpenDurationMs"),
                pitchDeg = num("pitch_deg", "pitchDeg"),
                pitchDeviationDurationMs = num("pitch_deviation_duration_ms", "pitchDeviationDurationMs"),
                pitchDeviationRatePerMin = num("pitch_deviation_rate_per_min", "pitchDeviationRatePerMin"),
                lstmDrowsinessProbability = num("lstm_drowsiness_probability", "lstmDrowsinessProbability"),
                warningScore = num("warning_score", "warningScore")
            )
        }
    }
}
