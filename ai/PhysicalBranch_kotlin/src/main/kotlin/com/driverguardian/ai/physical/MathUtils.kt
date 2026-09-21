package com.driverguardian.ai.physical

import kotlin.math.*

/**
 * High-performance numerical and mathematical utilities designed for
 * computer vision, statistical learning, and signal processing in pure Kotlin.
 */
object MathUtils {

    /**
     * Computes the p-th percentile of a 1D DoubleArray using linear interpolation,
     * identical to numpy.percentile(x, p, method='linear').
     *
     * @param data Array of finite numbers.
     * @param p Percentile in range [0.0, 100.0].
     */
    fun percentile(data: DoubleArray, p: Double): Double {
        require(data.isNotEmpty()) { "Cannot compute percentile of an empty array" }
        require(p in 0.0..100.0) { "Percentile must be between 0 and 100, got $p" }
        if (data.size == 1) return data[0]

        val sorted = data.clone()
        sorted.sort()

        val n = sorted.size
        val index = (n - 1) * (p / 100.0)
        val k = index.toInt()
        val d = index - k

        if (k >= n - 1) return sorted[n - 1]
        return sorted[k] + d * (sorted[k + 1] - sorted[k])
    }

    /**
     * Computes the median (50th percentile) of a 1D DoubleArray.
     */
    fun median(data: DoubleArray): Double = percentile(data, 50.0)

    /**
     * Computes the arithmetic mean of a 1D DoubleArray.
     */
    fun mean(data: DoubleArray): Double {
        require(data.isNotEmpty()) { "Cannot compute mean of empty array" }
        return data.sum() / data.size
    }

    /**
     * Computes the population variance (ddof = 0) of a 1D DoubleArray,
     * identical to numpy.var(x).
     */
    fun variance(data: DoubleArray, ddof: Int = 0): Double {
        require(data.size > ddof) { "Not enough elements to compute variance with ddof=$ddof" }
        val m = mean(data)
        var sumSq = 0.0
        for (v in data) {
            val diff = v - m
            sumSq += diff * diff
        }
        return sumSq / (data.size - ddof)
    }

    /**
     * Stable Log-Sum-Exp computation over a 1D array:
     * ln(sum(exp(x_i))) = max(x) + ln(sum(exp(x_i - max(x))))
     */
    fun logSumExp(values: DoubleArray): Double {
        if (values.isEmpty()) return Double.NEGATIVE_INFINITY
        var maxVal = values[0]
        for (i in 1 until values.size) {
            if (values[i] > maxVal) maxVal = values[i]
        }
        if (!maxVal.isFinite()) return maxVal

        var sumExp = 0.0
        for (v in values) {
            sumExp += exp(v - maxVal)
        }
        return maxVal + ln(max(sumExp, 1e-300))
    }

    /**
     * Mathematical modulo keeping the result in range [0, divisor).
     */
    fun mod(value: Double, divisor: Double): Double {
        val r = value % divisor
        return if (r < 0) r + divisor else r
    }

    /**
     * Wraps an angle in degrees into [-180, 180).
     * Equivalent to Python's: (val + 180) % 360 - 180
     */
    fun wrapAngle180(angleDeg: Double): Double = mod(angleDeg + 180.0, 360.0) - 180.0

    /**
     * Converts a 3x3 rotation matrix R to Euler angles (pitch, yaw, roll) in degrees.
     * Matches the formula in PitchFSM.py.
     */
    fun rotationMatrixToEulerAngles(R: Array<DoubleArray>): HeadPoseAngles {
        val r21 = R[2][1].coerceIn(-1.0, 1.0)
        val pitchRad = asin(r21)

        val yawRad: Double
        val rollRad: Double

        if (abs(R[2][1]) < 0.9999) {
            yawRad = atan2(-R[2][0], R[2][2])
            rollRad = atan2(-R[0][1], R[1][1])
        } else {
            yawRad = atan2(R[0][2], R[0][0])
            rollRad = 0.0
        }

        return HeadPoseAngles(
            pitch = Math.toDegrees(pitchRad),
            yaw = Math.toDegrees(yawRad),
            roll = Math.toDegrees(rollRad)
        )
    }

    /**
     * Converts a 3D rotation vector (Rodrigues vector) to a 3x3 rotation matrix.
     * Equivalent to cv2.Rodrigues(rvec).
     */
    fun rodriguesVectorToMatrix(rvec: DoubleArray): Array<DoubleArray> {
        val theta = sqrt(rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2])
        val R = Array(3) { DoubleArray(3) }
        if (theta < 1e-12) {
            for (i in 0..2) R[i][i] = 1.0
            return R
        }

        val rx = rvec[0] / theta
        val ry = rvec[1] / theta
        val rz = rvec[2] / theta

        val c = cos(theta)
        val s = sin(theta)
        val c1 = 1.0 - c

        R[0][0] = c + rx * rx * c1
        R[0][1] = rx * ry * c1 - rz * s
        R[0][2] = rx * rz * c1 + ry * s

        R[1][0] = ry * rx * c1 + rz * s
        R[1][1] = c + ry * ry * c1
        R[1][2] = ry * rz * c1 - rx * s

        R[2][0] = rz * rx * c1 - ry * s
        R[2][1] = rz * ry * c1 + rx * s
        R[2][2] = c + rz * rz * c1

        return R
    }

    /**
     * Converts a 3x3 rotation matrix to a 3D rotation vector (Rodrigues vector).
     * Equivalent to cv2.Rodrigues(R).
     */
    fun rotationMatrixToRodriguesVector(R: Array<DoubleArray>): DoubleArray {
        val trace = R[0][0] + R[1][1] + R[2][2]
        val cosTheta = ((trace - 1.0) / 2.0).coerceIn(-1.0, 1.0)
        val theta = acos(cosTheta)

        if (theta < 1e-6) {
            return doubleArrayOf(0.0, 0.0, 0.0)
        }

        val factor = theta / (2.0 * sin(theta))
        return doubleArrayOf(
            (R[2][1] - R[1][2]) * factor,
            (R[0][2] - R[2][0]) * factor,
            (R[1][0] - R[0][1]) * factor
        )
    }

    /**
     * Decomposes a 3x3 rotation matrix using RQ decomposition to extract Euler angles.
     * Equivalent to cv2.RQDecomp3x3(R)[0].
     *
     * Returns HeadPoseAngles(pitch, yaw, roll) in degrees.
     */
    fun rqDecomp3x3(R: Array<DoubleArray>): HeadPoseAngles {
        // Find x-rotation (pitch) to zero out R[2][1]
        var rx = atan2(R[2][1], R[2][2])
        val cX = cos(rx)
        val sX = sin(rx)

        // R1 = R * Qx^T
        val r1_20 = R[2][0]
        val r1_22 = R[2][1] * sX + R[2][2] * cX

        // Find y-rotation (yaw) to zero out R1[2][0]
        var ry = atan2(-r1_20, r1_22)
        val cY = cos(ry)
        val sY = sin(ry)

        val r1_01 = R[0][1] * cX - R[0][2] * sX
        val r1_11 = R[1][1] * cX - R[1][2] * sX
        val r1_00 = R[0][0]
        val r1_10 = R[1][0]
        val r1_02 = R[0][1] * sX + R[0][2] * cX
        val r1_12 = R[1][1] * sX + R[1][2] * cX

        val r2_00 = r1_00 * cY - r1_02 * sY
        val r2_10 = r1_10 * cY - r1_12 * sY

        // Find z-rotation (roll) to zero out R2[1][0]
        var rz = atan2(r2_10, r2_00)

        // Return degrees
        return HeadPoseAngles(
            pitch = Math.toDegrees(rx),
            yaw = Math.toDegrees(ry),
            roll = Math.toDegrees(rz)
        )
    }

    /**
     * Projects a 3D point into 2D camera pixel coordinates using camera intrinsic matrix,
     * rotation matrix, and translation vector.
     */
    fun projectPoint(
        point: Point3D,
        R: Array<DoubleArray>,
        t: DoubleArray,
        cameraMatrix: Array<DoubleArray>
    ): Point2D {
        // X_cam = R * point + t
        val xc = R[0][0] * point.x + R[0][1] * point.y + R[0][2] * point.z + t[0]
        val yc = R[1][0] * point.x + R[1][1] * point.y + R[1][2] * point.z + t[1]
        val zc = R[2][0] * point.x + R[2][1] * point.y + R[2][2] * point.z + t[2]

        val z = if (abs(zc) < 1e-6) 1e-6 else zc
        val u = cameraMatrix[0][0] * (xc / z) + cameraMatrix[0][2]
        val v = cameraMatrix[1][1] * (yc / z) + cameraMatrix[1][2]
        return Point2D(u, v)
    }
}
