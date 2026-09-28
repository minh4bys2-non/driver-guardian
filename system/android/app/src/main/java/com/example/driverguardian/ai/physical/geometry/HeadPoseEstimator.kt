package com.example.driverguardian.ai.physical.geometry

import com.example.driverguardian.ai.physical.model.HeadPose
import com.example.driverguardian.ai.physical.model.LandmarkPoint
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Head pose estimator computing 3D Euler angles (Pitch, Yaw, Roll) from 6 2D facial landmarks.
 *
 * Implements a pure-Kotlin non-linear Levenberg-Marquardt PnP solver and analytical RQ decomposition,
 * matching OpenCV's cv2.solvePnPRefineLM and cv2.RQDecomp3x3 without requiring native OpenCV dependencies.
 */
class HeadPoseEstimator(
    val imageWidth: Int,
    val imageHeight: Int
) {
    init {
        require(imageWidth > 0 && imageHeight > 0) { "Image dimensions must be positive" }
    }

    companion object {
        val LANDMARKS = intArrayOf(1, 152, 263, 33, 291, 61)

        val MODEL = arrayOf(
            doubleArrayOf(0.0, 0.0, 0.0),        // 1: Nose tip
            doubleArrayOf(0.0, 330.0, 65.0),     // 152: Chin
            doubleArrayOf(225.0, -170.0, 135.0), // 263: Left eye outer corner
            doubleArrayOf(-225.0, -170.0, 135.0),// 33: Right eye outer corner
            doubleArrayOf(150.0, 150.0, 125.0),  // 291: Left mouth corner
            doubleArrayOf(-150.0, 150.0, 125.0)  // 61: Right mouth corner
        )
    }

    private val fx = imageWidth.toDouble()
    private val fy = imageWidth.toDouble()
    private val cx = imageWidth / 2.0
    private val cy = imageHeight / 2.0

    /**
     * Estimates head pose from the 6 specific 2D landmarks.
     *
     * @param landmarks6 6 landmark points corresponding to LANDMARKS [1, 152, 263, 33, 291, 61]
     * @return HeadPose with pitch, yaw, and roll in degrees.
     * @throws IllegalArgumentException on degenerate or non-finite inputs.
     */
    fun estimate(landmarks6: List<LandmarkPoint>): HeadPose {
        require(landmarks6.size == 6) { "Expected six 2D landmarks" }
        val pts2d = Array(6) { i ->
            val p = landmarks6[i]
            val x = p.x.toDouble()
            val y = p.y.toDouble()
            require(x.isFinite() && y.isFinite()) { "Expected six finite 2D landmarks" }
            doubleArrayOf(x, y)
        }

        // Degeneracy check: matrix rank of centered points must be >= 2
        var meanX = 0.0
        var meanY = 0.0
        for (p in pts2d) {
            meanX += p[0]
            meanY += p[1]
        }
        meanX /= 6.0
        meanY /= 6.0

        var cxx = 0.0
        var cxy = 0.0
        var cyy = 0.0
        for (p in pts2d) {
            val dx = p[0] - meanX
            val dy = p[1] - meanY
            cxx += dx * dx
            cxy += dx * dy
            cyy += dy * dy
        }
        val det = cxx * cyy - cxy * cxy
        val trace = cxx + cyy
        if (trace < 1e-8 || det / (trace * trace) < 1e-5) {
            throw IllegalArgumentException("Degenerate landmarks")
        }

        // Initial depth estimate from outer eye distance (model width = 450.0)
        val eyeDist = hypot(pts2d[2][0] - pts2d[3][0], pts2d[2][1] - pts2d[3][1])
        require(eyeDist > 1e-4) { "Degenerate eye distance" }
        val tzInit = (450.0 * fx) / eyeDist
        val txInit = (pts2d[0][0] - cx) * tzInit / fx
        val tyInit = (pts2d[0][1] - cy) * tzInit / fy

        // Multi-seed Levenberg-Marquardt to avoid local minima
        val seeds = arrayOf(
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(0.2, 0.0, 0.0),
            doubleArrayOf(-0.2, 0.0, 0.0)
        )

        var bestX: DoubleArray? = null
        var bestCost = Double.MAX_VALUE

        for (rInit in seeds) {
            val x = doubleArrayOf(rInit[0], rInit[1], rInit[2], txInit, tyInit, tzInit)
            var lambda = 1e-3

            for (iter in 0 until 35) {
                val r = doubleArrayOf(x[0], x[1], x[2])
                val t = doubleArrayOf(x[3], x[4], x[5])
                val proj = project(r, t)
                val res = DoubleArray(12)
                var cost = 0.0
                for (i in 0 until 6) {
                    val du = proj[i][0] - pts2d[i][0]
                    val dv = proj[i][1] - pts2d[i][1]
                    res[2 * i] = du
                    res[2 * i + 1] = dv
                    cost += du * du + dv * dv
                }

                // Numerical Jacobian (12 x 6)
                val jac = Array(12) { DoubleArray(6) }
                val eps = 1e-6
                for (j in 0 until 6) {
                    val xPlus = x.clone()
                    xPlus[j] += eps
                    val rPlus = doubleArrayOf(xPlus[0], xPlus[1], xPlus[2])
                    val tPlus = doubleArrayOf(xPlus[3], xPlus[4], xPlus[5])
                    val projPlus = project(rPlus, tPlus)
                    for (i in 0 until 6) {
                        jac[2 * i][j] = (projPlus[i][0] - proj[i][0]) / eps
                        jac[2 * i + 1][j] = (projPlus[i][1] - proj[i][1]) / eps
                    }
                }

                // Normal equations: (J^T J + lambda * diag) dx = -J^T res
                val h = Array(6) { DoubleArray(6) }
                val g = DoubleArray(6)
                for (i in 0 until 6) {
                    for (k in 0 until 12) {
                        g[i] += jac[k][i] * res[k]
                        for (l in 0 until 6) {
                            h[i][l] += jac[k][i] * jac[k][l]
                        }
                    }
                }

                for (i in 0 until 6) {
                    h[i][i] += lambda * (h[i][i] + 1e-6)
                }

                val dx = solve6x6(h, g) ?: break

                val xNew = DoubleArray(6) { k -> x[k] - dx[k] }
                val projNew = project(doubleArrayOf(xNew[0], xNew[1], xNew[2]), doubleArrayOf(xNew[3], xNew[4], xNew[5]))
                var costNew = 0.0
                for (i in 0 until 6) {
                    val du = projNew[i][0] - pts2d[i][0]
                    val dv = projNew[i][1] - pts2d[i][1]
                    costNew += du * du + dv * dv
                }

                if (costNew < cost) {
                    System.arraycopy(xNew, 0, x, 0, 6)
                    lambda /= 10.0
                    var maxDx = 0.0
                    for (v in dx) {
                        val av = kotlin.math.abs(v)
                        if (av > maxDx) maxDx = av
                    }
                    if (maxDx < 1e-8) break
                } else {
                    lambda *= 10.0
                    if (lambda > 1e12) break
                }
            }

            val rFinal = doubleArrayOf(x[0], x[1], x[2])
            val tFinal = doubleArrayOf(x[3], x[4], x[5])
            val projFinal = project(rFinal, tFinal)
            var finalCost = 0.0
            for (i in 0 until 6) {
                val du = projFinal[i][0] - pts2d[i][0]
                val dv = projFinal[i][1] - pts2d[i][1]
                finalCost += du * du + dv * dv
            }

            if (finalCost < bestCost) {
                bestCost = finalCost
                bestX = x.clone()
            }
        }

        val sol = bestX ?: throw IllegalArgumentException("No valid pose solution")
        val r = doubleArrayOf(sol[0], sol[1], sol[2])
        val t = doubleArrayOf(sol[3], sol[4], sol[5])

        val rotMatrix = rodrigues(r)

        // Check if all model points are in front of the camera (Z > 0)
        for (i in 0 until 6) {
            val z = MODEL[i][0] * rotMatrix[2][0] + MODEL[i][1] * rotMatrix[2][1] + MODEL[i][2] * rotMatrix[2][2] + t[2]
            if (z <= 0.0) {
                throw IllegalArgumentException("Pose behind camera")
            }
        }

        // Closed-form RQ decomposition matching OpenCV RQDecomp3x3
        val cyRot = hypot(rotMatrix[2][1], rotMatrix[2][2])
        val pitch = Math.toDegrees(atan2(rotMatrix[2][1], rotMatrix[2][2]))
        val yaw = Math.toDegrees(atan2(-rotMatrix[2][0], cyRot))
        val roll = Math.toDegrees(atan2(rotMatrix[1][0], rotMatrix[0][0]))

        return HeadPose(pitch, yaw, roll)
    }

    private fun project(r: DoubleArray, t: DoubleArray): Array<DoubleArray> {
        val rot = rodrigues(r)
        return Array(6) { i ->
            val x3 = MODEL[i][0] * rot[0][0] + MODEL[i][1] * rot[0][1] + MODEL[i][2] * rot[0][2] + t[0]
            val y3 = MODEL[i][0] * rot[1][0] + MODEL[i][1] * rot[1][1] + MODEL[i][2] * rot[1][2] + t[1]
            val z3 = MODEL[i][0] * rot[2][0] + MODEL[i][1] * rot[2][1] + MODEL[i][2] * rot[2][2] + t[2]
            val safeZ = if (z3 != 0.0) z3 else 1e-6
            doubleArrayOf(fx * x3 / safeZ + cx, fy * y3 / safeZ + cy)
        }
    }

    private fun rodrigues(r: DoubleArray): Array<DoubleArray> {
        val th = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
        if (th < 1e-12) {
            return Array(3) { i -> DoubleArray(3) { j -> if (i == j) 1.0 else 0.0 } }
        }
        val kx = r[0] / th
        val ky = r[1] / th
        val kz = r[2] / th
        val c = cos(th)
        val s = sin(th)
        val v = 1.0 - c

        return arrayOf(
            doubleArrayOf(c + kx * kx * v, kx * ky * v - kz * s, kx * kz * v + ky * s),
            doubleArrayOf(ky * kx * v + kz * s, c + ky * ky * v, ky * kz * v - kx * s),
            doubleArrayOf(kz * kx * v - ky * s, kz * ky * v + kx * s, c + kz * kz * v)
        )
    }

    private fun solve6x6(a: Array<DoubleArray>, b: DoubleArray): DoubleArray? {
        val n = 6
        val aug = Array(n) { i -> DoubleArray(n + 1) { j -> if (j < n) a[i][j] else b[i] } }

        for (i in 0 until n) {
            var maxRow = i
            var maxVal = kotlin.math.abs(aug[i][i])
            for (k in i + 1 until n) {
                val av = kotlin.math.abs(aug[k][i])
                if (av > maxVal) {
                    maxVal = av
                    maxRow = k
                }
            }
            if (maxVal < 1e-14) return null
            val temp = aug[i]
            aug[i] = aug[maxRow]
            aug[maxRow] = temp

            val pivot = aug[i][i]
            for (j in i until n + 1) {
                aug[i][j] /= pivot
            }

            for (k in 0 until n) {
                if (k != i) {
                    val factor = aug[k][i]
                    for (j in i until n + 1) {
                        aug[k][j] -= factor * aug[i][j]
                    }
                }
            }
        }

        return DoubleArray(n) { i -> aug[i][n] }
    }
}
