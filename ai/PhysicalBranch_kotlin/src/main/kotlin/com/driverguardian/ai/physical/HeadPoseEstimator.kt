package com.driverguardian.ai.physical

import kotlin.math.*

/**
 * Head pose estimator interface for 2D and 3D facial landmarks.
 */
interface HeadPoseEstimator {
    fun estimate(landmarks2D: Array<Point2D>): HeadPoseAngles
}

/**
 * 3D Model landmarks and camera geometry reference.
 * Corresponds to HeadPoseEstimator in ai/PhysicalBranch/head_pose_estimation.py.
 */
object HeadPoseConstants {
    /** MediaPipe landmark indices used for head pose estimation: nose, chin, left eye, right eye, left mouth, right mouth */
    val LANDMARKS = intArrayOf(1, 152, 263, 33, 291, 61)

    /** 3D canonical facial model coordinates (in millimeters) */
    val MODEL = arrayOf(
        Point3D(0.0, 0.0, 0.0),          // 1: Nose tip
        Point3D(0.0, 330.0, 65.0),       // 152: Chin
        Point3D(225.0, -170.0, 135.0),   // 263: Left eye outer corner
        Point3D(-225.0, -170.0, 135.0),  // 33: Right eye outer corner
        Point3D(150.0, 150.0, 125.0),    // 291: Left mouth corner
        Point3D(-150.0, 150.0, 125.0)    // 61: Right mouth corner
    )

    /**
     * Alternative model coordinates with Z matching camera optical axis convention
     * (used in PitchFSM.py).
     */
    val MODEL_POINTS_3D = arrayOf(
        Point3D(0.0, 0.0, 0.0),          // Nose tip
        Point3D(0.0, 330.0, -65.0),      // Chin
        Point3D(225.0, -170.0, -135.0),  // Left eye outer corner
        Point3D(-225.0, -170.0, -135.0), // Right eye outer corner
        Point3D(150.0, 150.0, -125.0),   // Left mouth corner
        Point3D(-150.0, 150.0, -125.0)   // Right mouth corner
    )

    /**
     * Constructs the standard pinhole camera intrinsic matrix K:
     * [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
     * where fx = fy = imageWidth, cx = imageWidth / 2, cy = imageHeight / 2.
     */
    fun getCameraMatrix(width: Double, height: Double): Array<DoubleArray> = arrayOf(
        doubleArrayOf(width, 0.0, width / 2.0),
        doubleArrayOf(0.0, width, height / 2.0),
        doubleArrayOf(0.0, 0.0, 1.0)
    )
}

/**
 * Pure Kotlin Direct 3D Geometry Head Pose Estimator.
 *
 * Implements the algorithm from PitchFSM.py:
 * Computes an orthonormal head coordinate frame (ux, uy, uz) directly from 3D facial landmarks
 * (nose tip, chin, forehead, left eye, right eye) without requiring external native C++ dependencies.
 */
class DirectGeometryHeadPoseEstimator(
    val imageWidth: Double,
    val imageHeight: Double
) {
    init {
        require(imageWidth > 0.0 && imageHeight > 0.0) { "Image dimensions must be positive" }
    }

    /**
     * Computes head pose Euler angles from MediaPipe 3D landmark points.
     *
     * @param landmarks Normalized 3D landmarks (x: 0..1, y: 0..1, z: relative depth).
     * @return HeadPoseAngles (pitch, yaw, roll in degrees).
     */
    fun estimate(landmarks: Map<Int, Point3D>): HeadPoseAngles {
        fun getPixel3D(idx: Int): Point3D {
            val lm = landmarks[idx] ?: throw IllegalArgumentException("Missing required landmark index $idx")
            return Point3D(lm.x * imageWidth, lm.y * imageHeight, lm.z * imageWidth)
        }

        val pNose = getPixel3D(1)
        val pChin = getPixel3D(152)
        val pForehead = getPixel3D(10)
        val pLeftEye = getPixel3D(263)
        val pRightEye = getPixel3D(33)

        // u_x: Vector from right eye to left eye
        val vx = pLeftEye - pRightEye
        val ux = vx.normalized()

        // u_y: Vector from forehead to chin
        val vy = pChin - pForehead
        val uyRaw = vy.normalized()

        // u_z: Cross product of u_x and u_y (pointing outward)
        val uz = ux.cross(uyRaw).normalized()

        // Re-orthogonalize u_y = u_z x u_x
        val uy = uz.cross(ux).normalized()

        // Construct 3x3 rotation matrix R = [ux, uy, uz] (columns)
        val R = arrayOf(
            doubleArrayOf(ux.x, uy.x, uz.x),
            doubleArrayOf(ux.y, uy.y, uz.y),
            doubleArrayOf(ux.z, uy.z, uz.z)
        )

        return MathUtils.rotationMatrixToEulerAngles(R)
    }

    /**
     * Overload taking full landmarks array of Point3D indexed by MediaPipe landmark ID.
     */
    fun estimate(landmarks: Array<Point3D>): HeadPoseAngles {
        require(landmarks.size >= 264) { "Landmarks array must contain at least 264 points, got ${landmarks.size}" }
        val map = HashMap<Int, Point3D>(5)
        for (idx in intArrayOf(1, 152, 10, 263, 33)) {
            map[idx] = landmarks[idx]
        }
        return estimate(map)
    }
}

/**
 * General HeadPoseEstimator supporting pure Kotlin geometric estimation
 * or plugging in OpenCV Calib3d solvePnP when available.
 */
class DefaultHeadPoseEstimator(
    val imageWidth: Double,
    val imageHeight: Double,
    cameraMatrix: Array<DoubleArray>? = null,
    distortionCoeffs: DoubleArray? = null
) : HeadPoseEstimator {

    val cameraMatrix: Array<DoubleArray> = cameraMatrix ?: HeadPoseConstants.getCameraMatrix(imageWidth, imageHeight)
    val distortion: DoubleArray = distortionCoeffs ?: DoubleArray(4)

    var rotationMatrix: Array<DoubleArray>? = null
    var translationVector: DoubleArray? = null

    init {
        require(imageWidth > 0.0 && imageHeight > 0.0) { "Image dimensions must be positive" }
        require(this.cameraMatrix.size == 3 && this.cameraMatrix[0].size == 3) { "Invalid camera matrix" }
    }

    /**
     * Estimates head pose angles from 6 2D facial landmarks:
     * [0: nose(1), 1: chin(152), 2: left eye(263), 3: right eye(33), 4: left mouth(291), 5: right mouth(61)].
     */
    override fun estimate(landmarks2D: Array<Point2D>): HeadPoseAngles {
        require(landmarks2D.size == 6) { "Expected six finite 2D landmarks, got ${landmarks2D.size}" }
        for (p in landmarks2D) {
            require(p.x.isFinite() && p.y.isFinite()) { "Landmark points must be finite" }
        }

        // Geometric approximation using eye line, nose projection, and symmetry
        val nose = landmarks2D[0]
        val chin = landmarks2D[1]
        val leftEye = landmarks2D[2]
        val rightEye = landmarks2D[3]
        val leftMouth = landmarks2D[4]
        val rightMouth = landmarks2D[5]

        val eyeCenter = Point2D((leftEye.x + rightEye.x) / 2.0, (leftEye.y + rightEye.y) / 2.0)
        val eyeDist = leftEye.distanceTo(rightEye)
        if (eyeDist < 1e-4) throw IllegalArgumentException("Degenerate landmarks: eye distance too small")

        // Roll: Angle of eye line relative to horizontal
        val rollRad = atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x)
        val rollDeg = Math.toDegrees(rollRad)

        // Yaw: Asymmetry between nose-to-left-eye and nose-to-right-eye
        val distToLeft = hypot(nose.x - leftEye.x, nose.y - leftEye.y)
        val distToRight = hypot(nose.x - rightEye.x, nose.y - rightEye.y)
        val asymmetry = (distToRight - distToLeft) / eyeDist
        val yawDeg = (asymmetry * 55.0).coerceIn(-90.0, 90.0)

        // Pitch: Ratio of eye-to-nose vs nose-to-chin distance
        val eyeToNose = hypot(eyeCenter.x - nose.x, eyeCenter.y - nose.y)
        val noseToChin = hypot(chin.x - nose.x, chin.y - nose.y)
        val expectedRatio = 0.72 // Neutral face ratio
        val ratio = if (noseToChin > 1e-4) eyeToNose / noseToChin else expectedRatio
        val pitchDeg = ((ratio - expectedRatio) * 60.0).coerceIn(-90.0, 90.0)

        return HeadPoseAngles(pitch = pitchDeg, yaw = yawDeg, roll = rollDeg)
    }

    /**
     * Projects 3D canonical axes onto image plane coordinates for debugging/visualization.
     * Returns origin point and (endPoint, label) pairs for X (red), Y (green), Z (blue).
     */
    fun projectAxes(
        R: Array<DoubleArray>,
        t: DoubleArray,
        axisLength: Double = 100.0
    ): Pair<Point2D, List<Pair<Point2D, String>>> {
        val origin = MathUtils.projectPoint(Point3D(0.0, 0.0, 0.0), R, t, cameraMatrix)
        val axisX = MathUtils.projectPoint(Point3D(axisLength, 0.0, 0.0), R, t, cameraMatrix)
        val axisY = MathUtils.projectPoint(Point3D(0.0, axisLength, 0.0), R, t, cameraMatrix)
        val axisZ = MathUtils.projectPoint(Point3D(0.0, 0.0, -axisLength), R, t, cameraMatrix)

        return Pair(
            origin,
            listOf(
                Pair(axisX, "X (Pitch)"),
                Pair(axisY, "Y (Yaw)"),
                Pair(axisZ, "Z (Roll)")
            )
        )
    }
}
