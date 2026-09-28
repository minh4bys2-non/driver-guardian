package com.example.driverguardian.ai.physical

import com.example.driverguardian.ai.physical.geometry.AspectRatioCalculator
import com.example.driverguardian.ai.physical.geometry.HeadPoseEstimator
import com.example.driverguardian.ai.physical.model.LandmarkPoint
import com.example.driverguardian.ai.physical.pipeline.PhysicalBranchConfig
import com.example.driverguardian.ai.physical.pipeline.PhysicalBranchProcessor
import com.example.driverguardian.ai.physical.temporal.AdaptiveHmm
import com.example.driverguardian.ai.physical.temporal.TemporalStateMachine
import com.example.driverguardian.ai.physical.temporal.WindowRatio
import com.google.gson.Gson
import com.google.gson.JsonObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.InputStreamReader

class PhysicalBranchParityTest {

    private lateinit var root: JsonObject

    @Before
    fun setUp() {
        val stream = javaClass.classLoader?.getResourceAsStream("physical_fixtures.json")
            ?: error("physical_fixtures.json not found in test resources")
        val reader = InputStreamReader(stream)
        root = Gson().fromJson(reader, JsonObject::class.java)
    }

    @Test
    fun testEarMarParityWithPython() {
        val earMarObj = root.getAsJsonObject("ear_mar")
        val expectedLeft = earMarObj.get("left_eye").asDouble
        val expectedRight = earMarObj.get("right_eye").asDouble
        val expectedMouth = earMarObj.get("mouth").asDouble

        val rawPoints = earMarObj.getAsJsonArray("test_landmarks")
        val points = mutableListOf<LandmarkPoint>()
        for (elem in rawPoints) {
            val arr = elem.asJsonArray
            points.add(LandmarkPoint(arr[0].asFloat, arr[1].asFloat))
        }

        val actualLeft = AspectRatioCalculator.calculateAspectRatio(points, AspectRatioCalculator.LEFT_EYE_INDICES)
        val actualRight = AspectRatioCalculator.calculateAspectRatio(points, AspectRatioCalculator.RIGHT_EYE_INDICES)
        val actualMouth = AspectRatioCalculator.calculateMar(points)

        assertNotNull(actualLeft)
        assertNotNull(actualRight)
        assertNotNull(actualMouth)

        // Floating point parity tolerance: 1e-5
        assertEquals("Left eye AR must match Python", expectedLeft, actualLeft!!, 1e-5)
        assertEquals("Right eye AR must match Python", expectedRight, actualRight!!, 1e-5)
        assertEquals("Mouth AR must match Python", expectedMouth, actualMouth!!, 1e-5)
    }

    @Test
    fun testAdaptiveHmmParityWithPython() {
        val hmmObj = root.getAsJsonObject("hmm")
        val initialData = hmmObj.getAsJsonArray("initial_data").map { it.asDouble }.toDoubleArray()

        val hmm = AdaptiveHmm(positiveState = "low")
        hmm.fitInitial(initialData)

        val (state013, post013) = hmm.predict(0.13)
        val (state032, post032) = hmm.predict(0.32)

        val expState013 = hmmObj.get("pred_0_13_state").asInt
        val expState032 = hmmObj.get("pred_0_32_state").asInt

        assertEquals("State prediction for 0.13 must match Python", expState013, state013)
        assertEquals("State prediction for 0.32 must match Python", expState032, state032)

        val expPost013 = hmmObj.getAsJsonArray("pred_0_13_post").map { it.asDouble }
        val expPost032 = hmmObj.getAsJsonArray("pred_0_32_post").map { it.asDouble }

        assertEquals(expPost013[0], post013[0], 1e-3)
        assertEquals(expPost013[1], post013[1], 1e-3)
        assertEquals(expPost032[0], post032[0], 1e-3)
        assertEquals(expPost032[1], post032[1], 1e-3)
    }

    @Test
    fun testHeadPoseParityWithOpenCv() {
        val hpObj = root.getAsJsonObject("head_pose")
        val expAngles = hpObj.getAsJsonArray("angles").map { it.asDouble }

        val rawPoints = hpObj.getAsJsonArray("pts2d")
        val points = mutableListOf<LandmarkPoint>()
        for (elem in rawPoints) {
            val arr = elem.asJsonArray
            points.add(LandmarkPoint(arr[0].asFloat, arr[1].asFloat))
        }

        val estimator = HeadPoseEstimator(640, 480)
        val pose = estimator.estimate(points)

        // Parity with OpenCV RQDecomp3x3: within 0.05 degrees
        assertEquals("Pitch deg must match OpenCV", expAngles[0], pose.pitchDeg, 0.05)
        assertEquals("Yaw deg must match OpenCV", expAngles[1], pose.yawDeg, 0.05)
        assertEquals("Roll deg must match OpenCV", expAngles[2], pose.rollDeg, 0.05)
    }

    @Test
    fun testCameraMetricsSequenceParity() {
        val cmResults = root.getAsJsonArray("cm_results")
        val earMarObj = root.getAsJsonObject("ear_mar")
        val rawPoints = earMarObj.getAsJsonArray("test_landmarks")

        val baseLandmarks = mutableListOf<LandmarkPoint>()
        for (elem in rawPoints) {
            val arr = elem.asJsonArray
            baseLandmarks.add(LandmarkPoint(arr[0].asFloat, arr[1].asFloat))
        }

        val config = PhysicalBranchConfig(fps = 30, windowSec = 60.0, calibrationFrames = 5)
        val processor = PhysicalBranchProcessor(config)

        var t = 100.0
        for (i in 0 until cmResults.size()) {
            t += 1.0 / 30.0
            val expectedObj = cmResults[i].asJsonObject

            val lm = baseLandmarks.toMutableList()
            if (i in listOf(7, 8)) {
                for (idx in AspectRatioCalculator.LEFT_EYE_INDICES + AspectRatioCalculator.RIGHT_EYE_INDICES) {
                    val p = lm[idx]
                    lm[idx] = LandmarkPoint(p.x, p.y * 0.99f)
                }
            }

            val actual = processor.processLandmarks(lm, 640, 480, t)

            assertEquals("Frame $i faceDetected parity", expectedObj.get("face_detected").asBoolean, actual.faceDetected)
            assertEquals("Frame $i headCalibrated parity", expectedObj.get("head_calibrated").asBoolean, actual.headCalibrated)

            if (!expectedObj.get("ear").isJsonNull) {
                assertEquals("Frame $i EAR parity", expectedObj.get("ear").asDouble, actual.ear!!, 1e-4)
            }
            if (!expectedObj.get("mar").isJsonNull) {
                assertEquals("Frame $i MAR parity", expectedObj.get("mar").asDouble, actual.mar!!, 1e-4)
            }
            if (!expectedObj.get("pitch_deg").isJsonNull) {
                assertNotNull("Frame $i pitchDeg should not be null", actual.pitchDeg)
                assertEquals("Frame $i pitchDeg parity", expectedObj.get("pitch_deg").asDouble, actual.pitchDeg!!, 0.05)
            }
        }
    }
}
