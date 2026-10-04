package com.example.driverguardian.ai.physical

import com.example.driverguardian.ai.physical.geometry.AspectRatioCalculator
import com.example.driverguardian.ai.physical.geometry.HeadPoseEstimator
import com.example.driverguardian.ai.physical.model.LandmarkPoint
import com.example.driverguardian.ai.physical.pipeline.PhysicalBranchConfig
import com.example.driverguardian.ai.physical.pipeline.PhysicalBranchProcessor
import com.example.driverguardian.ai.physical.temporal.AdaptiveHmm
import com.google.gson.Gson
import com.google.gson.JsonArray
import com.google.gson.JsonObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.InputStreamReader
import kotlin.math.abs

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
    fun testHeadPoseMultiPoseParityWithOpenCv() {
        val stream = javaClass.classLoader?.getResourceAsStream("head_pose_golden.json")
            ?: error("head_pose_golden.json not found in test resources")
        val cases = Gson().fromJson(InputStreamReader(stream), JsonArray::class.java)

        var maxMeasuredErrorDeg = 0.0

        for (elem in cases) {
            val obj = elem.asJsonObject
            val name = obj.get("name").asString
            val w = obj.get("w").asInt
            val h = obj.get("h").asInt
            val expEuler = obj.getAsJsonArray("expected_euler").map { it.asDouble }
            val rawLm = obj.getAsJsonArray("landmarks")

            val lmList = mutableListOf<LandmarkPoint>()
            for (p in rawLm) {
                val ptArr = p.asJsonArray
                lmList.add(LandmarkPoint(ptArr[0].asFloat, ptArr[1].asFloat))
            }

            val estimator = HeadPoseEstimator(w, h)
            val actual = estimator.estimate(lmList)

            val pitchErr = abs(actual.pitchDeg - expEuler[0])
            val yawErr = abs(actual.yawDeg - expEuler[1])
            val rollErr = abs(actual.rollDeg - expEuler[2])
            val caseMaxErr = maxOf(pitchErr, maxOf(yawErr, rollErr))
            if (caseMaxErr > maxMeasuredErrorDeg) {
                maxMeasuredErrorDeg = caseMaxErr
            }

            assertEquals("Case $name pitch parity with OpenCV", expEuler[0], actual.pitchDeg, 0.05)
            assertEquals("Case $name yaw parity with OpenCV", expEuler[1], actual.yawDeg, 0.05)
            assertEquals("Case $name roll parity with OpenCV", expEuler[2], actual.rollDeg, 0.05)
        }

        // Bounded numerical tolerance across all 11 poses
        assertTrue("Max head pose error across all cases ($maxMeasuredErrorDeg) must be < 0.01 deg", maxMeasuredErrorDeg < 0.01)
    }

    @Test
    fun testAdaptiveHmmFullParityWithPython() {
        val stream = javaClass.classLoader?.getResourceAsStream("hmm_golden.json")
            ?: error("hmm_golden.json not found in test resources")
        val hmmGolden = Gson().fromJson(InputStreamReader(stream), JsonObject::class.java)

        val trainData = hmmGolden.getAsJsonArray("train_data").map { it.asDouble }.toDoubleArray()
        val expFit = hmmGolden.getAsJsonObject("fit_initial")

        val hmm = AdaptiveHmm(
            nStates = 2,
            learningRate = 0.01,
            adaptInterval = 300,
            positiveState = "low",
            minVariance = 1e-5,
            emIterations = 20
        )
        hmm.fitInitial(trainData)

        // Compare fitInitial parameters: means, vars, A, pi
        val expMeans = expFit.getAsJsonArray("means").map { it.asDouble }
        val expVars = expFit.getAsJsonArray("vars").map { it.asDouble }
        val expA = expFit.getAsJsonArray("A")
        val expPi = expFit.getAsJsonArray("pi").map { it.asDouble }

        assertEquals("Means[0] parity", expMeans[0], hmm.means[0], 1e-4)
        assertEquals("Means[1] parity", expMeans[1], hmm.means[1], 1e-4)
        assertEquals("Vars[0] parity", expVars[0], hmm.vars[0], 1e-4)
        assertEquals("Vars[1] parity", expVars[1], hmm.vars[1], 1e-4)
        assertEquals("Pi[0] parity", expPi[0], hmm.pi[0], 1e-3)

        val row0 = expA[0].asJsonArray.map { it.asDouble }
        val row1 = expA[1].asJsonArray.map { it.asDouble }
        assertEquals("A[0][0] parity", row0[0], hmm.a[0][0], 1e-3)
        assertEquals("A[0][1] parity", row0[1], hmm.a[0][1], 1e-3)
        assertEquals("A[1][0] parity", row1[0], hmm.a[1][0], 1e-3)
        assertEquals("A[1][1] parity", row1[1], hmm.a[1][1], 1e-3)

        // Compare predictions
        val predictions = hmmGolden.getAsJsonArray("predictions")
        for (elem in predictions) {
            val pObj = elem.asJsonObject
            val v = pObj.get("value").asDouble
            val expState = pObj.get("state").asInt
            val expPost = pObj.getAsJsonArray("posterior").map { it.asDouble }

            val (actState, actPost) = hmm.predict(v)
            assertEquals("Prediction state for $v", expState, actState)
            assertEquals("Posterior[0] for $v", expPost[0], actPost[0], 1e-3)
            assertEquals("Posterior[1] for $v", expPost[1], actPost[1], 1e-3)
        }

        // Test Online Adaptation with default adaptInterval=300
        val adaptData = hmmGolden.getAsJsonArray("adapt_data").map { it.asDouble }.toDoubleArray()
        var updatedStep: Int? = null
        for (i in adaptData.indices) {
            hmm.predict(adaptData[i])
            val didUpdate = hmm.updateOnline(adaptData[i])
            if (didUpdate) {
                updatedStep = i
            }
        }

        // Must update at step 299 (after exactly 300 samples)
        assertEquals("Online update step parity", 299, updatedStep)

        // Compare post-adapt parameters
        val expPostAdapt = hmmGolden.getAsJsonObject("post_adapt")
        val expPostMeans = expPostAdapt.getAsJsonArray("means").map { it.asDouble }
        val expPostVars = expPostAdapt.getAsJsonArray("vars").map { it.asDouble }

        assertEquals("Post-adapt means[0]", expPostMeans[0], hmm.means[0], 1e-4)
        assertEquals("Post-adapt means[1]", expPostMeans[1], hmm.means[1], 1e-4)
        assertEquals("Post-adapt vars[0]", expPostVars[0], hmm.vars[0], 1e-4)
        assertEquals("Post-adapt vars[1]", expPostVars[1], hmm.vars[1], 1e-4)
    }

    @Test
    fun testLongRunProcessorParity500Frames() {
        val stream = javaClass.classLoader?.getResourceAsStream("long_run_physical_fixtures.json")
            ?: error("long_run_physical_fixtures.json not found in test resources")
        val dataset = Gson().fromJson(InputStreamReader(stream), JsonObject::class.java)

        val inputs = dataset.getAsJsonArray("frames_input")
        val outputs = dataset.getAsJsonArray("frames_output")

        val config = PhysicalBranchConfig(
            fps = 30,
            windowSec = 60.0,
            calibrationFrames = 30,
            angleLimits = doubleArrayOf(20.0, 25.0, 20.0),
            nodPitchDeg = 14.0,
            nodReleaseDeg = 8.0,
            nodDirection = 1,
            nodDurationMinMs = 800.0,
            nodDurationMaxMs = 3500.0,
            maxGapSec = 0.25,
            eyeInitDurationSec = 5.0,
            mouthInitDurationSec = 5.0,
            learningRate = 0.01,
            adaptInterval = 300
        )
        val processor = PhysicalBranchProcessor(config)

        var totalFramesCompared = 0

        for (i in 0 until inputs.size()) {
            val inObj = inputs[i].asJsonObject
            val outObj = outputs[i].asJsonObject.getAsJsonObject("metrics")

            val timestamp = inObj.get("timestamp").asDouble
            val lmJson = inObj.get("landmarks")

            val lmList: List<LandmarkPoint>? = if (lmJson.isJsonNull) {
                null
            } else {
                lmJson.asJsonArray.map { ptElem ->
                    val arr = ptElem.asJsonArray
                    LandmarkPoint(arr[0].asFloat, arr[1].asFloat)
                }
            }

            val actual = processor.processLandmarks(lmList, 640, 480, timestamp)
            totalFramesCompared++

            // Exact boolean / state / null assertions
            assertEquals("Step $i faceDetected", outObj.get("face_detected").asBoolean, actual.faceDetected)
            assertEquals("Step $i eyeReady", outObj.get("eye_ready").asBoolean, actual.eyeReady)
            assertEquals("Step $i mouthReady", outObj.get("mouth_ready").asBoolean, actual.mouthReady)
            assertEquals("Step $i headReady", outObj.get("head_ready").asBoolean, actual.headReady)
            assertEquals("Step $i headCalibrated", outObj.get("head_calibrated").asBoolean, actual.headCalibrated)
            assertEquals("Step $i blinkDetected", outObj.get("blink_detected").asBoolean, actual.blinkDetected)
            assertEquals("Step $i yawnDetected", outObj.get("yawn_detected").asBoolean, actual.yawnDetected)
            assertEquals("Step $i nodDetected", outObj.get("nod_detected").asBoolean, actual.nodDetected)

            // State nullability & values
            if (outObj.get("eye_state").isJsonNull) {
                assertNull("Step $i eyeState null", actual.eyeState)
            } else {
                assertEquals("Step $i eyeState", outObj.get("eye_state").asInt, actual.eyeState)
            }

            if (outObj.get("mouth_state").isJsonNull) {
                assertNull("Step $i mouthState null", actual.mouthState)
            } else {
                assertEquals("Step $i mouthState", outObj.get("mouth_state").asInt, actual.mouthState)
            }

            if (outObj.get("over_angle").isJsonNull) {
                assertNull("Step $i overAngle null", actual.overAngle)
            } else {
                assertEquals("Step $i overAngle", outObj.get("over_angle").asBoolean, actual.overAngle)
            }

            if (outObj.get("nodding").isJsonNull) {
                assertNull("Step $i nodding null", actual.nodding)
            } else {
                assertEquals("Step $i nodding", outObj.get("nodding").asBoolean, actual.nodding)
            }

            // Floating point metrics with explicit tolerance
            if (!outObj.get("ear").isJsonNull) {
                assertNotNull("Step $i ear non-null", actual.ear)
                assertEquals("Step $i ear", outObj.get("ear").asDouble, actual.ear!!, 1e-5)
            }

            if (!outObj.get("mar").isJsonNull) {
                assertNotNull("Step $i mar non-null", actual.mar)
                assertEquals("Step $i mar", outObj.get("mar").asDouble, actual.mar!!, 1e-5)
            }

            if (!outObj.get("p80_ear_threshold").isJsonNull) {
                assertNotNull("Step $i p80EarThreshold non-null", actual.p80EarThreshold)
                assertEquals("Step $i p80EarThreshold", outObj.get("p80_ear_threshold").asDouble, actual.p80EarThreshold!!, 1e-4)
            }

            if (!outObj.get("perclos_pct").isJsonNull) {
                assertNotNull("Step $i perclosPct non-null", actual.perclosPct)
                assertEquals("Step $i perclosPct", outObj.get("perclos_pct").asDouble, actual.perclosPct!!, 1e-3)
            }

            if (!outObj.get("pom_pct").isJsonNull) {
                assertNotNull("Step $i pomPct non-null", actual.pomPct)
                assertEquals("Step $i pomPct", outObj.get("pom_pct").asDouble, actual.pomPct!!, 1e-3)
            }

            if (!outObj.get("over_angle_pct").isJsonNull) {
                assertNotNull("Step $i overAnglePct non-null", actual.overAnglePct)
                assertEquals("Step $i overAnglePct", outObj.get("over_angle_pct").asDouble, actual.overAnglePct!!, 1e-3)
            }

            if (!outObj.get("pitch_deg").isJsonNull) {
                assertNotNull("Step $i pitchDeg non-null", actual.pitchDeg)
                assertEquals("Step $i pitchDeg", outObj.get("pitch_deg").asDouble, actual.pitchDeg!!, 0.05)
                assertEquals("Step $i yawDeg", outObj.get("yaw_deg").asDouble, actual.yawDeg!!, 0.05)
                assertEquals("Step $i rollDeg", outObj.get("roll_deg").asDouble, actual.rollDeg!!, 0.05)
            }

            if (!outObj.get("eye_closure_duration_ms").isJsonNull) {
                assertEquals("Step $i eyeClosureDurationMs", outObj.get("eye_closure_duration_ms").asDouble, actual.eyeClosureDurationMs!!, 1.0)
            }

            if (!outObj.get("mouth_open_duration_ms").isJsonNull) {
                assertEquals("Step $i mouthOpenDurationMs", outObj.get("mouth_open_duration_ms").asDouble, actual.mouthOpenDurationMs!!, 1.0)
            }

            if (!outObj.get("nod_duration_ms").isJsonNull) {
                assertEquals("Step $i nodDurationMs", outObj.get("nod_duration_ms").asDouble, actual.nodDurationMs!!, 1.0)
            }
        }

        assertEquals("All 500 frames must be evaluated", 500, totalFramesCompared)
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

