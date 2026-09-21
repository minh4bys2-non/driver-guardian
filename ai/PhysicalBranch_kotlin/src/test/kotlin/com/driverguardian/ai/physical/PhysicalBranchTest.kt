package com.driverguardian.ai.physical

import com.driverguardian.ai.AIResult
import kotlin.math.abs

/**
 * Self-contained unit tests and verification assertions for all PhysicalBranch Kotlin components.
 * Can be run via JUnit or as a main() entry point.
 */
object PhysicalBranchTest {

    fun assertEquals(expected: Double, actual: Double, epsilon: Double = 1e-5, msg: String = "") {
        if (abs(expected - actual) > epsilon) {
            throw AssertionError("$msg: expected $expected, but got $actual (diff: ${abs(expected - actual)})")
        }
    }

    fun assertEquals(expected: Any?, actual: Any?, msg: String = "") {
        if (expected != actual) {
            throw AssertionError("$msg: expected $expected, but got $actual")
        }
    }

    fun assertTrue(condition: Boolean, msg: String = "") {
        if (!condition) {
            throw AssertionError("$msg: expected true, got false")
        }
    }

    fun testAIResult() {
        println("Testing AIResult...")
        val res = AIResult(
            timestampSec = 12.34,
            windowSec = 60.0,
            ear = 0.285,
            blinkRatePerMin = 18.0,
            eyeClosureDurationMs = 120.0,
            perclosPct = 8.5,
            mar = 0.12,
            pitchDeg = -2.5
        )
        val json = res.toJson()
        assertTrue(json.contains("\"timestamp_sec\": 12.34"), "JSON contains timestamp_sec")
        assertTrue(json.contains("\"ear\": 0.285"), "JSON contains ear")
        assertTrue(json.contains("\"perclos_pct\": 8.5"), "JSON contains perclos_pct")
        assertTrue(json.contains("\"lstm_drowsiness_probability\": null"), "JSON contains null field")

        val map = res.toMap()
        assertEquals(12.34, map["timestamp_sec"])
        assertEquals(0.285, map["ear"])

        val reconstructed = AIResult.fromMap(map)
        assertEquals(res.timestampSec, reconstructed.timestampSec)
        assertEquals(res.ear, reconstructed.ear)
        assertEquals(res.perclosPct, reconstructed.perclosPct)
        println("AIResult passed!")
    }

    fun testMathUtils() {
        println("Testing MathUtils...")
        val data = doubleArrayOf(10.0, 20.0, 30.0, 40.0, 50.0)
        assertEquals(30.0, MathUtils.percentile(data, 50.0), 1e-6, "Median 50%")
        assertEquals(10.0, MathUtils.percentile(data, 0.0), 1e-6, "Min 0%")
        assertEquals(50.0, MathUtils.percentile(data, 100.0), 1e-6, "Max 100%")
        assertEquals(20.0, MathUtils.percentile(data, 25.0), 1e-6, "25%")

        val evenData = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        assertEquals(2.5, MathUtils.median(evenData), 1e-6, "Median of 1,2,3,4")

        assertEquals(0.0, MathUtils.wrapAngle180(360.0), 1e-6, "Wrap 360")
        assertEquals(-170.0, MathUtils.wrapAngle180(190.0), 1e-6, "Wrap 190")
        assertEquals(170.0, MathUtils.wrapAngle180(-190.0), 1e-6, "Wrap -190")

        val lse = MathUtils.logSumExp(doubleArrayOf(1.0, 2.0, 3.0))
        assertEquals(3.40760596, lse, 1e-4, "LogSumExp of 1, 2, 3")

        val rvec = doubleArrayOf(0.1, 0.2, -0.3)
        val R = MathUtils.rodriguesVectorToMatrix(rvec)
        val rvecBack = MathUtils.rotationMatrixToRodriguesVector(R)
        assertEquals(rvec[0], rvecBack[0], 1e-4, "Rodrigues rvec X")
        assertEquals(rvec[1], rvecBack[1], 1e-4, "Rodrigues rvec Y")
        assertEquals(rvec[2], rvecBack[2], 1e-4, "Rodrigues rvec Z")
        println("MathUtils passed!")
    }

    fun testAdaptiveHMM() {
        println("Testing AdaptiveHMM...")
        val hmm = AdaptiveHMM(
            positiveState = PositiveState.LOW,
            emIterations = 10
        )
        // Synthetic data with two modes: high normal (0.35) and low blink (0.15)
        val samples = DoubleArray(100) { i ->
            if (i % 5 == 0) 0.15 else 0.35
        }
        hmm.fitInitial(samples)
        assertTrue(hmm.initialized, "HMM should be initialized")
        assertTrue(hmm.means[0] > hmm.means[1], "For PositiveState.LOW, state 0 mean must be > state 1 mean")

        val (stateOpen, postOpen) = hmm.predict(0.36)
        assertEquals(0, stateOpen, "High EAR should predict state 0 (open)")

        val (stateClosed, postClosed) = hmm.predict(0.12)
        assertEquals(1, stateClosed, "Low EAR should predict state 1 (closed)")
        println("AdaptiveHMM passed!")
    }

    fun testStateMachine() {
        println("Testing StateMachine...")
        val fsm = StateMachine(
            mode = FsmMode.EYE,
            minDurationMs = 100.0,
            maxDurationMs = 500.0,
            windowSizeSec = 60.0
        )

        var res = fsm.process(0, 1.0)
        assertEquals(false, res.eventDone)
        assertEquals(0.0, res.currentDurationMs)

        // Event starts at t=1.1
        res = fsm.process(1, 1.1)
        assertEquals(false, res.eventDone)

        // Event continues at t=1.2 (duration 100ms)
        res = fsm.process(1, 1.2)
        assertEquals(false, res.eventDone)
        assertEquals(100.0, res.currentDurationMs, 1e-3)

        // Event finishes at t=1.3 (duration 200ms -> valid blink!)
        res = fsm.process(0, 1.3)
        assertEquals(true, res.eventDone, "Event should be marked done")
        assertEquals(200.0, res.lastEventDurationMs, 1e-3)
        assertEquals(1.0, res.ratePerMinute, 1e-3) // 1 blink in 60s window = 1 blink/min
        println("StateMachine passed!")
    }

    fun testWindowRatio() {
        println("Testing WindowRatio...")
        val wr = WindowRatio(windowSec = 10.0, maxGapSec = 0.25)

        // Continuous active intervals with delta = 0.1s (<= 0.25s)
        wr.process(true, 1.0)
        wr.process(true, 1.1)
        wr.process(true, 1.2)
        wr.process(false, 1.3)
        val res = wr.process(false, 1.4)

        // Active from 1.0 to 1.2 (0.2s), total observed 1.0 to 1.4 (0.4s)
        // Ratio = 0.2 / 0.4 * 100 = 50.0%
        assertEquals(0.4, res.observedSeconds, 1e-3)
        assertEquals(75.0, res.percentage ?: 0.0, 1e-2)
        println("WindowRatio passed!")
    }

    fun testCameraMetrics() {
        println("Testing CameraMetrics...")
        val cm = CameraMetrics(
            calibrationFrames = 5,
            eyeInitDurationSec = 0.5,
            fps = 10.0
        )

        // Generate synthetic face landmarks (468 points)
        val landmarks = Array(468) { i ->
            Point2D(100.0 + (i % 20) * 5.0, 100.0 + (i / 20) * 5.0)
        }

        // Set up distinct eye landmarks for non-zero EAR
        // Left eye: 362, 385, 387, 263, 373, 380
        landmarks[362] = Point2D(100.0, 150.0)
        landmarks[263] = Point2D(140.0, 150.0) // width = 40
        landmarks[385] = Point2D(115.0, 140.0)
        landmarks[380] = Point2D(115.0, 160.0) // h1 = 20
        landmarks[387] = Point2D(125.0, 140.0)
        landmarks[373] = Point2D(125.0, 160.0) // h2 = 20
        // Expected EAR = (20 + 20) / (2 * 40) = 40 / 80 = 0.5

        val earLeft = CameraMetrics.aspectRatio(landmarks, CameraMetrics.LEFT_EYE)
        assertEquals(0.5, earLeft ?: 0.0, 1e-4, "Calculated EAR")

        var time = 1.0
        for (i in 0 until 10) {
            val result = cm.processLandmarks(landmarks, 640.0, 480.0, time)
            time += 0.1
            assertTrue(result.faceDetected, "Face must be detected")
        }

        val lastResult = cm.processLandmarks(landmarks, 640.0, 480.0, time)
        val aiResult = lastResult.toAIResult()
        assertEquals(time, aiResult.timestampSec)
        assertEquals(lastResult.ear, aiResult.ear)
        assertEquals(lastResult.perclosPct, aiResult.perclosPct)
        println("CameraMetrics passed!")
    }

    fun testPitchFsm() {
        println("Testing PitchFsm...")
        val pitchFsm = PitchFsm(
            pitchDownThreshold = 14.0,
            nodMinDurationMs = 800.0,
            nodMaxDurationMs = 3500.0,
            windowSizeSec = 60.0
        )

        // Normal head pose
        var res = pitchFsm.update(pitch = 5.0, yaw = 0.0, roll = 0.0, currentTimeSec = 1.0)
        assertEquals(PitchFsmState.NORMAL, res.state)
        assertEquals("Looking Center", res.poseStatus)

        // Head drops down (pitch = 20.0 > 14.0)
        res = pitchFsm.update(pitch = 20.0, yaw = 0.0, roll = 0.0, currentTimeSec = 2.0)
        assertEquals(PitchFsmState.PITCH_DOWN, res.state)

        // Still down at t=3.0 (duration 1000ms >= 800ms)
        res = pitchFsm.update(pitch = 20.0, yaw = 0.0, roll = 0.0, currentTimeSec = 3.0)
        assertEquals(PitchFsmState.PITCH_DOWN, res.state)
        assertTrue(res.eventLog.contains("WARNING: DROWSY HEAD DROPPING DETECTED"))

        // Head lifts back up at t=3.2 (duration 1200ms -> valid drowsy nod!)
        res = pitchFsm.update(pitch = 5.0, yaw = 0.0, roll = 0.0, currentTimeSec = 3.2)
        assertEquals(PitchFsmState.NORMAL, res.state)
        assertEquals(1, res.nodFrequencyPerMin)
        assertTrue(res.eventLog.contains("DROWSY HEAD NOD DETECTED"))
        assertTrue(res.isWarning)
        println("PitchFsm passed!")
    }

    @JvmStatic
    fun main(args: Array<String>) {
        println("Running PhysicalBranch Kotlin verification tests...")
        testAIResult()
        testMathUtils()
        testAdaptiveHMM()
        testStateMachine()
        testWindowRatio()
        testPitchFsm()
        testCameraMetrics()
        println("ALL TESTS PASSED SUCCESSFULLY!")
    }
}
