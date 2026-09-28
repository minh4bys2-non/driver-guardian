package com.example.driverguardian.ai.physical

import com.example.driverguardian.ai.physical.temporal.AdaptiveHmm
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AdaptiveHmmTest {

    @Test
    fun testFitInitialAndPredictLowPositiveState() {
        val hmm = AdaptiveHmm(positiveState = "low")
        val trainData = DoubleArray(30) { i ->
            if (i % 5 == 0) 0.12 else 0.32
        }

        hmm.fitInitial(trainData)
        assertTrue(hmm.initialized)

        // For positiveState="low", state 0 should be normal open eye (higher mean), state 1 closed (lower mean)
        assertTrue("State 0 mean should be greater than State 1 mean", hmm.means[0] > hmm.means[1])

        val (stateOpen, postOpen) = hmm.predict(0.33)
        assertEquals("Higher EAR should predict state 0 (open)", 0, stateOpen)
        assertTrue(postOpen[0] > postOpen[1])

        val (stateClosed, postClosed) = hmm.predict(0.10)
        assertEquals("Lower EAR should predict state 1 (closed)", 1, stateClosed)
        assertTrue(postClosed[1] > postOpen[1])
    }

    @Test
    fun testFitInitialAndPredictHighPositiveState() {
        // Mouth mode: positiveState = "high", state 0 is closed (lower MAR), state 1 is yawning (higher MAR)
        val hmm = AdaptiveHmm(positiveState = "high")
        val trainData = DoubleArray(30) { i ->
            if (i % 5 == 0) 0.65 else 0.20
        }

        hmm.fitInitial(trainData)
        assertTrue(hmm.initialized)
        assertTrue("State 0 mean should be lower than State 1 mean", hmm.means[0] < hmm.means[1])

        val (stateResting, _) = hmm.predict(0.18)
        assertEquals("Lower MAR should predict state 0 (resting)", 0, stateResting)

        val (stateYawn, _) = hmm.predict(0.70)
        assertEquals("Higher MAR should predict state 1 (yawning)", 1, stateYawn)
    }

    @Test
    fun testOnlineAdaptationTrigger() {
        val hmm = AdaptiveHmm(adaptInterval = 10)
        val trainData = doubleArrayOf(0.3, 0.3, 0.3, 0.3, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3)
        hmm.fitInitial(trainData)

        // Buffer 9 items (with both open and closed samples) -> should not adapt yet
        val sequence = doubleArrayOf(0.31, 0.31, 0.12, 0.31, 0.31, 0.12, 0.31, 0.31, 0.12, 0.31)
        for (i in 0 until 9) {
            val adapted = hmm.updateOnline(sequence[i])
            assertFalse(adapted)
        }

        // 10th item triggers updateOnline with both states having >= 2 occurrences
        val adapted = hmm.updateOnline(sequence[9])
        assertTrue("10th frame must trigger adaptation", adapted)
        assertTrue("Buffer must clear after adaptation", hmm.buffer.isEmpty())
    }

    @Test(expected = IllegalStateException::class)
    fun testPredictBeforeInitializationThrows() {
        val hmm = AdaptiveHmm()
        hmm.predict(0.25)
    }

    @Test(expected = IllegalArgumentException::class)
    fun testPredictNonFiniteThrows() {
        val hmm = AdaptiveHmm()
        hmm.fitInitial(doubleArrayOf(0.1, 0.1, 0.3, 0.3))
        hmm.predict(Double.NaN)
    }
}
