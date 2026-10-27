package com.example.driverguardian.ai.monitoring.camera

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class BrightnessBalancerTest {

    @Test
    fun testDefaultStateIsDisabled() {
        val balancer = BrightnessBalancer()
        assertFalse(balancer.enabled.get())
    }

    @Test
    fun testZeroConversionWhenDisabled() {
        val balancer = BrightnessBalancer(targetLuma = 70.0)
        balancer.enabled.set(false)

        val sample = ByteArray(640 * 480 * 4) { 10 }
        val copy = sample.clone()

        balancer.processInPlaceIfDark(sample, 640, 480)
        assertArrayEquals("Buffer must remain unchanged when balancer is disabled", copy, sample)
    }

    @Test
    fun testFastExitWhenEnvironmentIsBright() {
        val balancer = BrightnessBalancer(targetLuma = 70.0)
        balancer.enabled.set(true)
        assertTrue(balancer.enabled.get())

        // Giá trị pixel R=120, G=120, B=120 -> Luma Y = 120 >= 70.0 (trời sáng)
        val brightSample = ByteArray(640 * 480 * 4) { 120.toByte() }
        val copy = brightSample.clone()

        balancer.processInPlaceIfDark(brightSample, 640, 480)
        assertArrayEquals("Buffer must not be touched if fast luma >= targetLuma", copy, brightSample)
    }
}
