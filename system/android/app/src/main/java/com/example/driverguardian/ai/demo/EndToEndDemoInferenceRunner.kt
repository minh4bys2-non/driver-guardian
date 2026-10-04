package com.example.driverguardian.ai.demo

import com.example.driverguardian.ai.contract.demo.EndToEndDemoModelContract
import com.example.driverguardian.ai.runtime.OnnxRuntimeEngine
import com.example.driverguardian.ai.runtime.RuntimeError
import com.example.driverguardian.ai.runtime.RuntimeResult
import com.example.driverguardian.ai.tensor.FloatTensor
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Background inference runner for the demo end-to-end ONNX model.
 *
 * Rules:
 * - Uses existing OnnxRuntimeEngine infrastructure.
 * - Always runs inference on background worker (Dispatchers.Default), NEVER on the UI thread.
 * - Bounded latest-work behavior: drops incoming inference requests if an inference is already running (no backlog).
 * - Extracts binary logits and computes softmax for [DemoOnnxOutput].
 * - Strict non-interference: does NOT connect to warning/danger state or backend events.
 */
class EndToEndDemoInferenceRunner(
    private val engine: OnnxRuntimeEngine,
    private val dispatcher: CoroutineDispatcher = Dispatchers.Default,
) {
    private val isInferring = AtomicBoolean(false)

    val isBusy: Boolean
        get() = isInferring.get()

    suspend fun runInference(videoTensor: FloatTensor): RuntimeResult<DemoOnnxOutput> = withContext(dispatcher) {
        if (!isInferring.compareAndSet(false, true)) {
            // Drop request to prevent frame backlog (bounded / latest-work behavior)
            return@withContext RuntimeResult.Failure(
                RuntimeError.InferenceFailure("Inference dropped: background worker is busy with previous frame")
            )
        }

        try {
            val seqLen = videoTensor.shape.getOrNull(1)?.toInt() ?: 1
            val inputs = mapOf(EndToEndDemoModelContract.EXPECTED_INPUT_NAME to videoTensor)
            val startNs = System.nanoTime()

            when (val runResult = engine.run(inputs)) {
                is RuntimeResult.Success -> {
                    val durationMs = (System.nanoTime() - startNs) / 1_000_000.0
                    val logitsOutput = runResult.value.outputs.firstOrNull {
                        it.name == EndToEndDemoModelContract.EXPECTED_OUTPUT_NAME
                    }
                    val floatValues = logitsOutput?.floatValues
                    if (floatValues == null || floatValues.size < 2) {
                        return@withContext RuntimeResult.Failure(
                            RuntimeError.InferenceFailure("Model did not return binary logits: expected at least 2 floats")
                        )
                    }

                    // Logits tensor shape is [1, seqLen, 2].
                    // Take the logits from the last timestep: [seqLen - 1]
                    val lastStepOffset = (seqLen - 1) * 2
                    if (lastStepOffset + 1 >= floatValues.size) {
                        return@withContext RuntimeResult.Failure(
                            RuntimeError.InferenceFailure("Logits tensor size ${floatValues.size} insufficient for seqLen $seqLen")
                        )
                    }

                    val alertLogit = floatValues[lastStepOffset]
                    val drowsyLogit = floatValues[lastStepOffset + 1]

                    val (alertProb, drowsyProb) = DemoOnnxOutput.computeSoftmax(alertLogit, drowsyLogit)

                    val output = DemoOnnxOutput(
                        alertProbability = alertProb,
                        drowsyProbability = drowsyProb,
                        inferenceMs = durationMs,
                        sequenceFrames = seqLen,
                        modelLoaded = true,
                        rawAlertLogit = alertLogit,
                        rawDrowsyLogit = drowsyLogit,
                    )
                    RuntimeResult.Success(output)
                }
                is RuntimeResult.Failure -> {
                    RuntimeResult.Failure(runResult.error)
                }
            }
        } finally {
            isInferring.set(false)
        }
    }
}
