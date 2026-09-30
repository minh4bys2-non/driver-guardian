package com.example.driverguardian.ai.runtime

import ai.onnxruntime.OrtEnvironment
import com.example.driverguardian.ai.tensor.TensorData

/** Owns one active ONNX session and exposes model-agnostic load/run/close operations. */
class OnnxRuntimeEngine : AutoCloseable {
    private val lock = Any()
    private val environment = EnvironmentHolder.environment
    private var modelSession: OnnxModelSession? = null
    private var closed = false

    var state: RuntimeState = RuntimeState.NotLoaded
        private set
    var diagnostics: RuntimeDiagnostics = RuntimeDiagnostics(runtimeVersion = environment.version)
        private set

    fun modelMetadata(): ModelMetadata? = synchronized(lock) { modelSession?.metadata }

    fun loadModel(modelName: String, bytes: ByteArray): RuntimeResult<ModelMetadata> = synchronized(lock) {
        if (closed) return@synchronized RuntimeResult.Failure(RuntimeError.ClosedRuntime)
        state = RuntimeState.Loading
        modelSession?.close()
        modelSession = null
        when (val created = OnnxModelSession.create(environment, modelName, bytes)) {
            is RuntimeResult.Success -> {
                modelSession = created.value
                val metadata = created.value.metadata
                state = RuntimeState.Ready(metadata)
                diagnostics = diagnostics.copy(modelName = modelName, loadDurationNanos = metadata.loadDurationNanos, lastError = null)
                RuntimeResult.Success(metadata)
            }
            is RuntimeResult.Failure -> {
                state = RuntimeState.Failed(created.error)
                diagnostics = diagnostics.copy(modelName = modelName, lastError = created.error)
                created
            }
        }
    }

    fun recordFailure(error: RuntimeError) = synchronized(lock) {
        if (!closed) {
            state = RuntimeState.Failed(error, modelSession?.metadata)
            diagnostics = diagnostics.copy(lastError = error)
        }
    }

    fun run(inputs: Map<String, TensorData>): RuntimeResult<RuntimeInferenceResult> = synchronized(lock) {
        if (closed) return@synchronized RuntimeResult.Failure(RuntimeError.ClosedRuntime)
        val active = modelSession ?: return@synchronized RuntimeResult.Failure(RuntimeError.InferenceFailure("No model is loaded"))
        state = RuntimeState.Running(active.metadata)
        when (val result = active.run(inputs)) {
            is RuntimeResult.Success -> {
                state = RuntimeState.Ready(active.metadata)
                diagnostics = diagnostics.copy(lastInferenceDurationNanos = result.value.durationNanos, runCount = diagnostics.runCount + 1, lastError = null)
                result
            }
            is RuntimeResult.Failure -> {
                state = RuntimeState.Failed(result.error, active.metadata)
                diagnostics = diagnostics.copy(lastError = result.error)
                result
            }
        }
    }

    override fun close() = synchronized(lock) {
        if (!closed) {
            modelSession?.close()
            modelSession = null
            closed = true
            state = RuntimeState.Closed
        }
    }

    private object EnvironmentHolder { val environment: OrtEnvironment = OrtEnvironment.getEnvironment() }
}
