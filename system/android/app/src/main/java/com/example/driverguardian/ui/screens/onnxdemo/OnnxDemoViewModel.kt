package com.example.driverguardian.ui.screens.onnxdemo

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.driverguardian.ai.contract.demo.EndToEndDemoModelContract
import com.example.driverguardian.ai.contract.drowsiness.*
import com.example.driverguardian.ai.demo.BoundedSequenceBuffer
import com.example.driverguardian.ai.demo.DemoOnnxOutput
import com.example.driverguardian.ai.demo.EndToEndDemoInferenceRunner
import com.example.driverguardian.ai.demo.EndToEndDemoPreprocessor
import com.example.driverguardian.ai.parity.*
import com.example.driverguardian.ai.runtime.*
import com.example.driverguardian.ai.tensor.DummyTensorFactory
import com.example.driverguardian.ai.tensor.FloatTensor
import com.example.driverguardian.ai.tensor.TensorShape
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.FileNotFoundException

data class OnnxDemoUiState(
    val runtimeState: RuntimeState = RuntimeState.NotLoaded,
    val diagnostics: RuntimeDiagnostics? = null,
    val metadata: ModelMetadata? = null,
    val contract: ContractValidationResult? = null,
    val lastInference: RuntimeInferenceResult? = null,
    val demoOutput: DemoOnnxOutput? = null,
    val dummyAvailable: Boolean = false,
    val dummyMessage: String = "Load a model to evaluate dummy input support.",
    val goldenAvailable: Boolean = false,
    val goldenTolerance: String = "Not configured",
    val goldenResults: List<Pair<String, ParityResult>> = emptyList(),
    val goldenMessage: String = "Golden vectors: NOT AVAILABLE\nExpected path: assets/onnx_test_vectors/manifest.json",
    val activeModelAsset: String = OnnxDemoViewModel.DEMO_MODEL_ASSET,
)

class OnnxDemoViewModel(application: Application) : AndroidViewModel(application) {
    private val engine = OnnxRuntimeEngine()
    private val demoContract = EndToEndDemoModelContract()
    private val legacyContract = DrowsinessModelContract(DrowsinessInputMapping.Unresolved)
    private val demoInferenceRunner = EndToEndDemoInferenceRunner(engine)
    private var goldenSuite: LoadedGoldenSuite? = null
    private val mutableState = MutableStateFlow(OnnxDemoUiState(diagnostics = engine.diagnostics))
    val uiState: StateFlow<OnnxDemoUiState> = mutableState.asStateFlow()

    init {
        inspectGoldenVectors()
    }

    fun loadModel(assetPath: String = mutableState.value.activeModelAsset) {
        if (mutableState.value.runtimeState is RuntimeState.Loading || mutableState.value.runtimeState is RuntimeState.Running) return
        mutableState.update { it.copy(runtimeState = RuntimeState.Loading, lastInference = null, activeModelAsset = assetPath) }
        viewModelScope.launch {
            val bytes = try {
                withContext(Dispatchers.IO) { getApplication<Application>().assets.open(assetPath).use { it.readBytes() } }
            } catch (_: FileNotFoundException) {
                val error = RuntimeError.ModelNotFound(assetPath)
                engine.recordFailure(error)
                mutableState.update { it.copy(runtimeState = engine.state, diagnostics = engine.diagnostics, metadata = null, contract = null) }
                return@launch
            } catch (throwable: Throwable) {
                val error = RuntimeError.ModelLoadFailure(throwable.message ?: "Could not read model asset", throwable)
                engine.recordFailure(error)
                mutableState.update { it.copy(runtimeState = engine.state, diagnostics = engine.diagnostics) }
                return@launch
            }
            when (val result = withContext(Dispatchers.Default) { engine.loadModel(assetPath, bytes) }) {
                is RuntimeResult.Success -> mutableState.update {
                    val meta = result.value
                    // Try demo contract first, fallback to legacy contract
                    val demoValidation = demoContract.validate(meta)
                    val validation = if (demoValidation.isRunnable) demoValidation else legacyContract.validate(meta)
                    val dummy = dummyEligibility(meta)
                    it.copy(
                        runtimeState = engine.state,
                        diagnostics = engine.diagnostics,
                        metadata = meta,
                        contract = validation,
                        dummyAvailable = dummy.first,
                        dummyMessage = dummy.second,
                    )
                }
                is RuntimeResult.Failure -> mutableState.update { it.copy(runtimeState = engine.state, diagnostics = engine.diagnostics) }
            }
        }
    }

    /**
     * Executes inference on the loaded end-to-end demo ONNX model using deterministic synthetic video inputs.
     *
     * @param sequenceLength T frames (default 1, supports dynamic T such as 2 or 10)
     * @param constantPixelValue float pixel value, e.g. 0.5f (normalized) or 0.0f
     */
    fun runDemoInference(sequenceLength: Int = 1, constantPixelValue: Float = 0.5f) {
        val metadata = mutableState.value.metadata ?: return
        if (mutableState.value.runtimeState !is RuntimeState.Ready) return

        viewModelScope.launch {
            mutableState.update { it.copy(runtimeState = RuntimeState.Running(metadata)) }

            // Build deterministic [1, T, 3, 640, 640] FloatTensor
            val frameElements = EndToEndDemoPreprocessor.FRAME_ELEMENT_COUNT
            val totalElements = sequenceLength * frameElements
            val buffer = FloatArray(totalElements) { constantPixelValue }
            val shape = listOf(1L, sequenceLength.toLong(), 3L, 640L, 640L)
            val tensor = FloatTensor(shape, buffer)

            val inferenceResult = demoInferenceRunner.runInference(tensor)

            mutableState.update {
                when (inferenceResult) {
                    is RuntimeResult.Success -> it.copy(
                        runtimeState = engine.state,
                        diagnostics = engine.diagnostics,
                        demoOutput = inferenceResult.value,
                    )
                    is RuntimeResult.Failure -> it.copy(
                        runtimeState = RuntimeState.Failed(inferenceResult.error, metadata),
                        diagnostics = engine.diagnostics,
                    )
                }
            }
        }
    }

    fun runDummyInference() {
        val metadata = mutableState.value.metadata ?: return
        if (mutableState.value.runtimeState !is RuntimeState.Ready || !mutableState.value.dummyAvailable) return

        // If this is the demo end-to-end model, run demo inference with T=1
        val demoValidation = demoContract.validate(metadata)
        if (demoValidation.isRunnable) {
            runDemoInference(sequenceLength = 1)
            return
        }

        viewModelScope.launch {
            mutableState.update { it.copy(runtimeState = RuntimeState.Running(metadata)) }
            val result = withContext(Dispatchers.Default) {
                when (val tensors = DummyTensorFactory.create(metadata)) {
                    is RuntimeResult.Success -> engine.run(tensors.value)
                    is RuntimeResult.Failure -> tensors
                }
            }
            mutableState.update {
                when (result) {
                    is RuntimeResult.Success -> it.copy(runtimeState = engine.state, diagnostics = engine.diagnostics, lastInference = result.value)
                    is RuntimeResult.Failure -> it.copy(runtimeState = RuntimeState.Failed(result.error, metadata), diagnostics = engine.diagnostics)
                }
            }
        }
    }

    fun runGoldenTests() {
        val metadata = mutableState.value.metadata ?: return
        if (mutableState.value.runtimeState !is RuntimeState.Ready || !mutableState.value.goldenAvailable) return
        viewModelScope.launch {
            val suite = goldenSuite ?: return@launch
            mutableState.update { it.copy(runtimeState = RuntimeState.Running(metadata), goldenResults = emptyList()) }
            val results = withContext(Dispatchers.Default) {
                suite.cases.map { testCase ->
                    val parity = when (val run = engine.run(testCase.inputs)) {
                        is RuntimeResult.Success -> GoldenVectorRunner.compare(testCase, run.value, suite.tolerance)
                        is RuntimeResult.Failure -> ParityResult(ParityStatus.ERROR, message = run.error.detail)
                    }
                    testCase.name to parity
                }
            }
            mutableState.update { it.copy(runtimeState = engine.state, diagnostics = engine.diagnostics, goldenResults = results, goldenMessage = "Executed ${results.size} golden case(s).") }
        }
    }

    private fun inspectGoldenVectors() {
        viewModelScope.launch {
            val suite = try {
                withContext(Dispatchers.IO) { GoldenVectorAssetLoader.load(getApplication<Application>().assets) }
            } catch (_: FileNotFoundException) {
                return@launch
            } catch (throwable: Throwable) {
                mutableState.update { it.copy(goldenMessage = "Golden vectors: INVALID — ${throwable.message}") }
                return@launch
            }
            goldenSuite = suite
            val tolerance = when (val configured = suite.tolerance) {
                NumericalTolerance.Unconfigured -> "Not configured"
                is NumericalTolerance.Configured -> "atol=${configured.absolute}, rtol=${configured.relative}"
            }
            mutableState.update {
                it.copy(
                    goldenAvailable = true,
                    goldenTolerance = tolerance,
                    goldenMessage = "Golden vectors: AVAILABLE (${suite.cases.size} case(s))",
                )
            }
        }
    }

    private fun dummyEligibility(metadata: ModelMetadata): Pair<Boolean, String> {
        val demoValidation = demoContract.validate(metadata)
        if (demoValidation.isRunnable) {
            return true to "Demo end-to-end model validated. Ready for synthetic video input [1, T, 3, 640, 640]."
        }

        var total = 0L
        for (input in metadata.inputs) {
            if (input.type != RuntimeTensorType.FLOAT) return false to "Input '${input.name}' is ${input.type}; FLOAT is required."
            if (input.hasDynamicDimensions) return false to "Input '${input.name}' requires concrete dimensions."
            val count = TensorShape.checkedElementCount(input.shape) ?: return false to "Input '${input.name}' has an invalid shape."
            if (count > DummyTensorFactory.MAX_ELEMENTS_PER_INPUT || total > DummyTensorFactory.MAX_TOTAL_ELEMENTS - count || count > Int.MAX_VALUE) {
                return false to "Input '${input.name}' exceeds the dummy allocation safety limit."
            }
            total += count
        }
        return true to "Zero-filled diagnostic input. Not a meaningful drowsiness prediction."
    }

    override fun onCleared() { engine.close(); super.onCleared() }

    companion object {
        const val DEMO_MODEL_ASSET = "models/driver_guardian_end2end_demo.onnx"
        const val LEGACY_MODEL_ASSET = "models/drowsiness_model.onnx"
        const val MODEL_ASSET = DEMO_MODEL_ASSET
    }
}
