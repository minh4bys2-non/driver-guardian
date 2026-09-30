package com.example.driverguardian.ui.screens.onnxdemo

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.driverguardian.ai.contract.drowsiness.*
import com.example.driverguardian.ai.parity.*
import com.example.driverguardian.ai.runtime.*
import com.example.driverguardian.ai.tensor.DummyTensorFactory
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
    val dummyAvailable: Boolean = false,
    val dummyMessage: String = "Load a model to evaluate dummy input support.",
    val goldenAvailable: Boolean = false,
    val goldenTolerance: String = "Not configured",
    val goldenResults: List<Pair<String, ParityResult>> = emptyList(),
    val goldenMessage: String = "Golden vectors: NOT AVAILABLE\nExpected path: assets/onnx_test_vectors/manifest.json",
)

class OnnxDemoViewModel(application: Application) : AndroidViewModel(application) {
    private val engine = OnnxRuntimeEngine()
    private val contract = DrowsinessModelContract(DrowsinessInputMapping.Unresolved)
    private var goldenSuite: LoadedGoldenSuite? = null
    private val mutableState = MutableStateFlow(OnnxDemoUiState(diagnostics = engine.diagnostics))
    val uiState: StateFlow<OnnxDemoUiState> = mutableState.asStateFlow()

    init { inspectGoldenVectors() }

    fun loadModel() {
        if (mutableState.value.runtimeState is RuntimeState.Loading || mutableState.value.runtimeState is RuntimeState.Running) return
        mutableState.update { it.copy(runtimeState = RuntimeState.Loading, lastInference = null) }
        viewModelScope.launch {
            val bytes = try {
                withContext(Dispatchers.IO) { getApplication<Application>().assets.open(MODEL_ASSET).use { it.readBytes() } }
            } catch (_: FileNotFoundException) {
                val error = RuntimeError.ModelNotFound(MODEL_ASSET)
                engine.recordFailure(error)
                mutableState.update { it.copy(runtimeState = engine.state, diagnostics = engine.diagnostics, metadata = null, contract = null) }
                return@launch
            } catch (throwable: Throwable) {
                val error = RuntimeError.ModelLoadFailure(throwable.message ?: "Could not read model asset", throwable)
                engine.recordFailure(error)
                mutableState.update { it.copy(runtimeState = engine.state, diagnostics = engine.diagnostics) }
                return@launch
            }
            when (val result = withContext(Dispatchers.Default) { engine.loadModel(MODEL_ASSET, bytes) }) {
                is RuntimeResult.Success -> mutableState.update {
                    val dummy = dummyEligibility(result.value)
                    it.copy(
                        runtimeState = engine.state,
                        diagnostics = engine.diagnostics,
                        metadata = result.value,
                        contract = contract.validate(result.value),
                        dummyAvailable = dummy.first,
                        dummyMessage = dummy.second,
                    )
                }
                is RuntimeResult.Failure -> mutableState.update { it.copy(runtimeState = engine.state, diagnostics = engine.diagnostics) }
            }
        }
    }

    fun runDummyInference() {
        val metadata = mutableState.value.metadata ?: return
        if (mutableState.value.runtimeState !is RuntimeState.Ready || !mutableState.value.dummyAvailable) return
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

    companion object { const val MODEL_ASSET = "models/drowsiness_model.onnx" }
}
