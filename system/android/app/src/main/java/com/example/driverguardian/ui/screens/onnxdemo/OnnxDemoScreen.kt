package com.example.driverguardian.ui.screens.onnxdemo

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.driverguardian.ai.runtime.RuntimeState
import com.example.driverguardian.ui.components.DashboardCard

@Composable
fun OnnxDemoScreen(viewModel: OnnxDemoViewModel = viewModel()) {
    val state by viewModel.uiState.collectAsState()
    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(bottom = 24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text("ONNX Runtime Demo", style = MaterialTheme.typography.headlineMedium)
        Text("Generic runtime inspection and offline smoke/parity checks. No camera pipeline is connected.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        DashboardCard("Runtime status") {
            Text(runtimeLabel(state.runtimeState), style = MaterialTheme.typography.titleMedium)
            Text("Runtime version: ${state.diagnostics?.runtimeVersion ?: "unknown"}")
            state.diagnostics?.loadDurationNanos?.let { Text("Model load: ${"%.2f".format(it / 1_000_000.0)} ms") }
            state.diagnostics?.lastInferenceDurationNanos?.let { Text("Last inference: ${"%.2f".format(it / 1_000_000.0)} ms") }
            Button(onClick = viewModel::loadModel, enabled = state.runtimeState !is RuntimeState.Loading && state.runtimeState !is RuntimeState.Running) { Text("Load model") }
        }
        DashboardCard("Model") {
            Text("Expected path: assets/${OnnxDemoViewModel.MODEL_ASSET}")
            Text(state.metadata?.modelName ?: "No model loaded")
        }
        DashboardCard("Input / output metadata") {
            val metadata = state.metadata
            if (metadata == null) Text("Load a model to inspect its nodes.") else {
                Text("Inputs", style = MaterialTheme.typography.titleMedium)
                metadata.inputs.forEach { Text("• ${it.name}: ${it.type} ${it.shape} ${symbolicSuffix(it.dimensionNames)}") }
                Spacer(Modifier.height(8.dp))
                Text("Outputs", style = MaterialTheme.typography.titleMedium)
                metadata.outputs.forEach { Text("• ${it.name}: ${it.type} ${it.shape} ${symbolicSuffix(it.dimensionNames)}") }
            }
        }
        DashboardCard("Drowsiness semantic contract") {
            val validation = state.contract
            if (validation == null) Text("Contract validation waits for model metadata.") else {
                Text("Runtime metadata: ${validation.metadataStatus}")
                Text("Semantic mapping: ${validation.semanticStatus}")
                validation.messages.forEach { Text("• $it") }
            }
        }
        DashboardCard("Dummy smoke inference") {
            Text(state.dummyMessage)
            Button(onClick = viewModel::runDummyInference, enabled = state.runtimeState is RuntimeState.Ready && state.dummyAvailable) { Text("Run dummy inference") }
            state.lastInference?.let { result ->
                Text("Duration: ${"%.2f".format(result.durationNanos / 1_000_000.0)} ms")
                result.outputs.forEach { output ->
                    Text("${output.name}: ${output.type} ${output.shape}")
                    Text(output.preview.values.joinToString(prefix = "[", postfix = if (output.preview.truncated) ", …]" else "]"))
                }
            }
        }
        DashboardCard("Golden parity") {
            Text("Convention: manifest.json plus raw little-endian float32 files under assets/onnx_test_vectors/.")
            Text("Tolerance: ${state.goldenTolerance}")
            Button(onClick = viewModel::runGoldenTests, enabled = state.runtimeState is RuntimeState.Ready && state.goldenAvailable) { Text("Run golden tests") }
            Text(state.goldenMessage)
            state.goldenResults.forEach { (name, result) -> Text("• $name: ${result.status}${result.message?.let { " — $it" } ?: ""}") }
        }
    }
}

private fun runtimeLabel(state: RuntimeState): String = when (state) {
    RuntimeState.NotLoaded -> "Not loaded"
    RuntimeState.Loading -> "Loading"
    is RuntimeState.Ready -> "Ready"
    is RuntimeState.Running -> "Running"
    is RuntimeState.Failed -> state.error.detail
    RuntimeState.Closed -> "Closed"
}

private fun symbolicSuffix(names: List<String?>): String = names.mapIndexedNotNull { index, name -> name?.let { "$index=$it" } }
    .takeIf { it.isNotEmpty() }?.joinToString(prefix = "symbols(", postfix = ")") ?: ""
