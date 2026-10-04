package com.example.driverguardian.ui.screens.onnxdemo

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.driverguardian.ai.runtime.RuntimeState
import com.example.driverguardian.ui.components.DashboardCard
import com.example.driverguardian.ui.theme.SafeGreen
import com.example.driverguardian.ui.theme.WarningYellow

@Composable
fun OnnxDemoScreen(viewModel: OnnxDemoViewModel = viewModel()) {
    val state by viewModel.uiState.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
            .padding(bottom = 24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text("ONNX Runtime Demo", style = MaterialTheme.typography.headlineMedium)
        Text(
            "Demo end-to-end ONNX model verification: Camera / synthetic preprocessing -> ONNX Runtime Android -> Logits. " +
                "Unvalidated demo artifact — not connected to safety triggers.",
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        // Runtime status card
        DashboardCard("Runtime Status") {
            Text(runtimeLabel(state.runtimeState), style = MaterialTheme.typography.titleMedium)
            Text("Runtime version: ${state.diagnostics?.runtimeVersion ?: "unknown"}")
            state.diagnostics?.loadDurationNanos?.let {
                Text("Model load: ${"%.2f".format(it / 1_000_000.0)} ms")
            }
            state.diagnostics?.lastInferenceDurationNanos?.let {
                Text("Last inference: ${"%.2f".format(it / 1_000_000.0)} ms")
            }
            Spacer(Modifier.height(8.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    onClick = { viewModel.loadModel(OnnxDemoViewModel.DEMO_MODEL_ASSET) },
                    enabled = state.runtimeState !is RuntimeState.Loading && state.runtimeState !is RuntimeState.Running,
                ) {
                    Text("Load Demo Model")
                }
            }
        }

        // Model card
        DashboardCard("Model Asset") {
            Text("Active asset: assets/${state.activeModelAsset}")
            Text("Loaded name: ${state.metadata?.modelName ?: "No model loaded"}")
        }

        // DEMO ONNX OUTPUT card (Phases 8 & 9)
        DashboardCard("DEMO ONNX OUTPUT") {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFF2A2A2A), RoundedCornerShape(8.dp))
                    .padding(12.dp),
            ) {
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(
                        "TECHNICAL VALIDATION ONLY",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                        color = WarningYellow,
                    )
                    Text(
                        "This output is from an unvalidated demo artifact. It is NOT certified driver state and MUST NOT trigger warning/danger alerts.",
                        style = MaterialTheme.typography.bodySmall,
                        color = Color.LightGray,
                    )
                }
            }

            Spacer(Modifier.height(8.dp))

            val demoOutput = state.demoOutput
            if (demoOutput != null) {
                Text("Model Loaded: ${if (demoOutput.modelLoaded) "YES" else "NO"}", fontWeight = FontWeight.SemiBold)
                Text("Sequence Length (T): ${demoOutput.sequenceFrames} frame(s)")
                Text("Inference Duration: ${"%.2f".format(demoOutput.inferenceMs)} ms")

                Spacer(Modifier.height(4.dp))
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Column {
                        Text("Alert Probability (Class 0):", style = MaterialTheme.typography.bodyMedium)
                        Text(
                            "${"%.2f".format(demoOutput.alertProbability * 100)}%",
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                            color = SafeGreen,
                        )
                        Text("Raw logit: ${"%.4f".format(demoOutput.rawAlertLogit)}", style = MaterialTheme.typography.bodySmall)
                    }
                    Column {
                        Text("Drowsy Probability (Class 1):", style = MaterialTheme.typography.bodyMedium)
                        Text(
                            "${"%.2f".format(demoOutput.drowsyProbability * 100)}%",
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                            color = WarningYellow,
                        )
                        Text("Raw logit: ${"%.4f".format(demoOutput.rawDrowsyLogit)}", style = MaterialTheme.typography.bodySmall)
                    }
                }
            } else {
                Text("No demo inference has been run yet. Load model and tap below.")
            }

            Spacer(Modifier.height(8.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    onClick = { viewModel.runDemoInference(sequenceLength = 1) },
                    enabled = state.runtimeState is RuntimeState.Ready,
                ) {
                    Text("Run Demo (T=1)")
                }
                OutlinedButton(
                    onClick = { viewModel.runDemoInference(sequenceLength = 10) },
                    enabled = state.runtimeState is RuntimeState.Ready,
                ) {
                    Text("Run Demo (T=10)")
                }
            }
        }

        // Input / output metadata
        DashboardCard("Tensor Metadata") {
            val metadata = state.metadata
            if (metadata == null) {
                Text("Load a model to inspect its tensor nodes.")
            } else {
                Text("Inputs", style = MaterialTheme.typography.titleMedium)
                metadata.inputs.forEach {
                    Text("• ${it.name}: ${it.type} ${it.shape} ${symbolicSuffix(it.dimensionNames)}")
                }
                Spacer(Modifier.height(8.dp))
                Text("Outputs", style = MaterialTheme.typography.titleMedium)
                metadata.outputs.forEach {
                    Text("• ${it.name}: ${it.type} ${it.shape} ${symbolicSuffix(it.dimensionNames)}")
                }
            }
        }

        // Contract validation card
        DashboardCard("Contract Validation") {
            val validation = state.contract
            if (validation == null) {
                Text("Contract validation waits for model metadata.")
            } else {
                Text("Runtime metadata: ${validation.metadataStatus}")
                Text("Semantic mapping: ${validation.semanticStatus}")
                validation.messages.forEach { Text("• $it") }
            }
        }

        // Golden parity card
        DashboardCard("Golden Parity") {
            Text("Convention: manifest.json plus raw little-endian float32 files under assets/onnx_test_vectors/.")
            Text("Tolerance: ${state.goldenTolerance}")
            Button(
                onClick = viewModel::runGoldenTests,
                enabled = state.runtimeState is RuntimeState.Ready && state.goldenAvailable,
            ) {
                Text("Run golden tests")
            }
            Text(state.goldenMessage)
            state.goldenResults.forEach { (name, result) ->
                Text("• $name: ${result.status}${result.message?.let { " — $it" } ?: ""}")
            }
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

private fun symbolicSuffix(names: List<String?>): String = names.mapIndexedNotNull { index, name ->
    name?.let { "$index=$it" }
}.takeIf { it.isNotEmpty() }?.joinToString(prefix = "symbols(", postfix = ")") ?: ""
