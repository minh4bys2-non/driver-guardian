package com.example.driverguardian.ai.parity

import android.content.res.AssetManager
import com.example.driverguardian.ai.tensor.FloatTensor
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class LoadedGoldenSuite(val tolerance: NumericalTolerance, val cases: List<GoldenTestCase>)

object GoldenVectorAssetLoader {
    fun load(assets: AssetManager, manifestPath: String = GoldenVectorConvention().manifestAsset): LoadedGoldenSuite {
        val manifest = JSONObject(assets.open(manifestPath).bufferedReader().use { it.readText() })
        val base = manifestPath.substringBeforeLast('/', "")
        val toleranceJson = manifest.optJSONObject("tolerance")
        val tolerance = if (toleranceJson == null) NumericalTolerance.Unconfigured else NumericalTolerance.Configured(
            toleranceJson.getDouble("atol"), toleranceJson.getDouble("rtol"),
        )
        val casesJson = manifest.getJSONArray("cases")
        val cases = (0 until casesJson.length()).map { caseIndex ->
            val item = casesJson.getJSONObject(caseIndex)
            GoldenTestCase(
                name = item.getString("name"),
                inputs = readTensors(assets, base, item.getJSONArray("inputs")).mapValues { FloatTensor(it.value.first, it.value.second) },
                expectedOutputs = readTensors(assets, base, item.getJSONArray("expectedOutputs")).mapValues { it.value.second },
            )
        }
        return LoadedGoldenSuite(tolerance, cases)
    }

    private fun readTensors(assets: AssetManager, base: String, array: org.json.JSONArray): Map<String, Pair<List<Long>, FloatArray>> =
        (0 until array.length()).associate { index ->
            val tensor = array.getJSONObject(index)
            val shapeJson = tensor.getJSONArray("shape")
            val shape = (0 until shapeJson.length()).map { shapeJson.getLong(it) }
            val expectedCount = shape.fold(1L, Long::times)
            require(expectedCount in 1..Int.MAX_VALUE) { "Invalid shape for ${tensor.getString("name")}" }
            val path = listOf(base, tensor.getString("file")).filter(String::isNotBlank).joinToString("/")
            val bytes = assets.open(path).use { it.readBytes() }
            require(bytes.size.toLong() == expectedCount * Float.SIZE_BYTES) { "$path has ${bytes.size} bytes; expected ${expectedCount * Float.SIZE_BYTES}" }
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            val values = FloatArray(expectedCount.toInt()).also(buffer::get)
            tensor.getString("name") to (shape to values)
        }
}
