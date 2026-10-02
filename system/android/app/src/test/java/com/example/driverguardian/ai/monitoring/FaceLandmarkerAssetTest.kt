package com.example.driverguardian.ai.monitoring

import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest
import kotlin.io.path.isReadable
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FaceLandmarkerAssetTest {
    @Test
    fun assetExistsIsReadableAndMatchesDocumentedSha256() {
        val asset = Path.of("src/main/assets/face_landmarker.task")

        assertTrue("face_landmarker.task must be committed under app/src/main/assets", Files.exists(asset))
        assertTrue("face_landmarker.task must be readable", asset.isReadable())
        val bytes = Files.readAllBytes(asset)
        assertEquals(3_758_596, bytes.size)
        assertArrayEquals(
            hex("64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
            MessageDigest.getInstance("SHA-256").digest(bytes),
        )
    }

    private fun hex(value: String): ByteArray =
        value.chunked(2).map { it.toInt(16).toByte() }.toByteArray()
}
