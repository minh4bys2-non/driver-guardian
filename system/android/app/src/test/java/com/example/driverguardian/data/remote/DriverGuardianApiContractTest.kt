package com.example.driverguardian.data.remote

import com.example.driverguardian.data.remote.dto.DrowsinessEventCreateDto
import com.example.driverguardian.data.remote.dto.DrivingSessionCreateDto
import com.google.gson.JsonParser
import kotlinx.coroutines.test.runTest
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Before
import org.junit.Test

class DriverGuardianApiContractTest {
    private lateinit var server: MockWebServer
    private lateinit var api: DriverGuardianApi

    @Before
    fun setUp() {
        server = MockWebServer()
        server.start()
        api = ApiClient.create(server.url("/").toString())
    }

    @After
    fun tearDown() = server.shutdown()

    @Test
    fun `GET drivers uses contract path and parses exact fields`() = runTest {
        server.enqueue(jsonResponse("""[{"driver_id":7,"driver_code":"DRV-007","full_name":"Lan","phone_number":null,"license_number":"B2-7","status":"ACTIVE","created_at":"2026-09-28T01:02:03"}]"""))
        val result = api.getDrivers()
        assertEquals("/drivers", server.takeRequest().path)
        assertEquals(7, result.single().driverId)
        assertEquals("DRV-007", result.single().driverCode)
        assertNull(result.single().phoneNumber)
    }

    @Test
    fun `GET vehicles uses contract path and preserves nullable fields`() = runTest {
        server.enqueue(jsonResponse("""[{"vehicle_id":8,"plate_number":"51A-123.45","vehicle_name":null,"vehicle_type":"BUS","device_code":null,"status":"ACTIVE","created_at":"2026-09-28T01:02:03"}]"""))
        val result = api.getVehicles()
        assertEquals("/vehicles", server.takeRequest().path)
        assertEquals(8, result.single().vehicleId)
        assertNull(result.single().vehicleName)
        assertNull(result.single().deviceCode)
    }

    @Test
    fun `GET active model uses contract path and parses exact fields`() = runTest {
        server.enqueue(jsonResponse("""{"model_version_id":9,"version_name":"drowsy-v1","model_type":null,"file_name":"model.onnx","description":null,"is_active":"Y","deployed_at":"2026-09-28T01:02:03"}"""))
        val result = api.getActiveModelVersion()
        assertEquals("/model-versions/active", server.takeRequest().path)
        assertEquals(9, result.modelVersionId)
        assertEquals("Y", result.isActive)
    }

    @Test
    fun `POST sessions sends real identity ids and parses response`() = runTest {
        server.enqueue(jsonResponse("""{"session_id":11,"driver_id":7,"vehicle_id":8,"model_version_id":null,"start_time":"2026-09-28T01:02:03","duration_seconds":0,"total_alerts":0,"status":"ACTIVE","sync_status":"SYNCED"}""", 201))
        val result = api.createSession(DrivingSessionCreateDto(7, 8, 9))
        val request = server.takeRequest()
        val body = JsonParser.parseString(request.body.readUtf8()).asJsonObject
        assertEquals("/sessions", request.path)
        assertEquals("POST", request.method)
        assertEquals(7, body["driver_id"].asInt)
        assertEquals(8, body["vehicle_id"].asInt)
        assertEquals(9, body["model_version_id"].asInt)
        assertEquals(11, result.sessionId)
        assertNull(result.modelVersionId)
    }

    @Test
    fun `POST events sends danger mapping including nullable fields`() = runTest {
        server.enqueue(jsonResponse("""{"event_id":12,"session_id":11,"event_time":"2026-09-28T01:03:03","driver_state":"DANGER","alert_level":2,"confidence":null,"duration_ms":null,"acknowledged":"N","sync_status":"SYNCED"}""", 201))
        val result = api.createEvent(DrowsinessEventCreateDto(11, "DANGER", 2, null, null))
        val request = server.takeRequest()
        val body = JsonParser.parseString(request.body.readUtf8()).asJsonObject
        assertEquals("/events", request.path)
        assertEquals("DANGER", body["driver_state"].asString)
        assertEquals(2, body["alert_level"].asInt)
        assertEquals(true, body.has("confidence") && body["confidence"].isJsonNull)
        assertEquals(true, body.has("duration_ms") && body["duration_ms"].isJsonNull)
        assertEquals(12, result.eventId)
    }

    private fun jsonResponse(body: String, code: Int = 200) = MockResponse()
        .setResponseCode(code)
        .setHeader("Content-Type", "application/json")
        .setBody(body)
}
