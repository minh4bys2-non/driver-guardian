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

    @Test
    fun `POST complete session uses path without client timing body and preserves null score`() = runTest {
        server.enqueue(jsonResponse("""{"session_id":11,"driver_id":7,"vehicle_id":8,"model_version_id":9,"start_time":"2026-09-28T01:02:03","end_time":"2026-09-28T02:02:03","duration_seconds":3600,"total_alerts":2,"safety_score":null,"status":"COMPLETED","sync_status":"SYNCED"}"""))
        val result = api.completeSession(11)
        val request = server.takeRequest()
        assertEquals("POST", request.method)
        assertEquals("/sessions/11/complete", request.path)
        assertEquals(0L, request.bodySize)
        assertEquals("COMPLETED", result.status)
        assertNull(result.safetyScore)
    }

    @Test
    fun `GET sessions sends driver filter and parses joined nullable response`() = runTest {
        server.enqueue(jsonResponse("""[{"session_id":11,"driver_id":7,"driver_name":"Lan","vehicle_id":8,"vehicle_name":null,"plate_number":"51A","model_version_id":9,"version_name":"v1","start_time":"2026-09-28T01:02:03","end_time":null,"duration_seconds":0,"total_alerts":0,"safety_score":null,"status":"ACTIVE","sync_status":"SYNCED"}]"""))
        val result = api.getSessions(driverId = 7)
        assertEquals("/sessions?driver_id=7", server.takeRequest().path)
        assertEquals("Lan", result.single().driverName)
        assertNull(result.single().vehicleName)
        assertNull(result.single().endTime)
        assertNull(result.single().safetyScore)
    }

    @Test
    fun `GET session detail uses real session id`() = runTest {
        server.enqueue(jsonResponse("""{"session_id":11,"driver_id":7,"driver_name":"Lan","vehicle_id":8,"vehicle_name":"Bus","plate_number":"51A","model_version_id":9,"version_name":"v1","start_time":"2026-09-28T01:02:03","end_time":"2026-09-28T02:02:03","duration_seconds":3600,"total_alerts":2,"safety_score":null,"status":"COMPLETED","sync_status":"SYNCED"}"""))
        val result = api.getSession(11)
        assertEquals("/sessions/11", server.takeRequest().path)
        assertEquals(11, result.sessionId)
        assertEquals("51A", result.plateNumber)
    }

    @Test
    fun `GET session events preserves all nullable metrics`() = runTest {
        server.enqueue(jsonResponse("""[{"event_id":12,"session_id":11,"event_time":"2026-09-28T01:03:03","driver_state":"WARNING","alert_level":1,"confidence":null,"drowsiness_score":null,"ear_value":null,"mar_value":null,"head_pose":null,"duration_ms":null,"acknowledged":"N","sync_status":"SYNCED"}]"""))
        val event = api.getSessionEvents(11).single()
        assertEquals("/sessions/11/events", server.takeRequest().path)
        assertNull(event.confidence)
        assertNull(event.drowsinessScore)
        assertNull(event.earValue)
        assertNull(event.marValue)
        assertNull(event.headPose)
        assertNull(event.durationMs)
    }

    @Test
    fun `POST acknowledge targets persisted event`() = runTest {
        server.enqueue(jsonResponse("""{"event_id":12,"session_id":11,"event_time":"2026-09-28T01:03:03","driver_state":"DANGER","alert_level":2,"confidence":null,"drowsiness_score":null,"ear_value":null,"mar_value":null,"head_pose":null,"duration_ms":null,"acknowledged":"Y","sync_status":"SYNCED"}"""))
        val result = api.acknowledgeEvent(12)
        val request = server.takeRequest()
        assertEquals("POST", request.method)
        assertEquals("/events/12/acknowledge", request.path)
        assertEquals("Y", result.acknowledged)
    }

    private fun jsonResponse(body: String, code: Int = 200) = MockResponse()
        .setResponseCode(code)
        .setHeader("Content-Type", "application/json")
        .setBody(body)
}
