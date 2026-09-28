package com.example.driverguardian.data.repository

import com.example.driverguardian.data.remote.ApiClient
import kotlinx.coroutines.test.runTest
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class DriverGuardianRepositoryTest {
    private lateinit var server: MockWebServer
    private lateinit var repository: NetworkDriverGuardianRepository

    @Before
    fun setUp() {
        server = MockWebServer()
        server.start()
        repository = NetworkDriverGuardianRepository(ApiClient.create(server.url("/").toString()))
    }

    @After
    fun tearDown() = server.shutdown()

    @Test
    fun `http failure becomes explicit safe repository error`() = runTest {
        server.enqueue(MockResponse().setResponseCode(500).setBody("{\"detail\":\"internal secret\"}"))
        val result = repository.getDrivers()
        assertTrue(result is RepositoryResult.Error)
        result as RepositoryResult.Error
        assertEquals(500, result.statusCode)
        assertEquals("Máy chủ không thể xử lý yêu cầu.", result.message)
    }

    @Test
    fun `active model 404 has endpoint specific safe message`() = runTest {
        server.enqueue(MockResponse().setResponseCode(404).setBody("{\"detail\":\"No active model version found\"}"))
        val result = repository.getActiveModelVersion()
        assertTrue(result is RepositoryResult.Error)
        result as RepositoryResult.Error
        assertEquals(404, result.statusCode)
        assertEquals("Không tìm thấy mô hình đang hoạt động.", result.message)
    }

    @Test
    fun `complete session http failure is not converted to success`() = runTest {
        server.enqueue(MockResponse().setResponseCode(500).setBody("{\"detail\":\"internal database failure\"}"))
        val result = repository.completeSession(11)
        assertTrue(result is RepositoryResult.Error)
        result as RepositoryResult.Error
        assertEquals(500, result.statusCode)
        assertEquals("Máy chủ không thể xử lý yêu cầu.", result.message)
    }
}
