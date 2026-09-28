package com.example.driverguardian.data.repository

import com.example.driverguardian.data.remote.DriverGuardianApi
import com.example.driverguardian.data.remote.dto.DrowsinessEventCreateDto
import com.example.driverguardian.data.remote.dto.DrivingSessionCreateDto
import com.example.driverguardian.data.remote.dto.toDomain
import com.example.driverguardian.domain.model.Driver
import com.example.driverguardian.domain.model.DrowsinessEvent
import com.example.driverguardian.domain.model.DrivingSession
import com.example.driverguardian.domain.model.ModelVersion
import com.example.driverguardian.domain.model.Vehicle
import java.net.SocketTimeoutException
import kotlinx.coroutines.CancellationException
import retrofit2.HttpException

sealed interface RepositoryResult<out T> {
    data class Success<T>(val value: T) : RepositoryResult<T>
    data class Error(
        val message: String,
        val statusCode: Int? = null,
        val cause: Throwable? = null
    ) : RepositoryResult<Nothing>
}

interface DriverGuardianRepository {
    suspend fun getDrivers(): RepositoryResult<List<Driver>>
    suspend fun getVehicles(): RepositoryResult<List<Vehicle>>
    suspend fun getActiveModelVersion(): RepositoryResult<ModelVersion>
    suspend fun createSession(driverId: Int, vehicleId: Int, modelVersionId: Int): RepositoryResult<DrivingSession>
    suspend fun createEvent(
        sessionId: Int,
        driverState: String,
        alertLevel: Int,
        confidence: Double?,
        durationMs: Int?
    ): RepositoryResult<DrowsinessEvent>
}

class NetworkDriverGuardianRepository(
    private val api: DriverGuardianApi
) : DriverGuardianRepository {
    override suspend fun getDrivers() = request { api.getDrivers().map { it.toDomain() } }
    override suspend fun getVehicles() = request { api.getVehicles().map { it.toDomain() } }

    override suspend fun getActiveModelVersion() = request(
        notFoundMessage = "Không tìm thấy mô hình đang hoạt động."
    ) { api.getActiveModelVersion().toDomain() }

    override suspend fun createSession(driverId: Int, vehicleId: Int, modelVersionId: Int) = request {
        api.createSession(DrivingSessionCreateDto(driverId, vehicleId, modelVersionId)).toDomain()
    }

    override suspend fun createEvent(
        sessionId: Int,
        driverState: String,
        alertLevel: Int,
        confidence: Double?,
        durationMs: Int?
    ) = request {
        api.createEvent(DrowsinessEventCreateDto(sessionId, driverState, alertLevel, confidence, durationMs)).toDomain()
    }

    private suspend fun <T> request(
        notFoundMessage: String = "Không tìm thấy dữ liệu yêu cầu.",
        block: suspend () -> T
    ): RepositoryResult<T> = try {
        RepositoryResult.Success(block())
    } catch (error: CancellationException) {
        throw error
    } catch (error: SocketTimeoutException) {
        RepositoryResult.Error("Kết nối máy chủ đã hết thời gian chờ.", cause = error)
    } catch (error: HttpException) {
        val message = when (error.code()) {
            400, 422 -> "Dữ liệu gửi lên không hợp lệ."
            404 -> notFoundMessage
            in 500..599 -> "Máy chủ không thể xử lý yêu cầu."
            else -> "Yêu cầu thất bại (HTTP ${error.code()})."
        }
        RepositoryResult.Error(message, error.code(), error)
    } catch (error: java.io.IOException) {
        RepositoryResult.Error("Không thể kết nối tới máy chủ Driver Guardian.", cause = error)
    } catch (error: Exception) {
        RepositoryResult.Error("Đã xảy ra lỗi khi xử lý phản hồi máy chủ.", cause = error)
    }
}
