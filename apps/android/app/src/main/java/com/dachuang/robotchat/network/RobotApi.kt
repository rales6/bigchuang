package com.dachuang.robotchat.network

import com.dachuang.robotchat.model.RobotSnapshot
import com.dachuang.robotchat.model.CarConfiguration
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URI
import java.net.URL

data class ApiResult(
    val ok: Boolean,
    val message: String,
    val rawJson: String = "",
)

class RobotApi {
    suspend fun fetchState(baseUrl: String): RobotSnapshot = withContext(Dispatchers.IO) {
        val body = request("GET", endpoint(baseUrl, "/state"))
        RobotStateParser.parse(body)
    }

    suspend fun sendInstruction(baseUrl: String, instruction: String): ApiResult =
        sendControl(baseUrl, JSONObject().put("type", "set_instruction").put("instruction", instruction))

    suspend fun setEmergencyStop(baseUrl: String, enabled: Boolean): ApiResult =
        sendControl(baseUrl, JSONObject().put("type", "emergency_stop").put("enabled", enabled))

    suspend fun resetTask(baseUrl: String): ApiResult =
        sendControl(baseUrl, JSONObject().put("type", "reset"))

    suspend fun checkHealth(baseUrl: String): ApiResult = withContext(Dispatchers.IO) {
        val body = request("GET", endpoint(baseUrl, "/health"))
        val json = JSONObject(body)
        ApiResult(json.optString("status") == "ok", json.optString("status", "未知响应"), body)
    }

    suspend fun saveCarConfiguration(baseUrl: String, value: CarConfiguration): ApiResult =
        post(baseUrl, "/config", JSONObject().put("config", configurationJson(value)))

    suspend fun startManagedService(baseUrl: String, service: String): ApiResult {
        require(service in setOf("mapping", "annotation")) { "不支持的服务类型" }
        return post(baseUrl, "/services/$service/start", JSONObject())
    }

    suspend fun stopManagedService(baseUrl: String): ApiResult =
        post(baseUrl, "/services/stop", JSONObject())

    suspend fun recoverBluetooth(baseUrl: String): ApiResult =
        post(baseUrl, "/bluetooth/recover", JSONObject())

    private suspend fun sendControl(baseUrl: String, payload: JSONObject): ApiResult =
        post(baseUrl, "/control", payload)

    private suspend fun post(baseUrl: String, path: String, payload: JSONObject): ApiResult =
        withContext(Dispatchers.IO) {
            val body = request("POST", endpoint(baseUrl, path), payload.toString())
            val json = JSONObject(body)
            val ok = json.optString("type") != "error" && json.optBoolean("ok", true)
            ApiResult(ok, json.optString("message", if (ok) "操作成功" else "操作失败"), body)
        }

    private fun configurationJson(value: CarConfiguration): JSONObject {
        val mapping = value.mapping
        val annotation = value.annotation
        val bluetooth = value.bluetooth
        return JSONObject()
            .put(
                "mapping",
                JSONObject()
                    .put("usable_fov", mapping.usableFov)
                    .put("mapping_max_distance", mapping.maxDistance)
                    .put("stationary_map_distance", mapping.stationaryDistance)
                    .put("stationary_map_angle", mapping.stationaryAngle)
                    .put("stationary_fusion_scans", mapping.fusionScans)
                    .put("stationary_min_support", mapping.minSupport)
                    .put("speed", mapping.speed)
                    .put("max_drive_speed", mapping.maxDriveSpeed)
                    .put("turn_speed", mapping.turnSpeed)
                    .put("max_turn_speed", mapping.maxTurnSpeed)
                    .put("stop_distance", mapping.stopDistance)
                    .put("clearance", mapping.clearance)
                    .put("max_runtime_min", mapping.runtimeMinutes)
                    .put("output", mapping.output),
            )
            .put(
                "annotation",
                JSONObject()
                    .put("map", annotation.mapPath)
                    .put("map_yaml", annotation.mapPath.removeSuffix(".pgm") + ".yaml")
                    .put("semantic_map", annotation.semanticMapPath)
                    .put("standoff", annotation.standoff)
                    .put("max_targets", annotation.maxTargets)
                    .put("photos_per_target", annotation.photosPerTarget)
                    .put("speed", annotation.speed)
                    .put("max_drive_speed", annotation.maxDriveSpeed)
                    .put("turn_speed", annotation.turnSpeed)
                    .put("stop_distance", annotation.stopDistance)
                    .put("clearance", annotation.clearance)
                    .put("settle_seconds", annotation.settleSeconds)
                    .put("max_runtime_min", annotation.runtimeMinutes),
            )
            .put(
                "bluetooth",
                JSONObject()
                    .put("device_name", bluetooth.deviceName)
                    .put("device_address", bluetooth.deviceAddress)
                    .put("auto_recover", bluetooth.autoRecover)
                    .put("failure_threshold", bluetooth.failureThreshold)
                    .put("max_recovery_attempts", bluetooth.maxRecoveryAttempts)
                    .put("startup_grace_seconds", bluetooth.startupGraceSeconds)
                    .put("recovery_cooldown_seconds", bluetooth.recoveryCooldownSeconds),
            )
            .put("audio", JSONObject().put("volume", value.audioVolume))
    }

    private fun endpoint(baseUrl: String, path: String): URL {
        val normalized = baseUrl.trim().trimEnd('/')
        val uri = runCatching { URI(normalized) }.getOrElse { throw IOException("服务地址格式不正确") }
        if (uri.scheme !in setOf("http", "https") || uri.host.isNullOrBlank()) {
            throw IOException("服务地址必须以 http:// 或 https:// 开头")
        }
        return URL("$normalized$path")
    }

    private fun request(method: String, url: URL, jsonBody: String? = null): String {
        val connection = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = 4_000
            readTimeout = 6_000
            useCaches = false
            setRequestProperty("Accept", "application/json")
            if (jsonBody != null) {
                doOutput = true
                setRequestProperty("Content-Type", "application/json; charset=utf-8")
            }
        }
        return try {
            if (jsonBody != null) {
                connection.outputStream.bufferedWriter(Charsets.UTF_8).use { it.write(jsonBody) }
            }
            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val body = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }.orEmpty()
            if (code !in 200..299) throw IOException("服务器返回 HTTP $code${body.takeIf { it.isNotBlank() }?.let { ": $it" }.orEmpty()}")
            if (body.isBlank()) throw IOException("服务器返回了空内容")
            body
        } finally {
            connection.disconnect()
        }
    }
}
