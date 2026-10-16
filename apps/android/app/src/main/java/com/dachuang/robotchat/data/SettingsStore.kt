package com.dachuang.robotchat.data

import android.content.Context
import android.content.SharedPreferences
import androidx.core.content.edit
import com.dachuang.robotchat.model.AppSettings
import com.dachuang.robotchat.model.AnnotationConfiguration
import com.dachuang.robotchat.model.CarConfiguration
import com.dachuang.robotchat.model.BluetoothConfiguration
import com.dachuang.robotchat.model.MappingConfiguration
import com.dachuang.robotchat.model.SpeechEnginePref
import com.dachuang.robotchat.model.UserMode

class SettingsStore(context: Context) {
    private val prefs: SharedPreferences =
        context.getSharedPreferences("robot_companion_settings", Context.MODE_PRIVATE)

    val pinSecurity = PinSecurity(prefs)

    fun load(): AppSettings = AppSettings(
        baseUrl = storedBaseUrl(),
        videoUrl = prefs.getString("video_url", "") ?: "",
        mode = runCatching {
            UserMode.valueOf(prefs.getString("mode", UserMode.ADULT.name)!!)
        }.getOrDefault(UserMode.ADULT),
        autoSpeak = prefs.getBoolean("auto_speak", false),
        dialectTag = prefs.getString("dialect", "zh-CN") ?: "zh-CN",
        notificationsEnabled = prefs.getBoolean("notifications", true),
        dndAlways = prefs.getBoolean("dnd_always", false),
        dndScheduled = prefs.getBoolean("dnd_scheduled", false),
        dndStartMinutes = prefs.getInt("dnd_start", 22 * 60),
        dndEndMinutes = prefs.getInt("dnd_end", 7 * 60),
        speechEngine = runCatching {
            SpeechEnginePref.valueOf(prefs.getString("speech_engine", SpeechEnginePref.AUTO.name)!!)
        }.getOrDefault(SpeechEnginePref.AUTO),
        carConfig = CarConfiguration(
            mapping = MappingConfiguration(
                usableFov = prefs.getFloat("map_usable_fov", 160f),
                maxDistance = prefs.getFloat("map_max_distance", 4f),
                stationaryDistance = prefs.getFloat("map_stationary_distance", .35f),
                stationaryAngle = prefs.getFloat("map_stationary_angle", 20f),
                fusionScans = prefs.getInt("map_fusion_scans", 5),
                minSupport = prefs.getInt("map_min_support", 3),
                speed = prefs.getInt("map_speed", 300),
                maxDriveSpeed = prefs.getInt("map_max_drive_speed", 420),
                turnSpeed = prefs.getInt("map_turn_speed", 1500),
                maxTurnSpeed = prefs.getInt("map_max_turn_speed", 2600),
                stopDistance = prefs.getFloat("map_stop_distance", .40f),
                clearance = prefs.getFloat("map_clearance", .30f),
                runtimeMinutes = prefs.getFloat("map_runtime", 15f),
                output = prefs.getString("map_output", "maps/ble_scene_17") ?: "maps/ble_scene_17",
            ),
            annotation = AnnotationConfiguration(
                mapPath = prefs.getString("annotation_map", "maps/ble_scene_17.pgm") ?: "maps/ble_scene_17.pgm",
                semanticMapPath = prefs.getString("annotation_semantic_map", "maps/ble_scene_17_semantics.json")
                    ?: "maps/ble_scene_17_semantics.json",
                standoff = prefs.getFloat("annotation_standoff", .65f),
                maxTargets = prefs.getInt("annotation_max_targets", 32),
                photosPerTarget = prefs.getInt("annotation_photos_per_target", 2),
                speed = prefs.getInt("annotation_speed", 300),
                maxDriveSpeed = prefs.getInt("annotation_max_drive_speed", 460),
                turnSpeed = prefs.getInt("annotation_turn_speed", 1500),
                stopDistance = prefs.getFloat("annotation_stop_distance", .42f),
                clearance = prefs.getFloat("annotation_clearance", .30f),
                settleSeconds = prefs.getFloat("annotation_settle", 1f),
                runtimeMinutes = prefs.getFloat("annotation_runtime", 30f),
            ),
            bluetooth = BluetoothConfiguration(
                deviceName = prefs.getString("ble_device_name", "ESP32-Robot-Car") ?: "ESP32-Robot-Car",
                deviceAddress = prefs.getString("ble_device_address", "AA:BB:CC:DD:EE:01") ?: "AA:BB:CC:DD:EE:01",
                autoRecover = prefs.getBoolean("ble_auto_recover", true),
                failureThreshold = prefs.getInt("ble_failure_threshold", 3),
                maxRecoveryAttempts = prefs.getInt("ble_max_recovery_attempts", 2),
                startupGraceSeconds = prefs.getFloat("ble_startup_grace_seconds", 45f),
                recoveryCooldownSeconds = prefs.getFloat("ble_recovery_cooldown_seconds", 2f),
            ),
            audioVolume = prefs.getFloat("audio_volume", .80f).coerceIn(0f, 1f),
        ),
    )

    private fun storedBaseUrl(): String {
        val fallback = AppSettings().baseUrl
        val stored = prefs.getString("base_url", fallback) ?: fallback
        return if (stored == "http://192.0.2.100:8000") fallback else stored
    }

    fun save(value: AppSettings) {
        prefs.edit {
            putString("base_url", value.baseUrl.trim().trimEnd('/'))
            putString("video_url", value.videoUrl.trim())
            putString("mode", value.mode.name)
            putBoolean("auto_speak", value.autoSpeak)
            putString("dialect", value.dialectTag)
            putBoolean("notifications", value.notificationsEnabled)
            putBoolean("dnd_always", value.dndAlways)
            putBoolean("dnd_scheduled", value.dndScheduled)
            putInt("dnd_start", value.dndStartMinutes)
            putInt("dnd_end", value.dndEndMinutes)
            putString("speech_engine", value.speechEngine.name)
            val mapping = value.carConfig.mapping
            putFloat("map_usable_fov", mapping.usableFov)
            putFloat("map_max_distance", mapping.maxDistance)
            putFloat("map_stationary_distance", mapping.stationaryDistance)
            putFloat("map_stationary_angle", mapping.stationaryAngle)
            putInt("map_fusion_scans", mapping.fusionScans)
            putInt("map_min_support", mapping.minSupport)
            putInt("map_speed", mapping.speed)
            putInt("map_max_drive_speed", mapping.maxDriveSpeed)
            putInt("map_turn_speed", mapping.turnSpeed)
            putInt("map_max_turn_speed", mapping.maxTurnSpeed)
            putFloat("map_stop_distance", mapping.stopDistance)
            putFloat("map_clearance", mapping.clearance)
            putFloat("map_runtime", mapping.runtimeMinutes)
            putString("map_output", mapping.output)
            val annotation = value.carConfig.annotation
            putString("annotation_map", annotation.mapPath)
            putString("annotation_semantic_map", annotation.semanticMapPath)
            putFloat("annotation_standoff", annotation.standoff)
            putInt("annotation_max_targets", annotation.maxTargets)
            putInt("annotation_photos_per_target", annotation.photosPerTarget)
            putInt("annotation_speed", annotation.speed)
            putInt("annotation_max_drive_speed", annotation.maxDriveSpeed)
            putInt("annotation_turn_speed", annotation.turnSpeed)
            putFloat("annotation_stop_distance", annotation.stopDistance)
            putFloat("annotation_clearance", annotation.clearance)
            putFloat("annotation_settle", annotation.settleSeconds)
            putFloat("annotation_runtime", annotation.runtimeMinutes)
            val bluetooth = value.carConfig.bluetooth
            putString("ble_device_name", bluetooth.deviceName)
            putString("ble_device_address", bluetooth.deviceAddress)
            putBoolean("ble_auto_recover", bluetooth.autoRecover)
            putInt("ble_failure_threshold", bluetooth.failureThreshold)
            putInt("ble_max_recovery_attempts", bluetooth.maxRecoveryAttempts)
            putFloat("ble_startup_grace_seconds", bluetooth.startupGraceSeconds)
            putFloat("ble_recovery_cooldown_seconds", bluetooth.recoveryCooldownSeconds)
            putFloat("audio_volume", value.carConfig.audioVolume.coerceIn(0f, 1f))
        }
    }
}
