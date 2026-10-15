package com.dachuang.robotchat.model

import java.time.LocalTime
import java.util.UUID

enum class UserMode { CHILD, ADULT, SENIOR }

enum class AppTab { CHAT, MONITOR, TASKS, SETTINGS }

enum class MessageSender { USER, ROBOT, SYSTEM }

enum class SpeechEnginePref { AUTO, SYSTEM }

enum class VoskModelStatus { NOT_INSTALLED, DOWNLOADING, READY, ERROR }

data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val text: String,
    val sender: MessageSender,
    val timestampMillis: Long = System.currentTimeMillis(),
    val isError: Boolean = false,
    // Voice commands are displayed as a voice bubble; text stays available
    // internally for the current text-only robot command API.
    val voiceDurationSeconds: Int? = null,
)

data class DialectOption(
    val tag: String,
    val label: String,
    val hint: String,
)

data class VoskModelState(
    val status: VoskModelStatus = VoskModelStatus.NOT_INSTALLED,
    val progress: Float = 0f,
    val message: String = "",
    val sizeBytes: Long = 0L,
) {
    val formattedSize: String
        get() = if (sizeBytes <= 0) "" else {
            val mb = sizeBytes / (1024.0 * 1024.0)
            "%.1f MB".format(mb)
        }
}

val SupportedDialects = listOf(
    DialectOption("zh-CN", "普通话", "中国大陆普通话"),
    DialectOption("zh-HK", "粤语", "需要系统语音引擎安装粤语语音"),
    DialectOption("zh-TW", "台湾国语", "需要系统语音引擎安装台湾中文语音"),
    DialectOption("en-US", "English", "US English"),
)

data class AppSettings(
    val baseUrl: String = "http://172.18.14.74:8770",
    val videoUrl: String = "",
    val mode: UserMode = UserMode.ADULT,
    val autoSpeak: Boolean = false,
    val dialectTag: String = "zh-CN",
    val notificationsEnabled: Boolean = true,
    val dndAlways: Boolean = false,
    val dndScheduled: Boolean = false,
    val dndStartMinutes: Int = 22 * 60,
    val dndEndMinutes: Int = 7 * 60,
    val speechEngine: SpeechEnginePref = SpeechEnginePref.AUTO,
    val carConfig: CarConfiguration = CarConfiguration(),
) {
    val resolvedVideoUrl: String
        get() = videoUrl.trim().ifEmpty { "${baseUrl.trimEnd('/')}/stream.mjpg" }
}

data class MappingConfiguration(
    val usableFov: Float = 160f,
    val maxDistance: Float = 4f,
    val stationaryDistance: Float = .35f,
    val stationaryAngle: Float = 20f,
    val fusionScans: Int = 5,
    val minSupport: Int = 3,
    val speed: Int = 300,
    val maxDriveSpeed: Int = 420,
    val turnSpeed: Int = 1500,
    val maxTurnSpeed: Int = 2600,
    val stopDistance: Float = .40f,
    val clearance: Float = .30f,
    val runtimeMinutes: Float = 15f,
    val output: String = "maps/ble_scene_17",
)

data class AnnotationConfiguration(
    val mapPath: String = "maps/ble_scene_17.pgm",
    val semanticMapPath: String = "maps/ble_scene_17_semantics.json",
    val standoff: Float = .65f,
    val maxTargets: Int = 32,
    val photosPerTarget: Int = 2,
    val speed: Int = 300,
    val maxDriveSpeed: Int = 460,
    val turnSpeed: Int = 1500,
    val stopDistance: Float = .42f,
    val clearance: Float = .30f,
    val settleSeconds: Float = 1f,
    val runtimeMinutes: Float = 30f,
)

data class BluetoothConfiguration(
    val deviceName: String = "ESP32-Robot-Car",
    val deviceAddress: String = "64:B7:08:69:9A:E6",
    val autoRecover: Boolean = true,
    val failureThreshold: Int = 3,
    val maxRecoveryAttempts: Int = 2,
    val startupGraceSeconds: Float = 45f,
    val recoveryCooldownSeconds: Float = 2f,
)

data class CarConfiguration(
    val mapping: MappingConfiguration = MappingConfiguration(),
    val annotation: AnnotationConfiguration = AnnotationConfiguration(),
    val bluetooth: BluetoothConfiguration = BluetoothConfiguration(),
    val audioVolume: Float = .80f,
)

object DndSchedule {
    fun isActive(settings: AppSettings, now: LocalTime = LocalTime.now()): Boolean {
        if (settings.dndAlways) return true
        if (!settings.dndScheduled) return false
        val current = now.hour * 60 + now.minute
        val start = settings.dndStartMinutes.coerceIn(0, 1439)
        val end = settings.dndEndMinutes.coerceIn(0, 1439)
        if (start == end) return true
        return if (start < end) current in start until end else current >= start || current < end
    }
}

data class RobotTaskState(
    val active: Boolean = false,
    val taskType: String = "none",
    val phase: String = "idle",
    val phaseLabel: String = "暂无任务",
    val completed: Boolean = false,
    val progress: Float = 0f,
    val completedTargets: Int = 0,
    val totalTargets: Int = 0,
    val currentTarget: String = "",
)

data class TaskPlanStep(
    val id: String,
    val title: String,
    val detail: String = "",
    val status: String = "pending",
)

data class RobotTaskPlan(
    val instruction: String = "",
    val status: String = "idle",
    val summary: String = "",
    val completedSteps: Int = 0,
    val totalSteps: Int = 0,
    val mapUrl: String = "/task/map.png",
    val updatedAt: Long = 0L,
    val steps: List<TaskPlanStep> = emptyList(),
)

data class RobotServiceState(
    val kind: String = "idle",
    val running: Boolean = false,
    val logPath: String = "",
    val lastExitCode: Int? = null,
)

data class RobotSnapshot(
    val connected: Boolean = false,
    val cameraConnected: Boolean = false,
    val groundingStatus: String = "等待连接",
    val emergencyStop: Boolean = false,
    val obstacleStatus: String = "未运行",
    val lidarStatus: String = "未运行",
    val robotCommand: String = "idle",
    val commandReason: String = "",
    val task: RobotTaskState = RobotTaskState(),
    val taskPlan: RobotTaskPlan = RobotTaskPlan(),
    val service: RobotServiceState = RobotServiceState(),
    val replyText: String = "",
    val replyId: String = "",
    val rawJson: String = "",
)

data class AppUiState(
    val settings: AppSettings = AppSettings(),
    val selectedTab: AppTab = AppTab.CHAT,
    val messages: List<ChatMessage> = emptyList(),
    val robot: RobotSnapshot = RobotSnapshot(),
    val isSending: Boolean = false,
    val isListening: Boolean = false,
    val speechPartial: String = "",
    val useVoiceInput: Boolean = false,
    val settingsMessage: String = "",
    val hasGuardianPin: Boolean = false,
    val microphoneTestRunning: Boolean = false,
    val microphoneTestResult: String = "尚未检测",
    val voskModel: VoskModelState = VoskModelState(),
    val showVoskPrompt: Boolean = false,
) {
    val dndActive: Boolean get() = DndSchedule.isActive(settings)
}
