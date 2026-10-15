package com.dachuang.robotchat

import android.app.Application
import android.os.SystemClock
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.dachuang.robotchat.data.ChatHistoryStore
import com.dachuang.robotchat.data.ChildSafetyPolicy
import com.dachuang.robotchat.data.SettingsStore
import com.dachuang.robotchat.model.AppSettings
import com.dachuang.robotchat.model.AppTab
import com.dachuang.robotchat.model.AppUiState
import com.dachuang.robotchat.model.ChatMessage
import com.dachuang.robotchat.model.CarConfiguration
import com.dachuang.robotchat.model.MessageSender
import com.dachuang.robotchat.model.SpeechEnginePref
import com.dachuang.robotchat.model.UserMode
import com.dachuang.robotchat.model.VoskModelState
import com.dachuang.robotchat.model.VoskModelStatus
import com.dachuang.robotchat.network.RobotApi
import com.dachuang.robotchat.speech.VoskModelManager
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class RobotViewModel(application: Application) : AndroidViewModel(application) {
    private val settingsStore = SettingsStore(application)
    private val historyStore = ChatHistoryStore(application)
    private val api = RobotApi()

    private val initialSettings = settingsStore.load()
    private val initialMessages = historyStore.load().ifEmpty {
        listOf(ChatMessage(text = "你好，我是小车助手。请在设置中填写树莓派服务地址。", sender = MessageSender.ROBOT))
    }
    private val _uiState = MutableStateFlow(
        AppUiState(
            settings = initialSettings,
            messages = initialMessages,
            useVoiceInput = initialSettings.mode == UserMode.SENIOR,
            hasGuardianPin = settingsStore.pinSecurity.hasPin(),
        ),
    )
    val uiState = _uiState.asStateFlow()

    private val _speechEvents = MutableSharedFlow<String>(extraBufferCapacity = 4)
    val speechEvents = _speechEvents.asSharedFlow()
    private val _notificationEvents = MutableSharedFlow<String>(extraBufferCapacity = 4)
    val notificationEvents = _notificationEvents.asSharedFlow()

    private var lastReplyKey = ""
    private var voskJob: Job? = null
    private var voiceStartedAtMs = 0L
    private var releasedVoiceDurationSeconds: Int? = null
    private var pendingVoiceText: String? = null
    private var voiceCancelled = false

    init {
        viewModelScope.launch { pollRobot() }
        refreshVoskStatus()
        checkFirstLaunchVoskPrompt()
    }

    private fun checkFirstLaunchVoskPrompt() {
        val app = getApplication<android.app.Application>()
        val prefs = app.getSharedPreferences("robot_companion_settings", android.content.Context.MODE_PRIVATE)
        val promptShown = prefs.getBoolean("vosk_prompt_shown", false)
        if (!promptShown && !VoskModelManager.isReady(app)) {
            _uiState.value = _uiState.value.copy(showVoskPrompt = true)
        }
    }

    fun dismissVoskPrompt(download: Boolean) {
        val app = getApplication<android.app.Application>()
        app.getSharedPreferences("robot_companion_settings", android.content.Context.MODE_PRIVATE)
            .edit().putBoolean("vosk_prompt_shown", true).apply()
        _uiState.value = _uiState.value.copy(showVoskPrompt = false)
        if (download) {
            downloadVoskModel()
        }
    }

    fun selectTab(tab: AppTab) {
        _uiState.value = _uiState.value.copy(selectedTab = tab)
    }

    fun setVoiceInput(enabled: Boolean) {
        _uiState.value = _uiState.value.copy(useVoiceInput = enabled)
    }

    fun beginVoiceCapture() {
        voiceStartedAtMs = SystemClock.elapsedRealtime()
        releasedVoiceDurationSeconds = null
        pendingVoiceText = null
        voiceCancelled = false
    }

    fun endVoiceCapture() {
        if (voiceStartedAtMs == 0L || voiceCancelled) return
        releasedVoiceDurationSeconds =
            ((SystemClock.elapsedRealtime() - voiceStartedAtMs + 999L) / 1000L).toInt().coerceAtLeast(1)
        voiceStartedAtMs = 0L
        pendingVoiceText?.let { recognized ->
            pendingVoiceText = null
            sendMessage(recognized, releasedVoiceDurationSeconds)
            releasedVoiceDurationSeconds = null
        }
    }

    fun cancelVoiceCapture() {
        voiceCancelled = true
        voiceStartedAtMs = 0L
        pendingVoiceText = null
        releasedVoiceDurationSeconds = null
        _uiState.value = _uiState.value.copy(isListening = false, speechPartial = "")
    }

    fun setListening(listening: Boolean) {
        _uiState.value = _uiState.value.copy(
            isListening = listening,
            speechPartial = if (listening) "" else _uiState.value.speechPartial,
        )
    }

    fun onSpeechPartial(text: String) {
        if (text.isNotBlank()) {
            _uiState.value = _uiState.value.copy(speechPartial = text)
        }
    }

    fun onSpeechResult(text: String) {
        _uiState.value = _uiState.value.copy(isListening = false, speechPartial = "")
        if (voiceCancelled || text.isBlank()) return
        val duration = releasedVoiceDurationSeconds
        if (duration == null) {
            pendingVoiceText = text
        } else {
            releasedVoiceDurationSeconds = null
            sendMessage(text, duration)
        }
    }

    fun onSpeechError(message: String) {
        _uiState.value = _uiState.value.copy(isListening = false, speechPartial = "")
        voiceStartedAtMs = 0L
        pendingVoiceText = null
        releasedVoiceDurationSeconds = null
        if (voiceCancelled) return
        addMessage(ChatMessage(text = message, sender = MessageSender.SYSTEM, isError = true))
    }

    fun setMicrophoneTestRunning(running: Boolean) {
        _uiState.value = _uiState.value.copy(
            microphoneTestRunning = running,
            microphoneTestResult = if (running) "检测中，请持续说话…" else _uiState.value.microphoneTestResult,
        )
    }

    fun setMicrophoneTestResult(result: String) {
        _uiState.value = _uiState.value.copy(
            microphoneTestRunning = false,
            microphoneTestResult = result,
        )
    }

    fun sendMessage(rawText: String, voiceDurationSeconds: Int? = null) {
        val text = rawText.trim()
        if (text.isEmpty() || _uiState.value.isSending) return
        if (_uiState.value.settings.mode == UserMode.CHILD) {
            val safety = ChildSafetyPolicy.validate(text)
            if (!safety.allowed) {
                addMessage(ChatMessage(text = safety.reason, sender = MessageSender.SYSTEM, isError = true))
                return
            }
        }
        addMessage(ChatMessage(text = text, sender = MessageSender.USER, voiceDurationSeconds = voiceDurationSeconds))
        _uiState.value = _uiState.value.copy(isSending = true)
        viewModelScope.launch {
            try {
                val result = api.sendInstruction(_uiState.value.settings.baseUrl, text)
                addMessage(
                    ChatMessage(
                        text = if (result.ok) "小车已接收：${result.message}" else result.message,
                        sender = MessageSender.SYSTEM,
                        isError = !result.ok,
                    ),
                )
            } catch (error: Exception) {
                addMessage(
                    ChatMessage(
                        text = "发送失败：${error.message ?: "无法连接树莓派"}",
                        sender = MessageSender.SYSTEM,
                        isError = true,
                    ),
                )
            } finally {
                _uiState.value = _uiState.value.copy(isSending = false)
            }
        }
    }

    fun speak(text: String) {
        _speechEvents.tryEmit(text)
    }

    fun updateConnection(baseUrl: String, videoUrl: String) {
        updateSettings(_uiState.value.settings.copy(baseUrl = baseUrl, videoUrl = videoUrl))
        setSettingsMessage("连接地址已保存")
    }

    fun testConnection() {
        setSettingsMessage("正在测试连接…")
        viewModelScope.launch {
            val message = runCatching { api.checkHealth(_uiState.value.settings.baseUrl) }
                .fold(
                    onSuccess = { if (it.ok) "连接成功" else "连接失败：${it.message}" },
                    onFailure = { "连接失败：${it.message ?: "未知错误"}" },
                )
            setSettingsMessage(message)
        }
    }

    fun setAutoSpeak(enabled: Boolean) = updateSettings(_uiState.value.settings.copy(autoSpeak = enabled))
    fun setCarConfiguration(value: CarConfiguration) {
        updateSettings(_uiState.value.settings.copy(carConfig = value))
    }

    fun saveCarConfiguration(value: CarConfiguration = _uiState.value.settings.carConfig) {
        setCarConfiguration(value)
        setSettingsMessage("正在保存小车配置…")
        viewModelScope.launch {
            val message = runCatching {
                api.saveCarConfiguration(_uiState.value.settings.baseUrl, value)
            }.fold(
                onSuccess = { if (it.ok) "小车配置已保存到树莓派" else "保存失败：${it.message}" },
                onFailure = { "保存失败：${it.message ?: "无法连接树莓派"}" },
            )
            setSettingsMessage(message)
        }
    }

    fun startCarService(kind: String, value: CarConfiguration = _uiState.value.settings.carConfig) {
        setCarConfiguration(value)
        setSettingsMessage("正在启动${if (kind == "mapping") "建图" else "标注"}服务…")
        viewModelScope.launch {
            val result = runCatching {
                val saved = api.saveCarConfiguration(_uiState.value.settings.baseUrl, value)
                if (!saved.ok) error(saved.message)
                api.startManagedService(_uiState.value.settings.baseUrl, kind)
            }
            val message = result.fold(
                onSuccess = { if (it.ok) "服务已启动" else "启动失败：${it.message}" },
                onFailure = { "启动失败：${it.message ?: "未知错误"}" },
            )
            setSettingsMessage(message)
        }
    }

    fun stopCarService() {
        setSettingsMessage("正在安全停车…")
        viewModelScope.launch {
            val message = runCatching { api.stopManagedService(_uiState.value.settings.baseUrl) }
                .fold(
                    onSuccess = { if (it.ok) "当前小车服务已停止" else "停止失败：${it.message}" },
                    onFailure = { "停止失败：${it.message ?: "未知错误"}" },
                )
            setSettingsMessage(message)
        }
    }

    fun recoverBluetooth() {
        setSettingsMessage("正在安全停止运动任务并重置树莓派蓝牙……")
        viewModelScope.launch {
            val message = runCatching {
                api.recoverBluetooth(_uiState.value.settings.baseUrl)
            }.fold(
                onSuccess = { if (it.ok) "蓝牙恢复已开始，手机服务保持在线" else "蓝牙恢复失败：${it.message}" },
                onFailure = { "蓝牙恢复请求失败：${it.message ?: "无法连接树莓派"}" },
            )
            setSettingsMessage(message)
        }
    }
    fun setDialect(tag: String) = updateSettings(_uiState.value.settings.copy(dialectTag = tag))
    fun setNotifications(enabled: Boolean) =
        updateSettings(_uiState.value.settings.copy(notificationsEnabled = enabled))

    fun setDndAlways(enabled: Boolean) = updateSettings(_uiState.value.settings.copy(dndAlways = enabled))
    fun setDndScheduled(enabled: Boolean) =
        updateSettings(_uiState.value.settings.copy(dndScheduled = enabled))

    fun setDndTimes(startMinutes: Int, endMinutes: Int) = updateSettings(
        _uiState.value.settings.copy(dndStartMinutes = startMinutes, dndEndMinutes = endMinutes),
    )

    fun setGuardianPin(pin: String, currentPin: String = "") {
        val security = settingsStore.pinSecurity
        if (security.hasPin() && !security.verify(currentPin)) {
            setSettingsMessage("原监护密码不正确")
            return
        }
        if (!security.setPin(pin)) {
            setSettingsMessage("监护密码必须是 6 位数字")
            return
        }
        _uiState.value = _uiState.value.copy(hasGuardianPin = true, settingsMessage = "监护密码已保存")
    }

    fun switchMode(target: UserMode, pin: String = "") {
        val current = _uiState.value.settings.mode
        val security = settingsStore.pinSecurity
        if (!security.hasPin()) {
            setSettingsMessage("请先设置 6 位监护密码，再进入少儿模式")
            return
        }
        if (target == current) {
            setSettingsMessage("当前已经是所选模式")
            return
        }
        if (!security.verify(pin)) {
            setSettingsMessage("监护密码不正确，无法修改使用模式")
            return
        }
        val settings = _uiState.value.settings.copy(
            mode = target,
            autoSpeak = if (target == UserMode.SENIOR) true else _uiState.value.settings.autoSpeak,
        )
        updateSettings(settings)
        _uiState.value = _uiState.value.copy(
            selectedTab = AppTab.CHAT,
            useVoiceInput = target == UserMode.SENIOR,
            settingsMessage = when (target) {
                UserMode.CHILD -> "已进入少儿模式"
                UserMode.ADULT -> "已进入成人模式"
                UserMode.SENIOR -> "已进入老人模式，已开启自动朗读"
            },
        )
    }

    fun verifyGuardianPin(pin: String): Boolean {
        val verified = settingsStore.pinSecurity.verify(pin)
        setSettingsMessage(if (verified) "身份验证通过" else "监护密码不正确")
        return verified
    }

    fun emergencyStop(enabled: Boolean) {
        viewModelScope.launch {
            val result = runCatching { api.setEmergencyStop(_uiState.value.settings.baseUrl, enabled) }
            val text = result.fold(
                onSuccess = { if (enabled) "已发送急停指令" else "已请求解除急停" },
                onFailure = { "急停指令发送失败：${it.message}" },
            )
            addMessage(ChatMessage(text = text, sender = MessageSender.SYSTEM, isError = result.isFailure))
        }
    }

    fun resetTask() {
        viewModelScope.launch {
            val result = runCatching { api.resetTask(_uiState.value.settings.baseUrl) }
            addMessage(
                ChatMessage(
                    text = result.fold({ "已请求重置当前任务" }, { "重置失败：${it.message}" }),
                    sender = MessageSender.SYSTEM,
                    isError = result.isFailure,
                ),
            )
        }
    }

    fun clearChat() {
        val greeting = ChatMessage(text = "对话已清空。", sender = MessageSender.SYSTEM)
        historyStore.clear()
        _uiState.value = _uiState.value.copy(messages = listOf(greeting))
        historyStore.save(listOf(greeting))
    }

    // --- Vosk model management ---

    fun refreshVoskStatus() {
        viewModelScope.launch {
            val state = withContext(Dispatchers.IO) {
                val app = getApplication<android.app.Application>()
                if (VoskModelManager.isReady(app)) {
                    VoskModelState(
                        status = VoskModelStatus.READY,
                        sizeBytes = VoskModelManager.installedSizeBytes(app),
                    )
                } else {
                    VoskModelState(status = VoskModelStatus.NOT_INSTALLED)
                }
            }
            _uiState.value = _uiState.value.copy(voskModel = state)
        }
    }

    fun downloadVoskModel() {
        if (voskJob?.isActive == true) return
        voskJob = viewModelScope.launch {
            val app = getApplication<android.app.Application>()
            _uiState.value = _uiState.value.copy(
                voskModel = VoskModelState(status = VoskModelStatus.DOWNLOADING, progress = 0f),
            )
            try {
                VoskModelManager.download(app) { progress ->
                    _uiState.value = _uiState.value.copy(
                        voskModel = _uiState.value.voskModel.copy(
                            status = VoskModelStatus.DOWNLOADING,
                            progress = progress,
                        ),
                    )
                }
                val size = VoskModelManager.installedSizeBytes(app)
                _uiState.value = _uiState.value.copy(
                    voskModel = VoskModelState(
                        status = VoskModelStatus.READY,
                        sizeBytes = size,
                        message = "下载完成",
                    ),
                )
            } catch (e: CancellationException) {
                _uiState.value = _uiState.value.copy(
                    voskModel = VoskModelState(status = VoskModelStatus.NOT_INSTALLED),
                )
                throw e
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    voskModel = VoskModelState(
                        status = VoskModelStatus.ERROR,
                        message = "下载失败：${e.message ?: "未知错误"}",
                    ),
                )
            }
        }
    }

    fun cancelVoskDownload() {
        voskJob?.cancel()
        voskJob = null
        _uiState.value = _uiState.value.copy(
            voskModel = VoskModelState(status = VoskModelStatus.NOT_INSTALLED),
        )
    }

    fun deleteVoskModel() {
        viewModelScope.launch {
            val app = getApplication<android.app.Application>()
            withContext(Dispatchers.IO) { VoskModelManager.deleteModel(app) }
            _uiState.value = _uiState.value.copy(
                voskModel = VoskModelState(status = VoskModelStatus.NOT_INSTALLED),
            )
        }
    }

    fun setSpeechEngine(pref: SpeechEnginePref) {
        updateSettings(_uiState.value.settings.copy(speechEngine = pref))
    }

    private suspend fun pollRobot() {
        while (viewModelScope.coroutineContext.isActive) {
            val baseUrl = _uiState.value.settings.baseUrl
            try {
                val snapshot = api.fetchState(baseUrl)
                _uiState.value = _uiState.value.copy(robot = snapshot)
                handleIncomingReply(snapshot.replyId, snapshot.replyText)
                delay(1_500)
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                val current = _uiState.value.robot
                _uiState.value = _uiState.value.copy(
                    robot = current.copy(
                        connected = false,
                        cameraConnected = false,
                        groundingStatus = error.message ?: "连接失败",
                    ),
                )
                delay(3_000)
            }
        }
    }

    private fun handleIncomingReply(id: String, text: String) {
        if (text.isBlank()) return
        val key = id.ifBlank { text }
        if (key == lastReplyKey) return
        lastReplyKey = key
        addMessage(ChatMessage(text = text, sender = MessageSender.ROBOT))
        val state = _uiState.value
        if (!state.dndActive) {
            if (state.settings.autoSpeak || state.settings.mode == UserMode.SENIOR) _speechEvents.tryEmit(text)
            if (state.settings.notificationsEnabled) _notificationEvents.tryEmit(text)
        }
    }

    private fun updateSettings(settings: AppSettings) {
        settingsStore.save(settings)
        _uiState.value = _uiState.value.copy(settings = settings)
    }

    private fun setSettingsMessage(message: String) {
        _uiState.value = _uiState.value.copy(settingsMessage = message)
    }

    private fun addMessage(message: ChatMessage) {
        val messages = (_uiState.value.messages + message).takeLast(100)
        _uiState.value = _uiState.value.copy(messages = messages)
        historyStore.save(messages)
    }
}
