package com.dachuang.robotchat.ui.screens

import android.app.TimePickerDialog
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowLeft
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.dachuang.robotchat.model.AppUiState
import com.dachuang.robotchat.model.CarConfiguration
import com.dachuang.robotchat.model.SpeechEnginePref
import com.dachuang.robotchat.model.SupportedDialects
import com.dachuang.robotchat.model.UserMode
import com.dachuang.robotchat.model.VoskModelStatus
import com.dachuang.robotchat.ui.components.CheckRow
import com.dachuang.robotchat.ui.components.Chevron
import com.dachuang.robotchat.ui.components.SwitchRow
import com.dachuang.robotchat.ui.components.ValueRow
import com.dachuang.robotchat.ui.components.WeChatCard
import com.dachuang.robotchat.ui.components.WeChatDivider
import com.dachuang.robotchat.ui.components.WeChatGroup
import com.dachuang.robotchat.ui.components.WeChatRow

private enum class SettingsPage { MAIN, MODE_AUTH, MODE, VOICE, CONNECTION, CAR_CONFIG, DND, PIN, DATA }

@Composable
fun SettingsScreen(
    state: AppUiState,
    onConnectionSave: (String, String) -> Unit,
    onTestConnection: () -> Unit,
    onModeChange: (UserMode, String) -> Unit,
    onVerifyGuardianPin: (String) -> Boolean,
    onSetPin: (String, String) -> Unit,
    onAutoSpeak: (Boolean) -> Unit,
    onDialect: (String) -> Unit,
    onNotifications: (Boolean) -> Unit,
    onDndAlways: (Boolean) -> Unit,
    onDndScheduled: (Boolean) -> Unit,
    onDndTimes: (Int, Int) -> Unit,
    onClearChat: () -> Unit,
    onSpeechEngine: (SpeechEnginePref) -> Unit,
    onDownloadVosk: () -> Unit,
    onCancelVosk: () -> Unit,
    onDeleteVosk: () -> Unit,
    onTestMicrophone: () -> Unit,
    onCarConfigSave: (CarConfiguration) -> Unit,
    onStartMapping: (CarConfiguration) -> Unit,
    onStartAnnotation: (CarConfiguration) -> Unit,
    onRecoverBluetooth: () -> Unit,
    onStopCarService: () -> Unit,
) {
    var currentPage by remember { mutableStateOf(SettingsPage.MAIN) }
    var authenticatedPin by remember { mutableStateOf("") }

    when (currentPage) {
        SettingsPage.MAIN -> MainSettingsPage(
            state = state,
            onNavigate = { currentPage = it },
        )
        SettingsPage.MODE_AUTH -> ModeAuthenticationPage(
            state = state,
            onVerify = { pin ->
                if (onVerifyGuardianPin(pin)) {
                    authenticatedPin = pin
                    currentPage = SettingsPage.MODE
                }
            },
            onBack = { currentPage = SettingsPage.MAIN },
        )
        SettingsPage.MODE -> ModePage(
            state = state,
            onModeChange = { mode -> onModeChange(mode, authenticatedPin) },
            onBack = {
                authenticatedPin = ""
                currentPage = SettingsPage.MODE_AUTH
            },
        )
        SettingsPage.VOICE -> VoicePage(
            state = state,
            onAutoSpeak = onAutoSpeak,
            onDialect = onDialect,
            onSpeechEngine = onSpeechEngine,
            onDownloadVosk = onDownloadVosk,
            onCancelVosk = onCancelVosk,
            onDeleteVosk = onDeleteVosk,
            onTestMicrophone = onTestMicrophone,
            onBack = { currentPage = SettingsPage.MAIN },
        )
        SettingsPage.CONNECTION -> ConnectionPage(
            state = state,
            onConnectionSave = onConnectionSave,
            onTestConnection = onTestConnection,
            onBack = { currentPage = SettingsPage.MAIN },
        )
        SettingsPage.CAR_CONFIG -> CarConfigScreen(
            configuration = state.settings.carConfig,
            service = state.robot.service,
            message = state.settingsMessage,
            onSave = onCarConfigSave,
            onStartMapping = onStartMapping,
            onStartAnnotation = onStartAnnotation,
            onRecoverBluetooth = onRecoverBluetooth,
            onStop = onStopCarService,
            onBack = { currentPage = SettingsPage.MAIN },
        )
        SettingsPage.DND -> DndPage(
            state = state,
            onNotifications = onNotifications,
            onDndAlways = onDndAlways,
            onDndScheduled = onDndScheduled,
            onDndTimes = onDndTimes,
            onBack = { currentPage = SettingsPage.MAIN },
        )
        SettingsPage.PIN -> PinPage(
            state = state,
            onSetPin = onSetPin,
            onBack = { currentPage = SettingsPage.MAIN },
        )
        SettingsPage.DATA -> DataPage(
            onClearChat = onClearChat,
            onBack = { currentPage = SettingsPage.MAIN },
        )
    }
}

// ==================== Main Settings Page ====================

@Composable
private fun MainSettingsPage(state: AppUiState, onNavigate: (SettingsPage) -> Unit) {
    val s = state.settings
    LazyColumn(
        modifier = Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        item {
            Text("设置", style = MaterialTheme.typography.headlineSmall, modifier = Modifier.padding(start = 4.dp, top = 4.dp))
        }

        if (state.settingsMessage.isNotBlank()) {
            item {
                WeChatCard {
                    Text(state.settingsMessage, modifier = Modifier.padding(16.dp), color = MaterialTheme.colorScheme.primary)
                }
            }
        }

        // 使用模式
        item {
            WeChatGroup {
                val modeLabel = when (s.mode) {
                    UserMode.CHILD -> "少儿模式"
                    UserMode.ADULT -> "成人模式"
                    UserMode.SENIOR -> "老人模式"
                }
                ValueRow("使用模式", modeLabel) {
                    onNavigate(if (state.hasGuardianPin) SettingsPage.MODE_AUTH else SettingsPage.PIN)
                }
            }
        }

        // 语音
        if (s.mode != UserMode.CHILD) {
            item {
                val dialectLabel = SupportedDialects.find { it.tag == s.dialectTag }?.label ?: s.dialectTag
                val voskStatus = when (state.voskModel.status) {
                    VoskModelStatus.READY -> " · 离线就绪"
                    VoskModelStatus.DOWNLOADING -> " · 下载中"
                    else -> ""
                }
                WeChatGroup {
                    ValueRow("语音", "$dialectLabel$voskStatus") { onNavigate(SettingsPage.VOICE) }
                }
            }
        }

        // 树莓派连接
        if (s.mode == UserMode.ADULT) {
            item {
                WeChatGroup {
                    ValueRow("树莓派连接", s.baseUrl) { onNavigate(SettingsPage.CONNECTION) }
                    WeChatDivider()
                    ValueRow("小车配置", "建图 · 标注 · 蓝牙") { onNavigate(SettingsPage.CAR_CONFIG) }
                }
            }
        }

        // 消息与免打扰
        item {
            val dndSummary = when {
                s.dndAlways -> "始终免打扰"
                s.dndScheduled -> {
                    val start = "%02d:%02d".format(s.dndStartMinutes / 60, s.dndStartMinutes % 60)
                    val end = "%02d:%02d".format(s.dndEndMinutes / 60, s.dndEndMinutes % 60)
                    "$start – $end"
                }
                else -> "关闭"
            }
            WeChatGroup {
                ValueRow("消息与免打扰", dndSummary) { onNavigate(SettingsPage.DND) }
            }
        }

        // 监护密码
        if (s.mode == UserMode.ADULT) {
            item {
                WeChatGroup {
                    ValueRow("监护密码", if (state.hasGuardianPin) "已设置" else "未设置") { onNavigate(SettingsPage.PIN) }
                }
            }
        }

        // 本地数据
        if (s.mode == UserMode.ADULT) {
            item {
                WeChatGroup {
                    WeChatRow(
                        label = "本地数据",
                        onClick = { onNavigate(SettingsPage.DATA) },
                    ) { Chevron() }
                }
            }
        }

        item { Spacer(Modifier.height(16.dp)) }
    }
}

// ==================== Sub-page header ====================

@Composable
private fun SubPageHeader(title: String, onBack: () -> Unit) {
    Surface(tonalElevation = 1.dp) {
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 4.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = onBack) {
                Icon(Icons.AutoMirrored.Filled.KeyboardArrowLeft, "返回")
            }
            Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Medium)
        }
    }
}

// ==================== Mode Sub-page ====================

@Composable
private fun ModeAuthenticationPage(
    state: AppUiState,
    onVerify: (String) -> Unit,
    onBack: () -> Unit,
) {
    var pin by remember { mutableStateOf("") }

    Column(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        SubPageHeader("验证监护身份", onBack)
        LazyColumn(
            contentPadding = androidx.compose.foundation.layout.PaddingValues(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            item {
                WeChatGroup("修改使用模式前需要验证") {
                    OutlinedTextField(
                        value = pin,
                        onValueChange = { pin = it.filter(Char::isDigit).take(6) },
                        label = { Text("6 位监护密码") },
                        visualTransformation = PasswordVisualTransformation(),
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.NumberPassword),
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
                        singleLine = true,
                    )
                    Button(
                        onClick = { onVerify(pin) },
                        enabled = pin.length == 6,
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
                    ) {
                        Text("验证并进入模式选择")
                    }
                    if (state.settingsMessage.contains("密码不正确")) {
                        Text(
                            state.settingsMessage,
                            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                            color = MaterialTheme.colorScheme.error,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ModePage(
    state: AppUiState,
    onModeChange: (UserMode) -> Unit,
    onBack: () -> Unit,
) {
    val mode = state.settings.mode

    Column(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        SubPageHeader("使用模式", onBack)
        LazyColumn(
            contentPadding = androidx.compose.foundation.layout.PaddingValues(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            item {
                WeChatCard {
                    Text(
                        "监护身份已验证，请选择要使用的模式。",
                        modifier = Modifier.padding(16.dp),
                        color = MaterialTheme.colorScheme.primary,
                    )
                }
            }
            item {
                WeChatGroup {
                    CheckRow("少儿模式", "限制运动控制指令，退出需密码", mode == UserMode.CHILD) {
                        onModeChange(UserMode.CHILD)
                    }
                    WeChatDivider()
                    CheckRow("成人模式", "完整聊天、视频、任务和连接设置", mode == UserMode.ADULT) {
                        onModeChange(UserMode.ADULT)
                    }
                    WeChatDivider()
                    CheckRow("老人模式", "大字号、语音优先、自动朗读", mode == UserMode.SENIOR) {
                        onModeChange(UserMode.SENIOR)
                    }
                }
            }
        }
    }
}

// ==================== Voice Sub-page ====================

@Composable
private fun VoicePage(
    state: AppUiState,
    onAutoSpeak: (Boolean) -> Unit,
    onDialect: (String) -> Unit,
    onSpeechEngine: (SpeechEnginePref) -> Unit,
    onDownloadVosk: () -> Unit,
    onCancelVosk: () -> Unit,
    onDeleteVosk: () -> Unit,
    onTestMicrophone: () -> Unit,
    onBack: () -> Unit,
) {
    var showDialectDialog by remember { mutableStateOf(false) }

    Column(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        SubPageHeader("语音", onBack)
        LazyColumn(
            contentPadding = androidx.compose.foundation.layout.PaddingValues(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            // Auto-speak
            item {
                WeChatGroup {
                    SwitchRow("自动朗读树莓派回复", checked = state.settings.autoSpeak, onChecked = onAutoSpeak)
                }
            }

            item {
                WeChatGroup("语音输入诊断") {
                    WeChatRow(
                        label = if (state.microphoneTestRunning) "正在检测，请持续说话" else "测试麦克风输入",
                        subtitle = state.microphoneTestResult,
                        onClick = { if (!state.microphoneTestRunning) onTestMicrophone() },
                    ) {
                        if (state.microphoneTestRunning) {
                            CircularProgressIndicator(Modifier.size(22.dp), strokeWidth = 2.dp)
                        } else {
                            Chevron()
                        }
                    }
                }
            }

            // Offline speech
            item {
                WeChatGroup {
                    SwitchRow(
                        "离线语音识别",
                        subtitle = when (state.voskModel.status) {
                            VoskModelStatus.READY -> "模型就绪 · ${state.voskModel.formattedSize}"
                            VoskModelStatus.DOWNLOADING -> "下载中…"
                            VoskModelStatus.ERROR -> state.voskModel.message
                            VoskModelStatus.NOT_INSTALLED -> "安装后可离线使用语音输入（约 42 MB）"
                        },
                        checked = state.settings.speechEngine == SpeechEnginePref.AUTO,
                        onChecked = { checked ->
                            onSpeechEngine(if (checked) SpeechEnginePref.AUTO else SpeechEnginePref.SYSTEM)
                            if (checked && state.voskModel.status == VoskModelStatus.NOT_INSTALLED) {
                                onDownloadVosk()
                            }
                        },
                    )

                    when (state.voskModel.status) {
                        VoskModelStatus.NOT_INSTALLED -> {
                            WeChatDivider()
                            WeChatRow(
                                label = "下载离线语音模型",
                                subtitle = "约 42 MB，下载后断网也可用",
                                onClick = onDownloadVosk,
                            ) { Chevron() }
                        }
                        VoskModelStatus.DOWNLOADING -> {
                            WeChatDivider()
                            Column(
                                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 10.dp),
                            ) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    Text("正在下载模型", style = MaterialTheme.typography.bodyMedium, modifier = Modifier.weight(1f))
                                    val p = state.voskModel.progress
                                    if (p >= 0f) {
                                        Text("${(p * 100).toInt()}%", style = MaterialTheme.typography.bodySmall)
                                    }
                                    Spacer(Modifier.width(8.dp))
                                    TextButton(onClick = onCancelVosk) {
                                        Text("取消", color = MaterialTheme.colorScheme.error)
                                    }
                                }
                                Spacer(Modifier.height(6.dp))
                                LinearProgressIndicator(
                                    progress = { state.voskModel.progress.coerceIn(0f, 1f) },
                                    modifier = Modifier.fillMaxWidth(),
                                )
                            }
                        }
                        VoskModelStatus.READY -> {
                            WeChatDivider()
                            WeChatRow(
                                label = "离线语音模型",
                                subtitle = "已安装 · ${state.voskModel.formattedSize}",
                            ) {
                                TextButton(onClick = onDeleteVosk) {
                                    Text("删除", color = MaterialTheme.colorScheme.error, fontSize = 14.sp)
                                }
                            }
                        }
                        VoskModelStatus.ERROR -> {
                            WeChatDivider()
                            WeChatRow(
                                label = "离线语音模型",
                                subtitle = state.voskModel.message,
                            ) {
                                TextButton(onClick = onDownloadVosk) {
                                    Text("重试", fontSize = 14.sp)
                                }
                            }
                        }
                    }
                }
            }

            // Dialect
            item {
                val current = SupportedDialects.find { it.tag == state.settings.dialectTag }
                WeChatGroup {
                    ValueRow("方言", current?.label ?: state.settings.dialectTag) { showDialectDialog = true }
                }
            }
        }
    }

    if (showDialectDialog) {
        AlertDialog(
            onDismissRequest = { showDialectDialog = false },
            title = { Text("选择方言") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    SupportedDialects.forEach { option ->
                        CheckRow(
                            label = option.label,
                            subtitle = option.hint,
                            selected = state.settings.dialectTag == option.tag,
                        ) { onDialect(option.tag); showDialectDialog = false }
                    }
                }
            },
            confirmButton = { TextButton(onClick = { showDialectDialog = false }) { Text("取消") } },
        )
    }
}

// ==================== Connection Sub-page ====================

@Composable
private fun ConnectionPage(
    state: AppUiState,
    onConnectionSave: (String, String) -> Unit,
    onTestConnection: () -> Unit,
    onBack: () -> Unit,
) {
    var baseUrl by remember { mutableStateOf(state.settings.baseUrl) }
    var videoUrl by remember { mutableStateOf(state.settings.videoUrl) }
    LaunchedEffect(state.settings.baseUrl, state.settings.videoUrl) {
        baseUrl = state.settings.baseUrl
        videoUrl = state.settings.videoUrl
    }

    Column(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        SubPageHeader("树莓派连接", onBack)
        LazyColumn(
            contentPadding = androidx.compose.foundation.layout.PaddingValues(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            item {
                WeChatGroup("服务地址") {
                    OutlinedTextField(
                        value = baseUrl,
                        onValueChange = { baseUrl = it },
                        label = { Text("服务地址") },
                        supportingText = { Text("示例：http://192.168.1.100:8000") },
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
                        singleLine = true,
                    )
                }
            }
            item {
                WeChatGroup("视频地址") {
                    OutlinedTextField(
                        value = videoUrl,
                        onValueChange = { videoUrl = it },
                        label = { Text("视频地址（可留空）") },
                        supportingText = { Text("留空时自动使用 服务地址/stream.mjpg") },
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
                        singleLine = true,
                    )
                }
            }
            item {
                Row(
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Button(onClick = {
                        onConnectionSave(baseUrl, videoUrl)
                    }, modifier = Modifier.weight(1f)) { Text("保存") }
                    Button(onClick = onTestConnection, modifier = Modifier.weight(1f)) { Text("测试连接") }
                }
            }
        }
    }
}

// ==================== DND Sub-page ====================

@Composable
private fun DndPage(
    state: AppUiState,
    onNotifications: (Boolean) -> Unit,
    onDndAlways: (Boolean) -> Unit,
    onDndScheduled: (Boolean) -> Unit,
    onDndTimes: (Int, Int) -> Unit,
    onBack: () -> Unit,
) {
    Column(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        SubPageHeader("消息与免打扰", onBack)
        LazyColumn(
            contentPadding = androidx.compose.foundation.layout.PaddingValues(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            item {
                WeChatGroup {
                    SwitchRow("新消息通知", checked = state.settings.notificationsEnabled, onChecked = onNotifications)
                }
            }
            item {
                WeChatGroup("免打扰方案") {
                    CheckRow("关闭", selected = !state.settings.dndAlways && !state.settings.dndScheduled) {
                        onDndAlways(false); onDndScheduled(false)
                    }
                    WeChatDivider()
                    CheckRow("始终免打扰", selected = state.settings.dndAlways) {
                        onDndAlways(true); onDndScheduled(false)
                    }
                    WeChatDivider()
                    CheckRow("按时间免打扰", selected = state.settings.dndScheduled) {
                        onDndAlways(false); onDndScheduled(true)
                    }
                }
            }
            if (state.settings.dndScheduled) {
                item {
                    WeChatGroup("免打扰时段") {
                        val context = LocalContext.current
                        Row(
                            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            Button(
                                onClick = {
                                    TimePickerDialog(
                                        context,
                                        { _, hour, minute -> onDndTimes(hour * 60 + minute, state.settings.dndEndMinutes) },
                                        state.settings.dndStartMinutes / 60,
                                        state.settings.dndStartMinutes % 60,
                                        true,
                                    ).show()
                                },
                                modifier = Modifier.weight(1f),
                            ) {
                                Text("开始 %02d:%02d".format(state.settings.dndStartMinutes / 60, state.settings.dndStartMinutes % 60), fontSize = 13.sp)
                            }
                            Button(
                                onClick = {
                                    TimePickerDialog(
                                        context,
                                        { _, hour, minute -> onDndTimes(state.settings.dndStartMinutes, hour * 60 + minute) },
                                        state.settings.dndEndMinutes / 60,
                                        state.settings.dndEndMinutes % 60,
                                        true,
                                    ).show()
                                },
                                modifier = Modifier.weight(1f),
                            ) {
                                Text("结束 %02d:%02d".format(state.settings.dndEndMinutes / 60, state.settings.dndEndMinutes % 60), fontSize = 13.sp)
                            }
                        }
                    }
                }
            }
        }
    }
}

// ==================== PIN Sub-page ====================

@Composable
private fun PinPage(
    state: AppUiState,
    onSetPin: (String, String) -> Unit,
    onBack: () -> Unit,
) {
    var currentPin by remember { mutableStateOf("") }
    var newPin by remember { mutableStateOf("") }

    Column(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        SubPageHeader("监护密码", onBack)
        LazyColumn(
            contentPadding = androidx.compose.foundation.layout.PaddingValues(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            item {
                WeChatGroup {
                    if (state.hasGuardianPin) {
                        OutlinedTextField(
                            value = currentPin,
                            onValueChange = { if (it.length <= 6) currentPin = it.filter(Char::isDigit) },
                            label = { Text("原 6 位密码") },
                            visualTransformation = PasswordVisualTransformation(),
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.NumberPassword),
                            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
                            singleLine = true,
                        )
                        WeChatDivider()
                    }
                    OutlinedTextField(
                        value = newPin,
                        onValueChange = { if (it.length <= 6) newPin = it.filter(Char::isDigit) },
                        label = { Text(if (state.hasGuardianPin) "新 6 位密码" else "设置 6 位密码") },
                        visualTransformation = PasswordVisualTransformation(),
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.NumberPassword),
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
                        singleLine = true,
                    )
                }
            }
            item {
                Button(
                    onClick = {
                        onSetPin(newPin, currentPin)
                        newPin = ""
                        currentPin = ""
                    },
                    enabled = newPin.length == 6,
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp),
                ) {
                    Text(if (state.hasGuardianPin) "修改监护密码" else "设置监护密码")
                }
            }
        }
    }
}

// ==================== Data Sub-page ====================

@Composable
private fun DataPage(
    onClearChat: () -> Unit,
    onBack: () -> Unit,
) {
    var showConfirm by remember { mutableStateOf(false) }

    Column(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        SubPageHeader("本地数据", onBack)
        LazyColumn(
            contentPadding = androidx.compose.foundation.layout.PaddingValues(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            item {
                WeChatGroup {
                    WeChatRow(
                        label = "清空聊天记录",
                        subtitle = "只清除手机中的最近聊天记录",
                        onClick = { showConfirm = true },
                    ) { Chevron() }
                }
            }
        }
    }

    if (showConfirm) {
        AlertDialog(
            onDismissRequest = { showConfirm = false },
            title = { Text("清空聊天记录") },
            text = { Text("确定要清除所有聊天记录吗？此操作不可恢复。") },
            confirmButton = {
                TextButton(onClick = { onClearChat(); showConfirm = false }) {
                    Text("清空", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = { TextButton(onClick = { showConfirm = false }) { Text("取消") } },
        )
    }
}
