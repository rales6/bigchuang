package com.dachuang.robotchat.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.ime
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.filled.Checklist
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Badge
import androidx.compose.material3.BadgedBox
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalDensity
import com.dachuang.robotchat.RobotViewModel
import com.dachuang.robotchat.model.AppTab
import com.dachuang.robotchat.model.UserMode
import com.dachuang.robotchat.ui.screens.ChatScreen
import com.dachuang.robotchat.ui.screens.MonitorScreen
import com.dachuang.robotchat.ui.screens.SettingsScreen
import com.dachuang.robotchat.ui.screens.TasksScreen
import com.dachuang.robotchat.ui.theme.RobotTheme

private data class NavItem(val tab: AppTab, val label: String, val icon: ImageVector)

private val allItems = listOf(
    NavItem(AppTab.CHAT, "对话", Icons.AutoMirrored.Filled.Chat),
    NavItem(AppTab.MONITOR, "监控", Icons.Default.Videocam),
    NavItem(AppTab.TASKS, "任务", Icons.Default.Checklist),
    NavItem(AppTab.SETTINGS, "设置", Icons.Default.Settings),
)

@Composable
fun RobotCompanionApp(
    viewModel: RobotViewModel,
    onStartVoice: () -> Unit,
    onStopVoice: () -> Unit,
    onCancelVoice: () -> Unit,
    onRequestNotificationPermission: () -> Unit,
    onTestMicrophone: () -> Unit,
) {
    val state by viewModel.uiState.collectAsState()
    val imeVisible = WindowInsets.ime.getBottom(LocalDensity.current) > 0
    RobotTheme {
        // First-launch Vosk model download prompt
        if (state.showVoskPrompt) {
            AlertDialog(
                onDismissRequest = { viewModel.dismissVoskPrompt(download = false) },
                title = { Text("离线语音识别") },
                text = {
                    Text(
                        "是否下载离线语音模型（约 42 MB）？\n\n" +
                            "下载后可在无网络、无语音服务的手机上使用语音输入，识别普通话。\n" +
                            "您也可以在设置中稍后下载。",
                    )
                },
                confirmButton = {
                    TextButton(onClick = { viewModel.dismissVoskPrompt(download = true) }) {
                        Text("立即下载")
                    }
                },
                dismissButton = {
                    TextButton(onClick = { viewModel.dismissVoskPrompt(download = false) }) {
                        Text("稍后再说")
                    }
                },
            )
        }

        Scaffold(
            containerColor = MaterialTheme.colorScheme.background,
            bottomBar = {
                if (!imeVisible) {
                    NavigationBar {
                        allItems.forEach { item ->
                            NavigationBarItem(
                                selected = state.selectedTab == item.tab,
                                onClick = { viewModel.selectTab(item.tab) },
                                icon = { NavIcon(item, state.robot.task.active) },
                                label = { Text(item.label) },
                            )
                        }
                    }
                }
            },
        ) { padding ->
            Box(Modifier.fillMaxSize().padding(padding)) {
                AppPage(
                    viewModel = viewModel,
                    onStartVoice = onStartVoice,
                    onStopVoice = onStopVoice,
                    onCancelVoice = onCancelVoice,
                    onRequestNotificationPermission = onRequestNotificationPermission,
                    onTestMicrophone = onTestMicrophone,
                )
            }
        }
    }
}

@Composable
private fun AppPage(
    viewModel: RobotViewModel,
    onStartVoice: () -> Unit,
    onStopVoice: () -> Unit,
    onCancelVoice: () -> Unit,
    onRequestNotificationPermission: () -> Unit,
    onTestMicrophone: () -> Unit,
) {
    val state by viewModel.uiState.collectAsState()
    when (state.selectedTab) {
        AppTab.CHAT -> ChatScreen(
            state = state,
            onSend = { viewModel.sendMessage(it) },
            onSpeak = viewModel::speak,
            onToggleInput = { viewModel.setVoiceInput(!state.useVoiceInput) },
            onStartVoice = onStartVoice,
            onStopVoice = onStopVoice,
            onCancelVoice = onCancelVoice,
        )
        AppTab.MONITOR -> if (state.settings.mode == UserMode.ADULT) {
            MonitorScreen(
                robot = state.robot,
                videoUrl = state.settings.resolvedVideoUrl,
                onEmergencyStop = viewModel::emergencyStop,
            )
        } else {
            Box(Modifier.fillMaxSize(), contentAlignment = androidx.compose.ui.Alignment.Center) {
                Text("监控画面仅在成人模式开放")
            }
        }
        AppTab.TASKS -> TasksScreen(
            robot = state.robot,
            mode = state.settings.mode,
            baseUrl = state.settings.baseUrl,
            onSpeak = viewModel::speak,
            onReset = viewModel::resetTask,
            onEmergencyStop = viewModel::emergencyStop,
        )
        AppTab.SETTINGS -> SettingsScreen(
            state = state,
            onConnectionSave = viewModel::updateConnection,
            onTestConnection = viewModel::testConnection,
            onModeChange = viewModel::switchMode,
            onVerifyGuardianPin = viewModel::verifyGuardianPin,
            onSetPin = viewModel::setGuardianPin,
            onAutoSpeak = viewModel::setAutoSpeak,
            onDialect = viewModel::setDialect,
            onNotifications = {
                viewModel.setNotifications(it)
                if (it) onRequestNotificationPermission()
            },
            onDndAlways = viewModel::setDndAlways,
            onDndScheduled = viewModel::setDndScheduled,
            onDndTimes = viewModel::setDndTimes,
            onClearChat = viewModel::clearChat,
            onSpeechEngine = viewModel::setSpeechEngine,
            onDownloadVosk = viewModel::downloadVoskModel,
            onCancelVosk = viewModel::cancelVoskDownload,
            onDeleteVosk = viewModel::deleteVoskModel,
            onTestMicrophone = onTestMicrophone,
            onCarConfigSave = viewModel::saveCarConfiguration,
            onStartMapping = { viewModel.startCarService("mapping", it) },
            onStartAnnotation = { viewModel.startCarService("annotation", it) },
            onRecoverBluetooth = viewModel::recoverBluetooth,
            onStopCarService = viewModel::stopCarService,
        )
    }
}

@Composable
private fun NavIcon(item: NavItem, taskActive: Boolean) {
    BadgedBox(badge = {
        if (item.tab == AppTab.TASKS && taskActive) Badge()
    }) {
        Icon(item.icon, contentDescription = item.label)
    }
}
