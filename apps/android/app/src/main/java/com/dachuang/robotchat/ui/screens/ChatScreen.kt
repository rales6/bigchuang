package com.dachuang.robotchat.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.automirrored.filled.VolumeUp
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Keyboard
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalLocale
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.dachuang.robotchat.model.AppUiState
import com.dachuang.robotchat.model.ChatMessage
import com.dachuang.robotchat.model.MessageSender
import com.dachuang.robotchat.model.UserMode
import java.text.SimpleDateFormat
import java.util.Date

@Composable
fun ChatScreen(
    state: AppUiState,
    onSend: (String) -> Unit,
    onSpeak: (String) -> Unit,
    onToggleInput: () -> Unit,
    onStartVoice: () -> Unit,
    onStopVoice: () -> Unit,
    onCancelVoice: () -> Unit,
) {
    var text by remember { mutableStateOf("") }
    var voicePressed by remember { mutableStateOf(false) }
    val listState = rememberLazyListState()
    val senior = state.settings.mode == UserMode.SENIOR
    val currentStartVoice by rememberUpdatedState(onStartVoice)
    val currentStopVoice by rememberUpdatedState(onStopVoice)
    val currentCancelVoice by rememberUpdatedState(onCancelVoice)
    val currentSending by rememberUpdatedState(state.isSending)
    LaunchedEffect(state.messages.size) {
        if (state.messages.isNotEmpty()) listState.animateScrollToItem(state.messages.lastIndex)
    }

    Column(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        ChatHeader(state)
        LazyColumn(
            state = listState,
            modifier = Modifier.weight(1f).fillMaxWidth(),
            contentPadding = androidx.compose.foundation.layout.PaddingValues(
                horizontal = if (senior) 16.dp else 12.dp,
                vertical = 12.dp,
            ),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            items(state.messages, key = { it.id }) { message ->
                MessageBubble(message, senior, onSpeak)
            }
        }
        HorizontalDivider()
        Surface(
            modifier = Modifier.fillMaxWidth().imePadding(),
            tonalElevation = 2.dp,
        ) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                IconButton(onClick = onToggleInput, modifier = Modifier.size(if (senior) 56.dp else 48.dp)) {
                    Icon(
                        if (state.useVoiceInput) Icons.Default.Keyboard else Icons.Default.Mic,
                        contentDescription = if (state.useVoiceInput) "切换文字输入" else "切换语音输入",
                    )
                }
                if (state.useVoiceInput) {
                    Surface(
                        modifier = Modifier
                            .weight(1f)
                            .heightIn(min = if (senior) 58.dp else 48.dp)
                            .semantics {
                                role = Role.Button
                                contentDescription = "按住说话，松开发送，滑开取消"
                            }
                            .pointerInput(Unit) {
                                detectTapGestures(
                                    onPress = {
                                        if (!currentSending) {
                                            voicePressed = true
                                            currentStartVoice()
                                            try {
                                                if (tryAwaitRelease()) currentStopVoice() else currentCancelVoice()
                                            } finally {
                                                voicePressed = false
                                            }
                                        }
                                    },
                                )
                            },
                        shape = RoundedCornerShape(8.dp),
                        color = if (voicePressed || state.isListening) {
                            MaterialTheme.colorScheme.surfaceVariant
                        } else {
                            MaterialTheme.colorScheme.surface
                        },
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp),
                            horizontalArrangement = Arrangement.Center,
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Icon(
                                if (voicePressed || state.isListening) Icons.Default.GraphicEq else Icons.Default.Mic,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.onSurface,
                            )
                            Spacer(Modifier.width(8.dp))
                            Text(
                                when {
                                    currentSending -> "发送中…"
                                    voicePressed || state.isListening -> "松开发送 · 滑开取消"
                                    else -> "按住说话"
                                },
                                color = MaterialTheme.colorScheme.onSurface,
                                fontSize = if (senior) 20.sp else 16.sp,
                                maxLines = 1,
                            )
                        }
                    }
                } else {
                    OutlinedTextField(
                        value = text,
                        onValueChange = { text = it },
                        modifier = Modifier.weight(1f),
                        placeholder = { Text("发送消息给小车") },
                        maxLines = 4,
                        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                        keyboardActions = KeyboardActions(onSend = {
                            if (text.isNotBlank()) {
                                onSend(text)
                                text = ""
                            }
                        }),
                        shape = RoundedCornerShape(20.dp),
                    )
                    Spacer(Modifier.width(6.dp))
                    IconButton(
                        onClick = {
                            onSend(text)
                            text = ""
                        },
                        enabled = text.isNotBlank() && !state.isSending,
                        modifier = Modifier.clip(CircleShape).background(MaterialTheme.colorScheme.primary),
                    ) {
                        if (state.isSending) {
                            CircularProgressIndicator(Modifier.size(20.dp), color = Color.White, strokeWidth = 2.dp)
                        } else {
                            Icon(Icons.AutoMirrored.Filled.Send, "发送", tint = Color.White)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ChatHeader(state: AppUiState) {
    Surface(tonalElevation = 1.dp) {
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                Modifier.size(10.dp).clip(CircleShape).background(
                    if (state.robot.connected) Color(0xFF07C160) else MaterialTheme.colorScheme.error,
                ),
            )
            Spacer(Modifier.width(10.dp))
            Column(Modifier.weight(1f)) {
                Text("小车助手", fontWeight = FontWeight.SemiBold, fontSize = 19.sp)
                Text(
                    when {
                        state.dndActive -> "免打扰中 · 消息仍会保留"
                        state.robot.connected -> "树莓派已连接"
                        else -> "未连接 · 请检查设置"
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            val mode = when (state.settings.mode) {
                UserMode.CHILD -> "少儿"
                UserMode.ADULT -> "成人"
                UserMode.SENIOR -> "老人"
            }
            Text(mode, color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Medium)
        }
    }
}

@Composable
private fun MessageBubble(message: ChatMessage, senior: Boolean, onSpeak: (String) -> Unit) {
    val locale = LocalLocale.current.platformLocale
    if (message.sender == MessageSender.SYSTEM) {
        Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
            Text(
                text = message.text,
                modifier = Modifier.clip(RoundedCornerShape(12.dp)).background(
                    if (message.isError) MaterialTheme.colorScheme.errorContainer
                    else MaterialTheme.colorScheme.surfaceVariant,
                ).padding(horizontal = 12.dp, vertical = 6.dp),
                fontSize = if (senior) 16.sp else 12.sp,
                color = if (message.isError) MaterialTheme.colorScheme.onErrorContainer
                else MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        return
    }
    val isUser = message.sender == MessageSender.USER
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start,
        verticalAlignment = Alignment.Bottom,
    ) {
        if (!isUser) {
            Box(
                Modifier.size(if (senior) 46.dp else 38.dp).clip(CircleShape)
                    .background(MaterialTheme.colorScheme.primaryContainer),
                contentAlignment = Alignment.Center,
            ) {
                Text("车", fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary)
            }
            Spacer(Modifier.width(8.dp))
        }
        Column(horizontalAlignment = if (isUser) Alignment.End else Alignment.Start) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (!isUser) {
                    IconButton(onClick = { onSpeak(message.text) }, modifier = Modifier.size(36.dp)) {
                        Icon(Icons.AutoMirrored.Filled.VolumeUp, "朗读消息", modifier = Modifier.size(20.dp))
                    }
                }
                if (isUser && message.voiceDurationSeconds != null) {
                    Row(
                        modifier = Modifier.widthIn(min = 104.dp, max = 220.dp)
                            .clip(RoundedCornerShape(10.dp))
                            .background(Color(0xFF95EC69))
                            .padding(horizontal = 14.dp, vertical = 12.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Icon(Icons.Default.GraphicEq, contentDescription = null, tint = Color(0xFF171717))
                        Text("语音指令  ${message.voiceDurationSeconds}″", color = Color(0xFF171717),
                            fontSize = if (senior) 19.sp else 15.sp)
                    }
                } else {
                    Text(
                        text = message.text,
                        modifier = Modifier.widthIn(max = if (senior) 520.dp else 440.dp)
                            .clip(RoundedCornerShape(12.dp))
                            .background(
                                if (isUser) Color(0xFF95EC69) else MaterialTheme.colorScheme.surface,
                            )
                            .padding(horizontal = 14.dp, vertical = if (senior) 13.dp else 10.dp),
                        color = Color(0xFF171717),
                        fontSize = if (senior) 21.sp else 16.sp,
                        lineHeight = if (senior) 30.sp else 23.sp,
                    )
                }
            }
            Text(
                SimpleDateFormat("HH:mm", locale).format(Date(message.timestampMillis)),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(horizontal = 4.dp, vertical = 2.dp),
            )
        }
    }
}
