package com.dachuang.robotchat.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowLeft
import androidx.compose.material.icons.filled.Map
import androidx.compose.material.icons.filled.StopCircle
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.VolumeUp
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.dachuang.robotchat.model.AnnotationConfiguration
import com.dachuang.robotchat.model.CarConfiguration
import com.dachuang.robotchat.model.BluetoothConfiguration
import com.dachuang.robotchat.model.MappingConfiguration
import com.dachuang.robotchat.model.RobotServiceState
import com.dachuang.robotchat.ui.components.WeChatCard

private enum class CarConfigPage { MAPPING, ANNOTATION, BLUETOOTH, AUDIO }

@Composable
fun CarConfigScreen(
    configuration: CarConfiguration,
    service: RobotServiceState,
    message: String,
    onSave: (CarConfiguration) -> Unit,
    onStartMapping: (CarConfiguration) -> Unit,
    onStartAnnotation: (CarConfiguration) -> Unit,
    onRecoverBluetooth: () -> Unit,
    onStop: () -> Unit,
    onBack: () -> Unit,
) {
    var draft by remember(configuration) { mutableStateOf(configuration) }
    var page by remember { mutableStateOf(CarConfigPage.MAPPING) }
    var advanced by remember { mutableStateOf(false) }
    Column(Modifier.fillMaxSize()) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
        IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.KeyboardArrowLeft, "返回设置") }
        Text("小车配置", style = MaterialTheme.typography.titleMedium)
    }
    LazyColumn(
        modifier = Modifier.weight(1f),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(14.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        item {
            Text(
                "参数先保存到树莓派，再启动受控服务。",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        item {
            Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(page == CarConfigPage.MAPPING, { page = CarConfigPage.MAPPING }, { Text("建图设置") })
                FilterChip(page == CarConfigPage.ANNOTATION, { page = CarConfigPage.ANNOTATION }, { Text("标注设置") })
                FilterChip(page == CarConfigPage.BLUETOOTH, { page = CarConfigPage.BLUETOOTH }, { Text("蓝牙") })
                FilterChip(page == CarConfigPage.AUDIO, { page = CarConfigPage.AUDIO }, { Text("语言音量") })
            }
        }
        if (page == CarConfigPage.MAPPING || page == CarConfigPage.ANNOTATION) item {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(!advanced, { advanced = false }, { Text("基础参数") })
                FilterChip(advanced, { advanced = true }, { Text("运动与安全") })
            }
        }
        item {
            WeChatCard {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("服务状态", fontWeight = FontWeight.SemiBold)
                        Text(
                            if (service.running) "运行中：${service.kind}" else "待机",
                            color = if (service.running) Color(0xFF07C160) else MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    if (service.logPath.isNotBlank()) Text("日志：${service.logPath}", style = MaterialTheme.typography.bodySmall)
                    if (message.isNotBlank()) Text(message, color = MaterialTheme.colorScheme.primary)
                    if (service.running) {
                        Button(
                            onClick = onStop,
                            colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error),
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Icon(Icons.Default.StopCircle, null)
                            Text("安全停止当前服务", Modifier.padding(start = 8.dp))
                        }
                    }
                }
            }
        }
        when (page) {
            CarConfigPage.MAPPING -> mappingItems(draft.mapping, advanced) { draft = draft.copy(mapping = it) }
            CarConfigPage.ANNOTATION -> annotationItems(draft.annotation, advanced) { draft = draft.copy(annotation = it) }
            CarConfigPage.BLUETOOTH -> bluetoothItems(draft.bluetooth) { draft = draft.copy(bluetooth = it) }
            CarConfigPage.AUDIO -> item {
                WeChatCard {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        Row {
                            Icon(Icons.Default.VolumeUp, null)
                            Text("语言输出音量 ${(draft.audioVolume * 100).toInt()}%", Modifier.padding(start = 8.dp))
                        }
                        Slider(
                            value = draft.audioVolume,
                            onValueChange = { draft = draft.copy(audioVolume = it.coerceIn(0f, 1f)) },
                        )
                        Text("该音量已接入 Android 文字转语音，后续机器人语言输出可复用。", style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedButton(onClick = { onSave(draft) }, modifier = Modifier.weight(1f)) {
                    Icon(Icons.Default.Tune, null)
                    Text("保存参数", Modifier.padding(start = 6.dp))
                }
                Button(
                    onClick = {
                        when (page) {
                            CarConfigPage.ANNOTATION -> onStartAnnotation(draft)
                            CarConfigPage.BLUETOOTH -> onRecoverBluetooth()
                            else -> onStartMapping(draft)
                        }
                    },
                    enabled = page == CarConfigPage.BLUETOOTH || (!service.running && page != CarConfigPage.AUDIO),
                    modifier = Modifier.weight(1f),
                ) {
                    Icon(Icons.Default.Map, null)
                    if (page == CarConfigPage.BLUETOOTH) {
                        Text("重置并重连", Modifier.padding(start = 6.dp))
                    } else
                    Text(if (page == CarConfigPage.ANNOTATION) "开始标注" else "开始建图", Modifier.padding(start = 6.dp))
                }
            }
        }
    }
    }
}

private fun androidx.compose.foundation.lazy.LazyListScope.mappingItems(
    value: MappingConfiguration,
    advanced: Boolean,
    update: (MappingConfiguration) -> Unit,
) {
    if (!advanced) {
        item { NumberField("可用视场角（30–180°）", value.usableFov) { update(value.copy(usableFov = it)) } }
        item { NumberField("地图最大测距（0.5–8m）", value.maxDistance) { update(value.copy(maxDistance = it)) } }
        item { NumberField("停车建图距离（m）", value.stationaryDistance) { update(value.copy(stationaryDistance = it)) } }
        item { NumberField("停车建图角度（°）", value.stationaryAngle) { update(value.copy(stationaryAngle = it)) } }
        item { IntField("融合扫描圈数", value.fusionScans) { update(value.copy(fusionScans = it)) } }
        item { IntField("最小支持圈数", value.minSupport) { update(value.copy(minSupport = it)) } }
        item { TextField("地图输出名", value.output) { update(value.copy(output = it)) } }
    } else {
        item { IntField("巡航速度（mm/s）", value.speed) { update(value.copy(speed = it)) } }
        item { IntField("最大直行速度（mm/s）", value.maxDriveSpeed) { update(value.copy(maxDriveSpeed = it)) } }
        item { IntField("转向速度（mrad/s）", value.turnSpeed) { update(value.copy(turnSpeed = it)) } }
        item { IntField("最大转向速度（mrad/s）", value.maxTurnSpeed) { update(value.copy(maxTurnSpeed = it)) } }
        item { NumberField("停车距离（m）", value.stopDistance) { update(value.copy(stopDistance = it)) } }
        item { NumberField("车体净空（m）", value.clearance) { update(value.copy(clearance = it)) } }
        item { NumberField("最长运行时间（min）", value.runtimeMinutes) { update(value.copy(runtimeMinutes = it)) } }
    }
}

private fun androidx.compose.foundation.lazy.LazyListScope.annotationItems(
    value: AnnotationConfiguration,
    advanced: Boolean,
    update: (AnnotationConfiguration) -> Unit,
) {
    if (!advanced) {
        item { IntField("每个停靠点拍摄组数（双摄每组两张）", value.photosPerTarget) { update(value.copy(photosPerTarget = it)) } }
        item { TextField("地图文件", value.mapPath) { update(value.copy(mapPath = it)) } }
        item { TextField("语义标注文件", value.semanticMapPath) { update(value.copy(semanticMapPath = it)) } }
        item { NumberField("拍摄距离（m）", value.standoff) { update(value.copy(standoff = it)) } }
        item { IntField("最多拍摄点", value.maxTargets) { update(value.copy(maxTargets = it)) } }
        item { NumberField("停车稳定时间（s）", value.settleSeconds) { update(value.copy(settleSeconds = it)) } }
    } else {
        item { IntField("巡航速度（mm/s）", value.speed) { update(value.copy(speed = it)) } }
        item { IntField("最大直行速度（mm/s）", value.maxDriveSpeed) { update(value.copy(maxDriveSpeed = it)) } }
        item { IntField("转向速度（mrad/s）", value.turnSpeed) { update(value.copy(turnSpeed = it)) } }
        item { NumberField("停车距离（m）", value.stopDistance) { update(value.copy(stopDistance = it)) } }
        item { NumberField("车体净空（m）", value.clearance) { update(value.copy(clearance = it)) } }
        item { NumberField("最长运行时间（min）", value.runtimeMinutes) { update(value.copy(runtimeMinutes = it)) } }
    }
}

private fun androidx.compose.foundation.lazy.LazyListScope.bluetoothItems(
    value: BluetoothConfiguration,
    update: (BluetoothConfiguration) -> Unit,
) {
    item { SectionTitle("BLE 设备与自动恢复") }
    item { TextField("设备名称", value.deviceName) { update(value.copy(deviceName = it)) } }
    item { TextField("固定 MAC 地址", value.deviceAddress) { update(value.copy(deviceAddress = it.uppercase())) } }
    item {
        WeChatCard {
            Row(
                Modifier.fillMaxWidth().padding(16.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Column(Modifier.weight(1f)) {
                    Text("连接失败时自动恢复", fontWeight = FontWeight.SemiBold)
                    Text("只停止运动任务，不会关闭手机桥接服务", style = MaterialTheme.typography.bodySmall)
                }
                Switch(value.autoRecover, { update(value.copy(autoRecover = it)) })
            }
        }
    }
    item { IntField("触发恢复的异常次数", value.failureThreshold) { update(value.copy(failureThreshold = it)) } }
    item { IntField("最多自动恢复次数", value.maxRecoveryAttempts) { update(value.copy(maxRecoveryAttempts = it)) } }
    item { NumberField("启动监控时间（秒）", value.startupGraceSeconds) { update(value.copy(startupGraceSeconds = it)) } }
    item { NumberField("恢复后等待时间（秒）", value.recoveryCooldownSeconds) { update(value.copy(recoveryCooldownSeconds = it)) } }
}

@Composable
private fun SectionTitle(text: String) {
    Text(text, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
}

@Composable
private fun NumberField(label: String, value: Float, onValue: (Float) -> Unit) {
    var text by remember(value) { mutableStateOf(value.toString()) }
    OutlinedTextField(
        value = text,
        onValueChange = { input -> text = input; input.toFloatOrNull()?.let(onValue) },
        label = { Text(label) },
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
        singleLine = true,
        modifier = Modifier.fillMaxWidth(),
    )
}

@Composable
private fun IntField(label: String, value: Int, onValue: (Int) -> Unit) {
    var text by remember(value) { mutableStateOf(value.toString()) }
    OutlinedTextField(
        value = text,
        onValueChange = { input -> text = input; input.toIntOrNull()?.let(onValue) },
        label = { Text(label) },
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
        singleLine = true,
        modifier = Modifier.fillMaxWidth(),
    )
}

@Composable
private fun TextField(label: String, value: String, onValue: (String) -> Unit) {
    OutlinedTextField(
        value = value,
        onValueChange = onValue,
        label = { Text(label) },
        singleLine = true,
        modifier = Modifier.fillMaxWidth(),
    )
}
