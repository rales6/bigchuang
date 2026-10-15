package com.dachuang.robotchat.ui.screens

import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.VolumeUp
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Pending
import androidx.compose.material.icons.filled.RadioButtonChecked
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.draw.clip
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.dachuang.robotchat.model.RobotSnapshot
import com.dachuang.robotchat.model.TaskPlanStep
import com.dachuang.robotchat.model.UserMode
import com.dachuang.robotchat.ui.components.WeChatCard
import com.dachuang.robotchat.ui.components.WeChatDivider
import com.dachuang.robotchat.ui.components.WeChatRow
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.net.HttpURLConnection
import java.net.URL

private enum class TaskPanel { STATUS, MAP, STEPS }

@Composable
fun TasksScreen(
    robot: RobotSnapshot,
    mode: UserMode,
    baseUrl: String,
    onSpeak: (String) -> Unit,
    onReset: () -> Unit,
    onEmergencyStop: (Boolean) -> Unit,
) {
    val senior = mode == UserMode.SENIOR
    var panel by remember { mutableStateOf(TaskPanel.STATUS) }
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(if (senior) 16.dp else 12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        item {
            Text(
                "任务",
                style = if (senior) MaterialTheme.typography.headlineMedium
                    else MaterialTheme.typography.headlineSmall,
            )
        }

        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(panel == TaskPanel.STATUS, { panel = TaskPanel.STATUS }, { Text("概览") })
                FilterChip(panel == TaskPanel.MAP, { panel = TaskPanel.MAP }, { Text("地图") })
                FilterChip(panel == TaskPanel.STEPS, { panel = TaskPanel.STEPS }, { Text("步骤") })
            }
        }

        if (panel == TaskPanel.STEPS) item {
            WeChatCard {
                Column(Modifier.padding(if (senior) 20.dp else 16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text("任务规划过程", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
                    if (robot.taskPlan.instruction.isNotBlank()) {
                        Text("用户指令：${robot.taskPlan.instruction}")
                    }
                    if (robot.taskPlan.steps.isEmpty()) {
                        Text("发送任务后，这里会显示规划步骤和执行状态。", color = MaterialTheme.colorScheme.onSurfaceVariant)
                    } else {
                        robot.taskPlan.steps.forEachIndexed { index, step ->
                            PlanStepRow(index + 1, step, senior)
                        }
                    }
                    if (robot.taskPlan.summary.isNotBlank()) {
                        Text("当前判断：${robot.taskPlan.summary}", color = MaterialTheme.colorScheme.primary)
                    }
                }
            }
        }

        if (panel == TaskPanel.MAP) item {
            WeChatCard {
                Column(Modifier.padding(if (senior) 20.dp else 16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text("任务点地图", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
                    Text(
                        "紫色框是语义目标，蓝色点是计划观察点，橙色是小车，红色叉号是发现目标的位置。",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    TaskMapImage(
                        url = resolveUrl(baseUrl, robot.taskPlan.mapUrl),
                        revision = robot.taskPlan.updatedAt,
                    )
                }
            }
        }

        // Task progress card
        if (panel == TaskPanel.STATUS) item {
            WeChatCard {
                Column(Modifier.padding(if (senior) 20.dp else 16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            if (robot.task.completed) Icons.Default.CheckCircle else Icons.Default.Pending,
                            null,
                            tint = if (robot.task.completed) Color(0xFF07C160) else MaterialTheme.colorScheme.primary,
                            modifier = Modifier.padding(end = 10.dp),
                        )
                        Column(Modifier.weight(1f)) {
                            Text(
                                robot.task.phaseLabel,
                                fontSize = if (senior) 25.sp else 20.sp,
                                fontWeight = FontWeight.SemiBold,
                            )
                            if (robot.task.currentTarget.isNotBlank()) {
                                Text(
                                    "当前目标：${robot.task.currentTarget}",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }
                        }
                    }
                    LinearProgressIndicator(
                        progress = { robot.task.progress },
                        modifier = Modifier.fillMaxWidth().height(if (senior) 12.dp else 8.dp),
                    )
                    Text(
                        if (robot.task.totalTargets > 0) {
                            "完成 ${robot.task.completedTargets} / ${robot.task.totalTargets}"
                        } else if (robot.task.active) {
                            "任务正在执行"
                        } else {
                            "发送一条指令后，进度会显示在这里"
                        },
                        fontSize = if (senior) 20.sp else 15.sp,
                    )
                }
            }
        }

        // Run details card
        if (panel == TaskPanel.STATUS) item {
            WeChatCard {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("连接 ${if (robot.connected) "在线" else "离线"}  ·  摄像头 ${if (robot.cameraConnected) "在线" else "离线"}")
                    Text("雷达 ${robot.lidarStatus}  ·  避障 ${robot.obstacleStatus}", maxLines = 2)
                    Text("当前动作：${robot.robotCommand}", maxLines = 2)
                    if (robot.commandReason.isNotBlank()) {
                        Text(robot.commandReason, style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant, maxLines = 2)
                    }
                }
            }
        }

        // Actions
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = { onSpeak(robot.task.phaseLabel) }, modifier = Modifier.weight(1f)) {
                    Icon(Icons.AutoMirrored.Filled.VolumeUp, null)
                    Text("朗读", Modifier.padding(start = 4.dp))
                }
                if (mode == UserMode.ADULT) {
                    OutlinedButton(onClick = onReset, modifier = Modifier.weight(1f)) {
                        Icon(Icons.Default.Refresh, null)
                        Text("重置", Modifier.padding(start = 4.dp))
                    }
                }
            }
        }

        // Emergency stop button
        item {
            Button(
                onClick = { onEmergencyStop(!robot.emergencyStop) },
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (robot.emergencyStop) Color(0xFF555555) else MaterialTheme.colorScheme.error,
                ),
                modifier = Modifier.fillMaxWidth().height(if (senior) 64.dp else 52.dp),
            ) {
                Icon(Icons.Default.Warning, null)
                Text(
                    if (robot.emergencyStop) "解除急停" else "紧急停止",
                    Modifier.padding(start = 8.dp),
                    fontSize = if (senior) 21.sp else 16.sp,
                )
            }
        }

    }
}

@Composable
private fun PlanStepRow(index: Int, step: TaskPlanStep, senior: Boolean) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.Top) {
        Icon(
            imageVector = when (step.status) {
                "completed" -> Icons.Default.CheckCircle
                "active" -> Icons.Default.RadioButtonChecked
                else -> Icons.Default.Pending
            },
            contentDescription = null,
            tint = when (step.status) {
                "completed" -> Color(0xFF07C160)
                "active" -> MaterialTheme.colorScheme.primary
                else -> MaterialTheme.colorScheme.outline
            },
            modifier = Modifier.padding(top = 2.dp, end = 10.dp),
        )
        Column(Modifier.weight(1f)) {
            Text("$index. ${step.title}", fontSize = if (senior) 20.sp else 16.sp, fontWeight = FontWeight.Medium)
            if (step.detail.isNotBlank()) {
                Text(step.detail, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

@Composable
private fun TaskMapImage(url: String, revision: Long) {
    val bitmap by produceState<android.graphics.Bitmap?>(null, url, revision) {
        value = withContext(Dispatchers.IO) {
            runCatching {
                val separator = if ('?' in url) '&' else '?'
                val connection = URL("$url${separator}v=$revision").openConnection() as HttpURLConnection
                connection.connectTimeout = 4_000
                connection.readTimeout = 8_000
                connection.useCaches = false
                try {
                    if (connection.responseCode !in 200..299) error("HTTP ${connection.responseCode}")
                    connection.inputStream.use(BitmapFactory::decodeStream)
                } finally {
                    connection.disconnect()
                }
            }.getOrNull()
        }
    }
    if (bitmap == null) {
        Column(
            Modifier.fillMaxWidth().aspectRatio(1.35f).clip(RoundedCornerShape(10.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text("任务地图尚未生成", color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    } else {
        Image(
            bitmap = bitmap!!.asImageBitmap(),
            contentDescription = "任务点地图",
            contentScale = ContentScale.Fit,
            modifier = Modifier.fillMaxWidth().aspectRatio(1.35f)
                .clip(RoundedCornerShape(10.dp))
                .background(Color(0xFFD0D0D0)),
        )
    }
}

private fun resolveUrl(baseUrl: String, path: String): String {
    if (path.startsWith("http://") || path.startsWith("https://")) return path
    return baseUrl.trimEnd('/') + "/" + path.trimStart('/')
}

@Composable
private fun DetailRow(label: String, value: String, senior: Boolean) {
    WeChatRow(label = label) {
        Text(
            text = value.ifBlank { "—" },
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            fontSize = if (senior) 19.sp else 14.sp,
            maxLines = 2,
        )
    }
}
