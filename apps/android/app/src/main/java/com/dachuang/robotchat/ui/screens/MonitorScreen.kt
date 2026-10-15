package com.dachuang.robotchat.ui.screens

import android.annotation.SuppressLint
import android.graphics.Color as AndroidColor
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.core.net.toUri
import androidx.core.text.htmlEncode
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.dachuang.robotchat.model.RobotSnapshot
import com.dachuang.robotchat.ui.components.WeChatCard
import com.dachuang.robotchat.ui.components.WeChatDivider
import com.dachuang.robotchat.ui.components.WeChatRow

@Composable
fun MonitorScreen(
    robot: RobotSnapshot,
    videoUrl: String,
    onEmergencyStop: (Boolean) -> Unit,
) {
    BoxWithConstraints(Modifier.fillMaxSize().padding(12.dp)) {
        val landscape = maxWidth >= 760.dp
        if (landscape) {
            Row(Modifier.fillMaxSize(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                VideoPanel(videoUrl, robot.cameraConnected, Modifier.weight(1.7f))
                MonitorDetails(robot, onEmergencyStop, Modifier.weight(1f))
            }
        } else {
            Column(
                modifier = if (maxHeight < 570.dp) Modifier.fillMaxSize().verticalScroll(rememberScrollState())
                    else Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                VideoPanel(videoUrl, robot.cameraConnected, Modifier.fillMaxWidth())
                MonitorDetails(robot, onEmergencyStop, Modifier.fillMaxWidth())
            }
        }
    }
}

@Composable
private fun VideoPanel(videoUrl: String, cameraConnected: Boolean, modifier: Modifier = Modifier) {
    var reloadKey by remember { mutableIntStateOf(0) }
    var webError by remember { mutableStateOf("") }
    WeChatCard(modifier) {
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                Modifier.size(9.dp).clip(CircleShape)
                    .background(if (cameraConnected) Color(0xFF07C160) else MaterialTheme.colorScheme.error),
            )
            Text(
                if (cameraConnected) "实时画面" else "等待摄像头",
                Modifier.padding(start = 8.dp).weight(1f),
            )
            IconButton(onClick = {
                webError = ""
                reloadKey += 1
            }) { Icon(Icons.Default.Refresh, "重新加载视频") }
        }
        Box(
            Modifier.fillMaxWidth().aspectRatio(16f / 9f).background(Color.Black),
            contentAlignment = Alignment.Center,
        ) {
            MjpegWebView(videoUrl, reloadKey) { webError = it }
            if (webError.isNotEmpty()) {
                Text(
                    webError,
                    color = Color.White,
                    modifier = Modifier.background(Color(0xAA000000)).padding(12.dp),
                )
            }
        }
    }
}

@SuppressLint("SetJavaScriptEnabled")
@Composable
private fun MjpegWebView(videoUrl: String, reloadKey: Int, onError: (String) -> Unit) {
    val safeUrl = videoUrl.htmlEncode()
    val baseUrl = runCatching {
        val uri = videoUrl.toUri()
        "${uri.scheme}://${uri.authority}/"
    }.getOrDefault("about:blank")
    val html = """
        <!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1" />
        <style>html,body{margin:0;width:100%;height:100%;background:#000;overflow:hidden}img{width:100%;height:100%;object-fit:contain}</style>
        </head><body><img src="$safeUrl" alt="robot camera" /></body></html>
    """.trimIndent()
    AndroidView(
        modifier = Modifier.fillMaxSize(),
        factory = { context ->
            WebView(context).apply {
                setBackgroundColor(AndroidColor.BLACK)
                settings.javaScriptEnabled = false
                settings.loadsImagesAutomatically = true
                settings.blockNetworkImage = false
                settings.builtInZoomControls = false
                webViewClient = object : WebViewClient() {
                    override fun onReceivedError(
                        view: WebView?,
                        request: WebResourceRequest?,
                        error: WebResourceError?,
                    ) {
                        if (request?.isForMainFrame == true) onError("视频加载失败：${error?.description ?: "网络错误"}")
                    }
                }
            }
        },
        update = { webView ->
            val loadTag = "$videoUrl#$reloadKey"
            if (webView.tag != loadTag) {
                webView.tag = loadTag
                webView.loadDataWithBaseURL(baseUrl, html, "text/html", "UTF-8", null)
            }
        },
        onRelease = { it.destroy() },
    )
}

@Composable
private fun MonitorDetails(
    robot: RobotSnapshot,
    onEmergencyStop: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(modifier, verticalArrangement = Arrangement.spacedBy(8.dp)) {
        WeChatCard {
            Column(Modifier.padding(horizontal = 14.dp, vertical = 10.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                Text(robot.task.phaseLabel, fontSize = 16.sp, fontWeight = FontWeight.Medium, maxLines = 1,
                    overflow = TextOverflow.Ellipsis)
                LinearProgressIndicator(
                    progress = { robot.task.progress },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }

        WeChatCard {
            Column(Modifier.padding(horizontal = 14.dp, vertical = 10.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    CompactStatus("连接", if (robot.connected) "已连接" else "未连接", Modifier.weight(1f))
                    CompactStatus("摄像头", if (robot.cameraConnected) "在线" else "离线", Modifier.weight(1f))
                }
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    CompactStatus("雷达", robot.lidarStatus, Modifier.weight(1f))
                    CompactStatus("避障", robot.obstacleStatus, Modifier.weight(1f))
                }
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    CompactStatus("视觉", robot.groundingStatus, Modifier.weight(1f))
                    CompactStatus("动作", robot.robotCommand, Modifier.weight(1f))
                }
            }
        }

        Button(
            onClick = { onEmergencyStop(!robot.emergencyStop) },
            modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = if (robot.emergencyStop) Color(0xFF555555) else MaterialTheme.colorScheme.error,
            ),
        ) {
            Icon(Icons.Default.Warning, null)
            Text(if (robot.emergencyStop) "解除急停" else "紧急停止", Modifier.padding(start = 8.dp))
        }
    }
}

@Composable
private fun CompactStatus(label: String, value: String, modifier: Modifier = Modifier) {
    Column(modifier) {
        Text(label, style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(
            text = value.ifBlank { "—" },
            style = MaterialTheme.typography.bodyMedium,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
    }
}
