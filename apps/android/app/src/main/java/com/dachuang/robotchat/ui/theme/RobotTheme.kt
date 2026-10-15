package com.dachuang.robotchat.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val LightColors = lightColorScheme(
    primary = Color(0xFF07C160),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFD9F9E6),
    onPrimaryContainer = Color(0xFF003919),
    secondary = Color(0xFF4E6355),
    background = Color(0xFFF5F5F5),
    surface = Color.White,
    surfaceVariant = Color(0xFFE9ECEA),
    error = Color(0xFFBA1A1A),
)

private val DarkColors = darkColorScheme(
    primary = Color(0xFF5BDB8A),
    primaryContainer = Color(0xFF005228),
    background = Color(0xFF111412),
    surface = Color(0xFF1A1C1A),
    surfaceVariant = Color(0xFF3F4942),
)

@Composable
fun RobotTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = if (isSystemInDarkTheme()) DarkColors else LightColors,
        content = content,
    )
}
