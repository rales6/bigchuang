package com.dachuang.robotchat.ui.components

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/**
 * A zero-elevation white card with 12dp rounded corners — matches WeChat card style.
 */
@Composable
fun WeChatCard(
    modifier: Modifier = Modifier,
    content: @Composable ColumnScope.() -> Unit,
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
        content = content,
    )
}

/**
 * A titled group wrapping a WeChatCard. Title is shown above the card in gray caption style.
 */
@Composable
fun WeChatGroup(
    title: String? = null,
    modifier: Modifier = Modifier,
    content: @Composable ColumnScope.() -> Unit,
) {
    Column(modifier = modifier.fillMaxWidth()) {
        if (title != null) {
            Text(
                text = title,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(start = 16.dp, bottom = 6.dp),
            )
        }
        WeChatCard(content = content)
    }
}

/**
 * Thin WeChat-style divider (#EDEDED, 0.5dp) with 16dp horizontal padding inside a card.
 */
@Composable
fun WeChatDivider() {
    HorizontalDivider(
        thickness = 0.5.dp,
        color = Color(0xFFEDEDED),
        modifier = Modifier.padding(horizontal = 16.dp),
    )
}

/**
 * A basic WeChat-style row: label on the left, optional subtitle below, trailing content on the right.
 * 52dp min height, 16dp horizontal padding. Optionally clickable.
 */
@Composable
fun WeChatRow(
    label: String,
    subtitle: String? = null,
    onClick: (() -> Unit)? = null,
    modifier: Modifier = Modifier,
    trailing: @Composable RowScope.() -> Unit = {},
) {
    val baseModifier = modifier
        .fillMaxWidth()
        .height(52.dp)
        .padding(horizontal = 16.dp)
    val rowModifier = if (onClick != null) baseModifier.clickable(onClick = onClick) else baseModifier
    Row(
        modifier = rowModifier,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = label,
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurface,
            )
            if (subtitle != null) {
                Text(
                    text = subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
        trailing()
    }
}

/**
 * A row with a Switch toggle as trailing content.
 */
@Composable
fun SwitchRow(
    label: String,
    subtitle: String? = null,
    checked: Boolean,
    onChecked: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
) {
    WeChatRow(
        label = label,
        subtitle = subtitle,
        modifier = modifier,
    ) {
        Switch(checked = checked, onCheckedChange = onChecked)
    }
}

/**
 * A row showing a value on the right with a chevron — taps open a dialog/picker.
 */
@Composable
fun ValueRow(
    label: String,
    value: String? = null,
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
) {
    WeChatRow(
        label = label,
        onClick = onClick,
        modifier = modifier,
    ) {
        if (!value.isNullOrBlank()) {
            Text(
                text = value,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1,
            )
            Spacer(modifier = Modifier.width(6.dp))
        }
        Chevron()
    }
}

/**
 * A selectable check row — radio-button style with a checkmark when selected.
 */
@Composable
fun CheckRow(
    label: String,
    subtitle: String? = null,
    selected: Boolean,
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
) {
    WeChatRow(
        label = label,
        subtitle = subtitle,
        onClick = onClick,
        modifier = modifier,
    ) {
        if (selected) {
            Icon(
                imageVector = Icons.Default.Check,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(20.dp),
            )
        }
    }
}

/**
 * A highlighted action row with customizable label color (e.g., red for destructive actions).
 */
@Composable
fun ActionRow(
    label: String,
    labelColor: Color = MaterialTheme.colorScheme.primary,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    WeChatRow(
        label = label,
        onClick = onClick,
        modifier = modifier,
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyLarge,
            color = labelColor,
        )
        Spacer(modifier = Modifier.width(4.dp))
        Chevron()
    }
}

/**
 * Standard WeChat right-chevron icon.
 */
@Composable
fun Chevron() {
    Icon(
        imageVector = Icons.AutoMirrored.Filled.KeyboardArrowRight,
        contentDescription = null,
        tint = Color(0xFFB0B0B0),
        modifier = Modifier.size(22.dp),
    )
}
