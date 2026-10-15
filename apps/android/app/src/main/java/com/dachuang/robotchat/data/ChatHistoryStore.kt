package com.dachuang.robotchat.data

import android.content.Context
import androidx.core.content.edit
import com.dachuang.robotchat.model.ChatMessage
import com.dachuang.robotchat.model.MessageSender
import org.json.JSONArray
import org.json.JSONObject

class ChatHistoryStore(context: Context) {
    private val prefs = context.getSharedPreferences("robot_companion_chat", Context.MODE_PRIVATE)

    fun load(): List<ChatMessage> = runCatching {
        val array = JSONArray(prefs.getString("messages", "[]"))
        buildList {
            for (index in 0 until array.length()) {
                val item = array.getJSONObject(index)
                add(
                    ChatMessage(
                        id = item.optString("id"),
                        text = item.optString("text"),
                        sender = MessageSender.valueOf(item.optString("sender", "SYSTEM")),
                        timestampMillis = item.optLong("time"),
                        isError = item.optBoolean("error"),
                        voiceDurationSeconds = item.optInt("voiceDurationSeconds", 0).takeIf { it > 0 },
                    ),
                )
            }
        }
    }.getOrDefault(emptyList())

    fun save(messages: List<ChatMessage>) {
        val array = JSONArray()
        messages.takeLast(100).forEach { message ->
            array.put(
                JSONObject()
                    .put("id", message.id)
                    .put("text", message.text)
                    .put("sender", message.sender.name)
                    .put("time", message.timestampMillis)
                    .put("error", message.isError)
                    .put("voiceDurationSeconds", message.voiceDurationSeconds),
            )
        }
        prefs.edit { putString("messages", array.toString()) }
    }

    fun clear() = prefs.edit { remove("messages") }
}
