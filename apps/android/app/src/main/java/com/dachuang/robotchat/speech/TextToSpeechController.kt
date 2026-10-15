package com.dachuang.robotchat.speech

import android.content.Context
import android.os.Bundle
import android.speech.tts.TextToSpeech
import java.util.Locale

class TextToSpeechController(context: Context) : TextToSpeech.OnInitListener {
    private val tts = TextToSpeech(context.applicationContext, this)
    private var ready = false
    private var pendingText: String? = null
    private var languageTag: String = "zh-CN"
    private var volume: Float = .8f

    override fun onInit(status: Int) {
        ready = status == TextToSpeech.SUCCESS
        if (ready) {
            applyLanguage()
            pendingText?.let(::speak)
            pendingText = null
        }
    }

    fun setLanguage(tag: String) {
        languageTag = tag
        if (ready) applyLanguage()
    }

    fun setVolume(value: Float) {
        volume = value.coerceIn(0f, 1f)
    }

    fun speak(text: String) {
        if (text.isBlank()) return
        if (!ready) {
            pendingText = text
            return
        }
        val parameters = Bundle().apply {
            putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, volume)
        }
        tts.speak(text, TextToSpeech.QUEUE_FLUSH, parameters, "robot-reply-${System.nanoTime()}")
    }

    fun stop() = tts.stop()

    fun shutdown() = tts.shutdown()

    private fun applyLanguage() {
        val requested = Locale.forLanguageTag(languageTag)
        val result = tts.setLanguage(requested)
        if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
            tts.setLanguage(Locale.SIMPLIFIED_CHINESE)
        }
    }
}
