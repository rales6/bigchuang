package com.dachuang.robotchat.speech

import android.content.Context

class SpeechInputController(
    context: Context,
    private val onListeningChanged: (Boolean) -> Unit,
    private val onResult: (String) -> Unit,
    private val onError: (String) -> Unit,
    private val onPartial: ((String) -> Unit)? = null,
) {
    private val appContext = context.applicationContext
    private var systemEngine: SystemSpeechEngine? = null
    private var systemLanguageTag: String? = null
    private var voskEngine: VoskSpeechEngine? = null
    private var active: SpeechEngine? = null

    fun start(languageTag: String, preferVosk: Boolean) {
        // A missed release or a late recognizer callback must not block the next press.
        active?.cancel()
        active = null
        val useVosk = preferVosk && languageTag == "zh-CN" && VoskModelManager.isReady(appContext)
        val engine: SpeechEngine = if (useVosk) {
            voskEngine ?: createVoskEngine().also { voskEngine = it }
        } else {
            if (systemEngine == null || systemLanguageTag != languageTag) {
                systemEngine?.destroy()
                systemEngine = createSystemEngine(languageTag)
                systemLanguageTag = languageTag
            }
            checkNotNull(systemEngine)
        }
        active = engine
        try {
            engine.start()
        } catch (e: Exception) {
            // If Vosk fails, silently fall back to system engine
            if (useVosk) {
                val fallback = systemEngine ?: createSystemEngine(languageTag).also {
                    systemEngine = it
                    systemLanguageTag = languageTag
                }
                active = fallback
                fallback.start()
            } else {
                onError("语音识别启动失败：${e.message}")
                onListeningChanged(false)
            }
        }
    }

    fun stop() {
        active?.stop()
        active = null
    }

    fun cancel() {
        active?.cancel()
        active = null
    }

    fun destroy() {
        active = null
        systemEngine?.destroy()
        voskEngine?.destroy()
        voskEngine = null
        systemEngine = null
        systemLanguageTag = null
    }

    private fun createVoskEngine(): VoskSpeechEngine {
        val modelPath = VoskModelManager.modelPath(appContext)
        return VoskSpeechEngine(
            modelPath = modelPath,
            onListeningChanged = onListeningChanged,
            onResult = onResult,
            onPartial = { text -> onPartial?.invoke(text) },
            onError = onError,
        )
    }

    private fun createSystemEngine(languageTag: String) = SystemSpeechEngine(
        context = appContext,
        languageTag = languageTag,
        onListeningChanged = onListeningChanged,
        onResult = onResult,
        onPartial = { text -> onPartial?.invoke(text) },
        onError = onError,
    )
}
