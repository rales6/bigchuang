package com.dachuang.robotchat.speech

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer

class SystemSpeechEngine(
    context: Context,
    private val languageTag: String,
    private val onListeningChanged: (Boolean) -> Unit,
    private val onResult: (String) -> Unit,
    private val onPartial: (String) -> Unit,
    private val onError: (String) -> Unit,
) : SpeechEngine {
    private val appContext = context.applicationContext
    private var recognizer: SpeechRecognizer? = null
    private var sessionId = 0L
    private var listening = false

    override fun start() {
        if (!SpeechRecognizer.isRecognitionAvailable(appContext)) {
            onListeningChanged(false)
            onError("当前手机没有可用的语音识别服务")
            return
        }

        // Each press gets a fresh session. Late callbacks from a previous press are ignored.
        val currentSession = ++sessionId
        recognizer?.cancel()
        recognizer?.destroy()
        recognizer = SpeechRecognizer.createSpeechRecognizer(appContext).also { speechRecognizer ->
            speechRecognizer.setRecognitionListener(listenerFor(currentSession))
        }

        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, languageTag)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_MINIMUM_LENGTH_MILLIS, 1_000L)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, 60_000L)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 60_000L)
        }

        try {
            listening = true
            recognizer?.startListening(intent)
            onListeningChanged(true)
        } catch (error: Exception) {
            finishSession(currentSession)
            onError("语音识别启动失败：${error.message ?: "未知错误"}")
        }
    }

    override fun stop() {
        if (listening) {
            listening = false
            recognizer?.stopListening()
        }
        onListeningChanged(false)
    }

    override fun cancel() {
        sessionId++
        listening = false
        recognizer?.cancel()
        recognizer?.destroy()
        recognizer = null
        onListeningChanged(false)
    }

    override fun destroy() {
        cancel()
    }

    private fun listenerFor(session: Long) = object : RecognitionListener {
        override fun onReadyForSpeech(params: Bundle?) = Unit
        override fun onBeginningOfSpeech() = Unit
        override fun onRmsChanged(rmsdB: Float) = Unit
        override fun onBufferReceived(buffer: ByteArray?) = Unit
        override fun onEndOfSpeech() = Unit
        override fun onEvent(eventType: Int, params: Bundle?) = Unit
        override fun onPartialResults(partialResults: Bundle?) {
            if (session != sessionId) return
            partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                ?.firstOrNull()
                ?.takeIf { it.isNotBlank() }
                ?.let(onPartial)
        }

        override fun onResults(results: Bundle?) {
            if (session != sessionId) return
            val text = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                ?.firstOrNull()
                .orEmpty()
            finishSession(session)
            if (text.isBlank()) onError("没有识别到清晰的语音，请重试") else onResult(text)
        }

        override fun onError(error: Int) {
            if (session != sessionId) return
            finishSession(session)
            val message = when (error) {
                SpeechRecognizer.ERROR_AUDIO -> "录音设备出现问题"
                SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "没有麦克风权限"
                SpeechRecognizer.ERROR_NETWORK,
                SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "语音识别网络不可用"
                SpeechRecognizer.ERROR_NO_MATCH -> "没有听清，请重试"
                SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "语音识别正忙，请重新按住说话"
                SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "没有检测到语音"
                else -> "语音识别失败（错误码 $error）"
            }
            onError(message)
        }
    }

    private fun finishSession(session: Long) {
        if (session != sessionId) return
        listening = false
        recognizer?.destroy()
        recognizer = null
        onListeningChanged(false)
    }
}
