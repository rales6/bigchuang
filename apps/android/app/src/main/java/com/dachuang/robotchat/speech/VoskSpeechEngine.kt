package com.dachuang.robotchat.speech

import org.vosk.Model
import org.vosk.android.SpeechService
import org.vosk.android.RecognitionListener
import java.io.File

class VoskSpeechEngine(
    modelPath: File,
    private val onListeningChanged: (Boolean) -> Unit,
    private val onResult: (String) -> Unit,
    private val onPartial: (String) -> Unit,
    private val onError: (String) -> Unit,
) : SpeechEngine {
    private val sampleRate = 16000f
    private var model: Model? = null
    private var speechService: SpeechService? = null
    private val finalizedSegments = mutableListOf<String>()
    private var latestPartial = ""
    private var resultDelivered = false
    private var sessionId = 0L

    init {
        model = Model(modelPath.absolutePath)
    }

    override fun start() {
        val currentSession = ++sessionId
        speechService?.stop()
        speechService = null
        finalizedSegments.clear()
        latestPartial = ""
        resultDelivered = false
        try {
            val recognizer = org.vosk.Recognizer(model, sampleRate).apply {
                setMaxAlternatives(1)
                setWords(false)
            }
            speechService = SpeechService(recognizer, sampleRate).apply {
                startListening(object : RecognitionListener {
                    override fun onResult(hypothesis: String?) {
                        if (currentSession != sessionId) return
                        val text = parseText(hypothesis)
                        if (!text.isNullOrBlank()) {
                            finalizedSegments += text
                        }
                    }

                    override fun onFinalResult(hypothesis: String?) {
                        if (currentSession != sessionId) return
                        deliverFinalResult(parseText(hypothesis))
                    }

                    override fun onPartialResult(hypothesis: String?) {
                        if (currentSession != sessionId) return
                        val text = parseText(hypothesis)
                        if (!text.isNullOrBlank()) {
                            latestPartial = text
                            onPartial(text)
                        }
                    }

                    override fun onError(exception: Exception?) {
                        if (currentSession != sessionId) return
                        if (resultDelivered) return
                        resultDelivered = true
                        onListeningChanged(false)
                        onError("离线语音识别出错：${exception?.message ?: "未知错误"}")
                    }

                    override fun onTimeout() {
                        if (currentSession != sessionId) return
                        onListeningChanged(false)
                    }
                })
            }
            onListeningChanged(true)
        } catch (e: Exception) {
            onListeningChanged(false)
            onError("离线语音启动失败：${e.message}")
        }
    }

    override fun stop() {
        speechService?.stop()
        speechService = null
        onListeningChanged(false)
    }

    override fun cancel() {
        sessionId++
        resultDelivered = true
        speechService?.stop()
        speechService = null
        finalizedSegments.clear()
        latestPartial = ""
        onListeningChanged(false)
    }

    override fun destroy() {
        cancel()
        model?.close()
        model = null
    }

    private fun parseText(json: String?): String? {
        if (json.isNullOrBlank()) return null
        return try {
            val value = org.json.JSONObject(json)
            value.optString("text", "")
                .ifBlank { value.optString("partial", "") }
                .trim()
                .ifBlank { null }
        } catch (_: Exception) {
            null
        }
    }

    private fun deliverFinalResult(lastSegment: String?) {
        if (resultDelivered) return
        resultDelivered = true
        if (!lastSegment.isNullOrBlank()) finalizedSegments += lastSegment
        val text = finalizedSegments.joinToString(" ").trim().ifBlank { latestPartial }
        onListeningChanged(false)
        if (text.isBlank()) {
            onError("没有检测到可识别语音，请到“设置 → 语音”运行麦克风检测")
        } else {
            onResult(text)
        }
    }
}
