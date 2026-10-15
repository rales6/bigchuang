package com.dachuang.robotchat.speech

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.sqrt

object MicrophoneDiagnostics {
    private const val sampleRate = 16_000
    private const val testDurationMillis = 1_400L

    suspend fun run(): String = withContext(Dispatchers.IO) {
        val recognition = measure(MediaRecorder.AudioSource.VOICE_RECOGNITION)
        val microphone = measure(MediaRecorder.AudioSource.MIC)

        if (recognition.error != null || microphone.error != null) {
            return@withContext "麦克风检测失败：${recognition.error ?: microphone.error}"
        }

        val levels = "识别音源 ${recognition.dbLabel}，普通麦克风 ${microphone.dbLabel}。"
        when {
            recognition.hasSignal -> "$levels 输入设备正常；若仍不出文字，更可能是模型识别或发音匹配问题。"
            microphone.hasSignal -> "$levels 普通麦克风正常，但 Vosk 使用的识别音源无信号，属于本机音源兼容问题；请暂时关闭离线识别。"
            else -> "$levels 两种音源都没有检测到有效声音，请检查麦克风权限、系统隐私开关及是否被其他应用占用。"
        }
    }

    @SuppressLint("MissingPermission")
    private fun measure(audioSource: Int): Measurement {
        val minimum = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        if (minimum <= 0) return Measurement(error = "手机不支持 16 kHz 单声道录音")

        val bufferSize = maxOf(minimum, sampleRate / 2) * 2
        val recorder = try {
            AudioRecord(
                audioSource,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize,
            )
        } catch (error: Exception) {
            return Measurement(error = error.message ?: "无法创建录音器")
        }

        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            recorder.release()
            return Measurement(error = "录音器初始化失败")
        }

        return try {
            recorder.startRecording()
            if (recorder.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
                return Measurement(error = "麦克风没有开始录音")
            }

            val buffer = ShortArray(maxOf(minimum / 2, 1_600))
            val deadline = System.nanoTime() + testDurationMillis * 1_000_000L
            var sampleCount = 0L
            var squareSum = 0.0
            var peak = 0
            while (System.nanoTime() < deadline) {
                val count = recorder.read(buffer, 0, buffer.size)
                if (count < 0) return Measurement(error = "读取麦克风失败（$count）")
                for (index in 0 until count) {
                    val value = buffer[index].toInt()
                    peak = maxOf(peak, abs(value))
                    squareSum += value.toDouble() * value.toDouble()
                }
                sampleCount += count
            }

            if (sampleCount == 0L) {
                Measurement(error = "麦克风没有返回音频数据")
            } else {
                val rms = sqrt(squareSum / sampleCount) / Short.MAX_VALUE.toDouble()
                val db = 20.0 * log10(rms.coerceAtLeast(1e-9))
                Measurement(db = db, peak = peak.toDouble() / Short.MAX_VALUE.toDouble())
            }
        } catch (error: SecurityException) {
            Measurement(error = "没有麦克风权限")
        } catch (error: Exception) {
            Measurement(error = error.message ?: "录音测试异常")
        } finally {
            if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) recorder.stop()
            recorder.release()
        }
    }

    private data class Measurement(
        val db: Double = -180.0,
        val peak: Double = 0.0,
        val error: String? = null,
    ) {
        val hasSignal: Boolean get() = db >= -55.0 || peak >= 0.01
        val dbLabel: String get() = if (error == null) "%.1f dB".format(db) else "失败"
    }
}
