package com.dachuang.robotchat.speech

interface SpeechEngine {
    fun start()
    fun stop()
    fun cancel()
    fun destroy()
}
