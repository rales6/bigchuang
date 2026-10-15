package com.dachuang.robotchat

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.dachuang.robotchat.notification.NotificationHelper
import com.dachuang.robotchat.speech.MicrophoneDiagnostics
import com.dachuang.robotchat.speech.SpeechInputController
import com.dachuang.robotchat.speech.TextToSpeechController
import com.dachuang.robotchat.ui.RobotCompanionApp
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    private lateinit var viewModel: RobotViewModel
    private lateinit var speechInput: SpeechInputController
    private lateinit var textToSpeech: TextToSpeechController
    private lateinit var notifications: NotificationHelper
    private var startVoiceAfterPermission = false
    private var testMicrophoneAfterPermission = false
    private var voiceButtonPressed = false

    private val microphonePermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) {
            when {
                testMicrophoneAfterPermission -> runMicrophoneTest()
                startVoiceAfterPermission && voiceButtonPressed -> startVoiceRecognition()
            }
        } else {
            if (testMicrophoneAfterPermission) {
                viewModel.setMicrophoneTestResult("没有麦克风权限，无法检测输入设备")
            } else {
                voiceButtonPressed = false
                viewModel.onSpeechError("需要麦克风权限才能使用语音输入")
            }
        }
        startVoiceAfterPermission = false
        testMicrophoneAfterPermission = false
    }

    private val notificationPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (!granted) viewModel.setNotifications(false)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewModel = ViewModelProvider(this)[RobotViewModel::class.java]
        notifications = NotificationHelper(this)
        textToSpeech = TextToSpeechController(this)
        speechInput = SpeechInputController(
            context = this,
            onListeningChanged = viewModel::setListening,
            onResult = viewModel::onSpeechResult,
            onError = viewModel::onSpeechError,
            onPartial = viewModel::onSpeechPartial,
        )

        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                launch { viewModel.speechEvents.collect(textToSpeech::speak) }
                launch { viewModel.notificationEvents.collect(notifications::showMessage) }
                launch {
                    viewModel.uiState
                        .map { it.settings.dialectTag }
                        .distinctUntilChanged()
                        .collect(textToSpeech::setLanguage)
                }
                launch {
                    viewModel.uiState
                        .map { it.settings.carConfig.audioVolume }
                        .distinctUntilChanged()
                        .collect(textToSpeech::setVolume)
                }
            }
        }

        setContent {
            RobotCompanionApp(
                viewModel = viewModel,
                onStartVoice = ::requestVoiceRecognition,
                onStopVoice = ::stopVoiceRecognition,
                onCancelVoice = ::cancelVoiceRecognition,
                onRequestNotificationPermission = ::requestNotificationPermission,
                onTestMicrophone = ::requestMicrophoneTest,
            )
        }
    }

    private fun requestVoiceRecognition() {
        voiceButtonPressed = true
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            startVoiceRecognition()
        } else {
            startVoiceAfterPermission = true
            microphonePermission.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    private fun startVoiceRecognition() {
        if (!voiceButtonPressed) return
        viewModel.beginVoiceCapture()
        val settings = viewModel.uiState.value.settings
        val preferVosk = settings.speechEngine == com.dachuang.robotchat.model.SpeechEnginePref.AUTO
        speechInput.start(settings.dialectTag, preferVosk)
    }

    private fun stopVoiceRecognition() {
        voiceButtonPressed = false
        startVoiceAfterPermission = false
        speechInput.stop()
        viewModel.endVoiceCapture()
        viewModel.setListening(false)
    }

    private fun cancelVoiceRecognition() {
        voiceButtonPressed = false
        startVoiceAfterPermission = false
        viewModel.cancelVoiceCapture()
        speechInput.cancel()
    }

    private fun requestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
    }

    private fun requestMicrophoneTest() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            runMicrophoneTest()
        } else {
            testMicrophoneAfterPermission = true
            microphonePermission.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    private fun runMicrophoneTest() {
        viewModel.setMicrophoneTestRunning(true)
        lifecycleScope.launch {
            val result = MicrophoneDiagnostics.run()
            viewModel.setMicrophoneTestResult(result)
        }
    }

    override fun onDestroy() {
        speechInput.destroy()
        textToSpeech.shutdown()
        super.onDestroy()
    }
}
