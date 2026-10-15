# 构建与验证记录

## 2026-09-30：四底栏与按住说话界面

- 本地执行 `testDebugUnitTest assembleDebug`：构建成功，9 项单元测试通过。
- 安装包：[dist/RobotCompanion-1.2.0-debug.apk](dist/RobotCompanion-1.2.0-debug.apk)，包名 `com.dachuang.robotchat`，版本 `1.2.0`（versionCode 7），APK v2 签名校验通过。
- SHA-256：`8C97075AE924F488683ADE935877C9F701B08BC33E6EA5F83BCE970087B94369`。
- 树莓派不在现场，未做手机安装、语音硬件或实车联调；四底栏、手势取消和屏幕尺寸仍需在手机上确认。


验证日期：2026-08-14（Asia/Shanghai）

## 自动化结果

执行任务：

```powershell
gradle testDebugUnitTest assembleDebug lintDebug --no-daemon
```

结果：

- Kotlin / Java 编译：通过。
- Debug APK 打包：通过。
- 单元测试：9 个测试全部通过，0 failure，0 error。
  - `ChildSafetyPolicyTest`：少儿模式安全停止放行、运动控制和超长消息拦截。
  - `DndScheduleTest`：跨午夜定时免打扰和始终免打扰。
  - `RobotStateParserTest`：服务状态、目标队列、嵌套回复，以及任务规划和托管进程解析。
  - `VoskModelManagerTest`：离线语音模型状态处理。
- Android Lint：0 error，16 warning。

Lint warning 包括依赖版本提示、Compose 参数顺序、第三方原生库 16KB 对齐和局域网 cleartext HTTP 等非阻断项。HTTP 是为当前可信局域网内树莓派 `/control`、`/state`、`/task/map.png` 和 MJPEG 兼容而保留；跨公网部署前必须改用 HTTPS 和设备鉴权。

## APK 校验

文件：`dist/RobotCompanion-1.1.0-debug.apk`

```text
包名:       com.dachuang.robotchat
版本:       1.1.0 (versionCode 6)
minSdk:     26
targetSdk:  36
compileSdk: 36
大小:       56,392,424 bytes
SHA-256:    C79C9ADFA23E45BB09963142406DCE3BA5895F054405A77AC8D2C7B66C6B972F
签名:       APK Signature Scheme v2，Android Debug RSA 2048
```

`aapt dump badging` 和 `apksigner verify --verbose --print-certs` 均通过。

## 尚需真机联调的部分

当前电脑没有连接 Android 手机或模拟器，因此没有执行安装后的端到端 UI/硬件测试。以下项目需要在真实树莓派和手机上确认：

- 手机厂商提供的普通话、粤语、台湾国语 TTS 音色可用性。
- 系统 `SpeechRecognizer` 在现场网络和噪声条件下的识别效果。
- 实际树莓派地址、`/state` 返回字段与 `/stream.mjpg` 帧率。
- 不同厂商 WebView 对持续 MJPEG 流的稳定性。
- Android 13+ 通知权限弹窗和各厂商后台限制。

这些设备相关项不影响当前源码编译、协议解析测试和 APK 完整性验证。
