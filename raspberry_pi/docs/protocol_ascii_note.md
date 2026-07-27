# ASCII 兼容帧说明

Pi UART 除了 Vehicle Link V2 二进制帧，也接受临时联调用 ASCII 帧。帧边界为 `#` 和 `!`，例如：

```text
#ARMF100U050L000!
#MOVF200L000!
#LEDON!
#BEEP003!
#STOP!
```

解析入口在 `car/protocol/ascii_commands.py`，执行入口在 `RobotController.handle_ascii_command()`。ASCII 帧和二进制帧共用同一条 Pi UART，但不会被转发到底板；ESP32 会把它们转换成现有控制层命令。

后期树莓派程序稳定后，仍推荐使用带地址、序号和 CRC 的 Vehicle Link V2 二进制帧作为正式协议。ASCII 帧主要用于人工调试、串口助手测试和早期 App/树莓派联调。
