# 部署、接线与标定

## 接线原则

- 树莓派 TX → ESP32 UART1 RX21；树莓派 RX ← ESP32 UART1 TX22；必须共地。
- 现有电机底板 RX ← ESP32 UART2 TX17；必须共地。若底板有 TX，可接 ESP32
  RX16，但当前旧协议适配器不依赖状态回复。
- 六个舵机同样由串口底板的 ID000–005 控制，不直接占用 ESP32 PWM GPIO。
- UART0 仅用于 REPL/日志，不传控制帧。
- BLE 备用链路广播名默认为 `ESP32-Robot-Car`，使用标准 Nordic UART Service；
  UART 与 BLE 传输完全相同的 Vehicle Link V2 帧。
- 所有信号必须是兼容的 3.3V 逻辑电平。电机电源与逻辑电源布线应按硬件规格
  处理，不能由本文的软件配置代替电气设计。
- GPIO2 默认状态灯，GPIO5 默认蜂鸣器；与实际板卡不同必须修改 config.py。

## 无树莓派手动测试

电脑无需经过树莓派，可使用 USB-TTL 模块临时占用 UART1：USB-TTL TX 接 GPIO21、
USB-TTL RX 接 GPIO22、GND 共地。电脑运行：

```bash
python -m pip install pyserial
python tools/manual_command_console.py COM7
```

开发板原生 USB 仍保留给烧录和 REPL；不要把 USB-TTL 接到 UART0，以免与 REPL
争用。测试结束后拔除 USB-TTL，再将 UART1 接回树莓派。

树莓派串口需要关闭串口控制台并启用对应 UART；具体方式依树莓派型号与系统
版本而异，请以 [Raspberry Pi 官方 UART 配置文档](https://www.raspberrypi.com/documentation/computers/configuration.html)
为准。

## 固件部署

只把 `car/` 目录中的内容上传到 ESP32 MicroPython 根目录。不要上传 `tests/`、
`tools/` 或 `legacy_backup/`。

当前 `ACTUATOR_BACKEND="legacy_uart_all"`，兼容原底板 ID000–005 舵机和
ID006–009 电机指令。`protocol_v2.md` 中 `0x50`–`0x54` 是未来可升级的底板接口，
不是当前硬件必须实现的内容。

## 树莓派 UART 与 BLE 备用链路

安装依赖并启用树莓派硬件串口：

```bash
python3 -m pip install -r requirements-pi.txt
```

`raspberry_pi/config.py` 的 `link_mode` 有三种取值：

- `auto`：默认值。先打开 `/dev/serial0`，请求超时或串口读写出错后切换 BLE；
- `uart`：只使用 UART，适合排除蓝牙变量；
- `ble`：只使用 BLE，适合验证备用链路。

建议按以下顺序联调，车轮测试前必须架空车体，机械臂周围必须无障碍：

```bash
# 1. 纯通信、状态查询（不会驱动车轮和机械臂）
python3 -m raspberry_pi.communication_test --link uart --count 10

# 2. 强制验证 BLE
python3 -m raspberry_pi.communication_test --link ble --count 10

# 3. UART 优先、BLE 自动备用，并测试指示灯/蜂鸣器
python3 -m raspberry_pi.communication_test --link auto --io-test

# 4. 明确确认安全后，分别测试车轮和机械臂
python3 -m raspberry_pi.communication_test --link auto --motion-test
python3 -m raspberry_pi.communication_test --link auto --arm-test
```

如果现场有多个同名 ESP32，可用 `--ble-address` 指定设备地址，或修改 ESP32
`car/config.py` 中的 `PI_BLE_NAME`，并让树莓派使用相同的 `--ble-name`。

自主建图以 450 ms 间隔刷新完全相同的速度命令，命令 TTL 为 700 ms。ESP32 超过
800 ms 没收到有效帧仍会立即停止底盘和机械臂；链路切换期间这项保护继续生效。

BLE 运动帧使用 NUS 的 write-without-response，再由 Vehicle Link V2 的 CRC、ACK、
序号幂等重试保证首次下发、变速和变向可靠。已经确认过且内容完全相同的 TTL 保活
帧不再要求应用层 ACK，以免每 450 ms 的重复通知堵塞 BLE；保活丢失时 TTL/失联
保护仍会停车。ESP32 默认关闭主动状态广播，状态只在树莓派查询时返回。树莓派会
缓存首次扫描到的 BLE 地址，短暂失败后直接重连，并将单次 GATT 操作限制在
0.8 秒、单次应答等待限制在 0.4 秒；设备扫描超时不再被错误地用于每次写操作。

## 首次标定项目

在 [`car/config.py`](../car/config.py) 中依次确认：

1. 两条 UART 的编号、引脚、波特率；
2. `MOTOR_DIRECTIONS` 四轮安装方向；
3. `TRACK_WIDTH_MM` 实际左右轮中心距；
4. `MOTOR_UNITS_PER_MM_S` 速度换算；
5. `MIN_EFFECTIVE_MOTOR_UNITS` 克服静摩擦的最小控制量；
6. `MAX_LINEAR_MM_S`、`MAX_WHEEL_MM_S` 安全上限；
7. 六个舵机的 `ARM_LIMITS_US` 与 `ARM_HOME_US`；
8. LED、蜂鸣器的有效电平。

四轮独立输出倍率不再写入 ESP32 的 `config.py`。自主建图时由树莓派的
`--front-left-gain`、`--front-right-gain`、`--rear-left-gain`、
`--rear-right-gain` 下发，顺序为左前、右前、左后、右后。

先用较低的 `WHEEL_KP/WHEEL_KI`。只有底板状态中的轮速确实是 mm/s 且刷新稳定
时才启用闭环效果；反馈超过 300 ms 未更新，ESP32 会自动退化为开环斜坡控制。
