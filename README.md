# ESP32 小车执行控制固件

> 树莓派镭神 N10 建图、Vehicle Link V2 客户端和联调测试见
> [`docs/raspberry_pi_mapping.md`](docs/raspberry_pi_mapping.md)。

本目录只保留 ESP32 的确定性执行功能：四轮差速底盘、六关节机械臂、蜂鸣器、
状态灯、安全状态机、树莓派命令链路和执行器总线。自然语言、千问模型、图像、
激光雷达定位、地图与任务规划全部属于树莓派，不在 ESP32 上运行。

原工程完整备份在
[`legacy_backup/2026-07-10_pre_executor_refactor`](legacy_backup/2026-07-10_pre_executor_refactor)，
其中包括 DeepSeek、K210 视觉、语音、PS2、ESP-NOW、自主抓取和旧的
`<...>` / `{...}` / `#...!` / `$...!` 协议实现。备份不会上传到 ESP32。

## 活跃固件结构

```text
car/
├─ boot.py                     # 不初始化外设
├─ main.py                     # 启动入口
├─ app.py                      # UART/BLE、执行器 UART 与主循环装配
├─ config.py                   # 引脚、速度、机械臂限制和控制参数
├─ core/
│  ├─ controller.py            # 指令分发、取消、状态、失联保护
│  └─ timebase.py
├─ drivers/
│  ├─ actuator_bus.py          # 单主机总线仲裁、轮询、重试
│  ├─ legacy_motor_bus.py      # 现有底板 #006–#009 UART 适配
│  ├─ drive_base.py            # 差速运动学、斜坡、反馈 PI
│  ├─ arm.py                   # 机械臂限位与抢占
│  ├─ pwm_arm.py               # 可选的直连 PWM 机械臂后端
│  └─ indicators.py            # 非阻塞 LED/蜂鸣器
├─ protocol/
│  ├─ frame.py                 # Vehicle Link V2 帧与 CRC
│  └─ messages.py              # 固定二进制消息
└─ transport/
   ├─ uart.py                  # 当前生产链路
   ├─ bluetooth.py             # ESP32 BLE Nordic UART Service
   └─ multiplex.py             # UART 优先、响应按来源返回
```

仓库其他内容：

- [`docs/command_set.md`](docs/command_set.md)：测试控制台可输入的完整指令集。
- [`docs/protocol_v2.md`](docs/protocol_v2.md)：统一帧和消息格式。
- [`docs/architecture.md`](docs/architecture.md)：App、树莓派、ESP32、底板边界。
- [`docs/deployment.md`](docs/deployment.md)：接线、烧录、标定和上车顺序。
- [`tools/pi_command_console.py`](tools/pi_command_console.py)：模拟树莓派。

## 快速测试

先确认车轮悬空、机械臂活动范围无障碍，并核对 [`car/config.py`](car/config.py)
中的 GPIO 和波特率。

没有树莓派时，用 USB-TTL 模块连接电脑与 ESP32 的 UART1：USB-TTL TX 接
ESP32 GPIO21、USB-TTL RX 接 GPIO22、GND 共地。ESP32 原生 USB 口仍可用于
烧录和 REPL。必须使用 3.3V TTL 电平。

```bash
python -m pip install pyserial
# Windows：将 COM7 改成 USB-TTL 实际端口
python tools/manual_command_console.py COM7
# Linux：将设备名改成实际端口
python3 tools/manual_command_console.py /dev/ttyUSB0
```

如果直接使用 Thonny 的 `MicroPython (ESP32)` 解释器且没有 USB-TTL，把
`car/` 中的文件上传到 ESP32 根目录，中止自动运行的 main.py 后，在 Shell 输入：

```python
import thonny_manual_test as test
test.send("beep_3")          # 单条同步测试，完成后返回提示符
test.run_auto_test("io")     # 自动测试 LED、蜂鸣器和通信
# 车轮悬空、机械臂周围无障碍后：
test.run_auto_test("all")
test.stop()                   # 释放测试 UART
```

有树莓派后，树莓派连接同一 UART1，并可继续使用：

```bash
python3 tools/pi_command_console.py /dev/serial0
```

树莓派默认采用 UART 优先、BLE 自动备用。完整安装、BLE 强制验证以及车轮/机械臂
分步测试命令见 [`docs/deployment.md`](docs/deployment.md)。BLE 广播名默认为
`ESP32-Robot-Car`，两条链路使用同一套带 CRC 与 ACK 的协议。

基于已经验证 BLE 控制链路的自主建图入口是
[`raspberry_pi/autonomous_mapping.py`](raspberry_pi/autonomous_mapping.py)；
无需连接 ESP32 的前 180°手推建图测试是
[`raspberry_pi/mapping_test.py`](raspberry_pi/mapping_test.py)。运行顺序和安全参数见
[`docs/raspberry_pi_mapping.md`](docs/raspberry_pi_mapping.md)。

进入控制台后可输入：

```text
car> ping
car> led_blink_fast
car> beep_3
car> forward_slow
car> cancel
car> arm_home
car> demo_all
```

任何演示序列都在后台运行；输入新动作或 `cancel` 会中途停止旧序列。底盘速度
命令每 250 ms 刷新，ESP32 收不到新帧超过 800 ms 时立即触发失联保护。

## 当前硬件适配

树莓派/电脑到 ESP32 继续使用 Vehicle Link V2。ESP32 内部根据现有实物转换：

- 四轮电机通过 UART2 发送底板实际支持的 `#006P...!`–`#009P...!`。
- 六个舵机通过同一 UART2 发送 `#000P...!`–`#005P...!`。
- 电机停止使用四通道 `P1500`，舵机抢占/停止使用原底板的 `#255PDST!`。
- 旧底板没有 ACK 和轮速/电压回传，因此 `battery_mv=0`、轮速反馈为零属于当前
  适配器的预期状态；`bus_errors` 应保持为零。

当前后端是 `ACTUATOR_BACKEND="legacy_uart_all"`。后续若底板固件升级，可改成
`vehicle_link_v2`，重新启用
地址轮询、ACK 和状态反馈，不影响树莓派侧协议。
