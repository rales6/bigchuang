# ESP32 小车测试指令集

无树莓派测试程序：[`tools/manual_command_console.py`](../tools/manual_command_console.py)。
它在电脑上运行，通过 USB-TTL 模块模拟树莓派，并使用专用 UART 向 ESP32 发命令。
原 [`tools/pi_command_console.py`](../tools/pi_command_console.py) 保留给后续树莓派端使用。

## 启动

```bash
python -m pip install pyserial
# Windows：COM7 替换为 USB-TTL 的实际端口
python tools/manual_command_console.py COM7
# Linux：/dev/ttyUSB0 替换为 USB-TTL 的实际端口
python3 tools/manual_command_console.py /dev/ttyUSB0
```

串口默认 `230400, 8-N-1`。测试工具后台每 250 ms 发送心跳或刷新底盘速度。

USB-TTL 接线：TX → ESP32 GPIO21（UART1 RX），RX ← ESP32 GPIO22（UART1 TX），
GND 共地。请使用 3.3V TTL；不要把 5V 串口电平直接接入 ESP32。

### 直接使用 Thonny 的 ESP32 Shell

如果 Thonny 右下角选择的是 `MicroPython (ESP32)`，电脑端 pyserial 工具不能在
ESP32 上运行。将 `car/` 目录内容上传到 ESP32 根目录，停止自动执行的 main.py，
然后在 Thonny Shell 逐行输入：

```python
import thonny_manual_test as test
test.send("ping")
test.send("led_blink_fast")
test.send("forward_slow")
test.send("stop_drive")
test.send("arm_home")
test.send("cancel")
test.diagnostics()             # 查看 UART/PWM 写入计数和最后一条电机帧
test.stop()
```

`send()` 会在当前 Thonny Shell 调用中同步推进控制循环，动作完成后再返回提示符。
它不创建后台线程，因此不会用异步输出破坏 Thonny raw-REPL。自动测试中的底盘
动作约 1.5 秒，机械臂动作约 1.8–3.2 秒；异常或 Ctrl+C 会先请求停止。

自动流程：

```python
test.run_auto_test("io")     # 默认安全：通信、LED、蜂鸣器
test.run_auto_test("drive")  # 必须先让四轮悬空
test.run_auto_test("arm")    # 必须清空机械臂活动范围
test.run_auto_test("all")    # 满足以上两项安全条件后运行
```

每项会输出 `[RUN]`、`[PASS]` 或 `[FAIL]`，最后输出通过/失败数量和 ESP32 状态。
任何一项异常都会停止本组并执行全部取消。

`[PASS] driver output confirmed` 表示 ESP32 已正确解析，并确认至少写出过电机
或舵机 UART 指令；它仍不能代替对真实运动的人工观察。当前旧底板没有
标准反馈，因此 `wheel_feedback=(0,0,0,0)`、`battery_mv=0` 是预期值，
`bus_errors` 应为零。机械臂 `joint_positions` 是 PWM 指令位置，不是传感器回读。
`test.diagnostics()` 中 `motor_drive_writes` 和 `servo_uart_writes` 应随对应测试
增加；`last_drive_frame`、`last_arm_frame` 和停止帧可用于核对实际发往底板的
`#000`–`#009` 字符串。

## 系统与安全

| 控制台指令 | 行为 |
|---|---|
| `help` | 显示文档位置 |
| `ping` | 心跳测试，成功显示 `pong` |
| `status` | 输出轮速、舵机、状态位、电压和总线错误 |
| `cancel` / `stop_all` | 中止底盘、机械臂、蜂鸣器并关闭 LED |
| `quit` / `exit` | 先发送全部取消，再关闭串口 |

## 四轮底盘

| 控制台指令 | 线速度 / 角速度 | 用途 |
|---|---:|---|
| `forward_slow` | `180 mm/s, 0` | 初次悬空测试 |
| `forward` | `350 mm/s, 0` | 正常前进 |
| `forward_fast` | `500 mm/s, 0` | 高速前进，标定后使用 |
| `backward` | `-300 mm/s, 0` | 后退 |
| `spin_left` | `0, +2200 mrad/s` | 原地左转 |
| `spin_right` | `0, -2200 mrad/s` | 原地右转 |
| `curve_left` | `300, +900 mrad/s` | 前进左弧线 |
| `curve_right` | `300, -900 mrad/s` | 前进右弧线 |
| `drive <v> <w> [ttl]` | 自定义 | 例：`drive 220 -500 600` |
| `stop_drive` | — | 只停止底盘 |
| `demo_drive` | 多段 | 前进→旋转→弧线→停止 |

速度会经过曲率保持限幅和加速度斜坡，不会把指令阶跃直接送到电机。当前底板
没有轮速反馈，所以 PI 自动退化为开环。若 600 ms 内没有续发，当前速度自动
失效；整条 Pi 链路 800 ms 无有效帧会触发立即停止。

## 六关节机械臂

| 控制台指令 | 行为 |
|---|---|
| `arm_home` | 六关节回到配置的安全初始位置 |
| `arm_left` / `arm_right` | 0 号底座关节左/右转 |
| `arm_up` / `arm_down` | 1–3 号关节协同抬升/下降 |
| `grip` / `release` | 5 号夹爪闭合/松开 |
| `arm <id> <pulse> [duration]` | 单关节目标，如 `arm 0 1800 600` |
| `arm_stop` | 保持当前机械臂位置 |
| `demo_arm` | 回零→左转→下降→夹取→回零 |

所有关节目标都会检查 [`car/config.py`](../car/config.py) 中的独立脉宽限制。只有
上一条机械臂命令仍在运动时，新命令才按“停止旧动作→发送新目标”的顺序抢占；
上一动作已经正常完成时直接发送新目标，避免旧底板被不必要的 `#255PDST!`
暂停。`arm_stop`、异常和 Ctrl+C 始终可以立即停止机械臂。

## 蜂鸣器与状态灯

| 控制台指令 | 行为 |
|---|---|
| `beep_short` | 短响一次 |
| `beep_3` | 短响三次 |
| `beep_long` | 长响一次 |
| `led_off` | 关闭状态灯 |
| `led_on` | 状态灯常亮 |
| `led_blink_slow` | 500 ms 翻转一次 |
| `led_blink_fast` | 100 ms 翻转一次 |
| `demo_io` | 快闪与三声提示，然后常亮、关闭 |

LED 和蜂鸣器使用非阻塞状态机，播放过程中仍然每 5 ms 检查树莓派数据。新模式
会立即替换旧模式。

## 综合测试

| 控制台指令 | 行为 |
|---|---|
| `demo_all` | 依次测试 IO、底盘和机械臂 |

演示序列在后台线程运行，因此控制台仍可输入。任何新的手动动作指令会取消正在
运行的演示；`cancel` 始终可以立即请求全部停止。

## 推荐上车顺序

1. 车轮悬空，只连接 ESP32 与底板，执行 `ping`、`status`。
2. 测 `led_on`、`beep_short`，确认 GPIO 高低电平配置。
3. 测 `forward_slow` 后立即 `stop_drive`，核对四轮正方向；错误时修改
   `MOTOR_DIRECTIONS`，不要改协议符号定义。
4. 逐一执行 `arm <id> ...`，确认每个关节限位，再执行 `demo_arm`。
5. 断开树莓派串口或结束测试进程，确认 800 ms 内底盘停止。
6. 最后落地测试 `demo_drive`，逐步调整轨距、速度换算、最小有效控制量和 PI。
