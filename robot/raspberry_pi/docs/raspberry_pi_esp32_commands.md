# 树莓派控制 ESP32 指令手册

本文说明树莓派如何通过 UART 或 BLE 控制 ESP32，包括通信测试、Python API、
指令参数、响应和安全要求。

## 1. 通信结构

树莓派使用 `raspberry_pi.esp32.Esp32Client` 发送 Vehicle Link V2 二进制帧，
ESP32 收到指令后执行控制并返回 ACK、状态或错误响应。

支持三种链路模式：

| 模式 | 说明 |
| --- | --- |
| `uart` | 只使用树莓派硬件串口，默认设备为 `/dev/serial0` |
| `ble` | 只使用 BLE Nordic UART Service，默认设备名为 `ESP32-Robot-Car` |
| `auto` | 优先使用 UART，失败时自动切换到 BLE |

UART 默认波特率为 `230400`。ESP32 和树莓派使用 UART 时必须共地。

## 2. 命令行通信测试

在树莓派项目目录中进入虚拟环境：

```bash
cd ~/dachuang/car
source .venv/bin/activate
```

### 2.1 基础心跳和状态测试

使用 BLE：

```bash
python3 -m raspberry_pi.communication_test \
  --link ble \
  --ble-name ESP32-Robot-Car \
  --count 10
```

使用指定 BLE 地址：

```bash
python3 -m raspberry_pi.communication_test \
  --link ble \
  --ble-address AA:BB:CC:DD:EE:FF \
  --count 10
```

使用 UART：

```bash
python3 -m raspberry_pi.communication_test \
  --link uart \
  --port /dev/serial0 \
  --baud 230400 \
  --count 10
```

自动选择链路：

```bash
python3 -m raspberry_pi.communication_test --link auto --count 10
```

该测试只发送心跳并查询状态，不会驱动车轮或机械臂。

### 2.2 LED 和蜂鸣器测试

```bash
python3 -m raspberry_pi.communication_test --link ble --io-test
```

该命令会：

1. 让 LED 以 `150 ms` 周期闪烁；
2. 让蜂鸣器响一次，响 `150 ms`、间隔 `100 ms`；
3. 等待约 `0.6 s`；
4. 关闭 LED。

### 2.3 底盘运动测试

运行前必须架空车轮，并确保能够立即切断电源：

```bash
python3 -m raspberry_pi.communication_test --link ble --motion-test
```

该测试发送 `100 mm/s` 的直线速度，持续约 `0.5 s`，随后发送停车指令。

### 2.4 机械臂测试

运行前确认机械臂周围没有障碍物：

```bash
python3 -m raspberry_pi.communication_test --link ble --arm-test
```

该测试将关节 0 从 `1500 us` 移到 `1550 us`，再回到 `1500 us`，最后停止机械臂。

命令行参数汇总：

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--link` | `auto` | 选择 `auto`、`uart` 或 `ble` |
| `--port` | `/dev/serial0` | UART 设备 |
| `--baud` | `230400` | UART 波特率 |
| `--ble-name` | `ESP32-Robot-Car` | BLE 广播名称 |
| `--ble-address` | 无 | 直接指定 BLE 地址 |
| `--count` | `10` | 心跳测试次数 |
| `--io-test` | 关闭 | 执行 LED 和蜂鸣器测试 |
| `--motion-test` | 关闭 | 执行短距离底盘测试 |
| `--arm-test` | 关闭 | 执行机械臂小幅运动测试 |

## 3. Python 控制程序基本写法

以下示例放在树莓派上运行：

```python
import time

from raspberry_pi.config import SerialConfig
from raspberry_pi.esp32 import Esp32Client


config = SerialConfig(
    link_mode="ble",
    ble_device_name="ESP32-Robot-Car",
)

client = Esp32Client(config)
try:
    client.heartbeat()
    status = client.query_status()
    print(status)
finally:
    client.close()
```

使用 UART 时：

```python
config = SerialConfig(
    link_mode="uart",
    port="/dev/serial0",
    baudrate=230400,
)
```

使用自动备用链路时：

```python
config = SerialConfig(
    link_mode="auto",
    port="/dev/serial0",
    baudrate=230400,
    ble_device_name="ESP32-Robot-Car",
)
```

程序必须在 `finally` 中调用 `client.close()`，以正确关闭串口或 BLE 连接。

## 4. 全部控制指令

### 4.1 `heartbeat()`：心跳

```python
client.heartbeat()
```

用途：

- 检查 ESP32 是否在线；
- 更新 ESP32 的树莓派活动时间；
- 防止通信失联保护被触发。

ESP32 长时间收不到有效指令时会进入失联保护并停止运动。持续控制车辆时，应用程序
必须定期发送心跳或有效控制帧。

### 4.2 `query_status()`：查询机器人状态

```python
status = client.query_status()
print(status)
```

返回字典字段：

| 字段 | 含义 |
| --- | --- |
| `uptime_ms` | ESP32 启动后的运行时间，单位 ms |
| `flags` | 当前状态位集合 |
| `linear_mm_s` | 当前线速度，单位 mm/s |
| `angular_mrad_s` | 当前角速度，单位 mrad/s |
| `left_output` | 左侧电机控制输出 |
| `right_output` | 右侧电机控制输出 |
| `wheel_feedback` | 四个车轮的反馈值 |
| `joint_positions` | 六个机械臂关节当前位置 |
| `battery_mv` | 电池电压，单位 mV |
| `bus_errors` | 执行器总线错误计数 |

示例：

```python
status = client.query_status()
print("电池电压:", status["battery_mv"], "mV")
print("关节位置:", status["joint_positions"])
print("总线错误:", status["bus_errors"])
```

### 4.3 `set_twist()`：控制底盘速度

```python
client.set_twist(linear_mm_s, angular_mrad_s, ttl_ms=600)
```

参数：

| 参数 | 含义 |
| --- | --- |
| `linear_mm_s` | 线速度；正数前进，负数后退 |
| `angular_mrad_s` | 角速度；正数和负数代表相反转向 |
| `ttl_ms` | 指令有效期；到期未续期时自动停止 |

直行：

```python
client.set_twist(200, 0, 600)
```

后退：

```python
client.set_twist(-150, 0, 600)
```

原地转向：

```python
client.set_twist(0, 800, 600)
```

边前进边转向：

```python
client.set_twist(200, 500, 600)
```

持续运动时需要在 TTL 到期前重复发送。例如：

```python
import time

deadline = time.monotonic() + 3.0
try:
    while time.monotonic() < deadline:
        client.set_twist(150, 0, 600)
        time.sleep(0.25)
finally:
    client.stop()
```

项目默认限制由 ESP32 的 `car/config.py` 决定，包括最大线速度、最大角速度、
最大轮速和最大 TTL。首次上车应使用低速值。

### 4.4 `stop()`：停止底盘

```python
client.stop()
```

该指令取消树莓派客户端保存的持续速度命令，并要求 ESP32 紧急停止底盘。

建议所有运动代码使用：

```python
try:
    client.set_twist(100, 0, 600)
    time.sleep(0.5)
finally:
    client.stop()
```

### 4.5 `cancel_all()`：取消所有操作

```python
client.cancel_all()
```

用于统一取消正在进行的操作，包括底盘、机械臂、LED 和蜂鸣器状态。发生异常、
程序退出或需要恢复到安全状态时可以调用。

安全停止示例：

```python
try:
    # 执行控制任务
    pass
finally:
    client.cancel_all()
    client.close()
```

### 4.6 `set_led()`：设置状态灯

```python
client.set_led(mode, period_ms=0)
```

LED 模式定义在 `car.protocol.messages`：

| 常量 | 作用 |
| --- | --- |
| `LED_OFF` | 关闭 |
| `LED_ON` | 常亮 |
| `LED_BLINK` | 闪烁 |

示例：

```python
from car.protocol.messages import LED_BLINK, LED_OFF, LED_ON

client.set_led(LED_ON)
client.set_led(LED_BLINK, 200)
client.set_led(LED_OFF)
```

闪烁模式的 `period_ms` 合法范围为 `50..10000 ms`。

### 4.7 `beep()`：控制蜂鸣器

```python
client.beep(repeat=1, on_ms=100, off_ms=100)
```

参数：

| 参数 | 合法范围 | 含义 |
| --- | --- | --- |
| `repeat` | `1..20` | 鸣叫次数 |
| `on_ms` | `20..5000` | 每次响的时间 |
| `off_ms` | `20..5000` | 两次鸣叫之间的时间 |

响一次：

```python
client.beep(1, 150, 100)
```

连续响三次：

```python
client.beep(3, 100, 100)
```

蜂鸣器由 ESP32 非阻塞控制，播放期间仍可接收其他指令。

### 4.8 `set_arm_joints()`：控制机械臂关节

```python
client.set_arm_joints(joints, duration_ms=800)
```

`joints` 是 `(关节编号, 目标脉宽)` 列表，关节编号为 `0..5`，脉宽单位为
微秒。

移动一个关节：

```python
client.set_arm_joints([(0, 1550)], duration_ms=500)
```

同时移动多个关节：

```python
client.set_arm_joints(
    [
        (0, 1500),
        (1, 1650),
        (2, 1950),
    ],
    duration_ms=800,
)
```

当前项目默认关节范围配置如下，最终以 ESP32 的 `car/config.py` 为准：

| 关节 | 最小值/us | 最大值/us | 复位值/us |
| --- | ---: | ---: | ---: |
| 0 | 500 | 2500 | 1500 |
| 1 | 800 | 1700 | 1700 |
| 2 | 1500 | 2200 | 2000 |
| 3 | 800 | 1500 | 1100 |
| 4 | 900 | 2100 | 1500 |
| 5 | 1100 | 1600 | 1200 |

不得发送超出机械结构实际安全范围的脉宽。软件允许的范围不能替代机械限位检查。

### 4.9 `arm_stop()`：停止机械臂

```python
client.arm_stop()
```

停止当前机械臂动作。机械臂控制代码也应使用 `try/finally`：

```python
try:
    client.set_arm_joints([(0, 1550)], duration_ms=500)
    time.sleep(0.8)
finally:
    client.arm_stop()
```

### 4.10 `start()`：启动后台链路维护

```python
client.start()
```

该方法在树莓派上启动后台心跳和速度续期线程，可重复调用。它不会单独向电机发送
运动速度；实际运动仍由 `set_twist()` 发起。在底盘连续控制场景中，应先启动后台
维护，再发送速度指令。

## 5. 完整示例

下面的示例依次完成连接、状态查询、LED、蜂鸣器、低速底盘测试和安全退出。
运行底盘部分前必须架空车轮。

```python
import time

from car.protocol.messages import LED_BLINK, LED_OFF
from raspberry_pi.config import SerialConfig
from raspberry_pi.esp32 import Esp32Client


config = SerialConfig(
    link_mode="ble",
    ble_device_name="ESP32-Robot-Car",
)

client = Esp32Client(config)
try:
    client.heartbeat()
    print(client.query_status())

    client.set_led(LED_BLINK, 200)
    client.beep(2, 100, 100)
    time.sleep(0.6)
    client.set_led(LED_OFF)

    # 只有车轮已经架空时才运行以下三行。
    client.start()
    client.set_twist(100, 0, 600)
    time.sleep(0.5)
    client.stop()
finally:
    try:
        client.cancel_all()
    finally:
        client.close()
```

## 6. 响应与异常处理

正常控制指令返回 ACK；状态查询返回机器人状态帧。常见树莓派异常：

| 异常 | 含义 |
| --- | --- |
| `LinkTimeoutError` | ESP32 在重试后仍未应答 |
| `LinkError` | 连接、读写或响应校验失败 |
| `ConnectionError` | BLE 已断开或传输层不可用 |

建议捕获链路异常并执行安全停止：

```python
from raspberry_pi.esp32.client import LinkError

try:
    client.set_twist(100, 0, 600)
except LinkError as exc:
    print("通信失败:", exc)
finally:
    try:
        client.cancel_all()
    except Exception:
        pass
    client.close()
```

出现超时时依次检查：

1. ESP32 的 `main.py` 是否仍在运行；
2. ESP32 串口是否有 Traceback；
3. BLE 是否连接到了正确的设备；
4. UART 接线、波特率和共地是否正确；
5. 修改后的 ESP32 文件是否已经上传并复位生效。

## 7. 安全要求

- 首次底盘测试必须架空车轮；
- 机械臂运动前必须清空工作范围；
- 运动指令必须设置有限 TTL，并定期续期；
- 所有控制程序必须在 `finally` 中发送停止或取消命令；
- 不得仅依赖软件停止，调试时应保留可立即断电的手段；
- ESP32 固件更新后必须复位，并先通过无运动心跳测试；
- LED 和蜂鸣器测试通过后，再进行底盘和机械臂测试。
