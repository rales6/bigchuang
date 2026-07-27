# Raspberry Pi ASCII 指令集

这是树莓派到 ESP32 的临时可读联调协议。正式长期通信仍建议使用 `docs/protocol_v2.md` 中带地址、序号和 CRC 的二进制 Vehicle Link V2；ASCII 指令适合现在没有树莓派或客户端还没写完时，用 Thonny/串口助手快速验证。

## 通用格式

```text
#<TYPE><FIELD><VALUE><FIELD><VALUE>...!
```

ESP32 收到后返回 `OK:...` 或 `ERR:...`。在 Thonny 中可以直接执行：

```python
import thonny_manual_test as test
test.send("#ARMF020U030L000C020!")
test.send("#MOVF300L000S080!")
test.send("#LEDB500!")
test.send("#BEEP003!")
test.send("#STOP!")
```

## 指令表

| 指令 | 含义 | 单位 | 示例 |
|---|---|---|---|
| `#ARMFxxxUyyyLzzzCnnn!` | 机械臂末端相对前伸、抬升、左旋，并设置夹爪开口 | mm, mm, degree, 0-100 | `#ARMF100U050L000C020!` |
| `#ARMBxxxDyyyRzzzCnnn!` | 机械臂末端相对后退、下降、右旋，并设置夹爪开口 | mm, mm, degree, 0-100 | `#ARMB030D020R015C080!` |
| `#MOVFxxxLzzzSnnn!` | 小车前进并左转，`S` 为速度档位 | mm, degree, 1-100 | `#MOVF300L000S080!` |
| `#MOVBxxxRzzzSnnn!` | 小车后退并右转，`S` 为速度档位 | mm, degree, 1-100 | `#MOVB100R090S060!` |
| `#LEDON!` / `#LEDOFF!` | 状态灯常亮/关闭 | - | `#LEDON!` |
| `#LEDBxxx!` | 状态灯闪烁周期 | ms | `#LEDB500!` |
| `#BEEPn!` | 蜂鸣器短响 n 次，n 限制在 1-9 | count | `#BEEP003!` |
| `#PING!` | 通信测试 | - | `#PING!` |
| `#STOP!` | 取消底盘、机械臂、蜂鸣器和 LED | - | `#STOP!` |

## 机械臂映射

`#ARM...!` 不会直接透传给底板。当前实车优先使用旧代码标定过的舵机增量映射：`U` 会让 1 号、2 号脉宽减小，3 号脉宽增大；`D` 反向；`C` 映射到 5 号夹爪舵机。最后逐条发送旧底板支持的 `#000PxxxxTxxxx!` 格式。

当前保守工作区：

| 参数 | 范围 |
|---|---:|
| 前后伸缩 reach | 220-300 mm |
| 高度 height | 50-130 mm |
| 左右旋转 yaw | -80 到 +80 degree |
| 夹爪开口 C | 0 闭合，100 最大张开 |

默认初始舵机值沿用旧工程：`(1500, 1700, 2000, 1100, 1500, 1200)`。夹爪 `C000` 对应约 `#005P1600...!`，`C100` 对应约 `#005P1200...!`。旧底板对 `{...}` 分组帧兼容性不稳定，所以多舵机动作现在改为逐条写入同一 UART。

## 底盘速度与中断

`MOV` 指令的 `S` 是速度档位，不是时间。ESP32 会根据距离、角度和速度档位自动估算 TTL 时间，并把目标速度交给底盘平滑控制器。不带 `S` 时默认 `S070`。

示例：

```text
#MOVF300L000S080!  前进 300 mm，80 档速度
#MOVB200L000S060!  后退 200 mm，60 档速度
#MOVF000L090S050!  原地左转 90 度，50 档速度
#STOP!             立即中断当前底盘/机械臂/蜂鸣器/灯动作
```

新 `MOV` 会覆盖旧的底盘目标，`#STOP!` 会立即取消。树莓派上层可以给任务加时间戳或任务 id，用于丢弃过期自然语言任务；但 ESP32 端不需要依赖时间戳才能中断。
