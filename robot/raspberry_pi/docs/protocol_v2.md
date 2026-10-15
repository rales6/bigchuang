# Vehicle Link V2 协议

## 帧格式

所有多字节整数均为小端序。CRC 使用 CRC-16/CCITT-FALSE，初值 `0xFFFF`、
多项式 `0x1021`，覆盖 `VER` 至 Payload。

| 字段 | 字节数 | 说明 |
|---|---:|---|
| SOF | 2 | 固定 `A5 5A` |
| VER | 1 | 固定 `02` |
| FLAGS | 1 | bit0 要求应答；bit1 响应；bit2 错误 |
| SRC | 1 | 源节点地址 |
| DST | 1 | 目标节点地址 |
| TYPE | 1 | 消息类型 |
| SEQ | 2 | 请求序号；重试必须保持相同序号和负载 |
| LEN | 2 | Payload 长度，最大 256 |
| Payload | LEN | 按 TYPE 定义的固定二进制数据 |
| CRC16 | 2 | 帧校验 |

节点地址：ESP32 `0x01`、树莓派 `0x10`、K210 预留 `0x20`、聚合底板
`0x30`、四轮独立节点 `0x31`–`0x34`、机械臂节点 `0x40`、广播 `0xFF`。

## 树莓派—ESP32 消息

| TYPE | 名称 | Payload |
|---:|---|---|
| `0x01` | HEARTBEAT | 空或 `uptime:u32` |
| `0x02` | ACK | `original_type:u8, status:u8` |
| `0x03` | ERROR | `original_type:u8, error_code:u8` |
| `0x10` | SET_TWIST | `linear_mm_s:i16, angular_mrad_s:i16, ttl_ms:u16` |
| `0x11` | STOP | 空；立即停止底盘 |
| `0x12` | CANCEL | `mask:u8` |
| `0x13` | SET_BALANCED_TWIST | `linear:i16, angular:i16, ttl:u16, LF/RF/LR/RR_gain_permille:4×u16` |
| `0x20` | SET_ARM_JOINTS | `count:u8 + count × (id:u8, pulse_us:u16, duration_ms:u16)` |
| `0x21` | ARM_STOP | 空 |
| `0x22` | SET_LED | `mode:u8, period_ms:u16` |
| `0x23` | BEEP | `repeat:u8, on_ms:u16, off_ms:u16` |
| `0x30` | QUERY_STATUS | 空 |
| `0x31` | ROBOT_STATUS | 见下表 |
| `0x32` | SET_DRIVE_CALIBRATION | `version:u8, trim_intercept:f32, trim_slope_per_mm_s:f32` |
| `0x33` | QUERY_DRIVE_CALIBRATION | 空 |
| `0x34` | DRIVE_CALIBRATION | 与 `SET_DRIVE_CALIBRATION` 相同 |
| `0x35` | RESET_DRIVE_CALIBRATION | 空 |

底盘补偿使用 `trim = intercept + slope × abs(wheel_speed_mm_s)`，运行时
`left *= 1-trim`、`right *= 1+trim`。ESP32只在底盘已经停车时接受写入，并将
参数原子保存到 `drive_calibration.json`；任何速度下的绝对补偿都被限制在15%。

取消掩码：底盘 `0x01`、机械臂 `0x02`、蜂鸣器 `0x04`、灯 `0x08`、全部
`0x0F`。LED mode：关闭 `0`、常亮 `1`、闪烁 `2`。

`SET_TWIST` 中正线速度表示前进，正角速度表示逆时针左转。TTL 到期后即使树莓
派进程还连接，ESP32 也会把目标速度降到零；测试工具每 250 ms 更新一次命令。
`SET_BALANCED_TWIST` 的四轮倍率由树莓派决定，允许范围为 `500..1500`
（即 `0.50..1.50`）；ESP32 只负责校验和执行，不在固件配置中保存实车倍率。
收到普通 `SET_TWIST` 时四轮倍率恢复为 `1.00`。

## ROBOT_STATUS（38 字节）

| 字段 | 类型 |
|---|---|
| uptime_ms | u32 |
| flags | u16 |
| linear_mm_s, angular_mrad_s | i16, i16 |
| left_output, right_output | i16, i16 |
| wheel_feedback[4] | 4 × i16，单位 mm/s |
| joint_positions[6] | 6 × u16，单位 μs |
| battery_mv | u16 |
| bus_errors | u16 |

状态位：底盘运动 `0x0001`、机械臂运动 `0x0002`、闭环轮速有效 `0x0004`、
Pi 链路在线 `0x0008`、执行器故障 `0x0010`、失联保护 `0x0020`、LED 亮
`0x0040`、蜂鸣器工作 `0x0080`。

## 可选的 ESP32—V2 执行器底板消息

以下消息仅在 `ACTUATOR_BACKEND="vehicle_link_v2"` 时启用。当前现有硬件使用
`legacy_uart_all`：电机转换成 `#006`–`#009` ASCII，舵机转换成 `#000`–`#005`
ASCII，并在同一 UART 调度器中串行发送。

| TYPE | 名称 | Payload |
|---:|---|---|
| `0x50` | ACTUATOR_DRIVE | 四轮 `i16` 控制量，顺序 FL/FR/RL/RR |
| `0x51` | ACTUATOR_ARM | 与 SET_ARM_JOINTS 相同 |
| `0x52` | ACTUATOR_STOP | mask：bit0 底盘、bit1 机械臂 |
| `0x53` | ACTUATOR_STATUS_REQUEST | 空 |
| `0x54` | ACTUATOR_STATUS | `4×wheel:i16, 6×joint:u16, battery_mv:u16, faults:u16` |

底板按各电机自身的原始符号执行和回报；ESP32 使用同一组
`MOTOR_DIRECTIONS` 同时修正控制量和反馈，使控制层始终采用“小车向前为正”的
坐标系。不得只修正输出而不修正反馈，否则 PI 控制会形成正反馈。

## 应答、重试与抢占

- 请求方超时后重发完全相同的 SEQ、TYPE 和 Payload。
- 接收方缓存最近 8 个请求，应答丢失时重放响应而不重复执行动作。
- ACK 表示“已接受”，不等于机械动作已完成；完成情况从 ROBOT_STATUS 判断。
- 新的底盘速度覆盖旧速度，不排队。
- 上一机械臂动作仍在运行时，新目标先发送 ARM_STOP，再发送新目标；上一动作已
  正常完成时直接发送目标，避免对旧底板重复发送全舵机停止命令。
- CANCEL 可在动作中途停止指定子系统。
- CRC 错误帧直接丢弃；发送端因收不到 ACK 而重试。
