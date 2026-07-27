"""ESP32 执行器固件的唯一板级配置入口。"""

# 树莓派专用全双工链路。不要与执行器底板共用 UART。
PI_UART_ID = 1
PI_UART_BAUDRATE = 230400
PI_UART_TX_PIN = 22
PI_UART_RX_PIN = 21
PI_UART_RX_BUFFER = 1024

# BLE 仅作为树莓派链路的备用通道。使用 Nordic UART Service (NUS)，因此树莓派
# 和 ESP32 仍传输完全相同的 Vehicle Link V2 字节帧。
PI_BLE_ENABLED = True
PI_BLE_NAME = "ESP32-Robot-Car"
PI_BLE_RX_BUFFER = 2048
PI_UNSOLICITED_STATUS_ENABLED = False

# 当前硬件：四个电机和六个舵机均通过 UART2 连接旧执行器底板。
FIRMWARE_BUILD = "2026.07.27-uart-ble-v15-stable-link"
DRIVE_CALIBRATION_PATH = "drive_calibration.json"
ACTUATOR_BACKEND = "legacy_uart_all"
ACTUATOR_UART_ID = 2
ACTUATOR_UART_BAUDRATE = 115200
ACTUATOR_UART_TX_PIN = 17
ACTUATOR_UART_RX_PIN = 16
ACTUATOR_UART_RX_BUFFER = 1024
ACTUATOR_NODE_ADDRESSES = (0x30,)

# 四轮顺序固定为：左前006、右前007、左后008、右后009。
# 右侧协议符号与左侧相反，但 DriveBase 内部仍统一使用“向前为正”。
LEGACY_MOTOR_IDS = (6, 7, 8, 9)
LEGACY_MOTOR_PROTOCOL_SIGNS = (1, -1, 1, -1)
LEGACY_MOTOR_COMMAND_TIME_MS = 0

MAIN_LOOP_MS = 5
PI_LINK_TIMEOUT_MS = 800
COMMAND_MAX_TTL_MS = 2500
# Unsolicited status is diagnostic data, not the command watchdog. Keeping it
# below 2 Hz leaves BLE airtime for acknowledged motion commands and reduces
# notification congestion on MicroPython/BlueZ links.
STATUS_REPORT_MS = 750
ASCII_MOVE_DEFAULT_SPEED_LEVEL = 70
ASCII_MOVE_MIN_DURATION_MS = 450

# 底盘标定参数。首次上车必须根据实测修改，不能直接照搬理论值。
TRACK_WIDTH_MM = 185
MAX_LINEAR_MM_S = 550
MAX_ANGULAR_MRAD_S = 3500
MAX_WHEEL_MM_S = 700
# 原地转向需要克服四轮侧向摩擦。该增益只放大角速度产生的左右差速，
# 不改变已经标定好的直行速度输出。若轮胎打滑明显，可逐步降到 1.5。
ANGULAR_OUTPUT_GAIN = 1.8
ACCEL_MM_S2 = 650
DECEL_MM_S2 = 1000
# 原地建图使用短 TTL 脉冲；提高角速度斜坡，使电机能在 200~350 ms
# 内越过静摩擦死区，而不是靠持续 700 ms 的高速命令启动。
ANGULAR_ACCEL_MRAD_S2 = 15000
CONTROL_PERIOD_MS = 20
MOTOR_REFRESH_MS = 100

# 将期望轮速换算为底板的 -1000..1000 电机控制量。
MOTOR_UNITS_PER_MM_S = 1.35
MIN_EFFECTIVE_MOTOR_UNITS = 105
MAX_MOTOR_UNITS = 1000
MOTOR_DIRECTIONS = (1, 1, 1, 1)
# 每轮独立输出倍率由树莓派运动命令携带，ESP32 固件不保存实车调参值。

# 轮速反馈 PI 参数；反馈超过此时间未更新时自动退化为开环。
WHEEL_KP = 0.20
WHEEL_KI = 0.04
WHEEL_INTEGRAL_LIMIT = 2500
FEEDBACK_STALE_MS = 300

# 六个机械臂关节脉宽限制、复位位置。
ARM_LIMITS_US = (
    (500, 2500),
    (800, 1700),
    (1500, 2200),
    (800, 1500),
    (900, 2100),
    (1100, 1600),
)
ARM_HOME_US = (1500, 1700, 2000, 1100, 1500, 1200)
ARM_DEFAULT_DURATION_MS = 800
ARM_PWM_PINS = (32, 33, 25, 26, 27, 14)
ARM_INITIAL_REACH_MM = 250
ARM_INITIAL_HEIGHT_MM = 80
ARM_INITIAL_YAW_DEG = 0
ARM_ASCII_CONTROL_MODE = "calibrated_delta"
ARM_UP_AXIS_SIGN = -1
ARM_YAW_US_PER_DEG = 10
ARM_FORWARD_JOINT_US_PER_MM = (-2, 3, -2)
ARM_UP_JOINT_US_PER_MM = (-4, -10, 8)
ARM_MIN_REACH_MM = 220
ARM_MAX_REACH_MM = 300
ARM_MIN_HEIGHT_MM = 50
ARM_MAX_HEIGHT_MM = 130
ARM_MAX_YAW_DEG = 80
ARM_CLAW_OPEN_VALUE_MAX = 100
ARM_CLAW_OPEN_US = 1200
ARM_CLAW_CLOSED_US = 1600

LED_PIN = 2
LED_ACTIVE_HIGH = True
BUZZER_PIN = 5
BUZZER_ACTIVE_HIGH = True
