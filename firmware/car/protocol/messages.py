"""Vehicle Link V2 的固定二进制消息定义。

自然语言、图像和地图只在树莓派侧处理；ESP32 接收有界、确定性的数值命令，
避免在实时控制器中解析 JSON 或自然语言。
"""

try:
    import ustruct as struct
except ImportError:
    import struct


MSG_HEARTBEAT = 0x01
MSG_ACK = 0x02
MSG_ERROR = 0x03

MSG_SET_TWIST = 0x10
MSG_STOP = 0x11
MSG_CANCEL = 0x12
MSG_SET_BALANCED_TWIST = 0x13
MSG_SET_ARM_JOINTS = 0x20
MSG_ARM_STOP = 0x21
MSG_SET_LED = 0x22
MSG_BEEP = 0x23
MSG_QUERY_STATUS = 0x30
MSG_ROBOT_STATUS = 0x31
MSG_SET_DRIVE_CALIBRATION = 0x32
MSG_QUERY_DRIVE_CALIBRATION = 0x33
MSG_DRIVE_CALIBRATION = 0x34
MSG_RESET_DRIVE_CALIBRATION = 0x35

# ESP32 与执行器底板之间也使用同一种帧，只是消息类型和地址不同。
MSG_ACTUATOR_DRIVE = 0x50
MSG_ACTUATOR_ARM = 0x51
MSG_ACTUATOR_STOP = 0x52
MSG_ACTUATOR_STATUS_REQUEST = 0x53
MSG_ACTUATOR_STATUS = 0x54

STATUS_OK = 0
ERR_BAD_SOURCE = 1
ERR_BAD_DESTINATION = 2
ERR_BAD_LENGTH = 3
ERR_BAD_VALUE = 4
ERR_UNSUPPORTED = 5
ERR_BUSY = 6
ERR_INTERNAL = 7

CANCEL_DRIVE = 0x01
CANCEL_ARM = 0x02
CANCEL_BUZZER = 0x04
CANCEL_LED = 0x08
CANCEL_ALL = 0x0F

LED_OFF = 0
LED_ON = 1
LED_BLINK = 2


def encode_twist(linear_mm_s, angular_mrad_s, ttl_ms):
    return struct.pack("<hhH", linear_mm_s, angular_mrad_s, ttl_ms)


def decode_twist(payload):
    _require_length(payload, 6)
    return struct.unpack("<hhH", payload)


def encode_balanced_twist(
    linear_mm_s,
    angular_mrad_s,
    ttl_ms,
    wheel_output_gains,
):
    """Encode a twist plus four Pi-owned per-wheel output multipliers."""
    gains = tuple(float(value) for value in wheel_output_gains)
    if len(gains) != 4 or any(
        value < 0.50 or value > 1.50 for value in gains
    ):
        raise ValueError(
            "wheel output gains must contain four values in 0.50..1.50"
        )
    scaled = tuple(int(value * 1000.0 + 0.5) for value in gains)
    return struct.pack(
        "<hhHHHHH",
        linear_mm_s,
        angular_mrad_s,
        ttl_ms,
        scaled[0],
        scaled[1],
        scaled[2],
        scaled[3],
    )


def decode_balanced_twist(payload):
    _require_length(payload, 14)
    values = struct.unpack("<hhHHHHH", payload)
    gains = tuple(value / 1000.0 for value in values[3:7])
    if any(value < 0.50 or value > 1.50 for value in gains):
        raise ValueError("wheel output gain is outside 0.50..1.50")
    return values[0], values[1], values[2], gains


def encode_led(mode, period_ms=0):
    return struct.pack("<BH", mode, period_ms)


def decode_led(payload):
    _require_length(payload, 3)
    return struct.unpack("<BH", payload)


def encode_beep(repeat, on_ms, off_ms):
    return struct.pack("<BHH", repeat, on_ms, off_ms)


def decode_beep(payload):
    _require_length(payload, 5)
    return struct.unpack("<BHH", payload)


def encode_arm_joints(joints, duration_ms):
    """joints 为 ``[(joint_id, pulse_us), ...]``。"""
    if not 1 <= len(joints) <= 6:
        raise ValueError("joint count must be in range 1..6")
    payload = bytearray((len(joints),))
    for joint_id, pulse_us in joints:
        payload.extend(struct.pack("<BHH", joint_id, pulse_us, duration_ms))
    return bytes(payload)


def decode_arm_joints(payload):
    if not payload:
        raise ValueError("arm payload is empty")
    count = payload[0]
    _require_length(payload, 1 + count * 5)
    joints = []
    offset = 1
    for _ in range(count):
        joint_id, pulse_us, duration_ms = struct.unpack("<BHH", payload[offset:offset + 5])
        joints.append((joint_id, pulse_us, duration_ms))
        offset += 5
    return joints


def encode_drive_setpoint(front_left, front_right, rear_left, rear_right):
    return struct.pack("<hhhh", front_left, front_right, rear_left, rear_right)


def decode_drive_setpoint(payload):
    _require_length(payload, 8)
    return struct.unpack("<hhhh", payload)


def decode_actuator_status(payload):
    """返回轮速、六关节位置、电池电压和底板故障位。"""
    _require_length(payload, 24)
    values = struct.unpack("<hhhh6HHH", payload)
    return {
        "wheels": values[0:4],
        "joints": values[4:10],
        "battery_mv": values[10],
        "fault_flags": values[11],
    }


def encode_actuator_status(wheels, joints, battery_mv, fault_flags):
    return struct.pack(
        "<hhhh6HHH",
        wheels[0], wheels[1], wheels[2], wheels[3],
        joints[0], joints[1], joints[2], joints[3], joints[4], joints[5],
        battery_mv, fault_flags,
    )


def encode_ack(original_type, status=STATUS_OK):
    return bytes((original_type & 0xFF, status & 0xFF))


def encode_error(original_type, error_code):
    return bytes((original_type & 0xFF, error_code & 0xFF))


def encode_robot_status(status):
    """编码 ESP32 汇总状态；字段布局详见 docs/protocol_v2.md。"""
    wheels = status["wheel_feedback"]
    joints = status["joint_positions"]
    return struct.pack(
        "<IHhhhhhhhh6HHH",
        status["uptime_ms"] & 0xFFFFFFFF,
        status["flags"] & 0xFFFF,
        status["linear_mm_s"], status["angular_mrad_s"],
        status["left_output"], status["right_output"],
        wheels[0], wheels[1], wheels[2], wheels[3],
        joints[0], joints[1], joints[2], joints[3], joints[4], joints[5],
        status["battery_mv"], status["bus_errors"],
    )


def decode_robot_status(payload):
    _require_length(payload, 38)
    values = struct.unpack("<IHhhhhhhhh6HHH", payload)
    return {
        "uptime_ms": values[0],
        "flags": values[1],
        "linear_mm_s": values[2],
        "angular_mrad_s": values[3],
        "left_output": values[4],
        "right_output": values[5],
        "wheel_feedback": values[6:10],
        "joint_positions": values[10:16],
        "battery_mv": values[16],
        "bus_errors": values[17],
    }


def encode_drive_calibration(trim_intercept, trim_slope_per_mm_s):
    """编码速度相关左右轮修正：left=1-trim，right=1+trim。"""
    return struct.pack(
        "<Bff",
        1,
        float(trim_intercept),
        float(trim_slope_per_mm_s),
    )


def decode_drive_calibration(payload):
    _require_length(payload, 9)
    version, intercept, slope = struct.unpack("<Bff", payload)
    if version != 1:
        raise ValueError("unsupported drive calibration version")
    return {
        "version": version,
        "trim_intercept": float(intercept),
        "trim_slope_per_mm_s": float(slope),
    }


def _require_length(payload, expected):
    if len(payload) != expected:
        raise ValueError("payload length must be {}".format(expected))
