"""现有电机/舵机底板的 ASCII UART 适配器。

树莓派侧仍使用 Vehicle Link V2；只有 ESP32 到现有底板这一段转换为底板实际
支持的 ``#nnnP....T....!`` 指令。该底板协议没有 ACK/状态响应，因此此适配器
只能确认字节已经写入 UART，不能声称执行器已经物理动作。
"""

from core.timebase import ticks_ms
from protocol.messages import decode_arm_joints


class LegacyMotorBus:
    def __init__(self, transport, motor_ids, protocol_signs, command_time_ms=0):
        if len(motor_ids) != 4 or len(protocol_signs) != 4:
            raise ValueError("legacy motor mapping must contain four entries")
        self.transport = transport
        self.motor_ids = tuple(motor_ids)
        self.protocol_signs = tuple(protocol_signs)
        self.command_time_ms = command_time_ms
        self.on_status = None
        self._latest_drive = None
        self._priority_stop = False
        self._release_drive_pending = False
        self._latest_arm = []
        self._arm_stop_pending = False
        self.error_count = 0
        self.timeout_count = 0
        self.unexpected_count = 0
        self.write_count = 0
        self.drive_write_count = 0
        self.nonzero_drive_write_count = 0
        self.stop_write_count = 0
        self.arm_write_count = 0
        self.arm_stop_write_count = 0
        self.received_bytes = 0
        self.last_write_ms = ticks_ms()
        self.last_write = b""
        self.last_drive_frame = b""
        self.last_nonzero_drive_frame = b""
        self.last_stop_frame = b""
        self.last_arm_frame = b""
        self.last_arm_stop_frame = b""

    @property
    def supports_feedback(self):
        return False

    def send_drive(self, wheel_units, destination=None):
        """只保留最新速度，旧速度不会在 UART 队列中累积。"""
        if len(wheel_units) != 4:
            raise ValueError("wheel command must contain four values")
        self._latest_drive = tuple(_clamp(int(value), -1000, 1000)
                                   for value in wheel_units)

    def stop(self, mask=0x03, destination=None):
        if mask & 0x01:
            self._latest_drive = None
            self._priority_stop = True
            self._release_drive_pending = True
        if mask & 0x02:
            self._latest_arm = []
            self._arm_stop_pending = True

    def send_arm(self, payload, destination=None):
        self._latest_arm = self._encode_arm(payload)

    def preempt_arm(self, payload, destination=None):
        # #255PDST! 是原底板支持的全部舵机停止命令；下一轮再发送新目标。
        self._arm_stop_pending = True
        self._latest_arm = self._encode_arm(payload)

    def poll(self, now=None):
        now = ticks_ms() if now is None else now
        # 读取并计数底板可能产生的旧格式数据，避免 UART RX 缓冲区持续堆积。
        incoming = self.transport.read()
        if incoming:
            self.received_bytes += len(incoming)

        if self._priority_stop:
            self._priority_stop = False
            self._write(self._encode_drive((0, 0, 0, 0), 100), now, "motor_stop")
        elif self._release_drive_pending:
            self._release_drive_pending = False
            self._write(self._encode_drive_release(), now, "motor_release")
        elif self._arm_stop_pending:
            self._arm_stop_pending = False
            self._write(b"#255PDST!", now, "arm_stop")
        elif self._latest_arm:
            raw = self._latest_arm.pop(0)
            self._write(raw, now, "arm")
        elif self._latest_drive is not None:
            values = self._latest_drive
            self._latest_drive = None
            kind = "drive_nonzero" if any(values) else "drive_zero"
            self._write(self._encode_drive(values, self.command_time_ms), now, kind)

    def _encode_drive(self, wheel_units, duration_ms):
        parts = []
        for index in range(4):
            pulse = 1500 + self.protocol_signs[index] * wheel_units[index]
            pulse = _clamp(pulse, 500, 2500)
            parts.append("#{:03d}P{:04d}T{:04d}!".format(
                self.motor_ids[index], pulse, duration_ms
            ))
        return "".join(parts).encode("ascii")

    def _encode_drive_release(self):
        """Disable only the four wheel channels after centring them."""
        return "".join(
            "#{:03d}PDST!".format(motor_id)
            for motor_id in self.motor_ids
        ).encode("ascii")

    @staticmethod
    def _encode_arm(payload):
        joints = decode_arm_joints(payload)
        return [
            "#{:03d}P{:04d}T{:04d}!".format(
                joint_id, pulse_us, duration_ms
            ).encode("ascii")
            for joint_id, pulse_us, duration_ms in joints
        ]

    def _write(self, raw, now, kind):
        try:
            written = self.transport.write(raw)
            if written is not None and written != len(raw):
                self.error_count += 1
            else:
                self.write_count += 1
                if kind in ("drive_nonzero", "drive_zero"):
                    self.drive_write_count += 1
                    self.last_drive_frame = raw
                    if kind == "drive_nonzero":
                        self.nonzero_drive_write_count += 1
                        self.last_nonzero_drive_frame = raw
                elif kind in ("motor_stop", "motor_release"):
                    self.stop_write_count += 1
                    self.last_stop_frame = raw
                elif kind == "arm":
                    self.arm_write_count += 1
                    self.last_arm_frame = raw
                elif kind == "arm_stop":
                    self.arm_stop_write_count += 1
                    self.last_arm_stop_frame = raw
            self.last_write_ms = now
            self.last_write = raw
        except Exception:
            self.error_count += 1
            raise


def _clamp(value, minimum, maximum):
    return min(max(value, minimum), maximum)
