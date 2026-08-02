"""线程安全的 ESP32 Vehicle Link V2 串口客户端。

协议编解码直接复用 ``car.protocol``，保证树莓派和 ESP32 使用同一份消息定义。
速度命令会在后台刷新；若树莓派进程卡死，ESP32 自身的 TTL 和失联保护仍会停车。
"""

import threading
import time

from car.protocol.frame import (
    ADDR_ESP32,
    ADDR_RASPBERRY_PI,
    FLAG_ACK_REQUIRED,
    FLAG_ERROR,
    FrameParser,
    encode_frame,
)
from car.protocol.messages import (
    CANCEL_ALL,
    MSG_ACK,
    MSG_BEEP,
    MSG_CANCEL,
    MSG_DRIVE_CALIBRATION,
    MSG_ERROR,
    MSG_HEARTBEAT,
    MSG_QUERY_STATUS,
    MSG_QUERY_DRIVE_CALIBRATION,
    MSG_RESET_DRIVE_CALIBRATION,
    MSG_ROBOT_STATUS,
    MSG_SET_LED,
    MSG_SET_DRIVE_CALIBRATION,
    MSG_SET_ARM_JOINTS,
    MSG_ARM_STOP,
    MSG_SET_BALANCED_TWIST,
    MSG_SET_TWIST,
    MSG_STOP,
    decode_robot_status,
    decode_drive_calibration,
    encode_beep,
    encode_led,
    encode_arm_joints,
    encode_balanced_twist,
    encode_twist,
    encode_drive_calibration,
)

from raspberry_pi.config import SerialConfig


class LinkError(RuntimeError):
    """ESP32 拒绝请求或返回了不符合协议的响应。"""


class LinkTimeoutError(LinkError):
    """在重试后仍未收到 ESP32 响应。"""


class Esp32Client:
    """Vehicle Link V2 高层客户端。

    ``serial_instance`` 用于单元测试或接入自定义串口对象；正常运行时只传
    ``SerialConfig`` 即可。
    """

    def __init__(self, config=None, serial_instance=None):
        self.config = config or SerialConfig()
        if serial_instance is None:
            from raspberry_pi.transport import create_transport
            serial_instance = create_transport(self.config)
        self.serial = serial_instance
        self.parser = FrameParser()
        self._sequence = 0
        self._request_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._running = False
        self._maintenance_thread = None
        self._active_drive_request = None
        self.last_status = None
        self.last_link_error = None

    def start(self):
        """启动心跳和速度命令刷新线程，可重复调用。"""
        if self._running:
            return
        self._running = True
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            name="esp32-link-maintenance",
            daemon=True,
        )
        self._maintenance_thread.start()

    def request(self, msg_type, payload=b"", timeout_s=None, retries=None):
        timeout_s = self.config.request_timeout_s if timeout_s is None else timeout_s
        retries = self.config.retries if retries is None else retries
        with self._request_lock:
            sequence = self._next_sequence()
            packet = encode_frame(
                msg_type,
                sequence,
                payload,
                src=ADDR_RASPBERRY_PI,
                dst=ADDR_ESP32,
                flags=FLAG_ACK_REQUIRED,
            )
            for attempt in range(retries + 1):
                try:
                    self.serial.write(packet)
                except Exception as exc:
                    self.last_link_error = exc
                    if attempt < retries and self._switch_to_fallback():
                        continue
                    raise LinkError("通信链路写入失败: {}".format(exc)) from exc
                deadline = time.monotonic() + timeout_s
                while time.monotonic() < deadline:
                    waiting = getattr(self.serial, "in_waiting", 0)
                    try:
                        data = self.serial.read(waiting or 1)
                    except Exception as exc:
                        self.last_link_error = exc
                        if attempt < retries and self._switch_to_fallback():
                            break
                        raise LinkError("通信链路读取失败: {}".format(exc)) from exc
                    for frame in self.parser.feed(data):
                        if frame.src != ADDR_ESP32 or frame.dst != ADDR_RASPBERRY_PI:
                            continue
                        if frame.msg_type == MSG_ROBOT_STATUS:
                            self.last_status = decode_robot_status(frame.payload)
                        if frame.seq != sequence or not frame.is_response:
                            continue
                        self._validate_response(frame, msg_type)
                        return frame
                if attempt == retries:
                    raise LinkTimeoutError(
                        "ESP32 未应答：type=0x{:02X}, seq={}, attempts={}".format(
                            msg_type, sequence, retries + 1
                        )
                    )
                self._switch_to_fallback()
        raise AssertionError("unreachable")

    def send_without_response(self, msg_type, payload=b""):
        """Send a best-effort frame without waiting for an application ACK.

        This is reserved for refreshing an already acknowledged motion
        command.  A new speed or direction must still use ``request``.
        """
        with self._request_lock:
            sequence = self._next_sequence()
            packet = encode_frame(
                msg_type,
                sequence,
                payload,
                src=ADDR_RASPBERRY_PI,
                dst=ADDR_ESP32,
                flags=0,
            )
            try:
                self.serial.write(packet)
            except Exception as exc:
                self.last_link_error = exc
                raise LinkError(
                    "通信链路无应答写入失败: {}".format(exc)
                ) from exc

    def heartbeat(self):
        return self.request(MSG_HEARTBEAT)

    def query_status(self):
        frame = self.request(MSG_QUERY_STATUS)
        if frame.msg_type != MSG_ROBOT_STATUS:
            raise LinkError("状态查询收到意外响应 0x{:02X}".format(frame.msg_type))
        self.last_status = decode_robot_status(frame.payload)
        return self.last_status

    def set_drive_calibration(self, trim_intercept, trim_slope_per_mm_s):
        return self.request(
            MSG_SET_DRIVE_CALIBRATION,
            encode_drive_calibration(
                trim_intercept,
                trim_slope_per_mm_s,
            ),
        )

    def query_drive_calibration(self):
        frame = self.request(MSG_QUERY_DRIVE_CALIBRATION)
        if frame.msg_type != MSG_DRIVE_CALIBRATION:
            raise LinkError(
                "底盘标定查询收到意外响应 0x{:02X}".format(frame.msg_type)
            )
        return decode_drive_calibration(frame.payload)

    def reset_drive_calibration(self):
        return self.request(MSG_RESET_DRIVE_CALIBRATION)

    def set_twist(self, linear_mm_s, angular_mrad_s, ttl_ms=600):
        msg_type, payload = self._encode_twist_request(
            linear_mm_s,
            angular_mrad_s,
            ttl_ms,
        )
        # A changed motion command remains acknowledged.  Only an identical
        # periodic refresh is allowed to use the non-blocking path below.
        frame = self.request(msg_type, payload)
        with self._state_lock:
            self._active_drive_request = (msg_type, payload)
        return frame

    def refresh_twist(self, linear_mm_s, angular_mrad_s, ttl_ms=600):
        """Refresh an unchanged, previously acknowledged motion command."""
        msg_type, payload = self._encode_twist_request(
            linear_mm_s,
            angular_mrad_s,
            ttl_ms,
        )
        self.send_without_response(msg_type, payload)
        with self._state_lock:
            self._active_drive_request = (msg_type, payload)

    def _encode_twist_request(
        self,
        linear_mm_s,
        angular_mrad_s,
        ttl_ms,
    ):
        command = (int(linear_mm_s), int(angular_mrad_s), int(ttl_ms))
        gains = tuple(float(value) for value in self.config.wheel_output_gains)
        if gains == (1.0, 1.0, 1.0, 1.0):
            msg_type = MSG_SET_TWIST
            payload = encode_twist(*command)
        else:
            msg_type = MSG_SET_BALANCED_TWIST
            payload = encode_balanced_twist(
                command[0],
                command[1],
                command[2],
                gains,
            )
        return msg_type, payload

    def stop(self):
        with self._state_lock:
            self._active_drive_request = None
        return self.request(MSG_STOP)

    def cancel_all(self):
        with self._state_lock:
            self._active_drive_request = None
        return self.request(MSG_CANCEL, bytes((CANCEL_ALL,)))

    def set_led(self, mode, period_ms=0):
        return self.request(MSG_SET_LED, encode_led(mode, period_ms))

    def beep(self, repeat=1, on_ms=100, off_ms=100):
        return self.request(MSG_BEEP, encode_beep(repeat, on_ms, off_ms))

    def set_arm_joints(self, joints, duration_ms=800):
        """设置一个或多个舵机脉宽，如 ``[(0, 1500), (1, 1700)]``。"""
        return self.request(
            MSG_SET_ARM_JOINTS, encode_arm_joints(joints, int(duration_ms))
        )

    def arm_stop(self):
        return self.request(MSG_ARM_STOP)

    @property
    def active_transport(self):
        return getattr(self.serial, "name", "uart")

    def close(self, send_cancel=True):
        """停止后台线程、请求全车取消并关闭串口。"""
        self._running = False
        thread = self._maintenance_thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        if send_cancel:
            try:
                self.cancel_all()
            except Exception:
                pass
        close = getattr(self.serial, "close", None)
        if close:
            close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()

    def _maintenance_loop(self):
        while self._running:
            started = time.monotonic()
            try:
                with self._state_lock:
                    drive_request = self._active_drive_request
                if drive_request is None:
                    self.heartbeat()
                else:
                    msg_type, payload = drive_request
                    self.request(msg_type, payload, retries=1)
                self.last_link_error = None
            except Exception as exc:  # 状态供主线程展示，后台不能悄悄退出
                self.last_link_error = exc
            elapsed = time.monotonic() - started
            time.sleep(max(0.0, self.config.keepalive_period_s - elapsed))

    def _validate_response(self, frame, requested_type):
        if frame.msg_type == MSG_ERROR or frame.flags & FLAG_ERROR:
            error_code = frame.payload[1] if len(frame.payload) >= 2 else -1
            raise LinkError(
                "ESP32 拒绝 type=0x{:02X}，错误码 {}".format(requested_type, error_code)
            )
        if requested_type == MSG_QUERY_STATUS:
            expected_type = MSG_ROBOT_STATUS
        elif requested_type == MSG_QUERY_DRIVE_CALIBRATION:
            expected_type = MSG_DRIVE_CALIBRATION
        else:
            expected_type = MSG_ACK
        if frame.msg_type != expected_type:
            raise LinkError(
                "请求 0x{:02X} 收到意外响应 0x{:02X}".format(
                    requested_type, frame.msg_type
                )
            )
        if frame.msg_type == MSG_ACK:
            if len(frame.payload) != 2 or frame.payload[0] != requested_type:
                raise LinkError("ACK 与请求类型不匹配")
            if frame.payload[1] != 0:
                raise LinkError("ESP32 ACK 状态码 {}".format(frame.payload[1]))

    def _next_sequence(self):
        sequence = self._sequence
        self._sequence = (sequence + 1) & 0xFFFF
        return sequence

    def _switch_to_fallback(self):
        switch = getattr(self.serial, "switch_to_fallback", None)
        if not switch:
            return False
        try:
            return bool(switch())
        except Exception as exc:
            self.last_link_error = exc
            return False
