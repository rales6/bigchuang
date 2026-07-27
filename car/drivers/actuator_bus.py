"""电机/舵机多节点 UART 总线调度器。

ESP32 是唯一总线主机；从节点禁止主动上报，只能在收到发给自己的请求后响应。
调度器保证任意时刻最多一个可靠请求在途，并轮询各节点，从协议层避免多个电机
同时发送导致的电气总线争用。
"""

from core.timebase import ticks_diff, ticks_ms
from protocol.frame import (
    ADDR_ESP32,
    FLAG_ACK_REQUIRED,
    FLAG_RESPONSE,
    FrameParser,
    encode_frame,
)
from protocol.messages import (
    MSG_ACK,
    MSG_ACTUATOR_ARM,
    MSG_ACTUATOR_DRIVE,
    MSG_ACTUATOR_STATUS,
    MSG_ACTUATOR_STATUS_REQUEST,
    MSG_ACTUATOR_STOP,
    decode_actuator_status,
    encode_drive_setpoint,
)


class ActuatorBus:
    def __init__(self, transport, node_addresses, response_timeout_ms=35,
                 max_retries=2, poll_interval_ms=100, guard_ms=2):
        self.transport = transport
        self.node_addresses = tuple(node_addresses)
        self.response_timeout_ms = response_timeout_ms
        self.max_retries = max_retries
        self.poll_interval_ms = poll_interval_ms
        self.guard_ms = guard_ms
        self.parser = FrameParser()
        self.on_status = None
        self._seq = 0
        self._pending = None
        self._reliable_queue = []
        self._priority_queue = []
        self._latest_drive = None
        self._last_tx_ms = ticks_ms()
        self._last_poll_ms = self._last_tx_ms
        self._poll_index = 0
        self.error_count = 0
        self.timeout_count = 0
        self.unexpected_count = 0

    @property
    def healthy(self):
        return self._pending is None or self._pending["retries"] <= self.max_retries

    def send_drive(self, wheel_units, destination=None):
        """合并未发送的速度帧，防止主循环积压过期速度。"""
        dst = destination or self.node_addresses[0]
        self._latest_drive = (dst, encode_drive_setpoint(*wheel_units))

    def send_arm(self, payload, destination=None):
        dst = destination or self.node_addresses[0]
        self._reliable_queue.append((dst, MSG_ACTUATOR_ARM, payload))

    def preempt_arm(self, payload, destination=None):
        """删除尚未执行的机械臂命令，先停止当前动作，再发送新目标。"""
        dst = destination or self.node_addresses[0]
        self._reliable_queue = [
            item for item in self._reliable_queue if item[1] != MSG_ACTUATOR_ARM
        ]
        self._priority_queue.insert(0, (dst, MSG_ACTUATOR_ARM, payload))
        self._priority_queue.insert(0, (dst, MSG_ACTUATOR_STOP, b"\x02"))

    def stop(self, mask=0x03, destination=None):
        """停止帧优先于普通队列；bit0=底盘，bit1=机械臂。"""
        dst = destination or self.node_addresses[0]
        self._latest_drive = None
        self._priority_queue.insert(0, (dst, MSG_ACTUATOR_STOP, bytes((mask,))))

    def poll(self, now=None):
        now = ticks_ms() if now is None else now
        data = self.transport.read()
        if data:
            for frame in self.parser.feed(data):
                self._handle_frame(frame, now)

        self._check_timeout(now)
        if self._pending is not None:
            return
        if ticks_diff(now, self._last_tx_ms) < self.guard_ms:
            return

        if self._priority_queue:
            self._start_reliable(self._priority_queue.pop(0), now)
        elif self._latest_drive is not None:
            dst, payload = self._latest_drive
            self._latest_drive = None
            self._write(dst, MSG_ACTUATOR_DRIVE, payload, flags=0, now=now)
        elif self._reliable_queue:
            self._start_reliable(self._reliable_queue.pop(0), now)
        elif (self.node_addresses and
              ticks_diff(now, self._last_poll_ms) >= self.poll_interval_ms):
            dst = self.node_addresses[self._poll_index]
            self._poll_index = (self._poll_index + 1) % len(self.node_addresses)
            self._last_poll_ms = now
            self._start_reliable((dst, MSG_ACTUATOR_STATUS_REQUEST, b""), now)

    def _start_reliable(self, item, now):
        dst, msg_type, payload = item
        seq = self._next_seq()
        raw = encode_frame(
            msg_type, seq, payload, src=ADDR_ESP32, dst=dst,
            flags=FLAG_ACK_REQUIRED,
        )
        self.transport.write(raw)
        self._last_tx_ms = now
        self._pending = {
            "dst": dst,
            "msg_type": msg_type,
            "seq": seq,
            "raw": raw,
            "sent_ms": now,
            "retries": 0,
        }

    def _write(self, dst, msg_type, payload, flags, now):
        raw = encode_frame(
            msg_type, self._next_seq(), payload, src=ADDR_ESP32, dst=dst, flags=flags
        )
        self.transport.write(raw)
        self._last_tx_ms = now

    def _handle_frame(self, frame, now):
        # 物理端口和逻辑地址双重校验，不能把树莓派数据误当作底板反馈。
        if frame.dst != ADDR_ESP32 or frame.src not in self.node_addresses:
            self.unexpected_count += 1
            return
        pending = self._pending
        if (pending is None or not frame.is_response or
                frame.seq != pending["seq"] or frame.src != pending["dst"]):
            self.unexpected_count += 1
            return

        if frame.msg_type == MSG_ACTUATOR_STATUS:
            try:
                status = decode_actuator_status(frame.payload)
                if self.on_status:
                    self.on_status(frame.src, status, now)
            except ValueError:
                self.error_count += 1
        elif (frame.msg_type != MSG_ACK or len(frame.payload) != 2 or
              frame.payload[0] != pending["msg_type"]):
            self.error_count += 1
        self._pending = None

    def _check_timeout(self, now):
        pending = self._pending
        if pending is None:
            return
        if ticks_diff(now, pending["sent_ms"]) < self.response_timeout_ms:
            return
        if pending["retries"] < self.max_retries:
            self.transport.write(pending["raw"])
            pending["sent_ms"] = now
            pending["retries"] += 1
            self._last_tx_ms = now
        else:
            self.timeout_count += 1
            self.error_count += 1
            self._pending = None

    def _next_seq(self):
        value = self._seq
        self._seq = (self._seq + 1) & 0xFFFF
        return value
