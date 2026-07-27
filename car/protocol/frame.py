"""Vehicle Link V2 帧编解码。

格式（所有多字节整数为小端序）：
    A5 5A | VER | FLAGS | SRC | DST | TYPE | SEQ:u16 | LEN:u16 |
    PAYLOAD | CRC16:u16

长度字段和 CRC 使解析器能从拆包、粘包或线路噪声中恢复；SRC/DST 配合独立
物理 UART，可明确区分树莓派和执行器总线的数据来源。
"""


SOF = b"\xA5\x5A"
VERSION = 2
MAX_PAYLOAD = 256

FLAG_ACK_REQUIRED = 0x01
FLAG_RESPONSE = 0x02
FLAG_ERROR = 0x04

ADDR_ESP32 = 0x01
ADDR_RASPBERRY_PI = 0x10
ADDR_K210 = 0x20       # 预留地址；当前 K210 不参与运行
ADDR_BASEBOARD = 0x30
ADDR_MOTOR_FL = 0x31
ADDR_MOTOR_FR = 0x32
ADDR_MOTOR_RL = 0x33
ADDR_MOTOR_RR = 0x34
ADDR_ARM = 0x40
ADDR_BROADCAST = 0xFF


class ProtocolError(Exception):
    pass


class Frame:
    def __init__(self, flags, src, dst, msg_type, seq, payload=b"", version=VERSION):
        self.version = version
        self.flags = flags
        self.src = src
        self.dst = dst
        self.msg_type = msg_type
        self.seq = seq
        self.payload = payload

    @property
    def ack_required(self):
        return bool(self.flags & FLAG_ACK_REQUIRED)

    @property
    def is_response(self):
        return bool(self.flags & FLAG_RESPONSE)


def crc16_ccitt(data):
    """CRC-16/CCITT-FALSE：初值 0xFFFF，多项式 0x1021。"""
    value = 0xFFFF
    for byte in data:
        value ^= byte << 8
        for _ in range(8):
            if value & 0x8000:
                value = ((value << 1) ^ 0x1021) & 0xFFFF
            else:
                value = (value << 1) & 0xFFFF
    return value


def encode_frame(msg_type, seq, payload=b"", src=ADDR_ESP32,
                 dst=ADDR_RASPBERRY_PI, flags=0):
    if payload is None:
        payload = b""
    if isinstance(payload, bytearray):
        payload = bytes(payload)
    if not isinstance(payload, bytes):
        raise ProtocolError("payload must be bytes")
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError("payload exceeds {} bytes".format(MAX_PAYLOAD))
    if not 0 <= seq <= 0xFFFF:
        raise ProtocolError("sequence must be in range 0..65535")

    length = len(payload)
    body = bytes((
        VERSION,
        flags & 0xFF,
        src & 0xFF,
        dst & 0xFF,
        msg_type & 0xFF,
        seq & 0xFF,
        (seq >> 8) & 0xFF,
        length & 0xFF,
        (length >> 8) & 0xFF,
    )) + payload
    checksum = crc16_ccitt(body)
    return SOF + body + bytes((checksum & 0xFF, checksum >> 8))


class FrameParser:
    MIN_FRAME_SIZE = 13

    def __init__(self, max_payload=MAX_PAYLOAD, max_buffer=1024):
        self.max_payload = max_payload
        self.max_buffer = max_buffer
        self.buffer = bytearray()
        self.crc_errors = 0
        self.format_errors = 0
        self.dropped_bytes = 0

    def feed(self, data):
        if data:
            self.buffer.extend(data)
        if len(self.buffer) > self.max_buffer:
            count = len(self.buffer) - self.max_buffer
            self.buffer = self.buffer[count:]
            self.dropped_bytes += count

        frames = []
        while True:
            start = self.buffer.find(SOF)
            if start < 0:
                keep = 1 if self.buffer and self.buffer[-1] == SOF[0] else 0
                self.dropped_bytes += len(self.buffer) - keep
                self.buffer = bytearray((SOF[0],)) if keep else bytearray()
                break
            if start:
                self.buffer = self.buffer[start:]
                self.dropped_bytes += start
            if len(self.buffer) < self.MIN_FRAME_SIZE:
                break

            version = self.buffer[2]
            payload_len = self.buffer[9] | (self.buffer[10] << 8)
            if version != VERSION or payload_len > self.max_payload:
                self.format_errors += 1
                self.buffer = self.buffer[1:]
                continue

            frame_size = self.MIN_FRAME_SIZE + payload_len
            if len(self.buffer) < frame_size:
                break
            payload_end = 11 + payload_len
            expected = self.buffer[payload_end] | (self.buffer[payload_end + 1] << 8)
            actual = crc16_ccitt(self.buffer[2:payload_end])
            if expected != actual:
                self.crc_errors += 1
                self.buffer = self.buffer[1:]
                continue

            seq = self.buffer[7] | (self.buffer[8] << 8)
            frames.append(Frame(
                self.buffer[3], self.buffer[4], self.buffer[5], self.buffer[6], seq,
                bytes(self.buffer[11:payload_end]), version,
            ))
            self.buffer = self.buffer[frame_size:]
        return frames
