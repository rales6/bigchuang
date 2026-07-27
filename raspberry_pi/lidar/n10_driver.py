"""镭神智能 N10 串口雷达驱动。

依据随附的《N10 数据输出协议 V1.1》实现，不使用 RPLIDAR 协议：

* 串口参数 230400, 8N1；
* 点云帧固定 58 字节，帧头为 ``A5 5A 3A``；
* 每帧 16 个距离/强度点，距离为大端毫米；
* 最后一字节是前 57 字节累加和的低 8 位。
"""

from dataclasses import dataclass
import math
import time

import numpy as np

from raspberry_pi.config import LidarConfig
from raspberry_pi.lidar.base import LaserScan


N10_HEADER = b"\xA5\x5A"
N10_PACKET_LENGTH = 58
N10_POINTS_PER_PACKET = 16


def n10_checksum(data):
    return sum(data) & 0xFF


@dataclass(frozen=True)
class N10Packet:
    encoder_period_us: int
    start_angle_deg: float
    end_angle_deg: float
    distances_mm: np.ndarray
    intensities: np.ndarray

    @property
    def scan_frequency_hz(self):
        # 协议说明：一圈包含 24 个码盘周期。
        if self.encoder_period_us <= 0:
            return 0.0
        return 1_000_000.0 / (self.encoder_period_us * 24.0)

    def samples(self):
        """返回按协议起止角均匀插值得到的 16 个样本。"""
        end = self.end_angle_deg
        if end < self.start_angle_deg:
            end += 360.0
        angles = np.linspace(self.start_angle_deg, end, N10_POINTS_PER_PACKET)
        return np.mod(angles, 360.0), self.distances_mm, self.intensities


def decode_n10_packet(packet):
    if len(packet) != N10_PACKET_LENGTH:
        raise ValueError("N10 packet must contain 58 bytes")
    if packet[:2] != N10_HEADER or packet[2] != N10_PACKET_LENGTH:
        raise ValueError("invalid N10 packet header or length")
    if packet[-1] != n10_checksum(packet[:-1]):
        raise ValueError("invalid N10 packet checksum")

    distances = np.empty(N10_POINTS_PER_PACKET, dtype=np.uint16)
    intensities = np.empty(N10_POINTS_PER_PACKET, dtype=np.uint8)
    for index in range(N10_POINTS_PER_PACKET):
        offset = 7 + index * 3
        distances[index] = (packet[offset] << 8) | packet[offset + 1]
        intensities[index] = packet[offset + 2]
    return N10Packet(
        encoder_period_us=(packet[3] << 8) | packet[4],
        start_angle_deg=((packet[5] << 8) | packet[6]) / 100.0,
        end_angle_deg=((packet[55] << 8) | packet[56]) / 100.0,
        distances_mm=distances,
        intensities=intensities,
    )


class N10PacketParser:
    """能从拆包、粘包、噪声和校验错误中恢复的流解析器。"""

    def __init__(self, max_buffer=4096):
        self.max_buffer = max_buffer
        self.buffer = bytearray()
        self.checksum_errors = 0
        self.format_errors = 0
        self.dropped_bytes = 0

    def feed(self, data):
        if data:
            self.buffer.extend(data)
        if len(self.buffer) > self.max_buffer:
            count = len(self.buffer) - self.max_buffer
            del self.buffer[:count]
            self.dropped_bytes += count

        packets = []
        while True:
            start = self.buffer.find(N10_HEADER)
            if start < 0:
                keep = 1 if self.buffer and self.buffer[-1] == N10_HEADER[0] else 0
                self.dropped_bytes += len(self.buffer) - keep
                self.buffer = bytearray((N10_HEADER[0],)) if keep else bytearray()
                break
            if start:
                del self.buffer[:start]
                self.dropped_bytes += start
            if len(self.buffer) < 3:
                break
            if self.buffer[2] != N10_PACKET_LENGTH:
                self.format_errors += 1
                del self.buffer[0]
                continue
            if len(self.buffer) < N10_PACKET_LENGTH:
                break
            raw = bytes(self.buffer[:N10_PACKET_LENGTH])
            if raw[-1] != n10_checksum(raw[:-1]):
                self.checksum_errors += 1
                del self.buffer[0]
                continue
            packets.append(decode_n10_packet(raw))
            del self.buffer[:N10_PACKET_LENGTH]
        return packets


class N10LidarDriver:
    MOTOR_COMMAND_LENGTH = 188

    def __init__(self, config=None, serial_instance=None):
        self.config = config or LidarConfig()
        if serial_instance is None:
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError(
                    "缺少 pyserial，请运行 pip install -r requirements-pi.txt"
                ) from exc
            serial_instance = serial.Serial(
                self.config.port,
                self.config.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
            )
        self.serial = serial_instance
        self.parser = N10PacketParser()
        self._closed = False
        self._motor_started = False
        self.last_scan_frequency_hz = 0.0

    def scans(self):
        if self.config.motor_control:
            self.start_motor()
        scan_angles = []
        scan_distances = []
        previous_raw_angle = None
        offset_rad = math.radians(self.config.angle_offset_deg)

        while not self._closed:
            waiting = getattr(self.serial, "in_waiting", 0)
            data = self.serial.read(waiting or 1)
            if not data:
                continue
            for packet in self.parser.feed(data):
                self.last_scan_frequency_hz = packet.scan_frequency_hz
                raw_angles, distances_mm, _intensities = packet.samples()
                for raw_angle, distance_mm in zip(raw_angles, distances_mm):
                    if (
                        previous_raw_angle is not None
                        and previous_raw_angle > 300.0
                        and raw_angle < 60.0
                    ):
                        if len(scan_angles) >= self.config.min_scan_points:
                            yield LaserScan(
                                np.asarray(scan_angles, dtype=np.float64),
                                np.asarray(scan_distances, dtype=np.float64),
                                time.monotonic(),
                            )
                        scan_angles = []
                        scan_distances = []

                    previous_raw_angle = float(raw_angle)
                    if distance_mm in (0, 0xFFFF):
                        continue
                    raw_rad = math.radians(float(raw_angle))
                    angle_rad = (-raw_rad if self.config.clockwise else raw_rad) + offset_rad
                    scan_angles.append(angle_rad)
                    scan_distances.append(float(distance_mm) / 1000.0)

    def start_motor(self):
        reset = getattr(self.serial, "reset_input_buffer", None)
        if reset:
            reset()
        self.serial.write(self._motor_command(True))
        self._motor_started = True

    def stop_motor(self):
        self.serial.write(self._motor_command(False))
        self._motor_started = False

    def close(self):
        if self._closed:
            return
        if self.config.motor_control and self._motor_started:
            try:
                self.stop_motor()
            except Exception:
                pass
        self._closed = True
        close = getattr(self.serial, "close", None)
        if close:
            close()

    @classmethod
    def _motor_command(cls, start):
        command = bytearray(cls.MOTOR_COMMAND_LENGTH)
        command[0:3] = b"\xA5\x5A\x55"
        command[184] = 0x01
        command[185] = 0x01 if start else 0x00
        command[186:188] = b"\xFA\xFB"
        return bytes(command)

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()

