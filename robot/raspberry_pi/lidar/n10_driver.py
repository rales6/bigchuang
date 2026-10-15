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
    MAX_PACKET_SPAN_DEG = 30.0
    # Valid packets contain samples about one degree apart. A missing 16-point
    # packet leaves a gap near 16 degrees and must invalidate the revolution.
    MAX_REVOLUTION_GAP_DEG = 6.0
    ANGLE_DEDUP_BIN_DEG = 1.0

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
        self.invalid_packet_spans = 0
        self.incomplete_revolutions = 0
        self._scan_watchdog_s = None

    def set_scan_watchdog(self, timeout_s):
        """Opt-in empty frames on outage so a controller can stop/save state.

        Mapping and initial localization retain the existing default stream.
        This only changes read-side timeout reporting, never drives motors.
        """
        if timeout_s is not None and not .05 <= float(timeout_s) <= 5.:
            raise ValueError('scan watchdog must be 0.05..5 seconds or None')
        previous = self._scan_watchdog_s
        self._scan_watchdog_s = None if timeout_s is None else float(timeout_s)
        return previous

    def scans(self):
        # The first revolutions after the motor command are often partial or
        # sampled while the rotor is still reaching its steady speed.  They
        # must not become the SLAM origin reference: a distorted first scan
        # can make the next stationary scan look 30--45 degrees rotated.
        startup_discard_left = 4 if self.config.motor_control else 0
        if self.config.motor_control:
            self.start_motor()
        scan_raw_angles = []
        scan_distances = []
        scan_sample_angles = []
        revolution_valid = True
        previous_raw_angle = None
        offset_rad = math.radians(self.config.angle_offset_deg)

        last_scan_s = time.monotonic()
        while not self._closed:
            if self._scan_watchdog_s is not None and time.monotonic() - last_scan_s >= self._scan_watchdog_s:
                scan_raw_angles, scan_distances, scan_sample_angles = [], [], []
                previous_raw_angle, revolution_valid = None, True
                last_scan_s = time.monotonic()
                yield LaserScan(np.empty(0), np.empty(0), last_scan_s)
            waiting = getattr(self.serial, "in_waiting", 0)
            data = self.serial.read(waiting or 1)
            if not data:
                continue
            for packet in self.parser.feed(data):
                self.last_scan_frequency_hz = packet.scan_frequency_hz
                packet_span = (
                    packet.end_angle_deg - packet.start_angle_deg
                ) % 360.0
                if not 1.0 <= packet_span <= self.MAX_PACKET_SPAN_DEG:
                    self.invalid_packet_spans += 1
                    revolution_valid = False
                    continue
                raw_angles, distances_mm, _intensities = packet.samples()
                for raw_angle, distance_mm in zip(raw_angles, distances_mm):
                    if (
                        previous_raw_angle is not None
                        and previous_raw_angle > 300.0
                        and raw_angle < 60.0
                    ):
                        completed_scan = self._complete_revolution(
                            scan_raw_angles,
                            scan_distances,
                            scan_sample_angles,
                            revolution_valid,
                            offset_rad,
                        )
                        if completed_scan is not None:
                            if startup_discard_left > 0:
                                startup_discard_left -= 1
                            else:
                                last_scan_s = time.monotonic()
                                yield completed_scan
                        else:
                            self.incomplete_revolutions += 1
                        scan_raw_angles = []
                        scan_distances = []
                        scan_sample_angles = []
                        revolution_valid = True

                    raw_angle = float(raw_angle)
                    if (
                        previous_raw_angle is not None
                        and raw_angle + 2.0 < previous_raw_angle
                        and not (
                            previous_raw_angle > 300.0
                            and raw_angle < 60.0
                        )
                    ):
                        # A backward angle inside one revolution indicates a
                        # duplicated or out-of-order packet. Do not let the
                        # resulting multi-range ray reach scan matching.
                        revolution_valid = False
                    previous_raw_angle = raw_angle
                    scan_sample_angles.append(raw_angle)
                    if distance_mm in (0, 0xFFFF):
                        continue
                    scan_raw_angles.append(raw_angle)
                    scan_distances.append(float(distance_mm) / 1000.0)

    def _complete_revolution(
        self,
        raw_angles,
        distances_m,
        sample_angles,
        revolution_valid,
        offset_rad,
    ):
        """Validate and de-duplicate one physical lidar revolution."""
        if (
            not revolution_valid
            or len(raw_angles) < self.config.min_scan_points
            or len(sample_angles) < self.config.min_scan_points
        ):
            return None
        coverage_angles = np.unique(np.asarray(
            sample_angles,
            dtype=np.float64,
        ))
        if len(coverage_angles) < self.config.min_scan_points:
            return None
        coverage_angles.sort()
        circular_gaps = np.diff(np.concatenate((
            coverage_angles,
            coverage_angles[:1] + 360.0,
        )))
        if (
            not len(circular_gaps)
            or float(np.max(circular_gaps))
            > self.MAX_REVOLUTION_GAP_DEG
        ):
            return None

        # A physical single-return lidar can provide only one range for one
        # direction in one revolution. Duplicate/overlapping packets are
        # reduced to a median range rather than forming a radial false wall.
        bins = {}
        for angle, distance in zip(raw_angles, distances_m):
            bin_index = int(round(
                float(angle) / self.ANGLE_DEDUP_BIN_DEG
            )) % int(round(360.0 / self.ANGLE_DEDUP_BIN_DEG))
            bins.setdefault(bin_index, []).append(float(distance))
        if len(bins) < self.config.min_scan_points:
            return None
        clean_raw_angles = np.asarray(
            [index * self.ANGLE_DEDUP_BIN_DEG for index in sorted(bins)],
            dtype=np.float64,
        )
        clean_distances = np.asarray(
            [float(np.median(bins[index])) for index in sorted(bins)],
            dtype=np.float64,
        )
        raw_radians = np.radians(clean_raw_angles)
        robot_angles = (
            -raw_radians if self.config.clockwise else raw_radians
        ) + offset_rad
        return LaserScan(
            robot_angles,
            clean_distances,
            time.monotonic(),
        )

    def start_motor(self):
        reset = getattr(self.serial, "reset_input_buffer", None)
        if reset:
            reset()
        self.serial.write(self._motor_command(True))
        self._motor_started = True

    def reset_scan_stream(self):
        """Discard serial backlog accumulated during a long link outage."""
        reset = getattr(self.serial, "reset_input_buffer", None)
        if reset:
            reset()
        self.parser = N10PacketParser()

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
