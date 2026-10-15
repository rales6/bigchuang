import math
import os
import sys
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from raspberry_pi.config import LidarConfig  # noqa: E402
from raspberry_pi.lidar.n10_driver import (  # noqa: E402
    N10LidarDriver,
    N10PacketParser,
    decode_n10_packet,
    n10_checksum,
)


def make_packet(start_deg, end_deg, distances=None, period_us=4167):
    distances = distances or [1000 + index for index in range(16)]
    packet = bytearray(58)
    packet[0:3] = b"\xA5\x5A\x3A"
    packet[3:5] = int(period_us).to_bytes(2, "big")
    packet[5:7] = int(round(start_deg * 100)).to_bytes(2, "big")
    for index, distance in enumerate(distances):
        offset = 7 + index * 3
        packet[offset:offset + 2] = int(distance).to_bytes(2, "big")
        packet[offset + 2] = 20 + index
    packet[55:57] = int(round(end_deg * 100)).to_bytes(2, "big")
    packet[57] = n10_checksum(packet[:57])
    return bytes(packet)


class FakeSerial:
    def __init__(self, rx=b""):
        self.rx = bytearray(rx)
        self.tx = bytearray()
        self.closed = False

    @property
    def in_waiting(self):
        return len(self.rx)

    def read(self, count):
        data = bytes(self.rx[:count])
        del self.rx[:count]
        return data

    def write(self, data):
        self.tx.extend(data)
        return len(data)

    def reset_input_buffer(self):
        self.rx.clear()

    def close(self):
        self.closed = True


class N10LidarTest(unittest.TestCase):
    def test_opt_in_watchdog_reports_empty_scan_on_missing_data(self):
        serial = FakeSerial()
        driver = N10LidarDriver(LidarConfig(motor_control=False), serial_instance=serial)
        self.assertIsNone(driver.set_scan_watchdog(.5))
        with patch('raspberry_pi.lidar.n10_driver.time.monotonic', side_effect=[0., .1, .2, .3, .6, .6]):
            scan = next(driver.scans())
        self.assertEqual(len(scan.distances_m), 0)
        self.assertEqual(scan.timestamp_s, .6)
        self.assertEqual(driver.set_scan_watchdog(None), .5)
        self.assertIsNone(driver._scan_watchdog_s)
        self.assertEqual(serial.tx, b'')
        driver.close()

    def test_watchdog_rejects_unbounded_or_invalid_settings(self):
        driver = N10LidarDriver(LidarConfig(motor_control=False), serial_instance=FakeSerial())
        for timeout in (0., -1., 10., math.nan):
            with self.assertRaises(ValueError):
                driver.set_scan_watchdog(timeout)
        driver.close()

    def test_decode_vendor_58_byte_frame(self):
        decoded = decode_n10_packet(make_packet(169.04, 184.04, period_us=4166))
        self.assertEqual(4166, decoded.encoder_period_us)
        self.assertAlmostEqual(169.04, decoded.start_angle_deg)
        self.assertAlmostEqual(184.04, decoded.end_angle_deg)
        self.assertEqual(1000, decoded.distances_mm[0])
        self.assertEqual(35, decoded.intensities[-1])
        self.assertAlmostEqual(10.0, decoded.scan_frequency_hz, places=2)

    def test_parser_recovers_from_noise_fragmentation_and_bad_checksum(self):
        damaged = bytearray(make_packet(0.0, 15.0))
        damaged[-1] ^= 0x01
        valid = make_packet(15.0, 30.0)
        parser = N10PacketParser()
        packets = []
        stream = b"noise" + damaged + valid
        for index in range(0, len(stream), 7):
            packets.extend(parser.feed(stream[index:index + 7]))
        self.assertEqual(1, len(packets))
        self.assertAlmostEqual(15.0, packets[0].start_angle_deg)
        self.assertEqual(1, parser.checksum_errors)
        self.assertGreaterEqual(parser.dropped_bytes, 5)

    def test_driver_builds_clockwise_n10_revolution_in_robot_coordinates(self):
        stream = b"".join(
            make_packet(angle, (angle + 15) % 360)
            for angle in range(0, 360, 15)
        )
        serial = FakeSerial(stream)
        config = LidarConfig(motor_control=False, min_scan_points=100)
        driver = N10LidarDriver(config, serial)
        scan = next(driver.scans())
        self.assertGreater(len(scan.angles_rad), 350)
        self.assertAlmostEqual(10.0, driver.last_scan_frequency_hz, delta=0.01)
        # 厂商角度顺时针增加，机器人坐标系中应转换成负角度。
        raw_90_index = min(
            range(len(scan.angles_rad)),
            key=lambda index: abs(scan.angles_rad[index] + math.pi / 2),
        )
        self.assertAlmostEqual(-math.pi / 2, scan.angles_rad[raw_90_index], delta=0.02)

    def test_driver_rejects_partial_revolution_even_with_many_points(self):
        # Starting midway through a revolution used to pass the point-count
        # gate and become a distorted SLAM frame.
        partial = b"".join(
            make_packet(angle, (angle + 15) % 360)
            for angle in range(120, 360, 15)
        )
        complete = b"".join(
            make_packet(angle, (angle + 15) % 360)
            for angle in range(0, 360, 15)
        )
        driver = N10LidarDriver(
            LidarConfig(motor_control=False, min_scan_points=100),
            FakeSerial(partial + complete),
        )

        scan = next(driver.scans())

        self.assertGreaterEqual(driver.incomplete_revolutions, 1)
        self.assertGreater(len(scan.angles_rad), 300)

    def test_driver_rejects_revolution_with_one_missing_packet(self):
        missing_packet = b"".join(
            make_packet(angle, (angle + 15) % 360)
            for angle in range(0, 360, 15)
            if angle != 180
        )
        complete = b"".join(
            make_packet(angle, (angle + 15) % 360)
            for angle in range(0, 360, 15)
        )
        driver = N10LidarDriver(
            LidarConfig(motor_control=False, min_scan_points=100),
            FakeSerial(missing_packet + complete),
        )

        scan = next(driver.scans())

        self.assertGreaterEqual(driver.incomplete_revolutions, 1)
        self.assertGreater(len(scan.angles_rad), 300)

    def test_driver_collapses_duplicate_ranges_at_one_angle(self):
        driver = N10LidarDriver(
            LidarConfig(motor_control=False, min_scan_points=3),
            FakeSerial(),
        )
        raw_angles = [float(index) for index in range(360)]
        distances = [2.0] * 360
        raw_angles.extend((90.0, 90.0, 90.0))
        distances.extend((0.4, 3.5, 6.0))

        scan = driver._complete_revolution(
            raw_angles,
            distances,
            [float(index) for index in range(360)],
            True,
            0.0,
        )

        self.assertIsNotNone(scan)
        self.assertEqual(360, len(scan.angles_rad))
        index = min(
            range(len(scan.angles_rad)),
            key=lambda item: abs(scan.angles_rad[item] + math.pi / 2),
        )
        self.assertAlmostEqual(2.75, scan.distances_m[index], places=2)

    def test_driver_rejects_impossible_packet_angle_span(self):
        invalid_revolution = b"".join((
            make_packet(0.0, 170.0),
            *(make_packet(angle, (angle + 15) % 360)
              for angle in range(15, 360, 15)),
        ))
        complete_revolution = b"".join(
            make_packet(angle, (angle + 15) % 360)
            for angle in range(0, 360, 15)
        )
        driver = N10LidarDriver(
            LidarConfig(motor_control=False, min_scan_points=100),
            FakeSerial(invalid_revolution + complete_revolution),
        )

        scan = next(driver.scans())

        self.assertEqual(1, driver.invalid_packet_spans)
        self.assertGreater(len(scan.angles_rad), 300)

    def test_motor_start_discards_four_unstable_revolutions(self):
        class NonClearingSerial(FakeSerial):
            def reset_input_buffer(self):
                # Preserve the deterministic fixture that represents bytes
                # arriving after the real serial buffer reset.
                pass

            def read(self, count):
                # A real UART is drained incrementally; avoid handing the
                # parser an artificial 8-KiB burst beyond its safety buffer.
                return super().read(min(count, 256))

        stream = b"".join(
            make_packet(
                angle,
                (angle + 15) % 360,
                distances=[1000 + revolution * 100] * 16,
            )
            for revolution in range(6)
            for angle in range(0, 360, 15)
        )
        serial = NonClearingSerial(stream)
        driver = N10LidarDriver(
            LidarConfig(motor_control=True, min_scan_points=100),
            serial,
        )

        scan = next(driver.scans())

        median_range = sorted(scan.distances_m)[len(scan.distances_m) // 2]
        self.assertAlmostEqual(1.4, median_range, delta=0.01)
        self.assertTrue(driver._motor_started)

    def test_motor_commands_match_vendor_control_frame(self):
        start = N10LidarDriver._motor_command(True)
        stop = N10LidarDriver._motor_command(False)
        self.assertEqual(188, len(start))
        self.assertEqual(b"\xA5\x5A\x55", start[:3])
        self.assertEqual(b"\x01\x01\xFA\xFB", start[184:188])
        self.assertEqual(b"\x01\x00\xFA\xFB", stop[184:188])


if __name__ == "__main__":
    unittest.main()
