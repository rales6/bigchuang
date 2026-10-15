import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "car"))

from protocol.frame import (  # noqa: E402
    ADDR_ESP32,
    ADDR_RASPBERRY_PI,
    FLAG_ACK_REQUIRED,
    FrameParser,
    crc16_ccitt,
    encode_frame,
)
from protocol.messages import (  # noqa: E402
    MSG_SET_BALANCED_TWIST,
    MSG_SET_TWIST,
    decode_balanced_twist,
    decode_drive_calibration,
    decode_robot_status,
    decode_twist,
    encode_robot_status,
    encode_balanced_twist,
    encode_twist,
    encode_drive_calibration,
)


class ProtocolV2Test(unittest.TestCase):
    def test_balanced_twist_round_trip(self):
        payload = encode_balanced_twist(
            300,
            -500,
            600,
            (1.15, 1.0, 0.90, 1.0),
        )
        self.assertEqual(14, len(payload))
        self.assertEqual(
            (300, -500, 600, (1.15, 1.0, 0.90, 1.0)),
            decode_balanced_twist(payload),
        )
        frame = encode_frame(
            MSG_SET_BALANCED_TWIST,
            99,
            payload,
        )
        self.assertEqual(
            MSG_SET_BALANCED_TWIST,
            FrameParser().feed(frame)[0].msg_type,
        )

    def test_known_crc_vector(self):
        self.assertEqual(0x29B1, crc16_ccitt(b"123456789"))

    def test_fragmented_and_sticky_frames(self):
        first = encode_frame(
            MSG_SET_TWIST, 100, encode_twist(300, -500, 600),
            src=ADDR_RASPBERRY_PI, dst=ADDR_ESP32,
            flags=FLAG_ACK_REQUIRED,
        )
        second = encode_frame(MSG_SET_TWIST, 101, encode_twist(0, 0, 100))
        parser = FrameParser()
        frames = []
        stream = b"noise" + first + second
        for index in range(0, len(stream), 3):
            frames.extend(parser.feed(stream[index:index + 3]))
        self.assertEqual([100, 101], [frame.seq for frame in frames])
        self.assertEqual((300, -500, 600), decode_twist(frames[0].payload))
        self.assertEqual(5, parser.dropped_bytes)

    def test_crc_damage_recovers(self):
        damaged = bytearray(encode_frame(MSG_SET_TWIST, 1, encode_twist(1, 2, 100)))
        damaged[-1] ^= 0x55
        valid = encode_frame(MSG_SET_TWIST, 2, encode_twist(3, 4, 100))
        parser = FrameParser()
        frames = parser.feed(damaged + valid)
        self.assertEqual([2], [frame.seq for frame in frames])
        self.assertEqual(1, parser.crc_errors)

    def test_robot_status_round_trip(self):
        status = {
            "uptime_ms": 1234,
            "flags": 3,
            "linear_mm_s": 200,
            "angular_mrad_s": -100,
            "left_output": 250,
            "right_output": 220,
            "wheel_feedback": (190, 188, 191, 187),
            "joint_positions": (1500, 1700, 2000, 1100, 1500, 1200),
            "battery_mv": 11800,
            "bus_errors": 2,
        }
        self.assertEqual(status, decode_robot_status(encode_robot_status(status)))

    def test_drive_calibration_round_trip(self):
        decoded = decode_drive_calibration(
            encode_drive_calibration(0.02, 0.0001)
        )
        self.assertAlmostEqual(0.02, decoded["trim_intercept"], places=6)
        self.assertAlmostEqual(
            0.0001, decoded["trim_slope_per_mm_s"], places=8
        )


if __name__ == "__main__":
    unittest.main()
