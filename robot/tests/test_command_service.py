import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "car"))

from core.controller import PiCommandService, RobotController  # noqa: E402
from protocol.frame import (  # noqa: E402
    ADDR_ESP32,
    ADDR_RASPBERRY_PI,
    FLAG_ACK_REQUIRED,
    FrameParser,
    encode_frame,
)
from protocol.messages import (  # noqa: E402
    CANCEL_ALL,
    MSG_CANCEL,
    MSG_HEARTBEAT,
    MSG_SET_BALANCED_TWIST,
    MSG_SET_DRIVE_CALIBRATION,
    MSG_SET_TWIST,
    encode_twist,
    encode_balanced_twist,
    encode_drive_calibration,
)


class FakeTransport:
    def __init__(self):
        self.rx = b""
        self.tx = bytearray()

    def read(self):
        data, self.rx = self.rx, b""
        return data

    def write(self, data):
        self.tx.extend(data)


class FakeDrive:
    moving = False
    closed_loop = False
    current_linear = 0
    current_angular = 0
    outputs = (0, 0, 0, 0)
    feedback = (0, 0, 0, 0)

    def __init__(self):
        self.stopped = 0
        self.twist = None
        self.calibration = (0.0, 0.0)

    def set_twist(
        self,
        linear,
        angular,
        ttl,
        now,
        motor_output_gains=None,
    ):
        self.twist = (linear, angular, ttl)
        self.motor_output_gains = motor_output_gains
        self.moving = True

    def stop(self, emergency=False):
        self.stopped += 1
        self.moving = False

    def update(self, now):
        pass

    def set_feedback(self, wheels, now):
        self.feedback = wheels

    def set_calibration(self, intercept, slope):
        if self.moving:
            raise ValueError("moving")
        self.calibration = (intercept, slope)

    def calibration_values(self):
        return self.calibration

    def reset_calibration(self):
        self.calibration = (0.0, 0.0)


class FakeArm:
    moving = False
    positions = [1500] * 6

    def __init__(self):
        self.stopped = 0

    def set_joints(self, joints):
        self.moving = True

    def move_cartesian_delta(self, forward_mm=0, up_mm=0, left_deg=0,
                             claw_open=None, duration_ms=None):
        self.delta = (forward_mm, up_mm, left_deg, claw_open)
        self.moving = True
        return 1000

    def stop(self):
        self.stopped += 1
        self.moving = False

    def set_feedback(self, positions):
        self.positions = positions


class FakeIndicators:
    led_value = False
    buzzer_active = False

    def __init__(self):
        self.cancelled = 0

    def update(self, now):
        pass

    def cancel_buzzer(self):
        self.cancelled += 1

    def cancel_led(self):
        self.cancelled += 1

    def set_led(self, *args):
        pass

    def beep(self, *args):
        pass


class FakeBus:
    error_count = 0
    on_status = None


class FakeConfig:
    PI_LINK_TIMEOUT_MS = 800
    MAX_LINEAR_MM_S = 550
    MAX_ANGULAR_MRAD_S = 3500
    COMMAND_MAX_TTL_MS = 2500
    ASCII_MOVE_DEFAULT_SPEED_LEVEL = 70
    ASCII_MOVE_MIN_DURATION_MS = 450


class CommandServiceTest(unittest.TestCase):
    def test_pi_owned_wheel_gains_reach_drive_base(self):
        drive = FakeDrive()
        controller = RobotController(
            drive,
            FakeArm(),
            FakeIndicators(),
            FakeBus(),
            FakeConfig(),
        )
        response_type, _payload = controller.handle_command(
            MSG_SET_BALANCED_TWIST,
            encode_balanced_twist(
                250,
                100,
                500,
                (1.15, 1.0, 0.90, 1.0),
            ),
            1000,
        )
        self.assertEqual(0x02, response_type)
        self.assertEqual((250, 100, 500), drive.twist)
        self.assertEqual(
            (1.15, 1.0, 0.90, 1.0),
            drive.motor_output_gains,
        )

    def test_new_cancel_frame_preempts_previous_drive(self):
        transport = FakeTransport()
        drive = FakeDrive()
        arm = FakeArm()
        indicators = FakeIndicators()
        controller = RobotController(drive, arm, indicators, FakeBus(), FakeConfig())
        service = PiCommandService(transport, controller)

        drive_frame = encode_frame(
            MSG_SET_TWIST, 1, encode_twist(300, 0, 500),
            src=ADDR_RASPBERRY_PI, dst=ADDR_ESP32, flags=FLAG_ACK_REQUIRED,
        )
        cancel_frame = encode_frame(
            MSG_CANCEL, 2, bytes((CANCEL_ALL,)),
            src=ADDR_RASPBERRY_PI, dst=ADDR_ESP32, flags=FLAG_ACK_REQUIRED,
        )
        transport.rx = drive_frame + cancel_frame
        service.poll(1000)

        self.assertFalse(drive.moving)
        self.assertGreaterEqual(drive.stopped, 1)
        self.assertGreaterEqual(arm.stopped, 1)
        self.assertEqual(2, indicators.cancelled)
        self.assertEqual(2, len(FrameParser().feed(transport.tx)))

    def test_wrong_physical_peer_address_is_ignored(self):
        transport = FakeTransport()
        controller = RobotController(
            FakeDrive(), FakeArm(), FakeIndicators(), FakeBus(), FakeConfig()
        )
        service = PiCommandService(transport, controller)
        transport.rx = encode_frame(
            MSG_SET_TWIST, 1, encode_twist(100, 0, 100),
            src=0x20, dst=ADDR_ESP32,
        )
        service.poll(1000)
        self.assertEqual(1, service.bad_source_count)
        self.assertEqual(b"", bytes(transport.tx))

    def test_ascii_command_is_accepted_on_pi_uart(self):
        transport = FakeTransport()
        arm = FakeArm()
        controller = RobotController(
            FakeDrive(), arm, FakeIndicators(), FakeBus(), FakeConfig()
        )
        service = PiCommandService(transport, controller)
        transport.rx = b"#ARMF100U050L000C020!"
        service.poll(1000)
        self.assertEqual((100, 50, 0, 20), arm.delta)
        self.assertIn(b"OK:ARM", bytes(transport.tx))

    def test_binary_sequence_35_is_not_mistaken_for_ascii_start(self):
        transport = FakeTransport()
        controller = RobotController(
            FakeDrive(), FakeArm(), FakeIndicators(), FakeBus(), FakeConfig()
        )
        service = PiCommandService(transport, controller)
        transport.rx = encode_frame(
            MSG_HEARTBEAT,
            35,
            src=ADDR_RASPBERRY_PI,
            dst=ADDR_ESP32,
            flags=FLAG_ACK_REQUIRED,
        )

        service.poll(1000)

        responses = FrameParser().feed(transport.tx)
        self.assertEqual([35], [frame.seq for frame in responses])
        self.assertEqual(bytearray(), service._ascii_buffer)

    def test_sequence_zero_still_works_after_sequence_35(self):
        transport = FakeTransport()
        controller = RobotController(
            FakeDrive(), FakeArm(), FakeIndicators(), FakeBus(), FakeConfig()
        )
        service = PiCommandService(transport, controller)
        transport.rx = (
            encode_frame(
                MSG_HEARTBEAT,
                35,
                src=ADDR_RASPBERRY_PI,
                dst=ADDR_ESP32,
                flags=FLAG_ACK_REQUIRED,
            )
            + encode_frame(
                MSG_HEARTBEAT,
                0,
                src=ADDR_RASPBERRY_PI,
                dst=ADDR_ESP32,
                flags=FLAG_ACK_REQUIRED,
            )
        )

        service.poll(1000)

        responses = FrameParser().feed(transport.tx)
        self.assertEqual([35, 0], [frame.seq for frame in responses])
        self.assertEqual(bytearray(), service._ascii_buffer)

    def test_ascii_move_speed_level_sets_stronger_twist(self):
        transport = FakeTransport()
        drive = FakeDrive()
        controller = RobotController(
            drive, FakeArm(), FakeIndicators(), FakeBus(), FakeConfig()
        )
        service = PiCommandService(transport, controller)
        transport.rx = b"#MOVF300L000S090!"
        service.poll(1000)
        self.assertEqual((495, 0, 606), drive.twist)
        self.assertIn(b"OK:MOV", bytes(transport.tx))

    def test_drive_calibration_command_updates_stopped_drive(self):
        transport = FakeTransport()
        drive = FakeDrive()
        controller = RobotController(
            drive, FakeArm(), FakeIndicators(), FakeBus(), FakeConfig()
        )
        service = PiCommandService(transport, controller)
        transport.rx = encode_frame(
            MSG_SET_DRIVE_CALIBRATION,
            7,
            encode_drive_calibration(0.02, 0.0001),
            src=ADDR_RASPBERRY_PI,
            dst=ADDR_ESP32,
            flags=FLAG_ACK_REQUIRED,
        )
        service.poll(1000)
        self.assertAlmostEqual(0.02, drive.calibration[0], places=6)
        self.assertAlmostEqual(0.0001, drive.calibration[1], places=8)


if __name__ == "__main__":
    unittest.main()
