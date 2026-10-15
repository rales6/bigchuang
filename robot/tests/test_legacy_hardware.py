import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "car"))

import config  # noqa: E402
from drivers.legacy_motor_bus import LegacyMotorBus  # noqa: E402
from drivers.arm import ArmController  # noqa: E402
from drivers.pwm_arm import PWMArmController  # noqa: E402
from protocol.messages import encode_arm_joints  # noqa: E402


class FakeTransport:
    def __init__(self):
        self.writes = []

    def read(self):
        return b""

    def write(self, data):
        self.writes.append(data)
        return len(data)


class FakePWM:
    def __init__(self):
        self.values = []

    def duty(self, value):
        self.values.append(value)


class LegacyMotorBusTest(unittest.TestCase):
    def test_forward_matches_original_bottom_board_protocol(self):
        transport = FakeTransport()
        bus = LegacyMotorBus(
            transport, (6, 7, 8, 9), (1, -1, 1, -1), 0
        )
        bus.send_drive((300, 300, 300, 300))
        bus.poll(100)
        self.assertEqual(
            b"#006P1800T0000!#007P1200T0000!#008P1800T0000!#009P1200T0000!",
            transport.writes[-1],
        )

    def test_stop_centres_all_motor_channels(self):
        transport = FakeTransport()
        bus = LegacyMotorBus(
            transport, (6, 7, 8, 9), (1, -1, 1, -1), 0
        )
        bus.stop(mask=0x01)
        bus.poll(100)
        self.assertEqual(
            b"#006P1500T0100!#007P1500T0100!#008P1500T0100!#009P1500T0100!",
            transport.writes[-1],
        )
        bus.poll(105)
        self.assertEqual(
            b"#006PDST!#007PDST!#008PDST!#009PDST!",
            transport.writes[-1],
        )

    def test_arm_preemption_uses_original_hash_protocol(self):
        transport = FakeTransport()
        bus = LegacyMotorBus(
            transport, (6, 7, 8, 9), (1, -1, 1, -1), 0
        )
        payload = encode_arm_joints([(0, 1800), (5, 1500)], 600)
        bus.preempt_arm(payload)
        bus.poll(100)
        bus.poll(105)
        self.assertEqual(b"#255PDST!", transport.writes[0])
        self.assertEqual(b"#000P1800T0600!", transport.writes[1])
        self.assertEqual(1, bus.arm_write_count)
        bus.poll(110)
        self.assertEqual(b"#005P1500T0600!", transport.writes[2])
        self.assertEqual(2, bus.arm_write_count)

    def test_completed_arm_action_does_not_send_redundant_global_stop(self):
        transport = FakeTransport()
        bus = LegacyMotorBus(
            transport, (6, 7, 8, 9), (1, -1, 1, -1), 0
        )
        arm = ArmController(bus, config)
        arm.set_joints([(0, 2200, 100)])
        bus.poll(100)
        started = arm._started_ms[0]
        arm.update(started + 100)
        arm.set_joints([(1, 1600, 100)])
        bus.poll(200)
        self.assertEqual(0, bus.arm_stop_write_count)
        self.assertEqual(b"#001P1600T0100!", transport.writes[-1])

    def test_running_arm_action_is_preempted_before_new_target(self):
        transport = FakeTransport()
        bus = LegacyMotorBus(
            transport, (6, 7, 8, 9), (1, -1, 1, -1), 0
        )
        arm = ArmController(bus, config)
        arm.set_joints([(0, 2200, 1000)])
        bus.poll(100)
        arm.set_joints([(0, 800, 1000)])
        bus.poll(200)
        bus.poll(205)
        self.assertEqual(b"#255PDST!", transport.writes[-2])
        self.assertEqual(b"#000P0800T1000!", transport.writes[-1])

    def test_cartesian_arm_command_can_include_claw_opening(self):
        transport = FakeTransport()
        bus = LegacyMotorBus(
            transport, (6, 7, 8, 9), (1, -1, 1, -1), 0
        )
        arm = ArmController(bus, config)
        arm.move_cartesian_delta(20, 30, 0, 20)
        for tick in range(10):
            bus.poll(100 + tick)
        self.assertIn(b"#005P1520T1060!", transport.writes)
        self.assertIn(b"#001P1540T1060!", transport.writes)
        self.assertIn(b"#002P1760T1060!", transport.writes)
        self.assertIn(b"#003P1300T1060!", transport.writes)


class PWMArmTest(unittest.TestCase):
    def test_joint_is_interpolated_without_blocking(self):
        channels = [FakePWM() for _ in range(6)]
        arm = PWMArmController(channels, config)
        arm.set_joints([(0, 2000, 100)])
        started = arm._started_ms[0]
        arm.update(started + 50)
        self.assertEqual(1750, arm.positions[0])
        self.assertEqual(1750 * 1024 // 20000, channels[0].values[-1])
        arm.update(started + 100)
        self.assertEqual(2000, arm.positions[0])
        self.assertFalse(arm.moving)

    def test_stop_holds_interpolated_position(self):
        channels = [FakePWM() for _ in range(6)]
        arm = PWMArmController(channels, config)
        arm.set_joints([(0, 2000, 1000)])
        started = arm._started_ms[0]
        arm.update(started + 250)
        arm.stop(started + 250)
        held = arm.positions[0]
        arm.update(started + 900)
        self.assertEqual(held, arm.positions[0])
        self.assertEqual(held, arm.targets[0])


if __name__ == "__main__":
    unittest.main()
