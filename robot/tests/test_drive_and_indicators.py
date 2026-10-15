import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "car"))

import config  # noqa: E402
from core.timebase import ticks_ms  # noqa: E402
from drivers.drive_base import DriveBase  # noqa: E402
from drivers.drive_calibration import DriveCalibration  # noqa: E402
from drivers.indicators import IndicatorController  # noqa: E402
from protocol.messages import LED_BLINK  # noqa: E402


class FakeBus:
    def __init__(self):
        self.drives = []
        self.stops = []

    def send_drive(self, values):
        self.drives.append(values)

    def stop(self, mask):
        self.stops.append(mask)


class DriveTest(unittest.TestCase):
    def test_acceleration_is_ramped(self):
        bus = FakeBus()
        drive = DriveBase(bus, config)
        base = drive.last_update
        drive.set_twist(500, 0, 600, base)
        drive.update(base + config.CONTROL_PERIOD_MS)
        first = drive.current_linear
        self.assertGreater(first, 0)
        self.assertLess(first, 500)
        drive.update(base + config.CONTROL_PERIOD_MS * 2)
        self.assertGreater(drive.current_linear, first)

    def test_curvature_limit_preserves_ratio(self):
        bus = FakeBus()
        drive = DriveBase(bus, config)
        drive.current_linear = config.MAX_LINEAR_MM_S
        drive.current_angular = config.MAX_ANGULAR_MRAD_S
        left, right = drive._wheel_targets()
        self.assertLessEqual(max(abs(left), abs(right)), config.MAX_WHEEL_MM_S)
        self.assertGreater(right, left)

    def test_emergency_stop_is_immediate(self):
        bus = FakeBus()
        drive = DriveBase(bus, config)
        drive.current_linear = 300
        drive.outputs = (400, 400, 400, 400)
        drive.stop(emergency=True)
        self.assertEqual((0, 0, 0, 0), drive.outputs)
        self.assertEqual([1], bus.stops)

    def test_angular_output_gain_only_amplifies_turning(self):
        bus = FakeBus()
        drive = DriveBase(bus, config)
        drive.current_linear = 0
        drive.current_angular = 3000
        left, right = drive._wheel_targets()
        expected = (
            3.0 * config.TRACK_WIDTH_MM / 2.0
            * config.ANGULAR_OUTPUT_GAIN
        )
        self.assertAlmostEqual(-expected, left)
        self.assertAlmostEqual(expected, right)

        drive.current_linear = 500
        drive.current_angular = 0
        self.assertEqual((500, 500), drive._wheel_targets())

    def test_stationary_output_is_not_repeated(self):
        bus = FakeBus()
        drive = DriveBase(bus, config)
        base = drive.last_update
        drive.update(base + config.CONTROL_PERIOD_MS)
        first_count = len(bus.drives)
        drive.update(base + config.CONTROL_PERIOD_MS * 2)
        self.assertEqual(first_count, len(bus.drives))

    def test_calibration_reduces_left_and_boosts_right(self):
        bus = FakeBus()
        calibration = DriveCalibration(0.05, 0.0)
        drive = DriveBase(bus, config, calibration)
        left, right = drive._speed_outputs(400, 400, 0.02, drive.last_update)
        self.assertLess(left, right)

    def test_individual_motor_gains_boost_front_left_and_limit_rear_left(self):
        bus = FakeBus()
        drive = DriveBase(bus, config)
        drive.set_twist(
            100,
            0,
            600,
            drive.last_update,
            motor_output_gains=(1.15, 1.0, 0.90, 1.0),
        )

        forward = drive._motor_outputs(400, 400)
        reverse = drive._motor_outputs(-400, -400)

        self.assertEqual((460, 400, 360, 400), forward)
        self.assertEqual((-460, -400, -360, -400), reverse)
        self.assertGreater(abs(forward[0]), abs(forward[2]))

    def test_individual_motor_gain_preserves_low_speed_deadzone(self):
        bus = FakeBus()
        drive = DriveBase(bus, config)

        outputs = drive._motor_outputs(90, 90)

        self.assertTrue(all(
            abs(value) >= config.MIN_EFFECTIVE_MOTOR_UNITS
            for value in outputs
        ))


class IndicatorTest(unittest.TestCase):
    def test_new_pattern_preempts_old_pattern(self):
        led_values = []
        beep_values = []
        indicator = IndicatorController(led_values.append, beep_values.append)
        base = ticks_ms()
        indicator.set_led(LED_BLINK, 100, base)
        indicator.beep(3, 100, 100, base)
        indicator.update(base + 100)
        self.assertTrue(indicator.buzzer_active)
        indicator.beep(1, 20, 20, base + 101)
        indicator.update(base + 121)
        self.assertFalse(indicator.buzzer_active)
        indicator.cancel_all()
        self.assertFalse(indicator.led_value)


if __name__ == "__main__":
    unittest.main()
