import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "car"))

from app import ExecutorApplication  # noqa: E402
from drivers.indicators import IndicatorController  # noqa: E402
from protocol.frame import (  # noqa: E402
    ADDR_ESP32,
    ADDR_RASPBERRY_PI,
    FLAG_ACK_REQUIRED,
    FrameParser,
    encode_frame,
)
from protocol.messages import (  # noqa: E402
    LED_ON,
    MSG_ACK,
    MSG_SET_LED,
    MSG_SET_TWIST,
    encode_led,
    encode_twist,
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
        return len(data)


class FakeArm:
    def __init__(self):
        self.positions = [1500, 1700, 2000, 1100, 1500, 1200]
        self.targets = list(self.positions)
        self.moving = False

    def set_joints(self, _joints):
        self.moving = True

    def move_cartesian_delta(self, forward_mm=0, up_mm=0, left_deg=0,
                             claw_open=None, duration_ms=None):
        self.delta = (forward_mm, up_mm, left_deg, claw_open)
        self.moving = True
        return 1200

    def stop(self):
        self.moving = False

    def update(self, _now):
        pass

    def set_feedback(self, positions):
        self.positions = list(positions)


class ApplicationIntegrationTest(unittest.TestCase):
    def _make_app(self):
        pi = FakeTransport()
        actuator = FakeTransport()
        indicators = IndicatorController(lambda _value: None, lambda _value: None)
        return ExecutorApplication(pi, actuator, indicators, arm=FakeArm()), pi, actuator

    def test_main_step_accepts_pi_command_without_blocking(self):
        pi = FakeTransport()
        actuator = FakeTransport()
        led = []
        buzzer = []
        indicators = IndicatorController(led.append, buzzer.append)
        app = ExecutorApplication(pi, actuator, indicators, arm=FakeArm())
        now = app.controller.started_ms
        pi.rx = encode_frame(
            MSG_SET_LED, 77, encode_led(LED_ON),
            src=ADDR_RASPBERRY_PI, dst=ADDR_ESP32,
            flags=FLAG_ACK_REQUIRED,
        )
        app.step(now)
        responses = FrameParser().feed(pi.tx)
        self.assertEqual(1, len(responses))
        self.assertEqual(MSG_ACK, responses[0].msg_type)
        self.assertTrue(indicators.led_value)

    def test_pi_twist_reaches_legacy_motor_uart(self):
        app, pi, actuator = self._make_app()
        now = max(app.controller.started_ms, app.drive.last_update, app.bus.last_write_ms)
        pi.rx = encode_frame(
            MSG_SET_TWIST, 9, encode_twist(250, 0, 600),
            src=ADDR_RASPBERRY_PI, dst=ADDR_ESP32,
            flags=FLAG_ACK_REQUIRED,
        )
        app.step(now)
        app.step(now + 20)
        app.step(now + 25)
        self.assertIn(b"#006P", actuator.tx)
        self.assertIn(b"#007P", actuator.tx)
        self.assertIn(b"#008P", actuator.tx)
        self.assertIn(b"#009P", actuator.tx)

    def test_unacknowledged_twist_refresh_executes_without_ble_reply(self):
        app, pi, actuator = self._make_app()
        now = max(
            app.controller.started_ms,
            app.drive.last_update,
            app.bus.last_write_ms,
        )
        pi.rx = encode_frame(
            MSG_SET_TWIST,
            10,
            encode_twist(300, 0, 700),
            src=ADDR_RASPBERRY_PI,
            dst=ADDR_ESP32,
            flags=0,
        )

        app.step(now)
        app.step(now + 20)

        self.assertEqual(b"", pi.tx)
        self.assertEqual(300.0, app.drive.target_linear)

    def test_ascii_arm_command_reaches_controller(self):
        app, pi, _actuator = self._make_app()
        pi.rx = b"#ARMF100U050L000C020!"
        app.step(app.controller.started_ms)
        self.assertEqual((100, 50, 0, 20), app.arm.delta)
        self.assertIn(b"OK:ARM", pi.tx)


if __name__ == "__main__":
    unittest.main()
