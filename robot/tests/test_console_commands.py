import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.pi_command_console import CommandConsole  # noqa: E402


class FakeClient:
    def __init__(self):
        self.calls = []

    def cancel(self, mask):
        self.calls.append(("cancel", mask))

    def set_drive(self, linear, angular, ttl=600):
        self.calls.append(("drive", linear, angular, ttl))

    def stop_drive(self):
        self.calls.append(("stop_drive",))

    def set_arm(self, joints, duration=800):
        self.calls.append(("arm", joints, duration))

    def set_led(self, mode, period=0):
        self.calls.append(("led", mode, period))

    def beep(self, repeat, on_ms, off_ms):
        self.calls.append(("beep", repeat, on_ms, off_ms))

    def request(self, msg_type, payload=b""):
        self.calls.append(("request", msg_type, payload))

    def query_status(self):
        return {
            "uptime_ms": 1, "flags": 0, "linear_mm_s": 0,
            "angular_mrad_s": 0, "left_output": 0, "right_output": 0,
            "wheel_feedback": (0, 0, 0, 0),
            "joint_positions": (1500,) * 6,
            "battery_mv": 0, "bus_errors": 0,
        }


class ConsoleCommandTest(unittest.TestCase):
    def test_drive_name_maps_to_expected_twist(self):
        client = FakeClient()
        console = CommandConsole(client)
        console.execute("forward")
        self.assertIn(("drive", 350, 0, 600), client.calls)

    def test_custom_arm_command_maps_to_joint_target(self):
        client = FakeClient()
        console = CommandConsole(client)
        console.execute("arm 2 1900 700")
        self.assertIn(("arm", [(2, 1900)], 700), client.calls)

    def test_io_command_mappings(self):
        client = FakeClient()
        console = CommandConsole(client)
        console.execute("led_blink_fast")
        console.execute("beep_3")
        self.assertIn(("led", 2, 100), client.calls)
        self.assertIn(("beep", 3, 120, 120), client.calls)


if __name__ == "__main__":
    unittest.main()

