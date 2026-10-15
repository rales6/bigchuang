import os
import sys
import unittest
import time
import threading
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from car.protocol.frame import (  # noqa: E402
    ADDR_ESP32,
    ADDR_RASPBERRY_PI,
    FLAG_RESPONSE,
    FrameParser,
    encode_frame,
)
from car.protocol.messages import (  # noqa: E402
    MSG_ACK,
    MSG_DRIVE_CALIBRATION,
    MSG_HEARTBEAT,
    MSG_QUERY_STATUS,
    MSG_QUERY_DRIVE_CALIBRATION,
    MSG_ROBOT_STATUS,
    MSG_SET_BALANCED_TWIST,
    MSG_SET_TWIST,
    MSG_STOP,
    decode_balanced_twist,
    encode_ack,
    encode_drive_calibration,
    encode_robot_status,
)
from raspberry_pi.config import SerialConfig  # noqa: E402
from raspberry_pi.esp32 import Esp32Client  # noqa: E402


STATUS = {
    "uptime_ms": 12345,
    "flags": 0x0008,
    "linear_mm_s": 0,
    "angular_mrad_s": 0,
    "left_output": 0,
    "right_output": 0,
    "wheel_feedback": (0, 0, 0, 0),
    "joint_positions": (1500, 1700, 2000, 1100, 1500, 1200),
    "battery_mv": 0,
    "bus_errors": 0,
}


class FakeEsp32Serial:
    def __init__(self):
        self.request_parser = FrameParser()
        self.requests = []
        self.rx = bytearray()
        self.closed = False

    @property
    def in_waiting(self):
        return len(self.rx)

    def write(self, data):
        for request in self.request_parser.feed(data):
            self.requests.append(request)
            if not request.ack_required:
                continue
            if request.msg_type == MSG_QUERY_STATUS:
                response_type = MSG_ROBOT_STATUS
                payload = encode_robot_status(STATUS)
            elif request.msg_type == MSG_QUERY_DRIVE_CALIBRATION:
                response_type = MSG_DRIVE_CALIBRATION
                payload = encode_drive_calibration(0.02, 0.0001)
            else:
                response_type = MSG_ACK
                payload = encode_ack(request.msg_type)
            self.rx.extend(
                encode_frame(
                    response_type,
                    request.seq,
                    payload,
                    src=ADDR_ESP32,
                    dst=ADDR_RASPBERRY_PI,
                    flags=FLAG_RESPONSE,
                )
            )
        return len(data)

    def read(self, count):
        data = bytes(self.rx[:count])
        del self.rx[:count]
        return data

    def close(self):
        self.closed = True


class Esp32ClientTest(unittest.TestCase):
    def test_bounded_command_installs_absolute_deadline_not_unlimited_refresh(self):
        client, serial = self.make_client()
        with patch('raspberry_pi.esp32.client.time.monotonic', return_value=10.):
            client.set_bounded_twist(420, 0, 350, 10.2)
            self.assertEqual(client._active_drive_refresh_deadline_s, 10.2)
        expected = client._active_drive_request
        client._running = True
        before = len(serial.requests)
        with patch('raspberry_pi.esp32.client.time.monotonic', return_value=10.3):
            self.assertFalse(client._refresh_active_request(expected))
        self.assertEqual(len(serial.requests), before)
        self.assertIsNone(client._active_drive_request)
        client.close(send_cancel=False)

    def test_slow_ack_cannot_rearm_expired_bounded_motion(self):
        client, _serial = self.make_client()
        now = [10.]
        def delayed_request(*_args):
            now[0] = 10.6
            return object()
        with patch.object(client, 'request', side_effect=delayed_request), patch(
            'raspberry_pi.esp32.client.time.monotonic', side_effect=lambda: now[0]
        ):
            client.set_bounded_twist(420, 0, 350, 10.2)
        self.assertIsNone(client._active_drive_request)
        client.close(send_cancel=False)

    def test_stop_during_bounded_ack_wins_and_no_old_refresh_follows(self):
        client, serial = self.make_client()
        entered, release = threading.Event(), threading.Event()
        original_write = serial.write
        def delayed_write(data):
            result = original_write(data)
            if serial.requests[-1].msg_type in (MSG_SET_BALANCED_TWIST, MSG_SET_TWIST):
                entered.set()
                release.wait(1.)
            return result
        serial.write = delayed_write
        errors = []
        def move():
            try:
                client.set_bounded_twist(420, 0, 350, time.monotonic() + .8)
            except Exception as exc:
                errors.append(exc)
        moving = threading.Thread(target=move)
        stopping = threading.Thread(target=client.stop)
        moving.start()
        self.assertTrue(entered.wait(1.))
        stopping.start()
        deadline = time.monotonic() + 1.
        while client._drive_request_generation < 2 and time.monotonic() < deadline:
            time.sleep(.001)
        release.set()
        moving.join(1.)
        stopping.join(1.)
        self.assertFalse(moving.is_alive() or stopping.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(serial.requests[-1].msg_type, MSG_STOP)
        self.assertIsNone(client._active_drive_request)
        client.close(send_cancel=False)

    def test_bounded_deadline_must_be_finite(self):
        client, serial = self.make_client()
        with self.assertRaises(ValueError):
            client.set_bounded_twist(420, 0, 350, float('inf'))
        self.assertEqual(serial.requests, [])
        client.close(send_cancel=False)

    def make_client(self):
        serial = FakeEsp32Serial()
        config = SerialConfig(request_timeout_s=0.01, retries=0)
        return Esp32Client(config, serial), serial

    def test_heartbeat_uses_shared_protocol(self):
        client, _serial = self.make_client()
        response = client.heartbeat()
        self.assertEqual(MSG_ACK, response.msg_type)
        self.assertEqual(MSG_HEARTBEAT, response.payload[0])

    def test_nonunity_wheel_gains_are_sent_by_raspberry_pi(self):
        serial = FakeEsp32Serial()
        config = SerialConfig(
            request_timeout_s=0.01,
            retries=0,
            wheel_output_gains=(1.15, 1.0, 0.90, 1.0),
        )
        client = Esp32Client(config, serial)

        client.set_twist(300, 0, 450)

        request = serial.requests[-1]
        self.assertEqual(MSG_SET_BALANCED_TWIST, request.msg_type)
        self.assertEqual(
            (300, 0, 450, (1.15, 1.0, 0.90, 1.0)),
            decode_balanced_twist(request.payload),
        )

    def test_identical_motion_refresh_does_not_wait_for_ack(self):
        serial = FakeEsp32Serial()
        config = SerialConfig(
            request_timeout_s=0.01,
            retries=0,
            wheel_output_gains=(1.15, 1.0, 0.90, 1.0),
        )
        client = Esp32Client(config, serial)

        client.set_twist(300, 0, 700)
        client.refresh_twist(300, 0, 700)

        self.assertTrue(serial.requests[-2].ack_required)
        self.assertFalse(serial.requests[-1].ack_required)
        self.assertEqual(0, serial.in_waiting)

    def test_bounded_turn_can_be_left_to_expire_without_background_refresh(self):
        serial = FakeEsp32Serial()
        config = SerialConfig(
            request_timeout_s=0.01,
            retries=0,
            keepalive_period_s=0.02,
            wheel_output_gains=(1.15, 1.0, 0.90, 1.0),
        )
        client = Esp32Client(config, serial)
        client.start()
        client.set_twist(0, 1250, 350)
        client.allow_twist_to_expire()
        turn_index = len(serial.requests) - 1
        time.sleep(0.07)
        self.assertFalse(any(
            frame.msg_type == MSG_SET_BALANCED_TWIST
            for frame in serial.requests[turn_index + 1:]
        ))
        client.close(send_cancel=False)

    def test_turn_refresh_window_supplies_torque_then_expires(self):
        serial = FakeEsp32Serial()
        config = SerialConfig(
            request_timeout_s=0.01,
            retries=0,
            keepalive_period_s=0.02,
            wheel_output_gains=(1.15, 1.0, 0.90, 1.0),
        )
        client = Esp32Client(config, serial)
        client.start()
        client.set_twist(0, 2400, 350)
        self.assertTrue(client.allow_twist_to_refresh_for(0.07))
        time.sleep(0.055)
        refreshes_during_window = sum(
            frame.msg_type == MSG_SET_BALANCED_TWIST
            and not frame.ack_required
            for frame in serial.requests
        )
        self.assertGreaterEqual(refreshes_during_window, 1)
        time.sleep(0.09)
        refreshes_after_expiry = sum(
            frame.msg_type == MSG_SET_BALANCED_TWIST
            and not frame.ack_required
            for frame in serial.requests
        )
        time.sleep(0.06)
        self.assertEqual(
            refreshes_after_expiry,
            sum(
                frame.msg_type == MSG_SET_BALANCED_TWIST
                and not frame.ack_required
                for frame in serial.requests
            ),
        )
        client.close(send_cancel=False)

    def test_background_keepalive_is_continuous_and_stop_wins_race(self):
        serial = FakeEsp32Serial()
        config = SerialConfig(
            request_timeout_s=0.01,
            retries=0,
            keepalive_period_s=0.02,
            wheel_output_gains=(1.15, 1.0, 0.90, 1.0),
        )
        client = Esp32Client(config, serial)
        self.assertFalse(client.maintenance_running)
        client.start()
        self.assertTrue(client.maintenance_running)
        client.set_twist(400, 0, 650)
        time.sleep(0.09)
        refreshes = [
            frame for frame in serial.requests
            if frame.msg_type == MSG_SET_BALANCED_TWIST and not frame.ack_required
        ]
        self.assertGreaterEqual(len(refreshes), 2)
        client.stop()
        stop_index = max(
            index for index, frame in enumerate(serial.requests)
            if frame.msg_type == MSG_STOP
        )
        time.sleep(0.06)
        self.assertFalse(any(
            frame.msg_type == MSG_SET_BALANCED_TWIST
            for frame in serial.requests[stop_index + 1:]
        ))
        client.close(send_cancel=False)
        self.assertFalse(client.maintenance_running)

    def test_query_status_decodes_payload(self):
        client, _serial = self.make_client()
        self.assertEqual(STATUS, client.query_status())

    def test_close_cancels_and_closes_serial(self):
        client, serial = self.make_client()
        client.close()
        self.assertTrue(serial.closed)

    def test_drive_calibration_query_uses_typed_response(self):
        client, _serial = self.make_client()
        calibration = client.query_drive_calibration()
        self.assertAlmostEqual(0.02, calibration["trim_intercept"], places=6)
        self.assertAlmostEqual(
            0.0001,
            calibration["trim_slope_per_mm_s"],
            places=8,
        )


if __name__ == "__main__":
    unittest.main()
