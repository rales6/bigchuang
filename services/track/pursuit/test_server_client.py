from __future__ import annotations

import json
import unittest

from pursuit.pursuit_config import CameraConfig
from pursuit.run_visual_pursuit import ServerClient, build_parser


class _FakeWebSocket:
    def __init__(self) -> None:
        self.timeout = 1.0
        self.sent: list[tuple[str, object]] = []

    def settimeout(self, value: float) -> None:
        self.timeout = value

    def recv(self) -> str:
        if self.timeout <= 0.02:
            raise TimeoutError("timed out")
        return json.dumps({"type": "frame_result", "frame_id": 1})

    def send(self, value: str) -> None:
        if self.timeout < 0.5:
            raise TimeoutError("send inherited recv timeout")
        self.sent.append(("text", value))

    def send_binary(self, value: bytes) -> None:
        if self.timeout < 0.5:
            raise TimeoutError("binary send inherited recv timeout")
        self.sent.append(("binary", value))


class ServerClientTimeoutTests(unittest.TestCase):
    def test_recv_timeout_is_restored_before_text_send(self) -> None:
        client = ServerClient("http://example", "ws://example")
        socket = _FakeWebSocket()
        client._ws = socket
        self.assertIsNone(client.recv_result())
        self.assertEqual(socket.timeout, 1.0)
        client.send_text({"type": "motion"})
        self.assertFalse(client.dead())
        self.assertEqual(socket.sent[0][0], "text")

    def test_binary_send_keeps_normal_send_timeout(self) -> None:
        client = ServerClient("http://example", "ws://example")
        socket = _FakeWebSocket()
        client._ws = socket
        self.assertTrue(client.send_frame(b"jpeg"))
        self.assertEqual(socket.timeout, 1.0)

    def test_vehicle_camera_mapping_matches_current_ports(self) -> None:
        self.assertTrue(CameraConfig().swap_left_right)
        args = build_parser().parse_args(["--instruction", "target"])
        self.assertIsNone(args.swap_cameras)
        args = build_parser().parse_args(
            ["--instruction", "target", "--no-swap-cameras"]
        )
        self.assertFalse(args.swap_cameras)


if __name__ == "__main__":
    unittest.main()
