import json
import os
import sys
import time
import unittest
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from raspberry_pi.mapping import OccupancyGrid  # noqa: E402
from raspberry_pi.realtime_map import RealtimeMapServer  # noqa: E402


class RealtimeMapServerTest(unittest.TestCase):
    @staticmethod
    def _wait_for_map(server, timeout_s=2.0):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if server._store.png() is not None:
                return
            time.sleep(0.01)
        raise AssertionError("asynchronous map render timed out")

    def _slam(self):
        grid = OccupancyGrid(40, 30, 0.05)
        grid.observed[12:18, 10:30] = True
        grid.log_odds[12, 10:30] = 4.0
        return SimpleNamespace(
            grid=grid,
            trajectory=[
                (1.0, 0.00, 0.00, 0.0),
                (2.0, 0.20, 0.00, 0.0),
            ],
        )

    def _status(self, scan_count=2):
        return {
            "running": True,
            "scan_count": scan_count,
            "accepted_count": scan_count,
            "rejected_count": 0,
            "accepted_ratio": 1.0,
            "accepted": True,
            "pose": {"x_m": 0.2, "y_m": 0.0, "yaw_deg": 0.0},
            "rmse_m": 0.03,
            "correspondences": 80,
            "scan_points": 120,
            "inlier_ratio": 0.8,
            "map_status": "integrated",
            "linear_speed_m_s": 0.1,
            "angular_speed_rad_s": 0.0,
            "command": {
                "state": "advancing",
                "reason": "test",
                "linear_mm_s": 250,
                "angular_mrad_s": 0,
            },
            "mapping": {"integrated_count": scan_count},
            "elapsed_s": 1.0,
        }

    def test_serves_dashboard_status_and_in_memory_png(self):
        server = RealtimeMapServer(
            bind="127.0.0.1",
            port=0,
            refresh_hz=10.0,
        )
        try:
            server.start()
            self.assertTrue(server.publish(
                self._slam(),
                self._status(),
                force=True,
            ))
            self._wait_for_map(server)
            with urlopen(server.local_url + "/", timeout=2.0) as response:
                html = response.read().decode("utf-8")
            self.assertIn("实车建图 · Live SLAM", html)
            self.assertIn("不使用仿真真值", html)

            with urlopen(
                server.local_url + "/api/status",
                timeout=2.0,
            ) as response:
                status = json.loads(response.read().decode("utf-8"))
            self.assertEqual(2, status["scan_count"])
            self.assertTrue(status["map_available"])
            self.assertGreater(status["map"]["observed_cells"], 0)
            self.assertGreater(status["map_version"], 0)

            with urlopen(
                server.local_url + "/api/map.png",
                timeout=2.0,
            ) as response:
                png = response.read()
            self.assertEqual(b"\x89PNG\r\n\x1a\n", png[:8])
        finally:
            server.stop()

    def test_throttles_png_but_keeps_metrics_current(self):
        server = RealtimeMapServer(
            bind="127.0.0.1",
            port=0,
            refresh_hz=0.2,
        )
        try:
            server.start()
            slam = self._slam()
            self.assertTrue(server.publish(
                slam,
                self._status(1),
                force=True,
            ))
            self._wait_for_map(server)
            self.assertFalse(server.publish(slam, self._status(3)))
            with urlopen(
                server.local_url + "/api/status",
                timeout=2.0,
            ) as response:
                status = json.loads(response.read().decode("utf-8"))
            self.assertEqual(3, status["scan_count"])
            self.assertEqual(1, status["map_version"])
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
