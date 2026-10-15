import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from raspberry_pi.lidar import LaserScan  # noqa: E402
from raspberry_pi.mapping_test import (  # noqa: E402
    build_argument_parser,
    filter_scan_view,
    run,
)


class FakeLidar:
    def __init__(self, scans):
        self._scans = scans
        self.closed = False

    def scans(self):
        yield from self._scans

    def close(self):
        self.closed = True


class MappingTestScriptTest(unittest.TestCase):
    def test_defaults_match_raspberry_pi_acm_device(self):
        args = build_argument_parser().parse_args([])
        self.assertEqual("/dev/ttyACM0", args.port)
        self.assertEqual(180.0, args.field_of_view)
        self.assertGreater(args.count, 0)

    def test_finite_test_saves_map_and_closes_lidar(self):
        angles = np.linspace(-1.5, 1.5, 140)
        distances = 2.0 + 0.2 * np.sin(angles * 3.0)
        scans = (
            LaserScan(angles, distances, 1.0),
            LaserScan(angles, distances, 2.0),
        )
        lidar = FakeLidar(scans)
        with tempfile.TemporaryDirectory() as directory:
            args = build_argument_parser().parse_args(
                [
                    "--count", "2",
                    "--map-size", "10",
                    "--output", os.path.join(directory, "map"),
                    "--save-every", "0",
                ]
            )
            paths = run(args, lidar=lidar)
            self.assertTrue(lidar.closed)
            self.assertTrue(all(path.exists() for path in paths))
            with paths[2].open(encoding="utf-8") as stream:
                self.assertEqual(3, sum(1 for _line in stream))

    def test_front_180_degree_filter_removes_rear_points(self):
        angles = np.radians((-180, -90, -45, 0, 45, 90, 135))
        scan = LaserScan(angles, np.ones(len(angles)), 1.0)
        filtered = filter_scan_view(scan, field_of_view_deg=180, center_deg=0)
        np.testing.assert_allclose(
            filtered.angles_rad,
            np.radians((-90, -45, 0, 45, 90)),
        )


if __name__ == "__main__":
    unittest.main()
