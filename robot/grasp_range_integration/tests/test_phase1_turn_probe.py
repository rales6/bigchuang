from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from grasp_range_integration.phase1.probe_bounded_turn import _turn_clearance


class TurnClearanceTests(unittest.TestCase):
    def test_repeated_clipped_self_returns_do_not_hide_valid_clearance(self):
        scan = SimpleNamespace(
            angles_rad=np.radians(np.linspace(-180, 180, 40)),
            distances_m=np.array([0.104] * 20 + [0.7] * 20),
        )
        clearance, count = _turn_clearance(scan)
        self.assertEqual(count, 20)
        self.assertAlmostEqual(clearance, 0.7)

    def test_real_close_obstacle_still_blocks_turn(self):
        scan = SimpleNamespace(
            angles_rad=np.radians(np.linspace(-180, 180, 41)),
            distances_m=np.array([0.104] * 20 + [0.22] + [0.7] * 20),
        )
        clearance, _ = _turn_clearance(scan)
        self.assertAlmostEqual(clearance, 0.22)

    def test_few_clipped_returns_are_not_ignored(self):
        scan = SimpleNamespace(
            angles_rad=np.radians([-10, 0, 10]),
            distances_m=np.array([0.104, 0.104, 0.8]),
        )
        clearance, _ = _turn_clearance(scan)
        self.assertAlmostEqual(clearance, 0.104)

    def test_lone_rear_chassis_echo_is_excluded_but_side_obstacle_is_not(self):
        angles = np.radians(np.concatenate((
            np.linspace(-180, -90, 20), [170, 0, 90], np.linspace(0, 150, 19)
        )))
        scan = SimpleNamespace(
            angles_rad=angles,
            distances_m=np.array([0.104] * 20 + [0.19, 0.7, 0.22] + [0.7] * 19),
        )
        # The side 0.22m obstacle still blocks the turn.
        clearance, _ = _turn_clearance(scan)
        self.assertAlmostEqual(clearance, 0.22)

    def test_narrow_side_chassis_echo_is_excluded_only_with_clipping_cluster(self):
        angles = np.radians(np.concatenate((
            np.linspace(-180, -110, 20), [-92, -80, 0], np.linspace(0, 150, 19)
        )))
        scan = SimpleNamespace(
            angles_rad=angles,
            distances_m=np.array([0.104] * 20 + [0.135, 0.30, 0.8] + [0.8] * 19),
        )
        clearance, _ = _turn_clearance(scan)
        self.assertAlmostEqual(clearance, 0.30)

        scan.distances_m[:20] = 0.8
        clearance, _ = _turn_clearance(scan)
        self.assertAlmostEqual(clearance, 0.135)


if __name__ == "__main__":
    unittest.main()
