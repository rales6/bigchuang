"""No-hardware regression checks for direct web turn control."""

from __future__ import annotations

import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from .bounded_motion import (
    _arc_clearance, _arc_next_scan_would_overshoot, _arc_stop_lead_deg,
    _next_turn_speed, _short_arc_retreat, _short_reverse_pulse_s,
    _verified_reverse_increment,
    _verified_settle_increment, _verified_turn_increment,
)


def matched_yaw(degrees: float, *, converged: bool = True,
                rmse_m: float = 0.04, inlier_ratio: float = 0.8):
    return SimpleNamespace(
        converged=converged, rmse_m=rmse_m, inlier_ratio=inlier_ratio,
        transform=SimpleNamespace(yaw_rad=math.radians(degrees)),
    )


class BoundedTurnTests(unittest.TestCase):
    def test_short_reverse_requires_backward_straight_displacement(self) -> None:
        match = SimpleNamespace(
            converged=True, rmse_m=0.04, inlier_ratio=0.8,
            transform=SimpleNamespace(x_m=-0.08, y_m=0.01, yaw_rad=0.01),
        )
        self.assertAlmostEqual(_verified_reverse_increment(match), 0.08)
        match.transform.x_m = 0.08
        self.assertEqual(_verified_reverse_increment(match), 0.0)
        match.transform.x_m = -0.08
        match.transform.y_m = 0.08
        self.assertEqual(_verified_reverse_increment(match), 0.0)
        self.assertLessEqual(_short_reverse_pulse_s(0.10), 0.22)
        self.assertGreaterEqual(_short_reverse_pulse_s(0.05), 0.15)

    def test_adaptive_stop_lead_uses_observed_coast_and_scan_margin(self) -> None:
        self.assertEqual(_arc_stop_lead_deg(180, []), 16.0)
        self.assertAlmostEqual(_arc_stop_lead_deg(180, [15.76, 17.46, 16.32]), 21.46)
        self.assertAlmostEqual(_arc_stop_lead_deg(30, []), 13.5)
        self.assertTrue(_arc_next_scan_would_overshoot(
            math.radians(157.41), math.radians(158.54),
            math.radians(149.74), math.radians(2.93),
        ))
        self.assertFalse(_arc_next_scan_would_overshoot(
            0.0, math.radians(10), 0.0, 0.0,
        ))

    def test_short_retreat_is_single_bounded_pulse_and_ignores_rear_echo(self) -> None:
        angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
        before_distances = np.ones(360, dtype=float)
        left = np.abs(np.arctan2(np.sin(angles - math.pi / 2), np.cos(angles - math.pi / 2))) < math.radians(25)
        rear = np.abs(np.arctan2(np.sin(angles - math.pi), np.cos(angles - math.pi))) < math.radians(25)
        before_distances[left] = 0.3
        before_distances[rear] = 0.2
        after_distances = np.ones(360, dtype=float)
        after_distances[rear] = 0.2
        before = SimpleNamespace(angles_rad=angles, distances_m=before_distances)
        after = SimpleNamespace(angles_rad=angles, distances_m=after_distances,
                                points=lambda *_: range(60))
        calls = []
        client = SimpleNamespace(
            stop=lambda: calls.append(("stop",)),
            set_twist=lambda *a, **kw: calls.append(("twist", *a, kw)),
        )
        match = SimpleNamespace(
            converged=True, rmse_m=0.04, inlier_ratio=0.8,
            transform=SimpleNamespace(x_m=0.05, y_m=0.0),
        )
        localizer = SimpleNamespace(relative_motion=lambda *_: match)
        with patch("final_work.pi.bounded_motion.time.sleep"):
            result = _short_arc_retreat(client, iter([after]), localizer, before, "left", 0.55)
        self.assertIs(result, after)
        self.assertEqual([call[0] for call in calls], ["stop", "twist", "stop"])
        self.assertEqual(calls[1][1:3], (-405, 0))
        self.assertEqual(calls[1][3]["ttl_ms"], 220)
        match.transform.x_m = 0.15
        with patch("final_work.pi.bounded_motion.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "超过允许范围"):
                _short_arc_retreat(client, iter([after]), localizer, before, "left", 0.55)

    def test_scan_jitter_and_wrong_direction_are_not_progress(self) -> None:
        self.assertEqual(_verified_turn_increment(matched_yaw(0.8), "left"), 0.0)
        self.assertEqual(_verified_turn_increment(matched_yaw(-2.0), "left"), 0.0)
        self.assertEqual(_verified_turn_increment(matched_yaw(2.0), "right"), 0.0)

    def test_only_good_directional_yaw_counts(self) -> None:
        self.assertAlmostEqual(math.degrees(_verified_turn_increment(matched_yaw(2.5), "left")), 2.5)
        self.assertAlmostEqual(math.degrees(_verified_turn_increment(matched_yaw(-2.5), "right")), 2.5)
        self.assertEqual(_verified_turn_increment(matched_yaw(30.0), "left"), 0.0)
        self.assertEqual(_verified_turn_increment(matched_yaw(2.5, converged=False), "left"), 0.0)

    def test_stop_latency_yaw_is_counted_before_another_segment(self) -> None:
        self.assertAlmostEqual(math.degrees(_verified_settle_increment(matched_yaw(10), "left")), 10)
        self.assertAlmostEqual(math.degrees(_verified_settle_increment(matched_yaw(-8), "right")), 8)
        self.assertEqual(_verified_settle_increment(matched_yaw(0.5), "left"), 0.0)
        with self.assertRaisesRegex(RuntimeError, "无法确认"):
            _verified_settle_increment(matched_yaw(10, converged=False), "left")
        with self.assertRaisesRegex(RuntimeError, "异常"):
            _verified_settle_increment(matched_yaw(30), "left")

    def test_static_friction_boost_is_bounded(self) -> None:
        self.assertEqual(_next_turn_speed(2100, observed_rate_deg_s=0, maximum=2400,
                                          remaining_deg=180), 2300)
        self.assertEqual(_next_turn_speed(2300, observed_rate_deg_s=2, maximum=2400,
                                          remaining_deg=170), 2400)
        self.assertEqual(_next_turn_speed(2400, observed_rate_deg_s=0, maximum=2400,
                                          remaining_deg=160), 2400)
        self.assertEqual(_next_turn_speed(2100, observed_rate_deg_s=12, maximum=2400,
                                          remaining_deg=180), 2100)
        self.assertEqual(_next_turn_speed(2300, observed_rate_deg_s=0, maximum=2400,
                                          remaining_deg=8), 2300)

    def test_arc_guard_checks_front_and_turning_side_not_rear(self) -> None:
        angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
        distances = np.ones(360, dtype=float)
        rear = np.abs(np.arctan2(np.sin(angles - math.pi), np.cos(angles - math.pi))) < math.radians(25)
        distances[rear] = 0.2  # Rear lidar can see the robot's own body.
        scan = SimpleNamespace(angles_rad=angles, distances_m=distances)
        self.assertTrue(_arc_clearance("left", scan, 0.55)[0])
        left = np.abs(np.arctan2(np.sin(angles - math.pi / 2), np.cos(angles - math.pi / 2))) < math.radians(25)
        distances[left] = 0.3
        self.assertFalse(_arc_clearance("left", scan, 0.55)[0])
        self.assertTrue(_arc_clearance("right", scan, 0.55)[0])


if __name__ == "__main__":
    unittest.main()
