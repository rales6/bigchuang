import math
import unittest

from car_sim.headless_autonomous import (
    _confined_motion_metrics,
    _obstacle_view_sectors,
    _truth_at_timestamp,
    configure_scene,
)


class HeadlessAutonomousScoreTests(unittest.TestCase):
    def tearDown(self):
        configure_scene("full")

    def test_rejected_slam_frames_are_aligned_by_timestamp(self):
        truth = [
            {"timestamp_s": 0.1, "yaw_rad": 0.0},
            {"timestamp_s": 0.2, "yaw_rad": 0.2},
            {"timestamp_s": 0.6, "yaw_rad": 1.1},
            {"timestamp_s": 0.7, "yaw_rad": 1.2},
        ]

        matched = _truth_at_timestamp(truth, 0.61)

        self.assertEqual(0.6, matched["timestamp_s"])
        self.assertEqual(1.1, matched["yaw_rad"])

    def test_obstacle_views_require_positions_around_multiple_sides(self):
        configure_scene("single-rectangle")
        truth = []
        for sector in range(8):
            angle = math.tau * (sector + 0.5) / 12.0
            truth.append({
                "x_m": 3.0 + 1.1 * math.cos(angle),
                "y_m": 2.0 + 1.1 * math.sin(angle),
            })

        self.assertEqual([8], _obstacle_view_sectors(truth))

    def test_confined_metric_detects_repeated_local_rotation(self):
        truth = [
            {
                "timestamp_s": index * 0.1,
                "x_m": 0.1 * math.cos(index * 0.2),
                "y_m": 0.1 * math.sin(index * 0.2),
                "yaw_rad": index * 0.2,
            }
            for index in range(201)
        ]

        confined_s, yaw_deg = _confined_motion_metrics(truth)

        self.assertAlmostEqual(20.0, confined_s)
        self.assertGreater(yaw_deg, 2000.0)

    def test_confined_metric_does_not_label_new_view_sectors_as_stall(self):
        truth = [
            {
                "timestamp_s": index * 0.1,
                "x_m": 0.1 * math.cos(index * 0.2),
                "y_m": 0.1 * math.sin(index * 0.2),
                "yaw_rad": index * 0.2,
            }
            for index in range(201)
        ]

        confined_s, yaw_deg = _confined_motion_metrics(
            truth,
            progress_indices=tuple(range(30, 201, 30)),
        )

        self.assertLessEqual(confined_s, 3.0)
        self.assertLess(yaw_deg, 360.0)


if __name__ == "__main__":
    unittest.main()
