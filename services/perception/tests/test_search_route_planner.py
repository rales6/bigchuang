from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from raspberry_pi.known_map_navigation import KnownOccupancyMap
from raspberry_pi.search_route_planner import build_search_route
from raspberry_pi.object_search_simulation import simulate_waypoint_tracking


class SearchRoutePlannerTests(unittest.TestCase):
    def setUp(self):
        image = np.full((140, 180), 254, dtype=np.uint8)
        image[[0, -1], :] = 0
        image[:, [0, -1]] = 0
        image[55:85, 78:102] = 0
        self.known = KnownOccupancyMap(image, 0.05, -4.5, -3.5, clearance_m=0.20)

    def test_around_landmark_produces_reachable_views(self):
        route = build_search_route(
            self.known,
            (-3.5, -2.5),
            {"mode": "around_landmark", "anchor_xy_m": [0.0, 0.0], "view_distance_m": 0.75, "maximum_views": 6},
        )
        self.assertGreaterEqual(len(route["views"]), 3)
        self.assertTrue(all(item["path_xy_m"] for item in route["views"]))
        self.assertFalse(route["trajectory_used"])

    def test_region_route_stays_in_requested_quadrant(self):
        route = build_search_route(
            self.known,
            (0.0, 0.0),
            {"mode": "coarse_region", "region_hint": "bottom_right", "maximum_views": 6},
        )
        self.assertGreater(len(route["views"]), 0)
        self.assertTrue(all(item["view_xy_m"][0] >= -0.55 for item in route["views"]))
        self.assertTrue(all(item["view_xy_m"][1] <= 0.5 for item in route["views"]))

    def test_semantic_search_region_generates_views_without_treating_center_as_goal(self):
        center = (1.5, 1.5)
        route = build_search_route(
            self.known,
            (0.0, 0.0),
            {
                "mode": "semantic_search_region",
                "center_xy_m": list(center),
                "radius_m": 1.0,
                "maximum_views": 8,
                "require_live_visual_recheck": True,
            },
        )
        self.assertGreater(len(route["views"]), 0)
        self.assertTrue(all(
            0.15 < ((item["view_xy_m"][0] - center[0]) ** 2
                    + (item["view_xy_m"][1] - center[1]) ** 2) ** 0.5 <= 1.1
            for item in route["views"]
        ))

    def test_low_speed_controller_reaches_all_route_waypoints(self):
        route = build_search_route(
            self.known,
            (-3.5, -2.5),
            {"mode": "around_landmark", "anchor_xy_m": [0.0, 0.0], "view_distance_m": 0.75, "maximum_views": 6},
        )
        result = simulate_waypoint_tracking(self.known, route)
        self.assertTrue(result["reached"], result)
        self.assertLessEqual(result["maximum_linear_mm_s"], 240)
        self.assertLessEqual(result["maximum_angular_mrad_s"], 1250)


if __name__ == "__main__":
    unittest.main()
