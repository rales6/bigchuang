from __future__ import annotations

import math
from pathlib import Path
import unittest

from grasp_range_integration.phase1 import visibility_route_planner
from grasp_range_integration.phase1.visibility_route_planner import (
    POLICY,
    VISITED_POINTS,
    build_visibility_route,
    work_area_path,
)
from grasp_range_integration.phase1.navigation_avoidance import reserve_path
from raspberry_pi.known_map_navigation import KnownOccupancyMap


ROOT = Path(__file__).resolve().parents[2]
RELEASE = (
    ROOT / "maps/semantic_release_20260919"
    if (ROOT / "maps/semantic_release_20260919/ble_scene_17.pgm").is_file()
    else ROOT
)


class VisibilityRoutePlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.known_map = KnownOccupancyMap.load(
            RELEASE / "ble_scene_17.pgm",
            RELEASE / "ble_scene_17.yaml",
            clearance_m=0.30,
        )
        cls.start = (-0.955, -2.765)
        cls.center = (0.8751325645044502, 1.0443147835391624)
        cls.request = {
            "mode": "semantic_search_region",
            "planning_policy": POLICY,
            "center_xy_m": list(cls.center),
            "radius_m": 0.95,
            "maximum_views": 8,
            "camera_horizontal_fov_deg": 82.0,
            "inspect_current_view_first": True,
            "stop_on_visibility_entry": True,
        }

    def setUp(self):
        VISITED_POINTS.clear()
        visibility_route_planner.OBSERVATIONS_COMPLETED = 0

    def tearDown(self):
        VISITED_POINTS.clear()
        visibility_route_planner.OBSERVATIONS_COMPLETED = 0

    def test_current_view_is_checked_before_motion(self):
        route = build_visibility_route(self.known_map, self.start, self.request)
        first = route["views"][0]
        self.assertEqual(first["view_xy_m"], list(self.start))
        self.assertEqual(first["path_length_m"], 0.0)
        self.assertEqual(
            first["visibility_zone"]["stop_rule"], "capture_before_route_motion"
        )
        transit = [view for view in route["views"] if view["visibility_zone"]["band"] == "opportunistic_transit"]
        self.assertEqual(transit, [])

    def test_real_start_remains_reachable_with_turn_clearance_reserved(self):
        from grasp_range_integration.phase1.navigation_avoidance import reserve_path
        known = KnownOccupancyMap.load(RELEASE/'ble_scene_17.pgm', RELEASE/'ble_scene_17.yaml', clearance_m=.30)
        original = known.path
        known.path = lambda start, goal: reserve_path(known, start, goal, original)
        before = known.traversable.copy()
        route = build_visibility_route(known, (-1.27, -2.78), self.request)
        paths = [v['path_xy_m'] for v in route['views'] if v.get('target_yaw_rad') is not None]
        self.assertTrue(paths)
        self.assertTrue(paths[0])
        import numpy as np
        np.testing.assert_array_equal(before, known.traversable)

    def test_latest_real_start_is_not_task_search_but_planned_station_is(self):
        actual_start = [-1.27, -2.778, 2.122]  # Recorded 20260926_113029 pose; offline only.
        route = build_visibility_route(self.known_map, tuple(actual_start[:2]), self.request)
        gate = visibility_route_planner.task_region_view_assessment(
            route, actual_start, known_map=self.known_map, view=route['views'][0])
        self.assertFalse(gate['eligible'])
        self.assertEqual(gate['estimated_work_area_coverage'], 0.)
        for view in route['views']:
            if view['visibility_zone']['band'] not in {'high', 'medium', 'low'}:
                continue
            station_pose = list(view['view_xy_m']) + [view['target_yaw_rad']]
            gate = visibility_route_planner.task_region_view_assessment(
                route, station_pose, known_map=self.known_map, view=view)
            self.assertTrue(gate['eligible'], gate)

    def test_periodic_transit_stops_are_opt_in(self):
        route = build_visibility_route(
            self.known_map, self.start,
            dict(self.request, inspect_transit_views=True),
        )
        transit = [view for view in route["views"] if view["visibility_zone"]["band"] == "opportunistic_transit"]
        self.assertGreaterEqual(len(transit), 1)

    def test_later_views_are_visibility_zones_not_region_center(self):
        route = build_visibility_route(self.known_map, self.start, self.request)
        self.assertEqual(route["planning_policy"], POLICY)
        self.assertGreaterEqual(len(route["views"]), 2)
        self.assertEqual(route["physical_station_count"], 1)
        confidences = []
        station_points = set()
        for view in route["views"][1:]:
            zone = view["visibility_zone"]
            confidences.append(zone["confidence"])
            if zone["band"] == "opportunistic_transit":
                self.assertEqual(zone["stop_rule"], "periodic_stopped_capture_on_approach")
                continue
            self.assertEqual(zone["stop_rule"], "turn_at_station_then_capture")
            station_points.add(tuple(view["view_xy_m"]))
            self.assertGreater(
                math.hypot(
                    view["view_xy_m"][0] - self.center[0],
                    view["view_xy_m"][1] - self.center[1],
                ),
                self.request["radius_m"],
            )
            self.assertNotEqual(view["view_xy_m"], list(self.center))
            center_bearing = math.atan2(
                self.center[1] - view["view_xy_m"][1],
                self.center[0] - view["view_xy_m"][0],
            )
            yaw_error = math.atan2(
                math.sin(view["target_yaw_rad"] - center_bearing),
                math.cos(view["target_yaw_rad"] - center_bearing),
            )
            self.assertLessEqual(abs(math.degrees(yaw_error)), 75.0)
        self.assertEqual(len(station_points), 1)
        station_views = [
            view for view in route["views"]
            if view["visibility_zone"]["band"] in {"high", "medium", "low"}
        ]
        self.assertGreaterEqual(len(station_views), 1)  # Farther station can cover the area in one heading.
        self.assertEqual(
            [view["visibility_zone"]["sweep_index"] for view in station_views],
            list(range(len(station_views))),
        )
        self.assertEqual(confidences, sorted(confidences, reverse=True))
        self.assertIn("high", [view["visibility_zone"]["band"] for view in route["views"]])

    def test_next_station_is_planned_only_after_first_station_exhaustion(self):
        first = build_visibility_route(self.known_map, self.start, self.request)
        station = tuple(first["views"][1]["view_xy_m"])
        self.assertTrue(first["next_station_available"])
        VISITED_POINTS.extend((self.start, station))
        visibility_route_planner.OBSERVATIONS_COMPLETED = len(first["views"])
        second = build_visibility_route(self.known_map, station, self.request)
        self.assertEqual(second["physical_station_count"], 1)
        self.assertNotEqual(tuple(second["views"][0]["view_xy_m"]), station)
        self.assertLessEqual(
            len(first["views"]) + len(second["views"]),
            self.request["maximum_views"],
        )

    def test_route_declares_search_prior_not_target_coordinate(self):
        route = build_visibility_route(self.known_map, self.start, self.request)
        self.assertTrue(route["target_region"]["coordinates_are_search_prior_only"])
        self.assertEqual(route["mode"], "semantic_visibility_regions")

    def test_work_area_views_have_bounded_recognition_range_and_coverage(self):
        route = build_visibility_route(self.known_map, self.start, self.request)
        self.assertEqual(route["target_region"]["recognition_distance_m"], [0.45, 1.8])
        work_views = [
            view for view in route["views"]
            if view["visibility_zone"]["band"] in {"high", "medium", "low"}
        ]
        self.assertTrue(work_views)
        for view in work_views:
            zone = view["visibility_zone"]
            self.assertEqual(zone["recognition_max_distance_m"], 1.8)
            self.assertGreaterEqual(zone["estimated_work_area_coverage"], 0.25)
            self.assertGreaterEqual(zone["standoff_from_region_m"], 0.8)

    def test_old_close_standoff_decision_is_upgraded_and_path_keeps_buffer(self):
        request = dict(self.request, visibility_bands=[
            {"name": "high", "confidence": 0.85, "standoff_m": 0.55, "fov_margin_deg": 10},
        ])
        original_mask = self.known_map.traversable.copy()
        route = build_visibility_route(self.known_map, self.start, request)
        for view in route["views"][1:]:
            self.assertGreaterEqual(view["visibility_zone"]["standoff_from_region_m"], 0.8)
            for point in view["path_xy_m"]:
                self.assertGreaterEqual(math.dist(point, self.center), self.request["radius_m"] + 0.8 - 1e-4)
        self.assertTrue((original_mask == self.known_map.traversable).all())

    def test_buffer_rejects_driving_to_region_but_allows_stationary_capture(self):
        self.assertEqual(work_area_path(self.known_map, self.start, self.center, self.center, 1.75), [])
        self.assertEqual(work_area_path(self.known_map, self.center, self.center, self.center, 1.75), [self.center])

    def test_unrealistically_short_recognition_range_rejects_distant_views(self):
        request = dict(self.request, recognition_max_distance_m=0.8)
        with self.assertRaisesRegex(RuntimeError, "没有可达"):
            build_visibility_route(self.known_map, self.start, request)

    def test_194052_blocked_pivot_replans_to_straight_in_range_view(self):
        # Historical strong pose: the .50m turn-reserve excludes the start,
        # but a .44m body sweep along the present heading is map-clear.
        known = KnownOccupancyMap.load(RELEASE/'ble_scene_17.pgm',
                                       RELEASE/'ble_scene_17.yaml',clearance_m=.30)
        start=(-1.377823,-1.212825)
        yaw=1.035421
        original=known.path
        self.assertEqual(reserve_path(known,start,(-.825,.325),original),[])
        known.path=lambda a,b:reserve_path(known,a,b,original,
            start_yaw_rad=yaw if math.dist(a,start)<.08 else None,
            pose_uncertainty_m=.01556)
        known._phase1_original_path=known.path
        VISITED_POINTS.extend([(-1.37,-3.08),(-1.4126,-1.3663)])
        route=build_visibility_route(known,start,
            dict(self.request,_phase1_forward_view_yaw_rad=yaw))
        self.assertEqual(len(route['views']),1)
        view=route['views'][0]
        self.assertEqual(view['visibility_zone']['stop_rule'],
                         'forward_approach_then_capture')
        self.assertGreaterEqual(view['visibility_zone']['estimated_work_area_coverage'],.25)
        self.assertGreaterEqual(math.dist(view['view_xy_m'],self.center),.95+.80)
        self.assertAlmostEqual(view['target_yaw_rad'],yaw,places=4)
        self.assertEqual(len(known.path(start,tuple(view['view_xy_m']))),2)

    def test_straight_approach_never_uses_a_blocked_or_unoriented_corridor(self):
        known = KnownOccupancyMap.load(RELEASE/'ble_scene_17.pgm',
                                       RELEASE/'ble_scene_17.yaml',clearance_m=.30)
        start=(-1.377823,-1.212825)
        yaw=1.035421
        goal=(start[0]+1.5*math.cos(yaw),start[1]+1.5*math.sin(yaw))
        original=known.path
        self.assertEqual(reserve_path(known,start,goal,original),[])
        self.assertEqual(reserve_path(known,start,goal,original,
                                      start_yaw_rad=yaw+math.pi),[])
        obstacle=(start[0]+.20*math.cos(yaw),start[1]+.20*math.sin(yaw))
        column,row=known.world_to_grid((obstacle,))[0]
        known.temporary_occupied[row,column]=True
        self.assertEqual(reserve_path(known,start,goal,original,
                                      start_yaw_rad=yaw,pose_uncertainty_m=.01556),[])


if __name__ == "__main__":
    unittest.main()
