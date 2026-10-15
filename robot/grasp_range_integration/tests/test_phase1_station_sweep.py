from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from grasp_range_integration.phase1 import visibility_route_planner
from grasp_range_integration.phase1.relationship_find_only_entry import (
    _phase1_directive,
    _install_phase1_supervisor,
)
from grasp_range_integration.phase1.turn_policy import install_turn_policy


class StationSweepTests(unittest.TestCase):
    def tearDown(self):
        visibility_route_planner.ACTIVE_VIEW = None
        visibility_route_planner.LATEST_ROUTE = None
        visibility_route_planner.VISITED_POINTS.clear()
        visibility_route_planner.OBSERVATIONS_COMPLETED = 0

    def test_exhausted_station_requests_next_station_only_if_available(self):
        state = {"remaining_view_indices": [], "latest_event": {}, "detections": []}
        directive = _phase1_directive(
            state,
            {"views": [], "next_station_available": True, "maximum_total_views": 8},
            allow_low=False,
        )
        self.assertEqual(directive["action"], "replan")
        directive = _phase1_directive(
            state, {"views": [], "next_station_available": False}, allow_low=False,
        )
        self.assertEqual(directive["action"], "report_failure")
        state["attempts"] = 8
        directive = _phase1_directive(
            state,
            {"views": [], "next_station_available": True, "maximum_total_views": 8},
            allow_low=False,
        )
        self.assertEqual(directive["action"], "report_failure")

    def test_supervisor_finishes_all_yaws_before_replanning(self):
        views = [
            {"view_xy_m": [1.0, 2.0], "target_yaw_rad": yaw,
             "visibility_zone": {"band": "high"}}
            for yaw in (0.2, 0.5)
        ]
        visibility_route_planner.LATEST_ROUTE = {
            "views": views,
            "next_station_available": True,
            "maximum_total_views": 8,
        }
        mission = SimpleNamespace(GpuMissionClient=type("Gpu", (), {}))
        _install_phase1_supervisor(mission)
        gpu = mission.GpuMissionClient()
        first = gpu.next({
            "remaining_view_indices": [1], "current_view_index": 0,
            "latest_event": {"type": "observation_complete"},
            "detections": [], "attempts": 1,
        })
        self.assertEqual(first["action"], "navigate_next_view")
        self.assertEqual(first["next_view_index"], 1)
        self.assertIs(visibility_route_planner.ACTIVE_VIEW, views[1])
        self.assertEqual(visibility_route_planner.OBSERVATIONS_COMPLETED, 1)
        exhausted = gpu.next({
            "remaining_view_indices": [], "current_view_index": 1,
            "latest_event": {"type": "observation_complete"},
            "detections": [], "attempts": 2,
        })
        self.assertEqual(exhausted["action"], "replan")

    def test_224442_observed_transit_is_not_reinserted_ahead_of_cabinet_station(self):
        from raspberry_pi.known_map_navigation import KnownOccupancyMap

        root = Path(__file__).resolve().parents[2]
        maps = root if (root / 'ble_scene_17.pgm').exists() else root / 'maps' / 'semantic_release_20260919'
        known_map = KnownOccupancyMap.load(maps / 'ble_scene_17.pgm', maps / 'ble_scene_17.yaml', clearance_m=.30)
        request = dict(center_xy_m=[.8751325645,1.0443147835], radius_m=.95,
            maximum_views=8, inspect_current_view_first=True, camera_horizontal_fov_deg=82,
            visibility_bands=[dict(name='high',confidence=.85,standoff_m=.55,fov_margin_deg=10),
                              dict(name='medium',confidence=.70,standoff_m=.9,fov_margin_deg=7),
                              dict(name='low',confidence=.55,standoff_m=1.3,fov_margin_deg=4)])
        original = visibility_route_planner.build_visibility_route(known_map,(-1.51,-3.08),request)
        self.assertEqual(original['views'][0]['visibility_zone']['band'],'opportunistic_current_view')
        self.assertEqual(original['views'][1]['visibility_zone']['band'],'high')
        visibility_route_planner.LATEST_ROUTE = original
        mission = SimpleNamespace(GpuMissionClient=type('Gpu',(),{}))
        _install_phase1_supervisor(mission)
        directive = mission.GpuMissionClient().next(dict(latest_event={'type':'observation_complete'},
            current_view_index=0,remaining_view_indices=[1],detections=[],attempts=1))
        self.assertEqual(directive['next_view_index'],1)
        self.assertIn((-1.51,-3.08),visibility_route_planner.VISITED_POINTS)
        # The run replanned after a real obstacle vote at this pose. The
        # executor filters a previously observed stop at 0.25m; the planner
        # must already have removed it or the two view indices diverge.
        replanned = visibility_route_planner.build_visibility_route(
            known_map,(-1.4176,-3.0534),request)
        executable = [view for view in replanned['views'] if
            math.hypot(view['view_xy_m'][0]+1.51,view['view_xy_m'][1]+3.08) >= .25]
        self.assertEqual(len(replanned['views']),len(executable))
        self.assertEqual(replanned['views'][0]['visibility_zone']['band'],'high')
        self.assertAlmostEqual(math.degrees(replanned['views'][0]['target_yaw_rad']),22.93,places=1)
        actual = visibility_route_planner.task_region_view_assessment(replanned,
            [-.914,.182,1.189],known_map=known_map,view=replanned['views'][0])
        turned = visibility_route_planner.task_region_view_assessment(replanned,
            [-.914,.182,replanned['views'][0]['target_yaw_rad']],known_map=known_map,
            view=replanned['views'][0])
        self.assertFalse(actual['eligible'])
        self.assertTrue(turned['eligible'])

    def test_arrival_at_station_turns_to_active_observation_yaw(self):
        class Controller:
            def observe(self, _update):
                pass

            def stopped(self):
                pass

            def ttl_for(self, _command, normal_ttl_ms):
                return normal_ttl_ms

        mission = SimpleNamespace(
            AdaptiveSpeedController=Controller,
            _strict_command_for_pose=lambda *_args: (0, 0, True),
        )
        install_turn_policy(mission)
        args = SimpleNamespace(
            waypoint_tolerance=0.12, heading_tolerance_deg=8.0,
            turn_speed=2600,
        )
        pose = SimpleNamespace(x_m=1.0, y_m=2.0, yaw_rad=0.0)
        waypoint = (1.0, 2.0)
        visibility_route_planner.ACTIVE_VIEW = {
            "view_xy_m": [1.0, 2.0], "target_yaw_rad": math.radians(45),
        }
        linear, angular, reached = mission._strict_command_for_pose(
            args, pose, waypoint,
        )
        self.assertEqual(linear, 0)
        self.assertGreater(angular, 0)
        self.assertFalse(reached)
        pose.yaw_rad = math.radians(40)
        self.assertEqual(
            mission._strict_command_for_pose(args, pose, waypoint),
            (0, 0, True),
        )


if __name__ == "__main__":
    unittest.main()
