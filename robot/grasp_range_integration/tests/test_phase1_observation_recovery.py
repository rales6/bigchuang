from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from grasp_range_integration.phase1.observation_recovery import (
    CandidateRecovery, EvidenceStore, candidate, candidate_yaw_step,
    red_visual_cues, stable_candidate,
    install_observation_adapter, reject_copied_stereo_boxes, rotation_clearance_safe,
)
from grasp_range_integration.phase1.relationship_find_only_entry import _install_phase1_supervisor
from grasp_range_integration.phase1 import turn_policy, visibility_route_planner as planner


def detection(camera="left", box=None):
    return {"camera_id": camera, "class_name": "block", "confidence": 0.95,
            "bbox_1000": box or [900, 200, 930, 230], "source": "qwen",
            "single_camera_match": True, "dual_camera_confirmed": False,
            "search_evidence": {"source_ok": True, "class_ok": True,
                                "colour_ok": True, "bbox_valid": True}}


class CandidateRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.recovery = CandidateRecovery()
        self.recovery.context = {"robot_pose": [1.0, 2.0, 0.3]}
        self.route = {"views": [{"view_xy_m": [1.0, 2.0]}, {"view_xy_m": [3.0, 4.0]}],
                      "maximum_total_views": 8}
        self.state = {"latest_event": {"type": "observation_complete"}, "current_view_index": 0,
                      "remaining_view_indices": [1], "attempts": 1, "detections": [detection()]}

    def test_single_camera_first_requeues_same_view_without_translation(self):
        directive = self.recovery.next(self.state, self.route)
        self.assertEqual(directive["next_view_index"], 0)
        self.assertEqual(self.state["remaining_view_indices"], [0, 1])
        self.assertFalse(directive["recovery"]["linear_motion_allowed"])
        self.assertEqual(directive["recovery"]["step_deg"], 0)
        self.assertEqual(self.route["views"][0]["view_xy_m"], [1, 2])

    def test_second_stable_frame_gets_bounded_visual_yaw(self):
        self.recovery.next(self.state, self.route)
        self.state["remaining_view_indices"].remove(0)
        directive = self.recovery.next(self.state, self.route)
        self.assertEqual(directive["recovery"]["step_deg"], -6)
        self.assertAlmostEqual(self.route["views"][0]["target_yaw_rad"], 0.3 - math.radians(6))
        self.assertGreater(candidate_yaw_step(detection(box=[100, 200, 130, 230])), 0)

    def test_partial_edge_target_can_trigger_recheck_but_tiny_noise_cannot(self):
        item = detection(box=[980, 200, 1000, 230])
        item["single_camera_match"] = False
        self.assertIsNotNone(candidate([item]))
        self.assertIsNone(candidate([detection(box=[990, 200, 995, 205])]))

    def test_malformed_candidate_never_authorizes_motion(self):
        self.assertIsNone(candidate([detection(box=["invalid", 0, 30, 30])]))
        self.assertIsNone(candidate([detection(box=[math.nan, 0, 30, 30])]))

    def test_lost_or_unstable_candidate_does_not_authorize_turn(self):
        self.recovery.next(self.state, self.route)
        self.state["detections"] = [detection(box=[100, 200, 130, 230])]
        self.assertIsNone(self.recovery.next(self.state, self.route))
        self.assertEqual(self.recovery.last_status, "unstable_candidate")
        self.state["detections"] = []
        self.assertIsNone(self.recovery.next(self.state, self.route))

    def test_close_candidate_stops_and_budgets_are_finite(self):
        self.state["detections"] = [detection(box=[700, 900, 730, 950])]
        self.recovery.next(self.state, self.route)
        self.assertEqual(self.recovery.next(self.state, self.route)["action"], "report_failure")
        self.state["detections"] = [detection()]
        self.state["attempts"] = 8
        self.assertIsNone(self.recovery.next(self.state, self.route))

    def test_no_more_than_five_alignment_steps(self):
        self.recovery.next(self.state, self.route)
        for _ in range(5):
            self.assertIsNotNone(self.recovery.next(self.state, self.route))
        self.assertIsNone(self.recovery.next(self.state, self.route))
        self.assertEqual(self.recovery.last_status, "seen_unconfirmed")

    def test_model_miss_can_recover_a_visible_red_region_without_claiming_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = {}
            for camera, box in (("left", (0, 88, 55, 148)),
                                ("right", (284, 142, 329, 187))):
                picture = Image.new("RGB", (1280, 720), (105, 105, 105))
                ImageDraw.Draw(picture).rectangle(box, fill=(175, 36, 42))
                paths[camera] = Path(temporary) / f"{camera}.jpg"
                picture.save(paths[camera])
            cues = red_visual_cues(paths)
            self.assertEqual({item["camera_id"] for item in cues}, {"left", "right"})
            selected = candidate(cues)
            self.assertEqual(selected["camera_id"], "right")
            self.assertTrue(selected["complete_in_frame"])
            self.assertFalse(selected["confirmed_match"])
            self.assertFalse(selected["dual_camera_confirmed"])
            self.assertTrue(selected["search_evidence"]["visual_cue_only"])
            grounded = detection("right", box=[230, 198, 265, 265])
            self.assertTrue(stable_candidate(selected, grounded))

    def test_empty_station_gets_another_forward_heading_before_failure(self):
        self.recovery.context.update(robot_pose=[1., 2., 0.3], task_region_view={"eligible": True})
        self.route.update(mode="semantic_visibility_regions",
                          target_region={"center_xy_m": [2., 2.]},
                          views=[{"view_xy_m": [1., 2.], "target_yaw_rad": 0.3,
                                  "visibility_zone": {"sweep_count": 1}}])
        self.state.update(remaining_view_indices=[], detections=[])
        with patch.object(planner, "task_region_view_assessment", return_value={
                "eligible": True, "estimated_work_area_coverage": .4,
                "minimum_work_area_coverage": .25}):
            directive = self.recovery.next(self.state, self.route)
        self.assertEqual(directive["action"], "navigate_next_view")
        self.assertEqual(self.state["remaining_view_indices"], [0])
        self.assertEqual(directive["recovery"]["linear_motion_allowed"], False)
        self.assertEqual(self.route["views"][0]["visibility_zone"]["stop_rule"],
                         "local_view_sweep")

    def test_copied_qwen_boxes_do_not_complete_mission(self):
        items = [detection("left"), detection("right")]
        for item in items:
            item.update(dual_camera_confirmed=True, confirmed_match=True, stereo_pair_id="pair")
        result = reject_copied_stereo_boxes(items)
        self.assertTrue(all(not item["dual_camera_confirmed"] for item in result))

    def test_rotation_requires_allround_real_clearance(self):
        scan = SimpleNamespace(angles_rad=np.linspace(-math.pi, math.pi, 80, endpoint=False),
                               distances_m=np.ones(80))
        self.assertTrue(rotation_clearance_safe(scan))
        scan.distances_m[40] = 0.22
        self.assertFalse(rotation_clearance_safe(scan))
        scan.distances_m[:] = math.inf
        self.assertFalse(rotation_clearance_safe(scan))

    def test_repeated_clipping_in_front_is_not_assumed_to_be_body(self):
        angles = np.linspace(-math.pi, math.pi, 160, endpoint=False)
        distances = np.ones(160)
        distances[::3] = 0.104
        self.assertFalse(rotation_clearance_safe(SimpleNamespace(angles_rad=angles, distances_m=distances)))


class EvidenceTests(unittest.TestCase):
    def test_deepseek_verified_pair_survives_both_confirmation_passes(self):
        import raspberry_pi.object_search_mission as core
        with tempfile.TemporaryDirectory() as temporary:
            images = {}
            for camera, x in (("left", 350), ("right", 284)):
                picture = Image.new("RGB", (1280, 720), (105, 105, 105))
                ImageDraw.Draw(picture).rectangle((x, 142, x+45, 187),
                                                  fill=(175, 36, 42))
                path = Path(temporary) / f"{camera}.jpg"
                picture.save(path)
                images[camera] = path

            class Camera:
                def capture(self, *_args, **_kwargs):
                    raise AssertionError("no hardware capture")

            class Gpu:
                def perceive(self, *_args, **_kwargs):
                    raise AssertionError("no model call")

            class Log:
                def event(self, *_args, **_kwargs):
                    pass

            mission = SimpleNamespace(GpuMissionClient=Gpu, MissionLog=Log,
                confirm_detections=core.confirm_detections,
                _same_stereo_target=core._same_stereo_target,
                _fuse_tracking_pose=lambda *_args: None)
            recovery = install_observation_adapter(mission, Camera)
            recovery.context = {"task_region_view": {"eligible": True}}
            recovery.previous = next(item for item in red_visual_cues(images)
                                     if item["camera_id"] == "right")
            verdict = {"accepted": True, "reason": "two_independent_reviews_confirmed",
                       "model": "deepseek-flash", "repeats": 2,
                       "minimum_confidence": .91}
            target = {"class_name": "block", "color": "red",
                      "detection_mode": "qwen_fallback"}
            with patch('grasp_range_integration.phase1.red_target_deepseek.verify_red_block_pair',
                       return_value=verdict) as review:
                first = mission.confirm_detections([], target, images)
                second = mission.confirm_detections(first, target, images)
            self.assertEqual(review.call_count, 1)
            self.assertEqual(len([item for item in second
                                  if item.get("dual_camera_confirmed")]), 2)
            self.assertTrue(all(not item.get("confirmed_match") for item in second
                                if item.get("source") == "phase1_red_visual_cue"))

    def test_replan_filename_reuse_preserves_first_raw_and_annotations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left, right = root / "view_01.jpg", root / "view_01_camera_2.jpg"
            Image.new("RGB", (128, 72), "red").save(left)
            Image.new("RGB", (128, 72), "blue").save(right)
            store = EvidenceStore(root)
            first = store.capture(left, {"camera_count": 2, "camera_2_image": right.name})
            digest = hashlib.sha256((store.directory / first["raw_images"]["left"]).read_bytes()).hexdigest()
            store.analyze(first, [detection()], {"robot_pose": [0, 0, 0]})
            Image.new("RGB", (128, 72), "green").save(left)
            second = store.capture(left, {"camera_count": 2, "camera_2_image": right.name})
            self.assertNotEqual(first["observation_id"], second["observation_id"])
            self.assertEqual(digest, hashlib.sha256((store.directory / first["raw_images"]["left"]).read_bytes()).hexdigest())
            self.assertTrue((store.directory / first["annotated_images"]["left"]).exists())
            self.assertTrue((store.directory / "index.html").exists())
            self.assertEqual(len((store.directory / "observations.jsonl").read_text(encoding="utf-8").splitlines()), 3)


class ProtectedLoopIntegrationTests(unittest.TestCase):
    """Mocked IO only: exercise the actual executor's list/pose handoff."""
    def tearDown(self):
        planner.LATEST_ROUTE = planner.ACTIVE_VIEW = None
        planner.LATEST_KNOWN_MAP = None
        planner.VISITED_POINTS.clear()
        planner.OBSERVATIONS_COMPLETED = 0
        turn_policy.LAST_REQUEST = (0, 0)
        turn_policy.TELEMETRY_CALLBACK = None

    def test_single_left_recheck_then_stereo_success_without_any_drive_command(self):
        self.run_single_camera_scenario(reject_turn=False)

    def test_rotation_veto_runs_executor_stop_without_painting_front_obstacles(self):
        self.run_single_camera_scenario(reject_turn=True)

    def test_outside_region_start_requests_navigation_not_candidate_recovery(self):
        self.run_single_camera_scenario(reject_turn=False, approach_start=True)

    def test_live_stereo_preemption_runs_frozen_loop_fresh_capture_before_search_success(self):
        self.run_single_camera_scenario(reject_turn=False,transit_preempt=True)

    def run_single_camera_scenario(self, *, reject_turn, approach_start=False,transit_preempt=False):
        import raspberry_pi.object_search_mission as mission
        from raspberry_pi.mapping import Pose2D
        from raspberry_pi.known_map_navigation import LocalizationResult
        from grasp_range_integration.phase1.relationship_find_only_entry import _install_waypoint_obstacle_guard
        from grasp_range_integration.phase1.rotation_guard import RotationClearanceError

        class FakeCamera:
            calls=0
            def capture(self, path):
                from PIL import ImageDraw
                self.calls+=1
                path = Path(path)
                # Match the deployed pursuit model's calibrated resolution;
                # small mock frames are correctly rejected on the Pi.
                left=Image.new("RGB", (1280,720),(110,110,110))
                if not transit_preempt or self.calls>2:  # Warm-up then initial observation.
                    ImageDraw.Draw(left).rectangle((896,288,1024,360),fill=(220,0,0))
                left.save(path)
                second = path.with_name(path.stem + "_camera_2.jpg")
                right=Image.new("RGB", (1280,720),(110,110,110))
                if not transit_preempt or self.calls>2:
                    ImageDraw.Draw(right).rectangle((870,288,998,360),fill=(220,0,0))
                right.save(second)
                return {"camera_count": 2, "camera_2_image": second.name}

        class FakeGpu:
            calls = 0
            def perceive(self, left, right, context):
                self.calls += 1
                if approach_start or transit_preempt and self.calls==1:
                    return {"detections": []}
                a = detection("left", [700, 400, 800, 500])
                a["attributes"] = {"color": "red"}
                items = [a]
                if self.calls > 1 and not reject_turn:
                    b = detection("right", [680, 400, 780, 500])
                    b["attributes"] = {"color": "red"}
                    items.append(b)
                return {"detections": items}

        stops = []
        requested = []
        class FirstDriveRequested(RuntimeError):
            pass
        class FakeClient:
            def stop(self): stops.append(True)
            def heartbeat(self): pass
            def start(self): pass
            def set_twist(self, *args, **kwargs):
                if reject_turn and args == (0, 0):
                    requested.append(args)
                    return
                if approach_start:
                    requested.append(args)
                    raise FirstDriveRequested('software test stops at first requested drive, no hardware')
                raise AssertionError("stationary recheck must not issue any drive/turn command")

        scan = SimpleNamespace(angles_rad=np.linspace(-math.pi, math.pi, 80, endpoint=False), distances_m=np.ones(80), timestamp_s=0.)
        if reject_turn:
            scan.distances_m[40] = .10  # Actual front point, outside installed mask.
        lidar = SimpleNamespace(scans=lambda: iter([scan] * 20))
        pose = Pose2D(0, 0, 0)

        class FakeLocalizer:
            def __init__(self, *_args, **_kwargs): pass
            def relative_motion(self, *_args):
                return SimpleNamespace(transform=Pose2D(0, 0, 0), converged=True,
                                       rmse_m=0.02, inlier_ratio=0.9, correspondences=100)
            def track(self, _scan, predicted):
                return LocalizationResult(predicted, 0.02, 0.9, 1, 1)

        def route(*_args):
            planner.LATEST_ROUTE = {"views": [{"view_xy_m": [0, 0]}, {"view_xy_m": [2, 2]}],
                                    "maximum_total_views": 8}
            planner.ACTIVE_VIEW = planner.LATEST_ROUTE["views"][0]
            if approach_start:
                planner.LATEST_ROUTE.update(mode='semantic_visibility_regions', target_region={
                    'center_xy_m': [3, 0], 'radius_m': .5, 'recognition_distance_m': [.45, 1.8],
                    'minimum_work_area_coverage': .25})
                planner.LATEST_ROUTE['views'][1] = {'view_xy_m': [1.7, 0]}
                planner.LATEST_KNOWN_MAP = fake_map
            return planner.LATEST_ROUTE

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            # A deployed API config must not let a real model replace FakeGpu
            # evidence (and turn a one-eye/obstacle scenario into success).
            stack.enter_context(patch(
                'grasp_range_integration.phase1.red_target_deepseek.verify_red_block_pair',
                return_value={'accepted':False,'reason':'offline_mock_no_model_call'}))
            root = Path(temporary)
            decision = root / "decision.json"
            decision.write_text(json.dumps({"status": "executable", "route_request": {},
                                           "target": {"class_name": "block", "color": "red", "detection_mode": "qwen fallback"}}), encoding="utf-8")
            args = mission.build_argument_parser().parse_args([
                "--decision", str(decision), "--output", str(root), "--enable-motion",
                "--no-visual-localization", "--settle-seconds", "0", "--camera-stream-base-url", "",
            ])
            fake_map = SimpleNamespace(path=lambda start, goal: [start, goal])
            if approach_start:
                from raspberry_pi.known_map_navigation import KnownOccupancyMap
                fake_map = KnownOccupancyMap(np.full((200, 200), 255, dtype=np.uint8), .05, -5, -5)
                fake_map.path = lambda start, goal: [start, goal]
            stack.enter_context(patch.object(mission.KnownOccupancyMap, "load", return_value=fake_map))
            stack.enter_context(patch.object(mission, "ScanMapLocalizer", FakeLocalizer))
            stack.enter_context(patch.object(mission, "SearchDashboardPublisher", lambda *_args: SimpleNamespace(update=lambda **_kwargs: None)))
            stack.enter_context(patch.object(mission, "build_search_route", route))
            stack.enter_context(patch.object(mission, "_localize", return_value=(pose, scan)))
            stack.enter_context(patch.object(mission.signal, "signal"))
            stack.enter_context(patch.object(mission, "GpuMissionClient", FakeGpu))
            stack.enter_context(patch.object(mission, "MissionLog", type("TestLog", (mission.MissionLog,), {})))
            stack.enter_context(patch.object(mission, "AdaptiveSpeedController", type("TestController", (mission.AdaptiveSpeedController,), {})))
            for name in ("confirm_detections", "_fuse_tracking_pose", "_strict_command_for_pose", "_obstacle_safe"):
                stack.enter_context(patch.object(mission, name, getattr(mission, name)))
            recovery = install_observation_adapter(mission, FakeCamera)
            turn_policy.install_turn_policy(mission)
            _install_waypoint_obstacle_guard(mission)
            _install_phase1_supervisor(mission, recovery)
            live_calls=[]
            if transit_preempt:
                def select(*a):live_calls.append(a);return {'action':'stopped_recheck'}
                stack.enter_context(patch.object(mission,'_phase1_live_observation',
                    SimpleNamespace(select=select,stopped_capture=lambda:None),create=True))
            if approach_start:
                gpu = FakeGpu()
                with self.assertRaises(FirstDriveRequested):
                    mission.run(args, client=FakeClient(), lidar=lidar, camera=FakeCamera(), gpu=gpu)
                report = json.loads(next(root.glob('*/mission.json')).read_text(encoding='utf-8'))
                self.assertTrue(requested)
                self.assertNotEqual(requested[0], (0, 0))
                self.assertEqual(gpu.calls, 1)  # Transit image is searched, no candidate means continue.
                events = [e for e in report['events'] if e['type'] == 'observation_complete']
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]['detections'], [])
                self.assertEqual(events[0]['evidence_status'], 'approaching_task_region')
                directives = [e['directive'] for e in report['events'] if e['type'] == 'supervisor_directive']
                self.assertEqual(directives[-1]['next_view_index'], 1)
                self.assertEqual(directives[-1]['candidate_recovery_status'], 'approaching_task_region')
                self.assertTrue(stops)
                return
            if reject_turn:
                with self.assertRaises(RotationClearanceError):
                    mission.run(args, client=FakeClient(), lidar=lidar, camera=FakeCamera(), gpu=FakeGpu())
                reports = list(root.glob('*/mission.json'))
                self.assertEqual(len(reports), 1)
                report = json.loads(reports[0].read_text(encoding='utf-8'))
                self.assertEqual(report['result'], 'error')
                self.assertTrue(stops)
                self.assertFalse(any(e['type'] == 'temporary_obstacle' for e in report['events']))
                self.assertTrue(requested)
                self.assertTrue(all(command == (0, 0) for command in requested))
                self.assertIn('未添加前方障碍', report['summary'])
                return
            directory = mission.run(args, client=FakeClient(), lidar=lidar, camera=FakeCamera(), gpu=FakeGpu())
            report = json.loads((directory / "mission.json").read_text(encoding="utf-8"))
            self.assertEqual(report["result"], "success")
            observations = [e for e in report["events"] if e["type"] == "observation_complete"]
            self.assertEqual([e["view_index"] for e in observations], [0, 1] if transit_preempt else [0, 0])
            if transit_preempt:
                self.assertGreaterEqual(len(live_calls),2)
                self.assertEqual(observations[0]['detections'],[])
                self.assertTrue(all(i.get('dual_camera_confirmed') for i in observations[1]['detections']))
            self.assertNotEqual(observations[0]["archived_images"], observations[1]["archived_images"])
            for record in observations:
                for image in record["archived_images"].values():
                    self.assertTrue((directory / image).is_file())


if __name__ == "__main__":
    unittest.main()
