import gzip
import json
import math
from pathlib import Path
import tempfile
import threading
import unittest
import sys
from unittest.mock import patch

from PIL import Image

from raspberry_pi.joint_mapping import JointMappingSession, main
from raspberry_pi.map_camera_inspection import _information_view_targets, _spaced_targets
from raspberry_pi.known_map_navigation import KnownOccupancyMap, order_targets
from raspberry_pi.mapping import OccupancyGrid, Pose2D
from raspberry_pi.mapping.slam import SlamUpdate
from raspberry_pi.mapping.explorer import FrontierExplorer, ExplorationConfig, MotionCommand
from raspberry_pi.autonomous_mapping import run, build_argument_parser
from tests.test_autonomous_mapping import FakeClient, FakeLidar, front_scan


class Camera:
    def capture(self, path):
        Image.new("RGB", (40, 30), "red").save(path)
        return {}


def update(pose=Pose2D(), integrated=True):
    return SlamUpdate(pose, True, .01, 100, 100, .99, map_integrated=integrated)


class JointMappingTest(unittest.TestCase):
    def test_small_new_observations_do_not_restart_no_path_completion_sweep(self):
        explorer = FrontierExplorer(ExplorationConfig(initial_spin_rad=0, min_completion_travel_m=0))
        explorer.state = "exploring"
        explorer._travelled_m = 5
        grid = OccupancyGrid(100, 100, .05)
        grid.observed[30:70, 30:70] = True
        grid.log_odds[30:70, 30:70] = -2
        complete = False
        with patch.object(explorer, "_find_exploration_path", return_value=[]), patch.object(explorer, "_turn_command", return_value=MotionCommand(0, 1000, "completion_scan", "test")), patch("raspberry_pi.mapping.explorer.has_unresolved_obstacle_shadows", return_value=False), patch("raspberry_pi.mapping.explorer._has_global_frontier", return_value=False):
            for i in range(32):
                # 10 new cells exceed the 0.02 m2 threshold, even though the
                # real path planner reports no reachable new opportunity.
                grid.observed[i, :10] = True
                pose = Pose2D(0, 0, i * math.radians(15))
                command = explorer.update(pose, front_scan(5, 1 + i), grid)
                if command.finished:
                    complete = True
                    break
        self.assertTrue(complete, "small edge discoveries perpetually reset the 360 degree sweep")

    def test_reachable_new_path_still_cancels_completion(self):
        explorer = FrontierExplorer(ExplorationConfig(initial_spin_rad=0, min_completion_travel_m=0))
        explorer.state = "exploring"
        explorer._no_frontier_count = 100
        explorer._rescan_accumulated = math.tau
        grid = OccupancyGrid(100, 100, .05)
        grid.observed[:, :] = True
        grid.log_odds[:, :] = -2
        with patch.object(explorer, "_find_exploration_path", return_value=[(1., 0.)]):
            command = explorer.update(Pose2D(), front_scan(5), grid)
        self.assertFalse(command.finished)
        self.assertEqual(0, explorer._no_frontier_count)
        self.assertEqual(0, explorer._rescan_accumulated)

    def test_fusion_at_viewpoint_does_not_consume_wrong_facing_view(self):
        explorer = FrontierExplorer()
        explorer._inspection_look_at = (1., 0.)
        explorer._path = [(0., 0.)]
        explorer.note_stationary_map_integrated(Pose2D(0, 0, math.pi))
        self.assertEqual((1., 0.), explorer._inspection_look_at)
        self.assertEqual([(0., 0.)], explorer._path)
        explorer.note_stationary_map_integrated(Pose2D())
        self.assertIsNone(explorer._inspection_look_at)

    def test_shadow_viewpoint_keeps_direction_to_unknown_not_just_position(self):
        grid = OccupancyGrid(100, 100, .05)
        grid.observed[:, :] = True
        grid.log_odds[:, :] = -2
        grid.observed[65:, :] = False
        explorer = FrontierExplorer()
        with patch("raspberry_pi.mapping.explorer.find_frontier_path", return_value=[]), patch("raspberry_pi.mapping.explorer.find_observation_path", return_value=[(.5, 0.)]):
            path = explorer._find_exploration_path(grid, Pose2D())
        self.assertEqual([(.5, 0.)], path)
        self.assertIsNotNone(explorer._inspection_look_at)
        self.assertGreater(explorer._inspection_look_at[1], .7)
        self.assertTrue(explorer._unknown_observation)
        self.assertFalse(explorer._record_inspection_sector(Pose2D(), grid))

    def test_confirmed_field_start_selects_upper_not_pi_arbitrary_left_view(self):
        root = Path(__file__).resolve().parents[1]
        known = KnownOccupancyMap.load(root / "maps/ble_scene_17.pgm", clearance_m=.30)
        start = (-.715, .825)
        targets = known.inspection_targets(.65, maximum_targets=24,
                                           views_per_wall=3, views_per_region=1,
                                           preferred_view_xy=start)
        wall = next(target for target in targets if target.target_id == "wall_3_3")
        # The field run selected (-0.725,-0.675) and entered the left/lower
        # passage. The start-aware view remains on the user's upper side.
        self.assertGreater(wall.view_xy_m[1], .30)
        self.assertLess(math.dist(start, wall.view_xy_m), 1.3)
        targets = _spaced_targets(targets, .40)
        targets, _wide = _information_view_targets(known, targets, 2.20)
        ordered = order_targets(known, start, targets,
                                start_yaw_rad=math.radians(-131),
                                turn_cost_m_per_rad=.35)
        self.assertEqual("wall_3_3", ordered[0][0].target_id)
        self.assertGreater(ordered[0][0].view_xy_m[1], .80)

    def test_reference_is_copied_never_loaded_as_localization_prior(self):
        with tempfile.TemporaryDirectory() as temp:
            coarse = Path(temp) / "old.pgm"
            coarse.write_bytes(b"deliberately invalid old map")
            coarse.with_suffix(".yaml").write_text("unregistered")
            session = JointMappingSession(Path(temp) / "session", Camera(), lambda *a: {}, coarse)
            session.close()
            self.assertEqual(b"deliberately invalid old map", coarse.read_bytes())
            self.assertFalse(session.metadata["coarse_used_for_localization_or_planning"])
            self.assertEqual(coarse.read_bytes(), (session.directory / "coarse_reference/old.pgm").read_bytes())

    def test_no_repeat_photos_and_no_whole_wall_label_promotion(self):
        with tempfile.TemporaryDirectory() as temp:
            clock = [0.0]
            done = threading.Event()
            def infer(*args):
                done.set()
                return {"landmarks": [{"type": "table", "confidence": .99}]}
            session = JointMappingSession(Path(temp) / "session", Camera(), infer, clock=lambda: clock[0])
            try:
                grid = OccupancyGrid(40, 40, .1)
                session.capture(front_scan(), update(), grid)
                done.wait(2)
                session.jobs.join()
                session.poll()
                clock[0] = 100
                self.assertFalse(session.due(Pose2D(.1, .1, .1)))
                self.assertTrue(session.due(Pose2D(0, 0, math.pi / 2)))
                self.assertTrue(session.due(Pose2D(1, 0, 0)))
                self.assertFalse(session.records[0]["object_position_confirmed"])
                self.assertEqual("analyzed", session.records[0]["status"])
                self.assertEqual([0., 0., 0.], session.records[0]["pose"])
            finally:
                session.close()

    def test_failed_camera_cooldown_measured_from_end(self):
        clock = [0.0]
        class Broken:
            def capture(self, path):
                clock[0] += 20
                raise RuntimeError("busy")
        with tempfile.TemporaryDirectory() as temp:
            session = JointMappingSession(Path(temp) / "run", Broken(), lambda *a: {}, clock=lambda: clock[0])
            try:
                session.capture(front_scan(), update(), OccupancyGrid(20, 20, .1))
                self.assertFalse(session.due(Pose2D()))
                self.assertEqual("capture_failed", session.records[0]["status"])
            finally:
                session.close()

    def test_slow_inference_bounded_queue_and_shutdown(self):
        release = threading.Event()
        with tempfile.TemporaryDirectory() as temp:
            def slow(*args):
                release.wait(5)
                return {}
            session = JointMappingSession(Path(temp) / "run", Camera(), slow)
            try:
                for i in range(5):
                    session.capture(front_scan(), update(Pose2D(i, 0, 0)), OccupancyGrid(100, 100, .1))
                self.assertLessEqual(session.jobs.qsize(), 2)
                self.assertTrue(any(r["status"] == "deferred_inference" for r in session.records))
                session.close(wait_s=.01)
                self.assertFalse(session.metadata["all_views_analyzed"])
            finally:
                release.set()
                session.worker.join(2)
                session.close()

    def test_raw_scan_keeps_rejected_data_and_commands(self):
        with tempfile.TemporaryDirectory() as temp:
            session = JointMappingSession(Path(temp) / "run", Camera(), lambda *a: {})
            scan = front_scan()
            scan.distances_m[0] = math.nan
            rejected = SlamUpdate(Pose2D(), False, .4, 20, 100, rejection_reason="test_jump")
            session.record_scan(scan, rejected, None)
            session.close()
            with gzip.open(session.directory / "scans.jsonl.gz", "rt") as stream:
                row = json.loads(stream.readline())
            self.assertIsNone(row["distances_m"][0])
            self.assertFalse(row["accepted"])
            self.assertEqual("test_jump", row["rejection_reason"])

    def test_photo_hook_stops_and_does_not_move_on_old_scan(self):
        client = FakeClient()
        class Observer:
            captured = False
            def record_scan(self, *args):
                client.events.append(("scan",))
            def due(self, pose):
                return not self.captured
            def capture(self, scan, current, grid):
                self.captured = True
                self.asserted_stop = client.events[-1] == ("stop",)
                client.events.append(("photo",))
        observer = Observer()
        scans = [front_scan(timestamp=i * .2) for i in range(14)]
        with tempfile.TemporaryDirectory() as temp:
            args = build_argument_parser().parse_args(["--enable-motion", "--map-size", "5", "--output", str(Path(temp) / "map")])
            run(args, client=client, lidar=FakeLidar(scans), mapping_observer=observer)
        self.assertTrue(observer.captured)
        self.assertTrue(observer.asserted_stop)
        photo = client.events.index(("photo",))
        self.assertNotEqual("command", client.events[photo + 1][0] if photo + 1 < len(client.events) else "end")

    def test_camera_reset_replaces_lidar_iterator_before_any_motion(self):
        client = FakeClient()
        class ResetLidar:
            generation = 0
            reset = False
            stale_consumed = False
            def scans(self):
                self.generation += 1
                if self.generation == 1:
                    for i in range(7):
                        yield front_scan(timestamp=1 + i * .2)
                    self.stale_consumed = True
                    raise AssertionError("old iterator consumed after camera")
                for i in range(7):
                    yield front_scan(timestamp=10 + i * .2)
            def reset_scan_stream(self):
                self.reset = True
            def close(self):
                pass
        lidar = ResetLidar()
        class Observer:
            captured = False
            def record_scan(self, *a):
                pass
            def due(self, pose):
                return not self.captured
            def capture(self, *a):
                self.captured = True
        observer = Observer()
        with tempfile.TemporaryDirectory() as temp:
            args = build_argument_parser().parse_args(["--enable-motion", "--map-size", "5", "--output", str(Path(temp) / "map")])
            run(args, client=client, lidar=lidar, mapping_observer=observer)
        self.assertTrue(lidar.reset)
        self.assertFalse(lidar.stale_consumed)
        self.assertEqual(2, lidar.generation)

    def test_resume_annotations_never_constructs_motion_client(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            Camera().capture(directory / "view.jpg")
            manifest = directory / "observations.json"
            manifest.write_text(json.dumps({"observations": [{"id": 1, "image": "view.jpg", "context": {}, "status": "deferred_inference"}]}))
            with patch.object(sys, "argv", ["joint_mapping", "--resume-annotations", temp]), patch("raspberry_pi.joint_mapping.QwenViews") as infer, patch("raspberry_pi.autonomous_mapping.Esp32Client") as motion:
                infer.return_value.return_value = {"objects": [{"type": "table"}]}
                main()
                motion.assert_not_called()
            record = json.loads(manifest.read_text())["observations"][0]
            self.assertEqual("analyzed", record["status"])
            self.assertFalse(record["object_position_confirmed"])


if __name__ == "__main__":
    unittest.main()
