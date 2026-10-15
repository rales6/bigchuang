"""No hardware/network: compile, dispatch, navigate handoff, arm order and stop."""
import ast
import json
import math
from pathlib import Path
import re
import tempfile
import threading
import types
import unittest
from unittest.mock import patch, MagicMock

from grasp_range_integration.track_web_integration.gpu.plan_adaptation import normalize_web_plan, adapt_plan_to_live_view
from grasp_range_integration.track_web_integration.gpu.robot_client import RobotClient
from grasp_range_integration.track_web_integration.pi.region_navigation import entities_for, select_region, save_json, anchor_seen, install_region_skill
from grasp_range_integration.track_web_integration.pi.robot_skills import RobotSkills
from grasp_range_integration.track_web_integration.pi.placement_extension import run_placement, validate_pose


def compiler():
    # Execute the actual pure compiler class without importing GPU-only web/
    # image dependencies on the Windows development machine.
    path = Path(__file__).parents[1] / "track_web_integration/gpu/qwen_hub.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SharedAgentPlanner")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ns = {"__package__": "grasp_range_integration.track_web_integration.gpu", "json": json, "re": re}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns["SharedAgentPlanner"](None)


class SkillTests(unittest.TestCase):
    def plan(self):
        return normalize_web_plan(compiler()._compile(
            "去柜子边夹取一个绿色小物块然后放在最近的椅子旁", {
                "intent": "pick_and_place", "target_query": "绿色小物块",
                "search_near_query": "柜子", "destination_query": "最近的椅子旁",
                "destination_landmark_id": "wrong_early_choice"}, "deepseek"))

    def test_complete_chain_no_duplicate_search_or_navigation(self):
        plan = self.plan()
        self.assertEqual([s["action"] for s in plan["steps"]], [
            "inspect_semantic_map", "patrol_search", "align_for_grasp", "model_grasp", "verify_grasp",
            "locate_destination", "navigate_to", "place_on_floor", "verify_completion", "report_result"])
        self.assertEqual(plan["steps"][1]["target"]["query"], "柜子")
        self.assertEqual(plan["steps"][1]["target"]["pickup_instruction"], "去柜子边上夹取一个绿色小物块")
        self.assertTrue(plan["steps"][4]["target"]["carry_after_grasp"])
        self.assertIsNone(plan["steps"][5]["target"]["landmark_id"])
        self.assertEqual(plan["steps"][5]["target"]["reference"], "current_pose_after_pickup")
        for i, step in enumerate(plan["steps"]):
            self.assertEqual(step["depends_on"], [] if i == 0 else [plan["steps"][i-1]["id"]])

    def test_named_anchor_cannot_be_skipped_by_other_visible_block(self):
        plan = adapt_plan_to_live_view(self.plan(), {"status": "visible_both", "inside_grasp_region": True})
        self.assertEqual(plan["steps"][1]["action"], "patrol_search")

    def test_unsupported_height_does_not_move(self):
        for destination in ("椅子上", "桌面上", "柜子顶部"):
            plan = compiler()._compile("夹取绿色小物块后放在" + destination, {
                "intent": "pick_and_place", "target_query": "绿色小物块", "destination_query": destination}, "test")
            self.assertEqual(plan["intent"], "unsupported")
            self.assertEqual([s["action"] for s in plan["steps"]], ["report_result"])

    def test_missing_destination_does_not_pick(self):
        plan = compiler()._compile("拿起来放好", {"intent": "pick_and_place"}, "test")
        self.assertEqual(plan["intent"], "unsupported")

    def test_place_without_regrasp(self):
        plan = compiler()._compile("把手里物品放在最近椅子旁", {
            "intent": "place", "destination_query": "最近椅子旁"}, "test")
        self.assertNotIn("model_grasp", [s["action"] for s in plan["steps"]])

    def test_old_chair_category_does_not_select_table(self):
        semantic = {"objects": [
            {"id": "chair", "name": "桌边椅子（右侧）", "category": "table", "map_xy": [1, 0]},
            {"id": "table", "name": "桌子", "category": "table", "map_xy": [0, 0]}]}
        self.assertEqual([s["id"] for s in entities_for(semantic, {"query": "最近椅子旁"})], ["chair"])
        self.assertEqual([s["id"] for s in entities_for(semantic, {"query": "桌子旁"})], ["table"])

    def test_region_unknown_not_random(self):
        with self.assertRaises(ValueError):
            entities_for({"objects": []}, {"query": "椅子"})

    def test_destination_single_eye_is_enough_but_bad_box_is_not(self):
        item = {"class_name": "chair", "camera_id": "left", "confidence": .85,
                "bbox_1000": [100, 200, 600, 700]}
        self.assertTrue(anchor_seen([item], "chair"))
        for change in ({"confidence": "bad"}, {"bbox_1000": [700, 500, 600, 100]},
                       {"bbox_1000": [1, 2, float('nan'), 4]}):
            self.assertFalse(anchor_seen([{**item, **change}], "chair"))

    def test_region_hook_keeps_guard_wrapper_and_recovers_missed_views(self):
        import numpy as np
        from grasp_range_integration.phase1 import visibility_route_planner as visibility
        scan = types.SimpleNamespace(angles_rad=np.linspace(-math.pi/4, math.pi/4, 80),
                                     distances_m=np.full(80, 1.2))
        class Driver:
            def __init__(self):
                self.stream = types.SimpleNamespace(latest_after=lambda *a, **kw: scan)
        class GPU:
            def perceive(self, *a):
                return {"detections": [{"class_name": "chair", "camera_id": "left",
                    "confidence": .85, "bbox_1000": [100, 200, 600, 700]}]}
        guard_wrapper = MagicMock()
        mission = types.SimpleNamespace(N10LidarDriver=Driver, GpuMissionClient=GPU,
            confirm_detections=lambda *a: [], _localize=lambda *a, **kw: None,
            build_search_route=guard_wrapper)
        class Matrix:
            def __getitem__(self, key): return True
        known = types.SimpleNamespace(world_to_grid=lambda xy: [(0, 0)],
            in_bounds=lambda *a: True, traversable=Matrix(), path=lambda start, goal: [start, goal])
        with tempfile.TemporaryDirectory() as root, patch.object(visibility, "REGION_ROUTE_BUILDER", None, create=True):
            job_path = Path(root) / "job.json"
            save_json(job_path, {"schema": "robot_region_job.v1", "carrying": True,
                "candidates": [{"id": "chair", "name": "椅子", "class_name": "chair", "center_xy_m": [2, 0]}]})
            install_region_skill(mission, types.SimpleNamespace(max_replans=2), job_path)
            self.assertIs(mission.build_search_route, guard_wrapper)
            route = visibility.REGION_ROUTE_BUILDER(known, (0, 0), {})
            mission.N10LidarDriver()
            gpu = mission.GpuMissionClient()
            self.assertEqual(len(route["views"]), 3)
            next_view = gpu.next({"remaining_view_indices": [1, 2], "latest_event": {}})
            self.assertEqual(next_view["next_view_index"], 1)
            retry = gpu.next({"remaining_view_indices": [1], "latest_event": {"type": "navigation_no_progress"}})
            self.assertEqual(retry["action"], "replan")
            self.assertFalse(retry["terminal"])
            goal = route["views"][0]["view_xy_m"]
            gpu.perceive("left", "right", {"robot_pose": [*goal, 0]})
            done = gpu.next({"remaining_view_indices": [2], "latest_event": {}})
            self.assertEqual(done["action"], "report_success")
            arrival = json.loads(job_path.read_text(encoding="utf-8"))["arrival"]
            self.assertTrue(arrival["anchor_visual_confirmed"])
            self.assertTrue(arrival["placement_clearance"])

    def test_nearest_uses_reachable_path_not_euclidean(self):
        fake = types.SimpleNamespace(
            world_to_grid=lambda points: [(0, 0)], in_bounds=lambda x, y: True,
            traversable=types.SimpleNamespace(__getitem__=lambda self, key: True))
        class Matrix:
            def __getitem__(self, key): return True
        fake.traversable = Matrix()
        def path(start, goal):
            return [] if goal[0] < 4 else [start, goal]
        fake.path = path
        selected = select_region(fake, (0, 0), [
            {"id": "unreachable_close", "center_xy_m": [1, 0]},
            {"id": "reachable", "center_xy_m": [6, 0]}])
        self.assertEqual(selected["id"], "reachable")
        self.assertGreaterEqual(math.dist(selected["approach_xy_m"], [6, 0]), .89)

    def test_resolve_does_not_dispatch_motion(self):
        client = RobotClient()
        calls = []
        client.request = lambda path, payload=None, **kw: calls.append(path) or {"ok": True}
        result = client.dispatch_step(self.plan()["steps"][5], "instruction")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(calls, ["/skills/resolve-destination"])

    def test_carry_verification_does_not_wait_for_human_click(self):
        client = RobotClient()
        client.request = lambda *a, **k: {"ok": True, "physical_pick_verified": False}
        result = client.dispatch_step(self.plan()["steps"][4], "instruction")
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["detail"]["physical_pick_verified"])

    def test_placement_dispatch_uses_new_queue_not_old_release(self):
        client = RobotClient()
        calls = []
        client.request = lambda path, payload=None, **kw: calls.append(path) or {"ok": True}
        result = client.dispatch_step(self.plan()["steps"][7], "instruction")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(calls, ["/actions/preview/start", "/skills/place-on-floor"])

    def arm(self):
        calls = []
        arm = types.SimpleNamespace(_exec_cancel_generation=0, _state_lock=threading.RLock(), _exec_status={},
            CLAW_OPEN=95, CLAW_CLOSE=0,
            SERVOS=[{"key": "s0", "home": 1500, "min": 500, "max": 2500},
                    {"key": "s1", "home": 1700, "min": 800, "max": 1700},
                    {"key": "s2", "home": 2000, "min": 1500, "max": 2200}])
        arm.send_grasp_pose = lambda send, pose: send("POSE") and calls.append(("pose", pose))
        arm.send_claw = lambda send, value: send("CLAW") and calls.append(("claw", value))
        arm._heartbeat_wait = lambda send, seconds: send("#PING!")
        return arm, calls

    def test_placement_order_and_preserve_mirrored_nudge(self):
        arm, calls = self.arm()
        target = {"s0": 1400, "s1": 1100, "s2": 1790}
        status = {}
        sent = []
        run_placement(arm, lambda cmd: sent.append(cmd) or "OK", target, status, 0)
        self.assertEqual(calls, [("pose", target), ("claw", 95),
                                 ("pose", {"s0": 1500, "s1": 1700, "s2": 2000})])
        self.assertNotIn("#HOME!", sent)
        self.assertEqual(status["phase"], "done")
        self.assertFalse(status["physical_place_verified"])

    def test_stale_ble_recovers_with_ping_before_any_arm_motion(self):
        arm, calls = self.arm()
        responses = iter(["ERROR", "SIM", "SIM", "OK", "OK"])
        commands = []
        def send(command):
            commands.append(command)
            if len(commands) <= 5:
                self.assertEqual(command, "#PING!")
                self.assertEqual(calls, [])
                return next(responses)
            return "OK"
        status = {}
        with patch("grasp_range_integration.track_web_integration.pi.placement_extension.time.sleep"):
            run_placement(arm, send, {"s0": 1400, "s1": 1100, "s2": 1780}, status, 0)
        self.assertEqual(status["phase"], "done")
        self.assertEqual(status["link_attempts"], 5)
        self.assertTrue(status["lower_started"])
        self.assertTrue(status["release_started"])
        self.assertEqual([v for k, v in calls if k == "claw"], [95])

    def test_link_timeout_is_proven_pre_lower_retryable(self):
        arm, calls = self.arm()
        status = {}
        with patch("grasp_range_integration.track_web_integration.pi.placement_extension.time.monotonic", side_effect=[0, 21]):
            run_placement(arm, lambda cmd: "SIM", {}, status, 0)
        self.assertEqual(calls, [])
        self.assertFalse(status["lower_started"])
        self.assertFalse(status["release_started"])
        self.assertTrue(status["retryable_before_lower"])
        self.assertFalse(arm._exec_status["active"])

    def test_cancel_during_handoff_never_moves_or_authorises_retry(self):
        arm, calls = self.arm()
        def send(command):
            arm._exec_cancel_generation += 1
            return "OK"
        status = {}
        run_placement(arm, send, {}, status, 0)
        self.assertEqual(calls, [])
        self.assertFalse(status["retryable_before_lower"])

    def test_lowering_fault_and_failed_stop_do_not_release_or_escape_queue(self):
        arm, calls = self.arm()
        def send(command):
            if command == "POSE": return "ERROR"
            if command == "#STOP!": raise ConnectionError("stop link unavailable")
            return "OK"
        status = {}
        run_placement(arm, send, {"s0": 1400, "s1": 1100, "s2": 1780}, status, 0)
        self.assertEqual(status["phase"], "error")
        self.assertTrue(status["lower_started"])
        self.assertFalse(status["release_started"])
        self.assertFalse(status["retryable_before_lower"])
        self.assertEqual(calls, [])
        self.assertEqual(status["stop_result"], "ConnectionError")

    def test_swallowed_heartbeat_fault_prevents_release(self):
        arm, calls = self.arm()
        def wait(send, seconds):
            try: send("#PING!")
            except ConnectionError: pass
        arm._heartbeat_wait = wait
        ping = 0
        def send(command):
            nonlocal ping
            if command == "#PING!":
                ping += 1
                return "ERROR" if ping == 4 else "OK"
            return "OK"
        status = {}
        run_placement(arm, send, {"s0": 1400, "s1": 1100, "s2": 1780}, status, 0)
        self.assertEqual(status["phase"], "error")
        self.assertEqual([k for k, v in calls], ["pose"])
        self.assertFalse(status["release_started"])

    def test_placement_error_reconciles_carry_only_before_lower(self):
        with tempfile.TemporaryDirectory() as root:
            skills = self.ledger(root)
            safe = {"phase": "error", "ok": False, "placement_protocol": 2,
                    "lower_started": False, "release_started": False,
                    "retryable_before_lower": True}
            for status, expected in [(safe, "carrying"),
                    ({**safe, "lower_started": True}, "placement_unknown"),
                    ({**safe, "release_started": True}, "placement_unknown"),
                    ({**safe, "retryable_before_lower": False}, "placement_unknown"),
                    ({"phase": "error", "error": "old ambiguous result"}, "placement_unknown")]:
                skills.data = {"status": "placing", "place_id": "old", "ground_pose": {"s2": 1780}}
                skills._placement_result(status)
                self.assertEqual(skills.data["status"], expected)
                self.assertEqual(skills.data["ground_pose"], {"s2": 1780})
                self.assertEqual(self.ledger(root).data["status"], expected)

    def test_photo_io_error_does_not_abort_or_repeat_successful_release(self):
        arm, calls = self.arm()
        def capture(stage, status): raise OSError("evidence storage unavailable")
        status = {}
        run_placement(arm, lambda cmd: "OK", {"s0": 1400, "s1": 1100, "s2": 1780}, status, 0, capture)
        self.assertEqual(status["phase"], "done")
        self.assertEqual(set(status["image_errors"]), {"before", "after"})
        self.assertEqual([v for k, v in calls if k == "claw"], [95])

    def test_place_waits_handoff_and_pre_lower_retry_has_a_new_id(self):
        with tempfile.TemporaryDirectory() as root:
            skills = self.ledger(root)
            skills.bridge._authorise_hardware = lambda headers: None
            skills.bridge._grasp_lock = threading.Lock()
            skills.bridge._grasp_cancel = threading.Event()
            skills.bridge.process = types.SimpleNamespace(snapshot=lambda: {"running": False})
            skills.data = {"status": "carrying", "carry_id": "a"*32,
                           "destination_query": "椅子", "ground_pose": {"s0": 1400,"s1": 1100,"s2": 1780}}
            skills.arrival = lambda: {"job_id": "b"*32}
            ids, polls = [], []
            def request(path, payload=None, **kw):
                if path.endswith("/api/status"):
                    return {"ble": True, "cameras": {side: {"frames": 10, "age_s": .1} for side in ("left","right")}}
                if path.endswith("/api/arm-teach/state"):
                    return {"state": {"known": True,"claw_value": 0}}
                if path.endswith("/api/arm-teach/place"):
                    ids.append(payload["job_id"])
                    return {"ok": True,"phase": "queued"}
                if "/place-status?" in path:
                    polls.append(path)
                    if len(polls) in (1,3):return {"ok": True,"phase": "handoff"}
                    if len(polls)==2:
                        return {"ok": False,"phase": "error","placement_protocol": 2,
                                "lower_started": False,"release_started": False,"retryable_before_lower": True}
                    return {"ok": True,"phase": "done"}
                raise AssertionError(path)
            with patch("grasp_range_integration.track_web_integration.pi.robot_skills.time.sleep"):
                first = skills.place({"query":"椅子"}, {}, request=request, base_url="http://local")
                self.assertFalse(first["ok"])
                self.assertEqual(skills.data["status"], "carrying")
                second = skills.place({"query":"椅子"}, {}, request=request, base_url="http://local")
            self.assertTrue(second["ok"])
            self.assertEqual(skills.data["status"], "placed")
            self.assertNotEqual(ids[0], ids[1])
            self.assertEqual(len(polls), 4)
            self.assertTrue(all(re.fullmatch(r"[a-f0-9_]{16,100}", value) for value in ids))

    def placement_app(self, root, arm, enqueue):
        try:
            from flask import Flask
        except ImportError:
            self.skipTest("Flask unavailable locally; run HTTP handler tests in the existing Pi environment")
        from grasp_range_integration.track_web_integration.pi.placement_extension import register_placement
        app = Flask(__name__)
        arm._arm_jobs_pending = 0
        arm._state = {"known": True,"claw_value": 0}
        arm.enqueue_command = enqueue
        with patch("grasp_range_integration.track_web_integration.pi.placement_extension.Path",
                   side_effect=lambda value: Path(root)/"evidence"):
            register_placement(app, arm, lambda cmd: "OK")
        return app.test_client()

    def test_persisted_pre_lower_failure_can_be_read_without_repeating_motion(self):
        with tempfile.TemporaryDirectory() as root:
            job_id = "a"*32
            directory = Path(root)/"evidence"/job_id
            directory.mkdir(parents=True)
            status = {"ok": False,"phase": "error","placement_protocol": 2,
                      "lower_started": False,"release_started": False,"retryable_before_lower": True}
            save_json(directory/"status.json", status)
            arm, calls = self.arm()
            enqueue = MagicMock()
            app = self.placement_app(root, arm, enqueue)
            result = app.get("/api/arm-teach/place-status?job_id="+job_id).get_json()
            self.assertEqual(result, status)
            duplicate = app.post("/api/arm-teach/place", json={"job_id":job_id}).get_json()
            self.assertEqual(duplicate, status)
            enqueue.assert_not_called()
            self.assertEqual(calls, [])

    def test_persisted_lower_or_release_remains_unknown_on_service_restart(self):
        with tempfile.TemporaryDirectory() as root:
            for field in ("lower_started", "release_started"):
                job_id = "a"*32
                directory = Path(root)/"evidence"/job_id
                directory.mkdir(parents=True, exist_ok=True)
                status = {"ok": False,"phase": "error","placement_protocol": 2,
                          "lower_started": False,"release_started": False,"retryable_before_lower": True}
                status[field] = True
                save_json(directory/"status.json", status)
                arm, calls = self.arm()
                enqueue = MagicMock()
                app = self.placement_app(root, arm, enqueue)
                result = app.get("/api/arm-teach/place-status?job_id="+job_id).get_json()
                self.assertEqual(result["phase"], "unknown")
                enqueue.assert_not_called()
                self.assertEqual(calls, [])

    def test_cancel_before_lower_does_not_release(self):
        arm, calls = self.arm()
        arm._exec_cancel_generation = 1
        status = {}
        run_placement(arm, lambda cmd: "OK", {}, status, 0)
        self.assertEqual(status["phase"], "error")
        self.assertEqual(calls, [])

    def test_bad_pose_not_clamped_silently(self):
        arm, _ = self.arm()
        with self.assertRaises(ValueError):
            validate_pose({"s0": 1500, "s1": 1700, "s2": 2300}, arm.SERVOS)

    def ledger(self, root):
        adapter = types.SimpleNamespace(runtime_root=Path(root) / "navigation",
            live_status=lambda: {"pose": {"x_m": 0, "y_m": 1, "yaw_deg": 90}})
        bridge = types.SimpleNamespace(map_navigation=adapter)
        return RobotSkills(bridge)

    def test_new_demo_pick_archives_previous_carry_without_blocking(self):
        with tempfile.TemporaryDirectory() as root:
            skills = self.ledger(root)
            result = {"execution": {"phase": "done", "params": {"s0": 1500, "s1": 1200, "s2": 1750}}}
            skills.record_pick("绿色小物块", result)
            self.assertEqual(skills.data["ground_pose"]["s2"], 1750)
            self.assertFalse(skills.data["physical_pick_verified"])
            skills.assert_empty()
            self.assertEqual(skills.snapshot()['status'], 'ready_for_pick')
            self.assertEqual(skills.snapshot()['previous_carry']['ground_pose']['s2'], 1750)
            self.assertFalse(skills.snapshot()['physical_empty_verified'])
            self.assertNotIn('ground_pose', skills.snapshot())
            self.assertEqual(self.ledger(root).snapshot()['status'], 'ready_for_pick')

    def test_active_or_unknown_placement_still_requires_reconciliation(self):
        with tempfile.TemporaryDirectory() as root:
            skills = self.ledger(root)
            for status in ('placing', 'placement_unknown'):
                skills.data = {'status': status, 'place_id': 'this-round'}
                with self.assertRaises(RuntimeError): skills.assert_empty()
                self.assertEqual(skills.snapshot()['place_id'], 'this-round')

    def empty_confirmation_ledger(self, root):
        skills = self.ledger(root)
        skills.bridge._authorise_hardware = MagicMock()
        skills.bridge._grasp_lock = threading.Lock()
        skills.bridge.process = types.SimpleNamespace(snapshot=lambda: {'running': False, 'kind': 'idle'})
        skills.data = {'status': 'carrying', 'carry_id': 'old-red', 'target': '红色方块',
                       'invalidated_by': 'emergency_stop'}
        skills.save()
        return skills

    def test_confirmed_empty_clears_persisted_stale_pick_without_arm(self):
        with tempfile.TemporaryDirectory() as root:
            skills = self.empty_confirmation_ledger(root)
            result = skills.confirm_empty({'confirmed_empty': True, 'expected_carry_id': 'old-red'}, {})
            self.assertTrue(result['ok'])
            self.assertFalse(result['hardware_commands_sent'])
            skills.assert_empty()
            self.assertEqual(self.ledger(root).snapshot()['status'], 'empty')
            self.assertEqual(skills.snapshot()['previous_carry']['target'], '红色方块')

    def test_no_confirmation_or_changed_carry_keeps_record(self):
        with tempfile.TemporaryDirectory() as root:
            skills = self.empty_confirmation_ledger(root)
            for payload in [{}, {'confirmed_empty': False},
                            {'confirmed_empty': True, 'expected_carry_id': 'new-green'}]:
                with self.subTest(payload=payload), self.assertRaises((ValueError, RuntimeError)):
                    skills.confirm_empty(payload, {})
                self.assertEqual(skills.snapshot()['carry_id'], 'old-red')

    def test_running_chassis_or_arm_prevents_metadata_reset(self):
        with tempfile.TemporaryDirectory() as root:
            skills = self.empty_confirmation_ledger(root)
            payload = {'confirmed_empty': True, 'expected_carry_id': 'old-red'}
            skills.bridge._grasp_lock.acquire()
            with self.assertRaises(RuntimeError):skills.confirm_empty(payload, {})
            skills.bridge._grasp_lock.release()
            skills.bridge.process.snapshot = lambda: {'running': True, 'kind': 'visual_pursuit'}
            with self.assertRaises(RuntimeError):skills.confirm_empty(payload, {})
            self.assertEqual(skills.snapshot()['carry_id'], 'old-red')

    def test_stop_and_service_restart_do_not_infer_empty(self):
        with tempfile.TemporaryDirectory() as root:
            skills = self.empty_confirmation_ledger(root)
            skills.invalidate('emergency_stop')
            restarted = self.ledger(root)
            self.assertEqual(restarted.snapshot()['status'], 'carrying')
            restarted.assert_empty()  # Explicit next pick, not service boot.
            self.assertEqual(restarted.snapshot()['status'], 'ready_for_pick')
            self.assertFalse(restarted.snapshot()['physical_empty_verified'])

    def test_arrival_requires_current_job_visual_and_floor_clearance(self):
        with tempfile.TemporaryDirectory() as root:
            skills = self.ledger(root)
            path = Path(root) / "job.json"
            skills.data = {"carry_id": "x", "region_boot_id": skills.boot_id, "region_job": str(path)}
            import time
            job = {"status": "arrived", "carry_id": "x", "arrived_at": time.time(),
                   "arrival": {"anchor_visual_confirmed": True, "placement_clearance": True}}
            save_json(path, job)
            self.assertEqual(skills.arrival()["carry_id"], "x")
            for key in ("anchor_visual_confirmed", "placement_clearance"):
                bad = json.loads(json.dumps(job)); bad["arrival"][key] = False; save_json(path, bad)
                with self.assertRaises(RuntimeError): skills.arrival()
            save_json(path, job); skills.boot_id = "restarted"
            with self.assertRaises(RuntimeError): skills.arrival()

    def test_placement_unknown_does_not_reissue_release(self):
        with tempfile.TemporaryDirectory() as root:
            skills = self.ledger(root)
            skills.bridge._authorise_hardware = lambda headers: None
            skills.data = {"status": "placement_unknown", "place_id": "task"}
            calls = []
            result = skills.place({}, {}, request=lambda path, **kw: calls.append(path) or {
                "ok": False, "phase": "unknown"}, base_url="http://local")
            self.assertFalse(result["ok"])
            self.assertEqual(calls, ["http://local/api/arm-teach/place-status?job_id=task"])


if __name__ == "__main__":
    unittest.main()
