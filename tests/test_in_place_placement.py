"""Offline release-only compilation/dispatch; all hardware requests are fakes."""
import ast
import json
from pathlib import Path
import re
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from services.track.final_work.gpu.planner import AgentPlanner
from services.track.final_work.gpu.robot_client import RobotClient
from services.track.final_work.gpu.robot_skill_catalog import is_in_place_release
from services.track.final_work.pi.robot_skills import RobotSkills


ROOT = Path(__file__).resolve().parents[1]


def compiler_class():
    # Load only the real compiler class; do not import the GPU server/models.
    path = ROOT / "services/track/final_work/gpu/qwen_hub.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SharedAgentPlanner")
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    scope = {"__package__": "services.track.final_work.gpu", "json": json, "re": re}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope["SharedAgentPlanner"]


class InPlacePlanningTests(unittest.TestCase):
    def test_release_phrases_compile_without_model_map_or_destination(self):
        backend = MagicMock()
        compiler = compiler_class()(backend)
        for instruction in ("放下", "原地放下", "直接原地放下", "放下小物块",
                "把小物块放下", "请把手里的绿色小物块直接原地放下",
                "把手里的物品放到地上", "放在这里", "就地放下物品吧", "put it down here",
                "原地放下物品", "将夹子中的方块放在地面上", "将夹子中的方块放在地面", "放下夹子中的方块"):
            with self.subTest(instruction=instruction):
                plan = compiler.plan(instruction, {})
                self.assertEqual(plan["intent"], "place")
                self.assertEqual([s["action"] for s in plan["steps"]],
                    ["place_on_floor", "verify_completion", "report_result"])
                for step in plan["steps"]:
                    self.assertEqual(step["target"]["selection"], "in_place")
                    self.assertFalse(step["target"]["allow_chassis_motion"])
                    self.assertNotIn("lidar", step["sensors"])
                    self.assertNotIn("semantic_map", step["sensors"])
                self.assertEqual(plan["source"], "deterministic_in_place_release")
        backend.generate.assert_not_called()

    def test_negation_questions_and_other_actions_not_deterministic_release(self):
        for text in ("不要放下", "别原地放下", "能不能原地放下？", "原地放下可以吗",
                "去柜子边放下", "把物品放在椅子旁", "放在桌面上", "先前进再放下",
                "去柜子边夹取绿色小物块然后原地放下", "拿起来放好"):
            with self.subTest(text=text):
                self.assertFalse(is_in_place_release(text))

    def test_explicit_release_overrides_untrusted_model_request_to_navigate(self):
        plan = compiler_class()(None)._compile("原地放下", {
            "intent": "navigate", "destination_query": "椅子", "target_query": "椅子"}, "test")
        self.assertEqual(plan["intent"], "place")
        self.assertEqual(plan["steps"][0]["target"]["query"], "原地")
        self.assertNotIn("navigate_to", [s["action"] for s in plan["steps"]])

    def test_named_place_still_requires_arrival_and_missing_place_not_guessed(self):
        compiler = compiler_class()(None)
        plan = compiler._compile("把物品放在椅子旁", {
            "intent": "place", "destination_query": "椅子旁"}, "test")
        self.assertIn("navigate_to", [s["action"] for s in plan["steps"]])
        self.assertEqual(plan["steps"][-3]["target"]["selection"], "named_region")
        vague = compiler._compile("拿起来放好", {"intent": "pick_and_place"}, "test")
        self.assertEqual(vague["intent"], "unsupported")

    def test_pick_then_here_keeps_pick_but_no_transport(self):
        plan = compiler_class()(None)._compile("去柜子边夹取绿色小物块，然后原地放下", {
            "intent": "pick_and_place", "target_query": "绿色小物块",
            "search_near_query": "柜子", "destination_selection": "in_place"}, "test")
        actions = [s["action"] for s in plan["steps"]]
        self.assertIn("patrol_search", actions)
        self.assertIn("model_grasp", actions)
        self.assertNotIn("navigate_to", actions)
        self.assertNotIn("locate_destination", actions)
        self.assertEqual(actions[-3:], ["place_on_floor", "verify_completion", "report_result"])

    def test_web_planner_does_not_require_api_credentials_for_plain_release(self):
        stub = SimpleNamespace(SharedAgentPlanner=compiler_class())
        with patch.dict("sys.modules", {"services.track.final_work.gpu.qwen_hub": stub}):
            planner = AgentPlanner()
            planner._ensure_planner = MagicMock(side_effect=AssertionError("model must not load"))
            plan = planner.plan("放下小物块", None, {"provider": "deepseek_flash_api"})
        self.assertEqual(plan["intent"], "place")
        self.assertEqual(plan["source"], "deterministic_in_place_release")

    def test_dispatch_starts_preview_then_place_not_navigation(self):
        step = compiler_class()(None).plan("原地放下", {})["steps"][0]
        client = RobotClient()
        calls = []
        client.request = lambda path, payload=None, **kw: calls.append((path, payload)) or {"ok": True}
        result = client.dispatch_step(step, "原地放下")
        self.assertEqual(result["status"], "completed")
        self.assertEqual([path for path, _ in calls], ["/actions/preview/start", "/skills/place-on-floor"])
        self.assertEqual(calls[-1][1]["selection"], "in_place")

    def test_release_copies_match_deployable_sources(self):
        for path in ("gpu/qwen_hub.py", "gpu/planner.py", "gpu/robot_skill_catalog.py", "pi/robot_skills.py", "pi/placement_extension.py"):
            self.assertEqual((ROOT / "services/track/final_work" / path).read_bytes(),
                (ROOT / "robot/grasp_range_integration/track_web_integration" / path).read_bytes(), path)


class InPlaceExecutionTests(unittest.TestCase):
    def skills(self, root):
        navigation = SimpleNamespace(runtime_root=Path(root) / "navigation",
                                     live_status=MagicMock(side_effect=AssertionError("no localization")))
        bridge = SimpleNamespace(map_navigation=navigation, _authorise_hardware=MagicMock(),
            _grasp_lock=threading.Lock(), _grasp_cancel=threading.Event(),
            process=SimpleNamespace(snapshot=lambda: {"running": False}))
        skills = RobotSkills(bridge)
        skills.data = {"status": "carrying", "carry_id": "a" * 32,
                       "ground_pose": {"s0": 1450, "s1": 1100, "s2": 1800},
                       "destination_query": "过期椅子", "region_boot_id": "old-boot"}
        skills.arrival = MagicMock(side_effect=AssertionError("no region arrival needed"))
        return skills

    def target(self):
        return compiler_class()(None).plan("原地放下", {})["steps"][0]["target"]

    def request(self, calls):
        def fake(path, payload=None, **kwargs):
            calls.append((path, payload))
            if path.endswith("/api/status"):
                return {"ble": True, "cameras": {side: {"frames": 10, "age_s": .1}
                                                     for side in ("left", "right")}}
            if path.endswith("/api/arm-teach/state"):
                return {"state": {"known": True, "claw_value": 0}}
            if path.endswith("/api/arm-teach/place"):
                return {"phase": "done", "ok": True}
            raise AssertionError(path)
        return fake

    def test_here_ignores_old_map_arrival_and_uses_ground_pose_only_once(self):
        with tempfile.TemporaryDirectory() as root:
            skills, calls = self.skills(root), []
            result = skills.place(self.target(), {}, request=self.request(calls), base_url="http://fake")
            self.assertTrue(result["ok"])
            self.assertFalse(result["physical_place_verified"])
            self.assertEqual(skills.data["placement_mode"], "in_place")
            payload = next(payload for path, payload in calls if path.endswith("/api/arm-teach/place"))
            self.assertEqual(payload["params"], {"s0": 1450, "s1": 1100, "s2": 1800})
            self.assertRegex(payload["job_id"], r"^[a-f0-9_]{16,100}$")
            skills.arrival.assert_not_called()
            skills.bridge.map_navigation.live_status.assert_not_called()
            skills.place(self.target(), {}, request=self.request(calls), base_url="http://fake")
            self.assertEqual(sum(path.endswith("/api/arm-teach/place") for path, _ in calls), 1)

    def test_running_drive_and_no_carry_do_not_enqueue_release(self):
        with tempfile.TemporaryDirectory() as root:
            skills = self.skills(root)
            request = MagicMock(side_effect=AssertionError("no arm command expected"))
            skills.bridge.process.snapshot = lambda: {"running": True, "kind": "region_navigation"}
            with self.assertRaises(RuntimeError):
                skills.place(self.target(), {}, request=request, base_url="http://fake")
            skills.bridge.process.snapshot = lambda: {"running": False}
            skills.data["status"] = "empty"
            with self.assertRaises(RuntimeError):
                skills.place(self.target(), {}, request=request, base_url="http://fake")
            request.assert_not_called()

    def test_ambiguous_release_result_queries_old_id_never_repeats_release(self):
        with tempfile.TemporaryDirectory() as root:
            skills = self.skills(root)
            skills.data.update(status="placement_unknown", place_id="old_job")
            request = MagicMock(return_value={"phase": "unknown", "ok": False})
            result = skills.place(self.target(), {}, request=request, base_url="http://fake")
            self.assertFalse(result["ok"])
            request.assert_called_once_with("http://fake/api/arm-teach/place-status?job_id=old_job", timeout=4)

    def test_here_cannot_silently_allow_drive_or_unmapped_height(self):
        with tempfile.TemporaryDirectory() as root:
            skills = self.skills(root)
            request = MagicMock()
            for update in ({"allow_chassis_motion": True}, {"surface": "table"}, {"reference": "somewhere"}):
                with self.assertRaises(ValueError):
                    skills.place({**self.target(), **update}, {}, request=request, base_url="http://fake")
            request.assert_not_called()

    def test_preview_restart_adopts_history_without_region_job_or_motor_commands(self):
        from flask import Flask
        from services.track.final_work.pi.placement_extension import register_placement
        with tempfile.TemporaryDirectory() as root:
            skills = self.skills(root)
            skills.data.update(placement_mode='in_place', in_place_requested_at=1000.)
            skills.save()
            arm = SimpleNamespace(_state_lock=threading.RLock(), _arm_jobs_pending=0,
                _exec_status={'active': False}, _state={'known': False}, CLAW_CLOSE=0,
                SERVOS=[{'key': key, 'home': home, 'min': low, 'max': high}
                        for key, home, low, high in (
                            ('s0', 1500, 500, 2500), ('s1', 1700, 800, 1700), ('s2', 2000, 1500, 2200))])
            arm.snapshot = lambda: dict(arm._state)
            send = MagicMock(side_effect=AssertionError('adopting metadata must not send BLE'))
            app = Flask(__name__)
            def local_path(value):
                if str(value).endswith('robot_skills_state.json'):
                    return skills.path
                return Path(root) / 'evidence'
            with patch('services.track.final_work.pi.placement_extension.Path', side_effect=local_path):
                register_placement(app, arm, send)
                with patch('services.track.final_work.pi.placement_extension.time.time', return_value=1001.):
                    response = app.test_client().post('/api/arm-teach/adopt-carry', json={})
            self.assertEqual(response.status_code, 200)
            self.assertFalse(response.get_json()['hardware_readback'])
            self.assertTrue(arm._state['known'])
            self.assertEqual(arm._state['claw_value'], 0)
            send.assert_not_called()

    def test_stale_here_request_cannot_adopt_carry_history(self):
        from flask import Flask
        from services.track.final_work.pi.placement_extension import register_placement
        with tempfile.TemporaryDirectory() as root:
            skills = self.skills(root)
            skills.data.update(placement_mode='in_place', in_place_requested_at=1.)
            skills.save()
            app = Flask(__name__)
            arm, send = SimpleNamespace(), MagicMock()
            def local_path(value):
                return skills.path if str(value).endswith('robot_skills_state.json') else Path(root) / 'evidence'
            with patch('services.track.final_work.pi.placement_extension.Path', side_effect=local_path):
                register_placement(app, arm, send)
                with patch('services.track.final_work.pi.placement_extension.time.time', return_value=1000.):
                    response = app.test_client().post('/api/arm-teach/adopt-carry', json={})
            self.assertEqual(response.status_code, 409)
            send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
