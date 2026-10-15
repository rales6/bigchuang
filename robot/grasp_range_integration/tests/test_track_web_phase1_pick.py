"""Offline tests for the two web adapters, without contacting Pi/3090."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch
import sys


ROOT = Path(__file__).resolve().parents[1] / "track_web_integration"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PLAN = {
    "intent": "pick", "source": "test", "steps": [
        {"action": "inspect_semantic_map", "target": {"query": "红色小物块"}},
        {"action": "patrol_search", "target": {
            "query": "柜子", "target_query": "红色小物块"}},
        {"action": "align_for_grasp", "target": {"query": "红色小物块"}},
        {"action": "model_grasp", "target": {"query": "红色小物块"}},
        {"action": "verify_grasp", "target": {"query": "红色小物块"}},
    ],
}


class WebPhase1PickTests(unittest.TestCase):
    def test_named_anchor_pick_marks_only_search_step(self):
        module = load("phase1_web_adaptation", ROOT / "gpu/plan_adaptation.py")
        value = module.normalize_web_plan(PLAN)
        patrol = value["steps"][1]
        self.assertEqual(patrol["target"]["integration_workflow"],
                         "phase1_deepseek_pick_v1")
        self.assertEqual(value["steps"][2]["action"], "align_for_grasp")
        self.assertNotIn("integration_workflow", value["steps"][2]["target"])
        self.assertNotIn("integration_workflow", PLAN["steps"][1]["target"])

    def test_visible_block_does_not_skip_named_cabinet(self):
        module = load("phase1_web_adaptation", ROOT / "gpu/plan_adaptation.py")
        value = module.adapt_plan_to_live_view(
            module.normalize_web_plan(PLAN),
            {"status": "visible_both", "inside_grasp_region": True,
             "observed_at_ms": 123},
        )
        self.assertEqual(value["steps"][1]["action"], "patrol_search")
        self.assertEqual(value["live_view_probe"]["skip_suppressed"],
                         "named_anchor_not_geographically_verified")

    def test_unanchored_pick_keeps_old_current_view_path(self):
        module = load("phase1_web_adaptation", ROOT / "gpu/plan_adaptation.py")
        plan = json.loads(json.dumps(PLAN, ensure_ascii=False))
        plan["steps"][1]["target"]["query"] = "红色小物块"
        value = module.normalize_web_plan(plan)
        self.assertNotIn("integration_workflow", value["steps"][1]["target"])

    def test_phase1_command_is_exclusive_result_gate(self):
        module = load("phase1_web_map_adapter", ROOT / "pi/map_navigation_adapter.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = module.MapNavigationAdapter()
            adapter.car1_root = root
            adapter.release_root = root / "release"
            adapter.runtime_root = root / "runtime"
            (root / ".venv-pi/bin").mkdir(parents=True)
            (root / ".venv-pi/bin/python3").touch()
            (adapter.release_root / "grasp_range_integration/phase1").mkdir(parents=True)
            (adapter.release_root / "grasp_range_integration/phase1/verified_search.py").touch()
            (adapter.release_root / "ble_scene_17.pgm").touch()
            (adapter.release_root / "ble_scene_17.yaml").touch()
            adapter.runtime_root.mkdir()
            decision = adapter.runtime_root / "one.json"
            decision.write_text("{}", encoding="utf-8")
            command, env = adapter.phase1_pick_command(decision)
            self.assertEqual(command[3], "grasp_range_integration.phase1.verified_search")
            self.assertIn("--execute-verified-search", command)
            self.assertNotIn("--enable-motion", command)
            self.assertEqual(env["CAR_ROOT"], str(root))

    def test_phase1_planner_rejects_generic_target_without_a_known_class(self):
        module = load("phase1_web_map_adapter", ROOT / "pi/map_navigation_adapter.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = module.MapNavigationAdapter()
            adapter.car1_root = root
            adapter.release_root = root / "release"
            adapter.runtime_root = root / "runtime"
            planner = adapter.release_root / "grasp_range_integration/phase1/plan_phase1_deepseek.sh"
            planner.parent.mkdir(parents=True)
            planner.touch()

            def fake_run(command, **_kwargs):
                Path(command[-1]).write_text(json.dumps({
                    "instruction": "去柜子找绿色小物块并夹取", "status": "executable",
                    "target": {"query": "绿色物品", "color": "green", "class_name": "any_object", "requested_count": 1},
                    "phase_control": {"requested_final_action": "grasp"},
                }, ensure_ascii=False), encoding="utf-8")
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch.dict(module.os.environ, {"FINAL_WORK_ALLOW_DEEPSEEK_SEMANTIC_UPLOAD": "1"}), \
                    patch.object(module.subprocess, "run", fake_run):
                with self.assertRaisesRegex(ValueError, "单个、明确类别"):
                    adapter.prepare_phase1_pick("去柜子找绿色小物块并夹取")

    def test_phase1_planner_requires_explicit_external_upload_opt_in(self):
        module = load("phase1_web_map_adapter", ROOT / "pi/map_navigation_adapter.py")
        with tempfile.TemporaryDirectory() as temporary:
            adapter = module.MapNavigationAdapter()
            adapter.runtime_root = Path(temporary) / "navigation"
            with patch.dict(module.os.environ, {}, clear=True), \
                    patch.object(module.subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "需明确允许发送"):
                    adapter.prepare_phase1_pick("去柜子边上夹取红色小物块")
                run.assert_not_called()

    def test_persisted_upload_consent_reaches_planner(self):
        module = load("phase1_web_map_adapter", ROOT / "pi/map_navigation_adapter.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = module.MapNavigationAdapter()
            adapter.car1_root = root
            adapter.release_root = root / "release"
            adapter.runtime_root = root / "navigation"
            planner = adapter.release_root / "grasp_range_integration/phase1/plan_phase1_deepseek.sh"
            planner.parent.mkdir(parents=True)
            planner.touch()
            (root / "deepseek_semantic_upload.allowed").write_text("1\n", encoding="utf-8")

            def fake_run(command, **_kwargs):
                Path(command[-1]).write_text(json.dumps({
                    "instruction": "去柜子边上夹取一个红色的小物块",
                    "status": "executable",
                    "target": {"query": "红色小物块", "color": "red", "class_name": "block", "requested_count": 1},
                    "phase_control": {"requested_final_action": "grasp"},
                }, ensure_ascii=False), encoding="utf-8")
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch.dict(module.os.environ, {}, clear=True), \
                    patch.object(module.subprocess, "run", fake_run):
                decision, _path = adapter.prepare_phase1_pick("去柜子边上夹取一个红色的小物块")
            self.assertEqual(decision["status"], "executable")

    def test_bridge_dispatches_marked_pick_to_verified_search(self):
        package = types.ModuleType("final_work")
        package.__path__ = []
        subpackage = types.ModuleType("final_work.pi")
        subpackage.__path__ = []
        adapter_module = load("final_work.pi.map_navigation_adapter",
                              ROOT / "pi/map_navigation_adapter.py")
        with patch.dict(sys.modules, {
                "final_work": package, "final_work.pi": subpackage,
                "final_work.pi.map_navigation_adapter": adapter_module}):
            bridge_module = load("final_work.pi.robot_bridge",
                                 ROOT / "pi/robot_bridge.py")
        bridge = bridge_module.RobotBridge.__new__(bridge_module.RobotBridge)
        bridge._grasp_lock = MagicMock()
        bridge._grasp_lock.locked.return_value = False
        bridge._authorise_hardware = MagicMock()
        bridge.map_navigation = MagicMock()
        bridge.map_navigation.prepare_phase1_pick.return_value = (
            {"status": "executable", "instruction": "去柜子边上夹取红色小物块"},
            Path("/safe/decision.json"))
        bridge.map_navigation.phase1_pick_command.return_value = (
            ["python", "-m", "verified_search"], {"CAR_ROOT": "/safe"})
        bridge.map_navigation.car1_root = Path("/safe")
        bridge.process = MagicMock()
        bridge.process.snapshot.return_value = {"running": False}
        bridge.process.start.return_value = {"running": True, "log": "/safe/run.log"}
        payload = {"instruction": "去柜子边上夹取红色小物块", "real": True,
                   "target": {"query": "柜子", "target_query": "红色小物块",
                              "integration_workflow": "phase1_deepseek_pick_v1",
                              "requested_intent": "pick"}}
        result = bridge.map_mission(payload, {}, purpose="observation_search")
        self.assertEqual(result["integration_workflow"], "phase1_deepseek_pick_v1")
        bridge.map_navigation.request_decision.assert_not_called()
        bridge.map_navigation.mission_command.assert_not_called()
        bridge.map_navigation.phase1_pick_command.assert_called_once()
        bridge._authorise_hardware.assert_called_once()


if __name__ == "__main__":
    unittest.main()
