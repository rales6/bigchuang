import tempfile
from pathlib import Path
import unittest

from raspberry_pi.companion_service import (
    DEFAULT_CONFIG,
    build_task_plan,
    mapping_command,
    save_config,
    validate_config,
    annotation_command,
    bluetooth_log_health,
)
from raspberry_pi.bluetooth_recovery import recover_bluetooth


class CompanionConfigurationTests(unittest.TestCase):
    def test_rejects_unknown_and_unsafe_configuration(self):
        with self.assertRaisesRegex(ValueError, "不支持的配置项"):
            validate_config({"mapping": {"shell": "rm"}})
        with self.assertRaisesRegex(ValueError, "必须在"):
            validate_config({"mapping": {"speed": 900}})
        with self.assertRaisesRegex(ValueError, "路径"):
            validate_config({"mapping": {"output": "../outside"}})
        with self.assertRaisesRegex(ValueError, "路径"):
            validate_config({"mapping": {"output": "maps/../outside"}})

    def test_configuration_round_trip_and_fixed_command(self):
        value = validate_config({"mapping": {"speed": 280, "max_drive_speed": 400}})
        with tempfile.TemporaryDirectory() as directory:
            saved = save_config(value, Path(directory) / "config.json")
        command = mapping_command(saved)
        self.assertIn("raspberry_pi.autonomous_mapping", command)
        self.assertIn("--enable-motion", command)
        self.assertEqual(command[command.index("--speed") + 1], "280")
        self.assertNotIn("shell", command)

    def test_bluetooth_and_dense_annotation_configuration_reach_commands(self):
        value = validate_config({
            "bluetooth": {
                "device_name": "Test-Car",
                "device_address": "AA:BB:CC:DD:EE:FF",
                "auto_recover": True,
            },
            "annotation": {"max_targets": 32, "photos_per_target": 3},
        })
        mapping = mapping_command(value)
        annotation = annotation_command(value)
        self.assertEqual(mapping[mapping.index("--ble-address") + 1], "AA:BB:CC:DD:EE:FF")
        self.assertEqual(annotation[annotation.index("--photos-per-target") + 1], "3")
        self.assertIn("--ignore-trajectory-pose", annotation)
        self.assertEqual(annotation[annotation.index("--localization-min-margin") + 1], "0.06")
        self.assertEqual(annotation[annotation.index("--localization-min-accepted") + 1], "4")
        self.assertEqual(annotation[annotation.index("--localization-required-moves") + 1], "0")

    def test_bluetooth_recovery_uses_fixed_argv_without_shell(self):
        calls = []

        class Completed:
            returncode = 0
            stdout = "ok"
            stderr = ""

        def run(command, **kwargs):
            calls.append((command, kwargs))
            return Completed()

        result = recover_bluetooth("64:b7:08:69:9a:e6", run=run)
        self.assertTrue(result.ok)
        self.assertEqual(calls[0][0][-1], "64:B7:08:69:9A:E6")
        self.assertNotIn("shell", calls[0][1])

    def test_runtime_disconnect_is_a_fatal_bluetooth_health_signal(self):
        health = bluetooth_log_health(
            "底盘链路连接成功 1/8\nConnectionError: BLE peripheral disconnected"
        )
        self.assertTrue(health["connected"])
        self.assertTrue(health["fatal"])
        self.assertGreaterEqual(health["failures"], 1)


class CompanionTaskPlanTests(unittest.TestCase):
    def test_task_plan_exposes_completed_and_active_steps(self):
        decision = {
            "instruction": "去柜子边找红色小物块",
            "steps": [
                {"id": "step_01", "action": "localize", "success": "定位成功"},
                {"id": "step_02", "action": "plan_search_route", "success": "路线生成"},
                {"id": "step_03", "action": "navigate_to_search_area", "success": "到达"},
            ],
        }
        mission = {"result": "running", "events": [{"type": "localized"}, {"type": "route_planned"}]}
        live = {"command": {"state": "navigating", "reason": "前往观察点"}}
        plan = build_task_plan(decision, mission, live)
        self.assertEqual([item["status"] for item in plan["steps"]], ["completed", "completed", "active"])
        self.assertEqual(plan["map_url"], "/task/map.png")

    def test_success_marks_every_step_complete(self):
        decision = {"steps": [{"action": "localize"}, {"action": "stop_and_report"}]}
        plan = build_task_plan(decision, {"result": "success", "events": []}, {})
        self.assertTrue(all(item["status"] == "completed" for item in plan["steps"]))

    def test_live_terminal_state_works_without_mission_file(self):
        decision = {"steps": [{"action": "localize"}, {"action": "stop_and_report"}]}
        live = {"command": {"state": "success", "reason": "已找到目标"}}
        plan = build_task_plan(decision, {}, live)
        self.assertEqual(plan["status"], "success")
        self.assertTrue(all(item["status"] == "completed" for item in plan["steps"]))


if __name__ == "__main__":
    unittest.main()
