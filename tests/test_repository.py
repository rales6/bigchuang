"""Monorepo smoke checks: no hardware, network, services or remote writes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_release_bytes_are_not_normalized_by_git(self):
        result = subprocess.run(['git', 'check-attr', 'text', '--',
                                 'maps/current/semantic_runtime.json'], cwd=ROOT,
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), 'maps/current/semantic_runtime.json: text: unset')

    def test_current_map_and_semantic_runtime_exist(self):
        base = ROOT/"maps/current"
        self.assertTrue((base/"ble_scene_17.pgm").read_bytes().startswith(b"P5"))
        semantic = json.loads((base/"semantic_runtime.json").read_text(encoding="utf-8"))
        self.assertIsInstance(semantic, dict)
        self.assertTrue(semantic.get("objects") or semantic.get("entities"))

    def test_history_is_preserved_without_committing_changes(self):
        result = subprocess.run(["git", "cat-file", "-t", "0fea8a9"], cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "commit")

    def test_android_source_not_build_artifacts(self):
        self.assertTrue((ROOT/"apps/android/app/src/main/AndroidManifest.xml").is_file())
        self.assertTrue((ROOT/"apps/android/gradle/wrapper/gradle-wrapper.jar").is_file())
        self.assertFalse((ROOT/"apps/android/app/build").exists())

    def test_required_runtime_boundaries(self):
        for path in ("firmware/car", "robot/raspberry_pi", "services/track/pursuit",
                     "services/grasp/pi", "services/grasp/gpu", "services/perception", "simulation/car_sim"):
            self.assertTrue((ROOT/path).is_dir(), path)

    def test_correction_module_loaded_without_motor_import(self):
        path = ROOT/"services/grasp/pi/grasp_pose_correction.py"
        spec = importlib.util.spec_from_file_location("local_pose_check", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = {"s0": 1410,"s1": 969,"s2": 1720}
        result = module.corrected_grasp_pose(target, [{"key":"s2","min":1500,"max":2200}],
                                             {"ARM_GRASP_S0_NUDGE_US":"-30","ARM_GRASP_S2_NUDGE_US":"20"})
        self.assertEqual(result, {**target,"s2":1740})
        self.assertEqual(target["s0"], result["s0"])

    def test_source_manifest_includes_both_machines(self):
        manifest = json.loads((ROOT/"docs/maintenance/source_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual({m["origin"] for m in manifest["files"].values()}, {"local","pi","gpu"})
        self.assertFalse(manifest["remote_files_modified"])


if __name__ == "__main__":
    unittest.main()
