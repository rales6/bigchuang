from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from final_work.pi.map_navigation_adapter import MapNavigationAdapter


class MapNavigationAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.car1 = root / "car1.0"
        self.release = self.car1 / "ble_scene_17_relationship_20260919"
        self.runtime = root / "track" / "final_work" / "data" / "navigation"
        (self.car1 / "maps").mkdir(parents=True)
        adapter = MapNavigationAdapter()
        adapter.car1_root = self.car1
        adapter.release_root = self.release
        adapter.runtime_root = self.runtime
        adapter.search_api_url = "http://gpu:8010"
        self.adapter = adapter

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_json(self, path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def test_falls_back_to_existing_qwen_semantics(self) -> None:
        path = self.car1 / "maps" / "ble_scene_17_semantics_qwen.json"
        self._write_json(path, {"entities": {"cabinet": {"name": "柜子"}}})
        value = self.adapter.semantic_map()
        self.assertEqual(value["_status"], "compatibility_fallback")
        self.assertEqual(value["_entity_count"], 1)
        self.assertFalse(value["_release_available"])

    def test_stable_release_has_priority(self) -> None:
        self._write_json(
            self.car1 / "maps" / "ble_scene_17_semantics_qwen.json",
            {"entities": {"old": {}}},
        )
        self._write_json(
            self.release / "semantic_runtime.json",
            {"landmarks": [{"id": "stable"}, {"id": "stable2"}]},
        )
        value = self.adapter.semantic_map()
        self.assertEqual(value["_status"], "stable_release")
        self.assertEqual(value["_entity_count"], 2)
        self.assertTrue(value["_release_available"])

    def test_layer_catalog_is_fixed_and_prefers_semantic(self) -> None:
        (self.car1 / "maps" / "ble_scene_17.pgm").write_bytes(b"P5\n1 1\n255\n\x00")
        (self.car1 / "maps" / "ble_scene_17_semantic_qwen.png").write_bytes(b"png")
        result = self.adapter.layers()
        ids = {item["id"] for item in result["layers"]}
        self.assertEqual(ids, {"base", "semantic_compat"})
        self.assertTrue(next(item for item in result["layers"] if item["id"] == "semantic_compat")["is_default"])
        self.assertIsNone(self.adapter.layer_path("../../etc/passwd"))

    def test_decisions_are_written_only_under_final_work(self) -> None:
        (self.release / "ble_scene_17.pgm").parent.mkdir(parents=True, exist_ok=True)
        (self.release / "ble_scene_17.pgm").write_bytes(b"map")
        (self.release / "ble_scene_17.yaml").write_text("image: ble_scene_17.pgm")
        (self.release / "semantic_localization_landmarks.json").write_text("{}")
        python = self.car1 / ".venv-pi" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text("")
        path = self.adapter.save_decision({"task_id": "search/../../bad", "status": "executable"})
        self.assertEqual(path.parent, self.runtime)
        command, environment = self.adapter.mission_command(path, real=False)
        self.assertEqual(command[-1], "--simulate")
        self.assertIn(str(self.release / "semantic_localization_landmarks.json"), command)
        self.assertEqual(environment["SEARCH_3090_URL"], "http://gpu:8010")

    def test_relationship_release_is_default_layer(self) -> None:
        self._write_json(self.release / "semantic_runtime.json", {"landmarks": [{"id": "desk"}]})
        (self.release / "ble_scene_17.pgm").write_bytes(b"map")
        (self.release / "semantic_relationship_map_20260919.png").write_bytes(b"png")
        result = self.adapter.layers()
        selected = next(item for item in result["layers"] if item["is_default"])
        self.assertEqual(selected["id"], "relationship_release")
        self.assertEqual(result["semantic"]["status"], "stable_release")

    def test_reviewed_qwen_label_overrides_stale_planning_label(self) -> None:
        source = {
            "entities": {
                "wall": {
                    "semantic_label": "cabinet",
                    "qwen_annotation": {
                        "name_zh": "墙面",
                        "semantic_label": "wall",
                        "confidence": 0.95,
                        "needs_human_review": False,
                    },
                }
            }
        }
        planned = self.adapter.planner_semantic_map(source)
        self.assertEqual(planned["entities"]["wall"]["semantic_label"], "wall")
        self.assertEqual(planned["entities"]["wall"]["original_semantic_label"], "cabinet")
        self.assertEqual(source["entities"]["wall"]["semantic_label"], "cabinet")


if __name__ == "__main__":
    unittest.main()
