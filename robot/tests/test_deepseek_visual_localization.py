import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from deepseek_visual_localization import (  # noqa: E402
    candidate_scene_descriptions,
    load_config,
    parse_rank_response,
)


class DeepSeekVisionLocalizationTests(unittest.TestCase):
    def test_placeholder_key_is_not_configured(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"api_key": "请填写你的DeepSeek_API_Key"}), encoding="utf-8")
            config = load_config(path, module_directory=ROOT)
            self.assertFalse(config.configured)

    def test_real_key_is_configured_without_exposing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"api_key": "REDACTED_API_KEY_USE_LOCAL_CONFIG"}), encoding="utf-8")
            config = load_config(path, module_directory=ROOT)
            self.assertTrue(config.configured)
            self.assertNotIn("REDACTED_API_KEY_USE_LOCAL_CONFIG", repr(config))

    def test_response_must_select_an_allowed_candidate(self):
        result = parse_rank_response(
            json.dumps({
                "selected_candidate_id": "C99",
                "confidence": 0.99,
                "second_candidate_id": "C01",
                "second_confidence": 0.2,
            }),
            {"C01", "C02"},
        )
        self.assertEqual("uncertain", result.selected_candidate_id)

    def test_candidate_description_uses_relative_structure_geometry(self):
        config_payload = {
            "api_key": "sk-test",
            "landmark_categories": ["glass_wall", "cabinet"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config_payload), encoding="utf-8")
            config = load_config(path, module_directory=ROOT)
        candidate = SimpleNamespace(
            pose=SimpleNamespace(x_m=0.0, y_m=0.0, yaw_rad=math.pi / 2),
            reliable=True,
            rmse_m=0.05,
            inlier_ratio=0.8,
        )
        semantic = {"objects": [{
            "name": "玻璃墙",
            "category": "glass_wall",
            "map_xy": [0.0, 3.0],
            "position_uncertainty_m": 0.3,
            "semantic_reviewed": True,
        }]}
        rows = candidate_scene_descriptions([candidate], semantic, config)
        self.assertEqual("C01", rows[0]["candidate_id"])
        self.assertEqual("center", rows[0]["expected_visible_structures"][0]["side"])

    def test_movable_furniture_is_context_only_and_absence_never_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"api_key": "sk-test"}), encoding="utf-8")
            config = load_config(path, module_directory=ROOT)
        candidate = SimpleNamespace(
            pose=SimpleNamespace(x_m=0.0, y_m=0.0, yaw_rad=0.0),
            reliable=True,
            rmse_m=0.05,
            inlier_ratio=0.8,
            score=1.0,
        )
        semantic = {"objects": [
            {
                "name": "玻璃墙",
                "category": "glass_wall",
                "map_xy": [2.0, 0.0],
                "position_uncertainty_m": 0.2,
                "semantic_reviewed": True,
            },
            {
                "name": "桌边椅子",
                "category": "table",
                "map_xy": [1.0, 0.0],
                "position_uncertainty_m": 0.2,
                "semantic_reviewed": True,
            },
        ]}
        row = candidate_scene_descriptions([candidate], semantic, config)[0]
        self.assertEqual(["玻璃墙"], [item["name"] for item in row["expected_visible_structures"]])
        self.assertEqual(["桌边椅子"], [item["name"] for item in row["context_only_structures"]])
        self.assertFalse(row["expected_visible_structures"][0]["absence_is_negative"])

    def test_structure_well_outside_camera_fov_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "api_key": "sk-test",
                "camera_horizontal_fov_deg": 60,
                "camera_bearing_margin_deg": 0,
            }), encoding="utf-8")
            config = load_config(path, module_directory=ROOT)
        candidate = SimpleNamespace(
            pose=SimpleNamespace(x_m=0.0, y_m=0.0, yaw_rad=0.0),
            reliable=True,
            rmse_m=0.05,
            inlier_ratio=0.8,
            score=1.0,
        )
        semantic = {"objects": [{
            "name": "侧后方窗户",
            "category": "window",
            "map_xy": [0.0, 2.0],
            "position_uncertainty_m": 0.1,
            "semantic_reviewed": True,
        }]}
        row = candidate_scene_descriptions([candidate], semantic, config)[0]
        self.assertEqual([], row["expected_visible_structures"])


if __name__ == "__main__":
    unittest.main()
