import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tools.build_semantic_release import build_release


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SemanticReleaseTests(unittest.TestCase):
    def test_release_keeps_sources_separate_and_does_not_promote_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_path = root / "map.pgm"
            yaml_path = root / "map.yaml"
            localization = root / "localization.json"
            observation = root / "observation.json"
            observation_image = root / "observation.png"
            qwen = root / "qwen.json"
            qwen_image = root / "qwen.png"
            qwen_report = root / "qwen_report.json"
            output = root / "release"

            Image.new("L", (20, 20), 255).save(map_path)
            yaml_path.write_text(
                "image: map.pgm\nresolution: 0.05\norigin: [0,0,0]\n",
                encoding="utf-8",
            )
            localization.write_text('{"entities":{}}', encoding="utf-8")
            observation.write_text(
                json.dumps({
                    "inspection_run": "20260917_190706",
                    "coverage": 0.884,
                    "stations": [{"station": 1, "x_m": 1.0, "y_m": 2.0}],
                }),
                encoding="utf-8",
            )
            Image.new("RGB", (20, 20), "white").save(observation_image)
            memory = {
                "schema": "semantic_object_memory.v1",
                "map_sha256": digest(map_path),
                "map_yaml_sha256": digest(yaml_path),
                "objects": [{
                    "id": "candidate",
                    "semantic_reviewed": False,
                    "position_verified": True,
                    "navigation_allowed": True,
                    "localization_allowed": True,
                }],
                "search_regions": [{"id": "candidate", "map_xy": [1, 2]}],
            }
            qwen.write_text(json.dumps(memory), encoding="utf-8")
            Image.new("RGB", (20, 20), "white").save(qwen_image)
            qwen_report.write_text("{}", encoding="utf-8")
            originals = {path: path.read_bytes() for path in (
                map_path, yaml_path, localization, observation, observation_image,
                qwen, qwen_image, qwen_report,
            )}

            report = build_release(
                map_path=map_path,
                yaml_path=yaml_path,
                localization_path=localization,
                observation_path=observation,
                observation_image=observation_image,
                qwen_path=qwen,
                qwen_image=qwen_image,
                qwen_report=qwen_report,
                output=output,
            )

            runtime = json.loads((output / "semantic_runtime.json").read_text(encoding="utf-8"))
            item = runtime["objects"][0]
            self.assertFalse(item["position_verified"])
            self.assertFalse(item["navigation_allowed"])
            self.assertFalse(item["localization_allowed"])
            self.assertTrue(item["requires_visual_recheck"])
            self.assertEqual(report["observation_station_count"], 1)
            self.assertEqual(report["search_region_count"], 1)
            self.assertTrue((output / "semantic_observation_20260917_190706.json").is_file())
            self.assertTrue((output / "qwen_quality_filtered_20260918.json").is_file())
            self.assertEqual(originals, {path: path.read_bytes() for path in originals})

    def test_release_rejects_wrong_inspection_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_path = root / "map.pgm"
            yaml_path = root / "map.yaml"
            localization = root / "localization.json"
            observation = root / "observation.json"
            observation_image = root / "observation.png"
            qwen = root / "qwen.json"
            qwen_image = root / "qwen.png"
            qwen_report = root / "qwen_report.json"
            Image.new("L", (5, 5), 255).save(map_path)
            yaml_path.write_text("resolution: 0.1\norigin: [0,0,0]\n", encoding="utf-8")
            localization.write_text("{}", encoding="utf-8")
            observation.write_text('{"inspection_run":"wrong"}', encoding="utf-8")
            Image.new("RGB", (5, 5)).save(observation_image)
            qwen.write_text(json.dumps({
                "schema": "semantic_object_memory.v1",
                "map_sha256": digest(map_path),
                "map_yaml_sha256": digest(yaml_path),
                "objects": [],
            }), encoding="utf-8")
            Image.new("RGB", (5, 5)).save(qwen_image)
            qwen_report.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "20260917_190706"):
                build_release(
                    map_path=map_path, yaml_path=yaml_path,
                    localization_path=localization, observation_path=observation,
                    observation_image=observation_image, qwen_path=qwen,
                    qwen_image=qwen_image, qwen_report=qwen_report,
                    output=root / "release",
                )


if __name__ == "__main__":
    unittest.main()
