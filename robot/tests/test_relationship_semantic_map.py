import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tools.build_relationship_semantic_map import build_memory


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RelationshipSemanticMapTests(unittest.TestCase):
    def test_reviewed_geometry_wins_over_every_annotation_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_path = root / "map.pgm"
            yaml_path = root / "map.yaml"
            baseline_path = root / "baseline.json"
            Image.new("L", (20, 20), 255).save(map_path)
            yaml_path.write_text("resolution: 0.05\norigin: [0,0,0]\n", encoding="utf-8")
            baseline_path.write_text(json.dumps({"objects": [{
                "name": "玻璃门", "x_m": -2.9, "y_m": -0.8,
                "uncertainty_m": 0.8, "mode": "wall_surface",
                "evidence": [{"photo": "door.jpg", "bbox": [10, 10, 80, 90]}],
            }]}), encoding="utf-8")
            annotations = []
            for index, wrong_xy in enumerate(([4.0, 3.0], [3.0, 2.0])):
                path = root / f"annotation_{index}.json"
                path.write_text(json.dumps({
                    "schema": "semantic_object_memory.v1",
                    "map_sha256": digest(map_path),
                    "map_yaml_sha256": digest(yaml_path),
                    "objects": [{
                        "id": f"wrong_{index}", "name": "门", "category": "door",
                        "map_xy": wrong_xy, "semantic_ambiguous": False,
                        "evidence": [{
                            "observation_id": f"obs_{index}", "station_id": index + 1,
                            "confidence": 0.9, "views": [{
                                "image": "door.jpg", "bbox_1000": [10, 10, 80, 90],
                            }],
                        }],
                    }],
                }), encoding="utf-8")
                annotations.append(path)

            memory = build_memory(map_path, yaml_path, baseline_path, annotations)

            self.assertEqual(memory["objects"][0]["map_xy"], [-2.9, -0.8])
            self.assertEqual(memory["objects"][0]["annotation_support"]["direct_view_matches"], 2)
            self.assertEqual(len(memory["annotation_sources"]), 2)
            self.assertFalse(memory["objects"][0]["navigation_allowed"])
            self.assertFalse(memory["objects"][0]["localization_allowed"])
            self.assertEqual(memory["search_regions"][0]["map_xy"], [-2.9, -0.8])


if __name__ == "__main__":
    unittest.main()
