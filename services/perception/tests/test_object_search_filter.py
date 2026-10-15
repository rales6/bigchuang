from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen_grounded_tracker.object_search_filter import detection_matches, filter_detections


class ObjectSearchFilterTests(unittest.TestCase):
    def test_yolo_class_and_red_pixels_both_required(self):
        image = Image.new("RGB", (200, 100), "white")
        ImageDraw.Draw(image).rectangle((20, 20, 80, 80), fill=(230, 20, 20))
        detection = {"camera_id": "left", "class_name": "bottle", "confidence": 0.8, "bbox_1000": [100, 200, 400, 800]}
        evidence = detection_matches(detection, {"class_name": "bottle", "color": "red"}, image)
        self.assertTrue(evidence["matched"])
        self.assertEqual(evidence["detected_colour"], "red")

    def test_wrong_class_is_rejected_even_if_red(self):
        image = Image.new("RGB", (100, 100), (230, 20, 20))
        detection = {"class_name": "cup", "confidence": 0.9, "bbox_1000": [0, 0, 1000, 1000]}
        self.assertFalse(detection_matches(detection, {"class_name": "bottle", "color": "red"}, image)["matched"])

    def test_filter_orders_matching_candidates(self):
        detections = [
            {"camera_id": "left", "class_name": "bottle", "confidence": 0.7, "attributes": {"color": "red"}},
            {"camera_id": "right", "class_name": "bottle", "confidence": 0.9, "attributes": {"color": "blue"}},
        ]
        result = filter_detections(detections, {"class_name": "bottle", "color": "red"})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["camera_id"], "left")

    def test_qwen_fallback_rejects_non_qwen_or_low_confidence(self):
        image = Image.new("RGB", (100, 100), (230, 20, 20))
        target = {"class_name": "block", "color": "red", "detection_mode": "qwen_fallback"}
        good = {
            "source": "qwen", "class_name": "block", "confidence": 0.8,
            "bbox_1000": [100, 100, 900, 900],
        }
        self.assertTrue(detection_matches(good, target, image)["matched"])
        self.assertFalse(detection_matches({**good, "source": "yolo"}, target, image)["matched"])
        self.assertFalse(detection_matches({**good, "confidence": 0.4}, target, image)["matched"])


if __name__ == "__main__":
    unittest.main()
