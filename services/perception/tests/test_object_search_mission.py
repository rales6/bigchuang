from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
SRC = ROOT / "src"
for path in (PROJECT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from raspberry_pi.object_search_mission import confirm_detections


class MissionDetectionTests(unittest.TestCase):
    def test_red_bottle_is_confirmed_but_blue_bottle_is_not(self):
        path = ROOT / "tests" / "_temporary_detection.jpg"
        right_path = ROOT / "tests" / "_temporary_detection_right.jpg"
        image = Image.new("RGB", (200, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 80, 80), fill=(230, 20, 20))
        draw.rectangle((120, 20, 180, 80), fill=(20, 20, 230))
        image.save(path)
        image.save(right_path)
        try:
            detections = [
                {"camera_id": "left", "class_name": "bottle", "confidence": 0.8, "bbox_1000": [100, 200, 400, 800]},
                {"camera_id": "right", "class_name": "bottle", "confidence": 0.8, "bbox_1000": [100, 200, 400, 800]},
                {"camera_id": "left", "class_name": "bottle", "confidence": 0.8, "bbox_1000": [600, 200, 900, 800]},
            ]
            result = confirm_detections(
                detections,
                {"class_name": "bottle", "color": "red"},
                {"left": path, "right": right_path},
            )
            self.assertTrue(result[0]["confirmed_match"])
            self.assertTrue(result[1]["confirmed_match"])
            self.assertFalse(result[2]["confirmed_match"])
        finally:
            path.unlink(missing_ok=True)
            right_path.unlink(missing_ok=True)

    def test_qwen_fallback_requires_qwen_source_confidence_bbox_and_colour(self):
        path = ROOT / "tests" / "_temporary_qwen_detection.jpg"
        right_path = ROOT / "tests" / "_temporary_qwen_detection_right.jpg"
        image = Image.new("RGB", (100, 100), "white")
        ImageDraw.Draw(image).rectangle((20, 20, 80, 80), fill=(230, 20, 20))
        image.save(path)
        image.save(right_path)
        target = {"class_name": "block", "color": "red", "detection_mode": "qwen_fallback"}
        try:
            good = {
                "camera_id": "left", "class_name": "block", "source": "qwen",
                "confidence": 0.78, "bbox_1000": [200, 200, 800, 800],
            }
            result = confirm_detections(
                [
                    good,
                    {**good, "camera_id": "right"},
                    {**good, "source": "yolo"},
                    {**good, "confidence": 0.45},
                ],
                target,
                {"left": path, "right": right_path},
            )
            self.assertTrue(result[0]["confirmed_match"])
            self.assertTrue(result[1]["confirmed_match"])
            self.assertFalse(result[2]["confirmed_match"])
            self.assertFalse(result[3]["confirmed_match"])
        finally:
            path.unlink(missing_ok=True)
            right_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
