import math
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from relationship_localization_guard import (  # noqa: E402
    filter_localization_detections,
    guarded_landmark_labels,
    parse_start_pose_prior,
)


class RelationshipLocalizationGuardTests(unittest.TestCase):
    def test_start_pose_prior(self):
        prior = parse_start_pose_prior("-1.45,-2.90,110,0.75,35")
        self.assertAlmostEqual(-1.45, prior.x_m)
        self.assertAlmostEqual(-2.90, prior.y_m)
        self.assertEqual(110.0, prior.yaw_deg)
        self.assertAlmostEqual(0.75, prior.radius_m)
        self.assertAlmostEqual(35.0, prior.yaw_tolerance_deg)
        self.assertIsNone(parse_start_pose_prior(""))

    def test_rejects_observed_full_width_false_cabinet(self):
        detections = [{"label": "cabinet", "bbox_1000": [0, 0, 1000, 100]}]
        self.assertEqual([], filter_localization_detections(detections))

    def test_lone_generic_furniture_is_not_localization_evidence(self):
        def labels(rows):
            return {row["label"] for row in rows}

        result = guarded_landmark_labels(
            labels,
            [{"label": "cabinet", "bbox_1000": [600, 120, 900, 700]}],
        )
        self.assertEqual(set(), result)

    def test_distinctive_structure_survives(self):
        def labels(rows):
            return {row["label"] for row in rows}

        result = guarded_landmark_labels(
            labels,
            [{"label": "glass", "bbox_1000": [180, 100, 780, 720]}],
        )
        self.assertEqual({"glass"}, result)


if __name__ == "__main__":
    unittest.main()
