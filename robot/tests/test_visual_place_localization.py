import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image

from raspberry_pi.visual_place_localization import (
    PermanentLandmark,
    ReferencePlace,
    detection_landmark_labels,
    fuse_visual_top_k,
    load_reference_places,
    rank_landmark_candidates,
    select_visual_match,
    visual_top_k_matches,
)
from raspberry_pi.mapping.geometry import Pose2D


class VisualReferenceTests(unittest.TestCase):
    def test_real_orb_database_top_k_and_debug_image(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("opencv-python-headless is not installed")
        with tempfile.TemporaryDirectory() as directory:
            from raspberry_pi.visual_place_localization import VisualPlaceRecognizer

            root = Path(directory)
            image_a = np.full((420, 620, 3), 235, dtype=np.uint8)
            for index in range(12):
                center = (45 + index * 46, 70 + (index % 4) * 78)
                cv2.circle(image_a, center, 15 + index % 5, (20, 40 + index * 9, 180), 3)
                cv2.putText(
                    image_a, f"A{index}", (center[0] - 18, center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
                )
            cv2.line(image_a, (20, 380), (590, 25), (0, 0, 0), 5)
            image_b = np.full((420, 620, 3), 45, dtype=np.uint8)
            for index in range(9):
                left = 30 + index * 62
                cv2.rectangle(
                    image_b, (left, 40), (left + 32, 370),
                    (40 + index * 18, 220, 60), 4,
                )
            cv2.putText(
                image_b, "DIFFERENT PLACE", (85, 220),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA,
            )
            matrix = cv2.getPerspectiveTransform(
                np.float32([[0, 0], [619, 0], [619, 419], [0, 419]]),
                np.float32([[12, 9], [604, 2], [616, 410], [3, 418]]),
            )
            query = cv2.warpPerspective(image_a, matrix, (620, 420))
            query = cv2.convertScaleAbs(query, alpha=0.92, beta=13)
            cv2.imwrite(str(root / "a.jpg"), image_a)
            cv2.imwrite(str(root / "b.jpg"), image_b)
            cv2.imwrite(str(root / "query.jpg"), query)
            (root / "inspection_manifest.json").write_text(json.dumps({
                "captures": [
                    {"target_id": "candidate_a", "image": "a.jpg", "x_m": 0, "y_m": 0, "yaw_deg": 0},
                    {"target_id": "candidate_b", "image": "b.jpg", "x_m": 2, "y_m": 1, "yaw_deg": 90},
                ]
            }), encoding="utf-8")
            builder = VisualPlaceRecognizer(root)
            database = builder.build_feature_database(root / "features")
            recognizer = VisualPlaceRecognizer(root, feature_database=database)
            ranked = recognizer.rank([root / "query.jpg"], top_k=2)
            debug = recognizer.save_debug_match(
                root / "query.jpg", root / "a.jpg", root / "debug.jpg"
            )
            payload = json.loads(database.read_text(encoding="utf-8"))
            self.assertEqual("candidate_a", ranked[0].target_id)
            self.assertTrue(ranked[0].reliable)
            self.assertGreaterEqual(ranked[0].inliers, 9)
            self.assertEqual(2, payload["image_count"])
            self.assertTrue(debug.is_file())

    def test_loads_latest_manifest_and_resolves_dual_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = root / "20260814_010203"
            batch.mkdir()
            (batch / "left.jpg").write_bytes(b"left")
            (batch / "right.jpg").write_bytes(b"right")
            (batch / "inspection_manifest.json").write_text(json.dumps({
                "captures": [{
                    "target_id": "wall_corner",
                    "image": "left.jpg",
                    "camera_2_image": "right.jpg",
                    "x_m": 1.2,
                    "y_m": -0.4,
                    "yaw_deg": 90,
                }]
            }), encoding="utf-8")
            places = load_reference_places(root)
        self.assertEqual(len(places), 1)
        self.assertEqual(places[0].target_id, "wall_corner")
        self.assertEqual(len(places[0].images), 2)
        self.assertAlmostEqual(places[0].pose.yaw_rad, math.pi / 2)

    def test_requires_score_margin_and_inliers(self):
        first = ReferencePlace("corner_a", Pose2D(0, 0, 0), (Path("a"),), Path("manifest"))
        second = ReferencePlace("corner_b", Pose2D(3, 2, 0), (Path("b"),), Path("manifest"))
        clear = select_visual_match(
            [(0.72, 24, 40, first), (0.45, 20, 35, second)],
            minimum_score=0.30,
            minimum_margin=0.05,
            minimum_inliers=9,
        )
        ambiguous = select_visual_match(
            [(0.72, 24, 40, first), (0.70, 22, 38, second)],
            minimum_score=0.30,
            minimum_margin=0.05,
            minimum_inliers=9,
        )
        self.assertTrue(clear.reliable)
        self.assertFalse(ambiguous.reliable)

    def test_top_k_keeps_candidates_and_history_fuses_frames(self):
        places = [
            ReferencePlace(
                f"candidate_{index}", Pose2D(index, 0, 0),
                (Path(f"{index}.jpg"),), Path("manifest"),
            )
            for index in range(1, 5)
        ]
        first = visual_top_k_matches(
            [
                (0.72, 22, 39, places[0]),
                (0.69, 20, 36, places[1]),
                (0.52, 15, 30, places[2]),
                (0.20, 5, 15, places[3]),
            ],
            top_k=3,
            minimum_score=0.30,
            minimum_margin=0.05,
            minimum_inliers=9,
        )
        self.assertEqual(
            ["candidate_1", "candidate_2", "candidate_3"],
            [item.target_id for item in first],
        )
        self.assertFalse(first[0].reliable)
        _rows, history = fuse_visual_top_k({}, first, current_weight=0.60)
        second = visual_top_k_matches(
            [
                (0.88, 35, 51, places[0]),
                (0.45, 14, 28, places[1]),
                (0.30, 9, 21, places[2]),
            ],
            top_k=3,
            minimum_score=0.30,
            minimum_margin=0.05,
            minimum_inliers=9,
        )
        fused, _history = fuse_visual_top_k(history, second, current_weight=0.60)
        self.assertEqual("candidate_1", fused[0]["candidate"])
        self.assertGreater(fused[0]["score"] - fused[1]["score"], 0.20)

    def test_builds_labelled_dual_reference_sheets(self):
        with tempfile.TemporaryDirectory() as directory:
            from raspberry_pi.visual_place_localization import VisualPlaceRecognizer

            root = Path(directory)
            batch = root / "batch"
            batch.mkdir()
            Image.new("RGB", (80, 60), "red").save(batch / "left.jpg")
            Image.new("RGB", (80, 60), "blue").save(batch / "right.jpg")
            (batch / "inspection_manifest.json").write_text(json.dumps({
                "captures": [{
                    "target_id": "corner_a", "image": "left.jpg",
                    "camera_2_image": "right.jpg", "x_m": 0, "y_m": 0, "yaw_deg": 0,
                }]
            }), encoding="utf-8")
            recognizer = VisualPlaceRecognizer(batch)
            left, right, references = recognizer.build_reference_sheets(root / "out")
            self.assertTrue(left.is_file())
            self.assertTrue(right.is_file())
            self.assertEqual(references[0]["index"], 1)

    def test_permanent_landmark_breaks_symmetric_radar_tie(self):
        candidates = [
            type("Candidate", (), {"pose": Pose2D(0, 0, 0)})(),
            type("Candidate", (), {"pose": Pose2D(4, 0, math.pi)})(),
        ]
        landmarks = [
            PermanentLandmark("cabinet_a", 1.5, 0.1, frozenset({"cabinet"})),
            PermanentLandmark("chair_b", 2.5, 0.0, frozenset({"chair"})),
        ]
        labels = detection_landmark_labels([{
            "class_name": "cabinet", "confidence": 0.91, "attributes": {},
        }])
        ranked = rank_landmark_candidates(
            candidates, landmarks, labels, maximum_distance_m=2.0
        )
        self.assertEqual(ranked[0].candidate_index, 0)
        self.assertEqual(ranked[0].matched_labels, ("cabinet",))
        self.assertGreater(ranked[0].score, ranked[1].score)

    def test_occluded_landmark_does_not_vote(self):
        class Map:
            @staticmethod
            def line_is_clear(start, end):
                return start[0] < 1.0

        candidates = [
            type("Candidate", (), {"pose": Pose2D(0, 0, 0)})(),
            type("Candidate", (), {"pose": Pose2D(2, 0, math.pi)})(),
        ]
        landmark = PermanentLandmark("cabinet", 1, 0, frozenset({"cabinet"}))
        ranked = rank_landmark_candidates(
            candidates,
            [landmark],
            frozenset({"cabinet"}),
            known_map=Map(),
        )
        self.assertEqual(ranked[0].candidate_index, 0)
        self.assertEqual(ranked[1].score, 0.0)


if __name__ == "__main__":
    unittest.main()
