from __future__ import annotations

import unittest

import cv2
import numpy as np

from final_work.gpu.qwen_hub import _stereo_order_geometry


class CameraOrderGeometryTests(unittest.TestCase):
    @staticmethod
    def _pair() -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        left = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(180):
            x = int(rng.integers(30, 610))
            y = int(rng.integers(30, 450))
            colour = tuple(int(v) for v in rng.integers(70, 255, size=3))
            cv2.circle(left, (x, y), int(rng.integers(2, 7)), colour, -1)
        # Synthetic positive disparity matching the operator-confirmed
        # background convention of this inward-toed physical rig.
        matrix = np.float32([[1, 0, -12], [0, 1, 0]])
        right = cv2.warpAffine(left, matrix, (640, 480))
        return left, right

    def test_positive_disparity_means_swapped_for_inward_toed_rig(self) -> None:
        left, right = self._pair()
        result = _stereo_order_geometry(left, right)
        self.assertEqual(result["order"], "swapped")
        self.assertGreater(result["matches"], 18)
        self.assertEqual(result["rig"], "inward_toed_operator_validated")

    def test_negative_disparity_means_normal_for_inward_toed_rig(self) -> None:
        left, right = self._pair()
        result = _stereo_order_geometry(right, left)
        self.assertEqual(result["order"], "normal")


if __name__ == "__main__":
    unittest.main()
