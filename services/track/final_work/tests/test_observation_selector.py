from __future__ import annotations

import unittest

from final_work.pi.ai_observation_selector import _rebuild_in_ranked_order


class _Map:
    @staticmethod
    def path(start, goal):
        return [goal]


class ObservationSelectorTests(unittest.TestCase):
    def test_ai_can_only_reorder_existing_safe_candidates(self) -> None:
        route = {
            "views": [
                {"view_xy_m": [1.0, 0.0], "path_xy_m": [], "path_length_m": 0},
                {"view_xy_m": [0.0, 2.0], "path_xy_m": [], "path_length_m": 0},
                {"view_xy_m": [-1.0, 0.0], "path_xy_m": [], "path_length_m": 0},
            ]
        }
        value = _rebuild_in_ranked_order(_Map(), (0.0, 0.0), route, [2, 99, 2, 0])
        self.assertEqual(value["ai_ranked_original_indices"], [2, 0, 1])
        self.assertEqual([item["view_xy_m"] for item in value["views"]], [
            [-1.0, 0.0], [1.0, 0.0], [0.0, 2.0],
        ])
        self.assertEqual(value["views"][0]["path_length_m"], 1.0)
        self.assertEqual(value["views"][1]["path_length_m"], 2.0)


if __name__ == "__main__":
    unittest.main()
