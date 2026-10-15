import tempfile
from pathlib import Path
import unittest

from car_sim.semantic_inspection_stress import StressProfile, run, simulate_seed


class SemanticInspectionStressTests(unittest.TestCase):
    def test_real_map_seed_exercises_disturbances_and_finishes_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            result = simulate_seed(
                0,
                StressProfile.named("realistic"),
                artifact_directory=Path(directory) / "artifacts",
            )
        self.assertTrue(result["passed"])
        self.assertEqual(result["target_count"], result["capture_count"])
        self.assertEqual(1.0, result["semantic_coverage_ratio"])
        self.assertGreater(result["actual_map_changed_cells"], 0)
        self.assertEqual(4, len(set(result["wheel_model"]["gains"].values())))
        self.assertGreater(result["wheel_model"]["slip_events"], 0)
        self.assertGreater(result["lidar"]["glass_leaked_rays"], 0)
        self.assertGreater(result["lidar"]["dropouts"], 0)
        self.assertGreater(result["lidar"]["short_speckles"], 0)
        self.assertGreater(result["lidar"]["multipath_returns"], 0)
        self.assertEqual(3, len(result["visual_localization"]["top_k"]))
        self.assertEqual(2, result["visual_localization"]["frames"])
        self.assertTrue(result["visual_localization"]["top_k_contains_closest"])
        self.assertLessEqual(result["localization"]["position_p95_m"], 0.13)
        self.assertEqual(0, result["mission"]["collision_contacts"])
        self.assertEqual(0, result["mission"]["target_timeouts"])
        self.assertLessEqual(result["mission"]["maximum_stationary_command_streak"], 8)
        self.assertLessEqual(result["mission"]["maximum_cross_track_error_m"], 0.40)
        self.assertGreaterEqual(
            result["wall_camera_coverage"]["wall_surface_coverage_ratio"], 0.90
        )
        self.assertGreaterEqual(
            result["wall_camera_coverage"]["visual_only_feature_coverage_ratio"],
            0.95,
        )
        # Current planner can already cover the wall without extra stops.
        # Test coverage and accounting, not a historical nonzero stop count.
        plan = result["wall_coverage_plan"]
        self.assertEqual(plan["supplement_count"], len(plan["supplements"]))
        self.assertGreaterEqual(
            plan["nominal_coverage_after_supplement"]["wall_surface_coverage_ratio"], 0.90)

    def test_batch_gate_requires_all_requirements_and_all_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(
                seed_count=2,
                profile_name="mild",
                output=Path(directory) / "result.json",
            )
        self.assertTrue(result["passed"])
        self.assertEqual(2, result["passed_seed_count"])
        self.assertTrue(all(result["all_requirements_exercised"].values()))


if __name__ == "__main__":
    unittest.main()
