from pathlib import Path
import tempfile
import unittest

from car_sim.simulate_post_mapping_semantic import run_simulation


class PostMappingSemanticTest(unittest.TestCase):
    def test_post_mapping_semantic_simulation(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_simulation(Path(directory) / "semantic_sim")

        self.assertIs(report["passed"], True)
        self.assertGreaterEqual(report["map"]["endpoint_recall"], 0.90)
        self.assertEqual(report["semantic"]["located"], report["semantic"]["expected"])
        self.assertLessEqual(report["semantic"]["maximum_error_m"], 0.30)


if __name__ == "__main__":
    unittest.main()
