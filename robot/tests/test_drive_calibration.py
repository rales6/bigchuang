import os
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "car"))

from drivers.drive_calibration import DriveCalibration  # noqa: E402


class DriveCalibrationTest(unittest.TestCase):
    def test_speed_dependent_trim(self):
        calibration = DriveCalibration(0.01, 0.0001)
        self.assertAlmostEqual(0.05, calibration.trim_for_speed(400))

    def test_rejects_unsafe_curve(self):
        with self.assertRaises(ValueError):
            DriveCalibration(0.14, 0.0001)

    def test_persists_and_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "drive_calibration.json")
            calibration = DriveCalibration(path=path)
            calibration.set(0.02, 0.0001)
            loaded = DriveCalibration.load(path)
        self.assertAlmostEqual(0.02, loaded.trim_intercept)
        self.assertAlmostEqual(0.0001, loaded.trim_slope_per_mm_s)


if __name__ == "__main__":
    unittest.main()
