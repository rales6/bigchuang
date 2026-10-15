import math
import sys
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raspberry_pi.motion_calibration import (  # noqa: E402
    CalibrationResult,
    build_argument_parser,
    build_trials,
    combine_calibration,
    fit_speed_dependent_trim,
    parse_positive_ints,
    save_results,
    straight_scale_suggestion,
    summarize_results,
    sweep_speeds,
    turn_gain_suggestion,
    validate_args,
)


def result(**overrides):
    values = dict(
        trial=1,
        mode="straight",
        command_linear_mm_s=300,
        command_angular_mrad_s=0,
        duration_s=1.0,
        x_m=0.30,
        y_m=0.0,
        distance_m=0.30,
        yaw_deg=0.0,
        accepted_scans=8,
        rejected_scans=0,
        mean_rmse_m=0.03,
        mean_inlier_ratio=0.8,
    )
    values.update(overrides)
    return CalibrationResult(**values)


class MotionCalibrationTest(unittest.TestCase):
    def test_parse_and_build_trials(self):
        self.assertEqual([250, 350], parse_positive_ints("250, 350", "speed"))
        trials = build_trials("all", [300], [2000], 2)
        self.assertEqual(
            [
                (1, "straight", 300, 0),
                (2, "straight", 300, 0),
                (1, "right", 0, -2000),
                (2, "right", 0, -2000),
            ],
            trials,
        )
        self.assertEqual([350, 375, 400], sweep_speeds(350, 25, 400))

    def test_validation_rejects_unsafe_values(self):
        args = build_argument_parser().parse_args([
            "--linear-speeds", "600",
            "--turn-speeds", "3600",
        ])
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_straight_scale_suggestion_reduces_faster_side(self):
        # 正偏航表示右轮累计路程更长，因此建议降低右侧倍率。
        scales = straight_scale_suggestion(
            result(yaw_deg=5.0),
            track_width_mm=185.0,
        )
        self.assertGreater(scales[0], 1.0)
        self.assertLess(scales[1], 1.0)

    def test_turn_gain_suggestion(self):
        measured = result(
            mode="right",
            command_linear_mm_s=0,
            command_angular_mrad_s=-2000,
            duration_s=0.5,
            yaw_deg=-40.0,
        )
        expected = math.degrees(1.0) / 40.0
        self.assertAlmostEqual(expected, turn_gain_suggestion(measured))

    def test_summary_uses_median(self):
        summaries = summarize_results(
            [
                result(yaw_deg=2.0),
                result(yaw_deg=4.0),
                result(yaw_deg=40.0),
            ],
            185.0,
        )
        self.assertEqual(1, len(summaries))
        self.assertEqual(4.0, summaries[0]["yaw_deg"])
        self.assertIn("left_scale", summaries[0])

    def test_save_results_writes_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            output = save_results(
                Path(directory) / "motion.csv",
                [result()],
            )
            text = output.read_text(encoding="utf-8")
        self.assertIn("command_linear_mm_s", text)
        self.assertIn("straight", text)

    def test_speed_dependent_trim_fit_and_damped_combine(self):
        fitted = fit_speed_dependent_trim(
            [
                result(command_linear_mm_s=350, yaw_deg=-4.0),
                result(command_linear_mm_s=400, yaw_deg=-6.0),
                result(command_linear_mm_s=450, yaw_deg=-8.0),
            ],
            185.0,
        )
        self.assertEqual(3, len(fitted["points"]))
        self.assertGreater(fitted["trim_slope_per_mm_s"], 0.0)
        combined = combine_calibration(
            {
                "trim_intercept": 0.01,
                "trim_slope_per_mm_s": 0.0,
            },
            fitted,
            0.5,
        )
        self.assertGreater(combined["trim_intercept"], -0.15)
        self.assertLess(
            abs(
                combined["trim_intercept"]
                + combined["trim_slope_per_mm_s"] * 550.0
            ),
            0.141,
        )


if __name__ == "__main__":
    unittest.main()
