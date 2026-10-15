import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from car_sim.mapping_report import generate_mapping_report  # noqa: E402


class MappingReportTest(unittest.TestCase):
    def _map_paths(self, directory):
        prefix = Path(directory) / "experiment"
        paths = (
            prefix.with_suffix(".pgm"),
            prefix.with_suffix(".yaml"),
            prefix.with_name(prefix.name + "_trajectory.csv"),
            prefix.with_suffix(".png"),
        )
        for path in paths:
            path.write_bytes(b"test map artifact")
        return prefix, paths

    def _summary(self, passed=True):
        return {
            "passed": passed,
            "requested_scans": 2,
            "completed_scans": 2,
            "accepted_scans": 2 if passed else 0,
            "rejected_scans": 0 if passed else 2,
            "accepted_ratio": 1.0 if passed else 0.0,
            "integrated_scans": 2 if passed else 0,
            "observed_cells": 500 if passed else 10,
            "occupied_cells": 30 if passed else 0,
            "mean_rmse_m": 0.04 if passed else 0.30,
            "median_rmse_m": 0.04 if passed else 0.30,
            "p95_rmse_m": 0.05 if passed else 0.32,
            "max_rmse_m": 0.05 if passed else 0.32,
            "elapsed_s": 1.5,
            "scans_per_second": 1.333,
            "trajectory_samples": 2,
            "trajectory_length_m": 0.5,
            "final_pose": {"x_m": 0.5, "y_m": 0.0, "yaw_deg": 0.0},
        }

    def _thresholds(self):
        return {
            "accepted_ratio_min": 0.55,
            "integrated_scans_min": 1,
            "observed_cells_min": 400,
            "occupied_cells_min": 20,
            "mean_rmse_max_m": 0.16,
        }

    def _samples(self, rejected=False):
        return [
            {
                "scan_index": 1,
                "timestamp_s": 1.0,
                "accepted": 0 if rejected else 1,
                "map_integrated": 0 if rejected else 1,
                "pose_x_m": 0.0,
                "pose_y_m": 0.0,
                "pose_yaw_deg": 0.0,
                "rmse_m": "" if rejected else 0.03,
                "correspondences": 0 if rejected else 80,
                "scan_points": 100,
                "inlier_ratio": 0.0 if rejected else 0.8,
                "translation_m": 0.0,
                "rotation_deg": 0.0,
                "linear_speed_m_s": 0.1,
                "angular_speed_rad_s": 0.0,
                "rejection_reason": "匹配点不足" if rejected else "",
                "map_status": "速度过高" if rejected else "已融合",
            },
            {
                "scan_index": 2,
                "timestamp_s": 2.0,
                "accepted": 0 if rejected else 1,
                "map_integrated": 0 if rejected else 1,
                "pose_x_m": 0.5,
                "pose_y_m": 0.0,
                "pose_yaw_deg": 0.0,
                "rmse_m": 0.32 if rejected else 0.05,
                "correspondences": 5 if rejected else 75,
                "scan_points": 100,
                "inlier_ratio": 0.05 if rejected else 0.75,
                "translation_m": 0.5,
                "rotation_deg": 0.0,
                "linear_speed_m_s": 0.1,
                "angular_speed_rad_s": 0.0,
                "rejection_reason": "匹配点不足" if rejected else "",
                "map_status": "速度过高" if rejected else "已融合",
            },
        ]

    def test_generates_html_json_and_per_scan_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix, map_paths = self._map_paths(directory)
            report_paths = generate_mapping_report(
                prefix,
                parameters={"扫描帧数": 2, "速度": 350},
                summary=self._summary(),
                thresholds=self._thresholds(),
                map_paths=map_paths,
                scan_samples=self._samples(),
            )

            self.assertTrue(all(path.exists() for path in report_paths))
            html = report_paths[0].read_text(encoding="utf-8")
            self.assertIn("CAR·LAB 建图实验报告", html)
            self.assertIn("experiment.png", html)
            self.assertIn("逐帧匹配误差", html)

            payload = json.loads(report_paths[1].read_text(encoding="utf-8"))
            self.assertEqual("PASS", payload["result"])
            self.assertEqual(2, payload["summary"]["completed_scans"])
            self.assertTrue(payload["files"]["map_png"]["exists"])
            self.assertTrue(payload["files"]["report_html"]["exists"])
            self.assertIn("仿真基准", payload["analysis"][-1])

            with report_paths[2].open(
                newline="",
                encoding="utf-8-sig",
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(2, len(rows))
            self.assertEqual("0.05", rows[1]["rmse_m"])

    def test_failed_report_includes_diagnostic_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix, map_paths = self._map_paths(directory)
            report_paths = generate_mapping_report(
                prefix,
                parameters={"扫描帧数": 2},
                summary=self._summary(passed=False),
                thresholds=self._thresholds(),
                map_paths=map_paths,
                scan_samples=self._samples(rejected=True),
            )

            payload = json.loads(report_paths[1].read_text(encoding="utf-8"))
            self.assertEqual("FAIL", payload["result"])
            self.assertEqual(2, payload["rejection_reason_counts"]["匹配点不足"])
            self.assertEqual(2, payload["map_status_counts"]["速度过高"])
            self.assertTrue(
                any("扫描匹配接受率偏低" in item for item in payload["analysis"])
            )


if __name__ == "__main__":
    unittest.main()
