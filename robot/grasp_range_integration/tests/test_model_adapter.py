from __future__ import annotations

import unittest

from grasp_range_integration.adapters.arm_teach_model import (
    ArmTeachModelClient,
    GraspModelError,
)


class ParameterValidationTests(unittest.TestCase):
    def test_accepts_boundaries(self) -> None:
        self.assertEqual(
            ArmTeachModelClient.validate_parameters(
                {"s0": 500, "s1": 1700, "s2": 1500}
            ),
            {"s0": 500, "s1": 1700, "s2": 1500},
        )

    def test_rejects_out_of_range(self) -> None:
        with self.assertRaises(GraspModelError):
            ArmTeachModelClient.validate_parameters(
                {"s0": 2501, "s1": 1000, "s2": 1800}
            )

    def test_rejects_missing_axis(self) -> None:
        with self.assertRaises(GraspModelError):
            ArmTeachModelClient.validate_parameters({"s0": 1500, "s1": 1000})


if __name__ == "__main__":
    unittest.main()
