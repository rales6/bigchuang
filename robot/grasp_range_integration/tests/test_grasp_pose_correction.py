"""Pure pose arithmetic; no BLE or arm movement."""
import unittest

from grasp_range_integration.track_web_integration.pi.grasp_pose_correction import corrected_grasp_pose


class PoseCorrectionTests(unittest.TestCase):
    servos = [{"key": "s0", "min": 500, "max": 2500},
              {"key": "s2", "min": 1500, "max": 2200}]
    target = {"s0": 1410, "s1": 969, "s2": 1720}

    def test_removed_right_correction_and_reversed_lower_once_from_model(self):
        env = {"ARM_GRASP_S0_NUDGE_US": "-30", "ARM_GRASP_S2_NUDGE_US": "20"}
        result = corrected_grasp_pose(self.target, self.servos, env)
        self.assertEqual(result, {"s0": 1410, "s1": 969, "s2": 1740})
        self.assertEqual(self.target["s0"], 1410)
        self.assertEqual(corrected_grasp_pose(self.target, self.servos, env), result)

    def test_both_disabled_and_defaults_unchanged(self):
        self.assertIsNone(corrected_grasp_pose(self.target, self.servos,
            {"ARM_GRASP_S0_NUDGE_US": "0", "ARM_GRASP_S2_NUDGE_US": "0"}))
        self.assertEqual(corrected_grasp_pose(self.target, self.servos, {}),
            {**self.target, "s2": 1740})

    def test_invalid_offsets_rejected_not_silently_clamped(self):
        for env in ({"ARM_GRASP_S2_NUDGE_US": "26"}, {"ARM_GRASP_S2_NUDGE_US": "bad"},
                    {"ARM_GRASP_S2_NUDGE_US": "-26"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                corrected_grasp_pose(self.target, self.servos, env)

    def test_servo_limits_checked_before_any_motion(self):
        self.assertEqual(corrected_grasp_pose({**self.target, "s0": 510}, self.servos,
                                 {"ARM_GRASP_S0_NUDGE_US": "-30"})["s0"], 510)
        with self.assertRaises(ValueError):
            corrected_grasp_pose({**self.target, "s2": 2190}, self.servos,
                                 {"ARM_GRASP_S2_NUDGE_US": "20"})


if __name__ == "__main__":
    unittest.main()
