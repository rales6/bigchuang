"""Recorded 2026-09-28 right-turn/left-edge loss, no hardware commands."""
import unittest

from pursuit.pursuit_config import PursuitConfig, StereoConfig
from pursuit.pursuit_controller import PursuitController, Perception
from raspberry_pi.lidar.base import make_scan


class TurnRollbackTests(unittest.TestCase):
    def setUp(self):
        self.c = PursuitController(PursuitConfig(), StereoConfig(), 1280,
                                   frame_height=720)
        self.c.on_ground_result(True, 100.0)
        # The previous right arc began with the target just right of the
        # calibrated body line. A subsequent pair reached the left edge.
        self.c._turn_rollback_baseline = (100.0, 710.0)
        self.c._recent_motion = [(100.0, 405, -112)]

    def scan(self, t):
        return make_scan([(0.0, 2.0), (-0.3, 2.0), (0.3, 2.0)], t)

    def left_edge(self, t, frame):
        return Perception(True, (3.4, 84.4, 43.6, 125.2), None,
            tracking2=True, bbox2=(1612.9, 138.7, 1651.9, 171.6),
            received_at=t, frame_id=frame)

    def loss(self, t, frame):
        return Perception(False, None, None, received_at=t, frame_id=frame)

    def test_recorded_right_arc_stops_then_reverse_turns_before_qwen(self):
        first = self.c.tick(self.left_edge(100.5, 10),
            self.scan(100.5), 100.5)
        self.assertEqual((first.motion.linear_mm_s,
                          first.motion.angular_mrad_s), (0, 0))
        self.assertFalse(first.reground_now)
        self.assertTrue(self.c._turn_rollback_active)
        second = self.c.tick(self.loss(100.8, 11),
            self.scan(100.8), 100.8)
        self.assertLess(second.motion.linear_mm_s, 0)
        self.assertEqual(second.motion.angular_mrad_s,
                         self.c.cfg.recovery_swept_path_turn_mrad_s)
        self.assertIn('turn-rollback', second.motion.reason)
        self.assertFalse(self.c._stationary_recheck_pending)

    def test_requires_fresh_bilateral_sweep_and_matching_forward_arc(self):
        self.c._recent_motion = [(100.0, 405, 112)]
        self.c._record_dual_history(self.left_edge(100.5, 10), 100.5)
        self.assertFalse(self.c._turn_rollback_evidence(100.5))
        self.c._recent_motion = [(100.0, 405, -112)]
        self.assertFalse(self.c._turn_rollback_evidence(103.0))
        self.c._recent_dual_centers = [(100.5, 600.0)]
        self.assertFalse(self.c._turn_rollback_evidence(100.5))

    def test_no_fresh_front_scan_means_no_reverse(self):
        self.c._tick_tracking(self.left_edge(100.5, 10),
            self.scan(100.5), 2.0, .1, 100.5)
        decision = self.c._tick_tracking(self.loss(100.8, 11),
            None, 2.0, .1, 100.8)
        self.assertEqual(decision.motion.linear_mm_s, 0)

    def test_budget_stays_bounded_and_recovery_requires_recentred_pair(self):
        self.c._tick_tracking(self.left_edge(100.5, 10),
            self.scan(100.5), 2.0, .1, 100.5)
        self.c._tick_tracking(self.loss(100.8, 11),
            self.scan(100.8), 2.0, .1, 100.8)
        self.assertLessEqual(self.c._recovery_swept_estimated_m, .10)
        self.c._recent_dual_centers.append((101.3, 630.0))
        self.assertTrue(self.c._turn_rollback_recovered())
        self.c._recent_dual_centers[-1] = (101.3, 300.0)
        self.assertFalse(self.c._turn_rollback_recovered())

    def test_two_short_inverse_arcs_then_stop_without_visual_reacquisition(self):
        self.c.tick(self.left_edge(100.5, 10), self.scan(100.5), 100.5)
        first = self.c.tick(self.loss(100.8, 11), self.scan(100.8), 100.8)
        self.assertLess(first.motion.linear_mm_s, 0)
        self.c.tick(self.loss(101.3, 12), self.scan(101.3), 101.3)
        second = self.c.tick(self.loss(101.6, 13), self.scan(101.6), 101.6)
        self.assertLess(second.motion.linear_mm_s, 0)
        self.assertEqual(self.c._recovery_swept_pulses, 2)
        self.assertLessEqual(self.c._recovery_swept_estimated_m, .100001)
        stopped = self.c.tick(self.loss(102.5, 14), self.scan(102.5), 102.5)
        self.assertTrue(stopped.finished)
        self.assertEqual(stopped.motion.linear_mm_s, 0)


if __name__ == '__main__':
    unittest.main()
