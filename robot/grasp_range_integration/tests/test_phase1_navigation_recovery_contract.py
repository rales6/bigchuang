"""Regression: weak tracking must reach recovery, not a zero-command loop."""

from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from grasp_range_integration.phase1.relationship_find_only_entry import _install_midrun_visual_recheck


class NavigationRecoveryContractTests(unittest.TestCase):
    def setUp(self):
        self.pose = NS(x_m=-1.344, y_m=-.413, yaw_rad=1.324)
        self.weak = self.result(.07, .60)
        self.strong = self.result(.04, .80)
        self.events, self.vision, self.bounded = [], [], []
        self.client = NS(stop=lambda: None)
        self.controller = NS(stopped=lambda: None)

    def result(self, rmse, inlier, score=1., dx=0):
        return NS(pose=NS(x_m=self.pose.x_m+dx, y_m=self.pose.y_m,
                          yaw_rad=self.pose.yaw_rad), rmse_m=rmse, inlier_ratio=inlier, score=score)

    def install(self, *, tracked=None, nearby=None, visual=None, status='confirmed'):
        def bounded(*args, **kwargs):
            self.bounded.append(kwargs)
            return nearby or self.weak

        def check(*args):
            self.vision.append(True)
            return status, visual or self.strong, object(), {}

        mission = NS(_phase1_verified_glass_required=True,
                     _tracking_usable_for_photo_mission=lambda r, a: r is not None and r.rmse_m <= .16 and r.inlier_ratio >= .4,
                     _visual_localization_check=check)
        localizer = NS(track=lambda *a: tracked or self.weak,
                       localize_near_position=lambda *a, **k: nearby or self.weak,
                       localize_bounded=bounded)
        _install_midrun_visual_recheck(mission, localizer, object(), self.client, self.controller,
                                      object(), object(), object(), Path('unused'),
                                      lambda name, **data: self.events.append((name, data)))
        return mission, localizer

    def test_weak_but_legacy_usable_fit_reaches_ds_on_second_frame(self):
        mission, localizer = self.install()
        self.assertFalse(mission._tracking_usable_for_photo_mission(self.weak, object()))
        self.assertIsNone(localizer.track(object(), self.pose))
        self.assertIs(localizer.track(object(), self.pose), self.strong)
        self.assertEqual(len(self.vision), 1)
        self.assertTrue(self.events[-1][1]['accepted'])

    def test_core_near_retry_cannot_bypass_quality_or_bounded_pose_contract(self):
        mission, localizer = self.install(nearby=self.result(.04, .8, dx=.25))
        self.assertIsNone(localizer.localize_near_position(object(), self.pose, radius_m=.35))
        self.assertEqual(self.bounded[-1], {'radius_m': .20, 'budget_s': .35, 'yaw_radius_deg': 10})
        self.assertFalse(mission._tracking_usable_for_photo_mission(self.result(.04, .8, score=-1), object()))

    def test_bounded_lidar_recovers_without_vision(self):
        _, localizer = self.install(nearby=self.strong)
        self.assertIs(localizer.track(object(), self.pose), self.strong)
        self.assertEqual(self.vision, [])

    def test_ds_abstention_is_one_attempt_then_explicit_failure_not_weak_pose(self):
        _, localizer = self.install(status='uncertain')
        for _ in range(5):
            self.assertIsNone(localizer.track(object(), self.pose))
        self.assertEqual(len(self.vision), 1)
        self.assertFalse([data for name, data in self.events if name == 'midrun_visual_recheck'][-1]['accepted'])

    def test_visual_cannot_approve_a_weak_lidar_fit_or_far_jump(self):
        for result in (self.weak, self.result(.04, .8, dx=.25)):
            with self.subTest(result=result):
                _, localizer = self.install(visual=result)
                self.assertIsNone(localizer.track(object(), self.pose))
                self.assertIsNone(localizer.track(object(), self.pose))

    def test_strong_map_tracking_stays_on_fast_path(self):
        _, localizer = self.install(tracked=self.strong)
        self.assertIs(localizer.track(object(), self.pose), self.strong)
        self.assertEqual(self.bounded, [])
        self.assertEqual(self.vision, [])


if __name__ == '__main__':
    unittest.main()
