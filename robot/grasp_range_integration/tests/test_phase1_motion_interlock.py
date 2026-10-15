import math
from types import SimpleNamespace
import unittest

from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult
from grasp_range_integration.phase1.motion_interlock import validate_turn_feedback, predict_scan_motion
from grasp_range_integration.phase1.relationship_find_only_entry import (
    _install_midrun_visual_recheck, _scan_odometry_usable)


class MotionInterlockTests(unittest.TestCase):
    def match(self, yaw=0., xy=0., rmse=.02, ratio=.9, converged=True):
        return MatchResult(Pose2D(xy,0.,math.radians(yaw)),rmse,100,converged,ratio)

    def test_valid_stall_and_small_turn_are_accepted(self):
        for yaw in (0.,5.,-15.,24.):
            self.assertIsNone(validate_turn_feedback({'valid':True},self.match(yaw=yaw)))

    def test_large_uncertain_and_unpaired_turns_are_locked(self):
        for fit in (self.match(yaw=46.981),self.match(yaw=-25),
                    self.match(rmse=.1),self.match(ratio=.64),self.match(converged=False),
                    self.match(yaw=float('nan'))):
            self.assertIsNotNone(validate_turn_feedback({'valid':True},fit))
        self.assertIsNotNone(validate_turn_feedback({'valid':False},self.match()))

    def test_reliable_slip_has_no_fixed_translation_exit_or_clamp(self):
        pose = Pose2D(1., 2., math.pi/2)
        for distance in (.03, .23, .5, 1.):
            fit = self.match(yaw=-13.325, xy=distance)
            self.assertIsNone(validate_turn_feedback({'valid':True}, fit))
            predicted, usable = predict_scan_motion(pose, fit)
            self.assertTrue(usable)
            expected = pose.compose(fit.transform)
            self.assertAlmostEqual(predicted.x_m, expected.x_m)
            self.assertAlmostEqual(predicted.y_m, expected.y_m)
            self.assertAlmostEqual(predicted.yaw_rad, expected.yaw_rad)

    def test_nonfinite_translation_and_bad_quality_cannot_update_pose(self):
        for fit in (self.match(xy=float('nan')), self.match(xy=float('inf')),
                    self.match(xy=.5, rmse=.2), self.match(xy=.5, ratio=.3)):
            predicted, usable = predict_scan_motion(Pose2D(), fit)
            self.assertFalse(usable)
            self.assertEqual(predicted, Pose2D())

    def test_unverified_47_degree_motion_requests_recovery_without_terminal_fault(self):
        from raspberry_pi.object_search_mission import _scan_motion_prediction
        calls=[]
        client=SimpleNamespace(stop=lambda:calls.append('stop'))
        controller=SimpleNamespace(stopped=lambda:calls.append('controller_stopped'))
        mission=SimpleNamespace(_scan_motion_prediction=_scan_motion_prediction)
        localizer=SimpleNamespace(track=lambda *a:calls.append('track'))
        _install_midrun_visual_recheck(mission,localizer,object(),client,controller,
            None,None,None,None,None)
        predicted,usable=mission._scan_motion_prediction(
            Pose2D(0.,0.,0.),self.match(yaw=46.981,xy=.1423))
        self.assertFalse(usable)
        self.assertAlmostEqual(predicted.yaw_rad,math.radians(46.981))
        self.assertTrue(mission._phase1_turn_relocalization_pending)
        self.assertEqual(calls,['stop','controller_stopped'])
        self.assertIsNone(getattr(mission,'_phase1_motion_fault',None))

    def test_independently_verified_large_turn_uses_full_measured_transform_once(self):
        from raspberry_pi.object_search_mission import _scan_motion_prediction
        calls=[]
        client=SimpleNamespace(stop=lambda:calls.append('stop'))
        controller=SimpleNamespace(stopped=lambda:calls.append('controller_stopped'))
        mission=SimpleNamespace(_scan_motion_prediction=_scan_motion_prediction)
        localizer=SimpleNamespace(track=lambda *a:calls.append('track'))
        _install_midrun_visual_recheck(mission,localizer,object(),client,controller,
            None,None,None,None,None)
        fit=self.match(yaw=-29.7246,xy=.1845,rmse=.0259,ratio=.796)
        mission._phase1_verified_large_turn_match=fit
        start=Pose2D(-1.5865,-2.9083,1.8318)
        predicted,usable=mission._scan_motion_prediction(start,fit)
        self.assertTrue(usable)
        expected=start.compose(fit.transform)
        self.assertAlmostEqual(predicted.x_m,expected.x_m)
        self.assertAlmostEqual(predicted.y_m,expected.y_m)
        self.assertAlmostEqual(predicted.yaw_rad,expected.yaw_rad)
        self.assertIsNone(mission._phase1_verified_large_turn_match)
        self.assertEqual(calls,[])

    def test_unresolved_turn_requests_wide_stationary_relocalization(self):
        from raspberry_pi.object_search_mission import _scan_motion_prediction
        calls=[]
        client=SimpleNamespace(stop=lambda:calls.append('stop'))
        controller=SimpleNamespace(stopped=lambda:calls.append('controller_stopped'))
        mission=SimpleNamespace(_scan_motion_prediction=_scan_motion_prediction)
        localizer=SimpleNamespace(track=lambda *a:None)
        _install_midrun_visual_recheck(mission,localizer,object(),client,controller,
            None,None,None,None,None)
        fit=self.match(yaw=-26.932,xy=.0505,rmse=.0687,ratio=.867)
        unusable=MatchResult(fit.transform,fit.rmse_m,fit.correspondences,
                             False,fit.inlier_ratio)
        mission._phase1_turn_relocalization_match=unusable
        mission._phase1_turn_relocalization_hint=fit
        start=Pose2D(-1.5656,-1.2752,1.4984)
        predicted,usable=mission._scan_motion_prediction(start,unusable)
        self.assertFalse(usable)
        self.assertAlmostEqual(predicted.yaw_rad,start.compose(fit.transform).yaw_rad)
        self.assertTrue(mission._phase1_turn_relocalization_pending)
        self.assertTrue(localizer._wide_track_once)
        self.assertIsNone(getattr(mission,'_phase1_motion_fault',None))
        self.assertEqual(calls,['stop','controller_stopped'])

    def test_verified_large_turn_is_allowed_only_in_bounded_bridge(self):
        fit=self.match(yaw=-26.932,xy=.0505,rmse=.0687,ratio=.867)
        self.assertFalse(_scan_odometry_usable(fit))
        self.assertTrue(_scan_odometry_usable(fit,verified_large_turn=True))

    def test_turn_relocalization_clears_only_after_strong_map_track(self):
        from raspberry_pi.object_search_mission import _scan_motion_prediction
        target=Pose2D(-1.5,-1.3,1.0)
        good=SimpleNamespace(pose=target,rmse_m=.04,inlier_ratio=.8,score=1.)
        mission=SimpleNamespace(
            _scan_motion_prediction=_scan_motion_prediction,
            _phase1_turn_relocalization_pending=True,
            _tracking_usable_for_photo_mission=lambda candidate,_args:
                candidate is good)
        localizer=SimpleNamespace(track=lambda *_:good)
        client=SimpleNamespace(stop=lambda:None)
        controller=SimpleNamespace(stopped=lambda:None)
        _install_midrun_visual_recheck(mission,localizer,object(),client,controller,
            None,None,None,None,None)
        self.assertIs(localizer.track(object(),target),good)
        self.assertFalse(mission._phase1_turn_relocalization_pending)


if __name__=='__main__':unittest.main()
