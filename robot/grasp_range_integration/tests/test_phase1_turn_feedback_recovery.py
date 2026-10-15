"""Independent relative evidence and correct scan-time handoff after turns."""
import math
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np

from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult
from grasp_range_integration.phase1 import turn_feedback_recovery as recovery


class AmbiguousTurnTests(unittest.TestCase):
    def setUp(self):
        self.now=0.;self.stops=[];self.events=[]
        self.before=NS(sequence=1)
        self.after=NS(sequence=2)
        self.fresh=NS(sequence=3,acquisition_start_s=.25,received_s=.3)
        self.first=MatchResult(Pose2D(.010,.043,math.radians(-22.28)),.03485,100,True,.573)
        self.strong=MatchResult(Pose2D(.012,.042,math.radians(-22.1)),.035,120,True,.82)
        self.still=MatchResult(Pose2D(.003,0,math.radians(.1)),.02,120,True,.91)
        self.window=dict(valid=True)
        self.anchor=(self.before,Pose2D(-1.66,-1.67,1.50))
        self.map_fit=NS(pose=self.anchor[1].compose(self.strong.transform),
            reliable=True,score=1.,rmse_m=.045,inlier_ratio=.81,
            range_agreement=.84,range_coverage=.72)

    def run_recovery(self, *, from_start=None, stopped=None, map_results=None,
                     pivot=None, anchor=True, raw=True, legacy_pivot=False):
        from_start=self.strong if from_start is None else from_start
        stopped=self.still if stopped is None else stopped
        scans=[]
        def stop():
            self.stops.append(self.now)
        def read(sequence,timeout):
            scans.append(sequence)
            self.now=.31
            return self.fresh
        def relative(a,b):
            return stopped if a is self.after else from_start
        results=iter(map_results or [self.map_fit,self.map_fit])
        localizer=NS(lidar_offset=(.14,0),max_distance_m=8.,
            localize_bounded=lambda *a,**kw:next(results))
        if pivot is not None:
            localizer.relative_turn_motion=(lambda a,b:pivot) if legacy_pivot else (lambda *a,**kw:pivot)
        with patch.object(recovery,'turn_range_evidence',return_value=dict(accepted=raw)):
            outcome=recovery.recover_ambiguous_turn(self.before,self.after,self.first,
                self.window,localizer,relative=relative,read_scan=read,stop=stop,
                anchor=self.anchor if anchor else None,
                emit=lambda name,**fields:self.events.append((name,fields)),clock=lambda:self.now)
        return outcome,scans

    def test_224311_weak_first_fit_can_be_verified_by_stopped_second_scan(self):
        recovered,scans=self.run_recovery()
        expected=self.strong.transform.compose(self.still.transform.inverse())
        self.assertAlmostEqual(recovered.transform.x_m,expected.x_m)
        self.assertAlmostEqual(recovered.transform.y_m,expected.y_m)
        self.assertAlmostEqual(recovered.transform.yaw_rad,expected.yaw_rad)
        self.assertEqual(scans,[2])
        self.assertEqual(len(self.stops),1)
        self.assertEqual(self.events[-1][1]['status'],'second_scan_raw_verified')

    def test_two_agreeing_map_frames_can_rescue_weak_relative_pair(self):
        weak=MatchResult(self.strong.transform,.04,100,True,.56)
        recovered,_=self.run_recovery(from_start=weak)
        self.assertIsNotNone(recovered)
        self.assertGreaterEqual(recovered.inlier_ratio,.65)
        self.assertAlmostEqual(recovered.transform.yaw_rad,self.strong.transform.yaw_rad)
        self.assertEqual(self.events[-1][1]['status'],'two_map_frames_verified')

    def test_deployed_two_argument_pivot_api_is_not_disabled_by_keyword_typeerror(self):
        recovered, scans = self.run_recovery(pivot=self.strong, legacy_pivot=True)
        self.assertIs(recovered, self.strong)
        self.assertEqual(scans, [])
        self.assertEqual(self.events[-1][1]['status'], 'pivot_raw_verified')

    def test_moving_after_stop_is_never_called_stationary_or_accepted(self):
        slip=MatchResult(Pose2D(.08,0,0),.02,100,True,.90)
        recovered,_=self.run_recovery(stopped=slip)
        self.assertIsNone(recovered)
        self.assertEqual(self.events[-1][1]['status'],'post_stop_motion_unverified')

    def test_no_map_anchor_cannot_promote_two_weak_scan_fits(self):
        weak=MatchResult(self.strong.transform,.04,100,True,.56)
        recovered,_=self.run_recovery(from_start=weak,anchor=False)
        self.assertIsNone(recovered)

    def test_stale_feedback_cannot_authorize_recheck(self):
        self.window['valid']=False
        recovered,scans=self.run_recovery()
        self.assertIsNone(recovered)
        self.assertEqual(scans,[])
        self.assertEqual(self.stops,[])

    def test_pivot_model_must_agree_with_original_and_raw_ranges(self):
        recovered,scans=self.run_recovery(pivot=self.strong)
        self.assertIs(recovered,self.strong)
        self.assertEqual(scans,[])
        self.assertEqual(self.events[-1][1]['status'],'pivot_raw_verified')
        displaced=MatchResult(Pose2D(.3,0,math.radians(-10)),.02,100,True,.8)
        recovered,_=self.run_recovery(pivot=displaced,raw=False)
        self.assertIsNone(recovered)

    def test_two_map_frames_must_agree_and_be_in_the_old_motion_basin(self):
        weak=MatchResult(self.strong.transform,.04,100,True,.56)
        far=NS(**vars(self.map_fit));far.pose=Pose2D(2,2,0)
        recovered,_=self.run_recovery(from_start=weak,map_results=[self.map_fit,far])
        self.assertIsNone(recovered)
        self.assertEqual(self.events[-1][1]['status'],'second_map_recheck_unverified')


class LargeTurnReanchorTests(unittest.TestCase):
    def setUp(self):
        self.now=1.2; self.stops=[]; self.events=[]
        self.before=NS(sequence=1)
        self.after=NS(sequence=2)
        self.fresh=NS(sequence=3,acquisition_start_s=1.30,received_s=1.32)
        self.first=MatchResult(Pose2D(.1843,-.0089,math.radians(-29.7246)),
                               .0259,160,True,.796)
        self.still=MatchResult(Pose2D(.002,0,math.radians(.1)),.02,150,True,.88)
        self.window=dict(valid=True,sent_s=1.)
        self.intent=dict(angular_mrad_s=-3500,ttl_ms=180,
                         output_multiplier=1.3043,recorded_s=1.1)
        self.anchor=(self.before,Pose2D(-1.5865,-2.9083,1.8318))
        self.map_fit=NS(pose=self.anchor[1].compose(self.first.transform),
            reliable=True,score=1.,rmse_m=.035,inlier_ratio=.83,
            range_agreement=.84,range_coverage=.70)

    def recover(self, *, anchor=True, intent=None, maps=None, raw=True, still=None):
        results=iter([self.map_fit] if maps is None else maps)
        def bounded(*args,**kwargs):
            return next(results)
        def read(sequence,timeout):
            self.now=1.33
            return self.fresh
        def relative(a,b):
            if a is self.after:
                return self.still
            if a is self.fresh and b is self.before:
                return MatchResult(self.first.transform.inverse(),.035,140,True,.82)
            return still or self.first
        localizer=NS(lidar_offset=(.14,0),max_distance_m=8.,
                     localize_bounded=bounded,
                     map=NS(world_to_grid=lambda _points:[(0,0)],
                            in_bounds=lambda _x,_y:True,
                            traversable=np.ones((1,1),dtype=bool)))
        raw_values=iter(raw) if isinstance(raw,(tuple,list)) else None
        def evidence(*args,**kwargs):
            return dict(accepted=next(raw_values) if raw_values is not None else raw)
        with patch.object(recovery,'turn_range_evidence',side_effect=evidence):
            return recovery.recover_large_turn(self.before,self.after,self.first,
                self.window,localizer,intent=self.intent if intent is None else intent,
                relative=relative,read_scan=read,
                stop=lambda:self.stops.append(self.now),
                anchor=self.anchor if anchor else None,
                emit=lambda name,**fields:self.events.append((name,fields)),
                clock=lambda:self.now)

    def test_recorded_174728_turn_is_map_reanchored_without_abort(self):
        result=self.recover()
        self.assertIsNotNone(result)
        self.assertAlmostEqual(math.degrees(result.transform.yaw_rad),-29.7246,places=3)
        self.assertAlmostEqual(math.hypot(result.transform.x_m,result.transform.y_m),.1845,places=3)
        self.assertEqual(self.stops,[])
        self.assertEqual(self.events[-1][1]['status'],'large_turn_map_reanchored')

    def test_second_fresh_scan_can_reanchor_when_first_map_fit_fails(self):
        result=self.recover(maps=[None,self.map_fit],raw=[False,True,True,True])
        self.assertIsNotNone(result)
        self.assertEqual(self.stops,[1.2])
        self.assertEqual(self.events[-1][1]['status'],'large_turn_stopped_map_reanchored')

    def test_map_timeout_uses_strong_bounded_scan_odometry_then_normal_tracking(self):
        result=self.recover(maps=[None],raw=True)
        self.assertIs(result,self.first)
        self.assertEqual(self.stops,[])
        self.assertEqual(self.events[-1][1]['status'],
                         'large_turn_bounded_scan_odometry')

    def test_183124_two_map_timeouts_use_reciprocal_fresh_lidar(self):
        result=self.recover(maps=[None,None],raw=[False,True])
        self.assertIsNotNone(result)
        self.assertAlmostEqual(math.degrees(result.transform.yaw_rad),-29.7246,places=3)
        self.assertEqual(self.stops,[1.2])
        self.assertEqual(self.events[-1][1]['status'],
                         'large_turn_reciprocal_odometry')

    def test_no_anchor_cannot_approve_large_turn(self):
        self.assertIsNone(self.recover(anchor=False))
        self.assertEqual(self.stops,[])

    def test_wrong_direction_or_unpaired_pulse_is_not_approved(self):
        self.assertIsNone(self.recover(intent=dict(self.intent,angular_mrad_s=3500)))
        self.assertEqual(self.events[-1][1]['status'],
                         'large_turn_outside_acknowledged_pulse')
        self.assertIsNone(self.recover(intent=dict(self.intent,recorded_s=3.)))
        self.assertEqual(self.events[-1][1]['status'],'large_turn_pulse_unpaired')

    def test_missing_raw_agreement_stays_stopped(self):
        self.assertIsNone(self.recover(maps=[self.map_fit,self.map_fit],raw=False))
        self.assertEqual(self.stops,[1.2])

    def test_motion_after_stop_is_not_called_stationary(self):
        moved=MatchResult(Pose2D(.08,0,0),.02,150,True,.9)
        self.still=moved
        self.assertIsNone(self.recover(maps=[None,self.map_fit],raw=False))
        self.assertEqual(self.events[-1][1]['status'],
                         'large_turn_post_stop_motion_unverified')


if __name__=='__main__': unittest.main()
