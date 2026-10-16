"""Replay timing of 20260926_193306; no camera, lidar or motor connections."""

import math
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult
from raspberry_pi.known_map_navigation import KnownOccupancyMap
from grasp_range_integration.phase1.fresh_motion_feedback import FreshScan, install_fresh_motion_feedback
from grasp_range_integration.phase1.navigation_freshness import ensure_navigation_freshness, effective_scan
from grasp_range_integration.phase1.relationship_find_only_entry import _install_waypoint_obstacle_guard
from grasp_range_integration.phase1.glass_boundary_guard import attach_boundaries
from grasp_range_integration.phase1 import turn_policy


def scan(received=10., sequence=123, start=9.8, near=False):
    a=np.deg2rad(np.arange(-180,180))
    r=np.full(len(a),3.)
    if near:r[180]=.20
    return FreshScan(a,r,received,None,start,received,sequence)


class FreshnessPolicyTests(unittest.TestCase):
    def setUp(self):
        self.now=10.77
        self.original=scan()
        self.context={'scan':self.original,'reliable':True,'updated':10.77,
                      'quality':{},'pose':Pose2D(0.,0.,0.)}
        self.events=[]
        self.calls=[]

    def refresh(self, reference):
        self.calls.append(reference)
        self.now=11.05
        return scan(11.01,130,10.83),True

    def ready(self, refresh=None):
        return ensure_navigation_freshness(self.context,self.refresh if refresh is None else refresh,
            lambda kind,**kw:self.events.append((kind,kw)),clock=lambda:self.now)

    def test_point_77_second_scan_replaced_by_real_post_stop_scan_not_threshold_change(self):
        pose=self.context['pose']
        self.assertTrue(self.ready())
        self.assertEqual(self.calls,[self.original])
        self.assertIs(self.context['scan'],self.original)
        self.assertIs(self.context['pose'],pose)
        self.assertEqual(effective_scan(self.context).sequence,130)
        self.assertEqual(self.context['updated'],11.01)
        self.assertEqual(self.context['bridge_uncertainty_m'],.05)
        self.assertTrue(self.events[-1][1]['odometry_baseline_unchanged'])

    def test_fresh_context_does_not_stop_or_refit(self):
        self.now=10.1
        self.context['updated']=10.1
        self.assertTrue(self.ready())
        self.assertEqual(self.calls,[])

    def test_stationary_refresh_adds_margin_without_erasing_odometry_uncertainty(self):
        self.context['bridge_uncertainty_m'] = .24
        self.assertTrue(self.ready())
        self.assertAlmostEqual(self.context['bridge_uncertainty_m'], .29)

    def test_cached_new_scan_does_not_trigger_second_stop_in_same_cycle(self):
        self.assertTrue(self.ready())
        self.assertTrue(self.ready())
        self.assertEqual(self.calls,[self.original])

    def test_missing_service_is_zero_speed_retry_not_exception(self):
        self.assertFalse(ensure_navigation_freshness(self.context,None,lambda *a,**k:None,clock=lambda:self.now))

    def test_missing_scan_and_overage_reference_do_not_get_fresh_timestamps(self):
        for value in (None,scan(8.,100,7.8)):
            self.context['scan']=value
            self.assertFalse(self.ready())
        self.assertEqual(self.calls,[])
        self.assertEqual(self.context['updated'],10.77)

    def test_timeout_retries_without_exiting_mission(self):
        def timeout(_):raise TimeoutError('no new scan')
        self.assertFalse(self.ready(timeout))
        self.assertEqual(self.context['freshness_wait_reason'],'post_stop_scan_unavailable')

    def test_real_motion_invalidates_old_pose_without_composing_twice(self):
        pose=self.context['pose']
        self.assertFalse(self.ready(lambda _: (scan(10.75,130,10.55),False)))
        self.assertFalse(self.context['reliable'])
        self.assertEqual(self.context['updated'],10.77)
        self.assertIs(self.context['pose'],pose)
        self.assertIsNone(self.context.get('clearance_pair'))

    def test_weak_pose_is_not_promoted_by_fresh_scan_only(self):
        self.context['reliable']=False
        self.assertFalse(self.ready())
        self.assertEqual(self.calls,[])

    def test_same_old_or_pre_reference_scan_cannot_extend_pose_lease(self):
        for new in (self.original,scan(10.70,130,9.90),scan(10.70,130,10.4)):
            self.now=11.3
            self.assertFalse(self.ready(lambda _:(new,True)))
        self.assertEqual(self.context['updated'],10.77)

    def test_future_timestamp_is_rejected(self):
        self.assertFalse(self.ready(lambda _:(scan(11.1,130,10.8),True)))

    def test_refreshes_compare_against_original_absolute_fit_scan_not_incremental_bridge(self):
        self.assertTrue(self.ready())
        self.now=11.52
        self.assertFalse(self.ready())  # Original absolute-fit scan now too old.
        self.assertEqual(self.calls,[self.original])

    @unittest.skipUnless((Path(__file__).parent / "fixtures").is_dir(), "Requires private recording bundle")
    def test_real_193306_timing_trace_no_longer_raises_glass_scan_stale(self):
        rows=[json.loads(line) for line in (Path(__file__).parent/'fixtures/navigation_start_20260926_193306.jsonl').read_text(encoding='utf-8').splitlines()]
        delivered=[r for r in rows if r['type']=='fresh_scan_delivered'][-1]
        assessed=[r for r in rows if r['type']=='navigation_turn_space_assessed'][-1]
        self.assertEqual(delivered['sequence'],123)
        self.assertTrue(assessed['safe'])
        age=assessed['scan_age_s']
        self.assertGreater(age,.5)
        self.assertLess(age,1.5)
        self.context['scan']=scan(delivered['received_s'],123,delivered['acquisition_start_s'])
        self.now=delivered['received_s']+age
        self.context['updated']=delivered['received_s']
        def refreshed(reference):
            stopped=self.now
            self.now+=.28
            return scan(self.now-.04,130,stopped+.06),True
        # Real timing, synthetic beams/quiet-motion result: this is a software
        # scheduling regression, NOT a replay of unrecorded physical scan arrays.
        self.assertTrue(self.ready(refreshed))
        self.assertTrue(self.events[-1][1]['stationary_pose_supported'])


class NavigationAdapterTests(unittest.TestCase):
    def setUp(self):
        self.now=[10.77];self.reads=[];self.guard_scans=[]
        self.old=scan();self.new=scan(11.01,130,10.83)
        self.motion=Pose2D(0.,0.,0.)
        self.request=(240,0,False)
        owner=self
        class Client:
            def __init__(self):self.calls=[]
            def stop(self):self.calls.append(('stop',))
            def set_twist(self,*a):self.calls.append(('regular',a))
            def set_bounded_twist(self,*a,**kw):self.calls.append(('bounded',a,kw));return 'ACK'
            def allow_twist_to_expire(self):pass
        class Driver:
            def scans(self):return iter(())
        class Localizer:
            lidar_offset=np.array([.14,0.])
            max_distance_m=8.
            def track(self,current,pose):return SimpleNamespace(pose=pose,rmse_m=.04,inlier_ratio=.85)
            def relative_motion(self,before,after):
                return MatchResult(owner.motion,.02,100,True,.9)
        def original_guard(current,linear,distance,**kw):
            self.guard_scans.append(current)
            return not (linear>0 and current.distances_m[180]<distance)
        def command(*args):
            turn_policy.LAST_REQUEST=self.request[:2]
            return self.request
        self.mission=SimpleNamespace(N10LidarDriver=Driver,Esp32Client=Client,ScanMapLocalizer=Localizer,
            _strict_command_for_pose=command,_obstacle_safe=original_guard,
            build_search_route=lambda *a:{},_fuse_tracking_pose=lambda pose,result:pose,
            _phase1_verified_glass_required=True)
        self.fence=install_fresh_motion_feedback(self.mission)
        _install_waypoint_obstacle_guard(self.mission)
        self.client=self.mission.Esp32Client();self.localizer=self.mission.ScanMapLocalizer()
        self.stream=SimpleNamespace(delivered=123,latest_after=self.read)
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.FreshScanStream',return_value=self.stream):
            self.mission.N10LidarDriver().scans()
        self.localizer.relative_motion(self.old,self.old) # Register live localizer, no output.
        known=KnownOccupancyMap(np.full((120,120),255,dtype=np.uint8),.05,-3.,-3.)
        attach_boundaries(known,dict(segments_xy_m=[[[-2.7,-2.7],[-2.7,2.7]]],boundary_uncertainty_m=.20))
        self.mission.build_search_route(known,(0.,0.),{})
        self.localizer.track(self.old,Pose2D(0.,0.,0.))
        result=SimpleNamespace(pose=Pose2D(0.,0.,0.),rmse_m=.04,inlier_ratio=.85)
        with patch('time.monotonic',side_effect=lambda:self.now[0]):
            self.mission._fuse_tracking_pose(result.pose,result)

    def tearDown(self):
        turn_policy.LAST_REQUEST=(0,0)
        turn_policy.TELEMETRY_CALLBACK=None

    def read(self,sequence,timeout_s):
        self.reads.append(sequence)
        self.now[0]=11.05
        return self.new

    def test_actual_400mm_start_kick_is_authorized_after_new_scan_and_complete_recheck(self):
        with patch('time.monotonic',side_effect=lambda:self.now[0]):
            self.assertEqual(self.mission._strict_command_for_pose(),(240,0,False))
            self.assertTrue(self.mission._obstacle_safe(self.old,400,.45))
            self.assertEqual(self.client.set_twist(400,0,650),'ACK')
        self.assertEqual(self.reads,[123])
        self.assertEqual(self.client.calls[0],('stop',))
        self.assertEqual(self.client.calls[-1][0],'bounded')
        self.assertEqual(self.client.calls[-1][1],(400,0,650))
        self.assertTrue(all(s is self.new for s in self.guard_scans))
        self.assertEqual(self.stream.delivered,123) # Independent read, no odometry consumption.
        self.assertIsNone(self.fence.pending)

    def use_estimate(self, pose=Pose2D(0.,0.,0.), uncertainty=.24):
        self.now[0] = 10.1
        self.localizer.track(self.old, pose)
        result = SimpleNamespace(pose=pose, rmse_m=.02, inlier_ratio=.9, score=-1.)
        self.mission._phase1_navigation_estimate = dict(result=result, started_s=10.,
                                                       uncertainty_m=uncertainty)
        with patch('time.monotonic',side_effect=lambda:self.now[0]):
            self.mission._fuse_tracking_pose(pose,result)

    def test_estimate_can_move_without_stop_but_actual_speed_and_lease_are_limited(self):
        self.use_estimate()
        with patch('time.monotonic',side_effect=lambda:self.now[0]):
            self.assertEqual(self.client.set_twist(800,0,650),'ACK')
        self.assertEqual(self.reads,[])
        self.assertEqual(self.client.calls[-1][1],(400,0,250))
        self.assertTrue(all(c[0]!='stop' for c in self.client.calls))

    def test_estimate_uncertainty_blocks_sweep_that_precise_pose_would_clear(self):
        # 0.90m from glass: normal radius .44 + margin .20 clears a short
        # forward sweep; .24m odometry uncertainty must block that sweep.
        self.use_estimate(Pose2D(-1.8,0.,math.pi))
        with patch('time.monotonic',side_effect=lambda:self.now[0]):
            self.client.set_twist(400,0,250)
        self.assertTrue(all(c[0]=='stop' for c in self.client.calls))

    def test_real_front_obstacle_still_blocks_estimated_navigation(self):
        self.old=scan(10.,123,9.8,near=True)
        self.use_estimate()
        with patch('time.monotonic',side_effect=lambda:self.now[0]):
            self.client.set_twist(400,0,250)
        self.assertTrue(all(c[0]=='stop' for c in self.client.calls))

    def test_estimate_last_command_cannot_overrun_remaining_distance_budget(self):
        self.use_estimate()
        self.mission._phase1_navigation_estimate['distance_m'] = .96
        with patch('time.monotonic',side_effect=lambda:self.now[0]):
            self.client.set_twist(400,0,650)
        actual = self.client.calls[-1][1]
        self.assertLessEqual(actual[2],100)

    def test_estimate_too_little_budget_defers_instead_of_invalid_firmware_ttl(self):
        self.use_estimate()
        self.mission._phase1_navigation_estimate['distance_m'] = .99
        with patch('time.monotonic',side_effect=lambda:self.now[0]):
            self.client.set_twist(400,0,650)
        self.assertTrue(all(c[0]=='stop' for c in self.client.calls))
        self.assertTrue(self.mission._phase1_navigation_estimate['command_budget_exhausted'])

    def test_estimate_expired_cannot_issue_new_motion_command(self):
        self.use_estimate()
        self.now[0] = 40.1
        with patch('time.monotonic',side_effect=lambda:self.now[0]):
            self.client.set_twist(400,0,650)
        self.assertTrue(all(c[0]=='stop' for c in self.client.calls))

    def test_speed_controller_records_actual_estimated_navigation_cap(self):
        from dataclasses import dataclass
        @dataclass
        class Command:
            linear_mm_s: int
            angular_mrad_s: int = 0
        class Controller:
            def apply(self, command):self.last_command=command;return command
            def stopped(self):self.last_command=None
            def observe(self, update):pass
            def ttl_for(self, command, ttl):return ttl
        self.mission.AdaptiveSpeedController=Controller
        turn_policy.install_turn_policy(self.mission)
        self.use_estimate()
        controller=Controller()
        result=controller.apply(Command(800))
        self.assertEqual(result.linear_mm_s,400)
        self.assertIs(controller.last_command,result)

    def test_new_front_obstacle_blocks_command_despite_old_scan_being_clear(self):
        self.new=scan(11.01,130,10.83,near=True)
        with patch('time.monotonic',side_effect=lambda:self.now[0]):
            self.client.set_twist(400,0,650)
        self.assertTrue(all(c[0]=='stop' for c in self.client.calls))
        self.assertIs(self.guard_scans[-1],self.new)

    def test_vehicle_moved_during_compute_holds_zero_for_next_localization(self):
        self.motion=Pose2D(.06,0.,0.)
        with patch('time.monotonic',side_effect=lambda:self.now[0]):
            self.assertEqual(self.mission._strict_command_for_pose(),(0,0,False))
        self.assertTrue(all(c[0]=='stop' for c in self.client.calls))
        self.assertEqual(turn_policy.LAST_REQUEST,(0,0))
        self.assertFalse(getattr(self.mission,'_phase1_motion_fault',False))
        # The next ordinary map fit, not the tiny-motion bridge, restores authority.
        self.localizer.track(self.new,Pose2D(.06,0.,0.))
        result=SimpleNamespace(pose=Pose2D(.06,0.,0.),rmse_m=.04,inlier_ratio=.85)
        with patch('time.monotonic',side_effect=lambda:self.now[0]):
            self.mission._fuse_tracking_pose(result.pose,result)
            self.assertEqual(self.mission._strict_command_for_pose(),(240,0,False))

    def test_scan_timeout_holds_without_fatal_glass_scan_stale(self):
        self.stream.latest_after=lambda *a,**k: (_ for _ in ()).throw(TimeoutError('no new scan'))
        with patch('time.monotonic',side_effect=lambda:self.now[0]):
            self.assertEqual(self.mission._strict_command_for_pose(),(0,0,False))

    def test_late_defer_resets_controller_not_counts_unsent_kick_as_stall(self):
        held=[]
        self.mission._phase1_authorize_motion=lambda *a:False
        self.mission._phase1_motion_hold=lambda:held.append(True)
        self.client.set_twist(400,0,650)
        self.assertEqual(held,[True])
        self.assertEqual(self.client.calls,[('stop',)])

    def test_stale_rotation_uses_same_refresh_only_once_and_preserves_pulse_fence(self):
        self.request=(0,1200,False)
        with patch('time.monotonic',side_effect=lambda:self.now[0]):
            self.assertEqual(self.mission._strict_command_for_pose(),(0,1200,False))
            self.assertTrue(self.mission._obstacle_safe(self.old,0,.45))
            self.client.set_twist(0,1200,70)
        self.assertEqual(self.reads,[123])
        self.assertEqual(self.client.calls[-1][1],(0,1200,70))
        self.assertIsNotNone(self.fence.pending)

    def test_raw_beam_disagreement_cannot_promote_pose_despite_good_icp(self):
        with patch('time.monotonic',side_effect=lambda:self.now[0]), \
             patch('grasp_range_integration.phase1.turn_range_feedback.turn_range_evidence',return_value={'accepted':False}):
            self.assertEqual(self.mission._strict_command_for_pose(),(0,0,False))
        self.assertTrue(all(c[0]=='stop' for c in self.client.calls))

    def test_fresh_scan_does_not_override_actual_glass_boundary(self):
        self.localizer.track(self.old,Pose2D(-2.,0.,math.pi))
        result=SimpleNamespace(pose=Pose2D(-2.,0.,math.pi),rmse_m=.04,inlier_ratio=.85)
        with patch('time.monotonic',side_effect=lambda:self.now[0]):
            self.mission._fuse_tracking_pose(result.pose,result)
            # At 400mm/s, the forward sweep reaches the glass at x=-2.7.
            self.client.set_twist(400,0,650)
        self.assertTrue(all(c[0]=='stop' for c in self.client.calls))

    def test_delayed_ack_refresh_route_also_uses_new_scan_authorization(self):
        with patch('time.monotonic',side_effect=lambda:self.now[0]):
            self.client.refresh_twist(400,0,650)
        self.assertEqual(self.reads,[123])
        self.assertEqual(self.client.calls[-1][0],'bounded')

    def test_applied_command_model_is_reset_when_output_gate_defers(self):
        class Controller:
            def __init__(self):self.last_command=None;self.stop_count=0
            def apply(self,command):self.last_command=command;return command
            def observe(self,update):pass
            def stopped(self):self.last_command=None;self.stop_count+=1
            def ttl_for(self,command,ttl):return ttl
        self.mission.AdaptiveSpeedController=Controller
        turn_policy.install_turn_policy(self.mission)
        controller=Controller()
        command=SimpleNamespace(linear_mm_s=400,angular_mrad_s=0)
        controller.apply(command)
        self.assertIs(controller.last_command,command)
        self.mission._phase1_authorize_motion=lambda *a:False
        self.client.set_twist(400,0,650)
        self.assertIsNone(controller.last_command)
        self.assertEqual(controller.stop_count,1)
        self.assertTrue(all(c[0]=='stop' for c in self.client.calls))


if __name__=='__main__':unittest.main()
