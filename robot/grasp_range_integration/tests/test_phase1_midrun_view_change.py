"""Stopped uncertainty -> safe new view -> exactly-once navigation handoff."""
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np

from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult
from grasp_range_integration.phase1 import startup_active_recovery as recovery
from grasp_range_integration.phase1.localization_handoff import recovery_baseline, recovery_prediction
from grasp_range_integration.phase1.relationship_find_only_entry import (
    _install_midrun_visual_recheck, _install_waypoint_obstacle_guard)
from grasp_range_integration.phase1.motion_experience import AngularProgress, ProgressTimeout


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.now=0.;self.x=0.;self.seq=0;self.commands=[];self.events=[];self.match_calls=[];self.reviews=0
        self.prior=NS(x_m=0.,y_m=0.,yaw_deg=0.,radius_m=.10,yaw_tolerance_deg=30.)
        self.settings=recovery.RecoverySettings(phase='midrun',time_budget_s=90.,
            heading_tolerance_deg=30.,operator_prompt=False,initial_fit=False)

    def scans(self):
        while True:
            self.now+=.1;self.seq+=1
            yield NS(sequence=self.seq,received_s=self.now,acquisition_start_s=self.now,sx=self.x,
                angles_rad=np.linspace(-math.pi,math.pi,360,endpoint=False),distances_m=np.full(360,2.))

    def run_recovery(self, *, safe=True, raw=True, weak_first=False, vision_delay=0.,
                     blocked_after_one=False, inlier_override=None, candidate_offset=.01,
                     blocked_until=0., dispatch_deferrals=0, pulse_error=None,
                      confirm_visual=False, visual_offset=.01, absolute_ranges=None,
                      absolute_probe_delay=0., accept_candidate=None, raw_stationary_valid=False):
        self.requests=[]
        def pulse(*a,**kw):
            self.requests.append((a[-3:],a[0].sequence,a[1].radius_m))
            if pulse_error is not None: raise pulse_error
            if len(self.requests) <= dispatch_deferrals:
                return dict(status='deferred',command_sent=False,
                            assessment=dict(safe=False,reason='startup_glass_clearance'))
            self.commands.append(a[-3:]);self.x+=.07
        def create(localizer,prior):
            count=len(self.match_calls)
            inlier=.64 if weak_first and count==0 else .8
            if inlier_override is not None: inlier=inlier_override
            self.match_calls.append([])
            def match(scan,*a):
                self.match_calls[count].append(scan.sequence)
                return NS(pose=Pose2D(scan.sx+candidate_offset,0,0),reliable=True,rmse_m=.04,inlier_ratio=inlier)
            return NS(match=match,diagnostic={},visual_candidates=[])
        def visual(*a):
            self.reviews+=1
            if confirm_visual:
                winner=NS(pose=Pose2D(self.x+visual_offset,0,0),reliable=True,rmse_m=.04,inlier_ratio=.8)
                return 'confirmed',winner,object(),{}
            if vision_delay:
                self.now+=vision_delay
                winner=NS(pose=Pose2D(self.x+.01,0,0),reliable=True,rmse_m=.04,inlier_ratio=.8)
                return 'confirmed',winner,object(),{}
            return 'uncertain',None,None,{}
        mission=NS(_localization_acceptance=lambda rows,args:(True,rows[-1],1.,4,4),
                   _visual_localization_check=lambda *a:None)
        localizer=NS(map=object(),lidar_offset=(.14,0),max_distance_m=8.,
            relative_motion=lambda before,after:MatchResult(Pose2D(after.sx-before.sx,0,0),.02,100,True,.9))
        if absolute_ranges is not None:
            localizer.range_profile=lambda scan,maximum: scan
            self.absolute_sequences=[]
            def probe(pose,profile,deadline_s):
                if not self.absolute_sequences:self.now+=absolute_probe_delay
                self.absolute_sequences.append(profile.sequence)
                return absolute_ranges
            localizer._range_consistency=probe
        with patch.object(recovery,'StartupMatcher',side_effect=create), \
             patch.object(recovery,'scan_geometry',return_value=dict(informative=True)), \
             patch.object(recovery,'assess_recovery_action',side_effect=lambda scan,m,p,v,w,ttl,**kw:
                          dict(safe=safe and v>0 and self.now>=blocked_until
                               and not (blocked_after_one and self.commands),
                               reason='front_clear' if safe and v>0 else 'blocked')), \
             patch.object(recovery,'turn_range_evidence',side_effect=lambda before,after,*a,**kw:
                 dict(accepted=raw or (raw_stationary_valid and before.sx==after.sx))), \
             patch.object(recovery,'review_startup',side_effect=visual) as reviewed:
            outcome=recovery.recover_startup(mission,NS(),NS(stop=lambda:None,startup_recovery_pulse=pulse),
                iter(self.scans()),localizer,NS(),lambda:False,self.prior,
                camera=None,recognizer=None,gpu=None,directory=None,
                emit=lambda name,**e:self.events.append((name,e)),clock=lambda:self.now,
                settings=self.settings,reference_scan=NS(sx=0.),
                 accept_candidate=accept_candidate,
                confirm=lambda:self.fail('midrun must not wait for startup clearance input'))
            self.reviews=reviewed.call_count
            return outcome

    def test_214129_blind_turn_can_choose_observed_safe_forward_then_refit(self):
        result=self.run_recovery()
        self.assertEqual(self.commands,[(400,0,400),(400,0,400)])
        self.assertEqual(len(self.match_calls),1)
        self.assertEqual(len(self.match_calls[0]),4)
        self.assertAlmostEqual(result.predicted_pose.x_m,.14)
        self.assertAlmostEqual(result.candidate.pose.x_m,.15)
        self.assertTrue(any(name=='midrun_recovery_prior' for name,e in self.events))
        self.assertFalse(any(name.startswith('startup_') for name,e in self.events))

    def test_uncertain_ds_seeks_another_new_view_not_repeat_same_photo_or_exit(self):
        result=self.run_recovery(weak_first=True)
        self.assertEqual(len(self.commands),4)
        self.assertEqual(len(self.match_calls),2)
        self.assertEqual(self.reviews,1)
        self.assertAlmostEqual(result.predicted_pose.x_m,.28)

    def test_visual_delay_requires_a_real_new_scan_before_navigation_handoff(self):
        result=self.run_recovery(weak_first=True,vision_delay=3.)
        self.assertEqual(len(self.match_calls[0]),5)
        self.assertLessEqual(self.now-result.scan.received_s,.5)
        self.assertEqual(self.reviews,1)

    def test_no_safe_action_retries_fresh_static_consensus_without_moving_or_prompting(self):
        result=self.run_recovery(safe=False)
        self.assertEqual(self.commands,[])
        self.assertEqual(len(self.match_calls[0]),4)
        self.assertLessEqual(self.now-result.scan.received_s,.5)
        self.assertTrue(any(name=='midrun_recovery_waiting_for_scan' and not e['terminal']
                            for name,e in self.events))

    def test_invalid_raw_evidence_cannot_seed_a_new_pose_or_motion(self):
        with self.assertRaisesRegex(RuntimeError,'预算耗尽'):
            self.run_recovery(raw=False)
        self.assertEqual(self.commands,[])

    def test_153617_strong_map_with_pending_raw_motion_can_reach_visual_review(self):
        result=self.run_recovery(raw=False,confirm_visual=True)
        self.assertEqual(self.reviews,1)
        self.assertEqual(self.commands,[])
        self.assertAlmostEqual(result.predicted_pose.x_m,.01)
        self.assertTrue(any(name=='midrun_recovery_feedback_resolved'
                            and e['source']=='stationary_map_and_visual_anchor' for name,e in self.events))

    def test_165830_pending_relative_feedback_uses_real_absolute_range_probe(self):
        result=self.run_recovery(raw=False,absolute_ranges=(.85,.07,.80))
        self.assertEqual(self.reviews,0)
        self.assertEqual(self.commands,[])
        self.assertAlmostEqual(result.predicted_pose.x_m,result.candidate.pose.x_m)
        self.assertTrue(any(name=='midrun_recovery_absolute_range_anchor' and e['accepted']
                            and e['sequence']==result.scan.sequence for name,e in self.events))
        self.assertTrue(any(name=='midrun_recovery_feedback_resolved'
                            and e['source']=='stationary_map_consensus' for name,e in self.events))

    def test_bad_absolute_range_probe_does_not_promote_unknown_motion(self):
        with self.assertRaisesRegex(RuntimeError,'预算耗尽'):
            self.run_recovery(raw=False,absolute_ranges=(.50,.40,.90))
        self.assertEqual(self.commands,[])

    def test_pending_motion_absolute_anchor_rechecks_real_post_fit_scan_without_unknown_relative(self):
        result=self.run_recovery(raw=False,absolute_ranges=(.85,.07,.80),absolute_probe_delay=.8)
        self.assertEqual(self.commands,[])
        self.assertEqual(self.reviews,0)
        self.assertEqual(len(self.absolute_sequences),2)
        self.assertGreater(self.absolute_sequences[1],self.absolute_sequences[0])
        self.assertEqual(result.scan.sequence,self.absolute_sequences[-1])
        self.assertLessEqual(self.now-result.scan.received_s,.5)

    def test_173117_route_veto_retains_absolute_pose_so_guarded_exploration_can_continue(self):
        route_checks=[]
        def ready(candidate):
            route_checks.append(candidate.pose)
            return len(route_checks)>1
        result=self.run_recovery(raw=False,absolute_ranges=(.95,.05,.86),accept_candidate=ready,
                                 raw_stationary_valid=True)
        self.assertGreater(len(self.commands),0)
        resolved=[e for n,e in self.events if n=='midrun_recovery_feedback_resolved'
                  and e['source']=='stationary_absolute_anchor_route_pending']
        self.assertEqual(len(resolved),1)
        self.assertTrue(resolved[0]['localization_retained'])
        self.assertFalse(resolved[0]['mission_navigation_enabled'])
        self.assertAlmostEqual(resolved[0]['radius_m'],.22)
        self.assertGreater(result.scan.sequence,resolved[0]['sequence'])

    def test_rejected_absolute_pose_or_range_cannot_commit_a_route_pending_anchor(self):
        with self.assertRaisesRegex(RuntimeError,'预算耗尽'):
            self.run_recovery(raw=False,absolute_ranges=(.50,.40,.86),accept_candidate=lambda _:False)
        self.assertFalse(any(n=='midrun_recovery_feedback_resolved' for n,e in self.events))
        self.assertEqual(self.commands,[])

    def test_pending_raw_motion_uncertain_vision_is_not_repeated_on_same_view(self):
        with self.assertRaisesRegex(RuntimeError,'预算耗尽'):
            self.run_recovery(raw=False)
        self.assertEqual(self.reviews,1)
        self.assertEqual(self.commands,[])

    def test_visual_cannot_reanchor_outside_motion_search_envelope(self):
        with self.assertRaisesRegex(RuntimeError,'预算耗尽'):
            self.run_recovery(raw=False,confirm_visual=True,visual_offset=1.)
        self.assertEqual(self.commands,[])

    def test_last_short_safe_view_is_checked_without_another_unsafe_move(self):
        result=self.run_recovery(blocked_after_one=True)
        self.assertEqual(len(self.commands),1)
        self.assertEqual(len(self.match_calls),1)
        self.assertAlmostEqual(result.predicted_pose.x_m,.07)
        self.assertTrue(any(name=='midrun_recovery_last_safe_view' for name,e in self.events))

    def test_failed_last_safe_view_can_refit_new_lidar_without_repeating_ds_photo(self):
        result=self.run_recovery(blocked_after_one=True,weak_first=True)
        self.assertEqual(len(self.commands),1)
        self.assertEqual(len(self.match_calls),2)
        self.assertEqual(self.reviews,1)
        self.assertAlmostEqual(result.predicted_pose.x_m,.07)

    def test_consistent_midrun_65_percent_lidar_does_not_require_ds(self):
        self.run_recovery(inlier_override=.67)
        self.assertEqual(self.reviews,0)

    def test_correction_inside_measured_prior_is_not_cut_off_at_20cm(self):
        self.prior.radius_m=.31
        result=self.run_recovery(candidate_offset=.29)
        self.assertAlmostEqual(result.candidate.pose.x_m-result.predicted_pose.x_m,.29)

    def test_correction_outside_bounded_prior_never_releases_navigation(self):
        with self.assertRaisesRegex(RuntimeError,'预算耗尽'):
            self.run_recovery(safe=False,candidate_offset=.5)
        self.assertEqual(self.commands,[])

    def test_persistent_blocked_scan_and_weak_lidar_keep_no_motion_until_timeout(self):
        with self.assertRaisesRegex(RuntimeError,'预算耗尽'):
            self.run_recovery(safe=False,inlier_override=.61)
        self.assertEqual(self.commands,[])
        self.assertEqual(self.reviews,0)
        self.assertGreater(len(self.match_calls),1)

    def test_pre_send_glass_veto_reacquires_scan_and_shortens_lease_without_fake_motion(self):
        result=self.run_recovery(dispatch_deferrals=1)
        self.assertEqual(self.requests[0][0],(400,0,400))
        self.assertEqual(self.requests[1][0],(400,0,250))
        self.assertGreater(self.requests[1][1],self.requests[0][1])
        self.assertAlmostEqual(self.requests[1][2],self.requests[0][2])
        # The shorter retry is not a permanent lease ceiling: a newly clear
        # frame may reauthorize the normal breakaway lease again.
        self.assertEqual(self.commands,[(400,0,250),(400,0,400)])
        self.assertAlmostEqual(result.predicted_pose.x_m,.14)
        self.assertTrue(any(name=='midrun_recovery_dispatch_deferred' and not e['terminal']
                            and not e['command_sent'] for name,e in self.events))

    def test_repeated_veto_never_lowers_speed_or_erases_guard_and_search_bounds(self):
        self.run_recovery(dispatch_deferrals=3)
        self.assertEqual([row[0] for row in self.requests[:4]],
                         [(400,0,400),(400,0,250),(400,0,200),(400,0,150)])
        self.assertEqual(len({row[1] for row in self.requests[:4]}),4)
        self.assertEqual(len(self.commands),2)

    def test_persistent_pre_send_veto_never_sends_motion_or_resets_deadline(self):
        with self.assertRaisesRegex(RuntimeError,'预算耗尽'):
            self.run_recovery(dispatch_deferrals=10000)
        self.assertEqual(self.commands,[])
        self.assertTrue(all(row[0][2]>=150 for row in self.requests))

    def test_transport_error_is_not_misreported_as_an_unsent_veto(self):
        with self.assertRaisesRegex(RuntimeError,'transport_failure'):
            self.run_recovery(pulse_error=RuntimeError('transport_failure'))
        self.assertEqual(len(self.requests),1)
        self.assertEqual(self.commands,[])


class HandoffAndWiringTests(unittest.TestCase):
    def setUp(self):
        self.old_pose=Pose2D(-1.4,-1.5,1.)
        self.source=NS(sequence=2,received_s=0.)
        self.final=NS(sequence=10,received_s=10.)
        self.prediction=self.old_pose.compose(Pose2D(.24,0,.1))
        self.strong=NS(pose=self.prediction,rmse_m=.04,inlier_ratio=.8,score=1.,reliable=True)
        self.weak=NS(pose=self.old_pose,rmse_m=.06,inlier_ratio=.56,score=1.,reliable=True)
        self.events=[];self.stops=[];self.vision=[]
        self.client=NS(stop=lambda:self.stops.append('stop'),startup_recovery_pulse=lambda *a,**k:None)
        self.controller=NS(stopped=lambda:None)
        self.progress=AngularProgress(clock=lambda:10.)

    def install(self, *, clearance=False):
        mission=NS(_phase1_verified_glass_required=True, LocalizationResult=lambda **kw:NS(**kw),
            _tracking_usable_for_photo_mission=lambda r,a:r is not None and r.rmse_m<=.16 and r.inlier_ratio>=.4,
            _visual_localization_check=lambda *a:(self.vision.append(1) or 'uncertain',self.weak,None,{}),
            _phase1_angular_progress=self.progress)
        localizer=NS(track=lambda *a:self.weak,
                     localize_near_position=lambda *a,**k:self.weak)
        if clearance:
            entered=[False]
            def tracked(*a):
                if entered[0]: return self.weak
                entered[0]=True
                try:
                    recovered=localizer._phase1_stationary_map_recheck(*a)
                    return recovered or self.weak
                finally:
                    entered[0]=False
            localizer.track=tracked
        _install_midrun_visual_recheck(mission,localizer,NS(),self.client,self.controller,
            object(),object(),object(),Path('unused'),lambda name,**e:self.events.append((name,e)),
            scan_iterator=iter(()))
        return mission,localizer

    def outcome(self): return recovery.RecoveryEvidence(self.strong,self.final,self.prediction)

    def test_weak_track_and_ds_abstention_enter_view_change_on_second_scan(self):
        mission,localizer=self.install()
        mission.LocalizationResult=lambda **kw:NS(**kw)
        with patch.object(recovery,'recover_startup',return_value=self.outcome()) as recover:
            self.assertIsNone(localizer.track(self.source,self.old_pose))
            self.assertIs(localizer.track(self.source,self.old_pose),self.strong)
        self.assertEqual(recover.call_count,1)
        settings=recover.call_args.kwargs['settings']
        self.assertFalse(settings.initial_fit)
        self.assertFalse(settings.operator_prompt)
        self.assertEqual(settings.heading_tolerance_deg,30.)
        self.assertIs(mission._phase1_localization_handoff['source_scan'],self.source)
        self.assertIsNone(mission._phase1_navigation_estimate)
        self.assertEqual(len(self.vision),1)

    def test_clearance_recheck_failure_changes_view_without_old_pose_rejection(self):
        mission,localizer=self.install(clearance=True)
        with patch.object(recovery,'recover_startup',return_value=self.outcome()) as recover:
            result=localizer.track(self.source,self.old_pose)
        self.assertIs(result,self.strong)
        self.assertEqual(recover.call_count,1)

    def test_obstacle_vote_static_callback_cannot_move_behind_mission_pose(self):
        _,localizer=self.install()
        with patch.object(recovery,'recover_startup') as recover:
            self.assertIsNone(localizer._phase1_stationary_map_recheck(self.source,self.old_pose))
        recover.assert_not_called()

    def test_pose_and_scan_handoff_are_consumed_once_and_no_duplicate_recovery_translation(self):
        mission,localizer=self.install()
        with patch.object(recovery,'recover_startup',return_value=self.outcome()):
            localizer.track(self.source,self.old_pose);localizer.track(self.source,self.old_pose)
        future=NS(sequence=11)
        with self.assertRaisesRegex(RuntimeError,'位姿尚未交接'):
            recovery_baseline(mission,localizer,self.source,future)
        prediction,handoff=recovery_prediction(mission,self.strong,self.old_pose)
        self.assertIs(prediction,self.prediction)
        self.assertIs(recovery_baseline(mission,localizer,self.source,future),self.final)
        self.assertIs(recovery_prediction(mission,self.strong,self.old_pose)[0],self.old_pose)
        self.assertIs(recovery_baseline(mission,localizer,self.source,future),self.source)

    def test_fusion_uses_measured_recovery_prediction_not_clipped_old_hint(self):
        from raspberry_pi.object_search_mission import _fuse_tracking_pose
        mission,localizer=self.install()
        mission._obstacle_safe=lambda *a:True
        mission.build_search_route=lambda *a:None
        mission._fuse_tracking_pose=_fuse_tracking_pose
        _install_waypoint_obstacle_guard(mission)
        with patch.object(recovery,'recover_startup',return_value=self.outcome()):
            localizer.track(self.source,self.old_pose);localizer.track(self.source,self.old_pose)
        fused=mission._fuse_tracking_pose(self.old_pose,self.strong)
        self.assertAlmostEqual(fused.x_m,self.prediction.x_m)
        self.assertAlmostEqual(fused.y_m,self.prediction.y_m)

    def test_verified_stationary_work_is_excluded_but_does_not_reset_travel_budget(self):
        now=[0.]
        mission,localizer=self.install()
        moved=NS(converged=True,rmse_m=.02,inlier_ratio=.9,transform=Pose2D(.1,0,0))
        localizer.relative_motion=lambda *a:moved
        localizer.map=NS(world_to_grid=lambda p:[(1,1)],in_bounds=lambda *a:True,
                         traversable=np.ones((2,2),bool))
        # Re-install with scan-to-scan prediction for this isolated test.
        localizer.track=lambda *a:self.weak
        _install_midrun_visual_recheck(mission,localizer,NS(),self.client,self.controller,
            object(),object(),object(),Path('unused'),lambda name,**e:self.events.append((name,e)),
            initial_scan=self.source)
        mission._phase1_refresh_rotation_evidence=lambda scan:(
            NS(sequence=30,received_s=now[0],acquisition_start_s=now[0]-.05),True)
        def slow_visual(*a):
            now[0]+=10
            return 'uncertain',self.weak,None,{}
        mission._visual_localization_check=slow_visual
        with patch('time.monotonic',side_effect=lambda:now[0]):
            first=localizer.track(self.source,self.old_pose)
            self.assertEqual(first.score,-1.)
            localizer._phase1_stationary_map_recheck(self.source,self.old_pose)
            now[0]+=5
            later=localizer.track(NS(sequence=31,received_s=now[0]),self.old_pose)
        self.assertEqual(later.score,-1.)
        self.assertAlmostEqual(mission._phase1_navigation_estimate['distance_m'],.2)
        self.assertTrue(any(name=='midrun_stationary_time_excluded' for name,e in self.events))

    def test_unverified_stop_cannot_exclude_wait_time_or_freshen_old_pose(self):
        mission,localizer=self.install()
        mission._phase1_refresh_rotation_evidence=lambda scan:(self.final,False)
        with patch('time.monotonic',return_value=10.):
            self.assertIsNone(localizer._phase1_stationary_map_recheck(self.source,self.old_pose))
        self.assertFalse(any(name=='midrun_stationary_time_excluded' for name,e in self.events))
        self.assertFalse(hasattr(mission,'_phase1_stationary_bridge_pair'))
        self.assertEqual(self.progress.suspended_intervals,[])

    def test_feedback_adapter_fits_from_recovered_baseline_not_whole_recovery_twice(self):
        from grasp_range_integration.phase1.fresh_motion_feedback import install_fresh_motion_feedback
        calls=[]
        class Localizer:
            def relative_motion(self,before,after):
                calls.append((before,after))
                return MatchResult(Pose2D(.01,0,0),.02,100,True,.9)
        mission=NS(N10LidarDriver=object,Esp32Client=object,ScanMapLocalizer=Localizer)
        install_fresh_motion_feedback(mission)
        localizer=mission.ScanMapLocalizer()
        self.final.angles_rad=np.zeros(100);self.final.distances_m=np.ones(100)
        future=NS(sequence=11,angles_rad=np.zeros(100),distances_m=np.ones(100))
        mission._phase1_localization_handoff=dict(result=self.strong,source_scan=self.source,
            final_scan=self.final,predicted_pose=self.prediction,localizer=localizer,
            pose_consumed=True,baseline_consumed=False)
        result=localizer.relative_motion(self.source,future)
        self.assertIs(calls[0][0],self.final)
        self.assertAlmostEqual(result.transform.x_m,.01)
        self.assertTrue(mission._phase1_localization_handoff['baseline_consumed'])


class DeadlineTests(unittest.TestCase):
    def test_recovery_is_not_navigation_stall_and_overlapping_pauses_are_counted_once(self):
        progress=AngularProgress();deadline=0.+ProgressTimeout(progress)
        progress.suspend(5,15);progress.suspend(10,20)
        self.assertFalse(49.>=deadline)
        self.assertTrue(50.>=deadline)
        # A new path deadline must not inherit time excluded before creation.
        self.assertTrue(60.>=25.+ProgressTimeout(progress))


class RecordedStopTests(unittest.TestCase):
    def setUp(self):
        import json
        d=json.loads((Path(__file__).parent/'fixtures'/'midrun_stop_20260927_214129.json').read_text())
        self.scan=NS(angles_rad=np.array(d['raw_angles_rad']),
                     distances_m=np.array(d['raw_distances_m'],float),received_s=0.)
        self.prior=NS(x_m=d['pose'][0],y_m=d['pose'][1],yaw_deg=math.degrees(d['pose'][2]),
                      radius_m=d['uncertainty_m'],yaw_tolerance_deg=30.)

    def assess(self, v=400, w=0, *, directional=True, glass=True):
        with patch.object(recovery,'map_rotation_evidence',return_value=dict(clear=False)), \
             patch('grasp_range_integration.phase1.rotation_guard.map_rotation_evidence',
                   return_value=dict(clear=False)), \
             patch.object(recovery,'swept_glass_clearance',return_value=dict(safe=glass)):
            with patch.object(recovery,'bounded_recovery_glass_sweep',return_value=dict(safe=glass)):
                return recovery.assess_recovery_action(self.scan,object(),self.prior,v,w,400,
                    directional_translation=directional,clock=lambda:.1)

    def test_real_214129_scan_can_translate_but_does_not_authorize_blind_rotation(self):
        self.assertFalse(self.assess(directional=False)['safe'])
        self.assertTrue(self.assess()['safe'])
        self.assertFalse(self.assess()['rear_turn_authorized'])
        self.assertFalse(self.assess(v=0,w=3500)['safe'])
        self.assertFalse(self.assess(v=0,w=-3500)['safe'])

    def test_glass_and_missing_side_rays_still_block_translation(self):
        self.assertFalse(self.assess(glass=False)['safe'])
        mask=np.abs(self.scan.angles_rad)<=math.radians(50)
        self.scan.angles_rad=self.scan.angles_rad[mask]
        self.scan.distances_m=self.scan.distances_m[mask]
        self.assertFalse(self.assess()['safe'])

    def test_real_new_frontal_obstacle_is_not_confused_with_old_side_return(self):
        i=np.argmin(abs(self.scan.angles_rad))
        self.scan.distances_m[i]=.35
        self.assertFalse(self.assess()['safe'])

    def test_actual_release_map_and_recorded_frame_support_forward_not_blind_turn(self):
        from raspberry_pi.known_map_navigation import KnownOccupancyMap
        from grasp_range_integration.phase1.glass_boundary_guard import load_verified_boundaries,attach_boundaries
        root=Path(__file__).resolve().parents[2]
        maps=root if (root/'ble_scene_17.pgm').exists() else root/'maps'/'semantic_release_20260919'
        if not (maps/'ble_scene_17.pgm').exists(): maps=root.parent/'maps'/'current'
        config=load_verified_boundaries(root/'grasp_range_integration/phase1/glass_boundaries.json',
            maps/'ble_scene_17.pgm',maps/'ble_scene_17.yaml')
        m=KnownOccupancyMap.load(maps/'ble_scene_17.pgm',maps/'ble_scene_17.yaml',clearance_m=.3)
        attach_boundaries(m,config)
        checks=[recovery.assess_recovery_action(self.scan,m,self.prior,v,w,ttl,
                    directional_translation=True,clock=lambda:.1)
                for v,w,ttl in ((0,3500,180),(400,0,400),(0,-3500,180))]
        self.assertEqual([v['safe'] for v in checks],[False,True,False])
        self.assertTrue(checks[1]['glass']['safe'])


class RecordedDispatchMarginTests(unittest.TestCase):
    def setUp(self):
        import json
        from raspberry_pi.known_map_navigation import KnownOccupancyMap
        from grasp_range_integration.phase1.glass_boundary_guard import load_verified_boundaries,attach_boundaries
        self.record=json.loads((Path(__file__).parent/'fixtures'/'dispatch_glass_20261001_204249.json').read_text())
        root=Path(__file__).resolve().parents[2]
        maps=root if (root/'ble_scene_17.pgm').exists() else root/'maps'/'semantic_release_20260919'
        if not (maps/'ble_scene_17.pgm').exists(): maps=root.parent/'maps'/'current'
        config=load_verified_boundaries(root/'grasp_range_integration/phase1/glass_boundaries.json',
                                       maps/'ble_scene_17.pgm',maps/'ble_scene_17.yaml')
        self.map=KnownOccupancyMap.load(maps/'ble_scene_17.pgm',maps/'ble_scene_17.yaml',clearance_m=.3)
        attach_boundaries(self.map,config)
        p=self.record['pose']
        self.prior=NS(x_m=p[0],y_m=p[1],yaw_deg=math.degrees(p[2]),
                      radius_m=self.record['position_uncertainty_m'],yaw_tolerance_deg=30.)
        # Test only the recorded glass scalars. This open synthetic scan does
        # not claim to reconstruct missing raw ranges or authorize live motion.
        self.scan=NS(angles_rad=np.linspace(-math.pi,math.pi,360,endpoint=False),
                     distances_m=np.full(360,2.),received_s=0.)

    def assess(self, ttl, age, margin=0.):
        return recovery.assess_recovery_action(self.scan,self.map,self.prior,400,0,ttl,
            directional_translation=True,clock=lambda:age,dispatch_margin_s=margin)

    def test_recorded_parallel_heading_no_longer_counts_travel_as_motion_toward_glass(self):
        selection=self.assess(400,self.record['selection_scan_age_s'])
        dispatch=self.assess(400,self.record['dispatch_scan_age_s'])
        self.assertTrue(selection['safe'])
        self.assertTrue(dispatch['safe'])
        self.assertEqual(dispatch['glass']['authority'],'bounded_heading_actual_command_sweep')
        self.assertFalse(dispatch['glass']['travel_double_counted'])
        self.assertLess(dispatch['glass']['required_boundary_distance_m'],self.record['dispatch_required_m'])

    def test_directional_sweep_keeps_safe_parallel_leases_despite_old_disk_margin(self):
        age=self.record['selection_scan_age_s']
        self.assertTrue(self.assess(400,age,.05)['safe'])
        self.assertTrue(self.assess(250,age,.05)['safe'])
        self.assertTrue(self.assess(250,self.record['dispatch_scan_age_s'])['safe'])


class ForwardCoverageDiagnosticsTests(unittest.TestCase):
    def assess(self, gap_degrees):
        # Float conversion must not turn an exact 10deg gap into a real gap.
        degrees=np.arange(-180.,180.)
        keep=~((degrees>20.) & (degrees<20.+gap_degrees))
        scan=NS(angles_rad=np.radians(degrees[keep]),distances_m=np.full(keep.sum(),2.),received_s=0.)
        prior=NS(x_m=0.,y_m=0.,yaw_deg=0.,radius_m=.1,yaw_tolerance_deg=30.)
        with patch.object(recovery,'map_rotation_evidence',return_value=dict(clear=False)), \
             patch.object(recovery,'swept_glass_clearance',return_value=dict(safe=True)):
            with patch.object(recovery,'bounded_recovery_glass_sweep',return_value=dict(safe=True)):
                return recovery.assess_recovery_action(scan,object(),prior,400,0,150,
                    directional_translation=True,clock=lambda:0.)

    def test_ten_degree_float_boundary_does_not_reject_clear_sweep(self):
        result=self.assess(10)
        self.assertTrue(result['safe'])
        self.assertTrue(result['coverage_complete'])
        self.assertAlmostEqual(result['coverage_max_gap_deg'],10.)

    def test_real_missing_rays_have_separate_reason_and_never_authorize_motion(self):
        result=self.assess(11)
        self.assertFalse(result['safe'])
        self.assertEqual(result['reason'],'forward_scan_coverage_incomplete')
        self.assertGreater(result['nearest_sweep_distance_m'],result['sweep_radius_m'])


if __name__=='__main__': unittest.main()
