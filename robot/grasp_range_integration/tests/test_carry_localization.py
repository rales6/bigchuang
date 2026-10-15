"""Pickup -> carry localization regressions, with no hardware/network calls."""
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from raspberry_pi.mapping import Pose2D
from grasp_range_integration.phase1 import startup_active_recovery as recovery
from grasp_range_integration.phase1 import visibility_route_planner as visibility
from grasp_range_integration.track_web_integration.pi import carry_localization as carry
from grasp_range_integration.track_web_integration.pi.region_navigation import install_region_skill


INSTALL = 'grasp_range_integration.phase1.relationship_find_only_entry._install_midrun_visual_recheck'


class CarryLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        self.cancel = False
        self.hint = [-.8167, .2901, .65547]  # 224635: before visual approach
        self.pose = Pose2D(-.61, .45, math.radians(-90))
        self.candidate = NS(pose=self.pose, reliable=True, rmse_m=.025, inlier_ratio=.9)
        self.events = []
        self.mission = NS(_phase1_recovery_mode='navigation', _phase1_motion_fault=None,
            _localization_acceptance=lambda rows,args:(True,self.candidate,1.,4,4))
        self.client = NS(stop=Mock(), startup_recovery_pulse=Mock())
        self.controller = NS(stopped=Mock())
        self.localizer = NS(map=object(), track=Mock())
        self.args = NS(max_replans=1)
        self.scans = self.new_scans()

    def new_scans(self):
        for sequence in range(1, 1000):
            self.now += .2
            yield NS(acquisition_start_s=self.now, received_s=self.now, sequence=sequence,
                angles_rad=np.linspace(-math.pi,math.pi,360,endpoint=False),
                distances_m=np.full(360,2.))

    def run_localizer(self, **options):
        return carry.localize_carry(self.mission,self.hint,self.args,self.client,self.scans,
            self.localizer,self.controller,lambda:self.cancel,
            event_callback=lambda name,**data:self.events.append((name,data)),**options)

    def test_real_recovery_strong_static_fix_has_no_compulsory_movement(self):
        matcher=NS(diagnostic={},visual_candidates=[],match=Mock(return_value=self.candidate))
        actual_recovery=recovery.recover_startup
        def timed(*args,**kw):
            self.assertEqual(kw['settings'].heading_tolerance_deg,180)
            self.assertTrue(kw['settings'].fresh_start_handoff)
            self.assertIsNone(kw.get('reference_scan'))
            self.assertEqual(args[7].radius_m,1.5)
            return actual_recovery(*args,clock=lambda:self.now,**kw)
        with patch.object(carry,'recover_startup',side_effect=timed), \
             patch.object(recovery,'StartupMatcher',return_value=matcher), \
             patch(INSTALL) as install:
            pose,scan=self.run_localizer()
        self.assertIs(pose,self.pose)
        self.assertEqual(scan.sequence,5)  # four fits + REAL post-fit fresh frame
        self.assertEqual(matcher.match.call_count,5)
        self.assertGreater(abs(pose.yaw_rad-self.hint[2]),math.radians(90))
        self.client.startup_recovery_pulse.assert_not_called()
        self.assertIs(install.call_args.args[-1],scan)
        self.assertIs(install.call_args.kwargs['scan_iterator'],self.scans)
        self.assertEqual(self.mission._phase1_recovery_mode,'navigation')
        self.assertTrue(any(n=='startup_carry_handoff_verified' for n,_ in self.events))

    def test_failed_consensus_uses_existing_adaptive_recovery_not_old_initializer(self):
        def recovered(*args,**kw):
            self.assertFalse(kw['settings'].operator_prompt)
            self.assertTrue(kw['settings'].initial_fit)
            self.assertEqual(self.mission._phase1_recovery_mode,'startup')
            return self.pose,NS(sequence=40)
        with patch.object(carry,'recover_startup',side_effect=recovered) as recover,patch(INSTALL):
            self.assertIs(self.run_localizer()[0],self.pose)
        recover.assert_called_once()
        self.client.stop.assert_called_once();self.controller.stopped.assert_called_once()

    def actual_with_match(self, match):
        actual=recovery.recover_startup
        def timed(*args,**kw):
            return actual(*args,clock=lambda:self.now,**kw)
        matcher=NS(diagnostic={},visual_candidates=[],match=Mock(side_effect=match))
        with patch.object(carry,'recover_startup',side_effect=timed), \
             patch.object(recovery,'StartupMatcher',return_value=matcher),patch(INSTALL) as install:
            result=self.run_localizer()
        return result,matcher,install

    def test_slow_post_fit_scoring_reacquires_real_scan_not_new_timestamp(self):
        def match(scan,deadline):
            if scan.sequence==5:
                self.now+=.6
            return self.candidate
        (pose,scan),matcher,_=self.actual_with_match(match)
        self.assertIs(pose,self.pose);self.assertEqual(scan.sequence,6)
        self.assertEqual(matcher.match.call_count,6)
        self.client.startup_recovery_pulse.assert_not_called()

    def test_changed_post_fit_frame_cannot_release_navigation(self):
        def match(scan,deadline):
            if scan.sequence==5:
                self.cancel=True
                return NS(pose=Pose2D(1,1,0),reliable=True,rmse_m=.02,inlier_ratio=.9)
            return self.candidate
        with self.assertRaises(RuntimeError):self.actual_with_match(match)
        self.assertTrue(any(n=='startup_carry_handoff_pending' for n,_ in self.events))
        self.assertFalse(any(n=='carry_localization_ready' for n,_ in self.events))
        self.client.startup_recovery_pulse.assert_not_called()

    def test_cancel_during_post_fit_score_cannot_release_navigation(self):
        def match(scan,deadline):
            if scan.sequence==5:self.cancel=True
            return self.candidate
        with self.assertRaises(RuntimeError):self.actual_with_match(match)
        self.assertFalse(any(n=='carry_localization_ready' for n,_ in self.events))
        self.client.startup_recovery_pulse.assert_not_called()

    def test_closed_scan_source_does_not_fake_post_fit_frame(self):
        import itertools
        self.scans=itertools.islice(self.scans,4)
        with self.assertRaises(StopIteration):self.actual_with_match(lambda *a:self.candidate)
        self.assertFalse(any(n=='carry_localization_ready' for n,_ in self.events))
        self.client.startup_recovery_pulse.assert_not_called()

    def test_weak_individual_scan_cannot_hide_inside_strong_average(self):
        def match(scan,deadline):
            if scan.sequence==2:
                return NS(pose=self.pose,reliable=True,rmse_m=.15,inlier_ratio=.45)
            if scan.sequence==4:self.cancel=True
            return self.candidate
        with self.assertRaises(RuntimeError):self.actual_with_match(match)
        self.assertFalse(any(n=='startup_carry_handoff_verified' for n,_ in self.events))
        self.client.startup_recovery_pulse.assert_not_called()

    def test_transport_failure_keeps_carry_state_and_restores_mode(self):
        self.mission.carry_state={'status':'carrying','carry_id':'keep'}
        with patch.object(carry,'recover_startup',side_effect=OSError('scan disconnected')),patch(INSTALL) as install:
            with self.assertRaisesRegex(OSError,'scan disconnected'):self.run_localizer()
        install.assert_not_called()
        self.assertEqual(self.mission.carry_state,{'status':'carrying','carry_id':'keep'})
        self.assertEqual(self.mission._phase1_recovery_mode,'navigation')
        self.client.stop.assert_called_once()

    def test_cancellation_before_start_cannot_move(self):
        self.cancel=True
        with patch.object(carry,'recover_startup') as recover:
            with self.assertRaises(RuntimeError):self.run_localizer()
        recover.assert_not_called();self.client.startup_recovery_pulse.assert_not_called()

    def test_cancellation_after_fit_cannot_enter_navigation(self):
        def cancel(*a,**k):
            self.cancel=True
            return self.pose,NS(sequence=5)
        with patch.object(carry,'recover_startup',side_effect=cancel),patch(INSTALL) as install:
            with self.assertRaises(RuntimeError):self.run_localizer()
        install.assert_not_called()

    def test_invalid_hint_never_becomes_navigation_pose(self):
        for hint in (None,[],[0,0],[0,0,float('nan')],['bad',0,0]):
            self.hint=hint
            with patch.object(carry,'recover_startup') as recover:
                with self.assertRaises(ValueError):self.run_localizer()
            recover.assert_not_called()

    def test_stop_failure_is_not_swallowed_and_mode_is_restored(self):
        self.client.stop.side_effect=OSError('stop disconnected')
        with patch.object(carry,'recover_startup',return_value=(self.pose,NS(sequence=5))),patch(INSTALL) as install:
            with self.assertRaisesRegex(OSError,'stop disconnected'):self.run_localizer()
        self.controller.stopped.assert_called_once()
        self.assertEqual(self.mission._phase1_recovery_mode,'navigation')
        install.assert_not_called()

    def install_skill(self, directory, *, carrying=True):
        path=Path(directory)/'job.json'
        path.write_text(json.dumps({'schema':'robot_region_job.v1','carrying':carrying,
            'candidates':[{'id':'chair'}],'pose_hint':self.hint}),encoding='utf-8')
        old=Mock(side_effect=AssertionError('frozen initializer must not run'))
        class Gpu:
            perceive=Mock()
        self.mission._localize=old
        self.mission.GpuMissionClient=Gpu
        self.mission.N10LidarDriver=type('Driver',(),{})
        self.mission.build_search_route=Mock(return_value={'views':[{}]})
        decision=Path(directory)/'decision.json'
        decision.write_text(json.dumps({'route_request':{'mode':'semantic_search_region',
            'planning_policy':visibility.POLICY}}),encoding='utf-8')
        self.args.decision=decision
        install_region_skill(self.mission,self.args,path)
        return old

    def test_carry_entry_bypasses_old_initializer_but_keeps_route_preflight(self):
        with TemporaryDirectory() as directory,patch.object(visibility,'REGION_ROUTE_BUILDER',None,create=True):
            old=self.install_skill(directory)
            fresh=NS(sequence=8)
            with patch.object(carry,'localize_carry',return_value=(self.pose,fresh)) as init:
                actual=self.mission._localize(self.args,self.client,self.scans,self.localizer,
                    self.controller,lambda:False)
            old.assert_not_called();init.assert_called_once()
            self.assertEqual(actual,(self.pose,fresh))
            self.mission.build_search_route.assert_called_once()
            self.assertEqual(self.mission.build_search_route.call_args.args[1],(self.pose.x_m,self.pose.y_m))

    def test_noncarry_entry_bypasses_old_initializer_and_keeps_route_preflight(self):
        from grasp_range_integration.track_web_integration.pi import region_localization as region
        with TemporaryDirectory() as directory,patch.object(visibility,'REGION_ROUTE_BUILDER',None,create=True):
            old=self.install_skill(directory,carrying=False)
            fresh=NS(sequence=12)
            with patch.object(carry,'localize_carry') as carry_init, \
                 patch.object(region,'localize_region',return_value=(self.pose,fresh)) as init:
                actual=self.mission._localize(self.args,self.client,self.scans,self.localizer,
                    self.controller,lambda:False)
            carry_init.assert_not_called();old.assert_not_called();init.assert_called_once()
            self.assertEqual(actual,(self.pose,fresh))
            self.mission.build_search_route.assert_called_once()
            self.assertEqual(self.mission.build_search_route.call_args.args[1],(self.pose.x_m,self.pose.y_m))

    def test_carry_route_recovery_hands_off_new_scan_without_stacking_trackers(self):
        from grasp_range_integration.phase1 import startup_route_recovery as route
        with TemporaryDirectory() as directory,patch.object(visibility,'REGION_ROUTE_BUILDER',None,create=True):
            old=self.install_skill(directory)
            raw=self.localizer.track
            old_scan=NS(sequence=5);new_scan=NS(sequence=20)
            moved=Pose2D(-.5,.55,-1.4)
            def initial(*args,**kw):
                self.localizer.track=Mock()
                return self.pose,old_scan
            self.mission.build_search_route.side_effect=[visibility.NoReachableVisibilityError('blocked'),
                                                         {'views':[{}]}]
            def recovered(*args,**kw):
                self.assertIs(kw['reference_scan'],old_scan)
                self.assertEqual(kw['settings'].phase,'midrun')
                self.assertTrue(kw['accept_candidate'](NS(pose=moved)))
                return NS(candidate=NS(pose=moved),scan=new_scan)
            with patch.object(carry,'localize_carry',side_effect=initial), \
                 patch.object(route,'recover_startup',side_effect=recovered),patch(INSTALL) as install:
                actual=self.mission._localize(self.args,self.client,self.scans,self.localizer,
                    self.controller,lambda:False)
            old.assert_not_called();self.assertEqual(actual,(moved,new_scan))
            self.assertIs(self.localizer.track,raw)
            self.assertIs(install.call_args.args[-1],new_scan)


if __name__=='__main__':unittest.main()
