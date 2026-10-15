"""Offline evidence-seeking startup tests; no sensor or motor connections."""
import math
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult
from grasp_range_integration.phase1 import startup_active_recovery as recovery


class StartupActiveRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        self.prior = NS(x_m=0., y_m=0., yaw_deg=100., radius_m=.7, yaw_tolerance_deg=40.)
        self.scan = NS(angles_rad=np.linspace(-math.pi, math.pi, 360, endpoint=False),
                       distances_m=np.full(360, 2.), received_s=0., sequence=1)
        self.candidate = NS(pose=Pose2D(), reliable=True, rmse_m=.02, inlier_ratio=.9)

    def assessment(self, *, map_clear=True, glass_clear=True, operator_clear=False):
        with patch.object(recovery, 'map_rotation_evidence', return_value=dict(clear=map_clear)), \
             patch.object(recovery, 'swept_glass_clearance', return_value=dict(safe=glass_clear)), \
             patch.object(recovery, 'assess_rotation', return_value=dict(
                 safe=map_clear, reason='clear' if map_clear else
                 'installation_edge_occlusion_without_safe_map_support',
                 min_distance_m=float(self.scan.distances_m.min()), exterior_count=360)):
            return recovery.assess_recovery_action(self.scan, object(), self.prior,
                                                  0, -3500, 180, operator_clear=operator_clear,
                                                  clock=lambda:self.now)

    def test_whole_envelope_can_authorize_view_change_without_strong_candidate(self):
        self.assertTrue(self.assessment()['safe'])

    def test_map_uncertainty_requests_operator_not_navigation_on_guessed_pose(self):
        result = self.assessment(map_clear=False)
        self.assertFalse(result['safe'])
        self.assertEqual(result['reason'], 'startup_envelope_requires_operator_clearance')

    def test_confirmed_clearance_can_support_supervised_probe_but_not_glass(self):
        self.assertTrue(self.assessment(map_clear=False, operator_clear=True)['safe'])
        self.assertFalse(self.assessment(map_clear=False, glass_clear=False, operator_clear=True)['safe'])

    def test_real_near_return_cannot_be_overridden_by_operator(self):
        self.scan.distances_m[180] = .2
        self.assertFalse(self.assessment(map_clear=False, operator_clear=True)['safe'])

    def test_old_scan_cannot_authorize_recovery(self):
        self.now = .4
        self.assertEqual(self.assessment()['reason'], 'startup_recovery_stale_scan')

    def test_measured_rotation_moves_prior_heading_instead_of_rejecting_new_heading(self):
        moved = recovery.propagate_prior(self.prior, Pose2D(.1, .02, math.radians(-12)))
        self.assertAlmostEqual(moved.yaw_deg, 88.)
        self.assertGreater(moved.radius_m, self.prior.radius_m)
        self.assertEqual(self.prior.yaw_deg, 100.)

    def run_recovery(self, *, ready=True, allow=True, feedback=True, cancelled=False,
                     initial_fit=False, motion=None, accept_candidate=None):
        self.commands, self.events = [], []
        def scans():
            for index in range(1000):
                self.now += .2
                yield NS(acquisition_start_s=self.now, received_s=self.now, sequence=index,
                         angles_rad=self.scan.angles_rad, distances_m=self.scan.distances_m)
        matcher = NS(diagnostic=dict(result='accepted_candidate'), visual_candidates=[],
                     match=lambda *a:self.candidate)
        mission = NS(_localization_acceptance=lambda rows,args:(
            ready and (initial_fit or len(self.commands)>=5),self.candidate,1.,4,4))
        client = NS(stop=lambda:None, startup_recovery_pulse=lambda *a,**kw:self.commands.append((a,kw)))
        motion = motion or MatchResult(Pose2D(.01,0,math.radians(-3)),.02,100,True,.9)
        localizer = NS(map=object(),relative_motion=lambda *a:motion,
                       lidar_offset=(.14,0),max_distance_m=8.)
        with patch.object(recovery,'assess_recovery_action',return_value=dict(safe=allow,reason='test')), \
             patch.object(recovery,'StartupMatcher',return_value=matcher), \
             patch.object(recovery,'scan_geometry',return_value=dict(informative=True)), \
             patch.object(recovery,'turn_range_evidence',return_value=dict(accepted=feedback)):
            return recovery.recover_startup(mission,NS(),client,iter(scans()),localizer,NS(),
                lambda:cancelled,self.prior,camera=None,recognizer=None,gpu=None,directory=None,
                emit=lambda name,**kw:self.events.append((name,kw)), clock=lambda:self.now,
                confirm=lambda:False, accept_candidate=accept_candidate)

    def test_good_fit_with_blocked_route_keeps_seeking_new_view(self):
        pose,scan=self.run_recovery(initial_fit=True,
            accept_candidate=lambda candidate:len(self.commands)>=5)
        self.assertEqual(pose,self.candidate.pose)
        self.assertGreaterEqual(len(self.commands),5)

    def test_move_then_new_strong_lidar_can_finish_startup_without_ds_or_target_search(self):
        pose, scan = self.run_recovery()
        self.assertEqual(pose, self.candidate.pose)
        self.assertGreaterEqual(len(self.commands),5)
        self.assertEqual(self.commands[0][0][-3:],(0,-3500,180))
        self.assertTrue(any(name=='startup_active_recovery' and e.get('status')=='localized'
                            for name,e in self.events))

    def test_informative_initial_scan_can_localize_without_any_motor_action(self):
        pose,scan=self.run_recovery(initial_fit=True)
        self.assertEqual(pose,self.candidate.pose)
        self.assertEqual(self.commands,[])
        self.assertTrue(any(name=='startup_scan_geometry' and e['initial_full_heading_check']
                            and e['localization_requested'] for name,e in self.events))

    def test_failed_initial_fit_is_not_repeated_for_22_jitter_only_actions(self):
        jitter=MatchResult(Pose2D(.001,0,math.radians(.02)),.02,100,True,.9)
        with self.assertRaisesRegex(RuntimeError,'没有实测运动进展'):
            self.run_recovery(ready=False,motion=jitter)
        fits=[e for name,e in self.events if name=='startup_recovery_scan']
        self.assertEqual(len(fits),4)

    def test_bad_feedback_does_not_guess_propagated_pose(self):
        with self.assertRaisesRegex(RuntimeError,'预算耗尽'):
            self.run_recovery(feedback=False)
        # Even before sending the first pulse, the stopped baseline alignment
        # now refuses unreliable feedback rather than guessing the new pose.
        self.assertEqual(len(self.commands),0)

    def test_cancelled_recovery_never_sends_motion(self):
        with self.assertRaisesRegex(RuntimeError,'已取消'):
            self.run_recovery(cancelled=True)
        self.assertEqual(self.commands,[])

    def test_blocked_clearance_never_sends_motion(self):
        with self.assertRaisesRegex(RuntimeError,'无安全换视角'):
            self.run_recovery(allow=False)
        self.assertEqual(self.commands,[])

    def test_failed_evidence_does_not_abort_after_first_view(self):
        with self.assertRaisesRegex(RuntimeError,'时间预算耗尽'):
            self.run_recovery(ready=False)
        self.assertGreater(len(self.commands),12)
        self.assertTrue(any(c[0][-3] == 400 for c in self.commands))

    def test_short_midrun_forward_accepts_historical_borderline_raw_mean(self):
        fit = MatchResult(Pose2D(.03751,.00714,math.radians(-1.468)),
                          .02125,100,True,.9804)
        evidence = {'accepted':False,
                    'forward':dict(supported=198,agreement=.89394,
                                   overlap=1.,supported_sectors=5,bad_sectors=0,
                                   mean_error_m=.160959),
                    'reverse':dict(supported=196,agreement=.90306,
                                   overlap=.99,supported_sectors=5,bad_sectors=0,
                                   mean_error_m=.1378)}
        self.assertTrue(recovery._strong_borderline_midrun_drive(fit,evidence))
        self.assertFalse(recovery._strong_borderline_midrun_drive(
            MatchResult(Pose2D(.03751,.00714,math.radians(-1.468)),
                        .02125,100,True,.70), evidence))
        evidence['forward']['mean_error_m'] = .181
        self.assertFalse(recovery._strong_borderline_midrun_drive(fit,evidence))


if __name__ == '__main__': unittest.main()
