import math
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np

from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult
from grasp_range_integration.phase1.startup_exploration import (
    scan_geometry, corridor_scores, ExplorationPolicy)
from grasp_range_integration.phase1.startup_active_recovery import assess_recovery_action


def scan(ranges=None, angles=None):
    angles = np.linspace(-math.pi, math.pi, 360, endpoint=False) if angles is None else angles
    return NS(angles_rad=angles, distances_m=np.full(len(angles),2.) if ranges is None else ranges,
              received_s=0., acquisition_start_s=0., sequence=1)


class GeometryTests(unittest.TestCase):
    def test_circle_is_not_discriminative_just_because_normals_have_two_axes(self):
        self.assertFalse(scan_geometry(scan())['informative'])

    def test_single_wall_is_not_two_direction_geometry(self):
        a=np.linspace(-math.radians(70),math.radians(70),141)
        self.assertFalse(scan_geometry(scan(2/np.cos(a),a))['informative'])

    def test_rectangular_room_has_worthwhile_structural_geometry(self):
        a=scan().angles_rad
        c,s=np.cos(a),np.sin(a)
        with np.errstate(divide='ignore'):
            r=np.minimum(2/np.maximum(np.abs(c),1e-9),
                         np.where(s>=0,3.,1.5)/np.maximum(np.abs(s),1e-9))
        result=scan_geometry(scan(r,a))
        self.assertTrue(result['informative'],result)
        self.assertGreater(result['normal_balance'],.12)

    def test_missing_rays_do_not_score_as_open_corridor(self):
        a=np.linspace(-.01,.01,20)
        self.assertTrue(all(not e['supported'] for e in corridor_scores(scan(np.full(20,5.),a))))

    def test_nan_scan_cannot_be_informative(self):
        self.assertFalse(scan_geometry(scan(np.full(360,np.nan)))['informative'])


class PolicyTests(unittest.TestCase):
    def setUp(self): self.policy=ExplorationPolicy()

    def test_3_degree_turns_do_not_request_full_localization_each_time(self):
        self.policy.fitted()
        for _ in range(4):
            self.policy.observe(Pose2D(0,0,math.radians(-3)),(0,-3500,220))
            self.assertFalse(self.policy.wants_fit(dict(informative=True)))
        self.policy.observe(Pose2D(0,0,math.radians(-4)),(0,-3500,250))
        self.assertTrue(self.policy.wants_fit(dict(informative=True)))

    def test_rich_initial_scan_requests_one_fit_then_requires_actual_new_view(self):
        self.assertTrue(self.policy.wants_fit(dict(informative=True)))
        self.policy.fitted()
        self.assertFalse(self.policy.wants_fit(dict(informative=True)))
        self.policy.observe(Pose2D(.13,0,0),(400,0,250))
        self.assertTrue(self.policy.wants_fit(dict(informative=True)))
        self.assertFalse(self.policy.wants_fit(dict(informative=False)))

    def test_millimetre_jitter_is_not_a_new_observation_location(self):
        self.policy.fitted()
        for _ in range(100): self.policy.observe(Pose2D(.001,0,math.radians(.02)),(400,0,250))
        self.assertFalse(self.policy.wants_fit(dict(informative=True)))

    def test_opposite_turns_do_not_count_as_new_view_after_returning_to_start(self):
        self.policy.fitted()
        for angle in (8,-8,8,-8):
            self.policy.observe(Pose2D(0,0,math.radians(angle)),(0,3500 if angle>0 else -3500,180))
        self.assertFalse(self.policy.wants_fit(dict(informative=True)))

    def test_tiny_real_turns_grow_duration_instead_of_resetting_breakaway(self):
        for _ in range(4):
            self.policy.observe(Pose2D(0,0,math.radians(2)),(0,3500,180))
        options,_=self.policy.candidates(scan())
        self.assertEqual(next(v[2] for v in options if v[1]>0),400)

    def test_stalled_drive_inherits_its_own_duration_and_speed_escalation(self):
        for _ in range(3): self.policy.observe(Pose2D(.001,0,0),(400,0,250))
        options,_=self.policy.candidates(scan())
        self.assertIn((460,0,400),options)

    def test_successful_slow_turn_does_not_reset_duration_to_180ms(self):
        self.policy.direction = -1
        self.policy.observe(Pose2D(0,0,math.radians(-5)),(0,-3500,400))
        options,_=self.policy.candidates(scan())
        self.assertEqual(next(v[2] for v in options if v[1]<0),400)
        self.assertEqual(next(v[2] for v in options if v[1]>0),180)

    def test_successful_fit_resets_view_novelty_not_direction_response(self):
        self.policy.observe(Pose2D(.2,0,0),(400,0,250))
        self.policy.fitted()
        self.assertFalse(self.policy.wants_fit(dict(informative=True)))
        self.assertEqual(self.policy.fit_attempts,1)

    def test_seeded_straight_start_reuses_successful_speed_and_continuous_time(self):
        policy=ExplorationPolicy(dict(speed_mm_s=460,budget_ms=2100))
        options,_=policy.candidates(scan())
        self.assertIn((460,0,400),options)
        self.assertEqual(policy.continuous_drive_budget(),2100)
        policy.observe(Pose2D(.001,0,0),(460,0,400))
        self.assertEqual(policy.continuous_drive_budget(),2400)
        self.assertIn((400,0,400),policy.candidates(scan())[0])


class ForwardSafetyTests(unittest.TestCase):
    def test_observed_forward_corridor_does_not_need_rear_clearance_input(self):
        prior=NS(x_m=0.,y_m=0.,yaw_deg=0.,radius_m=.7,yaw_tolerance_deg=180.)
        with patch('grasp_range_integration.phase1.startup_active_recovery.map_rotation_evidence',
                   return_value=dict(clear=False)), \
             patch('grasp_range_integration.phase1.startup_active_recovery.swept_glass_clearance',
                   return_value=dict(safe=True)):
            result=assess_recovery_action(scan(),object(),prior,400,0,250,clock=lambda:0.)
        self.assertTrue(result['safe'])
        self.assertFalse(result['operator_clear'])

    def test_glass_still_blocks_observed_clear_forward_corridor(self):
        prior=NS(x_m=0.,y_m=0.,yaw_deg=0.,radius_m=.7,yaw_tolerance_deg=180.)
        with patch('grasp_range_integration.phase1.startup_active_recovery.map_rotation_evidence',
                   return_value=dict(clear=False)), \
             patch('grasp_range_integration.phase1.startup_active_recovery.swept_glass_clearance',
                   return_value=dict(safe=False)):
            self.assertFalse(assess_recovery_action(scan(),object(),prior,400,0,250,clock=lambda:0.)['safe'])

    def test_front_margin_includes_speed_lease_age_and_stopping_not_only_60cm(self):
        prior=NS(x_m=0.,y_m=0.,yaw_deg=0.,radius_m=.7,yaw_tolerance_deg=180.)
        with patch('grasp_range_integration.phase1.startup_active_recovery.map_rotation_evidence',
                   return_value=dict(clear=False)), \
             patch('grasp_range_integration.phase1.startup_active_recovery.swept_glass_clearance',
                   return_value=dict(safe=True)):
            close=scan(np.full(360,.7))
            self.assertFalse(assess_recovery_action(close,object(),prior,400,0,400,clock=lambda:0.)['safe'])
            # Latest failed run had ~0.897m ahead. Fresh 400/400 remains
            # possible, while an older scan at 460 must not authorize it.
            actual=scan(np.full(360,.897))
            self.assertTrue(assess_recovery_action(actual,object(),prior,400,0,400,clock=lambda:0.)['safe'])
            self.assertFalse(assess_recovery_action(actual,object(),prior,460,0,400,clock=lambda:.1)['safe'])


class LiveExplorationPulseTests(unittest.TestCase):
    def test_final_glass_veto_returns_unsent_deferred_after_stop_without_wheel_command(self):
        from grasp_range_integration.phase1.fresh_motion_feedback import install_fresh_motion_feedback
        calls=[]
        class Client:
            def stop(self): calls.append('stop')
            def set_bounded_twist(self,*a,**k): calls.append('wheel')
        class Driver:
            def scans(self): return iter(())
        class Localizer:
            def relative_motion(self,*a): return None
        mission=NS(N10LidarDriver=Driver,Esp32Client=Client,ScanMapLocalizer=Localizer,
                   _phase1_recovery_mode='midrun')
        fence=install_fresh_motion_feedback(mission)
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.FreshScanStream',
                   return_value=NS()), \
             patch('grasp_range_integration.phase1.startup_active_recovery.assess_recovery_action',
                   return_value=dict(safe=False,reason='startup_glass_clearance')):
            mission.N10LidarDriver().scans()
            value=mission.Esp32Client().startup_recovery_pulse(scan(),NS(),object(),400,0,400,
                localizer=Localizer(),recovery_phase='midrun')
        self.assertEqual(calls,['stop'])
        self.assertEqual(value['status'],'deferred')
        self.assertFalse(value['command_sent'])
        self.assertIsNone(fence.pending)

    def test_longer_pulse_brakes_at_measured_angle_without_renewal(self):
        from raspberry_pi.config import SerialConfig
        from grasp_range_integration.phase1.fresh_motion_feedback import install_fresh_motion_feedback
        now=[0.]
        calls=[]
        class Client:
            def __init__(self): self.config=SerialConfig()
            def set_bounded_twist(self,*a,**kw): calls.append(('send',a,kw));now[0]+=.04
            def stop(self): calls.append(('stop',));now[0]+=.02
        class Driver:
            def scans(self): return iter(())
        class Localizer:
            lidar_offset=(.14,0);max_distance_m=8.
            def relative_motion(self,*a): return MatchResult(Pose2D(0,0,math.radians(13)),.02,100,True,.9)
        mission=NS(N10LidarDriver=Driver,Esp32Client=Client,ScanMapLocalizer=Localizer)
        fence=install_fresh_motion_feedback(mission)
        def read(*a):
            now[0]+=.08
            return scan()
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.FreshScanStream',
                   return_value=NS(latest_after=read)), \
             patch('time.monotonic',side_effect=lambda:now[0]), \
             patch('grasp_range_integration.phase1.startup_active_recovery.assess_recovery_action',
                   return_value=dict(safe=True)), \
             patch('grasp_range_integration.phase1.turn_range_feedback.turn_range_evidence',
                   return_value=dict(accepted=True)):
            mission.N10LidarDriver().scans()
            client=mission.Esp32Client()
            client.startup_recovery_pulse(scan(),NS(x_m=0.,y_m=0.,yaw_deg=0.,
                radius_m=.7,yaw_tolerance_deg=180.),object(),0,-3500,400,localizer=Localizer())
        self.assertEqual([v[0] for v in calls],['send','stop'])
        self.assertLess(now[0],.4)
        self.assertLess(calls[0][2]['refresh_deadline_s'],0.)
        self.assertIsNone(fence.pending)


if __name__=='__main__': unittest.main()
