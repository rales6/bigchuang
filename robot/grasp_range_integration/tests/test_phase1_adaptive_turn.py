"""Replay the 19:04 stalled-turn failure with current ACK/scan feedback."""
import math
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult
from grasp_range_integration.phase1.adaptive_turn import AdaptiveTurnResponse
from grasp_range_integration.phase1.turn_start_recovery import run_turn_start_recovery
from grasp_range_integration.phase1.turn_policy import TurnConvergence, install_turn_policy
from grasp_range_integration.phase1 import turn_policy


class AdaptiveResponseTests(unittest.TestCase):
    def setUp(self):
        self.now=0.
        self.events=[]
        self.model=AdaptiveTurnResponse(lambda name,**kw:self.events.append((name,kw)),
                                        clock=lambda:self.now)
        self.error=math.radians(23.)

    def test_failed_92ms_pulse_grows_actual_duration_instead_of_repeating_20_times(self):
        self.model.directions[1].rate=2.504721964
        durations=[]
        speed,ttl=3500,92
        for _ in range(4):
            self.model.feedback(speed,ttl,math.radians(.025),.001,True,self.error)
            speed,ttl,_,_=self.model.plan(self.error,2600,650)
            durations.append(ttl)
        self.assertEqual(durations,[132,212,250,250])
        self.assertEqual(speed,3500)
        self.assertLess(self.model.directions[1].rate,.5)

    def test_left_feedback_does_not_change_right_recent_rate(self):
        self.model.feedback(-3500,180,math.radians(-3.),.002,True,-self.error)
        right=self.model.directions[-1].rate
        self.model.feedback(3500,113,math.radians(16.2),.002,True,self.error)
        self.assertEqual(self.model.directions[-1].rate,right)
        self.assertNotEqual(self.model.directions[1].rate,right)

    def test_reliable_pivot_translation_does_not_disable_recent_rate_learning(self):
        self.model.feedback(3500,113,math.radians(16.217),.0511,True,self.error)
        self.assertAlmostEqual(self.model.directions[1].rate,math.radians(16.217)/.113)
        self.assertEqual(self.model.directions[1].recent_samples,1)

    def test_recent_rate_can_decrease_not_only_grow(self):
        self.model.feedback(3500,180,math.radians(15),.002,True,self.error)
        fast=self.model.directions[1].rate
        self.model.feedback(3500,180,math.radians(3),.002,True,self.error)
        self.assertLess(self.model.directions[1].rate,fast)

    def test_learned_start_preserves_tested_gains_and_failure_does_not_lower_torque(self):
        learned=dict(fastest_rate_rad_s=1.,successes=10)
        speed,ttl,stage,_=self.model.plan(self.error,2600,650,learned=learned)
        self.assertEqual(stage,6)
        self.model.feedback(speed,ttl,0.,.001,True,self.error)
        speed,next_ttl,next_stage,_=self.model.plan(self.error,2600,650,learned=learned)
        self.assertEqual(speed,3500)
        self.assertGreater(next_ttl,ttl)
        self.assertGreaterEqual(next_stage,stage)

    def test_183124_slow_inherited_rate_cannot_request_full_180ms_near_goal(self):
        error=math.radians(-19.196)
        learned=dict(fastest_rate_rad_s=.462,successes=10)
        self.model.directions[-1].rate=.462
        speed,ttl,_,reused=self.model.plan(error,-1200,650,learned=learned)
        self.assertTrue(reused)
        self.assertEqual(speed,-3500)
        self.assertLessEqual(ttl,85)
        self.model.feedback(speed,ttl,0.,.001,True,error)
        _,retry_ttl,_,_=self.model.plan(error,-1200,650,learned=learned)
        self.assertGreater(retry_ttl,ttl)

    def test_small_correction_does_not_inherit_high_torque_large_angle_start(self):
        learned=dict(fastest_rate_rad_s=1.,successes=10)
        speed,ttl,stage,reused=self.model.plan(math.radians(6),1200,650,learned=learned)
        self.assertFalse(reused)
        self.assertEqual(speed,1200)
        self.assertEqual(stage,2)

    def test_reversing_after_long_successful_turn_does_not_inherit_opposite_stall_clock(self):
        self.model.plan(self.error,2600,650)
        self.model.feedback(3500,180,math.radians(3.),.001,True,self.error)
        self.now=31.
        speed,ttl,_,_=self.model.plan(-self.error,-2600,650)
        self.model.feedback(speed,ttl,0.,.001,True,-self.error)
        self.assertEqual(self.model.directions[-1].low_progress,1)

    def test_twitch_point_9_degree_does_not_reset_recovery_for_23deg_error(self):
        self.model.feedback(3500,92,0.,.002,True,self.error)
        self.model.feedback(3500,132,math.radians(.9),.002,True,self.error)
        self.assertEqual(self.model.directions[1].low_progress,2)

    def test_small_alignment_also_grows_time_after_real_failures(self):
        speed,ttl=1200,50
        for _ in range(4):
            self.model.feedback(speed,ttl,0.,.001,True,math.radians(6))
            speed,ttl,_,_=self.model.plan(math.radians(6),1200,650)
        self.assertEqual(ttl,180)
        self.assertGreater(speed,1200)
        self.assertLessEqual(speed,3500)

    def test_unreliable_feedback_cannot_authorize_escalation(self):
        self.model.feedback(3500,92,0.,.001,False,self.error)
        self.assertEqual(self.model.directions[1].low_progress,0)

    def test_parent_budget_and_hardware_speed_are_not_exceeded(self):
        self.model.feedback(3500,180,0.,.001,True,self.error)
        speed,ttl,_,_=self.model.plan(self.error,2600,100)
        self.assertEqual(ttl,100)
        self.assertEqual(speed,3500)

    def test_only_elapsed_real_stall_not_attempt_count_has_fault_watchdog(self):
        for _ in range(50):
            self.model.feedback(3500,250,0.,.001,True,self.error)
            self.now+=.1
        self.now=31.
        with self.assertRaisesRegex(RuntimeError,'30秒'):
            self.model.feedback(3500,250,0.,.001,True,self.error)

    def test_overshoot_reverses_and_shrinks_pulse(self):
        learned=dict(fastest_rate_rad_s=1.,successes=10)
        _,before,_,_=self.model.plan(self.error,2600,650,learned=learned)
        self.model.reverse_correction()
        speed,after,_,_=self.model.plan(-self.error,-2600,650,learned=learned)
        self.assertLess(after,before)
        self.assertLess(abs(speed),3500)
        self.assertLess(speed,0)

    def test_three_shrinking_overshoots_are_corrections_not_terminal_failure(self):
        monitor=TurnConvergence(corrective_reversals=True)
        for index,angle in enumerate((20,-12,8,-5,3,-2)):
            monitor.check('station',math.radians(angle),now=index)
        self.assertEqual(monitor.crossings,5)

    def test_nonshrinking_oscillation_and_wrong_direction_still_fail(self):
        monitor=TurnConvergence(corrective_reversals=True)
        with self.assertRaises(RuntimeError):
            for index,angle in enumerate((20,-10,11,-12,13)):
                monitor.check('station',math.radians(angle),now=index)


class LiveTurnTests(unittest.TestCase):
    def run_live(self, *, safe=True, angle=0., reliable=True, delay=0., ttl=250, shift=0.):
        self.now=0.;self.calls=[];self.events=[]
        def read(sequence,timeout):
            if sequence is not None:self.now+=.08
            return NS(sequence=1+int(self.now*100),received_s=self.now)
        def send(speed,ttl,deadline):
            self.calls.append(('send',speed,ttl,deadline));self.now+=delay
        return run_turn_start_recovery(error_rad=math.radians(23),
            read_scan=read,measure=lambda *a:MatchResult(Pose2D(shift,0,math.radians(angle)),
                                                       .02,100,reliable,.9),
            assess=lambda s:dict(safe=safe,reason='clear' if safe else 'exterior_near_obstacle'),
            send=send,renew=lambda *a:self.fail('No renewal'),
            stop=lambda:self.calls.append(('stop',)),
            emit=lambda name,**kw:self.events.append((name,kw)),clock=lambda:self.now,
            maximum_ttl_ms=ttl,stalled_scans=8,measured_rate_rad_s=2.5047,
            adaptive_start=True,pulse_speed_mrad_s=3500)

    def test_real_helper_does_not_recap_dynamic_250ms_to_old_92ms(self):
        self.assertEqual(self.run_live(), 'single_pulse_complete')
        self.assertEqual(self.calls[0][1:3],(3500,250))
        self.assertEqual(self.calls[-1],('stop',))
        self.assertLess(self.calls[0][3],0.)

    def test_live_measured_8deg_brakes_before_long_lease_expires(self):
        self.assertEqual(self.run_live(angle=9.),'live_feedback_brake')
        self.assertLess(self.now,.25)
        self.assertEqual(self.calls[-1],('stop',))

    def test_live_reliable_translation_alone_does_not_brake(self):
        self.assertEqual(self.run_live(angle=2., shift=.25), 'single_pulse_complete')
        self.assertGreaterEqual(self.now, .25)

    def test_unreliable_live_scan_brakes_and_does_not_renew(self):
        self.assertEqual(self.run_live(reliable=False),'live_motion_uncertain')
        self.assertEqual([c[0] for c in self.calls],['send','stop'])

    def test_obstacle_blocks_all_output(self):
        self.assertEqual(self.run_live(safe=False),'near_obstacle_recheck')
        self.assertEqual(self.calls,[('stop',)])

    def test_late_ack_does_not_arm_renewal_or_send_second_pulse(self):
        self.run_live(delay=.4)
        self.assertEqual([c[0] for c in self.calls],['send','stop'])

    def test_actual_entry_enables_dynamic_not_legacy_policy(self):
        from pathlib import Path
        source=(Path(__file__).parents[1]/'phase1/relationship_find_only_entry.py').read_text('utf-8')
        self.assertIn('install_turn_policy(mission, dynamic_feedback=True)',source)

    def test_ack_client_routes_dynamic_budget_to_actual_output_and_retains_fence(self):
        import numpy as np
        from raspberry_pi.config import SerialConfig
        from grasp_range_integration.phase1.fresh_motion_feedback import (
            FreshScan, install_fresh_motion_feedback)
        class Client:
            def __init__(self):self.config=SerialConfig();self.calls=[]
            def set_bounded_twist(self,*a,**kw):self.calls.append((a,kw));return 'ACK'
            def set_twist(self,*a):raise AssertionError('No renewable turns')
            def stop(self):self.calls.append(('stop',))
            def allow_twist_to_expire(self):pass
            def allow_twist_to_refresh_for(self,*a):raise AssertionError('No renewal')
        class Driver:
            def scans(self):return iter(())
        class Localizer:
            def relative_motion(self,*a):return MatchResult(Pose2D(),.02,100,True,.9)
        mission=NS(N10LidarDriver=Driver,Esp32Client=Client,ScanMapLocalizer=Localizer,
                   _phase1_rotation_assess=lambda frame:dict(safe=True))
        fence=install_fresh_motion_feedback(mission)
        frame=FreshScan(np.zeros(100),np.ones(100),9.9,None,9.8,9.9,1)
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.FreshScanStream',
                   return_value=NS(latest_after=lambda *a:frame)):
            mission.N10LidarDriver().scans()
        mission.ScanMapLocalizer().relative_motion(frame,frame)
        client=mission.Esp32Client()
        turn_policy.TURN_START_REQUEST=dict(adaptive_start=True,mode='adaptive_feedback_turn',
            error_rad=math.radians(23),angular_mrad_s=2600,pulse_speed_mrad_s=3500,
            maximum_ttl_ms=250,stalled_scans=6,actual_stalled_scans=3)
        self.addCleanup(setattr,turn_policy,'TURN_START_REQUEST',None)
        self.addCleanup(setattr,turn_policy,'TELEMETRY_CALLBACK',None)
        actual=[]
        def helper(**kw):
            actual.append(kw)
            kw['send'](kw['pulse_speed_mrad_s'],kw['maximum_ttl_ms'],9.)
            kw['stop']()
        with patch('time.monotonic',return_value=10.), \
             patch('grasp_range_integration.phase1.turn_start_recovery.run_turn_start_recovery',
                   side_effect=helper):
            client.set_twist(0,2600,250)
        self.assertTrue(actual[0]['adaptive_start'])
        self.assertEqual(actual[0]['actual_stalled_scans'],3)
        self.assertEqual(client.calls[0][0],(0,3500,250))
        self.assertLess(client.calls[0][1]['refresh_deadline_s'],10.)
        self.assertIsNotNone(fence.pending)
        with self.assertRaisesRegex(RuntimeError,'故障已锁定'):
            client.set_twist(400,0,650)

    def test_installed_dynamic_policy_escalates_20_actual_quiet_frames_without_old_exit(self):
        class Controller:
            last_command=None
            def observe(self,update):pass
            def stopped(self):self.last_command=None
            def ttl_for(self,command,ttl):return ttl
        mission=NS(AdaptiveSpeedController=Controller,
                   _strict_command_for_pose=lambda *a:(0,2600,False))
        install_turn_policy(mission,dynamic_feedback=True)
        self.addCleanup(setattr,turn_policy,'TURN_START_REQUEST',None)
        controller=Controller()
        pose=NS(x_m=0.,y_m=0.,yaw_rad=math.radians(-23))
        mission._strict_command_for_pose(NS(),pose,(1.,0.))
        command=NS(linear_mm_s=0,angular_mrad_s=2600)
        quiet=NS(accepted=True,rmse_m=.02,inlier_ratio=.9,rotation_rad=0.,translation_m=.001)
        durations=[]
        for _ in range(21):
            controller.last_command=command
            ttl=controller.ttl_for(command,650)
            request=turn_policy.TURN_START_REQUEST
            self.assertTrue(request['adaptive_start'])
            mission._phase1_turn_pulse_sent(dict(angular_mrad_s=request['pulse_speed_mrad_s'],ttl_ms=ttl))
            durations.append(ttl)
            controller.observe(quiet)
        self.assertEqual(durations[-1],250)
        self.assertGreater(durations[2],durations[1])
        self.assertEqual(request['pulse_speed_mrad_s'],3500)


if __name__=='__main__':unittest.main()
