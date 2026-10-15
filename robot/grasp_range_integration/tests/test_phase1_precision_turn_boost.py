"""Regression for 20260926_213241: 6-degree align never entered boost."""

import math
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from raspberry_pi.config import SerialConfig
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult
from grasp_range_integration.phase1 import turn_policy, visibility_route_planner as planner
from grasp_range_integration.phase1.fresh_motion_feedback import FreshScan, install_fresh_motion_feedback


class PrecisionTurnBoostTests(unittest.TestCase):
    def tearDown(self):
        turn_policy.TURN_START_REQUEST=None
        turn_policy.TELEMETRY_CALLBACK=None
        turn_policy.LAST_REQUEST=(0,0)
        planner.ACTIVE_VIEW=None

    def install_policy(self):
        class Controller:
            def __init__(self): self.last_command=None
            def observe(self, update): pass
            def stopped(self): self.last_command=None
            def ttl_for(self, command, ttl): return ttl
        mission=NS(AdaptiveSpeedController=Controller,
                   _strict_command_for_pose=lambda *a:(0,0,True))
        turn_policy.install_turn_policy(mission)
        planner.ACTIVE_VIEW={'view_xy_m':[0,0], 'target_yaw_rad':math.radians(-6.91),
                             'visibility_zone':{'stop_rule':'single_camera_yaw_alignment'}}
        mission._strict_command_for_pose(NS(waypoint_tolerance=.12, heading_tolerance_deg=8,
                                            turn_speed=2600), Pose2D(),(0,0))
        return mission, mission.AdaptiveSpeedController()

    def test_six_degree_stalls_reach_all_boost_tiers_not_abort_before_last_tier(self):
        _, controller=self.install_policy()
        command=NS(linear_mm_s=0, angular_mrad_s=-2600)
        feedback=NS(accepted=True, rmse_m=.04, inlier_ratio=.9,
                    rotation_rad=math.radians(-.035), translation_m=.0024)
        stages=[]
        for count in range(8):
            controller.last_command=command
            self.assertEqual(controller.ttl_for(command,650),50)
            request=turn_policy.TURN_START_REQUEST
            if count < 2:
                self.assertIsNone(request)
            else:
                self.assertEqual(request['mode'],'precision_output_boost')
                self.assertEqual(request['angular_mrad_s'],-2600)
                self.assertEqual(request['maximum_ttl_ms'],50)
                stages.append(request['stalled_scans'])
            if count < 7:
                controller.observe(feedback)
            else:
                with self.assertRaisesRegex(RuntimeError,'脉冲上限'):
                    controller.observe(feedback)
        self.assertEqual(stages,[2,3,4,5,6,7])

    def test_movement_or_unreliable_feedback_clears_precision_boost(self):
        for reliable, rotation in ((True,-.9),(False,0)):
            with self.subTest(reliable=reliable):
                _, controller=self.install_policy()
                command=NS(linear_mm_s=0, angular_mrad_s=-2600)
                quiet=NS(accepted=True,rmse_m=.04,inlier_ratio=.9,rotation_rad=0,translation_m=0)
                controller.last_command=command
                controller.ttl_for(command,650)
                controller.observe(quiet);controller.observe(quiet)
                controller.ttl_for(command,650)
                self.assertIsNotNone(turn_policy.TURN_START_REQUEST)
                controller.observe(NS(accepted=reliable,rmse_m=.04,inlier_ratio=.9,
                                      rotation_rad=math.radians(rotation),translation_m=0))
                controller.ttl_for(command,650)
                self.assertIsNone(turn_policy.TURN_START_REQUEST)

    def adapter(self, *, safe=True, fail_ack=False):
        class Client:
            def __init__(self): self.config=SerialConfig();self.calls=[]
            def set_bounded_twist(self,*a,**k):
                self.calls.append(('bounded',a,k,self.config.wheel_output_gains))
                if fail_ack: raise RuntimeError('ACK failed')
                return 'ACK'
            def set_twist(self,*a,**k): self.calls.append(('normal',a))
            def allow_twist_to_expire(self): self.calls.append(('expire',))
            def stop(self): self.calls.append(('stop',))
        class Driver:
            def scans(self): return iter(())
        class Localizer:
            def relative_motion(self,*a): return MatchResult(Pose2D(),.02,100,True,.9)
        mission=NS(Esp32Client=Client,N10LidarDriver=Driver,ScanMapLocalizer=Localizer,
                   _phase1_rotation_assess=lambda s:dict(safe=safe,reason='clear' if safe else 'exterior_near_obstacle'))
        fence=install_fresh_motion_feedback(mission)
        frame=FreshScan(np.linspace(-math.pi,math.pi,100),np.ones(100),9.9,
                        acquisition_start_s=9.8,received_s=9.9,sequence=1)
        stream=NS(latest_after=lambda *a:frame)
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.FreshScanStream',return_value=stream):
            mission.N10LidarDriver().scans()
        localizer=mission.ScanMapLocalizer()
        localizer.relative_motion(frame,frame)
        turn_policy.TURN_START_REQUEST={'mode':'precision_output_boost','angular_mrad_s':-2600,
            'error_rad':math.radians(-6.91),'maximum_ttl_ms':50,'stalled_scans':6}
        return mission,mission.Esp32Client(),fence,localizer,frame

    def test_real_adapter_sends_boosted_short_pulse_and_keeps_post_turn_fence(self):
        mission,client,fence,localizer,frame=self.adapter()
        normal=client.config
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.time.monotonic',return_value=10.), \
             patch('grasp_range_integration.phase1.turn_start_recovery.run_turn_start_recovery',
                   side_effect=AssertionError('6 degrees must not use large-angle start helper')):
            self.assertEqual(client.set_twist(0,-2600,50),'ACK')
        self.assertEqual(client.calls[0],('bounded',(0,-2600,50),{'refresh_deadline_s':10.},(1.5,1.5,1.5,1.5)))
        self.assertEqual(client.calls[-1],('expire',))
        self.assertIs(client.config,normal)
        self.assertIsNotNone(fence.pending)
        self.assertEqual(fence.pending['ttl_ms'],50)
        # Forward is still blocked until actual post-turn feedback is consumed.
        with self.assertRaisesRegex(RuntimeError,'故障已锁定'):
            client.set_twist(400,0,650)

    def test_obstacle_or_ack_failure_stops_without_losing_normal_config(self):
        for safe,fail_ack in ((False,False),(True,True)):
            with self.subTest(safe=safe):
                _,client,fence,_,_=self.adapter(safe=safe,fail_ack=fail_ack)
                normal=client.config
                with self.assertRaises(RuntimeError): client.set_twist(0,-2600,50)
                self.assertIs(client.config,normal)
                self.assertEqual(client.calls[-1],('stop',))
                self.assertIsNone(fence.pending)
                if not safe: self.assertFalse(any(c[0]=='bounded' for c in client.calls))

    def test_caller_cannot_enlarge_precision_ttl(self):
        _,client,fence,_,_=self.adapter()
        with self.assertRaisesRegex(RuntimeError,'短脉冲预算'): client.set_twist(0,-2600,100)
        self.assertEqual(client.calls,[('stop',)])
        self.assertIsNone(fence.pending)

    def test_complete_feedback_consumes_fence_before_normal_forward_output(self):
        _,client,fence,localizer,before=self.adapter()
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.time.monotonic',return_value=10.):
            client.set_twist(0,-2600,50)
        after=FreshScan(np.linspace(-math.pi,math.pi,100),np.ones(100),10.4,
                       acquisition_start_s=10.2,received_s=10.4,sequence=2)
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.time.monotonic',return_value=10.5):
            self.assertTrue(localizer.relative_motion(before,after).converged)
        self.assertIsNone(fence.pending)
        client.set_twist(200,0,650)
        self.assertEqual(client.calls[-1],('normal',(200,0,650)))
        self.assertEqual(client.config.wheel_output_gains,(1.,1.,1.,1.))


if __name__=='__main__': unittest.main()
