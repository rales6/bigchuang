"""Regression for final 23.318deg view turn in 20260927_155806."""
import math
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from grasp_range_integration.phase1 import turn_policy, visibility_route_planner as planner


class StationHeadingTests(unittest.TestCase):
    def setUp(self):
        self.saved=(planner.ACTIVE_VIEW,planner.LATEST_ROUTE)
        planner.ACTIVE_VIEW={'view_xy_m':[-.825,.325], 'target_yaw_rad':math.radians(22.93166436),
                             'sweep_count':1, 'visibility_zone':{}}
        planner.LATEST_ROUTE={'mode':'semantic_visibility_regions'}
        class Controller:
            def __init__(self):self.last_command=None
            def observe(self,update):pass
            def stopped(self):self.last_command=None
            def ttl_for(self,command,ttl):return ttl
        self.mission=SimpleNamespace(AdaptiveSpeedController=Controller,
                                    _strict_command_for_pose=lambda *a:(0,0,True))
        turn_policy.install_turn_policy(self.mission)
        self.controller=Controller()
        self.args=SimpleNamespace(waypoint_tolerance=.12,heading_tolerance_deg=8,turn_speed=2600)
        self.pose=SimpleNamespace(x_m=-.91975508,y_m=.28509635,yaw_rad=.807231733)
        self.command=SimpleNamespace(linear_mm_s=0,angular_mrad_s=-2600)
        self.quiet=SimpleNamespace(accepted=True,rmse_m=.05965,inlier_ratio=.857,
                                 rotation_rad=math.radians(.00535),translation_m=.0009)

    def tearDown(self):
        planner.ACTIVE_VIEW,planner.LATEST_ROUTE=self.saved
        turn_policy.TURN_START_REQUEST=None
        turn_policy.TELEMETRY_CALLBACK=None

    def prepare(self):
        return self.mission._strict_command_for_pose(self.args,self.pose,(-.825,.325))

    def test_arrival_23deg_uses_180ms_not_precision_70ms_or_8_stall_exit(self):
        command=self.prepare()
        self.assertEqual(command[0],0)
        self.assertFalse(command[2])
        durations=[]
        for _ in range(9):
            self.controller.last_command=self.command
            durations.append(self.controller.ttl_for(self.command,650))
            self.controller.observe(self.quiet)
        self.assertEqual(durations,[70,70,100,100,103,103,180,180,180])
        self.assertTrue(turn_policy.TURN_START_REQUEST['calibrated_start'])
        self.assertNotEqual(turn_policy.TURN_START_REQUEST.get('mode'),'precision_output_boost')

    def test_actual_motion_cancels_start_escalation(self):
        self.prepare();self.controller.ttl_for(self.command,650)
        self.controller.last_command=self.command
        self.controller._phase1_turn_stalled_scans=6
        self.controller._phase1_previous_ttl_ms=180
        moving=SimpleNamespace(accepted=True,rmse_m=.02,inlier_ratio=.9,
                               rotation_rad=math.radians(-2.85),translation_m=.002)
        self.controller.observe(moving)
        self.assertEqual(self.controller._phase1_turn_stalled_scans,0)
        self.assertEqual(self.controller.ttl_for(self.command,650),70)
        self.assertIsNone(turn_policy.TURN_START_REQUEST)

    def test_single_camera_6deg_does_not_inherit_arrival_breakaway(self):
        planner.ACTIVE_VIEW['view_xy_m']=[self.pose.x_m,self.pose.y_m]
        planner.ACTIVE_VIEW['target_yaw_rad']=self.pose.yaw_rad-math.radians(6)
        planner.ACTIVE_VIEW['visibility_zone']={'stop_rule':'single_camera_yaw_alignment'}
        self.mission._strict_command_for_pose(self.args,self.pose,(self.pose.x_m,self.pose.y_m))
        self.controller.ttl_for(self.command,650)
        self.controller._phase1_turn_stalled_scans=6
        self.assertEqual(self.controller.ttl_for(self.command,650),50)
        self.assertEqual(turn_policy.TURN_START_REQUEST['mode'],'precision_output_boost')

    def test_actual_usable_view_can_complete_14deg_trim_then_require_camera(self):
        self.pose.yaw_rad=planner.ACTIVE_VIEW['target_yaw_rad']+math.radians(14)
        with patch.object(planner,'task_region_view_assessment',return_value={'eligible':True}) as assess:
            self.assertEqual(self.prepare(),(0,0,True))
        self.assertEqual(assess.call_args.args[1],[self.pose.x_m,self.pose.y_m,self.pose.yaw_rad])

    def test_missing_coverage_cannot_skip_trim(self):
        self.pose.yaw_rad=planner.ACTIVE_VIEW['target_yaw_rad']+math.radians(14)
        with patch.object(planner,'task_region_view_assessment',return_value={'eligible':False}):
            self.assertFalse(self.prepare()[2])

    def test_multi_direction_sweep_cannot_be_skipped_by_generic_coverage(self):
        self.pose.yaw_rad=planner.ACTIVE_VIEW['target_yaw_rad']+math.radians(14)
        planner.ACTIVE_VIEW['sweep_count']=2
        with patch.object(planner,'task_region_view_assessment',return_value={'eligible':True}) as assess:
            self.assertFalse(self.prepare()[2])
            assess.assert_not_called()

    def test_real_nested_sweep_metadata_cannot_be_skipped(self):
        self.pose.yaw_rad=planner.ACTIVE_VIEW['target_yaw_rad']+math.radians(14)
        planner.ACTIVE_VIEW['visibility_zone']['sweep_count']=2
        with patch.object(planner,'task_region_view_assessment',return_value={'eligible':True}) as assess:
            self.assertFalse(self.prepare()[2])
            assess.assert_not_called()
