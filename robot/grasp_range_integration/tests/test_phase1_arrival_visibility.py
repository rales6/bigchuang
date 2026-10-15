"""Regressions using the recorded 20260927_171537 final state (no IO)."""
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from grasp_range_integration.phase1.arrival_visibility import (
    add_inspection_gate, arrival_action, linear_arrival_ttl,
)
from grasp_range_integration.phase1.observation_recovery import CandidateRecovery
from grasp_range_integration.phase1 import turn_policy, visibility_route_planner as planner
from raspberry_pi.object_search_mission import _strict_command_for_pose
from raspberry_pi.mapping import AdaptiveSpeedController,AdaptiveSpeedConfig,MotionCommand


class ArrivalTests(unittest.TestCase):
    def setUp(self):
        self.view={"view_xy_m":[-.825,.325],"target_yaw_rad":.40025,
                   "visibility_zone":{"sweep_count":1}}
        self.pose=[-.9247057507101135,.23740941735713383,.587943992051752]
        self.route={"mode":"semantic_visibility_regions","views":[self.view],"maximum_total_views":8}
        self.gate={"eligible":False,"reason":"approaching_task_region",
                   "estimated_work_area_coverage":.2353,"minimum_work_area_coverage":.25}
        self.assess=lambda r,p,view: add_inspection_gate(self.gate,p,view)

    def test_2353_percent_near_station_can_take_one_diagnostic_not_succeed(self):
        action=arrival_action(self.route,self.pose,self.view,self.assess)
        self.assertEqual(action["action"],"diagnostic_capture")
        self.assertFalse(action["assessment"]["eligible"])
        self.assertEqual(arrival_action(self.route,self.pose,self.view,self.assess)["action"],"adjust_on_route")

    def test_initial_seeing_red_far_from_station_does_not_allow_recheck(self):
        for pose,coverage in (([-1.47,-2.86,2.44],0),([-1.266,-.482,1.233],0),
                              ([-1.266,-.482,1.233],.2353)):
            gate=dict(self.gate,estimated_work_area_coverage=coverage)
            self.assertFalse(add_inspection_gate(gate,pose,self.view)["borderline_inspection_allowed"])

    def test_missing_invalid_or_too_low_coverage_never_allows_inspection(self):
        for gate in ({"eligible":False},{**self.gate,"estimated_work_area_coverage":.1},
                     {**self.gate,"reason":"invalid_region_contract"}):
            self.assertFalse(add_inspection_gate(gate,self.pose,self.view)["borderline_inspection_allowed"])

    def test_multiple_sweep_never_skips_angles_for_borderline_view(self):
        self.view["visibility_zone"]["sweep_count"]=2
        self.assertFalse(add_inspection_gate(self.gate,self.pose,self.view)["borderline_inspection_allowed"])
        self.assertEqual(arrival_action(self.route,self.pose,self.view,self.assess)["action"],"heading")

    def test_yaw_only_fix_not_sufficient_selects_existing_route_adjustment(self):
        self.view["arrival_diagnostic_done"]=True
        action=arrival_action(self.route,self.pose,self.view,self.assess)
        self.assertEqual(action["action"],"adjust_on_route")
        self.assertAlmostEqual(action["distance_m"],.132715285,places=7)

    def test_if_heading_will_make_coverage_valid_turn_instead_of_translate(self):
        self.view["arrival_diagnostic_done"]=True
        def assess(r,p,view):
            return {"eligible":True} if p[2]==view["target_yaw_rad"] else self.gate
        self.assertEqual(arrival_action(self.route,self.pose,self.view,assess)["action"],"heading")

    def test_no_unlimited_station_adjustment_if_actual_station_is_unusable(self):
        self.view["arrival_diagnostic_done"]=True
        with self.assertRaisesRegex(RuntimeError,"视野仍不足"):
            arrival_action(self.route,[-.825,.325,.40025],self.view,self.assess)

    def test_linear_adjustment_caps_real_400_start_kick_distance_not_planner_230(self):
        ttl=linear_arrival_ttl(.13291,400,650)
        self.assertEqual(ttl,269)
        self.assertLessEqual(ttl*.4/1000,.13291-.025)
        self.assertLessEqual(linear_arrival_ttl(.3,230,650),350)
        with self.assertRaises(RuntimeError): linear_arrival_ttl(.04,400,650)


class BorderlineCandidateTests(unittest.TestCase):
    setUp = ArrivalTests.setUp
    # Only the added test methods are specific to candidate orchestration.
    def candidate_state(self, confidence=.95):
        det={"camera_id":"left","class_name":"block","confidence":confidence,
             "bbox_1000":[617,54,640,85],"source":"qwen","complete_in_frame":True,
             "task_region_eligible":False,"transit_candidate_allowed":True,
             "dual_camera_confirmed":False,
             "search_evidence":{"source_ok":True,"class_ok":True,"colour_ok":True,"bbox_valid":True}}
        recovery=CandidateRecovery()
        recovery.context={"robot_pose":self.pose,"task_region_view":self.assess(self.route,self.pose,self.view)}
        self.view["arrival_diagnostic_done"]=True
        state={"latest_event":{"type":"observation_complete"},"current_view_index":0,
               "remaining_view_indices":[],"attempts":1,"detections":[det]}
        return recovery,state

    def test_trusted_left_candidate_gets_stationary_recheck_not_task_success(self):
        recovery,state=self.candidate_state()
        action=recovery.next(state,self.route)
        self.assertFalse(action["recovery"]["linear_motion_allowed"])
        self.assertEqual(action["recovery"]["step_deg"],0)
        self.assertEqual(self.view["view_xy_m"],self.pose[:2])
        action=recovery.next(state,self.route)
        self.assertEqual(action["recovery"]["step_deg"],-6)

    def test_candidate_lost_restores_original_station_and_heading(self):
        recovery,state=self.candidate_state()
        original=deepcopy(self.view)
        recovery.next(state,self.route)
        state["detections"]=[]
        directive=recovery.next(state,self.route)
        self.assertEqual(directive["next_view_index"],0)
        self.assertEqual(self.view["view_xy_m"],original["view_xy_m"])
        self.assertEqual(self.view["target_yaw_rad"],original["target_yaw_rad"])
        self.assertEqual(recovery.last_status,"resume_view_coverage_adjustment")

    def test_no_candidate_also_resumes_coverage_with_finite_retry_budget(self):
        recovery,state=self.candidate_state(); state["detections"]=[]
        self.assertIsNotNone(recovery.next(state,self.route))
        self.assertIsNotNone(recovery.next(state,self.route))
        self.assertIsNone(recovery.next(state,self.route))

    def test_low_confidence_borderline_does_not_steer_towards_box(self):
        recovery,state=self.candidate_state(.7)
        directive=recovery.next(state,self.route)
        self.assertNotIn("recovery",directive)
        self.assertEqual(self.view["view_xy_m"],[-.825,.325])

    def test_exhausted_observation_budget_cannot_requeue_forever(self):
        recovery,state=self.candidate_state();state["attempts"]=8
        self.assertIsNone(recovery.next(state,self.route))

    def test_borderline_recovery_cannot_restart_every_time_candidate_reappears(self):
        recovery,state=self.candidate_state()
        recovery.next(state,self.route)
        state["detections"]=[]
        recovery.next(state,self.route)
        state["detections"]=self.candidate_state()[1]["detections"]
        directive=recovery.next(state,self.route)
        self.assertNotIn("recovery",directive)


class InstalledPolicyTests(unittest.TestCase):
    def setUp(self):
        self.old=(planner.ACTIVE_VIEW,planner.LATEST_ROUTE)
        class Controller(AdaptiveSpeedController): pass
        self.mission=SimpleNamespace(AdaptiveSpeedController=Controller,
            _strict_command_for_pose=_strict_command_for_pose)
        turn_policy.install_turn_policy(self.mission)
        self.ctrl=Controller(2600,460,AdaptiveSpeedConfig(initial_turn_floor_mrad_s=2600,
            initial_turn_limit_mrad_s=2600))
        self.args=SimpleNamespace(waypoint_tolerance=.18,heading_tolerance_deg=8,
                                  turn_speed=2600,slow_speed=230,speed=240)

    def tearDown(self):
        planner.ACTIVE_VIEW,planner.LATEST_ROUTE=self.old
        turn_policy.TURN_START_REQUEST=None;turn_policy.TELEMETRY_CALLBACK=None

    def test_actual_1075deg_error_after_two_stalls_uses_low_speed_output_boost(self):
        planner.ACTIVE_VIEW={"view_xy_m":[0,0],"target_yaw_rad":0}
        planner.LATEST_ROUTE={}
        pose=SimpleNamespace(x_m=0,y_m=0,yaw_rad=math.radians(10.75))
        self.mission._strict_command_for_pose(self.args,pose,(0,0))
        quiet=SimpleNamespace(accepted=True,rmse_m=.02,inlier_ratio=.9,rotation_rad=0,
            translation_m=0,angular_speed_rad_s=0,rejection_reason="")
        for _ in range(7):
            command=self.ctrl.apply(MotionCommand(0,-650,"turning","test"))
            self.assertEqual(command.angular_mrad_s,-1200)
            ttl=self.ctrl.ttl_for(command,650)
            self.assertTrue(50 <= ttl <= 70)
            self.ctrl.observe(quiet)
        self.assertEqual(turn_policy.TURN_START_REQUEST["mode"],"precision_output_boost")
        self.assertEqual(turn_policy.TURN_START_REQUEST["angular_mrad_s"],-1200)

    def test_position_tolerance_does_not_finish_unusable_station_after_diagnostic(self):
        planner.ACTIVE_VIEW={"view_xy_m":[-.825,.325],"target_yaw_rad":.40025,
                             "arrival_diagnostic_done":True,"visibility_zone":{"sweep_count":1}}
        planner.LATEST_ROUTE={"mode":"semantic_visibility_regions"}
        pose=SimpleNamespace(x_m=-.92470575,y_m=.23740942,yaw_rad=.58794399)
        gate={"eligible":False,"reason":"approaching_task_region","estimated_work_area_coverage":.2353}
        with patch.object(planner,"task_region_view_assessment",return_value=gate):
            linear,angular,reached=self.mission._strict_command_for_pose(self.args,pose,(-.825,.325))
        self.assertGreater(linear,0); self.assertFalse(reached)
        self.assertAlmostEqual(self.mission._phase1_arrival_distance_m,.132715285,places=7)
