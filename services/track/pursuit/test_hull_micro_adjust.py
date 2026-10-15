"""Pure decision tests; none of these tests can open BLE or move the chassis."""
from __future__ import annotations

import tempfile
import threading
import unittest
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pursuit.hull_micro_adjust import (
    PULSE_FIRST_S, PULSE_MAX_S, ScanGuard, align, choose_adjustment,
    next_pulse_duration, adjustment_policy, SweptRetreatBudget, TRIAL_FIRST_S,
    select_adjustment, measured_response, AdjustmentPlan,
)


class HullMicroAdjustTests(unittest.TestCase):
    def test_inside_hull_requires_no_motion(self):
        observation = {
            "ok": True, "inside_teaching_hull": True,
            "outside_distance_px": 0.0,
            "centers": {"left": [380, 360], "right": [240, 280]},
        }
        self.assertIsNone(choose_adjustment(observation))
        with tempfile.TemporaryDirectory() as directory, \
             patch("pursuit.hull_micro_adjust._observe", return_value=observation):
            result = align("红色方块", real=False, log_dir=Path(directory))
        self.assertTrue(result.ok)
        self.assertEqual(result.pulses, 0)

    def test_large_hull_gap_never_moves(self):
        observation = {
            "ok": True, "inside_teaching_hull": False,
            "outside_distance_px": 240.4,
            "centers": {"left": [336, 242], "right": [414, 271]},
            "nearest_hull_centers": {
                "left": [415, 317], "right": [345, 241],
            },
        }
        with tempfile.TemporaryDirectory() as directory, \
             patch("pursuit.hull_micro_adjust._observe", return_value=observation), \
             patch("pursuit.hull_micro_adjust.verify_model_camera_order"), \
             patch("pursuit.hull_micro_adjust._pulse") as pulse:
            result = align("红色方块", real=True, log_dir=Path(directory))
        self.assertFalse(result.ok)
        self.assertIn("超出近距离微调上限", result.reason)
        pulse.assert_not_called()

    @staticmethod
    def observation(distance=180.4, offset=0.0):
        return {
            "ok": True, "inside_teaching_hull": False,
            "outside_distance_px": distance,
            "centers": {"left": [283.2 + offset, 283.0],
                        "right": [417.0 - offset, 316.8]},
            "nearest_hull_centers": {
                "left": [392.0, 330.4], "right": [315.7, 256.6]},
        }

    def simulate(self, observations, **kwargs):
        # Every hardware and network entry point is mocked. Even real=True
        # here cannot open a camera, serial port, HTTP connection or BLE.
        with tempfile.TemporaryDirectory() as directory, \
             patch("pursuit.hull_micro_adjust._observe", side_effect=observations), \
             patch("pursuit.hull_micro_adjust.verify_model_camera_order"), \
             patch("pursuit.hull_micro_adjust._ready"), \
             patch("pursuit.hull_micro_adjust.REAR_SECTOR_USABLE", True), \
             patch("pursuit.hull_micro_adjust.ScanGuard"), \
             patch("pursuit.hull_micro_adjust._pulse", return_value=0.8) as pulse:
            result = align("红色方块", real=True, log_dir=Path(directory), **kwargs)
            events = [json.loads(line) for line in
                      Path(result.log_path).read_text(encoding="utf-8").splitlines()]
            return result, pulse.call_args_list, events

    def test_policy_separates_trial_band_and_close_band(self):
        self.assertEqual(adjustment_policy(180.4), ("boundary_trial", 0.8, 0.8))
        self.assertEqual(adjustment_policy(180.0)[0], "normal")
        self.assertEqual(adjustment_policy(240.0)[0], "boundary_trial")
        self.assertEqual(adjustment_policy(42.0), ("close", 0.8, 0.8))
        with self.assertRaisesRegex(RuntimeError, "超出近距离"):
            adjustment_policy(240.01)
        with self.assertRaisesRegex(RuntimeError, "超出近距离"):
            adjustment_policy(180.4, trial_band_px=0.0)
        for invalid in (float("nan"), float("inf"), -1.0, 0.0):
            with self.assertRaisesRegex(RuntimeError, "无效"):
                adjustment_policy(invalid)

    def test_recorded_180_point_4_gap_gets_small_retreat_not_arm_success(self):
        outside = self.observation()
        improved = self.observation(168.0, offset=10.0)
        result, calls, events = self.simulate(
            [outside, outside, improved], max_pulses=1)
        self.assertFalse(result.ok)  # trial permission is NOT grasp permission
        self.assertEqual(result.pulses, 1)
        self.assertEqual(calls[0].args[2:4], ("backward", 0.8))
        self.assertEqual(result.distance_px, 168.0)
        planned = next(event for event in events if event["event"] == "planned_pulse")
        self.assertEqual(planned["stage"], "boundary_trial")

    def test_trial_can_complete_only_after_inside_verdict(self):
        outside = self.observation()
        inside = {**self.observation(0.0, 10.0), "inside_teaching_hull": True}
        result, calls, _ = self.simulate([outside, outside, inside])
        self.assertTrue(result.ok)
        self.assertEqual(len(calls), 1)

    def test_visible_motion_without_improvement_stops_after_one_trial(self):
        outside = self.observation()
        result, calls, _ = self.simulate(
            [outside, outside, self.observation(180.0, 10.0)])
        self.assertFalse(result.ok)
        self.assertIn("未改善", result.reason)
        self.assertEqual(len(calls), 1)

    def test_worsening_distance_stops_even_with_small_center_motion(self):
        outside = self.observation()
        result, calls, events = self.simulate(
            [outside, outside, self.observation(184.0, 1.0)])
        self.assertFalse(result.ok)
        self.assertIn("未改善", result.reason)
        self.assertEqual(len(calls), 1)
        self.assertEqual(events[-1]["distance_px"], 184.0)

    def test_stationary_trial_never_escalates_to_far_stage_long_pulse(self):
        outside = self.observation()
        result, calls, _ = self.simulate([outside] * 7)
        self.assertFalse(result.ok)
        self.assertEqual([call.args[3] for call in calls], [0.8])
        self.assertEqual({call.args[2] for call in calls}, {"backward"})

    def test_total_time_budget_prevents_unbounded_stationary_attempts(self):
        outside = self.observation()
        # Each accepted step visibly approaches the hull. Even an explicitly
        # larger pulse count must not remove the cumulative on-time ceiling.
        followups = [self.observation(170 - 10 * i, offset=10 * (i + 1)) for i in range(8)]
        result, calls, _ = self.simulate([outside, outside, *followups], max_pulses=10)
        self.assertFalse(result.ok)
        self.assertLessEqual(sum(call.args[3] for call in calls), 3.0 + 1e-9)
        self.assertIn("时间预算", result.reason)

    def test_near_hull_reuses_verified_approach_start_window(self):
        outside = self.observation(42.0)
        result, calls, _ = self.simulate(
            [outside, outside, self.observation(30.0, 10.0)], max_pulses=1)
        self.assertFalse(result.ok)
        self.assertEqual(calls[0].args[3], 0.8)

    def test_disabled_trial_band_preserves_strict_entry_limit(self):
        outside = self.observation()
        result, calls, _ = self.simulate([outside], trial_band_px=0.0)
        self.assertFalse(result.ok)
        self.assertEqual(calls, [])

    def test_invalid_distance_never_moves(self):
        for invalid in (float("nan"), float("inf"), -1.0, 0.0):
            with self.subTest(distance=invalid):
                result, calls, _ = self.simulate([self.observation(invalid)])
                self.assertFalse(result.ok)
                self.assertEqual(calls, [])

    def test_post_pulse_lost_target_stops_without_retry(self):
        outside = self.observation()
        result, calls, _ = self.simulate(
            [outside, outside, RuntimeError("双目目标中心无效")])
        self.assertFalse(result.ok)
        self.assertEqual(len(calls), 1)
        self.assertIn("双目目标中心无效", result.reason)

    def test_opposite_measured_disparity_stops_without_retry(self):
        outside = self.observation()
        result, calls, _ = self.simulate(
            [outside, outside, self.observation(160.0, -10.0)])
        self.assertFalse(result.ok)
        self.assertEqual(len(calls), 1)
        self.assertIn("视差变化", result.reason)

    def test_visible_improvement_does_not_lengthen_the_next_pulse(self):
        outside = self.observation(170.0)
        improved = self.observation(160.0, 10.0)
        inside = {**self.observation(0.0, 20.0), "inside_teaching_hull": True}
        result, calls, _ = self.simulate([outside, outside, improved, inside])
        self.assertTrue(result.ok)
        self.assertEqual([call.args[3] for call in calls], [0.8, 0.8])

    def test_visual_feedback_shortens_next_step_but_keeps_startup_floor(self):
        outside = self.observation(170.0)
        improved = self.observation(80.0, 50.0)
        inside = {**self.observation(0.0, 60.0), "inside_teaching_hull": True}
        result, calls, _ = self.simulate([outside, outside, improved, inside])
        self.assertTrue(result.ok)
        self.assertEqual([call.args[3] for call in calls], [0.8, 0.68])

    def test_shared_ble_payload_uses_approach_speed_and_no_odometry(self):
        guard = SimpleNamespace(clearance=lambda *args, **kwargs: 1.0)
        with patch('pursuit.hull_micro_adjust.TOKEN_PATH') as token, \
             patch('pursuit.hull_micro_adjust._post_json', return_value={'ok': True}) as post, \
             patch('pursuit.hull_micro_adjust.time.sleep'):
            from pursuit.hull_micro_adjust import _pulse
            token.read_text.return_value = 'test-only'
            _pulse('http://unit.test', guard, 'forward', 0.8)
        payload = post.call_args_list[0].args[1]
        self.assertEqual(payload['speed_mm_s'], 405)
        self.assertEqual(payload['duration_s'], 0.8)
        self.assertNotIn('distance_m', payload)

    def test_trial_does_not_bypass_front_obstacle_guard(self):
        outside = self.observation()
        with tempfile.TemporaryDirectory() as directory, \
             patch("pursuit.hull_micro_adjust._observe", return_value=outside), \
             patch("pursuit.hull_micro_adjust.verify_model_camera_order"), \
             patch("pursuit.hull_micro_adjust._ready"), \
             patch("pursuit.hull_micro_adjust.REAR_SECTOR_USABLE", True), \
             patch("pursuit.hull_micro_adjust.TOKEN_PATH") as token, \
             patch("pursuit.hull_micro_adjust.ScanGuard") as guard, \
             patch("pursuit.hull_micro_adjust._post_json") as post:
            token.read_text.return_value = "test-only"
            guard.return_value.clearance.side_effect = RuntimeError("forward 前方障碍")
            result = align("红色方块", real=True, log_dir=Path(directory))
        self.assertFalse(result.ok)
        self.assertEqual(result.pulses, 0)
        self.assertIn("前方障碍", result.reason)
        post.assert_not_called()

    def test_forward_direction_uses_both_eyes(self):
        observation = {
            "inside_teaching_hull": False,
            "centers": {"left": [380, 270], "right": [300, 240]},
            "nearest_hull_centers": {
                "left": [385, 320], "right": [295, 260],
            },
        }
        self.assertEqual(choose_adjustment(observation), (405, 0, "forward"))

    def test_toe_in_opposite_x_errors_mean_axial_retreat(self):
        observation = {
            "inside_teaching_hull": False,
            "centers": {"left": [368, 277], "right": [487, 300]},
            "nearest_hull_centers": {
                "left": [464, 321], "right": [388, 254],
            },
        }
        self.assertEqual(choose_adjustment(observation), (-405, 0, "backward"))

    def test_toe_in_opposite_x_errors_can_mean_forward(self):
        observation = {
            "inside_teaching_hull": False,
            "centers": {"left": [480, 270], "right": [340, 260]},
            "nearest_hull_centers": {
                "left": [390, 320], "right": [430, 305]},
        }
        self.assertEqual(choose_adjustment(observation), (405, 0, "forward"))

    def test_vertical_errors_are_not_overridden_by_disparity_alone(self):
        observation = {
            "inside_teaching_hull": False,
            "centers": {"left": [368, 250], "right": [487, 260]},
            "nearest_hull_centers": {
                "left": [464, 300], "right": [388, 310],
            },
        }
        self.assertEqual(choose_adjustment(observation), (405, 0, "forward"))

    @staticmethod
    def recorded_trial():
        before = {
            "ok": True, "inside_teaching_hull": False, "outside_distance_px": 138.2,
            "centers": {"left": [460.8, 260.4], "right": [515.5, 277.7]},
            "nearest_hull_centers": {
                "left": [512.7, 329.5], "right": [426.6, 273.6]},
        }
        after = {
            "ok": True, "inside_teaching_hull": False, "outside_distance_px": 149.8,
            "centers": {"left": [457.6, 225.4], "right": [467.2, 245.3]},
            "nearest_hull_centers": {
                "left": [474.3, 322.6], "right": [396.0, 258.1]},
        }
        return before, after

    def test_recorded_case_no_longer_blindly_retreats(self):
        before, after = self.recorded_trial()
        self.assertEqual(select_adjustment(before).action, "turn_right")
        axis, response = measured_response('backward', 0.8,
            (460.8, 260.4, 515.5, 277.7), (457.6, 225.4, 467.2, 245.3))
        plan = select_adjustment(after, responses={axis: response},
                                 blocked_actions={'backward'})
        self.assertEqual(plan.action, 'forward')
        self.assertEqual(plan.duration_s, 0.68)
        self.assertGreaterEqual(plan.predicted_improvement_px, 2.0)
        # A preceding 0.4s yaw must not make the learned linear action fail
        # its 0.68s startup floor when changing axes.
        next_plan = select_adjustment(after, responses={axis: response},
                                      blocked_actions={'backward'}, duration_s=0.4)
        self.assertEqual(next_plan.action, 'forward')
        self.assertEqual(next_plan.duration_s, 0.68)

    def simulate_recorded_wrong_trial(self, observations):
        # Reproduce OLD code's first wrong action, then let the NEW selector
        # learn from real recorded centres. All hardware/network stays mocked.
        original = select_adjustment
        count = 0
        def first_old_action(observation, **kwargs):
            nonlocal count
            count += 1
            if count == 1:
                return AdjustmentPlan('backward', 0.8, 'test_old_decision')
            return original(observation, **kwargs)
        with patch('pursuit.hull_micro_adjust.select_adjustment', side_effect=first_old_action):
            return self.simulate(observations)

    def test_bad_trial_rechecks_corrects_then_continues_to_hull(self):
        before, after = self.recorded_trial()
        # Final inside verdict is synthetic: this is control-flow testing,
        # NOT a claim that replaying the real test would complete a grasp.
        inside = {**before, 'inside_teaching_hull': True, 'outside_distance_px': 0.0}
        result, calls, events = self.simulate_recorded_wrong_trial(
            [before, before, after, after, before, inside])
        self.assertTrue(result.ok)
        self.assertEqual([call.args[2] for call in calls],
                         ['backward', 'forward', 'turn_right'])
        self.assertEqual([call.args[3] for call in calls], [0.8, 0.68, 0.4])
        self.assertEqual(sum(event['event'] == 'corrective_replan' for event in events), 1)
        self.assertIn('measured_4d_response', [event.get('decision_source') for event in events])

    def test_second_bad_pulse_stops_instead_of_oscillating(self):
        before, after = self.recorded_trial()
        worse = {**after, 'outside_distance_px': 154.0,
                 'centers': {'left': [460.0, 235.4], 'right': [477.2, 255.3]}}
        result, calls, _ = self.simulate_recorded_wrong_trial(
            [before, before, after, after, worse])
        self.assertFalse(result.ok)
        self.assertEqual(len(calls), 2)
        self.assertIn('未改善', result.reason)

    def test_corrective_recheck_blocks_moving_target(self):
        before, after = self.recorded_trial()
        moved = {**after, 'centers': {'left': [497.6, 225.4], 'right': [507.2, 245.3]}}
        result, calls, _ = self.simulate_recorded_wrong_trial(
            [before, before, after, moved])
        self.assertFalse(result.ok)
        self.assertEqual(len(calls), 1)
        self.assertIn('不稳定', result.reason)

    def test_large_worsening_does_not_get_corrective_motion(self):
        before, after = self.recorded_trial()
        result, calls, _ = self.simulate_recorded_wrong_trial(
            [before, before, {**after, 'outside_distance_px': 170.0}])
        self.assertFalse(result.ok)
        self.assertEqual(len(calls), 1)

    def test_measured_axis_does_not_fall_back_to_guess_when_floor_overshoots(self):
        observation = {
            'inside_teaching_hull': False,
            'centers': {'left': [380, 270], 'right': [300, 240]},
            'nearest_hull_centers': {'left': [379, 272], 'right': [301, 242]},
        }
        self.assertIsNone(select_adjustment(observation,
            responses={'axial': (-20, 100, 20, 100)}))

    def test_recorded_improving_retreat_can_continue_with_ramp_aware_small_step(self):
        # 2026-09-26 12:29: 197.4 -> 161.4px after 0.8s retreat.
        before=(230.1,278.4,380.5,322.3)
        after=(260.5,242.4,340.5,284.4)
        observation={
            'inside_teaching_hull':False,
            'centers':{'left':after[:2],'right':after[2:]},
            'nearest_hull_centers':{'left':[339.5,322.1],'right':[273.0,237.1]},
        }
        axis,response=measured_response('backward',.8,before,after)
        plan=select_adjustment(observation,responses={axis:response})
        self.assertIsNotNone(plan)
        self.assertEqual(plan.action,'backward')
        self.assertEqual(plan.duration_s,.68)
        self.assertGreaterEqual(plan.predicted_improvement_px,2)

    def test_response_samples_are_not_shared_between_sessions(self):
        before, _ = self.recorded_trial()
        select_adjustment(before, responses={'yaw': (100, 0, 100, 0)})
        self.assertEqual(select_adjustment(before).source, 'nominal_joint_4d_trial')

    def test_same_side_lateral_error_uses_in_place_turn(self):
        observation = {
            "inside_teaching_hull": False,
            "centers": {"left": [300, 320], "right": [200, 280]},
            "nearest_hull_centers": {
                "left": [340, 320], "right": [235, 280],
            },
        }
        self.assertEqual(choose_adjustment(observation), (0, 450, "turn_left"))

    def test_rear_obstacle_blocks_toe_in_retreat(self):
        guard = ScanGuard.__new__(ScanGuard)
        guard.latest = object()
        guard.updated_at = 100.0
        guard.error = ""
        with patch("pursuit.hull_micro_adjust.time.monotonic", return_value=100.1), \
             patch("pursuit.hull_micro_adjust.REAR_SECTOR_USABLE", True), \
             patch("pursuit.hull_micro_adjust.sector_min_distance",
                   return_value=0.08):
            with self.assertRaisesRegex(RuntimeError, "backward.*障碍"):
                guard.clearance("backward")

    def test_turn_checks_all_four_lidar_sides(self):
        guard = ScanGuard.__new__(ScanGuard)
        guard.latest = object()
        guard.updated_at = 100.0
        guard.error = ""
        with patch("pursuit.hull_micro_adjust.time.monotonic", return_value=100.1), \
             patch("pursuit.hull_micro_adjust.REAR_SECTOR_USABLE", True), \
             patch("pursuit.hull_micro_adjust.sector_min_distance",
                   side_effect=[1.0, 1.0, 0.08, 1.0]) as sector:
            with self.assertRaisesRegex(RuntimeError, "turn_left.*180°.*障碍"):
                guard.clearance("turn_left")
        self.assertEqual(sector.call_count, 3)

    def test_occluded_rear_is_not_sampled_for_budgeted_retreat(self):
        guard = ScanGuard.__new__(ScanGuard)
        guard.latest, guard.updated_at, guard.error = object(), 100.0, ""
        with patch("pursuit.hull_micro_adjust.time.monotonic", return_value=100.1), \
             patch("pursuit.hull_micro_adjust.sector_min_distance", return_value=1.0) as sector:
            self.assertEqual(guard.clearance("backward", front_only_retreat=True), 1.0)
        self.assertEqual([call.kwargs['center_deg'] for call in sector.call_args_list], [0.0])

    def test_pursuit_does_not_read_chassis_occluded_rear_sector(self):
        from pursuit.pursuit_controller import PursuitController
        from pursuit.pursuit_config import PursuitConfig
        controller = PursuitController.__new__(PursuitController)
        controller.cfg = PursuitConfig()
        with patch('pursuit.pursuit_controller.sector_min_distance') as sector:
            self.assertIsNone(controller._lidar_rear(object()))
        sector.assert_not_called()

    def test_occluded_rear_never_grants_unbounded_reverse(self):
        guard = ScanGuard.__new__(ScanGuard)
        guard.latest, guard.updated_at, guard.error = object(), 100.0, ""
        with patch("pursuit.hull_micro_adjust.time.monotonic", return_value=100.1), \
             patch("pursuit.hull_micro_adjust.sector_min_distance") as sector:
            with self.assertRaisesRegex(RuntimeError, "缺少原路短退预算"):
                guard.clearance("backward")
        sector.assert_not_called()

    def test_turn_only_samples_observable_front_fan(self):
        guard = ScanGuard.__new__(ScanGuard)
        guard.latest, guard.updated_at, guard.error = object(), 100.0, ""
        with patch("pursuit.hull_micro_adjust.time.monotonic", return_value=100.1), \
             patch("pursuit.hull_micro_adjust.sector_min_distance", return_value=1.0) as sector:
            guard.clearance("turn_left")
        self.assertEqual([call.kwargs['center_deg'] for call in sector.call_args_list],
                         [0.0, 45.0, 315.0])

    def test_retreat_budget_is_bounded_on_time_not_fake_metric_displacement(self):
        budget = SweptRetreatBudget(time.monotonic() + 90, "test-run")
        self.assertEqual(budget.reserve(0.8), (0.8, 0.8))
        self.assertEqual(budget.reserve(0.8), (0.8, 1.6))
        with self.assertRaisesRegex(RuntimeError, "预算已用尽"):
            budget.reserve(0.8)

    def test_expired_or_invalidated_retreat_evidence_never_moves(self):
        for budget in (SweptRetreatBudget(time.monotonic() - 1, "expired"),
                       SweptRetreatBudget(time.monotonic() + 90, "turned", invalidated=True)):
            with self.assertRaisesRegex(RuntimeError, "凭据过期或预算"):
                budget.reserve(0.8)

    def test_hull_handoff_uses_short_budgeted_retreat(self):
        outside = self.observation(148.0)
        inside = {**self.observation(0.0, 10.0), "inside_teaching_hull": True}
        result, calls, events = self.simulate([outside, outside, inside],
            retreat_budget=SweptRetreatBudget(time.monotonic() + 90, "matching-pursuit"))
        self.assertTrue(result.ok)
        self.assertEqual(calls[0].args[2:4], ("backward", 0.8))
        self.assertTrue(calls[0].kwargs['swept_retreat'])
        reserved = next(row for row in events if row['event'] == 'swept_retreat_reserved')
        self.assertFalse(reserved['rear_sector_checked'])

    def test_pulse_time_increases_only_when_stationary_and_is_bounded(self):
        self.assertEqual(next_pulse_duration(PULSE_FIRST_S, 0), 0.8)
        self.assertEqual(next_pulse_duration(PULSE_MAX_S, 0), PULSE_MAX_S)
        self.assertEqual(next_pulse_duration(0.70, 20,
                         minimum_s=TRIAL_FIRST_S), 0.68)

    def test_real_alignment_rechecks_then_stops_after_hull_entry(self):
        outside = {
            "ok": True, "inside_teaching_hull": False,
            "outside_distance_px": 42.0,
            "centers": {"left": [380, 275], "right": [300, 245]},
            "nearest_hull_centers": {
                "left": [382, 310], "right": [300, 270],
            },
        }
        inside = {
            "ok": True, "inside_teaching_hull": True,
            "outside_distance_px": 0.0,
            "centers": {"left": [382, 310], "right": [300, 270]},
        }
        with tempfile.TemporaryDirectory() as directory, \
             patch("pursuit.hull_micro_adjust._observe",
                   side_effect=[outside, outside, inside]), \
             patch("pursuit.hull_micro_adjust.verify_model_camera_order"), \
             patch("pursuit.hull_micro_adjust._ready"), \
             patch("pursuit.hull_micro_adjust.REAR_SECTOR_USABLE", True), \
             patch("pursuit.hull_micro_adjust.ScanGuard") as guard, \
             patch("pursuit.hull_micro_adjust._pulse", return_value=0.8) as pulse:
            result = align("红色方块", real=True, log_dir=Path(directory))
        self.assertTrue(result.ok)
        self.assertEqual(result.pulses, 1)
        pulse.assert_called_once()
        guard.return_value.close.assert_called_once()

    def test_moving_target_blocks_real_pulse(self):
        outside = {
            "ok": True, "inside_teaching_hull": False,
            "outside_distance_px": 42.0,
            "centers": {"left": [380, 275], "right": [300, 245]},
            "nearest_hull_centers": {
                "left": [382, 310], "right": [300, 270]},
        }
        moved = {**outside, "centers": {
            "left": [405, 275], "right": [325, 245],
        }}
        with tempfile.TemporaryDirectory() as directory, \
             patch("pursuit.hull_micro_adjust._observe",
                   side_effect=[outside, moved]), \
             patch("pursuit.hull_micro_adjust.verify_model_camera_order"), \
             patch("pursuit.hull_micro_adjust._pulse") as pulse:
            result = align("红色方块", real=True, log_dir=Path(directory))
        self.assertFalse(result.ok)
        self.assertIn("目标在移动", result.reason)
        pulse.assert_not_called()

    def test_emergency_cancel_prevents_any_observation_or_motion(self):
        cancelled = threading.Event()
        cancelled.set()
        with tempfile.TemporaryDirectory() as directory, \
             patch("pursuit.hull_micro_adjust._observe") as observe, \
             patch("pursuit.hull_micro_adjust._pulse") as pulse:
            result = align("红色方块", real=True, log_dir=Path(directory),
                           cancel_event=cancelled)
        self.assertFalse(result.ok)
        self.assertIn("急停", result.reason)
        observe.assert_not_called()
        pulse.assert_not_called()


if __name__ == "__main__":
    unittest.main()
