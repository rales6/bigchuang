import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from grasp_range_integration.phase1.motion_experience import (
    AngularProgress, ProgressTimeout, TurnExperience,
)
from grasp_range_integration.phase1 import turn_policy, visibility_route_planner


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        self.progress = AngularProgress(lambda: self.now)
        self.timeout = ProgressTimeout(self.progress)
        self.deadline = 0. + self.timeout
        self.progress.heading("waypoint", math.radians(-80))

    def advance(self, now, rotation, remaining, reliable=True):
        self.now = now
        self.progress.feedback(math.radians(rotation), math.radians(-80), reliable)
        self.progress.heading("waypoint", math.radians(remaining))

    def test_actual_latest_run_turn_progress_does_not_spend_replan(self):
        # 22 scans / 9.6deg physical turn used to expire solely on XY distance.
        for frame in range(1, 23):
            self.advance(frame*2, -9.6/22, -80+frame*9.6/22)
        self.assertFalse(44. >= self.deadline)
        self.assertTrue(90. >= self.deadline)

    def test_real_progress_has_absolute_waypoint_budget(self):
        for second in range(10, 121, 10):
            self.advance(second, -3, -80+second*.3)
        self.assertTrue(120. >= self.deadline)

    def test_no_motion_drift_and_unreliable_motion_do_not_grant_time(self):
        for reliable, rotation in ((True, 0), (False, -5), (True, 5)):
            self.setUp()
            self.advance(30, rotation, -70, reliable)
            self.assertTrue(35. >= self.deadline)

    def test_rotation_without_heading_improvement_does_not_extend(self):
        self.advance(30, -5, -85)
        self.assertTrue(35. >= self.deadline)

    def test_repeated_oscillation_cannot_refresh_same_best_heading(self):
        self.advance(20, -3, -77)
        self.advance(30, 3, -80)
        self.advance(40, -3, -77)
        self.assertTrue(55. >= self.deadline)

    def test_new_waypoint_resets_angular_evidence(self):
        self.advance(30, -3, -77)
        self.progress.heading("next", math.radians(-70))
        self.assertTrue(35. >= self.deadline)

    def test_original_xy_refresh_still_works(self):
        self.deadline = 30. + self.timeout
        self.assertFalse(60. >= self.deadline)
        self.assertTrue(65. >= self.deadline)

    def test_angular_lease_gives_time_to_start_driving_not_indefinite(self):
        self.advance(30, -3, -77)
        self.assertFalse(64. >= self.deadline)
        self.assertTrue(65. >= self.deadline)


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/"runtime"/"motion_experience.json"
        self.now = 1000000.
        self.exp = TurnExperience(self.path, "robot-config", clock=lambda: self.now)
        self.pulse = dict(angular_mrad_s=-3500, ttl_ms=180,
            wheel_output_gains=[1.5, 1.5/1.15, .9*1.5/1.15, 1.5/1.15])

    def learn(self, count=2):
        for _ in range(count):
            self.exp.record(self.pulse, math.radians(-3), .001, True)

    def test_two_successes_survive_restart_and_directions_are_independent(self):
        self.learn(1)
        self.assertIsNone(self.exp.ready(-1))
        self.learn(1)
        loaded = TurnExperience(self.path, "robot-config", clock=lambda: self.now)
        self.assertEqual(loaded.ready(-1)["successes"], 2)
        self.assertIsNone(loaded.ready(1))

    def test_configuration_change_invalidates_previous_robot_profile(self):
        self.learn()
        self.assertIsNone(TurnExperience(self.path, "new-config", clock=lambda:self.now).ready(-1))

    def test_explicit_audited_contract_migration_preserves_fastest_response(self):
        self.learn()
        migrated=TurnExperience(self.path,"new-contract",clock=lambda:self.now,
                               compatible_bindings=("robot-config",))
        self.assertEqual(migrated.ready(-1)["successes"],2)
        self.assertEqual(migrated.binding,"new-contract")

    def test_contract_migration_cannot_accept_out_of_range_pulse(self):
        self.learn()
        data=json.loads(self.path.read_text());data["directions"]["right"]["speed_mrad_s"]=-5000
        self.path.write_text(json.dumps(data))
        self.assertIsNone(TurnExperience(self.path,"new-contract",clock=lambda:self.now,
            compatible_bindings=("robot-config",)).ready(-1))

    def test_expiry_and_new_success_does_not_revive_old_count(self):
        self.learn()
        self.now += 8*86400
        self.assertIsNone(self.exp.ready(-1))
        self.learn(1)
        self.assertIsNone(self.exp.ready(-1))

    def test_wrong_way_zero_unreliable_translation_never_learn(self):
        for rotation, xy, reliable in ((3,0,True),(0,0,True),(-3,0,False),(-3,.04,True),(-30,0,True)):
            self.exp.record(self.pulse, math.radians(rotation), xy, reliable)
        self.assertEqual(self.exp.entries, {})

    def test_unknown_output_or_unproven_duration_never_learn(self):
        for changed in ({"ttl_ms":350}, {"angular_mrad_s":-4000},
                        {"wheel_output_gains":[1,1,1,1]}, {"wheel_output_gains":[math.nan]*4}):
            self.exp.record(dict(self.pulse, **changed), math.radians(-3), 0, True)
        self.assertEqual(self.exp.entries, {})

    def test_corrupt_files_are_ignored_without_blocking_start(self):
        self.path.parent.mkdir()
        for value in ("bad", "[]", '{"schema":1,"binding":"robot-config","directions":{"right":[]}}'):
            self.path.write_text(value)
            self.assertEqual(TurnExperience(self.path,"robot-config",clock=lambda:self.now).entries,{})

    def test_saved_out_of_range_profile_cannot_authorize_motion(self):
        self.learn()
        data = json.loads(self.path.read_text())
        data["directions"]["right"]["ttl_ms"] = 1000
        self.path.write_text(json.dumps(data))
        self.assertIsNone(TurnExperience(self.path,"robot-config",clock=lambda:self.now).ready(-1))

    def test_fastest_response_is_retained(self):
        self.learn()
        self.exp.record(self.pulse, math.radians(-10), 0, True)
        self.exp.record(self.pulse, math.radians(-1), 0, True)
        self.assertAlmostEqual(self.exp.ready(-1)["fastest_rate_rad_s"], math.radians(10)/.18)

    def history(self, valid=True):
        return [dict(self.pulse,type="turn_motor_output_boost",time=self.now-2),
            dict(type="turn_feedback_window",valid=valid,reason="post_pulse_fresh_scan",
                 measured_rotation_deg=-3),
            dict(type="measured_motion",time=self.now-1,reliable=True,rotation_deg=-3,translation_m=.001)]

    def test_audited_history_can_seed_last_proven_response(self):
        self.exp.import_audited_history(self.history()*2)
        self.assertEqual(self.exp.ready(-1)["successes"],2)
        self.assertEqual(self.exp.ready(-1)["updated_s"],self.now-1)

    def test_unverified_historical_feedback_cannot_seed(self):
        self.exp.import_audited_history(self.history(False)*2)
        self.assertEqual(self.exp.entries,{})
        events=self.history()
        events.insert(2,dict(type="turn_start_failed"))
        self.exp.import_audited_history(events*2)
        self.assertEqual(self.exp.entries,{})

    def test_mismatched_history_or_expired_history_cannot_seed(self):
        events=self.history()
        events[1]["measured_rotation_deg"]=-10
        self.exp.import_audited_history(events*2)
        self.assertEqual(self.exp.entries,{})
        events=self.history()*2
        self.now+=8*86400
        self.exp.import_audited_history(events)
        self.assertEqual(self.exp.entries,{})


class PolicyIntegrationTests(unittest.TestCase):
    def setUp(self):
        class Controller:
            def __init__(self): self.last_command = None
            def observe(self, update): pass
            def stopped(self): self.last_command = None
            def ttl_for(self, command, ttl): return ttl
        self.mission = SimpleNamespace(AdaptiveSpeedController=Controller,
            _strict_command_for_pose=lambda *a:(0,-2600,False))
        turn_policy.install_turn_policy(self.mission)
        self.exp = self.mission._phase1_turn_experience = TurnExperience(None, "test")
        self.controller = Controller()
        self.cmd = SimpleNamespace(linear_mm_s=0, angular_mrad_s=-2600)
        self.pose = SimpleNamespace(x_m=0, y_m=0, yaw_rad=math.radians(30))
        self.mission._strict_command_for_pose(object(), self.pose, (1,0))

    def tearDown(self):
        turn_policy.TELEMETRY_CALLBACK = None
        turn_policy.TURN_START_REQUEST = None
        visibility_route_planner.ACTIVE_VIEW = None

    def learn(self, degrees=-3):
        self.controller.last_command = self.cmd
        pulse = dict(angular_mrad_s=-3500,ttl_ms=180,
            wheel_output_gains=[1.5,1.5/1.15,.9*1.5/1.15,1.5/1.15])
        for _ in range(2):
            self.mission._phase1_turn_pulse_sent(pulse)
            self.controller.observe(SimpleNamespace(accepted=True,rmse_m=.02,inlier_ratio=.9,
                rotation_rad=math.radians(degrees),translation_m=.001))

    def test_reuses_success_without_six_failed_short_pulses_and_after_stop(self):
        self.learn()
        self.controller.stopped()
        self.assertEqual(self.controller.ttl_for(self.cmd,650),180)
        self.assertEqual(turn_policy.TURN_START_REQUEST["stalled_scans"],6)
        self.assertEqual(self.controller._phase1_turn_stalled_scans,0)

    def test_near_goal_uses_precision_not_large_learned_start(self):
        self.learn()
        self.pose.yaw_rad=math.radians(6)
        self.mission._strict_command_for_pose(object(),self.pose,(1,0))
        self.assertLessEqual(self.controller.ttl_for(self.cmd,650),70)
        self.assertIsNone(turn_policy.TURN_START_REQUEST)

    def test_fast_learned_response_shrinks_start_pulse(self):
        self.learn(-20)
        self.assertLess(self.controller.ttl_for(self.cmd,650),180)

    def test_changed_terrain_invalidates_seed_after_three_real_stalls(self):
        self.learn()
        quiet=SimpleNamespace(accepted=True,rmse_m=.02,inlier_ratio=.9,
                              rotation_rad=0,translation_m=0)
        for _ in range(3):
            self.controller.ttl_for(self.cmd,650)
            self.controller.observe(quiet)
        self.assertIsNone(self.exp.ready(-1))
        self.assertEqual(self.controller._phase1_turn_stalled_scans,3)

    def test_command_request_without_acknowledged_pulse_cannot_learn(self):
        self.controller.last_command=self.cmd
        for _ in range(10):
            self.controller.ttl_for(self.cmd,650)
            self.controller.observe(SimpleNamespace(accepted=True,rmse_m=.02,inlier_ratio=.9,
                rotation_rad=math.radians(-3),translation_m=.001))
        self.assertEqual(self.exp.entries,{})
