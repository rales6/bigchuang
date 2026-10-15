from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

from grasp_range_integration.phase1.turn_policy import (
    bounded_turn_ttl_ms, install_turn_policy, TurnConvergence,
)


class TurnPolicyTests(unittest.TestCase):
    def test_live_visual_bearing_is_not_overridden_by_old_station_yaw(self):
        from grasp_range_integration.phase1 import visibility_route_planner
        class Controller:
            ttl_for=lambda *a:100
            observe=lambda *a:None
            stopped=lambda *a:None
        mission=SimpleNamespace(AdaptiveSpeedController=Controller,
            _strict_command_for_pose=lambda *a:(200,-100,False))
        install_turn_policy(mission)
        visibility_route_planner.ACTIVE_VIEW={
            'view_xy_m':[.1,0],'target_yaw_rad':math.pi/2,'visibility_zone':{}}
        args=SimpleNamespace(waypoint_tolerance=.18,heading_tolerance_deg=2,
                             turn_speed=2600,_phase1_visual_guidance=True)
        pose=SimpleNamespace(x_m=0.,y_m=0.,yaw_rad=0.)
        self.assertEqual(mission._strict_command_for_pose(args,pose,(.1,0)),(200,-100,False))

    def tearDown(self):
        from grasp_range_integration.phase1 import visibility_route_planner, turn_policy
        visibility_route_planner.ACTIVE_VIEW = None
        turn_policy.LAST_REQUEST = (0, 0)
        turn_policy.TELEMETRY_CALLBACK = None
        turn_policy.TURN_START_REQUEST = None

    def test_pulse_duration_tracks_remaining_angle(self):
        self.assertEqual(bounded_turn_ttl_ms(math.radians(90), 2600, 450), 140)
        self.assertEqual(bounded_turn_ttl_ms(math.radians(20), 2600, 450), 70)
        self.assertEqual(bounded_turn_ttl_ms(math.radians(90), 2600, 100), 100)

    def test_repeated_stall_extends_pulse_but_respects_angle_and_ttl(self):
        self.assertEqual(
            bounded_turn_ttl_ms(math.radians(20), 2600, 500, stalled_scans=1), 70,
        )
        self.assertEqual(
            bounded_turn_ttl_ms(math.radians(20), 2600, 500, stalled_scans=4), 70,
        )
        self.assertLessEqual(
            bounded_turn_ttl_ms(math.radians(6), 2600, 500, stalled_scans=5), 50,
        )
        self.assertEqual(
            bounded_turn_ttl_ms(math.radians(90), 2600, 500, stalled_scans=5), 350,
        )
        self.assertEqual(
            bounded_turn_ttl_ms(math.radians(90), 2600, 100, stalled_scans=5), 100,
        )

    def test_adapter_only_shortens_pure_turn_commands(self):
        class Controller:
            def __init__(self):
                self.last_command = None

            def observe(self, _update):
                pass

            def stopped(self):
                self.last_command = None

            def ttl_for(self, _command, normal_ttl_ms):
                return normal_ttl_ms

        state = {"command": (0, 650, False)}
        mission = SimpleNamespace(
            AdaptiveSpeedController=Controller,
            _strict_command_for_pose=lambda *_args: state["command"],
        )
        install_turn_policy(mission)
        controller = Controller()
        pose = SimpleNamespace(x_m=0.0, y_m=0.0, yaw_rad=0.0)
        target = (0.0, 1.0)
        mission._strict_command_for_pose(object(), pose, target)
        pure_turn = SimpleNamespace(linear_mm_s=0, angular_mrad_s=2600)
        self.assertEqual(controller.ttl_for(pure_turn, 450), 140)
        state["command"] = (200, 0, False)
        mission._strict_command_for_pose(object(), pose, target)
        straight = SimpleNamespace(linear_mm_s=200, angular_mrad_s=0)
        self.assertEqual(controller.ttl_for(straight, 650), 650)

    def test_adapter_escalates_only_after_reliable_pure_turn_stalls(self):
        class Controller:
            def __init__(self):
                self.last_command = None

            def observe(self, _update):
                pass

            def stopped(self):
                self.last_command = None

            def ttl_for(self, _command, normal_ttl_ms):
                return normal_ttl_ms

        mission = SimpleNamespace(
            AdaptiveSpeedController=Controller,
            _strict_command_for_pose=lambda *_args: (0, 2600, False),
        )
        install_turn_policy(mission)
        controller = Controller()
        pose = SimpleNamespace(x_m=0.0, y_m=0.0, yaw_rad=0.0)
        mission._strict_command_for_pose(object(), pose, (0.0, 1.0))
        command = SimpleNamespace(linear_mm_s=0, angular_mrad_s=2600)
        controller.last_command = command
        base = controller.ttl_for(command, 500)
        stalled = SimpleNamespace(
            accepted=True, rmse_m=0.04, inlier_ratio=0.8, rotation_rad=0.0,
        )
        controller.observe(stalled)
        self.assertEqual(controller.ttl_for(command, 500), base)

        controller.observe(stalled)
        # The actual start-assist pulse now includes its speed and 20-degree
        # feedback budget, instead of logging a fictitious 250ms parent.
        assisted = controller.ttl_for(command, 500)
        self.assertGreater(assisted, 70)
        self.assertLessEqual(assisted * 2.6 / 1000, math.radians(20))
        moving = SimpleNamespace(
            accepted=True, rmse_m=0.04, inlier_ratio=0.8,
            rotation_rad=math.radians(3.0),
        )
        controller.observe(moving)
        self.assertEqual(controller.ttl_for(command, 500), base)
        controller.observe(stalled)
        controller.observe(stalled)
        bad_match = SimpleNamespace(
            accepted=False, rmse_m=0.2, inlier_ratio=0.1, rotation_rad=0.0,
        )
        controller.observe(bad_match)
        self.assertEqual(controller.ttl_for(command, 500), base)
        controller.observe(stalled)
        controller.observe(stalled)
        controller.stopped()
        self.assertEqual(controller.ttl_for(command, 500), base)

    def test_overshoot_brakes_and_repeated_oscillation_fails(self):
        monitor = TurnConvergence()
        self.assertFalse(monitor.check("station", math.radians(20), now=0))
        self.assertTrue(monitor.check("station", math.radians(-10), now=1))
        self.assertTrue(monitor.check("station", math.radians(8), now=2))
        with self.assertRaisesRegex(RuntimeError, "超调"):
            monitor.check("station", math.radians(-8), now=3)

    def test_fast_measured_motion_shortens_next_pulse(self):
        self.assertLess(
            bounded_turn_ttl_ms(math.radians(40), 2600, 450, measured_rate_rad_s=8),
            bounded_turn_ttl_ms(math.radians(40), 2600, 450),
        )

    def test_logged_stationary_map_drift_is_not_executed_turn(self):
        monitor = TurnConvergence()
        for index, degrees in enumerate((81.76, 86.98, 92.00, 96.25, 100, 104, 108, 112)):
            self.assertFalse(monitor.check('left_route', math.radians(degrees), now=index, active=False))
        self.assertEqual(monitor.total, 0)
        self.assertEqual(monitor.worsening, 0)
        self.assertFalse(monitor.check('left_route', math.radians(105), now=8, active=True))
        self.assertAlmostEqual(math.degrees(monitor.total), 7)

    def test_hold_pauses_timer_but_does_not_erase_proven_overshoot(self):
        monitor = TurnConvergence()
        monitor.check('turn', math.radians(20), now=0)
        monitor.check('turn', math.radians(-10), now=1)
        monitor.check('turn', math.radians(-15), now=50, active=False)
        self.assertEqual(monitor.crossings, 1)
        self.assertEqual(monitor.active_seconds, 1)
        monitor.check('turn', math.radians(8), now=51)
        with self.assertRaisesRegex(RuntimeError, '超调'):
            monitor.check('turn', math.radians(-8), now=52)

    def test_actual_turn_still_has_timeout_and_wrong_direction_limit(self):
        monitor = TurnConvergence()
        monitor.check('turn', math.radians(90), now=0)
        with self.assertRaises(RuntimeError):
            monitor.check('turn', math.radians(89), now=46)
        monitor = TurnConvergence()
        monitor.check('turn', math.radians(80), now=0)
        monitor.check('turn', math.radians(85), now=1)
        monitor.check('turn', math.radians(90), now=2)
        with self.assertRaises(RuntimeError):
            monitor.check('turn', math.radians(95), now=3)

    def test_installed_policy_still_rejects_actual_wrong_direction_turns(self):
        class Controller:
            def __init__(self): self.last_command = None
            def observe(self, update): pass
            def stopped(self): self.last_command = None
            def ttl_for(self, command, ttl): return ttl
        mission = SimpleNamespace(AdaptiveSpeedController=Controller,
                                  _strict_command_for_pose=lambda *a: (0, 2600, False))
        install_turn_policy(mission)
        controller = Controller()
        pose = lambda degrees: SimpleNamespace(x_m=0., y_m=0., yaw_rad=math.radians(-degrees))
        mission._strict_command_for_pose(object(), pose(80), (1., 0.))
        controller.last_command = SimpleNamespace(linear_mm_s=0, angular_mrad_s=2600)
        feedback = SimpleNamespace(accepted=True, rmse_m=.02, inlier_ratio=.9,
                                   rotation_rad=math.radians(-5), translation_m=0.)
        for degrees in (85, 90):
            controller.observe(feedback)
            mission._strict_command_for_pose(object(), pose(degrees), (1., 0.))
        controller.observe(feedback)
        with self.assertRaisesRegex(RuntimeError, '未收敛'):
            mission._strict_command_for_pose(object(), pose(95), (1., 0.))

    def test_precision_branch_avoids_floor_until_reliable_stall(self):
        from grasp_range_integration.phase1 import visibility_route_planner
        from raspberry_pi.mapping import AdaptiveSpeedController, AdaptiveSpeedConfig, MotionCommand
        class Controller(AdaptiveSpeedController):
            pass
        mission = SimpleNamespace(AdaptiveSpeedController=Controller,
                                  _strict_command_for_pose=lambda *_args: (0, 0, True))
        install_turn_policy(mission)
        controller = Controller(2600, 460, AdaptiveSpeedConfig(
            minimum_turn_floor_mrad_s=2600, initial_turn_floor_mrad_s=2600, initial_turn_limit_mrad_s=2600))
        visibility_route_planner.ACTIVE_VIEW = {
            "view_xy_m": [0, 0], "target_yaw_rad": math.radians(6),
            "visibility_zone": {"stop_rule": "single_camera_yaw_alignment"},
        }
        args = SimpleNamespace(waypoint_tolerance=0.12, heading_tolerance_deg=8, turn_speed=2600)
        linear, angular, reached = mission._strict_command_for_pose(
            args, SimpleNamespace(x_m=0, y_m=0, yaw_rad=0), (0, 0))
        self.assertFalse(reached)
        command = MotionCommand(linear, angular, "turning", "test")
        applied = controller.apply(command)
        self.assertEqual(applied.linear_mm_s, 0)
        self.assertEqual(applied.angular_mrad_s, 1200)
        self.assertGreaterEqual(controller.ttl_for(applied, 650), 50)
        stalled = SimpleNamespace(accepted=True, rmse_m=0.02, inlier_ratio=0.9, rotation_rad=0.0,
                                  translation_m=0.0, rejection_reason="", angular_speed_rad_s=0.0)
        controller.observe(stalled)
        controller.observe(stalled)
        escalated = controller.apply(command)
        self.assertEqual(escalated.angular_mrad_s, 2600)
        self.assertEqual(controller.ttl_for(escalated, 650), 50)

    def test_ttl_obeys_firmware_minimum(self):
        with self.assertRaises(ValueError):
            bounded_turn_ttl_ms(math.radians(6), 2600, 20)

    def test_large_angle_recovery_has_multiple_stages_without_exceeding_angle(self):
        stages = [bounded_turn_ttl_ms(math.radians(90),2600,650,stalled_scans=n)
                  for n in (0,2,4,6,8)]
        self.assertEqual(stages,[140,250,350,450,500])
        # The actual failed run had roughly 50 degrees left: do not blindly
        # apply 500 ms, which could sweep past that angle at nominal speed.
        stages = [bounded_turn_ttl_ms(math.radians(50),2600,650,stalled_scans=n)
                  for n in (0,2,4,6,8)]
        self.assertEqual(stages[:2],[140,250])
        self.assertGreater(stages[-1],250)
        self.assertLessEqual(stages[-1]*2.6/1000,math.radians(48)*.9)
        self.assertEqual(stages[-1],stages[-2])

    def test_large_angle_does_not_stop_at_six_short_stalls(self):
        class Controller:
            def __init__(self): self.last_command=None
            def observe(self,update): pass
            def stopped(self): self.last_command=None
            def ttl_for(self,command,ttl): return ttl
        mission=SimpleNamespace(AdaptiveSpeedController=Controller,
                                _strict_command_for_pose=lambda *a:(0,2600,False))
        install_turn_policy(mission)
        controller=Controller()
        command=SimpleNamespace(linear_mm_s=0,angular_mrad_s=2600)
        mission._strict_command_for_pose(object(),SimpleNamespace(x_m=0,y_m=0,yaw_rad=0),(0,1))
        update=SimpleNamespace(accepted=True,rmse_m=.02,inlier_ratio=.9,rotation_rad=0.,translation_m=0.)
        pulses=[]
        for _ in range(6):
            controller.last_command=command
            pulses.append(controller.ttl_for(command,650))
            controller.observe(update)
        self.assertEqual(pulses,[140,140,116,116,107,107])
        # At least three reliable failures of the final permitted pulse, not
        # three unexecuted requested commands, are needed to end coarse recovery.
        for _ in range(4):
            controller.last_command=command
            controller.ttl_for(command,650)
            controller.observe(update)
        controller.last_command=command
        self.assertEqual(controller.ttl_for(command,650),180)
        controller.observe(update)
        self.assertEqual(controller._phase1_turn_stalled_scans,11)
        from grasp_range_integration.phase1 import turn_policy
        self.assertIsNotNone(turn_policy.TURN_START_REQUEST)
        for _ in range(8):
            controller.ttl_for(command,650)
            controller.observe(update)
        with self.assertRaisesRegex(RuntimeError,'脉冲上限'):
            controller.observe(update)

    def test_zero_clearance_hold_freezes_but_does_not_increase_stall_evidence(self):
        class Controller:
            def __init__(self): self.last_command=None
            def observe(self,update): pass
            def stopped(self): self.last_command=None
            def ttl_for(self,command,ttl): return ttl
        mission=SimpleNamespace(AdaptiveSpeedController=Controller,
                                _strict_command_for_pose=lambda *a:(0,2600,False))
        install_turn_policy(mission)
        controller=Controller()
        command=SimpleNamespace(linear_mm_s=0,angular_mrad_s=2600)
        mission._strict_command_for_pose(object(),SimpleNamespace(x_m=0,y_m=0,yaw_rad=0),(0,1))
        controller.last_command=command
        controller.ttl_for(command,650)
        update=SimpleNamespace(accepted=True,rmse_m=.02,inlier_ratio=.9,rotation_rad=0.,translation_m=0.)
        controller.observe(update); controller.observe(update)
        controller.last_command=SimpleNamespace(linear_mm_s=0,angular_mrad_s=0)
        controller.observe(update)
        self.assertEqual(controller._phase1_turn_stalled_scans,2)
        controller.ttl_for(command,650)
        from grasp_range_integration.phase1 import turn_policy
        self.assertIsNotNone(turn_policy.TURN_START_REQUEST)

    def test_21_degree_navigation_start_uses_recovery_and_not_six_short_failures(self):
        class Controller:
            def __init__(self):self.last_command=None
            def observe(self,update):pass
            def stopped(self):pass
            def ttl_for(self,command,ttl):return ttl
        mission=SimpleNamespace(AdaptiveSpeedController=Controller,
                                _strict_command_for_pose=lambda *a:(0,2600,False))
        install_turn_policy(mission)
        controller=Controller();command=SimpleNamespace(linear_mm_s=0,angular_mrad_s=2600)
        mission._strict_command_for_pose(object(),SimpleNamespace(x_m=0,y_m=0,yaw_rad=math.radians(21)),(1,0))
        update=SimpleNamespace(accepted=True,rmse_m=.02,inlier_ratio=.9,rotation_rad=0.,translation_m=0.)
        for _ in range(6):
            controller.last_command=command
            controller.ttl_for(command,650);controller.observe(update)
        controller.ttl_for(command,650)
        from grasp_range_integration.phase1 import turn_policy
        self.assertIsNotNone(turn_policy.TURN_START_REQUEST)

    def test_recovery_still_shortens_after_fast_motion_and_near_goal(self):
        self.assertEqual(bounded_turn_ttl_ms(math.radians(20),2600,650,stalled_scans=20),70)
        self.assertLess(bounded_turn_ttl_ms(math.radians(50),2600,650,stalled_scans=8,
                                          measured_rate_rad_s=8),140)
        self.assertEqual(bounded_turn_ttl_ms(math.radians(90),2600,100,stalled_scans=8),100)

    def test_recovery_never_translates_to_correct_pose_drift(self):
        from grasp_range_integration.phase1 import visibility_route_planner
        class Controller:
            def observe(self, update): pass
            def stopped(self): pass
            def ttl_for(self, command, ttl): return ttl
        mission = SimpleNamespace(AdaptiveSpeedController=Controller,
                                  _strict_command_for_pose=lambda *_args: (240, 0, False))
        install_turn_policy(mission)
        visibility_route_planner.ACTIVE_VIEW = {
            "view_xy_m": [0, 0], "target_yaw_rad": math.radians(6),
            "visibility_zone": {"stop_rule": "single_camera_yaw_alignment"},
        }
        args = SimpleNamespace(waypoint_tolerance=0.12, heading_tolerance_deg=8, turn_speed=2600)
        command = mission._strict_command_for_pose(args, SimpleNamespace(x_m=0.05, y_m=0, yaw_rad=0), (0.08, 0))
        self.assertEqual(command[0], 0)
        with self.assertRaisesRegex(RuntimeError, "移位超限"):
            mission._strict_command_for_pose(args, SimpleNamespace(x_m=0.2, y_m=0, yaw_rad=0), (0, 0))

    def test_latest_right_turn_stalls_extend_actual_budget_then_motion_resets_it(self):
        from grasp_range_integration.phase1 import turn_policy
        class Controller:
            def __init__(self): self.last_command=None
            def observe(self, update): pass
            def stopped(self): self.last_command=None
            def ttl_for(self, command, ttl): return ttl
        mission=SimpleNamespace(AdaptiveSpeedController=Controller,
                                _strict_command_for_pose=lambda *a:(0,-2600,False))
        install_turn_policy(mission)
        controller=Controller()
        command=SimpleNamespace(linear_mm_s=0, angular_mrad_s=-2600)
        mission._strict_command_for_pose(object(),
            SimpleNamespace(x_m=0, y_m=0, yaw_rad=math.radians(23.317)), (1,0))
        quiet=SimpleNamespace(accepted=True,rmse_m=.02,inlier_ratio=.9,
                              rotation_rad=math.radians(.029),translation_m=.0017)
        durations=[]
        for _ in range(9):
            controller.last_command=command
            durations.append(controller.ttl_for(command,650))
            controller.observe(quiet)
        self.assertEqual(durations,[70,70,100,100,103,103,180,180,180])
        self.assertEqual(turn_policy.TURN_START_REQUEST['maximum_ttl_ms'],180)
        # Actual 0.973-degree turn from that same run is movement, NOT a
        # failed start. Drop the high-torque start request on the next pulse.
        moving=SimpleNamespace(accepted=True,rmse_m=.02,inlier_ratio=.9,
                               rotation_rad=math.radians(-.973),translation_m=.001)
        controller.observe(moving)
        self.assertEqual(controller._phase1_turn_stalled_scans,0)
        self.assertEqual(controller.ttl_for(command,650),70)
        self.assertIsNone(turn_policy.TURN_START_REQUEST)

    def test_breakaway_does_not_extend_fine_visual_alignment(self):
        self.assertEqual(bounded_turn_ttl_ms(math.radians(6),2600,650,
                         stalled_scans=20,breakaway=True),50)
        self.assertEqual(bounded_turn_ttl_ms(math.radians(23),3000,650,
                         stalled_scans=1,breakaway=True),70)


if __name__ == "__main__":
    unittest.main()
