import math
import unittest
from types import SimpleNamespace
from grasp_range_integration.phase1.turn_calibration import calibrated_start_ttl_ms
from grasp_range_integration.phase1.turn_policy import bounded_turn_ttl_ms
from grasp_range_integration.phase1.turn_start_recovery import run_turn_start_recovery
from grasp_range_integration.phase1.turn_policy import install_turn_policy


class CalibratedStartTests(unittest.TestCase):
    def test_exact_tested_right_pulse_not_recapped_to_99(self):
        self.assertEqual(bounded_turn_ttl_ms(math.radians(-44.319), -3500, 650,
            stalled_scans=6, breakaway=True, calibrated_start=True), 180)

    def test_no_wrong_direction_small_angle_duration_or_speed_extrapolation(self):
        for angle,speed,stalled,parent in [(-44,3500,6,650),(-6,-3500,6,650),
                (-44,-3250,6,650),(-44,-3500,4,650),(-44,-3500,6,100)]:
            self.assertIsNone(calibrated_start_ttl_ms(math.radians(angle),speed,stalled,parent))

    def test_left_uses_its_own_measurement_and_larger_headroom(self):
        self.assertEqual(calibrated_start_ttl_ms(math.radians(44),3500,6,650),180)
        self.assertIsNone(calibrated_start_ttl_ms(math.radians(12),3500,6,650))
        self.assertEqual(calibrated_start_ttl_ms(math.radians(-12),-3500,6,650),180)

    def test_fast_feedback_immediately_shortens_allowance(self):
        self.assertEqual(calibrated_start_ttl_ms(math.radians(-20),-3500,6,650,2.),104)
        self.assertIsNone(calibrated_start_ttl_ms(math.radians(-20),-3500,6,650,8.))

    def test_duration_never_exceeds_measured_180ms(self):
        self.assertEqual(calibrated_start_ttl_ms(math.radians(-90),-3500,20,650,.01),180)

    def test_slow_patch_does_not_erase_fast_patch_feedback(self):
        class Controller:
            def __init__(self):
                self.last_command=SimpleNamespace(linear_mm_s=0,angular_mrad_s=-2600)
                self._phase1_previous_ttl_ms=180
            def observe(self,update): pass
            def ttl_for(self,*args): return 650
            def stopped(self): pass
        mission=SimpleNamespace(AdaptiveSpeedController=Controller,
            _strict_command_for_pose=lambda *args:(0,-2600,False))
        install_turn_policy(mission)
        controller=Controller()
        for angle in (-15,-3):
            controller.observe(SimpleNamespace(accepted=True,rmse_m=.02,inlier_ratio=.9,
                rotation_rad=math.radians(angle),translation_m=.001))
        self.assertAlmostEqual(controller._phase1_measured_rate, math.radians(15)/.180)

    def test_opt_out_keeps_old_budget(self):
        self.assertLessEqual(bounded_turn_ttl_ms(math.radians(-44),-3500,650,
            stalled_scans=6,breakaway=True), 100)

    def test_helper_actually_sends_one_180ms_and_stops_after_late_ack(self):
        state={'now':1., 'calls':[]}
        def send(speed,ttl,deadline):
            state['calls'].append((speed,ttl,deadline)); state['now']+=.202
        result=run_turn_start_recovery(error_rad=math.radians(-44.319),
            read_scan=lambda *a:SimpleNamespace(sequence=1, received_s=state['now']),
            measure=lambda *a: self.fail('internal fit'),
            renew=lambda *a: self.fail('renew'),
            assess=lambda s:{'safe':True}, send=send,
            stop=lambda:state['calls'].append('stop'),emit=lambda *a,**kw:None,
            clock=lambda:state['now'],maximum_ttl_ms=180,stalled_scans=6,
            calibrated_start=True)
        self.assertEqual(result,'single_pulse_complete')
        self.assertEqual(state['calls'][0][:2],(-3500,180))
        self.assertEqual(state['calls'][1],'stop')
        self.assertLess(state['calls'][0][2],1.)

    def test_helper_preserves_dynamic_shortening_and_parent_cap(self):
        calls=[]
        state={'now':1.}
        def send(speed,ttl,deadline):
            calls.append((speed,ttl));state['now']+=.2
        run_turn_start_recovery(error_rad=math.radians(-20),
            read_scan=lambda *a:SimpleNamespace(sequence=1,received_s=state['now']),
            measure=lambda *a:None,renew=lambda *a:self.fail('no renewal'),
            assess=lambda s:{'safe':True}, send=send,stop=lambda:None,
            emit=lambda *a,**kw:None,clock=lambda:state['now'],
            maximum_ttl_ms=104,stalled_scans=6,measured_rate_rad_s=2.,calibrated_start=True)
        self.assertEqual(calls,[(-3500,104)])
