"""Regression: 25.45-degree/70ms turn must never become 3 x 350ms."""
import math
from types import SimpleNamespace
import unittest

from grasp_range_integration.phase1.turn_start_recovery import run_turn_start_recovery


class TurnStartTests(unittest.TestCase):
    def run_recovery(self, *, error_deg=25.45, stalled=2, cap=70,
                     safe=True, fail_send=False, ack_delay=0., stale=False, measured_rate=0.):
        self.now=0.; self.calls=[]; self.events=[]
        def read(sequence, timeout):
            if sequence is not None: self.now += .1
            return SimpleNamespace(sequence=int(self.now*10)+1,
                received_s=self.now-(1. if stale else 0.), acquisition_start_s=self.now-.1)
        def send(speed, ttl, deadline):
            self.calls.append(('send', speed, ttl, deadline, self.now))
            self.now += ack_delay
            if fail_send: raise RuntimeError('ACK failed')
        def forbidden(*a):
            raise AssertionError('No internal fit/renewal may authorize escalation')
        return run_turn_start_recovery(error_rad=math.radians(error_deg),
            read_scan=read, measure=forbidden, renew=forbidden,
            assess=lambda _:dict(safe=safe, reason='exterior_near_obstacle'),
            send=send, stop=lambda:self.calls.append(('stop',self.now)),
            emit=lambda kind,**kw:self.events.append((kind,kw)), clock=lambda:self.now,
            maximum_ttl_ms=cap, stalled_scans=stalled, measured_rate_rad_s=measured_rate)

    def test_collision_angle_never_overrides_parent_budget(self):
        self.assertEqual(self.run_recovery(), 'single_pulse_complete')
        sends=[c for c in self.calls if c[0]=='send']
        self.assertEqual(len(sends),1)
        self.assertLessEqual(sends[0][2],70)
        self.assertLess(sends[0][3],sends[0][4])
        self.assertEqual(self.calls[-1][0],'stop')
        self.assertIsNone(self.events[-1][1]['actual_rotation_deg'])
        self.assertTrue(self.events[-1][1]['post_stop_rotation_pending'])

    def test_torque_escalates_only_across_independent_stopped_observations(self):
        speeds=[]
        for stalled in (2,4,6,20):
            self.run_recovery(stalled=stalled)
            sends=[c for c in self.calls if c[0]=='send']
            self.assertEqual(len(sends),1)
            self.assertLessEqual(sends[0][2],70)
            speeds.append(sends[0][1])
        self.assertEqual(speeds,[3000,3250,3500,3500])

    def test_large_turn_still_inherits_shorter_parent_budget(self):
        self.run_recovery(error_deg=90,cap=100,stalled=10)
        self.assertLessEqual(self.calls[0][2],100)
        self.assertLess(abs(self.calls[0][1])/1000.*self.calls[0][2]/1000.,math.radians(20))

    def test_negative_error_preserves_direction(self):
        self.run_recovery(error_deg=-25.45,stalled=4)
        self.assertEqual(self.calls[0][1],-3250)

    def test_delayed_ack_never_arms_refresh(self):
        self.run_recovery(ack_delay=.8)
        self.assertEqual([c[0] for c in self.calls],['send','stop'])
        self.assertLess(self.calls[0][3],0)

    def test_real_obstacle_has_no_command(self):
        self.assertEqual(self.run_recovery(safe=False),'near_obstacle_recheck')
        self.assertEqual([c[0] for c in self.calls],['stop'])

    def test_stale_scan_has_no_command(self):
        with self.assertRaisesRegex(RuntimeError,'过时'):self.run_recovery(stale=True)
        self.assertEqual([c[0] for c in self.calls],['stop'])

    def test_failed_ack_still_stops(self):
        with self.assertRaisesRegex(RuntimeError,'ACK failed'):self.run_recovery(fail_send=True)
        self.assertEqual([c[0] for c in self.calls],['send','stop'])

    def test_precision_error_cannot_enable_start_assist(self):
        with self.assertRaises(ValueError):self.run_recovery(error_deg=10)

    def test_invalid_parent_budget_rejected(self):
        with self.assertRaises(ValueError):self.run_recovery(cap=49)

    def test_latest_23_degree_failure_gets_longer_but_angle_bounded_single_pulse(self):
        # 20260926_203012: -23.317 degrees, 19 stalls, 3000mrad/s,
        # 70ms TTL and post-stop lidar rotation 0.029 degrees.
        self.run_recovery(error_deg=-23.317, stalled=19, cap=650)
        sends=[c for c in self.calls if c[0]=='send']
        self.assertEqual(len(sends),1)
        self.assertEqual(sends[0][1],-3500)
        self.assertEqual(sends[0][2],95)
        self.assertLess(abs(sends[0][1])*sends[0][2]/1e6, math.radians(20))
        self.assertLess(sends[0][3], sends[0][4])
        self.assertEqual(self.calls[-1][0],'stop')

    def test_longer_pulse_never_overrides_a_shorter_parent(self):
        self.run_recovery(error_deg=23, stalled=19, cap=80)
        self.assertEqual(self.calls[0][2],80)

    def test_fast_measured_rate_reduces_start_assist_duration(self):
        self.run_recovery(error_deg=23, stalled=19, cap=650, measured_rate=4.)
        self.assertEqual(self.calls[0][2],65)

    def test_unsafe_minimum_ttl_does_not_send_or_renew(self):
        with self.assertRaisesRegex(RuntimeError,'最小50ms'):
            self.run_recovery(error_deg=23, stalled=19, cap=650, measured_rate=8.)
        self.assertEqual([c[0] for c in self.calls],['stop'])


if __name__=='__main__':unittest.main()
