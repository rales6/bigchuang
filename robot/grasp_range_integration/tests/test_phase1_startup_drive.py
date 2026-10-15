"""Replay breakaway timing without motor/sensor access."""
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from raspberry_pi.mapping import Pose2D
from grasp_range_integration.phase1.startup_drive import run_startup_drive
from grasp_range_integration.phase1.startup_drive_experience import StartupDriveExperience


class ContinuousStartTests(unittest.TestCase):
    def setUp(self):
        self.now=0.
        self.commands=[]
        self.events=[]
        self.seq=0
        self.cancelled=False
        self.check=dict(safe=True)
        self.reliable=True
        self.late=False

    def run_drive(self, *, progress_at=1.45, **kw):
        def start(): self.commands.append(('start',self.now));self.now+=.06
        def stop(): self.commands.append(('stop',self.now))
        def renew(*a): self.commands.append(('renew',self.now));return True
        def read(sequence,timeout):
            self.now+=.08 if not self.late else .45
            self.seq+=1
            return NS(sequence=self.seq,acquisition_start_s=self.now-.07,received_s=self.now)
        def measure(*a):
            # In 203859 motion was just jitter for ~1.4s at 400mm/s,
            # then 4.5cm followed by 9cm. Short STOPped pulses never got here.
            x=.11 if self.now>=progress_at+.12 else .045 if self.now>=progress_at else .001
            return NS(transform=Pose2D(x,0,0),reliable=self.reliable,raw_range=dict(accepted=True))
        args=dict(ttl_ms=400,budget_ms=1800,start=start,renew=renew,read_scan=read,
                  measure=measure,assess=lambda *a:self.check,stop=stop,
                  cancelled=lambda:self.cancelled,clock=lambda:self.now,
                  emit=lambda name,**e:self.events.append((name,e)))
        args.update(kw)
        return run_startup_drive(NS(sequence=0),**args)

    def test_203859_style_delayed_breakaway_stays_continuous_past_1_4_seconds(self):
        result=self.run_drive()
        self.assertEqual(result['reason'],'measured_local_step_complete')
        self.assertGreater(result['breakaway_s'],1.4)
        self.assertGreater(result['renewals'],10)
        self.assertEqual([x[0] for x in self.commands].count('stop'),1)
        self.assertEqual(self.commands[-1][0],'stop')
        self.assertLess(self.now,1.8)
        times=[x[1] for x in self.commands if x[0] in ('start','renew')]
        self.assertLess(max(b-a for a,b in zip(times,times[1:])),.4)

    def test_already_mobile_car_stops_early_instead_of_waiting_for_breakaway_budget(self):
        result=self.run_drive(progress_at=.01)
        self.assertLess(result['elapsed_s'],.4)
        self.assertEqual(result['reason'],'measured_local_step_complete')

    def test_no_motion_never_extends_firmware_lease_beyond_hard_deadline(self):
        self.run_drive(progress_at=10.)
        renews=[x[1] for x in self.commands if x[0]=='renew']
        self.assertTrue(all(t+.4<=1.8 for t in renews))
        self.assertLessEqual(self.now,1.88)
        self.assertEqual(self.commands[-1][0],'stop')

    def test_slow_ack_cannot_be_used_as_permission_to_restart_expired_lease(self):
        def delayed(): self.commands.append(('start',0));self.now+=.6
        result=self.run_drive(start=delayed)
        self.assertEqual(result['renewals'],0)
        self.assertEqual(result['reason'],'lease_expired_before_fresh_feedback')

    def test_stale_scan_is_never_renewed(self):
        self.late=True
        result=self.run_drive()
        self.assertEqual(result['renewals'],0)

    def test_bad_motion_or_live_obstacle_or_glass_stops_before_renewal(self):
        for reason in ('front_not_clear','startup_glass_clearance'):
            self.setUp();self.check=dict(safe=False,reason=reason)
            result=self.run_drive()
            self.assertEqual(result['reason'],reason)
            self.assertEqual(result['renewals'],0)
        self.setUp();self.reliable=False
        self.assertEqual(self.run_drive()['reason'],'unreliable_drive_feedback')
        self.assertEqual([x[0] for x in self.commands],['start','stop'])

    def test_cancellation_stops_without_sending_motion(self):
        self.cancelled=True
        self.assertEqual(self.run_drive()['reason'],'cancelled')
        self.assertEqual([x[0] for x in self.commands],['stop'])

    def test_sensor_failure_always_stops(self):
        def error(*a): raise TimeoutError('lost lidar')
        with self.assertRaises(TimeoutError): self.run_drive(read_scan=error)
        self.assertEqual(self.commands[-1][0],'stop')

    def test_short_monitor_wait_stops_pulse_but_returns_for_post_stop_recovery(self):
        from grasp_range_integration.phase1.fresh_motion_feedback import FreshScanWaitTimeout
        def missed(*a):
            self.now += .2
            raise FreshScanWaitTimeout('short lease wait')
        result = self.run_drive(read_scan=missed, ttl_ms=200, budget_ms=900, short_probe=True)
        self.assertEqual(result['reason'], 'fresh_scan_wait_expired')
        self.assertEqual(result['renewals'], 0)
        self.assertEqual([x[0] for x in self.commands], ['start', 'stop'])
        self.assertTrue(any(n == 'startup_pulse_feedback_wait' and not e['terminal']
                            for n,e in self.events))

    def test_closed_or_failed_source_is_not_treated_as_short_monitor_wait(self):
        def failed(*a): raise RuntimeError('source closed')
        with self.assertRaisesRegex(RuntimeError, 'source closed'):
            self.run_drive(read_scan=failed)
        self.assertEqual(self.commands[-1][0], 'stop')

    def test_external_stop_cannot_be_followed_by_a_renewal(self):
        result=self.run_drive(renew=lambda *a:False)
        self.assertEqual(result['reason'],'drive_cancelled_before_renewal')
        self.assertEqual(result['renewals'],0)

    def test_scans_straddling_first_command_do_not_authorize_renewal(self):
        def old(*a):
            self.now+=.1
            return NS(sequence=1,acquisition_start_s=-.1,received_s=self.now)
        self.assertEqual(self.run_drive(read_scan=old)['renewals'],0)

    def test_heading_drift_is_not_mistaken_for_straight_start_success(self):
        fit=NS(transform=Pose2D(.02,0,math.radians(13)),reliable=True,raw_range={})
        result=self.run_drive(measure=lambda *a:fit)
        self.assertEqual(result['reason'],'drive_heading_changed')
        self.assertEqual(result['renewals'],0)

    def test_stale_received_timestamp_is_not_renewed_even_before_ttl_expiry(self):
        def stale(*a):
            self.now+=.08
            return NS(sequence=1,acquisition_start_s=.01,received_s=-1.)
        result=self.run_drive(read_scan=stale)
        self.assertEqual(result['reason'],'stale_drive_scan')
        self.assertEqual(result['renewals'],0)


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name)/'startup_drive_experience.json'
        self.now=1000000.
        self.exp=StartupDriveExperience(self.path,'chassis',clock=lambda:self.now)
        self.command=(460,0,400)
        self.result=dict(reason='measured_local_step_complete',elapsed_s=1.8,renewals=12)
        self.motion=Pose2D(.1,.01,0)

    def test_real_confirmed_success_persists_actual_speed_and_duration(self):
        self.exp.record(self.command,self.result,self.motion,True)
        loaded=StartupDriveExperience(self.path,'chassis',clock=lambda:self.now)
        self.assertEqual(loaded.ready()['speed_mm_s'],460)
        self.assertEqual(loaded.ready()['budget_ms'],2100)
        self.assertIsNone(StartupDriveExperience(self.path,'other-chassis',clock=lambda:self.now).ready())
        self.now+=8*86400
        self.assertIsNone(loaded.ready())

    def test_request_only_stall_expired_or_unreliable_motion_does_not_persist(self):
        for reason in ('breakaway_window_complete','lease_expired_before_fresh_feedback','cancelled'):
            self.exp.record(self.command,dict(self.result,reason=reason),self.motion,True)
        self.exp.record(self.command,self.result,self.motion,False)
        self.exp.record(self.command,self.result,Pose2D(.001,0,0),True)
        self.exp.record(self.command,self.result,Pose2D(-.1,0,0),True)
        self.assertIsNone(self.exp.ready())
        self.assertFalse(self.path.exists())

    def test_corrupt_or_out_of_contract_profile_cannot_seed_startup(self):
        import json
        for value in ('bad','[]',json.dumps(dict(schema=1,binding='chassis',entry=dict(
                speed_mm_s=5000,budget_ms=10000,successes=3,updated_s=self.now)))):
            self.path.write_text(value,encoding='utf-8')
            self.assertIsNone(StartupDriveExperience(self.path,'chassis',clock=lambda:self.now).ready())


class ActualClientIntegrationTests(unittest.TestCase):
    def test_checked_refresh_uses_same_payload_without_arming_background_and_stop_invalidates_it(self):
        import numpy as np
        from raspberry_pi.config import SerialConfig
        from raspberry_pi.esp32.client import Esp32Client
        from raspberry_pi.mapping.scan_matcher import MatchResult
        from grasp_range_integration.phase1.fresh_motion_feedback import install_fresh_motion_feedback
        now=[0.];calls=[]
        class Client(Esp32Client):
            def __init__(self): super().__init__(SerialConfig(),serial_instance=NS())
            def request(self,kind,payload=b'',**kw):
                calls.append(('ack',kind,payload));now[0]+=.04
            def send_without_response(self,kind,payload=b''):
                calls.append(('refresh',kind,payload))
        class Driver:
            def scans(self): return iter(())
        class Localizer:
            lidar_offset=(.14,0);max_distance_m=8.
            def relative_motion(self,*a): return MatchResult(Pose2D(.001,0,0),.02,100,True,.9)
        mission=NS(N10LidarDriver=Driver,Esp32Client=Client,ScanMapLocalizer=Localizer)
        install_fresh_motion_feedback(mission)
        seq=[0]
        def read(*a):
            now[0]+=.08;seq[0]+=1
            if seq[0]==4: client.stop()  # Simulate concurrent user STOP.
            return NS(sequence=seq[0],received_s=now[0],acquisition_start_s=now[0]-.05)
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.FreshScanStream',
                   return_value=NS(latest_after=read)), \
             patch('time.monotonic',side_effect=lambda:now[0]), \
             patch('grasp_range_integration.phase1.startup_active_recovery.assess_recovery_action',
                   return_value=dict(safe=True)), \
             patch('grasp_range_integration.phase1.turn_range_feedback.turn_range_evidence',
                   return_value=dict(accepted=True)):
            mission.N10LidarDriver().scans();client=mission.Esp32Client()
            result=client.startup_recovery_pulse(NS(sequence=0),
                NS(x_m=0,y_m=0,yaw_deg=0,radius_m=.7,yaw_tolerance_deg=180),object(),
                400,0,400,localizer=Localizer())
        renews=[x for x in calls if x[0]=='refresh']
        self.assertEqual(len(renews),3)
        self.assertTrue(all(x[1:]==calls[0][1:] for x in renews))
        self.assertEqual(result['reason'],'drive_cancelled_before_renewal')
        self.assertIsNone(client._active_drive_request)
        self.assertEqual([x[0] for x in calls],['ack','refresh','refresh','refresh','ack','ack'])


if __name__=='__main__': unittest.main()
