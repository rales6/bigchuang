"""No-hardware regressions for scan backlog and ACK/feedback pairing."""

import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from raspberry_pi.lidar.base import LaserScan
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult
from grasp_range_integration.phase1.fresh_motion_feedback import (
    FreshScan, FreshScanStream, FreshScanWaitTimeout, MotionFeedbackFence, install_fresh_motion_feedback,
)


def scan(start=0., end=0., sequence=1):
    return FreshScan(np.zeros(100), np.ones(100), end, None, start, end, sequence)


class FenceTests(unittest.TestCase):
    def test_only_post_ack_expiry_complete_scan_is_valid(self):
        fence=MotionFeedbackFence()
        fence.acknowledge(1.,1.1,250,-2600)
        self.assertAlmostEqual(fence.not_before(),1.4)
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.time.monotonic',return_value=1.6):
            result=fence.assess(scan(.8,.9),scan(1.41,1.51,2))
        self.assertTrue(result['valid'])
        self.assertIsNone(fence.assess(scan(),scan()))

    def test_new_decode_timestamp_does_not_make_old_acquisition_valid(self):
        for before,after in [(scan(.8,.9),scan(1.2,1.51,2)),
                             (LaserScan(np.zeros(100),np.ones(100),1.51),scan(1.41,1.51,2)),
                             (scan(-1.,-.9),scan(1.41,1.51,2))]:
            with self.subTest(before=type(before).__name__):
                fence=MotionFeedbackFence();fence.acknowledge(1.,1.1,250,-2600)
                with patch('grasp_range_integration.phase1.fresh_motion_feedback.time.monotonic',return_value=1.6):
                    self.assertFalse(fence.assess(before,after)['valid'])

    def test_delayed_ack_moves_feedback_barrier_and_stop_cancels_it(self):
        fence=MotionFeedbackFence();fence.acknowledge(1.,2.,350,-2600)
        self.assertAlmostEqual(fence.not_before(),2.4)
        fence.cancel()
        self.assertEqual(fence.not_before(),0.)
        self.assertIsNone(fence.assess(scan(),scan()))

    def test_stale_delivery_cannot_be_used_for_stall(self):
        fence=MotionFeedbackFence();fence.acknowledge(1.,1.1,250,-2600)
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.time.monotonic',return_value=3.):
            self.assertFalse(fence.assess(scan(.8,.9),scan(1.41,1.51,2))['valid'])


class PumpTests(unittest.TestCase):
    def stream(self, source, stop, fence=None, timeout=.8):
        driver=SimpleNamespace(last_scan_frequency_hz=10.,close=stop.set)
        stream=FreshScanStream(source,driver,fence or MotionFeedbackFence(),timeout_s=timeout)
        self.addCleanup(stream.close)
        return stream

    def test_fast_buffered_burst_is_discarded(self):
        stop=threading.Event()
        def source():
            yield scan(sequence=1)
            yield scan(sequence=2)
            while not stop.wait(.06):
                yield scan(sequence=3)
        stream=self.stream(source(),stop)
        result=next(stream)
        self.assertGreaterEqual(stream.discarded_burst,2)
        self.assertGreaterEqual(result.received_s-result.acquisition_start_s,.05)

    def test_paused_consumer_receives_newest_frame_not_old_queue(self):
        stop=threading.Event()
        def source():
            while not stop.wait(.06):yield scan()
        stream=self.stream(source(),stop)
        first=next(stream)
        with stream.condition:
            stream.condition.wait_for(lambda:stream.sequence>=first.sequence+3,timeout=1.)
        second=next(stream)
        self.assertGreaterEqual(second.sequence,first.sequence+3)
        self.assertLess(time.monotonic()-second.received_s,.5)

    def test_next_after_turn_waits_for_full_post_expiry_scan(self):
        stop=threading.Event();fence=MotionFeedbackFence()
        def source():
            while not stop.wait(.06):yield scan()
        stream=self.stream(source(),stop,fence)
        before=next(stream)
        sent=time.monotonic();fence.acknowledge(sent,sent,50,-2600)
        after=next(stream)
        self.assertGreaterEqual(after.acquisition_start_s,fence.not_before())
        self.assertTrue(fence.assess(before,after)['valid'])

    def test_no_fresh_frame_times_out_instead_of_returning_old_frame(self):
        stop=threading.Event()
        def source():
            while not stop.wait(.005):yield scan()
        stream=self.stream(source(),stop,timeout=.05)
        with self.assertRaises(TimeoutError):next(stream)

    def test_monitor_deadline_is_distinct_from_failed_or_closed_source(self):
        stream = FreshScanStream.__new__(FreshScanStream)
        stream.condition = threading.Condition()
        stream.latest = None
        stream.closed = False
        stream.error = None
        with self.assertRaises(FreshScanWaitTimeout):
            stream.latest_after(0, .001)
        stream.error = OSError('lidar disconnected')
        with self.assertRaisesRegex(RuntimeError, '读取失败'):
            stream.latest_after(0, .001)
        stream.error = None
        stream.closed = True
        with self.assertRaisesRegex(RuntimeError, '已关闭'):
            stream.latest_after(0, .001)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        class Client:
            def __init__(self):self.calls=[]
            def set_bounded_twist(self,*a,**kw):self.calls.append(('bounded',a,kw));return 'ACK'
            def set_twist(self,*a):self.calls.append(('regular',a));return 'ACK'
            def allow_twist_to_expire(self):self.calls.append(('expire',))
            def stop(self):self.calls.append(('stop',))
        class Localizer:
            def relative_motion(self,*a):return getattr(self,'match',MatchResult(Pose2D(0.,0.,0.),.02,100,True,.9))
        self.mission=SimpleNamespace(N10LidarDriver=object,Esp32Client=Client,ScanMapLocalizer=Localizer)
        self.fence=install_fresh_motion_feedback(self.mission)

    def tearDown(self):
        from grasp_range_integration.phase1 import turn_policy
        turn_policy.TURN_START_REQUEST=None

    def test_no_live_stream_cannot_enable_continuous_torque(self):
        from grasp_range_integration.phase1 import turn_policy
        turn_policy.TURN_START_REQUEST={'angular_mrad_s':-2600,'error_rad':1.}
        client=self.mission.Esp32Client()
        client.set_twist(0,-2600,250)
        self.assertEqual(client.calls[0][0],'bounded')
        self.assertEqual(client.calls[1],('expire',))

    def test_ack_arms_fence_without_arming_background_refresh(self):
        client=self.mission.Esp32Client()
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.time.monotonic',side_effect=[1.,1.2]):
            self.assertEqual(client.set_twist(0,-2600,250),'ACK')
        self.assertEqual(client.calls[0][0],'bounded')
        self.assertEqual(client.calls[0][2]['refresh_deadline_s'],1.)
        self.assertEqual(client.calls[1],('expire',))
        self.assertAlmostEqual(self.fence.not_before(),1.5)
        client.stop()
        self.assertIsNone(self.fence.pending)

    def test_unpaired_motion_requests_recovery_without_latching_mission_fault(self):
        self.fence.acknowledge(1.,1.1,250,-2600)
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.time.monotonic',return_value=1.6):
            result=self.mission.ScanMapLocalizer().relative_motion(scan(.8,.9),scan(1.2,1.51,2))
        self.assertFalse(result.converged)
        self.assertIs(result,self.mission._phase1_turn_relocalization_match)
        self.assertIsNone(getattr(self.mission,'_phase1_motion_fault',None))
        client=self.mission.Esp32Client()
        self.assertIsNone(client.set_twist(400,0,650))
        self.assertEqual(client.calls,[('stop',)])

    def test_valid_motion_keeps_original_quality_and_forward_command_unchanged(self):
        self.fence.acknowledge(1.,1.1,250,-2600)
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.time.monotonic',return_value=1.6):
            result=self.mission.ScanMapLocalizer().relative_motion(scan(.8,.9),scan(1.41,1.51,2))
        self.assertTrue(result.converged)
        client=self.mission.Esp32Client()
        client.set_twist(240,0,650)
        self.assertEqual(client.calls,[('regular',(240,0,650))])
        self.assertIsNone(self.fence.pending)

    def test_one_slow_frame_defers_only_that_command_then_new_scan_can_resume(self):
        estimate = dict(policy='continuous_lidar_odometry', started_s=0.,
                        last_evidence_s=100., uncertainty_m=.20, distance_m=1.3)
        self.mission._phase1_navigation_estimate = estimate
        client = self.mission.Esp32Client()
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.time.monotonic', return_value=100.51):
            self.assertIsNone(client.set_twist(400, 0, 650))
        self.assertEqual(client.calls, [('stop',)])
        self.assertFalse(estimate['command_budget_exhausted'])
        estimate['last_evidence_s'] = 100.6
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.time.monotonic', return_value=100.6):
            client.set_twist(400, 0, 650)
        self.assertEqual(client.calls[-1], ('regular', (400, 0, 250)))
        self.assertEqual(estimate['uncertainty_m'], .20)

    def test_excess_uncertainty_still_latches_budget_not_fresh_scan_wait(self):
        estimate = dict(policy='continuous_lidar_odometry', started_s=0.,
                        last_evidence_s=100., uncertainty_m=.34, distance_m=1.3)
        self.mission._phase1_navigation_estimate = estimate
        client = self.mission.Esp32Client()
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.time.monotonic', return_value=100.):
            self.assertIsNone(client.set_twist(400, 0, 650))
        self.assertTrue(estimate['command_budget_exhausted'])
        self.assertEqual(client.calls, [('stop',)])

    def test_unconsumed_turn_feedback_cannot_be_cancelled_by_forward_command(self):
        client=self.mission.Esp32Client()
        self.fence.acknowledge(1.,1.1,70,2600)
        with self.assertRaisesRegex(RuntimeError,'故障已锁定'):client.set_twist(400,0,650)
        self.assertEqual(client.calls,[('stop',)])

    def test_zero_command_may_stop_but_cannot_unlock_missing_turn_feedback(self):
        client=self.mission.Esp32Client()
        self.fence.acknowledge(1.,1.1,70,2600)
        client.set_twist(0,0,650)
        self.assertIsNone(self.fence.pending)
        with self.assertRaisesRegex(RuntimeError,'故障已锁定'):client.set_twist(400,0,650)
        self.assertEqual(client.calls,[('regular',(0,0,650)),('stop',)])

    def test_unverified_47_degree_feedback_enters_recovery_without_exact_map_anchor(self):
        import math
        client=self.mission.Esp32Client()
        localizer=self.mission.ScanMapLocalizer()
        localizer.match=MatchResult(Pose2D(.1423,0.,math.radians(46.981)),.02,100,True,.9)
        self.fence.acknowledge(1.,1.1,70,2600)
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.time.monotonic',return_value=1.6):
            result=localizer.relative_motion(scan(.8,.9),scan(1.41,1.51,2))
        self.assertFalse(result.converged)
        self.assertIs(result,self.mission._phase1_turn_relocalization_match)
        self.assertIsNone(getattr(self.mission,'_phase1_motion_fault',None))
        self.assertIsNone(client.set_twist(400,281,650))
        self.assertTrue(all(c[0]=='stop' for c in client.calls))

    def test_valid_large_turn_with_map_anchor_enters_relocalization_not_fault(self):
        import math
        class Driver:
            def scans(self): return iter(())
        class Client:
            def __init__(self): self.calls=[]
            def stop(self): self.calls.append(('stop',))
            def set_twist(self,*a): self.calls.append(('regular',a))
        class Localizer:
            def relative_motion(self,*_):
                return MatchResult(Pose2D(.0505,0.,math.radians(-26.932)),
                                   .0687,160,True,.867)
        mission=SimpleNamespace(N10LidarDriver=Driver,Esp32Client=Client,
                                ScanMapLocalizer=Localizer)
        fence=install_fresh_motion_feedback(mission)
        before,after=scan(.8,.9),scan(1.41,1.51,2)
        mission._phase1_turn_pose_anchor=(before,Pose2D())
        stream=SimpleNamespace(latest_after=lambda *_a,**_k:after)
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.FreshScanStream',
                   return_value=stream):
            mission.N10LidarDriver().scans()
        client=mission.Esp32Client()
        fence.acknowledge(1.,1.1,70,-1200)
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.time.monotonic',
                   return_value=1.6), patch(
                       'grasp_range_integration.phase1.turn_feedback_recovery.recover_large_turn',
                       return_value=None):
            result=mission.ScanMapLocalizer().relative_motion(before,after)
        self.assertFalse(result.converged)
        self.assertIs(result,mission._phase1_turn_relocalization_match)
        self.assertIsNone(getattr(mission,'_phase1_motion_fault',None))
        self.assertIsNone(client.set_twist(400,0,650))
        self.assertTrue(all(call[0]=='stop' for call in client.calls))

    def test_refresh_path_also_checks_fault_and_glass(self):
        client=self.mission.Esp32Client()
        self.mission._phase1_authorize_motion=lambda *a: (_ for _ in ()).throw(RuntimeError('glass_sweep_blocked'))
        with self.assertRaisesRegex(RuntimeError,'glass_sweep_blocked'):client.refresh_twist(400,0,650)
        self.assertEqual(client.calls,[('stop',)])

    def test_verified_glass_drive_has_no_background_renewal(self):
        client=self.mission.Esp32Client()
        self.mission._phase1_verified_glass_required=True
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.time.monotonic',return_value=1.):
            client.refresh_twist(400,0,650)
        self.assertEqual(client.calls[0][0],'bounded')
        self.assertLess(client.calls[0][2]['refresh_deadline_s'],1.)

    def test_failed_ack_never_creates_feedback_permission(self):
        client=self.mission.Esp32Client()
        with patch.object(client.__class__.__mro__[1],'set_bounded_twist',side_effect=RuntimeError('BLE failure')):
            with self.assertRaisesRegex(RuntimeError,'BLE failure'):client.set_twist(0,-2600,250)
        self.assertIsNone(self.fence.pending)

    def test_slow_fit_refresh_requires_stop_and_post_stop_acquisition(self):
        class Driver:
            def scans(self): return iter(())
        client_class = self.mission.Esp32Client.__mro__[1]
        localizer_class = self.mission.ScanMapLocalizer
        mission = SimpleNamespace(N10LidarDriver=Driver, Esp32Client=client_class,
                                  ScanMapLocalizer=localizer_class)
        install_fresh_motion_feedback(mission)
        client = mission.Esp32Client()
        localizer = mission.ScanMapLocalizer()
        localizer.lidar_offset = np.array([.14, 0.]); localizer.max_distance_m = 8.
        reference = scan(9.5, 9.7, 1)
        old_inflight = scan(9.95, 10.10, 2)
        after_stop = scan(10.06, 10.20, 3)
        readings = iter([old_inflight, after_stop])
        stream = SimpleNamespace(latest_after=lambda *a, **k: next(readings))
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.FreshScanStream', return_value=stream):
            mission.N10LidarDriver().scans()
        localizer.relative_motion(reference, reference)
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.time.monotonic', return_value=10.), \
             patch('grasp_range_integration.phase1.turn_range_feedback.turn_range_evidence', return_value={'accepted':True}):
            fresh, supported = mission._phase1_refresh_rotation_evidence(reference)
        self.assertIs(fresh, after_stop)
        self.assertTrue(supported)
        self.assertEqual(client.calls, [('stop',)])


if __name__=='__main__':unittest.main()
