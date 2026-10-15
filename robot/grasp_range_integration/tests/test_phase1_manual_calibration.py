import unittest
from types import SimpleNamespace
from grasp_range_integration.phase1.probe_live_rotation import calibration_pulse, stationary_preflight


class CalibrationTests(unittest.TestCase):
    def run_pulse(self, ttl=180, safe=True, ack=.2):
        state = dict(now=1., sends=[], stops=0, events=[])
        def scan(*a):
            state['now'] += .1
            return SimpleNamespace(received_s=state['now'], sequence=1)
        def send(speed, ttl, deadline):
            state['sends'].append((speed, ttl, deadline))
            state['now'] += ack
        def stop(): state['stops'] += 1
        try:
            calibration_pulse(ttl_ms=ttl, speed=-3500, read_scan=scan,
                assess=lambda s: dict(safe=safe), send=send, stop=stop,
                emit=lambda k, **kw: state['events'].append((k, kw)),
                clock=lambda: state['now'])
        except Exception as exc:
            state['error'] = exc
        return state

    def test_one_lease_even_late_ack(self):
        result = self.run_pulse()
        self.assertEqual(len(result['sends']), 1)
        self.assertEqual(result['sends'][0][:2], (-3500, 180))
        self.assertLess(result['sends'][0][2], 1.1)
        self.assertEqual(result['stops'], 1)
        self.assertFalse(result['events'][0][1]['autonomous_permission'])

    def test_unsafe_does_not_send(self):
        result = self.run_pulse(safe=False)
        self.assertEqual(result['sends'], [])
        self.assertEqual(result['stops'], 1)
        self.assertIsInstance(result['error'], RuntimeError)

    def test_invalid_duration(self):
        result = self.run_pulse(ttl=350)
        self.assertEqual(result['sends'], [])
        self.assertIsInstance(result['error'], ValueError)

    def test_wait_uses_no_refresh(self):
        result = self.run_pulse(ack=0.)
        self.assertEqual(len(result['sends']), 1)
        self.assertEqual(result['stops'], 1)

    def test_preflight_transient_resets_streak_and_records_all(self):
        frames = iter([True, True, False, True, True, True, True, True])
        records = []
        stationary_preflight(lambda: next(frames), lambda frame: {'safe': frame}, records.append)
        self.assertEqual(len(records), 8)
        self.assertFalse(records[2])

    def test_persistent_obstacle_never_passes(self):
        records = []
        with self.assertRaisesRegex(RuntimeError, '未连接底盘'):
            stationary_preflight(lambda: False, lambda frame: {'safe': frame}, records.append)
        self.assertEqual(len(records), 30)

    def test_nonconsecutive_safe_frames_cannot_pass(self):
        frames = iter([True]*4+[False]+[True]*4+[False])
        with self.assertRaises(RuntimeError):
            stationary_preflight(lambda: next(frames), lambda frame: {'safe': frame},
                                 lambda frame: None, maximum=10)
