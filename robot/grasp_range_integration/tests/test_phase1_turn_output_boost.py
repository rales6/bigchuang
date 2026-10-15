from types import SimpleNamespace
import math
import unittest

from raspberry_pi.config import SerialConfig
from grasp_range_integration.phase1.turn_output_boost import boosted_gains, send_boosted_turn


class TurnOutputBoostTests(unittest.TestCase):
    def test_real_wheel_output_boost_grows_to_existing_protocol_limit(self):
        self.assertEqual([boosted_gains((1,1,1,1), n)[1] for n in (0,2,4,6,20)],
                         [1.,1.2,1.35,1.5,1.5])

    def test_balance_is_preserved_without_exceeding_any_wheel_limit(self):
        before=(1.2,1.,.8,.9)
        after, gain=boosted_gains(before, 20)
        self.assertEqual(gain,1.25)
        self.assertLessEqual(max(after),1.5)
        self.assertAlmostEqual(after[0]/after[1], before[0]/before[1])

    def test_only_turn_payload_is_boosted_normal_config_is_restored(self):
        normal=SerialConfig()
        client=SimpleNamespace(config=normal)
        calls, events=[],[]
        def send(*args, **kwargs):
            calls.append((args,kwargs,client.config.wheel_output_gains))
            return 'ACK'
        value=send_boosted_turn(client,send,-3500,95,-1.,6,
                               lambda event,**data:events.append((event,data)))
        self.assertEqual(value,'ACK')
        self.assertEqual(calls,[((0,-3500,95),{'refresh_deadline_s':-1.},(1.5,1.5,1.5,1.5))])
        self.assertIs(client.config,normal)
        self.assertTrue(events[-1][1]['straight_drive_unchanged'])

    def test_ack_failure_restores_config_and_never_falls_back_to_unboosted_drive(self):
        normal=SerialConfig()
        client=SimpleNamespace(config=normal)
        def fail(*a,**k): raise RuntimeError('rejected')
        with self.assertRaisesRegex(RuntimeError,'rejected'):
            send_boosted_turn(client,fail,3500,95,-1.,6,lambda *a,**k:None)
        self.assertIs(client.config,normal)

    def test_bad_gains_or_excess_speed_are_rejected_before_send(self):
        for gains in [(1.6,1,1,1),(math.nan,1,1,1),(1,1,1)]:
            with self.assertRaises(ValueError): boosted_gains(gains,6)
        with self.assertRaises(ValueError):
            send_boosted_turn(SimpleNamespace(config=SerialConfig()),
                              lambda *a,**k:self.fail('must not send'),4000,95,-1.,6,lambda *a,**k:None)


if __name__=='__main__': unittest.main()
