"""Offline contract and lifecycle checks; no cameras, BLE, APIs or arm calls."""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

from grasp_range_integration.track_web_integration.pi import pursuit_recovery as policy
from grasp_range_integration.track_web_integration.pi import pursuit_session as session


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'logs/pursuit/run'
        self.run.mkdir(parents=True)
        (self.run.parent / 'LATEST.txt').write_text(str(self.run), encoding='utf-8')
        stamp = lambda t: datetime.fromtimestamp(t, timezone.utc).isoformat()
        self.data = dict(pid=16303, mode='esp32', instruction='绿色小物块',
            active_instruction='绿色小物块', status='give_up', final_state='GIVE_UP',
            final_result='give_up', final_detail=policy.LOSS_DETAIL, error='',
            started_at=stamp(1000), ended_at=stamp(1049))
        (self.run / 'telemetry.csv').write_text(
            'state,tracking,tracking2\nTRACKING,False,True\nSEARCHING,False,False\n', encoding='utf-8')

    def check(self, **options):
        (self.run / 'run.json').write_text(json.dumps(self.data), encoding='utf-8')
        args = dict(target='绿色小物块', child_pid=16303, launched_at=1000,
                    returncode=2, now=1050)
        args.update(options)
        return policy.identified_loss_run(self.root, **args)

    def test_real_green_failure_is_retryable(self):
        self.assertEqual(self.check()['reason'], policy.LOSS_DETAIL)

    def test_color_is_preserved_not_hardcoded(self):
        self.data.update(instruction='黄色小物块', active_instruction='黄色小物块')
        self.assertIsNotNone(self.check(target='黄色小物块'))

    def test_wrong_pid_target_mode_or_result_rejected(self):
        for key, value in [('pid', 7), ('mode', 'dry-run'), ('instruction', '黄色小物块'),
                ('active_instruction', '红色小物块'), ('status', 'interrupted'),
                ('final_state', 'TRACKING'), ('final_result', 'success'),
                ('final_detail', '障碍物阻挡'), ('error', 'LinkError')]:
            old = self.data[key]
            self.data[key] = value
            with self.subTest(key=key):
                self.assertIsNone(self.check())
            self.data[key] = old

    def test_old_future_or_previous_launch_rejected(self):
        for options in [dict(now=1065), dict(now=1048), dict(launched_at=1030),
                        dict(now=float('nan')), dict(returncode=0), dict(returncode=130)]:
            with self.subTest(options=options):
                self.assertIsNone(self.check(**options))

    def test_unseen_target_does_not_repeat_blind_search(self):
        (self.run / 'telemetry.csv').write_text('state,tracking,tracking2\nSEARCHING,True,True\n', encoding='utf-8')
        self.assertIsNone(self.check())

    def test_escaped_latest_rejected(self):
        (self.run.parent / 'LATEST.txt').write_text(str(self.root), encoding='utf-8')
        self.assertIsNone(self.check())

    def test_missing_or_malformed_evidence_rejected(self):
        self.check()
        (self.run / 'run.json').write_text('[]', encoding='utf-8')
        self.assertIsNone(policy.identified_loss_run(self.root, target='绿色小物块',
            child_pid=16303, launched_at=1000, returncode=2, now=1050))
        (self.run / 'telemetry.csv').unlink()
        self.assertIsNone(self.check())


class PolicyTests(unittest.TestCase):
    def test_shared_count_and_time_budget(self):
        clock = Mock(return_value=10)
        budget = policy.LossRecoveryBudget(clock=clock)
        cancel = threading.Event()
        self.assertTrue(budget.reserve(cancel))
        self.assertTrue(budget.reserve(cancel))
        self.assertFalse(budget.reserve(cancel))
        budget = policy.LossRecoveryBudget(clock=clock)
        self.assertTrue(budget.reserve(cancel))
        clock.return_value = 190
        self.assertFalse(budget.reserve(cancel))

    def test_cancel_never_reserves(self):
        cancel = threading.Event(); cancel.set()
        self.assertFalse(policy.LossRecoveryBudget().reserve(cancel))

    def test_alignment_only_exact_target_loss(self):
        for reason in ['camera offline', '目标未同时出现在双目画面；BLE断开',
                       '达到 16 次微调上限，未进入示教凸包', '取消', '近障碍']:
            self.assertFalse(policy.alignment_target_lost(dict(ok=False, reason=reason)))
        self.assertTrue(policy.alignment_target_lost(dict(ok=False, reason=policy.ALIGNMENT_LOSS)))

    def recover(self, align, prepare=None, real=True, cancel=None):
        return policy.align_with_loss_recovery('黄色小物块', align=align,
            prepare=prepare or Mock(), real=real, cancelled=cancel or threading.Event(), emit=Mock())

    def test_hull_loss_then_success_reobserves(self):
        prepare = Mock()
        align = Mock(side_effect=[dict(ok=False, reason=policy.ALIGNMENT_LOSS), dict(ok=True)])
        self.assertTrue(self.recover(align, prepare)['ok'])
        self.assertEqual(align.call_count, 2); prepare.assert_called_once()

    def test_hull_fault_does_not_retry(self):
        align = Mock(return_value=dict(ok=False, reason='雷达失联'))
        prepare = Mock()
        self.assertFalse(self.recover(align, prepare)['ok'])
        self.assertEqual(align.call_count, 1); prepare.assert_not_called()

    def test_dry_run_no_motion_retry(self):
        align = Mock(return_value=dict(ok=False, reason=policy.ALIGNMENT_LOSS))
        self.recover(align, real=False)
        self.assertEqual(align.call_count, 1)

    def test_hull_loss_exhausts_three_attempts(self):
        align = Mock(return_value=dict(ok=False, reason=policy.ALIGNMENT_LOSS))
        prepare = Mock()
        self.assertFalse(self.recover(align, prepare)['ok'])
        self.assertEqual(align.call_count, 3); self.assertEqual(prepare.call_count, 2)

    def test_cancel_between_hull_attempts(self):
        cancel = threading.Event()
        align = Mock(return_value=dict(ok=False, reason=policy.ALIGNMENT_LOSS))
        self.assertFalse(self.recover(align, cancel.set, cancel=cancel)['ok'])
        self.assertEqual(align.call_count, 1)

    def test_cancel_during_successful_alignment_is_not_success(self):
        cancel = threading.Event()
        def align():
            cancel.set()
            return dict(ok=True)
        self.assertFalse(self.recover(align, cancel=cancel)['ok'])

    def test_prepare_exception_does_not_launch_second_alignment(self):
        align = Mock(return_value=dict(ok=False, reason=policy.ALIGNMENT_LOSS))
        with self.assertRaises(RuntimeError):
            self.recover(align, Mock(side_effect=RuntimeError('停止请求未确认')))
        self.assertEqual(align.call_count, 1)


class SessionTests(unittest.TestCase):
    def run_session(self, codes, *, losses=None, hulls=None, grasp=False, prepare=None,
                    dry=False, arm=None):
        children = [Mock(pid=i+100, wait=Mock(return_value=code)) for i, code in enumerate(codes)]
        argv = ['session', *(['--grasp'] if grasp else []), '--', '--instruction', '黄色小物块',
                '--dry-run' if dry else '--esp32', '--exit-on-finish']
        self.events = []; self.order = []
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', argv))
            stack.enter_context(patch.object(session.signal, 'signal'))
            self.stop = stack.enter_context(patch.object(session, '_stop_preview',
                side_effect=lambda: self.order.append('release')))
            self.restore = stack.enter_context(patch.object(session, '_restore_preview'))
            self.prepare = stack.enter_context(patch.object(session, '_prepare_loss_recovery',
                side_effect=prepare or (lambda cancel: self.order.append('fresh'))))
            self.spawn = stack.enter_context(patch.object(session.subprocess, 'Popen', side_effect=children))
            self.evidence = stack.enter_context(patch.object(session, 'identified_loss_run',
                side_effect=losses or [None]*len(codes)))
            self.retreat = stack.enter_context(patch.object(session, '_recent_pursuit_retreat', return_value=None))
            self.hull = stack.enter_context(patch.object(session, '_finish_at_exact_hull',
                side_effect=hulls or [dict(ok=True)]))
            self.arm = stack.enter_context(patch.object(session, '_grasp_after_alignment',
                side_effect=arm or (lambda *args: 0)))
            stack.enter_context(patch.object(session, '_event', side_effect=lambda event, **fields:
                self.events.append((event, fields))))
            return session.main()

    def test_green_continue_behavior_automatic_and_target_unchanged(self):
        self.assertEqual(self.run_session([2, 0], losses=[dict(reason=policy.LOSS_DETAIL), None], grasp=True), 0)
        self.assertEqual(self.spawn.call_count, 2)
        self.assertEqual(self.spawn.call_args_list[0], self.spawn.call_args_list[1])
        self.assertEqual(self.order, ['release', 'fresh', 'release'])
        self.retreat.assert_called_once_with('黄色小物块', 101)
        self.arm.assert_called_once()
        self.restore.assert_called_once()

    def test_standalone_align_does_not_execute_arm(self):
        self.assertEqual(self.run_session([2, 0], losses=[dict(reason=policy.LOSS_DETAIL), None]), 0)
        self.arm.assert_not_called()

    def test_successful_child_but_hull_loss_restarts_pursuit(self):
        self.assertEqual(self.run_session([0, 0], hulls=[dict(ok=False, reason=policy.ALIGNMENT_LOSS), dict(ok=True)]), 0)
        self.assertEqual(self.spawn.call_count, 2)
        self.assertEqual(self.retreat.call_args_list[-1].args, ('黄色小物块', 101))

    def test_cancel_during_prepare_no_second_child(self):
        self.assertEqual(self.run_session([2], losses=[dict(reason=policy.LOSS_DETAIL)],
                                        prepare=lambda cancel: cancel.set()), 130)
        self.assertEqual(self.spawn.call_count, 1)

    def test_stop_not_confirmed_no_second_child(self):
        self.assertEqual(self.run_session([2], losses=[dict(reason=policy.LOSS_DETAIL)],
            prepare=Mock(side_effect=RuntimeError('STOP failed'))), 2)
        self.assertEqual(self.spawn.call_count, 1)

    def test_unknown_failure_no_retry(self):
        self.assertEqual(self.run_session([2]), 2)
        self.prepare.assert_not_called(); self.arm.assert_not_called()

    def test_exhausted_target_loss_is_not_success(self):
        self.assertEqual(self.run_session([2, 2, 2], losses=[dict(reason=policy.LOSS_DETAIL)]*3), 2)
        self.assertEqual(self.spawn.call_count, 3)
        self.assertEqual(self.prepare.call_count, 2)
        self.arm.assert_not_called()
        self.assertTrue(any(event == 'pursuit_loss_recovery_exhausted' for event, _ in self.events))

    def test_dry_run_no_recovery_or_arm(self):
        self.assertEqual(self.run_session([2], dry=True), 2)
        self.evidence.assert_not_called(); self.prepare.assert_not_called(); self.arm.assert_not_called()

    def test_arm_failure_never_repeats_accepted_sequence(self):
        self.assertEqual(self.run_session([0], grasp=True, arm=Mock(side_effect=RuntimeError('arm failed'))), 2)
        self.arm.assert_called_once(); self.assertEqual(self.spawn.call_count, 1)
        self.prepare.assert_not_called()


class OwnershipTests(unittest.TestCase):
    def prepare(self, *, state=None, stopped=None, ready=True):
        fake = types.ModuleType('pursuit.hull_micro_adjust')
        fake._get_json = Mock(return_value={'status': state if state is not None else {'active': False}})
        fake._post_json = Mock(return_value=stopped if stopped is not None else {'ok': True, 'result': 'OK'})
        with patch.dict(sys.modules, {'pursuit.hull_micro_adjust': fake}), \
             patch.object(session, '_restore_preview'), \
             patch.object(session, '_wait_preview_ready', return_value=ready):
            session._prepare_loss_recovery(threading.Event())
        return fake

    def test_stop_ack_is_required(self):
        for stopped in [dict(ok=True, result='NO_BLE'), dict(ok=True), dict(ok=False, result='OK')]:
            with self.subTest(stopped=stopped), self.assertRaises(RuntimeError):
                self.prepare(stopped=stopped)

    def test_active_or_unknown_arm_not_cancelled_by_retry(self):
        for state in [dict(active=True), {}, dict(active=None)]:
            with self.subTest(state=state), self.assertRaises(RuntimeError):
                self.prepare(state=state)

    def test_fresh_preview_required(self):
        with self.assertRaises(RuntimeError):
            self.prepare(ready=False)

    def test_idle_and_stop_confirmed(self):
        fake = self.prepare()
        self.assertEqual(fake._post_json.call_args.args[0], 'http://127.0.0.1:8080/api/arm-teach/stop')
        self.assertNotIn('predict-grasp', str(fake._post_json.call_args))

    def test_bridge_real_alignment_retries_before_arm_only(self):
        from grasp_range_integration.track_web_integration.pi import robot_bridge
        fake = types.ModuleType('pursuit.hull_micro_adjust')
        fake.verify_model_camera_order = Mock()
        fake.align = Mock(side_effect=[types.SimpleNamespace(ok=False, reason=policy.ALIGNMENT_LOSS),
                                      types.SimpleNamespace(ok=True, reason='inside')])
        with patch.dict(sys.modules, {'pursuit.hull_micro_adjust': fake}), \
             patch.object(session, '_prepare_loss_recovery') as prepare, \
             patch('grasp_range_integration.track_web_integration.pi.motion_session._event'):
            self.assertTrue(robot_bridge._align_teaching_hull('黄色小物块', real=True)['ok'])
        self.assertEqual(fake.align.call_count, 2); prepare.assert_called_once()
        self.assertTrue(all(call.args == ('黄色小物块',) for call in fake.align.call_args_list))


if __name__ == '__main__':
    unittest.main()
