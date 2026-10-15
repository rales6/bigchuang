"""Offline pre-enqueue rejection replay; no camera, API, motor or arm IO."""
import io
import json
import threading
import unittest
from urllib.error import HTTPError
from unittest.mock import Mock, patch

from grasp_range_integration.track_web_integration.pi import grasp_execution as grasp


MESSAGE = ("目标尚未同时进入左右二维示教区或25px外沿，机械臂保持停止；"
           "左右误差 {'left': 25.77728538425835, 'right': 0.0}px，"
           "目标中心 {'left': [287.4, 299.8], 'right': [207.4, 257.5]}")


def refusal(wrapped=False, message=MESSAGE, code=422):
    e=HTTPError('http://test/api/arm-teach/predict-grasp',code,'refused',{},
                io.BytesIO(json.dumps(dict(ok=False,error=message)).encode()))
    if not wrapped:return e
    r=RuntimeError(f'上游服务 {code}: {message}')
    r.__cause__=e
    return r


class Gateway:
    def __init__(self, replies):
        self.replies=list(replies)
        self.state=dict(active=False,phase='idle',message='',params={'s2':2000})
        self.calls=[]
        self.accepted=0
        self.polls=0
        self.pending=0
        self.on_refusal=lambda:None

    def __call__(self,url,payload=None,timeout=None):
        self.calls.append(url.rsplit('/',1)[-1])
        if url.endswith('/exec'):
            if self.accepted:
                self.polls+=1
                return {'status':dict(self.state,active=self.polls==1,
                                      phase='grasping' if self.polls==1 else 'done')}
            return {'status':dict(self.state)}
        if url.endswith('/state'):return {'pending':self.pending}
        if url.endswith('/stop'):return {'ok':True,'result':'OK'}
        if url.endswith('/predict-grasp'):
            r=self.replies.pop(0)
            if isinstance(r,Exception):
                self.on_refusal()
                raise r
            if r.get('ok'):self.accepted+=1
            else:self.on_refusal()
            return r
        raise AssertionError(url)


class HullRecheckTests(unittest.TestCase):
    def run_grasp(self,gateway,**kw):
        return grasp.execute_model_grasp('绿色小方块',request=gateway,
            cancel_event=kw.pop('cancel_event',threading.Event()),base_url='http://test',
            **kw)

    def test_real_recorded_25_78_rejection_then_success_both_http_forms(self):
        for wrapped in (False,True):
            with self.subTest(wrapped=wrapped),patch.object(grasp.time,'sleep'):
                g=Gateway([refusal(wrapped),{'ok':True}])
                align=Mock(return_value={'ok':True,'distance_px':20})
                prepare=Mock();emit=Mock()
                result=self.run_grasp(g,realign=align,prepare_recovery=prepare,emit=emit)
                self.assertTrue(result['ok'])
                self.assertEqual(result['hull_recheck_retries'],1)
                self.assertEqual(g.accepted,1)
                self.assertEqual(g.calls.count('predict-grasp'),2)
                self.assertNotIn('stop',g.calls)  # no accepted arm sequence cancelled
                prepare.assert_called_once();align.assert_called_once()
                self.assertFalse(emit.call_args_list[0].kwargs['arm_sequence_enqueued'])

    def test_plain_json_refusal_uses_same_recovery(self):
        g=Gateway([{'ok':False,'error':MESSAGE},{'ok':True}])
        with patch.object(grasp.time,'sleep'):
            self.assertTrue(self.run_grasp(g,realign=lambda d:{'ok':True},prepare_recovery=lambda:None)['ok'])
        self.assertEqual(g.accepted,1)

    def test_unknown_or_not_near_hull_failure_never_retried(self):
        for e in (refusal(message='预测期间收到急停或复位，取消本次模型夹取'),
                  refusal(code=500),refusal(message=MESSAGE.replace('25.77728538425835','65.0')),
                  TimeoutError('prediction response lost'),RuntimeError(MESSAGE)):
            g=Gateway([e]);align=Mock();prepare=Mock()
            with self.subTest(error=str(e)),self.assertRaises(Exception):
                self.run_grasp(g,realign=align,prepare_recovery=prepare)
            align.assert_not_called();prepare.assert_not_called()
            self.assertEqual(g.calls.count('predict-grasp'),1)

    def test_malformed_geometry_not_retryable(self):
        for m in (MESSAGE.replace('25.77728538425835','nan'),
                  MESSAGE.replace('[287.4, 299.8]','[287.4]'),
                  MESSAGE.replace("'right': 0.0","'right': -1.0")):
            self.assertIsNone(grasp._hull_decline({'ok':False,'error':m}))

    def test_no_retry_if_arm_became_active_queued_or_completed(self):
        for change in ('active','pending','phase'):
            g=Gateway([refusal()]);align=Mock();prepare=Mock()
            def changed():
                if change=='pending':g.pending=1
                elif change=='active':g.state['active']=True
                else:g.state['phase']='done'
            g.on_refusal=changed
            with self.subTest(change=change),self.assertRaises(RuntimeError):
                self.run_grasp(g,realign=align,prepare_recovery=prepare)
            align.assert_not_called();prepare.assert_not_called()

    def test_cancel_during_prepare_or_realign_no_second_submission(self):
        for stage in ('prepare','align'):
            cancel=threading.Event();g=Gateway([refusal()])
            def prep():
                if stage=='prepare':cancel.set()
            def align(d):
                if stage=='align':cancel.set()
                return {'ok':True}
            with self.subTest(stage=stage),self.assertRaises(RuntimeError):
                self.run_grasp(g,realign=align,prepare_recovery=prep,cancel_event=cancel)
            self.assertEqual(g.calls.count('predict-grasp'),1)

    def test_prepare_failure_and_alignment_failure_do_not_submit_arm(self):
        for prepare,align in ((Mock(side_effect=RuntimeError('STOP failed')),Mock()),
                              (Mock(),Mock(return_value={'ok':False,'reason':'lost'}))):
            g=Gateway([refusal()])
            with self.assertRaises(RuntimeError):self.run_grasp(g,realign=align,prepare_recovery=prepare)
            self.assertEqual(g.calls.count('predict-grasp'),1)

    def test_persistent_boundary_retries_are_bounded_and_not_arm_attempts(self):
        g=Gateway([refusal() for _ in range(4)])
        align=Mock(return_value={'ok':True});prepare=Mock()
        with self.assertRaisesRegex(RuntimeError,'已完成重新识别及微调重试'):
            self.run_grasp(g,realign=align,prepare_recovery=prepare)
        self.assertEqual(align.call_count,3)
        self.assertEqual(g.accepted,0)

    def test_accepted_arm_error_is_never_resubmitted(self):
        g=Gateway([{'ok':True}]);align=Mock();prepare=Mock()
        def broken(url,*a,**k):
            if url.endswith('/exec') and g.accepted:
                return {'status':dict(g.state,active=False,phase='error',message='BLE failed')}
            return g(url,*a,**k)
        with self.assertRaisesRegex(RuntimeError,'机械臂夹取失败'):
            self.run_grasp(broken,realign=align,prepare_recovery=prepare)
        self.assertEqual(g.calls.count('predict-grasp'),1)
        align.assert_not_called();prepare.assert_not_called()
        self.assertIn('stop',g.calls)

    def test_without_motion_permission_no_recovery(self):
        g=Gateway([refusal(True)])
        with self.assertRaises(RuntimeError):self.run_grasp(g)
        self.assertEqual(g.calls.count('predict-grasp'),1)

    def test_shared_recovery_deadline_not_reset(self):
        g=Gateway([refusal(),refusal()]);align=Mock(return_value={'ok':True})
        with patch.object(grasp.time,'monotonic',side_effect=[0,1,2,181]),self.assertRaises(RuntimeError):
            self.run_grasp(g,realign=align,prepare_recovery=lambda:None)
        self.assertEqual(align.call_count,1)
        self.assertEqual(align.call_args.args[0],180)

    def test_web_model_entry_recovers_and_records_carry_once(self):
        from grasp_range_integration.track_web_integration.pi import robot_bridge as bridge_module
        from grasp_range_integration.track_web_integration.pi import pursuit_session as session
        bridge=bridge_module.RobotBridge.__new__(bridge_module.RobotBridge)
        bridge._authorise_hardware=Mock()
        bridge.skills=Mock()
        bridge.skills.assert_empty=Mock()
        bridge._grasp_cancel=threading.Event()
        bridge._phase1_verified_target='绿色小方块'
        bridge.process=Mock()
        bridge.process.snapshot.return_value={'running':False,'kind':'idle'}
        bridge._probe=Mock(return_value={'connected':True,'data':{'ble':True,
            'cameras':{s:{'age_s':0.1,'frames':100} for s in ('left','right')}}})
        g=Gateway([refusal(True),{'ok':True}])
        with patch.object(bridge_module,'_http_json',side_effect=g), \
             patch.object(bridge_module,'_align_teaching_hull',return_value={'ok':True}) as align, \
             patch.object(session,'_prepare_loss_recovery') as prepare, \
             patch('grasp_range_integration.track_web_integration.pi.motion_session._event'), \
             patch.object(grasp.time,'sleep'):
            result=bridge._grasp_impl('model',{'target':'绿色小方块','auto_align':True},{})
        self.assertTrue(result['ok'])
        self.assertEqual(align.call_count,2)
        self.assertIn('deadline',align.call_args.kwargs)
        prepare.assert_called_once()
        bridge.skills.record_pick.assert_called_once()
        self.assertIsNone(bridge._phase1_verified_target)
        self.assertEqual(g.accepted,1)

    def test_combined_pursuit_grasp_entry_uses_realign_without_new_navigation(self):
        import sys
        import types
        from grasp_range_integration.track_web_integration.pi import pursuit_session as session
        g=Gateway([refusal(),{'ok':True}])
        fake=types.ModuleType('pursuit.hull_micro_adjust')
        fake.GRAB_URL='http://test'
        fake._get_json=g
        fake._post_json=g
        with patch.dict(sys.modules,{'pursuit.hull_micro_adjust':fake}), \
             patch.object(session,'_wait_preview_ready',return_value=True), \
             patch.object(session,'_finish_at_exact_hull',return_value={'ok':True}) as align, \
             patch.object(session,'_prepare_loss_recovery') as prepare, \
             patch.object(session,'_event'),patch.object(grasp.time,'sleep'):
            self.assertEqual(session._grasp_after_alignment('绿色小方块',threading.Event()),0)
        align.assert_called_once();prepare.assert_called_once()
        self.assertTrue(align.call_args.kwargs['return_result'])
        self.assertIn('deadline',align.call_args.kwargs)
        self.assertEqual(g.accepted,1)


if __name__=='__main__':unittest.main()
