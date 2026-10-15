"""Recover pre-enqueue hull jitter; never repeat an accepted arm sequence."""
import ast
import json
import math
import re
import time
from urllib.error import HTTPError


def _hull_decline(value):
    """Only the frozen endpoint's exact, pre-enqueue 25px refusal is retryable."""
    if isinstance(value, dict):
        if value.get('ok') is not False:
            return None
        message = str(value.get('error') or '')
    elif isinstance(value, HTTPError) and value.code == 422:
        try:
            body = json.loads(value.read(16384).decode('utf-8'))
        except (ValueError, OSError):
            return None
        return _hull_decline(body)
    elif (isinstance(value, RuntimeError) and isinstance(value.__cause__, HTTPError)
          and value.__cause__.code == 422 and str(value).startswith('上游服务 422: ')):
        message = str(value)[len('上游服务 422: '):]
    else:
        return None
    match = re.fullmatch(
        r'目标尚未同时进入左右二维示教区或25px外沿，机械臂保持停止；'
        r'左右误差 (\{[^\n]+\})px，目标中心 (\{[^\n]+\})', message)
    if match is None:
        return None
    try:
        distances, centers = map(ast.literal_eval, match.groups())
        if set(distances) != {'left', 'right'} or set(centers) != {'left', 'right'}:
            return None
        if not all(isinstance(centers[s], (tuple, list)) and len(centers[s]) == 2
                   for s in ('left', 'right')):
            return None
        values = [float(distances[side]) for side in ('left', 'right')]
        xy = [float(v) for side in ('left', 'right') for v in centers[side]]
        if (len(xy) != 4 or not all(math.isfinite(v) for v in values + xy)
                or min(values) < 0 or not 25 < max(values) <= 35):
            return None
    except (ValueError, TypeError, KeyError, SyntaxError, OverflowError):
        return None
    return {'reason': message, 'view_hull_distances_px': distances, 'centers': centers}


def _idle_before_hull_retry(request, base_url, prior):
    """A refusal alone cannot prove there is no other queued arm command."""
    state = request(base_url + '/api/arm-teach/exec', timeout=3.0).get('status') or {}
    queue = request(base_url + '/api/arm-teach/state', timeout=3.0)
    if (state.get('active') is not False or type(queue.get('pending')) is not int
            or queue['pending'] != 0 or state.get('phase') not in {'idle', 'done', 'error'}
            or any(state.get(k) != prior.get(k) for k in ('phase', 'message', 'params'))):
        raise RuntimeError('夹取复拍拒绝后机械臂状态或队列发生变化，不重复提交夹取')
    return state


def execute_model_grasp(target, *, request, cancel_event, base_url,
                        prior_execution=None, emit=None, timeout_s=75.0,
                        realign=None, prepare_recovery=None, max_hull_retries=3,
                        recovery_timeout_s=180.0):
    prior = (prior_execution if prior_execution is not None else
             request(base_url + '/api/arm-teach/exec', timeout=3.0).get('status') or {})
    if prior.get('active'):
        raise RuntimeError('已有机械臂任务执行中，不能重复夹取')
    if cancel_event.is_set():
        raise RuntimeError('任务已取消，禁止启动夹取')
    retries = 0
    latest_alignment = None
    recovery_deadline = None
    while True:
        if cancel_event.is_set():
            raise RuntimeError('任务已取消，禁止启动夹取')
        # Alignment may have updated STOP metadata since the caller's probe.
        # Capture the state before this particular prediction, not an old one.
        if realign is not None and prepare_recovery is not None:
            prior = request(base_url + '/api/arm-teach/exec', timeout=3.0).get('status') or {}
            prior = _idle_before_hull_retry(request, base_url, prior)
        failure = None
        try:
            accepted = request(base_url + '/api/arm-teach/predict-grasp',
                               {'target': target}, timeout=130.0)
            decline = _hull_decline(accepted)
        except (RuntimeError, HTTPError) as exc:
            decline = _hull_decline(exc)
            if decline is None:
                raise
            failure = exc
        if decline is None:
            break
        if cancel_event.is_set():
            raise RuntimeError('任务已取消，未启动机械臂')
        if realign is None or prepare_recovery is None:
            if failure is not None:
                raise failure
            return accepted
        if recovery_deadline is None:
            recovery_deadline = time.monotonic() + recovery_timeout_s
        if retries >= max_hull_retries or time.monotonic() + 12 >= recovery_deadline:
            raise RuntimeError('夹取前复拍仍未进入示教区：已完成重新识别及微调重试，未启动机械臂')
        prior = _idle_before_hull_retry(request, base_url, prior)
        retries += 1
        if emit is not None:
            emit('grasp_hull_recheck_recovery', target=target, retry=retries,
                 fresh_stereo_required=True, arm_sequence_enqueued=False, **decline)
        prepare_recovery()
        if cancel_event.is_set():
            raise RuntimeError('任务已取消，未启动机械臂')
        prior = request(base_url + '/api/arm-teach/exec', timeout=3.0).get('status') or {}
        prior = _idle_before_hull_retry(request, base_url, prior)
        alignment = realign(recovery_deadline)
        latest_alignment = alignment
        if cancel_event.is_set():
            raise RuntimeError('任务已取消，未启动机械臂')
        if alignment.get('ok') is not True:
            raise RuntimeError('夹取前重新微调未完成：' + str(alignment.get('reason') or '原因未知'))
        if time.monotonic() >= recovery_deadline:
            raise RuntimeError('夹取前微调恢复已超时，未启动机械臂')
        # Re-read after alignment/STOP: do not reuse pre-alignment arm state.
        prior = request(base_url + '/api/arm-teach/exec', timeout=3.0).get('status') or {}
        prior = _idle_before_hull_retry(request, base_url, prior)
    if not accepted.get('ok'):
        return accepted
    completed = False
    saw_active = False
    phase_before = None
    deadline = time.monotonic() + timeout_s
    try:
        while not cancel_event.is_set() and time.monotonic() < deadline:
            state = request(base_url + '/api/arm-teach/exec', timeout=3.0).get('status') or {}
            phase = str(state.get('phase') or '')
            saw_active = saw_active or bool(state.get('active'))
            if phase != phase_before and emit is not None:
                emit('grasp_phase', status=state)
            phase_before = phase
            if phase == 'done' and saw_active and not state.get('active'):
                completed = True
                return {'ok': True, 'execution': state, 'prediction': accepted,
                        'message': state.get('message'), 'physical_pick_verified': False,
                        'hull_recheck_retries': retries,
                        'hull_recheck_alignment': latest_alignment}
            if phase == 'error' and not state.get('active') and (
                    saw_active or state.get('message') != prior.get('message')):
                raise RuntimeError(f"机械臂夹取失败：{state.get('message') or '执行错误'}")
            time.sleep(.25)
        raise RuntimeError('夹取已取消或超过时限，已请求停止')
    finally:
        if not completed:
            try:
                request(base_url + '/api/arm-teach/stop', {}, timeout=3.0)
            except Exception as exc:
                if emit is not None:
                    emit('grasp_stop_unconfirmed', error=str(exc))
                # Do not report successful stopping if the stop API failed.
                raise RuntimeError(f'机械臂执行异常，停止请求也失败，请立即检查：{exc}') from exc

