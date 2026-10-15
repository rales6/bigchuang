"""Recording follows the WHOLE web task, not each navigation/pursuit step."""
import uuid


def prepare_recording(execution):
    execution['recording']={'session_id':uuid.uuid4().hex,'started':False,'finished':False}
    return execution


def start_recording(robot,execution,instruction):
    recording=execution.get('recording')
    if not isinstance(recording,dict) or recording.get('finished'):
        prepare_recording(execution);recording=execution['recording']
    if not recording.get('started'):
        result=robot.request('/recording/start',dict(session_id=recording['session_id'],
                             instruction=instruction),timeout=75)
        if result.get('ok') is not True:
            raise RuntimeError('整轮任务录像未启动：'+str(result.get('error') or result))
        recording.update(started=True,detail=result)
    return recording


def finish_recording(robot,task):
    recording=(task.get('execution') or {}).get('recording') or {}
    if (task.get('status') not in {'completed','blocked','failed','paused','cancelled'}
            or not recording.get('started') or recording.get('finished')):
        return False
    result=robot.request('/recording/finish',dict(session_id=recording['session_id'],
                          reason=task['status']),timeout=5)
    if result.get('ok') is True:
        recording.update(finished=True,finish_result=result)
        return True
    return False


def persist_finished_recording(history, task):
    """Merge only recording metadata; a polling request must not undo a step."""
    latest = history.get_task(task['id'])
    execution = dict(latest['execution'])
    recording = task['execution']['recording']
    if (execution.get('recording') or {}).get('session_id') != recording['session_id']:
        return latest
    execution['recording'] = recording
    return history.update_task(task['id'], status=latest['status'], execution=execution)
