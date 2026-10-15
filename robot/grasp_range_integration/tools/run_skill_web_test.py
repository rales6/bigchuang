"""Continue ONE explicitly authorized web test through its existing step APIs.

This is a test runner, not a motor controller. Never repeats an uncertain
POST or marks a failed step complete. All recovery remains in robot adapters.
"""
import argparse
import json
import time
import urllib.request


def request(base, path, body=None):
    data=None if body is None else json.dumps(body,ensure_ascii=False).encode('utf-8')
    req=urllib.request.Request(base+path,data=data,headers={'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(req,timeout=450 if body is not None else 15))


def run(task_id, *, base='http://127.0.0.1:8030', budget_s=1800):
    deadline=time.monotonic()+budget_s
    previous=None
    dispatched=set()
    while time.monotonic()<deadline:
        task=request(base,'/api/tasks/'+task_id)['task']
        steps=task.get('plan',{}).get('steps',[])
        states=task.get('execution',{}).get('steps',{})
        summary={str(s['id']):states.get(str(s['id']),{}).get('status','pending') for s in steps}
        if summary!=previous:
            print(json.dumps({'task_id':task_id,'status':task['status'],'steps':summary},ensure_ascii=False),flush=True)
            previous=summary
        if task['status']=='completed':return True
        if task['status'] not in ('confirmed','running'):
            raise RuntimeError('Task is not authorized to continue: '+task['status'])
        for step in steps:
            sid=str(step['id']);status=summary[sid]
            if status=='completed':continue
            if status in ('failed','blocked','awaiting_verification'):
                raise RuntimeError('Step needs attention; not repeated: '+sid+' '+status)
            if status=='running':break
            if sid in dispatched:
                raise RuntimeError('Dispatch outcome uncertain; not repeated: '+sid)
            if not all(summary.get(str(dep))=='completed' for dep in step.get('depends_on',[])):
                raise RuntimeError('Dependency is incomplete: '+sid)
            dispatched.add(sid)  # Before POST, even if response is lost.
            print('DISPATCH '+sid+' '+str(step.get('action')),flush=True)
            result=request(base,'/api/tasks/'+task_id+'/steps/'+sid+'/execute',{})
            if result.get('ok') is not True:
                print(json.dumps(result,ensure_ascii=False),flush=True)
                raise RuntimeError('Step dispatch failed; not repeated: '+sid)
            break
        time.sleep(2)
    raise RuntimeError('Test runner budget exhausted; inspect task; no dispatch retry')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-id',required=True)
    p.add_argument('--budget-s',type=float,default=1800)
    a=p.parse_args()
    run(a.task_id,budget_s=a.budget_s)
