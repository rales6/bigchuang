"""Explicit, bounded real-car yaw startup test with stop-and-look feedback.

At most three 0.4s left-turn pulses (900/1600/2100 mrad/s); no forward or
reverse, no arm command. Front fan guard and the shared BLE cancellation
remain active. Requires --esp32 and an idle bridge/arm.
"""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from pursuit.hull_micro_adjust import (
    TRACK_ROOT, GRAB_URL, ScanGuard, _observe, _center_vector, _get_json,
    _pulse, verify_model_camera_order,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--esp32',action='store_true')
    parser.add_argument('--target',required=True)
    parser.add_argument('--hold-s',type=float,default=.4)
    parser.add_argument('--extended',action='store_true')
    parser.add_argument('--single-speed',type=int,choices=(900,1600,2100,2400))
    parser.add_argument('--action',choices=('turn_left','turn_right'),default='turn_left')
    args=parser.parse_args()
    if not args.esp32: parser.error('requires explicit --esp32')
    if not .4 <= args.hold_s <= .8: parser.error('hold must be 0.4..0.8s')
    state=_get_json('http://127.0.0.1:8780/state')['managed']
    if state['running'] and state['kind']!='camera_preview':
        raise RuntimeError('another chassis task is running')
    if _get_json(GRAB_URL+'/api/arm-teach/exec')['status']['active']:
        raise RuntimeError('arm active')
    verify_model_camera_order()
    root=TRACK_ROOT/'logs'/'turn_startup'
    root.mkdir(parents=True,exist_ok=True)
    path=root/(datetime.now().strftime('%Y%m%d_%H%M%S')+'.jsonl')
    def record(event,**fields):
        value={'event':event,**fields}
        with path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(value,ensure_ascii=False)+'\n')
        print(json.dumps(value,ensure_ascii=False),flush=True)
    guard=None
    try:
        before=_observe(args.target)
        check=_observe(args.target)
        stable=math.dist(_center_vector(before),_center_vector(check))
        if stable>15: raise RuntimeError('target not stable before yaw trial')
        before=check
        record('initial',observation=before,log_path=str(path))
        guard=ScanGuard()
        guard.wait_ready()
        speeds=((args.single_speed,) if args.single_speed else
                ((1600,2100,2400) if args.extended else (900,1600,2100)))
        for speed in speeds:
            feedback={}
            _pulse(GRAB_URL,guard,args.action,args.hold_s,turn_mrad_s=speed,
                   feedback_sink=feedback)
            after=_observe(args.target)
            a,b=_center_vector(before),_center_vector(after)
            delta=tuple(y-x for x,y in zip(a,b))
            record('trial',action=args.action,angular_mrad_s=speed,on_time_s=args.hold_s,
                   delta_px=delta,observation=after,pulse_feedback=feedback)
            if math.dist(a,b)>100: raise RuntimeError('yaw step exceeded visual limit')
            sign=1 if args.action=='turn_left' else -1
            body_flow=feedback.get('visual_feedback') or {}
            background=body_flow.get('movement_px') or {}
            if (feedback.get('termination')=='visual_motion_limit'
                    and len(background)==2
                    and min(sign*delta[0],sign*delta[2])>=4
                    and max(abs(delta[1]),abs(delta[3]))<30):
                record('verified',angular_mrad_s=speed,
                       message='dual-eye consistent lateral image movement; not metric yaw')
                return 0
            if min(sign*delta[0],sign*delta[2])<=-8:
                raise RuntimeError('measured yaw sign contradicts command')
            before=after
        raise RuntimeError('no confirmed yaw progress at bounded test settings')
    except Exception as exc:
        record('stopped',reason=str(exc))
        return 2
    finally:
        if guard is not None: guard.close()

if __name__=='__main__': raise SystemExit(main())
