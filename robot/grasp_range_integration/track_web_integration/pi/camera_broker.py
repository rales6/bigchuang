"""Local single-owner V4L2 fan-out and bounded whole-task dual recording."""
import argparse
import json
import os
from pathlib import Path
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs

from .task_video import VideoRecorder


class CameraBroker:
    def __init__(self,root,cv,devices):
        self.cv, self.devices=cv,devices
        self.condition=threading.Condition();self.frames={};self.errors={}
        self.stopped=threading.Event()
        self.recorder=VideoRecorder(root/'final_work/task_videos',cv)
        self.threads=[threading.Thread(target=self.capture,args=(side,index),daemon=True)
                      for side,index in devices.items()]
        for thread in self.threads:thread.start()

    def capture(self,side,index):
        while not self.stopped.is_set():
            cap=self.cv.VideoCapture(index,self.cv.CAP_V4L2)
            try:
                cap.set(self.cv.CAP_PROP_FOURCC,self.cv.VideoWriter_fourcc(*'MJPG'))
                cap.set(self.cv.CAP_PROP_FRAME_WIDTH,1280);cap.set(self.cv.CAP_PROP_FRAME_HEIGHT,720)
                cap.set(self.cv.CAP_PROP_FPS,15);cap.set(self.cv.CAP_PROP_BUFFERSIZE,3)
                if not cap.isOpened():raise RuntimeError('V4L2 camera unavailable')
                failures=0
                while not self.stopped.is_set():
                    ok,frame=cap.read()
                    if not ok or frame is None:
                        failures+=1
                        if failures>=15:raise RuntimeError('camera lost frames')
                        self.stopped.wait(.05);continue
                    if frame.shape != (720,1280,3):
                        raise RuntimeError('实际双摄分辨率不符合1280x720模型契约')
                    failures=0
                    with self.condition:
                        seq=(self.frames.get(side) or {}).get('sequence',0)+1
                        self.frames[side]=dict(frame=frame,raw=frame.tobytes(),sequence=seq,
                                              captured_s=time.monotonic())
                        self.errors.pop(side,None);self.condition.notify_all()
            except Exception as exc:
                with self.condition:self.errors[side]=str(exc)
            finally:cap.release()
            self.stopped.wait(1)

    def snapshot(self):
        with self.condition:
            cameras={str(index):dict(ready=side in self.frames and
                0 <= time.monotonic()-self.frames[side]['captured_s'] <= .5,
                error=self.errors.get(side,'')) for side,index in self.devices.items()}
        return dict(ready=all(v['ready'] for v in cameras.values()),cameras=cameras,
                    recording=self.recorder.status())

    def frame(self,index,after):
        side=next((s for s,i in self.devices.items() if i==index),None)
        if side is None:return None
        deadline=time.monotonic()+1
        with self.condition:
            while not self.stopped.is_set():
                entry=self.frames.get(side)
                if entry and entry['sequence']>after and 0<=time.monotonic()-entry['captured_s']<=.5:
                    return entry
                remaining=deadline-time.monotonic()
                if remaining<=0:return None
                self.condition.wait(min(.1,remaining))

    def record_loop(self):
        while not self.stopped.wait(.02):
            with self.condition:frames=dict(self.frames)
            self.recorder.tick(frames)

    def close(self):
        self.stopped.set()
        status=self.recorder.status()
        if status.get('recording'):self.recorder.finish(status['session_id'],'camera_service_stopped')
        for thread in self.threads:thread.join(timeout=2)


def handler(broker):
    class Handler(BaseHTTPRequestHandler):
        def reply(self,value,code=200):
            raw=json.dumps(value,ensure_ascii=False).encode()
            self.send_response(code);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            parsed=urlparse(self.path)
            if parsed.path=='/status':return self.reply(broker.snapshot())
            if not parsed.path.startswith('/frame/'):return self.reply(dict(error='not_found'),404)
            try:
                frame=broker.frame(int(parsed.path.rsplit('/',1)[1]),
                    int((parse_qs(parsed.query).get('after') or ['0'])[0]))
                if frame is None:return self.reply(dict(error='no_fresh_frame'),503)
                self.send_response(200)
                self.send_header('Content-Type','application/octet-stream')
                for name,value in [('Sequence',frame['sequence']),('Captured-Monotonic',frame['captured_s']),
                        ('Width',frame['frame'].shape[1]),('Height',frame['frame'].shape[0])]:
                    self.send_header('X-'+name,str(value))
                self.send_header('Content-Length',str(len(frame['raw'])))
                self.end_headers();self.wfile.write(frame['raw'])
            except (BrokenPipeError,ConnectionResetError):pass
            except ValueError:self.reply(dict(error='invalid_frame_request'),400)
        def do_POST(self):
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096:raise ValueError('payload size')
                data=json.loads(self.rfile.read(length))
                if not isinstance(data,dict):raise ValueError('payload must be an object')
                session=str(data.get('session_id') or '')
                if not session or len(session)>160:raise ValueError('session_id')
                if self.path=='/recording/start':
                    if not broker.snapshot()['ready']:raise RuntimeError('both cameras not ready')
                    result=broker.recorder.start(session,str(data.get('instruction') or '')[:500])
                elif self.path=='/recording/finish':
                    result=broker.recorder.finish(session,str(data.get('reason') or 'finished')[:500])
                else:return self.reply(dict(error='not_found'),404)
                self.reply(dict(ok=True,**result))
            except (ValueError,RuntimeError,OSError) as exc:self.reply(dict(ok=False,error=str(exc)),400)
        def log_message(self,*args):pass
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track-root',type=Path,required=True)
    args=parser.parse_args()
    import cv2
    from pursuit.camera_contract import model_devices
    left,right=model_devices(require_ports=True)
    broker=CameraBroker(args.track_root,cv2,{'left':left,'right':right})
    server=ThreadingHTTPServer(('127.0.0.1',18881),handler(broker))
    marker=args.track_root/'final_work/data/camera_broker.json'
    marker.parent.mkdir(parents=True,exist_ok=True)
    marker.write_text(json.dumps(dict(owner='final_work.camera_broker.v1',url='http://127.0.0.1:18881',
                                     pid=os.getpid())),encoding='utf-8')
    thread=threading.Thread(target=broker.record_loop,daemon=True);thread.start()
    def stop(*args):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:server.serve_forever(poll_interval=.1)
    except KeyboardInterrupt:pass
    finally:
        broker.close();server.server_close();thread.join(timeout=2)
        if marker.is_file():marker.unlink()


if __name__=='__main__':main()
