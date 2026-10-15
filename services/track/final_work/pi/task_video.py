"""One task's bounded videos; fixed-file retention, no recursive deletion."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from urllib.request import Request, urlopen

OWNER = 'final_work.task_video.v1'
FILES = ('left.avi', 'right.avi', 'timeline.jsonl', 'session.json')
URL = 'http://127.0.0.1:18881'


def clean_latest(root):
    root = Path(root)
    directory = root / 'latest'
    if root.name != 'task_videos' or root.is_symlink() or directory.is_symlink():
        raise RuntimeError('拒绝清理非任务专用录像目录')
    root.mkdir(parents=True,exist_ok=True)
    directory.mkdir(exist_ok=True)
    if directory.resolve().parent != root.resolve():
        raise RuntimeError('录像目录越界')
    owned = directory / 'session.json'
    existing = [directory/name for name in FILES
                if (directory/name).exists() or (directory/name).is_symlink()]
    if existing and (not owned.is_file() or owned.is_symlink()
            or json.loads(owned.read_text(encoding='utf-8')).get('owner') != OWNER):
        raise RuntimeError('旧录像缺少本功能所有权标记；未删除或覆盖')
    for file in existing:
        if file.is_symlink() or file.resolve().parent != directory.resolve():
            raise RuntimeError('录像文件越界，未清理')
    for file in existing:
        file.unlink()
    return directory, [file.name for file in existing]


class VideoRecorder:
    def __init__(self, root, cv, *, fps=5, maximum_bytes=512*1024*1024,
                 minimum_free_bytes=256*1024*1024):
        self.root, self.cv, self.fps = Path(root), cv, fps
        self.maximum_bytes, self.minimum_free_bytes = maximum_bytes, minimum_free_bytes
        self.lock = threading.RLock()
        self.writers = {}; self.timeline = None; self.metadata = {}
        self.frame_index = 0; self.next_frame_s = 0

    def _save(self):
        if self.metadata:
            path = self.root/'latest/session.json'
            path.write_text(json.dumps(self.metadata,ensure_ascii=False,indent=2),encoding='utf-8')

    def start(self, session_id, instruction):
        with self.lock:
            if self.metadata.get('recording'):
                if self.metadata['session_id'] == session_id:
                    return self.status()
                raise RuntimeError('上一轮任务仍在录制，不能覆盖正在写入的视频')
            directory, removed = clean_latest(self.root)
            self.metadata = dict(owner=OWNER,session_id=session_id,instruction=instruction,
                recording=False,started_at=time.time(),fps=self.fps,size=[640,360],
                removed_previous_files=removed,frames=0,paths={s:str(directory/(s+'.avi')) for s in ('left','right')})
            self._save()
            try:
                for side in ('left','right'):
                    writer = self.cv.VideoWriter(str(directory/(side+'.avi')),
                        self.cv.VideoWriter_fourcc(*'MJPG'),self.fps,(640,360))
                    if not writer.isOpened():
                        writer.release()
                        raise RuntimeError('无法创建 '+side+' 录像')
                    self.writers[side] = writer
                self.timeline = (directory/'timeline.jsonl').open('w',encoding='utf-8')
                self.frame_index=0; self.next_frame_s=time.monotonic()
                self.metadata['recording']=True
                self._save()
            except Exception as exc:
                self.finish(session_id, 'writer_error: '+str(exc))
                raise
            return self.status()

    def status(self):
        with self.lock:
            return dict(self.metadata)

    def finish(self, session_id, reason):
        with self.lock:
            if not session_id or self.metadata.get('session_id') != session_id:
                return dict(ignored=True,reason='another_session')
            for writer in self.writers.values():
                writer.release()
            self.writers.clear()
            if self.timeline is not None:
                self.timeline.close(); self.timeline=None
            self.metadata.update(recording=False,finished_at=time.time(),finish_reason=reason,
                                 frames=self.frame_index)
            self._save()
            return self.status()

    def tick(self, frames, *, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if not self.metadata.get('recording') or now < self.next_frame_s:
                return
            try:
                used=sum((self.root/'latest'/name).stat().st_size for name in FILES
                         if (self.root/'latest'/name).is_file())
                if used >= self.maximum_bytes or shutil.disk_usage(self.root).free < self.minimum_free_bytes:
                    self.finish(self.metadata['session_id'],'storage_budget_reached; cameras_continue')
                    return
                import numpy as np
                times = {}
                for side in ('left','right'):
                    entry = frames.get(side)
                    frame = entry['frame'] if entry else np.zeros((360,640,3),np.uint8)
                    picture = self.cv.resize(frame,(640,360))
                    stale = entry is None or not 0 <= now-entry['captured_s'] <= .5
                    self.cv.putText(picture, f"{self.frame_index/self.fps:.1f}s {side}"+(' CAMERA STALE' if stale else ''),
                        (8,24),self.cv.FONT_HERSHEY_SIMPLEX,.55,(0,0,255) if stale else (255,255,255),1)
                    self.writers[side].write(picture)
                    times[side] = dict(sequence=entry['sequence'] if entry else None,
                        captured_s=entry['captured_s'] if entry else None,stale=stale)
                self.timeline.write(json.dumps(dict(frame=self.frame_index,wall_time=time.time(),
                    elapsed_s=now-self.next_frame_s+self.frame_index/self.fps,cameras=times))+'\n')
                self.frame_index += 1
                self.next_frame_s += 1/self.fps
                self.metadata['frames']=self.frame_index
                if self.frame_index % self.fps == 0:
                    self.timeline.flush(); self._save()
            except Exception as exc:
                self.finish(self.metadata['session_id'],'recording_error: '+str(exc))


class TaskVideoManager:
    """Bridge-owned daemon lifecycle; cameras survive preview handoffs."""
    def __init__(self, track_root, python):
        self.track_root, self.python = Path(track_root), str(python)
        self.process=None; self.log=None; self.lock=threading.RLock()

    def request(self,path,payload=None,timeout=3):
        req=Request(URL+path,data=None if payload is None else json.dumps(payload).encode(),
                    headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=timeout) as response:
            return json.load(response)

    def ensure_camera_owner(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return
            if self.process is not None:
                self.close()
            from .motion_session import _stop_preview
            _stop_preview()
            root=self.track_root/'final_work/task_videos'
            root.mkdir(parents=True,exist_ok=True)
            env=dict(os.environ)
            env.pop('FINAL_WORK_CAMERA_BROKER_URL',None)
            # The daemon MUST open physical cameras, not recursively proxy itself.
            env['PYTHONPATH']=str(self.track_root)
            self.log=(root/'broker.log').open('w',encoding='utf-8')
            self.process=subprocess.Popen([self.python,'-u','-m','final_work.pi.camera_broker',
                '--track-root',str(self.track_root)],cwd=str(self.track_root),env=env,
                stdout=self.log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+15
            while time.monotonic()<deadline:
                if self.process.poll() is not None:
                    self.close()
                    from .motion_session import _restore_preview
                    _restore_preview()
                    raise RuntimeError('双摄共享采集进程未能启动，详情见 task_videos/broker.log')
                try:
                    status=self.request('/status')
                    if status.get('ready') is True:
                        from .motion_session import _restore_preview
                        _restore_preview()
                        return
                except Exception:
                    pass
                time.sleep(.2)
            self.close()
            from .motion_session import _restore_preview
            _restore_preview()
            raise RuntimeError('双摄共享采集未在15秒内获得两路新画面')

    def start(self,session_id,instruction):
        self.ensure_camera_owner()
        return self.request('/recording/start',dict(session_id=session_id,instruction=instruction))

    def finish(self,session_id,reason):
        return self.request('/recording/finish',dict(session_id=session_id,reason=reason))

    def status(self):
        if self.process is None or self.process.poll() is not None:
            return dict(recording=False,available=False)
        try:
            return self.request('/status').get('recording') or {}
        except Exception as exc:
            return dict(recording=False,error=str(exc))

    def close(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=2)
            if self.log is not None:self.log.close();self.log=None
            marker=self.track_root/'final_work/data/camera_broker.json'
            if marker.is_file():
                data=json.loads(marker.read_text(encoding='utf-8'))
                if (data.get('owner')=='final_work.camera_broker.v1' and self.process is not None
                        and data.get('pid')==self.process.pid):
                    marker.unlink()
