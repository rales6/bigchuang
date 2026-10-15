"""Process-local OpenCV adapter for the task's single camera owner.

Only camera nodes are proxied. Movies and unrelated captures retain OpenCV.
A missing/stale broker frame never falls back to opening the USB device.
"""
import json
import os
from pathlib import Path
import time
from urllib.request import urlopen


def camera_environment(base=None):
    env = dict(os.environ if base is None else base)
    root = Path(env.get('FINAL_WORK_TRACK_ROOT', '/home/dachuang/dachuang/track'))
    marker = root / 'final_work/data/camera_broker.json'
    if marker.is_file():
        data = json.loads(marker.read_text(encoding='utf-8'))
        if data.get('owner') != 'final_work.camera_broker.v1' or data.get('url') != 'http://127.0.0.1:18881':
            raise RuntimeError('摄像头共享入口配置无效')
        env['FINAL_WORK_CAMERA_BROKER_URL'] = data['url']
        bootstrap = root / 'final_work/pi/camera_bootstrap'
        # Keep the caller's navigation/core import priority. The bootstrap
        # needs to run first, but track must not shadow car1.0's core again.
        paths = [str(bootstrap), *env.get('PYTHONPATH', '').split(os.pathsep), str(root)]
        env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(p for p in paths if p))
    return env


def camera_index(source):
    if isinstance(source, int):
        return source
    if isinstance(source, (str, Path)) and str(source).startswith('/dev/'):
        name = Path(source).resolve().name
        if name.startswith('video') and name[5:].isdigit():
            return int(name[5:])
    return None


def require_broker_ready():
    env = camera_environment()
    if env.get('FINAL_WORK_CAMERA_BROKER_URL') != 'http://127.0.0.1:18881':
        raise RuntimeError('摄像头共享入口未启用')
    with urlopen(env['FINAL_WORK_CAMERA_BROKER_URL']+'/status', timeout=3) as response:
        status = json.load(response)
    if status.get('ready') is not True:
        raise RuntimeError('双摄共享采集尚无两路新画面')
    return status


class SharedCapture:
    def __init__(self, source, *, url, cv):
        self.index = camera_index(source)
        self.url, self.cv = url, cv
        self.closed = False
        self.sequence = 0
        self.width, self.height, self.fps = 1280, 720, 15
        self.pending = None

    def isOpened(self):
        if self.closed:
            return False
        try:
            with urlopen(self.url+'/status', timeout=1.0) as response:
                status = json.load(response)
            return (status.get('cameras') or {}).get(str(self.index), {}).get('ready') is True
        except Exception:
            return False

    def read(self):
        if self.closed:
            return False, None
        try:
            import numpy as np
            with urlopen(f'{self.url}/frame/{self.index}?after={self.sequence}', timeout=1.5) as response:
                sequence = int(response.headers['X-Sequence'])
                captured = float(response.headers['X-Captured-Monotonic'])
                width, height = int(response.headers['X-Width']), int(response.headers['X-Height'])
                if not (0 <= time.monotonic()-captured <= .5) or sequence <= self.sequence:
                    return False, None
                raw = response.read(16*1024*1024)
            if len(raw) != width*height*3 or not 0 <= time.monotonic()-captured <= .5:
                return False, None
            self.sequence = sequence
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height,width,3).copy()
            if (width, height) != (self.width, self.height):
                frame = self.cv.resize(frame, (self.width,self.height))
            return True, frame
        except Exception:
            return False, None

    def grab(self):
        ok, self.pending = self.read()
        return ok

    def retrieve(self, *args):
        return self.pending is not None, self.pending

    def release(self):
        self.closed = True
        self.pending = None

    def set(self, prop, value):
        for field, key in [('width','CAP_PROP_FRAME_WIDTH'), ('height','CAP_PROP_FRAME_HEIGHT')]:
            if prop == getattr(self.cv,key):
                if not 16 <= int(value) <= 4096:
                    return False
                setattr(self,field,int(value))
        # Physical capture remains 1280x720/15fps MJPG. Client requests never
        # reconfigure another reader's sensor or degrade taught geometry.
        return True

    def get(self, prop):
        return {self.cv.CAP_PROP_FRAME_WIDTH:self.width,
                self.cv.CAP_PROP_FRAME_HEIGHT:self.height,
                self.cv.CAP_PROP_FPS:self.fps}.get(prop,0.)


def install_camera_proxy():
    url = os.environ.get('FINAL_WORK_CAMERA_BROKER_URL')
    if not url:
        return False
    if url != 'http://127.0.0.1:18881':
        raise RuntimeError('共享摄像头只允许本机固定入口')
    install_preview_pipe_proxy()
    import cv2
    if getattr(cv2.VideoCapture, '_final_work_proxy', False):
        return True
    original = cv2.VideoCapture
    def capture(source=0, *args, **kwargs):
        if camera_index(source) is not None:
            return SharedCapture(source,url=url,cv=cv2)
        return original(source,*args,**kwargs)
    capture._final_work_proxy = True
    cv2.VideoCapture = capture
    return True


def install_preview_pipe_proxy():
    """Adapt grab_visual's exact MJPEG pipe without changing its frozen Cam.

    Cam uses ffmpeg, not cv2.VideoCapture. Only its known V4L2-to-stdout
    command is replaced; unrelated encoders, movies and processes stay intact.
    The replacement is a real subprocess with the same stdout/terminate API.
    """
    import subprocess
    import sys
    if getattr(subprocess.Popen, '_final_work_camera_pipe', False):
        return
    original = subprocess.Popen

    def popen(args, *positional, **kwargs):
        if (not kwargs.get('shell') and isinstance(args, (list, tuple)) and len(args) == 16
                and Path(str(args[0])).name == 'ffmpeg'
                and list(args[1:6]) == ['-f','v4l2','-input_format','mjpeg','-video_size']
                and args[7] == '-framerate' and args[9] == '-i'
                and list(args[11:]) == ['-c:v','copy','-f','mjpeg','pipe:1']
                and camera_index(args[10]) is not None):
            # Preserve requested image size and camera identity. A missing
            # broker is an explicit shared-reader failure, never USB fallback.
            args = [sys.executable, '-u', '-m', 'final_work.pi.shared_mjpeg_pipe',
                    '--camera', str(args[10]), '--size', str(args[6]), '--fps', str(args[8])]
        return original(args, *positional, **kwargs)

    popen._final_work_camera_pipe = True
    subprocess.Popen = popen
