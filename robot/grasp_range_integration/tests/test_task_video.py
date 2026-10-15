import copy
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer
import cv2
import numpy as np
from grasp_range_integration.track_web_integration.pi.task_video import VideoRecorder,clean_latest,OWNER
from grasp_range_integration.track_web_integration.pi.camera_proxy import SharedCapture,camera_environment
from grasp_range_integration.track_web_integration.pi.camera_broker import handler
from grasp_range_integration.track_web_integration.gpu.task_recording import (
    prepare_recording,start_recording,finish_recording,persist_finished_recording)


class Recording(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'task_videos'
        self.video=VideoRecorder(self.root,cv2,minimum_free_bytes=0)
        self.addCleanup(lambda:self.video.finish(self.video.metadata.get('session_id'),'test_end'))

    def frame(self,now):
        return {side:dict(frame=np.full((720,1280,3),120,np.uint8),sequence=1,captured_s=now)
                for side in ('left','right')}

    def test_stereo_video_roundtrip_and_new_round_retention(self):
        self.video.start('one','找黄色方块');start=self.video.next_frame_s
        for i in range(3):self.video.tick(self.frame(start+i*.21),now=start+i*.21)
        self.video.finish('one','completed')
        for side in ('left','right'):
            cap=cv2.VideoCapture(str(self.root/'latest'/f'{side}.avi'))
            try:
                self.assertTrue(cap.isOpened());self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT),3)
                ok,frame=cap.read();self.assertTrue(ok);self.assertEqual(frame.shape,(360,640,3))
            finally:cap.release()
        unrelated=self.root/'latest/photo.jpg';unrelated.write_bytes(b'preserve')
        new=self.video.start('two','another task')
        self.assertEqual(new['removed_previous_files'],['left.avi','right.avi','timeline.jsonl','session.json'])
        self.assertEqual(unrelated.read_bytes(),b'preserve');self.assertEqual(new['frames'],0)

    def test_same_active_round_does_not_erase_clip(self):
        self.video.start('one','x');now=self.video.next_frame_s
        self.video.tick(self.frame(now),now=now)
        self.assertEqual(self.video.start('one','x')['frames'],1)
        with self.assertRaises(RuntimeError):self.video.start('two','x')
        self.assertTrue(self.video.status()['recording'])
        self.assertTrue(self.video.finish('wrong','completed')['ignored'])

    def test_unowned_files_are_never_deleted(self):
        directory=self.root/'latest';directory.mkdir(parents=True)
        file=directory/'left.avi';file.write_bytes(b'private')
        with self.assertRaises(RuntimeError):clean_latest(self.root)
        self.assertEqual(file.read_bytes(),b'private')

    def test_wrong_root_never_cleaned(self):
        with self.assertRaises(RuntimeError):clean_latest(Path(self.tmp.name))

    def test_storage_limit_only_stops_recording(self):
        self.video.start('one','x');self.video.maximum_bytes=0
        now=self.video.next_frame_s;self.video.tick(self.frame(now),now=now)
        self.assertFalse(self.video.status()['recording'])
        self.assertIn('cameras_continue',self.video.status()['finish_reason'])

    def test_stale_or_missing_camera_is_visible_in_timeline(self):
        self.video.start('one','x');now=self.video.next_frame_s
        frames=self.frame(now-2);frames.pop('right')
        self.video.tick(frames,now=now);self.video.finish('one','done')
        row=json.loads((self.root/'latest/timeline.jsonl').read_text())
        self.assertTrue(row['cameras']['left']['stale']);self.assertTrue(row['cameras']['right']['stale'])


class SharedCamera(unittest.TestCase):
    def test_real_local_http_delivers_full_geometry_and_never_usb(self):
        now=time.monotonic()
        frame=dict(frame=np.zeros((720,1280,3),np.uint8),raw=bytes(720*1280*3),
                   sequence=2,captured_s=now)
        broker=NS(snapshot=lambda:dict(ready=True,cameras={'2':{'ready':True}}),
                  frame=lambda index,after:frame if after<2 else None)
        server=ThreadingHTTPServer(('127.0.0.1',0),handler(broker))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            cap=SharedCapture(2,url=f'http://127.0.0.1:{server.server_port}',cv=cv2)
            self.assertTrue(cap.isOpened());ok,picture=cap.read()
            self.assertTrue(ok);self.assertEqual(picture.shape,(720,1280,3))
            self.assertFalse(cap.read()[0]);cap.release();self.assertFalse(cap.isOpened())
        finally:server.shutdown();server.server_close();thread.join(timeout=2)

    def test_frame_stale_after_slow_response_is_rejected(self):
        response=NS(headers={'X-Sequence':'2','X-Captured-Monotonic':'10','X-Width':'16','X-Height':'16'},
                    read=lambda n:bytes(16*16*3))
        from contextlib import nullcontext
        prefix='grasp_range_integration.track_web_integration.pi.camera_proxy'
        with patch(prefix+'.urlopen',return_value=nullcontext(response)), \
             patch(prefix+'.time.monotonic',side_effect=[10.1,10.8]):
            cap=SharedCapture(2,url='x',cv=cv2)
            self.assertFalse(cap.read()[0]);self.assertEqual(cap.sequence,0)

    def test_environment_hook_requires_owned_fixed_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/'final_work/data/camera_broker.json';marker.parent.mkdir(parents=True)
            base=dict(FINAL_WORK_TRACK_ROOT=tmp)
            self.assertNotIn('FINAL_WORK_CAMERA_BROKER_URL',camera_environment(base))
            marker.write_text(json.dumps(dict(owner='bad',url='http://elsewhere')))
            with self.assertRaises(RuntimeError):camera_environment(base)
            marker.write_text(json.dumps(dict(owner='final_work.camera_broker.v1',url='http://127.0.0.1:18881')))
            result=camera_environment(base)
            self.assertIn('camera_bootstrap',result['PYTHONPATH'])
            self.assertEqual(result['FINAL_WORK_CAMERA_BROKER_URL'],'http://127.0.0.1:18881')


class TaskLifecycle(unittest.TestCase):
    def test_start_once_for_all_steps_finish_once_at_terminal(self):
        calls=[];robot=NS(request=lambda *a,**kw:calls.append(a) or dict(ok=True))
        execution=prepare_recording({});task=dict(status='running',execution=execution)
        for i in range(8):
            start_recording(robot,execution,'x');self.assertFalse(finish_recording(robot,task))
        self.assertEqual(len(calls),1)
        task['status']='completed';self.assertTrue(finish_recording(robot,task))
        self.assertFalse(finish_recording(robot,task));self.assertEqual(len(calls),2)
        previous=execution['recording']['session_id'];prepare_recording(execution)
        self.assertNotEqual(previous,execution['recording']['session_id'])

    def test_rejected_recording_prevents_dispatch(self):
        execution={};robot=NS(request=lambda *a,**kw:dict(ok=False,error='camera missing'))
        with self.assertRaises(RuntimeError):start_recording(robot,execution,'x')
        self.assertFalse(execution['recording']['started'])

    def test_poll_metadata_merge_cannot_restore_old_step(self):
        task=dict(id='a',status='completed',execution=prepare_recording({}))
        task['execution']['recording']['finished']=True
        latest=copy.deepcopy(task);latest['status']='running';latest['execution']['active_step']='next'
        history=NS(get_task=lambda _:latest,update_task=lambda i,**kw:kw)
        result=persist_finished_recording(history,task)
        self.assertEqual(result['status'],'running');self.assertEqual(result['execution']['active_step'],'next')


if __name__=='__main__':unittest.main()
