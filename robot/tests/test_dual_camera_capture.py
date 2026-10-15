"""Stopped dual-UVC recovery tests; no hardware or motion outputs."""
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from raspberry_pi.map_camera_inspection import CameraCapture


class FakeCamera:
    def __init__(self, clock, grabs=(), frame_value=100, grab_delay=.05, retrieves=()):
        self.clock = clock
        self.grabs = list(grabs)
        self.retrieves = list(retrieves)
        self.value = frame_value
        self.delay = grab_delay
        self.released = 0

    def grab(self):
        self.clock[0] += self.delay
        return self.grabs.pop(0) if self.grabs else True

    def retrieve(self):
        self.clock[0] += .01
        if self.retrieves and not self.retrieves.pop(0):
            return False, None
        return True, np.full((8, 8, 3), self.value, dtype=np.uint8)

    def get(self, prop):
        return {1: 1280, 2: 720, 3: 30, 4: 1196444237}.get(prop, 0)

    def release(self):
        self.released += 1

    def isOpened(self):
        return self.released == 0


CV = SimpleNamespace(CAP_PROP_FRAME_WIDTH=1, CAP_PROP_FRAME_HEIGHT=2,
                     CAP_PROP_FPS=3, CAP_PROP_FOURCC=4)


class DualCameraCaptureTests(unittest.TestCase):
    def test_dual_open_uses_three_buffers_without_changing_resolution(self):
        settings = []
        cap = SimpleNamespace(set=lambda prop, value: settings.append((prop, value)),
                              isOpened=lambda: True, release=lambda: None)
        fake_cv = SimpleNamespace(**vars(CV), CAP_V4L2=200, CAP_PROP_BUFFERSIZE=5,
            VideoCapture=lambda *_a: cap, VideoWriter_fourcc=lambda *_a: 1196444237)
        helper = CameraCapture('opencv', 0, 1280, 720, 12., dual_usb_cameras=True)
        with patch.dict('sys.modules', {'cv2': fake_cv}):
            self.assertIs(helper._open_opencv_device(0, '/dev/video0'), cap)
        self.assertIn((5, 3), settings)
        self.assertIn((1, 1280), settings)
        self.assertIn((2, 720), settings)

    def collect(self, cameras, clock, opener=None, timeout=1., keep_open=False):
        helper = CameraCapture('opencv', 0, 1280, 720, timeout, dual_usb_cameras=True, keep_open=keep_open)
        with patch('raspberry_pi.map_camera_inspection.time.monotonic', side_effect=lambda: clock[0]), patch(
                'raspberry_pi.map_camera_inspection.time.sleep', side_effect=lambda s: clock.__setitem__(0, clock[0] + s)), patch.object(
                helper, '_opencv_device_index', side_effect=lambda p: (int(p[-1]), p)), patch.object(
                helper, '_open_opencv_device', side_effect=opener or (lambda *_a: None)), contextlib.redirect_stdout(io.StringIO()):
            return helper._collect_dual_frames(cameras, ['/dev/video0', '/dev/video2'], CV)

    def test_alternating_grab_failures_keep_each_successful_frame(self):
        clock = [0.]
        cameras = [FakeCamera(clock, [True, False], 90), FakeCamera(clock, [False, True], 150)]
        frames, diagnostic = self.collect(cameras, clock)
        self.assertEqual([int(frame[0, 0, 0]) for frame in frames], [90, 150])
        self.assertFalse(diagnostic['hardware_synchronized'])
        self.assertLessEqual(diagnostic['grab_completion_skew_s'], .75)
        self.assertEqual([s['reopen_count'] for s in diagnostic['streams']], [0, 0])

    def test_decodes_first_stream_before_grabbing_second(self):
        clock, order = [0.], []
        class OrderedCamera(FakeCamera):
            def __init__(self, index):
                super().__init__(clock)
                self.index = index
            def grab(self):
                order.append(('grab', self.index))
                return super().grab()
            def retrieve(self):
                order.append(('retrieve', self.index))
                return super().retrieve()
        self.collect([OrderedCamera(0), OrderedCamera(1)], clock)
        self.assertEqual(order, [('grab', 0), ('retrieve', 0), ('grab', 1), ('retrieve', 1)])

    def test_blocking_grab_past_deadline_cannot_return_a_pair(self):
        clock = [0.]
        cameras = [FakeCamera(clock, grab_delay=2.), FakeCamera(clock)]
        with self.assertRaisesRegex(RuntimeError, '双摄像头取帧超时'):
            self.collect(cameras, clock, timeout=1.)

    def test_persistent_capture_drains_queued_old_frames(self):
        clock = [0.]
        class QueuedCamera(FakeCamera):
            def __init__(self):
                super().__init__(clock)
                self.values = [10, 20, 30, 180]
            def retrieve(self):
                self.value = self.values.pop(0)
                return super().retrieve()
        frames, diagnostic = self.collect([QueuedCamera(), QueuedCamera()], clock, keep_open=True)
        self.assertEqual([int(f[0, 0, 0]) for f in frames], [180, 180])
        self.assertEqual(diagnostic['required_fresh_frame_count'], 4)
        self.assertEqual([s['frame_count'] for s in diagnostic['streams']], [4, 4])

    def test_persistent_capture_reuses_streams_and_explicit_close_releases(self):
        clock = [0.]
        caps = [FakeCamera(clock), FakeCamera(clock)]
        helper = CameraCapture('opencv', 0, 1280, 720, 1., dual_usb_cameras=True, keep_open=True)
        fake_cv = SimpleNamespace(**vars(CV), imwrite=lambda *_a: True)
        quality = {'brightness': 100, 'contrast': 20, 'dark_fraction': 0,
                   'edge_fraction': .1, 'possibly_blurred_or_flat': False}
        with patch.dict('sys.modules', {'cv2': fake_cv}), patch.object(
                helper, '_stable_usb_capture_nodes', return_value=['/dev/video0', '/dev/video2']), patch.object(
                helper, '_opencv_device_index', side_effect=lambda p: (int(p[-1]), p)), patch.object(
                helper, '_open_opencv_device', side_effect=caps) as opener, patch.object(
                helper, '_image_quality', return_value=quality), patch(
                'raspberry_pi.map_camera_inspection.time.monotonic', side_effect=lambda: clock[0]), patch(
                'raspberry_pi.map_camera_inspection.time.sleep', side_effect=lambda s: clock.__setitem__(0, clock[0] + s)):
            for i in range(3):
                result = helper._opencv_capture_dual(Path('unused%d.jpg' % i))
                self.assertEqual(result['camera_count'], 2)
            self.assertEqual(opener.call_count, 2)
            self.assertEqual([c.released for c in caps], [0, 0])
        helper.close()
        helper.close()
        self.assertEqual([c.released for c in caps], [1, 1])

    def test_one_retrieve_failure_does_not_discard_other_stream(self):
        clock = [0.]
        cameras = [FakeCamera(clock, retrieves=[True, False]), FakeCamera(clock, retrieves=[False, True])]
        _, diagnostic = self.collect(cameras, clock)
        self.assertEqual([s['retrieve_failures'] for s in diagnostic['streams']], [1, 1])

    def test_reopens_only_failed_stream_once_and_recovers(self):
        clock = [0.]
        healthy = FakeCamera(clock)
        failed = FakeCamera(clock, [False, False])
        recovered = FakeCamera(clock, frame_value=160)
        cameras = [healthy, failed]
        calls = []
        frames, diagnostic = self.collect(cameras, clock,
            opener=lambda index, device: (calls.append((index, device)) or recovered))
        self.assertEqual(calls, [(2, '/dev/video2')])
        self.assertEqual(failed.released, 1)
        self.assertEqual(healthy.released, 0)
        self.assertEqual(int(frames[1][0, 0, 0]), 160)
        self.assertEqual(diagnostic['streams'][1]['reopen_count'], 1)

    def test_stale_frame_cannot_satisfy_pair_after_other_stream_is_late(self):
        clock = [0.]
        cameras = [FakeCamera(clock, [True] + [False] * 20), FakeCamera(clock, [False, True], grab_delay=.9)]
        with self.assertRaisesRegex(RuntimeError, '双摄像头取帧超时'):
            self.collect(cameras, clock, timeout=2.)

    def test_camera_that_never_recovers_is_not_reopened_indefinitely(self):
        clock = [0.]
        cameras = [FakeCamera(clock), FakeCamera(clock, [False] * 30)]
        replacement = FakeCamera(clock, [False] * 30)
        calls = []
        with self.assertRaisesRegex(RuntimeError, 'grab_failures'):
            self.collect(cameras, clock, opener=lambda *_a: (calls.append(True) or replacement))
        self.assertEqual(len(calls), 1)

    def test_failed_reopen_does_not_mask_timeout_with_none_release_error(self):
        clock = [0.]
        healthy, bad = FakeCamera(clock), FakeCamera(clock, [False] * 30)
        helper = CameraCapture('opencv', 0, 1280, 720, 1., dual_usb_cameras=True)
        with patch.dict('sys.modules', {'cv2': CV}), patch.object(
                helper, '_stable_usb_capture_nodes', return_value=['/dev/video0', '/dev/video2']), patch.object(
                helper, '_opencv_device_index', side_effect=lambda p: (int(p[-1]), p)), patch.object(
                helper, '_open_opencv_device', side_effect=[healthy, bad, None]), patch(
                'raspberry_pi.map_camera_inspection.time.monotonic', side_effect=lambda: clock[0]), patch(
                'raspberry_pi.map_camera_inspection.time.sleep', side_effect=lambda s: clock.__setitem__(0, clock[0] + s)), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'reopen_failed'):
                helper._opencv_capture_dual(Path('unused.jpg'))
        self.assertEqual(healthy.released, 1)
        self.assertEqual(bad.released, 1)

    def test_disconnected_device_resolution_preserves_stream_diagnostics(self):
        clock = [0.]
        cameras = [FakeCamera(clock), FakeCamera(clock, [False] * 30)]
        helper = CameraCapture('opencv', 0, 1280, 720, 1., dual_usb_cameras=True)
        with patch('raspberry_pi.map_camera_inspection.time.monotonic', side_effect=lambda: clock[0]), patch(
                'raspberry_pi.map_camera_inspection.time.sleep', side_effect=lambda s: clock.__setitem__(0, clock[0] + s)), patch.object(
                helper, '_opencv_device_index', side_effect=RuntimeError('USB device disconnected')):
            with self.assertRaisesRegex(RuntimeError, 'reopen_failed: USB device disconnected'):
                helper._collect_dual_frames(cameras, ['/dev/video0', '/dev/video2'], CV)
        self.assertIsNone(cameras[1])


if __name__ == '__main__':
    unittest.main()
