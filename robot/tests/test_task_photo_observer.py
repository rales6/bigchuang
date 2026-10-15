from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import io
from PIL import Image

from raspberry_pi.task_photo_observer import QwenTaskPhotoObserver, _GroundHttpSession
from raspberry_pi.task_semantic_localization import CameraCalibration


class TaskPhotoObserverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.photo = Path(self.temp.name) / 'photo.jpg'
        Image.new('RGB', (1000, 500)).save(self.photo)
        self.anchor = {'id': 'sign1', 'category': 'sign', 'name': '出口标牌',
            'position_verified': True, 'localization_allowed': True,
            'mobility': 'fixed', 'behind_glass': False}
        self.camera = CameraCalibration(1000, 500, 500, verified=True)

    def test_valid_grounding_uses_pixel_box_and_does_not_trust_qwen_id(self):
        class Response:
            def raise_for_status(self): pass
            def json(self):
                return {'ok': True, 'found': True, 'bbox': [450, 50, 550, 100],
                        'confidence': .95, 'object_id': 'invented'}
        class Session:
            def post(self, url, **kwargs):
                self.kwargs = kwargs
                return Response()
        session = Session()
        observer = QwenTaskPhotoObserver('http://example.invalid/agent/ground', session=session)
        result = observer.observe(self.photo, 1, self.camera, [self.anchor])
        self.assertEqual(len(result['detections']), 1)
        self.assertEqual(result['detections'][0].bearing_deg, 0)
        self.assertIsNone(result['detections'][0].object_id)
        self.assertEqual(session.kwargs['timeout'], 12)

    def test_repeated_class_and_unverified_camera_do_not_query(self):
        class Session:
            def post(self, *args, **kwargs): raise AssertionError('Unexpected network call')
        observer = QwenTaskPhotoObserver('http://example.invalid', session=Session())
        result = observer.observe(self.photo, 1, self.camera, [self.anchor, dict(self.anchor, id='sign2')])
        self.assertEqual(result['http_calls'], 0)
        result = observer.observe(self.photo, 1, CameraCalibration(1000, 500, 500), [self.anchor])
        self.assertEqual(result['http_calls'], 0)

    def test_timeout_is_partial_failure_not_motor_recovery(self):
        class Session:
            def post(self, *args, **kwargs): raise TimeoutError('test')
        result = QwenTaskPhotoObserver('http://example.invalid', session=Session()).observe(
            self.photo, 1, self.camera, [self.anchor])
        self.assertEqual(result['detections'], [])
        self.assertEqual(result['failures'][0]['error'], 'TimeoutError')

    def test_default_client_needs_no_requests_and_no_queries_for_zero_anchors(self):
        observer = QwenTaskPhotoObserver('http://example.invalid')
        self.assertEqual(observer.observe(self.photo, 1, self.camera, [])['http_calls'], 0)

    def test_stdlib_multipart_includes_chinese_instruction_and_image(self):
        session = _GroundHttpSession()
        response = io.BytesIO(b'{"ok":true,"found":false}')
        with patch.object(session.opener, 'open', return_value=response) as opened:
            result = session.post('http://example.invalid/agent/ground',
                files={'image': ('task_photo.jpg', io.BytesIO(b'JPEGDATA'), 'image/jpeg')},
                data={'instruction': '找插座'}, timeout=12)
        request = opened.call_args[0][0]
        self.assertIn('找插座'.encode('utf-8'), request.data)
        self.assertIn(b'JPEGDATA', request.data)
        self.assertIn(b'name="image"', request.data)
        self.assertEqual(opened.call_args.kwargs['timeout'], 12)
        self.assertFalse(result.json()['found'])


if __name__ == '__main__':
    unittest.main()
