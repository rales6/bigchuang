"""Optional task-only Qwen photo observer; never starts a server or moves robot."""
from pathlib import Path
import json
import urllib.request
import uuid

from raspberry_pi.task_semantic_localization import detections_from_qwen


class _JsonResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass  # urllib raises HTTPError before construction for failed status.

    def json(self):
        return json.loads(self.body.decode('utf-8'))


class _GroundHttpSession:
    """Small multipart client without requests dependency or inherited proxy."""
    def __init__(self):
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def post(self, url, *, files, data, timeout):
        boundary = 'task-ground-' + uuid.uuid4().hex
        parts = []
        for name, value in data.items():
            parts.append(('--' + boundary + '\r\nContent-Disposition: form-data; name="'
                          + name + '"\r\n\r\n' + str(value) + '\r\n').encode('utf-8'))
        for name, (filename, stream, content_type) in files.items():
            parts.append(('--' + boundary + '\r\nContent-Disposition: form-data; name="'
                          + name + '"; filename="' + filename + '"\r\nContent-Type: '
                          + content_type + '\r\n\r\n').encode('utf-8'))
            parts.extend([stream.read(), b'\r\n'])
        parts.append(('--' + boundary + '--\r\n').encode('ascii'))
        request = urllib.request.Request(url, data=b''.join(parts), method='POST',
            headers={'Content-Type': 'multipart/form-data; boundary=' + boundary})
        with self.opener.open(request, timeout=timeout) as response:
            return _JsonResponse(response.read())


class QwenTaskPhotoObserver:
    def __init__(self, ground_url, timeout_s=12., max_queries=3, session=None):
        if timeout_s <= 0 or max_queries < 1:
            raise ValueError('Invalid inference budget')
        if session is None:
            session = _GroundHttpSession()
        self.url, self.timeout_s, self.max_queries, self.session = ground_url, timeout_s, max_queries, session

    def observe(self, photo_path, camera_index, calibration, anchors):
        """Query at most three UNIQUE verified classes, with per-request timeout.

        Run on a worker, not the 5–10 Hz motion/UI loop. Caller must preserve
        capture timestamp/pose and reject results after vehicle movement.
        """
        from PIL import Image
        path = Path(photo_path)
        with Image.open(path) as image:
            width, height = image.size
        if width != calibration.width_px or not calibration.verified:
            return {'detections': [], 'failures': ['camera_uncalibrated_or_image_resized'], 'http_calls': 0}
        categories = {}
        for anchor in anchors:
            if (anchor.get('position_verified') is True and anchor.get('localization_allowed') is True
                    and anchor.get('mobility') == 'fixed' and anchor.get('behind_glass') is not True):
                categories.setdefault(anchor['category'], []).append(anchor)
        selected = [(category, items[0]) for category, items in sorted(categories.items()) if len(items) == 1]
        objects, failures = [], []
        calls = 0
        for category, anchor in selected[:self.max_queries]:
            calls += 1
            try:
                with path.open('rb') as stream:
                    response = self.session.post(self.url,
                        files={'image': ('task_photo.jpg', stream, 'image/jpeg')},
                        data={'instruction': '找出照片中可见的' + str(anchor.get('name', category))
                              + '，返回紧致框；没有或不能确定就返回未找到，不要猜测。'},
                        timeout=self.timeout_s)
                response.raise_for_status()
                result = response.json()
                if result.get('ok') is not True:
                    raise ValueError('Qwen returned failure')
                candidates = result.get('candidates') or ([result] if result.get('found') is True else [])
                # Multiple detections of a supposedly unique class are ambiguous.
                if len(candidates) != 1:
                    continue
                candidate = candidates[0]
                box = candidate.get('bbox', [])
                if not isinstance(box, list) or len(box) != 4:
                    continue
                objects.append({'category': category, 'confidence': candidate.get('confidence', 0),
                    'views': [{'image_index': camera_index, 'bbox_1000': [
                        box[0] * 1000 / width, box[1] * 1000 / height,
                        box[2] * 1000 / width, box[3] * 1000 / height]}]})
            except Exception as exc:
                failures.append({'category': category, 'error': type(exc).__name__})
        return {'detections': detections_from_qwen(objects, {camera_index: calibration}),
                'failures': failures, 'http_calls': calls}
