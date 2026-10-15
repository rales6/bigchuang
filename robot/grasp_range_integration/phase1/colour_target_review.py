"""Classify independently pixel-grounded green cues; never generate coordinates."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from urllib import request as urlrequest

from .red_target_deepseek import _configuration, _image_url, _crop_url


def verify_green_cues(images, cues, *, config=None, transport=None):
    result = {"accepted_cameras": [], "same_object": False, "reason": "review_unavailable"}
    try:
        cameras = [c for c in ("left", "right") if c in cues]
        if not cameras:
            return dict(result, reason="no_pixel_candidates")
        for c in cameras:
            cue = cues[c]
            box = [float(v) for v in cue["bbox_1000"]]
            if (cue.get("source") != "phase1_green_visual_cue"
                    or cue.get("complete_in_frame") is not True
                    or len(box) != 4 or not all(math.isfinite(v) for v in box)
                    or not (15 <= box[0] < box[2] <= 985 and 15 <= box[1] < box[3] <= 985)):
                return dict(result, reason="invalid_pixel_candidate")
        paths = {c: Path(images[c]) for c in cameras}
        if len(cameras) == 2 and (paths['left'].resolve() == paths['right'].resolve()
                or hashlib.sha256(paths['left'].read_bytes()).digest()
                == hashlib.sha256(paths['right'].read_bytes()).digest()):
            return dict(result, reason="duplicate_camera_images")
        config = config or _configuration()
        if not config.configured:
            return dict(result, reason="deepseek_not_configured")
        content = []
        for c in cameras:
            content += [{"type": "text", "text": c.upper() + " FULL IMAGE"}, _image_url(paths[c]),
                        {"type": "text", "text": c.upper() + " FIXED PIXEL CROP"},
                        _crop_url(paths[c], cues[c]['bbox_1000'])]
        content.append({"type": "text", "text":
            "分别判断每个相机给定裁剪框中心的绿色区域是否为地面上的绿色小物块。"
            "颜色斑、反光、地板纹理、柜门图案不算物块。只判定给定像素框，不另选框、"
            "不从另一张图片补全缺失物体，不输出坐标。没有提供裁剪的相机必须回答false。"
            "只有两摄均真实看到同一个物块时same_object才为true。输出JSON："
            '{"left":{"matches_target":false,"confidence":0.0},'
            '"right":{"matches_target":false,"confidence":0.0},"same_object":false}'})
        payload = {"model": config.model, "temperature": 0, "max_tokens": 300,
                   "thinking": {"type": "disabled"}, "response_format": {"type": "json_object"},
                   "messages": [{"role": "user", "content": content}]}
        votes = []
        for _ in range(2):
            req = urlrequest.Request(config.base_url + "/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode(),
                headers={"Authorization": "Bearer " + config.api_key, "Content-Type": "application/json"})
            if transport:
                reply = transport(req)
            else:
                with urlrequest.urlopen(req, timeout=min(config.timeout_s, 25)) as response:
                    reply = json.load(response)
            raw = reply['choices'][0]['message']['content'].strip()
            if raw.startswith('```'):
                raw = raw.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
            vote = json.loads(raw)
            for c in cameras:
                row = vote[c];score = row['confidence']
                if (isinstance(score, bool) or not isinstance(score, (int, float))
                        or not math.isfinite(score) or not 0 <= score <= 1):
                    raise ValueError('invalid_confidence')
            votes.append(vote)
        accepted = [c for c in cameras if all(v[c].get('matches_target') is True
                                             and v[c]['confidence'] >= .80 for v in votes)]
        return {"accepted_cameras": accepted,
                "same_object": len(accepted) == 2 and all(v.get('same_object') is True for v in votes),
                "reason": "consistent_pixel_crop_reviews" if accepted else "review_abstained",
                "minimum_confidence": min(v[c]['confidence'] for v in votes for c in (accepted or cameras)),
                "camera_confidences": {c: min(v[c]['confidence'] for v in votes) for c in accepted},
                "model": config.model, "repeats": len(votes)}
    except (ImportError, OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        return dict(result, error_type=type(exc).__name__)
