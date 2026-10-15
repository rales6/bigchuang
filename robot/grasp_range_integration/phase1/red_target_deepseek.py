"""Optional DeepSeek review of two independently grounded red image crops.

This reviews a stopped camera pair. The caller supplies pixel-derived boxes;
the model may classify them but cannot choose coordinates or move the robot.
"""

from __future__ import annotations

import base64
from io import BytesIO
import json
import math
import os
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

from PIL import Image


def _configuration():
    import deepseek_visual_localization as vision

    module_dir = Path(vision.__file__).resolve().parent
    source = Path(os.environ.get(
        "DEEPSEEK_VISION_CONFIG",
        str(Path(os.environ.get("CAR_ROOT", module_dir.parent)) /
            "secrets" / "deepseek_vision.json")))
    return vision.load_config(source, module_directory=module_dir)


def _image_url(path: Path) -> dict:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {
        "url": "data:image/jpeg;base64," + data, "detail": "high"}}


def _crop_url(path: Path, box: list[float]) -> dict:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        x1, y1, x2, y2 = (float(box[i]) / 1000 * (width if i % 2 == 0 else height)
                          for i in range(4))
        pad = max(36, int(max(x2 - x1, y2 - y1) * 1.7))
        crop = image.crop((max(0, int(x1 - pad)), max(0, int(y1 - pad)),
                           min(width, int(x2 + pad)), min(height, int(y2 + pad))))
        buffer = BytesIO()
        crop.save(buffer, format="JPEG", quality=88)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {
        "url": "data:image/jpeg;base64," + data, "detail": "high"}}


def _parse(content):
    text = str(content or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("response_not_object")
    votes = {}
    for camera in ("left", "right"):
        row = value.get(camera)
        if not isinstance(row, dict):
            raise ValueError("missing_camera_result")
        confidence = row.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("invalid_confidence")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("invalid_confidence")
        votes[camera] = {"is_red_block": row.get("is_red_block") is True,
                         "confidence": confidence}
    votes["same_object"] = value.get("same_object") is True
    return votes


def verify_red_block_pair(images: dict, cues: dict, *, config=None, transport=None) -> dict:
    """Return a bounded classification verdict; API/network failures abstain."""
    try:
        config = config or _configuration()
    except (ImportError, OSError, ValueError) as exc:
        return {"accepted": False, "reason": "deepseek_config_unavailable",
                "error_type": type(exc).__name__}
    if not config.configured:
        return {"accepted": False, "reason": "deepseek_not_configured"}
    if set(cues) != {"left", "right"} or not all(
            cues[camera].get("complete_in_frame") for camera in ("left", "right")):
        return {"accepted": False, "reason": "two_complete_crops_required"}
    try:
        paths = {camera: Path(images[camera]) for camera in ("left", "right")}
        content = [
            {"type": "text", "text": "LEFT FULL IMAGE"}, _image_url(paths["left"]),
            {"type": "text", "text": "LEFT RED REGION CROP"},
            _crop_url(paths["left"], cues["left"]["bbox_1000"]),
            {"type": "text", "text": "RIGHT FULL IMAGE"}, _image_url(paths["right"]),
            {"type": "text", "text": "RIGHT RED REGION CROP"},
            _crop_url(paths["right"], cues["right"]["bbox_1000"]),
            {"type": "text", "text": (
                "分别检查左右相机的完整照片与对应裁剪区域。只判断给定裁剪框里的物体，"
                "不要另选框或从另一相机猜测：它是否是地面上的红色小立方体/小物块？"
                "两个相机中的物体是否为同一个？若有任何不确定，回答 false。"
                "只输出 JSON：{\"left\":{\"is_red_block\":false,\"confidence\":0.0},"
                "\"right\":{\"is_red_block\":false,\"confidence\":0.0},"
                "\"same_object\":false}。confidence 表示各相机独立判断的可靠程度。")},
        ]
        payload = {"model": config.model, "thinking": {"type": "disabled"},
                   "temperature": 0.0, "max_tokens": 300,
                   "response_format": {"type": "json_object"},
                   "messages": [{"role": "user", "content": content}]}
        votes = []
        for _ in range(2):
            request = urllib_request.Request(
                config.base_url + "/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Authorization": "Bearer " + config.api_key,
                         "Content-Type": "application/json"}, method="POST")
            if transport is None:
                with urllib_request.urlopen(request, timeout=min(config.timeout_s, 25.)) as response:
                    reply = json.loads(response.read().decode("utf-8"))
            else:
                reply = transport(request)
            votes.append(_parse(reply["choices"][0]["message"]["content"]))
    except (OSError, TimeoutError, urllib_error.URLError, ValueError, KeyError,
            IndexError, TypeError, json.JSONDecodeError) as exc:
        return {"accepted": False, "reason": "deepseek_review_unavailable",
                "error_type": type(exc).__name__}
    accepted = (len(votes) >= 2 and all(
        vote["same_object"] and all(vote[camera]["is_red_block"]
                                    and vote[camera]["confidence"] >= .80
                                    for camera in ("left", "right"))
        for vote in votes))
    return {"accepted": accepted,
            "reason": "two_independent_reviews_confirmed" if accepted else "review_abstained",
            "model": config.model, "repeats": len(votes),
            "minimum_confidence": min(vote[camera]["confidence"] for vote in votes
                                      for camera in ("left", "right"))}
