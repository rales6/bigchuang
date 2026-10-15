from __future__ import annotations

import colorsys
from typing import Any

import numpy as np
from PIL import Image


COLOUR_HUE_RANGES = {
    "red": ((0.0, 0.06), (0.94, 1.0)),
    "orange": ((0.04, 0.12),),
    "yellow": ((0.10, 0.20),),
    "green": ((0.20, 0.47),),
    "blue": ((0.50, 0.72),),
    "purple": ((0.70, 0.90),),
}


def _normalise(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ").replace("-", " ")


def _bbox_pixels(detection: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int] | None:
    bbox = detection.get("bbox_1000") or detection.get("bbox_2d") or detection.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        values = [float(bbox[index]) for index in range(4)]
    except (TypeError, ValueError):
        return None
    if max(values) <= 1.5:
        x1, y1, x2, y2 = values[0] * width, values[1] * height, values[2] * width, values[3] * height
    elif max(values) <= 1000.0:
        x1, y1, x2, y2 = values[0] * width / 1000.0, values[1] * height / 1000.0, values[2] * width / 1000.0, values[3] * height / 1000.0
    else:
        x1, y1, x2, y2 = values
    x1, x2 = sorted((max(0, min(width, int(x1))), max(0, min(width, int(x2)))))
    y1, y2 = sorted((max(0, min(height, int(y1))), max(0, min(height, int(y2)))))
    return (x1, y1, x2, y2) if x2 - x1 >= 3 and y2 - y1 >= 3 else None


def estimate_colour(image: Image.Image, detection: dict[str, Any]) -> tuple[str, float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    box = _bbox_pixels(detection, image.width, image.height)
    if box is None:
        return "unknown", 0.0
    x1, y1, x2, y2 = box
    crop = rgb[y1:y2, x1:x2]
    if not crop.size:
        return "unknown", 0.0
    pixels = crop.reshape(-1, 3)
    # Ignore very dark, white and nearly grey pixels; they are usually
    # background, highlights or shadows rather than useful colour evidence.
    maximum = pixels.max(axis=1)
    minimum = pixels.min(axis=1)
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-6)
    mask = (maximum >= 0.18) & (maximum <= 0.98) & (saturation >= 0.28)
    pixels = pixels[mask]
    if len(pixels) < 20:
        return "unknown", 0.0
    hues = np.asarray([colorsys.rgb_to_hsv(*pixel)[0] for pixel in pixels], dtype=np.float32)
    scores = {}
    for colour, ranges in COLOUR_HUE_RANGES.items():
        scores[colour] = float(np.mean([any(low <= hue <= high for low, high in ranges) for hue in hues]))
    colour, score = max(scores.items(), key=lambda item: item[1])
    return (colour, score) if score >= 0.30 else ("unknown", score)


def detection_matches(
    detection: dict[str, Any],
    target: dict[str, Any],
    image: Image.Image | None = None,
    minimum_colour_score: float = 0.40,
) -> dict[str, Any]:
    required_class = _normalise(target.get("class_name"))
    detected_class = _normalise(detection.get("class_name") or detection.get("label"))
    generic = required_class in {"any yolo object", "any object", ""}
    class_ok = bool(detected_class) and (generic or detected_class == required_class)
    attributes = detection.get("attributes")
    if not isinstance(attributes, dict):
        attributes = {}
    required_colour = _normalise(target.get("color"))
    detected_colour = _normalise(attributes.get("color"))
    colour_score = float(detection.get("colour_score") or 0.0)
    colour_source = "model_attribute" if detected_colour else "none"
    if required_colour and image is not None:
        pixel_colour, pixel_score = estimate_colour(image, detection)
        if pixel_score >= colour_score:
            detected_colour, colour_score, colour_source = pixel_colour, pixel_score, "bbox_pixels"
    colour_ok = not required_colour or (
        detected_colour == required_colour
        and (colour_source == "model_attribute" or colour_score >= minimum_colour_score)
    )
    return {
        "matched": bool(class_ok and colour_ok),
        "class_ok": class_ok,
        "colour_ok": colour_ok,
        "detected_class": detected_class,
        "detected_colour": detected_colour or "unknown",
        "colour_score": round(colour_score, 4),
        "colour_source": colour_source,
        "confidence": float(detection.get("confidence") or 0.0),
    }


def filter_detections(
    detections: list[dict[str, Any]],
    target: dict[str, Any],
    images: dict[str, Image.Image] | None = None,
) -> list[dict[str, Any]]:
    result = []
    images = images or {}
    for detection in detections:
        camera_id = str(detection.get("camera_id") or "")
        evidence = detection_matches(detection, target, images.get(camera_id))
        if evidence["matched"]:
            result.append({**detection, "search_match": evidence})
    return sorted(
        result,
        key=lambda item: (
            float(item["search_match"]["confidence"]),
            float(item["search_match"]["colour_score"]),
        ),
        reverse=True,
    )
