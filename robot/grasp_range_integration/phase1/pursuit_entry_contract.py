"""Check whether a phase-one stereo observation can enter the frozen tracker.

The pursuit server refuses initial left boxes with centre x < 250/1280 of
the frame and right boxes with centre x > 1000/1280.  Its second-eye check
also rejects vertical centres more than 150 px apart at 720 px height.
These are entry conditions, not object-depth or grasp authorization.
"""

from __future__ import annotations

import math
from typing import Any


ENTRY_MARGIN_PX = 15.0


def assess_pursuit_entry(
    detections: list[dict[str, Any]],
    image_sizes: dict[str, tuple[int, int]],
) -> dict[str, Any] | None:
    """Return a diagnostic for a confirmed pair, or None without one."""
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for item in detections:
        if not isinstance(item, dict) or not (item.get("confirmed_match") and
                                            item.get("dual_camera_confirmed")):
            continue
        pair_id = str(item.get("stereo_pair_id") or "")
        camera = str(item.get("camera_id") or "")
        if pair_id and camera in {"left", "right"}:
            pairs.setdefault(pair_id, {})[camera] = item
    complete_pairs = [(name, value) for name, value in pairs.items()
                      if set(value) == {"left", "right"}]
    if not complete_pairs:
        return None
    results = [_assess_pair(pair_id, pair, image_sizes)
               for pair_id, pair in complete_pairs]
    return next((result for result in results if result["ready"]), results[0])


def _assess_pair(pair_id: str, pair: dict[str, dict[str, Any]],
                 image_sizes: dict[str, tuple[int, int]]) -> dict[str, Any]:
    centres: dict[str, tuple[float, float]] = {}
    for camera in ("left", "right"):
        box = pair[camera].get("bbox_1000") or []
        width, height = image_sizes.get(camera, (0, 0))
        try:
            coordinates = [float(value) for value in box]
        except (TypeError, ValueError):
            coordinates = []
        if (len(coordinates) != 4 or width <= 0 or height <= 0 or
                not all(math.isfinite(value) for value in coordinates) or
                not (0 <= coordinates[0] < coordinates[2] <= 1000 and
                     0 <= coordinates[1] < coordinates[3] <= 1000)):
            return {"ready": False, "reason": "invalid_handoff_box_or_image",
                    "stereo_pair_id": pair_id, "steering_camera": None}
        centres[camera] = ((coordinates[0] + coordinates[2]) * width / 2000.0,
                           (coordinates[1] + coordinates[3]) * height / 2000.0)
    left_width, left_height = image_sizes["left"]
    right_width, right_height = image_sizes["right"]
    left_min = (250.0 + ENTRY_MARGIN_PX) * left_width / 1280.0
    right_max = (1000.0 - ENTRY_MARGIN_PX) * right_width / 1280.0
    vertical_limit = (150.0 - ENTRY_MARGIN_PX) * min(left_height, right_height) / 720.0
    vertical_gap = abs(centres["left"][1] - centres["right"][1])
    if centres["left"][0] < left_min:
        reason, steering_camera = "left_outside_tracker_entry_roi", "left"
    elif centres["right"][0] > right_max:
        reason, steering_camera = "right_outside_tracker_entry_roi", "right"
    elif vertical_gap > vertical_limit:
        reason, steering_camera = "stereo_vertical_mismatch", None
    else:
        reason, steering_camera = "ready", None
    return {
        "ready": reason == "ready",
        "reason": reason,
        "stereo_pair_id": pair_id,
        "steering_camera": steering_camera,
        "left_center_px": [round(value, 1) for value in centres["left"]],
        "right_center_px": [round(value, 1) for value in centres["right"]],
        "vertical_gap_px": round(vertical_gap, 1),
        "left_min_center_px": round(left_min, 1),
        "right_max_center_px": round(right_max, 1),
        "vertical_limit_px": round(vertical_limit, 1),
    }
