"""Phase-1 single-camera reacquisition and append-only camera evidence.

Recovery requeues the *same* view in the protected executor's live remaining
list. It issues no hardware commands and never invents a target distance.
The executor remains responsible for localization, guarded turns and STOP.
"""

from __future__ import annotations

from copy import deepcopy
import html
import json
import math
from pathlib import Path
import shutil
import time
import threading

import numpy as np
from PIL import Image, ImageDraw


RED_CUE_SOURCE = "phase1_red_visual_cue"
GREEN_CUE_SOURCE = "phase1_green_visual_cue"


def red_visual_cues(images: dict, *, minimum_pixels: int = 120) -> list[dict]:
    return colour_visual_cues(images, "red", minimum_pixels=minimum_pixels)


def colour_visual_cues(images: dict, colour: str, *, minimum_pixels: int = 120) -> list[dict]:
    """Find red image regions for another look, never for mission success.

    The model's reported box can miss a plainly visible small red block. A
    colour component is only a bearing cue: it has no verified object class,
    depth or stereo identity and must not enter the protected success gate.
    """
    if colour not in {"red", "green"}:
        return []
    source_id = RED_CUE_SOURCE if colour == "red" else GREEN_CUE_SOURCE
    try:
        import cv2
    except ImportError:
        return []
    cues = []
    for camera in ("left", "right"):
        path = images.get(camera)
        if path is None:
            continue
        try:
            with Image.open(path) as source:
                rgb = np.asarray(source.convert("RGB"))
        except (OSError, ValueError):
            continue
        height, width = rgb.shape[:2]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        red, green, blue = (rgb[:, :, channel].astype(np.int16) for channel in range(3))
        if colour == "red":
            mask = (((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 170))
                    & (hsv[:, :, 1] >= 95) & (hsv[:, :, 2] >= 55)
                    & (red * 100 >= green * 135)
                    & (red * 100 >= blue * 120)).astype(np.uint8)
        else:
            mask = ((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 90)
                    & (hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 35)
                    & (green * 100 >= red * 125)
                    & (green * 100 >= blue * 110)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        _count, _labels, components, _centers = cv2.connectedComponentsWithStats(mask, 8)
        proposals = []
        for x, y, box_width, box_height, pixels in components[1:]:
            x, y, box_width, box_height, pixels = map(int, (x, y, box_width, box_height, pixels))
            aspect = box_width / max(1, box_height)
            fill = pixels / max(1, box_width * box_height)
            # A thin strip at the top/bottom is useful for choosing a new
            # station, but is never a complete object or stereo evidence.
            vertical_fragment = ((y <= 2 or y + box_height >= height - 2)
                                 and pixels >= 24 and box_width >= 12
                                 and fill >= .40 and box_width <= width * .15
                                 and box_height <= height * .06)
            if not vertical_fragment and (pixels < max(minimum_pixels, int(width * height * .00013))
                    or box_width < max(12, int(width * .012))
                    or box_height < max(12, int(height * .019))
                    or not .45 <= aspect <= 2.1 or fill < .45
                    or box_width > width * .35 or box_height > height * .35):
                continue
            complete = x >= width * .015 and x + box_width <= width * .985
            complete &= y >= height * .015 and y + box_height <= height * .985
            proposals.append((pixels, complete, x, y, box_width, box_height, fill, vertical_fragment))
        for pixels, complete, x, y, box_width, box_height, fill, fragment in sorted(
                proposals, reverse=True)[:2]:
            cues.append({
                "camera_id": camera, "class_name": colour + "_region",
                "source": source_id, "confidence": .70 if complete else .55,
                "bbox_1000": [round(1000 * x / width, 2), round(1000 * y / height, 2),
                              round(1000 * (x + box_width) / width, 2),
                              round(1000 * (y + box_height) / height, 2)],
                "attributes": {"color": colour, "description": "unclassified " + colour + " image region"},
                "single_camera_match": False, "confirmed_match": False,
                "dual_camera_confirmed": False, "complete_in_frame": bool(complete),
                "search_evidence": {
                    "visual_cue_only": True, "source_ok": False, "class_ok": False,
                    "colour_ok": True, "bbox_valid": True, "bbox_complete": bool(complete),
                    colour + "_pixel_count": pixels, colour + "_bbox_fill": round(fill, 3),
                    "mission_success_allowed": False,
                    "vertical_edge_fragment": bool(fragment),
                },
            })
    return cues


def candidate(detections: list[dict]) -> dict | None:
    valid = []
    for item in detections:
        if not isinstance(item, dict):
            continue
        evidence = item.get("search_evidence") or {}
        if item.get("task_region_eligible") is False and not item.get("transit_candidate_allowed"):
            continue
        try:
            box = [float(v) for v in item.get("bbox_1000") or []]
            confidence = float(item.get("confidence") or 0)
        except (TypeError, ValueError):
            continue
        if (
            item.get("camera_id") not in {"left", "right"}
            or len(box) != 4
            or not math.isfinite(confidence)
            or not all(math.isfinite(v) for v in box)
            or not (all(evidence.get(key) for key in
                        ("source_ok", "class_ok", "colour_ok", "bbox_valid"))
                    or (item.get("source") in {RED_CUE_SOURCE, GREEN_CUE_SOURCE}
                        and evidence.get("visual_cue_only") is True
                        and evidence.get("colour_ok") is True
                        and evidence.get("bbox_valid") is True))
            or min(float(box[2]) - float(box[0]), float(box[3]) - float(box[1])) < 12
        ):
            continue
        # Partly clipped candidates may trigger a stationary recheck, not
        # mission success. The protected stereo completeness gate is intact.
        normalized = dict(item)
        normalized["bbox_1000"] = box
        valid.append(normalized)
    return max(valid, key=lambda item: float(item.get("confidence") or 0), default=None)


def stable_candidate(previous: dict, current: dict, *, after_turn=False) -> bool:
    if previous.get("camera_id") != current.get("camera_id"):
        return False
    class_changed = previous.get("class_name") != current.get("class_name")
    cue_to_block = (any({previous.get("class_name"), current.get("class_name")}
                       == {colour + "_region", "block"} for colour in ("red", "green"))
                    and bool({RED_CUE_SOURCE, GREEN_CUE_SOURCE}
                             & {previous.get("source"), current.get("source")}))
    if class_changed and not cue_to_block:
        return False
    a, b = previous["bbox_1000"], current["bbox_1000"]
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    limit = 140 if after_turn else 60
    centering_ok = not after_turn or abs(bx - 500) <= abs(ax - 500) + 30
    return (math.hypot(ax - bx, ay - by) <= limit and abs(ay - by) <= 60
            and centering_ok and min(area_a, area_b) / max(area_a, area_b) >= 0.4)


def rotation_clearance_safe(scan, radius_m=0.30, **context) -> bool:
    from grasp_range_integration.phase1.rotation_guard import assess_rotation
    return bool(assess_rotation(scan, radius_m, **context)["safe"])


def candidate_yaw_step(item: dict, fov_deg: float = 82.0) -> float:
    """Image-right is vehicle-right for the audited forward, unmirrored pair.

    This is an approximate bearing for reacquisition only, not depth. A
    direction/centering validation must precede unattended real-car use.
    """
    box = item["bbox_1000"]
    offset = ((float(box[0]) + float(box[2])) / 2 - 500) / 500
    bearing = -math.atan(offset * math.tan(math.radians(fov_deg / 2)))
    if abs(math.degrees(bearing)) < 3:
        return 0.0
    return max(-math.radians(6), min(math.radians(6), bearing))


def reject_copied_stereo_boxes(detections: list[dict]) -> list[dict]:
    """A model copying a box to both cameras is not stereo grounding."""
    paired = [item for item in detections if item.get("dual_camera_confirmed")]
    if len(paired) == 2 and all("qwen" in str(i.get("source", "")).lower() for i in paired):
        if paired[0].get("bbox_1000") == paired[1].get("bbox_1000"):
            for item in paired:
                item["confirmed_match"] = item["dual_camera_confirmed"] = False
                # Copied coordinates must not select a steering camera.
                item["single_camera_match"] = False
                item.pop("stereo_pair_id", None)
                item["search_evidence"]["stereo_rejected_reason"] = "identical_qwen_boxes"
                item["search_evidence"]["bbox_valid"] = False
    return detections


class CandidateRecovery:
    def __init__(self):
        self.previous = None
        self.rechecks = 0
        self.turns = 0
        self.turn_budget_rad = 0.0
        self.started = None
        self.last_status = None
        self.context = {}
        self.borderline_original = None
        self.empty_sweeps = {}
        self.target_lock = None

    def _locked_recovery(self, state, route, item):
        """Feedback-driven viewing only; no depth inference or arm command.

        Return (handled, directive). A lock is scoped to one live station and
        requires two consistent, independently grounded single-eye frames.
        The normal executor still checks every actual wheel command.
        """
        from grasp_range_integration.phase1 import visibility_route_planner as planner
        views = route.get("views") or []
        index = state.get("current_view_index")
        pose = self.context.get("robot_pose") or []
        if (route.get("mode") != "semantic_visibility_regions"
                or not isinstance(index, int) or not 0 <= index < len(views)
                or len(pose) != 3 or not all(math.isfinite(float(v)) for v in pose)):
            self.target_lock = None
            return False, None
        scope = (id(views), index)
        if self.target_lock and self.target_lock["scope"] != scope:
            self.target_lock = None
        if self.target_lock is None:
            review = ((item or {}).get("search_evidence") or {}).get("deepseek_review") or {}
            independently_reviewed = (int(review.get("repeats") or 0) >= 2
                and (item or {}).get("source") in {"phase1_deepseek_green_block", "phase1_deepseek_red_block"}
                and (item or {}).get("camera_id") in (review.get("accepted_cameras") or []))
            if (item is None or not item.get("single_camera_match")
                    or float(item.get("confidence") or 0) < .85
                    or not (independently_reviewed or self.previous is not None
                            and stable_candidate(self.previous, item))):
                return False, None
            self.target_lock = {"scope": scope, "item": deepcopy(item),
                "pending": None, "best": None, "bad_signs": set(),
                "shifts": 0, "looks": 0, "origin_yaw": float(pose[2]),
                "yaw_goals": []}
            # Reserve actual target-recovery photos separately from the
            # ordinary station sweep; do not replenish this on every frame.
            if not route.get("locked_target_reserve_granted"):
                route["locked_target_extra_views"] = 12
                route["maximum_total_views"] = int(route.get("maximum_total_views") or 8) + 12
                route["locked_target_reserve_granted"] = True
        lock = self.target_lock
        view = views[index]
        detections = state.get("detections") or []
        # Keep the same camera/appearance instead of switching to an unrelated
        # equal-confidence model box when one eye loses the locked target.
        prior = lock["item"]
        current = next((candidate([d]) for d in detections
                        if isinstance(d, dict) and d.get("camera_id") == prior["camera_id"]
                        and candidate([d]) is not None
                        and d.get("single_camera_match")
                        and stable_candidate(prior, candidate([d]), after_turn=True)), None)
        complete_eyes = {d.get("camera_id") for d in detections
                         if isinstance(d, dict) and d.get("single_camera_match") and d.get("complete_in_frame")
                         and candidate([d]) is not None}

        def enqueue(xy, yaw, status, **detail):
            self.last_status = status
            shift_distance = math.hypot(xy[0]-pose[0], xy[1]-pose[1])
            view.update(view_xy_m=list(xy), target_yaw_rad=float(yaw), arrival_diagnostic_done=True)
            view["path_xy_m"] = [list(pose[:2]), list(xy)] if shift_distance > .04 else []
            view["path_length_m"] = round(shift_distance, 3)
            view.setdefault("visibility_zone", {}).update(stop_rule=status,
                recovery_camera=prior["camera_id"], locked_target=True, **detail)
            if index not in state["remaining_view_indices"]:
                state["remaining_view_indices"].insert(0, index)
            self.previous = deepcopy(current or prior)
            return True, {"action": "navigate_next_view", "next_view_index": index,
                "terminal": False, "reason": status,
                "recovery": {"camera": prior["camera_id"],
                    "step_deg": round(math.degrees(planner._angle_error(yaw, float(pose[2]))), 2),
                    "linear_motion_allowed": shift_distance > .04,
                    "station_shift_m": round(shift_distance, 3),
                    "locked_bbox_1000": list((current or prior)["bbox_1000"]),
                    "best_heading_deg": round(math.degrees(lock["best"]["yaw"]), 2)
                        if lock["best"] is not None else None,
                    **detail}}

        def yaw_probe(yaw, status="single_camera_locked_yaw_alignment"):
            # Planned angle is evidence to revisit, not a timed motor pulse.
            # The existing executor still checks fresh clearance and actual yaw.
            step = planner._angle_error(yaw, float(pose[2]))
            if abs(step) <= math.radians(2.1):
                return None  # Executor's 2 degree arrival window would swallow it.
            lock["yaw_goals"].append(float(yaw))
            lock["pending"] = {"kind": "yaw", "item": deepcopy(current or prior),
                "complete_eyes": len(complete_eyes), "yaw": float(pose[2]),
                "sign": 1 if step > 0 else -1}
            return enqueue(pose[:2], yaw, status,
                           target_alignment_priority=True,
                           local_yaw_probe_count=len(lock["yaw_goals"]))

        def next_yaw_probe():
            # A centered single-eye target is NOT an empty semantic region.
            # Inspect nearby headings at this station before asking for a new
            # global station. Never spin 360 degrees or infer target depth.
            preferred = 1 if candidate_yaw_step(current or prior) >= 0 else -1
            for degrees in (3, -3, 6, -6, 9, -9):
                yaw = lock["origin_yaw"] + preferred * math.radians(degrees)
                if any(abs(planner._angle_error(yaw, old)) < math.radians(2)
                       for old in lock["yaw_goals"]):
                    continue
                directive = yaw_probe(yaw, "single_camera_locked_local_probe")
                if directive is not None:
                    return directive
            return None

        def next_station(reason, *, allow_probe=True):
            # Do not send the frozen executor into an uncaught global route
            # exception after seeing a real target. Check availability first;
            # keep remaining local angular recovery ahead of global coverage.
            if int(state.get("attempts") or 0) >= int(route.get("maximum_total_views") or 8):
                self.last_status = "locked_target_local_recovery_exhausted"
                return True, {"action": "report_failure", "terminal": True,
                    "reason": "单摄目标证据已保留；本轮多角度复拍仍未完成双摄确认，未启动夹取"}
            directive = next_yaw_probe() if allow_probe else None
            if directive is not None:
                return directive
            region = route.get("target_region") or {}
            if planner.LATEST_KNOWN_MAP is not None and len(region.get("center_xy_m") or []) == 2:
                try:
                    planner.build_visibility_route(planner.LATEST_KNOWN_MAP, tuple(pose[:2]), {
                        "center_xy_m": region["center_xy_m"], "radius_m": region.get("radius_m", .95),
                        "maximum_views": route.get("requested_maximum_views", 8),
                        "recognition_min_distance_m": (region.get("recognition_distance_m") or [.45, 1.8])[0],
                        "recognition_max_distance_m": (region.get("recognition_distance_m") or [.45, 1.8])[1],
                        "minimum_work_area_coverage": region.get("minimum_work_area_coverage", .25),
                        "camera_horizontal_fov_deg": region.get("camera_horizontal_fov_deg", 82),
                    })
                except planner.NoReachableVisibilityError:
                    self.last_status = "locked_target_local_recovery_exhausted"
                    return True, {"action": "report_failure", "terminal": True,
                        "reason": "已发现单摄目标；已复拍附近有效朝向，但双摄仍未确认且无安全换站路线，保留照片等待调整；未启动夹取",
                        "recovery": {"camera": prior["camera_id"], "step_deg": 0.,
                            "linear_motion_allowed": False, "local_yaw_probe_count": len(lock["yaw_goals"])}}
            self.last_status = "locked_target_next_station"
            return True, {"action": "replan", "terminal": False, "reason": reason}

        pending = lock["pending"]
        if pending and pending["kind"] == "yaw":
            before = pending["item"]["bbox_1000"]
            after = current["bbox_1000"] if current else None
            worse = (after is None or len(complete_eyes) < pending["complete_eyes"]
                     or abs((after[0]+after[2])/2-500) > abs((before[0]+before[2])/2-500)+30)
            lock["pending"] = None
            if worse:
                lock["bad_signs"].add(pending["sign"])
                # Undo the measured heading change, not another guessed pulse.
                # Hold CURRENT XY so incidental turn translation is not chased.
                return enqueue(pose[:2], lock["best"]["yaw"], "single_camera_return_best_heading",
                               target_lost=current is None)
        elif pending:
            lock["pending"] = None
            lock["origin_yaw"] = float(pose[2])
            lock["yaw_goals"] = []
        if current is None:
            return next_station("回到有效朝向后仍未见锁定目标，改选观察站，不用旧框宣告找到")
        lock["item"] = deepcopy(current)
        box = current["bbox_1000"]
        quality = (len(complete_eyes), -abs((box[0]+box[2])/2-500))
        if lock["best"] is None or quality > lock["best"]["quality"]:
            lock["best"] = {"quality": quality, "yaw": float(pose[2])}
        lock["looks"] += 1
        if (lock["looks"] >= 12 or int(state.get("attempts") or 0) >=
                int(route.get("maximum_total_views") or 8)):
            return next_station("本站目标恢复无进一步进展，改选观察站而非重复自转", allow_probe=False)
        peer_fragments = [d for d in detections if isinstance(d, dict)
                          and d.get("camera_id") != current["camera_id"]
                          and d.get("source") in {RED_CUE_SOURCE, GREEN_CUE_SOURCE}
                          and (d.get("search_evidence") or {}).get("vertical_edge_fragment")
                          and (d.get("attributes") or {}).get("color") ==
                              (current.get("attributes") or {}).get("color")
                          and abs((d["bbox_1000"][0]+d["bbox_1000"][2])/2-
                                  (box[0]+box[2])/2) <= 350]
        upper_edge = box[1] <= 25
        lower_edge = box[3] >= 920
        yaw_step = candidate_yaw_step(current)
        sign = 1 if yaw_step > 0 else -1
        # Confirmed primary-eye geometry has priority. An UNVERIFIED colour
        # strip in the peer may be another object; it must not veto yaw and
        # force a station shift/global replan (actual 210758 failure).
        goal = float(pose[2]) + max(-math.radians(3), min(math.radians(3), yaw_step))
        tried_goal = any(abs(planner._angle_error(goal, old)) < math.radians(2)
                         for old in lock["yaw_goals"])
        change_station = (upper_edge or lower_edge or not yaw_step or sign in lock["bad_signs"]
                          or tried_goal or abs(planner._angle_error(goal, lock["origin_yaw"])) > math.radians(9.01))
        if (not change_station and not tried_goal
                and abs(planner._angle_error(goal, lock["origin_yaw"])) <= math.radians(9.01)):
            directive = yaw_probe(goal)
            if directive is not None:
                return directive
        directive = next_yaw_probe()
        if directive is not None:
            return directive
        if change_station:
            if not lower_edge and lock["shifts"] < 3 and planner.LATEST_KNOWN_MAP is not None:
                region = route.get("target_region") or {}
                center = region.get("center_xy_m") or []
                radius = float(region.get("radius_m") or 0) + float(
                    region.get("minimum_standoff_from_region_m", planner.MINIMUM_STANDOFF_M))
                if len(center) == 2 and radius > 0:
                    rejected = []
                    for distance in (.12, .10, .08, .06):
                        xy = (float(pose[0]) + distance*math.cos(float(pose[2])),
                              float(pose[1]) + distance*math.sin(float(pose[2])))
                        path = planner.short_forward_view_path(planner.LATEST_KNOWN_MAP,
                            tuple(pose[:2]), xy, tuple(center), radius)
                        # Same forward reserve and work-area mask, not a
                        # shortened collision distance or a snapped detour.
                        if (path and math.hypot(path[-1][0]-xy[0], path[-1][1]-xy[1]) <= .04
                                and all(abs((point[0]-pose[0])*math.sin(float(pose[2]))-
                                            (point[1]-pose[1])*math.cos(float(pose[2]))) <= .04 for point in path)
                                and planner._path_length(tuple(pose[:2]), path) <= distance+.04):
                            lock["shifts"] += 1
                            lock["pending"] = {"kind": "station"}
                            return enqueue(xy, pose[2], "locked_target_station_shift",
                                vertical_edge=bool(upper_edge or peer_fragments), shift_count=lock["shifts"],
                                rejected_forward_steps_m=rejected)
                        rejected.append(distance)
        return next_station("已确认单摄目标，本站角度已复拍；保留证据并改选地图可达观察站")

    def _next_empty_sweep(self, state, route):
        """Use another forward work-area view before declaring a station empty."""
        if (route.get("mode") != "semantic_visibility_regions"
                or (self.context.get("task_region_view") or {}).get("eligible") is not True
                or state.get("remaining_view_indices")
                or int(state.get("attempts") or 0) >= int(route.get("maximum_total_views") or 8)):
            return None
        index = state.get("current_view_index")
        views = route.get("views") or []
        pose = self.context.get("robot_pose") or []
        center = (route.get("target_region") or {}).get("center_xy_m") or []
        if (not isinstance(index, int) or not 0 <= index < len(views)
                or len(pose) != 3 or len(center) != 2
                or not all(math.isfinite(float(v)) for v in list(pose) + list(center))):
            return None
        # The dictionary identity stays stable while one live station is
        # requeued, and changes when the planner publishes another route.
        origin = self.empty_sweeps.setdefault(id(views[index]),
                                             {"yaw": float(pose[2]), "used": []})
        center_yaw = math.atan2(float(center[1]) - float(pose[1]),
                                float(center[0]) - float(pose[0]))
        from grasp_range_integration.phase1 import visibility_route_planner as planner
        choices = []
        for degrees in (15., -15., 30., -30.):
            yaw = math.atan2(math.sin(origin["yaw"] + math.radians(degrees)),
                             math.cos(origin["yaw"] + math.radians(degrees)))
            separation = abs(math.atan2(math.sin(yaw - float(pose[2])),
                                             math.cos(yaw - float(pose[2]))))
            forward = abs(math.atan2(math.sin(yaw - center_yaw), math.cos(yaw - center_yaw)))
            if (separation < math.radians(8) or forward > math.radians(60)
                    or any(abs(math.atan2(math.sin(yaw - old), math.cos(yaw - old)))
                           < math.radians(12) for old in origin["used"])):
                continue
            probe = planner.task_region_view_assessment(
                route, [float(pose[0]), float(pose[1]), yaw], view=views[index])
            coverage = float(probe.get("estimated_work_area_coverage") or 0.)
            if probe.get("eligible") is True:
                choices.append((coverage, -abs(degrees), yaw, probe))
        if not choices or len(origin["used"]) >= 2:
            return None
        _coverage, _cost, yaw, probe = max(choices)
        origin["used"].append(yaw)
        view = views[index]
        view["view_xy_m"] = [float(pose[0]), float(pose[1])]
        view["target_yaw_rad"] = yaw
        view.setdefault("visibility_zone", {}).update(
            stop_rule="local_view_sweep", sweep_count=max(2, int(
                view.get("visibility_zone", {}).get("sweep_count") or 1)))
        state["remaining_view_indices"].insert(0, index)
        self.reset("local_view_sweep")
        return {"action": "navigate_next_view", "next_view_index": index,
                "reason": "当前站位还有未检查的工作区朝向，原地转向复拍",
                "terminal": False,
                "recovery": {"camera": "work_area", "step_deg": round(math.degrees(
                    math.atan2(math.sin(yaw - float(pose[2])),
                               math.cos(yaw - float(pose[2])))), 2),
                    "linear_motion_allowed": False,
                    "estimated_work_area_coverage": probe.get("estimated_work_area_coverage")}}

    def _resume_coverage(self, state, route):
        index = state.get("current_view_index")
        views = route.get("views") or []
        if not isinstance(index,int) or not 0 <= index < len(views):
            return None
        view = views[index]
        if self.borderline_original is not None:
            index, original = self.borderline_original
            view = views[index]
            view.clear()
            view.update(deepcopy(original))
            self.borderline_original = None
        gate = self.context.get("task_region_view") or {}
        if (not view.get("arrival_diagnostic_done") or gate.get("eligible") is not False
                or gate.get("reason") != "approaching_task_region"
                or int(state.get("attempts") or 0) >= int(route.get("maximum_total_views") or 8)
                or int(view.get("coverage_resume_count") or 0) >= 2):
            return None
        view["coverage_resume_count"] = int(view.get("coverage_resume_count") or 0)+1
        if index not in state["remaining_view_indices"]:
            state["remaining_view_indices"].insert(0,index)
        self.reset("resume_view_coverage_adjustment")
        return {"action":"navigate_next_view","next_view_index":index,"terminal":False,
                "reason":"临界观察未完成确认，恢复原观察站视野调整；不追逐未知距离的物块"}

    def next(self, state, route):
        gate = self.context.get("task_region_view") or {}
        item = candidate(state.get("detections") or [])
        borderline = bool(gate.get("borderline_inspection_allowed") and item
                          and float(item.get("confidence") or 0) >= .85
                          and item.get("complete_in_frame") is True)
        if borderline and self.borderline_original is None:
            index=state.get("current_view_index")
            views=route.get("views") or []
            if (isinstance(index,int) and 0 <= index < len(views)
                    and not views[index].get("borderline_recovery_done")):
                self.borderline_original=(index,deepcopy(views[index]))
                self.borderline_original[1]["arrival_diagnostic_done"] = True
                self.borderline_original[1]["borderline_recovery_done"] = True
            elif self.borderline_original is None:
                borderline=False  # Do not restart the whole 60s recovery on every photo.
        directive = self._next(state,route,borderline=borderline)
        if directive is None and (state.get("latest_event") or {}).get("type") == "observation_complete":
            directive = self._resume_coverage(state,route)
        if directive and directive.get("action") == "replan":
            # The saved borderline view belongs to the OLD route. Restoring
            # it by the same numeric index would overwrite the new station.
            self.borderline_original = None
            self.previous = None
        if self.last_status == "stereo_confirmed":
            self.borderline_original = None
        return directive

    def reset(self, status: str):
        self.previous = None
        self.rechecks = self.turns = 0
        self.turn_budget_rad = 0.0
        self.started = None
        self.last_status = status

    def _next(self, state: dict, route: dict, *, borderline=False) -> dict | None:
        if (state.get("latest_event") or {}).get("type") != "observation_complete":
            self.reset("navigation_recovery")
            return None
        detections = state.get("detections") or []
        if self.target_lock and self.target_lock["scope"] != (
                id(route.get("views") or []), state.get("current_view_index")):
            self.target_lock = None
        if ((self.context.get("task_region_view") or {}).get("eligible") is False
                and not borderline and self.target_lock is None):
            # Still perceive/archive a grounded image candidate, but do not
            # mutate the observation station or steal the approach route.
            # 20260927_154557 never sent any linear command: a 30x30-pixel
            # transit candidate requeued the start and demanded a 6deg turn.
            self.reset("transit_candidate_archived_continue_approach" if candidate(detections)
                       else "approaching_task_region")
            return None
        if any(isinstance(item, dict) and item.get("dual_camera_confirmed")
               for item in detections):
            self.target_lock = None
            self.reset("stereo_confirmed")
            return None
        # A confirmed but edge-clipped *handoff* remains a search candidate.
        # Steer using the camera that violates the pursuit entry window, not
        # whichever of two equal-confidence model boxes happened to come first.
        item = next((value for value in detections
                     if isinstance(value, dict)
                     and value.get("handoff_alignment_camera") == value.get("camera_id")
                     and candidate([value]) is not None), None) or candidate(detections)
        handled, locked_directive = self._locked_recovery(state, route, item)
        if handled:
            return locked_directive
        if item is None:
            sweep = self._next_empty_sweep(state, route)
            if sweep is not None:
                return sweep
            self.reset("candidate_lost" if self.previous else
                       "approaching_task_region" if (self.context.get("task_region_view") or {}).get("eligible") is False else
                       "no_candidate")
            return None
        views = route.get("views") or []
        index = state.get("current_view_index")
        pose = self.context.get("robot_pose") or []
        maximum = int(route.get("maximum_total_views") or 8)
        if not isinstance(index, int) or not 0 <= index < len(views) or len(pose) != 3:
            self.reset("missing_view_or_pose")
            return None
        now = time.monotonic()
        self.started = now if self.started is None else self.started
        if int(state.get("attempts") or 0) >= maximum or now - self.started > 60:
            self.reset("recovery_budget_exhausted")
            return None
        step = 0.0
        if self.previous is None:
            self.rechecks = 1
            status = "single_camera_stationary_recheck"
        elif not stable_candidate(self.previous, item, after_turn=self.turns > 0):
            # Do not steer towards a newly jumping/model-hallucinated box.
            self.reset("unstable_candidate")
            return None
        else:
            box = item["bbox_1000"]
            if box[3] >= 920 or (box[2] - box[0]) * (box[3] - box[1]) >= 150000:
                self.reset("possibly_too_close")
                return {"action": "report_failure", "terminal": True,
                        "reason": "目标候选过大或处于画面底部，距离不明；保持停车，请人工检查"}
            entry = (item.get("search_evidence") or {}).get("pursuit_entry") or {}
            step = (0.0 if entry.get("reason") in {
                "stereo_vertical_mismatch", "invalid_handoff_box_or_image",
                "handoff_image_unavailable",
            } else candidate_yaw_step(item))
            if self.turns >= 5 or self.rechecks >= 2 and step == 0:
                self.reset("seen_unconfirmed")
                return None
            if step:
                self.turns += 1
                self.turn_budget_rad += abs(step)
                if self.turn_budget_rad > math.radians(30.01):
                    self.reset("recovery_angle_exhausted")
                    return None
                status = "single_camera_yaw_alignment"
            else:
                self.rechecks += 1
                status = "single_camera_stationary_recheck"
        self.previous = deepcopy(item)
        self.last_status = status
        view = views[index]
        # Updating the existing dictionary keeps the protected executor's
        # live view references and indices valid. No new motion executor.
        view["view_xy_m"] = [float(pose[0]), float(pose[1])]
        view["target_yaw_rad"] = math.atan2(math.sin(float(pose[2]) + step),
                                           math.cos(float(pose[2]) + step))
        view.setdefault("visibility_zone", {}).update({
            "stop_rule": status, "recovery_turn": self.turns,
            "recovery_camera": item["camera_id"],
            "recovery_step_deg": round(math.degrees(step), 2),
        })
        remaining = state["remaining_view_indices"]  # The executor owns this live list.
        if index not in remaining:
            remaining.insert(0, index)
        return {"action": "navigate_next_view", "next_view_index": index,
                "reason": status, "terminal": False,
                "recovery": {"camera": item["camera_id"], "step_deg": round(math.degrees(step), 2),
                             "linear_motion_allowed": False}}


class EvidenceStore:
    def __init__(self, directory: Path):
        self.directory = directory / "observations"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.by_working_path = {}

    def append(self, record: dict):
        with (self.directory / "observations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def capture(self, path: Path, info: dict) -> dict:
        number = len(self.records) + 1
        while (self.directory / f"obs_{number:04d}_left_raw.jpg").exists():
            number += 1
        record = {"observation_id": f"obs_{number:04d}", "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "stage": "captured", "working_image": path.name, "raw_images": {},
                  "capture_info": deepcopy(info)}
        sources = {"left": path}
        if info.get("camera_2_image"):
            sources["right"] = path.with_name(str(info["camera_2_image"]))
        for camera, source in sources.items():
            destination = self.directory / f"{record['observation_id']}_{camera}_raw.jpg"
            shutil.copy2(source, destination)
            record["raw_images"][camera] = destination.name
            self.by_working_path[str(source.resolve())] = record
        self.records.append(record)
        self.append(deepcopy(record))
        self.index()
        return record

    def analyze(self, record: dict, detections: list[dict], context: dict, *, error=None):
        record.update({"stage": "analyzed", "context": deepcopy(context),
                       "detections": deepcopy(detections), "error": error,
                       "status": "perception_error" if error else
                       "stereo_confirmed" if any(i.get("dual_camera_confirmed") for i in detections) else
                       "red_visual_cue" if candidate(detections) and
                       candidate(detections).get("source") == RED_CUE_SOURCE else
                       "single_camera_candidate" if candidate(detections) else
                       "approaching_task_region" if (context.get("task_region_view") or {}).get("eligible") is False else
                       "no_valid_candidate"})
        if (not error and (context.get("task_region_view") or {}).get("eligible") is False
                and candidate(detections)):
            record["status"] = ("borderline_single_camera_candidate" if
                (context.get("task_region_view") or {}).get("borderline_inspection_allowed")
                else "transit_candidate_archived_continue_approach")
        record["annotated_images"] = {}
        record["candidate_crops"] = []
        for camera, name in record["raw_images"].items():
            with Image.open(self.directory / name) as raw:
                picture = raw.convert("RGB")
            draw = ImageDraw.Draw(picture)
            for item in detections:
                if item.get("camera_id") != camera:
                    continue
                box = item.get("bbox_1000") or []
                if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
                    continue
                xy = [max(0, min(1000, float(v))) / 1000 *
                      (picture.width if i % 2 == 0 else picture.height) for i, v in enumerate(box)]
                if xy[2] <= xy[0] or xy[3] <= xy[1]:
                    continue
                colour = "lime" if item.get("dual_camera_confirmed") else "orange"
                draw.rectangle(xy, outline=colour, width=3)
                label = f"{item.get('class_name', '')} {float(item.get('confidence') or 0):.2f}"
                evidence = item.get("search_evidence") or {}
                failed = [key for key in ("source_ok", "class_ok", "colour_ok", "bbox_complete")
                          if evidence.get(key) is False]
                label += " " + ("reject:" + ",".join(failed) if failed else
                                 "stereo" if item.get("dual_camera_confirmed") else "candidate")
                draw.text((xy[0], max(0, xy[1] - 14)), label.encode("ascii", "replace").decode(), fill=colour)
                if candidate([item]):
                    crop_name = f"{record['observation_id']}_{camera}_candidate_{len(record['candidate_crops']) + 1}.jpg"
                    with Image.open(self.directory / name) as raw:
                        raw.crop((max(0, int(xy[0]) - 20), max(0, int(xy[1]) - 20),
                                  min(raw.width, int(xy[2]) + 20), min(raw.height, int(xy[3]) + 20))).save(self.directory / crop_name)
                    record["candidate_crops"].append(crop_name)
            draw.text((8, 8), record["status"], fill="yellow")
            annotated = f"{record['observation_id']}_{camera}_annotated.jpg"
            picture.save(self.directory / annotated)
            record["annotated_images"][camera] = annotated
        (self.directory / f"{record['observation_id']}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        self.append(deepcopy(record))
        self.index()

    def index(self):
        cards = []
        for record in self.records:
            links = []
            for camera, raw in record["raw_images"].items():
                display = record.get("annotated_images", {}).get(camera, raw)
                links.append(f'<a href="{raw}">{camera} raw</a> <img width="400" src="{display}">')
            details = f'<a href="{record["observation_id"]}.json">detection details</a>' if record.get("stage") == "analyzed" else ""
            crops = " ".join(f'<a href="{name}">candidate crop</a>' for name in record.get("candidate_crops", []))
            cards.append(f"<section><h2>{record['observation_id']} {html.escape(record.get('status', 'captured'))}</h2>"
                         f"<p>{record['time']} {html.escape(record['working_image'])} {details} {crops}</p>" + " ".join(links) + "</section>")
        (self.directory / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Phase 1 observations</title>'
            '<h1>Phase 1 observations — raw images are never annotated or overwritten</h1>' + "\n".join(cards),
            encoding="utf-8")


def install_observation_adapter(mission, camera_class) -> CandidateRecovery:
    recovery = CandidateRecovery()
    stores = {}
    capture = camera_class.capture
    perceive = mission.GpuMissionClient.perceive
    confirm = mission.confirm_detections
    event = mission.MissionLog.event
    latest = {"record": None}
    frame_review_cache = {}
    green_review_cache = {}
    capture_lock = threading.RLock()

    def unarchived_capture(camera, path):
        # Background transit photos must not replace the stopped observation
        # adapter's latest record or open the same V4L2 nodes concurrently.
        with capture_lock:
            return capture(camera,path)

    mission._phase1_capture_unarchived = unarchived_capture
    mission._phase1_raw_perceive = perceive
    mission._phase1_raw_confirm = confirm

    def store_for(path):
        root = Path(path).parent
        if root not in stores:
            stores[root] = EvidenceStore(root)
        return stores[root]

    def archived_capture(camera, path, *args, **kwargs):
        observer=getattr(mission,'_phase1_live_observation',None)
        if observer is not None:
            observer.stopped_capture()
        with capture_lock:
            info = capture(camera, path, *args, **kwargs)
            latest["record"] = store_for(path).capture(Path(path), info)
        return info

    def guarded_confirm(detections, target, images):
        grounded = [item for item in detections if item.get("source") not in
                    (RED_CUE_SOURCE, GREEN_CUE_SOURCE, "phase1_deepseek_red_block",
                     "phase1_deepseek_green_block")]
        checked = reject_copied_stereo_boxes(confirm(grounded, target, images))
        gate = recovery.context.get("task_region_view") or {}
        from grasp_range_integration.phase1 import visibility_route_planner as planner
        from grasp_range_integration.phase1.search_handoff import (
            near_region_geometry, ground_colour_boxes, live_target_gate)
        colour = str(target.get('color') or '').lower()
        cues = colour_visual_cues(images,colour) if colour in {'red','green'} else []
        if colour in {'red','green'} and str(target.get('class_name') or '').lower()=='block':
            checked = ground_colour_boxes(checked,cues)
            geometry = near_region_geometry(planner.LATEST_ROUTE or {},recovery.context.get('robot_pose'))
            if (geometry['near'] and gate.get('reason')=='approaching_task_region'
                    and any(c.get('complete_in_frame') for c in cues)):
                # Pixels authorize MODEL REVIEW, not region admission/success.
                gate = dict(gate,borderline_inspection_allowed=True,
                            pixel_review_near_region=True,**geometry)
        if gate.get('eligible') is False and str(target.get('color') or '').lower() in {'red','green'}:
            from grasp_range_integration.phase1 import visibility_route_planner as planner
            from grasp_range_integration.phase1.search_handoff import live_target_gate
            admitted = live_target_gate(gate, planner.LATEST_ROUTE or {},
                recovery.context.get('robot_pose'), checked,
                colour_visual_cues(images, str(target['color']).lower()))
            if admitted is not gate:
                gate = admitted
                recovery.context['task_region_view'] = gate
                recovery.context['search_phase'] = 'task_region_observation'
                print('目标已进入实际相机视野：近区域像素与模型候选一致，优先复核；不再因估计位置的临界差值丢弃。', flush=True)
        if (str(target.get("color") or "").lower() == "green"
                and str(target.get("class_name") or "").lower() == "block"
                and (gate.get("eligible") is True or gate.get("borderline_inspection_allowed") is True)
                and not any(item.get("dual_camera_confirmed") for item in checked)):
            # A copied Qwen left box must not suppress a real right-eye cue.
            # Coordinates come from each image's pixels, not from DeepSeek.
            try:
                key = tuple((str(Path(images[c]).resolve()), Path(images[c]).stat().st_mtime_ns)
                            for c in ("left", "right"))
            except (KeyError, OSError, TypeError):
                key = None
            if key is not None:
                if key not in green_review_cache:
                    cues = colour_visual_cues(images, "green")
                    pair = {c: next((i for i in cues if i['camera_id'] == c
                                    and i['complete_in_frame']), None) for c in ('left', 'right')}
                    pair = {c: i for c, i in pair.items() if i is not None}
                    from grasp_range_integration.phase1.colour_target_review import verify_green_cues
                    review = verify_green_cues(images, pair)
                    green_review_cache.clear()
                    green_review_cache[key] = (cues, pair, review)
                    print("绿色目标像素复核：候选相机={}，模型确认相机={}；未见的相机不补框。".format(
                        ','.join(pair) or '无', ','.join(review['accepted_cameras']) or '无'), flush=True)
                cues, pair, review = green_review_cache[key]
                for cue in deepcopy(cues):
                    cue['search_evidence']['deepseek_review'] = dict(review)
                    checked.append(cue)
                confirmed = []
                for c in review['accepted_cameras']:
                    base = pair[c]
                    confirmed.append({
                        'camera_id': c, 'class_name': 'block', 'source': 'phase1_deepseek_green_block',
                        'confidence': review.get('camera_confidences', {}).get(c, review['minimum_confidence']),
                        'bbox_1000': list(base['bbox_1000']),
                        'attributes': {'color': 'green'}, 'single_camera_match': True,
                        'complete_in_frame': True, 'confirmed_match': False, 'dual_camera_confirmed': False,
                        'search_evidence': {'source_ok': True, 'class_ok': True, 'colour_ok': True,
                            'bbox_valid': True, 'bbox_complete': True,
                            'pixel_geometry_source': GREEN_CUE_SOURCE, 'deepseek_review': dict(review)}})
                if (len(confirmed) == 2 and review['same_object']
                        and mission._same_stereo_target(confirmed[0], confirmed[1])):
                    for item in confirmed:
                        item.update(confirmed_match=True, dual_camera_confirmed=True,
                                    stereo_pair_id='deepseek_green_pixel_pair')
                checked.extend(confirmed)
        if (str(target.get("color") or "").lower() == "red"
                and str(target.get("class_name") or "").lower() == "block"
                and (gate.get("eligible") is True
                     or gate.get("borderline_inspection_allowed") is True)
                and not any(item.get("dual_camera_confirmed") for item in checked)):
            try:
                frame_key = tuple((str(Path(images[camera]).resolve()),
                                   Path(images[camera]).stat().st_mtime_ns)
                                  for camera in ("left", "right"))
            except (KeyError, OSError, TypeError):
                frame_key = None
            if frame_key is not None:
                if frame_key not in frame_review_cache:
                    cues = red_visual_cues(images)
                    pair = {camera: next((item for item in cues
                        if item["camera_id"] == camera and item["complete_in_frame"]), None)
                        for camera in ("left", "right")}
                    review = {"accepted": False, "reason": "two_complete_crops_required"}
                    prior = recovery.previous
                    current = next((item for item in cues
                                    if prior is not None
                                    and item["camera_id"] == prior.get("camera_id")), None)
                    corroborated = (prior is not None and current is not None
                                    and stable_candidate(prior, current,
                                                         after_turn=recovery.turns > 0))
                    if (all(pair.values()) and corroborated
                            and mission._same_stereo_target(pair["left"], pair["right"])):
                        from grasp_range_integration.phase1.red_target_deepseek import verify_red_block_pair
                        review = verify_red_block_pair(images, pair)
                    elif all(pair.values()):
                        review = {"accepted": False,
                                  "reason": "waiting_for_second_stable_observation"}
                    frame_review_cache.clear()  # Only the current pair may be reused.
                    frame_review_cache[frame_key] = (cues, pair, review)
                cues, pair, review = frame_review_cache[frame_key]
                for cue in cues:
                    current_cue = deepcopy(cue)
                    current_cue["search_evidence"]["deepseek_review"] = dict(review)
                    checked.append(current_cue)
                if review["accepted"] and (gate.get("eligible") is True
                                          or gate.get('pixel_review_near_region') is True):
                    for camera in ("left", "right"):
                        base = pair[camera]
                        checked.append({
                            "camera_id": camera, "class_name": "block",
                            "source": "phase1_deepseek_red_block",
                            "confidence": review["minimum_confidence"],
                            "bbox_1000": list(base["bbox_1000"]),
                            "attributes": {"color": "red", "description": "red small block on floor"},
                            "appearance_key": "red_block_deepseek_pair",
                            "single_camera_match": True, "complete_in_frame": True,
                            "confirmed_match": True, "dual_camera_confirmed": True,
                            "stereo_pair_id": "deepseek_red_pair_1",
                            "search_evidence": {
                                "source_ok": True, "class_ok": True, "colour_ok": True,
                                "bbox_valid": True, "bbox_complete": True,
                                "pixel_geometry_source": RED_CUE_SOURCE,
                                "deepseek_review": dict(review),
                                "independent_camera_frames": True,
                            },
                        })
        gate = live_target_gate(gate,planner.LATEST_ROUTE or {},
            recovery.context.get('robot_pose'),checked,cues)
        recovery.context['task_region_view'] = gate
        if gate.get('eligible'):
            recovery.context['search_phase'] = 'task_region_observation'
        for item in checked:
            if "eligible" in gate:
                item["task_region_eligible"] = bool(gate["eligible"])
                item.setdefault("search_evidence", {})["task_region_ok"] = bool(gate["eligible"])
            if gate.get("eligible") is False:
                # A real image may still supply archived candidate evidence.
                # Route interruption requires entering the task region view.
                item.update(confirmed_match=False, dual_camera_confirmed=False,
                            transit_candidate_allowed=True,
                            candidate_recovery_allowed=bool(gate.get("borderline_inspection_allowed")))
                item.pop("stereo_pair_id", None)
        if gate.get("eligible") is True:
            from grasp_range_integration.phase1.pursuit_entry_contract import assess_pursuit_entry
            try:
                with Image.open(images["left"]) as left_image, Image.open(images["right"]) as right_image:
                    image_sizes = {"left": left_image.size, "right": right_image.size}
                entry = assess_pursuit_entry(checked, image_sizes)
            except (KeyError, OSError, ValueError):
                entry = {"ready": False, "reason": "handoff_image_unavailable",
                         "stereo_pair_id": None, "steering_camera": None}
            if entry is not None and not entry["ready"]:
                for item in checked:
                    if (item.get("dual_camera_confirmed") and
                            (entry["stereo_pair_id"] is None or
                             item.get("stereo_pair_id") == entry["stereo_pair_id"])):
                        item["confirmed_match"] = False
                        item["dual_camera_confirmed"] = False
                        item["handoff_alignment_camera"] = entry["steering_camera"]
                        item.setdefault("search_evidence", {})["pursuit_entry"] = dict(entry)
                print("双摄找到目标，但后续追踪入口视野未就绪：{}；原站位换角度复拍，不提前交接。".format(
                    entry["reason"]), flush=True)
            elif entry is not None:
                for item in checked:
                    if item.get("stereo_pair_id") == entry["stereo_pair_id"]:
                        item.setdefault("search_evidence", {})["pursuit_entry"] = dict(entry)
        return checked

    def recorded_perceive(gpu, left, right, context):
        from grasp_range_integration.phase1 import visibility_route_planner as planner
        gate = planner.task_region_view_assessment(planner.LATEST_ROUTE or {}, context.get("robot_pose"),
                                                 view=planner.ACTIVE_VIEW)
        active = planner.ACTIVE_VIEW or {}
        stop_rule = str((active.get("visibility_zone") or {}).get("stop_rule") or "")
        pose = context.get("robot_pose") or []
        station = active.get("view_xy_m") or []
        colour = str((context.get("target") or {}).get("color") or "").lower()
        if (gate.get("eligible") is False
                and colour in {"red", "green"}
                and str((context.get("target") or {}).get("class_name") or "").lower() == "block"
                and stop_rule.startswith("single_camera")
                and recovery.previous is not None and len(pose) == 3 and len(station) == 2
                and math.hypot(float(pose[0]) - float(station[0]),
                               float(pose[1]) - float(station[1])) <= .25):
            region = (planner.LATEST_ROUTE or {}).get("target_region") or {}
            band = region.get("recognition_distance_m") or []
            center_distance = float(gate.get("region_center_distance_m") or math.inf)
            if len(band) == 2 and float(band[0]) <= center_distance <= float(band[1]) + .15:
                cues = colour_visual_cues({"left": Path(left), "right": Path(right)}, colour)
                peer = next((item for item in cues if item["camera_id"] ==
                             recovery.previous.get("camera_id")), None)
                if peer is not None and stable_candidate(
                        recovery.previous, peer, after_turn=recovery.turns > 0 or recovery.target_lock is not None):
                    gate = dict(gate, eligible=True,
                                reason="live_" + colour + "_reacquisition_at_station",
                                visual_evidence_requires_confirmation=True)
        context = deepcopy(context)
        context["task_region_view"] = gate
        context["search_phase"] = "task_region_observation" if gate["eligible"] else "approaching_task_region"
        recovery.context = deepcopy(context)
        store = store_for(left)
        record = store.by_working_path.get(str(Path(left).resolve()))
        request_context = deepcopy(context)
        if gate["eligible"] is False:
            request_context["search_phase"] = "transit_candidate_observation"
            request_context["candidate_only"] = True
            print("地图任务视野尚未满足：coverage={:.1%}；识别并存候选，临界观察站允许有限复查，其他区域不抢占导航。".format(
                float(gate.get("estimated_work_area_coverage") or 0)), flush=True)
        if record:
            # Unique ID prevents server-side reuse across replan/recheck frames.
            request_context["observation_id"] = f"{Path(left).parent.name}_{record['observation_id']}"
        request_context["camera_grounding_contract"] = "Independent boxes per camera; do not copy coordinates between images."
        if record:
            record["model_called"] = True
        try:
            value = perceive(gpu, left, right, request_context)
            checked = guarded_confirm(list(value.get("detections") or []), context.get("target") or {},
                                      {"left": Path(left), "right": Path(right)})
            gate = recovery.context.get('task_region_view') or gate
            context['task_region_view'] = gate
            context['search_phase'] = recovery.context.get('search_phase', context['search_phase'])
        except Exception as exc:
            if record:
                store.analyze(record, [], context, error=str(exc))
            raise
        if record:
            store.analyze(record, checked, context)
        value.update(detections=checked, task_region_view=gate, perception_skipped=False)
        return value

    def recorded_event(log, event_type, **fields):
        from grasp_range_integration.phase1 import turn_policy

        def telemetry(kind, **data):
            value = {"time": time.time(), "type": kind, **data}
            with (log.directory / "phase1_motion.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

        turn_policy.TELEMETRY_CALLBACK = telemetry
        record = latest["record"]
        if event_type == "observation_complete" and record:
            fields.update({"observation_id": record["observation_id"],
                           "archived_images": {k: "observations/" + v for k, v in record["raw_images"].items()},
                           "evidence_status": record.get("status")})
            if record.get("status") == "single_camera_candidate":
                fields["confirmation_reason"] = "单摄候选已归档；适配层先在当前站位复查，不直接前往下一站"
            if record.get("status") == "red_visual_cue":
                fields["confirmation_reason"] = "图像中有红色区域但类别未经模型确认；先原站位换角度复拍，不宣告找到"
            if record.get("status") == "borderline_single_camera_candidate":
                fields["confirmation_reason"] = "临界视野可信单摄候选，允许原地有限复查；不能直接宣告找到"
                fields["task_region_view"] = record.get("context",{}).get("task_region_view")
            if record.get("status") == "approaching_task_region":
                fields["confirmation_reason"] = "柜子工作区尚未进入有效视野；起点仅留证，继续导航"
                fields["task_region_view"] = record.get("context", {}).get("task_region_view")
            if record.get("status") == "transit_candidate_archived_continue_approach":
                fields["confirmation_reason"] = "区域外候选已留档；不抢占导航、不改写观察站位，继续前往柜子有效视野"
                fields["task_region_view"] = record.get("context", {}).get("task_region_view")
        return event(log, event_type, **fields)

    fuse = mission._fuse_tracking_pose

    def recorded_fuse(predicted_pose, result, *args, **kwargs):
        fused = fuse(predicted_pose, result, *args, **kwargs)
        from grasp_range_integration.phase1.turn_policy import emit
        emit("pose_feedback", predicted_pose=[predicted_pose.x_m, predicted_pose.y_m, predicted_pose.yaw_rad],
             fused_pose=[fused.x_m, fused.y_m, fused.yaw_rad],
             map_pose=[result.pose.x_m, result.pose.y_m, result.pose.yaw_rad] if result else None)
        return fused

    camera_class.capture = archived_capture
    mission.confirm_detections = guarded_confirm
    mission.GpuMissionClient.perceive = recorded_perceive
    mission.MissionLog.event = recorded_event
    mission._fuse_tracking_pose = recorded_fuse
    return recovery
