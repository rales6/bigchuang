"""Region-arrival skill layered on the released, adaptive phase-1 driver.

No second motor controller: the existing mission owns lidar, BLE, speed
experience, relocalization, avoidance and the continuous glass boundary.
This adapter replaces search completion with region arrival + anchor review.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import subprocess
import re


def entity_colors(text):
    text = str(text or "").lower()
    names = {"red": "红", "green": "绿", "blue": "蓝", "yellow": "黄",
             "white": "白", "black": "黑", "orange": "橙", "purple": "紫", "gray": "灰"}
    return {english for english, chinese in names.items()
            if chinese in text or re.search(r"\b" + english + r"\b", text)}


def entities_for(semantic, target):
    query = str(target.get("query") or "").strip().lower()
    requested = str(target.get("landmark_id") or "")
    items = semantic.get("objects") or semantic.get("landmarks") or []
    if not items and isinstance(semantic.get("entities"), dict):
        items = [{"id": k, **v} for k, v in semantic["entities"].items() if isinstance(v, dict)]
    group = ("chair" if "椅" in query or "chair" in query else
             "cabinet" if "柜" in query or "cabinet" in query else
             "table" if "桌" in query or "table" in query else "")
    rows = []
    required_colors = entity_colors(query)
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("behind_glass") is True or item.get("search_region_allowed") is False:
            continue  # An explicit model ID cannot override scene access restrictions.
        aliases = item.get("aliases") or []
        text = " ".join(str(item.get(k) or "") for k in ("name", "label", "semantic_label", "category", "color"))
        text = (text + " " + " ".join(str(a) for a in aliases if isinstance(a, str))).lower()
        kind = str(item.get("category") or item.get("type") or "").lower()
        # Reviewed chair names override stale geometric category=table.
        chair = "椅" in text or "chair" in text
        matches = ((chair if group == "chair" else "柜" in text or "cabinet" in text
                    if group == "cabinet" else ("桌" in text or "table" in text) and not chair
                    if group == "table" else query and query in text))
        if requested:
            matches = str(item.get("id") or "") == requested
        if group and kind in {"marker", "sign", "door", "window", "wall", "glass_wall"}:
            matches = False  # A mark ON a cabinet is not a cabinet.
        colors = entity_colors(text)
        if required_colors and colors and not required_colors <= colors:
            continue  # Never silently replace yellow with blue/green.
        xy = item.get("map_xy") or item.get("xy")
        if matches and isinstance(xy, (list, tuple)) and len(xy) >= 2:
            point = [float(xy[0]), float(xy[1])]
            if all(math.isfinite(v) for v in point):
                rows.append({"id": str(item.get("id") or ""), "name": str(item.get("name") or text),
                             "center_xy_m": point, "class_name": group or str(item.get("category") or ""),
                             "position_verified": item.get("position_verified") is True,
                             "position_uncertainty_m": float(item.get("position_uncertainty_m") or .85),
                             "requested_color": next(iter(required_colors), ""),
                             "color_in_map": sorted(colors),
                             "requires_color_recheck": bool(required_colors and not colors)})
    if required_colors and rows:
        explicit = [row for row in rows if not row["requires_color_recheck"]]
        if explicit:
            rows = explicit
        elif len(rows) > 1:
            raise ValueError("地图有多个未标颜色的候选，不能忽略颜色随意选地点：" + query)
    if not rows:
        raise ValueError("地图没有与放置/导航区域匹配的实体：" + query)
    return rows


def select_region(known_map, start_xy, candidates, *, excluded=()):
    """Nearest reachable FREE stand-off, never navigate to furniture centre."""
    choices = []
    for entity in candidates:
        cx, cy = entity["center_xy_m"]
        for radius in (.90, 1.10, 1.30, 1.50, 1.70):
            for angle in range(0, 360, 15):
                yaw = math.radians(angle)
                goal = (cx + radius * math.cos(yaw), cy + radius * math.sin(yaw))
                if any(math.dist(goal, old) < .25 for old in excluded):
                    continue
                # path() can snap an unsafe goal; validate goal and final cell.
                col, row = known_map.world_to_grid((goal,))[0]
                if not known_map.in_bounds(col, row) or not known_map.traversable[row, col]:
                    continue
                path = known_map.path(start_xy, goal)
                if not path or math.dist(path[-1], goal) > .12:
                    continue
                length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
                goal = tuple(float(v) for v in path[-1])
                choices.append({**entity, "approach_xy_m": list(goal),
                                "path_xy_m": [[float(x), float(y)] for x, y in path],
                                "path_length_m": float(length), "standoff_m": math.dist(goal, (cx, cy)),
                                "expanded_station_search": radius > 1.30,
                                "target_yaw_rad": math.atan2(cy - goal[1], cx - goal[0])})
    if not choices:
        from grasp_range_integration.phase1.visibility_route_planner import NoReachableVisibilityError
        raise NoReachableVisibilityError("区域旁站位暂不可达；交回换视角/定位恢复，不把家具中心当作行驶目标")
    preferred = [row for row in choices if not row["expanded_station_search"]]
    return min(preferred or choices, key=lambda value: (value["path_length_m"], value["id"], value["standoff_m"]))


def load_map(release):
    from raspberry_pi.known_map_navigation import KnownOccupancyMap
    from grasp_range_integration.phase1.glass_boundary_guard import load_verified_boundaries, attach_boundaries
    pgm, yaml = release / "ble_scene_17.pgm", release / "ble_scene_17.yaml"
    known_map = KnownOccupancyMap.load(pgm, yaml, clearance_m=.30)
    attach_boundaries(known_map, load_verified_boundaries(
        release / "grasp_range_integration/phase1/glass_boundaries.json", pgm, yaml))
    # Match the live planner: exclude the glass half-plane from A* too.
    known_map.traversable &= ~known_map._phase1_glass_keepout
    return known_map


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def matching_anchors(detections, expected_class, requested_color="", *, allow_warm_cabinet=False):
    matches = []
    for item in detections:
        if not isinstance(item, dict):
            continue
        box = item.get("bbox_1000")
        try:
            confidence = float(item.get("confidence") or 0)
            valid_box = (isinstance(box, (list, tuple)) and len(box) == 4
                         and all(math.isfinite(float(v)) and 0 <= float(v) <= 1000 for v in box)
                         and float(box[0]) < float(box[2]) and float(box[1]) < float(box[3]))
        except (ValueError, TypeError):
            continue
        attributes = item.get("attributes") or {}
        if not isinstance(attributes, dict):
            attributes = {}
        # A label ON a door is not the door/cabinet's body colour. Structured
        # body colour takes precedence over a stale appearance fingerprint;
        # never mine arbitrary description words for the requested colour.
        body_color = attributes.get("color") or item.get("color")
        observed_colors = entity_colors(body_color or str(item.get("appearance_key") or "").replace("_", " "))
        # The bound indoor wooden cabinet is called yellow/beige/light_brown
        # across otherwise identical captures. Do not apply this equivalence
        # to blocks, arbitrary furniture, white bodies or description labels.
        warm_body = str(body_color or "").lower().replace("_", " ").replace("-", " ").strip()
        warm_match = (allow_warm_cabinet and expected_class == "cabinet" and requested_color == "yellow"
                      and not observed_colors - {"yellow"}
                      and warm_body in {"beige", "tan", "light brown", "light wood", "natural wood",
                                        "wood", "wood color", "木色", "原木色", "浅木色", "浅棕色", "米色", "米黄色"}
                      and confidence >= .80)
        if requested_color and requested_color not in observed_colors and not warm_match:
            continue  # A generic cabinet/blue cabinet does not confirm a yellow one.
        subject = re.split(r"[，。；,;]", str(attributes.get("description") or "").lower(), maxsplit=1)[0]
        cabinet_subject = "柜" in subject or bool(re.search(r"\b(cabinet|cupboard|wardrobe)\b", subject))
        other_subject = ("门" in subject or "窗" in subject or "墙" in subject
                         or bool(re.search(r"\b(door|window|wall)\b", subject)))
        if expected_class == "cabinet" and other_subject and not cabinet_subject:
            continue  # Explicit class/subject contradiction in the model output.
        if (str(item.get("class_name") or "").lower() == expected_class
                and math.isfinite(confidence) and .55 <= confidence <= 1.0
                and item.get("camera_id") in {"left", "right"} and valid_box):
            if (float(box[2])-float(box[0]))*(float(box[3])-float(box[1])) > 900000:
                continue  # An entire-frame cabinet box is not object evidence.
            matches.append(item)
    return matches


def anchor_seen(detections, expected_class, requested_color=""):
    return bool(matching_anchors(detections, expected_class, requested_color))


def target_view_evidence(pose, selected):
    if len(pose) != 3 or not all(math.isfinite(float(v)) for v in pose):
        return {"eligible": False, "reason": "missing_actual_pose"}
    dx, dy = (selected["center_xy_m"][0]-pose[0], selected["center_xy_m"][1]-pose[1])
    distance = math.hypot(dx, dy)
    bearing = math.atan2(math.sin(math.atan2(dy, dx)-pose[2]), math.cos(math.atan2(dy, dx)-pose[2]))
    uncertainty = max(0., min(1.5, float(selected.get("position_uncertainty_m", .85))))
    allowance = 41.+15.+min(28., math.degrees(math.atan2(uncertainty, max(.2, distance))))
    return {"eligible": abs(math.degrees(bearing)) <= allowance,
            "target_bearing_deg": round(math.degrees(bearing), 2),
            "bearing_allowance_deg": round(allowance, 2), "distance_to_map_anchor_m": round(distance, 3),
            "map_entity_id": selected["id"], "metric_distance_is_map_estimate": True}


def arrival_proximity(pose, selected):
    """Destination is a usable neighbourhood, not one exact station cell."""
    evidence = target_view_evidence(pose, selected)
    if "distance_to_map_anchor_m" not in evidence:
        return False
    return (.45 <= evidence["distance_to_map_anchor_m"] <= 1.8
            or math.dist(pose[:2], selected["approach_xy_m"]) <= .60)


def install_region_skill(mission, args, job_path):
    """Called AFTER all released navigation/recovery hooks are installed."""
    from grasp_range_integration.phase1 import visibility_route_planner as visibility
    path = Path(job_path).resolve(strict=True)
    job = json.loads(path.read_text(encoding="utf-8"))
    if job.get("schema") != "robot_region_job.v1" or not job.get("candidates"):
        raise ValueError("区域导航任务契约无效")
    state = {"selected": None, "observation": None, "visited": [], "captures": 0, "driver": None}
    # This skill has a freshly initialized pose and keeps consuming real
    # adjacent scans. Do not expire a sound estimate just because the route
    # has taken 30 seconds; uncertainty/live-obstacle limits still apply.
    mission._phase1_region_continuous_odometry = True
    perceive = mission.GpuMissionClient.perceive
    driver_class = mission.N10LidarDriver

    class RegionDriver(driver_class):
        def __init__(self, *values, **options):
            super().__init__(*values, **options)
            state["driver"] = self

    mission.N10LidarDriver = RegionDriver

    def region_localize(*positional, **keywords):
        # Do not wrap global_localize and then enter the frozen initializer:
        # its compulsory cross-position moves reject an already strong fix.
        from .carry_localization import localize_carry
        from .region_localization import localize_region
        from grasp_range_integration.phase1.startup_route_recovery import wrap_startup_route_recovery
        if job.get("carrying"):
            initializer = lambda *values, **options: localize_carry(
                mission, job.get("pose_hint"), *values, **options)
        else:
            initializer = lambda *values, **options: localize_region(mission, *values, **options)
        pose, scan = wrap_startup_route_recovery(mission, initializer)(*positional, **keywords)
        job['initial_localization'] = {**getattr(mission, '_region_start_localization', {}),
            'pose': [pose.x_m, pose.y_m, pose.yaw_rad], 'yaw_deg': math.degrees(pose.yaw_rad),
            'final_sequence': getattr(scan, 'sequence', None)}
        save_json(path, job)
        return pose, scan

    def route(known_map, start_xy, request):
        if hasattr(known_map, "_phase1_glass_keepout"):
            known_map.traversable &= ~known_map._phase1_glass_keepout
        selected = select_region(known_map, start_xy, job["candidates"], excluded=state["visited"])
        # Freeze the nearest ENTITY after the first fresh localization. A
        # recovery changes station/angle, not which chair is "nearest".
        if state["selected"] is None:
            job["candidates"] = [row for row in job["candidates"] if row["id"] == selected["id"]]
            job["selection_pose_xy_m"] = list(start_xy)
        state["selected"] = selected
        state["observation"] = None
        job.update(status="navigating", selected=selected)
        # Keep the original route as well as replans; replacing `selected`
        # must not hide the path on which an execution was actually based.
        job.setdefault("route_history", []).append({"planned_at": time.time(),
            "map_entity_id": selected["id"], "start_xy_m": list(start_xy),
            "goal_xy_m": selected["approach_xy_m"], "path_xy_m": selected["path_xy_m"],
            "path_length_m": selected["path_length_m"]})
        job["route_history"] = job["route_history"][-24:]
        save_json(path, job)
        # Two views at a station cover the anchor with modest yaw changes;
        # only generate a new station after these fail.
        views = []
        for offset in (0, -25, 25):
            views.append({"view_xy_m": selected["approach_xy_m"],
                          "path_xy_m": selected["path_xy_m"] if not views else [],
                          "path_length_m": selected["path_length_m"] if not views else 0.,
                          "target_yaw_rad": selected["target_yaw_rad"] + math.radians(offset),
                          "capture_after_stop": True,
                          "visibility_zone": {"stop_rule": "turn_at_station_then_capture",
                              "target_region_center_xy_m": selected["center_xy_m"],
                              "target_region_radius_m": .30, "camera_horizontal_fov_deg": 82.,
                              "recognition_min_distance_m": .45, "recognition_max_distance_m": 1.8,
                              "minimum_work_area_coverage": .1}})
        result = {"schema": "robot_region_route.v1", "views": views, "next_station_available": True,
                  "start_xy_m": list(start_xy), "target_region": {"center_xy_m": selected["center_xy_m"],
                      "radius_m": .30, "recognition_distance_m": [.45, 1.80],
                      "minimum_work_area_coverage": .1, "camera_horizontal_fov_deg": 82.}}
        visibility.LATEST_ROUTE, visibility.ACTIVE_VIEW = result, views[0]
        print("区域导航：{}，自由站位距区域{:.2f}m；不重新抓取或松爪。".format(
            selected["name"], selected["standoff_m"]), flush=True)
        return result

    def review(gpu, left, right, context):
        selected = state["selected"]
        # Localization is not destination recognition: preserve its landmark
        # request, including the first startup capture before a route exists.
        if selected is None or context.get("search_phase") == "localization" or context.get("purpose") == "permanent_landmark_localization":
            return perceive(gpu, left, right, context)
        context = {**context, "target": {
            "query": (job.get("target") or {}).get("query") or selected["name"],
            "class_name": selected["class_name"], "color": selected.get("requested_color") or None,
            "semantic_anchor_name": selected["name"], "map_entity_id": selected["id"]}}
        try:
            raw = perceive(gpu, left, right, context)
        except Exception as exc:
            # Archive already records the error. Consume this angle, then
            # inspect the next angle/station instead of aborting the mission.
            raw = {"detections": [], "perception_error": type(exc).__name__ + ": " + str(exc)}
        detections = raw.get("detections") or []
        # Region identity only needs ONE camera. No stereo hull requirement
        # is appropriate for a chair: it is not going into the gripper.
        pose = context.get("robot_pose") or []
        anchors = matching_anchors(detections, selected["class_name"], selected.get("requested_color", ""),
                                   allow_warm_cabinet=bool(selected.get("requires_color_recheck")))
        view_evidence = target_view_evidence(pose, selected)
        matched = bool(anchors) and view_evidence["eligible"]
        near = arrival_proximity(pose, selected)
        placement_clearance = False
        clearance_detail = {}
        if job.get("carrying"):
            import numpy as np
            from grasp_range_integration.phase1.rotation_guard import installed_body_mask
            try:
                scan = state["driver"].stream.latest_after(None, timeout_s=.5)
                angles, distances = np.asarray(scan.angles_rad), np.asarray(scan.distances_m)
                bearings = np.arctan2(np.sin(angles), np.cos(angles))
                front = ((np.abs(bearings) <= math.radians(45)) & np.isfinite(distances)
                         & (distances > .05) & ~installed_body_mask(angles, distances))
                minimum = float(distances[front].min()) if front.any() else 0.
                placement_clearance = int(front.sum()) >= 20 and minimum >= .70
                clearance_detail = {"minimum_external_forward_range_m": minimum,
                                    "samples": int(front.sum()), "required_range_m": .70}
            except (RuntimeError, AttributeError, ValueError) as exc:
                clearance_detail = {"unverified": str(exc)}
        state["observation"] = {"near": near, "anchor_visual_confirmed": matched,
                                "map_entity_id": selected["id"], "target_view_evidence": view_evidence,
                                "matched_anchors": anchors,
                                "requested_color": selected.get("requested_color", ""),
                                "arrival_evidence": "neighbourhood_and_single_camera_target" if near and matched else "unconfirmed",
                                "placement_clearance": placement_clearance, "clearance": clearance_detail,
                                "pose": pose, "images": {"left": str(left), "right": str(right)}}
        state["captures"] += 1
        job["last_observation"] = state["observation"]
        save_json(path, job)
        return raw

    def next_step(_gpu, context):
        observed = state["observation"] or {}
        # A coarse layout point is a search prior, not evidence of arrival.
        # This applies equally to empty-gripper navigation and carrying.
        if (observed.get("near") and observed.get("anchor_visual_confirmed")
                and observed.get("map_entity_id") == (state["selected"] or {}).get("id")
                and (not job.get("carrying") or observed.get("placement_clearance"))):
            job.update(status="arrived", arrived_at=time.time(), arrival=observed)
            save_json(path, job)
            return {"action": "report_success", "terminal": True, "reason": "区域站位与指定目标视觉复核完成"}
        remaining = context.get("remaining_view_indices") or []
        event = context.get("latest_event") or {}
        # A blocked station must replan, not spend all retries at the same spot.
        blocked = event.get("type") in {"navigation_no_progress", "path_invalid", "temporary_obstacle"}
        if remaining and not blocked:
            index = remaining[0]
            visibility.ACTIVE_VIEW = visibility.LATEST_ROUTE["views"][index]
            return {"action": "navigate", "terminal": False, "next_view_index": index,
                    "reason": "先完成本站其余观察方向"}
        if state["selected"]:
            state["visited"].append(state["selected"]["approach_xy_m"])
        if state["captures"] >= 12:
            return {"action": "report_failure", "terminal": True,
                    "reason": "多站复核仍不能确认放置地点；保留闭爪携带状态，请检查区域标注/视觉证据"}
        return {"action": "replan", "terminal": False, "reason": "换可达站位继续区域复核"}

    mission._localize = region_localize
    original_command = getattr(mission, '_strict_command_for_pose', None)
    if callable(original_command):
        def trace_command(command_args, pose, waypoint):
            selected = state['selected']
            current = [pose.x_m, pose.y_m, pose.yaw_rad]
            views = (visibility.LATEST_ROUTE or {}).get('views') or []
            first_view = bool(views) and visibility.ACTIVE_VIEW is views[0]
            # Only the final approach, never skip route intermediates or the
            # later intentional sweeps. Capture first when the target area is
            # already in view; perception, not this shortcut, decides arrival.
            inspect_here = (selected is not None and first_view
                            and math.dist(waypoint, selected['approach_xy_m']) <= .25
                            and math.dist(current[:2], selected['approach_xy_m']) <= .60
                            and arrival_proximity(current, selected)
                            and target_view_evidence(current, selected)['eligible'])
            command = (0, 0, True) if inspect_here else original_command(command_args, pose, waypoint)
            if selected is not None:
                trace = job.setdefault('execution_trace', [])
                previous = trace[-1] if trace else None
                angle = (abs(math.atan2(math.sin(current[2]-previous['pose'][2]),
                         math.cos(current[2]-previous['pose'][2]))) if previous else math.inf)
                if (previous is None or math.dist(current[:2], previous['pose'][:2]) >= .05
                        or angle >= math.radians(5) or list(waypoint) != previous['waypoint_xy_m']):
                    trace.append({'time': time.time(), 'pose': current, 'waypoint_xy_m': list(waypoint),
                        'map_entity_id': selected['id'], 'linear_mm_s': command[0],
                        'angular_mrad_s': command[1], 'waypoint_reached': bool(command[2]),
                        'region_inspection_shortcut': inspect_here,
                        'pose_is_estimate': True})
                    job['execution_trace'] = trace[-2000:]
                    # Keep control-loop I/O small while binocular recording
                    # is active. Observations/route changes also flush this job.
                    if time.monotonic()-state.get('trace_saved_at', -math.inf) >= 2.:
                        save_json(path, job)
                        state['trace_saved_at'] = time.monotonic()
            return command
        mission._strict_command_for_pose = trace_command
    # Do NOT replace mission.build_search_route: its wrapper updates the
    # rotation guard's known-map context and dynamic-obstacle path reserve.
    visibility.REGION_ROUTE_BUILDER = route
    mission.GpuMissionClient.perceive = review
    mission.GpuMissionClient.next = next_step
    # Archive remains installed; do not apply grasp-target stereo gating to
    # destination furniture. No fabricated detections are introduced.
    mission.confirm_detections = lambda detections, target, images: detections
    args.max_replans = max(args.max_replans, 12)
    print("机器人技能模式：区域到达；继承自适应速度、避障和换视角恢复。", flush=True)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--resolve", type=Path)
    mode.add_argument("--run-job", type=Path)
    args = parser.parse_args()
    job_path = args.resolve or args.run_job
    job = json.loads(job_path.read_text(encoding="utf-8"))
    release = Path(os.environ["FINAL_WORK_SEMANTIC_RELEASE"])
    if args.resolve:
        result = select_region(load_map(release), job["pose_hint"][:2], job["candidates"])
        print(json.dumps(result, ensure_ascii=False))
        return
    # Only the local adapter can create this fixed navigation decision. It
    # exposes no arm authority and is not a model-provided motor program.
    first = job["candidates"][0]
    decision = {"schema": "robot_object_search_plan.v1", "status": "executable",
        "task_id": "region_" + job["job_id"], "instruction": "前往" + str(job["target"]["query"]),
        "target": {"query": first["name"], "class_name": first["class_name"], "requested_count": 1},
        "anchor": {"search_region_id": first["id"], "position_verified": False},
        "route_request": {"mode": "semantic_search_region", "center_xy_m": first["center_xy_m"],
            "planning_policy": "semantic_visibility_regions.v1", "inspect_current_view_first": True,
            "stop_on_visibility_entry": True, "radius_m": .3, "maximum_views": 12},
        "steps": [{"action": "localize"}, {"action": "navigate_to_search_area"}, {"action": "capture_at_stop"}],
        "capabilities": {"grasp_enabled": False, "arm_commands_allowed": False},
        "phase_control": {"phase": "phase1_find_only", "requested_final_action": "none",
                          "terminal_condition": "target_found_and_vehicle_stopped"}}
    decision_path = job_path.with_name(job_path.stem + "_navigation.json")
    save_json(decision_path, decision)
    env = os.environ.copy()
    env["SEARCH_START_POSE_HINT"] = ""
    env["FINAL_WORK_REGION_JOB"] = str(job_path.resolve())
    script = release / "grasp_range_integration/phase1/run_phase1_find_only.sh"
    code = subprocess.call(["bash", str(script), str(decision_path), "--execute-find-only"], env=env)
    final = json.loads(job_path.read_text(encoding="utf-8"))
    if code or final.get("status") != "arrived":
        final.update(status="not_arrived", returncode=code,
                     message="区域导航未完成；保留携带状态，不能自动松爪")
        save_json(job_path, final)
        raise SystemExit(code or 2)
    print("区域到达交接通过：" + json.dumps(final["selected"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
