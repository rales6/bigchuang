"""Plan stopped camera views of a semantic confidence region, not target points."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


POLICY = "semantic_visibility_regions.v1"
LATEST_ROUTE: dict[str, Any] | None = None
LATEST_KNOWN_MAP = None
ACTIVE_VIEW: dict[str, Any] | None = None
VISITED_POINTS: list[tuple[float, float]] = []
OBSERVATIONS_COMPLETED = 0
APPROACH_OBSERVATIONS = 0
MINIMUM_STANDOFF_M = 0.80


class NoReachableVisibilityError(RuntimeError):
    """No safe global station; distinct from invalid inputs or hardware faults."""


def task_region_view_assessment(route, robot_pose, *, known_map=None, view=None):
    """Authorize task perception by actual stopped pose, never a planned point.

Generic/non-semantic adapter callers retain their old behavior. Semantic
phase-one tasks fail closed without region, pose or map evidence. This is an
approximate visibility prior, not proof that a detection lies in the region.
"""
    if route.get("mode") != "semantic_visibility_regions":
        return {"eligible": True, "reason": "non_semantic_route"}
    region = route.get("target_region") or {}
    center = region.get("center_xy_m") or []
    result = {"eligible": False, "reason": "missing_region_pose_or_map"}
    known_map = LATEST_KNOWN_MAP if known_map is None else known_map
    if len(center) < 2 or len(robot_pose or []) != 3 or known_map is None:
        return result
    try:
        x, y, yaw = (float(v) for v in robot_pose)
        cx, cy = (float(v) for v in center[:2])
        radius = float(region["radius_m"])
        low, high = (float(v) for v in region["recognition_distance_m"])
        minimum = float(region["minimum_work_area_coverage"])
        zone = (view or {}).get("visibility_zone") or {}
        fov = float(zone.get("camera_horizontal_fov_deg", region.get("camera_horizontal_fov_deg", 82)))
        margin = float(zone.get("fov_margin_deg", region.get("fov_margin_deg", 10)))
    except (TypeError, ValueError, KeyError):
        return result
    if (not all(math.isfinite(v) for v in (x, y, yaw, cx, cy, radius, low, high, minimum, fov, margin))
            or radius <= 0 or not 0 < low < high <= 3 or not .1 <= minimum <= .75
            or not 45 <= fov <= 120 or not 0 <= margin < fov / 2):
        result["reason"] = "invalid_region_contract"
        return result
    if route.get('navigation_stage') == 'approach' and math.hypot(x-cx,y-cy) > radius+1.45:
        return {"eligible": False, "reason": "approaching_task_region",
                "navigation_stage": "approach", "estimated_work_area_coverage": 0.,
                "actual_robot_pose": [x,y,yaw]}
    coverage = _region_camera_coverage(
        known_map, (x, y), yaw, (cx, cy), radius,
        min_range_m=low, max_range_m=high,
        max_heading_error=math.radians(fov / 2 - margin),
    )
    result.update(eligible=coverage >= minimum,
                  reason="task_region_in_usable_view" if coverage >= minimum else "approaching_task_region",
                  estimated_work_area_coverage=round(coverage, 4), minimum_work_area_coverage=minimum,
                  recognition_distance_m=[low, high],
                  region_center_distance_m=round(math.hypot(x - cx, y - cy), 3),
                  actual_robot_pose=[x, y, yaw])
    from grasp_range_integration.phase1.arrival_visibility import add_inspection_gate
    return add_inspection_gate(result, [x,y,yaw], view)


def work_area_path(known_map, start, goal, center, radius_m, *, original=None,
                   exact_forward_standoff=False):
    """Exclude the uncertain floor-object region from both A* and smoothing.

    Only the in-memory planning mask is temporarily restricted; occupied
    lidar geometry and localization are unchanged. The same hook is used by
    the protected executor when it rebuilds a route to a station.
    """
    handoff = getattr(known_map, '_phase1_route_handoff', None)
    if (handoff and handoff.get('deferred_prediction') and not handoff.get('pose_consumed')
            and math.dist(tuple(start), handoff['old_route_start']) <= .08):
        # The frozen loop rebuilds its path before reading its next scan.
        # Use the recovered start for that rebuild as well as for prediction;
        # do not navigate back toward waypoints based on the pre-recovery pose.
        p = handoff['result'].pose
        start = (p.x_m, p.y_m)
    path_function = original or getattr(known_map, "_phase1_original_path", known_map.path)
    distance = math.hypot(start[0] - goal[0], start[1] - goal[1])
    recovery = str(((ACTIVE_VIEW or {}).get("visibility_zone") or {}).get("stop_rule", "")).startswith("single_camera")
    if distance <= 0.01 or recovery and distance <= 0.06:
        return [tuple(start)]  # Stationary capture/turn, including an initial near view.
    if any(math.hypot(point[0] - center[0], point[1] - center[1]) < radius_m
           for point in (start, goal)):
        return []
    # A verified straight-ahead candidate is checked continuously below;
    # its endpoint may sit in the last map cell outside the physical .80 m
    # standoff. Ordinary multi-turn routes retain the extra grid-cell margin.
    grid_margin = 0. if exact_forward_standoff else known_map.resolution_m
    key = (tuple(center), radius_m, grid_margin, known_map.traversable.shape)
    cached = getattr(known_map, "_phase1_buffer_mask", None)
    if cached is None or cached[0] != key:
        rows, columns = np.indices(known_map.traversable.shape)
        world = known_map.grid_to_world(np.stack((columns, rows), axis=-1))
        outside = np.hypot(world[..., 0] - center[0], world[..., 1] - center[1]) >= radius_m + grid_margin
        known_map._phase1_buffer_mask = (key, outside)
    else:
        outside = cached[1]
    previous = known_map.traversable
    try:
        known_map.traversable = previous & outside
        path = path_function(start, goal)
    finally:
        known_map.traversable = previous
    # Snapped endpoints and every continuous segment must obey the buffer.
    points = [tuple(start)] + list(path) + [tuple(goal)]
    if not path:
        return []
    for a, b in zip(points, points[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length2 = dx * dx + dy * dy
        fraction = 0 if length2 == 0 else max(0, min(1, ((center[0] - a[0]) * dx + (center[1] - a[1]) * dy) / length2))
        if math.hypot(a[0] + fraction * dx - center[0], a[1] + fraction * dy - center[1]) < radius_m:
            return []
    return path


def _angle_error(left: float, right: float) -> float:
    return math.atan2(math.sin(left - right), math.cos(left - right))


def short_forward_view_path(known_map, start, goal, center, radius_m):
    """Ask the SAME map wrapper for a short straight viewpoint adjustment.

    The reserve wrapper returns exact world endpoints only with a continuous
    check of its all-round inflated mask. It is a proposed path, NOT a motor
    authorization. Nothing relaxes unknown/glass/occupancy or the work-area
    mask. Restore this planning-only flag on every exit.
    """
    previous = getattr(known_map, "_phase1_short_view_probe", False)
    try:
        known_map._phase1_short_view_probe = True
        return work_area_path(known_map, start, goal, center, radius_m)
    finally:
        known_map._phase1_short_view_probe = previous


def _path_length(start: tuple[float, float], path: list[tuple[float, float]]) -> float:
    previous = start
    length = 0.0
    for point in path:
        length += math.hypot(float(point[0]) - previous[0], float(point[1]) - previous[1])
        previous = (float(point[0]), float(point[1]))
    return length


def _target_region_visible(
    known_map,
    viewpoint: tuple[float, float],
    center: tuple[float, float],
    region_radius_m: float,
    *,
    hit_region_center: tuple[float, float] | None = None,
) -> bool:
    """Accept a clear ray or a first mapped surface inside the target region."""
    start_cell = known_map.world_to_grid((viewpoint,))[0]
    target_cell = known_map.world_to_grid((center,))[0]
    samples = max(2, int(math.ceil(math.hypot(*(target_cell - start_cell)))))
    first_hit = None
    for fraction in np.linspace(0.0, 1.0, samples + 1):
        column = int(round(start_cell[0] + (target_cell[0] - start_cell[0]) * fraction))
        row = int(round(start_cell[1] + (target_cell[1] - start_cell[1]) * fraction))
        if not known_map.in_bounds(column, row):
            return False
        if known_map.occupied[row, column]:
            first_hit = known_map.grid_to_world(((column, row),))[0]
            break
    if first_hit is None:
        return True
    accepted_center = center if hit_region_center is None else hit_region_center
    return math.hypot(
        float(first_hit[0]) - accepted_center[0],
        float(first_hit[1]) - accepted_center[1],
    ) <= region_radius_m + 0.30


def _region_camera_coverage(
    known_map,
    viewpoint: tuple[float, float],
    heading: float,
    center: tuple[float, float],
    region_radius_m: float,
    *,
    min_range_m: float,
    max_range_m: float,
    max_heading_error: float,
) -> float:
    """Fraction of a sampled work area inside the usable camera cone.

    The range is to the sampled work surface, not merely to the region's
    centre. A first mapped surface within the work area is acceptable; an
    intervening mapped wall outside it is not. This is a planning estimate,
    followed by a stopped live-camera detection rather than a claim that an
    object is already visible.
    """
    samples = [center]
    for scale in (0.50, 0.90):
        for angle in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
            samples.append((
                center[0] + region_radius_m * scale * math.cos(float(angle)),
                center[1] + region_radius_m * scale * math.sin(float(angle)),
            ))
    count = 0
    for point in samples:
        distance = math.hypot(point[0] - viewpoint[0], point[1] - viewpoint[1])
        if not min_range_m <= distance <= max_range_m:
            continue
        bearing = math.atan2(point[1] - viewpoint[1], point[0] - viewpoint[0])
        if abs(_angle_error(bearing, heading)) > max_heading_error:
            continue
        if _target_region_visible(
            known_map, viewpoint, point, region_radius_m,
            hit_region_center=center,
        ):
            count += 1
    return count / len(samples)


def _final_path_heading(
    start: tuple[float, float], path: list[tuple[float, float]]
) -> float | None:
    points = [start] + [(float(point[0]), float(point[1])) for point in path]
    for previous, current in zip(reversed(points[:-1]), reversed(points[1:])):
        dx, dy = current[0] - previous[0], current[1] - previous[1]
        if math.hypot(dx, dy) >= 0.08:
            return math.atan2(dy, dx)
    return None


def _candidate_points(
    known_map,
    center: tuple[float, float],
    region_radius_m: float,
    standoff_m: float,
) -> list[tuple[float, float]]:
    ring_radius = region_radius_m + standoff_m
    result: list[tuple[float, float]] = []
    for angle in np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False):
        raw = (
            center[0] + ring_radius * math.cos(float(angle)),
            center[1] + ring_radius * math.sin(float(angle)),
        )
        cell = known_map._nearest_traversable(
            tuple(known_map.world_to_grid((raw,))[0]), maximum_radius=8
        )
        if cell is None:
            continue
        world_raw = known_map.grid_to_world((cell,))[0]
        world = (float(world_raw[0]), float(world_raw[1]))
        if all(math.hypot(world[0] - old[0], world[1] - old[1]) >= 0.28 for old in result):
            result.append(world)
    return result


def _first_visibility_entry(
    known_map,
    start: tuple[float, float],
    path: list[tuple[float, float]],
    center: tuple[float, float],
    region_radius_m: float,
    standoff_m: float,
    max_heading_error: float,
    min_range_m: float,
    max_range_m: float,
    minimum_coverage: float,
) -> tuple[tuple[float, float], list[tuple[float, float]]] | None:
    """Find the first safe place on a route whose forward view covers the region."""
    previous = start
    traversed: list[tuple[float, float]] = []
    for end_raw in path:
        end = (float(end_raw[0]), float(end_raw[1]))
        segment = math.hypot(end[0] - previous[0], end[1] - previous[1])
        if segment < 0.08:
            previous = end
            continue
        heading = math.atan2(end[1] - previous[1], end[0] - previous[0])
        for step in range(1, int(math.ceil(segment / 0.15)) + 1):
            fraction = min(1.0, step * 0.15 / segment)
            point = (
                previous[0] + (end[0] - previous[0]) * fraction,
                previous[1] + (end[1] - previous[1]) * fraction,
            )
            if math.hypot(point[0] - start[0], point[1] - start[1]) < 0.45:
                continue
            if math.hypot(point[0] - center[0], point[1] - center[1]) < region_radius_m + MINIMUM_STANDOFF_M:
                continue
            coverage = _region_camera_coverage(
                known_map, point, heading, center, region_radius_m,
                min_range_m=min_range_m, max_range_m=max_range_m,
                max_heading_error=max_heading_error,
            )
            if coverage < minimum_coverage:
                continue
            return point, traversed + [point]
        traversed.append(end)
        previous = end
    return None


def _visibility_band_candidates(
    known_map,
    start: tuple[float, float],
    center: tuple[float, float],
    region_radius_m: float,
    band: dict[str, Any],
    min_range_m: float,
    max_range_m: float,
    minimum_coverage: float,
) -> list[dict[str, Any]]:
    fov_deg = float(band["camera_horizontal_fov_deg"])
    max_heading_error = math.radians(max(12.0, fov_deg / 2.0 - float(band["fov_margin_deg"])))
    candidates = []
    for point in _candidate_points(
        known_map, center, region_radius_m, float(band["standoff_m"])
    ):
        if math.hypot(point[0] - center[0], point[1] - center[1]) < region_radius_m + MINIMUM_STANDOFF_M:
            continue
        path = work_area_path(known_map, start, point, center, region_radius_m + MINIMUM_STANDOFF_M)
        if not path:
            continue
        entry = _first_visibility_entry(
            known_map,
            start,
            path,
            center,
            region_radius_m,
            float(band["standoff_m"]),
            max_heading_error,
            min_range_m,
            max_range_m,
            minimum_coverage,
        )
        if entry is not None:
            point, path = entry
        target_heading = math.atan2(center[1] - point[1], center[0] - point[0])
        final_heading = _final_path_heading(start, path)
        heading_error = (
            0.0 if final_heading is None else abs(_angle_error(target_heading, final_heading))
        )
        coverage = _region_camera_coverage(
            known_map, point, target_heading, center, region_radius_m,
            min_range_m=min_range_m, max_range_m=max_range_m,
            max_heading_error=max_heading_error,
        )
        if coverage < minimum_coverage:
            continue
        distance = _path_length(start, path)
        candidates.append({
            "point": point,
            "path": path,
            "path_length_m": distance,
            "target_yaw_rad": target_heading,
            "arrival_heading_error_deg": math.degrees(heading_error),
            "band": str(band["name"]),
            "confidence": float(band["confidence"]),
            "standoff_m": math.hypot(point[0] - center[0], point[1] - center[1]) - region_radius_m,
            "fov_deg": fov_deg,
            "fov_margin_deg": float(band["fov_margin_deg"]),
            "work_area_coverage": coverage,
        })
    return sorted(
        candidates,
        key=lambda item: (
            item["path_length_m"], item["arrival_heading_error_deg"]
        ),
    )


def _forward_approach_view(known_map, start, heading, center, region_radius,
                           recognition_min, recognition_max, minimum_coverage,
                           band, previous):
    """Find an in-range work-area view by driving straight, without a pivot.

    Used only after a live rotation-clearance replan. The route stays outside
    the object/work-area standoff and requires a fully map-clear body sweep;
    every actual lease still needs a fresh front scan and glass authorization.
    No pixel box is treated as an object coordinate.
    """
    if heading is None or not math.isfinite(float(heading)):
        return None
    heading = float(heading)
    margin = float(band['fov_margin_deg'])
    half_fov = math.radians(max(12., float(band['camera_horizontal_fov_deg'])/2-margin))
    for distance in np.arange(.4, 2.01, .1):
        point = (start[0]+float(distance)*math.cos(heading),
                 start[1]+float(distance)*math.sin(heading))
        if math.hypot(point[0]-center[0],point[1]-center[1]) < region_radius+MINIMUM_STANDOFF_M:
            continue
        if any(math.hypot(point[0]-old[0],point[1]-old[1]) < .45 for old in previous):
            continue
        coverage = _region_camera_coverage(
            known_map, point, heading, center, region_radius,
            min_range_m=recognition_min, max_range_m=recognition_max,
            max_heading_error=half_fov)
        if coverage < minimum_coverage:
            continue
        path = work_area_path(known_map,start,point,center,
                              region_radius+MINIMUM_STANDOFF_M,
                              exact_forward_standoff=True)
        if (len(path) != 2 or math.hypot(path[0][0]-start[0],path[0][1]-start[1]) > .03
                or math.hypot(path[1][0]-point[0],path[1][1]-point[1]) > .03):
            continue  # A detour would require the very pivot that was denied.
        return dict(point=point,path=path,path_length_m=float(distance),
                    target_yaw_rad=heading,arrival_heading_error_deg=0.,
                    band=band['name'],confidence=band['confidence'],
                    standoff_m=math.hypot(point[0]-center[0],point[1]-center[1])-region_radius,
                    fov_deg=band['camera_horizontal_fov_deg'],fov_margin_deg=margin,
                    work_area_coverage=coverage,headings=[(heading,coverage)],
                    union_coverage=coverage,forward_approach=True)
    return None


def _station_sweep_headings(
    known_map,
    point: tuple[float, float],
    center: tuple[float, float],
    region_radius_m: float,
    *,
    min_range_m: float,
    max_range_m: float,
    fov_deg: float,
    fov_margin_deg: float,
    maximum_headings: int,
) -> tuple[list[tuple[float, float]], float]:
    """Cover visible work-area samples with a short forward-facing yaw sweep."""
    center_yaw = math.atan2(center[1] - point[1], center[0] - point[0])
    half_fov = math.radians(max(12.0, fov_deg / 2.0 - fov_margin_deg))
    samples = [center]
    for scale in (0.50, 0.90):
        for angle in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
            samples.append((
                center[0] + region_radius_m * scale * math.cos(float(angle)),
                center[1] + region_radius_m * scale * math.sin(float(angle)),
            ))
    visible_bearings = {}
    for index, sample in enumerate(samples):
        distance = math.hypot(sample[0] - point[0], sample[1] - point[1])
        bearing = math.atan2(sample[1] - point[1], sample[0] - point[0])
        # Never spend this station's search budget looking behind the work area.
        if not min_range_m <= distance <= max_range_m:
            continue
        if abs(_angle_error(bearing, center_yaw)) > math.radians(75.0):
            continue
        if _target_region_visible(
            known_map, point, sample, region_radius_m,
            hit_region_center=center,
        ):
            visible_bearings[index] = bearing
    if not visible_bearings:
        return [], 0.0

    candidate_yaws = [center_yaw] + list(visible_bearings.values())
    chosen: list[tuple[float, set[int]]] = []
    covered: set[int] = set()
    for _ in range(maximum_headings):
        ranked = []
        for yaw_index, yaw in enumerate(candidate_yaws):
            if any(abs(_angle_error(yaw, old_yaw)) < math.radians(16.0)
                   for old_yaw, _old_coverage in chosen):
                continue
            hits = {
                index for index, bearing in visible_bearings.items()
                if abs(_angle_error(bearing, yaw)) <= half_fov
            }
            new_hits = hits - covered
            if not new_hits:
                continue
            ranked.append((
                len(new_hits), len(hits),
                -abs(_angle_error(yaw, center_yaw)), -yaw_index, yaw, hits,
            ))
        if not ranked:
            break
        _gain, _total, _center_bias, _index, yaw, hits = max(ranked)
        chosen.append((yaw, hits))
        covered.update(hits)
        if len(covered) == len(visible_bearings):
            break
    # Observe the main work area first, then sweep only as far as needed.
    chosen.sort(key=lambda item: abs(_angle_error(item[0], center_yaw)))
    return [(yaw, len(hits) / len(samples)) for yaw, hits in chosen], len(covered) / len(samples)


def _choose_spread(
    candidates: list[dict[str, Any]], maximum: int, previous: list[tuple[float, float]],
    minimum_spacing_m: float = 0.45,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in candidates:
        point = item["point"]
        if all(
            math.hypot(point[0] - old[0], point[1] - old[1]) >= minimum_spacing_m
            for old in previous + [value["point"] for value in selected]
        ):
            selected.append(item)
            if len(selected) >= maximum:
                break
    return selected


def _transit_observations(
    start: tuple[float, float],
    path: list[tuple[float, float]],
    *,
    spacing_m: float,
    maximum: int,
) -> list[dict[str, Any]]:
    """Stop on the approach before an unmapped object can be passed unseen."""
    if maximum < 1 or not path:
        return []
    points = [start] + [(float(x), float(y)) for x, y in path]
    total = _path_length(start, path)
    # Keep the final approach for the high confidence observation itself.
    distances = [
        spacing_m * index
        for index in range(1, maximum + 1)
        if spacing_m * index < total - 0.45
    ]
    result = []
    for wanted in distances:
        covered = 0.0
        for left, right in zip(points, points[1:]):
            length = math.hypot(right[0] - left[0], right[1] - left[1])
            if length < 1e-6:
                continue
            if covered + length >= wanted:
                fraction = (wanted - covered) / length
                xy = (
                    left[0] + (right[0] - left[0]) * fraction,
                    left[1] + (right[1] - left[1]) * fraction,
                )
                result.append({
                    "view_xy_m": [round(xy[0], 4), round(xy[1], 4)],
                    "path_xy_m": [],
                    "path_length_m": round(wanted, 3),
                    "capture_after_stop": True,
                    "visibility_zone": {
                        "band": "opportunistic_transit",
                        "confidence": 0.50,
                        "reason": "沿途停车检查实时视野中的目标物",
                        "stop_rule": "periodic_stopped_capture_on_approach",
                    },
                })
                break
            covered += length
    return result


def build_visibility_route(
    known_map,
    start_xy: tuple[float, float],
    route_request: dict[str, Any],
) -> dict[str, Any]:
    global LATEST_KNOWN_MAP
    LATEST_KNOWN_MAP = known_map
    raw = route_request.get("center_xy_m")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise ValueError("visibility planning requires center_xy_m")
    center = (float(raw[0]), float(raw[1]))
    region_radius = max(0.40, min(2.50, float(route_request.get("radius_m") or 1.2)))
    maximum = max(2, min(12, int(route_request.get("maximum_views") or 8)))
    # Carry a ONE-TIME verified-target photo reserve across station replans.
    # Without this, restoring the original request's 8-photo ceiling would
    # end the mission as soon as recovery asks for another physical station.
    previous = LATEST_ROUTE or {}
    previous_region = previous.get("target_region") or {}
    inherited_target_reserve = bool(previous.get("locked_target_reserve_granted")
        and previous.get("mode") == "semantic_visibility_regions"
        and previous_region.get("center_xy_m") == list(center)
        and previous_region.get("radius_m") == region_radius)
    if inherited_target_reserve:
        maximum += int(previous.get("locked_target_extra_views") or 6)
    station_spacing = .18 if inherited_target_reserve else .45
    remaining_budget = maximum - OBSERVATIONS_COMPLETED
    if remaining_budget <= 0:
        raise RuntimeError("观察次数已达到本次任务上限")
    fov = max(45.0, min(120.0, float(route_request.get("camera_horizontal_fov_deg") or 82.0)))
    recognition_min = max(0.25, min(1.0, float(route_request.get("recognition_min_distance_m") or 0.45)))
    recognition_max = max(0.8, min(3.0, float(route_request.get("recognition_max_distance_m") or 1.80)))
    if recognition_max <= recognition_min + 0.30:
        raise ValueError("recognition distance interval is too narrow")
    minimum_coverage = max(0.10, min(0.75, float(route_request.get("minimum_work_area_coverage") or 0.25)))
    configured_bands = route_request.get("visibility_bands")
    if not isinstance(configured_bands, list) or not configured_bands:
        configured_bands = [
            {"name": "high", "confidence": 0.85, "standoff_m": 0.90, "fov_margin_deg": 10.0},
            {"name": "medium", "confidence": 0.70, "standoff_m": 1.10, "fov_margin_deg": 7.0},
            {"name": "low", "confidence": 0.55, "standoff_m": 1.30, "fov_margin_deg": 4.0},
        ]
    bands = []
    for row in configured_bands:
        if not isinstance(row, dict):
            continue
        bands.append({
            "name": str(row.get("name") or "low"),
            "confidence": max(0.0, min(1.0, float(row.get("confidence") or 0.0))),
            # Old decision JSONs with 0.55 m must not bypass the new policy.
            "standoff_m": max(0.90, min(2.00, float(row.get("standoff_m") or 0.9))),
            "fov_margin_deg": max(0.0, min(20.0, float(row.get("fov_margin_deg") or 6.0))),
            "camera_horizontal_fov_deg": fov,
        })
    bands.sort(key=lambda row: row["confidence"], reverse=True)

    forward_candidate = _forward_approach_view(
        known_map,start_xy,route_request.get('_phase1_forward_view_yaw_rad'),
        center,region_radius,recognition_min,recognition_max,minimum_coverage,
        bands[0],VISITED_POINTS) if bands else None
    views: list[dict[str, Any]] = []
    inspect_current = (forward_candidate is None
                       and bool(route_request.get("inspect_current_view_first", True)))
    if inspect_current and remaining_budget >= 2 and all(
        math.hypot(start_xy[0] - old[0], start_xy[1] - old[1]) >= 0.30
        for old in VISITED_POINTS
    ):
        views.append({
            "view_xy_m": [round(float(start_xy[0]), 4), round(float(start_xy[1]), 4)],
            "path_xy_m": [[round(float(start_xy[0]), 4), round(float(start_xy[1]), 4)]],
            "path_length_m": 0.0,
            "capture_after_stop": True,
            "visibility_zone": {
                "band": "opportunistic_current_view",
                "confidence": 0.45,
                "reason": "定位后先检查当前双摄，避免漏过已经进入视野的目标",
                "stop_rule": "capture_before_route_motion",
            },
        })

    # Candidate generation may inspect the map globally, but this route
    # publishes only one physical station. A later build chooses another only
    # after all useful headings at this station have been observed.
    station_candidates: list[dict[str, Any]] = []
    rejected_counts: dict[str, int] = {}
    for band in bands:
        candidates = _visibility_band_candidates(
            known_map, start_xy, center, region_radius, band,
            recognition_min, recognition_max, minimum_coverage,
        )
        rejected_counts[band["name"]] = max(0, 32 - len(candidates))
        if not VISITED_POINTS:
            start_yaw = math.atan2(center[1] - start_xy[1], center[0] - start_xy[0])
            start_coverage = _region_camera_coverage(
                known_map, start_xy, start_yaw, center, region_radius,
                min_range_m=recognition_min, max_range_m=recognition_max,
                max_heading_error=math.radians(max(
                    12.0, fov / 2.0 - band["fov_margin_deg"]
                )),
            )
            if (start_coverage >= minimum_coverage
                    and math.hypot(start_xy[0] - center[0], start_xy[1] - center[1]) >= region_radius + MINIMUM_STANDOFF_M):
                candidates.append({
                    "point": start_xy,
                    "path": [start_xy],
                    "path_length_m": 0.0,
                    "target_yaw_rad": start_yaw,
                    "arrival_heading_error_deg": 0.0,
                    "band": band["name"],
                    "confidence": band["confidence"],
                    "standoff_m": 0.0,
                    "fov_deg": fov,
                    "fov_margin_deg": band["fov_margin_deg"],
                    "work_area_coverage": start_coverage,
                })
        for item in candidates:
            point = item["point"]
            if any(math.hypot(point[0] - old[0], point[1] - old[1]) < station_spacing
                   for old in VISITED_POINTS):
                continue
            headings, union_coverage = _station_sweep_headings(
                known_map, point, center, region_radius,
                min_range_m=recognition_min, max_range_m=recognition_max,
                fov_deg=fov, fov_margin_deg=item["fov_margin_deg"],
                maximum_headings=min(4, remaining_budget - len(views)),
            )
            if not headings or union_coverage < minimum_coverage:
                continue
            station_candidates.append({
                **item, "headings": headings,
                "union_coverage": union_coverage,
            })
    if forward_candidate is not None:
        station_candidates.append(forward_candidate)
    if not station_candidates:
        raise NoReachableVisibilityError("地图中没有可达且朝向目标区域的相机视野")
    broad_coverage = max(0.40, min(0.60, minimum_coverage * 2.0))
    best_confidence = max(item["confidence"] for item in station_candidates)
    eligible = [
        item for item in station_candidates
        if item["confidence"] == best_confidence
    ]
    broad = [item for item in eligible if item["union_coverage"] >= broad_coverage]
    eligible = broad or eligible
    earliest_length = min(item["path_length_m"] for item in eligible)
    near_earliest = [
        item for item in eligible
        if item["path_length_m"] <= earliest_length + 0.45
    ]
    selected = forward_candidate or max(near_earliest, key=lambda item: (
        item["union_coverage"], -item["path_length_m"],
    ))
    arrival_yaw = _final_path_heading(start_xy, selected["path"])
    if arrival_yaw is not None and len(selected["headings"]) > 1:
        center_yaw = math.atan2(
            center[1] - selected["point"][1],
            center[0] - selected["point"][0],
        )
        ordered = sorted(
            selected["headings"], key=lambda row: _angle_error(row[0], center_yaw),
        )

        def turn_cost(sequence):
            previous = arrival_yaw
            cost = 0.0
            for yaw, _coverage in sequence:
                cost += abs(_angle_error(yaw, previous))
                previous = yaw
            return cost

        selected["headings"] = min((ordered, list(reversed(ordered))), key=turn_cost)
    point = selected["point"]
    next_station_available = any(
        math.hypot(item["point"][0] - point[0], item["point"][1] - point[1]) >= station_spacing
        for item in station_candidates if item is not selected
    )
    if bool(route_request.get("inspect_transit_views", False)):
        views.extend(_transit_observations(
            start_xy, selected["path"],
            spacing_m=max(0.55, min(1.0, float(route_request.get("transit_observation_spacing_m") or 0.75))),
            maximum=max(0, remaining_budget - len(views) - len(selected["headings"])),
        ))
    for sweep_index, (yaw, coverage) in enumerate(selected["headings"]):
        views.append({
            "view_xy_m": [round(point[0], 4), round(point[1], 4)],
            "path_xy_m": [
                [round(float(x), 4), round(float(y), 4)]
                for x, y in selected["path"]
            ] if sweep_index == 0 else [],
            "path_length_m": round(float(selected["path_length_m"]), 3)
            if sweep_index == 0 else 0.0,
            "capture_after_stop": True,
            "target_yaw_rad": round(float(yaw), 5),
            "visibility_zone": {
                "band": selected["band"],
                "confidence": round(float(selected["confidence"]), 3),
                "target_region_center_xy_m": [center[0], center[1]],
                "target_region_radius_m": region_radius,
                "camera_horizontal_fov_deg": selected["fov_deg"],
                "fov_margin_deg": selected["fov_margin_deg"],
                "arrival_heading_error_deg": round(
                    float(selected["arrival_heading_error_deg"]), 2
                ),
                "standoff_from_region_m": selected["standoff_m"],
                "recognition_min_distance_m": recognition_min,
                "recognition_max_distance_m": recognition_max,
                "minimum_work_area_coverage": minimum_coverage,
                "estimated_work_area_coverage": round(float(coverage), 3),
                "station_union_coverage": round(float(selected["union_coverage"]), 3),
                "sweep_index": sweep_index,
                "sweep_count": len(selected["headings"]),
                "stop_rule": "forward_approach_then_capture" if selected.get('forward_approach')
                             else "turn_at_station_then_capture",
            },
        })
    return {
        "schema": "semantic_visibility_search_route.v1",
        "localization_required_before_execution": True,
        "start_xy_m": [round(float(start_xy[0]), 4), round(float(start_xy[1]), 4)],
        "mode": "semantic_visibility_regions",
        "planning_policy": POLICY,
        "target_region": {
            "center_xy_m": [center[0], center[1]],
            "radius_m": region_radius,
            "coordinates_are_search_prior_only": True,
            "recognition_distance_m": [recognition_min, recognition_max],
            "minimum_work_area_coverage": minimum_coverage,
            "camera_horizontal_fov_deg": fov,
            "fov_margin_deg": 10.0,
            "minimum_standoff_from_region_m": MINIMUM_STANDOFF_M,
        },
        "views": views,
        "physical_station_count": 1,
        "next_station_available": next_station_available,
        "maximum_total_views": maximum,
        "requested_maximum_views": max(2, min(12, int(route_request.get("maximum_views") or 8))),
        "locked_target_extra_views": int(previous.get("locked_target_extra_views") or 6) if inherited_target_reserve else 0,
        "locked_target_reserve_granted": inherited_target_reserve,
        "minimum_station_spacing_m": station_spacing,
        "confidence_order": [row["name"] for row in bands],
        "rejected_candidate_counts": rejected_counts,
        "trajectory_used": bool(selected.get('forward_approach')),
    }


def build_approach_route(known_map, start_xy, route_request):
    """Navigate to a safe region perimeter WITHOUT requiring camera visibility.

    This is a transit checkpoint, not an object coordinate or an observation
    station. All existing map/glass/turn reserve and object-region masks still
    apply. Only after arrival is camera heading/coverage planned.
    """
    global LATEST_KNOWN_MAP
    LATEST_KNOWN_MAP = known_map
    raw = route_request.get("center_xy_m")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise ValueError("approach planning requires center_xy_m")
    center = tuple(float(v) for v in raw[:2])
    start = tuple(float(v) for v in start_xy)
    radius = max(.40, min(2.50, float(route_request.get("radius_m") or 1.2)))
    if not all(math.isfinite(v) for v in (*center, *start, radius)):
        raise ValueError("non-finite approach coordinates")
    candidates = []
    for standoff in (.95, 1.15, 1.35):
        for i in range(48):
            angle = 2 * math.pi * i / 48
            point = (center[0] + (radius + standoff) * math.cos(angle),
                     center[1] + (radius + standoff) * math.sin(angle))
            if math.hypot(point[0]-start[0], point[1]-start[1]) < .35:
                continue
            if any(math.hypot(point[0]-old[0], point[1]-old[1]) < .30 for old in VISITED_POINTS):
                continue
            path = work_area_path(known_map, start, point, center,
                                  radius + MINIMUM_STANDOFF_M)
            if not path:
                continue
            # The executor navigates to grid-snapped path endpoints. Publish
            # that SAME endpoint, rather than a ring point it cannot reach.
            goal = tuple(float(v) for v in path[-1])
            if any(math.hypot(goal[0]-old[0], goal[1]-old[1]) < .30 for old in VISITED_POINTS):
                continue
            if math.hypot(goal[0]-start[0], goal[1]-start[1]) < .35:
                continue
            if not radius + MINIMUM_STANDOFF_M <= math.hypot(goal[0]-center[0], goal[1]-center[1]) <= radius + 1.45:
                continue
            candidates.append((_path_length(start, path), goal, path))
    if not candidates:
        raise NoReachableVisibilityError("目标区域附近暂无地图可通行的接近路径；需要换视角恢复路线，非相机视野失败")
    length, goal, path = min(candidates, key=lambda row: row[0])
    return {
        "schema": "semantic_visibility_search_route.v1", "mode": "semantic_visibility_regions",
        "planning_policy": POLICY, "navigation_stage": "approach",
        "localization_required_before_execution": True, "start_xy_m": list(start),
        "target_region": {"center_xy_m": list(center), "radius_m": radius,
                          "coordinates_are_search_prior_only": True,
                          "minimum_standoff_from_region_m": MINIMUM_STANDOFF_M,
                          "recognition_distance_m": [max(.25,min(1.,float(route_request.get('recognition_min_distance_m') or .45))),
                                                     max(.8,min(3.,float(route_request.get('recognition_max_distance_m') or 1.8)))],
                          "minimum_work_area_coverage": max(.1,min(.75,float(route_request.get('minimum_work_area_coverage') or .25))),
                          "camera_horizontal_fov_deg": max(45.,min(120.,float(route_request.get('camera_horizontal_fov_deg') or 82.))),
                          "fov_margin_deg": 10.},
        "views": [{"view_xy_m": list(goal), "path_xy_m": [list(p) for p in path],
                   "path_length_m": round(length, 3), "capture_after_stop": True,
                   "visibility_zone": {"band": "transit_checkpoint", "confidence": .7,
                                       "stop_rule": "approach_region_then_plan_observation"}}],
        "physical_station_count": 0, "next_station_available": True,
        "maximum_total_views": max(2, min(12, int(route_request.get("maximum_views") or 8))),
    }


def build_staged_route(known_map, start_xy, route_request):
    center = route_request.get("center_xy_m") or []
    if len(center) < 2:
        raise ValueError("semantic navigation requires center_xy_m")
    radius = max(.4, min(2.5, float(route_request.get("radius_m") or 1.2)))
    distance = math.hypot(float(start_xy[0])-float(center[0]), float(start_xy[1])-float(center[1]))
    if distance > radius + 1.45:
        return build_approach_route(known_map, start_xy, route_request)
    try:
        return build_visibility_route(known_map, start_xy, route_request)
    except NoReachableVisibilityError:
        # Near the region but no useful camera station: try another genuinely
        # reachable perimeter position. Never pretend an occupied cell is free.
        return build_approach_route(known_map, start_xy, route_request)


def install_visibility_planner() -> None:
    """Replace only semantic phase-1 routing in the process-local module."""
    import raspberry_pi.search_route_planner as route_module

    original = route_module.build_search_route

    def guarded_build(known_map, start_xy, route_request):
        global LATEST_ROUTE, ACTIVE_VIEW
        region_builder = globals().get("REGION_ROUTE_BUILDER")
        if region_builder is not None:
            # Process-local destination-arrival skill. Keep the OUTER
            # obstacle/map-context wrapper installed on mission.build_search_route.
            return region_builder(known_map, start_xy, route_request)
        if (
            str(route_request.get("mode") or "") == "semantic_search_region"
            and str(route_request.get("planning_policy") or "") == POLICY
        ):
            center = tuple(float(v) for v in route_request["center_xy_m"][:2])
            radius = max(0.40, min(2.50, float(route_request.get("radius_m") or 1.2))) + MINIMUM_STANDOFF_M
            if not hasattr(known_map, "_phase1_original_path"):
                known_map._phase1_original_path = known_map.path

            def guarded_path(start, goal):
                return work_area_path(known_map, start, goal, center, radius,
                                      original=known_map._phase1_original_path)

            known_map.path = guarded_path
            LATEST_ROUTE = build_staged_route(known_map, start_xy, route_request)
            ACTIVE_VIEW = LATEST_ROUTE["views"][0]
            return LATEST_ROUTE
        return original(known_map, start_xy, route_request)

    route_module.build_search_route = guarded_build
