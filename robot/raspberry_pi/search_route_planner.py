from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from raspberry_pi.known_map_navigation import KnownOccupancyMap


REGION_RANGES = {
    "top_left": (0.00, 0.55, 0.45, 1.00),
    "top_right": (0.45, 1.00, 0.45, 1.00),
    "bottom_left": (0.00, 0.55, 0.00, 0.55),
    "bottom_right": (0.45, 1.00, 0.00, 0.55),
    "center": (0.25, 0.75, 0.25, 0.75),
}


def _known_bounds(known_map: KnownOccupancyMap) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(known_map.free | known_map.occupied)
    if not len(rows):
        return 0, known_map.width - 1, 0, known_map.height - 1
    return int(columns.min()), int(columns.max()), int(rows.min()), int(rows.max())


def _sample_spread(points: list[tuple[float, float]], maximum: int) -> list[tuple[float, float]]:
    if len(points) <= maximum:
        return points
    selected = [points[len(points) // 2]]
    remaining = list(points)
    remaining.remove(selected[0])
    while remaining and len(selected) < maximum:
        item = max(
            remaining,
            key=lambda point: min(math.hypot(point[0] - old[0], point[1] - old[1]) for old in selected),
        )
        selected.append(item)
        remaining.remove(item)
    return selected


def landmark_viewpoints(
    known_map: KnownOccupancyMap,
    anchor_xy: tuple[float, float],
    distance_m: float,
    maximum: int,
) -> list[tuple[float, float]]:
    def target_visible(world: tuple[float, float]) -> bool:
        start_cell = known_map.world_to_grid((world,))[0]
        target_cell = known_map.world_to_grid((anchor_xy,))[0]
        # Follow the ray from the camera toward the semantic target.  The
        # first occupied cell is the visible surface; cells behind that
        # surface are irrelevant.  Reject rays that encounter an unrelated
        # obstacle far away from the target coordinate.
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
            # A semantic point can be slightly outside the occupied pixels.
            return known_map.line_is_clear(world, anchor_xy, mask=~known_map.occupied)
        return math.hypot(first_hit[0] - anchor_xy[0], first_hit[1] - anchor_xy[1]) <= max(
            0.45, distance_m * 0.85
        )

    candidates: list[tuple[float, float]] = []
    for radius in (distance_m, distance_m + 0.25, max(0.45, distance_m - 0.20)):
        for angle in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
            point = (anchor_xy[0] + radius * math.cos(angle), anchor_xy[1] + radius * math.sin(angle))
            cell = known_map._nearest_traversable(tuple(known_map.world_to_grid((point,))[0]), maximum_radius=10)
            if cell is None:
                continue
            world = tuple(float(value) for value in known_map.grid_to_world((cell,))[0])
            if not target_visible(world):
                continue
            if all(math.hypot(world[0] - old[0], world[1] - old[1]) >= 0.35 for old in candidates):
                candidates.append(world)
    return _sample_spread(candidates, maximum)


def region_viewpoints(
    known_map: KnownOccupancyMap,
    region: str,
    maximum: int,
) -> list[tuple[float, float]]:
    if region not in REGION_RANGES:
        raise ValueError(f"unsupported region hint: {region}")
    left, right, bottom, top = REGION_RANGES[region]
    min_c, max_c, min_r, max_r = _known_bounds(known_map)
    c0 = int(round(min_c + (max_c - min_c) * left))
    c1 = int(round(min_c + (max_c - min_c) * right))
    r0 = int(round(min_r + (max_r - min_r) * bottom))
    r1 = int(round(min_r + (max_r - min_r) * top))
    spacing = max(4, int(round(0.55 / known_map.resolution_m)))
    cells = []
    for row in range(r0, r1 + 1, spacing):
        for column in range(c0, c1 + 1, spacing):
            nearest = known_map._nearest_traversable((column, row), maximum_radius=max(4, spacing // 2))
            if (
                nearest is not None
                and c0 <= nearest[0] <= c1
                and r0 <= nearest[1] <= r1
                and nearest not in cells
            ):
                cells.append(nearest)
    points = [tuple(float(value) for value in known_map.grid_to_world((cell,))[0]) for cell in cells]
    return _sample_spread(points, maximum)


def build_search_route(
    known_map: KnownOccupancyMap,
    start_xy: tuple[float, float],
    route_request: dict[str, Any],
) -> dict[str, Any]:
    mode = str(route_request.get("mode") or "")
    maximum = max(1, min(16, int(route_request.get("maximum_views") or 6)))
    if mode == "around_landmark":
        raw = route_request.get("anchor_xy_m")
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            raise ValueError("around_landmark route requires anchor_xy_m")
        candidates = landmark_viewpoints(
            known_map,
            (float(raw[0]), float(raw[1])),
            float(route_request.get("view_distance_m") or 0.75),
            maximum,
        )
    elif mode == "coarse_region":
        candidates = region_viewpoints(known_map, str(route_request.get("region_hint") or ""), maximum)
    else:
        raise ValueError(f"unsupported route mode: {mode}")
    remaining = list(candidates)
    current = start_xy
    ordered = []
    rejected = []
    while remaining:
        scored = []
        for point in remaining:
            path = known_map.path(current, point)
            if path:
                length = sum(
                    math.hypot(b[0] - a[0], b[1] - a[1])
                    for a, b in zip([current] + path[:-1], path)
                )
                scored.append((length, point, path))
        if not scored:
            rejected.extend(remaining)
            break
        length, point, path = min(scored, key=lambda item: item[0])
        ordered.append(
            {
                "view_xy_m": [round(point[0], 4), round(point[1], 4)],
                "path_xy_m": [[round(x, 4), round(y, 4)] for x, y in path],
                "path_length_m": round(length, 3),
                "capture_after_stop": True,
            }
        )
        current = point
        remaining.remove(point)
    return {
        "schema": "known_map_search_route.v1",
        "localization_required_before_execution": True,
        "start_xy_m": [round(start_xy[0], 4), round(start_xy[1], 4)],
        "mode": mode,
        "views": ordered,
        "rejected_unreachable_views": [[round(x, 4), round(y, 4)] for x, y in rejected],
        "trajectory_used": False,
    }


def render_search_route(
    known_map: KnownOccupancyMap,
    route: dict[str, Any],
    output_path: str | Path,
    route_request: dict[str, Any] | None = None,
) -> None:
    """Render only the planned route, never the old mapping trajectory."""
    image = Image.fromarray(np.flipud(known_map.image), mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)

    def pixel(world: tuple[float, float] | list[float]) -> tuple[int, int]:
        column, row = known_map.world_to_grid((world,))[0]
        return int(column), int(known_map.height - 1 - row)

    start = pixel(route["start_xy_m"])
    radius = max(3, int(round(0.12 / known_map.resolution_m)))
    draw.ellipse((start[0] - radius, start[1] - radius, start[0] + radius, start[1] + radius), fill=(0, 180, 0))
    previous = tuple(route["start_xy_m"])
    for index, view in enumerate(route.get("views", []), 1):
        path_world = [previous] + [tuple(item) for item in view.get("path_xy_m", [])]
        path_pixels = [pixel(item) for item in path_world]
        if len(path_pixels) >= 2:
            draw.line(path_pixels, fill=(0, 90, 255), width=max(2, radius // 2))
        view_pixel = pixel(view["view_xy_m"])
        draw.ellipse(
            (
                view_pixel[0] - radius,
                view_pixel[1] - radius,
                view_pixel[0] + radius,
                view_pixel[1] + radius,
            ),
            fill=(0, 90, 255),
        )
        draw.text((view_pixel[0] + radius + 1, view_pixel[1] - radius), str(index), fill=(20, 20, 20))
        previous = tuple(view["view_xy_m"])
    if route_request and route_request.get("anchor_xy_m"):
        anchor = pixel(route_request["anchor_xy_m"])
        draw.rectangle(
            (anchor[0] - radius, anchor[1] - radius, anchor[0] + radius, anchor[1] + radius),
            outline=(255, 0, 0),
            width=max(2, radius // 2),
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan a safe A* search route on a saved map; no motion commands.")
    parser.add_argument("--map", required=True)
    parser.add_argument("--map-yaml", required=True)
    parser.add_argument("--decision", required=True, help="Search decision JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--render", default=None, help="Optional PNG of this route without map trajectory")
    parser.add_argument("--start-x", type=float, default=None)
    parser.add_argument("--start-y", type=float, default=None)
    parser.add_argument("--clearance", type=float, default=0.30)
    args = parser.parse_args()
    decision = json.loads(Path(args.decision).read_text(encoding="utf-8"))
    if decision.get("status") != "executable":
        raise RuntimeError("Only an executable decision may be routed")
    if args.start_x is None or args.start_y is None:
        raise RuntimeError(
            "--start-x and --start-y must come from current localization; "
            "saved build trajectories are deliberately not used"
        )
    start = (args.start_x, args.start_y)
    known_map = KnownOccupancyMap.load(args.map, args.map_yaml, clearance_m=args.clearance)
    result = build_search_route(known_map, start, decision["route_request"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.render:
        render_search_route(known_map, result, args.render, decision.get("route_request"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
