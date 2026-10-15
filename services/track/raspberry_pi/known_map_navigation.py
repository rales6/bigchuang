"""Read-only localization and path planning against a saved occupancy map.

This module deliberately has no dependency on :mod:`autonomous_mapping` and
never writes to the input PGM/YAML pair.  It is used by the camera inspection
mission after mapping has already finished.
"""

from collections import deque
from dataclasses import dataclass
import heapq
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image

from raspberry_pi.mapping.geometry import Pose2D, normalize_angle
from raspberry_pi.mapping.scan_matcher import IcpScanMatcher


@dataclass(frozen=True)
class LocalizationResult:
    pose: Pose2D
    rmse_m: float
    inlier_ratio: float
    score: float
    ambiguity_margin: float = math.inf

    @property
    def reliable(self):
        return self.rmse_m <= 0.18 and self.inlier_ratio >= 0.42


@dataclass(frozen=True)
class InspectionTarget:
    target_id: str
    kind: str
    target_xy_m: tuple
    view_xy_m: tuple
    yaw_rad: float
    component_cells: int


def _parse_map_yaml(path):
    text = Path(path).read_text(encoding="utf-8")
    resolution_match = re.search(
        r"^\s*resolution\s*:\s*([-+0-9.eE]+)", text, re.MULTILINE
    )
    origin_match = re.search(
        r"^\s*origin\s*:\s*\[\s*([-+0-9.eE]+)\s*,\s*"
        r"([-+0-9.eE]+)",
        text,
        re.MULTILINE,
    )
    if not resolution_match or not origin_match:
        raise ValueError("地图 YAML 缺少 resolution 或 origin")
    return (
        float(resolution_match.group(1)),
        float(origin_match.group(1)),
        float(origin_match.group(2)),
    )


def _dilate(mask, radius):
    radius = max(0, int(radius))
    result = np.asarray(mask, dtype=np.bool_).copy()
    if radius == 0:
        return result
    source = result.copy()
    height, width = source.shape
    for dy in range(-radius, radius + 1):
        span = int(math.floor(math.sqrt(max(0, radius * radius - dy * dy))))
        sy0 = max(0, -dy)
        sy1 = min(height, height - dy)
        for dx in range(-span, span + 1):
            sx0 = max(0, -dx)
            sx1 = min(width, width - dx)
            result[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] |= source[
                sy0:sy1, sx0:sx1
            ]
    return result


def _distance_to_mask(mask):
    """Approximate Euclidean distance in cells using an 8-neighbour Dijkstra."""
    mask = np.asarray(mask, dtype=np.bool_)
    height, width = mask.shape
    distance = np.full(mask.shape, np.inf, dtype=np.float64)
    heap = []
    for row, column in np.argwhere(mask):
        distance[row, column] = 0.0
        heap.append((0.0, int(row), int(column)))
    heapq.heapify(heap)
    neighbours = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
    )
    while heap:
        current, row, column = heapq.heappop(heap)
        if current != distance[row, column]:
            continue
        for dy, dx, cost in neighbours:
            nr, nc = row + dy, column + dx
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            candidate = current + cost
            if candidate < distance[nr, nc]:
                distance[nr, nc] = candidate
                heapq.heappush(heap, (candidate, nr, nc))
    return distance


class KnownOccupancyMap:
    """A ROS-style PGM/YAML map held in the map's original world frame."""

    def __init__(
        self,
        image,
        resolution_m,
        origin_x_m,
        origin_y_m,
        clearance_m=0.30,
    ):
        # ROS PGM rows run from high world y to low world y.  Internally row
        # increases with world y to match OccupancyGrid.
        self.image = np.flipud(np.asarray(image, dtype=np.uint8))
        self.height, self.width = self.image.shape
        self.resolution_m = float(resolution_m)
        self.origin_x_m = float(origin_x_m)
        self.origin_y_m = float(origin_y_m)
        self.occupied = self.image <= 65
        self.free = self.image >= 250
        self.unknown = ~(self.occupied | self.free)
        self.clearance_cells = max(
            1, int(math.ceil(float(clearance_m) / self.resolution_m))
        )
        self.inflated_occupied = _dilate(
            self.occupied | self.unknown,
            self.clearance_cells,
        )
        # Runtime observations live on a separate volatile layer.  They are
        # used only by planning and are never fed into localization or written
        # back to the permanent PGM/YAML pair.
        self.temporary_occupied = np.zeros_like(self.occupied)
        self.temporary_obstacle_hits = np.zeros_like(
            self.occupied, dtype=np.uint8
        )
        self.temporary_obstacle_updates = 0
        self._refresh_traversable()
        self.obstacle_distance_cells = _distance_to_mask(self.occupied)

    @classmethod
    def load(cls, pgm_path, yaml_path=None, clearance_m=0.30):
        pgm_path = Path(pgm_path)
        yaml_path = (
            pgm_path.with_suffix(".yaml")
            if yaml_path is None
            else Path(yaml_path)
        )
        if not pgm_path.exists():
            raise FileNotFoundError("找不到已建地图：{}".format(pgm_path))
        if not yaml_path.exists():
            raise FileNotFoundError("找不到地图坐标文件：{}".format(yaml_path))
        resolution, origin_x, origin_y = _parse_map_yaml(yaml_path)
        with Image.open(pgm_path) as image:
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
        return cls(gray, resolution, origin_x, origin_y, clearance_m)

    def world_to_grid(self, xy):
        values = np.asarray(xy, dtype=np.float64)
        columns = np.floor(
            (values[..., 0] - self.origin_x_m) / self.resolution_m
        ).astype(int)
        rows = np.floor(
            (values[..., 1] - self.origin_y_m) / self.resolution_m
        ).astype(int)
        return np.stack((columns, rows), axis=-1)

    def grid_to_world(self, cells):
        values = np.asarray(cells, dtype=np.float64)
        x = self.origin_x_m + (values[..., 0] + 0.5) * self.resolution_m
        y = self.origin_y_m + (values[..., 1] + 0.5) * self.resolution_m
        return np.stack((x, y), axis=-1)

    def in_bounds(self, column, row):
        return 0 <= column < self.width and 0 <= row < self.height

    def occupied_points(self, maximum=3500):
        rows_columns = np.argwhere(self.occupied)
        if not len(rows_columns):
            return np.empty((0, 2), dtype=np.float64)
        if len(rows_columns) > maximum:
            indices = np.linspace(0, len(rows_columns) - 1, maximum, dtype=int)
            rows_columns = rows_columns[indices]
        return self.grid_to_world(rows_columns[:, ::-1])

    def _refresh_traversable(self):
        temporary_inflated = _dilate(
            self.temporary_occupied,
            self.clearance_cells,
        )
        self.traversable = (
            self.free
            & ~self.inflated_occupied
            & ~temporary_inflated
        )

    def mark_temporary_obstacles(
        self,
        world_points,
        radius_m=0.10,
        minimum_hits=1,
    ):
        """Add session-only obstacle evidence and rebuild the planning mask."""
        points = np.asarray(world_points, dtype=np.float64).reshape(-1, 2)
        if not len(points):
            return 0
        cells = self.world_to_grid(points)
        radius = max(0, int(math.ceil(radius_m / self.resolution_m)))
        minimum_hits = max(1, int(minimum_hits))
        before = int(np.count_nonzero(self.temporary_occupied))
        observation = np.zeros_like(self.temporary_occupied)
        # One scan contributes at most one vote per cell. Dilating this vote
        # by one cell tolerates ordinary range jitter between confirmations,
        # without allowing several adjacent beams in a single scan to satisfy
        # the multi-scan threshold by themselves.
        for column, row in np.unique(cells, axis=0):
            column, row = int(column), int(row)
            if not self.in_bounds(column, row):
                continue
            observation[
                max(0, row - 1):min(self.height, row + 2),
                max(0, column - 1):min(self.width, column + 2),
            ] = True
        hit_values = self.temporary_obstacle_hits[observation].astype(np.uint16)
        self.temporary_obstacle_hits[observation] = np.minimum(
            255, hit_values + 1
        ).astype(np.uint8)
        confirmed = observation & (
            self.temporary_obstacle_hits >= minimum_hits
        )
        for row, column in np.argwhere(confirmed):
            for dy in range(-radius, radius + 1):
                span = int(math.floor(math.sqrt(max(0, radius * radius - dy * dy))))
                target_row = row + dy
                if not 0 <= target_row < self.height:
                    continue
                left = max(0, column - span)
                right = min(self.width, column + span + 1)
                self.temporary_occupied[target_row, left:right] = True
        added = int(np.count_nonzero(self.temporary_occupied)) - before
        if added > 0:
            self.temporary_obstacle_updates += 1
            self._refresh_traversable()
        return added

    def clear_temporary_obstacles(self):
        """Clear only volatile obstacles; permanent map data is untouched."""
        self.temporary_occupied.fill(False)
        self.temporary_obstacle_hits.fill(0)
        self.temporary_obstacle_updates = 0
        self._refresh_traversable()

    def line_is_clear(self, start_xy, end_xy, mask=None):
        mask = self.traversable if mask is None else mask
        start = self.world_to_grid(np.asarray((start_xy,)))[0]
        end = self.world_to_grid(np.asarray((end_xy,)))[0]
        for column, row in _bresenham(tuple(start), tuple(end)):
            if not self.in_bounds(column, row) or not mask[row, column]:
                return False
        return True

    def path(self, start_xy, goal_xy):
        start = tuple(self.world_to_grid(np.asarray((start_xy,)))[0])
        goal = tuple(self.world_to_grid(np.asarray((goal_xy,)))[0])
        start = self._nearest_traversable(start)
        goal = self._nearest_traversable(goal)
        if start is None or goal is None:
            return []
        cells = _astar(self.traversable, start, goal)
        if not cells:
            return []
        cells = self._smooth_path(cells)
        return [tuple(value) for value in self.grid_to_world(cells)]

    def _nearest_traversable(self, cell, maximum_radius=20):
        column, row = int(cell[0]), int(cell[1])
        if self.in_bounds(column, row) and self.traversable[row, column]:
            return column, row
        for radius in range(1, maximum_radius + 1):
            candidates = []
            for dy in range(-radius, radius + 1):
                for dx in (-radius, radius):
                    candidates.append((column + dx, row + dy))
            for dx in range(-radius + 1, radius):
                for dy in (-radius, radius):
                    candidates.append((column + dx, row + dy))
            for candidate_column, candidate_row in candidates:
                if (
                    self.in_bounds(candidate_column, candidate_row)
                    and self.traversable[candidate_row, candidate_column]
                ):
                    return candidate_column, candidate_row
        return None

    def _smooth_path(self, cells):
        if len(cells) <= 2:
            return cells
        result = [cells[0]]
        anchor = 0
        while anchor < len(cells) - 1:
            selected = anchor + 1
            for candidate in range(len(cells) - 1, anchor, -1):
                # Require a one-cell corridor around the smoothed segment.
                # Center-line-only smoothing can place the path directly on
                # the edge of the inflated obstacle mask; a real chassis or
                # small localization error then cuts the wall corner.
                corridor = _erode_one(self.traversable)
                if _cell_line_clear(corridor, cells[anchor], cells[candidate]):
                    selected = candidate
                    break
            result.append(cells[selected])
            anchor = selected
        return result

    def inspection_targets(
        self,
        standoff_m=0.65,
        minimum_component_area_m2=0.015,
        maximum_targets=16,
    ):
        """Select safe views of wall zones and distinct occupied components."""
        components = _components(self.occupied)
        minimum_cells = max(
            3,
            int(math.ceil(minimum_component_area_m2 / self.resolution_m ** 2)),
        )
        components = [item for item in components if len(item) >= minimum_cells]
        components.sort(key=len, reverse=True)
        if not components:
            return []

        targets = []
        for component_index, component in enumerate(components):
            component_array = np.asarray(component, dtype=int)
            component_mask = np.zeros_like(self.occupied)
            component_mask[component_array[:, 0], component_array[:, 1]] = True
            boundary = component_mask & ~_erode_one(component_mask)
            # Photograph the face visible from known free space.  Sampling
            # the unknown/outside face of a room wall produces viewpoints
            # whose sight line necessarily crosses that same wall.
            boundary &= _dilate(self.free, 1)
            boundary_cells = np.argwhere(boundary)
            if not len(boundary_cells):
                continue
            extent_x = np.ptp(component_array[:, 1]) + 1
            extent_y = np.ptp(component_array[:, 0]) + 1
            touches_edge = bool(
                np.any(component_array[:, 0] <= 1)
                or np.any(component_array[:, 0] >= self.height - 2)
                or np.any(component_array[:, 1] <= 1)
                or np.any(component_array[:, 1] >= self.width - 2)
            )
            long_component = max(extent_x, extent_y) * self.resolution_m >= 1.5
            kind = (
                "perimeter_wall"
                if component_index == 0 or touches_edge or long_component
                else "interior_black_region"
            )
            estimated_views = int(math.ceil(len(boundary_cells) ** 0.5 / 6.0))
            desired_views = (
                min(6, max(4, estimated_views))
                if kind == "perimeter_wall"
                else min(4, max(2, estimated_views))
            )
            selected_boundaries = _farthest_point_samples(
                boundary_cells, desired_views
            )
            for local_index, boundary_cell in enumerate(selected_boundaries):
                view = self._view_for_boundary(boundary_cell, standoff_m)
                if view is None:
                    continue
                target_world = self.grid_to_world((
                    (int(boundary_cell[1]), int(boundary_cell[0])),
                ))[0]
                view_world = self.grid_to_world((view,))[0]
                yaw = math.atan2(
                    target_world[1] - view_world[1],
                    target_world[0] - view_world[0],
                )
                targets.append(InspectionTarget(
                    target_id="{}_{}_{}".format(
                        "wall" if kind == "perimeter_wall" else "object",
                        component_index + 1,
                        local_index + 1,
                    ),
                    kind=kind,
                    target_xy_m=(float(target_world[0]), float(target_world[1])),
                    view_xy_m=(float(view_world[0]), float(view_world[1])),
                    yaw_rad=yaw,
                    component_cells=len(component),
                ))
        return _spread_targets(targets, maximum_targets)

    def _view_for_boundary(self, boundary_row_column, standoff_m):
        row, column = (int(value) for value in boundary_row_column)
        desired_cells = max(1, int(round(standoff_m / self.resolution_m)))
        minimum_cells = max(self.clearance_cells, int(round(0.42 / self.resolution_m)))
        candidates = np.argwhere(self.traversable)
        if not len(candidates):
            return None
        squared = (
            (candidates[:, 0] - row) ** 2
            + (candidates[:, 1] - column) ** 2
        )
        radius = np.sqrt(squared)
        eligible = (
            (radius >= minimum_cells)
            & (radius <= desired_cells + max(3, self.clearance_cells))
        )
        candidates = candidates[eligible]
        radius = radius[eligible]
        if not len(candidates):
            return None
        order = np.argsort(np.abs(radius - desired_cells))
        target_xy = self.grid_to_world(((column, row),))[0]
        for index in order[:80]:
            candidate_row, candidate_column = candidates[index]
            candidate_xy = self.grid_to_world(((candidate_column, candidate_row),))[0]
            # The ray may end on an occupied target; require only intermediate
            # cells to be known and not occupied.
            cells = _bresenham(
                (int(candidate_column), int(candidate_row)), (column, row)
            )
            # A rendered wall may be several cells thick.  Permit the final
            # occupied run belonging to the target itself, while still
            # rejecting any intervening obstacle between camera and target.
            target_depth = max(2, self.clearance_cells // 2)
            visible_part = cells[:-target_depth] if len(cells) > target_depth else ()
            if all(
                self.in_bounds(c, r) and not self.occupied[r, c]
                for c, r in visible_part
            ):
                return int(candidate_column), int(candidate_row)
        return None


class ScanMapLocalizer:
    """Endpoint-to-map localizer with global startup and bounded tracking."""

    def __init__(
        self,
        known_map,
        lidar_offset_x_m=0.14,
        lidar_offset_y_m=0.0,
        min_distance_m=0.12,
        max_distance_m=8.0,
    ):
        self.map = known_map
        self.lidar_offset = np.asarray(
            (lidar_offset_x_m, lidar_offset_y_m), dtype=np.float64
        )
        self.min_distance_m = float(min_distance_m)
        self.max_distance_m = float(max_distance_m)
        self.motion_matcher = IcpScanMatcher(
            max_correspondence_m=0.35,
            min_points=35,
            coarse_yaw_range_deg=50.0,
            coarse_yaw_step_deg=3.0,
            trim_fraction=0.80,
        )

    def points(self, scan, maximum=300):
        points = scan.points(self.min_distance_m, self.max_distance_m)
        if len(points) > maximum:
            indices = np.linspace(0, len(points) - 1, maximum, dtype=int)
            points = points[indices]
        return points + self.lidar_offset

    def range_profile(self, scan, maximum=36):
        """Keep calibrated beam angles and metric ranges for map ray checks."""
        angles = np.asarray(scan.angles_rad, dtype=np.float64)
        distances = np.asarray(scan.distances_m, dtype=np.float64)
        valid = (
            np.isfinite(angles) & np.isfinite(distances)
            & (distances >= self.min_distance_m)
            & (distances <= self.max_distance_m)
        )
        angles, distances = angles[valid], distances[valid]
        if not len(angles):
            return np.empty((0, 2), dtype=np.float64)
        order = np.argsort(angles)
        angles, distances = angles[order], distances[order]
        if len(angles) > maximum:
            indices = np.linspace(0, len(angles) - 1, maximum, dtype=int)
            angles, distances = angles[indices], distances[indices]
        return np.column_stack((angles, distances))

    def global_localize(self, scan, hint=None):
        points = self.points(scan, maximum=220)
        range_profile = self.range_profile(scan)
        if len(points) < 80:
            return None
        candidates = []
        if hint is not None:
            candidates.extend(self._pose_lattice(
                hint,
                translation_radius_m=0.8,
                translation_step_m=0.20,
                yaw_radius_deg=45.0,
                yaw_step_deg=10.0,
            ))
        else:
            candidates.extend(self._global_lattice())
        coarse = self._rank(points, candidates, keep=8, range_profile=range_profile)
        if not coarse:
            return None
        refined_candidates = []
        for _score, pose, _metrics in coarse:
            refined_candidates.extend(self._pose_lattice(
                pose,
                translation_radius_m=0.24,
                translation_step_m=0.04,
                yaw_radius_deg=12.0,
                yaw_step_deg=2.0,
            ))
        ranked = self._rank(points, refined_candidates, keep=12, range_profile=range_profile)
        # A trajectory hint accelerates the search but must not hide a
        # symmetric alternative elsewhere in the room. Compare the accepted
        # local basin with a cheaper whole-map lattice before allowing motion.
        if hint is not None and ranked:
            global_coarse = self._rank(
                points,
                self._global_lattice(position_step_m=0.45, yaw_step_deg=30.0),
                keep=5,
                range_profile=range_profile,
            )
            global_refinement = []
            for _score, global_pose, _metrics in global_coarse:
                global_refinement.extend(self._pose_lattice(
                    global_pose,
                    translation_radius_m=0.24,
                    translation_step_m=0.06,
                    yaw_radius_deg=15.0,
                    yaw_step_deg=3.0,
                ))
            global_ranked = self._rank(
                points, global_refinement, keep=8, range_profile=range_profile
            )
            ranked.extend(global_ranked)
            ranked.sort(key=lambda item: item[0], reverse=True)
        return self._result_from_ranked(ranked)

    def global_localization_candidates(self, scan, maximum=4):
        """Return separated whole-map pose basins for visual disambiguation."""
        points = self.points(scan, maximum=220)
        range_profile = self.range_profile(scan)
        if len(points) < 80:
            return []
        coarse = self._rank(
            points, self._global_lattice(), keep=12, range_profile=range_profile
        )
        refined = []
        for _score, pose, _metrics in coarse:
            refined.extend(self._pose_lattice(
                pose,
                translation_radius_m=0.24,
                translation_step_m=0.04,
                yaw_radius_deg=12.0,
                yaw_step_deg=2.0,
            ))
        ranked = self._rank(points, refined, keep=32, range_profile=range_profile)
        basins = []
        for score, pose, (rmse, ratio) in ranked:
            if any(
                math.hypot(pose.x_m - old.pose.x_m, pose.y_m - old.pose.y_m) < 0.45
                and abs(normalize_angle(pose.yaw_rad - old.pose.yaw_rad)) < math.radians(25.0)
                for old in basins
            ):
                continue
            next_score = ranked[len(basins) + 1][0] if len(ranked) > len(basins) + 1 else score
            basins.append(LocalizationResult(
                pose=pose,
                rmse_m=rmse,
                inlier_ratio=ratio,
                score=score,
                ambiguity_margin=max(0.0, score - next_score),
            ))
            if len(basins) >= max(1, int(maximum)):
                break
        return basins

    def localize_near_position(self, scan, hint, radius_m=0.9):
        """Search every heading near a known physical start position."""
        points = self.points(scan, maximum=220)
        range_profile = self.range_profile(scan)
        if len(points) < 80:
            return None
        candidates = []
        position_step = 0.18
        offsets = np.arange(
            -radius_m, radius_m + position_step * 0.5, position_step
        )
        for dx in offsets:
            for dy in offsets:
                cell = self.map.world_to_grid(
                    ((hint.x_m + dx, hint.y_m + dy),)
                )[0]
                if (
                    not self.map.in_bounds(int(cell[0]), int(cell[1]))
                    or not self.map.traversable[int(cell[1]), int(cell[0])]
                ):
                    continue
                for yaw_deg in np.arange(-180.0, 180.0, 15.0):
                    candidates.append(Pose2D(
                        hint.x_m + float(dx),
                        hint.y_m + float(dy),
                        math.radians(float(yaw_deg)),
                    ))
        coarse = self._rank(points, candidates, keep=10, range_profile=range_profile)
        refined = []
        for _score, pose, _metrics in coarse:
            refined.extend(self._pose_lattice(
                pose,
                translation_radius_m=0.22,
                translation_step_m=0.04,
                yaw_radius_deg=12.0,
                yaw_step_deg=2.0,
            ))
        return self._result_from_ranked(
            self._rank(points, refined, keep=12, range_profile=range_profile)
        )

    def _global_lattice(
        self,
        position_step_m=0.28,
        yaw_step_deg=15.0,
    ):
        free_cells = np.argwhere(self.map.traversable)
        stride = max(1, int(round(position_step_m / self.map.resolution_m)))
        free_cells = free_cells[
            (free_cells[:, 0] % stride == 0)
            & (free_cells[:, 1] % stride == 0)
        ]
        positions = self.map.grid_to_world(free_cells[:, ::-1])
        candidates = []
        for x_m, y_m in positions:
            for yaw_deg in np.arange(-180.0, 180.0, yaw_step_deg):
                candidates.append(Pose2D(
                    float(x_m), float(y_m), math.radians(float(yaw_deg))
                ))
        return candidates

    def track(self, scan, previous_pose):
        points = self.points(scan)
        range_profile = self.range_profile(scan, maximum=28)
        if len(points) < 60:
            return None
        candidates = self._pose_lattice(
            previous_pose,
            translation_radius_m=0.12,
            translation_step_m=0.03,
            yaw_radius_deg=8.0,
            yaw_step_deg=2.0,
        )
        ranked = self._rank(
            points,
            candidates,
            keep=4,
            range_profile=range_profile,
            range_shortlist_min=12,
        )
        return self._result_from_ranked(ranked, ambiguity=False)

    def relative_motion(self, reference_scan, current_scan):
        """Estimate robot motion independently of the permanent map.

        The returned transform aligns the current scan with the reference
        scan and therefore represents the chassis motion from the reference
        pose to the current pose.  Startup localization uses it to reject a
        map pose that is merely stable in a symmetric room.
        """
        reference = self.points(reference_scan, maximum=360)
        current = self.points(current_scan, maximum=360)
        return self.motion_matcher.match(reference, current)

    def _pose_lattice(
        self,
        center,
        translation_radius_m,
        translation_step_m,
        yaw_radius_deg,
        yaw_step_deg,
    ):
        values = []
        offsets = np.arange(
            -translation_radius_m,
            translation_radius_m + translation_step_m * 0.5,
            translation_step_m,
        )
        yaw_offsets = np.arange(
            -yaw_radius_deg,
            yaw_radius_deg + yaw_step_deg * 0.5,
            yaw_step_deg,
        )
        for dx in offsets:
            for dy in offsets:
                cell = self.map.world_to_grid(((center.x_m + dx, center.y_m + dy),))[0]
                if (
                    not self.map.in_bounds(int(cell[0]), int(cell[1]))
                    or not self.map.traversable[int(cell[1]), int(cell[0])]
                ):
                    continue
                for dyaw in yaw_offsets:
                    values.append(Pose2D(
                        center.x_m + float(dx),
                        center.y_m + float(dy),
                        normalize_angle(center.yaw_rad + math.radians(float(dyaw))),
                    ))
        return values

    def _rank(
        self,
        points,
        candidates,
        keep,
        range_profile=None,
        range_shortlist_min=72,
    ):
        scored = []
        for pose in candidates:
            metrics = self._evaluate(points, pose)
            if metrics is None:
                continue
            score, rmse_m, inlier_ratio = metrics
            scored.append((score, pose, (rmse_m, inlier_ratio)))
        scored.sort(key=lambda item: item[0], reverse=True)
        # Full ray casting is reserved for the endpoint score's best basins.
        # This keeps whole-map startup practical while adding a genuinely
        # metric test: predicted wall range at each angle must match the scan.
        if range_profile is not None and len(range_profile):
            shortlist_count = min(
                len(scored),
                max(int(range_shortlist_min), int(keep) * 3),
            )
            rescored = []
            for score, pose, metrics in scored[:shortlist_count]:
                range_metrics = self._range_consistency(pose, range_profile)
                if range_metrics is not None:
                    agreement, mean_error, coverage = range_metrics
                    score += (
                        0.36 * agreement
                        - 0.40 * min(mean_error, 0.80)
                        + 0.06 * coverage
                    )
                rescored.append((score, pose, metrics))
            scored[:shortlist_count] = rescored
            scored.sort(key=lambda item: item[0], reverse=True)
        # Non-maximum suppression keeps alternative pose basins alive. Without
        # it, the top list is filled by millimetre-scale variants of one pose,
        # hiding an equally plausible solution on the opposite side of a
        # symmetric room.
        selected = []
        for item in scored:
            pose = item[1]
            if selected and any(
                math.hypot(
                    pose.x_m - old[1].x_m,
                    pose.y_m - old[1].y_m,
                ) < 0.10
                and abs(normalize_angle(
                    pose.yaw_rad - old[1].yaw_rad
                )) < math.radians(4.0)
                for old in selected
            ):
                continue
            selected.append(item)
            if len(selected) >= keep:
                break
        return selected

    def _range_consistency(self, pose, range_profile):
        errors = []
        for local_angle, observed_range in np.asarray(range_profile):
            predicted = self._raycast_range(pose, float(local_angle))
            if predicted is not None:
                errors.append(abs(float(observed_range) - predicted))
        minimum = max(8, int(len(range_profile) * 0.30))
        if len(errors) < minimum:
            return None
        errors = np.sort(np.asarray(errors, dtype=np.float64))
        robust = errors[:max(minimum, int(len(errors) * 0.80))]
        agreement = float(np.mean(robust <= 0.20))
        mean_error = float(np.mean(robust))
        coverage = float(len(errors)) / max(1, len(range_profile))
        return agreement, mean_error, coverage

    def _raycast_range(self, pose, local_angle, maximum_range=None):
        """Return first permanent occupied-cell distance for one LiDAR beam."""
        sensor = pose.transform_points(self.lidar_offset.reshape(1, 2))[0]
        angle = pose.yaw_rad + float(local_angle)
        direction = np.asarray((math.cos(angle), math.sin(angle)))
        step = max(0.01, self.map.resolution_m * 0.55)
        maximum = self.max_distance_m if maximum_range is None else float(maximum_range)
        distance = max(self.min_distance_m, step)
        while distance <= maximum:
            point = sensor + direction * distance
            column, row = self.map.world_to_grid((point,))[0]
            column, row = int(column), int(row)
            if not self.map.in_bounds(column, row):
                return None
            if self.map.occupied[row, column]:
                return float(distance)
            distance += step
        return None

    def _evaluate(self, local_points, pose):
        world = pose.transform_points(local_points)
        cells = self.map.world_to_grid(world)
        columns, rows = cells[:, 0], cells[:, 1]
        inside = (
            (columns >= 0) & (columns < self.map.width)
            & (rows >= 0) & (rows < self.map.height)
        )
        if np.count_nonzero(inside) < max(35, len(local_points) * 0.35):
            return None
        distances = np.full(len(local_points), 0.60, dtype=np.float64)
        distances[inside] = np.minimum(
            0.60,
            self.map.obstacle_distance_cells[rows[inside], columns[inside]]
            * self.map.resolution_m,
        )
        inliers = distances <= 0.16
        inlier_ratio = float(np.count_nonzero(inliers)) / max(1, len(distances))
        if np.any(inliers):
            rmse = float(np.sqrt(np.mean(distances[inliers] ** 2)))
        else:
            rmse = math.inf
        robust = np.sort(distances)[:max(1, int(len(distances) * 0.75))]
        score = inlier_ratio - float(np.mean(robust)) * 1.4
        return score, rmse, inlier_ratio

    @staticmethod
    def _result_from_ranked(ranked, ambiguity=True):
        if not ranked:
            return None
        score, pose, (rmse, ratio) = ranked[0]
        margin = math.inf
        if ambiguity:
            for other_score, other_pose, _metrics in ranked[1:]:
                separated = (
                    math.hypot(
                        pose.x_m - other_pose.x_m,
                        pose.y_m - other_pose.y_m,
                    ) > 0.45
                    or abs(normalize_angle(
                        pose.yaw_rad - other_pose.yaw_rad
                    )) > math.radians(25.0)
                )
                if separated:
                    margin = score - other_score
                    break
        return LocalizationResult(pose, rmse, ratio, score, margin)


def consistent_localization(results, position_tolerance_m=0.22, yaw_tolerance_deg=9.0):
    reliable = [item for item in results if item is not None and item.reliable]
    if len(reliable) < 3:
        return None
    anchor = reliable[-1]
    compatible = [
        item for item in reliable
        if math.hypot(
            item.pose.x_m - anchor.pose.x_m,
            item.pose.y_m - anchor.pose.y_m,
        ) <= position_tolerance_m
        and abs(normalize_angle(
            item.pose.yaw_rad - anchor.pose.yaw_rad
        )) <= math.radians(yaw_tolerance_deg)
    ]
    if len(compatible) < 3:
        return None
    x_m = float(np.median([item.pose.x_m for item in compatible]))
    y_m = float(np.median([item.pose.y_m for item in compatible]))
    sine = float(np.mean([math.sin(item.pose.yaw_rad) for item in compatible]))
    cosine = float(np.mean([math.cos(item.pose.yaw_rad) for item in compatible]))
    return LocalizationResult(
        Pose2D(x_m, y_m, math.atan2(sine, cosine)),
        float(np.median([item.rmse_m for item in compatible])),
        float(np.median([item.inlier_ratio for item in compatible])),
        float(np.median([item.score for item in compatible])),
        min(item.ambiguity_margin for item in compatible),
    )


def order_targets(known_map, start_xy, targets):
    """Nearest-reachable ordering; unreachable viewpoints are omitted."""
    pending = list(targets)
    ordered = []
    current = tuple(start_xy)
    while pending:
        choices = []
        for index, target in enumerate(pending):
            path = known_map.path(current, target.view_xy_m)
            if path:
                length = sum(
                    math.hypot(b[0] - a[0], b[1] - a[1])
                    for a, b in zip(path, path[1:])
                )
                choices.append((length, index, path))
        if not choices:
            break
        _length, selected, path = min(choices, key=lambda item: item[0])
        target = pending.pop(selected)
        ordered.append((target, path))
        current = target.view_xy_m
    return ordered


def _components(mask):
    mask = np.asarray(mask, dtype=np.bool_)
    visited = np.zeros_like(mask)
    result = []
    height, width = mask.shape
    for row, column in np.argwhere(mask):
        if visited[row, column]:
            continue
        queue = deque(((int(row), int(column)),))
        visited[row, column] = True
        component = []
        while queue:
            current_row, current_column = queue.popleft()
            component.append((current_row, current_column))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dx or dy):
                        continue
                    nr, nc = current_row + dy, current_column + dx
                    if (
                        0 <= nr < height and 0 <= nc < width
                        and mask[nr, nc] and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        queue.append((nr, nc))
        result.append(component)
    return result


def _erode_one(mask):
    padded = np.pad(mask, 1, constant_values=False)
    result = np.ones_like(mask)
    for dy in range(3):
        for dx in range(3):
            result &= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return result


def _farthest_point_samples(points, count):
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= count:
        return points.astype(int)
    center = points.mean(axis=0)
    selected = [int(np.argmax(np.sum((points - center) ** 2, axis=1)))]
    minimum_squared = np.sum((points - points[selected[0]]) ** 2, axis=1)
    while len(selected) < count:
        index = int(np.argmax(minimum_squared))
        selected.append(index)
        minimum_squared = np.minimum(
            minimum_squared,
            np.sum((points - points[index]) ** 2, axis=1),
        )
    return points[selected].astype(int)


def _spread_targets(targets, maximum):
    if len(targets) <= maximum:
        return targets
    points = np.asarray([item.target_xy_m for item in targets])
    center = points.mean(axis=0)
    selected = [int(np.argmax(np.sum((points - center) ** 2, axis=1)))]
    minimum_squared = np.sum((points - points[selected[0]]) ** 2, axis=1)
    while len(selected) < maximum:
        minimum_squared[selected] = -1.0
        index = int(np.argmax(minimum_squared))
        selected.append(index)
        minimum_squared = np.minimum(
            minimum_squared,
            np.sum((points - points[index]) ** 2, axis=1),
        )
    return [targets[index] for index in selected]


def _bresenham(start, end):
    x0, y0 = start
    x1, y1 = end
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    cells = []
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return cells
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy


def _cell_line_clear(mask, start, end):
    return all(mask[row, column] for column, row in _bresenham(start, end))


def _astar(mask, start, goal):
    if start == goal:
        return [start]
    height, width = mask.shape
    queue = [(0.0, 0.0, start)]
    came_from = {}
    costs = {start: 0.0}
    neighbours = (
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
    )
    while queue:
        _priority, cost, current = heapq.heappop(queue)
        if cost != costs.get(current):
            continue
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))
        column, row = current
        for dx, dy, step in neighbours:
            nc, nr = column + dx, row + dy
            if not (0 <= nc < width and 0 <= nr < height and mask[nr, nc]):
                continue
            if dx and dy and not (mask[row, nc] and mask[nr, column]):
                continue
            candidate = cost + step
            neighbour = (nc, nr)
            if candidate >= costs.get(neighbour, math.inf):
                continue
            costs[neighbour] = candidate
            came_from[neighbour] = current
            heuristic = math.hypot(goal[0] - nc, goal[1] - nr)
            heapq.heappush(queue, (candidate + heuristic, candidate, neighbour))
    return []
