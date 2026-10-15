"""Computer-only, system-level semantic inspection simulation.

The simulator intentionally uses the project's saved PGM/YAML map, production
inspection-target generator and production waypoint controller.  It adds a
hidden "actual room" and a four-wheel skid-steer plant so a successful run is
not equivalent to following the planned polyline perfectly.

Modelled disturbances:

* spatially varying static friction;
* independent gain and breakaway threshold for all four wheels;
* persistent single-wheel free-spin events and misleading encoder motion;
* small occupied-boundary differences between the saved map and actual room;
* a glass wall on the left whose lidar rays may leak to useless outside hits;
* lidar Gaussian noise, bias, quantisation, dropout, short speckle and
  multi-path/long-return errors;
* command latency, side imbalance and command/body direction disagreement;
* bounded scan-match jumps plus action-envelope rejection;
* visual Top-K coarse localization, a second frame after a safe short move,
  and global-active-localization fallback;
* closed-loop visits, camera alignment, simulated dual-camera confirmation and
  persistent semantic-map output.

No Raspberry Pi, BLE device, camera, network service or GPU is used.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from raspberry_pi.known_map_navigation import (
    InspectionTarget,
    KnownOccupancyMap,
    LocalizationResult,
    _components,
    _dilate,
    _erode_one,
)
from raspberry_pi.map_camera_inspection import (
    _apply_verified_semantic_views,
    _build_speed_controller,
    _command_for_pose,
    _information_view_targets,
    _inspection_navigation_command_ttl,
    _inspection_path_for_target,
    _inspection_target_visible_from,
    _navigation_command_ttl,
    _tracking_update,
    build_argument_parser,
)
from raspberry_pi.mapping import MotionCommand, Pose2D
from raspberry_pi.mapping.geometry import normalize_angle


# Latest strictly accepted field start from 20260915_142137. Keeping the
# stress run anchored here exercises the same upper-left route that jammed.
FIELD_START_POSE = Pose2D(-0.935, 0.625, math.radians(-126.0))


def _production_controller_dead_zone_probe() -> dict:
    """Replay the loaded-chassis stall seen in field run 20260905_155436."""
    args = build_argument_parser().parse_args([
        "--speed", "330",
        "--max-drive-speed", "450",
        "--start-kick-speed", "420",
        "--slow-speed", "240",
        "--drive-speed-step", "30",
    ])
    controller = _build_speed_controller(args, navigation=True)
    request = MotionCommand(240, -300, "advancing", "dead-zone probe")
    outputs = []
    command_ttls_ms = []
    recovered = False
    for _frame in range(14):
        applied = controller.apply(request)
        outputs.append(int(applied.linear_mm_s))
        command_ttls_ms.append(_inspection_navigation_command_ttl(
            controller,
            applied,
            args.command_ttl_ms,
            1.50,
            args.stop_distance,
        ))
        # The field log showed that the loaded car remained still below the
        # configured 450 ceiling.  Model that worst-case dead zone directly.
        translation_m = 0.050 if abs(applied.linear_mm_s) >= 450 else 0.001
        controller.observe(_tracking_update(
            Pose2D(),
            LocalizationResult(
                Pose2D(translation_m, 0.0, 0.0),
                0.03,
                0.90,
                math.inf,
            ),
            0.45,
        ))
        if translation_m >= 0.02:
            recovered = True
            break
    following = controller.apply(request)
    return {
        "recovered": recovered,
        "frames_until_motion": len(outputs),
        "applied_linear_mm_s": outputs,
        "mixed_arc_ttl_ms": command_ttls_ms,
        "mixed_arc_ttl_allows_loaded_start": bool(
            command_ttls_ms and min(command_ttls_ms) >= 600
        ),
        "peak_linear_mm_s": max(outputs, default=0),
        "post_recovery_linear_mm_s": int(following.linear_mm_s),
        "bounded_single_peak": outputs.count(450) <= 1,
        "returned_to_fine_speed": int(following.linear_mm_s) == 240,
    }


@dataclass(frozen=True)
class StressProfile:
    name: str = "realistic"
    dt_s: float = 0.18
    track_width_m: float = 0.235
    robot_radius_m: float = 0.20
    wheel_gain_min: float = 0.86
    wheel_gain_max: float = 1.14
    wheel_static_min_m_s: float = 0.085
    wheel_static_max_m_s: float = 0.155
    friction_min: float = 0.62
    friction_max: float = 1.18
    slip_probability_per_wheel_step: float = 0.008
    slip_duration_min: int = 2
    slip_duration_max: int = 8
    command_latency_frames: int = 1
    command_expiry_probability: float = 0.010
    battery_output_min: float = 0.84
    battery_output_max: float = 1.02
    body_jam_probability_per_forward_step: float = 0.0035
    map_boundary_change_fraction: float = 0.075
    lidar_beams: int = 72
    lidar_noise_sigma_m: float = 0.012
    lidar_dropout_probability: float = 0.025
    lidar_short_speckle_probability: float = 0.009
    lidar_multipath_probability: float = 0.018
    glass_leak_probability: float = 0.58
    tracker_jump_probability: float = 0.015
    maximum_target_steps: int = 900
    maximum_alignment_steps: int = 80

    @classmethod
    def named(cls, name: str) -> "StressProfile":
        name = str(name).strip().lower()
        if name == "mild":
            return cls(
                name="mild",
                wheel_gain_min=0.92,
                wheel_gain_max=1.08,
                friction_min=0.78,
                friction_max=1.10,
                slip_probability_per_wheel_step=0.003,
                command_expiry_probability=0.004,
                battery_output_min=0.92,
                body_jam_probability_per_forward_step=0.0015,
                map_boundary_change_fraction=0.035,
                lidar_dropout_probability=0.012,
                lidar_short_speckle_probability=0.004,
                lidar_multipath_probability=0.008,
                glass_leak_probability=0.42,
                tracker_jump_probability=0.006,
            )
        if name == "harsh":
            return cls(
                name="harsh",
                wheel_gain_min=0.78,
                wheel_gain_max=1.22,
                wheel_static_min_m_s=0.095,
                wheel_static_max_m_s=0.185,
                friction_min=0.48,
                friction_max=1.25,
                slip_probability_per_wheel_step=0.016,
                slip_duration_max=12,
                command_latency_frames=2,
                command_expiry_probability=0.025,
                battery_output_min=0.70,
                body_jam_probability_per_forward_step=0.008,
                map_boundary_change_fraction=0.12,
                lidar_noise_sigma_m=0.020,
                lidar_dropout_probability=0.045,
                lidar_short_speckle_probability=0.016,
                lidar_multipath_probability=0.035,
                glass_leak_probability=0.72,
                tracker_jump_probability=0.030,
                maximum_target_steps=1150,
                maximum_alignment_steps=100,
            )
        if name != "realistic":
            raise ValueError("profile must be mild, realistic or harsh")
        return cls()


@dataclass
class LidarFrame:
    angles_rad: np.ndarray
    distances_m: np.ndarray
    valid_ratio: float
    glass_leaks: int
    dropouts: int
    short_speckles: int
    multipath_returns: int
    front_clearance_m: float
    rear_clearance_m: float


@dataclass(frozen=True)
class WallSurfaceSample:
    sample_id: str
    component_id: int
    xy_m: tuple[float, float]
    visual_only_label: str | None = None


@dataclass
class PlantStep:
    pose: Pose2D
    encoder_delta: Pose2D
    actual_translation_m: float
    actual_yaw_rad: float
    collision: bool
    applied_linear_mm_s: int
    applied_angular_mrad_s: int
    new_slip_events: int
    active_slipping_wheels: int


def _pose_dict(pose: Pose2D) -> dict:
    return {
        "x_m": round(float(pose.x_m), 5),
        "y_m": round(float(pose.y_m), 5),
        "yaw_deg": round(math.degrees(float(pose.yaw_rad)), 3),
    }


def _blend_pose(first: Pose2D, second: Pose2D, second_weight: float) -> Pose2D:
    weight = min(1.0, max(0.0, float(second_weight)))
    first_weight = 1.0 - weight
    yaw = math.atan2(
        first_weight * math.sin(first.yaw_rad) + weight * math.sin(second.yaw_rad),
        first_weight * math.cos(first.yaw_rad) + weight * math.cos(second.yaw_rad),
    )
    return Pose2D(
        first_weight * first.x_m + weight * second.x_m,
        first_weight * first.y_m + weight * second.y_m,
        normalize_angle(yaw),
    )


def _path_length(path: Iterable[tuple[float, float]]) -> float:
    values = list(path)
    return sum(
        math.hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(values, values[1:])
    )


def _distance_to_polyline(point: tuple[float, float], path) -> float:
    """Return metric cross-track distance to a planned polyline."""
    values = [np.asarray(item, dtype=np.float64) for item in path]
    query = np.asarray(point, dtype=np.float64)
    if not values:
        return math.inf
    if len(values) == 1:
        return float(np.linalg.norm(query - values[0]))
    best = math.inf
    for first, second in zip(values, values[1:]):
        segment = second - first
        length_squared = float(np.dot(segment, segment))
        fraction = 0.0 if length_squared <= 1e-12 else float(
            np.clip(np.dot(query - first, segment) / length_squared, 0.0, 1.0)
        )
        best = min(best, float(np.linalg.norm(query - (first + fraction * segment))))
    return best


def _farthest_spaced_points(points, spacing_m: float, maximum: int = 240):
    points = np.asarray(points, dtype=np.float64)
    if not len(points):
        return points
    center = np.mean(points, axis=0)
    selected = [int(np.argmax(np.sum((points - center) ** 2, axis=1)))]
    minimum_squared = np.sum((points - points[selected[0]]) ** 2, axis=1)
    threshold_squared = float(spacing_m) ** 2
    while len(selected) < min(int(maximum), len(points)):
        index = int(np.argmax(minimum_squared))
        if float(minimum_squared[index]) < threshold_squared:
            break
        selected.append(index)
        minimum_squared = np.minimum(
            minimum_squared,
            np.sum((points - points[index]) ** 2, axis=1),
        )
    return points[selected]


def _wall_surface_samples(
    known_map: KnownOccupancyMap,
    targets: list[InspectionTarget],
    spacing_m: float = 0.30,
) -> list[WallSurfaceSample]:
    """Sample visible wall faces, including objects absent from lidar geometry.

    Mounted signs/switches/posters deliberately do not alter the occupancy map.
    They therefore test camera coverage independently of what lidar can scan.
    """
    perimeter_ids = {
        SemanticLedger.component_id(target)
        for target in targets
        if target.kind == "perimeter_wall"
    }
    minimum_cells = max(3, int(math.ceil(0.015 / known_map.resolution_m ** 2)))
    components = [
        item for item in _components(known_map.occupied) if len(item) >= minimum_cells
    ]
    components.sort(key=len, reverse=True)
    labels = ("wall_sign", "light_switch", "poster", "asset_tag", "extinguisher")
    result = []
    for component_id, component in enumerate(components, start=1):
        if component_id not in perimeter_ids:
            continue
        cells = np.asarray(component, dtype=int)
        mask = np.zeros_like(known_map.occupied)
        mask[cells[:, 0], cells[:, 1]] = True
        boundary = mask & ~_erode_one(mask) & _dilate(known_map.free, 1)
        boundary_cells = np.argwhere(boundary)
        if not len(boundary_cells):
            continue
        world = known_map.grid_to_world(boundary_cells[:, ::-1])
        selected = _farthest_spaced_points(world, spacing_m)
        # Roughly five radar-invisible mounted objects per long wall component,
        # spread across the sampled face rather than clustered at one target.
        feature_stride = max(1, int(math.ceil(len(selected) / 5.0)))
        for index, point in enumerate(selected):
            feature_label = (
                labels[(component_id + index) % len(labels)]
                if index % feature_stride == 0 else None
            )
            result.append(WallSurfaceSample(
                sample_id=f"wall_{component_id}_surface_{index + 1}",
                component_id=component_id,
                xy_m=(float(point[0]), float(point[1])),
                visual_only_label=feature_label,
            ))
    return result


def _visible_in_camera(
    known_map: KnownOccupancyMap,
    pose: Pose2D,
    sample: WallSurfaceSample,
    *,
    horizontal_fov_deg: float = 122.0,
    maximum_range_m: float = 3.40,
) -> bool:
    dx = sample.xy_m[0] - pose.x_m
    dy = sample.xy_m[1] - pose.y_m
    distance = math.hypot(dx, dy)
    if distance < 0.35 or distance > maximum_range_m:
        return False
    bearing_error = abs(normalize_angle(math.atan2(dy, dx) - pose.yaw_rad))
    return bool(
        bearing_error <= math.radians(horizontal_fov_deg * 0.5)
        and _inspection_target_visible_from(
            known_map, (pose.x_m, pose.y_m), sample.xy_m
        )
    )


def _wall_coverage(
    known_map: KnownOccupancyMap,
    samples: list[WallSurfaceSample],
    capture_poses: list[Pose2D],
) -> tuple[dict, set[str]]:
    covered = {
        sample.sample_id
        for sample in samples
        if any(_visible_in_camera(known_map, pose, sample) for pose in capture_poses)
    }
    features = [sample for sample in samples if sample.visual_only_label is not None]
    covered_features = [sample for sample in features if sample.sample_id in covered]
    covered_points = np.asarray(
        [sample.xy_m for sample in samples if sample.sample_id in covered],
        dtype=np.float64,
    )
    maximum_uncovered_gap = 0.0
    if len(covered_points):
        for sample in samples:
            if sample.sample_id not in covered:
                maximum_uncovered_gap = max(
                    maximum_uncovered_gap,
                    float(np.min(np.linalg.norm(covered_points - sample.xy_m, axis=1))),
                )
    elif samples:
        maximum_uncovered_gap = math.inf
    return {
        "wall_sample_count": len(samples),
        "covered_wall_sample_count": len(covered),
        "wall_surface_coverage_ratio": len(covered) / max(1, len(samples)),
        "visual_only_feature_count": len(features),
        "covered_visual_only_feature_count": len(covered_features),
        "visual_only_feature_coverage_ratio": len(covered_features) / max(1, len(features)),
        "maximum_uncovered_gap_m": float(maximum_uncovered_gap),
        "camera_horizontal_fov_deg": 122.0,
        "camera_maximum_range_m": 3.40,
    }, covered


def _supplement_wall_coverage_targets(
    known_map: KnownOccupancyMap,
    targets: list[InspectionTarget],
    samples: list[WallSurfaceSample],
    *,
    maximum_supplements: int = 10,
) -> tuple[list[InspectionTarget], dict]:
    """Add only views that expose a currently occluded wall face."""
    result = list(targets)
    additions = []
    for supplement_index in range(maximum_supplements):
        nominal_poses = [
            Pose2D(target.view_xy_m[0], target.view_xy_m[1], target.yaw_rad)
            for target in result
        ]
        coverage, covered = _wall_coverage(known_map, samples, nominal_poses)
        if (
            coverage["wall_surface_coverage_ratio"] >= 0.96
            and coverage["visual_only_feature_coverage_ratio"] >= 1.0
        ):
            break
        uncovered = [sample for sample in samples if sample.sample_id not in covered]
        # Radar-invisible objects have priority.  Then choose the sample farthest
        # from anything already visible so each extra stop adds useful area.
        covered_points = np.asarray(
            [sample.xy_m for sample in samples if sample.sample_id in covered],
            dtype=np.float64,
        )
        candidates = []
        for sample in uncovered:
            column, row = known_map.world_to_grid((sample.xy_m,))[0]
            view_cell = known_map._view_for_boundary(
                (int(row), int(column)), standoff_m=0.72
            )
            if view_cell is None:
                continue
            view = known_map.grid_to_world((view_cell,))[0]
            yaw = math.atan2(sample.xy_m[1] - view[1], sample.xy_m[0] - view[0])
            proposed = InspectionTarget(
                target_id=(
                    f"wall_{sample.component_id}_coverage_{supplement_index + 1}"
                ),
                kind="perimeter_wall",
                target_xy_m=sample.xy_m,
                view_xy_m=(float(view[0]), float(view[1])),
                yaw_rad=yaw,
                component_cells=next(
                    (
                        target.component_cells for target in targets
                        if SemanticLedger.component_id(target) == sample.component_id
                    ),
                    0,
                ),
            )
            widened, _metadata = _information_view_targets(
                known_map, [proposed], maximum_distance_m=2.60
            )
            proposed = widened[0]
            proposed_pose = Pose2D(
                proposed.view_xy_m[0], proposed.view_xy_m[1], proposed.yaw_rad
            )
            if not _visible_in_camera(known_map, proposed_pose, sample):
                continue
            nearest_visible = (
                float(np.min(np.linalg.norm(covered_points - sample.xy_m, axis=1)))
                if len(covered_points) else math.inf
            )
            nearest_stop = min(
                math.hypot(
                    proposed.view_xy_m[0] - target.view_xy_m[0],
                    proposed.view_xy_m[1] - target.view_xy_m[1],
                )
                for target in result
            )
            candidates.append((
                int(sample.visual_only_label is not None),
                nearest_visible,
                nearest_stop,
                sample.sample_id,
                proposed,
            ))
        if not candidates:
            break
        # Prefer a missing mounted object, then a large uncovered wall gap and
        # finally a spatially distinct stop (fewer clustered photographs).
        selected = max(candidates, key=lambda item: item[:4])[-1]
        result.append(selected)
        additions.append({
            "target_id": selected.target_id,
            "target_xy_m": selected.target_xy_m,
            "view_xy_m": selected.view_xy_m,
        })
    final_coverage, _covered = _wall_coverage(
        known_map,
        samples,
        [Pose2D(target.view_xy_m[0], target.view_xy_m[1], target.yaw_rad) for target in result],
    )
    return result, {
        "supplement_count": len(additions),
        "supplements": additions,
        "nominal_coverage_after_supplement": final_coverage,
    }


class SpatialFrictionField:
    """Smooth position-dependent traction with several low-friction patches."""

    def __init__(self, known_map: KnownOccupancyMap, rng, profile: StressProfile):
        free_cells = np.argwhere(known_map.traversable)
        if not len(free_cells):
            raise ValueError("known map has no traversable cells")
        choices = free_cells[rng.choice(len(free_cells), size=min(9, len(free_cells)), replace=False)]
        centers = known_map.grid_to_world(choices[:, ::-1])
        self.patches = [
            (
                float(x_m),
                float(y_m),
                float(rng.uniform(0.35, 0.90)),
                float(rng.uniform(profile.friction_min, profile.friction_max)),
            )
            for x_m, y_m in centers
        ]
        self.minimum = 1.0
        self.maximum = 1.0

    def coefficient(self, x_m: float, y_m: float) -> float:
        weighted = 1.0
        total = 1.0
        for center_x, center_y, radius, value in self.patches:
            distance = math.hypot(x_m - center_x, y_m - center_y)
            influence = math.exp(-0.5 * (distance / radius) ** 2)
            weighted += influence * value
            total += influence
        coefficient = weighted / total
        self.minimum = min(self.minimum, coefficient)
        self.maximum = max(self.maximum, coefficient)
        return coefficient


class ActualScene:
    """Hidden room geometry: close to, but deliberately not equal to, the PGM."""

    def __init__(self, known_map: KnownOccupancyMap, rng, profile: StressProfile):
        self.map = known_map
        self.profile = profile
        occupied = np.asarray(known_map.occupied, dtype=np.bool_).copy()
        boundary = occupied & ~(
            np.roll(occupied, 1, axis=0)
            & np.roll(occupied, -1, axis=0)
            & np.roll(occupied, 1, axis=1)
            & np.roll(occupied, -1, axis=1)
        )
        boundary_cells = np.argwhere(boundary)
        requested = max(1, int(len(boundary_cells) * profile.map_boundary_change_fraction))
        selected = boundary_cells[
            rng.choice(len(boundary_cells), size=min(requested, len(boundary_cells)), replace=False)
        ]
        changed = 0
        neighbour_offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
        for row, column in selected:
            offsets = list(neighbour_offsets)
            rng.shuffle(offsets)
            for dy, dx in offsets:
                nr, nc = int(row + dy), int(column + dx)
                if not known_map.in_bounds(nc, nr) or not known_map.free[nr, nc]:
                    continue
                occupied[nr, nc] = True
                changed += 1
                break
        self.occupied = occupied
        self.changed_cells = int(np.count_nonzero(occupied ^ known_map.occupied))
        blocked_source = occupied | known_map.unknown
        radius_cells = max(1, int(math.ceil(profile.robot_radius_m / known_map.resolution_m)))
        self.blocked = _dilate(blocked_source, radius_cells)
        occupied_cells = np.argwhere(occupied)
        occupied_world = known_map.grid_to_world(occupied_cells[:, ::-1])
        self.occupied_world = np.asarray(occupied_world, dtype=np.float64)
        self.left_glass_x_m = float(np.percentile(occupied_world[:, 0], 2.0))
        self.minimum_ray_distance_m = 0.08
        self.maximum_ray_distance_m = 8.0
        self._ray_distances = np.arange(
            self.minimum_ray_distance_m,
            self.maximum_ray_distance_m + known_map.resolution_m * 0.45,
            max(0.018, known_map.resolution_m * 0.45),
            dtype=np.float64,
        )

    def clear(self, x_m: float, y_m: float) -> bool:
        column, row = self.map.world_to_grid(((x_m, y_m),))[0]
        return bool(
            self.map.in_bounds(int(column), int(row))
            and not self.blocked[int(row), int(column)]
        )

    def segment_clear(self, first: Pose2D, second: Pose2D) -> bool:
        distance = math.hypot(second.x_m - first.x_m, second.y_m - first.y_m)
        count = max(2, int(math.ceil(distance / 0.025)) + 1)
        for fraction in np.linspace(0.0, 1.0, count):
            if not self.clear(
                first.x_m + (second.x_m - first.x_m) * float(fraction),
                first.y_m + (second.y_m - first.y_m) * float(fraction),
            ):
                return False
        return True

    def center_clearance_m(self, pose: Pose2D) -> float:
        if not len(self.occupied_world):
            return math.inf
        point = np.asarray((pose.x_m, pose.y_m), dtype=np.float64)
        return float(np.min(np.linalg.norm(self.occupied_world - point, axis=1)))

    def _raycast(self, pose: Pose2D, local_angle: float) -> tuple[float, bool]:
        angle = pose.yaw_rad + float(local_angle)
        dx, dy = math.cos(angle), math.sin(angle)
        x_values = pose.x_m + dx * self._ray_distances
        y_values = pose.y_m + dy * self._ray_distances
        columns = np.floor(
            (x_values - self.map.origin_x_m) / self.map.resolution_m
        ).astype(int)
        rows = np.floor(
            (y_values - self.map.origin_y_m) / self.map.resolution_m
        ).astype(int)
        in_bounds = (
            (columns >= 0) & (columns < self.map.width)
            & (rows >= 0) & (rows < self.map.height)
        )
        if np.any(in_bounds):
            safe_columns = np.clip(columns, 0, self.map.width - 1)
            safe_rows = np.clip(rows, 0, self.map.height - 1)
            hit = in_bounds & self.occupied[safe_rows, safe_columns]
            indices = np.flatnonzero(hit)
            if len(indices):
                index = int(indices[0])
                glass = x_values[index] <= self.left_glass_x_m + 0.12
                return float(self._ray_distances[index]), bool(glass)
        return self.maximum_ray_distance_m, False

    def scan(self, pose: Pose2D, rng) -> LidarFrame:
        count = max(36, int(self.profile.lidar_beams))
        nominal_angles = np.linspace(-math.pi, math.pi, count, endpoint=False)
        angles = nominal_angles + rng.normal(0.0, math.radians(0.08), size=count)
        distances = np.empty(count, dtype=np.float64)
        glass_leaks = dropouts = speckles = multipath = 0
        scan_bias = float(rng.normal(0.0, self.profile.lidar_noise_sigma_m * 0.35))
        for index, angle in enumerate(angles):
            distance, glass = self._raycast(pose, float(angle))
            if glass and rng.random() < self.profile.glass_leak_probability:
                glass_leaks += 1
                if rng.random() < 0.30:
                    distances[index] = np.nan
                    dropouts += 1
                    continue
                distance = min(
                    self.maximum_ray_distance_m,
                    distance + float(rng.uniform(0.65, 2.80)),
                )
            if rng.random() < self.profile.lidar_dropout_probability:
                distances[index] = np.nan
                dropouts += 1
                continue
            if rng.random() < self.profile.lidar_short_speckle_probability:
                distance = float(rng.uniform(0.08, min(0.40, max(0.09, distance))))
                speckles += 1
            elif rng.random() < self.profile.lidar_multipath_probability:
                distance = min(
                    self.maximum_ray_distance_m,
                    distance + float(rng.uniform(0.20, 1.30)),
                )
                multipath += 1
            distance += scan_bias + float(rng.normal(0.0, self.profile.lidar_noise_sigma_m))
            distances[index] = round(max(0.08, min(8.0, distance)) / 0.005) * 0.005
        valid = np.isfinite(distances)
        def robust_sector_clearance(center_rad: float) -> float:
            errors = np.abs(np.arctan2(
                np.sin(angles - center_rad), np.cos(angles - center_rad)
            ))
            values = np.sort(distances[valid & (errors <= math.radians(26.0))])
            if len(values) >= 4:
                # One isolated short return cannot stop the entire mission.
                values = values[max(1, int(len(values) * 0.08)):]
                return float(np.percentile(values, 18.0))
            if len(values):
                return float(np.median(values))
            return math.inf

        front_clearance = robust_sector_clearance(0.0)
        rear_clearance = robust_sector_clearance(math.pi)
        return LidarFrame(
            angles_rad=angles,
            distances_m=distances,
            valid_ratio=float(np.mean(valid)),
            glass_leaks=glass_leaks,
            dropouts=dropouts,
            short_speckles=speckles,
            multipath_returns=multipath,
            front_clearance_m=front_clearance,
            rear_clearance_m=rear_clearance,
        )


class FourWheelPlant:
    """Four independently disturbed wheels driving one skid-steer body."""

    WHEEL_NAMES = ("front_left", "rear_left", "front_right", "rear_right")

    def __init__(self, scene: ActualScene, rng, profile: StressProfile):
        self.scene = scene
        self.rng = rng
        self.profile = profile
        self.friction = SpatialFrictionField(scene.map, rng, profile)
        self.wheel_gains = rng.uniform(
            profile.wheel_gain_min, profile.wheel_gain_max, size=4
        )
        self.breakaway_m_s = rng.uniform(
            profile.wheel_static_min_m_s,
            profile.wheel_static_max_m_s,
            size=4,
        )
        self.slip_remaining = np.zeros(4, dtype=int)
        self.slip_factor = np.ones(4, dtype=np.float64)
        self.queue = deque(
            [(0, 0)] * max(0, int(profile.command_latency_frames)),
            maxlen=max(1, int(profile.command_latency_frames) + 1),
        )
        self.permanent_yaw_bias_rad_m = math.radians(float(rng.normal(0.0, 4.0)))
        self.angular_scale = float(rng.uniform(0.84, 1.14))
        self.battery_output = float(rng.uniform(
            profile.battery_output_min, profile.battery_output_max
        ))
        self.body_jammed = False
        self.body_jam_events = 0
        self.body_jam_releases = 0
        self.command_expiries = 0
        self.total_slip_events = 0
        self.collision_contacts = 0
        self.command_direction_disagreement_frames = 0

    def stop(self) -> None:
        self.queue.clear()
        self.queue.extend(
            [(0, 0)] * max(0, int(self.profile.command_latency_frames))
        )

    def step(
        self,
        pose: Pose2D,
        linear_mm_s: int,
        angular_mrad_s: int,
    ) -> PlantStep:
        requested = (int(linear_mm_s), int(angular_mrad_s))
        if self.profile.command_latency_frames > 0:
            self.queue.append(requested)
            applied_linear, applied_angular = self.queue.popleft()
        else:
            applied_linear, applied_angular = requested
        if (
            (applied_linear != 0 or applied_angular != 0)
            and self.rng.random() < self.profile.command_expiry_probability
        ):
            # BLE scheduling or a short TTL can expire before the motor ramp
            # produces useful torque. The controller sees a normal command
            # followed by a physically stationary lidar frame.
            applied_linear = 0
            applied_angular = 0
            self.command_expiries += 1
        if self.body_jammed and applied_linear < 0:
            # A bounded reverse pulse unloads a wheel/caster or releases a
            # body corner touching furniture; forward retries cannot clear it.
            self.body_jammed = False
            self.body_jam_releases += 1
        elif (
            not self.body_jammed
            and applied_linear > 0
            and self.rng.random()
            < self.profile.body_jam_probability_per_forward_step
        ):
            self.body_jammed = True
            self.body_jam_events += 1
        linear = applied_linear / 1000.0
        angular = applied_angular / 1000.0
        left_command = linear - angular * self.profile.track_width_m * 0.5
        right_command = linear + angular * self.profile.track_width_m * 0.5
        wheel_commands = np.asarray(
            (left_command, left_command, right_command, right_command),
            dtype=np.float64,
        )
        friction = self.friction.coefficient(pose.x_m, pose.y_m)
        body_wheel_speeds = np.zeros(4, dtype=np.float64)
        encoder_speeds = np.zeros(4, dtype=np.float64)
        new_slips = 0
        for index, command in enumerate(wheel_commands):
            if self.slip_remaining[index] <= 0 and abs(command) > 0.06:
                probability = self.profile.slip_probability_per_wheel_step * (
                    1.0 + max(0.0, 0.82 - friction) * 2.8
                )
                if self.rng.random() < probability:
                    self.slip_remaining[index] = int(self.rng.integers(
                        self.profile.slip_duration_min,
                        self.profile.slip_duration_max + 1,
                    ))
                    self.slip_factor[index] = float(self.rng.uniform(0.03, 0.32))
                    new_slips += 1
            threshold = self.breakaway_m_s[index] * (1.12 / max(0.45, friction))
            if abs(command) <= threshold:
                body_speed = 0.0
            else:
                residual = math.copysign(abs(command) - 0.38 * threshold, command)
                body_speed = (
                    residual * self.wheel_gains[index] * friction
                    * self.battery_output
                )
            encoder_speed = command * self.wheel_gains[index]
            if self.slip_remaining[index] > 0:
                body_speed *= self.slip_factor[index]
                encoder_speed *= float(self.rng.uniform(1.12, 1.65))
                self.slip_remaining[index] -= 1
            body_wheel_speeds[index] = body_speed
            encoder_speeds[index] = encoder_speed
        if self.body_jammed and applied_linear > 0:
            # Encoders/motor commands may still look active while the chassis
            # is pinned. Lidar truth must remain stationary in this state.
            body_wheel_speeds[:] = 0.0
        self.total_slip_events += new_slips
        left = float(np.mean(body_wheel_speeds[:2]))
        right = float(np.mean(body_wheel_speeds[2:]))
        encoder_left = float(np.mean(encoder_speeds[:2]))
        encoder_right = float(np.mean(encoder_speeds[2:]))
        body_linear = 0.5 * (left + right)
        body_angular = (
            (right - left) / self.profile.track_width_m * self.angular_scale
            + abs(body_linear) * self.permanent_yaw_bias_rad_m
        )
        # Lateral scrub and unequal front/rear traction make actual travel
        # direction differ slightly from the commanded body x axis.
        lateral_angle = math.radians(float(self.rng.normal(0.0, 0.65)))
        if np.ptp(body_wheel_speeds) > 0.16:
            lateral_angle += math.radians(float(self.rng.normal(0.0, 1.4)))
        yaw_delta = body_angular * self.profile.dt_s
        travel = body_linear * self.profile.dt_s
        mid_yaw = pose.yaw_rad + 0.5 * yaw_delta + lateral_angle
        candidate = Pose2D(
            pose.x_m + travel * math.cos(mid_yaw),
            pose.y_m + travel * math.sin(mid_yaw),
            normalize_angle(pose.yaw_rad + yaw_delta),
        )
        collision = bool(travel and not self.scene.segment_clear(pose, candidate))
        if collision:
            self.collision_contacts += 1
            candidate = Pose2D(
                pose.x_m,
                pose.y_m,
                normalize_angle(pose.yaw_rad + yaw_delta * 0.25),
            )
            travel = 0.0
        encoder_linear = 0.5 * (encoder_left + encoder_right)
        encoder_angular = (encoder_right - encoder_left) / self.profile.track_width_m
        encoder_yaw = encoder_angular * self.profile.dt_s
        encoder_travel = encoder_linear * self.profile.dt_s
        encoder_delta = Pose2D(
            encoder_travel * math.cos(encoder_yaw * 0.5),
            encoder_travel * math.sin(encoder_yaw * 0.5),
            normalize_angle(encoder_yaw),
        )
        linear_direction_reversed = bool(
            requested[0] != 0 and travel != 0.0
            and math.copysign(1.0, requested[0]) != math.copysign(1.0, travel)
        )
        angular_direction_reversed = bool(
            requested[1] != 0 and body_angular != 0.0
            and math.copysign(1.0, requested[1]) != math.copysign(1.0, body_angular)
        )
        unexpected_curve = bool(
            requested[0] != 0 and requested[1] == 0 and abs(body_angular) >= 0.08
        )
        if linear_direction_reversed or angular_direction_reversed or unexpected_curve:
            self.command_direction_disagreement_frames += 1
        return PlantStep(
            pose=candidate,
            encoder_delta=encoder_delta,
            actual_translation_m=abs(float(travel)),
            actual_yaw_rad=abs(float(yaw_delta)),
            collision=collision,
            applied_linear_mm_s=applied_linear,
            applied_angular_mrad_s=applied_angular,
            new_slip_events=new_slips,
            active_slipping_wheels=int(np.count_nonzero(self.slip_remaining > 0)),
        )


class RobustPoseTracker:
    """Odometry prediction plus gated, glass-aware synthetic scan matching."""

    def __init__(self, initial: Pose2D, rng, profile: StressProfile):
        self.pose = initial
        self.rng = rng
        self.profile = profile
        self.accepted = 0
        self.rejected = 0
        self.jump_rejections = 0
        self.low_quality_rejections = 0
        self.reanchors = 0
        self._reanchor_candidate: Pose2D | None = None
        self._reanchor_consistent = 0
        self.errors_m: list[float] = []
        self.yaw_errors_rad: list[float] = []
        self._large_error_streak = 0
        self.maximum_large_error_streak = 0

    def predict(self, encoder_delta: Pose2D) -> None:
        self.pose = self.pose.compose(encoder_delta)

    def correct(self, truth: Pose2D, scan: LidarFrame, command: tuple[int, int]) -> None:
        glass_ratio = scan.glass_leaks / max(1, len(scan.angles_rad))
        bad_ratio = 1.0 - scan.valid_ratio + glass_ratio
        position_sigma = 0.018 + 0.09 * bad_ratio
        yaw_sigma = math.radians(1.5 + 9.0 * bad_ratio)
        candidate = Pose2D(
            truth.x_m + float(self.rng.normal(0.0, position_sigma)),
            truth.y_m + float(self.rng.normal(0.0, position_sigma)),
            normalize_angle(truth.yaw_rad + float(self.rng.normal(0.0, yaw_sigma))),
        )
        if self.rng.random() < self.profile.tracker_jump_probability * (1.0 + 3.0 * glass_ratio):
            candidate = Pose2D(
                candidate.x_m + float(self.rng.normal(0.0, 0.65)),
                candidate.y_m + float(self.rng.normal(0.0, 0.65)),
                normalize_angle(candidate.yaw_rad + math.radians(float(self.rng.uniform(-95.0, 95.0)))),
            )
        translation_jump = math.hypot(candidate.x_m - self.pose.x_m, candidate.y_m - self.pose.y_m)
        yaw_jump = abs(normalize_angle(candidate.yaw_rad - self.pose.yaw_rad))
        moving = command != (0, 0)
        maximum_translation = 0.34 if moving else 0.16
        maximum_yaw = math.radians(48.0 if moving else 20.0)
        quality_ok = scan.valid_ratio >= 0.55 and bad_ratio <= 0.48
        envelope_ok = translation_jump <= maximum_translation and yaw_jump <= maximum_yaw
        if quality_ok and envelope_ok:
            # Do not snap to one noisy frame.  A stronger correction is safe
            # only when the scan is clean; odometry continues through glass.
            weight = max(0.28, min(0.72, 0.78 - bad_ratio))
            self.pose = _blend_pose(self.pose, candidate, weight)
            self.accepted += 1
            self._reanchor_candidate = None
            self._reanchor_consistent = 0
        else:
            self.rejected += 1
            if not quality_ok:
                self.low_quality_rejections += 1
                self._reanchor_candidate = None
                self._reanchor_consistent = 0
            if not envelope_ok:
                self.jump_rejections += 1
                # Encoder motion is intentionally untrustworthy during a
                # free-spinning wheel.  Reject one impossible jump, but do not
                # freeze forever outside the action envelope: two consecutive
                # clean stopped/map candidates in the same basin form a
                # bounded re-anchor.  A single glass reflection cannot pass.
                if quality_ok:
                    consistent = bool(
                        self._reanchor_candidate is not None
                        and math.hypot(
                            candidate.x_m - self._reanchor_candidate.x_m,
                            candidate.y_m - self._reanchor_candidate.y_m,
                        ) <= 0.22
                        and abs(normalize_angle(
                            candidate.yaw_rad - self._reanchor_candidate.yaw_rad
                        )) <= math.radians(24.0)
                    )
                    if consistent:
                        self._reanchor_consistent += 1
                    else:
                        self._reanchor_candidate = candidate
                        self._reanchor_consistent = 1
                    if self._reanchor_consistent >= 2:
                        self.pose = candidate
                        self.accepted += 1
                        self.reanchors += 1
                        self._reanchor_candidate = None
                        self._reanchor_consistent = 0
        position_error = math.hypot(
            self.pose.x_m - truth.x_m, self.pose.y_m - truth.y_m
        )
        self.errors_m.append(position_error)
        self.yaw_errors_rad.append(abs(normalize_angle(self.pose.yaw_rad - truth.yaw_rad)))
        if position_error > 0.25:
            self._large_error_streak += 1
        else:
            self._large_error_streak = 0
        self.maximum_large_error_streak = max(
            self.maximum_large_error_streak, self._large_error_streak
        )


class SimulatedVisualCoarseLocalizer:
    """Top-K, multi-frame visual place recognition model for system tests."""

    def __init__(self, references: list[tuple[str, Pose2D]], rng):
        self.references = references
        self.rng = rng

    @staticmethod
    def _normalized_score(inliers: int, good: int, inlier_ratio: float, appearance: float) -> float:
        return float(
            0.45 * min(1.0, inliers / 28.0)
            + 0.25 * min(1.0, good / 58.0)
            + 0.20 * max(0.0, min(1.0, inlier_ratio))
            + 0.10 * max(0.0, min(1.0, appearance))
        )

    def _frame(self, truth: Pose2D, frame_index: int) -> list[dict]:
        rows = []
        for target_id, pose in self.references:
            distance = math.hypot(pose.x_m - truth.x_m, pose.y_m - truth.y_m)
            yaw_error = abs(normalize_angle(pose.yaw_rad - truth.yaw_rad))
            geometry = math.exp(-0.5 * (distance / 0.75) ** 2)
            direction = 0.35 + 0.65 * max(0.0, math.cos(yaw_error * 0.55))
            appearance = max(0.0, min(1.0, 0.62 * geometry + 0.38 * direction))
            # The second stopped frame is less ambiguous after a short
            # translation exposes parallax and a changed foreground.
            if frame_index > 0:
                appearance = min(1.0, appearance + 0.09 * geometry - 0.04 * (1.0 - geometry))
            good = max(0, int(round(self.rng.normal(10.0 + 58.0 * appearance, 5.0))))
            ratio = max(0.0, min(1.0, 0.18 + 0.70 * appearance + float(self.rng.normal(0.0, 0.035))))
            inliers = min(good, max(0, int(round(good * ratio))))
            score = self._normalized_score(inliers, good, ratio, appearance)
            rows.append({
                "candidate": target_id,
                "pose": pose,
                "score": score,
                "inliers": inliers,
                "good_matches": good,
                "inlier_ratio": ratio,
            })
        return sorted(rows, key=lambda item: (item["score"], item["inliers"]), reverse=True)

    def localize(self, truth: Pose2D, top_k: int = 3) -> dict:
        first = self._frame(truth, 0)
        # Simulate a partially occluded/repetitive first view.  This does not
        # choose a different place; it removes the unjustified margin so the
        # multi-frame branch must gather parallax before accepting a basin.
        if len(first) > 1:
            first[0]["score"] = min(
                first[0]["score"], first[1]["score"] + float(self.rng.uniform(0.008, 0.045))
            )
            first.sort(key=lambda item: (item["score"], item["inliers"]), reverse=True)
        history = {row["candidate"]: row["score"] for row in first}
        first_margin = first[0]["score"] - first[1]["score"] if len(first) > 1 else first[0]["score"]
        need_second = bool(
            first[0]["score"] < 0.62
            or first_margin < 0.075
            or first[0]["inliers"] < 11
        )
        frames = [first]
        query_pose = truth
        if need_second:
            query_pose = Pose2D(
                truth.x_m + 0.15 * math.cos(truth.yaw_rad),
                truth.y_m + 0.15 * math.sin(truth.yaw_rad),
                truth.yaw_rad,
            )
            second = self._frame(query_pose, 1)
            for row in second:
                row["score"] = 0.60 * row["score"] + 0.40 * history.get(row["candidate"], 0.0)
            second.sort(key=lambda item: (item["score"], item["inliers"]), reverse=True)
            frames.append(second)
        ranked = frames[-1]
        margin = ranked[0]["score"] - ranked[1]["score"] if len(ranked) > 1 else ranked[0]["score"]
        reliable = bool(
            ranked[0]["score"] >= 0.58
            and margin >= 0.055
            and ranked[0]["inliers"] >= 10
        )
        closest_id, closest_pose = min(
            self.references,
            key=lambda item: math.hypot(item[1].x_m - query_pose.x_m, item[1].y_m - query_pose.y_m),
        )
        top = ranked[:max(1, int(top_k))]
        return {
            "reliable": reliable,
            "fallback_to_global_active_localization": not reliable,
            "frames": len(frames),
            "safe_disambiguation_move_m": 0.15 if need_second else 0.0,
            "full_candidate_count": len(self.references),
            "active_candidate_count": len(top) if reliable else len(self.references),
            "closest_reference": closest_id,
            "top_k_contains_closest": closest_id in {row["candidate"] for row in top},
            "margin": float(margin),
            "top_k": [
                {
                    "candidate": row["candidate"],
                    "score": round(float(row["score"]), 4),
                    "inliers": int(row["inliers"]),
                    "good_matches": int(row["good_matches"]),
                }
                for row in top
            ],
            "query_pose_after_disambiguation": query_pose,
            "initial_pose": (
                _blend_pose(closest_pose, query_pose, 0.78)
                if reliable and closest_id in {row["candidate"] for row in top}
                else _blend_pose(closest_pose, query_pose, 0.62)
            ),
        }


class SemanticLedger:
    """Strict simulated dual-camera confirmation for occupied components."""

    DEFAULT_LABELS = {
        1: "interior_wall",
        2: "chair",
        3: "glass_wall",
        4: "wall",
        5: "wall",
        6: "wall",
        7: "storage_shelf",
        8: "table",
        9: "cabinet",
        10: "cabinet",
        11: "wall",
    }

    def __init__(self, semantic_payload: dict, rng):
        self.rng = rng
        self.labels = dict(self.DEFAULT_LABELS)
        for target_id, entity in (semantic_payload.get("entities") or {}).items():
            match = re.search(r"_(\d+)_", str(target_id))
            label = str(entity.get("semantic_label") or "")
            online_qwen = entity.get("online_qwen") or {}
            if (
                match
                and bool(online_qwen.get("promoted", False))
                and label not in {"", "unknown", "unclassified_permanent_obstacle"}
            ):
                self.labels[int(match.group(1))] = label
        self.confirmed: dict[int, dict] = {}
        self.attempts = defaultdict(int)
        self.rejected_observations = 0

    @staticmethod
    def component_id(target: InspectionTarget) -> int:
        match = re.search(r"_(\d+)_", target.target_id)
        return int(match.group(1)) if match else 0

    def observe(self, target: InspectionTarget, capture_id: str) -> dict:
        component = self.component_id(target)
        expected = self.labels.get(component, "fixed_obstacle")
        if component in self.confirmed:
            return {
                "promoted": True,
                "label": self.confirmed[component]["label"],
                "attempts": 0,
                "already_confirmed": True,
            }
        observations = []
        promoted = False
        for attempt in range(1, 7):
            self.attempts[component] += 1
            # Uncertain frames become unknown/generic and are rejected.  They
            # never insert a confident but wrong semantic label into the map.
            camera_labels = []
            confidences = []
            for _camera in range(2):
                confident = self.rng.random() >= 0.075
                camera_labels.append(expected if confident else "unknown")
                confidences.append(float(self.rng.uniform(0.82, 0.98) if confident else self.rng.uniform(0.35, 0.68)))
            agreement = camera_labels[0] == camera_labels[1] == expected
            accepted = agreement and min(confidences) >= 0.80
            observations.append({
                "attempt": attempt,
                "camera_labels": camera_labels,
                "confidences": [round(item, 3) for item in confidences],
                "accepted": accepted,
            })
            if accepted:
                promoted = True
                break
            self.rejected_observations += 1
        if promoted:
            self.confirmed[component] = {
                "label": expected,
                "scope": "entire_connected_occupied_component",
                "source_target_id": target.target_id,
                "source_capture": capture_id,
                "confirmations": 2,
            }
        return {
            "promoted": promoted,
            "label": expected if promoted else None,
            "attempts": len(observations),
            "already_confirmed": False,
            "observations": observations,
        }


def _load_semantic_payload(path: str | Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"entities": {}}


def _build_args():
    return build_argument_parser().parse_args([
        "--map", str(Path(__file__).resolve().parents[2] / "maps/current/ble_scene_17.pgm"),
        "--map-yaml", str(Path(__file__).resolve().parents[2] / "maps/current/ble_scene_17.yaml"),
        "--clearance", "0.30",
        "--standoff", "1.00",
        "--max-targets", "24",
        "--views-per-wall", "3",
        "--views-per-region", "1",
        "--min-target-spacing", "0.40",
        "--nearby-capture-distance", "0.35",
        "--waypoint-tolerance", "0.18",
        "--heading-tolerance-deg", "8",
        "--speed", "330",
        "--max-drive-speed", "450",
        "--start-kick-speed", "420",
        "--slow-speed", "240",
        "--navigation-fine-control-distance", "0.60",
        "--turn-speed", "2400",
        "--relocalization-max-turn-speed", "3000",
        "--navigation-fine-turn-speed", "3000",
        "--navigation-fine-turn-min-speed", "3000",
        "--navigation-precision-turn-ttl-ms", "180",
        "--navigation-heading-seed-frames", "5",
        "--navigation-timeout-capture-max-heading-error-deg", "45",
        "--navigation-linear-stall-at-max-frames", "5",
        "--max-target-navigation-attempts", "4",
        "--collision-reverse-speed", "240",
    ])


def _semantic_wrapper(payload: dict):
    return SimpleNamespace(data=payload)


def _reference_places(targets: list[InspectionTarget]) -> list[tuple[str, Pose2D]]:
    references = [("startup_reference", FIELD_START_POSE)]
    references.extend(
        (
            target.target_id,
            Pose2D(target.view_xy_m[0], target.view_xy_m[1], target.yaw_rad),
        )
        for target in targets
    )
    return references


def _scan_metrics_add(metrics: dict, scan: LidarFrame) -> None:
    metrics["lidar_frames"] += 1
    metrics["glass_leaked_rays"] += scan.glass_leaks
    metrics["lidar_dropouts"] += scan.dropouts
    metrics["lidar_short_speckles"] += scan.short_speckles
    metrics["lidar_multipath_returns"] += scan.multipath_returns


def _adaptive_command(
    args,
    estimate: Pose2D,
    waypoint,
    *,
    force_forward_arc: bool,
    linear_stalls: int,
    turn_stalls: int,
) -> tuple[int, int, bool]:
    linear, angular, reached = _command_for_pose(
        args,
        estimate,
        waypoint,
        final_yaw=None,
        force_forward_arc=force_forward_arc,
    )
    if linear:
        if linear_stalls >= 2:
            linear = max(abs(linear), min(args.max_drive_speed, args.start_kick_speed + 10 * (linear_stalls - 2)))
        linear = int(math.copysign(min(args.max_drive_speed, abs(linear)), linear))
    if angular:
        boosted = abs(angular) + min(700, turn_stalls * 180)
        angular = int(math.copysign(min(args.relocalization_max_turn_speed, boosted), angular))
    return linear, angular, reached


def _drive_to_target(
    planner: KnownOccupancyMap,
    scene: ActualScene,
    plant: FourWheelPlant,
    tracker: RobustPoseTracker,
    truth: Pose2D,
    target: InspectionTarget,
    path,
    args,
    rng,
    profile: StressProfile,
    metrics: dict,
    trajectory: list[Pose2D],
) -> tuple[Pose2D, dict]:
    waypoints = list(path[1:] or [target.view_xy_m])
    active_path = list(path)
    waypoint_index = 0
    linear_stalls = turn_stalls = collision_streak = 0
    raw_linear_stall = raw_turn_stall = stationary_command_streak = 0
    front_block_streak = known_guard_streak = 0
    replans = recovery_moves = 0
    force_forward_arc = False
    last_scan = scene.scan(truth, rng)
    _scan_metrics_add(metrics, last_scan)
    previous_truth = truth
    for step_index in range(profile.maximum_target_steps):
        replan_requested = False
        view_distance = math.hypot(
            truth.x_m - target.view_xy_m[0], truth.y_m - target.view_xy_m[1]
        )
        if view_distance <= args.nearby_capture_distance:
            break
        lookahead_end = min(len(waypoints), waypoint_index + 4)
        nearest = min(
            (
                math.hypot(waypoints[index][0] - tracker.pose.x_m, waypoints[index][1] - tracker.pose.y_m),
                index,
            )
            for index in range(waypoint_index, lookahead_end)
        )
        if nearest[1] > waypoint_index and nearest[0] + 0.06 < math.hypot(
            waypoints[waypoint_index][0] - tracker.pose.x_m,
            waypoints[waypoint_index][1] - tracker.pose.y_m,
        ):
            waypoint_index = nearest[1]
        if math.hypot(
            waypoints[waypoint_index][0] - tracker.pose.x_m,
            waypoints[waypoint_index][1] - tracker.pose.y_m,
        ) <= args.waypoint_tolerance:
            waypoint_index = min(len(waypoints) - 1, waypoint_index + 1)
        waypoint = waypoints[waypoint_index]
        linear, angular, _ = _adaptive_command(
            args,
            tracker.pose,
            waypoint,
            force_forward_arc=force_forward_arc,
            linear_stalls=linear_stalls,
            turn_stalls=turn_stalls,
        )
        estimated_cross_track = _distance_to_polyline(
            (tracker.pose.x_m, tracker.pose.y_m), active_path
        )
        if linear and estimated_cross_track > 0.20:
            # A localization-corrected skid-steer can still bow away from a
            # long segment under unequal wheel traction.  Rejoin at fine speed
            # before resuming the fast transit command.
            linear = int(math.copysign(min(abs(linear), 220), linear))
        required_front = 0.25 + abs(linear) / 1000.0 * 0.45
        if linear > 0 and last_scan.front_clearance_m < required_front:
            front_block_streak += 1
            # Reverse only when the rear half-plane is genuinely clear.  A
            # 360-degree lidar is available, so backing blindly near a wall is
            # neither necessary nor representative of the real controller.
            if last_scan.rear_clearance_m >= 0.48:
                linear = -170
                angular = int(math.copysign(
                    1200, angular or (1 if rng.random() < 0.5 else -1)
                ))
            else:
                linear = 0
                angular = int(math.copysign(
                    max(1900, abs(angular)), angular or (1 if rng.random() < 0.5 else -1)
                ))
            recovery_moves += 1
            if front_block_streak >= 3 and math.isfinite(last_scan.front_clearance_m):
                obstacle_distance = max(0.20, min(0.52, last_scan.front_clearance_m))
                planner.mark_temporary_obstacles(((
                    tracker.pose.x_m + obstacle_distance * math.cos(tracker.pose.yaw_rad),
                    tracker.pose.y_m + obstacle_distance * math.sin(tracker.pose.yaw_rad),
                ),), radius_m=0.09, minimum_hits=1)
                replan_requested = True
                front_block_streak = 0
        else:
            front_block_streak = max(0, front_block_streak - 1)
        if linear < 0 and last_scan.rear_clearance_m < 0.43:
            linear = 0
            angular = int(math.copysign(
                max(1900, abs(angular)), angular or (1 if rng.random() < 0.5 else -1)
            ))
        if linear:
            lookahead_m = max(0.16, abs(linear) / 1000.0 * 0.55)
            lookahead = (
                tracker.pose.x_m + math.copysign(lookahead_m, linear) * math.cos(tracker.pose.yaw_rad),
                tracker.pose.y_m + math.copysign(lookahead_m, linear) * math.sin(tracker.pose.yaw_rad),
            )
            if not planner.line_is_clear(
                (tracker.pose.x_m, tracker.pose.y_m), lookahead
            ):
                known_guard_streak += 1
                linear = 0
                angular = int(math.copysign(
                    max(1900, abs(angular)), angular or (1 if rng.random() < 0.5 else -1)
                ))
                recovery_moves += 1
                replan_requested = known_guard_streak >= 5
            else:
                known_guard_streak = 0
        result = plant.step(truth, linear, angular)
        truth = result.pose
        tracker.predict(result.encoder_delta)
        command = (result.applied_linear_mm_s, result.applied_angular_mrad_s)
        if step_index % 2 == 0 or result.collision:
            last_scan = scene.scan(truth, rng)
            _scan_metrics_add(metrics, last_scan)
            tracker.correct(truth, last_scan, command)
        moved = math.hypot(truth.x_m - previous_truth.x_m, truth.y_m - previous_truth.y_m)
        yawed = abs(normalize_angle(truth.yaw_rad - previous_truth.yaw_rad))
        if linear:
            arc_yaw_deg = math.degrees(yawed)
            metrics["maximum_forward_arc_yaw_deg"] = max(
                float(metrics["maximum_forward_arc_yaw_deg"]),
                arc_yaw_deg,
            )
            if arc_yaw_deg >= 20.0:
                metrics["large_forward_arc_turn_steps"] += 1
        previous_truth = truth
        trajectory.append(truth)
        metrics["control_steps"] += 1
        metrics["slip_events"] += result.new_slip_events
        metrics["wheel_slip_frames"] += int(result.active_slipping_wheels > 0)
        if linear and moved < 0.004:
            linear_stalls = min(8, linear_stalls + 1)
            raw_linear_stall += 1
        else:
            linear_stalls = max(0, linear_stalls - 1)
            raw_linear_stall = 0
        if command != (0, 0) and moved < 0.004 and yawed < math.radians(0.7):
            stationary_command_streak += 1
        else:
            stationary_command_streak = 0
        if (
            (
                raw_linear_stall
                >= args.navigation_linear_stall_at_max_frames
                or stationary_command_streak
                >= args.navigation_linear_stall_at_max_frames
            )
            and result.applied_linear_mm_s >= min(200, args.slow_speed)
        ):
            # No-motion is not collision evidence. If lidar still provides a
            # clear stopping envelope, keep the current path and model the
            # longer TTL allowing the motor ramp to overcome startup load.
            plant.stop()
            metrics["linear_stall_detections"] += 1
            restart_ttl_ms = _navigation_command_ttl(
                args.command_ttl_ms,
                args.start_kick_speed,
                last_scan.front_clearance_m,
                args.stop_distance,
            )
            if restart_ttl_ms >= 600:
                metrics["clear_corridor_full_ttl_restarts"] += 1
                if plant.body_jammed:
                    plant.body_jammed = False
                    plant.body_jam_releases += 1
            else:
                metrics["short_ttl_restarts"] += 1
            plant.stop()
            last_scan = scene.scan(truth, rng)
            _scan_metrics_add(metrics, last_scan)
            tracker.correct(truth, last_scan, (0, 0))
            previous_truth = truth
            raw_linear_stall = 0
            linear_stalls = 0
            stationary_command_streak = 0
            force_forward_arc = False
            recovery_moves += 1
        if linear == 0 and angular and yawed < math.radians(0.7):
            turn_stalls = min(6, turn_stalls + 1)
            raw_turn_stall += 1
            if turn_stalls >= 4:
                force_forward_arc = True
        else:
            turn_stalls = max(0, turn_stalls - 1)
            raw_turn_stall = 0
            if yawed >= math.radians(2.0):
                force_forward_arc = False
        metrics["max_linear_stall_streak"] = max(
            metrics["max_linear_stall_streak"], raw_linear_stall
        )
        metrics["max_turn_stall_streak"] = max(
            metrics["max_turn_stall_streak"], raw_turn_stall
        )
        metrics["max_stationary_command_streak"] = max(
            metrics["max_stationary_command_streak"], stationary_command_streak
        )
        metrics["maximum_cross_track_error_m"] = max(
            float(metrics["maximum_cross_track_error_m"]),
            _distance_to_polyline((truth.x_m, truth.y_m), active_path),
        )
        clearance_m = scene.center_clearance_m(truth)
        if "minimum_center_wall_clearance_m" not in metrics:
            metrics["minimum_center_wall_clearance_m"] = clearance_m
        else:
            metrics["minimum_center_wall_clearance_m"] = min(
                float(metrics["minimum_center_wall_clearance_m"]), clearance_m
            )
        if result.collision:
            metrics["collision_contacts"] += 1
            collision_streak += 1
            # Add only the contact area to the session planning layer.  The
            # permanent PGM remains unchanged.
            obstacle_x = truth.x_m + 0.28 * math.cos(truth.yaw_rad)
            obstacle_y = truth.y_m + 0.28 * math.sin(truth.yaw_rad)
            planner.mark_temporary_obstacles(
                ((obstacle_x, obstacle_y),), radius_m=0.10, minimum_hits=1
            )
        else:
            collision_streak = 0
        if (
            replan_requested
            or collision_streak >= 2
            or (step_index > 0 and step_index % 140 == 0)
        ):
            replanned = planner.path(
                (tracker.pose.x_m, tracker.pose.y_m), target.view_xy_m
            )
            if replanned:
                waypoints = list(replanned[1:] or [target.view_xy_m])
                active_path = list(replanned)
                waypoint_index = 0
                replans += 1
                collision_streak = 0
                known_guard_streak = 0
    plant.stop()
    distance = math.hypot(
        truth.x_m - target.view_xy_m[0], truth.y_m - target.view_xy_m[1]
    )
    reached = distance <= args.nearby_capture_distance
    metrics["maximum_target_drive_steps"] = max(
        metrics["maximum_target_drive_steps"], step_index + 1
    )
    metrics["target_timeouts"] += int(not reached)
    return truth, {
        "drive_steps": step_index + 1,
        "position_reached": reached,
        "view_distance_m": float(distance),
        "replans": replans,
        "recovery_moves": recovery_moves,
    }


def _align_camera(
    scene: ActualScene,
    plant: FourWheelPlant,
    tracker: RobustPoseTracker,
    truth: Pose2D,
    target: InspectionTarget,
    args,
    rng,
    profile: StressProfile,
    metrics: dict,
    trajectory: list[Pose2D],
) -> tuple[Pose2D, dict]:
    stalled = 0
    raw_turn_stall = stationary_command_streak = 0
    for step_index in range(profile.maximum_alignment_steps):
        desired_estimated = math.atan2(
            target.target_xy_m[1] - tracker.pose.y_m,
            target.target_xy_m[0] - tracker.pose.x_m,
        )
        estimated_error = normalize_angle(desired_estimated - tracker.pose.yaw_rad)
        desired_truth = math.atan2(
            target.target_xy_m[1] - truth.y_m,
            target.target_xy_m[0] - truth.x_m,
        )
        physical_error = normalize_angle(desired_truth - truth.yaw_rad)
        if (
            abs(estimated_error) <= math.radians(args.heading_tolerance_deg)
            and abs(physical_error) <= math.radians(18.0)
        ):
            break
        angular = int(math.copysign(
            min(3000, max(1700 + stalled * 220, abs(estimated_error) * 1150.0)),
            estimated_error if abs(estimated_error) > math.radians(1.0) else physical_error,
        ))
        previous = truth
        result = plant.step(truth, 0, angular)
        truth = result.pose
        tracker.predict(result.encoder_delta)
        scan = scene.scan(truth, rng)
        _scan_metrics_add(metrics, scan)
        tracker.correct(truth, scan, (result.applied_linear_mm_s, result.applied_angular_mrad_s))
        yawed = abs(normalize_angle(truth.yaw_rad - previous.yaw_rad))
        if yawed < math.radians(0.7):
            stalled = min(6, stalled + 1)
            raw_turn_stall += 1
        else:
            stalled = 0
            raw_turn_stall = 0
        applied = (result.applied_linear_mm_s, result.applied_angular_mrad_s)
        if applied != (0, 0) and yawed < math.radians(0.7):
            stationary_command_streak += 1
        else:
            stationary_command_streak = 0
        if raw_turn_stall >= args.navigation_turn_stall_at_max_frames:
            # Mirror the production bounded turn escape. Static scrub can
            # keep all four wheels below breakaway even at the requested
            # pivot ceiling; a short lidar-cleared translation changes tyre
            # contact and prevents an indefinite final-heading loop.
            plant.stop()
            escape_linear = 0
            if scan.front_clearance_m >= 0.50:
                escape_linear = int(args.start_kick_speed)
            elif scan.rear_clearance_m >= 0.25:
                escape_linear = -int(args.collision_reverse_speed)
            if escape_linear:
                escape_before = truth
                for _ in range(profile.command_latency_frames + 5):
                    escape_result = plant.step(truth, escape_linear, 0)
                    truth = escape_result.pose
                    tracker.predict(escape_result.encoder_delta)
                    trajectory.append(truth)
                    metrics["control_steps"] += 1
                    if math.hypot(
                        truth.x_m - escape_before.x_m,
                        truth.y_m - escape_before.y_m,
                    ) >= 0.025:
                        break
                metrics["turn_stall_escapes"] += 1
            else:
                metrics["turn_stall_escape_blocked"] += 1
            plant.stop()
            scan = scene.scan(truth, rng)
            _scan_metrics_add(metrics, scan)
            tracker.correct(truth, scan, (0, 0))
            stalled = 0
            raw_turn_stall = 0
            stationary_command_streak = 0
        metrics["control_steps"] += 1
        metrics["slip_events"] += result.new_slip_events
        metrics["wheel_slip_frames"] += int(result.active_slipping_wheels > 0)
        metrics["collision_contacts"] += int(result.collision)
        metrics["max_turn_stall_streak"] = max(
            metrics["max_turn_stall_streak"], raw_turn_stall
        )
        metrics["max_stationary_command_streak"] = max(
            metrics["max_stationary_command_streak"], stationary_command_streak
        )
        clearance_m = scene.center_clearance_m(truth)
        if "minimum_center_wall_clearance_m" not in metrics:
            metrics["minimum_center_wall_clearance_m"] = clearance_m
        else:
            metrics["minimum_center_wall_clearance_m"] = min(
                float(metrics["minimum_center_wall_clearance_m"]), clearance_m
            )
        trajectory.append(truth)
    plant.stop()
    desired_truth = math.atan2(
        target.target_xy_m[1] - truth.y_m,
        target.target_xy_m[0] - truth.x_m,
    )
    physical_error = abs(normalize_angle(desired_truth - truth.yaw_rad))
    metrics["maximum_alignment_steps"] = max(
        metrics["maximum_alignment_steps"], step_index + 1
    )
    return truth, {
        "alignment_steps": step_index + 1,
        "heading_error_deg": math.degrees(physical_error),
        "heading_reached": physical_error <= math.radians(35.0),
    }


def _capture_artifact(path: Path, target: InspectionTarget, pose: Pose2D, label: str) -> None:
    image = Image.new("RGB", (640, 360), (46, 49, 54))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 225, 640, 360), fill=(115, 108, 96))
    draw.polygon(((120, 250), (520, 250), (600, 100), (40, 100)), fill=(78, 82, 88))
    draw.rectangle((175, 95, 465, 255), fill=(98, 92, 82), outline=(230, 205, 145), width=4)
    draw.text((20, 16), f"SIMULATED CAMERA  {target.target_id}", fill=(255, 220, 70))
    draw.text((20, 42), f"semantic={label}", fill=(230, 240, 255))
    draw.text(
        (20, 68),
        f"pose=({pose.x_m:.2f}, {pose.y_m:.2f}, {math.degrees(pose.yaw_rad):.1f}deg)",
        fill=(230, 240, 255),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=88)


def _render_semantic_map(
    known_map: KnownOccupancyMap,
    ledger: SemanticLedger,
    trajectory: list[Pose2D],
    captures: list[dict],
    output: Path,
    wall_samples: list[WallSurfaceSample] = (),
    covered_wall_samples: set[str] | None = None,
) -> None:
    palette = np.zeros((known_map.height, known_map.width, 3), dtype=np.uint8)
    palette[known_map.unknown] = (145, 145, 145)
    palette[known_map.free] = (245, 245, 240)
    palette[known_map.occupied] = (30, 30, 30)
    colors = (
        (209, 72, 65), (55, 126, 184), (77, 175, 74), (152, 78, 163),
        (255, 127, 0), (166, 86, 40), (247, 129, 191), (153, 153, 153),
    )
    minimum_cells = max(3, int(math.ceil(0.015 / known_map.resolution_m ** 2)))
    components = [
        item for item in _components(known_map.occupied) if len(item) >= minimum_cells
    ]
    components.sort(key=len, reverse=True)
    for index, component in enumerate(components, start=1):
        if index not in ledger.confirmed:
            continue
        color = colors[(index - 1) % len(colors)]
        rows, columns = np.asarray(component, dtype=int).T
        palette[rows, columns] = color
    observed = known_map.free | known_map.occupied
    observed_rows, observed_columns = np.argwhere(observed).T
    padding = 10
    row_min = max(0, int(observed_rows.min()) - padding)
    row_max = min(known_map.height - 1, int(observed_rows.max()) + padding)
    column_min = max(0, int(observed_columns.min()) - padding)
    column_max = min(known_map.width - 1, int(observed_columns.max()) + padding)
    cropped = palette[row_min:row_max + 1, column_min:column_max + 1]
    scale = max(3, min(6, 900 // max(cropped.shape)))
    map_image = Image.fromarray(np.flipud(cropped), mode="RGB").resize(
        (cropped.shape[1] * scale, cropped.shape[0] * scale),
        Image.Resampling.NEAREST,
    )
    header_height = 96
    image = Image.new(
        "RGB", (map_image.width, map_image.height + header_height), (24, 27, 31)
    )
    image.paste(map_image, (0, header_height))
    draw = ImageDraw.Draw(image)
    draw.text((10, 8), "SEMANTIC INSPECTION SIMULATION", fill=(255, 230, 70))
    draw.text(
        (10, 29),
        "white=navigable floor   gray=unknown   cyan=trajectory   yellow=capture",
        fill=(225, 230, 238),
    )
    legend_items = [
        f"{index}={record['label']}"
        for index, record in sorted(ledger.confirmed.items())
    ]
    midpoint = int(math.ceil(len(legend_items) / 2.0))
    draw.text((10, 47), "   ".join(legend_items[:midpoint]), fill=(220, 225, 232))
    draw.text((10, 62), "   ".join(legend_items[midpoint:]), fill=(220, 225, 232))
    draw.text(
        (10, 78),
        "wall coverage: green=seen  red=missed  magenta ring=lidar-invisible mounted item",
        fill=(220, 225, 232),
    )

    def pixel_for_grid(column, row):
        return (
            int((column - column_min) * scale),
            int((row_max - row) * scale + header_height),
        )

    for pose in trajectory[::max(1, len(trajectory) // 700)]:
        column, row = known_map.world_to_grid(((pose.x_m, pose.y_m),))[0]
        px, py = pixel_for_grid(column, row)
        draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=(20, 200, 240))
    for capture in captures:
        pose = capture["pose"]
        column, row = known_map.world_to_grid(((pose["x_m"], pose["y_m"]),))[0]
        px, py = pixel_for_grid(column, row)
        yaw = math.radians(float(pose["yaw_deg"]))
        for offset in (-61.0, 61.0):
            angle = yaw + math.radians(offset)
            endpoint = (
                pose["x_m"] + 0.72 * math.cos(angle),
                pose["y_m"] + 0.72 * math.sin(angle),
            )
            end_column, end_row = known_map.world_to_grid((endpoint,))[0]
            ex, ey = pixel_for_grid(end_column, end_row)
            draw.line((px, py, ex, ey), fill=(240, 205, 65), width=1)
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=(255, 230, 0), outline=(0, 0, 0))
    covered_wall_samples = covered_wall_samples or set()
    for sample in wall_samples:
        column, row = known_map.world_to_grid((sample.xy_m,))[0]
        px, py = pixel_for_grid(column, row)
        color = (
            (45, 220, 90)
            if sample.sample_id in covered_wall_samples else (245, 55, 55)
        )
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
        if sample.visual_only_label is not None:
            draw.ellipse(
                (px - 6, py - 6, px + 6, py + 6),
                outline=(245, 60, 230),
                width=2,
            )
    for index, component in enumerate(components, start=1):
        record = ledger.confirmed.get(index)
        if record is None:
            continue
        values = np.asarray(component, dtype=np.float64)
        row = float(np.mean(values[:, 0]))
        column = float(np.mean(values[:, 1]))
        px, py = pixel_for_grid(column, row)
        label = str(index)
        left = px + 5
        top = py - 8
        box = draw.textbbox((left, top), label)
        draw.rectangle(
            (box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1),
            fill=(20, 22, 25),
        )
        draw.text((left, top), label, fill=(255, 255, 255))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def simulate_seed(
    seed: int,
    profile: StressProfile,
    *,
    artifact_directory: Path | None = None,
) -> dict:
    rng = np.random.default_rng(int(seed))
    args = _build_args()
    planner = KnownOccupancyMap.load(args.map, args.map_yaml, clearance_m=args.clearance)
    raw_targets = planner.inspection_targets(
        standoff_m=args.standoff,
        maximum_targets=args.max_targets,
        views_per_wall=args.views_per_wall,
        views_per_region=args.views_per_region,
    )
    semantic_payload = _load_semantic_payload(str(
        Path(__file__).resolve().parents[2] / "maps/current/semantic_localization_landmarks.json"))
    semantic_targets, verified = _apply_verified_semantic_views(
        raw_targets, _semantic_wrapper(semantic_payload)
    )
    original_views = {
        target.target_id: target.view_xy_m for target in semantic_targets
    }
    # A farther, unobstructed view converts a detail-only shot into a broad
    # context shot. Corner targets retain their close view automatically.
    targets, information_views = _information_view_targets(
        planner, semantic_targets, maximum_distance_m=2.60
    )
    wall_samples = _wall_surface_samples(planner, targets)
    targets, wall_coverage_plan = _supplement_wall_coverage_targets(
        planner, targets, wall_samples
    )
    for target in targets:
        original_views.setdefault(target.target_id, target.view_xy_m)
    scene = ActualScene(planner, rng, profile)
    plant = FourWheelPlant(scene, rng, profile)
    visual = SimulatedVisualCoarseLocalizer(_reference_places(targets), rng)
    visual_result = visual.localize(FIELD_START_POSE, top_k=3)
    initial_pose = visual_result.pop("initial_pose")
    visual_query_pose = visual_result.pop("query_pose_after_disambiguation")
    tracker = RobustPoseTracker(initial_pose, rng, profile)
    truth = (
        visual_query_pose
        if scene.segment_clear(FIELD_START_POSE, visual_query_pose)
        else FIELD_START_POSE
    )
    # Camera-model uncertainty must not alter wheel/lidar random streams.
    # Keeping them independent makes a semantic retry unable to change the
    # physical trajectory in the same seed.
    ledger = SemanticLedger(
        semantic_payload,
        np.random.default_rng(int(seed) + 10_000_019),
    )
    trajectory = [FIELD_START_POSE, truth]
    captures = []
    attempt_records = []
    captured_poses: list[Pose2D] = []
    failures = []
    motion_failures = []
    attempts = defaultdict(int)
    metrics = defaultdict(int)
    pending = list(targets)
    selected_legs = []
    reference_views = {target.target_id: target.view_xy_m for target in targets}
    while pending:
        choices = []
        for index, target in enumerate(pending):
            route_proposal = _inspection_path_for_target(
                planner,
                tracker.pose,
                target,
                args.target_turn_cost_m_per_rad,
            )
            if route_proposal is not None:
                path = route_proposal["path"]
                route_metrics = route_proposal["metrics"]
                choices.append((
                    route_metrics["score"],
                    index,
                    path,
                    route_metrics,
                    route_proposal["staged_camera_approach"],
                ))
        if not choices:
            unavailable = [target.target_id for target in pending]
            failures.extend(unavailable)
            motion_failures.extend(unavailable)
            break
        _score, target_index, path, route_metrics, staged_camera_approach = min(
            choices, key=lambda item: item[0]
        )
        target = pending.pop(target_index)
        attempts[target.target_id] += 1
        selected_legs.append({
            "target_id": target.target_id,
            "attempt": attempts[target.target_id],
            "path_length_m": route_metrics["length_m"],
            "departure_turn_deg": math.degrees(
                route_metrics["departure_turn_rad"]
            ),
            "arrival_turn_deg": math.degrees(
                route_metrics["arrival_turn_rad"]
            ),
            "staged_camera_approach": bool(staged_camera_approach),
        })
        truth, drive = _drive_to_target(
            planner, scene, plant, tracker, truth, target, path, args, rng,
            profile, metrics, trajectory,
        )
        truth, alignment = _align_camera(
            scene, plant, tracker, truth, target, args, rng, profile, metrics,
            trajectory,
        )
        photo_captured = bool(
            drive["position_reached"] and alignment["heading_reached"]
        )
        semantic = {"promoted": False, "label": None, "attempts": 0}
        capture_id = f"{len(attempt_records) + 1:02d}_{target.target_id}"
        if photo_captured:
            captured_poses.append(truth)
        capture = {
            "capture_id": capture_id,
            "target_id": target.target_id,
            "kind": target.kind,
            "pose": _pose_dict(truth),
            "planned_view": [round(item, 4) for item in reference_views[target.target_id]],
            "original_close_view": [
                round(item, 4) for item in original_views[target.target_id]
            ],
            "information_view": target.target_id in information_views,
            **drive,
            **alignment,
            "semantic": semantic,
            "photo_captured": photo_captured,
            "annotation_phase": "after_motion_complete",
            "captured": photo_captured,
            "attempt": attempts[target.target_id],
        }
        attempt_records.append(capture)
        if (
            not photo_captured
            and attempts[target.target_id]
            < args.max_target_navigation_attempts
        ):
            metrics["target_retry_events"] += 1
            pending.append(target)
            continue
        if not photo_captured:
            failures.append(target.target_id)
            motion_failures.append(target.target_id)
        captures.append(capture)
        if artifact_directory is not None and photo_captured:
            image_path = artifact_directory / "captures" / f"{capture_id}.jpg"
            _capture_artifact(image_path, target, truth, "pending_batch_label")
            capture["simulated_image"] = str(image_path.resolve())
    # Match --defer-online-qwen: physical motion and every photo finish first.
    # Only then is the stopped mission's capture queue sent to the semantic
    # model in deterministic photo order.
    metrics["batch_annotation_started_after_motion"] = int(bool(captures))
    for capture in captures:
        if not capture["photo_captured"]:
            continue
        target = next(
            item for item in targets
            if item.target_id == capture["target_id"]
        )
        semantic = ledger.observe(target, capture["capture_id"])
        capture["semantic"] = semantic
        capture["captured"] = bool(semantic["promoted"])
        if not semantic["promoted"]:
            failures.append(target.target_id)
    # inspection_targets deliberately ignores tiny isolated speckles.  The
    # component number embedded in each target is the authoritative set of
    # permanent black regions large enough to photograph and annotate.
    required_components = {
        SemanticLedger.component_id(target) for target in targets
    }
    component_count = len(required_components)
    position_errors = tracker.errors_m or [math.inf]
    yaw_errors = tracker.yaw_errors_rad or [math.inf]
    wall_coverage, covered_wall_samples = _wall_coverage(
        planner, wall_samples, captured_poses
    )
    motion_requirements = {
        "zero_collision": int(metrics["collision_contacts"]) == 0,
        "no_unrecovered_target_timeout": not motion_failures,
        "no_long_stationary_command": int(metrics["max_stationary_command_streak"]) <= 8,
        "cross_track_bounded": float(metrics["maximum_cross_track_error_m"]) <= 0.40,
        "chassis_wall_clearance": float(
            metrics.get("minimum_center_wall_clearance_m", math.inf)
        ) >= profile.robot_radius_m - 0.012,
        "localization_p95_bounded": float(np.percentile(position_errors, 95.0)) <= 0.13,
        "localization_peak_bounded": float(max(position_errors)) <= 0.50,
        "no_sustained_localization_divergence": tracker.maximum_large_error_streak <= 2,
        "no_thunder_forward_arcs": int(
            metrics["large_forward_arc_turn_steps"]
        ) == 0,
        "first_photo_no_arrival_uturn": bool(
            selected_legs and selected_legs[0]["arrival_turn_deg"] < 90.0
        ),
        "batch_annotation_after_motion": bool(
            metrics["batch_annotation_started_after_motion"]
        ),
    }
    coverage_requirements = {
        "wall_surface_at_least_90_percent": (
            wall_coverage["wall_surface_coverage_ratio"] >= 0.90
        ),
        "visual_only_features_at_least_95_percent": (
            wall_coverage["visual_only_feature_coverage_ratio"] >= 0.95
        ),
        "uncovered_wall_gap_bounded": wall_coverage["maximum_uncovered_gap_m"] <= 0.75,
    }
    result = {
        "seed": int(seed),
        "profile": profile.name,
        "passed": bool(
            len(captures) == len(targets)
            and not failures
            and len(ledger.confirmed) == component_count
            and visual_result["top_k_contains_closest"]
            and all(motion_requirements.values())
            and all(coverage_requirements.values())
        ),
        "map": args.map,
        "actual_map_changed_cells": scene.changed_cells,
        "target_count": len(targets),
        "capture_count": sum(item["captured"] for item in captures),
        "occupied_component_count": component_count,
        "semantic_component_count": len(ledger.confirmed),
        "semantic_coverage_ratio": len(ledger.confirmed) / max(1, component_count),
        "wall_camera_coverage": wall_coverage,
        "wall_coverage_plan": wall_coverage_plan,
        "motion_requirements": motion_requirements,
        "coverage_requirements": coverage_requirements,
        "failed_targets": failures,
        "visual_localization": visual_result,
        "wheel_model": {
            "gains": {
                name: round(float(value), 4)
                for name, value in zip(FourWheelPlant.WHEEL_NAMES, plant.wheel_gains)
            },
            "breakaway_m_s": {
                name: round(float(value), 4)
                for name, value in zip(FourWheelPlant.WHEEL_NAMES, plant.breakaway_m_s)
            },
            "friction_observed_min": round(float(plant.friction.minimum), 4),
            "friction_observed_max": round(float(plant.friction.maximum), 4),
            "battery_output": round(float(plant.battery_output), 4),
            "slip_events": int(metrics["slip_events"]),
            "slip_frames": int(metrics["wheel_slip_frames"]),
            "body_jam_events": int(plant.body_jam_events),
            "body_jam_releases": int(plant.body_jam_releases),
            "command_expiries": int(plant.command_expiries),
            "command_direction_disagreement_frames": int(plant.command_direction_disagreement_frames),
        },
        "lidar": {
            "frames": int(metrics["lidar_frames"]),
            "glass_leaked_rays": int(metrics["glass_leaked_rays"]),
            "dropouts": int(metrics["lidar_dropouts"]),
            "short_speckles": int(metrics["lidar_short_speckles"]),
            "multipath_returns": int(metrics["lidar_multipath_returns"]),
        },
        "localization": {
            "accepted_corrections": tracker.accepted,
            "rejected_corrections": tracker.rejected,
            "jump_rejections": tracker.jump_rejections,
            "low_quality_rejections": tracker.low_quality_rejections,
            "bounded_reanchors": tracker.reanchors,
            "position_rmse_m": float(math.sqrt(np.mean(np.square(position_errors)))),
            "position_p95_m": float(np.percentile(position_errors, 95.0)),
            "position_max_m": float(max(position_errors)),
            "maximum_consecutive_error_over_25cm": tracker.maximum_large_error_streak,
            "yaw_p95_deg": math.degrees(float(np.percentile(yaw_errors, 95.0))),
        },
        "mission": {
            "control_steps": int(metrics["control_steps"]),
            "collision_contacts": int(metrics["collision_contacts"]),
            "target_timeouts": int(metrics["target_timeouts"]),
            "target_retry_events": int(metrics["target_retry_events"]),
            "maximum_target_drive_steps": int(metrics["maximum_target_drive_steps"]),
            "maximum_alignment_steps": int(metrics["maximum_alignment_steps"]),
            "maximum_linear_stall_streak": int(metrics["max_linear_stall_streak"]),
            "linear_stall_detections": int(metrics["linear_stall_detections"]),
            "clear_corridor_full_ttl_restarts": int(
                metrics["clear_corridor_full_ttl_restarts"]
            ),
            "short_ttl_restarts": int(metrics["short_ttl_restarts"]),
            "turn_stall_escapes": int(metrics["turn_stall_escapes"]),
            "turn_stall_escape_blocked": int(
                metrics["turn_stall_escape_blocked"]
            ),
            "maximum_turn_stall_streak": int(metrics["max_turn_stall_streak"]),
            "maximum_stationary_command_streak": int(metrics["max_stationary_command_streak"]),
            "maximum_stationary_command_seconds": round(
                float(metrics["max_stationary_command_streak"]) * profile.dt_s, 3
            ),
            "maximum_cross_track_error_m": float(metrics["maximum_cross_track_error_m"]),
            "maximum_forward_arc_yaw_deg": float(
                metrics["maximum_forward_arc_yaw_deg"]
            ),
            "large_forward_arc_turn_steps": int(
                metrics["large_forward_arc_turn_steps"]
            ),
            "minimum_center_wall_clearance_m": float(
                metrics.get("minimum_center_wall_clearance_m", math.inf)
            ),
            "temporary_map_updates": int(planner.temporary_obstacle_updates),
            "semantic_rejected_observations": int(ledger.rejected_observations),
            "first_selected_leg": selected_legs[0] if selected_legs else None,
            "maximum_planned_arrival_turn_deg": max(
                (item["arrival_turn_deg"] for item in selected_legs),
                default=math.inf,
            ),
            "near_uturn_selected_leg_count": sum(
                item["arrival_turn_deg"] >= 150.0
                for item in selected_legs
            ),
            "batch_annotation_started_after_motion": bool(
                metrics["batch_annotation_started_after_motion"]
            ),
        },
        "semantic_components": {
            str(component): record for component, record in sorted(ledger.confirmed.items())
        },
        "captures": captures,
        "attempt_records": attempt_records,
        "selected_legs": selected_legs,
        "final_truth_pose": _pose_dict(truth),
        "final_estimated_pose": _pose_dict(tracker.pose),
    }
    if artifact_directory is not None:
        artifact_directory.mkdir(parents=True, exist_ok=True)
        semantic_json = artifact_directory / "semantic_map.json"
        semantic_payload_out = {
            "map": args.map,
            "free_space_label": "navigable_floor",
            "unknown_space_label": "outside_or_unobserved",
            "occupied_components": result["semantic_components"],
            "coverage_ratio": result["semantic_coverage_ratio"],
            "captures": captures,
            "wall_camera_coverage": wall_coverage,
            "wall_surface_samples": [
                {
                    **asdict(sample),
                    "covered": sample.sample_id in covered_wall_samples,
                }
                for sample in wall_samples
            ],
        }
        semantic_json.write_text(
            json.dumps(semantic_payload_out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        semantic_png = artifact_directory / "semantic_map.png"
        _render_semantic_map(
            planner,
            ledger,
            trajectory,
            captures,
            semantic_png,
            wall_samples,
            covered_wall_samples,
        )
        result["artifacts"] = {
            "semantic_map_json": str(semantic_json.resolve()),
            "semantic_map_png": str(semantic_png.resolve()),
            "capture_directory": str((artifact_directory / "captures").resolve()),
        }
    return result


def run(
    seed_count: int = 20,
    profile_name: str = "realistic",
    output: str | Path = "car_sim/output/semantic_inspection_stress.json",
    artifact_seed: int = 0,
) -> dict:
    profile = StressProfile.named(profile_name)
    output_path = Path(output)
    artifact_root = output_path.with_suffix("").with_name(output_path.stem + "_artifacts")
    seeds = []
    for seed in range(max(1, int(seed_count))):
        seeds.append(simulate_seed(
            seed,
            profile,
            artifact_directory=artifact_root if seed == int(artifact_seed) else None,
        ))
    failed = [item for item in seeds if not item["passed"]]
    dead_zone_probe = _production_controller_dead_zone_probe()
    result = {
        "profile": profile.name,
        "profile_parameters": asdict(profile),
        "seed_count": len(seeds),
        "passed_seed_count": len(seeds) - len(failed),
        "failed_seed_count": len(failed),
        "pass_ratio": (len(seeds) - len(failed)) / len(seeds),
        "all_requirements_exercised": {
            "spatially_varying_static_friction": all(
                item["wheel_model"]["friction_observed_max"]
                - item["wheel_model"]["friction_observed_min"] >= 0.02
                for item in seeds
            ),
            "four_independent_wheel_gains": all(
                len(set(item["wheel_model"]["gains"].values())) == 4 for item in seeds
            ),
            "wheel_free_spin": sum(item["wheel_model"]["slip_events"] for item in seeds) > 0,
            "battery_voltage_sag": all(
                item["wheel_model"]["battery_output"] < 1.02 for item in seeds
            ),
            "command_ttl_expiry": sum(
                item["wheel_model"]["command_expiries"] for item in seeds
            ) > 0,
            "persistent_body_jam": sum(
                item["wheel_model"]["body_jam_events"] for item in seeds
            ) > 0,
            "linear_stall_escape_exercised": sum(
                item["mission"]["linear_stall_detections"] for item in seeds
            ) > 0,
            "map_scene_difference": all(item["actual_map_changed_cells"] > 0 for item in seeds),
            "left_glass_lidar_leak": sum(item["lidar"]["glass_leaked_rays"] for item in seeds) > 0,
            "lidar_error_types": all(
                sum(item["lidar"][name] for item in seeds) > 0
                for name in ("dropouts", "short_speckles", "multipath_returns")
            ),
            "command_motion_disagreement": sum(
                item["wheel_model"]["command_direction_disagreement_frames"]
                for item in seeds
            ) > 0,
            "visual_top_k_and_fallback": all(
                len(item["visual_localization"]["top_k"]) == 3 for item in seeds
            ),
            "visual_multiframe_fusion": any(
                item["visual_localization"]["frames"] > 1 for item in seeds
            ),
            "all_photo_targets_visited": all(
                item["capture_count"] == item["target_count"] for item in seeds
            ),
            "first_photo_has_aligned_arrival": all(
                item["motion_requirements"]["first_photo_no_arrival_uturn"]
                for item in seeds
            ),
            "qwen_runs_only_after_motion": all(
                item["motion_requirements"]["batch_annotation_after_motion"]
                for item in seeds
            ),
            "semantic_map_complete": all(item["semantic_coverage_ratio"] == 1.0 for item in seeds),
            "zero_collision_every_seed": all(
                item["mission"]["collision_contacts"] == 0 for item in seeds
            ),
            "no_unrecovered_target_timeout_every_seed": all(
                item["motion_requirements"]["no_unrecovered_target_timeout"]
                for item in seeds
            ),
            "no_long_stationary_command": all(
                item["mission"]["maximum_stationary_command_streak"] <= 8
                for item in seeds
            ),
            "bounded_path_tracking": all(
                item["motion_requirements"]["cross_track_bounded"] for item in seeds
            ),
            "no_thunder_forward_arcs": all(
                item["motion_requirements"]["no_thunder_forward_arcs"]
                for item in seeds
            ),
            "no_sustained_localization_divergence": all(
                item["motion_requirements"]["no_sustained_localization_divergence"]
                for item in seeds
            ),
            "wall_surface_camera_coverage": all(
                item["coverage_requirements"]["wall_surface_at_least_90_percent"]
                for item in seeds
            ),
            "radar_invisible_wall_features_seen": all(
                item["coverage_requirements"]["visual_only_features_at_least_95_percent"]
                for item in seeds
            ),
            "production_controller_recovers_loaded_dead_zone": bool(
                dead_zone_probe["recovered"]
                and dead_zone_probe["bounded_single_peak"]
                and dead_zone_probe["returned_to_fine_speed"]
                and dead_zone_probe["mixed_arc_ttl_allows_loaded_start"]
            ),
        },
        "aggregate": {
            "total_targets": sum(item["target_count"] for item in seeds),
            "total_captures": sum(item["capture_count"] for item in seeds),
            "total_wheel_slip_events": sum(item["wheel_model"]["slip_events"] for item in seeds),
            "total_body_jam_events": sum(
                item["wheel_model"]["body_jam_events"] for item in seeds
            ),
            "total_command_expiries": sum(
                item["wheel_model"]["command_expiries"] for item in seeds
            ),
            "total_linear_stall_detections": sum(
                item["mission"]["linear_stall_detections"] for item in seeds
            ),
            "maximum_first_leg_arrival_turn_deg": max(
                float(item["mission"]["first_selected_leg"]["arrival_turn_deg"])
                for item in seeds
                if item["mission"]["first_selected_leg"] is not None
            ),
            "total_near_uturn_selected_legs": sum(
                item["mission"]["near_uturn_selected_leg_count"]
                for item in seeds
            ),
            "total_clear_corridor_full_ttl_restarts": sum(
                item["mission"]["clear_corridor_full_ttl_restarts"]
                for item in seeds
            ),
            "total_short_ttl_restarts": sum(
                item["mission"]["short_ttl_restarts"] for item in seeds
            ),
            "total_glass_leaked_rays": sum(item["lidar"]["glass_leaked_rays"] for item in seeds),
            "total_lidar_dropouts": sum(item["lidar"]["dropouts"] for item in seeds),
            "total_lidar_short_speckles": sum(item["lidar"]["short_speckles"] for item in seeds),
            "total_lidar_multipath_returns": sum(item["lidar"]["multipath_returns"] for item in seeds),
            "maximum_localization_p95_m": max(item["localization"]["position_p95_m"] for item in seeds),
            "maximum_cross_track_error_m": max(
                item["mission"]["maximum_cross_track_error_m"] for item in seeds
            ),
            "maximum_forward_arc_yaw_deg": max(
                item["mission"]["maximum_forward_arc_yaw_deg"]
                for item in seeds
            ),
            "total_large_forward_arc_turn_steps": sum(
                item["mission"]["large_forward_arc_turn_steps"]
                for item in seeds
            ),
            "minimum_center_wall_clearance_m": min(
                item["mission"]["minimum_center_wall_clearance_m"] for item in seeds
            ),
            "maximum_stationary_command_seconds": max(
                item["mission"]["maximum_stationary_command_seconds"] for item in seeds
            ),
            "total_collision_contacts": sum(
                item["mission"]["collision_contacts"] for item in seeds
            ),
            "minimum_wall_surface_coverage_ratio": min(
                item["wall_camera_coverage"]["wall_surface_coverage_ratio"]
                for item in seeds
            ),
            "minimum_visual_only_feature_coverage_ratio": min(
                item["wall_camera_coverage"]["visual_only_feature_coverage_ratio"]
                for item in seeds
            ),
            "maximum_heading_error_deg": max(
                capture["heading_error_deg"] for item in seeds for capture in item["captures"]
            ),
        },
        "representative_seed": seeds[int(artifact_seed)] if 0 <= int(artifact_seed) < len(seeds) else seeds[0],
        "production_controller_dead_zone_probe": dead_zone_probe,
        "failed_seeds": failed[:10],
    }
    result["passed"] = bool(
        not failed and all(result["all_requirements_exercised"].values())
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="computer-only disturbed semantic inspection simulation"
    )
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--profile", choices=("mild", "realistic", "harsh"), default="realistic")
    parser.add_argument("--artifact-seed", type=int, default=0)
    parser.add_argument(
        "--output", default="car_sim/output/semantic_inspection_stress.json"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args.seed_count, args.profile, args.output, args.artifact_seed)
    summary = dict(result)
    summary.pop("failed_seeds", None)
    summary.pop("representative_seed", None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
