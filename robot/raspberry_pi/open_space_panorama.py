"""Low-frequency map correction and open-space panorama inspection.

This mode deliberately does not build a map or plan a route to pre-generated
photo targets.  The permanent occupancy map is used only as a read-only
geometric guard and as a low-rate correction reference.  Between corrections
the displayed pose follows bounded scan-to-scan odometry at lidar rate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from raspberry_pi.known_map_navigation import InspectionTarget, LocalizationResult
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.geometry import normalize_angle
from raspberry_pi.lidar.base import LaserScan


# Conservative compatibility floor, not a reason to extend stale-scan motion.
# Field run 20260917_152248: balanced twist with TTL=46 ms was rejected.
MINIMUM_MOTION_TTL_MS = 100


@dataclass(frozen=True)
class PanoramaConfig:
    minimum_station_spacing_m: float = 1.00
    ideal_station_spacing_m: float = 1.60
    maximum_station_spacing_m: float = 2.50
    minimum_stations: int = 3
    maximum_stations: int = 16
    angular_step_deg: float = 60.0
    camera_fov_deg: float = 82.0
    camera_range_m: float = 4.0
    coverage_target: float = 0.90
    open_minimum_clearance_m: float = 0.48
    open_direction_clearance_m: float = 0.72
    open_required_directions: int = 3
    map_correction_distance_m: float = 1.20
    map_correction_gain: float = 0.20
    map_correction_confirmations: int = 2
    map_correction_radius_m: float = 0.65
    drive_speed_mm_s: int = 380
    slow_speed_mm_s: int = 240
    turn_speed_mrad_s: int = 2600
    turn_max_speed_mrad_s: int = 3000
    status_interval_s: float = 0.20
    self_return_radius_m: float = 0.0
    self_return_half_width_m: float = 0.125
    stop_distance_m: float = 0.42
    lidar_offset_x_m: float = 0.14
    lidar_offset_y_m: float = 0.0
    startup_window_s: float = 0.70
    drive_progress_window_s: float = 1.80
    maximum_scan_age_s: float = 0.50
    startup_speed_mm_s: int = 420
    body_length_m: float = 0.24
    body_width_m: float = 0.22
    footprint_margin_m: float = 0.05

    @property
    def swept_half_width_m(self):
        return self.body_width_m / 2.0 + self.footprint_margin_m

    @property
    def rotation_clearance_m(self):
        return math.hypot(self.body_length_m / 2.0, self.body_width_m / 2.0) + self.footprint_margin_m

    def map_clearance_m(self, requested_m):
        # Never allow a script override to inflate less than the actual body.
        return max(float(requested_m), math.ceil(self.rotation_clearance_m * 100.0) / 100.0)

    @classmethod
    def from_args(cls, args):
        return cls(
            minimum_station_spacing_m=float(args.panorama_min_station_spacing),
            ideal_station_spacing_m=float(args.panorama_ideal_station_spacing),
            maximum_station_spacing_m=float(args.panorama_max_station_spacing),
            minimum_stations=int(args.panorama_min_stations),
            maximum_stations=int(args.panorama_max_stations),
            angular_step_deg=float(args.panorama_angle_step_deg),
            camera_fov_deg=float(args.panorama_camera_fov_deg),
            camera_range_m=float(args.panorama_camera_range),
            coverage_target=float(args.panorama_coverage_target),
            open_minimum_clearance_m=float(args.panorama_open_min_clearance),
            open_direction_clearance_m=float(
                args.panorama_open_direction_clearance
            ),
            open_required_directions=int(args.panorama_open_required_directions),
            map_correction_distance_m=float(
                args.panorama_map_correction_distance
            ),
            map_correction_gain=float(args.panorama_map_correction_gain),
            map_correction_confirmations=int(
                args.panorama_map_correction_confirmations
            ),
            map_correction_radius_m=float(args.panorama_map_correction_radius),
            drive_speed_mm_s=int(args.panorama_drive_speed),
            slow_speed_mm_s=int(args.panorama_slow_speed),
            turn_speed_mrad_s=int(args.panorama_turn_speed),
            turn_max_speed_mrad_s=int(args.panorama_turn_max_speed),
            status_interval_s=float(args.panorama_status_interval_seconds),
            self_return_radius_m=float(getattr(args, "panorama_self_return_radius", 0.0)),
            self_return_half_width_m=float(getattr(args, "panorama_self_return_half_width", 0.125)),
            stop_distance_m=float(getattr(args, "stop_distance", .42)),
            lidar_offset_x_m=float(getattr(args, "lidar_offset_x", .14)),
            lidar_offset_y_m=float(getattr(args, "lidar_offset_y", 0)),
            startup_window_s=float(getattr(args, "panorama_startup_window_seconds", .70)),
            drive_progress_window_s=float(getattr(args, "panorama_progress_window_seconds", 1.80)),
            maximum_scan_age_s=float(getattr(args, "panorama_max_scan_age_seconds", .50)),
            startup_speed_mm_s=int(getattr(args, "start_kick_speed", 420)),
            body_length_m=float(getattr(args, "panorama_body_length", .24)),
            body_width_m=float(getattr(args, "panorama_body_width", .22)),
            footprint_margin_m=float(getattr(args, "panorama_footprint_margin", .05)),
        )

    def validate(self):
        if not (.15 <= self.body_length_m <= .80 and .15 <= self.body_width_m <= .80
                and .04 <= self.footprint_margin_m <= .15):
            raise ValueError("实车长宽须为 0.15～0.80 m，尺寸/定位余量须为 0.04～0.15 m")
        if not .4 <= self.startup_window_s <= 1.0 or not 1.2 <= self.drive_progress_window_s <= 3.0:
            raise ValueError("起步窗口须为 0.4～1 s，运动窗口须为 1.2～3 s")
        if not .2 <= self.maximum_scan_age_s <= .8 or not 160 <= self.startup_speed_mm_s <= 450:
            raise ValueError("扫描最大年龄须为 0.2～0.8 s，起步输出须为 160～450")
        if not (
            0.30 <= self.minimum_station_spacing_m
            < self.ideal_station_spacing_m
            < self.maximum_station_spacing_m <= 4.0
        ):
            raise ValueError(
                "全景站点间距必须满足 0.30 <= minimum < ideal < maximum <= 4.0"
            )
        if not 3 <= self.minimum_stations <= self.maximum_stations <= 16:
            raise ValueError("全景站点数量必须满足 3 <= minimum <= maximum <= 16")
        if not 30.0 <= self.angular_step_deg <= 90.0:
            raise ValueError("全景拍摄角度步长必须在 30～90 度之间")
        photo_count = 360.0 / self.angular_step_deg
        if abs(photo_count - round(photo_count)) > 1e-6:
            raise ValueError("360 度必须能被全景拍摄角度步长整除")
        if not 0.70 <= self.coverage_target <= 0.98:
            raise ValueError("全景覆盖率目标必须在 0.70～0.98 之间")
        if self.open_required_directions not in (2, 3, 4):
            raise ValueError("开阔方向数量必须为 2、3 或 4")
        if not 2 <= self.map_correction_confirmations <= 4:
            raise ValueError("低频地图纠错需要连续 2～4 个一致候选")
        if not (
            self.angular_step_deg + 8.0 <= self.camera_fov_deg <= 180.0
            and 1.5 <= self.camera_range_m <= 8.0
        ):
            raise ValueError("全景相机视场必须有至少 8 度重叠，距离在 1.5～8 m 内")
        if not 0.05 <= self.map_correction_gain <= 0.35:
            raise ValueError("全景地图纠错融合比例必须在 0.05～0.35 内")
        if not 0.35 <= self.map_correction_radius_m <= 0.80:
            raise ValueError("全景局部地图纠错半径必须在 0.35～0.80 m 内")
        if not 160 <= self.slow_speed_mm_s <= self.drive_speed_mm_s <= 450:
            raise ValueError("全景慢速和直行速度必须满足 160 <= slow <= drive <= 450")
        if not 1800 <= self.turn_speed_mrad_s <= self.turn_max_speed_mrad_s <= 3000:
            raise ValueError("全景转向速度必须满足 1800 <= turn <= maximum <= 3000")
        if not 0.10 <= self.status_interval_s <= 0.50:
            raise ValueError("全景状态输出间隔必须在 0.10～0.50 秒内")
        if not 0.0 <= self.self_return_radius_m <= 0.20:
            raise ValueError("已确认车体回波遮罩半径必须在 0～0.20 m 内，须按传感器安装实测确认")
        if not 0.0 < self.self_return_half_width_m <= 0.14:
            raise ValueError("已确认车体回波遮罩半宽必须在 0～0.14 m 内，须按传感器安装实测确认")


class _BodyFilteredScan:
    """Preserve scan metadata while removing explicitly configured body returns.

    Mask only the sensor's rear half-plane inside the chassis inner disk.
    NaN means occluded/unknown, never a fabricated long free-space return.
    """

    def __init__(self, source, masked):
        self._source = source
        self.angles_rad = source.angles_rad
        self.distances_m = np.asarray(source.distances_m, dtype=np.float64).copy()
        self.distances_m[masked] = np.nan
        self.body_return_count = int(np.count_nonzero(masked))

    def __getattr__(self, name):
        return getattr(self._source, name)

    def points(self, min_distance_m=0.12, max_distance_m=8.0):
        return LaserScan.points(self, min_distance_m, max_distance_m)


class _ReceivedScan:
    """Receipt clock plus untouched acquisition timestamp/source arrays."""

    def __init__(self, source):
        self._source = source
        self.received_monotonic_s = time.monotonic()

    def __getattr__(self, name):
        return getattr(self._source, name)


def _scan_age_s(scan):
    now = time.monotonic()
    stamp = float(getattr(scan, "timestamp_s", 0) or 0)
    received = float(getattr(scan, "received_monotonic_s", now))
    # N10 stamps acquisition with this monotonic clock. Zero is explicitly
    # unavailable (legacy unit/offline scans), never a fabricated acquisition.
    if not math.isfinite(stamp) or stamp < 0:
        return math.inf
    if stamp > 0:
        if not math.isfinite(stamp) or stamp > now + .05:
            return math.inf
        return max(0., now - stamp, now - received)
    return max(0., now - received)


def _scan_has_data(scan):
    angles, distances = np.asarray(scan.angles_rad), np.asarray(scan.distances_m)
    return int(np.count_nonzero(np.isfinite(angles) & np.isfinite(distances)
        & (distances > 0.) & (distances <= 8.))) >= 30


def _navigation_stop_distance(config, speed_mm_s=240):
    # Keep the braking allowance; replace only the old fixed 30-cm body term.
    # Sector ranges are sensor-relative, swept/map ranges are center-relative:
    # use the more conservative forward extent for their shared threshold.
    forward_extent = max(config.body_length_m / 2., config.lidar_offset_x_m, 0.)
    return max(config.stop_distance_m + forward_extent + config.footprint_margin_m,
               config.open_minimum_clearance_m) + max(0., (abs(speed_mm_s) - 240) / 1000. * .35)


def _advance_assessment(scan, pose, known_map, config, relative_deg=0., speed_mm_s=240):
    lidar_m, support = _robust_sector_clearance(scan, relative_deg)
    guard = _nearest_map_guard_pose(known_map, pose)
    map_m = _map_ray_clearance(known_map, guard, relative_deg)
    # Rotate points AND the sensor offset into the candidate's corridor.
    angle = math.radians(relative_deg)
    candidate_scan = LaserScan(np.asarray(scan.angles_rad) - angle, np.asarray(scan.distances_m), 0.)
    offset_x = config.lidar_offset_x_m * math.cos(angle) + config.lidar_offset_y_m * math.sin(angle)
    offset_y = -config.lidar_offset_x_m * math.sin(angle) + config.lidar_offset_y_m * math.cos(angle)
    sweep = _swept_clearance_details(candidate_scan, half_width_m=config.swept_half_width_m,
                                     offset_x_m=offset_x, offset_y_m=offset_y)
    swept = sweep['clearance_m']
    nearest = sweep['nearest_point']
    if nearest is not None:
        nearest = {**_scan_point_details(scan, nearest['scan_index'],
                                        config.lidar_offset_x_m, config.lidar_offset_y_m),
                   'candidate_relative_deg': float(relative_deg),
                   'candidate_forward_m': swept}
    effective = min(lidar_m, map_m, swept)
    stop_m = _navigation_stop_distance(config, speed_mm_s)
    horizon_m = min(.35, max(.12, abs(speed_mm_s) / 1000. * config.maximum_scan_age_s + .08))
    heading = guard.yaw_rad + angle
    endpoint = (guard.x_m + horizon_m * math.cos(heading), guard.y_m + horizon_m * math.sin(heading))
    map_clear = bool(known_map.line_is_clear((guard.x_m, guard.y_m), endpoint))
    age = _scan_age_s(scan)
    reason = ('scan_stale' if age > config.maximum_scan_age_s
              else 'lidar_support_missing' if support < 3
              else 'swept_or_distance_blocked' if effective <= stop_m
              else 'map_short_segment_blocked' if not map_clear else '')
    raw = getattr(scan, '_source', scan)
    if isinstance(raw, _BodyFilteredScan):
        raw = raw._source
    raw_lidar, raw_support = _robust_sector_clearance(raw, relative_deg)
    return {'can_advance': not bool(reason), 'reason': reason,
            'lidar_m': lidar_m, 'raw_lidar_m': raw_lidar, 'raw_support': raw_support,
            'support': support, 'map_m': map_m, 'swept_m': swept,
            'effective_m': effective, 'stop_m': stop_m,
            'horizon_m': horizon_m, 'map_short_clear': map_clear,
            'scan_age_s': age, 'speed_mm_s': int(speed_mm_s),
            'nearest_swept_point': nearest,
            'swept_half_width_m': config.swept_half_width_m}


class _DriveProgressWindow:
    """Evaluate an energized drive window, not low-motion scan counts."""

    def __init__(self, config):
        self.config = config
        self.pause()

    def pause(self):
        self.started_s = None
        self.distance_m = 0.
        self.known = 0
        self.unknown = 0
        self.energized_s = 0.
        self.last_observed_s = None
        self.interrupted = False

    def begin(self, now):
        if self.started_s is None:
            self.started_s = now

    def starting(self, now):
        return self.started_s is None or (self.energized_s < self.config.startup_window_s and self.distance_m < .025)

    def observe(self, now, distance_m, accepted, energized_s=None):
        if self.started_s is None:
            return 'paused'
        delta = max(0., now - (self.last_observed_s if self.last_observed_s is not None else self.started_s))
        if energized_s is not None and delta > energized_s + .08:
            self.interrupted = True
        self.energized_s += delta if energized_s is None else max(0., min(delta, energized_s))
        self.last_observed_s = now
        if accepted:
            self.known += 1
            self.distance_m += max(0., distance_m)
        else:
            self.unknown += 1
        if self.distance_m >= .025:
            self.pause()
            self.begin(now - self.config.startup_window_s)
            self.energized_s = self.config.startup_window_s
            return 'progress'
        if self.energized_s < self.config.drive_progress_window_s:
            if now - self.started_s >= self.config.drive_progress_window_s * 2:
                return 'unknown'
            return 'starting' if self.starting(now) else 'waiting'
        return 'unknown' if self.unknown or self.interrupted else 'stalled'


def _json_finite(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {k: _json_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value


class _MotionAuditClient:
    """Record finite commands; gate background refresh on the last safe scan."""

    def __init__(self, client, config, directory):
        self.client, self.config, self.directory = client, config, directory
        self.scan = None
        self.last = None
        self.current_reason = 'controller_stop'
        self.last_render_s = 0.

    def __getattr__(self, name):
        return getattr(self.client, name)

    def _append(self, record, name='panorama_motion.jsonl'):
        with (self.directory / name).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(_json_finite(record), ensure_ascii=False, allow_nan=False) + '\n')

    def set_twist(self, linear, angular, ttl_ms):
        started = time.monotonic()
        age = _scan_age_s(self.scan) if self.scan is not None else math.inf
        if age > self.config.maximum_scan_age_s or not _scan_has_data(self.scan) or getattr(self.client, 'last_link_error', None) is not None:
            self.current_reason = 'scan_stale_or_link_error'
            self.stop()
            return False
        # A command + its last possible background refresh must expire no
        # later than the freshness budget; changed outputs remain ACKed.
        requested_ttl_ms = int(ttl_ms)
        budget_ms = max(0, int((self.config.maximum_scan_age_s - age) * 1000))
        ttl_ms = min(requested_ttl_ms, budget_ms)
        if ttl_ms < MINIMUM_MOTION_TTL_MS:
            self._append({'event': 'command_deferred', 'reason': 'scan_budget_too_short',
                          'monotonic_s': started, 'scan_age_s': age,
                          'requested_ttl_ms': requested_ttl_ms, 'available_ttl_ms': budget_ms,
                          'minimum_ttl_ms': MINIMUM_MOTION_TTL_MS,
                          'requested_linear_mm_s': int(linear),
                          'requested_angular_mrad_s': int(angular)})
            self.current_reason = 'scan_budget_too_short'
            self.stop()
            # stop() resets the log reason; callers need this retryable cause.
            self.current_reason = 'scan_budget_too_short'
            return False
        self.last = {'issued_monotonic_s': started, 'requested_linear_mm_s': int(linear),
            'requested_angular_mrad_s': int(angular), 'sent_linear_mm_s': int(linear),
            'sent_angular_mrad_s': int(angular), 'ttl_ms': ttl_ms, 'scan_age_s': age,
            'scan_timestamp_s': getattr(self.scan, 'timestamp_s', None),
            'scan_received_monotonic_s': getattr(self.scan, 'received_monotonic_s', None),
            'motor_feedback': None, 'supply_feedback': None}
        bounded = getattr(self.client, 'set_bounded_twist', None)
        refresh_deadline = started + max(0., self.config.maximum_scan_age_s - age - ttl_ms / 1000.) if linear else started
        try:
            if bounded is not None:
                reply = bounded(linear, angular, ttl_ms=ttl_ms, refresh_deadline_s=refresh_deadline)
            else:
                reply = self.client.set_twist(linear, angular, ttl_ms=ttl_ms)
        except Exception as exc:
            self.last['error'] = type(exc).__name__
            self.last['error_message'] = str(exc)
            self._append(self.last)
            self.current_reason = 'command_link_error'
            self.stop()
            raise
        self.last['ack_elapsed_s'] = time.monotonic() - started
        if reply is False:
            self.last['error'] = 'command_denied'
            self._append(self.last)
            self.current_reason = 'command_denied'
            self.stop()
            return False
        remaining = max(0., self.config.maximum_scan_age_s - _scan_age_s(self.scan) - ttl_ms / 1000.)
        refresh = getattr(self.client, 'allow_twist_to_refresh_for', None)
        expire = getattr(self.client, 'allow_twist_to_expire', None)
        if bounded is not None:
            remaining = max(0., refresh_deadline - time.monotonic())
        elif linear and remaining > 0 and refresh is not None:
            refresh(remaining)
        elif expire is not None:
            expire()
        self.last['refresh_window_s'] = remaining if linear and refresh is not None else 0.
        self._append(self.last)
        return True

    def stop(self):
        started = time.monotonic()
        if self.last is not None:
            self.last['stopped_monotonic_s'] = started
        result = self.client.stop()
        self._append({'event': 'stop', 'monotonic_s': started,
                      'stop_reason': self.current_reason, 'ack_elapsed_s': time.monotonic() - started})
        self.current_reason = 'controller_stop'
        return result

    def stop_event(self, reason, scan, assessment=None):
        if reason == 'turn_clearance_blocked' and assessment is None:
            assessment = getattr(self, '_panorama_turn_guard', None)
        self.current_reason = reason
        self.stop()
        raw = getattr(scan, '_source', scan)
        if isinstance(raw, _BodyFilteredScan):
            raw = raw._source
        self._append({'event': 'stop_trigger', 'reason': reason, 'monotonic_s': time.monotonic(),
            'assessment': assessment, 'source_timestamp_s': getattr(scan, 'timestamp_s', None),
            'raw_angles_rad': np.asarray(raw.angles_rad).tolist(),
            'raw_distances_m': np.asarray(raw.distances_m).tolist(),
            'filtered_distances_m': np.asarray(scan.distances_m).tolist()}, 'panorama_stop_events.jsonl')

def _filter_body_returns(scan, radius_m, offset_x_m=0.0, offset_y_m=0.0, half_width_m=0.125):
    """Apply the independently confirmed self-return mask, not body dimensions.

    Approximate collision footprint dimensions must not clip this installed
    sensor mask. Only the configured rear-plane/disk/strip intersection is
    masked; outside points remain visible and masked directions stay unknown.
    """
    if radius_m <= 0.0:
        return scan
    angles = np.asarray(scan.angles_rad, dtype=np.float64)
    distances = np.asarray(scan.distances_m, dtype=np.float64)
    sensor_x = distances * np.cos(angles)
    body_x = sensor_x + float(offset_x_m)
    body_y = distances * np.sin(angles) + float(offset_y_m)
    masked = (
        np.isfinite(angles) & np.isfinite(distances) & (distances > 0.0)
        & (sensor_x <= 1e-9)
        & (np.hypot(body_x, body_y) < float(radius_m))
        & (np.abs(body_y) < float(half_width_m))
    )
    return _BodyFilteredScan(scan, masked)


def _angle_delta(a, b):
    return normalize_angle(float(a) - float(b))


def _robust_sector_clearance(
    scan,
    center_deg,
    width_deg=38.0,
    minimum_m=0.12,
    maximum_m=8.0,
):
    angles = np.asarray(scan.angles_rad, dtype=np.float64)
    distances = np.asarray(scan.distances_m, dtype=np.float64)
    relative = (
        angles - math.radians(float(center_deg)) + math.pi
    ) % (2.0 * math.pi) - math.pi
    selected = (
        (np.abs(relative) <= math.radians(float(width_deg)) * 0.5)
        & np.isfinite(distances)
        & (distances >= float(minimum_m))
        & (distances <= float(maximum_m))
    )
    values = np.sort(distances[selected])
    if len(values) < 3:
        return math.inf, int(len(values))
    if values[0] <= 0.55:
        # A single very close return may be a speckle, but treating it as open
        # can put a map-unrecorded thin object inside the chassis envelope.
        return float(values[0]), int(len(values))
    # Ignore one isolated near-field speckle, but a real obstacle cluster is
    # still represented by the next-lowest return.
    index = min(len(values) - 1, max(1, int(math.floor(len(values) * 0.12))))
    return float(values[index]), int(len(values))


def forward_swept_clearance(scan, half_width_m=0.28, offset_x_m=0.0, offset_y_m=0.0, reverse=False):
    """Nearest return inside the forward chassis corridor, not a centre ray.

    Include side-front returns: a clear centre beam does not imply that the
    wheels will clear a table leg. Missing returns never count as free space.
    """
    return _swept_clearance_details(scan, half_width_m, offset_x_m, offset_y_m, reverse)['clearance_m']


def _scan_point_details(scan, index, offset_x_m, offset_y_m):
    angle, distance = float(scan.angles_rad[index]), float(scan.distances_m[index])
    raw = scan
    while getattr(raw, '_source', None) is not None:
        raw = raw._source
    return {'scan_index': int(index),
            'sensor_angle_deg': math.degrees(normalize_angle(angle)),
            'raw_distance_m': float(raw.distances_m[index]),
            'body_x_m': distance * math.cos(angle) + float(offset_x_m),
            'body_y_m': distance * math.sin(angle) + float(offset_y_m),
            'filtered_out': False}


def _swept_clearance_details(scan, half_width_m=.28, offset_x_m=0., offset_y_m=0., reverse=False):
    angles = np.asarray(scan.angles_rad, dtype=np.float64)
    distances = np.asarray(scan.distances_m, dtype=np.float64)
    x = distances * np.cos(angles) + float(offset_x_m)
    y = distances * np.sin(angles) + float(offset_y_m)
    if reverse:
        x = -x
    valid = np.isfinite(angles) & np.isfinite(distances) & (distances >= 0.08) & (distances <= 8.0)
    corridor = valid & (x > 0.0) & (np.abs(y) <= float(half_width_m))
    if not np.any(corridor):
        return {'clearance_m': 0., 'nearest_point': None}
    indices = np.flatnonzero(corridor)
    index = int(indices[np.argmin(x[indices])])
    return {'clearance_m': float(x[index]),
            'nearest_point': _scan_point_details(scan, index, offset_x_m, offset_y_m)}


def _map_ray_clearance(known_map, pose, relative_deg, maximum_m=4.0):
    angle = float(pose.yaw_rad) + math.radians(float(relative_deg))
    step = max(0.025, float(known_map.resolution_m) * 0.65)
    distance = step
    while distance <= float(maximum_m):
        point = (
            pose.x_m + distance * math.cos(angle),
            pose.y_m + distance * math.sin(angle),
        )
        column, row = known_map.world_to_grid((point,))[0]
        column, row = int(column), int(row)
        if not known_map.in_bounds(column, row):
            return distance
        if known_map.occupied[row, column] or known_map.unknown[row, column]:
            return distance
        distance += step
    return float(maximum_m)


def _nearest_map_guard_pose(known_map, pose, maximum_radius=12):
    column, row = known_map.world_to_grid(((pose.x_m, pose.y_m),))[0]
    column, row = int(column), int(row)
    if (
        known_map.in_bounds(column, row)
        and known_map.traversable[row, column]
    ):
        return pose
    nearest = known_map._nearest_traversable(
        (column, row), maximum_radius=maximum_radius
    )
    if nearest is None:
        return pose
    guard_xy = known_map.grid_to_world((nearest,))[0]
    return Pose2D(float(guard_xy[0]), float(guard_xy[1]), pose.yaw_rad)


def directional_clearances(scan, known_map, pose):
    result = {}
    for name, angle in (
        ("front", 0.0),
        ("left", 90.0),
        ("rear", 180.0),
        ("right", -90.0),
        ("front_left", 45.0),
        ("front_right", -45.0),
    ):
        lidar_m, support = _robust_sector_clearance(scan, angle)
        map_m = _map_ray_clearance(known_map, pose, angle)
        # The permanent map protects glass walls when lidar rays leak through.
        # Missing lidar support does not convert a mapped wall into open space.
        effective = min(lidar_m, map_m)
        result[name] = {
            "lidar_m": lidar_m,
            "map_m": map_m,
            "effective_m": effective,
            "support": support,
        }
    return result


def open_space_assessment(clearances, config):
    cardinal = [
        float(clearances[name]["effective_m"])
        for name in ("front", "left", "rear", "right")
    ]
    open_count = sum(
        value >= config.open_direction_clearance_m for value in cardinal
    )
    minimum = min(cardinal)
    return {
        "is_open": bool(
            minimum >= config.open_minimum_clearance_m
            and open_count >= config.open_required_directions
        ),
        "minimum_clearance_m": minimum,
        "open_direction_count": int(open_count),
        "cardinal_clearances_m": cardinal,
    }


class SceneCoverageTracker:
    """Photo-only coverage overlay for occupied boundaries in the saved map."""

    def __init__(self, known_map, sample_spacing_m=0.16):
        occupied = np.asarray(known_map.occupied, dtype=np.bool_)
        free = np.asarray(known_map.free, dtype=np.bool_)
        adjacent_free = np.zeros_like(occupied)
        adjacent_free[1:, :] |= free[:-1, :]
        adjacent_free[:-1, :] |= free[1:, :]
        adjacent_free[:, 1:] |= free[:, :-1]
        adjacent_free[:, :-1] |= free[:, 1:]
        boundary = occupied & adjacent_free
        cells = np.argwhere(boundary)
        stride = max(
            1,
            int(round(float(sample_spacing_m) / known_map.resolution_m)),
        )
        if len(cells):
            keep = (
                (cells[:, 0] % stride == 0)
                | (cells[:, 1] % stride == 0)
            )
            cells = cells[keep]
        self.known_map = known_map
        self.cells = cells
        self.points = (
            known_map.grid_to_world(cells[:, ::-1])
            if len(cells) else np.empty((0, 2), dtype=np.float64)
        )
        self.covered = np.zeros(len(self.points), dtype=np.bool_)

    @property
    def ratio(self):
        if not len(self.covered):
            return 1.0
        return float(np.mean(self.covered))

    def _visible_mask(self, pose, yaw_rad, fov_deg, maximum_range_m):
        if not len(self.points):
            return np.zeros(0, dtype=np.bool_)
        deltas = self.points - np.asarray((pose.x_m, pose.y_m))
        ranges = np.linalg.norm(deltas, axis=1)
        headings = np.arctan2(deltas[:, 1], deltas[:, 0])
        errors = np.abs(
            (headings - float(yaw_rad) + math.pi) % (2.0 * math.pi) - math.pi
        )
        candidates = np.flatnonzero(
            (ranges >= 0.20)
            & (ranges <= float(maximum_range_m))
            & (errors <= math.radians(float(fov_deg)) * 0.5)
        )
        visible = np.zeros(len(self.points), dtype=np.bool_)
        for index in candidates:
            if self._line_visible(pose, self.points[index], ranges[index]):
                visible[index] = True
        return visible

    def _line_visible(self, pose, target_xy, target_range_m):
        # The endpoint is the wall sample itself.  Occupancy is allowed only
        # in the last 12 cm; an earlier occupied or unknown cell occludes it.
        samples = max(4, int(math.ceil(target_range_m / 0.05)))
        for fraction in np.linspace(0.05, 1.0, samples, endpoint=True):
            point = (
                pose.x_m + (target_xy[0] - pose.x_m) * fraction,
                pose.y_m + (target_xy[1] - pose.y_m) * fraction,
            )
            column, row = self.known_map.world_to_grid((point,))[0]
            column, row = int(column), int(row)
            if not self.known_map.in_bounds(column, row):
                return False
            remaining = target_range_m * (1.0 - fraction)
            if remaining > 0.12 and (
                self.known_map.occupied[row, column]
                or self.known_map.unknown[row, column]
            ):
                return False
        return True

    def observe(self, pose, camera_yaw_rad, config):
        visible = self._visible_mask(
            pose,
            camera_yaw_rad,
            config.camera_fov_deg,
            config.camera_range_m,
        )
        newly = int(np.count_nonzero(visible & ~self.covered))
        self.covered |= visible
        return newly

    def expected_panorama_gain(self, pose, config):
        if not len(self.points):
            return 0
        # Direction selection only needs a cheap ranking signal.  A complete
        # panorama covers every bearing, so rank candidate endpoints by the
        # number of still-unseen boundary samples in range.  Exact map-ray
        # occlusion remains enforced when a real photo is recorded.
        ranges = np.linalg.norm(
            self.points - np.asarray((pose.x_m, pose.y_m)), axis=1
        )
        potentially_visible = (
            (ranges >= 0.20) & (ranges <= config.camera_range_m)
        )
        return int(np.count_nonzero(potentially_visible & ~self.covered))

    def summary(self):
        uncovered = np.flatnonzero(~self.covered)
        return {
            "boundary_sample_count": int(len(self.covered)),
            "covered_sample_count": int(np.count_nonzero(self.covered)),
            "coverage_ratio": round(self.ratio, 6),
            "uncovered_samples": [
                [round(float(self.points[index][0]), 3),
                 round(float(self.points[index][1]), 3)]
                for index in uncovered[:500]
            ],
        }


def _observe_photo_coverage(verified, provisional, pose, camera_yaw, config, pose_uncertain):
    """Uncertain photos guide exploration but never certify map coverage."""
    provisional_new = provisional.observe(pose, camera_yaw, config)
    verified_new = 0 if pose_uncertain else verified.observe(pose, camera_yaw, config)
    return verified_new, provisional_new


class OpenSpaceDirectionSelector:
    """Choose only a local heading; it never builds an A* target path."""

    CANDIDATE_DEGREES = (0.0, 15.0, -15.0, 35.0, -35.0, 70.0, -70.0, 110.0, -110.0, 145.0, -145.0, 180.0)

    def __init__(self):
        self.last_world_heading = None
        self.visited_stations = []
        self.rejected_directions = deque(maxlen=24)
        self.travel_history = deque(maxlen=240)

    def record_translation(self, pose):
        point = (float(pose.x_m), float(pose.y_m))
        if not self.travel_history or math.hypot(point[0] - self.travel_history[-1][0],
                                               point[1] - self.travel_history[-1][1]) >= .12:
            self.travel_history.append(point)

    def revisit_cost(self, endpoint):
        # Exclude the last short segment so ordinary continued travel is not
        # penalized simply for being near the current position.
        return min(6, sum(math.hypot(endpoint.x_m - x, endpoint.y_m - y) < .35
                          for x, y in list(self.travel_history)[:-4]))

    def mark_station(self, pose):
        self.visited_stations.append((float(pose.x_m), float(pose.y_m)))

    def reject_direction(self, pose, world_heading_rad):
        self.rejected_directions.append((
            float(pose.x_m), float(pose.y_m), float(world_heading_rad)
        ))

    def choose(self, scan, pose, known_map, coverage, config):
        scored = []
        guard_pose = _nearest_map_guard_pose(known_map, pose)
        uncovered_points = coverage.points[~coverage.covered]
        uncovered_heading = None
        if len(uncovered_points):
            # A coarse bearing toward the remaining photo gap is not a goal
            # point or route.  It only breaks ties between locally safe lidar
            # directions so blind walking gradually leaves already-seen areas.
            if self.visited_stations:
                station_values = np.asarray(
                    self.visited_stations, dtype=np.float64
                )
                distances = np.min(
                    np.linalg.norm(
                        uncovered_points[:, None, :] - station_values[None, :, :],
                        axis=2,
                    ),
                    axis=1,
                )
                centroid = uncovered_points[int(np.argmax(distances))]
            else:
                centroid = np.mean(uncovered_points, axis=0)
            uncovered_heading = math.atan2(
                float(centroid[1]) - pose.y_m,
                float(centroid[0]) - pose.x_m,
            )
        for relative_deg in self.CANDIDATE_DEGREES:
            advance = _advance_assessment(scan, pose, known_map, config, relative_deg, config.slow_speed_mm_s)
            lidar_m, support = advance['lidar_m'], advance['support']
            map_m = min(advance['map_m'], 2.2)
            clearance_m = min(lidar_m, map_m)
            if support < 3 and map_m < 0.70:
                continue
            travel_m = min(0.95, max(0.0, clearance_m - 0.48))
            if travel_m < 0.12:
                continue
            world_heading = normalize_angle(
                pose.yaw_rad + math.radians(relative_deg)
            )
            # Shorten the local observation horizon, never the map's chassis
            # inflation. A long straight ray can cross a corner although its
            # next safe segment leads out of it.
            endpoint = None
            for horizon_m in (travel_m, min(travel_m, .35), min(travel_m, .20), min(travel_m, .12)):
                if horizon_m < 0.12:
                    continue
                candidate_endpoint = Pose2D(
                    guard_pose.x_m + horizon_m * math.cos(world_heading),
                    guard_pose.y_m + horizon_m * math.sin(world_heading),
                    world_heading,
                )
                if known_map.line_is_clear(
                    (guard_pose.x_m, guard_pose.y_m),
                    (candidate_endpoint.x_m, candidate_endpoint.y_m),
                ):
                    endpoint = candidate_endpoint
                    break
            if endpoint is None:
                continue
            gain = coverage.expected_panorama_gain(endpoint, config)
            repeat_penalty = 0.0
            if self.last_world_heading is not None:
                repeat_penalty = max(
                    0.0,
                    1.0 - abs(_angle_delta(world_heading, self.last_world_heading))
                    / math.pi,
                )
            turn_penalty = abs(math.radians(relative_deg)) / math.pi
            coverage_heading_bonus = 0.0
            if uncovered_heading is not None:
                coverage_heading_bonus = 2.20 * math.cos(
                    _angle_delta(world_heading, uncovered_heading)
                )
            station_novelty = 0.0
            if self.visited_stations:
                nearest_station_m = min(
                    math.hypot(endpoint.x_m - x_m, endpoint.y_m - y_m)
                    for x_m, y_m in self.visited_stations
                )
                station_novelty = min(1.6, nearest_station_m) * 1.20
            blocked_direction_penalty = 0.0
            if any(
                math.hypot(pose.x_m - x_m, pose.y_m - y_m) < 0.55
                and abs(_angle_delta(world_heading, old_heading))
                < math.radians(30.0)
                for x_m, y_m, old_heading in self.rejected_directions
            ):
                blocked_direction_penalty = 5.0
            score = (
                min(clearance_m, 2.2) * 1.5
                + gain * 0.035
                + coverage_heading_bonus
                + station_novelty
                - turn_penalty * 1.50
                + repeat_penalty * 0.22
                - blocked_direction_penalty
            )
            revisit_cost = self.revisit_cost(endpoint)
            score -= revisit_cost * 2.5
            forward_eligible = (
                advance['can_advance']
                and blocked_direction_penalty == 0.0
            )
            scored.append((score, relative_deg, world_heading, clearance_m, gain, forward_eligible, revisit_cost))
        if not scored:
            return None
        # Coverage cannot buy a large turn when a supported forward course is
        # clear. Real swept-footprint and map guards still run before driving.
        straight = [item for item in scored if item[1] == 0.0 and item[5]]
        near_forward = [item for item in scored if abs(item[1]) <= 15.0 and item[5]]
        forward = [item for item in scored if abs(item[1]) <= 35.0 and item[5]]
        pool = straight or near_forward or forward or scored
        policy = ('safe_straight' if straight else 'near_forward' if near_forward
                  else 'forward_escape' if forward else 'wide_escape')
        if straight and straight[0][6] >= 2:
            novel_exit = [item for item in scored if item[5] and abs(item[1]) <= 70.
                          and item[6] + 2 <= straight[0][6]]
            if novel_exit:
                pool, policy = novel_exit, 'leave_visited_area'
        selected = max(pool)
        self.last_world_heading = selected[2]
        return {
            "score": float(selected[0]),
            "relative_heading_deg": float(selected[1]),
            "world_heading_rad": float(selected[2]),
            "clearance_m": float(selected[3]),
            "expected_new_samples": int(selected[4]),
            "selection_policy": policy,
            'revisit_cost': int(selected[6]),
        }


def _recent_safe_backtrack_heading(trail, pose, scan, known_map):
    guard_pose = _nearest_map_guard_pose(known_map, pose)
    for old_pose in reversed(list(trail)[:-1]):
        distance = math.hypot(
            old_pose.x_m - guard_pose.x_m,
            old_pose.y_m - guard_pose.y_m,
        )
        if not 0.45 <= distance <= 0.90:
            continue
        heading = math.atan2(
            old_pose.y_m - guard_pose.y_m,
            old_pose.x_m - guard_pose.x_m,
        )
        relative_deg = math.degrees(_angle_delta(heading, pose.yaw_rad))
        lidar_m, support = _robust_sector_clearance(scan, relative_deg)
        if support < 3 or lidar_m < distance + 0.25:
            continue
        if not known_map.line_is_clear(
            (guard_pose.x_m, guard_pose.y_m),
            (old_pose.x_m, old_pose.y_m),
        ):
            continue
        return {
            "world_heading_rad": heading,
            "relative_heading_deg": relative_deg,
            "clearance_m": lidar_m,
            "expected_new_samples": 0,
            "local_backtrack": True,
            "backtrack_origin_xy": (pose.x_m, pose.y_m),
        }
    return None


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _render_live_map(known_map, output, pose, trail, stations, coverage):
    image = Image.fromarray(np.flipud(known_map.image)).convert("RGB")
    draw = ImageDraw.Draw(image)

    def pixel(xy):
        column, row = known_map.world_to_grid((xy,))[0]
        return int(column), int(known_map.height - 1 - row)

    for old_pose in list(trail)[-250:]:
        x, y = pixel((old_pose.x_m, old_pose.y_m))
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(30, 190, 235))
    for station in stations:
        x, y = pixel((station.x_m, station.y_m))
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 205, 20))
    if len(coverage.points):
        for index in range(0, len(coverage.points), max(1, len(coverage.points) // 700)):
            x, y = pixel(coverage.points[index])
            color = (35, 220, 80) if coverage.covered[index] else (235, 55, 55)
            draw.point((x, y), fill=color)
    x, y = pixel((pose.x_m, pose.y_m))
    draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(20, 255, 20), outline=(0, 0, 0))
    tip = (
        x + int(13 * math.cos(pose.yaw_rad)),
        y - int(13 * math.sin(pose.yaw_rad)),
    )
    draw.line((x, y, tip[0], tip[1]), fill=(0, 0, 0), width=2)
    image.save(output)


def _send_twist(client, linear, angular, ttl_ms):
    accepted = client.set_twist(int(linear), int(angular), ttl_ms=int(ttl_ms)) is not False
    if not isinstance(client, _MotionAuditClient):
        expire = getattr(client, "allow_twist_to_expire", None)
        if expire is not None:
            expire()
    return accepted


def _motion_translation_m(match):
    """Read displacement from the real ICP MatchResult transform."""
    transform = getattr(match, "transform", None)
    if transform is None:
        return math.inf
    return math.hypot(float(transform.x_m), float(transform.y_m))


class _LocalHeadingLock:
    """A short body-relative course; map corrections cannot rotate its target."""

    def __init__(self, relative_rad):
        self.error_rad = normalize_angle(relative_rad)
        self.distance_m = 0.0
        self.drive_started_s = None
        self.aligning = False

    def observe(self, transform):
        self.error_rad = normalize_angle(self.error_rad - transform.yaw_rad)
        self.distance_m += math.hypot(transform.x_m, transform.y_m)

    def needs_alignment(self, front_blocked=False):
        error = abs(self.error_rad)
        if self.aligning:
            self.aligning = error > math.radians(25.0)
        elif error > math.radians(35.0):
            self.aligning = True
        return self.aligning or (front_blocked and error > math.radians(12.0))

    def mark_drive(self, now):
        if self.drive_started_s is None:
            self.drive_started_s = now

    def expired(self, now):
        # Time alone while turning/waiting must not alternate selected sides.
        return self.distance_m >= .65 or (
            self.drive_started_s is not None and self.distance_m >= .15
            and now - self.drive_started_s >= 4.0
        )


def _local_drive_angular(error_rad, left_m, right_m):
    corridor = max(-.25, min(.25, (left_m - right_m) * .35))
    error = max(-math.radians(25), min(math.radians(25), error_rad))
    return int(max(-450, min(450, error * 450.0 + corridor * 900.0)))


def _rotation_clearance(scan, known_map, pose, config):
    return _rotation_assessment(scan, known_map, pose, config)['clearance_m']


def _rotation_assessment(scan, known_map, pose, config):
    """Minimum clearance around the body center, not the offset lidar origin."""
    angles, distances = np.asarray(scan.angles_rad), np.asarray(scan.distances_m)
    valid = np.isfinite(angles) & np.isfinite(distances) & (distances > 0.) & (distances <= 8.)
    if not np.any(valid):
        return {'clearance_m': 0., 'reason': 'lidar_support_missing', 'nearest_point': None}
    x = distances[valid] * np.cos(angles[valid]) + config.lidar_offset_x_m
    y = distances[valid] * np.sin(angles[valid]) + config.lidar_offset_y_m
    radii = np.hypot(x, y)
    local_index = int(np.argmin(radii))
    lidar_clearance = float(radii[local_index])
    nearest = _scan_point_details(scan, int(np.flatnonzero(valid)[local_index]),
                                  config.lidar_offset_x_m, config.lidar_offset_y_m)
    # Check every occupied/unknown cell near the actual pose, including corners
    # and glass walls. Do not snap to a different, more convenient free pose.
    resolution = known_map.resolution_m
    col, row = map(int, known_map.world_to_grid(((pose.x_m, pose.y_m),))[0])
    if not known_map.in_bounds(col, row):
        return {'clearance_m': 0., 'reason': 'map_boundary', 'nearest_point': nearest}
    # The caller also uses this clearance to suppress steering below .40 m;
    # cover that neighborhood so a short search bound cannot zero all steering.
    radius = int(math.ceil(max(config.rotation_clearance_m, .50) / resolution)) + 2
    x0, x1 = max(0, col - radius), min(known_map.width, col + radius + 1)
    y0, y1 = max(0, row - radius), min(known_map.height, row + radius + 1)
    rows, cols = np.nonzero(known_map.occupied[y0:y1, x0:x1] | known_map.unknown[y0:y1, x0:x1])
    boundary = min(pose.x_m - known_map.origin_x_m,
                   known_map.origin_x_m + known_map.width * resolution - pose.x_m,
                   pose.y_m - known_map.origin_y_m,
                   known_map.origin_y_m + known_map.height * resolution - pose.y_m)
    # Cells outside this small checked window are not claimed free: cap the
    # diagnostic to the distance to its edge (a conservative local bound).
    window_edge = min(pose.x_m - (known_map.origin_x_m + x0 * resolution),
                      known_map.origin_x_m + x1 * resolution - pose.x_m,
                      pose.y_m - (known_map.origin_y_m + y0 * resolution),
                      known_map.origin_y_m + y1 * resolution - pose.y_m)
    map_clearance = min(boundary, window_edge)
    if len(rows):
        centers = known_map.grid_to_world(np.column_stack((cols + x0, rows + y0)))
        distance = np.min(np.hypot(centers[:, 0] - pose.x_m, centers[:, 1] - pose.y_m))
        map_clearance = min(map_clearance, max(0., float(distance) - resolution / math.sqrt(2.)))
    clearance = min(lidar_clearance, map_clearance)
    return {'clearance_m': clearance, 'lidar_m': lidar_clearance, 'map_m': map_clearance,
            'map_distance_is_conservative_local_bound': True,
            'required_m': config.rotation_clearance_m, 'nearest_point': nearest,
            'reason': ('' if clearance >= config.rotation_clearance_m else
                       'lidar_near_point' if lidar_clearance <= map_clearance else 'map_rotation_guard')}


class _EscapeTurnBudget:
    """Legacy thresholds retained for audit and the optional course subclass.

    Obstacle escape control no longer uses reason() to authorize turns or end
    the mission. Independent live safety and real no-progress guards remain.
    """

    maximum_angle_rad = math.radians(200.)
    maximum_steps = 12
    maximum_duration_s = 20.

    def __init__(self):
        self.reset()

    def reset(self):
        self.started_s = None
        self.angle_rad = 0.
        self.steps = 0
        self.no_response_steps = 0

    def begin(self, now):
        if self.started_s is None:
            self.started_s = now

    def observe(self, measured_rad):
        self.steps += 1
        amount = abs(float(measured_rad))
        if not math.isfinite(amount):
            amount = 0.
        self.angle_rad += amount
        self.no_response_steps = self.no_response_steps + 1 if amount < math.radians(1.5) else 0

    def reason(self, now):
        if self.no_response_steps >= 6:
            return 'turn_no_response'
        if self.angle_rad >= self.maximum_angle_rad - math.radians(4.):
            return 'turn_angle_limit'
        if self.steps >= self.maximum_steps:
            return 'turn_step_limit'
        if self.started_s is not None and now - self.started_s >= self.maximum_duration_s:
            return 'turn_time_limit'
        return ''

    def pause(self, duration_s):
        if self.started_s is not None:
            self.started_s += max(0., duration_s)

    def snapshot(self, now):
        return {'actual_angle_deg': math.degrees(self.angle_rad), 'steps': self.steps,
                'no_response_steps': self.no_response_steps,
                'elapsed_s': 0. if self.started_s is None else max(0., now - self.started_s),
                'exhaustion_reason': self.reason(now)}


class _CourseTurnBudget(_EscapeTurnBudget):
    """Bound optional course alignment without consuming obstruction credit.

    Exhaustion means take the guarded clear front, not terminate the mission.
    Reselecting a heading alone must not replenish this allowance.
    """
    maximum_angle_rad = math.radians(80.)
    maximum_steps = 4
    maximum_duration_s = 8.

    def reason(self, now):
        if self.no_response_steps >= 2:
            return 'course_no_response'
        return super().reason(now)


def _local_motion_action(front_m, stop_m, turn_clearance_m, error_rad, turn_steps,
                         required_turn_clearance_m=.30):
    """Translation first; absolute/course yaw never blocks a clear front."""
    if front_m > stop_m:
        return "drive"
    if turn_clearance_m >= required_turn_clearance_m and abs(error_rad) > math.radians(12) and turn_steps < 6:
        return "turn"
    return "blocked"


def _bounded_motion(localizer, before_scan, scan, pose):
    pending = getattr(localizer, "_panorama_pending_motion", None)
    turning = getattr(localizer, '_panorama_rotation_only', False)
    if pending is not None:
        before_scan, pose = pending
        turning = getattr(localizer, '_panorama_pending_rotation_only', turning)
    icp_started = time.monotonic()
    matcher = localizer.relative_motion
    if turning:
        matcher = getattr(localizer, 'relative_turn_motion', matcher)
    match = matcher(before_scan, scan)
    translation = _motion_translation_m(match)
    transform = getattr(match, "transform", None)
    yaw = float(transform.yaw_rad) if transform is not None else math.inf
    rmse = float(getattr(match, 'rmse_m', math.inf))
    inliers = float(getattr(match, 'inlier_ratio', 0.))
    turn_model = getattr(localizer, '_panorama_turn_model_diagnostic', None) if turning else None
    reason = (
        "scan_match_failed" if not bool(getattr(match, "converged", False))
        else "non_finite_transform" if not all(math.isfinite(v) for v in (translation, yaw))
        else "translation_exceeds_envelope" if translation > 0.38
        # Limited field trial: tolerate the observed 48.1-degree turn while
        # retaining translation, finite-transform and larger-yaw rejection.
        else "yaw_exceeds_envelope" if abs(yaw) > math.radians(60.0)
        else "scan_match_quality_low" if not (math.isfinite(rmse) and math.isfinite(inliers)
            and rmse <= .12 and inliers >= .55
            and int(getattr(match, 'correspondences', 0)) >= 35)
        else "turn_translation_unverified" if turning and translation > .08
            and not (isinstance(turn_model, dict) and turn_model.get('translation_verified'))
        else ""
    )
    localizer._panorama_motion_diagnostic = {
        "accepted": not bool(reason),
        "rejection_reason": reason,
        "translation_m": translation if math.isfinite(translation) else None,
        "yaw_deg": math.degrees(yaw) if math.isfinite(yaw) else None,
        "correspondences": getattr(match, "correspondences", None),
        'rmse_m': getattr(match, 'rmse_m', None),
        'inlier_ratio': getattr(match, 'inlier_ratio', None),
        'turn_model': turn_model,
        "icp_elapsed_s": time.monotonic() - icp_started,
    }
    if not reason:
        localizer._panorama_pending_motion = None
        localizer._panorama_pending_rotation_only = False
        localizer._panorama_turn_since_calibration_rad = (
            getattr(localizer, '_panorama_turn_since_calibration_rad', 0.) + abs(yaw)
        )
        heading_lock = getattr(localizer, "_panorama_local_heading_lock", None)
        if heading_lock is not None:
            heading_lock.observe(match.transform)
        return pose.compose(match.transform), match, True
    localizer._panorama_pending_motion = (before_scan, pose)
    localizer._panorama_pending_rotation_only = turning
    return pose, match, False


def _blend_pose(trusted, candidate, gain):
    return Pose2D(
        trusted.x_m + (candidate.x_m - trusted.x_m) * gain,
        trusted.y_m + (candidate.y_m - trusted.y_m) * gain,
        normalize_angle(
            trusted.yaw_rad
            + _angle_delta(candidate.yaw_rad, trusted.yaw_rad) * gain
        ),
    )


def _front_blockage_kind(advance):
    """Distinguish a live obstruction from map-only or missing evidence."""
    if advance.get('can_advance'):
        return 'clear'
    if advance.get('reason') in ('scan_stale', 'lidar_support_missing'):
        return 'sensor_unknown'
    lidar, swept = advance['lidar_m'], advance['swept_m']
    if not math.isfinite(lidar) or not math.isfinite(swept) or swept <= 0:
        return 'sensor_unknown'
    if min(lidar, swept) <= advance['stop_m']:
        return 'lidar_obstacle'
    return 'map_only'


def _local_stop_diagnostic(advance):
    kind = _front_blockage_kind(advance)
    if kind == 'map_only':
        return 'MAP_GUARD_STOP', 'map_guard_blocked'
    if kind == 'sensor_unknown':
        return 'WAIT_LIDAR_EVIDENCE', advance.get('reason') or 'lidar_support_missing'
    if advance.get('swept_m', math.inf) <= advance['stop_m']:
        return 'LIDAR_NEAR_POINT_STOP', 'lidar_swept_point_blocked'
    return 'LIDAR_FRONT_STOP', 'lidar_front_sector_blocked'


def _reverse_escape_assessment(scan, pose, config, known_map, args):
    rear, support = _robust_sector_clearance(scan, 180.)
    swept = forward_swept_clearance(scan, half_width_m=config.swept_half_width_m, offset_x_m=args.lidar_offset_x,
                                   offset_y_m=args.lidar_offset_y, reverse=True)
    guard = _nearest_map_guard_pose(known_map, pose)
    map_m = _map_ray_clearance(known_map, guard, 180.)
    # Only guard the short escape and stopping margin, not a 20-cm route.
    endpoint = (guard.x_m - .12 * math.cos(guard.yaw_rad),
                guard.y_m - .12 * math.sin(guard.yaw_rad))
    map_clear = known_map.line_is_clear((guard.x_m, guard.y_m), endpoint)
    reason = ('scan_stale' if _scan_age_s(scan) > config.maximum_scan_age_s
              else 'scan_incomplete' if not _scan_has_data(scan)
              else 'rear_support_missing' if support < 3
              else 'rear_lidar_blocked' if not math.isfinite(rear) or rear < max(.60, args.collision_reverse_rear_clearance)
              else 'rear_swept_blocked' if not math.isfinite(swept) or swept < .50
              else 'rear_map_or_glass_guard' if map_m < .60 or not map_clear else '')
    return {'can_reverse': not bool(reason), 'reason': reason,
            'rear_lidar_m': rear, 'rear_support': support, 'rear_swept_m': swept,
            'rear_map_m': map_m, 'map_short_clear': bool(map_clear)}


def _short_reverse(client, iterator, localizer, scan, pose, config, known_map, args, status, should_stop):
    """Allow command latency/loaded start while rechecking every lidar frame."""
    client.stop()
    if should_stop() or not _reverse_escape_assessment(scan, pose, config, known_map, args)['can_reverse']:
        client.stop()
        if not should_stop():
            # An unavailable reverse must still consume a fresh stopped scan.
            # Returning the same scan let idle recovery reuse one close speckle
            # repeatedly without ever observing whether it had cleared.
            reference = scan
            scan = next(iterator)
            pose, _match, _accepted = _bounded_motion(localizer, reference, scan, pose)
        return scan, pose, 0.0
    moved = 0.0
    deadline = time.monotonic() + 0.60
    if not _send_twist(client, -max(config.slow_speed_mm_s, 360), 0, 600):
        return scan, pose, 0.0
    try:
        while not should_stop() and time.monotonic() < deadline and moved < 0.08:
            reference = scan
            scan = next(iterator)
            new_pose, match, accepted = _bounded_motion(localizer, reference, scan, pose)
            if not accepted:
                client.stop()
                scan = next(iterator)
                new_pose, match, accepted = _bounded_motion(localizer, reference, scan, pose)
                if accepted:
                    pose = new_pose
                    if abs(match.transform.yaw_rad) <= math.radians(12) and abs(match.transform.y_m) <= .03:
                        moved += max(0., -match.transform.x_m)
                break
            pose = new_pose
            # Rotation, lateral ICP drift and forward motion are not reverse
            # progress and must never replenish the escape allowance.
            if abs(match.transform.yaw_rad) > math.radians(12) or abs(match.transform.y_m) > .03:
                break
            moved += max(0., -match.transform.x_m)
            status(pose, "LIDAR_CLEARED_SHORT_REVERSE", scan)
            if not _reverse_escape_assessment(scan, pose, config, known_map, args)['can_reverse']:
                break
    finally:
        client.stop()
    return scan, pose, moved


def _candidate_quality(result):
    return bool(
        result is not None
        and math.isfinite(float(result.rmse_m))
        and math.isfinite(float(result.inlier_ratio))
        and float(result.rmse_m) <= 0.12
        and float(result.inlier_ratio) >= 0.55
        and all(math.isfinite(v) for v in (
            result.pose.x_m, result.pose.y_m, result.pose.yaw_rad))
    )


def _candidate_in_envelope(candidate, pose, config):
    return bool(
        _candidate_quality(candidate)
        and math.hypot(candidate.pose.x_m - pose.x_m, candidate.pose.y_m - pose.y_m)
        <= config.map_correction_radius_m
        and abs(_angle_delta(candidate.pose.yaw_rad, pose.yaw_rad)) <= math.radians(35.0)
    )


@dataclass(frozen=True)
class _CorrectionObservation:
    """Map estimate paired with the odometric pose of the SAME scan."""

    result: object
    odometry_pose: Pose2D

    @property
    def pose(self):
        return self.result.pose


def _correction_residual(first, second):
    expected = first.pose
    if isinstance(first, _CorrectionObservation) and isinstance(second, _CorrectionObservation):
        motion = first.odometry_pose.inverse().compose(second.odometry_pose)
        expected = expected.compose(motion)
    return (
        math.hypot(expected.x_m - second.pose.x_m, expected.y_m - second.pose.y_m),
        abs(_angle_delta(expected.yaw_rad, second.pose.yaw_rad)),
    )


def _correction_consistent(first, second):
    position_error, yaw_error = _correction_residual(first, second)
    return bool(
        position_error <= 0.14 and yaw_error <= math.radians(12.0)
    )


def _correction_rejection_reason(candidate, pose, radius_m, yaw_limit_deg):
    if candidate is None:
        return 'no_candidate'
    if not _candidate_quality(candidate):
        return 'candidate_quality_low'
    if math.hypot(candidate.pose.x_m - pose.x_m, candidate.pose.y_m - pose.y_m) > radius_m:
        return 'candidate_translation_outside_window'
    if abs(_angle_delta(candidate.pose.yaw_rad, pose.yaw_rad)) > math.radians(yaw_limit_deg):
        return 'candidate_yaw_outside_window'
    evidence = [getattr(candidate, key, None) for key in
                ('range_agreement', 'range_mean_error_m', 'range_coverage')]
    displacement = math.hypot(candidate.pose.x_m - pose.x_m, candidate.pose.y_m - pose.y_m)
    if displacement > .20 or any(value is not None for value in evidence):
        if not all(value is not None and math.isfinite(float(value)) for value in evidence):
            return 'candidate_metric_evidence_missing'
        if evidence[0] < .65 or evidence[1] > .22 or evidence[2] < .60:
            return 'candidate_metric_distance_mismatch'
    if displacement > .20:
        margin = float(getattr(candidate, 'ambiguity_margin', math.nan))
        if not math.isfinite(margin) or margin < .02:
            return 'candidate_uniqueness_unproven'
    return ''


def _candidate_diagnostic(candidate, pose):
    if candidate is None:
        return {}
    return {
        'candidate_pose': {'x_m': candidate.pose.x_m, 'y_m': candidate.pose.y_m,
                           'yaw_deg': math.degrees(candidate.pose.yaw_rad)},
        'candidate_rmse_m': candidate.rmse_m,
        'candidate_inlier_ratio': candidate.inlier_ratio,
        'candidate_ambiguity_margin': getattr(candidate, 'ambiguity_margin', None),
        'candidate_range_agreement': getattr(candidate, 'range_agreement', None),
        'candidate_range_mean_error_m': getattr(candidate, 'range_mean_error_m', None),
        'candidate_range_coverage': getattr(candidate, 'range_coverage', None),
        'candidate_translation_delta_m': math.hypot(
            candidate.pose.x_m - pose.x_m, candidate.pose.y_m - pose.y_m),
        'candidate_yaw_delta_deg': math.degrees(_angle_delta(candidate.pose.yaw_rad, pose.yaw_rad)),
    }


def _transport_candidate(candidate, origin, current):
    if candidate is None:
        return None
    aligned = candidate.pose.compose(origin.inverse().compose(current))
    return LocalizationResult(aligned, candidate.rmse_m, candidate.inlier_ratio,
                              getattr(candidate, 'score', 0.),
                              getattr(candidate, 'ambiguity_margin', math.inf),
                              getattr(candidate, 'range_agreement', None),
                              getattr(candidate, 'range_mean_error_m', None),
                              getattr(candidate, 'range_coverage', None))


def _stopped_heading_calibration(client, scan_iterator, localizer, scan, pose, config,
                                 should_stop=lambda: False):
    """One short two-scan yaw check; never issue a movement or global search.

    Keep translation unchanged by map fusion. Compensate real stopped-tail
    motion and apply at most 12 degrees of heading correction, without feeding
    that synthetic correction into the body-frame steering lock.
    """
    started = time.monotonic()
    deadline = started + 2.
    observations = []
    reason = 'bounded_localizer_unavailable'
    correction_deg = 0.
    candidate = None
    client.stop()
    bounded = getattr(localizer, 'localize_bounded', None)
    try:
        if bounded is None:
            return scan, pose, False
        for _ in range(2):
            if should_stop() or time.monotonic() >= deadline:
                reason = 'heading_check_cancelled_or_budget'
                return scan, pose, False
            scan, pose, fresh = _refresh_stopped_scan(
                client, scan_iterator, localizer, scan, pose, config, should_stop, deadline_s=deadline)
            if not fresh:
                reason = 'refresh_failed'
                return scan, pose, False
            scan_time = float(getattr(scan, 'acquired_monotonic_s', scan.timestamp_s))
            if observations and scan_time <= observations[-1][0]:
                reason = 'duplicate_scan'
                return scan, pose, False
            candidate = bounded(scan, pose, radius_m=.18, budget_s=.35, yaw_radius_deg=45.)
            reason = _correction_rejection_reason(candidate, pose, .18, 45.)
            if not reason and (candidate.rmse_m > .12 or candidate.inlier_ratio < .55):
                reason = 'heading_candidate_quality_low'
            if not reason and getattr(candidate, 'ambiguity_margin', math.inf) < .01:
                reason = 'heading_candidate_ambiguous'
            if reason:
                return scan, pose, False
            observations.append((scan_time, _CorrectionObservation(candidate, pose)))
        if not _correction_consistent(observations[0][1], observations[1][1]):
            reason = 'motion_compensated_candidates_inconsistent'
            return scan, pose, False
        observation = observations[-1][1]
        scan, pose, fresh = _refresh_stopped_scan(
            client, scan_iterator, localizer, scan, pose, config, should_stop, deadline_s=deadline)
        if not fresh or should_stop() or time.monotonic() >= deadline:
            reason = 'post_matching_refresh_failed_or_budget'
            return scan, pose, False
        aligned = observation.pose.compose(observation.odometry_pose.inverse().compose(pose))
        change = _angle_delta(aligned.yaw_rad, pose.yaw_rad) * .35
        change = max(-math.radians(12.), min(math.radians(12.), change))
        correction_deg = math.degrees(change)
        pose = Pose2D(pose.x_m, pose.y_m, normalize_angle(pose.yaw_rad + change))
        # Old candidate odometry frames are invalid after this fusion.
        reason = 'yaw_blended'
        return scan, pose, True
    finally:
        client.stop()
        localizer._panorama_heading_diagnostic = {
            'event': 'heading_calibration', 'result': reason,
            'elapsed_s': time.monotonic() - started, 'correction_deg': correction_deg,
            **_candidate_diagnostic(candidate, pose),
            'search_diagnostic': getattr(localizer, '_last_bounded_diagnostic', None),
        }
        if isinstance(client, _MotionAuditClient):
            client._append(localizer._panorama_heading_diagnostic, 'panorama_motion.jsonl')


def _refresh_stopped_scan(client, scan_iterator, localizer, scan, pose, config,
                          should_stop=lambda: False, deadline_s=None):
    """Refresh after blocking work, retaining the stopped-motion reference.

    Empty watchdog frames and stale queued frames are not obstacle evidence.
    Never send movement or apply an unaccepted transform while refreshing.
    """
    client.stop()
    reference = scan
    deadline = time.monotonic() + 2.
    if deadline_s is not None:
        deadline = min(deadline, deadline_s)
    for _ in range(6):
        if should_stop() or time.monotonic() >= deadline:
            break
        current = next(scan_iterator)
        if _scan_age_s(current) > config.maximum_scan_age_s or not _scan_has_data(current):
            continue
        refreshed_pose, _, accepted = _bounded_motion(localizer, reference, current, pose)
        if accepted:
            return current, refreshed_pose, True
    if isinstance(client, _MotionAuditClient):
        client.stop_event('stopped_refresh_failed', reference)
    return reference, pose, False


def _recheck_local_obstruction(client, scan_iterator, localizer, scan, pose,
                              config, assessment, should_stop=lambda: False):
    """Stop first; require two consecutive new acquisitions, never receipts.

    A clear result retains the full map/lidar guard. Persistent obstacles are
    not filtered away. Unknown frames or rejected odometry cannot authorize
    motion or a claim that the obstacle persists.
    """
    client.stop()
    deadline = time.monotonic() + 2.
    stamp = float(getattr(scan, 'acquired_monotonic_s', scan.timestamp_s) or 0.)
    last_kind, consecutive = None, 0
    for _ in range(6):
        if should_stop() or time.monotonic() >= deadline:
            break
        current = next(scan_iterator)
        current_stamp = float(getattr(current, 'acquired_monotonic_s', current.timestamp_s) or 0.)
        if (not math.isfinite(current_stamp) or current_stamp <= stamp
                or _scan_age_s(current) > config.maximum_scan_age_s
                or not _scan_has_data(current)):
            consecutive, last_kind = 0, None
            continue
        stamp = current_stamp
        new_pose, _, accepted = _bounded_motion(localizer, scan, current, pose)
        scan = current
        if not accepted:
            consecutive, last_kind = 0, None
            continue
        pose = new_pose
        kind = _front_blockage_kind(assessment(scan, pose))
        if kind == 'sensor_unknown':
            consecutive, last_kind = 0, None
            continue
        consecutive = consecutive + 1 if kind == last_kind else 1
        last_kind = kind
        if consecutive >= 2:
            return scan, pose, kind
    return scan, pose, 'sensor_unknown'


def _recover_pending_turn_heading(client, scan_iterator, localizer, scan, pose,
                                  config, should_stop=lambda: False):
    """Never drop an unresolved turn by restarting scans at the old yaw."""
    client.stop()
    pending = getattr(localizer, '_panorama_pending_motion', None)
    recover = getattr(localizer, 'recover_turn_heading', None)
    if pending is None or recover is None:
        return scan, pose, False
    reference, anchor_pose = pending
    deadline = time.monotonic() + 3.
    previous_yaw = None
    confirmations = 0
    stamp = float(getattr(scan, 'acquired_monotonic_s', scan.timestamp_s) or 0.)
    for _ in range(6):
        if should_stop() or time.monotonic() >= deadline:
            break
        scan = next(scan_iterator)
        current_stamp = float(getattr(scan, 'acquired_monotonic_s', scan.timestamp_s) or 0.)
        if (not math.isfinite(current_stamp) or current_stamp <= stamp
                or _scan_age_s(scan) > config.maximum_scan_age_s or not _scan_has_data(scan)):
            confirmations, previous_yaw = 0, None
            continue
        stamp = current_stamp
        try:
            fit = recover(reference, scan, anchor_pose)
        except TimeoutError:
            fit = None
        if (fit is None or not fit.converged or not math.isfinite(fit.rmse_m)
                or fit.rmse_m > .08 or not math.isfinite(fit.inlier_ratio)
                or fit.inlier_ratio < .70 or fit.correspondences < 60
                or _motion_translation_m(fit) > .001 or not math.isfinite(fit.transform.yaw_rad)):
            confirmations, previous_yaw = 0, None
            continue
        yaw = fit.transform.yaw_rad
        confirmations = confirmations + 1 if (previous_yaw is not None
            and abs(_angle_delta(yaw, previous_yaw)) <= math.radians(4.)) else 1
        previous_yaw = yaw
        # Three new stopped scans: two confirming observations plus a tail
        # check. Never reuse the frame which originally failed tracking.
        if confirmations >= 3 and not should_stop() and time.monotonic() < deadline:
            pose = anchor_pose.compose(Pose2D(yaw_rad=yaw))
            localizer._panorama_pending_motion = None
            localizer._panorama_pending_rotation_only = False
            localizer._panorama_local_heading_lock = None
            return scan, pose, True
    return scan, pose, False


def _turn_pulse_ttl_ms(remaining, speed, observed_rate, starting, stalled=0):
    # Preserve torque output, but bound its duration by the requested angle.
    # A 350-ms startup pulse at 3000 mrad/s can itself turn about 60 degrees.
    rate = max(abs(speed) / 1000., observed_rate * 1.4, .1)
    angle_budget = max(math.radians(3.), remaining - math.radians(4.))
    ttl = max(120, min(350 if starting else 220, int(angle_budget / rate * 700)))
    if starting and stalled >= 2:
        # Loaded static friction needs a finite uninterrupted torque window.
        # Boost duration only after fresh accepted scans show no rotation;
        # once moving, return to the angle/rate-limited short pulse.
        ttl = max(ttl, min(220, 120 + 25 * stalled))
    return ttl


def _turn_increment(client, scan_iterator, localizer, scan, pose, requested_rad,
                    config, update_status, motion_is_safe=None,
                    turn_state='PANORAMA_TURN', translation_is_safe=None):
    previous = getattr(localizer, '_panorama_rotation_only', False)
    localizer._panorama_rotation_only = True
    try:
        return _turn_increment_impl(client, scan_iterator, localizer, scan, pose, requested_rad,
            config, update_status, motion_is_safe, turn_state, translation_is_safe)
    finally:
        localizer._panorama_rotation_only = previous


def _turn_increment_impl(
    client,
    scan_iterator,
    localizer,
    scan,
    pose,
    requested_rad,
    config,
    update_status,
    motion_is_safe=None,
    turn_state="PANORAMA_TURN",
    translation_is_safe=None,
):
    client.stop()
    direction = 1 if requested_rad >= 0.0 else -1
    required = abs(float(requested_rad))
    accumulated = 0.0
    stalled = 0
    steps = 0
    deadline = time.monotonic() + min(16.0, max(5.0, math.degrees(required) / 8.0))
    turn_started_s = time.monotonic()
    last_measurement_s = turn_started_s
    observed_rate = 0.
    while accumulated < required - math.radians(4.0) and time.monotonic() < deadline:
        if _scan_age_s(scan) > config.maximum_scan_age_s or (isinstance(client, _MotionAuditClient) and not _scan_has_data(scan)):
            before_refresh_yaw = pose.yaw_rad
            scan, pose, fresh = _refresh_stopped_scan(client, scan_iterator, localizer, scan, pose, config)
            if not fresh:
                break
            stopped_yaw = _angle_delta(pose.yaw_rad, before_refresh_yaw)
            if stopped_yaw * direction > 0:
                accumulated += abs(stopped_yaw)
            if accumulated >= required - math.radians(7):
                break
        if motion_is_safe is not None and not motion_is_safe(scan, pose):
            if isinstance(client, _MotionAuditClient):
                client.stop_event('turn_clearance_blocked', scan)
            else:
                client.stop()
            # Stop immediately, then allow up to three stopped frames to
            # distinguish transient boundary noise from persistent obstacles.
            # Never keep output energized while waiting for confirmation.
            for _ in range(3):
                reference = scan
                scan = next(scan_iterator)
                recovered_pose, guard_match, guard_ok = _bounded_motion(localizer, reference, scan, pose)
                if not guard_ok:
                    return scan, pose, direction * accumulated, False
                pose = recovered_pose
                guard_yaw = float(guard_match.transform.yaw_rad)
                if guard_yaw * direction > 0.0:
                    accumulated += abs(guard_yaw)
                if motion_is_safe(scan, pose):
                    break
            else:
                update_status(pose, "TURN_CLEARANCE_STOP", scan)
                return scan, pose, direction * accumulated, False
        remaining = required - accumulated
        speed = config.turn_speed_mrad_s
        if stalled >= 2:
            speed = config.turn_max_speed_mrad_s
        if remaining < math.radians(18.0):
            speed = min(speed, max(1800, int(config.turn_speed_mrad_s * 0.78)))
        before = scan
        local_turn = turn_state == "LOCAL_HEADING_TURN"
        # A finite loaded-start window, renewed only with fresh safe scans;
        # not 50-ms pulses which stop before static friction is overcome.
        ttl_ms = _turn_pulse_ttl_ms(remaining, speed, observed_rate,
                                  accumulated < math.radians(.8), stalled)
        if not _send_twist(client, 0, direction * speed, ttl_ms):
            if getattr(client, 'current_reason', '') == 'scan_budget_too_short':
                previous_pose = pose
                scan, pose, refreshed = _refresh_stopped_scan(
                    client, scan_iterator, localizer, before, pose, config)
                if not refreshed:
                    return scan, pose, direction * accumulated, False
                tail_yaw = normalize_angle(pose.yaw_rad - previous_pose.yaw_rad)
                if tail_yaw * direction > 0.:
                    accumulated += abs(tail_yaw)
                update_status(pose, 'TURN_WAIT_FRESH_SCAN', scan)
                continue
            break
        scan = next(scan_iterator)
        pose, match, accepted = _bounded_motion(localizer, before, scan, pose)
        if not accepted:
            # Do not send another turn while its previous angle is unknown.
            # A stopped rescan against the same reference can recover the
            # entire missing increment without losing it from the accumulator.
            client.stop()
            recovery_scan = next(scan_iterator)
            pose, match, accepted = _bounded_motion(
                localizer, before, recovery_scan, pose
            )
            scan = recovery_scan
            if not accepted:
                break
        measured = 0.0
        if accepted:
            measured = float(match.transform.yaw_rad)
            if measured * direction > 0.0:
                accumulated += abs(measured)
        measured_at = time.monotonic()
        dt = max(.02, measured_at - last_measurement_s)
        last_measurement_s = measured_at
        observed_rate = .5 * observed_rate + .5 * min(4., abs(measured) / dt)
        lead = max(math.radians(4), min(math.radians(12), observed_rate * .10))
        can_leave = local_turn and translation_is_safe is not None and translation_is_safe(scan, pose)
        brake_now = required - accumulated <= lead or can_leave
        if brake_now and accepted:
            client.stop()
            # Consume the stopped tail before the next decision so residual
            # rotation is not lost when the following command takes its anchor.
            stopped_scan = next(scan_iterator)
            stopped_pose, tail, tail_ok = _bounded_motion(localizer, scan, stopped_scan, pose)
            scan = stopped_scan
            if not tail_ok:
                break
            pose = stopped_pose
            tail_yaw = float(tail.transform.yaw_rad)
            measured += tail_yaw
            if tail_yaw * direction > 0:
                accumulated += abs(tail_yaw)
            if can_leave or accumulated >= required - math.radians(7):
                update_status(pose, turn_state, scan)
                break
        if abs(measured) < math.radians(0.8):
            stalled += 1
        else:
            stalled = 0
        steps += 1
        update_status(pose, turn_state, scan)
        if (stalled >= 6 and time.monotonic() - turn_started_s >= config.drive_progress_window_s) or (steps >= 28 and time.monotonic() - turn_started_s >= config.drive_progress_window_s):
            break
    client.stop()
    success = accumulated >= required - math.radians(7.0)
    return scan, pose, direction * accumulated, success


def run_open_space_panorama(
    *,
    args,
    known_map,
    localizer,
    initial_pose,
    initial_scan,
    scan_iterator,
    client,
    camera,
    report,
    online_qwen,
    should_stop,
):
    """Execute the no-waypoint open-space panorama state machine."""
    config = PanoramaConfig.from_args(args)
    config.validate()
    localizer._panorama_pending_motion = None
    localizer._panorama_pending_rotation_only = False
    localizer._panorama_motion_diagnostic = None
    localizer._panorama_local_heading_lock = None
    localizer._panorama_turn_since_calibration_rad = 0.
    localizer._panorama_rotation_only = False
    client = _MotionAuditClient(client, config, report.directory)
    # Use one consistent scan stream for clearance, odometry and corrections.
    initial_scan = _ReceivedScan(_filter_body_returns(
        initial_scan, config.self_return_radius_m,
        args.lidar_offset_x, args.lidar_offset_y,
        config.self_return_half_width_m,
    ))
    raw_scan_iterator = scan_iterator
    def received_scans():
        for source in raw_scan_iterator:
            current = _ReceivedScan(_filter_body_returns(source, config.self_return_radius_m,
                args.lidar_offset_x, args.lidar_offset_y, config.self_return_half_width_m))
            client.scan = current
            yield current
    scan_iterator = received_scans()
    client.scan = initial_scan
    if config.self_return_radius_m > 0:
        print(
            "车体回波过滤：只排除雷达后半圈、半径 {:.2f}m 且半宽 {:.3f}m 内的点；"
            "独立于近似车身尺寸；初始排除 {} 点。遮挡仍视为未知，遮罩外障碍保护保留。".format(
                config.self_return_radius_m, config.self_return_half_width_m,
                getattr(initial_scan, "body_return_count", 0)
            ),
            flush=True,
        )
    # Leave a small coverage margin for bounded pose/FOV errors. The required
    # effective coverage remains the user's target, not a fixed photo count.
    coverage_stop_threshold = min(1.0, config.coverage_target + 0.02)
    coverage = SceneCoverageTracker(known_map)
    # A photographed sector can discourage redundant exploration without
    # becoming verified map coverage or a trusted semantic coordinate.
    exploration_coverage = SceneCoverageTracker(known_map)
    selector = OpenSpaceDirectionSelector()
    pose = initial_pose
    scan = initial_scan
    trail = deque((pose,), maxlen=1200)
    stations = []
    station_attempt_count = 0
    partial_station_origins = []
    deferred_annotations = []
    diagnostics_path = report.directory / "panorama_navigation.jsonl"
    live_path = report.directory / "live_pose.json"
    live_map_path = report.directory / "inspection_plan.png"
    coverage_path = report.directory / "panorama_coverage.json"
    correction_candidates = deque(maxlen=config.map_correction_confirmations)
    correction_due = False
    correction_attempts = 0
    distance_since_correction = 0.0
    distance_since_station = 0.0
    selected_heading = None
    locked_heading_selection = None
    state = "SEARCH_OPEN_SPACE"
    last_status_s = -math.inf
    last_render_s = -math.inf
    mission_started_s = time.monotonic()
    obstacle_reverse_attempts = 0
    spacing_backtrack_attempted = False
    last_trusted_map_pose = initial_pose
    correction_failure_count = 0
    translation_progress_pose = pose
    last_translation_progress_s = mission_started_s
    local_turn_steps = 0
    escape_turn_budget = _EscapeTurnBudget()
    course_turn_budget = _CourseTurnBudget()
    obstruction_unknown_started_s = None
    obstruction_recheck_pending = False
    forward_distance_since_escape_reset = 0.0
    departure_distance_m = 0.0
    last_guard_correction_s = -math.inf
    finish_reason = "requested_stop"
    progress_window = _DriveProgressWindow(config)
    drive_recovery_count = 0
    pending_recovery_started_s = None
    odometry_reanchors = 0
    pose_uncertain = False
    no_safe_exit_started_s = None
    stale_started_s = None
    last_map_conflict_s = -math.inf
    last_periodic_correction_s = -math.inf
    last_map_only_check_s = -math.inf
    reverse_retry_after_s = -math.inf
    last_reverse_denied_s = -math.inf
    last_heading_calibration_s = -math.inf

    def offer_correction(candidate, window=None, yaw_limit_deg=35.):
        """Log acceptance separately from search success; compare scan-time poses."""
        if window is None:
            window = correction_candidates
        reason = _correction_rejection_reason(
            candidate, pose, config.map_correction_radius_m, yaw_limit_deg)
        residual = {}
        if not reason:
            observation = _CorrectionObservation(candidate, pose)
            if window:
                position_error, yaw_error = _correction_residual(window[-1], observation)
                residual = {'motion_compensated_position_error_m': position_error,
                            'motion_compensated_yaw_error_deg': math.degrees(yaw_error)}
                if not _correction_consistent(window[-1], observation):
                    window.clear()
                    reason = 'motion_compensated_candidates_inconsistent'
            window.append(observation)
        else:
            window.clear()
        client._append({'event': 'map_correction_validation',
                        'result': reason or ('confirmed_window' if len(window) >= config.map_correction_confirmations
                                             else 'waiting_for_confirmation'),
                        'candidate_count': len(window), 'yaw_limit_deg': yaw_limit_deg,
                        **_candidate_diagnostic(candidate, pose), **residual}, 'panorama_motion.jsonl')
        return not reason or reason == 'motion_compensated_candidates_inconsistent'

    def attempt_reverse_escape(reason, assessment):
        nonlocal scan, pose, obstacle_reverse_attempts, reverse_retry_after_s
        nonlocal distance_since_correction, selected_heading, local_turn_steps
        nonlocal forward_distance_since_escape_reset, drive_recovery_count, correction_due
        nonlocal no_safe_exit_started_s
        nonlocal last_reverse_denied_s
        if should_stop() or obstacle_reverse_attempts >= 3 or time.monotonic() < reverse_retry_after_s:
            return False
        rear = _reverse_escape_assessment(scan, pose, config, known_map, args)
        if not rear['can_reverse']:
            if time.monotonic() - last_reverse_denied_s >= 2.:
                last_reverse_denied_s = time.monotonic()
                client._append({'event': 'reverse_escape_denied', 'cause': reason,
                                'monotonic_s': last_reverse_denied_s, 'rear_escape': rear}, 'panorama_motion.jsonl')
            return False
        client.stop_event(reason, scan, {**assessment, 'rear_escape': rear})
        progress_window.pause()
        obstacle_reverse_attempts += 1
        scan, pose, reverse_distance = _short_reverse(
            client, scan_iterator, localizer, scan, pose, config, known_map, args,
            update_status, should_stop)
        distance_since_correction += reverse_distance
        trail.append(pose)
        selected_heading = None
        localizer._panorama_local_heading_lock = None
        # Backward progress restores turn capacity, but never the independent
        # reverse-attempt allowance. Only 20 cm of forward travel restores it.
        if reverse_distance >= .04:
            local_turn_steps = 0
            escape_turn_budget.reset()
            course_turn_budget.reset()
            forward_distance_since_escape_reset = 0.
            drive_recovery_count = 0
            correction_due = True
            no_safe_exit_started_s = None
        reverse_retry_after_s = time.monotonic() + 2.
        client._append({'event': 'reverse_escape_result', 'cause': reason,
                        'distance_m': reverse_distance, 'attempt': obstacle_reverse_attempts,
                        'turn_budget_reset': reverse_distance >= .04}, 'panorama_motion.jsonl')
        update_status(pose, 'REVERSE_FIRST_REASSESS', scan)
        return True

    def stopped_map_correction(radius_m):
        nonlocal scan, pose, last_translation_progress_s
        started = time.monotonic()
        candidate = None
        matching_pose = pose
        result_reason = 'refresh_failed'
        client.stop()
        progress_window.pause()
        try:
            scan, pose, fresh = _refresh_stopped_scan(
                client, scan_iterator, localizer, scan, pose, config, should_stop)
            if fresh and not should_stop():
                # Do not fall back to the startup all-heading search.
                bounded_localize = getattr(localizer, 'localize_bounded', None)
                result_reason = 'bounded_localizer_unavailable'
                if bounded_localize is not None:
                    matching_started = time.monotonic()
                    matching_pose = pose
                    candidate = bounded_localize(scan, pose, radius_m=radius_m, budget_s=.35)
                    result_reason = 'candidate' if candidate is not None else 'no_candidate_or_budget'
                    if time.monotonic() - matching_started > .5:
                        candidate = None
                        result_reason = 'budget_overrun'
            # Fetch a new scan before handing control back to motion. A
            # receipt timestamp must not hide acquisition age during matching.
            if fresh and not should_stop():
                scan, pose, refreshed = _refresh_stopped_scan(
                    client, scan_iterator, localizer, scan, pose, config, should_stop)
                if not refreshed:
                    candidate = None
                    result_reason = 'post_matching_refresh_failed'
            candidate = _transport_candidate(candidate, matching_pose, pose)
            return candidate
        finally:
            stopped_duration = time.monotonic() - started
            # Exclude intentional stopped processing, not turning, failed
            # driving, or repeated attempts to escape a physical obstacle.
            last_translation_progress_s += stopped_duration
            client._append({'event': 'map_correction', 'elapsed_s': stopped_duration,
                            'result': result_reason,
                            'search_diagnostic': getattr(localizer, '_last_bounded_diagnostic', None),
                            **_candidate_diagnostic(candidate, pose)}, 'panorama_motion.jsonl')

    def turn_is_safe(current_scan, current_pose):
        if should_stop() or _scan_age_s(current_scan) > config.maximum_scan_age_s or not _scan_has_data(current_scan):
            return False
        assessment = _rotation_assessment(current_scan, known_map, current_pose, config)
        if assessment['clearance_m'] < config.rotation_clearance_m:
            client._panorama_turn_guard = assessment
            return False
        client._panorama_turn_guard = None
        return True

    def update_status(current_pose, current_state, current_scan, extra=None):
        nonlocal last_status_s, last_render_s
        now = time.monotonic()
        clearances = directional_clearances(
            current_scan, known_map, current_pose
        )
        payload = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "monotonic_s": now,
            "scan_acquired_monotonic_s": getattr(current_scan, 'timestamp_s', None) or None,
            "scan_received_monotonic_s": getattr(current_scan, 'received_monotonic_s', None),
            "scan_age_s": _scan_age_s(current_scan),
            "motion_command": client.last,
            "render_elapsed_s": client.last_render_s,
            "drive_recovery_count": drive_recovery_count,
            "odometry_reanchors": odometry_reanchors,
            "pose_uncertain": pose_uncertain,
            "mode": "open_space_panorama",
            "physical_footprint": {
                "length_m": config.body_length_m, "width_m": config.body_width_m,
                "margin_m": config.footprint_margin_m,
                "swept_half_width_m": config.swept_half_width_m,
                "rotation_clearance_m": config.rotation_clearance_m,
            },
            "state": current_state,
            "pose_source": "high_rate_scan_odometry_with_low_rate_map_correction",
            "x_m": round(current_pose.x_m, 4),
            "y_m": round(current_pose.y_m, 4),
            "yaw_deg": round(math.degrees(current_pose.yaw_rad), 2),
            "last_trusted_map_pose": {
                "x_m": round(last_trusted_map_pose.x_m, 4),
                "y_m": round(last_trusted_map_pose.y_m, 4),
                "yaw_deg": round(math.degrees(last_trusted_map_pose.yaw_rad), 2),
            },
            "localization_confidence_score": round(min(.45 if pose_uncertain else 1., max(
                0.25,
                0.95 - distance_since_correction * 0.10
                - min(0.30, correction_failure_count * 0.05),
            )), 3),
            "distance_since_station_m": round(distance_since_station, 3),
            "distance_since_map_correction_m": round(distance_since_correction, 3),
            "map_correction_due": bool(correction_due),
            "panorama_station_count": len(stations),
            "station_departure_remaining_m": round(departure_distance_m, 3),
            "local_escape_turn_steps": local_turn_steps,
            'local_escape_budget': {**escape_turn_budget.snapshot(now),
                'legacy_limit_reason': escape_turn_budget.reason(now),
                'exhaustion_reason': '', 'enforced': False},
            'local_escape_limit_policy': 'telemetry_only_no_mission_cutoff',
            'local_course_budget': course_turn_budget.snapshot(now),
            'local_direction_policy': selected_heading.get('selection_policy') if selected_heading else None,
            'local_direction_revisit_cost': selected_heading.get('revisit_cost') if selected_heading else None,
            "coverage_ratio": round(coverage.ratio, 5),
            'provisional_photo_coverage_ratio': round(exploration_coverage.ratio, 5),
            "directional_clearance_m": {
                name: round(float(item["effective_m"]), 3)
                for name, item in clearances.items()
            },
            "directional_clearance_details": {
                name: {
                    key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                    for key, value in item.items()
                }
                for name, item in clearances.items()
            },
            "body_return_filter": {
                "independent_of_collision_footprint": True,
                "radius_m": config.self_return_radius_m,
                "half_width_m": config.self_return_half_width_m,
                "excluded_point_count": getattr(current_scan, "body_return_count", 0),
                "rear_half_only": True,
            },
            "front_swept_clearance_m": forward_swept_clearance(
                current_scan, half_width_m=config.swept_half_width_m, offset_x_m=args.lidar_offset_x,
                offset_y_m=args.lidar_offset_y,
            ),
            "front_swept_details": _swept_clearance_details(current_scan,
                config.swept_half_width_m, config.lidar_offset_x_m, config.lidar_offset_y_m),
            "rotation_guard": _rotation_assessment(current_scan, known_map, current_pose, config),
        }
        if extra:
            payload.update(extra)
        motion_diagnostic = getattr(localizer, "_panorama_motion_diagnostic", None)
        if motion_diagnostic is not None:
            payload["scan_odometry"] = motion_diagnostic
        payload['last_heading_calibration'] = getattr(localizer, '_panorama_heading_diagnostic', None)
        heading_lock = getattr(localizer, "_panorama_local_heading_lock", None)
        payload["local_heading_error_deg"] = (
            round(math.degrees(heading_lock.error_rad), 2) if heading_lock is not None else None
        )
        _atomic_json(live_path, _json_finite(payload))
        if now - last_render_s >= 0.50:
            render_started = time.monotonic()
            _render_live_map(
                known_map, live_map_path, current_pose, trail, stations, coverage
            )
            client.last_render_s = time.monotonic() - render_started
            last_render_s = now
        if now - last_status_s >= config.status_interval_s:
            print(
                "全景巡检：state={} pose=({:+.2f},{:+.2f},{:+.1f}°) "
                "距上站={:.2f}m 覆盖={:.1f}% 站点={}。".format(
                    current_state,
                    current_pose.x_m,
                    current_pose.y_m,
                    math.degrees(current_pose.yaw_rad),
                    distance_since_station,
                    coverage.ratio * 100.0,
                    len(stations),
                ),
                flush=True,
            )
            last_status_s = now
        with diagnostics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_json_finite(payload), ensure_ascii=False, allow_nan=False) + "\n")

    def capture_station(current_pose, current_scan):
        nonlocal scan, station_attempt_count
        station_attempt_count += 1
        station_index = station_attempt_count
        station_start_yaw = current_pose.yaw_rad
        accumulated_yaw = 0.0
        photo_count = int(round(360.0 / config.angular_step_deg))
        completed = True
        for photo_index in range(photo_count):
            client.stop()
            if should_stop():
                completed = False
                break
            time.sleep(min(0.8, max(0.2, float(args.settle_seconds) * 0.35)))
            # Integrate the stopping tail before writing a photo's direction.
            # The last scan during a turn can precede delivery of STOP.
            before_settle_yaw = current_pose.yaw_rad
            current_scan, settled_pose, settled_ok = _refresh_stopped_scan(
                client, scan_iterator, localizer, current_scan, current_pose, config, should_stop)
            if not settled_ok:
                completed = False
                break
            current_pose = settled_pose
            if photo_index == 0:
                station_start_yaw = current_pose.yaw_rad
            else:
                accumulated_yaw += _angle_delta(current_pose.yaw_rad, before_settle_yaw)
            image_name = "panorama_s{:02d}_d{:03d}.jpg".format(
                station_index,
                int(round(photo_index * config.angular_step_deg)),
            )
            image_path = report.directory / image_name
            camera_info = camera.capture(image_path)
            clearances = directional_clearances(
                current_scan, known_map, current_pose
            )
            camera_yaw = normalize_angle(
                current_pose.yaw_rad + math.radians(args.camera_yaw_offset_deg)
            )
            newly_seen, provisional_newly_seen = _observe_photo_coverage(
                coverage, exploration_coverage, current_pose, camera_yaw, config, pose_uncertain
            )
            target_xy = (
                current_pose.x_m + 2.0 * math.cos(camera_yaw),
                current_pose.y_m + 2.0 * math.sin(camera_yaw),
            )
            target = InspectionTarget(
                target_id="panorama_s{:02d}_d{:03d}".format(
                    station_index,
                    int(round(photo_index * config.angular_step_deg)),
                ),
                kind="panorama_sector",
                target_xy_m=target_xy,
                view_xy_m=(current_pose.x_m, current_pose.y_m),
                yaw_rad=camera_yaw,
                component_cells=0,
            )
            record = {
                "target_id": target.target_id,
                "kind": target.kind,
                "image": image_name,
                "photo_index": photo_index + 1,
                "photos_per_target": photo_count,
                "panorama_station_id": station_index,
                "relative_panorama_yaw_deg": round(
                    math.degrees(accumulated_yaw), 2
                ),
                "x_m": round(current_pose.x_m, 4),
                "y_m": round(current_pose.y_m, 4),
                "yaw_deg": round(math.degrees(camera_yaw), 2),
                "target_x_m": round(target_xy[0], 4),
                "target_y_m": round(target_xy[1], 4),
                "localization_confidence": "uncertain_reanchor" if pose_uncertain else "odometry_bounded",
                "localization_confidence_score": round(min(.45 if pose_uncertain else 1., max(
                    0.25,
                    0.95 - distance_since_correction * 0.10
                    - min(0.30, correction_failure_count * 0.05),
                )), 3),
                "low_confidence_pose": bool(
                    pose_uncertain or distance_since_correction > config.map_correction_distance_m * 2.0
                    or correction_failure_count >= 3
                ),
                "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "front_clearance_m": round(
                    clearances["front"]["effective_m"], 3
                ),
                "new_coverage_samples": newly_seen,
                'provisional_new_coverage_samples': provisional_newly_seen,
                'provisional_coverage_ratio_after_capture': round(exploration_coverage.ratio, 6),
                "coverage_ratio_after_capture": round(coverage.ratio, 6),
                "capture_mode": "open_space_panorama",
                "capture_note": "direction_recorded; map_read_only",
                "inspection_subject": "whole_scene_sector",
                "review_question": "识别该方向可见物体并保留不确定候选",
                **camera_info,
            }
            report.add(record)
            deferred_annotations.append((target, record, image_path))
            update_status(current_pose, "PANORAMA_CAPTURE", current_scan)
            if photo_index + 1 >= photo_count:
                break
            before_refresh_yaw = current_pose.yaw_rad
            current_scan, current_pose, refreshed = _refresh_stopped_scan(
                client, scan_iterator, localizer, current_scan, current_pose, config, should_stop)
            if not refreshed:
                completed = False
                break
            accumulated_yaw += _angle_delta(current_pose.yaw_rad, before_refresh_yaw)
            current_scan, current_pose, measured, success = _turn_increment(
                client,
                scan_iterator,
                localizer,
                current_scan,
                current_pose,
                math.radians((photo_index + 1) * config.angular_step_deg)
                - accumulated_yaw,
                config,
                update_status,
                turn_is_safe,
            )
            accumulated_yaw += measured
            trail.append(current_pose)
            if not success:
                completed = False
                print(
                    "全景站点 {} 转向没有取得足够实际角度；已停车，"
                    "保留已拍照片但不虚报完整一圈。".format(station_index),
                    flush=True,
                )
                break
        # The photographs already cover the circle. Do not add a compulsory
        # return-to-start turn before departing; retain the actual final yaw.
        client.stop()
        station_pose = Pose2D(
            current_pose.x_m,
            current_pose.y_m,
            station_start_yaw,
        )
        return current_scan, current_pose, station_pose, completed

    update_status(pose, state, scan)
    while not should_stop():
        if _scan_age_s(scan) > config.maximum_scan_age_s or not _scan_has_data(scan):
            client.stop_event('scan_stale_or_incomplete', scan)
            progress_window.pause()
            if stale_started_s is None:
                stale_started_s = time.monotonic()
            if time.monotonic() - stale_started_s >= 3.0:
                finish_reason = 'lidar_data_unavailable'
                break
            scan = next(scan_iterator)
            update_status(pose, 'WAIT_FRESH_SCAN', scan)
            continue
        stale_started_s = None
        if getattr(localizer, "_panorama_pending_motion", None) is not None:
            client.stop()
            progress_window.pause()
            if pending_recovery_started_s is None:
                pending_recovery_started_s = time.monotonic()
            reference = scan
            scan = next(scan_iterator)
            pose, _pending_match, pending_ok = _bounded_motion(localizer, reference, scan, pose)
            update_status(pose, "STOPPED_ODOMETRY_RECOVERY", scan)
            if not pending_ok:
                if time.monotonic() - pending_recovery_started_s >= 1.0:
                    if getattr(localizer, '_panorama_pending_rotation_only', False):
                        scan, pose, heading_recovered = _recover_pending_turn_heading(
                            client, scan_iterator, localizer, scan, pose, config, should_stop)
                        pose_uncertain = True
                        selected_heading = None
                        correction_candidates.clear()
                        client._append({'event': 'lost_turn_heading_recovery',
                            'recovered': heading_recovered,
                            'diagnostic': getattr(localizer, '_panorama_turn_recovery_diagnostic', None)},
                            'panorama_motion.jsonl')
                        update_status(pose, 'TURN_HEADING_RECOVERED' if heading_recovered
                                      else 'TURN_HEADING_UNRESOLVED', scan)
                        if not heading_recovered:
                            finish_reason = 'turn_heading_unresolved'
                            break
                        pending_recovery_started_s = None
                        correction_due = True
                        continue
                    odometry_reanchors += 1
                    if odometry_reanchors > 3:
                        finish_reason = 'odometry_unavailable_after_stopped_recovery'
                        break
                    # Do not apply the rejected transform. Restart relative
                    # tracking from a stopped scan, with explicit uncertainty.
                    localizer._panorama_pending_motion = None
                    localizer._panorama_pending_rotation_only = False
                    localizer._panorama_local_heading_lock = None
                    selected_heading = None
                    pending_recovery_started_s = None
                    correction_failure_count += 1
                    pose_uncertain = True
                    correction_candidates.clear()
                    update_status(pose, 'ODOMETRY_REANCHORED_UNCERTAIN', scan)
                continue
            pending_recovery_started_s = None
            selected_heading = None
        if (getattr(localizer, '_panorama_turn_since_calibration_rad', 0.) >= math.radians(45.)
                and time.monotonic() - last_heading_calibration_s >= 8.):
            started_heading_check = time.monotonic()
            progress_window.pause()
            update_status(pose, 'STOPPED_HEADING_CHECK', scan)
            scan, pose, heading_corrected = _stopped_heading_calibration(
                client, scan_iterator, localizer, scan, pose, config, should_stop)
            last_heading_calibration_s = time.monotonic()
            localizer._panorama_turn_since_calibration_rad = 0.
            last_translation_progress_s += last_heading_calibration_s - started_heading_check
            escape_turn_budget.pause(last_heading_calibration_s - started_heading_check)
            course_turn_budget.pause(last_heading_calibration_s - started_heading_check)
            if heading_corrected:
                correction_candidates.clear()
            print('转弯后短时朝向校准：{}；不额外转圈、不等待重定位。'.format(
                localizer._panorama_heading_diagnostic['result']), flush=True)
            update_status(pose, 'HEADING_CHECK_COMPLETE', scan)
            continue
        now = time.monotonic()
        if math.hypot(pose.x_m - translation_progress_pose.x_m,
                      pose.y_m - translation_progress_pose.y_m) >= 0.08:
            translation_progress_pose = pose
            last_translation_progress_s = now
        elif now - last_translation_progress_s >= 45.0:
            client.stop()
            finish_reason = "no_translation_progress"
            print(
                "局部恢复 45 秒未产生至少 0.08m 的位置进展；转角变化不算脱困。"
                "已停车，后方雷达遮挡时不会依赖旧图盲目倒车。",
                flush=True,
            )
            break
        if stations:
            distance_since_station = math.hypot(
                pose.x_m - stations[-1].x_m,
                pose.y_m - stations[-1].y_m,
            )
        elapsed = time.monotonic() - mission_started_s
        if elapsed >= float(args.max_runtime_min) * 60.0:
            finish_reason = "runtime_limit"
            print("开阔点全景巡检达到运行时间上限，停车并保存未覆盖区域。")
            break
        if (
            len(stations) >= config.minimum_stations
            and coverage.ratio >= coverage_stop_threshold
        ):
            finish_reason = "coverage_complete"
            print(
                "照片覆盖率 {:.1f}% 已达到目标 {:.1f}%，结束运动。".format(
                    coverage.ratio * 100.0,
                    coverage_stop_threshold * 100.0,
                ),
                flush=True,
            )
            break
        if len(stations) >= config.maximum_stations:
            finish_reason = "station_limit"
            print(
                "已达到最多 {} 个全景站点；停止运动并保留未覆盖区域清单。"
                .format(config.maximum_stations),
                flush=True,
            )
            break

        clearances = directional_clearances(scan, known_map, pose)
        openness = open_space_assessment(clearances, config)
        suitable_open_space = bool(
            openness["is_open"]
            or (
                distance_since_station > config.maximum_station_spacing_m
                and openness["minimum_clearance_m"] >= 0.45
                and openness["open_direction_count"] >= 2
            )
        )
        station_ready = (
            suitable_open_space
            and departure_distance_m <= 0.0
            and all(
                math.hypot(pose.x_m - x_m, pose.y_m - y_m) >= 0.45
                for x_m, y_m in partial_station_origins
            )
            and (
                not stations
                or min(
                    math.hypot(pose.x_m - old.x_m, pose.y_m - old.y_m)
                    for old in stations
                ) >= config.minimum_station_spacing_m
            )
        )
        if station_ready:
            state = "PANORAMA_CAPTURE"
            localizer._panorama_local_heading_lock = None
            client.stop()
            progress_window.pause()
            localizer._panorama_rotation_only = True
            try:
                scan, pose, station_pose, completed = capture_station(pose, scan)
            finally:
                localizer._panorama_rotation_only = False
            departure_distance_m = .80
            local_turn_steps = 0
            escape_turn_budget.reset()
            course_turn_budget.reset()
            forward_distance_since_escape_reset = 0.0
            drive_recovery_count = 0
            # Intentional stationary photography is not a driving stall.
            last_translation_progress_s = time.monotonic()
            translation_progress_pose = pose
            if completed:
                stations.append(station_pose)
                selector.mark_station(station_pose)
                distance_since_station = 0.0
                spacing_backtrack_attempted = False
                selected_heading = None
                correction_due = True
                correction_attempts = 0
                correction_candidates.clear()
                # The station is the preferred low-rate correction location:
                # collect only a small stopped window before driving again.
                client.stop()
                update_status(pose, "PERIODIC_LOCALIZATION", scan)
                stopped_candidates = []
                for candidate_index in range(config.map_correction_confirmations):
                    if candidate_index:
                        scan = next(scan_iterator)
                    candidate = stopped_map_correction(config.map_correction_radius_m)
                    offer_correction(candidate, stopped_candidates)
                if (
                    len(stopped_candidates) == config.map_correction_confirmations
                    and all(
                        _correction_consistent(
                            stopped_candidates[index - 1],
                            stopped_candidates[index],
                        )
                        for index in range(1, len(stopped_candidates))
                    )
                ):
                    last_trusted_map_pose = stopped_candidates[-1].pose
                    pose = _blend_pose(
                        pose, last_trusted_map_pose, config.map_correction_gain
                    )
                    correction_failure_count = 0
                    pose_uncertain = False
                else:
                    correction_failure_count += 1
                    print(
                        "全景后地图候选未连续一致；保留里程计位姿继续，"
                        "不执行全局重新定位。",
                        flush=True,
                    )
                correction_due = False
                distance_since_correction = 0.0
            else:
                # A partial station does not satisfy station count or reset
                # spacing; choose a new open area instead of retrying in place.
                selected_heading = None
                partial_station_origins.append((pose.x_m, pose.y_m))
            _atomic_json(coverage_path, coverage.summary())
            update_status(pose, "DEPART_STATION", scan)
            continue

        if (
            distance_since_station > config.maximum_station_spacing_m
            and not spacing_backtrack_attempted
        ):
            spacing_backtrack_attempted = True
            rear_m = clearances["rear"]["effective_m"]
            if rear_m >= max(0.75, float(args.collision_reverse_rear_clearance)):
                state = "BACKTRACK_TO_OPEN_SPACE"
                scan, pose, reverse_distance = _short_reverse(client, scan_iterator, localizer, scan, pose, config, known_map, args, update_status, should_stop)
                distance_since_correction += reverse_distance
                trail.append(pose)
                update_status(pose, state, scan)
            selected_heading = None

        heading_lock = getattr(localizer, "_panorama_local_heading_lock", None)
        # A short escape can be safe even when no long candidate course is
        # available. Check live obstruction before asking for a new direction.
        early_advance = _advance_assessment(scan, pose, known_map, config, speed_mm_s=config.slow_speed_mm_s)
        if obstruction_recheck_pending or _front_blockage_kind(early_advance) in ('lidar_obstacle', 'sensor_unknown'):
            obstruction_recheck_pending = True
            client.stop_event('local_obstruction_recheck', scan, early_advance)
            progress_window.pause()
            scan, pose, rechecked_kind = _recheck_local_obstruction(
                client, scan_iterator, localizer, scan, pose, config,
                lambda s, p: _advance_assessment(s, p, known_map, config,
                    speed_mm_s=config.slow_speed_mm_s), should_stop)
            trail.append(pose)
            update_status(pose, 'LOCAL_OBSTRUCTION_RECHECK', scan,
                          {'obstruction_recheck_result': rechecked_kind})
            if rechecked_kind == 'sensor_unknown':
                if obstruction_unknown_started_s is None:
                    obstruction_unknown_started_s = time.monotonic()
                if time.monotonic() - obstruction_unknown_started_s >= 3.:
                    finish_reason = 'obstruction_evidence_unavailable'
                    break
                continue
            obstruction_unknown_started_s = None
            obstruction_recheck_pending = False
            early_advance = _advance_assessment(scan, pose, known_map, config,
                                               speed_mm_s=config.slow_speed_mm_s)
        if (_front_blockage_kind(early_advance) == 'lidar_obstacle'
                and attempt_reverse_escape('front_obstacle_reverse_first', early_advance)):
            continue
        if selected_heading is locked_heading_selection and heading_lock is not None and heading_lock.expired(time.monotonic()):
            selected_heading = None
        if selected_heading is None:
            localizer._panorama_local_heading_lock = None
            locked_heading_selection = None
            state = "SELECT_DIRECTION"
            selected_heading = selector.choose(
                scan, pose, known_map, exploration_coverage, config
            )
            if selected_heading is None:
                client.stop()
                progress_window.pause()
                if no_safe_exit_started_s is None:
                    no_safe_exit_started_s = time.monotonic()
                if time.monotonic() - no_safe_exit_started_s >= 8.0:
                    finish_reason = 'no_safe_exit'
                    break
                # The odometric icon can drift just outside the map's
                # inflated free mask even while lidar shows real free space.
                # Treat this as one low-rate correction trigger, not as a
                # reason to globally replan or spin in place.
                # This trigger is rate limited even if the guard stays blocked.
                now = time.monotonic()
                if now - last_guard_correction_s < 5.0:
                    scan = next(scan_iterator)
                    update_status(pose, state, scan)
                    continue
                last_guard_correction_s = now
                candidate = stopped_map_correction(config.map_correction_radius_m)
                if offer_correction(candidate):
                    if (
                        len(correction_candidates)
                        >= config.map_correction_confirmations
                        and all(
                            _correction_consistent(
                                correction_candidates[index - 1],
                                correction_candidates[index],
                            )
                            for index in range(1, len(correction_candidates))
                        )
                    ):
                        pose = _blend_pose(
                            pose,
                            correction_candidates[-1].pose,
                            config.map_correction_gain,
                        )
                        last_trusted_map_pose = correction_candidates[-1].pose
                        correction_failure_count = 0
                        correction_candidates.clear()
                        distance_since_correction = 0.0
                else:
                    correction_candidates.clear()
                print(
                    "当前没有满足永久地图与雷达净空的局部方向；"
                    "停车采集低频局部纠错候选，不执行全局重规划。",
                    flush=True,
                )
                scan = next(scan_iterator)
                update_status(pose, state, scan)
                continue
            print(
                "局部方向选择：相对 {:+.0f}°，净空 {:.2f}m，"
                "预计新增覆盖采样 {}；不生成目标点或全局路径。".format(
                    selected_heading["relative_heading_deg"],
                    selected_heading["clearance_m"],
                    selected_heading["expected_new_samples"],
                ),
                flush=True,
            )

        if selected_heading is not locked_heading_selection:
            # Select with current lidar, then keep the course in the body frame.
            # Absolute map yaw remains useful for coverage, not alignment control.
            localizer._panorama_local_heading_lock = _LocalHeadingLock(
                math.radians(selected_heading["relative_heading_deg"])
            )
            locked_heading_selection = selected_heading
        heading_lock = localizer._panorama_local_heading_lock
        no_safe_exit_started_s = None
        heading_error = heading_lock.error_rad
        advance = _advance_assessment(scan, pose, known_map, config, speed_mm_s=config.slow_speed_mm_s)
        blockage_kind = _front_blockage_kind(advance)
        if blockage_kind == 'map_only' and time.monotonic() - last_map_only_check_s >= 5.:
            client.stop_event('map_only_check_pose', scan, advance)
            candidate = stopped_map_correction(config.map_correction_radius_m)
            last_map_only_check_s = time.monotonic()
            if offer_correction(candidate):
                if (len(correction_candidates) >= config.map_correction_confirmations
                        and all(_correction_consistent(a, b) for a, b in
                                zip(list(correction_candidates)[:-1], list(correction_candidates)[1:]))):
                    last_trusted_map_pose = correction_candidates[-1].pose
                    pose = _blend_pose(pose, last_trusted_map_pose, config.map_correction_gain)
                    correction_candidates.clear()
                    correction_failure_count = 0
                    pose_uncertain = False
                    selected_heading = None
            else:
                correction_candidates.clear()
                correction_failure_count += 1
            update_status(pose, 'MAP_ONLY_POSE_CHECK', scan)
            # Freshly corrected geometry is evaluated on the next decision;
            # never reverse just because an old-map ray is short.
            continue
        map_conflict = advance['lidar_m'] - advance['map_m'] > .60 and advance['swept_m'] - advance['map_m'] > .60
        if map_conflict and time.monotonic() - last_map_conflict_s >= 5.0:
            last_map_conflict_s = time.monotonic()
            correction_due = True
            pose_uncertain = True
            client._append({'event': 'map_lidar_conflict', 'monotonic_s': last_map_conflict_s,
                'x_m': pose.x_m, 'y_m': pose.y_m, 'yaw_rad': pose.yaw_rad,
                'assessment': advance, 'status': 'suspicious_not_map_overwrite'}, 'panorama_map_conflicts.jsonl')
        front_m = advance['effective_m']
        turn_clearance_m = _rotation_clearance(scan, known_map, pose, config)
        blind_stop_distance_m = advance['stop_m']
        # Obstruction prohibits translation, not a lidar-cleared pure turn.
        # Otherwise the stop branch continually discards escape headings.
        action = _local_motion_action(front_m if advance['can_advance'] else 0., blind_stop_distance_m, turn_clearance_m,
                                      heading_error, 0,
                                      config.rotation_clearance_m)
        if (selected_heading.get('selection_policy') == 'leave_visited_area'
                and advance['can_advance']
                and abs(heading_error) > math.radians(25.)
                and turn_clearance_m >= config.rotation_clearance_m
                and not course_turn_budget.reason(time.monotonic())):
            action = 'turn'
        if action == "drive" and (abs(heading_error) > math.radians(35)
                or (selected_heading.get('selection_policy') == 'leave_visited_area'
                    and course_turn_budget.reason(time.monotonic()))):
            # Do not chase a rearward coverage target with a long arc/circle.
            # Commit to the actually clear front, then reassess after moving.
            heading_lock.error_rad = 0.0
            heading_error = 0.0
            selected_heading["world_heading_rad"] = pose.yaw_rad
            selected_heading["relative_heading_deg"] = 0.0
        if action == "turn":
            progress_window.pause()
            course_alignment = bool(advance['can_advance'])
            budget = course_turn_budget if course_alignment else escape_turn_budget
            state = "ALIGN_CLEAR_COURSE" if course_alignment else "TURN_TO_LOCAL_DIRECTION"
            if not course_alignment:
                local_turn_steps += 1
            budget.begin(time.monotonic())
            turn_amount = math.copysign(min(abs(heading_error), math.radians(20.0),
                budget.maximum_angle_rad - budget.angle_rad if course_alignment else math.radians(20.)), heading_error)
            scan, pose, _measured, _success = _turn_increment(
                client, scan_iterator, localizer, scan, pose, turn_amount,
                config, update_status, turn_is_safe,
                "LOCAL_COURSE_ALIGNMENT" if course_alignment else "LOCAL_HEADING_TURN",
                None if course_alignment else lambda s, p: _advance_assessment(s, p, known_map, config,
                    speed_mm_s=config.slow_speed_mm_s)['can_advance'],
            )
            budget.observe(_measured)
            trail.append(pose)
            continue
        if front_m <= blind_stop_distance_m or not advance['can_advance']:
            stop_state, stop_reason = _local_stop_diagnostic(advance)
            client.stop_event(stop_reason, scan, advance)
            progress_window.pause()
            # Cumulative turn history is diagnostic, not permission to move
            # and not a terminal condition. Stop, reject this course and
            # select from CURRENT safe geometry. Real no-progress, unavailable
            # sensors, and unresolved heading retain their independent guards.
            selector.reject_direction(
                pose, selected_heading["world_heading_rad"]
            )
            selected_heading = None
            state = stop_state
            scan = next(scan_iterator)
            update_status(pose, state, scan)
            continue
        state = "DEPART_STATION" if departure_distance_m > 0.0 else "BLIND_DRIVE"
        speed = config.drive_speed_mm_s
        if (
            front_m < 0.90
            or abs(heading_error) > math.radians(20.0)
            or distance_since_station >= config.ideal_station_spacing_m
            or selected_heading.get("local_backtrack")
        ):
            speed = config.slow_speed_mm_s
        if selected_heading.get("local_backtrack"):
            speed = max(config.slow_speed_mm_s, 300)
        if progress_window.starting(time.monotonic()):
            # Highest bounded loaded-start output that the same safety guard
            # permits; e.g. reduce 420 to 400 rather than 240 at 0.78 m.
            for startup_speed in range(config.startup_speed_mm_s, config.slow_speed_mm_s - 1, -20):
                if _advance_assessment(scan, pose, known_map, config,
                        speed_mm_s=startup_speed)['can_advance']:
                    speed = startup_speed
                    state = 'STARTING_DRIVE'
                    break
        speed_advance = _advance_assessment(scan, pose, known_map, config, speed_mm_s=speed)
        if not speed_advance['can_advance']:
            # Keep a genuinely safe low-speed course instead of selecting a
            # rearward coverage target just because startup output is unsafe.
            speed = config.slow_speed_mm_s
            speed_advance = advance
        left_m = clearances["front_left"]["effective_m"]
        right_m = clearances["front_right"]["effective_m"]
        # Positive angular velocity turns left, so steer toward the side with
        # more clearance rather than toward the nearer obstacle.
        angular = _local_drive_angular(heading_error, left_m, right_m)
        if turn_clearance_m < 0.40:
            angular = max(-350, min(350, angular))
        if angular > 0 and clearances["left"]["effective_m"] < 0.40:
            angular = 0
        elif angular < 0 and clearances["right"]["effective_m"] < 0.40:
            angular = 0
        guard_pose = _nearest_map_guard_pose(known_map, pose)
        short_endpoint = (
            guard_pose.x_m + speed_advance['horizon_m'] * math.cos(guard_pose.yaw_rad),
            guard_pose.y_m + speed_advance['horizon_m'] * math.sin(guard_pose.yaw_rad),
        )
        if not known_map.line_is_clear(
            (guard_pose.x_m, guard_pose.y_m), short_endpoint
        ):
            client.stop_event('map_short_segment_blocked', scan, speed_advance)
            progress_window.pause()
            selector.reject_direction(
                pose, selected_heading["world_heading_rad"]
            )
            selected_heading = None
            scan = next(scan_iterator)
            update_status(pose, "MAP_GUARD_STOP", scan)
            continue
        before = scan
        ttl_ms = (
            650
            if front_m >= 1.0 and not selected_heading.get("local_backtrack")
            else 350
        )
        ttl_ms = min(ttl_ms, int(config.maximum_scan_age_s * 1000))
        heading_lock.mark_drive(time.monotonic())
        if should_stop():
            client.stop()
            break
        if not _send_twist(client, speed, angular, ttl_ms):
            progress_window.pause()
            scan = next(scan_iterator)
            update_status(pose, 'WAIT_FRESH_SCAN', scan)
            continue
        progress_window.begin(client.last['issued_monotonic_s'])
        scan = next(scan_iterator)
        new_pose, match, accepted = _bounded_motion(localizer, before, scan, pose)
        if not accepted:
            client.stop_event('drive_odometry_rejected', scan, localizer._panorama_motion_diagnostic)
            progress_window.pause()
            recovery_scan = next(scan_iterator)
            new_pose, match, accepted = _bounded_motion(
                localizer, before, recovery_scan, pose
            )
            scan = recovery_scan
        if accepted:
            moved = _motion_translation_m(match)
            forward_moved = max(0., float(match.transform.x_m))
            departure_distance_m = max(0.0, departure_distance_m - forward_moved)
            forward_distance_since_escape_reset += forward_moved
            if forward_distance_since_escape_reset >= .20:
                local_turn_steps = 0
                escape_turn_budget.reset()
                course_turn_budget.reset()
                obstacle_reverse_attempts = 0
                forward_distance_since_escape_reset = 0.0
            pose = new_pose
            selector.record_translation(pose)
            distance_since_station += moved
            distance_since_correction += moved
            if moved >= .025:
                drive_recovery_count = 0
                odometry_reanchors = 0
        else:
            moved = 0.0
        trail.append(pose)
        motion_now = time.monotonic()
        energized_s = max(0., min(min(motion_now, client.last.get('stopped_monotonic_s', motion_now)) - client.last['issued_monotonic_s'],
            client.last['ttl_ms'] / 1000. + client.last['refresh_window_s']))
        motion_state = progress_window.observe(motion_now, moved, accepted, energized_s)
        if motion_state in ('stalled', 'unknown'):
            client.stop_event('drive_' + motion_state, scan, speed_advance)
            progress_window.pause()
            if motion_state == 'stalled':
                drive_recovery_count += 1
                if drive_recovery_count >= 3:
                    # Three complete energized windows, not a single slow
                    # startup frame. This is not a collision diagnosis.
                    if attempt_reverse_escape('persistent_drive_no_response', speed_advance):
                        continue
                    finish_reason = 'actuator_no_response'
                    break
                if drive_recovery_count >= 2:
                    selector.reject_direction(pose, selected_heading['world_heading_rad'])
                    selected_heading = None
            update_status(pose, 'RETRY_START' if motion_state == 'stalled' else 'MOTION_UNKNOWN', scan)
            continue

        if distance_since_correction >= config.map_correction_distance_m:
            correction_due = True
        if (correction_due and getattr(localizer, "_panorama_pending_motion", None) is None
                and time.monotonic() - last_periodic_correction_s >= 5.):
            client.stop()
            progress_window.pause()
            candidate = stopped_map_correction(
                min(.35, config.map_correction_radius_m) if map_conflict else config.map_correction_radius_m)
            last_periodic_correction_s = time.monotonic()
            correction_attempts += 1
            offer_correction(candidate, yaw_limit_deg=10.0 if map_conflict else 35.0)
            if (
                len(correction_candidates) >= config.map_correction_confirmations
                and all(
                    _correction_consistent(
                        correction_candidates[index - 1],
                        correction_candidates[index],
                    )
                    for index in range(1, len(correction_candidates))
                )
            ):
                candidate = correction_candidates[-1]
                before_correction = pose
                pose = _blend_pose(
                    pose, candidate.pose, config.map_correction_gain
                )
                last_trusted_map_pose = candidate.pose
                correction_failure_count = 0
                pose_uncertain = False
                print(
                    "低频地图纠错：连续候选一致，平滑修正 {:.2f}m/{:.1f}°，"
                    "融合比例 {:.0f}%。".format(
                        math.hypot(
                            candidate.pose.x_m - before_correction.x_m,
                            candidate.pose.y_m - before_correction.y_m,
                        ),
                        math.degrees(abs(_angle_delta(
                            candidate.pose.yaw_rad, before_correction.yaw_rad
                        ))),
                        config.map_correction_gain * 100.0,
                    ),
                    flush=True,
                )
                correction_due = False
                correction_attempts = 0
                correction_candidates.clear()
                distance_since_correction = 0.0
            elif correction_attempts >= 4:
                correction_failure_count += 1
                if correction_failure_count >= 3:
                    pose_uncertain = True
                print(
                    "低频地图纠错候选不连续；保留里程计位姿继续行驶，"
                    "不会跳到对称区域。",
                    flush=True,
                )
                correction_due = False
                correction_attempts = 0
                correction_candidates.clear()
                distance_since_correction = 0.0
        update_status(pose, state, scan)
        if accepted and moved >= 0.006 and selected_heading is not None:
            if selected_heading.get("local_backtrack"):
                origin = selected_heading["backtrack_origin_xy"]
                if math.hypot(pose.x_m - origin[0], pose.y_m - origin[1]) >= 0.30:
                    selected_heading = None

    client.stop()
    _atomic_json(coverage_path, coverage.summary())
    update_status(pose, "FINISH_MOTION", scan, {'finish_reason': finish_reason})
    motion_summary = {
        'mode': 'open_space_panorama', 'station_count': len(stations),
        'station_attempt_count': station_attempt_count, 'photo_count': len(deferred_annotations),
        'coverage': coverage.summary(), 'finish_reason': finish_reason,
        'provisional_photo_coverage': exploration_coverage.summary(),
        'motion_complete': finish_reason == 'coverage_complete',
        'pose_uncertain': pose_uncertain, 'annotation_pending': bool(deferred_annotations and online_qwen.url),
        'final_pose': {'x_m': pose.x_m, 'y_m': pose.y_m, 'yaw_deg': math.degrees(pose.yaw_rad)},
    }
    _atomic_json(report.directory / 'panorama_motion_summary.json', motion_summary)
    semantics = {
        "format_version": 1,
        "map": str(args.map),
        "map_is_read_only": True,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "observations": [],
    }
    if deferred_annotations and online_qwen.url:
        print(
            "小车已经停车；开始按拍摄顺序统一分析 {} 张全景照片。".format(
                len(deferred_annotations)
            ),
            flush=True,
        )
        for target, record, image_path in deferred_annotations:
            second_name = record.get("camera_2_image")
            if not second_name:
                inference = {
                    "enabled": True,
                    "promoted": False,
                    "error": "dual_camera_pair_required",
                }
            else:
                inference = online_qwen.analyze_panorama(
                    record,
                    image_path,
                    image_path.with_name(str(second_name)),
                )
            record["online_qwen"] = inference
            semantics["observations"].append({
                "target_id": target.target_id,
                "image": str(image_path.resolve()),
                "x_m": record["x_m"],
                "y_m": record["y_m"],
                "yaw_deg": record["yaw_deg"],
                "result": inference,
                "status": (
                    "confirmed" if inference.get("promoted")
                    else "suspicious_or_unconfirmed"
                ),
            })
            report.save()
    _atomic_json(report.directory / "panorama_semantics.json", semantics)
    result = {
        "mode": "open_space_panorama",
        "station_count": len(stations),
        "station_attempt_count": station_attempt_count,
        "photo_count": len(deferred_annotations),
        "coverage": coverage.summary(),
        "provisional_photo_coverage": exploration_coverage.summary(),
        "coverage_target": config.coverage_target,
        "coverage_stop_threshold": coverage_stop_threshold,
        "finish_reason": finish_reason,
        "motion_complete": finish_reason == "coverage_complete",
        "coverage_target_reached": coverage.ratio >= config.coverage_target,
        "map_was_not_rebuilt": True,
        "pose_uncertain": pose_uncertain,
        "final_pose": {
            "x_m": pose.x_m,
            "y_m": pose.y_m,
            "yaw_deg": math.degrees(pose.yaw_rad),
        },
    }
    _atomic_json(report.directory / "panorama_summary.json", result)
    return result
