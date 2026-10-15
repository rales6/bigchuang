"""Task-only visual pose corrections. No motors, SLAM writes or inspection imports.

Angles are degrees, positive counterclockwise in the map frame. Photo bearings
are positive left of the vehicle. Inputs and candidates refer to capture time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
from typing import Callable


def wrap(degrees: float) -> float:
    return (degrees + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class TaskPose:
    x_m: float
    y_m: float
    yaw_deg: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.x_m, self.y_m, self.yaw_deg)):
            raise ValueError('Pose must be finite')


@dataclass(frozen=True)
class PoseCandidate:
    pose: TaskPose
    radar_supported: bool = False
    name: str = 'odometry'


@dataclass(frozen=True)
class CameraCalibration:
    width_px: int
    fx_px: float
    cx_px: float
    yaw_offset_deg: float = 0.0
    verified: bool = False
    x_offset_m: float = 0.0
    y_offset_m: float = 0.0

    def __post_init__(self):
        if (self.width_px <= 0 or self.fx_px <= 0 or not all(math.isfinite(v)
                for v in (self.fx_px, self.cx_px, self.yaw_offset_deg, self.x_offset_m, self.y_offset_m))
                or not 0 <= self.cx_px <= self.width_px):
            raise ValueError('Invalid camera calibration')

    def bearing(self, bbox_1000):
        if len(bbox_1000) != 4 or not all(math.isfinite(v) and 0 <= v <= 1000 for v in bbox_1000):
            raise ValueError('Invalid normalized image box')
        left, top, right, bottom = bbox_1000
        if left >= right or top >= bottom:
            raise ValueError('Image box must have positive area')
        u = (left + right) * self.width_px / 2000
        return wrap(self.yaw_offset_deg + math.degrees(math.atan((self.cx_px - u) / self.fx_px)))


@dataclass(frozen=True)
class SemanticDetection:
    category: str
    bearing_deg: float
    confidence: float
    object_id: str | None = None
    bearing_calibrated: bool = False
    camera_x_m: float = 0.0
    camera_y_m: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.bearing_deg, self.confidence, self.camera_x_m, self.camera_y_m)) or not 0 <= self.confidence <= 1:
            raise ValueError('Detection contains invalid numeric values')


@dataclass(frozen=True)
class LocalizationConfig:
    min_confidence: float = .65
    min_anchor_separation_deg: float = 20.0
    max_residual_deg: float = 12.0
    candidate_yaw_refinement_deg: float = 45.0
    max_candidate_translation_m: float = .6
    max_candidate_yaw_deg: float = 65.0
    max_step_translation_m: float = .15
    max_step_yaw_deg: float = 8.0
    confirm_frames: int = 2
    max_photo_age_s: float = 30.0
    min_frame_interval_s: float = .4
    ambiguity_margin_deg: float = 3.0

    def __post_init__(self):
        if (not 0 < self.min_confidence <= 1 or self.confirm_frames < 2
                or any(not math.isfinite(v) or v <= 0 for key, v in asdict(self).items()
                       if key not in {'min_confidence', 'confirm_frames'})):
            raise ValueError('Invalid localization thresholds')


class TaskSemanticLocalizer:
    """Resolve nearby radar hypotheses or refine yaw, requiring independent frames.

    Class names cannot identify repeated outlets/cabinets. Explicit object IDs
    must come from a verified reference match, not arbitrary Qwen assignment.
    Translation needs a radar-supported candidate AND an injected swept-footprint
    validator. A successful decision is a proposal; the task executor applies it.
    """
    def __init__(self, memory: dict, map_sha256: str, config=None,
                 pose_validator: Callable[[TaskPose, TaskPose], bool] | None = None):
        if memory.get('schema') != 'semantic_object_memory.v1' or memory.get('map_sha256') != map_sha256:
            raise ValueError('Semantic memory schema/map hash mismatch')
        self.config = config or LocalizationConfig()
        self.pose_validator = pose_validator
        self.anchors = []
        for item in memory.get('objects', []):
            xy = item.get('map_xy')
            if (item.get('position_verified') is True and item.get('localization_allowed') is True
                    and item.get('mobility') == 'fixed' and item.get('behind_glass') is not True
                    and isinstance(xy, (list, tuple)) and len(xy) == 2
                    and all(isinstance(v, (int, float)) and math.isfinite(v) for v in xy)):
                self.anchors.append(item)
        if len({a['id'] for a in self.anchors}) != len(self.anchors):
            raise ValueError('Duplicate verified anchor IDs')
        self._pending = None
        self._count = 0
        self._last_frame = None
        self._last_capture_s = None

    def _reset(self):
        self._pending = None
        self._count = 0

    def _associate(self, detections):
        rows = {}
        for detection in detections:
            if not detection.bearing_calibrated or detection.confidence < self.config.min_confidence:
                continue
            matches = [a for a in self.anchors if a.get('category') == detection.category
                       and (detection.object_id is None or a['id'] == detection.object_id)]
            if len(matches) != 1:
                continue  # Repeated class is not a unique physical landmark.
            anchor = matches[0]
            prior = rows.get(anchor['id'])
            if prior is None or detection.confidence > prior[1].confidence:
                rows[anchor['id']] = (anchor, detection)
        return list(rows.values())

    def evaluate(self, *, frame_id: str, captured_at_s: float, now_s: float,
                 capture_pose: TaskPose, current_pose: TaskPose,
                 detections: list[SemanticDetection], candidates: list[PoseCandidate]):
        cfg = self.config
        result = {'accepted': False, 'pose': asdict(current_pose), 'reason': '',
                  'matched_ids': [], 'rankings': [], 'confirmation_frames': 0,
                  'action': 'keep_existing_task_local_control', 'motor_command': None}

        def reject(reason, reset=True):
            if reset:
                self._reset()
            result['reason'] = reason
            return result

        if not frame_id or not all(math.isfinite(v) for v in (captured_at_s, now_s)):
            return reject('invalid_photo_timestamp')
        if frame_id == self._last_frame or (self._last_capture_s is not None
                and captured_at_s - self._last_capture_s < cfg.min_frame_interval_s):
            return reject('duplicate_or_out_of_order_frame', reset=False)
        self._last_frame, self._last_capture_s = frame_id, captured_at_s
        if not 0 <= now_s - captured_at_s <= cfg.max_photo_age_s:
            return reject('stale_photo')
        if (math.hypot(current_pose.x_m - capture_pose.x_m, current_pose.y_m - capture_pose.y_m) > .1
                or abs(wrap(current_pose.yaw_deg - capture_pose.yaw_deg)) > 10):
            return reject('vehicle_moved_since_capture')
        pairs = self._associate(detections)
        result['matched_ids'] = [a['id'] for a, _ in pairs]
        if len(pairs) < 2:
            return reject('insufficient_verified_calibrated_anchors')
        if max(abs(wrap(d.bearing_deg - e.bearing_deg)) for _, d in pairs for _, e in pairs) < cfg.min_anchor_separation_deg:
            return reject('poor_bearing_geometry')
        pool = [PoseCandidate(capture_pose)] + list(candidates)
        evaluated = []
        for candidate in pool:
            pose = candidate.pose
            distance = math.hypot(pose.x_m - capture_pose.x_m, pose.y_m - capture_pose.y_m)
            if distance > cfg.max_candidate_translation_m or abs(wrap(pose.yaw_deg - capture_pose.yaw_deg)) > cfg.max_candidate_yaw_deg:
                continue
            if distance > .001 and (len(pairs) < 3 or not candidate.radar_supported or self.pose_validator is None):
                continue
            def residuals_at(yaw):
                values = []
                radians = math.radians(yaw)
                for anchor, detection in pairs:
                    camera_x = pose.x_m + detection.camera_x_m * math.cos(radians) - detection.camera_y_m * math.sin(radians)
                    camera_y = pose.y_m + detection.camera_x_m * math.sin(radians) + detection.camera_y_m * math.cos(radians)
                    dx, dy = anchor['map_xy'][0] - camera_x, anchor['map_xy'][1] - camera_y
                    if math.hypot(dx, dy) < .2:
                        return []
                    expected = math.degrees(math.atan2(dy, dx)) - yaw
                    values.append(wrap(expected - detection.bearing_deg))
                return values

            yaw = pose.yaw_deg
            for _ in range(6):
                residuals = residuals_at(yaw)
                if not residuals:
                    break
                origin = residuals[0]
                yaw = wrap(yaw + wrap(statistics.median(origin + wrap(r - origin) for r in residuals)))
            residuals = residuals_at(yaw)
            if not residuals or abs(wrap(yaw - pose.yaw_deg)) > cfg.candidate_yaw_refinement_deg:
                continue
            refined = TaskPose(pose.x_m, pose.y_m, yaw)
            if abs(wrap(refined.yaw_deg - capture_pose.yaw_deg)) > cfg.max_candidate_yaw_deg:
                continue
            errors = [abs(r) for r in residuals]
            score = sum(errors) / len(errors)
            if max(errors) > cfg.max_residual_deg:
                continue  # Do not hide an inconsistent landmark behind a median.
            if distance > .001 and not self.pose_validator(capture_pose, refined):
                continue
            if any(math.hypot(refined.x_m - row[1].x_m, refined.y_m - row[1].y_m) < .02
                   and abs(wrap(refined.yaw_deg - row[1].yaw_deg)) < 1 for row in evaluated):
                continue
            evaluated.append((score, refined, candidate.name))
        evaluated.sort(key=lambda row: row[0])
        result['rankings'] = [{'mean_bearing_error_deg': s, 'pose': asdict(p), 'source': n} for s, p, n in evaluated]
        if not evaluated:
            return reject('no_consistent_nearby_candidate')
        if len(evaluated) > 1 and evaluated[1][0] - evaluated[0][0] < cfg.ambiguity_margin_deg:
            return reject('ambiguous_pose_candidates')
        target = evaluated[0][1]
        signature = (target, tuple(sorted(result['matched_ids'])))
        if (self._pending is not None and self._pending[1] == signature[1]
                and math.hypot(target.x_m - self._pending[0].x_m, target.y_m - self._pending[0].y_m) < .15
                and abs(wrap(target.yaw_deg - self._pending[0].yaw_deg)) < 8):
            self._count += 1
        else:
            self._count = 1
        self._pending = signature
        result['confirmation_frames'] = self._count
        if self._count < cfg.confirm_frames:
            return reject('waiting_for_independent_confirmation', reset=False)
        dx, dy = target.x_m - capture_pose.x_m, target.y_m - capture_pose.y_m
        length = math.hypot(dx, dy)
        scale = min(1., cfg.max_step_translation_m / length) if length else 1.
        yaw_delta = max(-cfg.max_step_yaw_deg, min(cfg.max_step_yaw_deg, wrap(target.yaw_deg - capture_pose.yaw_deg)))
        corrected = TaskPose(current_pose.x_m + dx * scale, current_pose.y_m + dy * scale, wrap(current_pose.yaw_deg + yaw_delta))
        if length > .001 and not self.pose_validator(current_pose, corrected):
            return reject('corrected_pose_fails_footprint_validation')
        result.update(accepted=True, pose=asdict(corrected), reason='bounded_semantic_correction',
                      action='task_executor_may_apply_pose_proposal')
        self._reset()
        return result


class TaskPhotoSchedule:
    """Low frequency, distance/time/event triggers. Failed vision never spins car."""
    def __init__(self, interval_s=45., distance_m=1.5, cooldown_s=10.):
        self.interval_s, self.distance_m, self.cooldown_s = interval_s, distance_m, cooldown_s
        self.last_request_s = None
        self.last_distance_m = 0.

    def should_capture(self, now_s, accumulated_distance_m, *, event=None, stationary=False):
        if not stationary:
            return False
        if self.last_request_s is not None and now_s - self.last_request_s < self.cooldown_s:
            return False
        return (self.last_request_s is None or now_s - self.last_request_s >= self.interval_s
                or accumulated_distance_m - self.last_distance_m >= self.distance_m
                or event in {'arrival', 'heading_discrepancy', 'radar_ambiguity'})

    def mark_requested(self, now_s, accumulated_distance_m):
        self.last_request_s, self.last_distance_m = now_s, accumulated_distance_m


def detections_from_qwen(objects, calibrations: dict[int, CameraCalibration]):
    """Convert inventory observations. Uncalibrated camera boxes are not bearings.

    object_id is deliberately ignored: generic Qwen output cannot establish IDs.
    Reference-matched IDs may be supplied separately as SemanticDetection inputs.
    """
    detections = []
    for item in objects:
        for view in item.get('views', []):
            calibration = calibrations.get(view.get('image_index'))
            if calibration is None or not calibration.verified:
                continue
            try:
                detections.append(SemanticDetection(item['category'],
                    calibration.bearing(view.get('bbox_1000', view.get('bbox_2d', []))),
                    float(item.get('confidence', 0)), bearing_calibrated=True,
                    camera_x_m=calibration.x_offset_m, camera_y_m=calibration.y_offset_m))
            except (KeyError, TypeError, ValueError):
                continue
    return detections
