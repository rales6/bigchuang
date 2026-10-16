"""Near-field, stop-and-look adjustment into the arm's *4D* teaching hull.

This is deliberately separate from pursuit: its observations are the same
640x480 grab_visual camera stream and Qwen/model service used by the arm.
Require BOTH independent taught 2D polygons or their 25px shells. The
original joint 4D hull is retained for pose projection, not a parking veto.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

TRACK_ROOT = Path(__file__).resolve().parents[1]
if str(TRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACK_ROOT))

from raspberry_pi.config import LidarConfig
from raspberry_pi.lidar.n10_driver import N10LidarDriver
from raspberry_pi.mapping.scan_filter import sector_min_distance
from pursuit.pursuit_config import PursuitConfig
from pursuit.forward_reference import reference_error,reference_payload
from pursuit.visual_approach import approach_evidence,recent_forward_valid
from integrations.grab_visual.teaching_region import polygon_distance, region_accepts


GRAB_URL = "http://127.0.0.1:8080"
MIN_FRONT_M = 0.25
MIN_REAR_M = 0.35
REAR_SECTOR_USABLE = False  # operator confirmed the chassis occludes this sector
_APPROACH_PROFILE = PursuitConfig()
PULSE_SPEED_MM_S = _APPROACH_PROFILE.visual_pulse_mm_s
PULSE_TURN_MRAD_S = _APPROACH_PROFILE.align_turn_pulse_mrad_s
APPROACH_HOLD_S = min(0.80, _APPROACH_PROFILE.visual_pulse_near_s)
PULSE_FIRST_S = _APPROACH_PROFILE.hull_micro_step_s
PULSE_STEP_S = _APPROACH_PROFILE.hull_micro_step_increment_s
PULSE_MAX_S = _APPROACH_PROFILE.hull_micro_step_max_s
REST_S = _APPROACH_PROFILE.visual_pulse_rest_s
TRIAL_BAND_PX = 60.0  # bounded recovery band, NOT a grasp tolerance
CLOSE_DISTANCE_PX = 60.0
TRIAL_FIRST_S = PULSE_FIRST_S
LINEAR_STARTUP_S = 0.0  # not a measured quantity; never force a startup floor
TURN_PULSE_S = 0.40
CLOSE_MAX_S = PULSE_MAX_S
TRIAL_MAX_S = PULSE_MAX_S
MAX_TOTAL_PULSE_S = 8.0
MAX_CORRECTABLE_WORSENING_PX = 60.0
MAX_CONSECUTIVE_CORRECTIONS = 3
MAX_LOW_CONFIDENCE_PROBES = 2
MIN_PROGRESS_PX = 2.0
SWEPT_RETREAT_MAX_PULSES = 2
SWEPT_RETREAT_MAX_ON_TIME_S = 1.28
TOKEN_PATH = Path(os.environ.get(
    "FINAL_WORK_HARDWARE_TOKEN_FILE",
    "/home/robot/dachuang/track/final_work/data/hardware_token"))


def visual_adjustment_phase(observation: dict, previous: str | None = None) -> str:
    """Fore/aft entry by image Y alone; independent of X and hull distance."""
    center = _center_vector(observation)
    margin = _APPROACH_PROFILE.hull_micro_exit_row_margin_px if previous=='micro' else 0.0
    if (center[1] >= _APPROACH_PROFILE.hull_micro_enter_left_y_px-margin and
            center[3] >= _APPROACH_PROFILE.hull_micro_enter_right_y_px-margin):
        return 'micro'
    return 'approach'


def approach_adjustment(observation: dict, *, duration_s: float,
                        maximum_s: float, blocked_actions: set) -> AdjustmentPlan | None:
    """Approach relative to calibrated body-forward lines, not hull X.

    Disabled calibration retains the legacy centre-third replay rule.
    """
    left_x,left_y,right_x,right_y = _center_vector(observation)
    if _APPROACH_PROFILE.forward_reference_enabled:
        left_error=reference_error(_APPROACH_PROFILE,'left',(left_x,left_y),640,480)
        right_error=reference_error(_APPROACH_PROFILE,'right',(right_x,right_y),640,480)
        common_error=(left_error+right_error)/2
        deadband=max(_APPROACH_PROFILE.angle_deadband_px*640/1280,
                     _APPROACH_PROFILE.hull_approach_turn_trigger_px)
        action=('turn_left' if common_error < -deadband else
                'turn_right' if common_error > deadband else 'forward')
        source='approach_body_line_'+action
    elif left_x < 640/3 and right_x < 640/3:
        action, source = 'turn_left', 'approach_recenter_left'
    elif left_x > 1280/3 and right_x > 1280/3:
        action, source = 'turn_right', 'approach_recenter_right'
    else:
        action, source = 'forward', 'approach_centre_third_forward'
    if action in blocked_actions:
        return None
    return AdjustmentPlan(action,min(maximum_s,max(PULSE_FIRST_S,duration_s)),source)


def verify_model_camera_order() -> None:
    """Model-coordinate audit matches physical-left/right input video2/video0.

    The old fixed-0/2 assumption yielded 239.7px for the recorded final target,
    versus 43.4px in 2/0 order. This check is Pi-only.
    """
    if sys.platform != "linux":
        return
    from pursuit.camera_contract import model_devices
    expected = model_devices(require_ports=True)
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / "cmdline").read_bytes().decode(
                "utf-8", "replace").split("\0")
        except (OSError, PermissionError):
            continue
        if not any(arg.endswith("/grab_visual/web_app.py") for arg in args):
            continue
        def camera(flag: str, default: int) -> int:
            equals = next((arg.split("=", 1)[1] for arg in args
                           if arg.startswith(flag + "=")), None)
            if equals is not None:
                return int(equals)
            if flag not in args:
                return default
            index = args.index(flag) + 1
            return int(args[index]) if index < len(args) else -1
        left, right = camera("--left-camera", 0), camera("--right-camera", 2)
        if (left, right) != expected:
            raise RuntimeError(
                f"抓取服务相机顺序为 video{left}/video{right}，"
                f"但模型绑定的物理左右为 video{expected[0]}/video{expected[1]}；禁止微调和夹取")
        return
    raise RuntimeError("无法确认抓取服务的左右相机顺序，禁止微调和夹取")


def _get_json(url: str, timeout: float = 4.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict, timeout: float = 100.0,
               token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if token is not None:
        headers["X-Robot-Safety-Token"] = token
    request = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _center_vector(observation: dict) -> tuple[float, float, float, float]:
    centers = observation.get("centers") or {}
    try:
        values = (*centers["left"], *centers["right"])
        if len(values) != 4 or not all(math.isfinite(float(x)) for x in values):
            raise ValueError("invalid centers")
        return tuple(float(x) for x in values)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("双目目标中心无效，禁止底盘微调") from exc


def _hull_vector(observation: dict) -> tuple[float, float, float, float]:
    nearest = (observation.get('view_hull_centers') if observation.get('grasp_region_mode')=='dual_2d_shell'
               else observation.get("nearest_hull_centers")) or {}
    try:
        values = (*nearest["left"], *nearest["right"])
        if len(values) != 4 or not all(math.isfinite(float(x)) for x in values):
            raise ValueError("invalid hull point")
        return tuple(float(x) for x in values)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("示教凸包最近点缺失，禁止底盘微调") from exc


def _weighted(vector) -> tuple[float, ...]:
    # Match model_state's metric: x/640, y/480, then residual * 640.
    return tuple(value * (640.0 / 480.0 if index % 2 else 1.0)
                 for index, value in enumerate(vector))


def _norm(vector) -> float:
    return math.sqrt(sum(value * value for value in vector))


@dataclass(frozen=True)
class AdjustmentPlan:
    action: str
    duration_s: float
    source: str
    predicted_improvement_px: float | None = None


def in_grasp_region(observation: dict) -> bool:
    """Accept model-declared jitter margin, never arbitrary outside projection."""
    if observation.get('grasp_region_mode')=='dual_2d_shell':
        return region_accepts(observation,_APPROACH_PROFILE.hull_grasp_margin_px)
    if observation.get("inside_teaching_hull") is True:
        return True
    if (observation.get("grasp_region_ok") is not True
            or observation.get("projected_to_teaching_boundary") is not True):
        return False
    try:
        gap = float(observation.get("outside_distance_px", float("inf")))
        return math.isfinite(gap) and 0 <= gap <= _APPROACH_PROFILE.hull_grasp_margin_px
    except (TypeError, ValueError):
        return False


def select_adjustment(observation: dict, *, responses: dict | None = None,
                      blocked_actions: set | None = None,
                      duration_s: float = PULSE_FIRST_S,
                      maximum_s: float = PULSE_MAX_S,
                      allow_probe: bool = False) -> AdjustmentPlan | None:
    """Use the full 4D residual, then learn a local image-space response.

    Nominal toe-in directions are only priors for a bounded first trial.
    Measured responses are session-local, never reused across camera restarts
    or targets. They select direction, not an assumed from-rest travel time.
    All actions are cut short by live image motion. No pixels become metres.
    """
    if in_grasp_region(observation):
        return None
    center, goal = _center_vector(observation), _hull_vector(observation)
    residual = _weighted(tuple(b - a for a, b in zip(center, goal)))
    length = _norm(residual)
    if length == 0:
        return None
    responses, blocked_actions = responses or {}, blocked_actions or set()
    measured, nominal, probes = [], [], []
    # Model-left is video2/physical-left; model-right video0/physical-right.
    # Forward increases both Y and physical-left minus physical-right X.
    # The relative gains are uncalibrated priors, not odometry.
    for axis, actions, prior in (
        ("axial", ("forward", "backward"), (0.5, 1.0, -0.5, 1.0)),
        ("yaw", ("turn_left", "turn_right"), (1.0, 0.0, 1.0, 0.0)),
    ):
        learned = axis in responses
        response = _weighted(responses[axis] if learned else prior)
        energy = sum(value * value for value in response)
        if energy < 1e-9:
            continue
        projection = sum(a * b for a, b in zip(residual, response))
        action = actions[0 if projection > 0 else 1]
        if action in blocked_actions:
            continue
        if learned:
            sign = 1 if projection > 0 else -1
            seconds = min(maximum_s, max(TRIAL_FIRST_S, duration_s))
            # Optical early stop makes duration-to-pixel extrapolation invalid.
            # Rank a conservative image-space direction, not a startup model.
            step_px = min(2.0 if allow_probe else 8.0, length * 0.5)
            delta = tuple(sign * step_px * value / math.sqrt(energy) for value in response)
            gain = length - _norm(tuple(a - b for a, b in zip(residual, delta)))
            minimum_gain = 0.1 if allow_probe else 0.5
            if gain >= minimum_gain and _norm(delta) <= 100.0:
                measured.append((gain, AdjustmentPlan(
                    action, seconds, "measured_4d_response", round(gain, 2))))
        else:
            # A turn corrects common-mode X, not opposite X errors. Axial
            # selection includes Y from BOTH eyes instead of disparity first.
            common_x = (goal[0] - center[0] + goal[2] - center[2]) / 2.0
            if axis == "yaw" and abs(common_x) < 12.0:
                continue
            cosine = abs(projection) / (length * math.sqrt(energy))
            if cosine >= 0.10:
                seconds = min(maximum_s, max(TRIAL_FIRST_S, duration_s))
                nominal.append((cosine, AdjustmentPlan(action, seconds,
                                                       "nominal_joint_4d_trial")))
            elif (allow_probe and axis == 'axial' and cosine >= 0.02
                  and _distance(observation) <= CLOSE_DISTANCE_PX
                  and visual_adjustment_phase(observation, 'micro') == 'micro'):
                # Toe-in residuals can nearly cancel at a convex-hull face.
                # A positive but weak projection is not a direction veto.
                # Only try the supported axial sign, never arbitrary yaw or
                # the opposite of a blocked action. The loop bounds trials
                # and retains optical early-stop + fresh stopped observations.
                probes.append((cosine, AdjustmentPlan(
                    action, min(maximum_s, TRIAL_FIRST_S), 'weak_axial_probe')))
    candidates = measured or nominal or probes
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def choose_adjustment(observation: dict) -> tuple[int, int, str] | None:
    plan = select_adjustment(observation)
    return _command_for(plan.action) if plan else None


def _command_for(action: str) -> tuple[int, int, str]:
    return {
        "forward": (PULSE_SPEED_MM_S, 0, "forward"),
        "backward": (-PULSE_SPEED_MM_S, 0, "backward"),
        "turn_left": (0, PULSE_TURN_MRAD_S, "turn_left"),
        "turn_right": (0, -PULSE_TURN_MRAD_S, "turn_right"),
    }[action]


def measured_response(action: str, seconds: float, before, after) -> tuple[str, tuple]:
    """Average local response used for direction only, not metric odometry."""
    if seconds <= 0 or not math.isfinite(seconds):
        raise ValueError("invalid pulse duration")
    sign = 1 if action in {"forward", "turn_left"} else -1
    axis = "axial" if action in {"forward", "backward"} else "yaw"
    window = seconds
    return axis, tuple(sign * (b - a) / window for a, b in zip(before, after))


def next_pulse_duration(previous_s: float, moved_px: float, *,
                        minimum_s: float = PULSE_FIRST_S,
                        maximum_s: float = PULSE_MAX_S) -> float:
    """Increase *time*, never motor speed, only when the last pulse did not move."""
    if moved_px < 5.0:
        return min(maximum_s, round(previous_s + PULSE_STEP_S, 2))
    return min(maximum_s, max(minimum_s, round(previous_s - PULSE_STEP_S, 2)))


class PulseWindowMemory:
    """Session-local effective windows, NOT measured motor startup times.

    Separate forward/backward, pure yaw and arcs. Optical evidence from both
    views plus at least one background excludes centre jitter alone.
    """
    def __init__(self):
        self.windows: dict[str,float] = {}

    def remember(self, action: str, requested_s: float, feedback: dict) -> bool:
        visual=feedback.get('visual_feedback') or {}
        bg=visual.get('movement_px') or {}
        roi=visual.get('target_region_movement_px') or {}
        values=[]
        for side in ('left','right'):
            background=float(bg.get(side,0))
            local=float(roi.get(side,0))
            if not all(math.isfinite(v) and v>=0 for v in (background,local)):
                return False
            if max(background,local)<0.8:
                return False
            values.append(background)
        if max(values)<0.8 or not math.isfinite(requested_s) or not PULSE_FIRST_S<=requested_s<=0.8:
            return False
        effective=float(feedback.get('until_stop_requested_s',requested_s))
        if not math.isfinite(effective) or effective<=0:
            return False
        self.windows[action]=min(PULSE_MAX_S,max(PULSE_FIRST_S,
            round(min(requested_s,effective),2)))
        return True

    def window(self, action: str, requested_s: float, cap_s: float) -> float:
        return min(cap_s,max(requested_s,self.windows.get(action,PULSE_FIRST_S)))

    def clear(self):
        self.windows.clear()


def adjustment_policy(distance_px: float, normal_limit_px: float = 180.0,
                      trial_band_px: float = TRIAL_BAND_PX) -> tuple[str, float, float]:
    """Separate a bounded trial envelope from the unchanged grasp gate.

    Distance is the model's normalized 4D residual expressed in pixel units,
    not an estimate of chassis travel. No mode here authorizes arm motion.
    """
    if (not all(math.isfinite(x) for x in
                (distance_px, normal_limit_px, trial_band_px))
            or distance_px <= 0 or normal_limit_px <= 0 or trial_band_px < 0):
        raise RuntimeError("凸包距离或微调范围无效，禁止微调")
    if distance_px > normal_limit_px + trial_band_px:
        raise RuntimeError(
            f"离示教凸包 {distance_px:.1f}px，超出近距离微调上限"
            f"（试探上限 {normal_limit_px + trial_band_px:.1f}px）")
    if distance_px > normal_limit_px:
        return "boundary_trial", PULSE_FIRST_S, TRIAL_MAX_S
    if distance_px <= min(CLOSE_DISTANCE_PX, normal_limit_px):
        return "close", PULSE_FIRST_S, CLOSE_MAX_S
    return "normal", PULSE_FIRST_S, PULSE_MAX_S


def _distance(observation: dict) -> float:
    if observation.get('grasp_region_mode')=='dual_2d_shell':
        try:
            values=[float(observation['view_hull_distances_px'][s]) for s in ('left','right')]
            if not all(math.isfinite(v) and v>=0 for v in values):
                raise ValueError('invalid per-eye distances')
            return max(values)
        except (KeyError,TypeError,ValueError) as exc:
            raise RuntimeError('二维凸包距离无效，禁止微调') from exc
    value = float(observation["outside_distance_px"])
    if not math.isfinite(value) or value < 0:
        raise RuntimeError("凸包距离无效，禁止微调")
    if not observation["inside_teaching_hull"] and value == 0:
        raise RuntimeError("凸包内外状态与距离不一致，禁止微调")
    return value


def _disparity(vector: tuple[float, float, float, float]) -> float:
    """Physical left (video2) minus physical right (video0)."""
    return vector[0] - vector[2]


def coherent_axial_motion(action: str, before, after) -> bool:
    """Both fresh target rows support the commanded axial direction.

    Not an identity proof or metric displacement: a large jump additionally
    requires a stopped re-observation and valid live background-flow feedback.
    """
    if action not in {"forward", "backward"}:
        return False
    sign = 1 if action == "forward" else -1
    return (all(sign * (after[i] - before[i]) >= 3.0 for i in (1, 3))
            and sign * (_disparity(after) - _disparity(before)) >= -4.0)


def forward_patience_candidate(observation: dict, before: tuple, feedback: dict,
                               previous_distance: float, anchor_distance: float, *,
                               before_observation=None,recent_forward=False) -> bool:
    """A bounded continuation, not a claim that the hull must get closer."""
    center=_center_vector(observation)
    gap=_distance(observation)
    if (gap>CLOSE_DISTANCE_PX or visible_too_close(observation)
            or gap-previous_distance>_APPROACH_PROFILE.hull_micro_forward_step_worsening_px
            or gap-anchor_distance>_APPROACH_PROFILE.hull_micro_forward_total_worsening_px
            or _norm(tuple(a-b for a,b in zip(center,before)))>100):
        return False
    if _APPROACH_PROFILE.hull_multicue_progress_enabled:
        previous=before_observation or {'centers':{'left':before[:2],'right':before[2:]}}
        cues=approach_evidence(previous,observation,feedback,recent_forward=recent_forward,
            jitter_px=_APPROACH_PROFILE.hull_center_jitter_px,
            growth_ratio=_APPROACH_PROFILE.hull_area_growth_ratio)
        if not cues['supported']:
            return False
    else:
        flow=(feedback.get('visual_feedback') or {}).get('movement_px') or {}
        if (not coherent_axial_motion('forward',before,center)
                or not all(math.isfinite(float(flow.get(side,-1))) and float(flow.get(side,-1))>=.8
                           for side in ('left','right'))):
            return False
    # Bearing is relative to the chassis lines, not opposite nearest-hull X.
    heading=approach_adjustment(observation,duration_s=PULSE_FIRST_S,
                                maximum_s=PULSE_MAX_S,blocked_actions=set())
    return heading is not None and heading.action=='forward'


def observed_motion_state(moved_px: float, distance_change_px: float, feedback: dict) -> str:
    """A stopped detection jitter is not a failed chassis correction."""
    visual=feedback.get('visual_feedback') or {}
    bg=visual.get('movement_px') or {}
    local=visual.get('target_region_movement_px') or {}
    values=[float(bg[side]) for side in ('left','right') if side in bg]
    values.extend(float(local[side]) for side in ('left','right') if side in local)
    if any(not math.isfinite(v) or v<0 for v in values):
        raise RuntimeError('视觉位移反馈无效，已停止')
    if (moved_px<=12.0 and abs(distance_change_px)<=12.0 and
            ((len(bg)==2 and values and max(values)<0.8)
             or not values and moved_px<5.0 and abs(distance_change_px)<=8.0)):
        return 'stationary'
    if moved_px<5.0 and abs(distance_change_px)<=8.0:
        return 'small_motion'
    return 'moved'


def approach_progress(action: str, before, after) -> float:
    if action=='forward':
        return min(after[1]-before[1],after[3]-before[3])
    if action=='turn_left':
        return min(after[0]-before[0],after[2]-before[2])
    if action=='turn_right':
        return min(before[0]-after[0],before[2]-after[2])
    return 0.0


def opposite_arc_repair(observation: dict, before, after, failed_action: str,
                        feedback: dict, *, duration_s: float,
                        maximum_s: float) -> AdjustmentPlan | None:
    """Use a bad forward arc to select a bounded opposite *pure yaw* probe.

    The arc's Y motion contains forward travel, so reversing its full 4D image
    displacement would be a false prediction for another forward arc. Rank
    only its observed X sign against the per-eye polygons; every later pulse
    still needs a fresh stopped observation and optical early-stop.
    """
    if failed_action not in {'turn_left', 'turn_right'} or duration_s <= 0:
        return None
    flow = (feedback.get('visual_feedback') or {}).get('movement_px') or {}
    try:
        if not all(math.isfinite(float(flow[side])) and float(flow[side]) >= .8
                   for side in ('left', 'right')):
            return None
        polygons = observation['view_hull_polygons_px']
        centers = observation['centers']
        current = max(polygon_distance(centers[side], polygons[side])[0]
                      for side in ('left', 'right'))
        correction_s = min(maximum_s, .50)
        ratio = correction_s / duration_s
        predicted = max(polygon_distance(
            (float(centers[side][0]) -
             (after[offset] - before[offset]) * ratio,
             float(centers[side][1])), polygons[side])[0]
            for side, offset in (('left', 0), ('right', 2)))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    gain = current - predicted
    if not math.isfinite(gain) or gain < 1.0:
        return None
    opposite = 'turn_right' if failed_action == 'turn_left' else 'turn_left'
    return AdjustmentPlan(opposite, correction_s,
                          'measured_opposite_yaw_repair', round(gain, 2))


@dataclass
class AlignmentResult:
    ok: bool
    reason: str
    observations: int
    pulses: int
    distance_px: float | None
    log_path: str


@dataclass
class SweptRetreatBudget:
    """One handoff's short straight rollback, never a general reverse permit.

    Created from a matching approach run or fresh micro-stage movement evidence.
    Visual feedback determines progress; pixel residual is NOT metric odometry.
    Bound pulse count and on-time, never claim a commanded duration proves cm.
    Dynamic rear obstacles remain unobservable without a rear sensor/operator.
    """
    expires_at: float
    source: str
    on_time_s: float = 0.0
    pulses: int = 0
    invalidated: bool = False
    heading_change_rad: float = 0.0  # command proxy, never actual odometry

    def available(self) -> bool:
        return (not self.invalidated and math.isfinite(self.expires_at)
                and time.monotonic()<=self.expires_at
                and self.pulses<SWEPT_RETREAT_MAX_PULSES
                and math.isfinite(self.on_time_s) and 0<=self.on_time_s<SWEPT_RETREAT_MAX_ON_TIME_S)

    def note_turn(self, angular_mrad_s: float, seconds: float):
        if (not math.isfinite(angular_mrad_s) or not math.isfinite(seconds)
                or seconds<0):
            self.invalidated=True
            return
        self.heading_change_rad+=abs(angular_mrad_s)*seconds/1000.0
        if (not math.isfinite(self.heading_change_rad)
                or self.heading_change_rad>_APPROACH_PROFILE.retreat_recent_max_heading_rad):
            self.invalidated=True

    def reserve(self, requested_s: float) -> tuple[float, float]:
        if (self.invalidated or not math.isfinite(self.expires_at)
                or time.monotonic() > self.expires_at
                or self.pulses >= SWEPT_RETREAT_MAX_PULSES
                or not math.isfinite(self.on_time_s)
                or not 0.0 <= self.on_time_s < SWEPT_RETREAT_MAX_ON_TIME_S):
            raise RuntimeError("原路短退凭据过期或预算已用尽，禁止无后向复测的后退")
        if not math.isfinite(requested_s) or not TRIAL_FIRST_S <= requested_s <= PULSE_MAX_S:
            raise RuntimeError("原路短退脉冲时长无效")
        seconds = min(requested_s, SWEPT_RETREAT_MAX_ON_TIME_S - self.on_time_s)
        if seconds < TRIAL_FIRST_S - 1e-9:
            raise RuntimeError("原路短退剩余通电时间预算不足，已停止")
        self.on_time_s = round(self.on_time_s + seconds, 3)
        self.pulses += 1  # reserve before sending; failures never refund motion
        return seconds, self.on_time_s


def visible_too_close(observation: dict) -> bool:
    """Lower-edge target rows beyond the taught pose: recovery, not grasp."""
    if in_grasp_region(observation):
        return False
    center,goal=_center_vector(observation),_hull_vector(observation)
    return (center[1]>=380 and center[3]>=320
            and all(center[i]-goal[i]>=12 for i in (1,3)))


class ScanGuard:
    def __init__(self) -> None:
        self.driver = N10LidarDriver(LidarConfig())
        self.latest = None
        self.updated_at = 0.0
        self.error = ""
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        try:
            for scan in self.driver.scans():
                if self.stop_event.is_set():
                    return
                self.latest = scan
                self.updated_at = time.monotonic()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def wait_ready(self, timeout_s: float = 3.0) -> None:
        deadline = time.monotonic() + timeout_s
        while self.latest is None and not self.error and time.monotonic() < deadline:
            time.sleep(0.05)
        if self.latest is None:
            raise RuntimeError(f"雷达没有新鲜扫描，禁止微调：{self.error or '等待超时'}")

    def clearance(self, action: str, *, front_only_retreat: bool = False) -> float:
        if self.error:
            raise RuntimeError(f"雷达故障，禁止微调：{self.error}")
        if self.latest is None or time.monotonic() - self.updated_at > 1.2:
            raise RuntimeError("没有新鲜雷达扫描，禁止微调")
        if action not in {"forward", "backward", "turn_left", "turn_right"}:
            raise RuntimeError(f"未知微调方向 {action}，禁止运动")
        # Only an explicitly budgeted same-path rollback may omit the rear
        # sector. Front sensing stays live throughout that short pulse.
        if action == "backward" and not REAR_SECTOR_USABLE and not front_only_retreat:
            raise RuntimeError("后向扇区被车身遮挡；缺少原路短退预算，禁止后退")
        if action == "backward" and front_only_retreat:
            sectors = ((0.0, MIN_FRONT_M),)
        elif action == "backward":
            sectors = ((180.0, MIN_REAR_M),)
        elif action == "forward":
            sectors = ((0.0, MIN_FRONT_M),)
        elif REAR_SECTOR_USABLE:
            sectors = ((0.0, MIN_FRONT_M), (90.0, MIN_REAR_M),
                       (180.0, MIN_REAR_M), (270.0, MIN_REAR_M))
        else:
            # Only the front fan is observable on this chassis. Do not read
            # the rear/occluded side sectors even during a short yaw pulse.
            sectors = ((0.0, MIN_FRONT_M), (45.0, MIN_FRONT_M),
                       (315.0, MIN_FRONT_M))
        clearances = []
        for center, limit in sectors:
            value = sector_min_distance(
                self.latest, center_deg=center, width_deg=55.0,
                min_distance_m=0.02, max_distance_m=8.0)
            if not math.isfinite(value):
                raise RuntimeError("运动方向雷达扇区无有效测距，禁止微调")
            if value < limit:
                raise RuntimeError(
                    f"{action} {center:.0f}° 障碍 {value:.2f}m < {limit:.2f}m，禁止微调")
            clearances.append(value)
        return min(clearances)

    def close(self) -> None:
        self.stop_event.set()
        self.driver.close()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)


def _ready(grab_url: str, *, require_ble: bool = True) -> None:
    status = _get_json(f"{grab_url}/api/status")
    cameras = status.get("cameras") or {}
    if (require_ble and not status.get("ble")) or not all(
        (cameras.get(side) or {}).get("frames", 0) > 0
        and float((cameras.get(side) or {}).get("age_s", 99)) < 1.0
        for side in ("left", "right")
    ):
        raise RuntimeError("双目或 ESP32 BLE 未就绪，禁止微调")
    if (status.get("grasp") or {}).get("active"):
        raise RuntimeError("机械臂任务正在执行，禁止移动底盘")


class TargetNotVisible(RuntimeError):
    """Only an explicit detector miss, never a network/model/hardware fault."""


def _observe(target: str, *, grab_url: str = GRAB_URL,
             timeout: float = 45.0,
             require_ble: bool = True) -> dict:
    _ready(grab_url, require_ble=require_ble)
    try:
        result = _post_json(f"{grab_url}/api/arm-teach/predict",
                            {"target": target}, timeout=timeout)
    except urllib.error.HTTPError as exc:
        try:
            result = json.loads(exc.read().decode('utf-8'))
        except (ValueError, UnicodeError):
            raise exc
        if 'target not visible in both views' in str(result.get('error', '')).lower():
            raise TargetNotVisible('目标未同时出现在双目画面') from exc
        raise exc
    if not result.get("ok") or not isinstance(result.get("inside_teaching_hull"), bool):
        if 'target not visible in both views' in str(result.get('error', '')).lower():
            raise TargetNotVisible('目标未同时出现在双目画面')
        raise RuntimeError(f"双目/凸包预测无效：{result.get('error', result)}")
    _center_vector(result)
    if (result.get('grasp_region_mode')!='dual_2d_shell'
            or result.get('grasp_margin_px')!=_APPROACH_PROFILE.hull_grasp_margin_px):
        raise RuntimeError('抓取服务尚未使用左右二维25px规则，请同步部署两端；禁止微调')
    _distance(result)
    if not result["inside_teaching_hull"]:
        _hull_vector(result)
    return result


def _pulse(grab_url: str, guard: ScanGuard, action: str, seconds: float,
           cancel_event: threading.Event | None = None, *,
           swept_retreat: bool = False,
           turn_mrad_s: int = PULSE_TURN_MRAD_S,
           arc_forward: bool = False,
           speed_mm_s: int = PULSE_SPEED_MM_S,
           feedback_sink: dict | None = None,
           visual_motion_limit_px: float | None = None,
           target_centers: dict | None = None) -> float:
    """Let grab_visual's existing BLE owner issue the bounded pulse.

    The caller keeps watching lidar while the HTTP worker is active.  A new
    obstacle or emergency sends the preview's STOP command immediately;
    the preview itself also stops in a finally block and the firmware has a
    short motion timeout if the Pi disappears.
    """
    if arc_forward and action not in {'turn_left','turn_right'}:
        raise ValueError('forward arc requires a turn action')
    wire_action = ('arc_left' if action=='turn_left' else 'arc_right') if arc_forward else action
    token = TOKEN_PATH.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("近距微调缺少硬件安全令牌")
    clearance = guard.clearance(action, front_only_retreat=swept_retreat)
    outcome: dict = {}
    pulse_id = uuid.uuid4().hex

    def send() -> None:
        try:
            outcome["response"] = _post_json(
                f"{grab_url}/api/arm-teach/chassis-pulse",
                {"action": wire_action, "duration_s": seconds,
                 "pulse_id": pulse_id, "speed_mm_s": speed_mm_s,
                 "angular_mrad_s": turn_mrad_s,
                 **({"visual_motion_limit_px": visual_motion_limit_px}
                    if visual_motion_limit_px is not None else {}),
                 **({'target_centers': target_centers} if target_centers is not None else {})},
                timeout=16.0, token=token)
        except urllib.error.HTTPError as exc:
            try:
                outcome["response"] = json.loads(exc.read().decode("utf-8"))
            except Exception:
                outcome["error"] = str(exc)
        except Exception as exc:
            outcome["error"] = str(exc)

    worker = threading.Thread(target=send, daemon=True)
    worker.start()
    while worker.is_alive():
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("收到急停，已中止凸包微调")
            guard.clearance(action, front_only_retreat=swept_retreat)
        except Exception:
            try:
                _post_json(
                    f"{grab_url}/api/arm-teach/chassis-cancel",
                    {"pulse_id": pulse_id}, timeout=3.0, token=token)
            finally:
                worker.join(timeout=2.0)
            raise
        worker.join(timeout=0.08)
    response = outcome.get("response") or {}
    if not response.get("ok"):
        raise RuntimeError("共享 BLE 微调失败：" + str(
            outcome.get("error") or response.get("error") or response))
    if feedback_sink is not None:
        feedback_sink.update(response.get('pulse') or {})
    time.sleep(REST_S)
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("收到急停，已中止凸包微调")
    return clearance


def align(target: str, *, max_pulses: int = _APPROACH_PROFILE.hull_micro_max_steps, max_distance_px: float = 180.0,
          trial_band_px: float = TRIAL_BAND_PX,
          deadline_s: float = 150.0, real: bool = False,
          log_dir: Path | None = None,
          grab_url: str = GRAB_URL,
          cancel_event: threading.Event | None = None,
          retreat_budget: SweptRetreatBudget | None = None) -> AlignmentResult:
    log_root = log_dir or TRACK_ROOT / "logs" / "hull_align"
    log_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    log_path = log_root / f"{stamp}_{'real' if real else 'dry_run'}.jsonl"
    observations = pulses = 0
    total_pulse_s = 0.0
    distance = None
    guard = None
    deadline = time.monotonic() + deadline_s

    def record(event: str, **fields) -> None:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": datetime.now().astimezone().isoformat(),
                                     "event": event, **fields}, ensure_ascii=False) + "\n")
        if event=='planned_pulse':
            label='近端微调' if fields['control_phase']=='micro' else '视觉逼近/居中'
            action_label={'arc_left':'前进左修','arc_right':'前进右修'}.get(fields.get('motor_action'),fields['action'])
            print(f"[phase] {label} 第{fields['step']}/{fields['step_limit']}步：{action_label} "
                  f"窗口{fields['duration_s']:.2f}s，视觉早停{fields['visual_motion_limit_px']:.0f}px；"
                  f"凸包误差{fields['distance_px']:.1f}px（非实际距离）",flush=True)
        elif event=='stationary_retry':
            print(f"[phase] 第{fields['step']}步未确认底盘移动；{fields['action']}同向重试，"
                  f"下步窗口{fields['next_window_s']:.2f}s",flush=True)
        elif event=='stopped':
            print('[phase] 任务停止: '+str(fields.get('reason','')),flush=True)
        elif event in {'visual_phase_changed','continue_alignment','corrective_replan','alignment_complete','stationary_turn_replan',
                       'visible_near_retreat','visible_near_recovered','lost_target_rollback','lost_target_retreat_retry',
                       'forward_start_window','forward_patience_continue'}:
            print('[phase] 视觉调整 '+json.dumps({'event':event,**fields},ensure_ascii=False),flush=True)

    def success(reason: str, distance_px: float) -> AlignmentResult:
        record("alignment_complete", reason=reason, observations=observations,
               pulses=pulses, distance_px=distance_px, total_pulse_s=total_pulse_s)
        return AlignmentResult(True, reason, observations, pulses,
                               distance_px, str(log_path))

    try:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("收到急停，禁止凸包微调")
        if real:
            verify_model_camera_order()
        observation = _observe(target, grab_url=grab_url,
                               timeout=min(45.0, deadline_s))
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("识别期间收到急停，已中止凸包微调")
        observations += 1
        distance = _distance(observation)
        record("observe", target=target, observation=observation)
        if observation["inside_teaching_hull"] or (not real and in_grasp_region(observation)):
            return success("目标已在示教抓取区域内（含模型检测余量）", distance)
        recover_visible=(real and retreat_budget is not None and retreat_budget.available()
                         and visible_too_close(observation))
        if not recover_visible:
            adjustment_policy(distance, max_distance_px, trial_band_px)
        # A person moving the object (or a mistaken hand lock) invalidates the
        # next open-loop pulse.  Require a second fresh, stable observation.
        if real:
            remaining = deadline - time.monotonic()
            if remaining < 10.0:
                raise RuntimeError("没有时间完成移动前双目稳定性复核")
            confirmed = _observe(target, grab_url=grab_url,
                                 timeout=min(45.0, remaining - 2.0))
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("复核期间收到急停，已中止凸包微调")
            observations += 1
            displacement = math.sqrt(sum(
                (a - b) ** 2 for a, b in zip(
                    _center_vector(observation), _center_vector(confirmed))))
            record("stability_check", displacement_px=round(displacement, 1),
                   observation=confirmed)
            if displacement > 15.0:
                raise RuntimeError(
                    f"目标在移动或双目识别不稳定 ({displacement:.1f}px)，禁止微调")
            observation = confirmed
            distance = _distance(observation)
            if in_grasp_region(observation):
                return success("稳定复核后进入示教抓取区域（含模型检测余量）", distance)
        # Over-close recovery precedes the normal 240px candidate cutoff.
        # Two straight-back pulses at most, with optical early stop and no
        # chassis-occluded rear scan. Windows cannot establish centimetres.
        def recover_visible_near():
            nonlocal observation,distance,pulses,total_pulse_s,observations,guard
            while (real and retreat_budget is not None and retreat_budget.available()
                   and visible_too_close(observation)):
                if pulses>=max_pulses or time.monotonic()+10>=deadline:
                    raise RuntimeError('近端短退达到次数或总时限，已停止')
                seconds=min(.64,.60+.04*retreat_budget.pulses)
                seconds,total_retreat=retreat_budget.reserve(seconds)
                if total_pulse_s+seconds>MAX_TOTAL_PULSE_S:
                    raise RuntimeError('近端短退达到总脉冲预算，已停止')
                if guard is None:
                    guard=ScanGuard()
                    guard.wait_ready()
                _ready(grab_url)
                feedback={}
                before=_center_vector(observation)
                old_distance=distance
                record('visible_near_retreat',duration_s=seconds,
                       source=retreat_budget.source,total_retreat_on_time_s=total_retreat,
                       rear_sector_checked=False,approximate_path=True,
                       centers=observation['centers'],distance_px=distance,
                       heading_command_proxy_rad=retreat_budget.heading_change_rad,
                       max_pulses=SWEPT_RETREAT_MAX_PULSES,
                       max_on_time_s=SWEPT_RETREAT_MAX_ON_TIME_S)
                _pulse(grab_url,guard,'backward',seconds,cancel_event=cancel_event,
                       swept_retreat=True,feedback_sink=feedback,
                       speed_mm_s=_APPROACH_PROFILE.hull_micro_speed_mm_s,
                       visual_motion_limit_px=2.0,target_centers=observation['centers'])
                pulses+=1
                total_pulse_s=round(total_pulse_s+seconds,3)
                record('visible_near_retreat_done',duration_s=seconds,pulse_feedback=feedback,pulses=pulses)
                try:
                    observation=_observe(target,grab_url=grab_url,
                                         timeout=min(30,max(1,deadline-time.monotonic()-3)),require_ble=True)
                except TargetNotVisible:
                    record('lost_target_stationary_recheck',after_action='visible_near_retreat')
                    try:
                        observation=_observe(target,grab_url=grab_url,
                            timeout=min(20,max(1,deadline-time.monotonic()-8)),require_ble=True)
                    except TargetNotVisible:
                        record('lost_target_retreat_retry',pulses=retreat_budget.pulses,
                               after_action='visible_near_retreat')
                        if (retreat_budget.available() and pulses<max_pulses
                                and time.monotonic()+15<deadline
                                and not (cancel_event is not None and cancel_event.is_set())):
                            continue  # one more straight retreat; never renew this budget
                        raise RuntimeError('近端短退后仍未找回目标，已停车')
                observations+=1
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError('短退复核时收到急停，已停止')
                after=_center_vector(observation)
                distance=_distance(observation)
                moved=_norm(tuple(a-b for a,b in zip(after,before)))
                record('visible_near_retreat_observe',observation=observation,
                       moved_px=round(moved,1),previous_distance_px=old_distance)
                if moved>100 or (moved>=5 and distance>old_distance+8):
                    raise RuntimeError('短退后目标位置变化异常或明显远离凸包，已停止')
                if in_grasp_region(observation):
                    return True
            if real and visible_too_close(observation):
                raise RuntimeError('目标仍过近，但有限短退凭据或预算不可用，已停止')
            return False

        if recover_visible_near():
            return success('近端短退后进入抓取区域；夹取前仍须复核',distance)
        stage, duration, pulse_cap = adjustment_policy(
            distance, max_distance_px, trial_band_px)
        record("alignment_policy", stage=stage, normal_limit_px=max_distance_px,
               distance_metric=('max_per_eye_2d_px' if observation.get('grasp_region_mode')=='dual_2d_shell' else 'legacy_joint_4d_px'),
               grasp_margin_px=_APPROACH_PROFILE.hull_grasp_margin_px,
               forward_reference=reference_payload(_APPROACH_PROFILE),
               trial_limit_px=max_distance_px + trial_band_px,
               total_pulse_budget_s=MAX_TOTAL_PULSE_S,
               speed_mm_s=PULSE_SPEED_MM_S, first_on_time_s=PULSE_FIRST_S,
               micro_speed_mm_s=_APPROACH_PROFILE.hull_micro_speed_mm_s,
               first_normal_forward_s=_APPROACH_PROFILE.hull_forward_first_s,
               forward_patience_steps=_APPROACH_PROFILE.hull_micro_forward_patience_steps,
               effective_window_memory='session_local_per_motor_action',
               minimum_linear_on_time_s=TRIAL_FIRST_S,
               linear_startup_assumption_s=LINEAR_STARTUP_S,
               max_on_time_s=PULSE_MAX_S, feedback="dual_visual_4d_hull")
        last_action = None
        last_center = _center_vector(observation)
        responses: dict[str, tuple] = {}
        blocked_actions: set[str] = (
            {"backward"} if real and not REAR_SECTOR_USABLE and retreat_budget is None
            else set())
        consecutive_corrections = 0
        stationary_at_cap = 0
        motion_limit = 4.0
        control_phase = None
        static_action = None
        static_turn_steps = 0
        turn_speed = PULSE_TURN_MRAD_S
        probe_trials = 0
        window_memory = PulseWindowMemory()
        first_forward_sent=False
        forward_patience_used=0
        forward_patience_anchor=None
        pending_forward_patience=False
        pending_repair: AdjustmentPlan | None = None
        arc_repair_action: str | None = None
        arc_repair_steps = 0
        last_reliable_forward_step=-100
        last_reliable_forward_at=0.0
        def has_recent_forward():
            return recent_forward_valid(pulses,last_reliable_forward_step,time.monotonic(),
                last_reliable_forward_at,_APPROACH_PROFILE.hull_forward_trend_max_age_s)
        for _ in range(max_pulses):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("收到急停，已中止凸包微调")
            if time.monotonic() + 8.0 >= deadline:
                raise RuntimeError("微调总时限已到，已停止底盘")
            stage, _, pulse_cap = adjustment_policy(
                distance, max_distance_px, trial_band_px)
            if pulses >= max_pulses:
                raise RuntimeError(f'达到 {max_pulses} 次微调上限，已停止')
            if (retreat_budget is not None and
                    (retreat_budget.invalidated or
                     time.monotonic() > retreat_budget.expires_at or
                     retreat_budget.pulses >= SWEPT_RETREAT_MAX_PULSES)):
                blocked_actions.add("backward")
            new_phase=visual_adjustment_phase(observation,control_phase)
            if new_phase!=control_phase:
                duration=PULSE_FIRST_S
                static_action=None
                static_turn_steps=0
                record('visual_phase_changed',previous=control_phase,phase=new_phase,
                       distance_px=distance, enter_left_y_px=_APPROACH_PROFILE.hull_micro_enter_left_y_px,
                       enter_right_y_px=_APPROACH_PROFILE.hull_micro_enter_right_y_px,
                       entry_signal='image_rows_only')
            control_phase=new_phase
            continuing_forward=pending_forward_patience
            pending_forward_patience=False
            if continuing_forward:
                plan=AdjustmentPlan('forward',min(pulse_cap,max(duration,_APPROACH_PROFILE.hull_forward_first_s)),
                                    'bounded_forward_continuation')
            elif pending_repair is not None:
                plan=pending_repair
                pending_repair=None
            elif static_action is not None and static_action not in blocked_actions:
                source=('measured_opposite_yaw_repair_static_retry'
                        if static_action==arc_repair_action else 'stationary_same_direction_retry')
                plan=AdjustmentPlan(static_action,min(pulse_cap,duration),source)
            elif control_phase=='approach' and not blocked_actions.intersection({'forward','turn_left','turn_right'}):
                plan=approach_adjustment(observation,duration_s=duration,
                                         maximum_s=pulse_cap,blocked_actions=blocked_actions)
            else:
                plan=select_adjustment(observation,responses=responses,
                    blocked_actions=blocked_actions,duration_s=duration,maximum_s=pulse_cap,
                    allow_probe=control_phase=='micro' and probe_trials<MAX_LOW_CONFIDENCE_PROBES)
            if plan is None:
                record('no_adjustment_candidate', observation=observation,
                       learned_responses=responses, blocked_actions=sorted(blocked_actions),
                       probe_trials=probe_trials, probe_limit=MAX_LOW_CONFIDENCE_PROBES)
                raise RuntimeError("视觉微调没有可用方向（含受限试探），已停车；详见候选诊断日志")
            v, w, action = _command_for(plan.action)
            arc_forward = (control_phase=='approach' and
                not plan.source.startswith('measured_opposite_yaw_repair') and
                action in {'turn_left','turn_right'})
            pulse_speed = (_APPROACH_PROFILE.hull_micro_speed_mm_s
                           if control_phase=='micro' else PULSE_SPEED_MM_S)
            if action in {'forward','backward'}:
                v=pulse_speed if action=='forward' else -pulse_speed
            if action in {'turn_left','turn_right'}:
                w=turn_speed if action=='turn_left' else -turn_speed
                if arc_forward:
                    v=PULSE_SPEED_MM_S
            duration = plan.duration_s
            if arc_forward and consecutive_corrections==0 and plan.source!='weak_axial_probe':
                requested=duration
                duration=min(pulse_cap,max(duration,_APPROACH_PROFILE.hull_approach_arc_first_s))
                if duration>requested:
                    record('approach_arc_start_window',action=action,
                           previous_window_s=requested,duration_s=duration,
                           speed_mm_s=PULSE_SPEED_MM_S,turn_mrad_s=abs(w),
                           source='operator_observed_small_arc_motion')
            if (action=='forward' and consecutive_corrections==0
                    and plan.source!='weak_axial_probe'):
                requested=duration
                duration=min(pulse_cap,max(duration,_APPROACH_PROFILE.hull_forward_first_s))
                if first_forward_sent and duration>requested:
                    record('forward_window_floor_retained',from_window_s=requested,
                           duration_s=duration,speed_mm_s=pulse_speed,
                           control_phase=control_phase,source='normal_forward_minimum',
                           short_probes_unchanged=True)
                if not first_forward_sent:
                    record('forward_start_window',duration_s=duration,speed_mm_s=pulse_speed,
                           source='sixth_step_203939',short_probes_unchanged=True)
                first_forward_sent=True
            motor_action = ('arc_left' if action=='turn_left' else 'arc_right') if arc_forward else action
            # Weak first trials and short rollback retain their original caps;
            # never borrow a forward window for a reverse or a turn.
            if action!='backward' and plan.source!='weak_axial_probe' and not continuing_forward:
                restored=window_memory.window(motor_action,duration,pulse_cap)
                if restored>duration:
                    record('effective_window_reused',motor_action=motor_action,
                           from_window_s=duration,to_window_s=restored,
                           control_phase=control_phase)
                duration=restored
            swept_retreat = action == "backward" and retreat_budget is not None
            if swept_retreat:
                duration, total_retreat_s = retreat_budget.reserve(duration)
                record("swept_retreat_reserved", source=retreat_budget.source,
                       duration_s=duration, total_retreat_on_time_s=total_retreat_s,
                       max_retreat_on_time_s=SWEPT_RETREAT_MAX_ON_TIME_S,
                       rear_sector_checked=False, feedback="stopped_dual_visual_hull")
            elif action in {"turn_left", "turn_right"} and retreat_budget is not None:
                retreat_budget.note_turn(w,duration)
                if retreat_budget.invalidated:
                    blocked_actions.add("backward")
            if total_pulse_s + duration > MAX_TOTAL_PULSE_S + 1e-9:
                raise RuntimeError("微调累计脉冲时间预算已用尽，已停止底盘")
            if plan.source=='weak_axial_probe':
                probe_trials += 1
            record("planned_pulse", action=action, v=v, w=w,
                   grasp_region_mode=observation.get('grasp_region_mode'),
                   view_hull_distances_px=observation.get('view_hull_distances_px'),
                   joint_hull_distance_px=observation.get('joint_hull_distance_px',observation.get('outside_distance_px')),
                   motor_action=motor_action,
                   effective_windows_s=dict(window_memory.windows),
                   duration_s=duration, distance_px=distance, stage=stage,
                   pulse_cap_s=pulse_cap, total_pulse_s=total_pulse_s,
                   decision_source=plan.source,
                   predicted_improvement_px=plan.predicted_improvement_px,
                   learned_responses=responses, blocked_actions=sorted(blocked_actions),
                   visual_motion_limit_px=min(motion_limit,2.0 if control_phase=='micro' else 4.0),
                   target_region_guard=True, control_phase=control_phase,
                   probe_trials=probe_trials, probe_limit=MAX_LOW_CONFIDENCE_PROBES,
                   step=pulses+1, step_limit=max_pulses)
            if not real:
                return AlignmentResult(False, "dry-run：仅记录建议脉冲，未移动底盘",
                                       observations, pulses, distance, str(log_path))
            if guard is None:
                guard = ScanGuard()
                guard.wait_ready()
            _ready(grab_url)
            pulse_feedback = {}
            pulse_observation=observation
            clearance = _pulse(grab_url, guard, action, duration,
                               cancel_event=cancel_event,
                               swept_retreat=swept_retreat, feedback_sink=pulse_feedback,
                               turn_mrad_s=turn_speed,
                               arc_forward=arc_forward,
                               speed_mm_s=pulse_speed,
                               visual_motion_limit_px=min(motion_limit, 2.0 if control_phase=='micro' else 4.0),
                               target_centers=observation['centers'])
            pulses += 1
            total_pulse_s = round(total_pulse_s + duration, 3)
            record("pulse_done", action=action, duration_s=duration,
                   clearance_m=clearance, total_pulse_s=total_pulse_s,
                   pulse_feedback=pulse_feedback)
            remaining = deadline - time.monotonic()
            if remaining <= 5.0:
                raise RuntimeError("微调总时限已到，已停止底盘")
            recovered = False
            try:
                observation = _observe(target, grab_url=grab_url,
                                       timeout=min(30.0, remaining - 2.0), require_ble=True)
            except TargetNotVisible:
                window_memory.clear()
                last_reliable_forward_step=-100
                last_reliable_forward_at=0.0
                record('lost_target_stationary_recheck', after_action=action)
                # First inspect while stopped. No rollback on malformed output,
                # camera/BLE faults, model errors or network timeouts.
                try:
                    observation = _observe(target, grab_url=grab_url,
                        timeout=min(20.0, max(1.0, deadline-time.monotonic()-8)), require_ble=True)
                except TargetNotVisible:
                    flow = (pulse_feedback.get('visual_feedback') or {}).get('movement_px') or {}
                    recent_approach=(action=='forward' or arc_forward)
                    modest_arc=(not arc_forward or abs(w)*duration/1000.0
                                <=_APPROACH_PROFILE.retreat_recent_max_heading_rad)
                    if (not recent_approach or not modest_arc or pulses >= max_pulses or
                            not all(math.isfinite(float(flow.get(side, -1)))
                                    and float(flow.get(side, -1)) >= 1.0 for side in ('left','right')) or
                            total_pulse_s+PULSE_FIRST_S > MAX_TOTAL_PULSE_S or
                            time.monotonic()+15 >= deadline or
                            (cancel_event is not None and cancel_event.is_set())):
                        raise RuntimeError('复检仍未找回目标，缺少近期接近/新鲜视觉回溯条件，已停止')
                    if retreat_budget is None:
                        retreat_budget = SweptRetreatBudget(time.monotonic()+60,
                                                            'micro_approach_with_live_flow',
                                                            heading_change_rad=abs(w)*duration/1000.0)
                    while True:
                        seconds, _ = retreat_budget.reserve(min(.64,PULSE_FIRST_S+.04*retreat_budget.pulses))
                        _ready(grab_url)
                        rollback_feedback = {}
                        _pulse(grab_url, guard, 'backward', seconds, cancel_event=cancel_event,
                               swept_retreat=True, feedback_sink=rollback_feedback,
                               visual_motion_limit_px=2.0)
                        pulses += 1
                        total_pulse_s = round(total_pulse_s+seconds,3)
                        record('lost_target_rollback', duration_s=seconds, pulses=pulses,
                               total_pulse_s=total_pulse_s, pulse_feedback=rollback_feedback,
                               rear_sector_checked=False,approximate_path=True,
                               after_action=motor_action,source=retreat_budget.source)
                        try:
                            observation = _observe(target, grab_url=grab_url,
                                timeout=min(30.0, max(1.0, deadline-time.monotonic()-2)), require_ble=True)
                            break
                        except TargetNotVisible:
                            record('lost_target_retreat_retry',pulses=retreat_budget.pulses,
                                   remaining_on_time_s=SWEPT_RETREAT_MAX_ON_TIME_S-retreat_budget.on_time_s)
                            if (not retreat_budget.available() or pulses>=max_pulses
                                    or total_pulse_s+.44>MAX_TOTAL_PULSE_S
                                    or time.monotonic()+15>=deadline
                                    or (cancel_event is not None and cancel_event.is_set())):
                                raise RuntimeError('有限短退仍未找回目标，已停止，不再继续盲退')
                    recovered = True
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("重新识别期间收到急停，已中止凸包微调")
            observations += 1
            new_distance = _distance(observation)
            center = _center_vector(observation)
            moved = math.sqrt(sum((a - b) ** 2 for a, b in zip(center, last_center)))
            disparity_change = _disparity(center) - _disparity(last_center)
            record("observe", observation=observation,
                   moved_px=round(moved, 1),
                   disparity_delta_px=round(disparity_change, 1),
                   hull_improvement_px=round(distance - new_distance, 1))
            previous_distance = distance
            distance = new_distance  # stopped logs must report the latest error
            if in_grasp_region(observation):
                return success("目标进入示教抓取区域（含模型检测余量）", new_distance)
            if visible_too_close(observation):
                flow=(pulse_feedback.get('visual_feedback') or {}).get('movement_px') or {}
                if (retreat_budget is None and (action=='forward' or arc_forward)
                        and abs(w)*duration/1000.0<=_APPROACH_PROFILE.retreat_recent_max_heading_rad
                        and all(math.isfinite(float(flow.get(side,-1)))
                                and float(flow.get(side,-1))>=1 for side in ('left','right'))):
                    retreat_budget=SweptRetreatBudget(time.monotonic()+60,
                        'visible_near_after_live_approach',heading_change_rad=abs(w)*duration/1000.0)
                confirmed=_observe(target,grab_url=grab_url,
                    timeout=min(20,max(1,deadline-time.monotonic()-8)),require_ble=True)
                observations+=1
                if _norm(tuple(a-b for a,b in zip(center,_center_vector(confirmed))))>15:
                    raise RuntimeError('近端回退前目标移动或识别不稳定，已停止')
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError('近端回退前收到急停，已停止')
                observation,distance=confirmed,_distance(confirmed)
                if in_grasp_region(confirmed) or recover_visible_near():
                    return success('近端复核/短退后进入抓取区域；夹取前仍须复核',distance)
                adjustment_policy(distance,max_distance_px,trial_band_px)
                last_center=_center_vector(observation)
                responses.clear()
                window_memory.clear()
                last_reliable_forward_step=-100
                last_reliable_forward_at=0.0
                static_action=None
                duration,motion_limit=PULSE_FIRST_S,2.0
                record('visible_near_recovered',observation=observation)
                continue
            adjustment_policy(new_distance, max_distance_px, trial_band_px)
            if recovered:
                # Combined forward+rollback is not a single response sample.
                record('rollback_reacquired', observation=observation)
                confirmed = _observe(target, grab_url=grab_url,
                    timeout=min(20.0, max(1.0, deadline-time.monotonic()-8)), require_ble=True)
                observations += 1
                if _norm(tuple(a-b for a,b in zip(center,_center_vector(confirmed)))) > 15:
                    raise RuntimeError('回溯找回后双目目标不稳定，已停止')
                distance = _distance(confirmed)
                if in_grasp_region(confirmed):
                    return success('回溯复核后目标进入示教抓取区域',distance)
                observation, last_center = confirmed, _center_vector(confirmed)
                duration, motion_limit = PULSE_FIRST_S, 2.0
                responses.pop('axial',None)
                last_action = 'backward'
                continue
            forward_cues=None
            if _APPROACH_PROFILE.hull_multicue_progress_enabled and action=='forward':
                forward_cues=approach_evidence(pulse_observation,observation,pulse_feedback,
                    recent_forward=has_recent_forward(),
                    jitter_px=_APPROACH_PROFILE.hull_center_jitter_px,
                    growth_ratio=_APPROACH_PROFILE.hull_area_growth_ratio)
                if forward_cues['learnable']:
                    last_reliable_forward_step=pulses
                    last_reliable_forward_at=time.monotonic()
                record('forward_visual_evidence',**forward_cues)
            elif action!='forward':
                last_reliable_forward_step=-100
                last_reliable_forward_at=0.0
            motion_state=observed_motion_state(moved,new_distance-previous_distance,pulse_feedback)
            if motion_state=='stationary':
                if duration>=pulse_cap-1e-9:
                    stationary_at_cap+=1
                    if stationary_at_cap>=2:
                        raise RuntimeError('连续两步在微调时长上限仍无视觉进展，已停止')
                else:
                    stationary_at_cap=0
                next_duration=next_pulse_duration(duration,0,minimum_s=PULSE_FIRST_S,maximum_s=pulse_cap)
                if action in {'turn_left','turn_right'}:
                    static_turn_steps+=1
                    turn_static_limit=(3 if plan.source.startswith('measured_opposite_yaw_repair')
                                       else _APPROACH_PROFILE.hull_micro_turn_static_steps)
                    if static_turn_steps>=turn_static_limit:
                        # Do not lock a stale turn for eleven unsuccessful
                        # pulses, or increase yaw until it suddenly launches.
                        # Approach will re-evaluate the body line; close micro
                        # blocks the failed yaw and selects a supported axial
                        # correction, or stops when none is justified.
                        if not arc_forward:
                            blocked_actions.add(action)
                            responses.pop('yaw',None)
                        record('stationary_turn_replan',action=action,
                               arc_forward=arc_forward,turn_mrad_s=turn_speed,
                               blocked_actions=sorted(blocked_actions),
                               window_s=next_duration)
                        motion_limit=2.0
                        static_turn_steps=0
                        static_action=None
                        last_action,last_center=action,center
                        duration=next_duration
                        continue
                else:
                    static_turn_steps=0
                record('stationary_retry',action=action,control_phase=control_phase,
                       moved_px=round(moved,1),distance_change_px=round(new_distance-previous_distance,1),
                       previous_window_s=duration,next_window_s=next_duration,
                       next_turn_mrad_s=turn_speed,
                       step=pulses,step_limit=max_pulses,pulse_feedback=pulse_feedback)
                static_action=action
                last_action,last_center=action,center
                duration=next_duration
                continue
            static_action=None
            static_turn_steps=0
            stationary_at_cap=0
            if motion_state=='small_motion':
                # With the micro optical limit at 2px, useful response samples
                # often fall below the old 5px learning threshold. Require
                # movement evidence in BOTH backgrounds; otherwise this can
                # just be Qwen centre jitter and must not become a direction.
                flow=(pulse_feedback.get('visual_feedback') or {}).get('movement_px') or {}
                if (not arc_forward and control_phase=='micro' and moved>=2.0
                        and (forward_cues is None or forward_cues['learnable']) and
                        all(math.isfinite(float(flow.get(side,-1))) and
                            float(flow.get(side,-1))>=0.8 for side in ('left','right'))):
                    axis,response=measured_response(action,duration,last_center,center)
                    responses[axis]=response
                    record('local_response',axis=axis,
                           average_pixels_per_requested_window_s=response,
                           response_use='direction_only',sample_source='small_motion_dual_flow')
                record('continue_alignment',reason='小幅运动/检测波动；继续停车后小步调整',
                       control_phase=control_phase,distance_px=distance,pulses=pulses)
                if window_memory.remember(motor_action,duration,pulse_feedback):
                    record('effective_window_learned',motor_action=motor_action,
                           window_s=window_memory.windows[motor_action],source='live_optical_motion')
                last_action,last_center=action,center
                duration=PULSE_FIRST_S
                continue
            coherent = coherent_axial_motion(action, last_center, center)
            if moved > 100.0:
                # No immediate next pulse after a large observation jump.
                # Distinguish a confirmed overrun from a hand/identity jump.
                live_flow = (pulse_feedback.get('visual_feedback') or {}).get('movement_px') or {}
                if (moved > 320.0 or not coherent or
                        not all(math.isfinite(float(live_flow.get(side, -1)))
                                and float(live_flow.get(side, -1)) >= 1.0
                                for side in ('left', 'right'))):
                    raise RuntimeError(f"单步视觉位移 {moved:.1f}px 过大且缺少可信运动反馈，已停止")
                confirmed = _observe(target, grab_url=grab_url,
                    timeout=min(30.0, max(1.0, deadline-time.monotonic()-8)), require_ble=True)
                observations += 1
                shift = _norm(tuple(a-b for a,b in zip(center,_center_vector(confirmed))))
                record('overrun_stability_check', displacement_px=round(shift,1), observation=confirmed)
                if shift > 15.0:
                    raise RuntimeError('大步移动后目标身份/位置不稳定，已停止')
                observation, center = confirmed, _center_vector(confirmed)
                new_distance = distance = _distance(confirmed)
                adjustment_policy(distance, max_distance_px, trial_band_px)
                if in_grasp_region(confirmed):
                    return success('停车复核后目标进入示教抓取区域', distance)
                coherent = coherent_axial_motion(action, last_center, center)
                if not coherent:
                    raise RuntimeError('大步移动复核与指令方向不一致，已停止')
                motion_limit = 2.0
                record('overrun_recovery_allowed', action=action, moved_px=round(moved,1),
                       next_visual_motion_limit_px=motion_limit)
            if action in {"forward", "backward"} and abs(disparity_change) >= 8.0:
                expected_sign = 1 if action == "forward" else -1
                if disparity_change * expected_sign < 0:
                    raise RuntimeError(
                        "实测双目视差变化与内八字运动模型相反，"
                        "可能相机顺序/目标身份变化，已停止")
            if (moved >= 5.0 and not arc_forward
                    and (forward_cues is None or forward_cues['learnable'])):
                axis, response = measured_response(action, duration, last_center, center)
                responses[axis] = response
                record("local_response", axis=axis,
                       average_pixels_per_requested_window_s=response,
                       response_use='direction_only')
            elif action=='forward' and forward_cues is not None and not forward_cues['learnable']:
                record('forward_response_retained',reason='weak_motion_or_center_jitter',
                       learned_responses=responses)
            if action == 'forward' and coherent:
                # This micro-stage itself has just traversed a straight path.
                # It may issue a short rollback without the chassis-occluded
                # rear sensor, even when the pursuit handoff lacked evidence.
                if retreat_budget is None:
                    retreat_budget = SweptRetreatBudget(time.monotonic()+60.0,
                                                        'confirmed_micro_forward')
                else:
                    retreat_budget.expires_at = time.monotonic()+60.0
                blocked_actions.discard('backward')
                record('micro_retreat_evidence', source=retreat_budget.source,
                       pulses_reserved=retreat_budget.pulses, on_time_s=retreat_budget.on_time_s)
            depth_progress=approach_progress(action,last_center,center)
            bad_progress = (new_distance > previous_distance + MIN_PROGRESS_PX
                            or moved >= 5.0 and
                            new_distance >= previous_distance - MIN_PROGRESS_PX)
            if control_phase=='approach' and action in {'forward','turn_left','turn_right'}:
                # Approach is judged by rows/recentering, not grasp-hull X.
                bad_progress=depth_progress<2.0 and moved>=5.0
                if forward_cues is not None and forward_cues['supported']:
                    bad_progress=False
                record('approach_visual_progress',action=action,progress_px=round(depth_progress,1),
                       hull_distance_diagnostic_px=new_distance)
            if bad_progress:
                reason = (f"微调后凸包距离未改善 "
                          f"({previous_distance:.1f}→{new_distance:.1f}px)")
                anchor=previous_distance if forward_patience_anchor is None else forward_patience_anchor
                can_continue=(control_phase=='micro' and action=='forward'
                        and forward_patience_used<_APPROACH_PROFILE.hull_micro_forward_patience_steps
                        and forward_patience_candidate(observation,last_center,pulse_feedback,
                            previous_distance,anchor,before_observation=pulse_observation,
                            recent_forward=has_recent_forward()))
                record('forward_patience_evaluation',eligible=can_continue,
                       used=forward_patience_used,limit=_APPROACH_PROFILE.hull_micro_forward_patience_steps,
                       action=action,control_phase=control_phase,previous_distance_px=previous_distance,
                       distance_px=new_distance,anchor_distance_px=anchor)
                if can_continue:
                    if time.monotonic()+15>=deadline:
                        raise RuntimeError(reason+'；没有时间完成继续直行前的复核，已停止')
                    confirmed=_observe(target,grab_url=grab_url,
                        timeout=min(20,max(1,deadline-time.monotonic()-8)),require_ble=True)
                    observations+=1
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError('继续微调前收到急停，已停止')
                    shift=_norm(tuple(a-b for a,b in zip(center,_center_vector(confirmed))))
                    record('forward_patience_recheck',displacement_px=round(shift,1),observation=confirmed)
                    if shift>15:
                        raise RuntimeError('继续微调前目标移动或识别不稳定，已停止')
                    if in_grasp_region(confirmed):
                        return success('继续微调前复核已进入抓取区域',_distance(confirmed))
                    if forward_patience_candidate(confirmed,last_center,pulse_feedback,
                            previous_distance,anchor,before_observation=pulse_observation,
                            recent_forward=has_recent_forward()):
                        forward_patience_used+=1
                        forward_patience_anchor=anchor
                        observation,distance=confirmed,_distance(confirmed)
                        last_action,last_center='forward',_center_vector(confirmed)
                        pending_forward_patience=True
                        static_action=None
                        motion_limit=2.0
                        weak=forward_cues is not None and not forward_cues['learnable']
                        duration=min(pulse_cap,max(_APPROACH_PROFILE.hull_forward_first_s,
                            round(duration+PULSE_STEP_S,2) if weak else _APPROACH_PROFILE.hull_forward_first_s))
                        record('forward_patience_continue',previous_distance_px=previous_distance,
                               distance_px=distance,anchor_distance_px=anchor,
                               extra_step=forward_patience_used,
                               extra_step_limit=_APPROACH_PROFILE.hull_micro_forward_patience_steps,
                               next_window_s=duration,next_visual_motion_limit_px=2.0)
                        continue
                blocked_actions.add(action)
                # Replan after each stopped observation, within finite budgets.
                repair = (opposite_arc_repair(observation,last_center,center,action,
                    pulse_feedback,duration_s=duration,maximum_s=pulse_cap)
                    if arc_forward and moved <= 100.0 else None)
                if repair is None or repair.action in blocked_actions:
                    repair = select_adjustment(observation, responses=responses,
                        blocked_actions=blocked_actions, duration_s=duration,
                        maximum_s=pulse_cap,
                        allow_probe=control_phase=='micro' and probe_trials<MAX_LOW_CONFIDENCE_PROBES)
                record('correction_candidate',failed_action=action,
                       next_action=repair.action if repair else None,
                       learned_responses=responses,blocked_actions=sorted(blocked_actions),
                       observation=observation)
                if (moved < 5.0 or consecutive_corrections >= MAX_CONSECUTIVE_CORRECTIONS
                        or new_distance - previous_distance > MAX_CORRECTABLE_WORSENING_PX
                        or repair is None
                        or total_pulse_s + repair.duration_s > MAX_TOTAL_PULSE_S + 1e-9):
                    raise RuntimeError(reason + "，已停止")
                remaining = deadline - time.monotonic()
                if remaining < 15.0:
                    raise RuntimeError(reason + "；没有时间复核纠正，已停止")
                confirmed = _observe(target, grab_url=grab_url,
                    timeout=min(45.0, remaining - 8.0), require_ble=True)
                observations += 1
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("纠正复核期间收到急停，已停止")
                displacement = _norm(tuple(a - b for a, b in zip(
                    center, _center_vector(confirmed))))
                record("correction_stability_check", displacement_px=round(displacement, 1),
                       observation=confirmed)
                if displacement > 15.0:
                    raise RuntimeError("纠正前目标移动或识别不稳定，已停止")
                distance = _distance(confirmed)
                if in_grasp_region(confirmed):
                    return success("复核确认目标进入示教抓取区域（含模型检测余量）", distance)
                adjustment_policy(distance, max_distance_px, trial_band_px)
                repair = (opposite_arc_repair(confirmed,last_center,_center_vector(confirmed),
                    action,pulse_feedback,duration_s=duration,maximum_s=pulse_cap)
                    if arc_forward and moved <= 100.0 else None)
                if repair is None or repair.action in blocked_actions:
                    repair = select_adjustment(confirmed, responses=responses,
                        blocked_actions=blocked_actions, duration_s=duration,
                        maximum_s=pulse_cap,
                        allow_probe=control_phase=='micro' and probe_trials<MAX_LOW_CONFIDENCE_PROBES)
                if repair is None:
                    raise RuntimeError(reason + "；复核后无可改善动作，已停止")
                consecutive_corrections += 1
                motion_limit = 2.0
                if repair.source == 'measured_opposite_yaw_repair':
                    arc_repair_action,arc_repair_steps=repair.action,0
                    record('opposite_yaw_repair_selected',failed_action=action,
                           next_action=repair.action,predicted_improvement_px=repair.predicted_improvement_px,
                           short_window_s=repair.duration_s,max_followup_steps=6)
                record("corrective_replan", failed_action=action,
                       next_action=repair.action, duration_s=TRIAL_FIRST_S,
                       next_visual_motion_limit_px=2.0,
                       decision_source=repair.source,
                       correction_number=consecutive_corrections,
                       predicted_improvement_px=repair.predicted_improvement_px)
                observation = confirmed
                pending_repair = repair if control_phase=='approach' else None
                last_action, last_center = action, _center_vector(confirmed)
                duration = TRIAL_FIRST_S  # corrective moves never reuse a long window
                continue
            record("continue_alignment", previous_distance_px=previous_distance,
                   distance_px=new_distance, improvement_px=round(
                       previous_distance - new_distance, 1), pulses=pulses)
            if window_memory.remember(motor_action,duration,pulse_feedback):
                record('effective_window_learned',motor_action=motor_action,
                       window_s=window_memory.windows[motor_action],source='live_optical_motion')
            if plan.source.startswith('measured_opposite_yaw_repair') and action==arc_repair_action:
                arc_repair_steps+=1
                if arc_repair_steps<6:
                    pending_repair=AdjustmentPlan(action,min(pulse_cap,.50),
                        'measured_opposite_yaw_repair_followup')
                    record('opposite_yaw_repair_continue',action=action,
                           successful_steps=arc_repair_steps,max_steps=6,
                           distance_px=new_distance)
                else:
                    arc_repair_action=None
            if last_action is not None and action != last_action and moved < 5.0:
                raise RuntimeError("方向反复变化但底盘未移动，已停止")
            _, _, next_cap = adjustment_policy(
                new_distance, max_distance_px, trial_band_px)
            if moved < 5.0 and duration >= next_cap:
                stationary_at_cap += 1
                if stationary_at_cap >= 2:
                    raise RuntimeError("连续两步在微调时长上限仍无视觉进展，已停止")
            else:
                stationary_at_cap = 0
            if previous_distance-new_distance >= MIN_PROGRESS_PX or control_phase=='approach' and depth_progress>=2.0:
                consecutive_corrections = 0
                if previous_distance-new_distance>=4.0:
                    forward_patience_used=0
                    forward_patience_anchor=None
                opposite = {'forward':'backward','backward':'forward',
                            'turn_left':'turn_right','turn_right':'turn_left'}[action]
                blocked_actions.discard(opposite)
            if moved >= 5.0 or pulse_feedback.get('termination')=='visual_motion_limit':
                # Next selection restores this action's optically verified
                # effective window. Never convert pixels to a startup model.
                duration = PULSE_FIRST_S
            else:
                duration = next_pulse_duration(duration, moved,
                    minimum_s=TRIAL_FIRST_S, maximum_s=next_cap)
            last_action, last_center, distance = action, center, new_distance
        raise RuntimeError(f"达到 {max_pulses} 次微调上限，未进入示教凸包")
    except Exception as exc:
        if retreat_budget is not None:
            retreat_budget.invalidated = True
        record("stopped", reason=str(exc), pulses=pulses, distance_px=distance,
               total_pulse_s=total_pulse_s)
        return AlignmentResult(False, str(exc), observations, pulses,
                               distance, str(log_path))
    finally:
        if guard is not None:
            try:
                guard.close()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--esp32", action="store_true", help="显式允许受限实车微调")
    parser.add_argument("--max-pulses", type=int, default=_APPROACH_PROFILE.hull_micro_max_steps)
    parser.add_argument("--max-distance-px", type=float, default=180.0,
                        help="常规微调范围，非抓取容差")
    parser.add_argument("--trial-band-px", type=float, default=TRIAL_BAND_PX,
                        help="常规范围外的受限试探带；设为 0 禁用")
    args = parser.parse_args()
    result = align(args.target, max_pulses=args.max_pulses,
                   max_distance_px=args.max_distance_px,
                   trial_band_px=args.trial_band_px, real=args.esp32)
    print(json.dumps(result.__dict__, ensure_ascii=False), flush=True)
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
