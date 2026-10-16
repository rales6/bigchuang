"""Execute one short, lidar-guarded relative chassis motion.

This module intentionally supports only four explicit primitives.  It never
accepts raw wheel speeds from the web planner: distance/angle are bounded here,
and every real run owns both the lidar and ESP32 link until it has stopped.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from raspberry_pi.config import LidarConfig, SerialConfig
from raspberry_pi.esp32 import Esp32Client
from raspberry_pi.known_map_navigation import ScanMapLocalizer
from raspberry_pi.lidar import N10LidarDriver
from raspberry_pi.bluetooth_recovery import recover_bluetooth


DIRECTIONS = {"forward", "backward", "left", "right"}


def _event(name: str, **fields) -> None:
    print(json.dumps({"event": name, "at": time.time(), **fields}, ensure_ascii=False), flush=True)


def _connect_client(args) -> Esp32Client:
    last_error: Exception | None = None
    for attempt in range(1, int(args.ble_retries) + 2):
        _event("ble_connect_attempt", attempt=attempt, address=args.ble_address)
        try:
            client = Esp32Client(SerialConfig(
                link_mode="ble",
                ble_device_name=args.ble_name,
                ble_address=args.ble_address,
                ble_connect_timeout_s=float(args.ble_connect_timeout),
            ))
            client.heartbeat()
            _event("ble_connected", attempt=attempt)
            return client
        except Exception as exc:
            last_error = exc
            _event("ble_connect_failed", attempt=attempt, error=str(exc))
            if attempt <= int(args.ble_retries):
                recovery = recover_bluetooth(args.ble_address)
                _event("ble_recovery", attempt=attempt, result=recovery.to_dict())
                time.sleep(float(args.ble_retry_delay))
    raise RuntimeError(f"ESP32 蓝牙连接失败，已重试 {int(args.ble_retries) + 1} 次：{last_error}")


def _next_scan(iterator):
    while True:
        scan = next(iterator)
        if len(scan.points(0.12, 8.0)) >= 60:
            return scan


def _sector_clearance(scan, center_rad: float, half_width_deg: float = 28.0) -> float:
    angles = np.asarray(scan.angles_rad, dtype=np.float64)
    distances = np.asarray(scan.distances_m, dtype=np.float64)
    delta = np.arctan2(np.sin(angles - center_rad), np.cos(angles - center_rad))
    valid = (
        np.isfinite(distances)
        & (distances >= 0.12)
        & (distances <= 8.0)
        & (np.abs(delta) <= math.radians(half_width_deg))
    )
    values = distances[valid]
    if len(values) < 5:
        return 0.0
    return float(np.percentile(values, 15.0))


def _safe_for(direction: str, scan, stop_distance_m: float) -> tuple[bool, float]:
    if direction == "forward":
        clearance = _sector_clearance(scan, 0.0)
        return clearance >= stop_distance_m, clearance
    if direction == "backward":
        clearance = _sector_clearance(scan, math.pi)
        return clearance >= stop_distance_m, clearance
    distances = np.asarray(scan.distances_m, dtype=np.float64)
    valid = distances[np.isfinite(distances) & (distances >= 0.12) & (distances <= 2.0)]
    clearance = float(np.percentile(valid, 8.0)) if len(valid) >= 12 else 0.0
    return clearance >= 0.30, clearance


def _valid_increment(match, direction: str) -> float:
    if not match.converged or not math.isfinite(match.rmse_m):
        return 0.0
    if match.rmse_m > 0.12 or match.inlier_ratio < 0.45:
        return 0.0
    if direction in {"forward", "backward"}:
        value = math.hypot(match.transform.x_m, match.transform.y_m)
        return value if value <= 0.35 else 0.0
    value = abs(match.transform.yaw_rad)
    return value if value <= math.radians(55.0) else 0.0


def _verified_reverse_increment(match) -> float:
    """Accept only a short, mostly straight backward scan displacement."""
    if not match.converged or not math.isfinite(match.rmse_m):
        return 0.0
    if match.rmse_m > 0.12 or match.inlier_ratio < 0.45:
        return 0.0
    motion = match.transform
    x = float(motion.x_m)
    y = float(motion.y_m)
    yaw = float(motion.yaw_rad)
    if not all(math.isfinite(value) for value in (x, y, yaw)):
        return 0.0
    if x >= -0.015 or abs(y) > 0.06 or abs(yaw) > math.radians(10.0):
        return 0.0
    return -x if -x <= 0.35 else 0.0


def _short_reverse_pulse_s(distance_m: float) -> float:
    return min(0.22, max(0.15, float(distance_m) / 0.405))


def _verified_turn_increment(match, direction: str) -> float:
    """Reject scan-matching jitter and rotation opposite to the command."""
    if not match.converged or not math.isfinite(match.rmse_m):
        return 0.0
    if match.rmse_m > 0.12 or match.inlier_ratio < 0.45:
        return 0.0
    yaw = float(match.transform.yaw_rad)
    signed = yaw if direction == "left" else -yaw
    # The old controller added |yaw| from every pair of consecutive scans.
    # Sub-degree matcher noise could therefore appear as tens of degrees of
    # progress even when static friction kept the chassis still.
    if not math.radians(1.2) <= signed <= math.radians(25.0):
        return 0.0
    return signed


def _verified_settle_increment(match, direction: str) -> float:
    """Account for rotation while the stop command takes effect.

    Unlike a moving scan, an untrusted stopped scan must abort the next
    segment; silently re-anchoring it loses real rotation on every pause.
    """
    if not match.converged or not math.isfinite(match.rmse_m):
        raise RuntimeError("停车后雷达转角无法确认，已停止后续转向")
    if match.rmse_m > 0.12 or match.inlier_ratio < 0.45:
        raise RuntimeError("停车后雷达匹配质量不足，已停止后续转向")
    yaw = float(match.transform.yaw_rad)
    signed = yaw if direction == "left" else -yaw
    if not math.isfinite(signed) or signed < -math.radians(2.0) or signed > math.radians(25.0):
        raise RuntimeError("停车后转角异常，已停止后续转向")
    return signed if signed >= math.radians(1.2) else 0.0


def _arc_stop_lead_deg(target_deg: float, settled_deg: list[float],
                       minimum_deg: float = 16.0) -> float:
    """Leave room for measured stop latency and a possible next scan jump."""
    observed = max(settled_deg[-3:], default=0.0)
    return min(target_deg * 0.45, max(minimum_deg, observed + 4.0))


def _arc_next_scan_would_overshoot(progress: float, segment_goal: float,
                                  segment_start: float, last_increment: float) -> bool:
    if progress - segment_start < math.radians(1.2):
        return False
    predicted_next = max(math.radians(3.0), last_increment)
    return progress + predicted_next >= segment_goal


def _next_turn_speed(current: int, *, observed_rate_deg_s: float,
                     maximum: int, remaining_deg: float) -> int:
    """Raise yaw command only when measured rotation is inadequate."""
    if observed_rate_deg_s >= 8.0 or remaining_deg <= 10.0:
        return current
    return min(maximum, current + 200)


def _arc_clearance(direction: str, scan, required_m: float) -> tuple[bool, float]:
    """Guard the forward and turning-side swept path, not the rear chassis."""
    front = _sector_clearance(scan, 0.0, half_width_deg=48.0)
    side = _sector_clearance(
        scan, math.pi / 2 if direction == "left" else -math.pi / 2,
        half_width_deg=42.0,
    )
    clearance = min(front, side)
    return front >= required_m and side >= required_m, clearance


def _short_arc_retreat(client, scans, localizer, before, direction: str,
                       required_m: float):
    """One bounded reverse pulse; the rear lidar is chassis-occluded.

    This is deliberately not a general backward-drive primitive.  It may be
    used only once within a turn, and never treats a rear scan as free space.
    """
    client.stop()
    _event("arc_retreat_started", direction=direction, speed_mm_s=405,
           pulse_s=0.18, nominal_limit_m=0.073,
           rear_lidar_used=False)
    try:
        # A short ESP32 TTL is a second bound if the host is delayed while
        # stopping the pulse.  Do not refresh this command.
        client.set_twist(-405, 0, ttl_ms=220)
        time.sleep(0.18)
    finally:
        client.stop()
    after = _next_scan(scans)
    match = localizer.relative_motion(before, after)
    moved_m = _valid_increment(match, "backward")
    safe, clearance = _arc_clearance(direction, after, required_m)
    _event("arc_retreat_result", direction=direction,
           measured_m=round(moved_m, 3),
           clearance_m=round(clearance, 3),
           recovered=bool(safe and 0.015 <= moved_m <= 0.11))
    if moved_m < 0.015:
        raise RuntimeError("左/右前方净空不足；短回退后未确认位移，已停车，不再盲倒")
    if moved_m > 0.11:
        raise RuntimeError(f"短回退实测 {moved_m:.2f}m 超过允许范围，已停车")
    if not safe:
        raise RuntimeError(f"短回退后转向空间仍不足：{clearance:.2f}m；已停车")
    return after


def _run_forward_arc_turn(args, direction: str, target: float) -> dict:
    """Steer a rolling chassis in short radar-checked arcs, never in place."""
    if not 250 <= int(args.arc_speed) <= 405:
        raise ValueError("前进弧线线速度必须在 250..405 mm/s")
    if not 500 <= int(args.arc_turn_speed) <= int(args.arc_max_turn_speed) <= 2100:
        raise ValueError("前进弧线角速度超出受限范围")
    if not 0.5 <= float(args.arc_max_travel_m) <= 1.6:
        raise ValueError("前进弧线总位移上限必须在 0.5..1.6m")
    if not math.isfinite(float(args.arc_clearance_m)) or not 0.45 <= float(args.arc_clearance_m) <= 1.2:
        raise ValueError("前进弧线净空阈值必须在 0.45..1.2m")

    client = None
    lidar = None
    localizer = ScanMapLocalizer(None)
    progress = 0.0
    traveled_m = 0.0
    segments = 0
    valid_updates = 0
    angular_speed = int(args.arc_turn_speed)
    settled_degrees: list[float] = []
    retreat_used = False
    direction_sign = 1 if direction == "left" else -1
    operation_started = time.monotonic()
    try:
        client = _connect_client(args)
        if args.probe_only:
            result = {"ok": True, "mode": "probe", "ble": "connected"}
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return result
        lidar = N10LidarDriver(LidarConfig(
            port=args.lidar_port, baudrate=230400, min_scan_points=45,
        ))
        client.start()
        scans = iter(lidar.scans())
        current = _next_scan(scans)
        safe, clearance = _arc_clearance(direction, current, float(args.arc_clearance_m))
        if not safe:
            current = _short_arc_retreat(
                client, scans, localizer, current, direction, float(args.arc_clearance_m),
            )
            retreat_used = True
            safe, clearance = _arc_clearance(direction, current, float(args.arc_clearance_m))
        started_at = time.monotonic()
        deadline = started_at + min(30.0, max(8.0, math.degrees(target) / 10.0 + 6.0))
        _event("arc_turn_started", direction=direction, target_deg=round(math.degrees(target), 2),
               linear_mm_s=int(args.arc_speed), angular_mrad_s=angular_speed,
               segment_deg=30, max_travel_m=float(args.arc_max_travel_m),
               clearance_m=round(clearance, 3),
               setup_s=round(started_at - operation_started, 2))
        while True:
            stop_lead_deg = _arc_stop_lead_deg(math.degrees(target), settled_degrees)
            stop_threshold = target - math.radians(stop_lead_deg)
            if progress >= stop_threshold:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"弧线转向超时，已停车；可信转角 {math.degrees(progress):.1f}/{math.degrees(target):.1f}°")
            safe, clearance = _arc_clearance(direction, current, float(args.arc_clearance_m))
            if not safe:
                if retreat_used:
                    raise RuntimeError(f"前进弧线净空不足：{clearance:.2f}m；短回退次数已用尽，已停车")
                current = _short_arc_retreat(
                    client, scans, localizer, current, direction, float(args.arc_clearance_m),
                )
                retreat_used = True
                continue
            segment_goal = min(stop_threshold, progress + math.radians(30.0))
            segment_start_angle = progress
            segment_start_travel = traveled_m
            yaw_anchor = current
            travel_anchor = current
            segment_started = time.monotonic()
            last_yaw_at = segment_started
            last_travel_at = segment_started
            rate_window_started = segment_started
            yaw_at_window = progress
            client.set_twist(int(args.arc_speed), direction_sign * angular_speed,
                             ttl_ms=args.command_ttl_ms)
            _event("arc_segment_started", segment=segments + 1,
                   target_deg=round(math.degrees(segment_goal), 2),
                   angular_mrad_s=direction_sign * angular_speed)
            while progress < segment_goal:
                current = _next_scan(scans)
                now = time.monotonic()
                safe, clearance = _arc_clearance(direction, current, float(args.arc_clearance_m))
                if not safe:
                    client.stop()
                    if retreat_used:
                        raise RuntimeError(f"弧线行驶中净空不足：{clearance:.2f}m；短回退次数已用尽，已停车")
                    current = _short_arc_retreat(
                        client, scans, localizer, current, direction, float(args.arc_clearance_m),
                    )
                    retreat_used = True
                    yaw_anchor = current
                    travel_anchor = current
                    last_yaw_at = now
                    last_travel_at = now
                    segment_start_travel = traveled_m
                    client.set_twist(int(args.arc_speed), direction_sign * angular_speed,
                                     ttl_ms=args.command_ttl_ms)
                    continue
                yaw_match = localizer.relative_motion(yaw_anchor, current)
                yaw_increment = _verified_turn_increment(yaw_match, direction)
                if yaw_increment:
                    progress += yaw_increment
                    valid_updates += 1
                    yaw_anchor = current
                    last_yaw_at = now
                travel_match = localizer.relative_motion(travel_anchor, current)
                translation = _valid_increment(travel_match, "forward")
                if 0.025 <= translation <= 0.22:
                    traveled_m += translation
                    travel_anchor = current
                    last_travel_at = now
                if traveled_m - segment_start_travel > 0.35:
                    raise RuntimeError("单段弧线已前进 0.35m 但转角不足，已停车")
                if traveled_m > float(args.arc_max_travel_m):
                    raise RuntimeError("弧线转向达到总前进距离上限，已停车")
                if now - last_travel_at > 2.0:
                    raise RuntimeError("弧线前进未克服静摩擦或雷达无法确认位移，已停车")
                if now - last_yaw_at > 3.0:
                    raise RuntimeError("弧线前进后仍无法确认转向，已停车")
                if now >= deadline:
                    raise RuntimeError("弧线转向达到总时间上限，已停车")
                if _arc_next_scan_would_overshoot(
                    progress, segment_goal, segment_start_angle, yaw_increment,
                ):
                    _event("arc_predictive_stop", segment=segments + 1,
                           angle_deg=round(math.degrees(progress), 2),
                           segment_goal_deg=round(math.degrees(segment_goal), 2),
                           stop_lead_deg=round(stop_lead_deg, 2),
                           last_scan_deg=round(math.degrees(yaw_increment), 2))
                    break
                boosted = False
                if now - rate_window_started >= 0.7:
                    rate = math.degrees(progress - yaw_at_window) / max(0.1, now - rate_window_started)
                    rate_window_started = now
                    yaw_at_window = progress
                    if rate < 8.0 and angular_speed < int(args.arc_max_turn_speed):
                        angular_speed = min(int(args.arc_max_turn_speed), angular_speed + 300)
                        client.set_twist(int(args.arc_speed), direction_sign * angular_speed,
                                         ttl_ms=args.command_ttl_ms)
                        boosted = True
                        _event("arc_turn_boost", angular_mrad_s=direction_sign * angular_speed,
                               recent_rate_deg_s=round(rate, 2),
                               verified_angle_deg=round(math.degrees(progress), 2))
                if not boosted and not getattr(client, "maintenance_running", False):
                    client.refresh_twist(int(args.arc_speed), direction_sign * angular_speed,
                                         ttl_ms=args.command_ttl_ms)
                _event("arc_turn_progress", segment=segments + 1,
                       direction=direction,
                       angle_deg=round(math.degrees(progress), 2),
                       target_deg=round(math.degrees(target), 2),
                       travel_m=round(traveled_m, 3), clearance_m=round(clearance, 3),
                       angular_mrad_s=direction_sign * angular_speed,
                       valid_updates=valid_updates)
            client.stop()
            segments += 1
            time.sleep(0.12)
            # Consume one potentially buffered revolution, but retain the
            # pre-stop anchor so its rotation is not discarded.
            _next_scan(scans)
            settled = _next_scan(scans)
            settle_match = localizer.relative_motion(yaw_anchor, settled)
            settle_yaw = _verified_settle_increment(settle_match, direction)
            progress += settle_yaw
            settled_degrees.append(math.degrees(settle_yaw))
            if settle_yaw:
                valid_updates += 1
            settle_translation = _valid_increment(
                localizer.relative_motion(travel_anchor, settled), "forward",
            )
            if 0.025 <= settle_translation <= 0.22:
                traveled_m += settle_translation
            current = settled
            _event("arc_segment_settled", segment=segments,
                   extra_deg=round(math.degrees(settle_yaw), 2),
                   angle_deg=round(math.degrees(progress), 2),
                   travel_m=round(traveled_m, 3))
            if traveled_m > float(args.arc_max_travel_m):
                raise RuntimeError("停车惯性后已达到弧线总位移上限，已停车")
            _event("arc_segment_finished", segment=segments,
                   angle_deg=round(math.degrees(progress), 2),
                   travel_m=round(traveled_m, 3))
        result = {
            "ok": True, "mode": "real", "turn_mode": "forward_arc",
            "direction": direction,
            "measured_angle_deg": round(math.degrees(progress), 2),
            "measured_distance_m": round(traveled_m, 3),
            "segments": segments, "valid_updates": valid_updates,
            "settled_deg": [round(value, 2) for value in settled_degrees],
            "short_retreat_used": retreat_used,
            "elapsed_s": round(time.monotonic() - started_at, 2),
        }
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result
    finally:
        if client is not None:
            try:
                if not args.probe_only:
                    client.stop()
            finally:
                client.close()
        if lidar is not None:
            lidar.close()


def _run_short_backward(args, target: float) -> dict:
    """One operator-requested <=10 cm reverse, never using rear self-echoes."""
    if not 0.05 <= target <= 0.10:
        raise ValueError("后向雷达被车身遮挡；单次后退只能为 0.05..0.10m")
    client = None
    lidar = None
    started_at = time.monotonic()
    try:
        client = _connect_client(args)
        if args.probe_only:
            result = {"ok": True, "mode": "probe", "ble": "connected"}
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return result
        lidar = N10LidarDriver(LidarConfig(
            port=args.lidar_port, baudrate=230400, min_scan_points=45,
        ))
        localizer = ScanMapLocalizer(None)
        client.start()
        scans = iter(lidar.scans())
        before = _next_scan(scans)
        pulse_s = _short_reverse_pulse_s(target)
        ttl_ms = int(pulse_s * 1000) + 40
        _event("short_reverse_started", target_m=target, speed_mm_s=405,
               pulse_s=round(pulse_s, 3), ttl_ms=ttl_ms,
               rear_lidar_used=False, rear_clearance_unverified=True)
        try:
            client.set_twist(-405, 0, ttl_ms=ttl_ms)
            time.sleep(pulse_s)
        finally:
            client.stop()
        time.sleep(0.12)
        _next_scan(scans)  # Discard a potentially buffered pre-stop scan.
        after = _next_scan(scans)
        moved_m = _verified_reverse_increment(localizer.relative_motion(before, after))
        _event("short_reverse_result", target_m=target,
               measured_m=round(moved_m, 3), rear_lidar_used=False,
               elapsed_s=round(time.monotonic() - started_at, 2))
        if moved_m > 0.12:
            raise RuntimeError(f"短后退实测 {moved_m:.2f}m 超过 0.12m 上限，已停车")
        if moved_m < max(0.03, target - 0.04):
            raise RuntimeError(f"短后退未确认足够位移（实测 {moved_m:.2f}/{target:.2f}m），已停车")
        result = {"ok": True, "mode": "real", "direction": "backward",
                  "guard_mode": "single_bounded_pulse_no_rear_lidar",
                  "requested_distance_m": target,
                  "measured_distance_m": round(moved_m, 3),
                  "elapsed_s": round(time.monotonic() - started_at, 2)}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result
    finally:
        if client is not None:
            try:
                if not args.probe_only:
                    client.stop()
            finally:
                client.close()
        if lidar is not None:
            lidar.close()


def run(args) -> dict:
    direction = str(args.direction)
    if direction not in DIRECTIONS:
        raise ValueError("不支持的运动方向")
    target = float(args.distance_m) if direction in {"forward", "backward"} else math.radians(float(args.angle_deg))
    if direction in {"forward", "backward"} and not 0.05 <= target <= 1.50:
        raise ValueError("单次直行距离必须在 0.05..1.50m")
    if direction == "backward" and target > 0.10:
        raise ValueError("后向雷达被车身遮挡；单次后退最多 0.10m")
    if direction in {"left", "right"} and not math.radians(5) <= target <= math.radians(180):
        raise ValueError("单次转向角度必须在 5..180°")
    if not 500 <= int(args.turn_speed) <= int(args.max_turn_speed) <= 2400:
        raise ValueError("转向命令必须满足 500 <= 起步值 <= 上限 <= 2400 mrad/s")
    if args.simulate:
        result = {"ok": True, "mode": "simulate", "direction": direction,
                  "distance_m": args.distance_m, "angle_deg": args.angle_deg,
                  "turn_mode": "forward_arc" if direction in {"left", "right"} else "straight",
                  "guard_mode": "single_bounded_pulse_no_rear_lidar" if direction == "backward" else "lidar_guarded"}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result

    _event(
        "motion_request", direction=direction, distance_m=args.distance_m,
        angle_deg=args.angle_deg,
        turn_mode="forward_arc" if direction in {"left", "right"} else "straight",
        guard_mode="single_bounded_pulse_no_rear_lidar" if direction == "backward" else "lidar_guarded",
        linear_speed_mm_s=args.arc_speed if direction in {"left", "right"} else args.speed,
        turn_speed_mrad_s=args.arc_turn_speed if direction in {"left", "right"} else 0,
        max_turn_speed_mrad_s=args.arc_max_turn_speed if direction in {"left", "right"} else 0,
        stop_distance_m=args.stop_distance,
        probe_only=bool(args.probe_only),
    )
    if direction in {"left", "right"}:
        return _run_forward_arc_turn(args, direction, target)
    if direction == "backward":
        return _run_short_backward(args, target)
    client = None
    lidar = None
    localizer = ScanMapLocalizer(None)
    progress = 0.0
    speed = int(args.speed)
    turn_speed = int(args.turn_speed)
    valid_updates = 0
    stalled_updates = 0
    operation_started = time.monotonic()
    motion_started = None
    try:
        client = _connect_client(args)
        if args.probe_only:
            result = {"ok": True, "mode": "probe", "ble": "connected"}
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return result
        lidar = N10LidarDriver(LidarConfig(port=args.lidar_port, baudrate=230400, min_scan_points=45))
        client.start()
        scans = iter(lidar.scans())
        previous = _next_scan(scans)
        safe, clearance = _safe_for(direction, previous, args.stop_distance)
        if not safe:
            raise RuntimeError(f"运动方向净空不足：{clearance:.2f}m")
        linear = speed * (1 if direction == "forward" else -1) if direction in {"forward", "backward"} else 0
        angular = turn_speed * (1 if direction == "left" else -1) if direction in {"left", "right"} else 0
        client.set_twist(linear, angular, ttl_ms=args.command_ttl_ms)
        motion_started = time.monotonic()
        if linear:
            # Keep a second, time-based travel bound even when scan matching
            # produces no usable increments.  At the current 405 mm/s command
            # a failed matcher must not let a 1 m request run for 12+ seconds.
            nominal_m_s = max(0.10, abs(linear) / 1000.0)
            motion_timeout = min(8.0, max(2.0, target / nominal_m_s * 1.4 + 0.8))
        else:
            motion_timeout = min(12.0, max(3.0, target / math.radians(25.0) * 1.3 + 1.5))
        deadline = motion_started + motion_timeout
        if linear:
            absolute_timeout = min(15.0, max(motion_timeout, target / 0.05 + 2.0))
            minimum_rate = 0.025
            meaningful_increment = 0.002
        else:
            absolute_timeout = min(24.0, max(8.0, motion_timeout, target / math.radians(10.0) + 4.0))
            minimum_rate = math.radians(8.0)
            meaningful_increment = math.radians(1.2)
        absolute_deadline = motion_started + absolute_timeout
        last_meaningful_progress_at = motion_started
        turn_anchor = previous
        last_turn_boost_at = motion_started
        turn_progress_at_boost = 0.0
        high_speed_started_at = None
        high_speed_start_progress = 0.0
        observed_rate = 0.0
        _event(
            "motion_started", direction=direction, target=target,
            timeout_s=round(motion_timeout, 2), absolute_timeout_s=round(absolute_timeout, 2),
            setup_s=round(motion_started - operation_started, 2),
            angular_mrad_s=angular,
        )
        while progress < target and time.monotonic() < deadline:
            current = _next_scan(scans)
            safe, clearance = _safe_for(direction, current, args.stop_distance)
            if not safe:
                raise RuntimeError(f"雷达发现障碍，已停车；剩余净空 {clearance:.2f}m")
            match = localizer.relative_motion(previous if linear else turn_anchor, current)
            increment = (_valid_increment(match, direction) if linear
                         else _verified_turn_increment(match, direction))
            if increment > 0:
                progress += increment
                valid_updates += 1
                stalled_updates = 0
                now = time.monotonic()
                if not linear:
                    turn_anchor = current
                if increment >= meaningful_increment:
                    last_meaningful_progress_at = now
                elapsed = max(0.1, now - motion_started)
                observed_rate = progress / elapsed
                if valid_updates >= 5 and observed_rate >= minimum_rate:
                    predicted_remaining_s = max(0.0, target - progress) / observed_rate
                    adaptive_deadline = now + predicted_remaining_s * 1.35 + 0.6
                    deadline = max(deadline, min(absolute_deadline, adaptive_deadline))
            else:
                stalled_updates += 1
            if linear:
                previous = current
            if stalled_updates >= 5 and linear:
                speed = min(int(args.max_speed), speed + 20)
                linear = speed * (1 if direction == "forward" else -1)
                stalled_updates = 0
                client.set_twist(linear, 0, ttl_ms=args.command_ttl_ms)
            now = time.monotonic()
            turn_boosted = False
            if angular and now - last_turn_boost_at >= 0.85:
                window_s = max(0.1, now - last_turn_boost_at)
                rate_deg_s = math.degrees(progress - turn_progress_at_boost) / window_s
                requested = _next_turn_speed(
                    turn_speed, observed_rate_deg_s=rate_deg_s,
                    maximum=int(args.max_turn_speed),
                    remaining_deg=math.degrees(max(0.0, target - progress)),
                )
                last_turn_boost_at = now
                turn_progress_at_boost = progress
                if requested > turn_speed:
                    turn_speed = requested
                    angular = turn_speed * (1 if direction == "left" else -1)
                    client.set_twist(0, angular, ttl_ms=args.command_ttl_ms)
                    turn_boosted = True
                    if turn_speed >= int(args.max_turn_speed):
                        high_speed_started_at = now
                        high_speed_start_progress = progress
                    _event("turn_startup_boost", angular_mrad_s=angular,
                           recent_rate_deg_s=round(rate_deg_s, 2),
                           verified_angle_deg=round(math.degrees(progress), 2))
            if not turn_boosted and not getattr(client, "maintenance_running", False):
                client.refresh_twist(linear, angular, ttl_ms=args.command_ttl_ms)
            if high_speed_started_at is not None and now - high_speed_started_at >= 1.2:
                high_speed_rate = math.degrees(progress - high_speed_start_progress) / (now - high_speed_started_at)
                if high_speed_rate < 3.0:
                    raise RuntimeError(
                        f"最高受限转速下仍未克服静摩擦或雷达无法确认转角"
                        f"（实测 {high_speed_rate:.1f}°/s），已停车"
                    )
            stall_limit_s = 2.2 if linear else 3.2
            if now - last_meaningful_progress_at >= stall_limit_s:
                unit_progress = progress if linear else math.degrees(progress)
                unit_target = target if linear else math.degrees(target)
                unit = "m" if linear else "°"
                raise RuntimeError(
                    f"雷达位移连续 {stall_limit_s:.1f} 秒无可信进展，已停车；累计 {unit_progress:.3f}/{unit_target:.3f}{unit}"
                )
            print(json.dumps({
                "event": "motion_progress", "direction": direction,
                "progress": round(progress if linear else math.degrees(progress), 3),
                "target": round(target if linear else math.degrees(target), 3),
                "clearance_m": round(clearance, 3),
                "linear_speed_mm_s": speed if linear else 0,
                "angular_mrad_s": angular,
                "valid_updates": valid_updates,
                "observed_rate": round(observed_rate if linear else math.degrees(observed_rate), 3),
                "deadline_remaining_s": round(max(0.0, deadline - now), 2),
            }, ensure_ascii=False), flush=True)
        if progress < target:
            unit_progress = progress if linear else math.degrees(progress)
            unit_target = target if linear else math.degrees(target)
            unit = "m" if linear else "°"
            raise RuntimeError(
                f"达到运动时间上限，已停车；雷达累计 {unit_progress:.3f}/{unit_target:.3f}{unit}"
            )
        result = {
            "ok": True, "mode": "real", "direction": direction,
            "measured_distance_m": round(progress, 3) if linear else 0.0,
            "measured_angle_deg": round(math.degrees(progress), 2) if angular else 0.0,
            "valid_updates": valid_updates,
            "elapsed_s": round(time.monotonic() - (motion_started or operation_started), 2),
            "setup_s": round((motion_started or operation_started) - operation_started, 2),
        }
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result
    finally:
        if client is not None:
            try:
                if not args.probe_only:
                    client.stop()
            finally:
                client.close()
        if lidar is not None:
            lidar.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="雷达保护的单次相对运动")
    parser.add_argument("--direction", required=True, choices=sorted(DIRECTIONS))
    parser.add_argument("--distance-m", type=float, default=0.30)
    parser.add_argument("--angle-deg", type=float, default=30.0)
    parser.add_argument("--speed", type=int, default=405)
    parser.add_argument("--max-speed", type=int, default=460)
    parser.add_argument("--turn-speed", type=int, default=2100)
    parser.add_argument("--max-turn-speed", type=int, default=2400)
    parser.add_argument("--arc-speed", type=int, default=405)
    parser.add_argument("--arc-turn-speed", type=int, default=1500)
    parser.add_argument("--arc-max-turn-speed", type=int, default=2100)
    parser.add_argument("--arc-max-travel-m", type=float, default=1.4)
    parser.add_argument("--arc-clearance-m", type=float, default=0.55)
    parser.add_argument("--stop-distance", type=float, default=0.45)
    parser.add_argument("--command-ttl-ms", type=int, default=650)
    parser.add_argument("--lidar-port", default="/dev/ttyACM0")
    parser.add_argument("--ble-name", default="ESP32-Robot-Car")
    parser.add_argument("--ble-address", default="AA:BB:CC:DD:EE:01")
    parser.add_argument("--ble-connect-timeout", type=float, default=10.0)
    parser.add_argument("--ble-retries", type=int, default=2)
    parser.add_argument("--ble-retry-delay", type=float, default=2.0)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
