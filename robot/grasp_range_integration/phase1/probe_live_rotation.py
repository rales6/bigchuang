"""Operator-supervised single bounded turn with immutable raw scan evidence.

Not a navigation bypass: rear blindness may use the operator's explicit
0.5 m all-around clearance confirmation ONLY for this manual diagnostic.
Only documented body returns may use that confirmation; unclassified near
rays never can. Autonomous navigation's map-support contract is unchanged.
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np

from grasp_range_integration.phase1.rotation_guard import (
    assess_rotation, installed_body_mask, body_supported_blind_arcs,
)


def calibration_pulse(*, ttl_ms, speed, read_scan, assess, send, stop, emit,
                      clock=time.monotonic):
    """Manual open-space measurement, NOT an autonomous angle authorization.

    One lease, no refresh. Do not infer acceleration or safety from target
    angular velocity. Operator-confirmed swept clearance is mandatory at CLI.
    """
    if not 100 <= ttl_ms <= 250 or not 1500 <= abs(speed) <= 3500:
        raise ValueError("Calibration limited to 100..250ms and 1500..3500mrad/s")
    try:
        anchor = read_scan(None, .25)
        check = assess(anchor)
        if not check["safe"] or not 0 <= clock()-anchor.received_s <= .30:
            raise RuntimeError("标定净空/扫描新鲜度不合格；不发送运动")
        issued = clock()
        send(speed, ttl_ms, issued-1.)
        emit("manual_calibration_pulse", angular_mrad_s=speed, ttl_ms=ttl_ms,
             renewable=False, autonomous_permission=False,
             ack_latency_s=clock()-issued)
        sequence = anchor.sequence
        while clock() < issued+ttl_ms/1000.:
            current = read_scan(sequence, .25)
            sequence = current.sequence
            if not assess(current)["safe"]:
                raise RuntimeError("标定中净空变化；立即停车")
        return "manual_single_pulse_complete"
    finally:
        stop()


def probe_clearance(scan, operator_clearance_confirmed=False):
    value = assess_rotation(scan)
    if value.get("min_distance_m", 0) < .50:
        return dict(value, safe=False, reason="probe_external_clearance_below_050m")
    if value["safe"]:
        return value
    angles, distances = np.asarray(scan.angles_rad), np.asarray(scan.distances_m)
    body = installed_body_mask(angles, distances)
    exterior = np.isfinite(angles) & np.isfinite(distances) & (distances > 0) & (distances <= 8) & ~body
    bearings = np.sort(angles[exterior] % (2 * math.pi))
    explained = False
    if len(bearings) >= 16:
        gap = np.max(np.diff(np.r_[bearings, bearings[0]+2*math.pi]))
        arcs, body_explained = body_supported_blind_arcs(angles, body, bearings)
        explained = gap <= math.radians(90) or body_explained
    if (operator_clearance_confirmed and explained
            and value["reason"] in {"unobserved_sector_without_safe_map_support",
                                    "installation_edge_occlusion_without_safe_map_support"}):
        return dict(value, safe=True, reason="operator_confirmed_rear_blind_clearance_manual_probe_only")
    return value


def stationary_preflight(read_scan, assess, record, *, required=5, maximum=30):
    """Wait stationary for consecutive safe frames; never mask close echoes.

    This retry is manual-probe-only. Every unsafe frame resets the streak;
    persistent obstacles cannot pass, and motion still checks each live frame.
    """
    streak = 0
    for _ in range(maximum):
        frame = read_scan()
        record(frame)
        assessment = assess(frame)
        streak = streak+1 if assessment["safe"] else 0
        if streak >= required:
            return frame
    raise RuntimeError("静止预检未获得连续{}帧净空；未连接底盘：{}".format(
        required, json.dumps(assessment)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--operator-clearance-confirmed", action="store_true")
    parser.add_argument("--map", required=True)
    parser.add_argument("--map-yaml", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-ttl-ms", type=int, default=None)
    parser.add_argument("--calibration-speed", type=int, default=-3500)
    args = parser.parse_args()
    if not args.enable_motion or not args.operator_clearance_confirmed:
        parser.error("Manual probe requires explicit motion and operator clearance confirmation")
    if args.calibration_ttl_ms is not None and not 100 <= args.calibration_ttl_ms <= 250:
        parser.error("Calibration TTL must be 100..250ms")
    if not 1500 <= abs(args.calibration_speed) <= 3500:
        parser.error("Calibration speed must be 1500..3500mrad/s")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    from raspberry_pi.config import LidarConfig, SerialConfig
    from raspberry_pi.esp32 import Esp32Client
    from raspberry_pi.known_map_navigation import KnownOccupancyMap, ScanMapLocalizer
    from raspberry_pi.lidar import N10LidarDriver
    from grasp_range_integration.phase1.fresh_motion_feedback import FreshScanStream, MotionFeedbackFence
    from grasp_range_integration.phase1.inspect_body_returns import describe
    from grasp_range_integration.phase1.turn_range_feedback import turn_range_evidence
    from grasp_range_integration.phase1.turn_start_recovery import run_turn_start_recovery
    known = KnownOccupancyMap.load(args.map, args.map_yaml, clearance_m=.30)
    localizer = ScanMapLocalizer(known, lidar_offset_x_m=.14, lidar_offset_y_m=0.)
    lidar = N10LidarDriver(LidarConfig(port="/dev/ttyACM0", baudrate=230400))
    stream, client = None, None
    data = {"operator_confirmed_clearance_m": .50, "translation_enabled": False,
            "maximum_attempt_seconds": 1.2, "frames": [], "events": []}
    def emit(kind, **fields):
        data["events"].append(dict(type=kind, time=time.time(), **fields))
        if kind in {"turn_start_feedback", "turn_start_complete", "turn_start_failure", "turn_start_lease"}:
            print(json.dumps(data["events"][-1], ensure_ascii=False), flush=True)
    try:
        stream = FreshScanStream(lidar.scans(), lidar, MotionFeedbackFence(), emit=emit)
        stationary_preflight(lambda: next(stream), lambda frame: probe_clearance(frame, True),
            lambda frame: data["frames"].append(dict(stage="before", **describe(frame))))
        client = Esp32Client(SerialConfig(link_mode="ble", ble_device_name="ESP32-Robot-Car",
                                        ble_address="AA:BB:CC:DD:EE:01",
                                        wheel_output_gains=(1.15, 1., .90, 1.)))
        client.heartbeat(); client.stop(); client.start()
        emit("robot_status", status=str(client.query_status()))
        def assess(frame):
            data["frames"].append(dict(stage="during", **describe(frame)))
            return probe_clearance(frame, True)
        def measure(before, after):
            from dataclasses import replace
            match = localizer.relative_motion(before, after)
            evidence = turn_range_evidence(before, after, match.transform,
                lidar_offset=localizer.lidar_offset, maximum_range=localizer.max_distance_m)
            emit("range_feedback", **evidence)
            return match if evidence["accepted"] else replace(match, converged=False)
        if args.calibration_ttl_ms is not None:
            from grasp_range_integration.phase1.turn_output_boost import send_boosted_turn
            data["result"] = calibration_pulse(ttl_ms=args.calibration_ttl_ms,
                speed=args.calibration_speed, read_scan=stream.latest_after,
                assess=assess, stop=client.stop, emit=emit,
                send=lambda speed, ttl, deadline: send_boosted_turn(client,
                    client.set_bounded_twist, speed, ttl, deadline, 6, emit))
        else:
            data["result"] = run_turn_start_recovery(error_rad=math.radians(-20),
            read_scan=stream.latest_after, measure=measure, assess=assess,
            send=lambda speed, ttl, deadline: client.set_bounded_twist(0, speed, ttl,
                                                       refresh_deadline_s=deadline),
            renew=client.allow_twist_to_refresh_for, stop=client.stop, emit=emit)
    except Exception as exc:
        data.update(result="error", error=str(exc))
        print("PROBE_ERROR", str(exc), flush=True)
    finally:
        try:
            if client is not None:
                client.stop()
            stopped = time.monotonic()
            if stream is not None:
                for _ in range(10):
                    frame = next(stream)
                    while frame.acquisition_start_s < stopped+.05:
                        frame = next(stream)
                    data["frames"].append(dict(stage="after", **describe(frame)))
                during = [f for f in data["frames"] if f["stage"] == "during"]
                if during:
                    from raspberry_pi.lidar.base import LaserScan
                    def restored(frame):
                        return LaserScan(np.asarray(frame["angles_rad"]),
                            np.asarray(frame["distances_m"], dtype=float), frame["timestamp_s"])
                    anchor = restored(during[0])
                    rows = []
                    for f in data["frames"]:
                        if f["stage"] != "after":
                            continue
                        current = restored(f)
                        match = localizer.relative_motion(anchor, current)
                        evidence = turn_range_evidence(anchor, current, match.transform,
                            lidar_offset=localizer.lidar_offset, maximum_range=localizer.max_distance_m)
                        rows.append({"icp_estimate_deg": math.degrees(match.transform.yaw_rad),
                                     "icp_converged": bool(match.converged),
                                     "rmse_m": float(match.rmse_m),
                                     "inlier_ratio": float(match.inlier_ratio),
                                     "raw_range_feedback": evidence})
                    data["post_stop_feedback"] = rows
                    emit("post_stop_rotation_estimate",
                         median_icp_deg=float(np.median([r["icp_estimate_deg"] for r in rows])),
                         validated_frames=sum(r["raw_range_feedback"]["accepted"] for r in rows),
                         total_frames=len(rows), control_permission=False)
                    print(json.dumps(data["events"][-1], ensure_ascii=False), flush=True)
        finally:
            if client is not None:
                client.close()
            if stream is not None:
                stream.close()
            else:
                lidar.close()
            with (output / "raw_rotation.json").open("x", encoding="utf-8") as destination:
                json.dump(data, destination, ensure_ascii=False, allow_nan=False)
            print("RAW_ROTATION", str(output / "raw_rotation.json"), flush=True)
    return 2 if data.get("result") == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
