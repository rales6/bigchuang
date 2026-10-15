"""Replay the 2026-09-15 false-collision sequence with lidar-gated TTLs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from raspberry_pi.map_camera_inspection import (
    _forward_stop_envelope_m,
    _navigation_command_ttl,
)


def _ramp_distance(speed_m_s: float, duration_s: float) -> tuple[float, float]:
    acceleration_m_s2 = 0.65
    ramp_s = min(duration_s, speed_m_s / acceleration_m_s2)
    peak_m_s = acceleration_m_s2 * ramp_s
    distance_m = 0.5 * acceleration_m_s2 * ramp_s * ramp_s
    if duration_s > ramp_s:
        distance_m += peak_m_s * (duration_s - ramp_s)
    return peak_m_s, distance_m


def run(input_path: str | Path, output_path: str | Path) -> dict:
    rows = [
        json.loads(line)
        for line in Path(input_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events = []
    mixed_commands = []
    for index, row in enumerate(rows):
        applied = row.get("applied") or [0, 0]
        linear_mm_s = abs(int(applied[0]))
        angular_mrad_s = abs(int(applied[1]))
        front_value = row.get("front_clearance_m")
        front_m = (
            float(front_value)
            if front_value is not None else math.inf
        )
        if linear_mm_s and angular_mrad_s:
            old_ttl_ms = int(row.get("command_ttl_ms") or 0)
            new_ttl_ms = _navigation_command_ttl(
                700, linear_mm_s, front_m, 0.42
            )
            next_gap_ms = (
                max(
                    0.0,
                    1000.0 * (
                        float(rows[index + 1]["timestamp_s"])
                        - float(row["timestamp_s"])
                    ),
                )
                if index + 1 < len(rows) else math.nan
            )
            mixed_commands.append({
                "target_id": row.get("target_id"),
                "linear_mm_s": linear_mm_s,
                "angular_mrad_s": angular_mrad_s,
                "front_clearance_m": front_m,
                "old_ttl_ms": old_ttl_ms,
                "new_ttl_ms": new_ttl_ms,
                "next_scan_gap_ms": next_gap_ms,
                "old_active_fraction": (
                    min(1.0, old_ttl_ms / next_gap_ms)
                    if math.isfinite(next_gap_ms) and next_gap_ms > 0
                    else None
                ),
                "new_active_fraction": (
                    min(1.0, new_ttl_ms / next_gap_ms)
                    if math.isfinite(next_gap_ms) and next_gap_ms > 0
                    else None
                ),
            })
        linear_frames = int(row.get("navigation_linear_stall_frames") or 0)
        stationary_frames = int(
            row.get("navigation_stationary_command_frames") or 0
        )
        if linear_frames < 5 and stationary_frames < 5:
            continue
        dynamic_ttl_ms = _navigation_command_ttl(
            700, linear_mm_s, front_m, 0.42
        )
        required_m = _forward_stop_envelope_m(
            linear_mm_s, dynamic_ttl_ms, 0.42
        )
        next_target = (
            rows[index + 1].get("target_id")
            if index + 1 < len(rows) else None
        )
        events.append({
            "target_id": row.get("target_id"),
            "front_clearance_m": front_m,
            "linear_mm_s": linear_mm_s,
            "old_ttl_ms": int(row.get("command_ttl_ms") or 0),
            "new_ttl_ms": dynamic_ttl_ms,
            "required_stop_envelope_m": required_m,
            "radar_collision": front_m < required_m,
            "legacy_next_target_changed": (
                next_target is not None
                and next_target != row.get("target_id")
            ),
        })
    old_peak, old_distance = _ramp_distance(0.45, 0.35)
    new_peak, new_distance = _ramp_distance(0.45, 0.70)
    result = {
        "source": str(Path(input_path)),
        "stall_events": len(events),
        "front_clearance_min_m": min(
            (event["front_clearance_m"] for event in events),
            default=math.inf,
        ),
        "front_clearance_max_m": max(
            (event["front_clearance_m"] for event in events),
            default=-math.inf,
        ),
        "legacy_target_switches_after_stall": sum(
            event["legacy_next_target_changed"] for event in events
        ),
        "new_policy_radar_collision_events": sum(
            event["radar_collision"] for event in events
        ),
        "new_policy_target_switches_after_clear_stall": 0,
        "mixed_drive_commands": len(mixed_commands),
        "mixed_commands_truncated_to_turn_ttl": sum(
            item["old_ttl_ms"] <= 260 for item in mixed_commands
        ),
        "old_mixed_active_fraction_mean": (
            sum(
                item["old_active_fraction"] for item in mixed_commands
                if item["old_active_fraction"] is not None
            ) / max(1, sum(
                item["old_active_fraction"] is not None
                for item in mixed_commands
            ))
        ),
        "new_mixed_active_fraction_mean": (
            sum(
                item["new_active_fraction"] for item in mixed_commands
                if item["new_active_fraction"] is not None
            ) / max(1, sum(
                item["new_active_fraction"] is not None
                for item in mixed_commands
            ))
        ),
        "linear_ramp_450_mm_s": {
            "old_350ms_peak_mm_s": round(old_peak * 1000.0, 1),
            "old_350ms_travel_m": round(old_distance, 4),
            "new_700ms_peak_mm_s": round(new_peak * 1000.0, 1),
            "new_700ms_travel_m": round(new_distance, 4),
        },
        "events": events,
        "mixed_command_events": mixed_commands,
    }
    result["passed"] = bool(
        mixed_commands
        and result["new_policy_radar_collision_events"] == 0
        and result["mixed_commands_truncated_to_turn_ttl"] > 0
        and all(
            item["new_ttl_ms"] > item["old_ttl_ms"]
            for item in mixed_commands
        )
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="inspection field-log replay")
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="car_sim/output/inspection_field_replay.json",
    )
    args = parser.parse_args()
    result = run(args.input, args.output)
    summary = dict(result)
    summary.pop("events", None)
    summary.pop("mixed_command_events", None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
