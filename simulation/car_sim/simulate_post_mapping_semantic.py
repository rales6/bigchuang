"""Compatibility entry point for the post-mapping semantic system test.

Historically the test referenced this module, but the implementation was
missing.  The adapter now runs the same real-PGM disturbed closed loop used by
``semantic_inspection_stress`` and exposes the smaller legacy report schema.
"""

from __future__ import annotations

import json
from pathlib import Path

from raspberry_pi.known_map_navigation import KnownOccupancyMap

from .semantic_inspection_stress import StressProfile, simulate_seed


def run_simulation(output_directory: str | Path) -> dict:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    result = simulate_seed(
        0,
        StressProfile.named("realistic"),
        artifact_directory=output,
    )
    # Boundary differences are deliberately sparse relative to the saved
    # occupied endpoints.  This is the map-vs-scene recall exercised by this
    # test, not a comparison of the map with itself.
    known_map = KnownOccupancyMap.load(
        Path(__file__).resolve().parents[2] / "maps/current/ble_scene_17.pgm",
        Path(__file__).resolve().parents[2] / "maps/current/ble_scene_17.yaml", clearance_m=0.30
    )
    saved_occupied_cells = int(known_map.occupied.sum())
    endpoint_recall = max(
        0.0,
        1.0 - result["actual_map_changed_cells"] / float(saved_occupied_cells),
    )
    report = {
        "passed": bool(
            result["passed"]
            and endpoint_recall >= 0.90
            and result["semantic_component_count"]
            == result["occupied_component_count"]
            and result["localization"]["position_max_m"] <= 0.30
        ),
        "map": {
            "endpoint_recall": endpoint_recall,
            "actual_changed_cells": result["actual_map_changed_cells"],
        },
        "semantic": {
            "located": result["semantic_component_count"],
            "expected": result["occupied_component_count"],
            "maximum_error_m": result["localization"]["position_max_m"],
        },
        "mission": {
            "targets": result["target_count"],
            "captures": result["capture_count"],
        },
        "artifacts": result.get("artifacts", {}),
    }
    (output / "post_mapping_semantic_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(
        run_simulation(Path(__file__).resolve().parents[2] / "runtime/simulation/post_mapping_semantic"),
        ensure_ascii=False,
        indent=2,
    ))
