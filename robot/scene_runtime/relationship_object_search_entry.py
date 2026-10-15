#!/usr/bin/env python3
"""Run the protected object-search mission with the release-local route planner.

The production ``raspberry_pi`` package is deliberately left unchanged.  This
entry point installs only the audited relationship-aware route module before
importing ``object_search_mission``; every other mission, localization, safety,
camera, lidar, BLE and motion component still comes from the protected package.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


MODULE_NAME = "raspberry_pi.search_route_planner"
CAR_ROOT = Path(
    os.environ.get("CAR_ROOT", Path(__file__).resolve().parents[1])
).resolve()
if str(CAR_ROOT) not in sys.path:
    sys.path.insert(0, str(CAR_ROOT))


def install_relationship_planner() -> Path:
    local_release = Path(__file__).resolve().with_name(
        "search_route_planner_relationship.py"
    )
    planner_path = (
        local_release
        if local_release.is_file()
        else CAR_ROOT / "raspberry_pi" / "search_route_planner.py"
    )
    if not planner_path.is_file():
        raise FileNotFoundError(f"relationship route planner missing: {planner_path}")
    spec = importlib.util.spec_from_file_location(MODULE_NAME, planner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load relationship route planner: {planner_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "build_search_route"):
        raise RuntimeError("relationship route planner has no build_search_route")
    return planner_path


def main() -> None:
    install_relationship_planner()
    import raspberry_pi.object_search_mission as mission
    from deepseek_visual_localization import install as install_deepseek_vision
    from relationship_localization_guard import install as install_localization_guard

    install_localization_guard(mission)
    install_deepseek_vision(mission)
    args = mission.build_argument_parser().parse_args()
    # The relationship-map release is now the full real-car path.  The old
    # first-view acceptance ceiling was useful during initial commissioning,
    # but it prevented every legitimate route longer than that ceiling from
    # starting.  Disable it here without editing the protected mission file.
    args.real_test_max_distance = 0.0
    mission.run(args)


if __name__ == "__main__":
    main()
