#!/usr/bin/env python3
"""Replay saved stationary images/candidates through the DeepSeek ranker.

No camera, lidar, ESP32, chassis, or arm device is opened by this tool.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

from raspberry_pi.mapping import Pose2D

from deepseek_visual_localization import (
    DeepSeekVisionClient,
    candidate_scene_descriptions,
    load_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=Path(
        os.environ.get(
            "DEEPSEEK_VISION_CONFIG",
            "/home/dachuang/dachuang/car1.0/secrets/deepseek_vision.json",
        )
    ))
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    release = Path(__file__).resolve().parent
    config = load_config(args.config, module_directory=release)
    if not config.configured:
        raise RuntimeError(f"DeepSeek configuration is not ready: {config.source_path}")
    camera = source["camera"]
    left, right = Path(camera["left"]), Path(camera["right"])
    candidates = []
    for row in source.get("radar_candidates") or []:
        candidates.append(SimpleNamespace(
            pose=Pose2D(
                float(row["x_m"]),
                float(row["y_m"]),
                math.radians(float(row["yaw_deg"])),
            ),
            reliable=bool(row.get("reliable")),
            rmse_m=float(row.get("rmse_m") or math.inf),
            inlier_ratio=float(row.get("inlier_ratio") or 0.0),
            score=float(row.get("score") or 0.0),
        ))
    semantic = json.loads(config.semantic_map.read_text(encoding="utf-8"))
    descriptions = candidate_scene_descriptions(candidates, semantic, config)
    result = DeepSeekVisionClient(config).rank(left, right, descriptions)
    reliable = bool(
        result.selected_candidate_id != "uncertain"
        and result.confidence >= config.minimum_confidence
        and result.margin >= config.minimum_margin
        and result.consistent_repeats == config.repeat_count
    )
    output = args.output or args.input.with_name("deepseek_visibility_replay.json")
    payload = {
        "schema": "stationary_deepseek_replay.v1",
        "source": str(args.input),
        "hardware_opened": False,
        "hardware_commands_sent": False,
        "model": config.model,
        "api_key_logged": False,
        "candidate_descriptions": descriptions,
        "result": result.to_dict(),
        "reliable": reliable,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "result": result.to_dict(),
        "reliable": reliable,
        "hardware_commands_sent": False,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
