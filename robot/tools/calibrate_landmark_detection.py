"""Offline smoke test for labelled-landmark localization using saved photos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from raspberry_pi.object_search_mission import GpuMissionClient
from raspberry_pi.known_map_navigation import KnownOccupancyMap
from raspberry_pi.visual_place_localization import (
    VisualPlaceRecognizer,
    detection_landmark_labels,
    rank_landmark_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", default="inspections/ble_scene_17")
    parser.add_argument("--semantic-map", default="maps/ble_scene_17_semantics_qwen.json")
    parser.add_argument("--endpoint", default="http://192.0.2.20:8011")
    parser.add_argument("--targets", default="object_2_1,wall_6_3,wall_6_4,wall_1_3")
    args = parser.parse_args()
    manifests = sorted(Path(args.reference_root).rglob("inspection_manifest.json"), key=lambda path: path.stat().st_mtime_ns)
    query_manifest = manifests[-2]
    payload = json.loads(query_manifest.read_text(encoding="utf-8"))
    requested = {value.strip() for value in args.targets.split(",") if value.strip()}
    recognizer = VisualPlaceRecognizer(
        args.reference_root,
        semantic_map=args.semantic_map,
    )
    known_map = KnownOccupancyMap.load(
        "maps/ble_scene_17.pgm", "maps/ble_scene_17.yaml", clearance_m=0.30
    )
    gpu = GpuMissionClient(args.endpoint, 120.0)
    for record in payload.get("captures") or []:
        target_id = str(record.get("target_id") or "")
        if target_id not in requested:
            continue
        left = query_manifest.parent / str(record.get("image"))
        right_name = record.get("camera_2_image") or record.get("image")
        right = query_manifest.parent / str(right_name)
        response = gpu.perceive(left, right, {
            "observation_id": "landmark-calibration-" + target_id,
            "purpose": "permanent_landmark_localization",
            "search_phase": "localization",
            "target": {},
        })
        labels = detection_landmark_labels(response.get("detections") or [])
        ranked = rank_landmark_candidates(
            recognizer.places,
            recognizer.landmarks,
            labels,
            known_map=known_map,
        )
        print(json.dumps({
            "query": target_id,
            "detected_labels": sorted(labels),
            "detections": [
                {
                    "class_name": item.get("class_name"),
                    "confidence": item.get("confidence"),
                    "description": (item.get("attributes") or {}).get("description"),
                }
                for item in (response.get("detections") or [])
            ],
            "top_candidates": [
                {
                    "target_id": recognizer.places[item.candidate_index].target_id,
                    "score": item.score,
                    "matched": item.matched_labels,
                    "expected": item.expected_labels,
                }
                for item in ranked[:4]
            ],
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
