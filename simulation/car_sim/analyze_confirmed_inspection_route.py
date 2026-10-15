"""Rebuild the current inspection target order from a human-confirmed pose."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from raspberry_pi.known_map_navigation import KnownOccupancyMap, Pose2D, order_targets
from raspberry_pi.map_camera_inspection import _information_view_targets, _spaced_targets
from raspberry_pi.joint_mapping import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--map", default=None)
    parser.add_argument("--semantic", default=None)
    parser.add_argument("--pose", nargs=3, type=float, default=(-.815, .565, -126.))
    args = parser.parse_args()
    output = Path(args.directory)
    map_path = Path(args.map) if args.map else output / "ble_scene_17.pgm"
    known = KnownOccupancyMap.load(map_path, clearance_m=.30)
    start = Pose2D(args.pose[0], args.pose[1], math.radians(args.pose[2]))
    targets = known.inspection_targets(standoff_m=.65, maximum_targets=24,
                                       views_per_wall=3, views_per_region=1,
                                       preferred_view_xy=(start.x_m, start.y_m))
    targets = _spaced_targets(targets, .40)
    semantic_path = Path(args.semantic) if args.semantic else output / "ble_scene_17_semantics_qwen.json"
    if not semantic_path.exists():
        semantic_path = Path("maps") / "ble_scene_17_semantics_qwen.json"
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    confirmed_components = {
        target.target_id.rsplit("_", 1)[0] for target in targets
        if target.kind == "interior_black_region"
        and (semantic.get("entities", {}).get(target.target_id, {}).get("online_qwen") or {}).get("promoted")
    }
    targets = [target for target in targets if not (
        target.kind == "interior_black_region"
        and target.target_id.rsplit("_", 1)[0] in confirmed_components)]
    targets, wide = _information_view_targets(known, targets, 2.20)
    ordered = order_targets(
        known, (start.x_m, start.y_m), targets,
        start_yaw_rad=start.yaw_rad, turn_cost_m_per_rad=.35,
    )
    rows = []
    current = (start.x_m, start.y_m)
    for index, (target, path) in enumerate(ordered, 1):
        distance = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
        rows.append({"order": index, "target_id": target.target_id, "kind": target.kind,
                     "from": list(current), "view": list(target.view_xy_m),
                     "subject": list(target.target_xy_m), "view_yaw_deg": math.degrees(target.yaw_rad),
                     "path_length_m": distance, "path": [list(p) for p in path],
                     "wide_view": target.target_id in wide})
        current = target.view_xy_m
    payload = {"map_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
               "start_pose": {"x_m": start.x_m, "y_m": start.y_m, "yaw_deg": args.pose[2],
                              "confirmed_by_user": True},
               "selection_policy": "A* path length plus 0.35m/rad final camera-heading turn cost",
               "skipped_preclassified_components": sorted(confirmed_components), "ordered": rows}
    atomic_json(output / "confirmed_start_and_route.json", payload)

    image = Image.fromarray(__import__("numpy").flipud(known.image)).convert("RGB").resize((known.width*2, known.height*2))
    draw = ImageDraw.Draw(image)
    def screen(xy):
        col, row = known.world_to_grid((xy,))[0]
        return int(col*2), int((known.height-1-row)*2)
    current = (start.x_m, start.y_m)
    for row in rows:
        points = [screen(current)] + [screen(p) for p in row["path"]]
        if len(points) > 1:
            draw.line(points, fill=(30, 100, 230), width=3)
        px, py = screen(row["view"])
        draw.ellipse((px-5, py-5, px+5, py+5), fill=(20, 80, 255))
        draw.text((px+5, py+5), str(row["order"]), fill=(220, 20, 20))
        current = tuple(row["view"])
    px, py = screen((start.x_m, start.y_m))
    draw.ellipse((px-7, py-7, px+7, py+7), fill=(0, 180, 0))
    draw.line((px, py, px+35*math.cos(start.yaw_rad), py-35*math.sin(start.yaw_rad)), fill=(0, 180, 0), width=4)
    image.save(output / "confirmed_route.png")
    print(json.dumps({"targets": len(rows), "first_five": [{k: r[k] for k in ("order", "target_id", "view", "path_length_m", "view_yaw_deg")} for r in rows[:5]]}, indent=2))


if __name__ == "__main__":
    main()
