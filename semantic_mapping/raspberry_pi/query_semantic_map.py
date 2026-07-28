from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semantic_mapping.common.semantic_map import SemanticMap


def main() -> int:
    parser = argparse.ArgumentParser(description="Query semantic landmarks by text.")
    parser.add_argument("--map", required=True, help="*_semantic.json path")
    parser.add_argument("--query", required=True, help="目标描述，例如 垃圾桶 / 门口 / 柜子")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    semantic_map = SemanticMap.load(args.map)
    matches = semantic_map.query(args.query, limit=args.limit)
    if not matches:
        print("没有找到匹配的语义位置。")
        return 1
    for item in matches:
        print(
            "{id} {name} type={type} xy=({x:.2f},{y:.2f}) stability={stability:.2f} seen={seen}".format(
                id=item.landmark_id,
                name=item.name,
                type=item.landmark_type,
                x=item.map_xy[0],
                y=item.map_xy[1],
                stability=item.stability,
                seen=item.seen_count,
            )
        )
        if item.visual_description:
            print("  visual:", item.visual_description)
        if item.lidar_description:
            print("  lidar:", item.lidar_description)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
