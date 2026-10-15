from __future__ import annotations

import argparse
import binascii
import json
import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _png_chunk(name: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + name + data + struct.pack(">I", binascii.crc32(name + data) & 0xFFFFFFFF)


def _save_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    stride = width * 3
    raw = b"".join(b"\x00" + bytes(pixels[y * stride : (y + 1) * stride]) for y in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(raw, 9))
    payload += _png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def _put(pixels: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < width and 0 <= y < height:
        offset = (y * width + x) * 3
        pixels[offset : offset + 3] = bytes(color)


def _rect(pixels: bytearray, width: int, height: int, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = box
    for y in range(max(0, y1), min(height, y2)):
        for x in range(max(0, x1), min(width, x2)):
            _put(pixels, width, height, x, y, color)


def _line(pixels: bytearray, width: int, height: int, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int], thickness: int = 1) -> None:
    x1, y1 = start
    x2, y2 = end
    dx, dy = abs(x2 - x1), -abs(y2 - y1)
    sx, sy = (1 if x1 < x2 else -1), (1 if y1 < y2 else -1)
    error = dx + dy
    while True:
        radius = max(0, thickness // 2)
        _rect(pixels, width, height, (x1 - radius, y1 - radius, x1 + radius + 1, y1 + radius + 1), color)
        if x1 == x2 and y1 == y2:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x1 += sx
        if twice <= dx:
            error += dx
            y1 += sy


def _circle(pixels: bytearray, width: int, height: int, center: tuple[int, int], radius: int, color: tuple[int, int, int]) -> None:
    cx, cy = center
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                _put(pixels, width, height, x, y, color)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render semantic cleanup demo map.")
    parser.add_argument("--scene", default="examples/demo_cleanup_scene.json")
    parser.add_argument("--output", default="outputs/demo_cleanup_scene.png")
    parser.add_argument("--trace", default="outputs/semantic_task_demo.json")
    args = parser.parse_args()
    scene_path = ROOT / args.scene if not Path(args.scene).is_absolute() else Path(args.scene)
    output_path = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    rows = scene["occupancy_grid"]["rows"]
    cell = 34
    margin = 24
    width, height = len(rows[0]) * cell + margin * 2, len(rows) * cell + margin * 2
    pixels = bytearray((248, 250, 252) * (width * height))
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            box = (margin + x * cell, margin + y * cell, margin + (x + 1) * cell, margin + (y + 1) * cell)
            _rect(pixels, width, height, box, (51, 65, 85) if value == "#" else (255, 255, 255))
            _line(pixels, width, height, (box[0], box[1]), (box[2], box[1]), (226, 232, 240))
            _line(pixels, width, height, (box[0], box[1]), (box[0], box[3]), (226, 232, 240))
    trace_path = ROOT / args.trace if not Path(args.trace).is_absolute() else Path(args.trace)
    if trace_path.exists():
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        points = trace.get("dry_run", {}).get("path_cells", [])
        if points:
            polyline = [(int(margin + (x + 0.5) * cell), int(margin + (y + 0.5) * cell)) for x, y in points]
            for first, second in zip(polyline, polyline[1:]):
                _line(pixels, width, height, first, second, (37, 99, 235), thickness=5)
    colors = {"furniture": (251, 146, 60), "container": (34, 197, 94), "landmark": (168, 85, 247)}
    for obj in scene.get("objects", []) + scene.get("simulated_detections", []):
        xy = obj.get("grid_xy")
        if not xy:
            continue
        x, y = xy
        cx, cy = int(margin + (x + 0.5) * cell), int(margin + (y + 0.5) * cell)
        color = (239, 68, 68) if "trash" in str(obj.get("category", "")) else colors.get(obj.get("category"), (15, 118, 110))
        _circle(pixels, width, height, (cx, cy), 10, color)
    start = scene["robot_pose"]["grid_xy"]
    cx, cy = int(margin + (start[0] + 0.5) * cell), int(margin + (start[1] + 0.5) * cell)
    _circle(pixels, width, height, (cx, cy), 12, (14, 165, 233))
    _line(pixels, width, height, (cx, cy), (cx, cy - 15), (8, 47, 73), thickness=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_png(output_path, width, height, pixels)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
