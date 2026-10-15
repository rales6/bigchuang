#!/usr/bin/env python3
"""Render a stationary lidar + dual-camera pose diagnostic board."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from raspberry_pi.render_semantic_map import _known_crop, _load_map_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, default=-1.38)
    parser.add_argument("--y", type=float, default=-2.7367)
    parser.add_argument("--yaw-deg", type=float, default=96.3)
    parser.add_argument("--uncertainty-m", type=float, default=0.45)
    parser.add_argument("--camera-fov-deg", type=float, default=78.0)
    parser.add_argument("--camera-range-m", type=float, default=3.5)
    parser.add_argument(
        "--camera-dir",
        type=Path,
        default=ROOT / "diagnostics" / "current_camera_lidar_pose_20260919",
    )
    parser.add_argument("--rmse-note", default="5.4–5.8 cm")
    parser.add_argument("--inlier-note", default="82.3%–83.3%")
    parser.add_argument(
        "--visual-note",
        default="两路均看到玻璃墙结构",
    )
    parser.add_argument(
        "--perception-note",
        default="3090 摘要：室内地面，有玻璃墙和柜子",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "diagnostics" / "current_camera_lidar_pose_20260919"
        / "current_camera_lidar_pose_map.png",
    )
    args = parser.parse_args()

    release = ROOT / "maps" / "semantic_release_20260919"
    map_path = release / "ble_scene_17.pgm"
    yaml_path = release / "ble_scene_17.yaml"
    base_path = release / "semantic_relationship_map_20260919.png"
    memory = json.loads((release / "semantic_runtime.json").read_text(encoding="utf-8"))
    camera_dir = args.camera_dir
    camera_paths = [camera_dir / "current_left.jpg", camera_dir / "current_left_camera_2.jpg"]

    config = _load_map_config(yaml_path)
    with Image.open(map_path) as source:
        gray = source.convert("L")
    crop = _known_crop(gray, padding=10)
    scale, left, top = 9, 150, 245

    def pixel(xy: tuple[float, float]) -> tuple[float, float]:
        x, y = xy
        return (
            left + ((x - config["origin"][0]) / config["resolution"] - crop[0]) * scale,
            top + (gray.height - 1 - (y - config["origin"][1]) / config["resolution"] - crop[1]) * scale,
        )

    base = Image.open(base_path).convert("RGBA")
    panel_width = 720
    board = Image.new("RGBA", (base.width + panel_width, base.height), "#eef2f5")
    board.paste(base, (0, 0))
    overlay = Image.new("RGBA", board.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    title_font = ImageFont.truetype(str(font_path), 36)
    font = ImageFont.truetype(str(font_path), 25)
    small = ImageFont.truetype(str(font_path), 21)

    pose = (args.x, args.y)
    center = pixel(pose)
    yaw = math.radians(args.yaw_deg)
    half_fov = math.radians(args.camera_fov_deg / 2.0)
    sector_world = [pose]
    for index in range(25):
        angle = yaw - half_fov + 2.0 * half_fov * index / 24.0
        sector_world.append((
            args.x + args.camera_range_m * math.cos(angle),
            args.y + args.camera_range_m * math.sin(angle),
        ))
    sector_pixels = [pixel(point) for point in sector_world]
    draw.polygon(sector_pixels, fill=(255, 164, 35, 48), outline=(230, 123, 15, 190))

    radius = args.uncertainty_m / config["resolution"] * scale
    draw.ellipse(
        (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
        fill=(225, 42, 42, 26),
        outline=(195, 30, 30, 210),
        width=5,
    )
    draw.ellipse(
        (center[0] - 14, center[1] - 14, center[0] + 14, center[1] + 14),
        fill=(205, 24, 24, 255),
        outline="white",
        width=4,
    )
    arrow_world = (
        args.x + 0.72 * math.cos(yaw),
        args.y + 0.72 * math.sin(yaw),
    )
    arrow = pixel(arrow_world)
    draw.line((*center, *arrow), fill=(205, 24, 24, 255), width=12)
    wing = 0.18
    for delta in (2.55, -2.55):
        tip = (
            arrow_world[0] + wing * math.cos(yaw + delta),
            arrow_world[1] + wing * math.sin(yaw + delta),
        )
        draw.line((*arrow, *pixel(tip)), fill=(205, 24, 24, 255), width=9)

    glass = next(item for item in memory["objects"] if item.get("category") == "glass_wall")
    glass_xy = tuple(float(value) for value in glass["map_xy"])
    glass_pixel = pixel(glass_xy)
    draw.line((*center, *glass_pixel), fill=(0, 129, 150, 215), width=5)
    draw.ellipse(
        (glass_pixel[0] - 11, glass_pixel[1] - 11, glass_pixel[0] + 11, glass_pixel[1] + 11),
        fill=(0, 129, 150, 255),
        outline="white",
        width=3,
    )
    label_box = (center[0] + 22, center[1] + 22, center[0] + 440, center[1] + 115)
    draw.rounded_rectangle(label_box, radius=10, fill=(255, 255, 255, 238), outline=(205, 24, 24, 255), width=3)
    draw.text((label_box[0] + 13, label_box[1] + 8), "当前小车（静止双模态定位）", font=font, fill="#b01818")
    draw.text(
        (label_box[0] + 13, label_box[1] + 48),
        f"x={args.x:.2f}  y={args.y:.2f}  yaw={args.yaw_deg:.1f}°",
        font=small,
        fill="#b01818",
    )
    board = Image.alpha_composite(board, overlay)

    panel_x = base.width
    panel = ImageDraw.Draw(board)
    panel.rectangle((panel_x, 0, board.width - 1, board.height - 1), fill="#f7f9fb", outline="#9aa9b5", width=3)
    panel.text((panel_x + 32, 34), "静止双模态定位复核", font=title_font, fill="#18334c")
    panel.text((panel_x + 32, 93), "未连接 BLE｜未发送运动命令", font=font, fill="#a52a2a")

    image_y = 150
    for index, path in enumerate(camera_paths, start=1):
        camera = Image.open(path).convert("RGB")
        camera.thumbnail((656, 369))
        x = panel_x + (panel_width - camera.width) // 2
        board.paste(camera, (x, image_y))
        panel.rectangle((x, image_y, x + camera.width, image_y + camera.height), outline="#415a70", width=3)
        panel.text((panel_x + 32, image_y + camera.height + 8), f"摄像头 {index}：玻璃墙/栏杆在前方", font=small, fill="#24485f")
        image_y += camera.height + 62

    glass_bearing = math.degrees(math.atan2(glass_xy[1] - args.y, glass_xy[0] - args.x))
    heading_error = abs((glass_bearing - args.yaw_deg + 180.0) % 360.0 - 180.0)
    notes = [
        "雷达三帧：位置变化 ≤ 0.06 m",
        f"RMSE：{args.rmse_note}",
        f"内点率：{args.inlier_note}",
        f"摄像头：{args.visual_note}",
        args.perception_note,
        f"玻璃墙方位约 {glass_bearing:.1f}°",
        f"车头与玻璃墙方向差约 {heading_error:.1f}°",
        "结论：地图下方，车头朝玻璃墙",
        f"红圈：本次三帧定位显示范围（约 {args.uncertainty_m:.2f} m）",
        "橙色：摄像头水平视野示意（尚未精确标定）",
    ]
    note_y = image_y + 18
    for line in notes:
        panel.text((panel_x + 32, note_y), line, font=small, fill="#263746")
        note_y += 43

    args.output.parent.mkdir(parents=True, exist_ok=True)
    board.convert("RGB").save(args.output, quality=94, dpi=(180, 180))
    print(args.output)


if __name__ == "__main__":
    main()
