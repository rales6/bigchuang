from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


LABEL_COLOURS = {
    "wall": (35, 105, 220),
    "wall_surface": (35, 105, 220),
    "glass_wall": (30, 170, 220),
    "door": (30, 160, 100),
    "window": (50, 180, 210),
    "cabinet": (150, 65, 210),
    "table": (170, 95, 35),
    "desk": (170, 95, 35),
    "chair": (35, 155, 70),
    "unknown": (235, 135, 25),
}


def _load_map_config(path: Path) -> dict[str, Any]:
    """Read the small ROS map YAML without requiring PyYAML."""
    values: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key, raw_value = key.strip(), raw_value.strip()
        if key == "origin":
            values[key] = json.loads(raw_value)
        elif key in {"resolution", "occupied_thresh", "free_thresh"}:
            values[key] = float(raw_value)
        elif key in {"negate"}:
            values[key] = int(raw_value)
        else:
            values[key] = raw_value.strip("'\"")
    return values


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render Qwen semantic labels on an occupancy map without trajectory."
    )
    parser.add_argument("--map", required=True, help="PGM occupancy map")
    parser.add_argument("--map-yaml", required=True, help="ROS map YAML")
    parser.add_argument("--semantic-map", required=True, help="Qwen semantic JSON")
    parser.add_argument("--output", required=True, help="Output PNG")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--font", default=None, help="Optional CJK TrueType font")
    parser.add_argument("--no-crop-known", action="store_true")
    return parser


def _load_font(font_path: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        font_path,
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _load_ascii_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _draw_segments(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    segments: list[tuple[str, ImageFont.ImageFont, tuple[int, int, int]]],
) -> None:
    x, y = xy
    for text, font, colour in segments:
        draw.text((x, y), text, font=font, fill=colour)
        box = draw.textbbox((x, y), text, font=font)
        x = box[2]


def _known_crop(gray: Image.Image, padding: int = 18) -> tuple[int, int, int, int]:
    # ROS maps normally use 205 for unknown; retain free and occupied cells.
    mask = gray.point(lambda value: 0 if 190 <= value <= 220 else 255)
    box = mask.getbbox()
    if box is None:
        return (0, 0, gray.width, gray.height)
    left, top, right, bottom = box
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(gray.width, right + padding),
        min(gray.height, bottom + padding),
    )


def _annotation_for_entity(entity: dict[str, Any]) -> dict[str, Any]:
    annotation = entity.get("qwen_annotation")
    if not isinstance(annotation, dict):
        annotation = {}
    needs_review = bool(annotation.get("needs_human_review", False))
    label = str(annotation.get("semantic_label") or entity.get("semantic_label") or "unknown")
    name = str(annotation.get("name_zh") or label)
    confidence = annotation.get(
        "confidence", entity.get("semantic_label_confidence", entity.get("qwen_candidate_confidence", 0.0))
    )
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "target_id": str(entity.get("target_id") or "unknown"),
        "x_m": float(entity["target_x_m"]),
        "y_m": float(entity["target_y_m"]),
        "label": label,
        "name": name,
        "confidence": confidence,
        "needs_review": needs_review,
    }


def _colour(label: str, needs_review: bool) -> tuple[int, int, int]:
    if needs_review:
        return LABEL_COLOURS["unknown"]
    return LABEL_COLOURS.get(label, (210, 55, 75))


def render_semantic_map(
    map_path: Path,
    yaml_path: Path,
    semantic_path: Path,
    output_path: Path,
    scale: int = 3,
    font_path: str | None = None,
    crop_known: bool = True,
) -> dict[str, Any]:
    if scale < 1:
        raise ValueError("scale must be at least 1")
    config = _load_map_config(yaml_path)
    resolution = float(config["resolution"])
    origin = config["origin"]
    origin_x, origin_y = float(origin[0]), float(origin[1])
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    entities = semantic.get("entities", {})
    if not isinstance(entities, dict):
        raise ValueError("semantic map entities must be an object")

    annotations = []
    for entity in entities.values():
        if not isinstance(entity, dict):
            continue
        if entity.get("target_x_m") is None or entity.get("target_y_m") is None:
            continue
        annotations.append(_annotation_for_entity(entity))
    annotations.sort(key=lambda item: (item["label"], item["target_id"]))

    with Image.open(map_path) as source:
        gray = source.convert("L")
    crop = _known_crop(gray) if crop_known else (0, 0, gray.width, gray.height)
    cropped = gray.crop(crop).convert("RGB")
    map_image = cropped.resize(
        (cropped.width * scale, cropped.height * scale),
        resample=Image.Resampling.NEAREST,
    )

    font = _load_font(font_path, max(16, 7 * scale))
    small_font = _load_font(font_path, max(13, 5 * scale))
    title_font = _load_font(font_path, max(20, 9 * scale))
    ascii_font = _load_ascii_font(max(16, 7 * scale))
    ascii_small_font = _load_ascii_font(max(13, 5 * scale))
    ascii_title_font = _load_ascii_font(max(20, 9 * scale))
    row_height = max(29, 11 * scale)
    legend_width = max(440, 165 * scale)
    top_padding = max(52, 19 * scale)
    output = Image.new(
        "RGB",
        (map_image.width + legend_width, max(map_image.height + top_padding, top_padding + row_height * (len(annotations) + 2))),
        (242, 242, 242),
    )
    output.paste(map_image, (0, top_padding))
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, output.width, top_padding), fill=(34, 38, 44))
    _draw_segments(
        draw,
        (16, 10),
        [
            ("ble_scene_17  Qwen ", ascii_title_font, (255, 255, 255)),
            ("语义标注地图（不含行驶轨迹）", title_font, (255, 255, 255)),
        ],
    )
    draw.line((map_image.width, top_padding, map_image.width, output.height), fill=(160, 160, 160), width=2)

    left, top, _right, _bottom = crop
    radius = max(8, 4 * scale)
    for number, item in enumerate(annotations, start=1):
        full_x = (item["x_m"] - origin_x) / resolution
        full_y = gray.height - 1 - (item["y_m"] - origin_y) / resolution
        x = int(round((full_x - left) * scale))
        y = int(round((full_y - top) * scale)) + top_padding
        colour = _colour(item["label"], item["needs_review"])
        if 0 <= x < map_image.width and top_padding <= y < top_padding + map_image.height:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour, outline="white", width=max(1, scale))
            number_text = str(number)
            number_box = draw.textbbox((0, 0), number_text, font=ascii_small_font)
            draw.text(
                (x - (number_box[2] - number_box[0]) / 2, y - (number_box[3] - number_box[1]) / 2 - 1),
                number_text,
                font=ascii_small_font,
                fill="white",
            )

    legend_x = map_image.width + max(16, 6 * scale)
    _draw_segments(
        draw,
        (legend_x, top_padding + 8),
        [
            ("编号  地图目标 → ", font, (25, 25, 25)),
            ("Qwen ", ascii_font, (25, 25, 25)),
            ("类别", font, (25, 25, 25)),
        ],
    )
    for row, (number, item) in enumerate(zip(range(1, len(annotations) + 1), annotations), start=1):
        y = top_padding + 10 + row * row_height
        colour = _colour(item["label"], item["needs_review"])
        draw.ellipse((legend_x, y + 3, legend_x + 18, y + 21), fill=colour)
        suffix = "  [待复核]" if item["needs_review"] else ""
        _draw_segments(
            draw,
            (legend_x + 27, y),
            [
                (f"{number:02d}  {item['target_id']} -> ", ascii_small_font, (25, 25, 25)),
                (item["name"], small_font, (25, 25, 25)),
                (f" ({item['confidence']:.0%})", ascii_small_font, (25, 25, 25)),
                (suffix, small_font, (25, 25, 25)),
            ],
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)
    return {
        "output": str(output_path.resolve()),
        "entity_count": len(annotations),
        "review_count": sum(item["needs_review"] for item in annotations),
        "trajectory_rendered": False,
        "crop": crop,
    }


def main() -> None:
    args = build_argument_parser().parse_args()
    result = render_semantic_map(
        Path(args.map),
        Path(args.map_yaml),
        Path(args.semantic_map),
        Path(args.output),
        scale=args.scale,
        font_path=args.font,
        crop_known=not args.no_crop_known,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
