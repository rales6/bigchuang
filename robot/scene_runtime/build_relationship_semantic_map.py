"""Fuse reviewed room layout with one or more semantic annotation batches.

The reviewed layout owns geometry.  Annotation batches may add evidence,
aliases and confidence, but they never overwrite a reviewed map position.
This keeps multi-view room relationships stable while retaining every source
for later audit and calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raspberry_pi.render_semantic_map import _known_crop, _load_map_config
from tools.render_inspection_semantic_overview import intersects, place_labels, wrap


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = ROOT / "maps" / "ble_scene_17.pgm"
DEFAULT_YAML = ROOT / "maps" / "ble_scene_17.yaml"
DEFAULT_BASELINE = (
    ROOT / "semantic_maps" / "ble_scene_17_20260917_190706"
    / "complete_objects" / "supplemented_object_map.json"
)
DEFAULT_OBSERVATION = (
    ROOT / "semantic_maps" / "ble_scene_17_20260917_190706"
    / "semantic_observation_map.json"
)
DEFAULT_ANNOTATION = (
    ROOT / "semantic_maps" / "qwen_reannotation_20260918_142739"
    / "results" / "quality_filtered" / "semantic_object_memory.json"
)
DEFAULT_OUTPUT_JSON = ROOT / "maps" / "semantic_release_20260919" / "semantic_runtime.json"
DEFAULT_OUTPUT_IMAGE = ROOT / "maps" / "semantic_release_20260919" / "semantic_relationship_map_20260919.png"


CATEGORY_STYLES = {
    "table": ("桌子", "#986522"),
    "chair": ("椅子", "#3f8545"),
    "cabinet": ("柜子", "#8251b8"),
    "glass_wall": ("玻璃墙", "#078c9d"),
    "door": ("门", "#256fb2"),
    "window": ("窗户", "#3885a9"),
    "broom": ("清扫工具", "#967441"),
    "stairs": ("楼梯", "#657a87"),
    "outlet": ("插座", "#c35c3d"),
    "sign": ("标牌", "#4c8a42"),
    "light": ("灯具", "#ab5c85"),
    "marker": ("标记", "#168ca5"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slug(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def category_for(name: str) -> str:
    if "玻璃墙" in name:
        return "glass_wall"
    if "玻璃门" in name or name == "门":
        return "door"
    if "窗户" in name:
        return "window"
    if "标牌" in name:
        return "sign"
    if "桌" in name:
        return "table"
    if "椅" in name:
        return "chair"
    if "柜" in name and "标记" not in name:
        return "cabinet"
    if "楼梯" in name or "栏杆" in name:
        return "stairs"
    if "插座" in name:
        return "outlet"
    if "扫帚" in name or "簸箕" in name:
        return "broom"
    if "灯" in name:
        return "light"
    return "marker"


def aliases_for(item: dict[str, Any], category: str) -> list[str]:
    aliases = {str(item["name"]), CATEGORY_STYLES[category][0], category}
    if category == "door":
        aliases.update({"门", "玻璃门", "door"})
    elif category == "broom":
        aliases.update({"扫帚", "簸箕", "清扫工具", "broom", "dustpan"})
    elif category == "stairs":
        aliases.update({"楼梯", "栏杆", "stairs"})
    elif category == "sign":
        aliases.update({"标牌", "出口标牌", "sign"})
    return sorted(aliases)


def source_photos(item: dict[str, Any]) -> set[str]:
    return {
        str(view.get("photo"))
        for view in item.get("evidence", [])
        if isinstance(view, dict) and view.get("photo")
    }


def annotation_support(
    baseline: dict[str, Any], category: str, batches: list[tuple[Path, dict[str, Any]]]
) -> dict[str, Any]:
    photos = source_photos(baseline)
    matched_views: list[dict[str, Any]] = []
    category_candidates = 0
    for path, batch in batches:
        for candidate in batch.get("objects", []):
            if not isinstance(candidate, dict) or candidate.get("semantic_ambiguous") is True:
                continue
            if candidate.get("category") != category:
                continue
            category_candidates += 1
            for evidence in candidate.get("evidence", []):
                if not isinstance(evidence, dict) or evidence.get("semantic_ambiguous") is True:
                    continue
                for view in evidence.get("views", []):
                    image = str(view.get("image") or "")
                    if image not in photos:
                        continue
                    bbox = view.get("bbox_1000")
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        continue
                    matched_views.append({
                        "annotation": path.name,
                        "candidate_id": candidate.get("id"),
                        "observation_id": evidence.get("observation_id"),
                        "station_id": evidence.get("station_id"),
                        "image": image,
                        "bbox_1000": bbox,
                        "confidence": evidence.get("confidence"),
                    })
    unique = {
        (row["annotation"], row["observation_id"], row["image"]): row
        for row in matched_views
    }
    rows = list(unique.values())
    return {
        "direct_view_matches": len(rows),
        "station_ids": sorted({row["station_id"] for row in rows if row["station_id"] is not None}),
        "category_candidate_count": category_candidates,
        "matched_views": rows,
    }


def room_relations(objects: list[dict[str, Any]]) -> list[dict[str, str]]:
    by_name = {item["name"]: item["id"] for item in objects}

    specifications = [
        ("玻璃门", "connected_to", "玻璃墙"),
        ("绿色出口标牌（位置待核）", "near_and_above", "玻璃门"),
        ("楼梯 / 栏杆（玻璃外）", "behind", "玻璃墙"),
        ("蓝色储物柜（玻璃外候选）", "behind", "玻璃墙"),
        ("玻璃外悬挂灯具", "behind", "玻璃墙"),
        ("桌边椅子（右侧）", "around", "中央桌子"),
        ("桌边椅子（下侧）", "around", "中央桌子"),
        ("窗户（位置待确认）", "opposite_room_side_from", "玻璃墙"),
    ]
    return [
        {"subject_id": by_name[subject], "relation": kind, "object_id": by_name[target]}
        for subject, kind, target in specifications
        if subject in by_name and target in by_name
    ]


def build_memory(
    map_path: Path,
    yaml_path: Path,
    baseline_path: Path,
    annotation_paths: list[Path],
) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    batches = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in annotation_paths]
    for path, batch in batches:
        if batch.get("schema") != "semantic_object_memory.v1":
            raise ValueError(f"Unsupported annotation schema: {path}")
        if batch.get("map_sha256") != digest(map_path):
            raise ValueError(f"Annotation uses another occupancy map: {path}")
    objects = []
    for source in baseline.get("objects", []):
        name = str(source["name"])
        category = category_for(name)
        support = annotation_support(source, category, batches)
        behind_glass = source.get("mode") == "beyond_glass"
        objects.append({
            "id": "layout_" + slug(name),
            "name": name,
            "aliases": aliases_for(source, category),
            "category": category,
            "type": category,
            "label": category,
            "map_xy": [float(source["x_m"]), float(source["y_m"])],
            "approach_xy": None,
            "position_status": "reviewed_layout_prior",
            "position_method": "reviewed_20260917_multiview_relations_and_lidar_surface",
            "position_uncertainty_m": float(source.get("illustrative_radius_m") or source.get("uncertainty_m") or 1.2),
            "mobility": "fixed" if category in {"glass_wall", "door", "window", "cabinet", "stairs", "outlet", "sign", "light", "marker"} else "unknown",
            "confidence": 0.85 if support["direct_view_matches"] else 0.7,
            "behind_glass": behind_glass,
            "semantic_reviewed": True,
            "position_verified": False,
            "approach_verified": False,
            "navigation_allowed": False,
            "localization_allowed": False,
            "grasp_allowed": False,
            "search_region_allowed": not behind_glass,
            "requires_visual_recheck": True,
            "reviewed_source": {
                "file": str(baseline_path),
                "evidence": source.get("evidence", []),
                "estimates": source.get("estimates", []),
            },
            "annotation_support": support,
        })
    relations = room_relations(objects)
    def search_radius(item: dict[str, Any]) -> float:
        minimum = 0.95 if item["category"] in {"table", "cabinet"} else (
            0.75 if item["category"] == "chair" else 0.6
        )
        return max(minimum, min(1.5, item["position_uncertainty_m"]))

    search_regions = [
        {
            "id": item["id"],
            "query": item["name"],
            "map_xy": item["map_xy"],
            "radius_m": search_radius(item),
            "source": "reviewed_layout_prior",
            "requires_visual_recheck": True,
        }
        for item in objects if item["search_region_allowed"]
    ]
    return {
        "schema": "semantic_object_memory.v1",
        "map_id": "ble_scene_17",
        "map_sha256": digest(map_path),
        "map_yaml_sha256": digest(yaml_path),
        "fusion_method": "reviewed_layout_geometry_with_multi_batch_annotation_evidence",
        "geometry_policy": "reviewed_layout_positions_are_immutable_during_annotation_fusion",
        "objects": objects,
        "scene_relations": relations,
        "search_regions": search_regions,
        "annotation_sources": [
            {"path": str(path), "sha256": digest(path), "object_count": len(batch.get("objects", []))}
            for path, batch in batches
        ],
        "reviewed_layout_source": {"path": str(baseline_path), "sha256": digest(baseline_path)},
        "safety": {
            "direct_navigation_to_prior": False,
            "localization_correction_from_unverified_prior": False,
            "direct_grasp_from_prior": False,
            "live_visual_recheck_required": True,
        },
    }


def render_map(
    memory: dict[str, Any], map_path: Path, yaml_path: Path, observation_path: Path | None,
    output: Path, font_path: Path,
) -> dict[str, Any]:
    config = _load_map_config(yaml_path)
    with Image.open(map_path) as source:
        gray = source.convert("L")
    crop = _known_crop(gray, padding=10)
    scale, left, top = 9, 150, 245
    base = gray.crop(crop).convert("RGB").resize(
        ((crop[2] - crop[0]) * scale, (crop[3] - crop[1]) * scale), Image.Resampling.NEAREST
    )
    width, height = base.width + 2 * left, base.height + top + 245
    image = Image.new("RGB", (width, height), "#f5f7fa")
    image.paste(base, (left, top))
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(str(font_path), 42)
    font = ImageFont.truetype(str(font_path), 26)
    small = ImageFont.truetype(str(font_path), 22)
    text_boxes: list[tuple[float, float, float, float]] = []

    def text(xy, value, face=font, color="#23384b", check=True):
        box = draw.textbbox(xy, value, font=face)
        if check and any(intersects(box, other, margin=2) for other in text_boxes):
            raise ValueError("Overlapping text: " + value)
        if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
            raise ValueError("Clipped text: " + value)
        if check:
            text_boxes.append(box)
        draw.text(xy, value, font=face, fill=color)

    def pixel(xy):
        x, y = xy
        return (
            left + ((x - config["origin"][0]) / config["resolution"] - crop[0]) * scale,
            top + (gray.height - 1 - (y - config["origin"][1]) / config["resolution"] - crop[1]) * scale,
        )

    text((36, 24), "ble_scene_17｜多组标注关系融合语义地图", title_font, "#18334c")
    text((36, 87), "0917 复核布局固定空间骨架；后续标注只补充证据与类别，不覆盖门、标牌、玻璃墙和窗户位置。", small, "#364b5e")
    text((36, 126), "圆圈为搜索范围；实心点为复核布局中心。用于任务搜索先验，强定位、导航终点与夹取仍需现场视觉确认。", small, "#805a27")
    text((36, 165), "浅灰连线表示门—标牌—玻璃墙、桌椅及玻璃后物体等场景关系。", small, "#526272")

    objects = memory["objects"]
    by_id = {item["id"]: item for item in objects}
    anchors = {item["id"]: pixel(item["map_xy"]) for item in objects}
    for relation in memory["scene_relations"]:
        a, b = anchors[relation["subject_id"]], anchors[relation["object_id"]]
        draw.line((*a, *b), fill="#9aa9b5", width=3)

    overlay = Image.new("RGBA", image.size)
    tint = ImageDraw.Draw(overlay)
    for item in objects:
        x, y = anchors[item["id"]]
        color = CATEGORY_STYLES[item["category"]][1]
        rgb = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
        radius = item["position_uncertainty_m"] / config["resolution"] * scale
        tint.ellipse((x - radius, y - radius, x + radius, y + radius), fill=rgb + (18,), outline=rgb + (75,), width=2)
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)

    if observation_path is not None:
        observation = json.loads(observation_path.read_text(encoding="utf-8"))
        for station in observation.get("stations", []):
            if station.get("x_m") is None or station.get("y_m") is None:
                continue
            x, y = pixel((float(station["x_m"]), float(station["y_m"])))
            draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill="#edf6ff", outline="#1768ac", width=2)
            label = "S" + str(station.get("station"))
            bounds = draw.textbbox((0, 0), label, font=small)
            draw.text((x - (bounds[2] - bounds[0]) / 2, y - 16), label, font=small, fill="#1768ac")

    label_lines, sizes, points = [], [], []
    for item in objects:
        support = item["annotation_support"]["direct_view_matches"]
        suffix = f"〔新标注证据 {support}〕" if support else "〔0917复核〕"
        lines = wrap(draw, item["name"] + suffix, font, 430)
        label_lines.append(lines)
        sizes.append((max(draw.textlength(line, font=font) for line in lines) + 26, len(lines) * 36 + 14))
        points.append(anchors[item["id"]])
    boxes = place_labels(points + list(points), sizes, (18, top - 25, width - 18, top + base.height + 25))
    label_text_boxes = []
    for item, point, box, lines in zip(objects, points, boxes, label_lines):
        color = CATEGORY_STYLES[item["category"]][1]
        x, y = point
        draw.line((x, y, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2), fill=color, width=3)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline="white", width=2)
        draw.rounded_rectangle(box, radius=8, fill="white", outline=color, width=2)
        for index, line in enumerate(lines):
            xy = (box[0] + 13, box[1] + 4 + index * 36)
            draw.text(xy, line, font=font, fill=color)
            label_text_boxes.append(draw.textbbox(xy, line, font=font))

    bar_y = top + base.height + 52
    bar_length = scale / config["resolution"]
    draw.line((left + 15, bar_y, left + 15 + bar_length, bar_y), fill="#263746", width=5)
    draw.line((left + 15, bar_y - 7, left + 15, bar_y + 7), fill="#263746", width=3)
    draw.line((left + 15 + bar_length, bar_y - 7, left + 15 + bar_length, bar_y + 7), fill="#263746", width=3)
    text((left + 30 + bar_length, bar_y - 17), "1 米", small)
    text((left + 370, bar_y - 17), "图上方为地图 +Y，右方为 +X", small)
    text((36, height - 92), f"融合 {len(memory['annotation_sources'])} 组后续标注；{len(objects)} 个复核布局对象；{len(memory['scene_relations'])} 条场景关系。", small, "#364b5e")
    text((36, height - 54), "自动候选的错误投影不会改变布局中心；原始占据地图和各批标注文件保持不变。", small, "#805a27")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, dpi=(180, 180))
    report = {
        "scope": "reviewed_layout_with_multi_batch_annotation_support",
        "object_count": len(objects),
        "relation_count": len(memory["scene_relations"]),
        "annotation_batch_count": len(memory["annotation_sources"]),
        "label_overlap_count": 0,
        "text_overlap_count": 0,
        "image_size": list(image.size),
        "label_boxes": boxes,
        "source_map_sha256": memory["map_sha256"],
    }
    output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def write_memory(path: Path, memory: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(memory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--map-yaml", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--annotation", type=Path, action="append", default=None,
                        help="Repeat for every annotation-batch semantic memory")
    parser.add_argument("--observation-map", type=Path, default=DEFAULT_OBSERVATION)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-image", type=Path, default=DEFAULT_OUTPUT_IMAGE)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    args = parser.parse_args()
    annotations = args.annotation or [DEFAULT_ANNOTATION]
    memory = build_memory(args.map, args.map_yaml, args.baseline, annotations)
    write_memory(args.output_json, memory)
    report = render_map(memory, args.map, args.map_yaml, args.observation_map, args.output_image, args.font)
    print(json.dumps({
        "output_json": str(args.output_json.resolve()),
        "output_image": str(args.output_image.resolve()),
        **report,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
