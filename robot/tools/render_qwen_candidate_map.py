"""Render the actual occupancy map and fresh Qwen candidate positions to PNG.

This is a spatial audit, not verified object coordinates or a navigation map.
No inference, remote connection, original map writes, or robot controls.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raspberry_pi.render_semantic_map import _known_crop, _load_map_config
from tools.render_inspection_semantic_overview import intersects, place_labels


STYLES = {
    'table': ('桌子', '#986522'), 'chair': ('椅子', '#3f8545'),
    'cabinet': ('柜子', '#8251b8'), 'door': ('门', '#256fb2'),
    'window': ('窗户', '#3885a9'), 'glass_wall': ('玻璃墙', '#078c9d'),
    'outlet': ('插座', '#c35c3d'), 'switch': ('开关', '#a36c2e'),
    'sign': ('标牌', '#6652a9'), 'light': ('灯具', '#ab5c85'),
    'broom': ('扫帚', '#967441'), 'dustpan': ('簸箕', '#a47b30'),
    'stairs': ('楼梯', '#657a87'), 'unknown_object': ('类别待确认', '#7a7a7a'),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_candidates(memory):
    groups, missing, invalid = defaultdict(list), [], []
    for item in memory.get('objects', []):
        if item.get('semantic_ambiguous') is True or item.get('map_xy') is None:
            missing.append(item)
            continue
        xy = item['map_xy']
        if (not isinstance(xy, (tuple, list)) or len(xy) != 2
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in xy)):
            invalid.append(item['id'])
            continue
        groups[item.get('category', 'unknown_object')].append(item)
    return dict(groups), missing, invalid


def representative(items):
    # Choose an existing hypothesis, never average different physical objects.
    return max(items, key=lambda item: (len(item.get('station_ids', [])),
                item.get('evidence_count', 0), item.get('confidence', 0), item['id']))


def render(map_path, yaml_path, memory_path, output, font_path, observation_map_path=None):
    sources = [map_path, yaml_path, memory_path]
    if observation_map_path is not None:
        sources.append(observation_map_path)
    for source in sources:
        if source.resolve() in (output.resolve(), output.with_suffix('.json').resolve()):
            raise ValueError('Output must not replace source inputs')
    before = {str(path): digest(path) for path in sources}
    memory = json.loads(memory_path.read_text(encoding='utf-8'))
    if memory.get('schema') != 'semantic_object_memory.v1':
        raise ValueError('Expected semantic_object_memory.v1')
    if memory.get('map_sha256') != digest(map_path) or memory.get('map_yaml_sha256') != digest(yaml_path):
        raise ValueError('Semantic candidates refer to a different original map')
    groups, missing, invalid = select_candidates(memory)
    observation = None
    if observation_map_path is not None:
        observation = json.loads(observation_map_path.read_text(encoding='utf-8'))
        if str(observation.get('inspection_run')) != '20260917_190706':
            raise ValueError('Observation map is not the successful 20260917_190706 run')
    categories = [category for category in STYLES if category in groups]
    categories += sorted(set(groups) - set(categories))
    config = _load_map_config(yaml_path)
    if abs(float(config['origin'][2])) > 1e-8:
        raise ValueError('Rotated map YAML is not supported')
    with Image.open(map_path) as source:
        gray = source.convert('L')
    crop = _known_crop(gray, padding=10)
    scale, left, top = 9, 105, 250
    base = gray.crop(crop).convert('RGB').resize(
        ((crop[2] - crop[0]) * scale, (crop[3] - crop[1]) * scale), Image.Resampling.NEAREST)
    map_width, map_height = base.size
    legend_left = left + map_width + 125
    width = legend_left + 650
    height = max(top + map_height + 265, top + 260 + len(categories) * 75 + 265)
    image = Image.new('RGB', (width, height), '#f5f7fa')
    image.paste(base, (left, top))
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(str(font_path), 49)
    font = ImageFont.truetype(str(font_path), 30)
    small = ImageFont.truetype(str(font_path), 26)
    tag_font = ImageFont.truetype(str(font_path), 28)
    text_boxes = []

    def text(xy, value, face=font, color='#23384b'):
        box = draw.textbbox(xy, value, font=face)
        if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
            raise ValueError('Clipped text: ' + value)
        if any(intersects(box, other, margin=3) for other in text_boxes):
            raise ValueError('Overlapping text: ' + value)
        text_boxes.append(box)
        draw.text(xy, value, font=face, fill=color)

    def pixel(xy):
        x, y = xy
        # The ROS image pixel-center convention agrees with the prior renderer.
        return (left + ((x - config['origin'][0]) / config['resolution'] - crop[0]) * scale,
                top + (gray.height - 1 - (y - config['origin'][1]) / config['resolution'] - crop[1]) * scale)

    located_count = sum(len(items) for items in groups.values())
    verified_count = sum(item.get('position_verified') is True and item.get('localization_allowed') is True
                         for items in groups.values() for item in items)
    title = ('ble_scene_17｜成功巡检站位 · Qwen 语义候选'
             if observation is not None else 'ble_scene_17｜真实雷达底图 · Qwen 语义候选')
    text((45, 25), title, title_font)
    subtitle = ('2026-09-17 成功巡检 + 2026-09-18 重标注｜33 组照片｜底图哈希一致'
                if observation is not None
                else '2026-09-18 重标注｜33 组照片｜底图与原始地图完全一致')
    text((45, 100), subtitle, font)
    text((45, 150), '空心点是估算搜索区域，不是精确物体坐标；同一物体可能出现多个候选。', small, '#805a27')
    text((45, 191), '每类只标注一个观测较强的候选位置；其余同色小点保留，便于检查偏差。', small, '#526272')

    all_anchors, representative_anchors, labels, reps = [], [], [], []
    station_points = []
    for station in (observation or {}).get('stations', []):
        try:
            anchor = pixel((float(station['x_m']), float(station['y_m'])))
        except (KeyError, TypeError, ValueError):
            continue
        if left <= anchor[0] <= left + map_width and top <= anchor[1] <= top + map_height:
            station_points.append((int(station['station']), anchor))
            all_anchors.append(anchor)
    displayed = []
    omitted_outside = []
    for category in categories:
        name, color = STYLES.get(category, (category, '#7a7a7a'))
        items = groups[category]
        for item in items:
            px, py = pixel(item['map_xy'])
            if not (left <= px <= left + map_width and top <= py <= top + map_height):
                omitted_outside.append(item['id'])
                continue
            all_anchors.append((px, py))
            radius = min(9, 4.5 + math.sqrt(max(0, item.get('evidence_count', 1))))
            # Every current point is estimated. Future verified points are solid.
            verified = item.get('position_verified') is True and item.get('localization_allowed') is True
            draw.ellipse((px - radius, py - radius, px + radius, py + radius),
                         fill=color if verified else '#ffffff', outline=color, width=3)
            displayed.append({'id': item['id'], 'category': category, 'map_xy': item['map_xy'],
                              'position_verified': verified, 'marker_xy': [px, py]})
        rep = representative(items)
        representative_anchors.append(pixel(rep['map_xy']))
        reps.append(rep)
        labels.append((name, color))
    sizes = [(draw.textlength(name + '（估算）', font=tag_font) + 28, 54) for name, _ in labels]
    boxes = place_labels(representative_anchors + all_anchors, sizes,
                         (25, top, legend_left - 45, top + map_height))
    for anchor, box, (_, color) in zip(representative_anchors, boxes, labels):
        ax, ay = anchor
        # Connect to the nearest label edge, not through the text.
        bx = min(max(ax, box[0]), box[2])
        by = min(max(ay, box[1]), box[3])
        draw.line((ax, ay, bx, by), fill=color, width=3)
    for box, (name, color) in zip(boxes, labels):
        draw.rounded_rectangle(box, radius=7, fill='white', outline=color, width=2)
        text((box[0] + 14, box[1] + 7), name + '（估算）', tag_font, color)

    for station_id, (px, py) in station_points:
        radius = 17
        draw.ellipse((px - radius, py - radius, px + radius, py + radius),
                     fill='#edf6ff', outline='#1768ac', width=3)
        label = f'S{station_id}'
        bounds = draw.textbbox((0, 0), label, font=small)
        draw.text((px - (bounds[2] - bounds[0]) / 2,
                   py - (bounds[3] - bounds[1]) / 2 - 2),
                  label, font=small, fill='#1768ac')

    text((legend_left, top), '图例与待核实项目', font)
    text((legend_left, top + 57), '黑：雷达占据 / 白：自由 / 灰：未知', small)
    text((legend_left, top + 104), f'有坐标候选 {located_count} · 未定位 {len(missing)}', small)
    text((legend_left, top + 149), f'可用于强定位的已验证地标：{verified_count}', small, '#805a27')
    if observation is not None:
        text((legend_left, top + 190), f'蓝色 S：成功巡检拍摄站位 {len(station_points)} 个', small, '#1768ac')
    station_offset = 45 if observation is not None else 0
    text((legend_left, top + 208 + station_offset), '下列数量是候选区数，不是物体数量：', small, '#526272')
    for index, category in enumerate(categories):
        name, color = STYLES.get(category, (category, '#7a7a7a'))
        y = top + 270 + station_offset + index * 75
        draw.ellipse((legend_left, y + 8, legend_left + 20, y + 28), fill='white', outline=color, width=3)
        text((legend_left + 36, y), f'{name}：{len(groups[category])} 个候选区', font, color)
    legend_bottom = top + 270 + station_offset + len(categories) * 75
    text((legend_left, legend_bottom + 15), '未定位/类别冲突的候选不上图。', small, '#805a27')
    text((legend_left, legend_bottom + 61), '插座与开关同框时仍需现场确认。', small, '#805a27')
    # Scale bar and Cartesian direction, not an invented geographical north.
    bar_y = top + map_height + 50
    bar_length = scale / config['resolution']
    draw.line((left + 20, bar_y, left + 20 + bar_length, bar_y), fill='#263746', width=5)
    draw.line((left + 20, bar_y - 8, left + 20, bar_y + 8), fill='#263746', width=3)
    draw.line((left + 20 + bar_length, bar_y - 8, left + 20 + bar_length, bar_y + 8), fill='#263746', width=3)
    text((left + 30 + bar_length, bar_y - 20), '1 米', small)
    text((left + 370, bar_y - 20), '图上方为地图 +Y，右方为 +X', small)
    text((45, height - 117), '这是实际雷达栅格上的语义审核图：位置仍受旧照片朝向、相机标定与深度缺失影响。', small)
    text((45, height - 72), '类别和坐标核实前不用于自动导航/强定位；原占据地图、语义文件和备份均未改写。', small, '#805a27')
    after = {str(path): digest(path) for path in sources}
    if after != before:
        raise RuntimeError('Source files changed during rendering')
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, dpi=(180, 180))
    report = {'scope': 'actual_occupancy_grid_with_unverified_semantic_hypotheses',
        'source_map_sha256': memory['map_sha256'], 'source_inputs_sha256': before,
        'image_size': list(image.size), 'displayed_candidate_count': len(displayed),
        'unlocalized_candidate_count': len(missing), 'invalid_coordinates': invalid,
        'outside_known_crop': omitted_outside, 'verified_localization_landmarks': verified_count,
        'semantic_label_count': len(boxes), 'observation_station_count': len(station_points),
        'label_overlap_count': 0, 'text_overlap_count': 0,
        'label_boxes': boxes, 'text_boxes': text_boxes, 'displayed_candidates': displayed,
        'label_representatives': [item['id'] for item in reps],
        'counts_are_not_real_object_counts': True, 'navigation_map': False}
    output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output.resolve()), 'size': list(image.size),
        'candidate_markers': len(displayed), 'category_labels': len(boxes),
        'text_overlap_count': 0}, ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--map-yaml', type=Path, required=True)
    parser.add_argument('--memory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font', type=Path, default=Path('C:/Windows/Fonts/msyh.ttc'))
    parser.add_argument('--observation-map', type=Path, default=None,
                        help='Optional successful 20260917_190706 observation-map JSON')
    args = parser.parse_args()
    render(args.map, args.map_yaml, args.memory, args.output, args.font, args.observation_map)
