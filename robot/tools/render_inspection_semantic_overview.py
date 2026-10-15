"""Render photo semantics without inventing object coordinates or overlapping text."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import statistics
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raspberry_pi.render_semantic_map import _known_crop, _load_map_config


TRANSLATIONS = {
    'floor': '地板', 'wall': '墙壁', 'door': '门', 'light': '灯具',
    'baseboard': '踢脚线', 'floor reflection': '反光', 'reflection': '反光',
    'debris': '地面碎屑', 'small object': '小物体', 'floor debris': '地面碎屑',
    'floor mark': '地面痕迹', 'floor texture': '地面纹理', 'corner': '墙角',
    'wall corner': '墙角', 'cabinet': '柜子', 'chair': '椅子', 'table': '桌子',
    'window': '窗户', 'stairs': '楼梯', 'glass wall': '玻璃墙',
    '墙': '墙壁', '地面': '地板', '玻璃边界': '玻璃',
    '柜子的底部': '柜子', '柜子底部': '柜子',
}
BACKGROUND = {'地板', '墙壁', '踢脚线', '墙角', '反光', '地面纹理'}


def normalize_name(value):
    text = str(value or '').strip()
    return TRANSLATIONS.get(text.lower(), TRANSLATIONS.get(text, text))


def wrap(draw, text, font, width):
    lines, line = [], ''
    for char in text:
        candidate = line + char
        if line and draw.textlength(candidate, font=font) > width:
            lines.append(line)
            line = char
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines or ['']


def intersects(a, b, margin=4):
    return not (a[2] + margin <= b[0] or b[2] + margin <= a[0]
                or a[3] + margin <= b[1] or b[3] + margin <= a[1])


def place_labels(anchors, sizes, bounds):
    """Deterministic placement with leaders; fail rather than overlap labels."""
    result = []
    for (x, y), (width, height) in zip(anchors, sizes):
        candidates = []
        for radius in range(24, int(max(bounds[2] - bounds[0], bounds[3] - bounds[1])), 16):
            for angle in range(0, 360, 30):
                cx = x + radius * math.cos(math.radians(angle))
                cy = y + radius * math.sin(math.radians(angle))
                box = (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
                if box[0] < bounds[0] or box[1] < bounds[1] or box[2] > bounds[2] or box[3] > bounds[3]:
                    continue
                if any(intersects(box, other) for other in result):
                    continue
                if any(box[0] - 9 <= ax <= box[2] + 9 and box[1] - 9 <= ay <= box[3] + 9 for ax, ay in anchors):
                    continue
                candidates.append((radius, box))
            if candidates:
                break
        if not candidates:
            raise ValueError('No collision-free label placement; increase rendering scale')
        result.append(min(candidates, key=lambda item: item[0])[1])
    return result


def aggregate(semantics):
    groups = defaultdict(list)
    for observation in semantics.get('observations', []):
        match = re.search(r'_s(\d+)_', observation.get('target_id', ''))
        if match:
            groups[int(match.group(1))].append(observation)
    cards = []
    for station, observations in sorted(groups.items()):
        counts, background = Counter(), Counter()
        for observation in observations:
            result = observation.get('result') or {}
            # Landmarks and objects can duplicate each other: count each name
            # at most once per photo pair, but do not drop objects-only results.
            visible = set()
            for item in list(result.get('landmarks') or []) + list(result.get('objects') or []):
                name = normalize_name(item.get('name') or item.get('type'))
                if name:
                    visible.add(name)
            for name in visible:
                (background if name in BACKGROUND else counts)[name] += 1
        cards.append({'station': station, 'photo_pairs': len(observations),
            'x_m': statistics.mean(float(o['x_m']) for o in observations),
            'y_m': statistics.mean(float(o['y_m']) for o in observations),
            'names': sorted(counts, key=lambda name: (-counts[name], name)),
            'background_names': sorted(background),
            'yaw_deg': [float(o['yaw_deg']) for o in observations],
            'empty_results': sum(not (o.get('result', {}).get('landmarks') or o.get('result', {}).get('objects')) for o in observations),
            'error_results': sum(bool(o.get('result', {}).get('error')) for o in observations),
            'names_are_photo_observations_not_object_coordinates': True})
    return cards


def render(snapshot, output, font_path):
    pi = snapshot / 'pi_snapshot'
    manifest = json.loads((snapshot / 'backup_manifest.json').read_text(encoding='utf-8'))
    run = manifest['inspection_run']
    directory = pi / 'inspections' / 'ble_scene_17' / run
    semantic_path = directory / 'panorama_semantics.json'
    semantics = json.loads(semantic_path.read_text(encoding='utf-8'))
    summary = json.loads((directory / 'panorama_summary.json').read_text(encoding='utf-8'))
    cards = aggregate(semantics)
    config = _load_map_config(pi / 'maps' / 'ble_scene_17.yaml')
    with Image.open(pi / 'maps' / 'ble_scene_17.pgm') as source:
        gray = source.convert('L')
    crop = _known_crop(gray, padding=12)
    scale = 7
    map_image = gray.crop(crop).convert('RGB').resize(
        ((crop[2] - crop[0]) * scale, (crop[3] - crop[1]) * scale), Image.Resampling.NEAREST)
    font = ImageFont.truetype(font_path, 25)
    small = ImageFont.truetype(font_path, 21)
    heading = ImageFont.truetype(font_path, 30)
    title = ImageFont.truetype(font_path, 38)
    temp = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    card_width = 860
    card_lines = []
    for card in cards:
        names = '、'.join(card['names']) or '没有有效物体名称'
        rows = wrap(temp, '可见候选：' + names, font, card_width - 40)
        rows += wrap(temp, '背景：' + '、'.join(card['background_names']), small, card_width - 40)
        if card['empty_results'] or card['error_results']:
            rows += wrap(temp, f"空结果 {card['empty_results']} 组；接口错误 {card['error_results']} 组", small, card_width - 40)
        card_lines.append(rows)
    card_heights = [70 + len(rows) * 34 for rows in card_lines]
    top, map_left = 180, 28
    map_top = top + 62
    sidebar_left = map_left + map_image.width + 38
    content_height = max(map_image.height + 92, sum(card_heights) + (len(cards) - 1) * 14)
    image = Image.new('RGB', (sidebar_left + card_width + 28, top + content_height + 130), '#f4f6f8')
    draw = ImageDraw.Draw(image)
    text_boxes = []

    def draw_text(xy, text, font, fill):
        box = draw.textbbox(xy, text, font=font)
        if box[0] < 0 or box[1] < 0 or box[2] > image.width or box[3] > image.height:
            raise ValueError('Text clipped: ' + text)
        if any(intersects(box, other, margin=0) for other in text_boxes):
            raise ValueError('Overlapping text: ' + text)
        text_boxes.append(box)
        draw.text(xy, text, font=font, fill=fill)

    draw_text((28, 22), 'ble_scene_17｜本轮语义观测地图', font=title, fill='#172b40')
    ratio = float(summary['coverage']['coverage_ratio'])
    draw_text((28, 82), f"巡检 {run}   ·   {len(semantics['observations'])} 组双摄照片   ·   视野估算覆盖 {ratio:.1%}", font=font, fill='#30465b')
    draw_text((28, 122), '蓝色编号为拍摄站位；右侧物体为该站照片的识别候选，不代表物体精确坐标。', font=font, fill='#79521a')
    draw_text((map_left, top + 10), '只读雷达地图 + 本轮拍摄站位', font=heading, fill='#172b40')
    draw_text((sidebar_left, top + 10), '各站可见内容（同名去重，保留模型候选）', font=heading, fill='#172b40')
    image.paste(map_image, (map_left, map_top))
    resolution = float(config['resolution'])
    origin_x, origin_y = config['origin'][:2]

    def pixel(x, y):
        return (map_left + ((x - origin_x) / resolution - crop[0]) * scale,
                map_top + (gray.height - 1 - (y - origin_y) / resolution - crop[1]) * scale)

    anchors = [pixel(card['x_m'], card['y_m']) for card in cards]
    labels = ['S' + str(card['station']) for card in cards]
    sizes = [(temp.textlength(text, font=heading) + 22, 45) for text in labels]
    boxes = place_labels(anchors, sizes, (map_left + 8, map_top + 8,
        map_left + map_image.width - 8, map_top + map_image.height - 8))
    uncovered = summary['coverage'].get('uncovered_samples', [])
    for x, y in uncovered:
        px, py = pixel(x, y)
        if map_left <= px < map_left + map_image.width and map_top <= py < map_top + map_image.height:
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill='#d87621')
    for (x, y), box in zip(anchors, boxes):
        draw.line((x, y, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2), fill='#276eb5', width=2)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill='#276eb5', outline='white', width=2)
    for text, box in zip(labels, boxes):
        draw.rounded_rectangle(box, radius=8, fill='white', outline='#276eb5', width=2)
        draw_text((box[0] + 11, box[1] + 2), text, font=heading, fill='#15558e')
    y = top + 62
    for card, rows, height in zip(cards, card_lines, card_heights):
        draw.rounded_rectangle((sidebar_left, y, sidebar_left + card_width, y + height), radius=10, fill='white', outline='#d8e1e9', width=1)
        draw_text((sidebar_left + 18, y + 10), f"S{card['station']}   ·   {card['photo_pairs']} 组照片   ·   站位约 ({card['x_m']:+.2f}, {card['y_m']:+.2f}) m", font=heading, fill='#15558e')
        for row_index, row in enumerate(rows):
            row_font = small if row.startswith(('背景：', '空结果')) else font
            draw_text((sidebar_left + 18, y + 55 + row_index * 34), row, font=row_font, fill='#273c4e')
        y += height + 14
    footer = image.height - 100
    draw_text((28, footer), '橙色点：尚未覆盖的边界采样。覆盖率是视野投影估算，不等于所有物体均已确认。', font=small, fill='#79521a')
    draw_text((28, footer + 34), '本轮因 actuator_no_response 停止；方向与站位来自运行日志，未独立校准。原始地图未被修改。', font=small, fill='#475d71')
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, dpi=(180, 180))
    report = {'image': str(output.resolve()), 'inspection_run': run,
        'size': list(image.size), 'station_labels': labels, 'label_boxes': boxes,
        'label_overlap_count': sum(intersects(a, b, margin=0) for index, a in enumerate(boxes) for b in boxes[index + 1:]),
        'text_overlap_count': 0, 'text_boxes': text_boxes,
        'semantic_scope': 'photo_observations_only', 'object_coordinates_inferred': False,
        'source': str(semantic_path), 'coverage': ratio,
        'motion_complete': summary.get('motion_complete'), 'stations': cards}
    output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['image', 'size', 'station_labels', 'label_overlap_count', 'coverage', 'motion_complete']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font', default='C:/Windows/Fonts/msyh.ttc')
    args = parser.parse_args()
    render(args.snapshot, args.output, args.font)
