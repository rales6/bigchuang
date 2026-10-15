"""Illustrative object positions from reviewed photos, bearings and map surfaces.

Never edits occupancy/semantic memory or turns these estimates into navigation
goals. The assumed FOV and logged yaw are not calibrated metric measurements.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raspberry_pi.render_semantic_map import _known_crop, _load_map_config
from tools.render_inspection_semantic_overview import intersects, place_labels, wrap


def bearing_offset(center_u, fov_deg):
    return math.degrees(math.atan((1 - 2 * center_u) * math.tan(math.radians(fov_deg / 2))))


def ray_surfaces(gray, config, x, y, yaw_deg):
    resolution = float(config['resolution'])
    origin_x, origin_y = config['origin'][:2]
    angle = math.radians(yaw_deg)
    hits = []
    previous = None
    for distance in np.arange(.15, 9., resolution / 2):
        xx, yy = x + distance * math.cos(angle), y + distance * math.sin(angle)
        column = int(math.floor((xx - origin_x) / resolution))
        row = gray.shape[0] - 1 - int(math.floor((yy - origin_y) / resolution))
        if not (0 <= column < gray.shape[1] and 0 <= row < gray.shape[0]):
            break
        if (column, row) == previous:
            continue
        previous = column, row
        if gray[row, column] < 90:
            hits.append({'x_m': float(xx), 'y_m': float(yy), 'column': column,
                         'row': row, 'distance_m': float(distance)})
    if not hits:
        raise ValueError('No map surface intersects reviewed bearing')
    return hits


def occupied_components(gray, config):
    seen, components = set(), []
    for row, column in np.argwhere(gray < 90):
        start = (int(column), int(row))
        if start in seen:
            continue
        queue, cells = deque([start]), []
        seen.add(start)
        while queue:
            column, row = queue.popleft()
            cells.append((column, row))
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    other = (column + dx, row + dy)
                    if (0 <= other[0] < gray.shape[1] and 0 <= other[1] < gray.shape[0]
                            and gray[other[1], other[0]] < 90 and other not in seen):
                        seen.add(other)
                        queue.append(other)
        if len(cells) >= 16:
            components.append(cells)
    return components


def render(snapshot, evidence_path, output):
    pi = snapshot / 'pi_snapshot'
    backup = json.loads((snapshot / 'backup_manifest.json').read_text(encoding='utf-8'))
    directory = pi / 'inspections/ble_scene_17' / backup['inspection_run']
    captures = json.loads((directory / 'inspection_manifest.json').read_text(encoding='utf-8'))['captures']
    records = {}
    for record in captures:
        records[record['image']] = record
        if record.get('camera_2_image'):
            records[record['camera_2_image']] = record
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    config = _load_map_config(pi / 'maps/ble_scene_17.yaml')
    with Image.open(pi / 'maps/ble_scene_17.pgm') as source:
        gray_image = source.convert('L')
    gray = np.array(gray_image)
    components = occupied_components(gray, config)
    objects = []
    for item in evidence['objects']:
        estimates = []
        for view in item['evidence']:
            if not (directory / view['photo']).is_file():
                raise FileNotFoundError(view['photo'])
            record = records[view['photo']]
            bbox = view['bbox']
            u = (bbox[0] + bbox[2]) / 2000
            yaw = float(record['yaw_deg']) + bearing_offset(u, evidence['camera_horizontal_fov_deg'])
            try:
                hits = ray_surfaces(gray, config, float(record['x_m']), float(record['y_m']), yaw)
            except ValueError:
                # Thin lidar walls have gaps. Search a small angular envelope,
                # explicitly preserving the chosen bearing in the sidecar.
                hits = None
                for correction in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6]:
                    try:
                        hits = ray_surfaces(gray, config, float(record['x_m']), float(record['y_m']), yaw + correction)
                        yaw += correction
                        break
                    except ValueError:
                        continue
                if hits is None:
                    raise ValueError('Cannot estimate reviewed object: ' + item['name'])
            if item['mode'] == 'furniture_component':
                # A wide table box does not determine a precise center bearing.
                # Test its angular envelope instead of snapping to a far wall.
                candidates = []
                for sample_u in np.linspace(bbox[0] / 1000, bbox[2] / 1000, 21):
                    sample_yaw = float(record['yaw_deg']) + bearing_offset(float(sample_u), evidence['camera_horizontal_fov_deg'])
                    try:
                        candidate = ray_surfaces(gray, config, float(record['x_m']), float(record['y_m']), sample_yaw)[0]
                        candidate['sample_bearing_deg'] = sample_yaw
                        candidates.append(candidate)
                    except ValueError:
                        continue
                free_rows, free_columns = np.where(gray > 240)
                free_center = np.array([np.median(free_columns), np.median(free_rows)])
                eligible = []
                for candidate in candidates:
                    for cells in components:
                        if ((candidate['column'], candidate['row']) in cells
                                and np.ptp(np.array(cells), axis=0).max() <= 45):
                            eligible.append((np.linalg.norm(np.mean(cells, axis=0) - free_center), candidate))
                if not eligible:
                    raise ValueError('Table angular envelope has no plausible furniture outline')
                hits = [min(eligible, key=lambda pair: pair[0])[1]]
                yaw = hits[0]['sample_bearing_deg']
            hit = hits[-1] if item['mode'] in {'wall_surface', 'beyond_glass'} else hits[0]
            estimates.append({'photo': view['photo'], 'bbox': bbox, 'bearing_deg': yaw,
                              'method': item['mode'], **hit})
        x = statistics.mean(hit['x_m'] for hit in estimates)
        y = statistics.mean(hit['y_m'] for hit in estimates)
        if item['mode'] == 'furniture_component':
            hit_cells = {(hit['column'], hit['row']) for hit in estimates}
            initial = [cells for cells in components if any(cell in hit_cells for cell in cells)]
            if not initial:
                raise ValueError('No furniture component for reviewed table')
            seed = initial[0]
            center = np.mean(seed, axis=0)
            # Join nearby pieces of a fragmented lidar outline, not wall chains.
            selected = [cells for cells in components
                if np.linalg.norm(np.mean(cells, axis=0) - center) * config['resolution'] <= 1.1
                and np.ptp(np.array(cells), axis=0).max() <= 45]
            if selected:
                center = np.mean([cell for cells in selected for cell in cells], axis=0)
                x = config['origin'][0] + (center[0] + .5) * config['resolution']
                y = config['origin'][1] + (gray.shape[0] - center[1] - .5) * config['resolution']
        if item['mode'] == 'beyond_glass':
            angle = math.radians(estimates[0]['bearing_deg'])
            # Symbolic offset only: no stereo depth or claim of measured range.
            x += .35 * math.cos(angle)
            y += .35 * math.sin(angle)
        spread = max(math.hypot(hit['x_m'] - x, hit['y_m'] - y) for hit in estimates)
        objects.append({**item, 'x_m': float(x), 'y_m': float(y), 'estimates': estimates,
            'illustrative_radius_m': max(item['uncertainty_m'], spread),
            'uncertainty_is_not_a_statistical_confidence_interval': True,
            'navigation_allowed': False, 'permanent_semantic_memory_updated': False})
    scale, padding, top = 9, 150, 195
    crop = _known_crop(gray_image, padding=9)
    map_image = gray_image.crop(crop).convert('RGB').resize(
        ((crop[2] - crop[0]) * scale, (crop[3] - crop[1]) * scale), Image.Resampling.NEAREST)
    width, height = map_image.width + padding * 2, map_image.height + top + padding + 175
    image = Image.new('RGB', (width, height), '#f5f7fa')
    left, map_top = padding, top + 50
    image.paste(map_image, (left, map_top))

    def pixel(x, y):
        return (left + ((x - config['origin'][0]) / config['resolution'] - crop[0]) * scale,
            map_top + (gray.shape[0] - 1 - (y - config['origin'][1]) / config['resolution'] - crop[1]) * scale)

    base_palette = ['#8d612f', '#468548', '#468548', '#8d4fc0', '#168ca5', '#2472b8', '#2472b8', '#d07926', '#b46b45', '#b46b45']
    palette = [base_palette[index % len(base_palette)] for index in range(len(objects))]
    overlay = Image.new('RGBA', image.size)
    tint = ImageDraw.Draw(overlay)
    for item, color in zip(objects, palette):
        x, y = pixel(item['x_m'], item['y_m'])
        radius = item['illustrative_radius_m'] / config['resolution'] * scale
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        tint.ellipse((x - radius, y - radius, x + radius, y + radius), fill=rgb + (20,), outline=rgb + (85,), width=2)
        if item['name'] in {'玻璃墙', '沿墙柜子'}:
            points = [pixel(hit['x_m'], hit['y_m']) for hit in item['estimates']]
            if len(points) > 1:
                tint.line(points, fill=rgb + (120,), width=12)
    # Keep illustrative uncertainty halos away from the title and footnotes.
    tint.rectangle((0, 0, width, top - 5), fill=(0, 0, 0, 0))
    tint.rectangle((0, height - 155, width, height), fill=(0, 0, 0, 0))
    image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 30)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 25)
    title = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 41)
    text_boxes = []

    def text(xy, value, face, color):
        box = draw.textbbox(xy, value, font=face)
        if any(intersects(box, other, margin=0) for other in text_boxes):
            raise ValueError('Overlapping text: ' + value)
        if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
            raise ValueError('Clipped text: ' + value)
        text_boxes.append(box)
        draw.text(xy, value, font=face, fill=color)

    text((36, 24), 'ble_scene_17｜物体大致位置语义地图', title, '#18334c')
    text((36, 89), '根据已核对照片、日志朝向与雷达边界估算；连线指向物体区域，不是拍照站位。', small, '#364b5e')
    text((36, 131), '淡色圆为示意性位置范围；玻璃外物体另作候选。未测量精确深度，不用于自动导航。', small, '#825d28')
    anchors = [pixel(item['x_m'], item['y_m']) for item in objects]
    label_lines = [wrap(draw, item['name'], font, 380) for item in objects]
    sizes = [(max(draw.textlength(line, font=font) for line in lines) + 28,
              len(lines) * 40 + 16) for lines in label_lines]
    boxes = place_labels(anchors, sizes, (18, top, width - 18, height - 160))
    for anchor, box, color in zip(anchors, boxes, palette):
        x, y = anchor
        draw.line((x, y, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2), fill=color, width=3)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline='white', width=2)
    for box, lines, color in zip(boxes, label_lines, palette):
        draw.rounded_rectangle(box, radius=9, fill='white', outline=color, width=2)
        for index, line in enumerate(lines):
            text((box[0] + 14, box[1] + 5 + index * 40), line, font, color)
    text((36, height - 112), '物体名称经选取照片复核；重复/误识别的地面痕迹、反光和桌板“柜子”未直接采用。', small, '#364b5e')
    text((36, height - 72), '这是粗粒度位置示意，方向误差仍会影响估算；原始地图、识别结果和之前备份均保持不变。', small, '#364b5e')
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, dpi=(180, 180))
    report = {'scope': evidence['scope'], 'source_snapshot': str(snapshot), 'objects': objects,
        'camera_horizontal_fov_deg': evidence['camera_horizontal_fov_deg'],
        'camera_intrinsics_calibrated': False, 'stereo_depth_measured': False,
        'label_overlap_count': 0, 'text_overlap_count': 0, 'label_boxes': boxes,
        'text_boxes': text_boxes, 'size': list(image.size)}
    output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output.resolve()), 'size': list(image.size),
        'object_count': len(objects), 'text_overlap_count': 0,
        'positions': [(o['name'], round(o['x_m'], 2), round(o['y_m'], 2)) for o in objects]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.snapshot, args.evidence, args.output)
