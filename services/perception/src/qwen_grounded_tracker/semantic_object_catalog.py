"""Evidence-bearing object memory. Coarse photo estimates are search priors only."""
from __future__ import annotations

import hashlib
from copy import deepcopy
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image

SCHEMA = 'semantic_object_memory.v1'
PROMPT_VERSION = 'all_visible_objects.v1'
ALIASES = {
    '插座': 'outlet', '电源插座': 'outlet', 'socket': 'outlet', '开关': 'switch',
    '标牌': 'sign', '出口标牌': 'sign', '灯具': 'light', '门': 'door',
    '玻璃门': 'door', '门框': 'door_frame', '门把手': 'door_handle',
    '玻璃': 'glass_wall', '玻璃边界': 'glass_wall', '玻璃墙': 'glass_wall',
    '窗户': 'window', '柜子': 'cabinet', '蓝色储物柜': 'cabinet',
    '柜子底部': 'cabinet', '柜子的底部': 'cabinet', '桌子': 'table',
    '椅子': 'chair', '楼梯': 'stairs', '扫帚': 'broom', '簸箕': 'dustpan',
    '涂鸦': 'surface_mark', '绿色痕迹': 'surface_mark', '黑色柱子': 'pillar',
    '鞋子': 'shoe', '人': 'person', '地板': 'floor', '地面': 'floor',
    '墙壁': 'wall', '墙': 'wall', '踢脚线': 'baseboard', '反射': 'reflection',
}
BACKGROUND = {'floor', 'wall', 'baseboard', 'reflection', 'floor reflection', 'floor texture'}
FIXED_TYPES = {'outlet', 'switch', 'sign', 'light', 'door', 'door_frame', 'door_handle',
    'glass_wall', 'window', 'cabinet', 'stairs', 'surface_mark', 'pillar'}
WALL_TYPES = FIXED_TYPES - {'pillar'}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value, default=None):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def category_for(name, category=None):
    if str(name) in ALIASES:
        return ALIASES[str(name)]
    if any(word in str(name).lower() for word in ['反光', '反射', '光斑', 'reflection']):
        return 'reflection'
    if any(word in str(name) for word in ['地板', '地面', '痕迹', '斑点', '小红点', '污渍', '标记线']):
        return 'floor_feature'
    raw = ALIASES.get(str(category), str(category or name or 'unknown').lower().replace(' ', '_'))
    if raw in {'floor_mark', 'floor_debris', 'floor_texture', 'debris', 'wall_corner', 'corner'}:
        return 'floor_feature'
    return raw if re.fullmatch(r'[a-z][a-z0-9_]*', raw) else 'unknown_object'


def build_object_prompt(image_count: int, details=False):
    focus = '本轮特别检查插座、开关、标牌、灯具、门框/把手及柜体上的可辨认标记。输入可能是原照片上部裁剪。' if details else '全面检查所有可见设施、家具和活动物体，不要只识别占画面最大的八个物体。'
    return f'''你是机器人的逐照片语义记录器。{focus}
输入是独立编号的 {image_count} 张照片，而不是左右拼接图。
只记录照片中确实可见的东西。不要根据地图目标名称猜类别；桌子侧板不是柜子。
看不清的墙面面板标为 uncertain_panel，别同时断言是插座和开关。
同一个物体可有多个 views，不同位置的同类物体分开；不能证明相同就不要强行合并。
category 用英文 snake_case，name_zh 用中文。区分 fixed、movable、unknown；
玻璃后的物体 behind_glass=true。不要输出地图坐标、深度、电机指令或永久确认。
views 的 image_index 是 1..{image_count}；bbox_2d 是该张输入图独立的 0..1000 相对坐标。
小物体必须用紧框，不要用整张图当插座框。无法框定时 views=[]，仍保留名称候选。
输出必须是一个 JSON 对象，不要 Markdown：
{{"objects":[{{"name_zh":"插座","category":"outlet","mobility":"fixed",
"confidence":0.85,"behind_glass":false,"appearance":"白色墙面上的面板",
"views":[{{"image_index":1,"bbox_2d":[200,10,260,80]}}]}}],
"possibly_missed":["有遮挡/看不清的区域"],"complete_visible_inventory":false}}
只允许对可见内容声称清单完整，绝不代表房间没有未拍到的物品。'''


def normalize_objects(payload, images, record, crop_fraction=1., source='qwen_object_inventory'):
    observations = []
    rows = payload.get('objects')
    if not isinstance(rows, list):
        raise ValueError('Qwen response must contain an objects list')
    for index, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        name = str(item.get('name_zh') or item.get('name') or '').strip()
        if not name:
            continue
        views = []
        for view in item.get('views') or []:
            if not isinstance(view, dict):
                continue
            number = finite_number(view.get('image_index'))
            if number is None or not number.is_integer() or not 1 <= number <= len(images):
                continue
            bbox = view.get('bbox_2d')
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            bbox = [finite_number(value) for value in bbox]
            if any(value is None or not 0 <= value <= 1000 for value in bbox):
                continue
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                continue
            bbox[1] *= crop_fraction
            bbox[3] *= crop_fraction
            views.append({'image_index': int(number), 'image': Path(images[int(number) - 1]).name,
                          'bbox_1000': bbox, 'crop_fraction': crop_fraction})
        identity = hashlib.sha256(json.dumps([record.get('target_id'), source, crop_fraction, index, name, views], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
        confidence = finite_number(item.get('confidence'), 0.)
        mobility = item.get('mobility') if item.get('mobility') in {'fixed', 'movable', 'unknown'} else 'unknown'
        observations.append({'observation_id': 'obs_' + identity,
            'capture_id': record.get('target_id'), 'station_id': record.get('panorama_station_id'),
            'captured_at': record.get('captured_at'), 'name': name,
            'category': category_for(name, item.get('category')), 'mobility': mobility,
            'confidence': min(1., max(0., confidence)),
            'behind_glass': item.get('behind_glass') is True,
            'appearance': str(item.get('appearance') or ''), 'views': views,
            'robot_pose': {key: finite_number(record.get(key)) for key in ['x_m', 'y_m', 'yaw_deg']},
            'source': source, 'camera_reference_uncertain': source == 'legacy_panorama',
            'semantic_reviewed': False, 'position_verified': False})
    return observations


def replay_panorama(payload, captures):
    records = {record['target_id']: record for record in captures}
    result = []
    for observation in payload.get('observations', []):
        record = records.get(observation.get('target_id'))
        if record is None:
            continue
        images = [record.get('image'), record.get('camera_2_image')]
        images = [value for value in images if value]
        inference = observation.get('result') or {}
        objects, seen = [], set()
        for item in list(inference.get('landmarks') or []) + list(inference.get('objects') or []):
            if not isinstance(item, dict):
                continue
            key = json.dumps([item.get('name'), item.get('camera_bbox')], ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            side = str(item.get('stereo_consistency') or '')
            image_index = 2 if side in {'right', 'right_only'} and len(images) > 1 else 1
            objects.append({**item, 'category': category_for(item.get('name'), item.get('type')),
                'mobility': 'fixed' if category_for(item.get('name'), item.get('type')) in FIXED_TYPES else 'unknown',
                'views': [{'image_index': image_index, 'bbox_2d': item.get('camera_bbox')}]})
        result.extend(normalize_objects({'objects': objects}, images, record, source='legacy_panorama'))
    return result


class ReadOnlyMap:
    def __init__(self, pgm, yaml):
        self.sha256 = file_sha256(pgm)
        self.yaml_sha256 = file_sha256(yaml)
        text = yaml.read_text(encoding='utf-8')
        self.resolution = float(re.search(r'^resolution:\s*([^\s]+)', text, re.M).group(1))
        origin = json.loads(re.search(r'^origin:\s*(\[[^\n]+\])', text, re.M).group(1))
        self.origin = origin[:2]
        if len(origin) > 2 and abs(float(origin[2])) > 1e-9:
            raise ValueError('Rotated map YAML not supported for coarse projection')
        with Image.open(pgm) as image:
            self.gray = np.array(image.convert('L'))

    def free(self, xy):
        if not isinstance(xy, (list, tuple)) or len(xy) != 2 or any(finite_number(v) is None for v in xy):
            return False
        column = math.floor((xy[0] - self.origin[0]) / self.resolution)
        row = self.gray.shape[0] - 1 - math.floor((xy[1] - self.origin[1]) / self.resolution)
        # Verify only point occupancy here; chassis clearance is still the
        # runtime planner's responsibility, never inferred by this exporter.
        return bool(0 <= row < self.gray.shape[0] and 0 <= column < self.gray.shape[1] and self.gray[row, column] > 240)

    def estimate(self, observation, fov=82.):
        pose = observation['robot_pose']
        if any(pose.get(key) is None for key in ['x_m', 'y_m', 'yaw_deg']):
            return None
        hits = []
        for view in observation['views']:
            bbox = view['bbox_1000']
            center = (bbox[0] + bbox[2]) / 2000
            bearing = math.radians(pose['yaw_deg']) + math.atan((1 - 2 * center) * math.tan(math.radians(fov / 2)))
            view_hits = []
            for distance in np.arange(.2, 8., self.resolution / 2):
                x, y = pose['x_m'] + distance * math.cos(bearing), pose['y_m'] + distance * math.sin(bearing)
                column = math.floor((x - self.origin[0]) / self.resolution)
                row = self.gray.shape[0] - 1 - math.floor((y - self.origin[1]) / self.resolution)
                if not (0 <= row < self.gray.shape[0] and 0 <= column < self.gray.shape[1]):
                    break
                if self.gray[row, column] < 90:
                    view_hits.append([float(x), float(y)])
            if view_hits:
                hits.append(view_hits[-1] if observation['category'] in WALL_TYPES else view_hits[0])
        if not hits:
            return None
        return [sum(point[axis] for point in hits) / len(hits) for axis in [0, 1]]


def build_catalog(observations, map_data: ReadOnlyMap, source_manifest, previous=None):
    if previous and previous.get('map_sha256') != map_data.sha256:
        raise ValueError('Previous catalog belongs to a different map')
    objects, background = [], []
    previous_objects = (previous or {}).get('objects') or []
    for observation in observations:
        if observation['category'] in BACKGROUND:
            background.append(observation)
            continue
        xy = map_data.estimate(observation)
        # Candidate clustering only. Do not claim this proves two detections
        # refer to the same real object, or count stationary frames as anchors.
        matches = [item for item in objects if item['category'] == observation['category']
            and item['behind_glass'] == observation['behind_glass']
            and xy is not None and item['map_xy'] is not None
            and math.dist(xy, item['map_xy']) <= .65]
        if matches:
            candidate = min(matches, key=lambda item: math.dist(xy, item['map_xy']))
            candidate['evidence'].append(observation)
            candidate['confidence'] = max(candidate['confidence'], observation['confidence'])
            candidate['aliases'] = sorted(set(candidate['aliases'] + [observation['name']]))
            continue
        prior = [item for item in previous_objects if item.get('category') == observation['category']
            and item.get('map_xy') is not None and xy is not None and math.dist(xy, item['map_xy']) <= .45]
        identity = (min(prior, key=lambda item: math.dist(xy, item['map_xy']))['id'] if prior
            else 'obj_' + hashlib.sha256((map_data.sha256 + observation['observation_id']).encode()).hexdigest()[:16])
        if any(item['id'] == identity for item in objects):
            identity += '_' + observation['observation_id'][-6:]
        new_item = {'id': identity, 'name': observation['name'], 'aliases': [observation['name']],
            'category': observation['category'], 'type': observation['category'], 'label': observation['category'],
            'map_xy': xy, 'approach_xy': None, 'position_status': 'estimated' if xy else 'unlocalized',
            'position_method': 'camera_bearing_map_surface_search_prior' if xy else 'no_reliable_surface_intersection',
            'position_uncertainty_m': 1.2 if xy else None, 'mobility': observation['mobility'],
            'confidence': observation['confidence'], 'behind_glass': observation['behind_glass'],
            'semantic_reviewed': False, 'position_verified': False, 'approach_verified': False,
            'navigation_allowed': False, 'localization_allowed': False, 'grasp_allowed': False,
            'requires_visual_recheck': True, 'evidence': [observation]}
        matching_prior = next((item for item in prior if item['id'] == identity), None)
        if matching_prior and matching_prior.get('position_verified') is True:
            for key in ['map_xy', 'approach_xy', 'position_status', 'position_method',
                        'position_uncertainty_m', 'semantic_reviewed', 'position_verified',
                        'approach_verified', 'navigation_allowed', 'localization_allowed', 'review']:
                if key in matching_prior:
                    new_item[key] = deepcopy(matching_prior[key])
            new_item['evidence'] = deepcopy(matching_prior.get('evidence', [])) + [observation]
        objects.append(new_item)
    seen_ids = {item['id'] for item in objects}
    suspicions = []
    for prior in previous_objects:
        if prior.get('id') in seen_ids:
            continue
        retained = deepcopy(prior)
        retained['unobserved_in_this_run'] = True
        retained['requires_visual_recheck'] = True
        objects.append(retained)
        suspicions.append({'id': 'recheck_' + retained['id'],
            'expected_item_id': retained['id'], 'expected_query': retained['name'],
            'map_xy': retained.get('map_xy'), 'last_change_type': 'unobserved_not_proven_absent',
            'requires_recheck_on_every_visit': True})
    for item in objects:
        item['station_ids'] = sorted({e['station_id'] for e in item['evidence'] if e['station_id'] is not None})
        item['evidence_count'] = len(item['evidence'])
    return {'schema': SCHEMA, 'map_id': 'map_' + map_data.sha256[:12],
        'candidate_cluster_counts_are_not_real_object_counts': True,
        'map_sha256': map_data.sha256, 'map_yaml_sha256': map_data.yaml_sha256,
        'source_manifest': str(source_manifest), 'camera_intrinsics_calibrated': False,
        'camera_extrinsics_calibrated': False, 'stereo_depth_measured': False,
        'complete_scene_inventory': False, 'objects': objects, 'background_observations': background,
        'entities': {item['id']: {'target_id': item['id'],
            'target_x_m': item['map_xy'][0], 'target_y_m': item['map_xy'][1],
            'semantic_label': item['category'], 'permanent': True,
            'position_verified': True, 'localization_allowed': True,
            'qwen_annotation': {'name_zh': item['name'], 'category': item['category']}}
            for item in objects if item.get('map_xy') is not None and item.get('position_verified') is True
            and item.get('localization_allowed') is True and item.get('mobility') == 'fixed'
            and item.get('behind_glass') is not True},
        'suspicious_locations': suspicions, 'search_regions': [
            {'id': item['id'], 'query': item['name'], 'map_xy': item['map_xy'],
             'radius_m': item['position_uncertainty_m'], 'requires_visual_recheck': True}
            for item in objects if item['map_xy'] is not None]}


def apply_reviews(catalog, reviews, map_data):
    if reviews.get('map_sha256') != catalog['map_sha256']:
        raise ValueError('Review map hash mismatch')
    by_id = {item['id']: item for item in catalog['objects']}
    for review in reviews.get('objects', []):
        item = by_id.get(review.get('id'))
        if item is None:
            raise ValueError('Review references unknown object')
        if not review.get('reviewer') or not review.get('reviewed_at'):
            raise ValueError('Review needs reviewer and timestamp')
        xy = review.get('verified_map_xy')
        if not isinstance(xy, (list, tuple)) or len(xy) != 2 or any(finite_number(v) is None for v in xy):
            raise ValueError('Review needs verified finite coordinates')
        xy = [float(v) for v in xy]
        if not (map_data.origin[0] <= xy[0] < map_data.origin[0] + map_data.gray.shape[1] * map_data.resolution
                and map_data.origin[1] <= xy[1] < map_data.origin[1] + map_data.gray.shape[0] * map_data.resolution):
            raise ValueError('Verified object coordinate is outside map frame')
        catalog['entities'].pop(item['id'], None)
        item.update(map_xy=[float(v) for v in xy], semantic_reviewed=True,
            position_verified=True, position_status='verified', review=review,
            mobility=review.get('mobility', item['mobility']))
        if review.get('name'):
            item['name'] = str(review['name'])
        item['localization_allowed'] = (item['mobility'] == 'fixed' and not item['behind_glass']
            and review.get('localization_allowed') is True)
        approach = review.get('verified_approach_xy')
        if approach is not None:
            if not map_data.free(approach):
                raise ValueError('Verified approach is not an old-map free point')
            item['approach_xy'] = [float(v) for v in approach]
            item['approach_verified'] = True
        item['navigation_allowed'] = bool(item['approach_verified'] and not item['behind_glass']
            and review.get('navigation_allowed') is True)
        # Grasp always requires live model alignment; no offline promotion.
        if item['localization_allowed']:
            catalog['entities'][item['id']] = {'target_id': item['id'],
                'target_x_m': item['map_xy'][0], 'target_y_m': item['map_xy'][1],
                'semantic_label': item['category'], 'permanent': True,
                'position_verified': True, 'localization_allowed': True,
                'qwen_annotation': {'name_zh': item['name'], 'category': item['category'],
                    'confidence': item['confidence'], 'needs_human_review': False}}
    return catalog
