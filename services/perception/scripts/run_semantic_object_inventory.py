"""Qwen inventory of all visible objects, including small wall fixtures.

Offline replay works without torch. Real inference reuses the project's Qwen
backend on the 3090, with independent overview and upper-image detail passes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from qwen_grounded_tracker.inspection_annotation import Qwen3VLInspectionBackend, extract_json_object
from qwen_grounded_tracker.semantic_object_catalog import (
    PROMPT_VERSION, ReadOnlyMap, apply_reviews, build_catalog, build_object_prompt,
    file_sha256, normalize_objects, replay_panorama,
)


class GroundServiceBackend:
    """Reuse the already loaded Qwen /agent/ground API, no second GPU copy.

    The interface returns pixel boxes, not normalized model-text boxes. This
    targeted supplement is not claimed to be an exhaustive room inventory.
    """
    def __init__(self, url, queries, timeout, detail_queries=None, session=None):
        if session is None:
            import requests
            session = requests.Session()
            session.trust_env = False
        self.session = session
        self.url, self.queries, self.timeout = url, queries, timeout
        self.detail_queries = detail_queries
        self.http_calls = 0

    def annotate(self, images, prompt):
        rows = []
        for image_index, path in enumerate(images, 1):
            with Image.open(path) as image:
                width, height = image.size
            queries = self.detail_queries if self.detail_queries is not None and '本轮特别检查' in prompt else self.queries
            for query in queries:
                with path.open('rb') as stream:
                    self.http_calls += 1
                    response = self.session.post(self.url, files={'image': (path.name, stream, 'image/jpeg')},
                        data={'instruction': '这是单路照片，不是左右拼接图。找出画面中可见的' + query + '（最多三个）；看不清或没有就返回未找到，不要把其他墙面面板猜成目标。'},
                        timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                if result.get('ok') is not True:
                    raise ValueError('Qwen grounding failed: ' + str(result.get('error')))
                candidates = (result.get('candidates') or [result]) if result.get('found') is True else []
                print(json.dumps({'qwen_http_call': self.http_calls, 'image_index': image_index,
                    'query': query, 'candidate_count': len(candidates)}, ensure_ascii=False), flush=True)
                for candidate in candidates:
                    try:
                        if float(candidate.get('confidence', 0)) < .65:
                            continue
                    except (TypeError, ValueError):
                        continue
                    box = candidate.get('bbox')
                    if not isinstance(box, list) or len(box) != 4:
                        continue
                    rows.append({'name_zh': query, 'category': query, 'mobility': 'unknown',
                        'confidence': candidate.get('confidence', 0.),
                        'appearance': 'Qwen 定向补扫候选，类别与位置需复核',
                        'views': [{'image_index': image_index, 'bbox_2d': [
                            box[0] / width * 1000, box[1] / height * 1000,
                            box[2] / width * 1000, box[3] / height * 1000]}]})
        return json.dumps({'objects': rows, 'complete_visible_inventory': False}, ensure_ascii=False)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def run(args, backend=None):
    manifest_path = args.inspection / 'inspection_manifest.json'
    captures = json.loads(manifest_path.read_text(encoding='utf-8'))['captures']
    all_capture_count = len(captures)
    if args.limit:
        captures = captures[:args.limit]
    for immutable in [args.inspection, args.map.parent]:
        if args.output.resolve() == immutable.resolve() or args.output.resolve().is_relative_to(immutable.resolve()):
            raise ValueError('Output must be separate from source photos and map directory')
    args.output.mkdir(parents=True, exist_ok=True)
    map_data = ReadOnlyMap(args.map, args.map_yaml)
    source_hash = file_sha256(manifest_path)
    observations, failures, backend_calls, cache_hits = [], [], 0, 0
    if args.replay:
        payload = json.loads(args.replay.read_text(encoding='utf-8'))
        observations = replay_panorama(payload, captures)
        source = {'mode': 'existing_qwen_results_replay_not_new_inference',
                  'path': str(args.replay), 'sha256': file_sha256(args.replay)}
    else:
        detail_queries = args.detail_queries.split(',') if args.detail_queries else None
        backend = backend or (GroundServiceBackend(args.ground_url, args.queries.split(','), args.timeout, detail_queries) if args.ground_url else Qwen3VLInspectionBackend(str(args.model_path), device=args.device,
            local_files_only=True, max_new_tokens=args.max_new_tokens,
            max_time_seconds=args.timeout, min_visual_tokens=128, max_visual_tokens=1024))
        source = {'mode': 'targeted_shared_qwen_grounding' if args.ground_url else 'new_qwen_inference',
                  'model_path': str(args.model_path), 'ground_url': args.ground_url,
                  'queries': args.queries if args.ground_url else None,
                  'detail_queries': args.detail_queries if args.ground_url else None,
                  'prompt_version': PROMPT_VERSION, 'passes': ['overview', 'upper_details']}
        if args.baseline:
            observations.extend(replay_panorama(json.loads(args.baseline.read_text(encoding='utf-8')), captures))
            source['baseline_sha256'] = file_sha256(args.baseline)
        checkpoint_dir = args.output / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        for capture_index, record in enumerate(captures, 1):
            images = []
            for name in [record.get('image'), record.get('camera_2_image')]:
                if not name:
                    continue
                if Path(name).name != name:
                    raise ValueError('Manifest image must be a local basename')
                images.append(args.inspection / name)
            if not images:
                failures.append({'capture_id': record.get('target_id'), 'error': 'no_images'})
                continue
            for details in [False, True]:
                pass_name = 'details' if details else 'overview'
                fraction = .45 if details else 1.
                checkpoint = checkpoint_dir / (f'{capture_index:04d}_{pass_name}.json')
                try:
                    image_hashes = [file_sha256(image) for image in images]
                    fingerprint = hashlib.sha256(json.dumps({'record': record,
                        'image_hashes': image_hashes, 'source': source,
                        'map_sha256': map_data.sha256, 'pass': pass_name},
                        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                    if checkpoint.exists() and not args.force:
                        cached = json.loads(checkpoint.read_text(encoding='utf-8'))
                        if cached.get('input_fingerprint') == fingerprint and cached.get('success') is True:
                            observations.extend(cached['observations'])
                            cache_hits += 1
                            continue
                    prepared = images
                    if details:
                        prepared = []
                        crop_dir = args.output / 'detail_crops'
                        crop_dir.mkdir(exist_ok=True)
                        for image in images:
                            crop_path = crop_dir / image.name
                            with Image.open(image) as original:
                                original.crop((0, 0, original.width, round(original.height * fraction))).save(crop_path)
                            prepared.append(crop_path)
                    print(f"Qwen {capture_index}/{len(captures)} {pass_name}: {record.get('target_id')}", flush=True)
                    backend_calls += 1
                    raw = backend.annotate(prepared, build_object_prompt(len(images), details))
                    payload = extract_json_object(raw)
                    normalized = normalize_objects(payload, images, record, fraction)
                    observations.extend(normalized)
                    write_json(checkpoint, {'success': True, 'input_fingerprint': fingerprint,
                        'source_images_sha256': image_hashes, 'raw_response': raw,
                        'possibly_missed': payload.get('possibly_missed', []),
                        'complete_visible_inventory': payload.get('complete_visible_inventory') is True,
                        'observations': normalized})
                except Exception as exc:
                    failures.append({'capture_id': record.get('target_id'), 'pass': pass_name, 'error': str(exc)})
                    print('推理失败，保留部分结果：' + str(exc), file=sys.stderr, flush=True)
    previous = json.loads(args.previous.read_text(encoding='utf-8')) if args.previous else None
    catalog = build_catalog(observations, map_data, manifest_path, previous)
    catalog['source'] = source
    catalog['source_manifest_sha256'] = source_hash
    if args.reviews:
        catalog = apply_reviews(catalog, json.loads(args.reviews.read_text(encoding='utf-8')), map_data)
    report = {'capture_count': len(captures), 'total_capture_count': all_capture_count,
        'observation_count': len(observations), 'candidate_cluster_count': len(catalog['objects']),
        'category_counts': {category: sum(item['category'] == category for item in catalog['objects'])
                            for category in sorted({item['category'] for item in catalog['objects']})},
        'new_backend_calls': backend_calls, 'cache_hits': cache_hits, 'source': source,
        'new_http_calls': getattr(backend, 'http_calls', 0),
        'counts_are_candidate_clusters_not_real_object_counts': True,
        'failures': failures, 'all_selected_inputs_processed': not failures,
        'complete_scene_inventory': False,
        'localized_verified_count': sum(item['localization_allowed'] for item in catalog['objects']),
        'navigation_verified_count': sum(item['navigation_allowed'] for item in catalog['objects'])}
    catalog['processing_report'] = report
    if file_sha256(args.map) != map_data.sha256 or file_sha256(args.map_yaml) != map_data.yaml_sha256 or file_sha256(manifest_path) != source_hash:
        raise RuntimeError('Read-only source changed during inventory processing')
    write_json(args.output / 'semantic_object_memory.json', catalog)
    write_json(args.output / 'inventory_report.json', report)
    with (args.output / 'object_inventory.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'name', 'category', 'x_m', 'y_m', 'position_status',
            'uncertainty_m', 'confidence', 'evidence_count', 'navigation_allowed', 'localization_allowed'])
        for item in catalog['objects']:
            xy = item['map_xy'] or ['', '']
            writer.writerow([item['id'], item['name'], item['category'], *xy,
                item['position_status'], item['position_uncertainty_m'], item['confidence'],
                item['evidence_count'], item['navigation_allowed'], item['localization_allowed']])
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def parser():
    result = argparse.ArgumentParser(description='Qwen 全可见物体清单、细小设施补扫与结构化语义地图旁车')
    result.add_argument('--inspection', type=Path, required=True)
    result.add_argument('--map', type=Path, required=True)
    result.add_argument('--map-yaml', type=Path, required=True)
    result.add_argument('--output', type=Path, required=True)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument('--replay', type=Path, help='整理现有 panorama_semantics.json，不重新推理')
    mode.add_argument('--model-path', type=Path, help='3090 上的本地 Qwen3-VL 模型路径')
    mode.add_argument('--ground-url', help='复用已运行的 Qwen /agent/ground；定向补扫而非全内容识别')
    result.add_argument('--queries', default='插座,开关,标牌,灯具', help='共享服务补扫类别，逗号分隔')
    result.add_argument('--detail-queries', help='上部细节裁剪只检查这些类别；省略则与整图相同')
    result.add_argument('--baseline', type=Path, help='共享服务补扫时保留既有 panorama_semantics.json 内容')
    result.add_argument('--device', default='cuda')
    result.add_argument('--max-new-tokens', type=int, default=4096)
    result.add_argument('--timeout', type=float, default=120.)
    result.add_argument('--limit', type=int, default=0)
    result.add_argument('--force', action='store_true')
    result.add_argument('--previous', type=Path, help='同一地图的旧旁车，用于尽量保留候选 ID')
    result.add_argument('--reviews', type=Path, help='人工/实测坐标确认记录，不接受 Qwen 自行批准')
    return result


if __name__ == '__main__':
    args = parser().parse_args()
    if args.limit < 0 or args.timeout <= 0 or args.max_new_tokens <= 0:
        raise ValueError('Invalid limit/token/timeout')
    raise SystemExit(bool(run(args)['failures']))
