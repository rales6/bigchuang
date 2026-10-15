"""Supplement reviewed map evidence, without overwriting prior artifacts."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.render_approximate_object_map import render


def build(root):
    base = root / 'semantic_maps/ble_scene_17_20260917_190706'
    evidence = json.loads((base / 'object_locations/approximate_object_evidence.json').read_text(encoding='utf-8'))
    evidence['scope'] = 'reviewed_major_objects_and_small_fixture_candidates_not_complete_room'
    evidence['objects'] += [
        {'name': '插座候选 A（已识别）', 'mode': 'wall_surface', 'uncertainty_m': 1.2,
         'evidence': [{'photo': 'panorama_s01_d000.jpg', 'bbox': [522, 14, 544, 52]}]},
        {'name': '插座候选 B（可能与 A 重复）', 'mode': 'wall_surface', 'uncertainty_m': 1.2,
         'evidence': [{'photo': 'panorama_s03_d180.jpg', 'bbox': [320, 8, 347, 50]}]},
        {'name': '绿色出口标牌（位置待核）', 'mode': 'wall_surface', 'uncertainty_m': 1.2,
         'evidence': [{'photo': 'panorama_s03_d180_camera_2.jpg', 'bbox': [540, 20, 570, 110]}]},
        {'name': '玻璃外悬挂灯具', 'mode': 'beyond_glass', 'uncertainty_m': 1.2,
         'evidence': [{'photo': 'panorama_s01_d180.jpg', 'bbox': [725, 0, 790, 110]}]},
        {'name': '柜面绿色标记', 'mode': 'first_surface', 'uncertainty_m': 1.0,
         'evidence': [{'photo': 'panorama_s01_d120.jpg', 'bbox': [580, 0, 670, 100]}]},
    ]
    evidence['unconfirmed_items'] = [
        {'name': '开关', 'reason': '旧识别与插座共用同一框，未证实为独立开关'},
        {'name': '黑色立柱', 'reason': '可能是柜体或门框，未确认独立物体'},
    ]
    evidence['parts'] = [{'parent': '玻璃门', 'names': ['门框', '把手'],
                          'reason': '保留部件关系，不编造独立精确坐标'}]
    output = base / 'complete_objects'
    output.mkdir(parents=True, exist_ok=True)
    evidence_path = output / 'supplemented_evidence.json'
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    render(root / 'backups/semantic_inspection_20260917_195037', evidence_path,
           output / 'supplemented_object_map.png')


if __name__ == '__main__':
    build(Path(__file__).resolve().parents[1])
