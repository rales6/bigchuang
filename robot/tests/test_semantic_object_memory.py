from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '3090/src'))
from qwen_grounded_tracker.semantic_object_catalog import (
    ReadOnlyMap, apply_reviews, build_catalog, normalize_objects, replay_panorama,
)
from qwen_grounded_tracker.semantic_task_planner import SemanticTaskPlanner, summarize_semantic_map
from qwen_grounded_tracker.search_task_planner import find_landmark
from raspberry_pi.visual_place_localization import load_permanent_landmarks

spec = importlib.util.spec_from_file_location('inventory_cli', ROOT / '3090/scripts/run_semantic_object_inventory.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class SemanticObjectMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.maps = self.root / 'maps'
        self.maps.mkdir()
        gray = np.full((100, 100), 255, dtype=np.uint8)
        gray[:, 70] = 0
        Image.fromarray(gray).save(self.maps / 'room.pgm')
        (self.maps / 'room.yaml').write_text('resolution: 0.1\norigin: [0, 0, 0]\n', encoding='utf-8')
        self.map = ReadOnlyMap(self.maps / 'room.pgm', self.maps / 'room.yaml')
        self.record = {'target_id': 'panorama_s01_d000', 'image': 'a.jpg', 'camera_2_image': 'b.jpg',
            'panorama_station_id': 1, 'x_m': 2, 'y_m': 5, 'yaw_deg': 0}
        self.payload = {'objects': [{'name_zh': '插座', 'category': 'outlet', 'confidence': .95,
            'mobility': 'fixed', 'views': [{'image_index': 1, 'bbox_2d': [450, 20, 550, 100]}]}]}

    def observations(self, payload=None):
        return normalize_objects(payload or self.payload, ['a.jpg', 'b.jpg'], self.record)

    def catalog(self):
        return build_catalog(self.observations(), self.map, 'manifest.json')

    def review(self, catalog, **extra):
        return {'map_sha256': self.map.sha256, 'objects': [{'id': catalog['objects'][0]['id'],
            'reviewer': 'test', 'reviewed_at': '2026-09-17', 'verified_map_xy': [7, 5],
            'verified_approach_xy': [6, 5], 'mobility': 'fixed',
            'localization_allowed': True, 'navigation_allowed': True, **extra}]}

    def test_crop_boxes_map_back_to_original_image(self):
        observations = normalize_objects(self.payload, ['a.jpg'], self.record, .45)
        self.assertEqual(observations[0]['views'][0]['bbox_1000'], [450, 9, 550, 45])

    def test_invalid_numeric_boxes_are_not_accepted(self):
        payload = deepcopy(self.payload)
        payload['objects'][0]['confidence'] = float('nan')
        payload['objects'][0]['views'][0]['bbox_2d'] = [0, 0, float('inf'), 100]
        observations = self.observations(payload)
        self.assertEqual(observations[0]['confidence'], 0)
        self.assertEqual(observations[0]['views'], [])

    def test_high_qwen_confidence_and_repeated_frames_cannot_enable_navigation(self):
        observations = self.observations()
        catalog = build_catalog(observations * 5, self.map, 'manifest.json')
        item = catalog['objects'][0]
        self.assertEqual(item['evidence_count'], 5)
        self.assertFalse(item['navigation_allowed'])
        self.assertFalse(item['localization_allowed'])
        self.assertFalse(item['grasp_allowed'])
        self.assertEqual(catalog['entities'], {})

    def test_missing_boxes_still_preserve_names_without_fake_location(self):
        payload = deepcopy(self.payload)
        payload['objects'][0]['views'] = []
        catalog = build_catalog(self.observations(payload), self.map, 'manifest.json')
        self.assertIsNone(catalog['objects'][0]['map_xy'])
        self.assertEqual(catalog['objects'][0]['position_status'], 'unlocalized')

    def test_review_enables_only_verified_fixture_and_free_approach(self):
        catalog = self.catalog()
        apply_reviews(catalog, self.review(catalog), self.map)
        self.assertTrue(catalog['objects'][0]['localization_allowed'])
        self.assertTrue(catalog['objects'][0]['navigation_allowed'])
        self.assertFalse(catalog['objects'][0]['grasp_allowed'])
        path = self.root / 'memory.json'
        path.write_text(json.dumps(catalog), encoding='utf-8')
        landmarks = load_permanent_landmarks(path)
        self.assertEqual(len(landmarks), 1)
        self.assertIn('outlet', landmarks[0].labels)

    def test_review_rejects_wrong_map_and_occupied_approach(self):
        catalog = self.catalog()
        review = self.review(catalog)
        review['map_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            apply_reviews(catalog, review, self.map)
        with self.assertRaisesRegex(ValueError, 'not an old-map free'):
            apply_reviews(catalog, self.review(catalog, verified_approach_xy=[7, 5]), self.map)

    def test_behind_glass_cannot_be_promoted_for_navigation_or_localization(self):
        catalog = self.catalog()
        catalog['objects'][0]['behind_glass'] = True
        apply_reviews(catalog, self.review(catalog), self.map)
        self.assertFalse(catalog['objects'][0]['navigation_allowed'])
        self.assertFalse(catalog['objects'][0]['localization_allowed'])

    def test_future_run_preserves_unseen_memory_as_recheck_not_deletion(self):
        previous = self.catalog()
        apply_reviews(previous, self.review(previous), self.map)
        current = build_catalog([], self.map, 'new_manifest.json', previous)
        self.assertEqual(current['objects'][0]['id'], previous['objects'][0]['id'])
        self.assertTrue(current['objects'][0]['unobserved_in_this_run'])
        self.assertEqual(current['suspicious_locations'][0]['last_change_type'], 'unobserved_not_proven_absent')
        self.assertEqual(len(current['entities']), 1)

    def test_task_planner_keeps_coarse_coordinates_as_search_prior(self):
        catalog = self.catalog()
        plan = SemanticTaskPlanner().plan('去插座旁边', catalog)
        target = plan.steps[0].target
        self.assertIsNone(target['landmark_id'])
        self.assertIsNone(target['expected_map_xy'])
        self.assertTrue(target['require_scene_recheck'])
        self.assertTrue(target['requires_verified_approach'])
        self.assertIn('position_uncertainty_m', summarize_semantic_map(catalog)['objects'][0])
        self.assertIsNone(find_landmark(catalog, '插座'))

    def test_localizer_rejects_unverified_new_schema_even_if_entity_is_present(self):
        catalog = self.catalog()
        catalog['entities']['bad'] = {'permanent': True, 'semantic_label': 'outlet',
            'target_x_m': 7, 'target_y_m': 5, 'position_verified': False,
            'localization_allowed': True}
        path = self.root / 'memory.json'
        path.write_text(json.dumps(catalog), encoding='utf-8')
        self.assertEqual(load_permanent_landmarks(path), [])

    def test_replay_preserves_objects_only_and_deduplicates_landmark_copy(self):
        inference = {'objects': [{'name': '插座', 'camera_bbox': [100, 10, 200, 50],
            'confidence': .9, 'stereo_consistency': 'right'}]}
        payload = {'observations': [{'target_id': self.record['target_id'], 'result': inference}]}
        observations = replay_panorama(payload, [self.record])
        self.assertEqual(len(observations), 1)
        self.assertTrue(observations[0]['camera_reference_uncertain'])
        self.assertEqual(observations[0]['views'][0]['image_index'], 2)

    def test_qwen_pipeline_checkpoint_resume_and_source_hash_invalidation(self):
        photos = self.root / 'photos'
        photos.mkdir()
        for name in ['a.jpg', 'b.jpg']:
            Image.new('RGB', (80, 60), 'white').save(photos / name)
        (photos / 'inspection_manifest.json').write_text(json.dumps({'captures': [self.record]}), encoding='utf-8')
        args = cli.parser().parse_args(['--inspection', str(photos), '--map', str(self.maps / 'room.pgm'),
            '--map-yaml', str(self.maps / 'room.yaml'), '--output', str(self.root / 'output'),
            '--model-path', 'fake_model'])
        class Backend:
            def annotate(inner, *_args):
                return json.dumps(self.payload)
        first = cli.run(args, Backend())
        self.assertEqual(first['new_backend_calls'], 2)
        second = cli.run(args, Backend())
        self.assertEqual(second['new_backend_calls'], 0)
        self.assertEqual(second['cache_hits'], 2)
        Image.new('RGB', (80, 60), 'black').save(photos / 'a.jpg')
        third = cli.run(args, Backend())
        self.assertEqual(third['new_backend_calls'], 2)


if __name__ == '__main__':
    unittest.main()
