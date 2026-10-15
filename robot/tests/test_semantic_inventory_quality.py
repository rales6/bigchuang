from copy import deepcopy
from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '3090/src'))
from qwen_grounded_tracker.semantic_inventory_quality import filter_observations, remove_ambiguous_world_positions


class SemanticInventoryQualityTests(unittest.TestCase):
    def row(self, category='outlet', confidence=.99, image='a.jpg', identity='one', box=None):
        return {'observation_id': identity, 'name': category, 'category': category, 'confidence': confidence,
                'views': [{'image': image, 'bbox_1000': box or [400, 10, 450, 100]}]}

    def test_zero_confidence_is_not_a_map_object(self):
        rows, report = filter_observations([self.row(confidence=0)])
        self.assertEqual(rows, [])
        self.assertEqual(report['rejected'][0]['reason'], 'low_confidence')

    def test_same_panel_outlet_switch_is_one_ambiguous_region(self):
        raw = [self.row(), self.row('switch', identity='two')]
        original = deepcopy(raw)
        rows, report = filter_observations(raw)
        self.assertEqual(raw, original)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]['semantic_ambiguous'])
        self.assertEqual(rows[0]['alternative_categories'], ['outlet', 'switch'])
        self.assertEqual(report['ambiguous_regions'], 1)

    def test_different_photos_or_regions_do_not_merge(self):
        rows, _ = filter_observations([self.row(), self.row('switch', image='b.jpg', identity='two')])
        self.assertEqual(len(rows), 2)
        rows, _ = filter_observations([self.row(), self.row('switch', identity='two', box=[700, 10, 750, 100])])
        self.assertEqual(len(rows), 2)

    def test_overview_detail_repeat_does_not_add_an_object(self):
        rows, _ = filter_observations([self.row(), self.row(identity='two')])
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0].get('semantic_ambiguous', False))

    def test_conflicted_label_cannot_leave_an_estimated_world_point(self):
        catalog = {'objects': [{'id': 'one', 'aliases': ['outlet'], 'map_xy': [1, 2],
            'evidence': [{'semantic_ambiguous': True, 'alternative_names': ['outlet', 'switch']}]}],
            'search_regions': [{'id': 'one', 'map_xy': [1, 2]}]}
        remove_ambiguous_world_positions(catalog)
        self.assertIsNone(catalog['objects'][0]['map_xy'])
        self.assertFalse(catalog['objects'][0]['localization_allowed'])
        self.assertEqual(catalog['search_regions'], [])

    def test_negative_ground_response_with_residual_boxes_is_ignored(self):
        spec = importlib.util.spec_from_file_location('quality_inventory_cli',
            Path(__file__).resolve().parents[1] / '3090/scripts/run_semantic_object_inventory.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        class Response:
            def raise_for_status(self): pass
            def json(self):
                return {'ok': True, 'found': False, 'candidates': [{'bbox': [1, 1, 20, 20], 'confidence': .99}]}
        class Session:
            def post(self, *args, **kwargs): return Response()
        backend = cli.GroundServiceBackend('http://example.invalid', ['插座'], 1, session=Session())
        with tempfile.TemporaryDirectory() as folder:
            photo = Path(folder) / 'a.jpg'
            Image.new('RGB', (100, 100)).save(photo)
            import json
            self.assertEqual(json.loads(backend.annotate([photo], ''))['objects'], [])

    def test_detail_pass_uses_only_selected_fixture_queries(self):
        spec = importlib.util.spec_from_file_location('detail_inventory_cli',
            Path(__file__).resolve().parents[1] / '3090/scripts/run_semantic_object_inventory.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        class Response:
            def raise_for_status(self): pass
            def json(self): return {'ok': True, 'found': False}
        class Session:
            def __init__(self): self.queries = []
            def post(self, *args, **kwargs):
                self.queries.append(kwargs['data']['instruction'])
                return Response()
        backend = cli.GroundServiceBackend('http://example.invalid', ['桌子', '插座'], 1, ['插座'], session=Session())
        with tempfile.TemporaryDirectory() as folder:
            photo = Path(folder) / 'a.jpg'
            Image.new('RGB', (100, 100)).save(photo)
            backend.annotate([photo], cli.build_object_prompt(1, True))
        self.assertEqual(backend.http_calls, 1)
        self.assertIn('插座', backend.session.queries[0])


if __name__ == '__main__':
    unittest.main()
