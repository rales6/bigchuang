import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from tools.render_qwen_candidate_map import render, representative, select_candidates


class QwenCandidateMapTests(unittest.TestCase):
    def item(self, identity='table1', xy=None, **extra):
        return {'id': identity, 'category': 'table', 'map_xy': [2, 2] if xy is None else xy,
                'station_ids': [1], 'evidence_count': 2, 'confidence': .99, **extra}

    def test_semantic_conflict_is_not_plotted_even_if_coordinates_remain(self):
        groups, missing, invalid = select_candidates({'objects': [self.item(semantic_ambiguous=True)]})
        self.assertEqual(groups, {})
        self.assertEqual(len(missing), 1)
        self.assertEqual(invalid, [])

    def test_representative_is_an_existing_candidate_not_mean_world_position(self):
        a, b = self.item(), self.item('table2', [4, 4], station_ids=[1, 2])
        self.assertIs(representative([a, b]), b)

    def test_render_preserves_originals_and_does_not_overlap_text(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            pgm, yaml, memory = folder / 'map.pgm', folder / 'map.yaml', folder / 'memory.json'
            Image.new('L', (100, 100), 255).save(pgm)
            yaml.write_text('resolution: 0.05\norigin: [0,0,0]\n', encoding='utf-8')
            data = {'schema': 'semantic_object_memory.v1', 'map_sha256': hashlib.sha256(pgm.read_bytes()).hexdigest(),
                    'map_yaml_sha256': hashlib.sha256(yaml.read_bytes()).hexdigest(), 'objects': [self.item()]}
            memory.write_text(json.dumps(data), encoding='utf-8')
            before = [path.read_bytes() for path in (pgm, yaml, memory)]
            report = render(pgm, yaml, memory, folder / 'map.png', Path('C:/Windows/Fonts/msyh.ttc'))
            self.assertEqual(report['displayed_candidate_count'], 1)
            self.assertEqual(report['text_overlap_count'], 0)
            self.assertEqual(before, [path.read_bytes() for path in (pgm, yaml, memory)])
            data['map_sha256'] = 'wrong'
            memory.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaises(ValueError):
                render(pgm, yaml, memory, folder / 'wrong.png', Path('C:/Windows/Fonts/msyh.ttc'))


if __name__ == '__main__':
    unittest.main()
