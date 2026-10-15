import unittest

from tools.render_inspection_semantic_overview import aggregate, intersects, place_labels


class SemanticOverviewTests(unittest.TestCase):
    def test_dense_station_labels_do_not_overlap(self):
        anchors = [(200, 200)] * 20
        boxes = place_labels(anchors, [(70, 45)] * 20, (0, 0, 600, 600))
        for index, box in enumerate(boxes):
            self.assertGreaterEqual(box[0], 0)
            self.assertLessEqual(box[2], 600)
            self.assertFalse(any(intersects(box, other) for other in boxes[index + 1:]))

    def test_objects_only_results_are_preserved_and_duplicates_removed(self):
        semantics = {'observations': [{'target_id': 'panorama_s02_d060',
            'x_m': 1, 'y_m': 2, 'yaw_deg': 30,
            'result': {'landmarks': [], 'objects': [{'name': 'chair'}, {'name': 'chair'}]}}]}
        card = aggregate(semantics)[0]
        self.assertEqual(card['names'], ['椅子'])
        self.assertEqual(card['station'], 2)
        self.assertTrue(card['names_are_photo_observations_not_object_coordinates'])

    def test_empty_model_result_is_not_reported_as_recognized(self):
        semantics = {'observations': [{'target_id': 'panorama_s01_d000',
            'x_m': 0, 'y_m': 0, 'yaw_deg': 0, 'result': {'error': 'timeout'}}]}
        card = aggregate(semantics)[0]
        self.assertEqual(card['names'], [])
        self.assertEqual(card['error_results'], 1)
        self.assertEqual(card['empty_results'], 1)


if __name__ == '__main__':
    unittest.main()
