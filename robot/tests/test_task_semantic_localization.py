from copy import deepcopy
import math
import unittest

from raspberry_pi.task_semantic_localization import (
    CameraCalibration, PoseCandidate, SemanticDetection, TaskPhotoSchedule,
    TaskPose, TaskSemanticLocalizer, detections_from_qwen, wrap,
)


class TaskSemanticLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.memory = {'schema': 'semantic_object_memory.v1', 'map_sha256': 'fixture',
            'objects': [{'id': name, 'category': name, 'map_xy': xy,
                         'position_verified': True, 'localization_allowed': True,
                         'mobility': 'fixed', 'behind_glass': False}
                        for name, xy in [('sign', [3, 2]), ('outlet', [4, -2]), ('window', [1, 4])]]}
        self.pose = TaskPose(0, 0, 20)

    def detections(self, actual=TaskPose(0, 0, 0)):
        return [SemanticDetection(a['category'], wrap(math.degrees(math.atan2(
            a['map_xy'][1] - actual.y_m, a['map_xy'][0] - actual.x_m)) - actual.yaw_deg),
            .95, bearing_calibrated=True) for a in self.memory['objects']]

    def evaluate(self, localizer, frame=1, **kwargs):
        return localizer.evaluate(frame_id=str(frame), captured_at_s=float(frame), now_s=float(frame),
            capture_pose=self.pose, current_pose=kwargs.pop('current_pose', self.pose),
            detections=kwargs.pop('detections', self.detections()), candidates=kwargs.pop('candidates', []), **kwargs)

    def localizer(self, **kwargs):
        return TaskSemanticLocalizer(self.memory, 'fixture', **kwargs)

    def test_yaw_sign_and_bounded_two_frame_correction(self):
        loc = self.localizer()
        self.assertFalse(self.evaluate(loc)['accepted'])
        result = self.evaluate(loc, 2)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['pose']['yaw_deg'], 12)
        self.assertEqual(result['pose']['x_m'], 0)
        self.assertIsNone(result['motor_command'])

    def test_duplicate_frame_cannot_confirm(self):
        loc = self.localizer()
        self.evaluate(loc)
        self.assertEqual(self.evaluate(loc)['reason'], 'duplicate_or_out_of_order_frame')
        self.assertTrue(self.evaluate(loc, 2)['accepted'])

    def test_wrong_map_rejected(self):
        with self.assertRaises(ValueError):
            TaskSemanticLocalizer(self.memory, 'another_map')

    def test_approximate_mobile_glass_anchors_are_not_used(self):
        for field, value in [('position_verified', False), ('mobility', 'movable'), ('behind_glass', True)]:
            memory = deepcopy(self.memory)
            for a in memory['objects']:
                a[field] = value
            result = self.evaluate(TaskSemanticLocalizer(memory, 'fixture'))
            self.assertEqual(result['reason'], 'insufficient_verified_calibrated_anchors')

    def test_repeated_class_requires_reference_identity(self):
        memory = deepcopy(self.memory)
        memory['objects'][1]['category'] = 'sign'
        memory['objects'][2]['category'] = 'sign'
        loc = TaskSemanticLocalizer(memory, 'fixture')
        detections = [SemanticDetection('sign', d.bearing_deg, .99, bearing_calibrated=True) for d in self.detections()]
        self.assertFalse(self.evaluate(loc, detections=detections)['accepted'])
        detections = [SemanticDetection('sign', d.bearing_deg, .99, a['id'], True)
                      for d, a in zip(self.detections(), memory['objects'])]
        self.evaluate(loc, 2, detections=detections)
        self.assertTrue(self.evaluate(loc, 3, detections=detections)['accepted'])

    def test_one_anchor_repeated_is_not_multiple_evidence(self):
        d = self.detections()[0]
        self.assertFalse(self.evaluate(self.localizer(), detections=[d] * 6)['accepted'])

    def test_outlier_or_opposite_heading_does_not_snap(self):
        detections = self.detections()
        detections[0] = SemanticDetection('sign', 130, .99, bearing_calibrated=True)
        self.assertFalse(self.evaluate(self.localizer(), detections=detections)['accepted'])
        self.assertFalse(self.evaluate(self.localizer(), detections=self.detections(TaskPose(0, 0, 180)))['accepted'])

    def test_stale_or_moving_camera_frame_is_rejected(self):
        loc = self.localizer()
        result = loc.evaluate(frame_id='old', captured_at_s=1, now_s=50,
            capture_pose=self.pose, current_pose=self.pose, detections=self.detections(), candidates=[])
        self.assertEqual(result['reason'], 'stale_photo')
        self.assertEqual(self.evaluate(self.localizer(), current_pose=TaskPose(.2, 0, 20))['reason'],
                         'vehicle_moved_since_capture')

    def test_translation_requires_radar_and_swept_footprint_check(self):
        actual = TaskPose(.55, .2, 0)
        candidates = [PoseCandidate(actual, True, 'radar')]
        detections = self.detections(actual)
        loc = self.localizer(pose_validator=lambda previous, proposed: True)
        self.evaluate(loc, detections=detections, candidates=candidates)
        result = self.evaluate(loc, 2, detections=detections, candidates=candidates)
        self.assertTrue(result['accepted'])
        self.assertAlmostEqual(math.hypot(result['pose']['x_m'], result['pose']['y_m']), .15)
        blocked = self.localizer(pose_validator=lambda previous, proposed: False)
        first = self.evaluate(blocked, detections=detections, candidates=candidates)
        self.assertFalse(any(r['source'] == 'radar' for r in first['rankings']))
        unvalidated = self.evaluate(self.localizer(), detections=detections, candidates=candidates)
        self.assertFalse(any(r['source'] == 'radar' for r in unvalidated['rankings']))

    def test_far_symmetric_candidate_is_not_used(self):
        result = self.evaluate(self.localizer(), candidates=[PoseCandidate(TaskPose(3, 0, 20), True, 'far')])
        self.assertFalse(any(r['source'] == 'far' for r in result['rankings']))

    def test_yaw_wraparound(self):
        self.pose = TaskPose(0, 0, 179)
        loc = self.localizer()
        detections = self.detections(TaskPose(0, 0, -179))
        self.evaluate(loc, detections=detections)
        result = self.evaluate(loc, 2, detections=detections)
        self.assertAlmostEqual(result['pose']['yaw_deg'], -179)

    def test_camera_mount_translation_is_compensated(self):
        detections = [SemanticDetection(a['category'], math.degrees(math.atan2(
            a['map_xy'][1] - .1, a['map_xy'][0] - .2)), .95,
            bearing_calibrated=True, camera_x_m=.2, camera_y_m=.1)
            for a in self.memory['objects']]
        loc = self.localizer()
        self.evaluate(loc, detections=detections)
        result = self.evaluate(loc, 2, detections=detections)
        self.assertTrue(result['accepted'])
        self.assertAlmostEqual(result['rankings'][0]['pose']['yaw_deg'], 0, places=4)

    def test_no_visual_result_never_generates_spin_or_throws_after_failures(self):
        loc = self.localizer()
        for frame in range(1, 30):
            result = self.evaluate(loc, frame, detections=[])
            self.assertFalse(result['accepted'])
            self.assertIsNone(result['motor_command'])

    def test_bearing_conversion_and_unverified_camera(self):
        rows = [{'category': 'sign', 'confidence': .9, 'object_id': 'invented',
                 'views': [{'image_index': 1, 'bbox_1000': [700, 100, 800, 200]}]}]
        unverified = CameraCalibration(1000, 500, 500)
        self.assertEqual(detections_from_qwen(rows, {1: unverified}), [])
        verified = CameraCalibration(1000, 500, 500, 10, True)
        detection = detections_from_qwen(rows, {1: verified})[0]
        self.assertAlmostEqual(detection.bearing_deg, 10 - math.degrees(math.atan(.5)))
        self.assertIsNone(detection.object_id)

    def test_schedule_is_stationary_and_cooldown_limited(self):
        schedule = TaskPhotoSchedule()
        self.assertFalse(schedule.should_capture(0, 0, stationary=False))
        self.assertTrue(schedule.should_capture(0, 0, stationary=True))
        schedule.mark_requested(0, 0)
        self.assertFalse(schedule.should_capture(5, 2, stationary=True, event='arrival'))
        self.assertTrue(schedule.should_capture(11, 2, stationary=True))
        self.assertTrue(schedule.should_capture(46, .5, stationary=True))

    def test_nonfinite_values_rejected(self):
        with self.assertRaises(ValueError):
            TaskPose(float('nan'), 0, 0)
        with self.assertRaises(ValueError):
            SemanticDetection('sign', 0, float('inf'))


if __name__ == '__main__':
    unittest.main()
