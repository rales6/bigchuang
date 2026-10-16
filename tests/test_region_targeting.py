"""Reproduce yellow-cabinet misselection and effective-turn output regressions."""
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from services.track.final_work.pi.region_navigation import entities_for, select_region, anchor_seen, install_region_skill, save_json, target_view_evidence, matching_anchors, arrival_proximity
from grasp_range_integration.phase1.adaptive_turn import AdaptiveTurnResponse
from grasp_range_integration.phase1.visibility_route_planner import NoReachableVisibilityError, POLICY
from grasp_range_integration.phase1.startup_route_recovery import wrap_startup_route_recovery


ROOT = Path(__file__).resolve().parents[1]


def public_sample_semantic():
    # Invented geometry and IDs: exercise filtering without a real room map.
    return {'objects': [
        {'id': 'sample_indoor_cabinet', 'name': '沿墙柜子', 'category': 'cabinet',
         'map_xy': [2., 1.], 'search_region_allowed': True},
        {'id': 'sample_blue_cabinet', 'name': '蓝色柜子', 'category': 'cabinet',
         'map_xy': [4., 3.], 'behind_glass': True, 'search_region_allowed': False},
        {'id': 'sample_sign', 'name': '柜子标牌', 'category': 'sign', 'map_xy': [2., 1.]}]}


class RegionTargetTests(unittest.TestCase):
    def test_actual_release_yellow_query_never_selects_blue_or_cabinet_marker(self):
        semantic = public_sample_semantic()
        rows = entities_for(semantic, {'query': '移动黄色柜子附近'})
        self.assertEqual([row['id'] for row in rows], ['sample_indoor_cabinet'])
        self.assertEqual(rows[0]['requested_color'], 'yellow')
        self.assertTrue(rows[0]['requires_color_recheck'])  # map itself has no yellow annotation
        self.assertEqual(rows[0]['color_in_map'], [])

    def test_explicit_blue_id_cannot_override_glass_or_search_prohibition(self):
        semantic = public_sample_semantic()
        with self.assertRaises(ValueError):
            entities_for(semantic, {'query': '黄色柜子', 'landmark_id': 'sample_blue_cabinet'})

    def test_explicit_color_beats_an_uncolored_candidate(self):
        semantic = {'objects': [
            {'id': 'unknown', 'name': '沿墙柜子', 'category': 'cabinet', 'map_xy': [0, 0]},
            {'id': 'yellow', 'name': '黄色柜子', 'category': 'cabinet', 'map_xy': [1, 0]},
            {'id': 'green', 'name': '绿色柜子', 'category': 'cabinet', 'map_xy': [2, 0]}]}
        self.assertEqual([r['id'] for r in entities_for(semantic, {'query': '黄色柜子'})], ['yellow'])

    def test_multiple_uncolored_entities_do_not_silently_drop_color(self):
        semantic = {'objects': [{'id': str(i), 'name': '柜子', 'category': 'cabinet',
                                 'map_xy': [i, 0]} for i in (1, 2)]}
        with self.assertRaisesRegex(ValueError, '颜色'):
            entities_for(semantic, {'query': '黄色柜子'})

    def test_color_alias_is_respected_without_renaming_the_map(self):
        semantic = {'objects': [{'id': 'cabinet', 'name': '柜子', 'aliases': ['yellow cabinet'],
                                'category': 'cabinet', 'map_xy': [1, 2]}]}
        row = entities_for(semantic, {'query': '黄色柜子'})[0]
        self.assertFalse(row['requires_color_recheck'])

    def known(self, path):
        class Matrix:
            def __getitem__(self, index): return True
        return NS(world_to_grid=lambda points: [(0, 0)], in_bounds=lambda *args: True,
                  traversable=Matrix(), path=path)

    def test_no_station_raises_recognized_recovery_type(self):
        known = self.known(lambda start, goal: [])
        with self.assertRaises(NoReachableVisibilityError):
            select_region(known, (0, 0), [{'id': 'cabinet', 'center_xy_m': [2, 2]}])

    def test_expanded_station_search_only_when_original_standoffs_are_unreachable(self):
        def path(start, goal):
            return [start, goal] if math.dist(goal, [2, 2]) > 1.4 else []
        row = select_region(self.known(path), (0, 0), [{'id': 'cabinet', 'center_xy_m': [2, 2]}])
        self.assertTrue(row['expanded_station_search'])
        self.assertGreater(row['standoff_m'], 1.4)
        self.assertLessEqual(row['standoff_m'], 1.8)

    def test_original_standoffs_stay_preferred_when_valid(self):
        row = select_region(self.known(lambda start, goal: [start, goal]),
                            (0, 0), [{'id': 'cabinet', 'center_xy_m': [2, 2]}])
        self.assertFalse(row['expanded_station_search'])
        self.assertLessEqual(row['standoff_m'], 1.30000001)

    def test_real_station_error_reaches_existing_startup_recovery(self):
        import tempfile
        from raspberry_pi.mapping import Pose2D
        mission = NS(_phase1_motion_fault=None)
        known = self.known(lambda start, goal: [])
        mission.build_search_route = lambda *args: select_region(known, (0, 0),
            [{'id': 'cabinet', 'center_xy_m': [2, 2]}])
        pose, scan = Pose2D(), NS(sequence=1)
        client = NS(stop=Mock(), startup_recovery_pulse=Mock())
        controller = NS(stopped=Mock())
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'decision.json'
            path.write_text(json.dumps({'route_request': {'planning_policy': POLICY,
                            'mode': 'semantic_search_region'}}), encoding='utf-8')
            adapted = wrap_startup_route_recovery(mission, lambda *a, **k: (pose, scan))
            recovery_path = 'grasp_range_integration.phase1.startup_route_recovery.recover_startup'
            # Entering the existing recovery (not escaping via RuntimeError)
            # is the regression under test; no actual motion is authorised.
            with patch(recovery_path, side_effect=LookupError('entered existing recovery')) as recover:
                with self.assertRaisesRegex(LookupError, 'entered existing recovery'):
                    adapted(NS(decision=path), client, iter([]), NS(map=object()), controller, lambda: False)
            recover.assert_called_once()
            client.startup_recovery_pulse.assert_not_called()

    def test_source_copies_match(self):
        self.assertEqual((ROOT / 'services/track/final_work/pi/region_navigation.py').read_bytes(),
            (ROOT / 'robot/grasp_range_integration/track_web_integration/pi/region_navigation.py').read_bytes())

    def test_color_is_evidence_not_just_a_requested_prompt(self):
        item = {'class_name': 'cabinet', 'camera_id': 'right', 'confidence': .85,
                'bbox_1000': [50, 100, 600, 800]}
        self.assertTrue(anchor_seen([item], 'cabinet'))
        self.assertFalse(anchor_seen([item], 'cabinet', 'yellow'))
        self.assertFalse(anchor_seen([{**item, 'attributes': {'color': 'blue'}}], 'cabinet', 'yellow'))
        self.assertTrue(anchor_seen([{**item, 'attributes': {'color': '黄色'}}], 'cabinet', 'yellow'))
        self.assertTrue(anchor_seen([{**item, 'appearance_key': 'yellow_wood_large'}], 'cabinet', 'yellow'))

    def test_actual_white_door_with_yellow_label_does_not_confirm_yellow_cabinet(self):
        item = {'class_name': 'cabinet', 'camera_id': 'left', 'confidence': .85,
                'bbox_1000': [874, 0, 999, 194], 'appearance_key': 'white_metal_large',
                'attributes': {'color': 'white', 'description': '一个白色的门，门框为金属材质，门上有一个黄色的标签。'}}
        self.assertFalse(anchor_seen([item], 'cabinet', 'yellow'))
        # Even a mistaken yellow body-colour field cannot override an
        # explicit door subject masquerading as class_name=cabinet.
        item['attributes']['color'] = 'yellow'
        self.assertFalse(anchor_seen([item], 'cabinet', 'yellow'))
        item['attributes']['description'] = '黄色柜子，有两个柜门和白色标签。'
        self.assertTrue(anchor_seen([item], 'cabinet', 'yellow'))

    def test_body_colour_overrides_fingerprint_and_whole_frame_is_not_an_object_box(self):
        item = {'class_name': 'cabinet', 'camera_id': 'right', 'confidence': .85,
                'bbox_1000': [100, 100, 600, 600], 'appearance_key': 'yellow_wood_large',
                'attributes': {'color': 'white'}}
        self.assertFalse(anchor_seen([item], 'cabinet', 'yellow'))
        item['attributes']['color'] = 'yellow'; item['bbox_1000'] = [0, 0, 1000, 1000]
        self.assertFalse(anchor_seen([item], 'cabinet', 'yellow'))

    def test_bound_yellow_cabinet_accepts_replayed_warm_body_colours_only(self):
        item = {'class_name': 'cabinet', 'camera_id': 'left', 'confidence': .92,
                'bbox_1000': [0, 0, 850, 240]}
        for color in ('beige', 'light_brown', '米色', '浅棕色', 'wood'):
            value = {**item, 'attributes': {'color': color}}
            self.assertFalse(matching_anchors([value], 'cabinet', 'yellow'))
            self.assertTrue(matching_anchors([value], 'cabinet', 'yellow', allow_warm_cabinet=True))
            self.assertFalse(matching_anchors([{**value, 'class_name': 'block'}], 'block', 'yellow',
                                             allow_warm_cabinet=True))
        for color in ('white', 'blue', 'green', 'red', 'gray', 'white_beige', ''):
            self.assertFalse(matching_anchors([{**item, 'attributes': {'color': color}}],
                                             'cabinet', 'yellow', allow_warm_cabinet=True))
        door = {**item, 'attributes': {'color': 'beige', 'description': '一扇门，上面有黄色标签。'}}
        self.assertFalse(matching_anchors([door], 'cabinet', 'yellow', allow_warm_cabinet=True))

    def test_arrival_neighbourhood_not_exact_station_or_remote_camera_sighting(self):
        target = {'id': 'cabinet', 'center_xy_m': [2, 0], 'approach_xy_m': [1, 0]}
        self.assertTrue(arrival_proximity([.6, .5, 0], target))
        self.assertFalse(arrival_proximity([-1, 0, 0], target))
        self.assertFalse(arrival_proximity([], target))

    def test_rear_map_target_cannot_be_confirmed_by_unrelated_front_cabinet(self):
        target = {'id': 'cabinet', 'center_xy_m': [2, 0], 'position_uncertainty_m': .85}
        self.assertTrue(target_view_evidence([0, 0, 0], target)['eligible'])
        self.assertFalse(target_view_evidence([0, 0, math.pi], target)['eligible'])
        self.assertFalse(target_view_evidence([], target)['eligible'])

    def test_reproduced_empty_gripper_cannot_report_arrived_at_glass_photo(self):
        import tempfile
        from grasp_range_integration.phase1 import visibility_route_planner as visibility
        reply = {'detections': []}
        contexts = []
        class GPU:
            def perceive(self, left, right, context):
                contexts.append(context)
                return reply
        mission = NS(N10LidarDriver=type('Driver', (), {}), GpuMissionClient=GPU,
                     build_search_route=Mock(), _strict_command_for_pose=Mock(return_value=(300, 0, False)))
        original_command = mission._strict_command_for_pose
        with tempfile.TemporaryDirectory() as root, patch.object(visibility, 'REGION_ROUTE_BUILDER', None, create=True):
            path = Path(root) / 'job.json'
            save_json(path, {'schema': 'robot_region_job.v1', 'carrying': False,
                'target': {'query': '黄色柜子附近'}, 'candidates': [{
                'id': 'cabinet', 'name': '沿墙柜子', 'class_name': 'cabinet',
                'center_xy_m': [2, 2], 'requested_color': 'yellow'}]})
            install_region_skill(mission, NS(max_replans=2), path)
            gpu = mission.GpuMissionClient()
            # Startup/permanent-landmark calls must not dereference selected=None
            # or have their empty target overwritten with the destination.
            localization = {'target': {}, 'search_phase': 'localization'}
            gpu.perceive('initial_left', 'initial_right', localization)
            self.assertEqual(contexts[-1], localization)
            route = visibility.REGION_ROUTE_BUILDER(self.known(lambda a, b: [a, b]), (0, 0), {})
            goal = route['views'][0]['view_xy_m']
            from raspberry_pi.mapping import Pose2D
            self.assertEqual(mission._strict_command_for_pose(NS(), Pose2D(0, 0, 0), goal), (300, 0, False))
            original_command.assert_called_once()
            saved = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(saved['route_history'][0]['map_entity_id'], 'cabinet')
            self.assertEqual(saved['execution_trace'][0]['waypoint_xy_m'], goal)
            for detections in ([], [{'class_name': 'cabinet', 'camera_id': 'left', 'confidence': .85,
                                    'bbox_1000': [0, 100, 600, 600], 'attributes': {'color': 'blue'}}]):
                reply['detections'] = detections
                gpu.perceive('glass_left', 'glass_right', {'robot_pose': [*goal, 0]})
                directive = gpu.next({'remaining_view_indices': [1, 2]})
                self.assertFalse(directive['terminal'])
                self.assertEqual(directive['next_view_index'], 1)
                self.assertNotEqual(json.loads(path.read_text(encoding='utf-8'))['status'], 'arrived')
            retry = gpu.next({'remaining_view_indices': []})
            self.assertEqual(retry['action'], 'replan')
            reply['detections'][0]['attributes']['color'] = 'yellow'
            gpu.perceive('yellow_left', 'yellow_right', {'robot_pose': [*goal, 0]})
            self.assertEqual(gpu.next({})['action'], 'report_success')
            arrival = json.loads(path.read_text(encoding='utf-8'))['arrival']
            self.assertEqual(arrival['requested_color'], 'yellow')
            self.assertEqual(arrival['arrival_evidence'], 'neighbourhood_and_single_camera_target')

    def test_final_approach_capture_does_not_skip_transit_or_other_sweep_angles(self):
        import tempfile
        from grasp_range_integration.phase1 import visibility_route_planner as visibility
        from raspberry_pi.mapping import Pose2D
        perception = Mock(return_value={'detections': []})
        mission = NS(N10LidarDriver=type('Driver', (), {}),
                     GpuMissionClient=type('GPU', (), {'perceive': perception}),
                     _strict_command_for_pose=Mock(return_value=(0, 1800, False)))
        original = mission._strict_command_for_pose
        with tempfile.TemporaryDirectory() as root, patch.object(visibility, 'REGION_ROUTE_BUILDER', None, create=True):
            path = Path(root) / 'job.json'
            save_json(path, {'schema': 'robot_region_job.v1', 'target': {'query': '黄色柜子附近'},
                'candidates': [{'id': 'cabinet', 'class_name': 'cabinet', 'name': '沿墙柜子',
                                'center_xy_m': [2, 0], 'requires_color_recheck': True,
                                'requested_color': 'yellow'}]})
            install_region_skill(mission, NS(max_replans=2), path)
            route = visibility.REGION_ROUTE_BUILDER(self.known(lambda a, b: [a, b]), (0, 0), {})
            xy = route['views'][0]['view_xy_m']
            pose = Pose2D(xy[0]-.30, xy[1], math.radians(20))
            command = mission._strict_command_for_pose(NS(), pose, xy)
            self.assertEqual(command, (0, 0, True))
            original.assert_not_called()
            self.assertEqual(mission._strict_command_for_pose(NS(), pose, (-1, -1)), (0, 1800, False))
            original.assert_called_once()
            visibility.ACTIVE_VIEW = route['views'][1]
            self.assertEqual(mission._strict_command_for_pose(NS(), pose, xy), (0, 1800, False))
            gpu = mission.GpuMissionClient()
            # Near the destination but >0.45m away from the exact station:
            # a genuine warm cabinet body now completes without extra laps.
            perception.return_value = {'detections': [{'class_name': 'cabinet', 'camera_id': 'left', 'confidence': .92,
                      'bbox_1000': [0, 0, 850, 240], 'attributes': {'color': 'light_brown'}}]}
            gpu.perceive('left', 'right', {'robot_pose': [xy[0]-.3, xy[1]+.4, 0]})
            self.assertEqual(gpu.next({})['action'], 'report_success')


class ProvenTurnTests(unittest.TestCase):
    def setUp(self):
        self.model = AdaptiveTurnResponse()
        self.error = math.radians(60)

    def proven_start(self):
        speed, ttl, stage, _ = self.model.plan(self.error, 2600, 650)
        for _ in range(2):
            self.model.feedback(speed, ttl, 0., .001, True, self.error)
            speed, ttl, stage, _ = self.model.plan(self.error, 2600, 650)
        self.model.feedback(speed, ttl, math.radians(3), .003, True, self.error)
        return speed, ttl, stage

    def test_success_does_not_drop_2300_236ms_back_to_1800_116ms(self):
        speed, ttl, stage = self.proven_start()
        self.assertEqual((speed, ttl), (2300, 236))
        next_speed, next_ttl, next_stage, _ = self.model.plan(self.error, 2600, 650)
        self.assertEqual((next_speed, next_ttl), (speed, ttl))
        self.assertGreaterEqual(next_stage, stage)

    def test_a_new_stall_grows_from_proven_command_not_floor(self):
        speed, ttl, _ = self.proven_start()
        self.model.feedback(speed, ttl, .0001, .001, True, self.error)
        next_speed, next_ttl, _, _ = self.model.plan(self.error, 2600, 650)
        self.assertGreater(next_speed, speed)
        self.assertGreaterEqual(next_ttl, ttl)
        self.assertLessEqual(next_ttl, 250)

    def test_opposite_direction_does_not_inherit_proven_output(self):
        self.proven_start()
        speed, ttl, _, _ = self.model.plan(-self.error, -1200, 650)
        self.assertEqual(speed, -1200)

    def test_small_angle_and_reverse_correction_do_not_latch_large_start(self):
        self.proven_start()
        speed, ttl, _, _ = self.model.plan(math.radians(6), 1200, 650)
        self.assertEqual(speed, 1200)
        self.model.reverse_correction()
        speed, ttl, _, _ = self.model.plan(-self.error, -1200, 650)
        self.assertLess(abs(speed), 1200)

    def test_fast_response_shrinks_duration_and_preserves_parent_budget(self):
        self.proven_start()
        self.model.feedback(2300, 236, math.radians(20), .001, True, self.error)
        speed, ttl, _, _ = self.model.plan(math.radians(15), 1200, 100)
        self.assertEqual(speed, 2300)
        self.assertLess(ttl, 100)


if __name__ == '__main__':
    unittest.main()
