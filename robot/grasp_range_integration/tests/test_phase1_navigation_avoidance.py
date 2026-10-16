import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from raspberry_pi.known_map_navigation import KnownOccupancyMap
from grasp_range_integration.phase1.navigation_avoidance import (
    exterior_points, forward_reserve, reserve_path, record_stopped_obstacles,
)


def scan(distance=3.):
    return SimpleNamespace(angles_rad=np.deg2rad(np.arange(-180, 180, 2)),
                           distances_m=np.full(180, distance), received_s=10.)


class NavigationAvoidanceTests(unittest.TestCase):
    def test_short_view_proposal_does_not_snap_or_erase_unknown_glass_or_obstacles(self):
        for kind in ('occupied', 'unknown', 'temporary_occupied', '_phase1_glass_keepout'):
            known = KnownOccupancyMap(np.full((100, 100), 255, dtype=np.uint8), .05, -2.5, -2.5)
            if kind == '_phase1_glass_keepout':
                known._phase1_glass_keepout = np.zeros_like(known.occupied)
            c, r = known.world_to_grid((.4, 0.))
            getattr(known, kind)[r, c] = True
            before = known.traversable.copy()
            self.assertEqual(reserve_path(known, (0., 0.), (.1, 0.), known.path,
                                         allow_short_forward=True), [])
            np.testing.assert_array_equal(before, known.traversable)

    @unittest.skipUnless((Path(__file__).parent / "fixtures").is_dir(), "Requires private recording bundle")
    def test_real_desk_scan_blocks_advance_and_preserves_obstacle(self):
        data = json.loads((Path(__file__).parent/'fixtures/rotation_desk_20260926_180247.json').read_text(encoding='utf-8'))
        frame = data['frame']
        current = SimpleNamespace(angles_rad=np.array(frame['angles_rad'], dtype=float),
                                  distances_m=np.array(frame['distances_m'], dtype=float), received_s=10.)
        self.assertEqual(data['assessment']['reason'], 'exterior_near_obstacle')
        before = current.distances_m.copy()
        value = forward_reserve(current, 400, -228, now=10.7)
        self.assertFalse(value['safe'])
        self.assertEqual(value['reason'], 'forward_turn_space_blocked')
        self.assertGreater(len(value['blocking_points_body_xy_m']), 10)
        np.testing.assert_array_equal(before, current.distances_m)

    def test_reserve_stops_before_existing_rotation_minimum(self):
        current = scan()
        current.distances_m[90] = .80  # .94 m from body origin, not yet close.
        self.assertFalse(forward_reserve(current, 400, now=10.7)['safe'])
        self.assertTrue(forward_reserve(current, 100, now=10.7)['safe'])

    def test_parallel_wall_outside_reserve_is_not_an_angular_sector_veto(self):
        current = scan()
        # Side wall .60 m away: angle width alone is not the criterion.
        current.distances_m[135] = .60
        self.assertTrue(forward_reserve(current, 400, now=10.7)['safe'])
        current.distances_m[135] = .35
        self.assertFalse(forward_reserve(current, 400, now=10.7)['safe'])

    def test_advance_warning_does_not_relabel_rear_echo_as_front_wall(self):
        current = scan()
        current.distances_m[38] = .157  # -104 deg, outside narrow installed edge.
        self.assertTrue(forward_reserve(current, 400, now=10.7)['safe'])
        from grasp_range_integration.phase1.rotation_guard import assess_rotation
        self.assertFalse(assess_rotation(current)['safe'])  # Actual turn still vetoed.

    def test_stale_missing_or_invalid_scan_never_approves_advance(self):
        self.assertFalse(forward_reserve(scan(), 400, now=12.)['safe'])
        current = scan(); current.distances_m[:] = np.nan
        self.assertFalse(forward_reserve(current, 400, now=10.)['safe'])

    def test_reserve_route_goes_around_table_without_mutating_map(self):
        known = KnownOccupancyMap(np.full((120, 120), 255, dtype=np.uint8), .05, -3, -3)
        for x in np.arange(-.3, .31, .05):
            for y in np.arange(-.3, .31, .05):
                c, r = known.world_to_grid(((x, y),))[0]
                known.occupied[r, c] = True
        known._refresh_traversable()
        before = known.traversable.copy()
        path = reserve_path(known, (-2., 0.), (2., 0.), known.path)
        self.assertTrue(path)
        self.assertTrue(any(abs(y) > .70 for x, y in path))
        np.testing.assert_array_equal(before, known.traversable)

    def test_reserve_does_not_snap_a_tight_start_to_invent_escape(self):
        known = KnownOccupancyMap(np.full((100, 100), 255, dtype=np.uint8), .05, -2.5, -2.5)
        c, r = known.world_to_grid(((.3, 0.),))[0]
        known.temporary_occupied[r, c] = True
        before = known.traversable.copy()
        self.assertEqual(reserve_path(known, (0., 0.), (1., 1.), known.path), [])
        np.testing.assert_array_equal(before, known.traversable)

    def test_confirmed_stopped_scan_records_real_right_obstacle_not_blind_arc(self):
        known = KnownOccupancyMap(np.full((100, 100), 255, dtype=np.uint8), .05, -2.5, -2.5)
        current = scan(); current.distances_m[:] = np.nan
        current.distances_m[78:83] = .24
        pose = SimpleNamespace(x_m=0., y_m=0., yaw_rad=0.)
        args = SimpleNamespace(temporary_obstacle_radius=.12, temporary_obstacle_min_hits=2)
        self.assertEqual(record_stopped_obstacles(current, pose, known, args), 0)
        self.assertGreater(record_stopped_obstacles(current, pose, known, args), 0)
        c, r = known.world_to_grid(((-.8, 0.),))[0]
        self.assertFalse(known.temporary_occupied[r, c])


if __name__ == '__main__': unittest.main()
