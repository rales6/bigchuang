import math
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from raspberry_pi.lidar import LaserScan  # noqa: E402
from raspberry_pi.config import LidarConfig, MappingConfig  # noqa: E402
from raspberry_pi.mapping import LidarSlam, OccupancyGrid, Pose2D  # noqa: E402
from raspberry_pi.mapping.map_visualization import (  # noqa: E402
    END_COLOR,
    START_COLOR,
    TRAJECTORY_COLOR,
    build_trajectory_rgb,
    save_trajectory_png,
)
from raspberry_pi.mapping.scan_matcher import IcpScanMatcher  # noqa: E402
from raspberry_pi.mapping.scan_matcher import MatchResult  # noqa: E402
from raspberry_pi.mapping.scan_filter import filter_scan_sector  # noqa: E402


class PiMappingTest(unittest.TestCase):
    def test_pose_compose_and_inverse(self):
        pose = Pose2D(1.0, -2.0, 0.4)
        points = np.array(((0.2, 0.3), (-1.0, 2.0)))
        restored = pose.inverse().transform_points(pose.transform_points(points))
        np.testing.assert_allclose(points, restored, atol=1e-9)

    def test_icp_recovers_current_pose_in_reference_frame(self):
        angles = np.linspace(-1.4, 1.4, 120)
        ranges = 2.0 + 0.35 * np.sin(angles * 2.7) + 0.15 * np.cos(angles * 5.1)
        reference = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))
        expected = Pose2D(0.08, -0.04, math.radians(2.0))
        # expected 把 current 变到 reference，因此用逆变换合成当前扫描。
        current = expected.inverse().transform_points(reference)
        result = IcpScanMatcher(max_correspondence_m=0.3, min_points=30).match(
            reference, current
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(expected.x_m, result.transform.x_m, delta=0.015)
        self.assertAlmostEqual(expected.y_m, result.transform.y_m, delta=0.015)
        self.assertAlmostEqual(expected.yaw_rad, result.transform.yaw_rad, delta=0.01)

    def test_icp_coarse_search_recovers_large_rotation(self):
        angles = np.linspace(-1.35, 1.35, 180)
        ranges = (
            2.4
            + 0.40 * np.sin(angles * 2.3)
            + 0.22 * np.cos(angles * 5.7)
            + 0.08 * np.sin(angles * 11.0)
        )
        reference = np.column_stack((
            ranges * np.cos(angles),
            ranges * np.sin(angles),
        ))
        expected = Pose2D(0.07, -0.03, math.radians(24.0))
        current = expected.inverse().transform_points(reference)
        result = IcpScanMatcher(
            max_correspondence_m=0.35,
            min_points=35,
        ).match(reference, current)
        self.assertTrue(result.converged)
        self.assertGreater(result.inlier_ratio, 0.75)
        self.assertAlmostEqual(expected.x_m, result.transform.x_m, delta=0.025)
        self.assertAlmostEqual(expected.y_m, result.transform.y_m, delta=0.025)
        self.assertAlmostEqual(expected.yaw_rad, result.transform.yaw_rad, delta=0.025)

    def test_icp_recovery_can_search_wider_than_normal_updates(self):
        angles = np.linspace(-math.pi, math.pi, 300, endpoint=False)
        ranges = (
            2.8
            + 0.55 * np.sin(angles * 2.0)
            + 0.24 * np.cos(angles * 5.0)
            + 0.12 * np.sin(angles * 9.0)
        )
        reference = np.column_stack((
            ranges * np.cos(angles),
            ranges * np.sin(angles),
        ))
        expected = Pose2D(yaw_rad=math.radians(52.0))
        current = expected.inverse().transform_points(reference)
        result = IcpScanMatcher(
            max_correspondence_m=0.35,
            min_points=35,
            coarse_yaw_range_deg=35.0,
        ).match(
            reference,
            current,
            coarse_yaw_range_deg=75.0,
        )

        self.assertTrue(result.converged)
        self.assertGreater(result.inlier_ratio, 0.80)
        self.assertAlmostEqual(
            expected.yaw_rad,
            result.transform.yaw_rad,
            delta=math.radians(2.0),
        )

    def test_icp_coarse_search_avoids_full_room_wall_slide(self):
        angles = np.linspace(
            -math.pi,
            math.pi,
            288,
            endpoint=False,
        )
        distances = []
        for angle in angles:
            cosine = math.cos(angle)
            sine = math.sin(angle)
            candidates = []
            for wall_x in (-1.2, 3.5):
                if abs(cosine) > 1e-9:
                    distance = wall_x / cosine
                    wall_y = distance * sine
                    if distance > 0.0 and -1.0 <= wall_y <= 2.2:
                        candidates.append(distance)
            for wall_y in (-1.0, 2.2):
                if abs(sine) > 1e-9:
                    distance = wall_y / sine
                    wall_x = distance * cosine
                    if distance > 0.0 and -1.2 <= wall_x <= 3.5:
                        candidates.append(distance)
            distances.append(min(candidates))
        reference = np.column_stack((
            np.asarray(distances) * np.cos(angles),
            np.asarray(distances) * np.sin(angles),
        ))
        expected = Pose2D(
            0.03,
            0.01,
            math.radians(8.5),
        )
        current = expected.inverse().transform_points(reference)
        matcher = IcpScanMatcher(
            max_correspondence_m=0.35,
            min_points=35,
            coarse_yaw_range_deg=35.0,
            coarse_yaw_step_deg=4.0,
        )

        result = matcher.match(reference, current)

        self.assertTrue(result.converged)
        self.assertAlmostEqual(
            expected.x_m,
            result.transform.x_m,
            delta=0.005,
        )
        self.assertAlmostEqual(
            expected.y_m,
            result.transform.y_m,
            delta=0.005,
        )
        self.assertAlmostEqual(
            expected.yaw_rad,
            result.transform.yaw_rad,
            delta=math.radians(0.3),
        )

    def test_icp_rotation_with_partial_front_fov_does_not_invent_translation(self):
        world_angles = np.linspace(-math.pi, math.pi, 720, endpoint=False)
        ranges = (
            2.5
            + 0.50 * np.sin(world_angles * 2.3)
            + 0.25 * np.cos(world_angles * 5.7)
            + 0.12 * np.sin(world_angles * 11.3)
        )
        world = np.column_stack((
            ranges * np.cos(world_angles),
            ranges * np.sin(world_angles),
        ))
        half_fov = math.radians(80.0)
        reference = world[np.abs(world_angles) <= half_fov]
        expected = Pose2D(yaw_rad=math.radians(20.0))
        relative_angles = (
            world_angles - expected.yaw_rad + math.pi
        ) % (2.0 * math.pi) - math.pi
        current_world = world[np.abs(relative_angles) <= half_fov]
        current = expected.inverse().transform_points(current_world)

        result = IcpScanMatcher().match(reference, current)
        self.assertTrue(result.converged)
        self.assertLess(math.hypot(
            result.transform.x_m,
            result.transform.y_m,
        ), 0.04)
        self.assertAlmostEqual(
            expected.yaw_rad,
            result.transform.yaw_rad,
            delta=0.025,
        )

    def test_grid_marks_free_and_occupied_and_saves(self):
        grid = OccupancyGrid(
            100, 100, 0.1, min_obstacle_area_m2=0.0
        )
        grid.update((0.0, 0.0), np.array(((2.0, 0.0), (0.0, 2.0))))
        origin = grid.world_to_grid(((0.0, 0.0),))[0]
        endpoint = grid.world_to_grid(((2.0, 0.0),))[0]
        self.assertLess(grid.log_odds[origin[1], origin[0]], 0.0)
        self.assertGreater(grid.log_odds[endpoint[1], endpoint[0]], 0.0)
        with tempfile.TemporaryDirectory() as directory:
            paths = grid.save(os.path.join(directory, "test_map"))
            self.assertTrue(all(path.exists() for path in paths))

    def test_floating_line_with_free_space_on_both_sides_is_removed(self):
        grid = OccupancyGrid(30, 30, 0.1)
        grid.observed[14:17, 10:16] = True
        grid.log_odds[14, 10:16] = -2.0
        grid.log_odds[15, 10:16] = 2.0
        grid.log_odds[16, 10:16] = -2.0
        removed = grid.prune_floating_obstacles()
        self.assertEqual(6, removed)
        self.assertTrue(np.all(grid.log_odds[15, 10:16] < 0.0))

    def test_wall_with_unknown_space_behind_is_preserved(self):
        grid = OccupancyGrid(30, 30, 0.1)
        grid.observed[14, 10:16] = True
        grid.log_odds[14, 10:16] = -2.0
        grid.observed[15, 10:16] = True
        grid.log_odds[15, 10:16] = 2.0
        removed = grid.prune_floating_obstacles()
        self.assertEqual(0, removed)
        self.assertTrue(np.all(grid.log_odds[15, 10:16] > 0.0))

    def test_render_bridges_short_unknown_gap_in_supported_wall(self):
        grid = OccupancyGrid(
            30,
            30,
            0.1,
            min_obstacle_area_m2=0.0,
            render_wall_gap_max_m=0.1,
            render_wall_support_m=0.2,
        )
        grid.observed[15, 8:10] = True
        grid.log_odds[15, 8:10] = 2.0
        grid.observed[15, 11:13] = True
        grid.log_odds[15, 11:13] = 2.0
        image = grid.render_image()
        self.assertEqual(0, image[15, 10])
        self.assertFalse(grid.observed[15, 10])

    def test_render_does_not_close_confirmed_free_wall_gap(self):
        grid = OccupancyGrid(
            30,
            30,
            0.1,
            min_obstacle_area_m2=0.0,
            render_wall_gap_max_m=0.1,
            render_wall_support_m=0.2,
        )
        grid.observed[15, 8:13] = True
        grid.log_odds[15, 8:10] = 2.0
        grid.log_odds[15, 10] = -2.0
        grid.log_odds[15, 11:13] = 2.0
        image = grid.render_image()
        self.assertEqual(254, image[15, 10])

    def test_render_removes_weak_scattered_hits_before_wall_bridging(self):
        grid = OccupancyGrid(
            30,
            30,
            0.1,
            min_obstacle_area_m2=0.0,
            render_wall_gap_max_m=0.2,
        )
        scattered = ((8, 8), (10, 10), (12, 12), (15, 20))
        for row, column in scattered:
            grid.observed[row, column] = True
            grid.log_odds[row, column] = grid.occupied_log_odds

        image = grid.render_image()

        for row, column in scattered:
            self.assertEqual(254, image[row, column])
        self.assertEqual(len(scattered), grid.render_filtered_noise_cells)

    def test_render_bridges_short_diagonal_wall_gap(self):
        grid = OccupancyGrid(
            30,
            30,
            0.1,
            min_obstacle_area_m2=0.0,
            render_wall_gap_max_m=0.1,
            render_wall_support_m=0.2,
        )
        for index in (8, 9, 11, 12):
            grid.observed[index, index] = True
            grid.log_odds[index, index] = 2.0

        image = grid.render_image()

        self.assertEqual(0, image[10, 10])
        self.assertFalse(grid.observed[10, 10])

    def test_render_closes_supported_three_sided_rectangle(self):
        grid = OccupancyGrid(
            70,
            70,
            0.05,
            min_obstacle_area_m2=0.0,
            render_wall_gap_max_m=0.0,
        )
        for start, end in (
            ((24, 24), (36, 24)),
            ((24, 24), (24, 36)),
            ((36, 24), (36, 36)),
        ):
            for column, row in grid._bresenham(start, end):
                grid.observed[row, column] = True
                grid.log_odds[row, column] = 3.0
        grid._obstacle_filter_dirty = True

        rendered = grid.render_obstacle_mask()

        self.assertTrue(rendered[36, 30])
        self.assertFalse(grid.observed[36, 30])
        self.assertGreater(grid.render_contour_cells_closed, 0)

    def test_render_does_not_close_rectangle_across_confirmed_free_mouth(self):
        grid = OccupancyGrid(
            70,
            70,
            0.05,
            min_obstacle_area_m2=0.0,
            render_wall_gap_max_m=0.0,
        )
        for start, end in (
            ((24, 24), (36, 24)),
            ((24, 24), (24, 36)),
            ((36, 24), (36, 36)),
        ):
            for column, row in grid._bresenham(start, end):
                grid.observed[row, column] = True
                grid.log_odds[row, column] = 3.0
        for row in range(34, 39):
            for column in range(26, 35):
                grid.observed[row, column] = True
                grid.log_odds[row, column] = -3.0
        grid._obstacle_filter_dirty = True

        rendered = grid.render_obstacle_mask()

        self.assertFalse(rendered[36, 30])
        self.assertFalse(grid._mask_encloses_pocket(
            rendered[20:43, 20:43]
        ))

    def test_multiview_registration_allows_supported_shape_completion(self):
        grid = OccupancyGrid(
            70,
            70,
            0.05,
            min_obstacle_area_m2=0.0,
            render_wall_gap_max_m=0.0,
        )
        for start, end in (
            ((24, 24), (36, 24)),
            ((24, 24), (24, 36)),
            ((36, 24), (36, 36)),
        ):
            for column, row in grid._bresenham(start, end):
                grid.observed[row, column] = True
                grid.log_odds[row, column] = 3.0
        for row in range(34, 39):
            for column in range(26, 35):
                grid.observed[row, column] = True
                grid.log_odds[row, column] = -3.0
        grid._obstacle_filter_dirty = True
        anchor = (
            grid.origin_x_m + 30.5 * grid.resolution_m,
            grid.origin_y_m + 30.5 * grid.resolution_m,
        )
        grid.register_structural_object(anchor, support_views=7)

        rendered = grid.render_obstacle_mask()

        self.assertTrue(grid._mask_encloses_pocket(
            rendered[20:43, 20:43]
        ))

    def test_render_merges_rotated_duplicate_long_wall_bands(self):
        grid = OccupancyGrid(
            140,
            140,
            0.05,
            min_obstacle_area_m2=0.0,
            render_wall_gap_max_m=0.0,
        )
        angle = math.radians(31.0)
        tangent = np.asarray((math.cos(angle), math.sin(angle)))
        normal = np.asarray((-math.sin(angle), math.cos(angle)))
        center = np.asarray((70.0, 70.0))
        for offset in (-2.0, 2.0):
            start = np.rint(
                center + normal * offset - tangent * 45.0
            ).astype(int)
            end = np.rint(
                center + normal * offset + tangent * 45.0
            ).astype(int)
            for column, row in grid._bresenham(tuple(start), tuple(end)):
                grid.observed[row, column] = True
                grid.log_odds[row, column] = 3.0
        grid._obstacle_filter_dirty = True

        rendered = grid.render_obstacle_mask()
        axis = grid._dominant_manhattan_axis(rendered)
        rows, columns = np.nonzero(rendered)
        along = np.rint(
            columns * math.cos(axis) + rows * math.sin(axis)
        ).astype(int)
        cross = np.rint(
            -columns * math.sin(axis) + rows * math.cos(axis)
        ).astype(int)
        center_along = int(round(
            70.0 * math.cos(axis) + 70.0 * math.sin(axis)
        ))
        central = np.abs(along - center_along) < 30

        self.assertLess(abs(math.degrees(axis) - 31.0), 2.0)
        # A diagonal one-cell Bresenham line can quantize into two adjacent
        # normal bins when its fitted axis is recomputed.  It is still one wall
        # layer as long as there is no separated parallel band.
        central_cross = np.unique(cross[central])
        self.assertLessEqual(int(np.ptp(central_cross)), 1)
        self.assertGreater(grid.render_parallel_wall_cells_merged, 0)

    def test_render_keeps_two_genuinely_separate_long_walls(self):
        grid = OccupancyGrid(
            120,
            120,
            0.05,
            min_obstacle_area_m2=0.0,
            render_wall_gap_max_m=0.0,
        )
        for row in (40, 60):
            grid.observed[row, 20:100] = True
            grid.log_odds[row, 20:100] = 3.0
        grid._obstacle_filter_dirty = True

        rendered = grid.render_obstacle_mask()

        self.assertTrue(np.any(rendered[38:43, 30:90]))
        self.assertTrue(np.any(rendered[58:63, 30:90]))

    def test_render_merges_sparse_duplicate_wall_with_sampling_holes(self):
        grid = OccupancyGrid(
            140,
            100,
            0.05,
            min_obstacle_area_m2=0.0,
            render_wall_gap_max_m=0.0,
        )
        grid.observed[40, 15:120] = True
        grid.log_odds[40, 15:120] = 3.0
        for start, end in ((15, 45), (50, 72), (78, 105)):
            grid.observed[43, start:end] = True
            grid.log_odds[43, start:end] = 3.0
        grid._obstacle_filter_dirty = True

        rendered = grid.render_obstacle_mask()
        rows, _columns = np.nonzero(rendered[:, 20:100])

        self.assertEqual(1, len(np.unique(rows)))
        self.assertGreater(grid.render_parallel_wall_cells_merged, 0)

    def test_render_preserves_short_endpoint_connected_real_wall_step(self):
        grid = OccupancyGrid(
            120,
            120,
            0.05,
            min_obstacle_area_m2=0.0,
            render_wall_gap_max_m=0.0,
        )
        grid.observed[40, 20:50] = True
        grid.log_odds[40, 20:50] = 3.0
        grid.observed[43, 49:85] = True
        grid.log_odds[43, 49:85] = 3.0
        grid.observed[40:44, 49] = True
        grid.log_odds[40:44, 49] = 3.0
        grid._obstacle_filter_dirty = True

        rendered = grid.render_obstacle_mask()

        self.assertTrue(rendered[40, 30])
        self.assertTrue(rendered[43, 70])
        self.assertTrue(np.all(rendered[40:44, 49]))
        self.assertEqual(0, grid.render_parallel_wall_cells_merged)

    def test_render_aligns_unconnected_staggered_long_wall_fragments(self):
        grid = OccupancyGrid(
            160,
            120,
            0.05,
            min_obstacle_area_m2=0.0,
            render_wall_gap_max_m=0.0,
        )
        grid.observed[40, 20:70] = True
        grid.log_odds[40, 20:70] = 3.0
        grid.observed[43, 75:125] = True
        grid.log_odds[43, 75:125] = 3.0
        grid._obstacle_filter_dirty = True

        rendered = grid.render_obstacle_mask()
        rows, _columns = np.nonzero(rendered[:, 25:120])

        self.assertEqual(1, len(np.unique(rows)))
        self.assertGreater(grid.render_parallel_wall_cells_merged, 0)

    def test_render_preserves_short_recess_in_an_otherwise_long_wall(self):
        grid = OccupancyGrid(
            140,
            100,
            0.05,
            min_obstacle_area_m2=0.0,
            render_wall_gap_max_m=0.0,
        )
        # This detour represents a genuine small cabinet/recess along a room
        # wall.  Its two long wall pieces are nearby and endpoint-connected,
        # but are not duplicate observations of the same tangent interval.
        grid.observed[60, 20:61] = True
        grid.log_odds[60, 20:61] = 3.0
        grid.observed[60, 75:120] = True
        grid.log_odds[60, 75:120] = 3.0
        grid.observed[55, 60:76] = True
        grid.log_odds[55, 60:76] = 3.0
        grid.observed[55:61, 60] = True
        grid.log_odds[55:61, 60] = 3.0
        grid.observed[55:61, 75] = True
        grid.log_odds[55:61, 75] = 3.0
        grid._obstacle_filter_dirty = True

        rendered = grid.render_obstacle_mask()

        self.assertTrue(np.all(rendered[55, 61:75]))
        self.assertTrue(np.all(rendered[56:60, 60]))
        self.assertTrue(np.all(rendered[56:60, 75]))
        self.assertFalse(np.any(rendered[60, 62:74]))

    def test_render_closes_supported_round_outline_gap(self):
        grid = OccupancyGrid(
            80,
            80,
            0.05,
            min_obstacle_area_m2=0.0,
            render_wall_gap_max_m=0.0,
        )
        center = 40
        radius = 8
        for sample in range(180):
            angle = math.tau * sample / 180
            if abs(math.atan2(
                math.sin(angle), math.cos(angle)
            )) < math.radians(35.0):
                continue
            column = int(round(center + radius * math.cos(angle)))
            row = int(round(center + radius * math.sin(angle)))
            grid.observed[row, column] = True
            grid.log_odds[row, column] = 3.0
        grid._obstacle_filter_dirty = True

        rendered = grid.render_obstacle_mask()

        self.assertTrue(grid._mask_encloses_pocket(
            rendered[28:53, 28:53]
        ))
        self.assertGreater(grid.render_contour_cells_closed, 0)

    def test_render_keeps_confirmed_free_round_opening(self):
        grid = OccupancyGrid(
            80,
            80,
            0.05,
            min_obstacle_area_m2=0.0,
            render_wall_gap_max_m=0.0,
        )
        center = 40
        radius = 8
        for sample in range(180):
            angle = math.tau * sample / 180
            if abs(math.atan2(
                math.sin(angle), math.cos(angle)
            )) < math.radians(35.0):
                continue
            column = int(round(center + radius * math.cos(angle)))
            row = int(round(center + radius * math.sin(angle)))
            grid.observed[row, column] = True
            grid.log_odds[row, column] = 3.0
        for opening_radius in range(radius - 2, radius + 3):
            for degrees in range(-35, 36):
                angle = math.radians(degrees)
                column = int(round(
                    center + opening_radius * math.cos(angle)
                ))
                row = int(round(
                    center + opening_radius * math.sin(angle)
                ))
                grid.observed[row, column] = True
                grid.log_odds[row, column] = -3.0
        grid._obstacle_filter_dirty = True

        rendered = grid.render_obstacle_mask()

        self.assertFalse(grid._mask_encloses_pocket(
            rendered[28:53, 28:53]
        ))
        self.assertEqual(0, grid.render_contour_cells_closed)

    def test_colored_map_contains_red_path_and_endpoint_stars(self):
        grid = OccupancyGrid(60, 60, 0.1)
        trajectory = [
            (1.0, -1.0, -1.0, 0.0),
            (2.0, 0.0, 0.0, 0.0),
            (3.0, 1.0, 1.0, 0.0),
        ]
        image = build_trajectory_rgb(grid, trajectory)
        for color in (TRAJECTORY_COLOR, START_COLOR, END_COLOR):
            self.assertTrue(np.any(np.all(image == color, axis=2)))
        with tempfile.TemporaryDirectory() as directory:
            path = save_trajectory_png(
                grid,
                trajectory,
                os.path.join(directory, "colored.png"),
            )
            self.assertEqual(
                b"\x89PNG\r\n\x1a\n",
                path.read_bytes()[:8],
            )

    def test_confirmed_obstacle_blocks_inconsistent_far_ray(self):
        grid = OccupancyGrid(
            100, 100, 0.1, min_obstacle_area_m2=0.0
        )
        near_hit = np.array(((1.0, 0.0),))
        grid.update((0.0, 0.0), near_hit)
        grid.update((0.0, 0.0), near_hit)
        obstacle = grid.world_to_grid(near_hit)[0]
        behind = grid.world_to_grid(((2.0, 0.0),))[0]
        grid.update((0.0, 0.0), np.array(((3.0, 0.0),)))
        self.assertGreaterEqual(
            grid.log_odds[obstacle[1], obstacle[0]],
            grid.ray_block_log_odds,
        )
        self.assertFalse(grid.observed[behind[1], behind[0]])

    def test_repeated_free_rays_retire_stale_confirmed_wall(self):
        grid = OccupancyGrid(
            100,
            100,
            0.1,
            contradiction_clear_hits=3,
            min_obstacle_area_m2=0.0,
        )
        near_hit = np.array(((1.0, 0.0),))
        far_hit = np.array(((3.0, 0.0),))
        grid.update((0.0, 0.0), near_hit)
        grid.update((0.0, 0.0), near_hit)
        obstacle = grid.world_to_grid(near_hit)[0]
        behind = grid.world_to_grid(((2.0, 0.0),))[0]

        for _index in range(3):
            grid.update((0.0, 0.0), far_hit)

        self.assertLess(
            grid.log_odds[obstacle[1], obstacle[0]],
            0.0,
        )
        self.assertTrue(grid.observed[behind[1], behind[0]])

    def test_many_rays_in_one_scan_count_as_one_wall_contradiction(self):
        grid = OccupancyGrid(
            100,
            100,
            0.1,
            contradiction_clear_hits=3,
            min_obstacle_area_m2=0.0,
        )
        near_hit = np.array(((1.0, 0.0),))
        far_hits = np.repeat(
            np.array(((3.0, 0.0),)),
            20,
            axis=0,
        )
        grid.update((0.0, 0.0), near_hit)
        grid.update((0.0, 0.0), near_hit)
        obstacle = grid.world_to_grid(near_hit)[0]

        grid.update((0.0, 0.0), far_hits)

        self.assertEqual(
            1,
            grid._contradiction_counts[
                obstacle[1], obstacle[0]
            ],
        )
        self.assertGreater(
            grid.log_odds[obstacle[1], obstacle[0]],
            0.0,
        )

    def test_many_hits_in_one_scan_count_as_one_occupied_observation(self):
        grid = OccupancyGrid(
            100,
            100,
            0.1,
            occupied_inflation_cells=1,
            min_obstacle_area_m2=0.0,
        )
        repeated_hits = np.repeat(
            np.array(((1.0, 0.0),)),
            30,
            axis=0,
        )

        grid.update(
            (0.0, 0.0),
            repeated_hits,
            occupied_evidence_scale=1.25,
        )

        endpoint = grid.world_to_grid(((1.0, 0.0),))[0]
        self.assertAlmostEqual(
            grid.occupied_log_odds * 1.25,
            float(grid.log_odds[endpoint[1], endpoint[0]]),
            places=5,
        )

    def test_repeated_hits_on_same_wall_are_not_contradictions(self):
        grid = OccupancyGrid(
            100,
            100,
            0.1,
            contradiction_clear_hits=3,
            min_obstacle_area_m2=0.0,
        )
        wall_hit = np.array(((1.0, 0.0),))
        for _index in range(8):
            grid.update((0.0, 0.0), wall_hit)
        obstacle = grid.world_to_grid(wall_hit)[0]
        self.assertGreater(
            grid.log_odds[obstacle[1], obstacle[0]],
            grid.ray_block_log_odds,
        )

    def test_confirmed_obstacle_closes_one_cell_ray_gap(self):
        grid = OccupancyGrid(
            100,
            100,
            0.1,
            ray_block_inflation_cells=1,
            min_obstacle_area_m2=0.0,
        )
        hit = np.array(((1.0, 0.1),))
        grid.update((0.0, 0.0), hit)
        grid.update((0.0, 0.0), hit)
        behind = grid.world_to_grid(((2.0, 0.0),))[0]
        grid.update((0.0, 0.0), np.array(((3.0, 0.0),)))
        self.assertFalse(grid.observed[behind[1], behind[0]])

    def test_blocked_ray_does_not_erase_previous_observation(self):
        grid = OccupancyGrid(
            100, 100, 0.1, min_obstacle_area_m2=0.0
        )
        far_hit = np.array(((3.0, 0.0),))
        behind = grid.world_to_grid(((2.0, 0.0),))[0]
        grid.update((0.0, 0.0), far_hit)
        self.assertTrue(grid.observed[behind[1], behind[0]])

        near_hit = np.array(((1.0, 0.0),))
        grid.update((0.0, 0.0), near_hit)
        grid.update((0.0, 0.0), near_hit)
        grid.update((0.0, 0.0), far_hit)
        self.assertTrue(grid.observed[behind[1], behind[0]])

    def test_small_obstacle_component_is_hidden_until_area_is_large(self):
        grid = OccupancyGrid(
            30,
            30,
            0.1,
            min_obstacle_area_m2=0.05,
        )
        grid.observed[10, 5:9] = True
        grid.log_odds[10, 5:9] = 2.0
        significant = grid.significant_obstacles()
        self.assertFalse(np.any(significant[10, 5:9]))
        self.assertEqual(4, grid.filtered_small_cells)

        grid.observed[10, 9:11] = True
        grid.log_odds[10, 9:11] = 2.0
        grid.refresh_obstacle_filter()
        significant = grid.significant_obstacles()
        self.assertTrue(np.all(significant[10, 5:11]))

    def test_grid_expands_without_moving_existing_world_cells(self):
        grid = OccupancyGrid(
            20,
            20,
            0.1,
            auto_expand=True,
            expansion_margin_m=0.2,
            min_obstacle_area_m2=0.0,
        )
        origin_cell_before = grid.world_to_grid(((0.0, 0.0),))[0]
        grid.observed[origin_cell_before[1], origin_cell_before[0]] = True
        grid.log_odds[origin_cell_before[1], origin_cell_before[0]] = -1.0

        grid.update(
            (0.0, 0.0),
            np.array(((2.5, 0.0),)),
        )

        origin_cell_after = grid.world_to_grid(((0.0, 0.0),))[0]
        endpoint = grid.world_to_grid(((2.5, 0.0),))[0]
        self.assertGreater(grid.width, 20)
        self.assertTrue(
            grid.observed[origin_cell_after[1], origin_cell_after[0]]
        )
        self.assertTrue(grid.observed[endpoint[1], endpoint[0]])

    def test_closed_black_boundary_does_not_invent_unknown_obstacles(self):
        grid = OccupancyGrid(
            20,
            20,
            0.1,
            min_obstacle_area_m2=0.0,
        )
        grid.observed[6, 6:14] = True
        grid.observed[13, 6:14] = True
        grid.observed[6:14, 6] = True
        grid.observed[6:14, 13] = True
        grid.log_odds[6, 6:14] = 3.0
        grid.log_odds[13, 6:14] = 3.0
        grid.log_odds[6:14, 6] = 3.0
        grid.log_odds[6:14, 13] = 3.0

        image = grid.render_image()

        self.assertEqual(205, image[9, 9])
        self.assertEqual(205, image[2, 2])
        self.assertEqual(0, grid.render_filled_enclosed_cells)

    def test_slam_rejects_large_single_frame_rotation(self):
        angles = np.linspace(-1.35, 1.35, 180)
        ranges = (
            2.4
            + 0.40 * np.sin(angles * 2.3)
            + 0.22 * np.cos(angles * 5.7)
        )
        reference = np.column_stack((
            ranges * np.cos(angles),
            ranges * np.sin(angles),
        ))
        movement = Pose2D(yaw_rad=math.radians(25.0))
        current = movement.inverse().transform_points(reference)
        current_angles = np.arctan2(current[:, 1], current[:, 0])
        current_ranges = np.linalg.norm(current, axis=1)
        slam = LidarSlam(
            MappingConfig(
                min_match_points=30,
                max_pose_angular_speed_rad_s=100.0,
                max_pose_rotation_step_deg=15.0,
            ),
            LidarConfig(max_distance_m=5.0),
        )
        slam.process(LaserScan(angles, ranges, 1.0))
        update = slam.process(
            LaserScan(current_angles, current_ranges, 2.0)
        )
        self.assertFalse(update.accepted)
        self.assertEqual(
            "rotation_step_jump",
            update.rejection_reason,
        )

    def test_slam_rejects_large_step_even_after_long_scan_gap(self):
        angles = np.linspace(-1.4, 1.4, 180)
        ranges = (
            2.0
            + 0.3 * np.sin(angles * 3.0)
            + 0.2 * np.cos(angles * 6.0)
        )
        reference = np.column_stack((
            ranges * np.cos(angles),
            ranges * np.sin(angles),
        ))
        movement = Pose2D(0.30, 0.0, 0.0)
        current = movement.inverse().transform_points(reference)
        current_angles = np.arctan2(current[:, 1], current[:, 0])
        current_ranges = np.linalg.norm(current, axis=1)
        slam = LidarSlam(
            MappingConfig(
                min_match_points=30,
                max_pose_linear_speed_m_s=10.0,
                max_pose_translation_step_m=0.12,
            ),
            LidarConfig(max_distance_m=5.0),
        )
        slam.process(LaserScan(angles, ranges, 1.0))
        update = slam.process(
            LaserScan(current_angles, current_ranges, 3.0)
        )
        self.assertFalse(update.accepted)
        self.assertEqual(
            "translation_step_jump",
            update.rejection_reason,
        )

    def test_global_map_uses_sparse_keyframes(self):
        slam = LidarSlam(MappingConfig(min_match_points=3))
        points = np.array((
            (1.0, 0.0),
            (1.0, 0.1),
            (1.0, 0.2),
        ))
        slam._add_to_global_map(points, Pose2D(), force=True)
        slam._add_to_global_map(points, Pose2D(0.05, 0.0, 0.0))
        self.assertEqual(1, len(slam._global_keyframes))
        slam._add_to_global_map(points, Pose2D(0.20, 0.0, 0.0))
        self.assertEqual(2, len(slam._global_keyframes))

    def test_global_reference_drops_returns_disproved_by_live_grid(self):
        slam = LidarSlam(MappingConfig(
            min_match_points=1,
            global_map_voxel_m=0.0,
        ))
        points = np.asarray(((1.0, 0.0), (1.0, 0.5)))
        slam._add_to_global_map(points, Pose2D(), force=True)
        contradicted = slam.grid.world_to_grid(((1.0, 0.0),))[0]
        slam.grid.observed[contradicted[1], contradicted[0]] = True
        slam.grid.log_odds[
            contradicted[1], contradicted[0]
        ] = slam.grid.free_log_odds

        slam._refresh_global_reference()

        self.assertEqual(1, len(slam._global_reference))
        np.testing.assert_allclose((1.0, 0.5), slam._global_reference[0])

    def test_periodic_global_jump_falls_back_to_valid_local_match(self):
        angles = np.linspace(-1.3, 1.3, 160)
        ranges = 2.0 + 0.25 * np.sin(angles * 3.0)
        first = LaserScan(angles, ranges, 1.0)
        second = LaserScan(angles, ranges, 1.1)
        slam = LidarSlam(MappingConfig(
            min_match_points=30,
            global_match_every_updates=1,
            manhattan_enabled=False,
        ))
        slam.process(first, stationary_mapping=False)
        reference = slam.previous_points.copy()
        slam._local_reference = reference.copy()
        slam._global_reference = reference.copy()

        results = iter((
            MatchResult(Pose2D(0.01, 0.0, 0.0), 0.030, 140, True, 0.90),
            MatchResult(Pose2D(0.01, 0.0, 0.0), 0.028, 140, True, 0.91),
            # Better ICP residual but an impossible single-frame correction.
            MatchResult(Pose2D(0.16, 0.0, 0.0), 0.010, 150, True, 0.96),
        ))
        slam.matcher.match = lambda *_args, **_kwargs: next(results)

        update = slam.process(second, stationary_mapping=False)

        self.assertTrue(update.accepted)
        self.assertEqual("", update.rejection_reason)
        self.assertAlmostEqual(0.01, update.pose.x_m, delta=1e-9)

    def test_laser_scan_filters_invalid_ranges(self):
        scan = LaserScan(
            np.array((0.0, 1.0, 2.0)),
            np.array((1.0, float("nan"), 10.0)),
            1.0,
        )
        self.assertEqual((1, 2), scan.points(0.1, 5.0).shape)

    def test_front_lidar_offset_moves_points_into_vehicle_frame(self):
        angles = np.linspace(-0.2, 0.2, 80)
        scan = LaserScan(angles, np.ones(80), 1.0)
        slam = LidarSlam(
            MappingConfig(
                min_match_points=20,
                lidar_offset_x_m=0.15,
            ),
            LidarConfig(max_distance_m=5.0),
        )
        points = slam._scan_points(scan)
        self.assertAlmostEqual(
            1.15,
            points[len(points) // 2, 0],
            delta=0.01,
        )

    def test_front_lidar_rotation_is_about_vehicle_centre(self):
        angles = np.linspace(-1.2, 1.2, 160)
        ranges = 2.0 + 0.3 * np.sin(angles * 3.1)
        reference = np.column_stack((
            ranges * np.cos(angles),
            ranges * np.sin(angles),
        ))
        expected = Pose2D(yaw_rad=math.radians(12.0))
        offset = np.array((0.15, 0.0))
        current_lidar = (
            expected.inverse().transform_points(reference) - offset
        )
        current_vehicle = current_lidar + offset
        result = IcpScanMatcher(
            max_correspondence_m=0.35,
            min_points=30,
        ).match(reference, current_vehicle)
        self.assertTrue(result.converged)
        self.assertLess(
            math.hypot(result.transform.x_m, result.transform.y_m),
            0.03,
        )
        self.assertAlmostEqual(
            expected.yaw_rad,
            result.transform.yaw_rad,
            delta=0.04,
        )

    def test_slam_accumulates_matched_pose(self):
        angles = np.linspace(-1.5, 1.5, 140)
        ranges = 2.2 + 0.25 * np.sin(3.1 * angles) + 0.1 * np.cos(7.0 * angles)
        reference = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))
        movement = Pose2D(0.06, 0.02, math.radians(-1.5))
        current = movement.inverse().transform_points(reference)
        current_angles = np.arctan2(current[:, 1], current[:, 0])
        current_ranges = np.linalg.norm(current, axis=1)

        slam = LidarSlam(
            MappingConfig(width_cells=120, height_cells=120, min_match_points=30),
            LidarConfig(max_distance_m=5.0),
        )
        first = slam.process(
            LaserScan(angles, ranges, 1.0),
            stationary_mapping=False,
        )
        second = slam.process(
            LaserScan(current_angles, current_ranges, 2.0),
            stationary_mapping=False,
        )
        self.assertTrue(first.accepted)
        self.assertTrue(second.accepted)
        self.assertAlmostEqual(movement.x_m, second.pose.x_m, delta=0.02)
        self.assertAlmostEqual(movement.y_m, second.pose.y_m, delta=0.02)
        self.assertAlmostEqual(
            movement.yaw_rad,
            second.pose.yaw_rad,
            delta=0.03,
        )
        self.assertEqual(2, len(slam.trajectory))

    def test_stationary_mapping_freezes_false_icp_heading(self):
        angles = np.linspace(-1.5, 1.5, 140)
        ranges = (
            2.2
            + 0.25 * np.sin(3.1 * angles)
            + 0.1 * np.cos(7.0 * angles)
        )
        reference = np.column_stack((
            ranges * np.cos(angles),
            ranges * np.sin(angles),
        ))
        false_motion = Pose2D(
            0.03,
            0.0,
            math.radians(-4.0),
        )
        current = false_motion.inverse().transform_points(reference)
        current_angles = np.arctan2(current[:, 1], current[:, 0])
        current_ranges = np.linalg.norm(current, axis=1)
        slam = LidarSlam(
            MappingConfig(
                width_cells=120,
                height_cells=120,
                min_match_points=30,
            ),
            LidarConfig(max_distance_m=5.0),
        )

        slam.process(LaserScan(angles, ranges, 1.0))
        update = slam.process(
            LaserScan(current_angles, current_ranges, 2.0),
            stationary_mapping=True,
        )

        self.assertTrue(update.accepted)
        self.assertAlmostEqual(0.0, update.pose.x_m, delta=1e-9)
        self.assertAlmostEqual(0.0, update.pose.y_m, delta=1e-9)
        self.assertAlmostEqual(0.0, update.pose.yaw_rad, delta=1e-9)

    def test_later_stationary_stop_tracks_small_residual_rotation(self):
        angles = np.linspace(-1.5, 1.5, 140)
        ranges = (
            2.2
            + 0.25 * np.sin(3.1 * angles)
            + 0.1 * np.cos(7.0 * angles)
        )
        reference = np.column_stack((
            ranges * np.cos(angles),
            ranges * np.sin(angles),
        ))
        coast = Pose2D(yaw_rad=math.radians(9.0))
        current = coast.inverse().transform_points(reference)
        current_angles = np.arctan2(current[:, 1], current[:, 0])
        current_ranges = np.linalg.norm(current, axis=1)
        slam = LidarSlam(
            MappingConfig(
                width_cells=120,
                height_cells=120,
                min_match_points=30,
            ),
            LidarConfig(max_distance_m=5.0),
        )
        slam._last_grid_pose = Pose2D()

        slam.process(
            LaserScan(angles, ranges, 1.0),
            stationary_mapping=False,
        )
        update = slam.process(
            LaserScan(current_angles, current_ranges, 2.0),
            stationary_mapping=True,
        )

        self.assertTrue(update.accepted)
        self.assertAlmostEqual(
            coast.yaw_rad,
            update.pose.yaw_rad,
            delta=0.03,
        )

    def test_stationary_scans_fuse_before_updating_grid(self):
        angles = np.linspace(-0.5, 0.5, 100)
        ranges = np.full(100, 2.0)
        slam = LidarSlam(MappingConfig(min_match_points=20))
        updates = [
            slam.process(LaserScan(
                angles,
                ranges,
                1.0 + index * 0.1,
            ))
            for index in range(7)
        ]
        self.assertEqual(7, len(slam.trajectory))
        self.assertTrue(all(
            not update.map_integrated for update in updates[:6]
        ))
        self.assertTrue(updates[-1].map_integrated)
        self.assertEqual("stationary_fused", updates[-1].map_status)
        self.assertEqual(1, slam.stationary_fusion_count)

    def test_moving_scans_localize_without_updating_grid(self):
        angles = np.linspace(-0.5, 0.5, 100)
        ranges = np.full(100, 2.0)
        slam = LidarSlam(MappingConfig(
            min_match_points=20,
            stationary_map_settle_scans=0,
            stationary_map_fusion_scans=3,
            stationary_map_min_support_scans=2,
        ))
        updates = [
            slam.process(
                LaserScan(angles, ranges, 1.0 + index * 0.1),
                stationary_mapping=False,
            )
            for index in range(4)
        ]
        self.assertTrue(all(update.accepted for update in updates))
        self.assertTrue(all(
            not update.map_integrated for update in updates
        ))
        self.assertTrue(all(
            update.map_status == "moving_localization_only"
            for update in updates
        ))
        self.assertFalse(np.any(slam.grid.observed))

    def test_stationary_fusion_removes_single_scan_range_noise(self):
        slam = LidarSlam(MappingConfig(
            stationary_map_fusion_scans=5,
            stationary_map_min_support_scans=3,
            stationary_map_angle_bin_deg=1.0,
            stationary_map_range_tolerance_m=0.10,
        ))
        scans = []
        for index in range(5):
            points = np.asarray(((2.0, 0.0),))
            if index == 2:
                points = np.asarray(((0.5, 0.0),))
            if index == 4:
                angle = math.radians(20.0)
                points = np.vstack((
                    points,
                    (1.0 * math.cos(angle), 1.0 * math.sin(angle)),
                ))
            scans.append(points)

        fused = slam._fuse_stationary_scans(scans)

        self.assertEqual(1, len(fused))
        self.assertAlmostEqual(2.0, np.linalg.norm(fused[0]), delta=0.02)

    def test_first_stationary_wall_is_provisional_until_confirmed(self):
        angles = np.linspace(
            math.radians(-30.0),
            math.radians(30.0),
            121,
        )
        ranges = 2.0 / np.cos(angles)
        slam = LidarSlam(
            MappingConfig(
                width_cells=160,
                height_cells=160,
                resolution_m=0.05,
                min_match_points=20,
                mapping_max_distance_m=6.0,
            ),
            LidarConfig(max_distance_m=8.0),
        )

        updates = [
            slam.process(LaserScan(
                angles,
                ranges,
                1.0 + index * 0.1,
            ))
            for index in range(7)
        ]
        provisional = slam.grid.render_image()

        self.assertTrue(updates[-1].map_integrated)
        self.assertEqual(0, np.count_nonzero(provisional == 0))

        # A second independent stopped observation promotes the same
        # endpoints above the occupied threshold.
        slam.pose = Pose2D(0.35, 0.0, 0.0)
        confirming_ranges = (2.0 - slam.pose.x_m) / np.cos(angles)
        points = slam._scan_points(LaserScan(
            angles,
            confirming_ranges,
            2.0,
        ))
        integrated, _status = slam._integrate(
            points,
            2.0,
            force=True,
        )
        confirmed = slam.grid.render_image()

        self.assertTrue(integrated)
        self.assertGreater(np.count_nonzero(confirmed == 0), 20)

    def test_first_view_radial_artifact_does_not_render_as_wall(self):
        slam = LidarSlam(MappingConfig(
            width_cells=160,
            height_cells=160,
            resolution_m=0.05,
            min_match_points=3,
            min_obstacle_area_m2=0.0,
        ))
        radial_artifact = np.column_stack((
            np.linspace(0.25, 3.5, 50),
            np.zeros(50),
        ))

        integrated, _status = slam._integrate(
            radial_artifact,
            1.0,
            force=True,
        )

        self.assertTrue(integrated)
        self.assertEqual(
            0,
            np.count_nonzero(slam.grid.render_image() == 0),
        )

    def test_stationary_mapping_is_due_after_observation_spacing(self):
        slam = LidarSlam(MappingConfig(
            stationary_map_translation_m=0.40,
            stationary_map_rotation_deg=18.0,
        ))
        self.assertTrue(slam.stationary_mapping_due())
        slam._last_grid_pose = Pose2D()
        slam.pose = Pose2D(0.39, 0.0, math.radians(17.0))
        self.assertFalse(slam.stationary_mapping_due())
        slam.pose = Pose2D(0.41, 0.0, 0.0)
        self.assertTrue(slam.stationary_mapping_due())

    def test_grid_integration_accepts_only_stable_low_speed_motion(self):
        slam = LidarSlam(MappingConfig(
            min_match_points=20,
            map_min_linear_speed_m_s=0.025,
            map_max_linear_speed_m_s=0.30,
            map_max_angular_speed_rad_s=1.20,
        ))
        points = np.array(((1.0, -0.2), (1.0, 0.2), (2.0, 0.0)))
        integrated, status = slam._integrate(
            points,
            1.0,
            force=True,
        )
        self.assertTrue(integrated)
        self.assertEqual("integrated", status)

        slam.pose = Pose2D(0.05, 0.0, 0.0)
        integrated, status = slam._integrate(
            points,
            2.0,
            linear_speed_m_s=0.05,
            angular_speed_rad_s=0.1,
        )
        self.assertTrue(integrated)

        slam.pose = Pose2D(0.10, 0.0, 0.0)
        integrated, status = slam._integrate(
            points,
            3.0,
            linear_speed_m_s=0.50,
            angular_speed_rad_s=0.1,
        )
        self.assertFalse(integrated)
        self.assertEqual("too_fast", status)

        integrated, status = slam._integrate(
            points,
            4.0,
            linear_speed_m_s=0.05,
            angular_speed_rad_s=1.5,
        )
        self.assertFalse(integrated)
        self.assertEqual("turning_too_fast", status)

        integrated, status = slam._integrate(
            points,
            5.0,
            linear_speed_m_s=0.0,
            angular_speed_rad_s=0.0,
        )
        self.assertFalse(integrated)
        self.assertEqual("stationary", status)

    def test_full_scan_localizes_while_front_sector_alone_updates_map(self):
        angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
        distances = np.full_like(angles, 2.0)
        full_scan = LaserScan(angles, distances, 1.0)
        mapping_scan = filter_scan_sector(full_scan, 0.0, 160.0)
        slam = LidarSlam(
            MappingConfig(
                width_cells=160,
                height_cells=160,
                resolution_m=0.05,
                min_match_points=30,
                min_obstacle_area_m2=0.0,
            ),
            LidarConfig(max_distance_m=8.0),
        )

        updates = [
            slam.process(
                LaserScan(
                    full_scan.angles_rad,
                    full_scan.distances_m,
                    1.0 + index * 0.1,
                ),
                mapping_scan=LaserScan(
                    mapping_scan.angles_rad,
                    mapping_scan.distances_m,
                    1.0 + index * 0.1,
                ),
            )
            for index in range(7)
        ]
        update = updates[-1]

        self.assertTrue(update.accepted)
        self.assertTrue(update.map_integrated)
        self.assertLess(float(np.min(slam.previous_points[:, 0])), -1.5)
        front_col, front_row = slam.grid.world_to_grid(((2.0, 0.0),))[0]
        rear_col, rear_row = slam.grid.world_to_grid(((-2.0, 0.0),))[0]
        self.assertTrue(np.any(slam.grid.observed[
            front_row - 1:front_row + 2,
            front_col - 1:front_col + 2,
        ]))
        self.assertFalse(slam.grid.observed[rear_row, rear_col])

    def test_far_mapping_echo_clears_to_limit_without_black_boundary(self):
        slam = LidarSlam(
            MappingConfig(
                width_cells=160,
                height_cells=160,
                resolution_m=0.05,
                mapping_max_distance_m=3.0,
                min_obstacle_area_m2=0.0,
            )
        )

        integrated, status = slam._integrate(
            np.asarray(((5.0, 0.0),)),
            1.0,
            force=True,
        )

        limit_col, limit_row = slam.grid.world_to_grid(((3.0, 0.0),))[0]
        beyond_col, beyond_row = slam.grid.world_to_grid(((3.5, 0.0),))[0]
        self.assertTrue(integrated)
        self.assertEqual("integrated", status)
        self.assertTrue(slam.grid.observed[limit_row, limit_col])
        self.assertLess(slam.grid.log_odds[limit_row, limit_col], 0.0)
        self.assertFalse(slam.grid.observed[beyond_row, beyond_col])

    def test_mapping_interpolates_safe_free_space_between_dense_beams(self):
        slam = LidarSlam(MappingConfig(
            resolution_m=0.05,
            width_cells=160,
            height_cells=160,
            map_free_fill_max_angle_deg=2.0,
        ))
        angles = np.radians((-0.8, 0.8))
        distance = 5.0
        endpoints = np.column_stack((
            distance * np.cos(angles),
            distance * np.sin(angles),
        ))

        inserted = slam._interpolate_free_endpoints(
            endpoints,
            np.asarray((0.0, 0.0)),
        )

        self.assertGreater(len(inserted), 0)
        self.assertTrue(np.any(np.abs(inserted[:, 1]) < 0.01))
        self.assertTrue(np.all(
            np.linalg.norm(inserted, axis=1) < distance
        ))

    def test_rebuild_requires_repeated_evidence_before_forgetting_old_free(self):
        slam = LidarSlam(
            MappingConfig(
                width_cells=100,
                height_cells=100,
                resolution_m=0.1,
                mapping_max_distance_m=3.0,
                min_obstacle_area_m2=0.0,
            )
        )
        slam._integrate(
            np.asarray(((2.0, 0.0),)),
            1.0,
            force=True,
        )
        old_free = slam.grid.world_to_grid(((1.0, 0.0),))[0]
        self.assertTrue(slam.grid.observed[old_free[1], old_free[0]])

        slam._integrate(
            np.asarray(((0.5, 0.0),)),
            2.0,
            force=True,
        )
        structural_anchor = (0.25, 0.30)
        slam.grid.register_structural_object(
            structural_anchor,
            support_views=7,
        )
        self.assertEqual(2, len(slam._mapping_keyframes))
        self.assertTrue(slam.rebuild_map())
        self.assertEqual(
            [structural_anchor],
            slam.grid._structural_object_anchors,
        )
        self.assertTrue(slam.grid.observed[old_free[1], old_free[0]])

        for index in range(2, 9):
            slam._integrate(
                np.asarray(((0.5, 0.0),)),
                float(index + 1),
                force=True,
            )
        self.assertEqual(4, len(slam._mapping_keyframes))
        self.assertGreater(slam.map_rebuild_count, 0)
        self.assertTrue(slam.rebuild_map())
        new_wall = slam.grid.world_to_grid(((0.5, 0.0),))[0]
        self.assertFalse(slam.grid.observed[old_free[1], old_free[0]])
        self.assertGreater(
            slam.grid.log_odds[new_wall[1], new_wall[0]],
            0.0,
        )

    def test_mapping_hold_skips_frames_after_relocalization(self):
        slam = LidarSlam(MappingConfig(min_match_points=3))
        points = np.array(((1.0, -0.2), (1.0, 0.0), (1.0, 0.2)))
        slam._integrate(points, 1.0, force=True)
        original = slam.grid.log_odds.copy()
        slam.suspend_mapping(2)
        slam.pose = Pose2D(0.05, 0.0, 0.0)

        first = slam._integrate(
            points,
            2.0,
            linear_speed_m_s=0.05,
        )
        second = slam._integrate(
            points,
            3.0,
            linear_speed_m_s=0.05,
        )

        self.assertEqual((False, "relocalizing"), first)
        self.assertEqual((False, "relocalizing"), second)
        np.testing.assert_allclose(original, slam.grid.log_odds)

    def test_slam_rejects_impossible_translation_jump(self):
        angles = np.linspace(-1.4, 1.4, 180)
        ranges = 2.0 + 0.3 * np.sin(angles * 3.0) + 0.2 * np.cos(angles * 6.0)
        reference = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))
        movement = Pose2D(0.24, 0.0, 0.0)
        current = movement.inverse().transform_points(reference)
        current_angles = np.arctan2(current[:, 1], current[:, 0])
        current_ranges = np.linalg.norm(current, axis=1)
        slam = LidarSlam(
            MappingConfig(
                width_cells=120,
                height_cells=120,
                min_match_points=30,
                pose_translation_margin_m=0.03,
            ),
            LidarConfig(max_distance_m=5.0),
        )
        slam.process(LaserScan(angles, ranges, 1.0))
        update = slam.process(LaserScan(current_angles, current_ranges, 1.1))
        self.assertFalse(update.accepted)
        self.assertEqual("translation_step_jump", update.rejection_reason)
        self.assertEqual(1, len(slam.trajectory))

    def test_isolated_scan_points_are_removed(self):
        cluster = np.array(((1.0, 0.0), (1.02, 0.01), (1.04, 0.02)))
        isolated = np.array(((4.0, 4.0),))
        filtered = LidarSlam._filter_isolated(
            np.vstack((cluster, isolated)),
            0.10,
        )
        self.assertEqual(3, len(filtered))

    def test_adaptive_cluster_filter_removes_short_spikes(self):
        wall = np.array((
            (2.00, -0.08),
            (2.00, -0.04),
            (2.00, 0.00),
            (2.00, 0.04),
            (2.00, 0.08),
        ))
        spike = np.array(((4.0, 1.5), (4.03, 1.5)))
        filtered = LidarSlam._filter_dense_clusters(
            np.vstack((wall, spike)),
            0.06,
            0.025,
            0.28,
            2,
        )
        self.assertEqual(5, len(filtered))

    def test_local_submap_anchors_absolute_pose(self):
        angles = np.linspace(-1.3, 1.3, 180)
        ranges = (
            2.2
            + 0.35 * np.sin(angles * 2.7)
            + 0.16 * np.cos(angles * 6.1)
        )
        world = np.column_stack((
            ranges * np.cos(angles),
            ranges * np.sin(angles),
        ))
        expected = Pose2D(0.08, -0.03, math.radians(3.0))
        current = expected.inverse().transform_points(world)
        slam = LidarSlam(MappingConfig(min_match_points=30))
        slam._add_to_local_submap(world, Pose2D())
        result = slam.matcher.match(
            slam._local_reference,
            current,
            initial=Pose2D(0.18, 0.04, math.radians(5.0)),
        )
        self.assertTrue(slam._local_match_is_reliable(result))
        self.assertAlmostEqual(expected.x_m, result.transform.x_m, delta=0.025)
        self.assertAlmostEqual(expected.y_m, result.transform.y_m, delta=0.025)

    def test_pure_rotation_constraint_caps_false_translation(self):
        angles = np.linspace(-1.2, 1.2, 120)
        points = np.column_stack((2.0 * np.cos(angles), 2.0 * np.sin(angles)))
        slam = LidarSlam()
        match = MatchResult(
            Pose2D(0.30, 0.0, 0.0),
            0.01,
            len(points),
            True,
            1.0,
        )
        constrained = slam._constrain_pure_rotation(match, points, points)
        self.assertAlmostEqual(
            slam.mapping_config.pure_rotation_max_translation_m,
            math.hypot(
                constrained.transform.x_m,
                constrained.transform.y_m,
            ),
        )
        self.assertLessEqual(constrained.inlier_ratio, 1.0)

    def test_lidar_turn_allows_arc_translation_but_rejects_large_step(self):
        angles = np.linspace(-1.4, 1.4, 180)
        ranges = (
            2.0
            + 0.3 * np.sin(angles * 3.0)
            + 0.2 * np.cos(angles * 6.0)
        )
        reference = np.column_stack((
            ranges * np.cos(angles),
            ranges * np.sin(angles),
        ))
        movement = Pose2D(0.095, 0.0, math.radians(6.0))
        current = movement.inverse().transform_points(reference)
        current_angles = np.arctan2(current[:, 1], current[:, 0])
        current_ranges = np.linalg.norm(current, axis=1)
        slam = LidarSlam(
            MappingConfig(
                min_match_points=30,
                max_pose_linear_speed_m_s=10.0,
                lidar_turn_max_translation_m=0.08,
                manhattan_enabled=False,
            ),
            LidarConfig(max_distance_m=5.0),
        )
        slam.process(LaserScan(angles, ranges, 1.0))

        update = slam.process(
            LaserScan(current_angles, current_ranges, 2.0)
        )

        self.assertFalse(update.accepted)
        self.assertEqual("turn_translation_jump", update.rejection_reason)

    def test_global_relocalization_updates_pose_without_writing_map(self):
        angles = np.linspace(-1.2, 1.2, 160)
        ranges = 2.0 + 0.25 * np.sin(angles * 3.3)
        scan = LaserScan(angles, ranges, 1.0)
        slam = LidarSlam(MappingConfig(
            min_match_points=30,
            relocalization_min_keyframes=1,
        ))
        slam.process(scan)
        trajectory_length = len(slam.trajectory)
        grid_before = slam.grid.log_odds.copy()
        slam.pose = Pose2D(0.10, -0.03, math.radians(2.0))
        relocated_scan = LaserScan(angles, ranges, 2.0)

        self.assertTrue(slam.relocalize(relocated_scan))
        self.assertAlmostEqual(0.0, slam.pose.x_m, delta=0.03)
        self.assertAlmostEqual(0.0, slam.pose.y_m, delta=0.03)
        self.assertEqual(trajectory_length, len(slam.trajectory))
        np.testing.assert_allclose(grid_before, slam.grid.log_odds)

    def test_manhattan_regularizer_applies_only_small_yaw_correction(self):
        x = np.linspace(0.5, 2.0, 80)
        points = np.column_stack((x, np.zeros_like(x)))
        slam = LidarSlam(MappingConfig(
            min_match_points=20,
            manhattan_enabled=True,
            manhattan_anchor_observations=1,
            manhattan_min_segments=20,
            manhattan_min_confidence=0.8,
            manhattan_max_correction_deg=0.8,
        ))
        slam._manhattan_axis_yaw = 0.0

        corrected = slam._regularize_manhattan_yaw(
            Pose2D(yaw_rad=math.radians(5.0)),
            points,
        )

        self.assertAlmostEqual(
            4.2,
            math.degrees(corrected.yaw_rad),
            delta=0.05,
        )
        self.assertEqual(1, slam.manhattan_correction_count)

    def test_reseed_replaces_reference_without_writing_map(self):
        angles = np.linspace(-1.2, 1.2, 120)
        first = LaserScan(angles, np.full(120, 2.0), 1.0)
        second = LaserScan(angles, np.full(120, 2.5), 2.0)
        slam = LidarSlam()
        slam.process(first)
        trajectory_length = len(slam.trajectory)
        self.assertTrue(slam.reseed(second))
        self.assertEqual(trajectory_length, len(slam.trajectory))
        self.assertEqual(2.0, slam.previous_timestamp_s)

    def test_bad_startup_reference_can_be_replaced_at_zero_pose(self):
        angles = np.linspace(-math.pi, math.pi, 180, endpoint=False)
        first = LaserScan(angles, np.full(180, 2.0), 1.0)
        replacement = LaserScan(angles, np.full(180, 2.7), 2.0)
        slam = LidarSlam()
        slam.process(first)
        slam.pose = Pose2D(0.4, -0.2, math.radians(35.0))

        self.assertTrue(slam.reset_startup_reference(replacement))

        self.assertEqual(Pose2D(), slam.pose)
        self.assertEqual(2.0, slam.previous_timestamp_s)
        self.assertEqual(1, len(slam._global_keyframes))
        self.assertIsNotNone(slam._global_reference)

    def test_startup_reference_cannot_reset_after_first_map_keyframe(self):
        angles = np.linspace(-math.pi, math.pi, 180, endpoint=False)
        scan = LaserScan(angles, np.full(180, 2.0), 1.0)
        slam = LidarSlam()
        slam.process(scan)
        previous_points = slam.previous_points.copy()
        slam.map_integrated_count = 1

        self.assertFalse(slam.reset_startup_reference(scan))
        np.testing.assert_allclose(previous_points, slam.previous_points)

    def test_required_reseed_does_not_discard_reference_on_failed_recovery(self):
        angles = np.linspace(-1.2, 1.2, 120)
        first = LaserScan(angles, np.full(120, 2.0), 1.0)
        unrelated = LaserScan(angles, np.full(120, 4.5), 2.0)
        slam = LidarSlam()
        slam.process(first)
        previous_points = slam.previous_points.copy()
        previous_timestamp = slam.previous_timestamp_s

        self.assertFalse(slam.reseed(
            unrelated,
            require_motion_recovery=True,
        ))
        np.testing.assert_allclose(previous_points, slam.previous_points)
        self.assertEqual(previous_timestamp, slam.previous_timestamp_s)

    def test_required_reseed_rejects_fast_false_wall_slide(self):
        angles = np.linspace(-1.2, 1.2, 120)
        ranges = 2.0 + 0.15 * np.sin(angles * 4.0)
        first = LaserScan(angles, ranges, 1.0)
        second = LaserScan(angles, ranges, 1.1)
        slam = LidarSlam(MappingConfig(
            max_pose_linear_speed_m_s=0.50,
            pose_translation_margin_m=0.01,
        ))
        slam.process(first)
        previous_points = slam.previous_points.copy()
        previous_timestamp = slam.previous_timestamp_s
        slam.matcher.match = lambda *_args, **_kwargs: MatchResult(
            Pose2D(0.18, 0.0, 0.0),
            0.02,
            100,
            True,
            0.90,
        )

        self.assertFalse(slam.reseed(
            second,
            require_motion_recovery=True,
        ))
        np.testing.assert_allclose(previous_points, slam.previous_points)
        self.assertEqual(previous_timestamp, slam.previous_timestamp_s)

    def test_stationary_anchor_rejects_ambiguous_wall_translation(self):
        points = np.column_stack((
            np.full(160, 2.0),
            np.linspace(-1.8, 1.8, 160),
        ))
        predicted = Pose2D()
        slam = LidarSlam(MappingConfig(
            manhattan_enabled=False,
            stationary_anchor_min_keyframes=3,
            stationary_anchor_large_correction_m=0.06,
            stationary_anchor_min_score_improvement=0.008,
        ))
        slam.pose = predicted
        slam._global_reference = points.copy()
        slam._global_keyframes = [points.copy() for _ in range(3)]

        def ambiguous_match(_reference, _current, initial=None, **_kwargs):
            if initial is not None and abs(initial.yaw_rad) < 1e-9:
                return MatchResult(predicted, 0.080, 140, True, 0.80)
            return MatchResult(
                Pose2D(0.18, 0.0, 0.0),
                0.078,
                140,
                True,
                0.80,
            )

        slam.matcher.match = ambiguous_match
        corrected = slam._anchor_stationary_pose(
            points,
            predicted,
            5.0,
        )

        self.assertEqual(predicted, corrected)
        self.assertEqual(0, slam.stationary_anchor_corrections)
        self.assertEqual(1, slam.stationary_anchor_rejections)

    def test_stationary_global_anchor_repairs_rolling_turn_yaw_loss(self):
        angles = np.linspace(-math.pi, math.pi, 260, endpoint=False)
        ranges = (
            2.5
            + 0.42 * np.sin(angles * 2.0)
            + 0.23 * np.cos(angles * 5.0)
            + 0.11 * np.sin(angles * 9.0)
        )
        points = np.column_stack((
            ranges * np.cos(angles),
            ranges * np.sin(angles),
        ))
        true_pose = Pose2D(0.45, -0.20, math.radians(35.0))
        predicted = Pose2D(0.45, -0.20, math.radians(14.0))
        reference = true_pose.transform_points(points)
        slam = LidarSlam(MappingConfig(
            manhattan_enabled=False,
            stationary_anchor_min_keyframes=3,
            stationary_anchor_max_translation_m=0.60,
            stationary_anchor_min_score_improvement=0.002,
        ))
        slam.pose = predicted
        slam._global_reference = reference
        slam._global_keyframes = [reference.copy() for _ in range(3)]

        corrected = slam._anchor_stationary_pose(points, predicted, 5.0)

        self.assertAlmostEqual(true_pose.x_m, corrected.x_m, delta=0.04)
        self.assertAlmostEqual(true_pose.y_m, corrected.y_m, delta=0.04)
        self.assertAlmostEqual(
            true_pose.yaw_rad,
            corrected.yaw_rad,
            delta=math.radians(2.0),
        )
        self.assertEqual(1, slam.stationary_anchor_corrections)


if __name__ == "__main__":
    unittest.main()
