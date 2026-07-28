"""扫描过滤、匹配、位姿累计和占据栅格更新。"""

from collections import deque
from dataclasses import dataclass
import csv
import math
from pathlib import Path

import numpy as np

from raspberry_pi.config import LidarConfig, MappingConfig
from .geometry import Pose2D, normalize_angle
from .map_visualization import save_trajectory_png
from .occupancy_grid import OccupancyGrid
from .scan_matcher import IcpScanMatcher, MatchResult


@dataclass(frozen=True)
class SlamUpdate:
    pose: Pose2D
    accepted: bool
    rmse_m: float
    correspondences: int
    scan_points: int
    inlier_ratio: float = 0.0
    translation_m: float = 0.0
    rotation_rad: float = 0.0
    rejection_reason: str = ""
    map_integrated: bool = False
    map_status: str = ""
    linear_speed_m_s: float = 0.0
    angular_speed_rad_s: float = 0.0


class LidarSlam:
    def __init__(self, mapping_config=None, lidar_config=None):
        self.mapping_config = mapping_config or MappingConfig()
        self.lidar_config = lidar_config or LidarConfig()
        self.grid = self._new_grid()
        self.matcher = IcpScanMatcher(
            self.mapping_config.max_correspondence_m,
            self.mapping_config.min_match_points,
            coarse_yaw_range_deg=self.mapping_config.coarse_yaw_range_deg,
            coarse_yaw_step_deg=self.mapping_config.coarse_yaw_step_deg,
            trim_fraction=self.mapping_config.icp_trim_fraction,
        )
        self.pose = Pose2D()
        # Vehicle heading is accumulated from the rotation of complete,
        # adjacent lidar point clouds, with the startup direction as zero.
        self._scan_heading_yaw = self.pose.yaw_rad
        self.previous_points = None
        self.previous_timestamp_s = None
        self.trajectory = []
        self._local_scans = deque(
            maxlen=max(1, self.mapping_config.local_submap_scan_count)
        )
        self._local_reference = None
        self._global_keyframes = []
        self._global_reference = None
        self._last_global_keyframe_pose = None
        self._mapping_keyframes = []
        self._match_updates = 0
        self._last_grid_pose = None
        self._last_grid_timestamp_s = None
        self._stationary_mapping_active = False
        self._stationary_settle_left = 0
        self._stationary_mapping_scans = []
        self._stationary_mapping_poses = []
        self._linear_speed_samples = deque(
            maxlen=max(1, self.mapping_config.map_speed_filter_window)
        )
        self._angular_speed_samples = deque(
            maxlen=max(1, self.mapping_config.map_speed_filter_window)
        )
        self._map_hold_frames = 0
        self._manhattan_observations = deque(
            maxlen=max(
                1,
                self.mapping_config.manhattan_anchor_observations,
            )
        )
        self._manhattan_axis_yaw = None
        self.global_relocalization_count = 0
        self.manhattan_correction_count = 0
        self.map_integrated_count = 0
        self.map_rebuild_count = 0
        self.map_keyframes_replaced = 0
        self.stationary_fusion_count = 0
        self.stationary_fusion_input_points = 0
        self.stationary_fusion_output_points = 0
        self.map_skip_counts = {}

    def _new_grid(self):
        return OccupancyGrid(
            self.mapping_config.width_cells,
            self.mapping_config.height_cells,
            self.mapping_config.resolution_m,
            occupied_inflation_cells=(
                self.mapping_config
                .stationary_map_occupied_inflation_cells
            ),
            contradiction_clear_hits=(
                self.mapping_config.map_contradiction_clear_hits
            ),
            min_obstacle_area_m2=(
                self.mapping_config.min_obstacle_area_m2
            ),
            render_wall_gap_max_m=(
                self.mapping_config.render_wall_gap_max_m
            ),
            render_wall_support_m=(
                self.mapping_config.render_wall_support_m
            ),
            auto_expand=self.mapping_config.map_auto_expand,
            expansion_margin_m=(
                self.mapping_config.map_expand_margin_m
            ),
        )

    def process(
        self,
        scan,
        pure_rotation=False,
        mapping_scan=None,
        stationary_mapping=True,
    ):
        """Localize with a complete scan and optionally map a restricted scan.

        ``scan`` always feeds the adjacent-frame, rolling-submap, and global
        localization references.  ``mapping_scan`` is allowed to contain only
        the trustworthy front sector and is used exclusively for occupancy-grid
        integration. Keeping these inputs separate prevents a mapping FOV
        limit from throwing away useful localization history.

        ``stationary_mapping`` is controlled by the autonomous observation
        scheduler. False means localization-only. True starts or continues a
        settle-and-fuse batch; no individual moving scan is written directly
        into the occupancy grid.
        """
        # ``pure_rotation`` is kept only for API compatibility. Wheel commands
        # are not odometry; turning is classified below from lidar motion.
        _ = pure_rotation
        points = self._scan_points(scan)
        mapping_points = (
            points
            if mapping_scan is None
            else self._scan_points(mapping_scan)
        )
        if len(points) < self.mapping_config.min_match_points:
            return SlamUpdate(
                self.pose,
                False,
                math.inf,
                0,
                len(points),
                rejection_reason="too_few_filtered_points",
            )

        if self.previous_points is None:
            self.previous_points = points
            self.previous_timestamp_s = scan.timestamp_s
            self.trajectory.append((
                scan.timestamp_s,
                self.pose.x_m,
                self.pose.y_m,
                self.pose.yaw_rad,
            ))
            map_integrated, map_status = self._stationary_map_update(
                mapping_points,
                scan.timestamp_s,
                stationary_mapping,
                match_rmse_m=0.0,
                match_inlier_ratio=1.0,
            )
            self._add_to_local_submap(points, self.pose)
            self._add_to_global_map(points, self.pose, force=True)
            return SlamUpdate(
                self.pose,
                True,
                0.0,
                len(points),
                len(points),
                inlier_ratio=1.0,
                map_integrated=map_integrated,
                map_status=map_status,
            )

        frame_match = self.matcher.match(self.previous_points, points)
        self._match_updates += 1
        match = frame_match
        match_is_global = False

        # First estimate motion from adjacent frames, then correct it against
        # several recent accepted scans expressed in world coordinates.
        if self._local_reference is not None:
            initial_pose = self.pose.compose(frame_match.transform)
            local_match = self.matcher.match(
                self._local_reference,
                points,
                initial=initial_pose,
            )
            if self._local_match_is_reliable(local_match):
                match = local_match
                match_is_global = True

        # A sparse, long-term point map limits cumulative drift when the car
        # revisits structure that has already left the rolling local submap.
        candidate_pose = (
            match.transform
            if match_is_global
            else self.pose.compose(match.transform)
        )
        if (
            self._global_reference is not None
            and self._match_updates
            % max(1, self.mapping_config.global_match_every_updates)
            == 0
        ):
            global_match = self.matcher.match(
                self._global_reference,
                points,
                initial=candidate_pose,
            )
            correction = candidate_pose.inverse().compose(
                global_match.transform
            )
            if (
                self._local_match_is_reliable(global_match)
                and math.hypot(correction.x_m, correction.y_m)
                <= self.mapping_config.global_match_max_correction_m
                and abs(correction.yaw_rad)
                <= math.radians(
                    self.mapping_config.global_match_max_correction_deg
                )
            ):
                match = global_match
                match_is_global = True

        relative_transform = (
            self.pose.inverse().compose(match.transform)
            if match_is_global
            else match.transform
        )
        translation_m = math.hypot(
            relative_transform.x_m,
            relative_transform.y_m,
        )
        rotation_rad = abs(relative_transform.yaw_rad)
        lidar_detected_turn = rotation_rad >= math.radians(
            self.mapping_config.lidar_turn_min_rotation_deg
        )
        elapsed_s = max(
            0.001,
            scan.timestamp_s - self.previous_timestamp_s,
        )
        translation_limit = (
            self.mapping_config.max_pose_linear_speed_m_s * elapsed_s
            + self.mapping_config.pose_translation_margin_m
        )
        rotation_limit = (
            self.mapping_config.max_pose_angular_speed_rad_s * elapsed_s
            + math.radians(self.mapping_config.pose_rotation_margin_deg)
        )

        rejection_reason = ""
        if not match.converged:
            rejection_reason = "icp_not_converged"
        elif match.rmse_m > self.mapping_config.max_match_rmse_m:
            rejection_reason = "rmse_too_high"
        elif match.inlier_ratio < self.mapping_config.min_match_inlier_ratio:
            rejection_reason = "inlier_ratio_too_low"
        elif translation_m > self.mapping_config.max_pose_translation_step_m:
            rejection_reason = "translation_step_jump"
        elif (
            lidar_detected_turn
            and translation_m
            > self.mapping_config.lidar_turn_max_translation_m
        ):
            rejection_reason = "turn_translation_jump"
        elif translation_m > translation_limit:
            rejection_reason = "translation_jump"
        elif rotation_rad > math.radians(
            self.mapping_config.max_pose_rotation_step_deg
        ):
            rejection_reason = "rotation_step_jump"
        elif (
            stationary_mapping
            and rotation_rad > math.radians(
                self.mapping_config
                .stationary_localization_max_rotation_step_deg
            )
        ):
            rejection_reason = "stationary_rotation_jump"
        elif rotation_rad > rotation_limit:
            rejection_reason = "rotation_jump"

        accepted = not rejection_reason
        map_integrated = False
        map_status = "localization_rejected"
        linear_speed_m_s = translation_m / elapsed_s
        angular_speed_rad_s = rotation_rad / elapsed_s
        if accepted:
            proposed_pose = (
                match.transform
                if match_is_global
                else self.pose.compose(match.transform)
            )
            adjacent_rotation = relative_transform.yaw_rad
            adjacent_reliable = (
                frame_match.converged
                and frame_match.rmse_m
                <= self.mapping_config.max_match_rmse_m
                and frame_match.inlier_ratio
                >= self.mapping_config.min_match_inlier_ratio
                and abs(frame_match.transform.yaw_rad)
                <= math.radians(
                    self.mapping_config.max_pose_rotation_step_deg
                )
            )
            if adjacent_reliable:
                adjacent_rotation = frame_match.transform.yaw_rad
            if stationary_mapping and self._last_grid_pose is None:
                # Before the first map keyframe there is no global geometric
                # anchor. Keep the user-defined startup direction instead of
                # accepting a symmetric-room ICP rotation as the origin yaw.
                fused_heading = self._scan_heading_yaw
            else:
                # On later stops the chassis may still coast for a few scans
                # after zero speed is sent. Continue estimating that real
                # residual rotation; the stationary stability gate below only
                # fuses scans after the pose has converged.
                predicted_heading = normalize_angle(
                    self._scan_heading_yaw + adjacent_rotation
                )
                fused_heading = predicted_heading
                if match_is_global:
                    correction = normalize_angle(
                        proposed_pose.yaw_rad - predicted_heading
                    )
                    maximum = math.radians(
                        self.mapping_config
                        .heading_submap_max_correction_deg
                    )
                    correction = max(
                        -maximum,
                        min(
                            maximum,
                            correction
                            * self.mapping_config
                            .heading_submap_correction_gain,
                        ),
                    )
                    fused_heading = normalize_angle(
                        predicted_heading + correction
                    )
            proposed_pose = Pose2D(
                proposed_pose.x_m,
                proposed_pose.y_m,
                fused_heading,
            )
            self.pose = self._regularize_manhattan_yaw(
                proposed_pose,
                points,
            )
            self._scan_heading_yaw = self.pose.yaw_rad
            self.previous_points = points
            self.previous_timestamp_s = scan.timestamp_s
            self.trajectory.append((
                scan.timestamp_s,
                self.pose.x_m,
                self.pose.y_m,
                self.pose.yaw_rad,
            ))
            self._linear_speed_samples.append(linear_speed_m_s)
            self._angular_speed_samples.append(angular_speed_rad_s)
            filtered_linear_speed = float(np.median(
                self._linear_speed_samples
            ))
            filtered_angular_speed = float(np.median(
                self._angular_speed_samples
            ))
            linear_speed_m_s = filtered_linear_speed
            angular_speed_rad_s = filtered_angular_speed
            map_integrated, map_status = self._stationary_map_update(
                mapping_points,
                scan.timestamp_s,
                stationary_mapping,
                match_rmse_m=match.rmse_m,
                match_inlier_ratio=match.inlier_ratio,
            )
            self._add_to_local_submap(points, self.pose)
            self._add_to_global_map(points, self.pose)
        elif stationary_mapping:
            # Never continue a fusion batch across a pose-rejected frame.
            # The next accepted scan must settle and start a fresh batch.
            self._reset_stationary_mapping()
            if rejection_reason == "stationary_rotation_jump":
                # A stopped single-wall scan can repeatedly fall into the same
                # wrong ICP minimum. Reseed adjacent/local references at the
                # held pose so the next unchanged scan can recover at zero
                # relative motion without corrupting the world map.
                self.previous_points = points
                self.previous_timestamp_s = scan.timestamp_s
                self._local_scans.clear()
                self._local_reference = None
                self._add_to_local_submap(points, self.pose)
        return SlamUpdate(
            self.pose,
            accepted,
            match.rmse_m,
            match.correspondences,
            len(points),
            match.inlier_ratio,
            translation_m,
            rotation_rad,
            rejection_reason,
            map_integrated,
            map_status,
            linear_speed_m_s,
            angular_speed_rad_s,
        )

    def reseed(self, scan):
        """停车后重建相邻帧参考，不改变全局位姿，也不写入地图。"""
        points = self._scan_points(scan)
        if len(points) < self.mapping_config.min_match_points:
            return False
        self.previous_points = points
        self.previous_timestamp_s = scan.timestamp_s
        self._local_scans.clear()
        self._local_reference = None
        self._reset_stationary_mapping()
        self._add_to_local_submap(points, self.pose)
        return True

    def relocalize(self, scan):
        """Relocalize a stopped vehicle against the long-term map.

        No occupancy cells are written here. The method only updates pose and
        rebuilds the rolling reference after a sufficiently reliable global
        match.
        """
        points = self._scan_points(scan)
        if (
            len(points) < self.mapping_config.min_match_points
            or self._global_reference is None
            or len(self._global_keyframes)
            < self.mapping_config.relocalization_min_keyframes
        ):
            return False
        match = self.matcher.match(
            self._global_reference,
            points,
            initial=self.pose,
        )
        correction = self.pose.inverse().compose(match.transform)
        reliable = (
            match.converged
            and match.rmse_m
            <= self.mapping_config.relocalization_max_rmse_m
            and match.inlier_ratio
            >= self.mapping_config.relocalization_min_inlier_ratio
            and math.hypot(correction.x_m, correction.y_m)
            <= self.mapping_config.relocalization_max_correction_m
            and abs(correction.yaw_rad)
            <= math.radians(
                self.mapping_config.relocalization_max_correction_deg
            )
        )
        if not reliable:
            return False
        self.pose = self._regularize_manhattan_yaw(
            match.transform,
            points,
        )
        self._scan_heading_yaw = self.pose.yaw_rad
        self.previous_points = points
        self.previous_timestamp_s = scan.timestamp_s
        self._local_scans.clear()
        self._local_reference = None
        self._reset_stationary_mapping()
        self._add_to_local_submap(points, self.pose)
        self._linear_speed_samples.clear()
        self._angular_speed_samples.clear()
        self.global_relocalization_count += 1
        return True

    def suspend_mapping(self, accepted_frames):
        self._map_hold_frames = max(
            self._map_hold_frames,
            max(0, int(accepted_frames)),
        )

    def _scan_points(self, scan):
        points = scan.points(
            self.lidar_config.min_distance_m,
            self.lidar_config.max_distance_m,
        )
        clustered = self._filter_dense_clusters(
            points,
            self.mapping_config.outlier_cluster_base_radius_m,
            self.mapping_config.outlier_cluster_radius_per_meter,
            self.mapping_config.outlier_cluster_max_radius_m,
            self.mapping_config.outlier_cluster_min_neighbors,
        )
        # In a very sparse scene, use the older one-neighbour filter rather
        # than discarding an otherwise usable complete frame.
        if len(clustered) >= self.mapping_config.min_match_points:
            points = clustered
        else:
            points = self._filter_isolated(
                points,
                self.mapping_config.outlier_neighbor_distance_m,
            )
        points = self._uniform_sample(
            points,
            self.mapping_config.max_scan_points,
        )
        return points + np.array((
            self.mapping_config.lidar_offset_x_m,
            self.mapping_config.lidar_offset_y_m,
        ))

    def save(self, output_prefix, rebuild=True):
        if rebuild:
            self.rebuild_map()
        pgm_path, yaml_path = self.grid.save(output_prefix)
        png_path = Path(output_prefix).with_suffix(".png")
        save_trajectory_png(
            self.grid,
            self.trajectory,
            png_path,
        )
        trajectory_path = Path(output_prefix).with_name(
            Path(output_prefix).name + "_trajectory.csv"
        )
        with trajectory_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("timestamp_s", "x_m", "y_m", "yaw_rad"))
            writer.writerows(self.trajectory)
        return pgm_path, yaml_path, trajectory_path, png_path

    def rebuild_map(self):
        """Replay replaceable keyframes into a clean occupancy grid.

        Unlike an append-only map, this deliberately forgets evidence from a
        superseded visit. Therefore free space may become unknown again and a
        wall may move when a newer observation replaces an older keyframe.
        """
        if not self._mapping_keyframes:
            return False
        rebuilt = self._new_grid()
        for timestamp_s, pose, local_points in self._mapping_keyframes:
            self._update_grid_from_local_points(
                rebuilt,
                local_points,
                pose,
                refresh_filter=False,
            )
        rebuilt.prune_floating_obstacles()
        self.grid = rebuilt
        timestamp_s, pose, _local_points = self._mapping_keyframes[-1]
        self._last_grid_pose = pose
        self._last_grid_timestamp_s = timestamp_s
        self.map_rebuild_count += 1
        return True

    def stationary_mapping_due(self):
        """Return whether a new stopped observation pose is required."""
        if self._last_grid_pose is None:
            return True
        relative = self._last_grid_pose.inverse().compose(self.pose)
        return (
            math.hypot(relative.x_m, relative.y_m)
            >= self.mapping_config.stationary_map_translation_m
            or abs(relative.yaw_rad)
            >= math.radians(
                self.mapping_config.stationary_map_rotation_deg
            )
        )

    @property
    def stationary_mapping_progress(self):
        required = max(
            1, int(self.mapping_config.stationary_map_fusion_scans)
        )
        return len(self._stationary_mapping_scans), required

    def _reset_stationary_mapping(self):
        self._stationary_mapping_active = False
        self._stationary_settle_left = 0
        self._stationary_mapping_scans = []
        self._stationary_mapping_poses = []

    def _stationary_map_update(
        self,
        local_points,
        timestamp_s,
        enabled,
        match_rmse_m=0.0,
        match_inlier_ratio=1.0,
    ):
        """Collect several stable scans and write one denoised keyframe."""
        if self._map_hold_frames > 0:
            self._map_hold_frames -= 1
            self._reset_stationary_mapping()
            return self._skip_map("relocalizing")
        if not enabled:
            self._reset_stationary_mapping()
            return self._skip_map("moving_localization_only")
        if (
            match_rmse_m
            > self.mapping_config.stationary_map_max_match_rmse_m
            or match_inlier_ratio
            < self.mapping_config
            .stationary_map_min_match_inlier_ratio
        ):
            self._stationary_mapping_scans = []
            self._stationary_mapping_poses = []
            self._stationary_settle_left = max(
                0,
                int(self.mapping_config.stationary_map_settle_scans),
            )
            return self._skip_map("stationary_match_unreliable")

        if not self._stationary_mapping_active:
            self._stationary_mapping_active = True
            self._stationary_settle_left = max(
                0,
                int(self.mapping_config.stationary_map_settle_scans),
            )
            self._stationary_mapping_scans = []
            self._stationary_mapping_poses = []

        if self._stationary_settle_left > 0:
            self._stationary_settle_left -= 1
            return self._skip_map("stationary_settling")

        if self._stationary_mapping_poses:
            anchor = self._stationary_mapping_poses[0]
            relative = anchor.inverse().compose(self.pose)
            unstable = (
                math.hypot(relative.x_m, relative.y_m)
                > self.mapping_config.stationary_map_max_pose_translation_m
                or abs(relative.yaw_rad)
                > math.radians(
                    self.mapping_config.stationary_map_max_pose_rotation_deg
                )
            )
            if unstable:
                self._stationary_mapping_scans = []
                self._stationary_mapping_poses = []
                self._stationary_settle_left = max(
                    0,
                    int(
                        self.mapping_config.stationary_map_settle_scans
                    ),
                )
                return self._skip_map("stationary_unstable")

        self._stationary_mapping_scans.append(
            np.asarray(local_points, dtype=np.float64).copy()
        )
        self._stationary_mapping_poses.append(self.pose)
        required = max(
            1, int(self.mapping_config.stationary_map_fusion_scans)
        )
        if len(self._stationary_mapping_scans) < required:
            return self._skip_map("stationary_collecting")

        fused_points = self._fuse_stationary_scans(
            self._stationary_mapping_scans
        )
        fused_pose = self._median_stationary_pose(
            self._stationary_mapping_poses
        )
        self.stationary_fusion_input_points += sum(
            len(points) for points in self._stationary_mapping_scans
        )
        self.stationary_fusion_output_points += len(fused_points)
        self._reset_stationary_mapping()
        if len(fused_points) < self.mapping_config.min_match_points:
            return self._skip_map("stationary_fusion_sparse")

        integrated, status = self._integrate(
            fused_points,
            timestamp_s,
            force=True,
            pose=fused_pose,
        )
        if integrated:
            self.stationary_fusion_count += 1
            return True, "stationary_fused"
        return integrated, status

    def _fuse_stationary_scans(self, scans):
        """Keep angular returns supported by several independent scans."""
        sensor_offset = np.asarray((
            self.mapping_config.lidar_offset_x_m,
            self.mapping_config.lidar_offset_y_m,
        ), dtype=np.float64)
        bin_width = math.radians(max(
            0.1,
            float(self.mapping_config.stationary_map_angle_bin_deg),
        ))
        maximum_range = self.mapping_config.mapping_max_distance_m
        clear_range = maximum_range + max(
            0.05,
            self.mapping_config.stationary_map_range_tolerance_m,
        )
        by_bin = {}
        for frame_index, points in enumerate(scans):
            relative = np.asarray(points, dtype=np.float64) - sensor_offset
            ranges = np.linalg.norm(relative, axis=1)
            angles = np.arctan2(relative[:, 1], relative[:, 0])
            finite = (
                np.isfinite(ranges)
                & np.isfinite(angles)
                & (ranges > 0.0)
            )
            frame_bins = {}
            for angle, distance in zip(angles[finite], ranges[finite]):
                bin_index = int(round(float(angle) / bin_width))
                frame_bins.setdefault(bin_index, []).append(
                    min(float(distance), clear_range)
                )
            for bin_index, values in frame_bins.items():
                by_bin.setdefault(bin_index, []).append((
                    frame_index,
                    float(np.median(values)),
                ))

        minimum_support = min(
            len(scans),
            max(
                1,
                int(
                    self.mapping_config
                    .stationary_map_min_support_scans
                ),
            ),
        )
        tolerance = max(
            0.01,
            float(
                self.mapping_config.stationary_map_range_tolerance_m
            ),
        )
        fused = []
        for bin_index, observations in by_bin.items():
            distances = np.asarray(
                [item[1] for item in observations],
                dtype=np.float64,
            )
            median_distance = float(np.median(distances))
            supported = distances[
                np.abs(distances - median_distance) <= tolerance
            ]
            if len(supported) < minimum_support:
                continue
            distance = float(np.median(supported))
            angle = bin_index * bin_width
            fused.append(
                sensor_offset
                + distance * np.asarray((
                    math.cos(angle),
                    math.sin(angle),
                ))
            )
        if not fused:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(fused, dtype=np.float64)

    @staticmethod
    def _median_stationary_pose(poses):
        x_m = float(np.median([pose.x_m for pose in poses]))
        y_m = float(np.median([pose.y_m for pose in poses]))
        sine = float(np.median([
            math.sin(pose.yaw_rad) for pose in poses
        ]))
        cosine = float(np.median([
            math.cos(pose.yaw_rad) for pose in poses
        ]))
        return Pose2D(x_m, y_m, math.atan2(sine, cosine))

    def _integrate(
        self,
        local_points,
        timestamp_s,
        linear_speed_m_s=0.0,
        angular_speed_rad_s=0.0,
        force=False,
        pose=None,
    ):
        integration_pose = self.pose if pose is None else pose
        if not force:
            if self._map_hold_frames > 0:
                self._map_hold_frames -= 1
                return self._skip_map("relocalizing")
            if (
                linear_speed_m_s
                < self.mapping_config.map_min_linear_speed_m_s
            ):
                return self._skip_map("stationary")
            if (
                linear_speed_m_s
                > self.mapping_config.map_max_linear_speed_m_s
            ):
                return self._skip_map("too_fast")
            if (
                angular_speed_rad_s
                > self.mapping_config.map_max_angular_speed_rad_s
            ):
                return self._skip_map("turning_too_fast")

        should_integrate = self._last_grid_pose is None
        if self._last_grid_pose is not None:
            relative = self._last_grid_pose.inverse().compose(
                integration_pose
            )
            elapsed_s = timestamp_s - self._last_grid_timestamp_s
            should_integrate = (
                math.hypot(relative.x_m, relative.y_m)
                >= self.mapping_config.map_keyframe_translation_m
                or abs(relative.yaw_rad)
                >= math.radians(
                    self.mapping_config.map_keyframe_rotation_deg
                )
                or elapsed_s
                >= self.mapping_config.map_keyframe_max_interval_s
            )
        if not should_integrate:
            return self._skip_map("keyframe_wait")
        self._update_grid_from_local_points(
            self.grid,
            local_points,
            integration_pose,
        )
        self.grid.prune_floating_obstacles()
        self._store_mapping_keyframe(
            local_points,
            timestamp_s,
            integration_pose,
        )
        self._last_grid_pose = integration_pose
        self._last_grid_timestamp_s = timestamp_s
        self.map_integrated_count += 1
        return True, "integrated"

    def _update_grid_from_local_points(
        self,
        grid,
        local_points,
        pose,
        refresh_filter=True,
    ):
        sensor_local = np.array(((
            self.mapping_config.lidar_offset_x_m,
            self.mapping_config.lidar_offset_y_m,
        ),))
        sensor_offset = sensor_local[0]
        relative_points = local_points - sensor_offset
        ranges = np.linalg.norm(relative_points, axis=1)
        maximum_range = self.mapping_config.mapping_max_distance_m
        near = ranges <= maximum_range
        mapping_hits = local_points[near]
        far_vectors = relative_points[~near]
        far_ranges = ranges[~near]
        if len(far_vectors):
            clear_endpoints = (
                sensor_offset
                + far_vectors
                * (maximum_range / far_ranges)[:, None]
            )
        else:
            clear_endpoints = np.empty((0, 2), dtype=np.float64)
        world_points = pose.transform_points(mapping_hits)
        clear_world_points = pose.transform_points(clear_endpoints)
        sensor_world = pose.transform_points(sensor_local)[0]
        grid.update(
            sensor_world,
            world_points,
            clear_points_m=clear_world_points,
            refresh_filter=refresh_filter,
            occupied_evidence_scale=(
                self.mapping_config
                .stationary_map_occupied_evidence_scale
            ),
        )

    def _store_mapping_keyframe(self, local_points, timestamp_s, pose):
        radius_m = self.mapping_config.map_revisit_replace_radius_m
        yaw_limit = math.radians(
            self.mapping_config.map_revisit_replace_yaw_deg
        )
        matching_indices = []
        for index, old_frame in enumerate(self._mapping_keyframes):
            _old_time, old_pose, _old_points = old_frame
            distance_m = math.hypot(
                old_pose.x_m - pose.x_m,
                old_pose.y_m - pose.y_m,
            )
            yaw_error = abs(normalize_angle(
                old_pose.yaw_rad - pose.yaw_rad
            ))
            if distance_m <= radius_m and yaw_error <= yaw_limit:
                matching_indices.append(index)
        retained = list(self._mapping_keyframes)
        replaced = 0
        evidence_frames = max(
            2, int(self.mapping_config.map_revisit_evidence_frames)
        )
        if len(matching_indices) >= evidence_frames:
            # Retire only the oldest observation in this direction. Repeated
            # scans must build a full evidence window before all older wall
            # placement can disappear from a rebuilt map.
            del retained[matching_indices[0]]
            replaced = 1
        retained.append((
            float(timestamp_s),
            pose,
            np.asarray(local_points, dtype=np.float64).copy(),
        ))
        maximum = max(
            1, int(self.mapping_config.map_rebuild_max_keyframes)
        )
        if len(retained) > maximum:
            retained = retained[-maximum:]
        self._mapping_keyframes = retained
        self.map_keyframes_replaced += replaced

    def _skip_map(self, reason):
        self.map_skip_counts[reason] = (
            self.map_skip_counts.get(reason, 0) + 1
        )
        return False, reason

    def mapping_summary(self):
        parts = ["integrated={}".format(self.map_integrated_count)]
        for reason in (
            "relocalizing",
            "moving_localization_only",
            "stationary_settling",
            "stationary_collecting",
            "stationary_unstable",
            "stationary_fusion_sparse",
        ):
            parts.append("{}={}".format(
                reason,
                self.map_skip_counts.get(reason, 0),
            ))
        parts.append("stationary_fusions={}".format(
            self.stationary_fusion_count
        ))
        parts.append("fusion_points={}/{}".format(
            self.stationary_fusion_output_points,
            self.stationary_fusion_input_points,
        ))
        parts.append("small_components_hidden={}".format(
            self.grid.filtered_small_components
        ))
        parts.append("small_cells_hidden={}".format(
            self.grid.filtered_small_cells
        ))
        parts.append("render_noise_hidden={}".format(
            self.grid.render_filtered_noise_cells
        ))
        parts.append("enclosed_unknown_filled={}".format(
            self.grid.render_filled_enclosed_cells
        ))
        parts.append("global_relocalizations={}".format(
            self.global_relocalization_count
        ))
        parts.append("manhattan_corrections={}".format(
            self.manhattan_correction_count
        ))
        parts.append("map_rebuilds={}".format(
            self.map_rebuild_count
        ))
        parts.append("map_keyframes={}".format(
            len(self._mapping_keyframes)
        ))
        parts.append("map_keyframes_replaced={}".format(
            self.map_keyframes_replaced
        ))
        return " ".join(parts)

    def _local_match_is_reliable(self, match):
        return (
            match.converged
            and match.rmse_m <= self.mapping_config.local_submap_max_rmse_m
            and match.inlier_ratio
            >= self.mapping_config.local_submap_min_inlier_ratio
        )

    def _add_to_local_submap(self, local_points, pose):
        self._local_scans.append(pose.transform_points(local_points))
        combined = np.vstack(tuple(self._local_scans))
        voxel = self.mapping_config.local_submap_voxel_m
        if voxel > 0 and len(combined):
            cells = np.floor(combined / voxel).astype(np.int64)
            _unique, indices = np.unique(cells, axis=0, return_index=True)
            combined = combined[np.sort(indices)]
        self._local_reference = self._uniform_sample(
            combined,
            self.mapping_config.local_submap_max_points,
        )

    def _add_to_global_map(self, local_points, pose, force=False):
        if not force and self._last_global_keyframe_pose is not None:
            relative = self._last_global_keyframe_pose.inverse().compose(pose)
            if (
                math.hypot(relative.x_m, relative.y_m)
                < self.mapping_config.global_keyframe_translation_m
                and abs(relative.yaw_rad)
                < math.radians(
                    self.mapping_config.global_keyframe_rotation_deg
                )
            ):
                return
        self._global_keyframes.append(pose.transform_points(local_points))
        self._last_global_keyframe_pose = pose
        combined = np.vstack(tuple(self._global_keyframes))
        voxel = self.mapping_config.global_map_voxel_m
        if voxel > 0 and len(combined):
            cells = np.floor(combined / voxel).astype(np.int64)
            _unique, indices = np.unique(cells, axis=0, return_index=True)
            combined = combined[np.sort(indices)]
        self._global_reference = self._uniform_sample(
            combined,
            self.mapping_config.global_map_max_points,
        )

    def _regularize_manhattan_yaw(self, pose, local_points):
        """Apply a small yaw correction when repeated wall evidence agrees.

        The room axis is learned from several scans. Corrections are capped
        below one degree per frame, so chairs and short furniture edges cannot
        abruptly rotate the complete map.
        """
        if not self.mapping_config.manhattan_enabled:
            return pose
        axis, confidence, segment_count = self._dominant_manhattan_axis(
            local_points
        )
        if (
            segment_count < self.mapping_config.manhattan_min_segments
            or confidence < self.mapping_config.manhattan_min_confidence
        ):
            return pose
        world_axis = self._quarter_turn_residual(axis + pose.yaw_rad)
        if self._manhattan_axis_yaw is None:
            self._manhattan_observations.append(world_axis)
            if (
                len(self._manhattan_observations)
                < self._manhattan_observations.maxlen
            ):
                return pose
            self._manhattan_axis_yaw = self._quarter_turn_mean(
                self._manhattan_observations
            )
            return pose

        error = self._quarter_turn_residual(
            self._manhattan_axis_yaw - world_axis
        )
        if abs(error) > math.radians(
            self.mapping_config.manhattan_max_error_deg
        ):
            return pose
        correction = error * self.mapping_config.manhattan_correction_gain
        maximum = math.radians(
            self.mapping_config.manhattan_max_correction_deg
        )
        correction = max(-maximum, min(maximum, correction))
        if abs(correction) < math.radians(0.03):
            return pose
        self.manhattan_correction_count += 1
        return Pose2D(
            pose.x_m,
            pose.y_m,
            normalize_angle(pose.yaw_rad + correction),
        )

    @staticmethod
    def _dominant_manhattan_axis(points):
        points = np.asarray(points, dtype=np.float64)
        if len(points) < 2:
            return 0.0, 0.0, 0
        differences = np.diff(points, axis=0)
        lengths = np.linalg.norm(differences, axis=1)
        keep = (lengths >= 0.015) & (lengths <= 0.30)
        differences = differences[keep]
        lengths = lengths[keep]
        if not len(differences):
            return 0.0, 0.0, 0
        angles = np.arctan2(differences[:, 1], differences[:, 0])
        cosine = float(np.sum(lengths * np.cos(4.0 * angles)))
        sine = float(np.sum(lengths * np.sin(4.0 * angles)))
        total = float(np.sum(lengths))
        confidence = math.hypot(cosine, sine) / max(1e-9, total)
        axis = 0.25 * math.atan2(sine, cosine)
        return axis, confidence, len(angles)

    @staticmethod
    def _quarter_turn_residual(angle):
        quarter = math.pi / 2.0
        return (angle + quarter / 2.0) % quarter - quarter / 2.0

    @staticmethod
    def _quarter_turn_mean(angles):
        cosine = sum(math.cos(4.0 * angle) for angle in angles)
        sine = sum(math.sin(4.0 * angle) for angle in angles)
        return 0.25 * math.atan2(sine, cosine)

    def _constrain_global_pure_rotation(self, match, current):
        constrained = Pose2D(
            self.pose.x_m,
            self.pose.y_m,
            match.transform.yaw_rad,
        )
        rmse, count, ratio = self.matcher.evaluate(
            self._local_reference,
            current,
            constrained,
        )
        return MatchResult(
            constrained,
            rmse,
            count,
            math.isfinite(rmse),
            ratio,
        )

    def _constrain_pure_rotation(self, match, reference, current):
        """原地转向时限制部分视场 ICP 虚构出的平移。"""
        limit = self.mapping_config.pure_rotation_max_translation_m
        distance = math.hypot(
            match.transform.x_m,
            match.transform.y_m,
        )
        if distance <= limit or distance <= 1e-9:
            return match
        scale = limit / distance
        constrained = Pose2D(
            match.transform.x_m * scale,
            match.transform.y_m * scale,
            match.transform.yaw_rad,
        )
        rmse, count, ratio = self.matcher.evaluate(
            reference,
            current,
            constrained,
        )
        return MatchResult(
            constrained,
            rmse,
            count,
            match.converged,
            ratio,
        )

    @staticmethod
    def _uniform_sample(points, maximum):
        if len(points) <= maximum:
            return points
        indices = np.linspace(0, len(points) - 1, maximum, dtype=int)
        return points[indices]

    @staticmethod
    def _filter_isolated(points, maximum_neighbor_distance_m):
        """删除两侧都没有近邻的孤立回波。"""
        points = np.asarray(points, dtype=np.float64)
        if len(points) < 3 or maximum_neighbor_distance_m <= 0:
            return points
        previous_distance = np.linalg.norm(
            points - np.roll(points, 1, axis=0),
            axis=1,
        )
        next_distance = np.linalg.norm(
            points - np.roll(points, -1, axis=0),
            axis=1,
        )
        keep = (
            (previous_distance <= maximum_neighbor_distance_m)
            | (next_distance <= maximum_neighbor_distance_m)
        )
        return points[keep]

    @staticmethod
    def _filter_dense_clusters(
        points,
        base_radius_m,
        radius_per_meter,
        maximum_radius_m,
        minimum_neighbors,
    ):
        """保留有局部点簇支撑的回波，滤掉单点、双点毛刺和射线端点。"""
        points = np.asarray(points, dtype=np.float64)
        minimum_neighbors = max(0, int(minimum_neighbors))
        if len(points) < minimum_neighbors + 1:
            return points
        ranges = np.linalg.norm(points, axis=1)
        radii = np.minimum(
            float(maximum_radius_m),
            float(base_radius_m) + float(radius_per_meter) * ranges,
        )
        differences = points[:, None, :] - points[None, :, :]
        squared = np.einsum("ijk,ijk->ij", differences, differences)
        nearby = squared <= radii[:, None] ** 2
        neighbor_counts = np.count_nonzero(nearby, axis=1) - 1
        return points[neighbor_counts >= minimum_neighbors]
