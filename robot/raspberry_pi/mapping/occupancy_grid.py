"""二维对数概率占据栅格及 ROS 兼容 PGM/YAML 导出。"""

from pathlib import Path
import math

import numpy as np


class OccupancyGrid:
    def __init__(
        self,
        width_cells=400,
        height_cells=400,
        resolution_m=0.05,
        origin_x_m=None,
        origin_y_m=None,
        free_log_odds=-0.38,
        occupied_log_odds=0.45,
        occupied_inflation_cells=0,
        ray_block_log_odds=0.75,
        ray_block_inflation_cells=1,
        contradiction_clear_hits=10,
        auto_expand=False,
        expansion_margin_m=1.0,
        min_obstacle_area_m2=0.03,
        render_wall_gap_max_m=0.10,
        render_wall_support_m=0.10,
        render_parallel_wall_merge_max_m=0.20,
        render_parallel_wall_min_span_m=1.80,
        render_contour_close_max_span_m=1.60,
        render_contour_min_side_support=0.35,
    ):
        if width_cells <= 0 or height_cells <= 0 or resolution_m <= 0:
            raise ValueError("map dimensions and resolution must be positive")
        self.width = int(width_cells)
        self.height = int(height_cells)
        self.resolution_m = float(resolution_m)
        self.origin_x_m = (
            -self.width * self.resolution_m / 2.0 if origin_x_m is None else origin_x_m
        )
        self.origin_y_m = (
            -self.height * self.resolution_m / 2.0 if origin_y_m is None else origin_y_m
        )
        self.free_log_odds = float(free_log_odds)
        self.occupied_log_odds = float(occupied_log_odds)
        self.occupied_inflation_cells = max(
            0, int(occupied_inflation_cells)
        )
        self.ray_block_log_odds = float(ray_block_log_odds)
        self.ray_block_inflation_cells = max(
            0, int(ray_block_inflation_cells)
        )
        self.contradiction_clear_hits = max(
            1, int(contradiction_clear_hits)
        )
        self.auto_expand = bool(auto_expand)
        self.expansion_margin_cells = max(
            0,
            int(math.ceil(
                float(expansion_margin_m) / self.resolution_m
            )),
        )
        self.min_obstacle_area_m2 = max(
            0.0, float(min_obstacle_area_m2)
        )
        self.render_wall_gap_max_cells = max(
            0,
            int(round(
                float(render_wall_gap_max_m) / self.resolution_m
            )),
        )
        self.render_wall_support_cells = max(
            1,
            int(round(
                float(render_wall_support_m) / self.resolution_m
            )),
        )
        self.render_parallel_wall_merge_cells = max(
            0,
            int(math.ceil(
                float(render_parallel_wall_merge_max_m)
                / self.resolution_m
            )),
        )
        self.render_parallel_wall_min_span_cells = max(
            2,
            int(math.ceil(
                float(render_parallel_wall_min_span_m)
                / self.resolution_m
            )),
        )
        self.render_contour_close_max_span_cells = max(
            0,
            int(math.ceil(
                float(render_contour_close_max_span_m)
                / self.resolution_m
            )),
        )
        self.render_contour_min_side_support = min(
            1.0,
            max(0.0, float(render_contour_min_side_support)),
        )
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)
        self.observed = np.zeros((self.height, self.width), dtype=np.bool_)
        self._significant_obstacles = np.zeros(
            (self.height, self.width),
            dtype=np.bool_,
        )
        self._contradiction_counts = np.zeros(
            (self.height, self.width),
            dtype=np.uint8,
        )
        self._contradiction_last_epoch = np.zeros(
            (self.height, self.width),
            dtype=np.uint32,
        )
        self._contradiction_epoch = 0
        self.filtered_small_components = 0
        self.filtered_small_cells = 0
        self.render_filtered_noise_cells = 0
        self.render_filled_enclosed_cells = 0
        self.render_parallel_wall_cells_merged = 0
        self.render_contour_cells_closed = 0
        self._render_obstacle_cache = None
        self._structural_object_anchors = []
        self._obstacle_filter_dirty = True

    def register_structural_object(self, anchor_xy_m, support_views=1):
        """Register a compact object backed by independent stopped views."""
        if int(support_views) < 1:
            return False
        anchor = (float(anchor_xy_m[0]), float(anchor_xy_m[1]))
        for index, old in enumerate(self._structural_object_anchors):
            if math.hypot(anchor[0] - old[0], anchor[1] - old[1]) > 0.80:
                continue
            moved = math.hypot(anchor[0] - old[0], anchor[1] - old[1])
            self._structural_object_anchors[index] = anchor
            if moved >= 0.01:
                self._render_obstacle_cache = None
            return moved >= 0.01
        self._structural_object_anchors.append(anchor)
        self._render_obstacle_cache = None
        return True

    def world_to_grid(self, points):
        points = np.asarray(points, dtype=np.float64)
        columns = np.floor((points[..., 0] - self.origin_x_m) / self.resolution_m).astype(int)
        rows = np.floor((points[..., 1] - self.origin_y_m) / self.resolution_m).astype(int)
        return np.stack((columns, rows), axis=-1)

    def in_bounds(self, column, row):
        return 0 <= column < self.width and 0 <= row < self.height

    def update(
        self,
        sensor_xy_m,
        hit_points_m,
        clear_points_m=None,
        refresh_filter=True,
        free_evidence_scale=1.0,
        occupied_evidence_scale=1.0,
    ):
        """将一圈雷达射线融合进栅格；超出地图的射线会被安全裁掉。"""
        self._advance_contradiction_epoch()
        if self.auto_expand:
            bounds_points = [
                np.asarray((sensor_xy_m,), dtype=np.float64),
                np.asarray(
                    hit_points_m,
                    dtype=np.float64,
                ).reshape(-1, 2),
            ]
            if clear_points_m is not None:
                bounds_points.append(np.asarray(
                    clear_points_m,
                    dtype=np.float64,
                ).reshape(-1, 2))
            self.expand_to_include(np.vstack(bounds_points))
        sensor_cell = self.world_to_grid(np.asarray((sensor_xy_m,), dtype=np.float64))[0]
        if not self.in_bounds(int(sensor_cell[0]), int(sensor_cell[1])):
            return
        hit_cells = self.world_to_grid(hit_points_m)
        free_columns = []
        free_rows = []
        occupied_columns = []
        occupied_rows = []
        start = (int(sensor_cell[0]), int(sensor_cell[1]))
        for endpoint in hit_cells:
            end = (int(endpoint[0]), int(endpoint[1]))
            cells = self._bresenham(start, end)
            if not cells:
                continue
            endpoint_inside = self.in_bounds(*end)
            free_part = cells[:-1] if endpoint_inside else cells
            blocked = False
            for column, row in free_part:
                if self.in_bounds(column, row):
                    endpoint_inflation = (
                        endpoint_inside
                        and max(
                            abs(column - end[0]),
                            abs(row - end[1]),
                        )
                        <= self.ray_block_inflation_cells
                    )
                    # A static obstacle confirmed by repeated hits blocks
                    # later inconsistent long rays. Do not clear cells behind
                    # a table/wall because of an occasional far-range echo.
                    if (
                        (column, row) != start
                        and not endpoint_inflation
                        and self._confirmed_obstacle_near(column, row)
                    ):
                        if not self._register_free_contradiction(
                            column, row
                        ):
                            blocked = True
                            break
                    free_columns.append(column)
                    free_rows.append(row)
            if endpoint_inside and not blocked:
                radius = self.occupied_inflation_cells
                for row in range(end[1] - radius, end[1] + radius + 1):
                    for column in range(
                        end[0] - radius,
                        end[0] + radius + 1,
                    ):
                        if self.in_bounds(column, row):
                            occupied_columns.append(column)
                            occupied_rows.append(row)

        if clear_points_m is not None:
            clear_cells = self.world_to_grid(clear_points_m)
            for endpoint in clear_cells:
                end = (int(endpoint[0]), int(endpoint[1]))
                for column, row in self._bresenham(start, end):
                    if not self.in_bounds(column, row):
                        continue
                    if (
                        (column, row) != start
                        and self._confirmed_obstacle_near(column, row)
                    ):
                        if not self._register_free_contradiction(
                            column, row
                        ):
                            break
                    free_columns.append(column)
                    free_rows.append(row)

        if free_columns:
            np.add.at(
                self.log_odds,
                (free_rows, free_columns),
                self.free_log_odds * max(
                    0.0, float(free_evidence_scale)
                ),
            )
            self.observed[free_rows, free_columns] = True
        if occupied_columns:
            # Adjacent beams and endpoint inflation frequently land in the
            # same grid cell. They belong to one lidar revolution, not to
            # independent observations, so that cell may receive occupied
            # evidence only once per update. Without this de-duplication a
            # single radial packet artifact can become a high-confidence wall.
            occupied_cells = np.unique(
                np.column_stack((occupied_rows, occupied_columns)),
                axis=0,
            )
            unique_rows = occupied_cells[:, 0]
            unique_columns = occupied_cells[:, 1]
            self.log_odds[unique_rows, unique_columns] += (
                self.occupied_log_odds * max(
                    0.0, float(occupied_evidence_scale)
                )
            )
            self.observed[unique_rows, unique_columns] = True
            self._contradiction_counts[
                unique_rows, unique_columns
            ] = 0
        np.clip(self.log_odds, -4.0, 4.0, out=self.log_odds)
        self._render_obstacle_cache = None
        self._obstacle_filter_dirty = True
        if refresh_filter:
            self.refresh_obstacle_filter()

    def _confirmed_obstacle_near(self, column, row):
        radius = self.ray_block_inflation_cells
        column_start = max(0, column - radius)
        column_end = min(self.width, column + radius + 1)
        row_start = max(0, row - radius)
        row_end = min(self.height, row + radius + 1)
        return bool(np.any(
            self._significant_obstacles[
                row_start:row_end,
                column_start:column_end,
            ]
        ))

    def _register_free_contradiction(self, column, row):
        """Let repeated free rays retire a formerly confirmed obstacle.

        A single inconsistent long echo still stops at the old wall.  Only
        repeated contradictions clear the conflicting occupied cells, after
        which later cells on the ray may be observed normally.
        """
        radius = self.ray_block_inflation_cells
        column_start = max(0, column - radius)
        column_end = min(self.width, column + radius + 1)
        row_start = max(0, row - radius)
        row_end = min(self.height, row + radius + 1)
        obstacle_region = self._significant_obstacles[
            row_start:row_end,
            column_start:column_end,
        ]
        obstacle_rows, obstacle_columns = np.nonzero(obstacle_region)
        if not len(obstacle_rows):
            return True
        obstacle_rows = obstacle_rows + row_start
        obstacle_columns = obstacle_columns + column_start
        counts = self._contradiction_counts[
            obstacle_rows, obstacle_columns
        ].astype(np.uint16)
        last_epochs = self._contradiction_last_epoch[
            obstacle_rows, obstacle_columns
        ]
        fresh = last_epochs != self._contradiction_epoch
        counts[fresh] = np.minimum(counts[fresh] + 1, 255)
        self._contradiction_counts[
            obstacle_rows, obstacle_columns
        ] = counts.astype(np.uint8)
        self._contradiction_last_epoch[
            obstacle_rows[fresh], obstacle_columns[fresh]
        ] = self._contradiction_epoch
        clear = counts >= self.contradiction_clear_hits
        if np.any(clear):
            clear_rows = obstacle_rows[clear]
            clear_columns = obstacle_columns[clear]
            self.log_odds[clear_rows, clear_columns] = min(
                self.free_log_odds,
                -0.01,
            )
            self.observed[clear_rows, clear_columns] = True
            self._significant_obstacles[
                clear_rows, clear_columns
            ] = False
            self._contradiction_counts[
                clear_rows, clear_columns
            ] = 0
            self._obstacle_filter_dirty = True
        remaining = self._significant_obstacles[
            row_start:row_end,
            column_start:column_end,
        ]
        return not bool(np.any(remaining))

    def _advance_contradiction_epoch(self):
        self._contradiction_epoch = (
            self._contradiction_epoch + 1
        ) & 0xFFFFFFFF
        if self._contradiction_epoch == 0:
            self._contradiction_last_epoch.fill(0)
            self._contradiction_epoch = 1

    def expand_to_include(self, points_m):
        """Grow all grid layers without changing existing world positions."""
        points = np.asarray(points_m, dtype=np.float64).reshape(-1, 2)
        if not len(points):
            return False
        points = points[np.all(np.isfinite(points), axis=1)]
        if not len(points):
            return False
        cells = self.world_to_grid(points)
        margin = self.expansion_margin_cells
        left = max(0, margin - int(np.min(cells[:, 0])))
        bottom = max(0, margin - int(np.min(cells[:, 1])))
        right = max(
            0,
            int(np.max(cells[:, 0])) + margin - (self.width - 1),
        )
        top = max(
            0,
            int(np.max(cells[:, 1])) + margin - (self.height - 1),
        )
        if not any((left, right, bottom, top)):
            return False
        padding = ((bottom, top), (left, right))
        self.log_odds = np.pad(self.log_odds, padding)
        self.observed = np.pad(self.observed, padding)
        self._significant_obstacles = np.pad(
            self._significant_obstacles,
            padding,
        )
        self._contradiction_counts = np.pad(
            self._contradiction_counts,
            padding,
        )
        self._contradiction_last_epoch = np.pad(
            self._contradiction_last_epoch,
            padding,
        )
        self.width += left + right
        self.height += bottom + top
        self.origin_x_m -= left * self.resolution_m
        self.origin_y_m -= bottom * self.resolution_m
        self._obstacle_filter_dirty = True
        self._render_obstacle_cache = None
        return True

    def probabilities(self):
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def significant_obstacles(self, occupied_probability=0.65):
        if occupied_probability == 0.65:
            if self._obstacle_filter_dirty:
                self.refresh_obstacle_filter()
            return self._significant_obstacles.copy()
        occupied = self.observed & (
            self.probabilities() >= occupied_probability
        )
        significant, _components, _cells = (
            self._filter_small_components(occupied)
        )
        return significant

    def refresh_obstacle_filter(self):
        occupied = self.observed & (self.probabilities() >= 0.65)
        (
            self._significant_obstacles,
            self.filtered_small_components,
            self.filtered_small_cells,
        ) = self._filter_small_components(occupied)
        self._obstacle_filter_dirty = False
        return self.filtered_small_cells

    def _filter_small_components(self, occupied):
        minimum_cells = max(
            1,
            int(np.ceil(
                self.min_obstacle_area_m2
                / (self.resolution_m * self.resolution_m)
            )),
        )
        if minimum_cells <= 1:
            return occupied.copy(), 0, 0
        remaining = set(map(tuple, np.argwhere(occupied)))
        significant = np.zeros_like(occupied)
        removed_components = 0
        removed_cells = 0
        while remaining:
            start = remaining.pop()
            component = [start]
            queue = [start]
            while queue:
                row, column = queue.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        neighbor = (row + dy, column + dx)
                        if neighbor not in remaining:
                            continue
                        remaining.remove(neighbor)
                        component.append(neighbor)
                        queue.append(neighbor)
            if len(component) >= minimum_cells:
                rows, columns = zip(*component)
                significant[rows, columns] = True
            else:
                removed_components += 1
                removed_cells += len(component)
        return significant, removed_components, removed_cells

    def floating_obstacle_mask(
        self,
        occupied_probability=0.65,
        free_probability=0.35,
    ):
        """Find zero-thickness occupied cells observed free on both sides."""
        probabilities = self.probabilities()
        occupied = self.observed & (
            probabilities > occupied_probability
        )
        free = self.observed & (probabilities < free_probability)
        padded = np.pad(free, 1, constant_values=False)
        left = padded[1:-1, :-2]
        right = padded[1:-1, 2:]
        up = padded[2:, 1:-1]
        down = padded[:-2, 1:-1]
        up_left = padded[2:, :-2]
        down_right = padded[:-2, 2:]
        up_right = padded[2:, 2:]
        down_left = padded[:-2, :-2]
        free_on_opposite_sides = (
            (left & right)
            | (up & down)
            | (up_left & down_right)
            | (up_right & down_left)
        )
        return occupied & free_on_opposite_sides

    def prune_floating_obstacles(self):
        """Remove floating lines but preserve walls backed by unknown space."""
        floating = self.floating_obstacle_mask()
        if np.any(floating):
            self.log_odds[floating] = min(
                -1.0,
                self.free_log_odds,
            )
            self.observed[floating] = True
            self._render_obstacle_cache = None
            self._obstacle_filter_dirty = True
        self.refresh_obstacle_filter()
        return int(np.count_nonzero(floating))

    def render_image(self):
        """Build a clean three-state image before the PGM y-axis flip.

        Weak one-off returns and small occupied components are removed before
        short wall gaps are bridged.  This order prevents scattered scan dots
        from being promoted into apparent walls by the output cleanup.
        """
        probabilities = self.probabilities()
        image = np.full(probabilities.shape, 205, dtype=np.uint8)
        image[self.observed] = 254
        occupied = self.render_obstacle_mask(probabilities)
        image[occupied] = 0
        return image

    def render_obstacle_mask(self, probabilities=None):
        """Return the cleaned occupied geometry used by PNG/PGM output.

        The probability grid remains untouched. This lets localization and
        planning retain raw evidence while the displayed geometry can merge
        duplicate long wall bands and close a strongly supported compact
        outline. Both structural operations are deliberately evidence-gated.
        """
        if self._render_obstacle_cache is not None:
            return self._render_obstacle_cache.copy()
        if probabilities is None:
            probabilities = self.probabilities()
        occupied = self.significant_obstacles()
        neighbor_count = np.zeros_like(
            self.log_odds,
            dtype=np.uint8,
        )
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                source_y0 = max(0, -dy)
                source_y1 = min(self.height, self.height - dy)
                source_x0 = max(0, -dx)
                source_x1 = min(self.width, self.width - dx)
                neighbor_count[
                    source_y0 + dy:source_y1 + dy,
                    source_x0 + dx:source_x1 + dx,
                ] += occupied[
                    source_y0:source_y1,
                    source_x0:source_x1,
                ]
        occupied = occupied & (
            (neighbor_count >= 2) | (probabilities > 0.85)
        )
        self.render_filtered_noise_cells = int(np.count_nonzero(
            self.observed
            & (probabilities >= 0.50)
            & ~occupied
        ))
        occupied = self._bridge_short_wall_gaps(
            occupied,
            self.observed & (probabilities < 0.35),
        )
        before_wall_consolidation = occupied.copy()
        axis_angle = self._dominant_manhattan_axis(occupied)
        # One pass can turn two overlapping drift layers into two staggered
        # long fragments.  Iterate to a fixed point so that those fragments are
        # aligned on the following pass.  Every pass retains the same long-span
        # gates, so compact real protrusions and cabinet recesses remain out of
        # scope.  Four passes are enough for the two Manhattan directions and
        # bound the rendering cost on the Pi.
        for _pass in range(4):
            consolidated = self._consolidate_parallel_wall_bands(
                occupied,
                axis_angle,
            )
            if np.array_equal(consolidated, occupied):
                break
            occupied = consolidated
            axis_angle = self._dominant_manhattan_axis(occupied)
        self.render_parallel_wall_cells_merged = int(np.count_nonzero(
            before_wall_consolidation ^ occupied
        ))
        occupied = self._close_supported_rectangular_contours(
            occupied,
            self.observed & (probabilities < 0.35),
            axis_angle,
        )
        occupied = self._close_supported_circular_contours(
            occupied,
            self.observed & (probabilities < 0.35),
            axis_angle,
        )
        occupied = self._close_compact_contour_gaps(
            occupied,
            self.observed & (probabilities < 0.35),
        )
        # The compact outline may be completed, but its interior is never
        # filled: gray/unknown evidence remains visible and auditable.
        self.render_filled_enclosed_cells = 0
        self._render_obstacle_cache = occupied.copy()
        return occupied

    @staticmethod
    def _dominant_manhattan_axis(mask, radius=12):
        """Estimate the two orthogonal wall directions from local support."""
        if int(np.count_nonzero(mask)) < 20:
            return 0.0
        height, width = mask.shape
        samples = []
        for dy in range(0, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx <= 0:
                    continue
                distance = math.hypot(dx, dy)
                if distance < 3.0 or distance > radius:
                    continue
                source_y0 = max(0, -dy)
                source_y1 = min(height, height - dy)
                source_x0 = max(0, -dx)
                source_x1 = min(width, width - dx)
                if source_y1 <= source_y0 or source_x1 <= source_x0:
                    continue
                overlap = int(np.count_nonzero(
                    mask[
                        source_y0:source_y1,
                        source_x0:source_x1,
                    ]
                    & mask[
                        source_y0 + dy:source_y1 + dy,
                        source_x0 + dx:source_x1 + dx,
                    ]
                ))
                if overlap <= 0:
                    continue
                angle = math.atan2(dy, dx) % (math.pi / 2.0)
                samples.append((angle, overlap * distance))
        if not samples:
            return 0.0
        bin_count = 180
        scores = np.zeros(bin_count, dtype=np.float64)
        for angle, weight in samples:
            index = int(round(
                angle / (math.pi / 2.0) * bin_count
            )) % bin_count
            scores[index] += weight
        smooth = sum(
            np.roll(scores, offset) for offset in range(-3, 4)
        )
        peak = int(np.argmax(smooth))
        peak_angle = (peak + 0.5) / bin_count * (math.pi / 2.0)
        sine = 0.0
        cosine = 0.0
        window = math.radians(5.0)
        for angle, weight in samples:
            error = abs(
                (angle - peak_angle + math.pi / 4.0)
                % (math.pi / 2.0)
                - math.pi / 4.0
            )
            if error > window:
                continue
            sine += weight * math.sin(4.0 * angle)
            cosine += weight * math.cos(4.0 * angle)
        if abs(sine) + abs(cosine) <= 1e-9:
            return peak_angle
        return (math.atan2(sine, cosine) / 4.0) % (math.pi / 2.0)

    def _consolidate_parallel_wall_bands(self, occupied, axis_angle):
        """Collapse duplicated long wall bands onto one supported centerline."""
        self.render_parallel_wall_cells_merged = 0
        if (
            self.render_parallel_wall_merge_cells <= 0
            or not np.any(occupied)
        ):
            return occupied.copy()
        original = occupied.copy()
        remove = np.zeros_like(original)
        add = np.zeros_like(original)
        for tangent in (axis_angle, axis_angle + math.pi / 2.0):
            self._consolidate_one_wall_direction(
                original,
                remove,
                add,
                tangent,
            )
        result = (original & ~remove) | add
        self.render_parallel_wall_cells_merged = int(np.count_nonzero(
            original ^ result
        ))
        return result

    def _close_supported_circular_contours(
        self,
        occupied,
        strong_free,
        axis_angle,
    ):
        """Complete a compact round outline from a supported majority arc."""
        maximum_span = self.render_contour_close_max_span_cells
        if maximum_span <= 0 or not np.any(occupied):
            return occupied.copy()
        result = occupied.copy()
        join_radius = max(
            1,
            int(math.ceil(0.15 / self.resolution_m)),
        )
        cosine = math.cos(axis_angle)
        sine = math.sin(axis_angle)
        for component in self._mask_components(
            occupied,
            join_radius=join_radius,
        ):
            rows = np.asarray([cell[0] for cell in component])
            columns = np.asarray([cell[1] for cell in component])
            structurally_confirmed = self._component_is_registered(
                rows,
                columns,
            )
            span_row = int(rows.max()) - int(rows.min()) + 1
            span_column = int(columns.max()) - int(columns.min()) + 1
            if (
                min(span_row, span_column) < 6
                or max(span_row, span_column) > maximum_span
                or min(span_row, span_column)
                / max(span_row, span_column) < 0.68
            ):
                continue
            margin = 3
            crop = result[
                max(0, int(rows.min()) - margin):
                min(self.height, int(rows.max()) + margin + 1),
                max(0, int(columns.min()) - margin):
                min(self.width, int(columns.max()) + margin + 1),
            ]
            if self._mask_encloses_pocket(crop):
                continue

            # A three-sided square is handled by the rectangular rule above.
            # Do not reinterpret a confirmed open rectangular mouth as round.
            along = columns * cosine + rows * sine
            normal = -columns * sine + rows * cosine
            minimum_along = float(np.min(along))
            maximum_along = float(np.max(along))
            minimum_normal = float(np.min(normal))
            maximum_normal = float(np.max(normal))
            face_band = max(1.0, 0.075 / self.resolution_m)
            corner_radius = max(2.0, 0.12 / self.resolution_m)
            corners = (
                (minimum_along, minimum_normal),
                (minimum_along, maximum_normal),
                (maximum_along, minimum_normal),
                (maximum_along, maximum_normal),
            )
            corners_present = sum(
                bool(np.any(np.hypot(
                    along - corner_along,
                    normal - corner_normal,
                ) <= corner_radius))
                for corner_along, corner_normal in corners
            )
            side_support = (
                len(np.unique(np.rint(along[
                    np.abs(normal - minimum_normal) <= face_band
                ]))),
                len(np.unique(np.rint(along[
                    np.abs(normal - maximum_normal) <= face_band
                ]))),
                len(np.unique(np.rint(normal[
                    np.abs(along - minimum_along) <= face_band
                ]))),
                len(np.unique(np.rint(normal[
                    np.abs(along - maximum_along) <= face_band
                ]))),
            )
            side_lengths = (
                maximum_along - minimum_along + 1.0,
                maximum_along - minimum_along + 1.0,
                maximum_normal - minimum_normal + 1.0,
                maximum_normal - minimum_normal + 1.0,
            )
            rectangular_sides = sum(
                support / max(1.0, length)
                >= self.render_contour_min_side_support
                for support, length in zip(side_support, side_lengths)
            )
            strong_rectangular_sides = sum(
                support / max(1.0, length) >= 0.60
                for support, length in zip(side_support, side_lengths)
            )
            if (
                corners_present >= 2
                and rectangular_sides >= 3
                and strong_rectangular_sides >= 3
            ):
                continue

            center_row = (float(rows.min()) + float(rows.max())) * 0.5
            center_column = (
                float(columns.min()) + float(columns.max())
            ) * 0.5
            radii = np.hypot(
                rows - center_row,
                columns - center_column,
            )
            radius = float(np.median(radii))
            if radius < 2.5:
                continue
            radial_residual = np.abs(radii - radius)
            if float(np.percentile(radial_residual, 80.0)) > max(
                1.5,
                radius * 0.35,
            ):
                continue
            sector_count = 24
            angles = (
                np.arctan2(
                    rows - center_row,
                    columns - center_column,
                )
                % math.tau
            )
            sectors = set(map(
                int,
                np.floor(angles / math.tau * sector_count),
            ))
            quadrants = {
                sector // (sector_count // 4) for sector in sectors
            }
            if len(sectors) < 16 or len(quadrants) < 3:
                continue

            sample_count = max(48, int(math.ceil(math.tau * radius * 3.0)))
            proposed_circle = []
            previous = None
            for sample in range(sample_count + 1):
                angle = math.tau * sample / sample_count
                point = (
                    int(round(center_column + radius * math.cos(angle))),
                    int(round(center_row + radius * math.sin(angle))),
                )
                if previous is not None:
                    proposed_circle.extend(self._bresenham(previous, point))
                previous = point
            proposed_circle = list(dict.fromkeys(
                (column, row)
                for column, row in proposed_circle
                if self.in_bounds(column, row)
                and not result[row, column]
            ))
            if not proposed_circle:
                continue
            near_free = sum(
                int(self._confirmed_opening_at(
                    strong_free,
                    column,
                    row,
                    column - center_column,
                    row - center_row,
                ))
                for column, row in proposed_circle
            )
            if (
                not structurally_confirmed
                and near_free / len(proposed_circle) >= 0.25
            ):
                continue
            candidate = result.copy()
            for column, row in proposed_circle:
                candidate[row, column] = True
            candidate_crop = candidate[
                max(0, int(rows.min()) - margin):
                min(self.height, int(rows.max()) + margin + 1),
                max(0, int(columns.min()) - margin):
                min(self.width, int(columns.max()) + margin + 1),
            ]
            if not self._mask_encloses_pocket(candidate_crop):
                continue
            for column, row in proposed_circle:
                result[row, column] = True
        self.render_contour_cells_closed += int(np.count_nonzero(
            result & ~occupied
        ))
        return result

    def _consolidate_one_wall_direction(
        self,
        occupied,
        remove,
        add,
        tangent,
    ):
        rows, columns = np.nonzero(occupied)
        if not len(rows):
            return
        cosine = math.cos(tangent)
        sine = math.sin(tangent)
        along = np.rint(columns * cosine + rows * sine).astype(int)
        normal = np.rint(-columns * sine + rows * cosine).astype(int)
        by_normal = {}
        for value_n, value_s in zip(normal, along):
            by_normal.setdefault(int(value_n), set()).add(int(value_s))
        records = []
        minimum_span = self.render_parallel_wall_min_span_cells
        minimum_segment_span = max(
            8,
            int(math.ceil(0.60 / self.resolution_m)),
        )
        projection_gap = max(
            2,
            int(math.ceil(0.35 / self.resolution_m)),
        )
        for value_n, values in by_normal.items():
            ordered = sorted(values)
            if not ordered:
                continue
            start = previous = ordered[0]
            count = 1
            for value in ordered[1:] + [None]:
                # A rasterized diagonal alternates between neighboring normal
                # bins and therefore leaves small holes in either individual
                # projected row. Join these sub-35 cm quantization/sampling
                # holes while extracting a line record; this does not itself
                # draw or bridge anything in the rendered map. A second long,
                # substantially overlapping wall is still required before any
                # cell can be consolidated.
                if (
                    value is not None
                    and value - previous <= projection_gap
                ):
                    previous = value
                    count += 1
                    continue
                span = previous - start + 1
                if (
                    span >= minimum_segment_span
                    and count / span >= 0.45
                ):
                    records.append((value_n, start, previous, count))
                if value is not None:
                    start = previous = value
                    count = 1
        if len(records) < 2:
            return

        parents = list(range(len(records)))
        staggered_parents = list(range(len(records)))

        def find(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left, right):
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        def find_staggered(index):
            while staggered_parents[index] != index:
                staggered_parents[index] = staggered_parents[
                    staggered_parents[index]
                ]
                index = staggered_parents[index]
            return index

        def union_staggered(left, right):
            left_root = find_staggered(left)
            right_root = find_staggered(right)
            if left_root != right_root:
                staggered_parents[right_root] = left_root

        minimum_overlap = max(
            4,
            int(math.ceil(0.40 / self.resolution_m)),
        )
        endpoint_gap = max(
            2,
            int(math.ceil(0.35 / self.resolution_m)),
        )
        for left in range(len(records)):
            n_left, start_left, end_left, _count_left = records[left]
            for right in range(left + 1, len(records)):
                n_right, start_right, end_right, _count_right = records[right]
                if abs(n_left - n_right) > (
                    self.render_parallel_wall_merge_cells
                ):
                    continue
                overlap = min(end_left, end_right) - max(
                    start_left, start_right
                ) + 1
                left_span = end_left - start_left + 1
                right_span = end_right - start_right + 1
                # A real wall step, corner protrusion or cabinet recess also
                # consists of nearby parallel segments, but those segments
                # meet at their endpoints instead of occupying the same long
                # tangent interval.  Only two long, substantially overlapping
                # observations are evidence of a duplicated SLAM wall layer.
                required_overlap = max(
                    minimum_overlap,
                    int(math.ceil(0.65 * min(left_span, right_span))),
                )
                if (
                    left_span >= minimum_span
                    and right_span >= minimum_span
                    and overlap >= required_overlap
                ):
                    union(left, right)
                    continue
                tangent_gap = max(
                    start_left - end_right - 1,
                    start_right - end_left - 1,
                    0,
                )
                # A long wall sometimes lands on one offset line for its first
                # half and a nearby line for its second half. Correct only this
                # long-wall seam. Local protrusions and cabinet recesses have
                # at least one short face and cannot satisfy both span gates.
                if (
                    left_span >= minimum_span
                    and right_span >= minimum_span
                    and tangent_gap <= endpoint_gap
                ):
                    union_staggered(left, right)

        groups = {}
        for index, record in enumerate(records):
            groups.setdefault(find(index), []).append(record)
        for group in groups.values():
            normal_values = {record[0] for record in group}
            if len(normal_values) < 2:
                continue
            group_start = min(record[1] for record in group)
            group_end = max(record[2] for record in group)
            duplicate_runs = []
            run_start = None
            for value_s in range(group_start, group_end + 2):
                active = [
                    record for record in group
                    if record[1] <= value_s <= record[2]
                ]
                if len(active) >= 2 and value_s <= group_end:
                    if run_start is None:
                        run_start = value_s
                    continue
                if run_start is None:
                    continue
                run_end = value_s - 1
                if run_end - run_start + 1 >= minimum_span:
                    duplicate_runs.append((run_start, run_end))
                run_start = None

            for start, end in duplicate_runs:
                active = [
                    record for record in group
                    if record[1] <= start and record[2] >= end
                ]
                if len(active) < 2:
                    continue
                # Keep the best-supported measured wall rather than drawing an
                # averaged line between two observations.  Averaging can move
                # a correct wall and disconnect a genuine corner feature.
                target_normal = max(
                    active,
                    key=lambda record: (
                        record[3],
                        record[2] - record[1] + 1,
                    ),
                )[0]
                # Remove only the long line bins that supplied the duplicate
                # evidence.  Do not clear the whole band between them: short
                # perpendicular connectors and recessed cabinet faces inside
                # that band are real scene geometry and must survive.
                active_normals = np.asarray(
                    [record[0] for record in active],
                    dtype=int,
                )
                # One raster cell beside a diagonal line is its quantization
                # fringe, not another physical surface.  Limit cleanup to this
                # one-cell fringe around verified long duplicate records; the
                # middle of the band and any larger step remain untouched.
                near_active_line = np.min(
                    np.abs(normal[:, None] - active_normals[None, :]),
                    axis=1,
                ) <= 1
                belongs = (
                    near_active_line
                    & (along >= start)
                    & (along <= end)
                )
                remove[rows[belongs], columns[belongs]] = True
                start_xy = (
                    int(round(start * cosine - target_normal * sine)),
                    int(round(start * sine + target_normal * cosine)),
                )
                end_xy = (
                    int(round(end * cosine - target_normal * sine)),
                    int(round(end * sine + target_normal * cosine)),
                )
                for column, row in self._bresenham(start_xy, end_xy):
                    if self.in_bounds(column, row):
                        add[row, column] = True

        staggered_groups = {}
        for index, record in enumerate(records):
            staggered_groups.setdefault(
                find_staggered(index),
                [],
            ).append(record)
        for group in staggered_groups.values():
            if len(group) < 2 or len({record[0] for record in group}) < 2:
                continue
            start = min(record[1] for record in group)
            end = max(record[2] for record in group)
            if end - start + 1 < minimum_span:
                continue
            target_normal = max(
                group,
                key=lambda record: (
                    record[3],
                    record[2] - record[1] + 1,
                ),
            )[0]
            belongs = np.zeros(len(rows), dtype=bool)
            for value_n, record_start, record_end, _count in group:
                belongs |= (
                    (np.abs(normal - value_n) <= 1)
                    & (along >= record_start)
                    & (along <= record_end)
                )
            remove[rows[belongs], columns[belongs]] = True
            start_xy = (
                int(round(start * cosine - target_normal * sine)),
                int(round(start * sine + target_normal * cosine)),
            )
            end_xy = (
                int(round(end * cosine - target_normal * sine)),
                int(round(end * sine + target_normal * cosine)),
            )
            for column, row in self._bresenham(start_xy, end_xy):
                if self.in_bounds(column, row):
                    add[row, column] = True

    def _close_supported_rectangular_contours(
        self,
        occupied,
        strong_free,
        axis_angle,
    ):
        """Complete a compact rectangle only from three sides and four corners."""
        self.render_contour_cells_closed = 0
        maximum_span = self.render_contour_close_max_span_cells
        if maximum_span <= 0 or not np.any(occupied):
            return occupied.copy()
        result = occupied.copy()
        cosine = math.cos(axis_angle)
        sine = math.sin(axis_angle)
        corner_radius = max(2.0, 0.12 / self.resolution_m)
        face_band = max(1.0, 0.075 / self.resolution_m)
        join_radius = max(
            1,
            int(math.ceil(0.15 / self.resolution_m)),
        )
        for component in self._mask_components(
            occupied,
            join_radius=join_radius,
        ):
            if len(component) < 6:
                continue
            rows = np.asarray([cell[0] for cell in component])
            columns = np.asarray([cell[1] for cell in component])
            structurally_confirmed = self._component_is_registered(
                rows,
                columns,
            )
            along = columns * cosine + rows * sine
            normal = -columns * sine + rows * cosine
            minimum_along = float(np.min(along))
            maximum_along = float(np.max(along))
            minimum_normal = float(np.min(normal))
            maximum_normal = float(np.max(normal))
            span_along = maximum_along - minimum_along
            span_normal = maximum_normal - minimum_normal
            if (
                min(span_along, span_normal) < 3.0
                or max(span_along, span_normal) > maximum_span
            ):
                continue
            corners = (
                (minimum_along, minimum_normal),
                (minimum_along, maximum_normal),
                (maximum_along, minimum_normal),
                (maximum_along, maximum_normal),
            )
            if not all(
                bool(np.any(np.hypot(
                    along - corner_along,
                    normal - corner_normal,
                ) <= corner_radius))
                for corner_along, corner_normal in corners
            ):
                continue
            side_support = (
                len(np.unique(np.rint(along[
                    np.abs(normal - minimum_normal) <= face_band
                ]))) / max(1.0, span_along + 1.0),
                len(np.unique(np.rint(along[
                    np.abs(normal - maximum_normal) <= face_band
                ]))) / max(1.0, span_along + 1.0),
                len(np.unique(np.rint(normal[
                    np.abs(along - minimum_along) <= face_band
                ]))) / max(1.0, span_normal + 1.0),
                len(np.unique(np.rint(normal[
                    np.abs(along - maximum_along) <= face_band
                ]))) / max(1.0, span_normal + 1.0),
            )
            if sum(
                value >= self.render_contour_min_side_support
                for value in side_support
            ) < 3:
                continue
            edges = (
                ((minimum_along, minimum_normal),
                 (maximum_along, minimum_normal)),
                ((minimum_along, maximum_normal),
                 (maximum_along, maximum_normal)),
                ((minimum_along, minimum_normal),
                 (minimum_along, maximum_normal)),
                ((maximum_along, minimum_normal),
                 (maximum_along, maximum_normal)),
            )
            proposed = []
            for start, end in edges:
                start_xy = (
                    int(round(start[0] * cosine - start[1] * sine)),
                    int(round(start[0] * sine + start[1] * cosine)),
                )
                end_xy = (
                    int(round(end[0] * cosine - end[1] * sine)),
                    int(round(end[0] * sine + end[1] * cosine)),
                )
                cells = [
                    (column, row)
                    for column, row in self._bresenham(start_xy, end_xy)
                    if self.in_bounds(column, row)
                ]
                missing = [
                    (column, row) for column, row in cells
                    if not result[row, column]
                ]
                edge_delta_column = end_xy[0] - start_xy[0]
                edge_delta_row = end_xy[1] - start_xy[1]
                near_free = sum(
                    int(self._confirmed_opening_at(
                        strong_free,
                        column,
                        row,
                        -edge_delta_row,
                        edge_delta_column,
                    ))
                    for column, row in missing
                )
                if (
                    not structurally_confirmed
                    and
                    missing
                    and near_free / len(missing) >= 0.25
                ):
                    # This particular edge is probably a real opening. It
                    # must not prevent other well-supported edges from having
                    # harmless raster gaps repaired.
                    continue
                proposed.extend(missing)
            for column, row in proposed:
                result[row, column] = True
        self.render_contour_cells_closed = int(np.count_nonzero(
            result & ~occupied
        ))
        return result

    def _component_is_registered(self, rows, columns):
        if not self._structural_object_anchors:
            return False
        center_x = self.origin_x_m + (
            float(np.mean(columns)) + 0.5
        ) * self.resolution_m
        center_y = self.origin_y_m + (
            float(np.mean(rows)) + 0.5
        ) * self.resolution_m
        return any(
            math.hypot(center_x - anchor[0], center_y - anchor[1])
            <= 0.80
            for anchor in self._structural_object_anchors
        )

    def _close_compact_contour_gaps(self, occupied, strong_free):
        """Close small raster gaps in a compact non-rectangular outline.

        This is a local morphological close, never a global image operation.
        It is accepted only when it creates an enclosed pocket and fewer than
        a quarter of the proposed bridge cells are backed by strong free-space
        evidence. Thus a small dropout in a round object closes, while a real
        observed opening remains open.
        """
        radius = max(
            1,
            int(math.ceil(0.10 / self.resolution_m)),
        )
        maximum_span = self.render_contour_close_max_span_cells
        if maximum_span <= 0 or not np.any(occupied):
            return occupied.copy()
        result = occupied.copy()
        join_radius = max(
            1,
            int(math.ceil(0.15 / self.resolution_m)),
        )
        for component in self._mask_components(
            occupied,
            join_radius=join_radius,
        ):
            rows = np.asarray([cell[0] for cell in component])
            columns = np.asarray([cell[1] for cell in component])
            span_row = int(rows.max()) - int(rows.min()) + 1
            span_column = int(columns.max()) - int(columns.min()) + 1
            if (
                min(span_row, span_column) < 5
                or max(span_row, span_column) > maximum_span
            ):
                continue
            margin = radius + 2
            row_start = max(0, int(rows.min()) - margin)
            row_end = min(self.height, int(rows.max()) + margin + 1)
            column_start = max(0, int(columns.min()) - margin)
            column_end = min(
                self.width,
                int(columns.max()) + margin + 1,
            )
            local = np.zeros(
                (row_end - row_start, column_end - column_start),
                dtype=bool,
            )
            local[
                rows - row_start,
                columns - column_start,
            ] = True
            closed = self._binary_erode(
                self._binary_dilate(local, radius),
                radius,
            )
            candidates = closed & ~local
            candidate_rows, candidate_columns = np.nonzero(candidates)
            if not len(candidate_rows):
                continue
            global_rows = candidate_rows + row_start
            global_columns = candidate_columns + column_start
            near_free = sum(
                int(self._mask_value_near(
                    strong_free,
                    int(column),
                    int(row),
                    radius=1,
                ))
                for row, column in zip(global_rows, global_columns)
            )
            if near_free / len(global_rows) >= 0.25:
                continue
            proposed = local | candidates
            if not self._mask_encloses_pocket(proposed):
                continue
            result[global_rows, global_columns] = True
        self.render_contour_cells_closed = int(np.count_nonzero(
            result & ~occupied
        )) + self.render_contour_cells_closed
        return result

    @staticmethod
    def _binary_dilate(mask, radius):
        padded = np.pad(mask, radius, constant_values=False)
        result = np.zeros_like(mask)
        for delta_row in range(-radius, radius + 1):
            for delta_column in range(-radius, radius + 1):
                row_start = radius + delta_row
                column_start = radius + delta_column
                result |= padded[
                    row_start:row_start + mask.shape[0],
                    column_start:column_start + mask.shape[1],
                ]
        return result

    @staticmethod
    def _binary_erode(mask, radius):
        padded = np.pad(mask, radius, constant_values=False)
        result = np.ones_like(mask)
        for delta_row in range(-radius, radius + 1):
            for delta_column in range(-radius, radius + 1):
                row_start = radius + delta_row
                column_start = radius + delta_column
                result &= padded[
                    row_start:row_start + mask.shape[0],
                    column_start:column_start + mask.shape[1],
                ]
        return result

    @staticmethod
    def _mask_encloses_pocket(mask):
        barrier = np.asarray(mask, dtype=bool)
        exterior = np.zeros_like(barrier)
        queue = []
        for row in range(barrier.shape[0]):
            for column in (0, barrier.shape[1] - 1):
                if not barrier[row, column] and not exterior[row, column]:
                    exterior[row, column] = True
                    queue.append((row, column))
        for column in range(barrier.shape[1]):
            for row in (0, barrier.shape[0] - 1):
                if not barrier[row, column] and not exterior[row, column]:
                    exterior[row, column] = True
                    queue.append((row, column))
        while queue:
            row, column = queue.pop()
            for delta_row, delta_column in (
                (-1, 0), (1, 0), (0, -1), (0, 1),
            ):
                next_row = row + delta_row
                next_column = column + delta_column
                if not (
                    0 <= next_row < barrier.shape[0]
                    and 0 <= next_column < barrier.shape[1]
                ):
                    continue
                if (
                    barrier[next_row, next_column]
                    or exterior[next_row, next_column]
                ):
                    continue
                exterior[next_row, next_column] = True
                queue.append((next_row, next_column))
        return int(np.count_nonzero(~barrier & ~exterior)) >= 2

    @staticmethod
    def _mask_value_near(mask, column, row, radius=1):
        row_start = max(0, int(row) - radius)
        row_end = min(mask.shape[0], int(row) + radius + 1)
        column_start = max(0, int(column) - radius)
        column_end = min(mask.shape[1], int(column) + radius + 1)
        return bool(np.any(mask[
            row_start:row_end,
            column_start:column_end,
        ]))

    @classmethod
    def _confirmed_opening_at(
        cls,
        strong_free,
        column,
        row,
        normal_column,
        normal_row,
    ):
        """Require free evidence on the boundary or on both of its sides."""
        if cls._mask_value_near(
            strong_free,
            column,
            row,
            radius=0,
        ):
            return True
        length = math.hypot(normal_column, normal_row)
        if length <= 1e-9:
            return False
        unit_column = normal_column / length
        unit_row = normal_row / length
        distance = 2.0
        positive = (
            int(round(column + unit_column * distance)),
            int(round(row + unit_row * distance)),
        )
        negative = (
            int(round(column - unit_column * distance)),
            int(round(row - unit_row * distance)),
        )
        return (
            cls._mask_value_near(
                strong_free,
                positive[0],
                positive[1],
                radius=0,
            )
            and cls._mask_value_near(
                strong_free,
                negative[0],
                negative[1],
                radius=0,
            )
        )

    @staticmethod
    def _mask_components(mask, join_radius=1):
        remaining = set(map(tuple, np.argwhere(mask)))
        radius = max(1, int(join_radius))
        while remaining:
            start = remaining.pop()
            component = [start]
            queue = [start]
            while queue:
                row, column = queue.pop()
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if dx == 0 and dy == 0:
                            continue
                        neighbor = (row + dy, column + dx)
                        if neighbor not in remaining:
                            continue
                        remaining.remove(neighbor)
                        component.append(neighbor)
                        queue.append(neighbor)
            yield component

    def _bridge_short_wall_gaps(self, occupied, strong_free):
        """Bridge short, supported horizontal, vertical, and diagonal gaps.

        This is an output-only cleanup. It never changes probabilities used
        by localization or path planning, and it refuses to close a gap that
        has repeatedly been observed as free.
        """
        maximum_gap = self.render_wall_gap_max_cells
        support = self.render_wall_support_cells
        if maximum_gap <= 0 or not np.any(occupied):
            return occupied.copy()
        result = occupied.copy()
        self._bridge_line_gaps(
            result,
            strong_free,
            maximum_gap,
            support,
        )
        transposed = result.T.copy()
        self._bridge_line_gaps(
            transposed,
            strong_free.T,
            maximum_gap,
            support,
        )
        result = transposed.T
        self._bridge_diagonal_gaps(
            result,
            strong_free,
            maximum_gap,
            support,
        )
        return result

    @classmethod
    def _bridge_diagonal_gaps(
        cls,
        mask,
        strong_free,
        maximum_gap,
        support,
    ):
        """Run the same conservative gap rule along both diagonal families."""
        height, width = mask.shape
        for flipped in (False, True):
            working = np.fliplr(mask) if flipped else mask
            working_free = np.fliplr(strong_free) if flipped else strong_free
            for offset in range(-height + 1, width):
                rows, columns = cls._diagonal_indices(
                    height,
                    width,
                    offset,
                )
                if len(rows) < support * 2 + 1:
                    continue
                values = working[rows, columns].copy()[None, :]
                free_values = working_free[rows, columns][None, :]
                cls._bridge_line_gaps(
                    values,
                    free_values,
                    maximum_gap,
                    support,
                )
                working[rows, columns] = values[0]

    @staticmethod
    def _diagonal_indices(height, width, offset):
        if offset >= 0:
            length = min(height, width - offset)
            rows = np.arange(length)
            columns = rows + offset
        else:
            length = min(height + offset, width)
            columns = np.arange(length)
            rows = columns - offset
        return rows, columns

    @staticmethod
    def _bridge_line_gaps(mask, strong_free, maximum_gap, support):
        width = mask.shape[1]
        for row in range(mask.shape[0]):
            column = support
            while column < width - support:
                if mask[row, column]:
                    column += 1
                    continue
                gap_start = column
                while column < width and not mask[row, column]:
                    column += 1
                gap_end = column
                gap_size = gap_end - gap_start
                if gap_end >= width or gap_size > maximum_gap:
                    continue
                left_start = gap_start - support
                right_end = gap_end + support
                if left_start < 0 or right_end > width:
                    continue
                left_supported = bool(np.all(
                    mask[row, left_start:gap_start]
                ))
                right_supported = bool(np.all(
                    mask[row, gap_end:right_end]
                ))
                gap_is_free = bool(np.any(
                    strong_free[row, gap_start:gap_end]
                ))
                if (
                    left_supported
                    and right_supported
                    and not gap_is_free
                ):
                    mask[row, gap_start:gap_end] = True

    def save(self, output_prefix):
        """保存 ``<prefix>.pgm``、``<prefix>.yaml`` 和轨迹之外的地图数据。"""
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        pgm_path = prefix.with_suffix(".pgm")
        yaml_path = prefix.with_suffix(".yaml")

        self.prune_floating_obstacles()
        image = self.render_image()

        # PGM 从上到下写；内部数组的 row 则随世界 y 增大，因此需要翻转。
        with pgm_path.open("wb") as stream:
            stream.write("P5\n{} {}\n255\n".format(self.width, self.height).encode("ascii"))
            stream.write(np.flipud(image).tobytes())
        yaml_path.write_text(
            "image: {}\nresolution: {:.6f}\norigin: [{:.6f}, {:.6f}, 0.0]\n"
            "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.35\n".format(
                pgm_path.name,
                self.resolution_m,
                self.origin_x_m,
                self.origin_y_m,
            ),
            encoding="utf-8",
        )
        return pgm_path, yaml_path

    @staticmethod
    def _bresenham(start, end):
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        cells = []
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            twice_error = 2 * error
            if twice_error >= dy:
                error += dy
                x0 += sx
            if twice_error <= dx:
                error += dx
                y0 += sy
        return cells
