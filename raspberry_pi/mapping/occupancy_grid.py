"""二维对数概率占据栅格及 ROS 兼容 PGM/YAML 导出。"""

from collections import deque
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
        ray_block_log_odds=0.75,
        ray_block_inflation_cells=1,
        contradiction_clear_hits=10,
        auto_expand=False,
        expansion_margin_m=1.0,
        min_obstacle_area_m2=0.03,
        render_wall_gap_max_m=0.10,
        render_wall_support_m=0.10,
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
        self._obstacle_filter_dirty = True

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
                occupied_columns.append(end[0])
                occupied_rows.append(end[1])

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
            np.add.at(self.log_odds, (free_rows, free_columns), self.free_log_odds)
            self.observed[free_rows, free_columns] = True
        if occupied_columns:
            np.add.at(
                self.log_odds,
                (occupied_rows, occupied_columns),
                self.occupied_log_odds,
            )
            self.observed[occupied_rows, occupied_columns] = True
            self._contradiction_counts[
                occupied_rows, occupied_columns
            ] = 0
        np.clip(self.log_odds, -4.0, 4.0, out=self.log_odds)
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
        enclosed_unknown = self._enclosed_unknown_mask(occupied)
        self.render_filled_enclosed_cells = int(np.count_nonzero(
            enclosed_unknown
        ))
        occupied = occupied | enclosed_unknown
        image[occupied] = 0
        return image

    def _enclosed_unknown_mask(self, occupied):
        """Fill only unknown cells sealed by black boundaries for rendering."""
        passable = ~occupied
        outside = np.zeros_like(passable)
        queue = deque()
        for column in range(self.width):
            for row in (0, self.height - 1):
                if passable[row, column] and not outside[row, column]:
                    outside[row, column] = True
                    queue.append((row, column))
        for row in range(self.height):
            for column in (0, self.width - 1):
                if passable[row, column] and not outside[row, column]:
                    outside[row, column] = True
                    queue.append((row, column))
        while queue:
            row, column = queue.popleft()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                next_row = row + dy
                next_column = column + dx
                if (
                    0 <= next_row < self.height
                    and 0 <= next_column < self.width
                    and passable[next_row, next_column]
                    and not outside[next_row, next_column]
                ):
                    outside[next_row, next_column] = True
                    queue.append((next_row, next_column))
        return ~self.observed & passable & ~outside

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
