"""Annotate temporary obstacle provenance without changing protected map masks.

The confirmed core includes the original one-cell jitter tolerance. Neither
this core nor its radius padding is silently cleared or ignored for turning.
"""

import numpy as np


def install_temporary_evidence(cls):
    if cls is None or getattr(cls, "_phase1_evidence_installed", False):
        return
    original_mark = cls.mark_temporary_obstacles
    original_clear = cls.clear_temporary_obstacles

    def mark(self, world_points, radius_m=0.10, minimum_hits=1):
        points = np.asarray(world_points, dtype=float).reshape(-1, 2)
        observation = np.zeros_like(self.temporary_occupied, dtype=bool)
        for column, row in np.unique(self.world_to_grid(points), axis=0):
            column, row = int(column), int(row)
            if self.in_bounds(column, row):
                observation[max(0, row-1):min(self.height, row+2),
                            max(0, column-1):min(self.width, column+2)] = True
        # Existing unexplained cells remain unknown provenance, never padding.
        if not hasattr(self, "_phase1_temporary_core"):
            self._phase1_temporary_core = np.zeros_like(observation)
            self._phase1_temporary_unclassified = self.temporary_occupied.copy()
        added = original_mark(self, points, radius_m=radius_m, minimum_hits=minimum_hits)
        self._phase1_temporary_core |= observation & (
            self.temporary_obstacle_hits >= max(1, int(minimum_hits)))
        self._phase1_temporary_padding_radius_m = float(radius_m)
        return added

    def clear(self):
        result = original_clear(self)
        self._phase1_temporary_core = np.zeros_like(self.temporary_occupied)
        self._phase1_temporary_unclassified = np.zeros_like(self.temporary_occupied)
        return result

    cls.mark_temporary_obstacles, cls.clear_temporary_obstacles = mark, clear
    cls._phase1_evidence_installed = True
