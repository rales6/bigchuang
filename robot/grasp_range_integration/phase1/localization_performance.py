"""Same-distance vector ray casting through an adapter, not weaker evidence."""
import math
import time
import numpy as np


def install_localization_performance(mission):
    cls = mission.ScanMapLocalizer
    original = cls._raycast_range

    def raycast(self, pose, local_angle, maximum_range=None):
        # Preserve sequential floating-point additions from the original
        # caster (np.arange can land on different cells at exact boundaries).
        step = max(.01, self.map.resolution_m*.55)
        maximum = self.max_distance_m if maximum_range is None else float(maximum_range)
        minimum = max(self.min_distance_m, step)
        if not all(math.isfinite(v) for v in (maximum, minimum, step)) or maximum > 20:
            return original(self, pose, local_angle, maximum_range)
        key = (minimum, maximum, step)
        cached = getattr(self, "_phase1_ray_distances", None)
        if cached is None or cached[0] != key:
            values, distance = [], minimum
            while distance <= maximum:
                values.append(distance)
                distance += step
            distances = np.asarray(values, dtype=np.float64)
            self._phase1_ray_distances = (key, distances)
        else:
            distances = cached[1]
        if not len(distances):
            return None
        sensor = pose.transform_points(self.lidar_offset.reshape(1,2))[0]
        angle = pose.yaw_rad+float(local_angle)
        direction = np.asarray((math.cos(angle),math.sin(angle)))
        cells = self.map.world_to_grid(sensor+direction*distances[:,None])
        columns, rows = cells[:,0], cells[:,1]
        inside = ((columns >= 0)&(columns < self.map.width)&(rows >= 0)&(rows < self.map.height))
        exits = np.flatnonzero(~inside)
        end = int(exits[0]) if len(exits) else len(cells)
        hits = np.flatnonzero(self.map.occupied[rows[:end],columns[:end]])
        return float(distances[hits[0]]) if len(hits) else None

    cls._raycast_range = raycast
    original_track = cls.track

    def track(self, *args, **kwargs):
        start = time.monotonic()
        result = original_track(self, *args, **kwargs)
        from grasp_range_integration.phase1.turn_policy import emit
        emit("map_tracking_timing", elapsed_ms=round((time.monotonic()-start)*1000,2),
             result_present=result is not None, same_ray_samples=True)
        return result

    cls.track = track
