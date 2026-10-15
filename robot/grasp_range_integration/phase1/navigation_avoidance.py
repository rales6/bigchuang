"""Early turn-space reservation and stopped obstacle evidence; no motor IO."""

import math
import time
from types import SimpleNamespace
import numpy as np
from grasp_range_integration.phase1.rotation_guard import installed_body_mask

ROTATION_RESERVE_M = .44  # Existing .30 m lidar envelope + .14 m installation offset.


def exterior_points(scan):
    angles = np.asarray(scan.angles_rad, dtype=float)
    ranges = np.asarray(scan.distances_m, dtype=float)
    if angles.ndim != 1 or ranges.shape != angles.shape:
        return np.empty((0, 2))
    valid = (np.isfinite(angles) & np.isfinite(ranges) & (ranges >= .08)
             & (ranges <= 8.) & ~installed_body_mask(angles, ranges))
    return np.column_stack((ranges[valid]*np.cos(angles[valid])+.14,
                            ranges[valid]*np.sin(angles[valid])))


def forward_reserve(scan, linear_mm_s, angular_mrad_s=0, *, now=None,
                    reserve_radius_m=ROTATION_RESERVE_M):
    """Check a swept disk, not a wider angular sector or imaginary rear rays.

    Reserve the already required turning space along the commanded advance.
    Include scan age, a 650 ms command lease and 200 ms reaction allowance.
    Missing/stale scans cannot approve translation. This is a conservative
    turn-space policy, not a calibrated vehicle-footprint collision model.
    """
    now = time.monotonic() if now is None else now
    received = getattr(scan, 'received_s', now)
    age = max(0., now - received)
    if not math.isfinite(reserve_radius_m) or reserve_radius_m < ROTATION_RESERVE_M:
        raise ValueError('forward reserve must enclose the physical turn radius')
    result = dict(safe=False, reason='stale_navigation_scan', scan_age_s=age,
                  reserve_radius_m=reserve_radius_m)
    if not math.isfinite(age) or age > 1.5:
        return result
    points = exterior_points(scan)
    if len(points) < 16:
        return dict(result, reason='insufficient_navigation_evidence')
    # This is additional ADVANCE lookahead, not authority for a rear sweep.
    # Rear near returns still veto actual turns in assess_rotation. Do not
    # turn an unclassified rear installation echo into a new front wall.
    points = points[points[:, 0] >= .14-1e-9]
    if len(points) < 8:
        return dict(result, reason='insufficient_forward_evidence')
    speed, rate = max(0, linear_mm_s)/1000., angular_mrad_s/1000.
    horizon = age + .65 + .20
    times = np.linspace(0., horizon, 21)
    if abs(rate) < 1e-6:
        centers = np.column_stack((speed*times, np.zeros_like(times)))
    else:
        centers = np.column_stack((speed*np.sin(rate*times)/rate,
                                   speed*(1-np.cos(rate*times))/rate))
    # Sampling margin bounds displacement between consecutive centers.
    radius = reserve_radius_m + speed*horizon/40.
    distances = np.linalg.norm(points[:, None, :]-centers[None, :, :], axis=2).min(axis=1)
    blocked = distances < radius
    return dict(result, safe=not bool(blocked.any()), reason='forward_turn_space_blocked' if blocked.any()
                else 'forward_turn_space_clear', lookahead_s=horizon,
                lookahead_m=speed*horizon, blocking_points_body_xy_m=points[blocked].tolist())


def turn_reserve_mask(known_map):
    """The same read-only geometry for startup candidates and A* routes."""
    from raspberry_pi.known_map_navigation import _dilate
    radius = int(math.ceil((ROTATION_RESERVE_M + known_map.resolution_m*math.sqrt(2)/2)
                           / known_map.resolution_m))
    key = (radius, getattr(known_map, 'temporary_obstacle_updates', 0),
           int(np.count_nonzero(known_map.temporary_occupied)))
    cached = getattr(known_map, '_phase1_turn_reserve_mask', None)
    if cached is None or cached[0] != key:
        blocked = known_map.occupied | known_map.unknown | known_map.temporary_occupied
        glass = getattr(known_map, '_phase1_glass_keepout', None)
        if glass is not None:
            blocked = blocked | glass
        cached = (key, ~_dilate(blocked, radius))
        known_map._phase1_turn_reserve_mask = cached
    mask = known_map.traversable & cached[1]
    # Unknown exterior of the map also blocks the swept disk.
    mask[:radius, :] = mask[-radius:, :] = False
    mask[:, :radius] = mask[:, -radius:] = False
    return mask


def reserve_path(known_map, start, goal, original, *, start_yaw_rad=None,
                 pose_uncertainty_m=0., allow_short_forward=False):
    """Plan on a temporary turn-clearance mask; never alter localization map."""
    mask = turn_reserve_mask(known_map)
    start_cell, goal_cell = known_map.world_to_grid((start, goal))
    for column, row in (start_cell, goal_cell):
        if not known_map.in_bounds(column, row):
            return []
    if not mask[goal_cell[1], goal_cell[0]]:
        return []  # Do not let nearest-free snapping invent a goal.
    can_advance_straight = (start_yaw_rad is not None and math.isfinite(start_yaw_rad)
        and 0. <= pose_uncertainty_m <= .03
        and known_map.traversable[start_cell[1], start_cell[0]])

    def straight_body_sweep_clear(distance_m):
        from grasp_range_integration.phase1.rotation_guard import map_rotation_evidence
        radius_m = ROTATION_RESERVE_M + max(.01, pose_uncertainty_m)
        for step in np.linspace(0., distance_m, int(math.ceil(distance_m/.025))+1):
            point = (float(start[0]) + float(step)*math.cos(start_yaw_rad),
                     float(start[1]) + float(step)*math.sin(start_yaw_rad))
            c, r = known_map.world_to_grid((point,))[0]
            if (not known_map.in_bounds(c, r) or not known_map.traversable[r, c]
                    or not map_rotation_evidence(known_map,
                        SimpleNamespace(x_m=point[0], y_m=point[1],
                                        yaw_rad=start_yaw_rad), radius_m)['clear']):
                return False
        return True

    distance_to_goal = math.hypot(float(goal[0])-float(start[0]),
                                  float(goal[1])-float(start[1]))
    if allow_short_forward and .06 - 1e-9 <= distance_to_goal <= .125:
        # Geometric proposal, not permission to issue a motor command. Avoid
        # A*'s grid-center zigzag for a small locked-target viewpoint move.
        # Every continuous sample must obey the SAME all-round inflated mask,
        # including unknown, glass and the outer work-area restriction.
        cells = known_map.world_to_grid(np.linspace(start, goal,
            max(3, int(math.ceil(distance_to_goal/(known_map.resolution_m/4)))+1)))
        if all(known_map.in_bounds(c, r) and mask[r, c] for c, r in cells):
            return [tuple(float(v) for v in start), tuple(float(v) for v in goal)]
    if can_advance_straight and .20 <= distance_to_goal <= 2.0:
        bearing = math.atan2(float(goal[1])-float(start[1]),
                             float(goal[0])-float(start[0]))
        yaw_error = math.atan2(math.sin(bearing-start_yaw_rad),
                               math.cos(bearing-start_yaw_rad))
        if abs(yaw_error) <= math.radians(4) and straight_body_sweep_clear(distance_to_goal):
            # This is a *straight* camera-approach path, not permission to
            # turn inside the blind rear sector. Live lidar and glass checks
            # still decide whether each short motor lease may be issued.
            return [tuple(float(v) for v in start), tuple(float(v) for v in goal)]
    if not mask[start_cell[1], start_cell[0]]:
        # The car can be inside the conservative *turn* reserve after real
        # motion although its actual .44 m body disk is still map-clear. Do
        # not demand a pivot in that very place, or erase the nearby wall.
        # Only a fresh strong map pose may supply the yaw, and the live front
        # lidar/glass output gates must separately authorize every command.
        if not can_advance_straight:
            return []
        bearing = math.atan2(float(goal[1])-float(start[1]),
                             float(goal[0])-float(start[0]))
        yaw_error = math.atan2(math.sin(bearing-start_yaw_rad),
                               math.cos(bearing-start_yaw_rad))
        if abs(yaw_error) > math.radians(18):
            return []  # Do not advance away from the goal just to escape.
        for distance_m in (.40, .45, .50):
            egress = (float(start[0]) + distance_m*math.cos(start_yaw_rad),
                      float(start[1]) + distance_m*math.sin(start_yaw_rad))
            column, row = known_map.world_to_grid((egress,))[0]
            if not known_map.in_bounds(column, row) or not mask[row, column]:
                continue
            if not straight_body_sweep_clear(distance_m):
                continue
            tail = reserve_path(known_map, egress, goal, original)
            if tail:
                return [tuple(float(v) for v in start), egress] + list(tail)
        return []
    previous = known_map.traversable
    try:
        known_map.traversable = mask
        path = original(start, goal)
        # Validate smoothed continuous segments, not only A* cell centers.
        points = [start] + list(path) + [goal]
        for a, b in zip(points, points[1:]):
            count = max(2, int(math.ceil(math.hypot(b[0]-a[0], b[1]-a[1])
                                        / (known_map.resolution_m/2)))+1)
            cells = known_map.world_to_grid(np.linspace(a, b, count))
            for column, row in cells:
                if not known_map.in_bounds(column, row) or not mask[row, column]:
                    return []
        return path
    finally:
        known_map.traversable = previous


def record_stopped_obstacles(scan, pose, known_map, args):
    """One fresh stopped scan contributes one vote, with original hit threshold.

    Only actual exterior endpoints within 1.5 m are marked in free map cells.
    No unobserved sectors or artificial front wall are generated. Caller must
    supply independently verified map pose and genuinely stopped fresh scan.
    """
    points = exterior_points(scan)
    points = points[np.linalg.norm(points, axis=1) <= 1.5]
    if not len(points): return 0
    yaw = pose.yaw_rad
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)],
                         [math.sin(yaw), math.cos(yaw)]])
    world = points @ rotation.T + np.array([pose.x_m, pose.y_m])
    cells = known_map.world_to_grid(world)
    keep = [known_map.in_bounds(c, r) and known_map.free[r, c] for c, r in cells]
    return known_map.mark_temporary_obstacles(world[np.asarray(keep, dtype=bool)],
        radius_m=args.temporary_obstacle_radius,
        minimum_hits=max(2, args.temporary_obstacle_min_hits))
