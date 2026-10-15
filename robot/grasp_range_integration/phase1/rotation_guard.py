"""Read-only turn assessment; never infer free space from body occlusion.

Keep the 0917 base mask. A separately bounded 0926 installation-edge patch
uses the user's front-mounted-lidar photo and recurring 110--114 degree
returns, not an automatic 'all stationary near points are chassis' rule.
New patch occlusion always requires reliable local-map rotation clearance.
"""

from __future__ import annotations

import math
import numpy as np


class RotationClearanceError(RuntimeError):
    """A turn veto is not a frontal obstacle or a reason to paint the map."""


BODY_MASK_PROFILE = "front_n10_20261001_contiguous_opposite_edge_v5"
BODY_OCCLUSION_POLICY = "recorded_rear_shoulder_20261003_v2"


def body_mask_components(angles, distances):
    angles, distances = np.asarray(angles, dtype=float), np.asarray(distances, dtype=float)
    with np.errstate(invalid="ignore"):
        sensor_x = distances * np.cos(angles)
        body_x = sensor_x + 0.14
        body_y = distances * np.sin(angles)
    legacy = (np.isfinite(angles) & np.isfinite(distances) & (distances > 0)
            & (sensor_x <= 1e-9) & (np.hypot(body_x, body_y) < 0.19)
            & (np.abs(body_y) < 0.125))
    bearings = (np.degrees(angles) + 180) % 360 - 180
    # Only the documented protruding edge, not the opposite side or full rear.
    # Photo supplies installation topology, NOT centimetre measurements;
    # these conservative limits bound the recorded points plus small jitter.
    patch = (np.isfinite(angles) & np.isfinite(distances) & ~legacy
             & (bearings >= 107) & (bearings <= 117)
             & (distances >= .135) & (distances <= .160)
             & (sensor_x <= 0) & (np.hypot(body_x, body_y) < .19)
             & (body_x >= .065) & (body_x <= .100)
             & (body_y >= .125) & (body_y <= .145))
    # Opposite edge: allow angle/range jitter around the confirmed chassis
    # returns, including -109deg/.133m and -105deg/.1405m from 20260929.
    opposite_edge = (np.isfinite(angles) & np.isfinite(distances) & ~legacy
                     & (bearings >= -110) & (bearings <= -102)
                     & (distances >= .128) & (distances <= .145)
                     & (sensor_x <= 0) & (np.hypot(body_x, body_y) < .19)
                     & (body_x >= .095) & (body_x <= .115)
                     # 20261001_173725: -105deg/.1415-.142m lies at
                     # y=-.13668..-.13716, between the old -.136 edge and
                     # -.138 confirmed strip. Keep angle/range/x bounds;
                     # bridge only the recorded -105deg edge, not the
                     # entire angle strip (-103deg/.141 remains exterior).
                     & ((body_y >= -.136) | ((bearings >= -106.5)
                         & (bearings <= -104.5) & (body_y >= -.141)))
                     & (body_y <= -.125))
    patch |= opposite_edge
    # Operator confirmed the highlighted -107deg / .156m point as chassis
    # in the raw stop plot for 20260927_162135 (not inferred from persistence).
    # Only +/-2deg and a bounded local edge rectangle; DO NOT enlarge the
    # entire side/rear strip or automatically classify all nearby rays.
    confirmed_edge = (np.isfinite(angles) & np.isfinite(distances) & ~legacy
                      & (bearings >= -109) & (bearings <= -105)
                      & (distances >= .145) & (distances <= .160)
                      & (sensor_x <= 0) & (np.hypot(body_x, body_y) < .19)
                      & (body_x >= .085) & (body_x <= .105)
                      & (body_y >= -.155) & (body_y <= -.138))
    patch |= confirmed_edge
    # Retain unknown isolated points when the installed rear body signature
    # is absent. Persistence alone is never evidence of a self return.
    supported = (np.count_nonzero(legacy) >= 5
                 and np.any(legacy & (bearings >= 90))
                 and np.any(legacy & (bearings <= -90)))
    return legacy, patch if supported else np.zeros_like(legacy)


def installed_body_mask(angles, distances):
    legacy, patch = body_mask_components(angles, distances)
    return legacy | patch


def body_supported_blind_arcs(angles, body, bearings):
    """Explain large exterior gaps with *raw* installed-body evidence.

Large arcs remain at most 180 degrees. Ordinary rear gaps retain the
45-degree rule. The three 20261003 runs also show a bounded shoulder whose
exterior edge varies down to 82 degrees and first dense body ray is at
145--150 degrees (up to 65 degrees apart). Allow at most 66 degrees ONLY
for that leading shoulder, with dense independent support in three rear
bands and no enlarged central/trailing gaps. The small 82--90 degree edge
extension also requires that same dense signature. This explains occlusion,
NOT exterior freedom: assess_rotation still requires a clear reliable map.
"""
    ends = np.r_[bearings[1:], bearings[0] + 2 * math.pi]
    arcs = []
    for start, end in zip(bearings, ends):
        gap = float(end - start)
        if gap <= math.radians(90) + 1e-9:
            continue
        support = np.sort((angles[body] - start) % (2 * math.pi))
        support = support[support < gap]
        intervals = np.diff(np.r_[0, support, gap])
        unsupported = float(np.max(intervals))
        rear = bool(np.all(np.cos(np.linspace(start, end, 32)) <= 1e-9))
        absolute = np.degrees(start + support)
        bands = [int(np.count_nonzero((absolute >= lo) & (absolute <= hi)))
                 for lo, hi in ((145, 165), (170, 195), (198, 225))]
        # Keep the original self-return geometry unchanged. This is an
        # angular continuity test, not permission to mask extra near points.
        dense_rear = bool(len(support) >= 24 and min(bands) >= 8)
        bounded_edge = bool(dense_rear
                            and 82 - 1e-9 <= math.degrees(start) <= 110 + 1e-9
                            and math.degrees(end) <= 270 + 1e-9)
        shoulder = bool(bounded_edge
                        and len(intervals) > 1
                        and 145 - 1e-9 <= absolute[0] <= 150 + 1e-9
                        and intervals[0] <= math.radians(66) + 1e-9
                        and np.max(intervals[1:]) <= math.radians(45) + 1e-9)
        explained = (gap <= math.pi + 1e-9 and (rear or bounded_edge) and len(support) >= 5
                     and (unsupported <= math.radians(45) + 1e-9 or shoulder))
        arcs.append({"start_deg": math.degrees(start), "width_deg": math.degrees(gap),
                     "body_support_count": len(support), "unsupported_gap_deg": math.degrees(unsupported),
                     "recorded_rear_shoulder": shoulder,
                     "recorded_rear_edge": bounded_edge,
                     "rear_band_support_counts": bands,
                     "installed_body_explained": explained})
    return arcs, bool(arcs) and all(arc["installed_body_explained"] for arc in arcs)


def map_rotation_evidence(known_map, pose, radius_m, *, exact_glass_geometry=False):
    """Check every intersecting cell, including unknown and volatile obstacles.

The disk is deliberately conservative: lidar-centred 0.30 m + its 0.14 m
offset encloses the original envelope at every heading. No map is modified.
"""
    result = {"clear": False, "radius_m": float(radius_m)}
    if known_map is None or pose is None:
        return dict(result, reason="missing_map_or_pose")
    if not all(math.isfinite(float(v)) for v in (pose.x_m, pose.y_m, pose.yaw_rad)):
        return dict(result, reason="invalid_pose")
    result["pose"] = [float(pose.x_m), float(pose.y_m), float(pose.yaw_rad)]
    resolution = float(known_map.resolution_m)
    if not math.isfinite(resolution) or resolution <= 0 or not math.isfinite(radius_m) or radius_m <= 0:
        return dict(result, reason="invalid_geometry")
    extent = radius_m + resolution * math.sqrt(2) / 2
    corners = known_map.world_to_grid(((pose.x_m - extent, pose.y_m - extent),
                                      (pose.x_m + extent, pose.y_m + extent)))
    c0, r0 = corners[0]
    c1, r1 = corners[1]
    if not known_map.in_bounds(c0, r0) or not known_map.in_bounds(c1, r1):
        return dict(result, reason="outside_map")
    rows, columns = np.mgrid[r0:r1 + 1, c0:c1 + 1]
    world = known_map.grid_to_world(np.stack((columns, rows), axis=-1))
    intersecting = np.hypot(world[..., 0] - pose.x_m, world[..., 1] - pose.y_m) <= extent
    layers = {}
    for name in ("occupied", "unknown", "temporary_occupied"):
        blocked = intersecting & getattr(known_map, name)[rows, columns]
        indices = np.argwhere(blocked)
        layers[name] = {"count": int(len(indices)), "nearest_distance_m": None, "sample_world_xy_m": []}
        if len(indices):
            points = world[blocked]
            distances = np.hypot(points[:, 0] - pose.x_m, points[:, 1] - pose.y_m)
            order = np.argsort(distances)
            layers[name].update(nearest_distance_m=float(distances[order[0]]),
                                sample_world_xy_m=points[order[:5]].tolist())
    clear = not any(layer["count"] for layer in layers.values())
    glass = getattr(known_map, "_phase1_glass_keepout", None)
    if glass is not None:
        count = int(np.count_nonzero(intersecting & glass[rows, columns]))
        layers["verified_glass"] = {"count": count,
            "boundary_policy": getattr(known_map, "_phase1_glass_policy", "unspecified"),
            "field_verified": bool(getattr(known_map, "_phase1_glass_field_verified", False))}
        clear = clear and count == 0
        if exact_glass_geometry:
            # Keep real occupied/unknown/temporary cells blocking. Only the
            # glass layer has a separately stored continuous boundary: its
            # raster already includes half a cell diagonal, so intersecting
            # it with a second cell-expanded disk double-counts discretization.
            # Recheck the SAME physical disk + boundary uncertainty directly;
            # neither a caller's Boolean verdict nor a modified map is trusted.
            from .glass_boundary_guard import swept_glass_clearance
            continuous = swept_glass_clearance(known_map, pose, 0, 0,
                reliable=True, pose_age_s=0., ttl_ms=1, radius_m=radius_m)
            result['continuous_glass_disk'] = continuous
            clear = not any(layers[name]['count'] for name in
                            ('occupied','unknown','temporary_occupied')) and continuous['safe']
    core = getattr(known_map, "_phase1_temporary_core", None)
    if core is not None:
        temporary = intersecting & known_map.temporary_occupied[rows, columns]
        confirmed = temporary & core[rows, columns]
        unclassified = temporary & getattr(known_map, "_phase1_temporary_unclassified")[rows, columns]
        layers["temporary_occupied"].update(
            confirmed_core_count=int(np.count_nonzero(confirmed)),
            padding_only_count=int(np.count_nonzero(temporary & ~confirmed & ~unclassified)),
            unclassified_count=int(np.count_nonzero(unclassified)),
            padding_radius_m=getattr(known_map, "_phase1_temporary_padding_radius_m", None),
            provenance="confirmed_core_includes_one_cell_jitter; padding_remains_blocking")
    else:
        layers["temporary_occupied"]["provenance"] = "unavailable; all_cells_remain_blocking"
    return dict(result, clear=clear, reason="clear" if clear else "blocked_cells",
                intersecting_cells=int(np.count_nonzero(intersecting)), layers=layers)


def map_rotation_clear(known_map, pose, radius_m):
    return map_rotation_evidence(known_map, pose, radius_m)["clear"]


def assess_rotation(scan, radius_m=0.30, *, known_map=None, pose=None,
                    map_pose_reliable=False, map_pose_uncertainty_m=0.,
                    exact_glass_geometry=False):
    angles = np.asarray(scan.angles_rad, dtype=float)
    distances = np.asarray(scan.distances_m, dtype=float)
    result = {"safe": False, "reason": "invalid_scan", "radius_m": radius_m}
    if angles.ndim != 1 or angles.shape != distances.shape or not math.isfinite(radius_m) or radius_m < 0.30:
        return result
    legacy, patch = body_mask_components(angles, distances)
    body = legacy | patch
    patch_indices = np.flatnonzero(patch)
    result.update(body_mask_profile=BODY_MASK_PROFILE,
                  body_occlusion_policy=BODY_OCCLUSION_POLICY,
                  legacy_self_return_count=int(np.count_nonzero(legacy)),
                  installation_patch_count=len(patch_indices),
                  installation_patch_points=[{
                      "index": int(i), "bearing_deg": float((math.degrees(angles[i])+180)%360-180),
                      "distance_m": float(distances[i]),
                      "body_xy_m": [float(distances[i]*math.cos(angles[i])+.14),
                                    float(distances[i]*math.sin(angles[i]))]
                  } for i in patch_indices])
    valid = (np.isfinite(angles) & np.isfinite(distances) & (distances > 0)
             & (distances <= 8) & ~body)
    indices = np.flatnonzero(valid)
    result.update(self_return_count=int(np.count_nonzero(body)), exterior_count=len(indices))
    if not len(indices):
        result["reason"] = "no_exterior_evidence"
        return result
    nearest = int(indices[np.argmin(distances[indices])])
    result.update(min_distance_m=float(distances[nearest]),
                  min_bearing_deg=float((math.degrees(angles[nearest]) + 180) % 360 - 180),
                  min_body_xy_m=[float(distances[nearest] * math.cos(angles[nearest]) + 0.14),
                                 float(distances[nearest] * math.sin(angles[nearest]))])
    if distances[nearest] < radius_m:
        result["reason"] = "exterior_near_obstacle"
        return result
    if len(indices) < 16:
        result["reason"] = "insufficient_exterior_evidence"
        return result
    checked_radius = radius_m+.14+max(0., map_pose_uncertainty_m)
    map_evidence = map_rotation_evidence(known_map, pose, checked_radius,
                                        exact_glass_geometry=exact_glass_geometry)
    result.update(map_pose_reliable=bool(map_pose_reliable), map_support=map_evidence)
    if getattr(known_map, "_phase1_glass_keepout", None) is not None:
        if not map_pose_reliable or "layers" not in map_evidence:
            result["reason"] = "glass_pose_unreliable"
            return result
        if (map_evidence["layers"]["verified_glass"]["count"]
                and not map_evidence.get('continuous_glass_disk',{}).get('safe',False)):
            result["reason"] = "glass_sweep_blocked"
            return result
    if len(patch_indices) and not (map_pose_reliable and map_evidence["clear"]):
        result["reason"] = "installation_edge_occlusion_without_safe_map_support"
        return result
    bearings = np.sort(angles[valid] % (2 * math.pi))
    max_gap = float(np.max(np.diff(np.r_[bearings, bearings[0] + 2 * math.pi])))
    result["max_unobserved_gap_deg"] = math.degrees(max_gap)
    arcs, body_explained = body_supported_blind_arcs(angles, body, bearings)
    result.update(blind_arcs=arcs, installed_body_blind_arc=body_explained)
    # Unlike fixed octants, this cannot fail just because rays straddle a bin
    # boundary. A gap >45 degrees is still unknown, never imaginary free space.
    if max_gap <= math.radians(45) + 1e-9:
        result.update(safe=True, reason="installation_edge_occlusion_with_local_map_support"
                      if len(patch_indices) else "exterior_lidar_clear")
        if len(patch_indices):
            result["map_checked_radius_m"] = checked_radius
    elif ((max_gap <= math.radians(90) + 1e-9 or body_explained) and map_pose_reliable
          and map_evidence["clear"]):
        result.update(safe=True, reason="installed_body_occlusion_with_local_map_support" if body_explained
                      else "lidar_with_local_map_support",
                      map_checked_radius_m=checked_radius)
    else:
        result["reason"] = "unobserved_sector_without_safe_map_support"
    return result
