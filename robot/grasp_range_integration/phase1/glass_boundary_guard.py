"""Map-bound glass keepouts: measured lines or user-designated whole-side policy."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_CONFIG = Path(__file__).with_name("glass_boundaries.json")
USER_LEFT_POLICY = "user_confirmed_left_side_half_plane"


def load_verified_boundaries(path, pgm_path, yaml_path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    user_side = data.get("boundary_policy") == USER_LEFT_POLICY
    if data.get("boundary_policy") not in (None, USER_LEFT_POLICY):
        raise ValueError("未知玻璃边界策略")
    if data.get("schema") != "phase1.verified_glass_boundaries.v1" or (data.get("verified") is not True and not user_side):
        raise RuntimeError("玻璃墙连续边界尚未现场核验；禁止自主运动。请填写并核验 " + str(path))
    if user_side:
        confirmation = data.get("user_confirmation") or {}
        if confirmation.get("source") != "user" or not str(confirmation.get("text", "")).strip():
            raise ValueError("整侧禁行策略必须有用户明确指定记录")
        if data.get("geometry_source") != "bound_map_occupied_outline_estimate":
            raise ValueError("整侧边界必须注明地图估计来源，不能冒充实测")
        reference = np.asarray(data.get("allowed_reference_xy_m"), dtype=float)
        if reference.shape != (2,) or not np.isfinite(reference).all():
            raise ValueError("整侧禁行策略缺少室内侧参考坐标")
    if not isinstance(data.get("verification_note"), str) or not data["verification_note"].strip():
        raise ValueError("玻璃边界缺少现场核验记录")
    for key, file in (("map_pgm_sha256", pgm_path), ("map_yaml_sha256", yaml_path)):
        if data.get(key) != hashlib.sha256(Path(file).read_bytes()).hexdigest():
            raise ValueError("玻璃边界与当前地图/坐标文件不匹配：" + key)
    segments = np.asarray(data.get("segments_xy_m"), dtype=float)
    if segments.ndim != 3 or segments.shape[1:] != (2, 2) or len(segments) == 0:
        raise ValueError("玻璃边界必须为非空的连续线段端点列表")
    if not np.isfinite(segments).all() or np.any(np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1) < .05):
        raise ValueError("玻璃边界坐标无效或线段过短")
    margin = float(data.get("boundary_uncertainty_m", float("nan")))
    if not math.isfinite(margin) or margin < .05:
        raise ValueError("玻璃边界误差余量至少为0.05m")
    if user_side and (len(segments) != 1 or margin < .20):
        raise ValueError("地图估计的整侧边界须为一条连续边且保留至少0.20m误差余量")
    data["boundary_uncertainty_m"] = margin
    return data


def distance_to_segments(points, segments):
    points = np.asarray(points, dtype=float)
    segments = np.asarray(segments, dtype=float)
    a, delta = segments[:, 0], segments[:, 1]-segments[:, 0]
    t = np.clip(np.sum((points[..., None, :]-a)*delta, axis=-1)/np.sum(delta*delta, axis=-1), 0., 1.)
    nearest = a+t[..., None]*delta
    return np.linalg.norm(points[..., None, :]-nearest, axis=-1).min(axis=-1)


def attach_boundaries(known_map, data):
    segments = np.asarray(data["segments_xy_m"], dtype=float)
    for column, row in known_map.world_to_grid(segments.reshape(-1, 2)):
        if not known_map.in_bounds(column, row):
            raise ValueError("玻璃边界端点在地图外；请检查坐标系")
    rows, columns = np.indices(known_map.occupied.shape)
    xy = known_map.grid_to_world(np.stack((columns, rows), axis=-1))
    extent = data["boundary_uncertainty_m"] + known_map.resolution_m*math.sqrt(2)/2
    if data.get("boundary_policy") == USER_LEFT_POLICY:
        a, b = segments[0]
        tangent = b-a
        normal = np.array([tangent[1], -tangent[0]])/np.linalg.norm(tangent)
        reference = np.asarray(data["allowed_reference_xy_m"], dtype=float)
        col, row = known_map.world_to_grid((reference,))[0]
        if not known_map.in_bounds(col, row) or not known_map.free[row, col]:
            raise ValueError("室内侧参考点不在地图自由区")
        if np.dot(reference-a, normal) < 0:
            normal = -normal
        if np.dot(reference-a, normal) <= extent:
            raise ValueError("室内侧参考点与玻璃估计边界过近")
        # Entire exterior half-plane, including beyond both endpoints. No
        # glass transmission or a route around a truncated segment can clear it.
        mask = (xy-a) @ normal <= extent
        known_map._phase1_glass_halfplane = (a.copy(), normal.copy())
    else:
        mask = distance_to_segments(xy, segments) <= extent
        if hasattr(known_map, "_phase1_glass_halfplane"):
            del known_map._phase1_glass_halfplane
    if not mask.any():
        raise ValueError("玻璃边界未覆盖任何地图栅格")
    known_map._phase1_glass_keepout = mask
    if hasattr(known_map, "_phase1_turn_reserve_mask"):
        del known_map._phase1_turn_reserve_mask
    known_map._phase1_glass_segments = segments
    known_map._phase1_glass_margin = float(data["boundary_uncertainty_m"])
    known_map._phase1_glass_policy = data.get("boundary_policy", "field_verified_segments")
    known_map._phase1_glass_field_verified = data.get("verified") is True
    # No localization, permanent occupancy or temporary-obstacle data changed.
    return mask


def swept_glass_clearance(known_map, pose, linear_mm_s, angular_mrad_s, *,
                          reliable, pose_age_s, ttl_ms=650, radius_m=.44):
    result = dict(safe=False, reason="glass_pose_unreliable")
    if pose is None or not reliable or not math.isfinite(pose_age_s) or not 0 <= pose_age_s <= .5:
        return result
    if not all(math.isfinite(v) for v in (pose.x_m, pose.y_m, pose.yaw_rad)):
        return result
    segments = getattr(known_map, "_phase1_glass_segments", None)
    if segments is None:
        return dict(result, reason="glass_boundary_missing")
    v, w = linear_mm_s/1000., angular_mrad_s/1000.
    horizon = pose_age_s + ttl_ms/1000. + .20
    times = np.linspace(0., horizon, 41)
    if abs(w) < 1e-6:
        local = np.column_stack((v*times, np.zeros_like(times)))
    else:
        local = np.column_stack((v*np.sin(w*times)/w, v*(1-np.cos(w*times))/w))
    c, s = math.cos(pose.yaw_rad), math.sin(pose.yaw_rad)
    world = local @ np.array([[c, s], [-s, c]]) + [pose.x_m, pose.y_m]
    halfplane = getattr(known_map, "_phase1_glass_halfplane", None)
    if halfplane is not None:
        a, normal = halfplane
        distance = float(((world-a) @ normal).min())
    else:
        distance = float(distance_to_segments(world, segments).min())
    reserve = radius_m + known_map._phase1_glass_margin + abs(v)*horizon/80.
    return dict(safe=distance > reserve, reason="glass_sweep_clear" if distance > reserve else "glass_sweep_blocked",
                boundary_policy=known_map._phase1_glass_policy,
                field_verified=known_map._phase1_glass_field_verified,
                minimum_boundary_distance_m=distance, required_boundary_distance_m=reserve,
                lookahead_s=horizon)


def bounded_recovery_glass_sweep(known_map, pose, linear, angular, *,
                                uncertainty_m, heading_tolerance_deg,
                                pose_age_s, ttl_ms, reserve_m, dispatch_margin_s=0.):
    """Enclose all bounded headings, without adding straight travel twice.

    Unbounded startup headings retain their full-disk check in the caller.
    Between sampled headings the extra chord reserve encloses the missing
    arcs, so this is not merely an optimistic center-line ray check.
    """
    tolerance = float(heading_tolerance_deg)
    if (not math.isfinite(tolerance) or not 0 <= tolerance <= 45
            or not math.isfinite(uncertainty_m) or uncertainty_m < 0):
        return dict(safe=False,reason='unbounded_recovery_heading')
    horizon = pose_age_s+ttl_ms/1000.+.20+dispatch_margin_s
    step = math.radians(2*tolerance/8)
    chord = abs(linear)/1000.*horizon*math.sin(step/2)
    radius = .44+uncertainty_m+reserve_m+chord+abs(linear)/1000.*dispatch_margin_s
    rows=[]
    from raspberry_pi.mapping import Pose2D
    for offset in np.linspace(-tolerance,tolerance,9):
        p=Pose2D(pose.x_m,pose.y_m,pose.yaw_rad+math.radians(float(offset)))
        rows.append(swept_glass_clearance(known_map,p,linear,angular,
                    reliable=True,pose_age_s=pose_age_s,ttl_ms=ttl_ms,radius_m=radius))
    worst=min(rows,key=lambda r:r.get('minimum_boundary_distance_m',-math.inf)
              -r.get('required_boundary_distance_m',math.inf))
    return dict(worst,safe=all(r['safe'] for r in rows),
                authority='bounded_heading_actual_command_sweep',
                heading_tolerance_deg=tolerance,heading_samples=9,
                interpolation_reserve_m=chord,position_uncertainty_m=uncertainty_m,
                translation_reserve_m=reserve_m,travel_double_counted=False)


def install_glass_boundary_guard(mission, data):
    original = mission.KnownOccupancyMap

    class GlassMap(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            attach_boundaries(self, data)

    mission.KnownOccupancyMap = GlassMap
    mission._phase1_verified_glass_required = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--map", required=True)
    parser.add_argument("--map-yaml", required=True)
    args = parser.parse_args()
    try:
        data = load_verified_boundaries(args.config, args.map, args.map_yaml)
        from raspberry_pi.known_map_navigation import KnownOccupancyMap
        attach_boundaries(KnownOccupancyMap.load(args.map, args.map_yaml), data)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, "未启动：" + str(exc) + "\n")
    if data.get("boundary_policy") == USER_LEFT_POLICY:
        print("玻璃边界配置通过：用户指定左侧整段及外侧禁行，几何取自地图估计（非现场实测）；未连接硬件。")
    else:
        print("玻璃连续边界核验配置通过；未连接底盘、雷达或摄像头。")


if __name__ == "__main__":
    main()
