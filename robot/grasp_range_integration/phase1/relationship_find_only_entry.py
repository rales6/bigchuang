#!/usr/bin/env python3
"""Release-local real-car entry with the trained stereo camera order.

This wrapper changes no protected source file.  It installs the already
released relationship planner/localization hooks and limits the direct V4L2
camera enumeration to LEFT=/dev/video2, RIGHT=/dev/video0 for this process.
"""

from __future__ import annotations

from collections import deque
import json
import math
import os
from pathlib import Path
import sys
import time


RELEASE_DIR = Path(__file__).resolve().parents[2]
CAR_ROOT = Path(os.environ.get("CAR_ROOT", RELEASE_DIR.parent)).resolve()
for path in (RELEASE_DIR, CAR_ROOT):
    # Camera bootstrap adds track/ to PYTHONPATH. It contains an OLD copy of
    # raspberry_pi; being present later is not sufficient to select car1.0.
    # Reorder before importing any localization/navigation dependency.
    sys.path[:] = [entry for entry in sys.path if entry != str(path)]
    sys.path.insert(0, str(path))

from grasp_range_integration.phase1.motion_interlock import (
    ESTIMATE_MAX_AGE_S, ESTIMATE_MAX_TRAVEL_M,
    ESTIMATE_MAX_UNCERTAINTY_M, estimate_within_budget)


def _camera_nodes() -> list[str]:
    left = Path(os.environ.get("PHASE1_LEFT_CAMERA", "/dev/video2"))
    right = Path(os.environ.get("PHASE1_RIGHT_CAMERA", "/dev/video0"))
    if left == right:
        raise RuntimeError("第一阶段左右摄像头不能是同一设备")
    missing = [str(path) for path in (left, right) if not path.exists()]
    if missing:
        raise RuntimeError("第一阶段摄像头不存在：" + ", ".join(missing))
    return [str(left), str(right)]


def _install_waypoint_obstacle_guard(mission) -> None:
    """Use the protected guard's intended narrow known-map corridor.

    The protected helper documents a 42-degree corridor when waypoint
    tracking supplies an angular value, but object_search_mission's call site
    currently omits that optional argument and therefore uses the legacy
    60-degree envelope.  In this narrow room that repeatedly classifies both
    mapped side walls as a new frontal obstacle.  The distance envelope is not
    changed; this process-local adapter only selects the documented waypoint
    corridor.
    """
    original = mission._obstacle_safe
    original_build = mission.build_search_route
    original_fuse = mission._fuse_tracking_pose
    from grasp_range_integration.phase1.temporary_obstacle_evidence import install_temporary_evidence
    install_temporary_evidence(getattr(mission, "KnownOccupancyMap", None))
    context = {"map": None, "pose": None, "reliable": False, "updated": 0.0}
    waiting = {"frames": 0, "since": 0.0, "rechecked": False,
               "pending_replan": False, "report_replan": False, "assessment": None,
               "category": None, "cause": "rotation_map_support_recovery",
               "forward_view_replan": False, "view_change_replan": False}
    obstacle_evidence = {"votes": [], "seen": set()}
    from grasp_range_integration.phase1.map_margin_egress import MapMarginEgress
    margin_egress = MapMarginEgress()
    original_remember = getattr(mission, "_remember_temporary_obstacle", None)
    log_class = getattr(mission, "MissionLog", None)
    can_replan = original_remember is not None and log_class is not None
    support_reasons = {"installation_edge_occlusion_without_safe_map_support",
                       "unobserved_sector_without_safe_map_support",
                       "stale_scan_after_stationary_recheck"}

    def reset_wait():
        waiting.update(frames=0, since=0.0, rechecked=False, category=None)

    def obstacle_replan_allowed():
        from grasp_range_integration.phase1 import visibility_route_planner
        rule = str(((visibility_route_planner.ACTIVE_VIEW or {}).get("visibility_zone") or {}).get("stop_rule", ""))
        # A single-camera object recheck is explicitly stationary. Never turn
        # a blocked precision look into a drive towards an unlocated object.
        return can_replan and not rule.startswith("single_camera")

    def effective_scan(scan):
        from grasp_range_integration.phase1.navigation_freshness import effective_scan as latest
        return latest(context, scan)

    def fresh_for_navigation():
        from grasp_range_integration.phase1.navigation_freshness import ensure_navigation_freshness
        from grasp_range_integration.phase1.turn_policy import emit
        return ensure_navigation_freshness(context,
            getattr(mission, "_phase1_refresh_rotation_evidence", None), emit,
            clock=time.monotonic)

    def hold_navigation():
        from grasp_range_integration.phase1 import turn_policy
        turn_policy.LAST_REQUEST = (0, 0)
        hold = getattr(mission, "_phase1_hold_turn_convergence", None)
        if hold is not None:
            hold()

    def add_obstacle_vote(scan):
        # Require strong fresh map pose and distinct stopped scan acquisition.
        # Weak localization cannot paint a desk at an invented world position.
        scan = effective_scan(scan)
        received = getattr(scan, "received_s", None)
        identity = getattr(scan, "sequence", id(scan))
        if (not context["reliable"] or context.get("quality", {}).get("estimated_navigation", False)
                or time.monotonic()-context["updated"] > 1.
                or context.get("map") is None or context.get("pose") is None
                or received is None or time.monotonic()-received > .5
                or not context.get("quality", {}).get("post_stop_small_motion", False)
                or identity in obstacle_evidence["seen"]):
            return
        obstacle_evidence["seen"].add(identity)
        obstacle_evidence["votes"].append((scan, context["pose"]))
        del obstacle_evidence["votes"][:-8]

    def forward_turn_reserve_m():
        # Live body-frame lidar measures physical turn clearance directly.
        # Map-pose uncertainty must not enlarge this radius: the 20260929
        # run stopped on a side return ~0.8 m away after it grew to .743 m.
        # Keep the physical .44 m reserve; the independent front-obstacle and
        # glass-boundary checks still gate each commanded advance.
        return .44

    def assess(scan):
        from grasp_range_integration.phase1.rotation_guard import assess_rotation
        pair = context.get("clearance_pair")
        if pair is not None and scan is pair[0]:
            scan = pair[1]
        age = max(0.0, time.monotonic() - context["updated"])
        value = assess_rotation(scan, known_map=context["map"], pose=context["pose"],
                                map_pose_reliable=context["reliable"] and age <= 1.0,
                                map_pose_uncertainty_m=context.get("bridge_uncertainty_m", 0.))
        value["pose_quality"] = dict(context.get("quality", {}), age_s=age,
                                     fresh=age <= 1.0, reliable=context["reliable"])
        if getattr(mission, '_phase1_verified_glass_required', False):
            from grasp_range_integration.phase1.navigation_rotation_support import navigation_rotation_support
            value = navigation_rotation_support(value, scan, context, now=time.monotonic())
        received = getattr(scan, "received_s", None)
        # Re-localization/vision can take time. Never authorize a turn using
        # the old frame that started that stopped check.
        if received is not None:
            value["scan_age_s"] = max(0.0, time.monotonic() - received)
            if value["scan_age_s"] > .5 and value["reason"] != "exterior_near_obstacle":
                value.update(safe=False, reason="stale_scan_after_stationary_recheck")
        directory = context.get("audit_directory")
        count = context.get("audit_count", 0)
        if not value["safe"] and directory is not None and count < 24:
            from grasp_range_integration.phase1.inspect_body_returns import describe
            try:
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"clearance_{count:03d}.json"
                with path.open("x", encoding="utf-8") as stream:
                    json.dump({"frame": describe(scan), "assessment": value}, stream,
                              ensure_ascii=False, allow_nan=False)
                context["audit_count"] = count + 1
                value["raw_scan_path"] = str(path)
            except Exception as exc:
                value["raw_scan_save_error"] = str(exc)
        return value

    def authorize_motion(linear, angular, ttl):
        if not getattr(mission, "_phase1_verified_glass_required", False):
            return
        from grasp_range_integration.phase1.glass_boundary_guard import swept_glass_clearance
        from grasp_range_integration.phase1.turn_policy import emit
        if not fresh_for_navigation():
            hold_navigation()
            return False
        # This is the final output gate, including actual start-kick speed.
        # Recheck on the NEW scan, not the pre-map-fitting front-clear result.
        current = effective_scan(context.get("scan"))
        if not waypoint_obstacle_safe(current, linear, context.get("stop_distance_m", .45)):
            return False
        if linear > 0:
            from grasp_range_integration.phase1.navigation_avoidance import forward_reserve
            actual_sweep = forward_reserve(current, linear, angular,
                                           reserve_radius_m=forward_turn_reserve_m())
            emit("navigation_actual_output_assessed", **actual_sweep)
            if not actual_sweep["safe"]:
                if can_replan:
                    waiting.update(pending_replan=True, assessment=actual_sweep, cause="navigation_turn_space_recovery")
                hold_navigation()
                return False
        assessment = swept_glass_clearance(context["map"], context["pose"], linear, angular,
            reliable=context["reliable"], pose_age_s=time.monotonic()-context["updated"], ttl_ms=ttl,
            radius_m=.44+context.get("bridge_uncertainty_m", 0.))
        emit("glass_motion_assessed", **assessment)
        if not assessment["safe"]:
            if assessment["reason"] == "glass_sweep_blocked" and obstacle_replan_allowed():
                waiting.update(pending_replan=True, assessment=assessment, cause="navigation_glass_boundary_recovery")
            hold_navigation()
            emit("navigation_motion_held", reason=assessment["reason"])
            return False
        return True

    mission._phase1_authorize_motion = authorize_motion

    def reject(assessment):
        from grasp_range_integration.phase1 import turn_policy
        from grasp_range_integration.phase1.rotation_guard import RotationClearanceError
        turn_policy.emit("rotation_clearance_rejected", **assessment)
        raise RotationClearanceError(
            "旋转净空未确认：{}；最近外部回波={}m，方向={}°。已停车，未添加前方障碍。".format(
                assessment["reason"], assessment.get("min_distance_m"), assessment.get("min_bearing_deg")))

    # Before applying a turn, allow at most three ZERO-motion frames to obtain
    # fresh evidence. A single noisy point is never discarded as 'probably
    # chassis'; persistent near points still stop the task. No extra scan IO.
    if hasattr(mission, "ScanMapLocalizer") and hasattr(mission, "_strict_command_for_pose"):
        cls = mission.ScanMapLocalizer
        original_track = cls.track
        original_command = mission._strict_command_for_pose

        def tracked(self, scan, *args, **kwargs):
            context["localizer"] = self
            context["clearance_pair"] = None
            value = original_track(self, scan, *args, **kwargs)
            context["scan"] = scan
            if (waiting["frames"] >= 3 and waiting["category"] in {"map_support", "exterior_near_obstacle"}
                    and not waiting["rechecked"]):
                waiting["rechecked"] = True
                recheck = getattr(self, "_phase1_stationary_map_recheck", None)
                if recheck is not None:
                    recovered = recheck(scan, args[0] if args else kwargs.get("predicted_pose"))
                    if recovered is not None:
                        value = recovered
            return value

        def guarded_command(*args, **kwargs):
            from grasp_range_integration.phase1 import turn_policy
            observer = getattr(mission,'_phase1_live_observation',None)
            if observer is not None and len(args)>=3:
                from grasp_range_integration.phase1 import visibility_route_planner as live_planner
                rule=str(((live_planner.ACTIVE_VIEW or {}).get('visibility_zone') or {}).get('stop_rule',''))
                if not rule.startswith('single_camera'):
                    choice=observer.select(args[1],live_planner.LATEST_ROUTE or {},context.get('map'))
                    if choice['action']=='stopped_recheck':
                        hold_navigation()
                        return 0,0,True  # Reach protocol only; not mission success.
                    if choice['action']=='guided_waypoint' and not waiting['pending_replan']:
                        from copy import copy
                        guided_args=copy(args[0]);guided_args.heading_tolerance_deg=choice['heading_tolerance_deg']
                        guided_args._phase1_visual_guidance=True
                        args=(guided_args,args[1],choice['waypoint'],*args[3:])
                        turn_policy.emit('transit_visual_guidance',**choice)
            if waiting["pending_replan"]:
                hold_navigation()
                return 0, 0, False
            from grasp_range_integration.phase1 import visibility_route_planner as planner
            if margin_egress.origin is not None:
                escape=margin_egress.select(context,effective_scan(context.get('scan')),
                                           planner.LATEST_ROUTE,clock=time.monotonic)
                if escape['safe']:
                    turn_policy.LAST_REQUEST=(400,0)
                    turn_policy.emit('map_margin_egress',**escape)
                    return 400,0,False
            from grasp_range_integration.phase1.arrival_visibility import ArrivalCoverageError
            try:
                linear, angular, reached = original_command(*args, **kwargs)
            except ArrivalCoverageError as exc:
                if not can_replan:
                    raise
                from grasp_range_integration.phase1 import visibility_route_planner as planner
                xy = (planner.ACTIVE_VIEW or {}).get("view_xy_m")
                if xy and tuple(xy) not in planner.VISITED_POINTS:
                    planner.VISITED_POINTS.append(tuple(xy))
                waiting.update(pending_replan=True,cause="navigation_visibility_recovery",
                    assessment={"safe":False,"reason":"arrival_view_insufficient",
                                "detail":str(exc),"added_cells":0})
                hold_navigation()
                return 0,0,False
            if (linear or angular) and getattr(mission, "_phase1_verified_glass_required", False):
                if not fresh_for_navigation():
                    hold_navigation()
                    return 0, 0, False
            if linear != 0 or angular == 0 or context.get("scan") is None:
                reset_wait()
                return linear, angular, reached
            scan = effective_scan(context["scan"])
            received = getattr(scan, "received_s", None)
            refresh = getattr(mission, "_phase1_refresh_rotation_evidence", None)
            if received is not None and time.monotonic()-received > .5 and refresh is not None:
                try:
                    fresh = refresh(scan)
                    if fresh is not None:
                        current, small_motion = fresh
                        context["clearance_pair"] = (context["scan"], current)
                        context["reliable"] = context["reliable"] and small_motion
                        context["quality"]["post_stop_small_motion"] = small_motion
                except Exception as exc:
                    turn_policy.emit("rotation_fresh_clearance_unavailable", error=str(exc))
            assessment = assess(context["scan"])
            if assessment["safe"]:
                reset_wait()
                obstacle_evidence["votes"].clear()
                obstacle_evidence["seen"].clear()
                return linear, angular, reached
            if can_replan and obstacle_replan_allowed():
                escape=margin_egress.select(context,scan,planner.LATEST_ROUTE,assessment,
                                           clock=time.monotonic)
                if escape['safe']:
                    reset_wait()
                    hold_navigation()
                    turn_policy.LAST_REQUEST=(400,0)
                    turn_policy.emit('map_margin_egress',**escape)
                    print('转弯仅被地图误差余量拦住：前方雷达走廊已通过，先短距离直行换视角；沿途继续定位，不原地等待精准匹配。',flush=True)
                    return 400,0,False
            now = time.monotonic()
            category = ("stale_scan" if assessment["reason"] == "stale_scan_after_stationary_recheck"
                        else "map_support" if assessment["reason"] in support_reasons
                        else assessment["reason"])
            if category != waiting["category"]:
                reset_wait()
                waiting["category"] = category
            if not waiting["frames"]:
                waiting["since"] = now
            waiting["frames"] += 1
            support_missing = assessment["reason"] in support_reasons and can_replan
            near_obstacle = assessment["reason"] == "exterior_near_obstacle" and obstacle_replan_allowed()
            if near_obstacle:
                add_obstacle_vote(context["scan"])
            weak_obstacle_pose = near_obstacle and not context["reliable"]
            limit_frames, limit_seconds = (8, 10.0) if support_missing or weak_obstacle_pose else (3, 3.0)
            if waiting["frames"] >= limit_frames or now - waiting["since"] > limit_seconds:
                if not support_missing and not near_obstacle:
                    reject(assessment)
                cause = "rotation_obstacle_recovery" if near_obstacle else "rotation_map_support_recovery"
                waiting.update(pending_replan=True, assessment=assessment, cause=cause)
                turn_policy.emit(cause + "_requested", **assessment)
            turn_policy.LAST_REQUEST = (0, 0)
            hold = getattr(mission, "_phase1_hold_turn_convergence", None)
            if hold is not None:
                hold()
            turn_policy.emit("rotation_clearance_wait", wait_frame=waiting["frames"],
                             requested_angular_mrad_s=angular, **assessment)
            print("旋转净空暂未确认：{}；第{}帧零速复核，持续近障碍将进入定位复核及绕行重规划。".format(
                assessment["reason"], waiting["frames"]), flush=True)
            return 0, 0, False

        cls.track = tracked
        mission._strict_command_for_pose = guarded_command

    def build_route(known_map, *args, **kwargs):
        context["map"] = known_map
        route_args = list(args)
        view_change_requested = waiting['view_change_replan']
        waiting['view_change_replan'] = False
        if waiting["forward_view_replan"] and len(route_args) >= 2 and isinstance(route_args[1], dict):
            pose = context.get("pose")
            quality = context.get("quality") or {}
            if (pose is not None and context.get("reliable")
                    and quality.get("map_match") is True
                    and 0 <= time.monotonic()-context.get("updated", 0.) <= 1.
                    and float(quality.get("fused_delta_m", 1.)) <= .03):
                request = dict(route_args[1])
                request["_phase1_forward_view_yaw_rad"] = float(pose.yaw_rad)
                route_args[1] = request
                from grasp_range_integration.phase1 import turn_policy
                turn_policy.emit("rotation_forward_approach_requested",
                                 pose=[pose.x_m,pose.y_m,pose.yaw_rad],
                                 object_coordinate_used=False,
                                 pivot_motion_allowed=False)
        waiting["forward_view_replan"] = False
        reset_wait()
        obstacle_evidence["votes"].clear()
        obstacle_evidence["seen"].clear()
        if not hasattr(known_map, "_phase1_turn_reserve_path") and hasattr(known_map, "occupied"):
            from grasp_range_integration.phase1.navigation_avoidance import reserve_path
            original_path = known_map.path
            def reserved_path(start, goal):
                from grasp_range_integration.phase1 import visibility_route_planner
                pose = context.get('pose')
                quality = context.get('quality') or {}
                # Never plan this special straight egress from bounded-only
                # odometry, a stale pose, or a different route start point.
                trusted = (pose is not None and context.get('reliable')
                    and quality.get('map_match') is True
                    and 0 <= time.monotonic()-context.get('updated', 0.) <= 1.
                    and math.hypot(float(start[0])-pose.x_m,
                                   float(start[1])-pose.y_m) <= .08)
                return reserve_path(known_map, start, goal, original_path,
                    start_yaw_rad=pose.yaw_rad if trusted else None,
                    pose_uncertainty_m=float(quality.get('fused_delta_m', 0.)) if trusted else 0.,
                    allow_short_forward=bool(getattr(known_map, '_phase1_short_view_probe', False)
                        or str(((visibility_route_planner.ACTIVE_VIEW or {}).get('visibility_zone') or {}).get(
                            'stop_rule', '')) == 'locked_target_station_shift'))
            known_map.path = reserved_path
            known_map._phase1_turn_reserve_path = True
        from grasp_range_integration.phase1.visibility_route_planner import NoReachableVisibilityError
        try:
            if (view_change_requested and callable(getattr(mission, '_phase1_recover_route', None))
                    and context.get('pose') is not None and context.get('scan') is not None):
                # A syntactically valid A* route is not progress when the
                # SAME live clearance veto keeps rejecting its first turn.
                # Enter guarded short-motion recovery, rather than rebuilding
                # that route until the frozen generic replan budget expires.
                raise NoReachableVisibilityError('当前路段转向证据持续不足；换视角后再规划')
            return original_build(known_map, *route_args, **kwargs)
        except NoReachableVisibilityError:
            recover_route = getattr(mission, '_phase1_recover_route', None)
            pose = context.get('pose')
            scan = getattr(mission, '_phase1_navigation_scan', None) or context.get('scan')
            # Startup uses its own route recovery before loop variables exist.
            # Midrun recovery must hand pose AND the scan baseline back together.
            if not callable(recover_route) or pose is None or scan is None or len(route_args) < 2:
                raise
            prepared = {}
            def route_ready(candidate):
                try:
                    prepared['route'] = original_build(known_map,
                        (candidate.pose.x_m, candidate.pose.y_m), route_args[1], **kwargs)
                except NoReachableVisibilityError:
                    return False
                return bool(prepared['route'].get('views'))
            candidate = recover_route(scan, pose, accept_candidate=route_ready)
            if candidate is None:
                raise
            handoff = mission._phase1_localization_handoff
            handoff['old_route_start'] = tuple(route_args[0])
            known_map._phase1_route_handoff = handoff
            context.update(pose=candidate.pose, scan=mission._phase1_localization_handoff['final_scan'],
                           reliable=False, quality={'map_match': False}, updated=time.monotonic())
            mission._phase1_navigation_pose = candidate.pose
            return prepared['route']

    def fuse_pose(predicted_pose, result, *args, **kwargs):
        from grasp_range_integration.phase1.localization_handoff import recovery_prediction
        predicted_pose, handoff = recovery_prediction(mission, result, predicted_pose)
        if handoff is not None:
            context['scan'] = handoff['final_scan']
            context['clearance_pair'] = (handoff['source_scan'], handoff['final_scan'])
        fused = original_fuse(predicted_pose, result, *args, **kwargs)
        reliable = False
        quality = {"reason": "no_map_result"}
        if result is not None:
            distance = math.hypot(fused.x_m - result.pose.x_m, fused.y_m - result.pose.y_m)
            angle = math.atan2(math.sin(fused.yaw_rad - result.pose.yaw_rad),
                               math.cos(fused.yaw_rad - result.pose.yaw_rad))
            map_match = getattr(result, "score", 0.0) != -1.0
            reliable = (map_match and math.isfinite(result.rmse_m) and result.rmse_m <= 0.09
                        and result.inlier_ratio >= 0.65 and distance <= 0.20
                        and abs(angle) <= math.radians(10))
            quality = {"map_match": map_match, "rmse_m": float(result.rmse_m),
                       "inlier_ratio": float(result.inlier_ratio), "fused_delta_m": distance,
                       "fused_delta_deg": math.degrees(angle),
                       "reason": "strong_map_match" if reliable else "map_quality_insufficient"}
        estimate = getattr(mission, "_phase1_navigation_estimate", None)
        estimated = bool(estimate and result is estimate.get("result")
                         and estimate_within_budget(estimate, time.monotonic()))
        if estimated:
            # Bounded odometry is safe-navigation support, NOT a strong map
            # fit. Carry its uncertainty into rotation and glass sweeps.
            reliable = True
            quality.update(estimated_navigation=True, map_match=False,
                           reason="bounded_scan_odometry",
                           uncertainty_m=estimate["uncertainty_m"])
        context.update(pose=fused, reliable=reliable, quality=quality, updated=time.monotonic())
        mission._phase1_navigation_pose = fused
        if reliable and quality.get('reason') == 'strong_map_match' and context.get('scan') is not None:
            # Bind an ambiguous-turn recheck to the exact pre-turn scan and
            # its independently verified absolute map pose.
            mission._phase1_turn_pose_anchor = (context['scan'], fused)
        pair = None
        if getattr(mission, "_phase1_verified_glass_required", False):
            # The pose belongs to the acquisition used for map fitting, not
            # to the later instant at which slow fitting finished.
            received = getattr(context.get("scan"), "received_s", None)
            if received is not None:
                context["updated"] = received
            pair = getattr(mission, '_phase1_stationary_bridge_pair', None)
            if pair is not None and context.get('scan') is pair[0]:
                context['clearance_pair'] = pair
                context['updated'] = pair[1].received_s
                context['quality'].update(post_stop_small_motion=True, navigation_stationary_bridge=True)
            if handoff is not None:
                reset_wait()
                waiting.update(pending_replan=False, assessment=None)
                obstacle_evidence['votes'].clear()
                obstacle_evidence['seen'].clear()
        context["bridge_uncertainty_m"] = (estimate["uncertainty_m"] if estimated
            else quality.get("fused_delta_m", 0.) if reliable else 0.)
        if pair is not None and context.get('scan') is pair[0]:
            context['bridge_uncertainty_m'] += .05
        return fused

    def waypoint_obstacle_safe(scan, linear_mm_s, stop_distance_m):
        from grasp_range_integration.phase1 import turn_policy
        context["stop_distance_m"] = stop_distance_m
        scan = effective_scan(scan)
        if waiting["pending_replan"]:
            # Zero command, STOP and the existing bounded replan branch. The
            # remember/log wrappers below prevent this being painted as a wall.
            if linear_mm_s != 0 or turn_policy.LAST_REQUEST != (0, 0):
                raise RuntimeError("地图支持恢复必须保持零速")
            return False
        if linear_mm_s == 0 and turn_policy.LAST_REQUEST[1] != 0:
            assessment = assess(scan)
            turn_policy.emit("rotation_clearance_assessed", **assessment)
            if not assessment["safe"]:
                # The executor's finally STOP still runs. Do NOT return False:
                # that branch paints FRONT obstacles and consumes replan budget.
                if assessment["reason"] == "exterior_near_obstacle" and obstacle_replan_allowed():
                    waiting.update(pending_replan=True, assessment=assessment, cause="rotation_obstacle_recovery")
                    add_obstacle_vote(scan)
                    turn_policy.LAST_REQUEST = (0, 0)
                    return False
                reject(assessment)
            return True
        forward_safe = original(
            scan,
            linear_mm_s,
            stop_distance_m,
            angular_mrad_s=0.0,
        )
        if not forward_safe or linear_mm_s <= 0 or not can_replan:
            return forward_safe
        from grasp_range_integration.phase1.navigation_avoidance import forward_reserve
        assessment = forward_reserve(scan, linear_mm_s, turn_policy.LAST_REQUEST[1],
                                     reserve_radius_m=forward_turn_reserve_m())
        turn_policy.emit("navigation_turn_space_assessed", **assessment)
        if assessment["safe"]:
            return True
        waiting.update(pending_replan=True, assessment=assessment, cause="navigation_turn_space_recovery")
        turn_policy.LAST_REQUEST = (0, 0)
        # Forward scan was acquired while moving: do not use it as a stopped
        # map vote. The next bounded localization/route rebuild uses the map.
        return False

    if can_replan:
        original_log_init = log_class.__init__

        def log_init(self, *args, **kwargs):
            original_log_init(self, *args, **kwargs)
            if hasattr(self, "directory"):
                context.update(audit_directory=Path(self.directory) / "rotation_scans", audit_count=0)

        log_class.__init__ = log_init

        def remember(*args, **kwargs):
            if waiting["pending_replan"]:
                waiting.update(pending_replan=False, report_replan=True)
                added = 0
                if waiting["cause"] in {"rotation_obstacle_recovery", "navigation_turn_space_recovery"} and len(args) >= 5:
                    from grasp_range_integration.phase1.navigation_avoidance import record_stopped_obstacles
                    # Already confirmed stopped votes, each with its own strong
                    # pose. Never reuse a moving scan or invent rear sectors.
                    required = max(2, int(args[4].temporary_obstacle_min_hits))
                    votes = obstacle_evidence["votes"]
                    refresh = getattr(mission, "_phase1_refresh_rotation_evidence", None)
                    recheck = getattr(context.get("localizer"), "_phase1_stationary_map_recheck", None)
                    reference = effective_scan(args[0])
                    # The executor has ACKed STOP before this callback. Obtain
                    # up to three stopped scans, not moving obstacle samples.
                    # A moved/weak pose requires the existing bounded radar/DS
                    # check before any world-map writes. No blind retreat.
                    checked = False
                    if refresh is not None:
                        for _ in range(min(3, max(0, required-len(votes)))):
                            try:
                                fresh = refresh(reference)
                                if fresh is None: break
                                current, small = fresh
                                if (not small or not context["reliable"]) and not checked and recheck is not None:
                                    checked = True
                                    result = recheck(current, context["pose"])
                                    if result is not None:
                                        fuse_pose(context["pose"], result)
                                        small = True  # Pose was checked on this stopped scan.
                                context["clearance_pair"] = (args[0], current)
                                context.setdefault("quality", {})["post_stop_small_motion"] = small
                                add_obstacle_vote(args[0])
                                reference = current
                            except Exception as exc:
                                from grasp_range_integration.phase1 import turn_policy
                                turn_policy.emit("stopped_obstacle_evidence_unavailable", error=str(exc))
                                break
                    if len(votes) >= required:
                        for scan, pose in votes:
                            added += record_stopped_obstacles(scan, pose, args[3], args[4])
                waiting["added_cells"] = added
                obstacle_evidence["votes"].clear()
                obstacle_evidence["seen"].clear()
                return added
            return original_remember(*args, **kwargs)

        original_event = log_class.event

        def event(self, event_type, **fields):
            if event_type == "temporary_obstacle" and waiting["report_replan"]:
                waiting["report_replan"] = False
                added = waiting.get("added_cells", 0)
                cause = waiting["cause"]
                original_event(self, cause, assessment=waiting["assessment"], added_cells=added)
                # The protected pre-first-observation loop recognizes this
                # existing non-obstacle recovery event, NOT arbitrary new
                # event types. Keep that protocol and log the precise cause.
                event_type = "navigation_no_progress"
                pose = context.get("pose")
                fields.update(assessment=waiting["assessment"], added_cells=added,
                              recovery_cause=cause,
                              robot_pose=None if pose is None else [pose.x_m, pose.y_m, pose.yaw_rad])
                waiting["forward_view_replan"] = cause == "rotation_map_support_recovery"
                waiting['view_change_replan'] = cause == 'rotation_map_support_recovery'
                reset_wait()
            return original_event(self, event_type, **fields)

        mission._remember_temporary_obstacle = remember
        log_class.event = event

    mission._obstacle_safe = waypoint_obstacle_safe
    mission.build_search_route = build_route
    mission._fuse_tracking_pose = fuse_pose
    mission._phase1_rotation_assess = assess


def _phase1_directive(state: dict, route: dict | None, *, allow_low: bool) -> dict:
    """Keep observations in confidence order and pause before low confidence search."""
    event = str((state.get("latest_event") or {}).get("type") or "")
    detections = state.get("detections") or []
    stereo: dict[str, set[str]] = {}
    for item in detections:
        if not isinstance(item, dict):
            continue
        if item.get("task_region_eligible") is False:
            continue
        if item.get("confirmed_match") and item.get("dual_camera_confirmed"):
            pair = str(item.get("stereo_pair_id") or "")
            camera = str(item.get("camera_id") or "").lower()
            if pair and camera in {"left", "right"}:
                stereo.setdefault(pair, set()).add(camera)
    if any(cameras == {"left", "right"} for cameras in stereo.values()):
        return {"action": "report_success", "reason": "双摄确认目标颜色与类别", "terminal": True}
    if event in {"mission_timeout", "localization_failed", "perception_unavailable"}:
        return {"action": "report_failure", "reason": f"不可恢复事件：{event}", "terminal": True}
    if event in {"temporary_obstacle", "path_invalid", "navigation_no_progress"}:
        return {"action": "replan", "reason": "安全障碍或路径变化，重新规划", "terminal": False}
    maximum = int((route or {}).get("maximum_total_views") or 0)
    if maximum and int(state.get("attempts") or 0) >= maximum:
        return {"action": "report_failure", "reason": "观察及候选复查已达任务上限；未完成双摄确认", "terminal": True}
    remaining = [int(value) for value in state.get("remaining_view_indices") or []]
    if not remaining:
        if (
            bool((route or {}).get("next_station_available"))
            and int(state.get("attempts") or 0)
            < int((route or {}).get("maximum_total_views") or 0)
        ):
            return {
                "action": "replan",
                "reason": "当前站位所需朝向均已观察，规划下一处站位",
                "terminal": False,
            }
        return {"action": "report_failure", "reason": "已检查全部允许的视野，未确认目标", "terminal": True}
    next_index = remaining[0]
    views = (route or {}).get("views") or []
    if next_index >= len(views):
        return {"action": "report_failure", "reason": "视野索引与当前路线不一致", "terminal": True}
    zone = views[next_index].get("visibility_zone") or {}
    if zone.get("band") == "low" and not allow_low:
        return {
            "action": "report_failure",
            "reason": "高、中置信视野均未确认目标；低置信区域需要用户同意后继续（PHASE1_ALLOW_LOW_CONFIDENCE=1）",
            "terminal": True,
        }
    return {
        "action": "navigate_next_view",
        "next_view_index": next_index,
        "reason": (
            f"当前站位第{int(zone['sweep_index']) + 1}个方向；到位后停车双摄识别"
            if zone.get("sweep_index") is not None else
            "前往沿途观察视野并停车双摄识别"
        ),
        "station_xy_m": views[next_index].get("view_xy_m"),
        "target_yaw_deg": (
            round(math.degrees(float(views[next_index]["target_yaw_rad"])), 1)
            if views[next_index].get("target_yaw_rad") is not None else None
        ),
        "sweep_index": zone.get("sweep_index"),
        "sweep_count": zone.get("sweep_count"),
        "terminal": False,
    }


def _install_phase1_supervisor(mission, recovery=None) -> None:
    from grasp_range_integration.phase1 import visibility_route_planner

    allow_low = os.environ.get("PHASE1_ALLOW_LOW_CONFIDENCE", "0") == "1"

    def next_view(_gpu_client, state):
        route = visibility_route_planner.LATEST_ROUTE or {}
        event = state.get("latest_event") or {}
        near_target_seen = any(d.get('task_region_eligible') is True
                              and (d.get('single_camera_match') or d.get('confirmed_match'))
                              for d in state.get('detections') or [])
        # Transit photos are diagnostic evidence, not exhausted target views.
        # Handle this before target-recovery/failure rules: we have not started
        # searching the work area yet, and must not turn at the distant start.
        if route.get("navigation_stage") == "approach" and event.get("type") == "observation_complete":
            key = (id(route), state.get("current_view_index"))
            if getattr(mission, "_phase1_last_approach_photo", None) != key:
                visibility_route_planner.APPROACH_OBSERVATIONS += 1
                mission._phase1_last_approach_photo = key
                index = state.get('current_view_index')
                views = route.get('views') or []
                if isinstance(index, int) and 0 <= index < len(views):
                    old = tuple(float(v) for v in views[index]['view_xy_m'])
                    if old not in visibility_route_planner.VISITED_POINTS:
                        visibility_route_planner.VISITED_POINTS.append(old)
            visibility_route_planner.ACTIVE_VIEW = None
            if near_target_seen:
                # The actual capture gate already established useful range /
                # work-area visibility. Single-camera recovery takes priority
                # over generating another global station. It is NOT grasp
                # authorization; the normal stereo contract remains in force.
                route['navigation_stage'] = 'observation'
                route['physical_station_count'] = 1
            else:
                return {"action": "replan", "terminal": False,
                        "reason": "已到目标区域接近位置；现在才规划观察方向并寻找物品"}
        state = dict(state, attempts=max(0, int(state.get("attempts") or 0)
                     - visibility_route_planner.APPROACH_OBSERVATIONS))
        visibility_route_planner.OBSERVATIONS_COMPLETED = state["attempts"]
        if (event.get("type") == "navigation_no_progress" and
                event.get("recovery_cause") in {
                    "rotation_map_support_recovery",
                    "navigation_turn_space_recovery",
                    "rotation_obstacle_recovery",
                }):
            # A replan to the same physical station reproduces the same
            # clearance veto until the generic replan budget is exhausted.
            # Exclude only that station for this mission; the planner may
            # choose another map-clear view, while live lidar/glass guards
            # still authorize every actual movement.
            index = event.get("view_index")
            views = (visibility_route_planner.LATEST_ROUTE or {}).get("views") or []
            if isinstance(index, int) and 0 <= index < len(views):
                blocked_view = views[index]
                stop_rule = str((blocked_view.get("visibility_zone") or {}).get("stop_rule") or "")
                point = blocked_view.get("view_xy_m") or []
                pose = event.get("robot_pose")
                if pose is None:
                    current = getattr(mission, "_phase1_navigation_pose", None)
                    pose = None if current is None else [current.x_m, current.y_m]
                at_station = (pose is not None and len(pose) >= 2 and len(point) == 2
                              and math.hypot(float(pose[0])-float(point[0]),
                                             float(pose[1])-float(point[1])) <= .45)
                if (stop_rule in {"turn_at_station_then_capture", "forward_approach_then_capture"}
                        and at_station):
                    old = (float(point[0]), float(point[1]))
                    if old not in visibility_route_planner.VISITED_POINTS:
                        visibility_route_planner.VISITED_POINTS.append(old)
                        print("当前观察站的行驶/转向净空未确认；下次规划改选其他观察站，不再原地重复同一路线。", flush=True)
        if event.get("type") == "observation_complete":
            index = state.get("current_view_index")
            views = (visibility_route_planner.LATEST_ROUTE or {}).get("views") or []
            if isinstance(index, int) and 0 <= index < len(views):
                point = views[index]["view_xy_m"]
                old = (float(point[0]), float(point[1]))
                # A photo taken outside useful recognition range still
                # consumes this physical viewpoint. The frozen executor also
                # records it in visited_view_points. Keep the planner's list
                # identical before a replan, or its opportunistic index 0 is
                # filtered only by the executor while ACTIVE_VIEW still names
                # that stale index (224442: reached cabinet but skipped yaw).
                if old not in visibility_route_planner.VISITED_POINTS:
                    visibility_route_planner.VISITED_POINTS.append(old)
        route = visibility_route_planner.LATEST_ROUTE or {}
        directive = recovery.next(state, route) if recovery is not None and route.get("navigation_stage") != "approach" else None
        if directive is None:
            directive = _phase1_directive(state, route, allow_low=allow_low)
        if recovery is not None:
            directive["candidate_recovery_status"] = recovery.last_status
            if directive.get("recovery"):
                detail = directive["recovery"]
                print("目标视野恢复：相机={}，角度调整={:+.1f}°，地图净空约束换站位={}m；不以单摄框估计物体距离。".format(
                    detail["camera"], detail["step_deg"],
                    detail.get("station_shift_m", 0) if detail.get("linear_motion_allowed") else 0), flush=True)
        index = directive.get("next_view_index")
        views = (visibility_route_planner.LATEST_ROUTE or {}).get("views") or []
        visibility_route_planner.ACTIVE_VIEW = (
            views[index] if isinstance(index, int) and 0 <= index < len(views)
            else None
        )
        return directive

    mission.GpuMissionClient.next = next_view


def _inside_start_prior(pose, prior) -> bool:
    from grasp_range_integration.phase1.startup_prior import inside_start_prior
    return inside_start_prior(pose, prior)


def _visual_abstention_allows_lidar_start(
    status, diagnostics, candidate, *, accepted: int, total: int, ratio: float,
) -> bool:
    """A two-camera abstention is not positive evidence against a strong start pose."""
    if status != "uncertain" or not isinstance(diagnostics, dict):
        return False
    result = diagnostics.get("deepseek_result")
    return bool(
        diagnostics.get("engine") == "deepseek_vision_candidate_ranker"
        and diagnostics.get("camera_count") == 2
        and not diagnostics.get("deepseek_error")
        and not diagnostics.get("rejected_reason")
        and isinstance(result, dict)
        and result.get("selected_candidate_id") == "uncertain"
        and int(result.get("consistent_repeats") or 0) >= 2
        and accepted >= 4
        and total >= 4
        and ratio >= 0.80
        and candidate.reliable
        and candidate.rmse_m <= 0.10
        and candidate.inlier_ratio >= 0.70
    )


def _bounded_tracking_candidate(result, predicted_pose, mission, args) -> bool:
    if result is not None and getattr(result, "score", 0.) == -1.:
        return False  # An estimate must never reset its own map-loss budget.
    if not mission._tracking_usable_for_photo_mission(result, args):
        return False
    distance = math.hypot(
        result.pose.x_m - predicted_pose.x_m,
        result.pose.y_m - predicted_pose.y_m,
    )
    yaw_error = abs(math.atan2(
        math.sin(result.pose.yaw_rad - predicted_pose.yaw_rad),
        math.cos(result.pose.yaw_rad - predicted_pose.yaw_rad),
    ))
    glass_required = getattr(mission, "_phase1_verified_glass_required", False)
    return (distance <= (0.20 if glass_required else 0.30)
            and yaw_error <= math.radians(10.0 if glass_required else 15.0))


def _scan_odometry_usable(match, *, verified_large_turn=False) -> bool:
    if match is None or not match.converged or not math.isfinite(match.rmse_m):
        return False
    return (
        match.inlier_ratio >= 0.50
        and match.rmse_m <= 0.12
        and all(math.isfinite(v) for v in
                (match.transform.x_m, match.transform.y_m, match.transform.yaw_rad))
        and (abs(match.transform.yaw_rad) <= math.radians(24.0)
             or verified_large_turn)
    )


def _pose_on_known_free_map(localizer, pose) -> bool:
    known_map = getattr(localizer, "map", None)
    if known_map is None:
        return False
    column, row = known_map.world_to_grid(((pose.x_m, pose.y_m),))[0]
    column, row = int(column), int(row)
    return bool(
        known_map.in_bounds(column, row)
        and known_map.traversable[row, column]
    )


def _install_midrun_visual_recheck(
    mission, localizer, args, client, speed_controller, camera,
    visual_recognizer, gpu, visual_directory, event_callback,
    initial_scan=None,
    scan_iterator=None, stopping=lambda: False,
) -> None:
    """Bridge brief map dropouts before stopping for lidar/vision recovery.

    The mission's ordinary near-pose lidar retry remains first. DeepSeek is
    invoked once per consecutive failure episode, after the estimate budget
    is exhausted and two unsuccessful recovery frames have accumulated.
    A visual winner is usable only when it is also a good lidar fit close to
    the motion-predicted pose; a distant basin cannot cause a pose jump.
    """
    original_track = localizer.track
    # Use the SAME map-quality contract in the protected executor's fallback
    # and our recovery adapter as in the final glass motion authorization.
    # Otherwise weak fits are continually accepted, DS is never reached and
    # the vehicle sits at zero speed until a navigation replan is exhausted.
    if (getattr(mission, "_phase1_verified_glass_required", False)
            and not getattr(mission, "_phase1_tracking_quality_installed", False)):
        original_usable = mission._tracking_usable_for_photo_mission

        def navigation_usable(result, check_args):
            estimate = getattr(mission, "_phase1_navigation_estimate", None)
            if (estimate and result is estimate.get("result")
                    and estimate_within_budget(estimate, time.monotonic())):
                return True
            return bool(original_usable(result, check_args) and result is not None
                        and getattr(result, "score", 0.0) != -1.0
                        and math.isfinite(result.rmse_m) and result.rmse_m <= .09
                        and result.inlier_ratio >= .65)

        mission._tracking_usable_for_photo_mission = navigation_usable
        mission._phase1_tracking_quality_installed = True
    if (getattr(mission, "_phase1_verified_glass_required", False)
            and not getattr(localizer, "_phase1_bounded_navigation_retry", False)):
        original_near = localizer.localize_near_position

        def navigation_near(scan, hint, radius_m=.35):
            bounded = getattr(localizer, "localize_bounded", None)
            candidate = (bounded(scan, hint, radius_m=min(radius_m, .20),
                                 budget_s=.35, yaw_radius_deg=10)
                         if callable(bounded) else original_near(scan, hint, radius_m=min(radius_m, .20)))
            return candidate if _bounded_tracking_candidate(candidate, hint, mission, args) else None

        localizer.localize_near_position = navigation_near
        localizer._phase1_bounded_navigation_retry = True
    failures = 0
    checked = False
    attempt = 100
    last_scan = initial_scan
    bridge_started = None
    bridge_distance_m = 0.0
    bridge_rotation_rad = 0.0
    bridge_locked = False
    bridge_uncertainty_m = .05
    latest_motion = {"match": None, "extra_uncertainty_m": 0.,
                     "verified_large_turn": False}
    tracking = {'depth': 0}
    recovery_enabled = (scan_iterator is not None
                        and getattr(mission, '_phase1_verified_glass_required', False)
                        and callable(getattr(client, 'startup_recovery_pulse', None)))

    def reset_bridge(candidate, prediction):
        nonlocal bridge_started, bridge_distance_m, bridge_rotation_rad, bridge_locked
        nonlocal bridge_uncertainty_m, failures, checked
        mission._phase1_navigation_estimate = None
        failures, checked, bridge_locked = 0, False, False
        bridge_started, bridge_distance_m, bridge_rotation_rad = None, 0., 0.
        bridge_uncertainty_m = .05+math.hypot(
            candidate.pose.x_m-prediction.x_m, candidate.pose.y_m-prediction.y_m)

    def active_view_change(scan, predicted_pose, *, initial_fit=False, accept_candidate=None,
                           deferred_prediction=False):
        nonlocal last_scan
        if not recovery_enabled:
            return None
        from types import SimpleNamespace
        from grasp_range_integration.phase1.startup_active_recovery import recover_startup, RecoverySettings
        from grasp_range_integration.phase1.turn_policy import emit as telemetry
        if stopping() or getattr(mission, '_phase1_motion_fault', None):
            raise RuntimeError('途中换视角已取消或运动反馈故障；未授权恢复动作')
        def emit(name, **fields):
            telemetry(name, **fields)
            if event_callback is not None:
                event_callback(name, **fields)
        prior = SimpleNamespace(x_m=predicted_pose.x_m, y_m=predicted_pose.y_m,
            yaw_deg=math.degrees(predicted_pose.yaw_rad),
            radius_m=max(.10, bridge_uncertainty_m), yaw_tolerance_deg=30.)
        emit('midrun_view_change_requested', pose=[prior.x_m,prior.y_m,predicted_pose.yaw_rad],
             uncertainty_m=prior.radius_m, source_sequence=getattr(scan,'sequence',None),
             mission_navigation_enabled=False)
        print('途中定位尚未恢复：按实时雷达安全走廊换角度/小幅前移，获得新视角后复核；暂不寻物、不夹取。', flush=True)
        previous_mode = getattr(mission, '_phase1_recovery_mode', None)
        mission._phase1_recovery_mode = 'midrun'
        episode = getattr(mission, '_phase1_view_change_episode', 0)+1
        mission._phase1_view_change_episode = episode
        started = time.monotonic()
        try:
            recovered = recover_startup(mission, args, client, scan_iterator, localizer,
                speed_controller, stopping, prior, camera=camera, recognizer=visual_recognizer,
                gpu=gpu, directory=visual_directory, emit=emit, reference_scan=scan,
                accept_candidate=accept_candidate,
                settings=RecoverySettings(phase='midrun', time_budget_s=90.,
                    heading_tolerance_deg=30., operator_prompt=False, initial_fit=initial_fit,
                    visual_attempt_base=1000+episode*100))
        finally:
            client.stop()
            speed_controller.stopped()
            mission._phase1_recovery_mode = previous_mode
            progress = getattr(mission, '_phase1_angular_progress', None)
            if progress is not None:
                progress.suspend(started, time.monotonic())
        # Move the mission's pose prediction and scan baseline TOGETHER.
        # The frozen loop still owns its old scan variable until the next
        # iteration; the feedback adapter consumes this handoff exactly once.
        mission._phase1_localization_handoff = dict(result=recovered.candidate,
            source_scan=scan, final_scan=recovered.scan, predicted_pose=recovered.predicted_pose,
            localizer=localizer, pose_consumed=False, baseline_consumed=False,
            deferred_prediction=deferred_prediction)
        mission._phase1_stationary_bridge_pair = None
        reset_bridge(recovered.candidate, recovered.predicted_pose)
        mission._phase1_turn_relocalization_pending = False
        latest_motion['match'] = None
        last_scan = recovered.scan
        emit('midrun_view_change_complete',
             pose=[recovered.candidate.pose.x_m,recovered.candidate.pose.y_m,recovered.candidate.pose.yaw_rad],
             measured_prediction=[recovered.predicted_pose.x_m,recovered.predicted_pose.y_m,recovered.predicted_pose.yaw_rad],
             final_sequence=getattr(recovered.scan,'sequence',None), navigation_resume=True)
        print('途中换视角定位已恢复：x={:.3f}m y={:.3f}m yaw={:.1f}°；新扫描基准已交接，继续原寻物任务。'.format(
            recovered.candidate.pose.x_m, recovered.candidate.pose.y_m,
            math.degrees(recovered.candidate.pose.yaw_rad)), flush=True)
        return recovered.candidate
    mission._phase1_recover_route = lambda scan, pose, accept_candidate: active_view_change(
        scan, pose, initial_fit=False, accept_candidate=accept_candidate, deferred_prediction=True)
    original_prediction = getattr(mission, "_scan_motion_prediction", None)
    if original_prediction is not None:
        def predict(pose, match):
            nonlocal bridge_uncertainty_m
            from grasp_range_integration.phase1.localization_handoff import deferred_recovery_pose
            pose = deferred_recovery_pose(mission, pose)
            from grasp_range_integration.phase1.motion_interlock import predict_scan_motion
            unresolved_turn = (match is not None and match is getattr(
                mission, '_phase1_turn_relocalization_match', None))
            if unresolved_turn:
                mission._phase1_turn_relocalization_match = None
                mission._phase1_turn_relocalization_pending = True
                mission._phase1_post_turn_match = None
                hint_match = getattr(mission,
                    '_phase1_turn_relocalization_hint',None)
                mission._phase1_turn_relocalization_hint = None
                # A plausible but uncorroborated relative fit is only a *search
                # hint*. No navigation command is authorized from it until
                # the following wide map fit or active re-localization wins.
                plausible_hint = (hint_match is not None and hint_match.converged
                               and math.isfinite(hint_match.rmse_m)
                               and hint_match.rmse_m <= .12
                               and hint_match.inlier_ratio >= .42
                               and all(math.isfinite(v) for v in
                                   (hint_match.transform.x_m,hint_match.transform.y_m,
                                    hint_match.transform.yaw_rad)))
                hinted = pose.compose(hint_match.transform) if plausible_hint else pose
                hint_distance = (math.hypot(hint_match.transform.x_m,hint_match.transform.y_m)
                                 if plausible_hint else 0.)
                bridge_uncertainty_m = max(bridge_uncertainty_m, .25+hint_distance)
                latest_motion.update(match=None, extra_uncertainty_m=.25,
                                     verified_large_turn=False)
                localizer._wide_track_once = True
                client.stop()
                speed_controller.stopped()
                if event_callback is not None:
                    event_callback('turn_relocalization_map_search',
                                   hint=[hinted.x_m,hinted.y_m,hinted.yaw_rad],
                                   hint_from_relative_fit=plausible_hint,
                                   motion_command_allowed=False)
                return hinted, False
            verified_large_turn = (match is not None and match is getattr(
                mission, '_phase1_verified_large_turn_match', None))
            mission._phase1_verified_large_turn_match = None
            if verified_large_turn:
                # The ACKed pulse, raw ranges and absolute map fit already
                # established this full SE(2) motion. Reapply neither the old
                # fixed 24-degree cap nor an old pose after a real overshoot.
                predicted, usable = pose.compose(match.transform), True
                if event_callback is not None:
                    event_callback('large_turn_prediction_applied',
                                   predicted_pose=[predicted.x_m,predicted.y_m,predicted.yaw_rad],
                                   observed_rotation_deg=math.degrees(match.transform.yaw_rad),
                                   observed_translation_m=math.hypot(match.transform.x_m,
                                                                     match.transform.y_m),
                                   evidence=getattr(mission,
                                       '_phase1_turn_feedback_recovery_status',None))
            else:
                predicted, usable = predict_scan_motion(pose, match)
            if (not usable and match is not None and match.converged
                    and abs(match.transform.yaw_rad) > math.radians(24)):
                # An unverified large increment must be re-estimated, but it
                # is not a terminal mission/hardware fault. The following
                # track pass uses a bounded search and can change viewpoint.
                mission._phase1_turn_relocalization_pending = True
                localizer._wide_track_once = True
                client.stop()
                speed_controller.stopped()
                plausible = (math.isfinite(match.rmse_m) and match.rmse_m <= .12
                    and match.inlier_ratio >= .42 and all(math.isfinite(v) for v in
                    (match.transform.x_m,match.transform.y_m,match.transform.yaw_rad)))
                hinted = pose.compose(match.transform) if plausible else pose
                distance = math.hypot(match.transform.x_m,match.transform.y_m) if plausible else 0.
                bridge_uncertainty_m = max(bridge_uncertainty_m, .25+distance)
                latest_motion.update(match=None, extra_uncertainty_m=.25,
                                     verified_large_turn=False)
                if event_callback is not None:
                    event_callback("motion_prediction_recheck", terminal=False,
                                   reason="large_increment_requires_new_evidence",
                                   search_hint=[hinted.x_m,hinted.y_m,hinted.yaw_rad])
                return hinted, False
            evidence_status = getattr(mission,
                '_phase1_turn_feedback_recovery_status',None)
            latest_motion["verified_large_turn"] = verified_large_turn
            latest_motion["extra_uncertainty_m"] = (
                .08 if verified_large_turn and evidence_status in
                ('large_turn_bounded_scan_odometry',
                 'large_turn_reciprocal_odometry') else 0.)
            bounded_turn = getattr(mission, '_phase1_bounded_turn_match', None)
            mission._phase1_bounded_turn_match = None
            if usable and bounded_turn is not None and match is bounded_turn[0]:
                latest_motion["extra_uncertainty_m"] += float(bounded_turn[1])
            post_turn = getattr(mission, "_phase1_post_turn_match", None)
            mission._phase1_post_turn_match = None
            if usable and match is post_turn:
                distance = math.hypot(match.transform.x_m, match.transform.y_m)
                if event_callback is not None:
                    event_callback("turn_translation_applied", raw_translation_m=distance,
                                   applied_translation_m=distance,
                                   dx_m=match.transform.x_m, dy_m=match.transform.y_m,
                                   rotation_deg=math.degrees(match.transform.yaw_rad),
                                   predicted_pose=[predicted.x_m, predicted.y_m, predicted.yaw_rad])
            latest_motion["match"] = match if usable else None
            return predicted, usable
        mission._scan_motion_prediction = predict

    def track(scan, predicted_pose):
        nonlocal failures, checked, attempt, last_scan
        nonlocal bridge_started, bridge_distance_m, bridge_rotation_rad, bridge_locked
        nonlocal bridge_uncertainty_m
        mission._phase1_navigation_scan = scan
        motion = latest_motion["match"]
        if original_prediction is None and last_scan is not None and hasattr(localizer, "relative_motion"):
            motion = localizer.relative_motion(last_scan, scan)
        last_scan = scan
        # The protected mission has already applied this adjacent-scan motion
        # to predicted_pose. The map tracker's own scan-to-scan prediction
        # would otherwise apply the same transform a second time.
        if hasattr(localizer, "_last_track_scan"):
            localizer._last_track_scan = None
        if getattr(mission,'_phase1_turn_relocalization_pending',False):
            localizer._wide_track_once = True
        tracking['depth'] += 1
        try:
            result = original_track(scan, predicted_pose)
        finally:
            tracking['depth'] -= 1
        handoff = getattr(mission, '_phase1_localization_handoff', None)
        if handoff is not None and result is handoff['result'] and not handoff.get('pose_consumed'):
            # A clearance-triggered recovery inside original_track already
            # verified the moved pose. Do not compare it with the OLD hint.
            return result
        if _bounded_tracking_candidate(result, predicted_pose, mission, args):
            if getattr(mission,'_phase1_turn_relocalization_pending',False):
                mission._phase1_turn_relocalization_pending = False
                if event_callback is not None:
                    event_callback('turn_relocalization_complete',
                                   source='wide_map_track',
                                   pose=[result.pose.x_m,result.pose.y_m,
                                         result.pose.yaw_rad])
            mission._phase1_navigation_estimate = None
            failures, checked = 0, False
            if bridge_started is not None and event_callback is not None:
                event_callback("midrun_odometry_bridge", status="map_recovered",
                               distance_m=round(bridge_distance_m, 3))
            bridge_started, bridge_distance_m, bridge_rotation_rad = None, 0.0, 0.0
            bridge_locked = False
            bridge_uncertainty_m = .05 + math.hypot(
                result.pose.x_m-predicted_pose.x_m, result.pose.y_m-predicted_pose.y_m)
            return result
        if event_callback is not None and getattr(mission, "_phase1_verified_glass_required", False):
            event_callback("navigation_map_recovery", status="weak_track",
                           rmse_m=getattr(result, "rmse_m", None),
                           inlier_ratio=getattr(result, "inlier_ratio", None),
                           next_check="bounded_odometry_then_lidar_visual")
        if (
            not bridge_locked
            and not getattr(mission, '_phase1_turn_relocalization_pending', False)
            and not (getattr(mission, "_phase1_navigation_estimate", None) or {}).get(
                "command_budget_exhausted", False)
            and _scan_odometry_usable(motion, verified_large_turn=
                                     latest_motion["verified_large_turn"])
            and _pose_on_known_free_map(localizer, predicted_pose)
            and (not getattr(mission, "_phase1_verified_glass_required", False)
                 or 0 <= time.monotonic()-getattr(scan, "received_s", -math.inf) <= .5)
        ):
            step_distance = math.hypot(motion.transform.x_m, motion.transform.y_m)
            step_rotation = abs(motion.transform.yaw_rad)
            now = time.monotonic()
            started = now if bridge_started is None else bridge_started
            continuous = bool(getattr(mission, '_phase1_region_continuous_odometry', False)
                and motion.rmse_m <= .06 and motion.inlier_ratio >= .80)
            # Strong adjacent-scan evidence allows a slower uncertainty
            # growth model; it is still accumulated, never reset by itself.
            translation_growth = .08 if continuous else .15
            uncertainty = (bridge_uncertainty_m + translation_growth*step_distance + .08*step_rotation
                           + latest_motion["extra_uncertainty_m"])
            if (
                (continuous or now - started <= ESTIMATE_MAX_AGE_S
                 and bridge_distance_m + step_distance <= ESTIMATE_MAX_TRAVEL_M)
                and uncertainty <= ESTIMATE_MAX_UNCERTAINTY_M
            ):
                if bridge_started is None and event_callback is not None:
                    event_callback("midrun_odometry_bridge", status="started",
                                   distance_budget_m=None if continuous else ESTIMATE_MAX_TRAVEL_M,
                                   time_budget_s=None if continuous else ESTIMATE_MAX_AGE_S,
                                   policy='continuous_lidar_odometry' if continuous else 'bounded_dropout',
                                   uncertainty_budget_m=ESTIMATE_MAX_UNCERTAINTY_M)
                if bridge_started is None:
                    print("途中地图匹配暂弱：持续用雷达估计行驶并尝试地图纠偏；按位姿不确定度和实时净空限速。", flush=True)
                bridge_started = started
                bridge_distance_m += step_distance
                bridge_rotation_rad += step_rotation
                bridge_uncertainty_m = uncertainty
                failures, checked = 0, False
                # score=-1 explicitly marks a scan-to-scan estimate rather
                # than a map match in mission events and replay diagnostics.
                bridged = mission.LocalizationResult(
                    pose=predicted_pose,
                    rmse_m=motion.rmse_m,
                    inlier_ratio=motion.inlier_ratio,
                    score=-1.0,
                    ambiguity_margin=0.0,
                )
                mission._phase1_navigation_estimate = dict(result=bridged, started_s=started,
                    uncertainty_m=uncertainty, distance_m=bridge_distance_m,
                    rotation_rad=bridge_rotation_rad,
                    policy='continuous_lidar_odometry' if continuous else 'bounded_dropout',
                    last_evidence_s=getattr(scan, 'received_s', now))
                if event_callback is not None:
                    event_callback("navigation_estimated_pose", distance_m=bridge_distance_m,
                                   uncertainty_m=uncertainty, map_match=False,
                                   linear_limit_mm_s=400, ttl_limit_ms=250)
                return bridged
        mission._phase1_navigation_estimate = None
        # A failed ordinary map frame alone no longer issues STOP. Only an
        # exhausted/invalid estimate needs the slower bounded lidar/DS check.
        if getattr(mission, "_phase1_verified_glass_required", False):
            client.stop()
            speed_controller.stopped()
        near = localizer.localize_near_position(scan, predicted_pose, radius_m=0.35)
        if _bounded_tracking_candidate(near, predicted_pose, mission, args):
            if getattr(mission,'_phase1_turn_relocalization_pending',False):
                mission._phase1_turn_relocalization_pending = False
                if event_callback is not None:
                    event_callback('turn_relocalization_complete',
                                   source='near_map_search',
                                   pose=[near.pose.x_m,near.pose.y_m,
                                         near.pose.yaw_rad])
            failures, checked = 0, False
            if bridge_started is not None and event_callback is not None:
                event_callback("midrun_odometry_bridge", status="map_recovered",
                               distance_m=round(bridge_distance_m, 3))
            bridge_started, bridge_distance_m, bridge_rotation_rad = None, 0.0, 0.0
            bridge_locked = False
            bridge_uncertainty_m = .05 + math.hypot(
                near.pose.x_m-predicted_pose.x_m, near.pose.y_m-predicted_pose.y_m)
            return near
        if getattr(mission, '_phase1_turn_relocalization_pending', False) and recovery_enabled:
            # Do not return None through a frozen loop that retains its old
            # pose and advances its scan baseline: it would lose the turn
            # hint. Complete the scan/pose handoff in this recovery episode.
            return active_view_change(scan, predicted_pose, initial_fit=True)
        if bridge_started is not None and event_callback is not None:
            event_callback("midrun_odometry_bridge", status="budget_exhausted",
                           distance_m=round(bridge_distance_m, 3))
        if bridge_started is not None:
            bridge_locked = True
        bridge_started, bridge_distance_m, bridge_rotation_rad = None, 0.0, 0.0
        failures += 1
        if failures < 2 or checked:
            return None
        checked = True
        attempt += 1
        client.stop()
        speed_controller.stopped()
        candidate = near or result
        if candidate is None:
            try:
                candidates = localizer.global_localization_candidates(scan, maximum=1)
            except Exception:
                candidates = []
            candidate = candidates[0] if candidates else None
        if candidate is None:
            if event_callback is not None:
                event_callback("midrun_visual_recheck", status="no_lidar_candidate", accepted=False)
            return active_view_change(scan, predicted_pose)
        try:
            status, visual, match, diagnostics = mission._visual_localization_check(
                args, candidate, scan, localizer, camera, visual_recognizer,
                gpu, visual_directory, attempt,
            )
        except Exception as exc:
            status, visual, match, diagnostics = "error", None, None, {"error": str(exc)}
        distance = math.inf
        yaw_error = math.inf
        if visual is not None:
            distance = math.hypot(
                visual.pose.x_m - predicted_pose.x_m,
                visual.pose.y_m - predicted_pose.y_m,
            )
            yaw_error = abs(math.atan2(
                math.sin(visual.pose.yaw_rad - predicted_pose.yaw_rad),
                math.cos(visual.pose.yaw_rad - predicted_pose.yaw_rad),
            ))
        accepted = (
            status in {"confirmed", "corrected"}
            and match is not None
            and _bounded_tracking_candidate(visual, predicted_pose, mission, args)
            and distance <= 0.30
            and yaw_error <= math.radians(15.0)
        )
        if event_callback is not None:
            event_callback(
                "midrun_visual_recheck", status=status, accepted=accepted,
                position_delta_m=round(distance, 3) if math.isfinite(distance) else None,
                yaw_delta_deg=(
                    round(math.degrees(yaw_error), 1)
                    if math.isfinite(yaw_error) else None
                ),
                diagnostics=diagnostics,
            )
        if accepted:
            failures, checked = 0, False
            bridge_locked = False
            bridge_uncertainty_m = .05 + distance
            return visual
        return active_view_change(scan, predicted_pose)

    localizer.track = track

    def stationary_map_recheck_once(scan, predicted_pose):
        """One stopped, bounded correction per clearance episode, not a spin."""
        if predicted_pose is None:
            return None
        client.stop()
        speed_controller.stopped()
        def strong(candidate):
            return bool(candidate is not None
                        and getattr(candidate, "score", 0.0) != -1.0
                        and _bounded_tracking_candidate(candidate, predicted_pose, mission, args)
                        and math.isfinite(candidate.rmse_m) and candidate.rmse_m <= .09
                        and candidate.inlier_ratio >= .65
                        and math.hypot(candidate.pose.x_m-predicted_pose.x_m,
                                       candidate.pose.y_m-predicted_pose.y_m) <= .20
                        and abs(math.atan2(math.sin(candidate.pose.yaw_rad-predicted_pose.yaw_rad),
                                           math.cos(candidate.pose.yaw_rad-predicted_pose.yaw_rad))) <= math.radians(10))
        bounded = getattr(localizer, "localize_bounded", None)
        try:
            candidate = (bounded(scan, predicted_pose, radius_m=.20, budget_s=.35, yaw_radius_deg=10)
                         if callable(bounded) else
                         localizer.localize_near_position(scan, predicted_pose, radius_m=.20))
        except Exception as exc:
            if event_callback is not None:
                event_callback("rotation_stationary_map_recheck", status="lidar_error", accepted=False,
                               diagnostics={"error": str(exc)})
            return None
        status, diagnostics = "lidar_recheck", {}
        if not strong(candidate) and camera is not None and visual_recognizer is not None and gpu is not None:
            nonlocal attempt
            attempt += 1
            try:
                # Vision ranks measured lidar candidates; it never invents XY.
                seed = candidate or original_track(scan, predicted_pose)
                if seed is not None:
                    status, visual, match, diagnostics = mission._visual_localization_check(
                        args, seed, scan, localizer, camera, visual_recognizer,
                        gpu, visual_directory, attempt)
                    candidate = visual if status in {"confirmed", "corrected"} and match is not None else None
            except Exception as exc:
                status, candidate, diagnostics = "error", None, {"error": str(exc)}
        accepted = strong(candidate)
        if event_callback is not None:
            event_callback("rotation_stationary_map_recheck", status=status, accepted=accepted,
                           diagnostics=diagnostics, translation_limit_m=.20, yaw_limit_deg=10)
        return candidate if accepted else None

    def stationary_map_recheck(scan, predicted_pose):
        nonlocal bridge_started
        if predicted_pose is None:
            return None
        started = time.monotonic()
        candidate = stationary_map_recheck_once(scan, predicted_pose)
        # Exclude slow stopped scoring only after verifying a FULL new scan
        # with <=3cm/2deg motion. Never reset distance/angle/uncertainty budgets.
        refresh = getattr(mission, '_phase1_refresh_rotation_evidence', None)
        if callable(refresh):
            try:
                pair = refresh(scan)
            except Exception:
                pair = None
            current = pair[0] if pair is not None else None
            fresh_stationary = (current is not None and current is not scan and pair[1]
                and getattr(current, 'acquisition_start_s', -math.inf) >= started
                and 0 <= time.monotonic()-getattr(current, 'received_s', -math.inf) <= .5)
            if fresh_stationary:
                current = pair[0]
                elapsed = time.monotonic()-started
                if bridge_started is not None:
                    bridge_started += elapsed
                    estimate = getattr(mission, '_phase1_navigation_estimate', None)
                    if estimate is not None:
                        estimate['started_s'] += elapsed
                mission._phase1_stationary_bridge_pair = (scan, current)
                progress = getattr(mission, '_phase1_angular_progress', None)
                if progress is not None:
                    progress.suspend(started, time.monotonic())
                if event_callback is not None:
                    event_callback('midrun_stationary_time_excluded', excluded_s=round(elapsed,3),
                                   distance_and_uncertainty_unchanged=True,
                                   scan_sequence=getattr(current,'sequence',None))
        if candidate is None:
            if not tracking['depth']:
                # The obstacle-voting callback also uses this static check.
                # It cannot hand a moved pose to the frozen navigation loop;
                # recovery is performed only inside normal track().
                return None
            return active_view_change(scan, predicted_pose)
        return candidate

    localizer._phase1_stationary_map_recheck = stationary_map_recheck


def _install_stationary_start_localization(mission, prior) -> None:
    """For a confirmed staging area, verify lidar and ask stereo before motion.

    A stereo abstention is recorded separately from confirmation; only strong
    repeated lidar inside the fixed start prior may proceed without it.
    """
    if prior is None:
        original_localize = mission._localize

        def active_localize(args, client, scan_iterator, localizer,
                            speed_controller, stopping, **kwargs):
            pose, scan = original_localize(
                args, client, scan_iterator, localizer,
                speed_controller, stopping, **kwargs,
            )
            if hasattr(localizer, "track"):
                _install_midrun_visual_recheck(
                    mission, localizer, args, client, speed_controller,
                    kwargs.get("camera"), kwargs.get("visual_recognizer"),
                    kwargs.get("gpu"),
                    kwargs.get("visual_directory") or Path("search_runs"),
                    kwargs.get("event_callback"), scan,
                    scan_iterator=scan_iterator, stopping=stopping,
                )
            return pose, scan

        mission._localize = active_localize
        return

    def localize(args, client, scan_iterator, localizer, speed_controller,
                 stopping, *, camera=None, visual_recognizer=None, gpu=None,
                 visual_directory=None, event_callback=None,
                 status_callback=None):
        if camera is None or visual_recognizer is None or not visual_recognizer.available:
            raise RuntimeError("固定起点静态定位需要双摄和视觉参考库")
        client.stop()
        results = deque(maxlen=max(4, args.localization_window))
        deadline = time.monotonic() + args.localization_timeout_seconds
        from grasp_range_integration.phase1.startup_localization import StartupMatcher
        matcher = StartupMatcher(localizer, prior,
                                 search_budget_s=float(os.environ.get("PHASE1_START_SEARCH_BUDGET_SECONDS", "20")),
                                 verify_budget_s=float(os.environ.get("PHASE1_START_VERIFY_BUDGET_SECONDS", "2")))
        latest_scan = None
        candidate = None
        ready = False
        converged = False
        failure_reason = "insufficient_consistent_scans"
        accepted, total, ratio = 0, 0, 0.
        prior_rejection = None
        scan_outcomes = {}
        # Four consistent scans plus an independent stereo/semantic check
        # replace the generic entry's mandatory active motion for this one
        # explicitly staged start area.
        for _ in range(max(6, args.localization_window + 2)):
            if stopping() or time.monotonic() >= deadline:
                failure_reason = "stopped" if stopping() else "startup_deadline_exceeded"
                break
            try:
                latest_scan = next(scan_iterator)
            except StopIteration:
                failure_reason = "scan_stream_ended"
                break
            result = matcher.match(latest_scan, deadline)
            outcome = matcher.diagnostic["result"]
            scan_outcomes[outcome] = scan_outcomes.get(outcome, 0) + 1
            if matcher.diagnostic["result"] == "outside_start_prior":
                prior_rejection = dict(matcher.diagnostic)
            if event_callback is not None:
                event_callback("startup_scan_scored", **matcher.diagnostic)
            print("启动雷达：{} 耗时{:.3f}s，结果={}，位姿={}，起点范围检查={}。".format(
                matcher.diagnostic["mode"], matcher.diagnostic["elapsed_s"], matcher.diagnostic["result"],
                matcher.diagnostic.get("pose"), matcher.diagnostic.get("prior_check")), flush=True)
            if time.monotonic() >= deadline:
                failure_reason = "startup_deadline_exceeded"
                break
            results.append(result)
            ready, candidate, ratio, accepted, total = mission._localization_acceptance(
                results, args
            )
            from grasp_range_integration.phase1.startup_prior import assess_start_prior
            aggregate_prior = (assess_start_prior(candidate.pose, prior)
                               if candidate is not None else None)
            if event_callback is not None:
                event_callback("startup_lidar_consensus", ready=ready, accepted=accepted,
                               total=total, ratio=ratio, prior_check=aggregate_prior,
                               pose=None if candidate is None else [candidate.pose.x_m,
                                    candidate.pose.y_m, math.degrees(candidate.pose.yaw_rad)])
            print(
                f"固定起点静态定位：accepted={accepted}/{total} ({ratio:.0%})",
                flush=True,
            )
            if ready and accepted >= 4 and total >= 4:
                if candidate is not None and aggregate_prior["inside"]:
                    converged = True
                    break
                failure_reason = "consensus_outside_start_prior"
            elif matcher.diagnostic["result"] == "outside_start_prior":
                failure_reason = matcher.diagnostic["prior_check"]["reason"]
            else:
                failure_reason = "insufficient_consistent_scans"
        if not converged or candidate is None or latest_scan is None:
            client.stop()
            if failure_reason == "insufficient_consistent_scans" and not any(row is not None for row in results):
                if prior_rejection is not None:
                    failure_reason = prior_rejection["prior_check"]["reason"]
                elif scan_outcomes.get("budget_exhausted"):
                    failure_reason = "startup_scoring_budget_exhausted"
                elif scan_outcomes.get("no_reliable_match"):
                    failure_reason = "no_reliable_lidar_match"
            if event_callback is not None:
                event_callback("startup_lidar_rejected", reason=failure_reason, accepted=accepted,
                               total=total, last_scan=matcher.diagnostic, prior_rejection=prior_rejection,
                               scan_outcomes=scan_outcomes)
            if "outside_start" in failure_reason:
                message = "雷达候选超出固定起点范围（位置或朝向），未进入双摄复核"
            elif failure_reason in {"startup_deadline_exceeded", "startup_scoring_budget_exhausted"}:
                message = "固定起点雷达定位超时"
            elif failure_reason == "no_reliable_lidar_match":
                message = "固定起点没有达到质量要求的雷达匹配"
            elif failure_reason == "stopped":
                message = "固定起点雷达定位已取消"
            elif failure_reason == "scan_stream_ended":
                message = "固定起点雷达扫描流提前结束"
            else:
                message = "固定起点雷达扫描未达成至少四帧一致"
            raise RuntimeError(f"{message}（reason={failure_reason}，accepted={accepted}/{total}）；已停车")
        if event_callback is not None:
            event_callback("startup_lidar_anchor", pose=[candidate.pose.x_m, candidate.pose.y_m,
                           math.degrees(candidate.pose.yaw_rad)], accepted=accepted, total=total,
                           rmse_m=candidate.rmse_m, inlier_ratio=candidate.inlier_ratio)
        print("启动雷达锚点：x={:.3f}m y={:.3f}m yaw={:.1f}°；双摄仅复核此起点范围。".format(
            candidate.pose.x_m, candidate.pose.y_m, math.degrees(candidate.pose.yaw_rad)), flush=True)

        if status_callback is not None:
            status_callback("visual_localizing", "静态雷达一致，正在用双摄复核固定起点")
        from grasp_range_integration.phase1.startup_visual_review import (
            review_startup, pose_delta,
        )
        ds_module = None
        if getattr(mission, "_phase1_deepseek_enabled", False):
            import deepseek_visual_localization as ds_module
        for attempt in (1, 2):
            if stopping() or time.monotonic() >= deadline:
                client.stop()
                raise RuntimeError("启动定位复核超过总时间预算或已请求停止；未授权运动")
            visual_started = time.monotonic()
            try:
                status, visual_candidate, match, diagnostics = review_startup(
                    mission._visual_localization_check,
                    (args, candidate, latest_scan, localizer, camera,
                     visual_recognizer, gpu, visual_directory or Path("search_runs"), attempt),
                    candidate, prior, matcher.visual_candidates, ds_module,
                )
            except Exception as exc:
                status, visual_candidate, match = "error", candidate, None
                diagnostics = {"error": str(exc)}
            if event_callback is not None:
                event_callback("visual_localization_checked", status=status, attempt=attempt,
                               elapsed_s=round(time.monotonic() - visual_started, 3), **diagnostics)
            if stopping() or time.monotonic() >= deadline:
                client.stop()
                raise RuntimeError("启动定位复核超过总时间预算或已请求停止；未授权运动")
            if visual_candidate is None:
                visual_candidate = candidate
                status, match = "error", None
                diagnostics["rejected_reason"] = "startup_missing_visual_candidate"
            distance, angle = pose_delta(visual_candidate.pose, candidate.pose)
            verified = (status in {"confirmed", "corrected"} and match is not None
                        and _inside_start_prior(visual_candidate.pose, prior)
                        and distance <= .55 and angle <= 30.)
            abstained = _visual_abstention_allows_lidar_start(
                status, diagnostics, candidate, accepted=accepted, total=total, ratio=ratio)
            if verified or abstained or attempt == 2:
                break
            client.stop()
            if event_callback is not None:
                event_callback("startup_visual_retry", reason=diagnostics.get("rejected_reason", status),
                               next_attempt=2, motion_enabled=False)
            print("启动双摄尚未确认：原地获取两帧新雷达并重新拍摄复核一次；不转向、不前进。", flush=True)
            anchor = candidate
            for _ in range(2):
                if stopping() or time.monotonic() >= deadline:
                    client.stop()
                    raise RuntimeError("启动重试已超时或停止；未授权运动")
                try:
                    latest_scan = next(scan_iterator)
                except StopIteration as exc:
                    raise RuntimeError("启动重试缺少新雷达扫描；未授权运动") from exc
                fresh = matcher.match(latest_scan, deadline)
                if event_callback is not None:
                    event_callback("startup_scan_scored", retry=True, **matcher.diagnostic)
                if fresh is None or not _inside_start_prior(fresh.pose, prior):
                    raise RuntimeError("启动重试雷达验证失败；未授权运动")
                shift, rotation = pose_delta(fresh.pose, anchor.pose)
                if shift > .12 or rotation > 6:
                    raise RuntimeError("原地启动重试位姿发生变化；未授权运动")
                results.append(fresh)
            ready, refreshed, ratio, accepted, total = mission._localization_acceptance(results, args)
            if not ready or refreshed is None or not _inside_start_prior(refreshed.pose, prior):
                raise RuntimeError("启动重试雷达一致性不足；未授权运动")
            candidate = refreshed
        position_delta = math.hypot(
            visual_candidate.pose.x_m - candidate.pose.x_m,
            visual_candidate.pose.y_m - candidate.pose.y_m,
        )
        yaw_delta = abs(math.atan2(
            math.sin(visual_candidate.pose.yaw_rad - candidate.pose.yaw_rad),
            math.cos(visual_candidate.pose.yaw_rad - candidate.pose.yaw_rad),
        ))
        visual_verified = (
            status in {"confirmed", "corrected"}
            and match is not None
            and _inside_start_prior(visual_candidate.pose, prior)
            and position_delta <= 0.55
            and yaw_delta <= math.radians(30.0)
        )
        lidar_only_approved = (
            not visual_verified
            and _visual_abstention_allows_lidar_start(
                status, diagnostics, candidate,
                accepted=accepted, total=total, ratio=ratio,
            )
        )
        approved = visual_verified or lidar_only_approved
        if event_callback is not None:
            event_callback(
                "stationary_start_checked",
                verified=visual_verified,
                approved=approved,
                basis=("lidar_and_stereo" if visual_verified else
                       "lidar_start_prior_visual_abstention" if lidar_only_approved else
                       "rejected"),
                lidar_accepted=accepted,
                lidar_total=total,
                lidar_ratio=round(ratio, 3),
                lidar_rmse_m=round(candidate.rmse_m, 4),
                lidar_inlier_ratio=round(candidate.inlier_ratio, 4),
                lidar_pose=[candidate.pose.x_m, candidate.pose.y_m,
                            round(math.degrees(candidate.pose.yaw_rad), 2)],
                visual_attempts=attempt,
                position_delta_m=round(position_delta, 3),
                yaw_delta_deg=round(math.degrees(yaw_delta), 1),
                prior=[prior.x_m, prior.y_m, prior.yaw_deg,
                       prior.radius_m, prior.yaw_tolerance_deg],
            )
        if not approved:
            client.stop()
            raise RuntimeError(
                f"固定起点双摄复核未通过（{status}，位移差{position_delta:.2f}m，"
                f"朝向差{math.degrees(yaw_delta):.1f}°，"
                f"原因={diagnostics.get('rejected_reason', status)}，复核次数={attempt}）；已停车"
            )
        client.stop()
        selected = visual_candidate if visual_verified else candidate
        if status_callback is not None:
            status_callback(
                "localized",
                "固定起点雷达与双摄复核通过，无需主动移动车体"
                if visual_verified else
                "固定起点雷达高质量一致；双摄未能区分候选，按起点先验继续",
            )
        print(
            "固定起点静态定位{}：x={:.3f} y={:.3f} yaw={:.1f}°；启动阶段未移动。".format(
                "双摄确认" if visual_verified else "雷达先验通过（双摄弃选）",
                selected.pose.x_m, selected.pose.y_m,
                math.degrees(selected.pose.yaw_rad),
            ),
            flush=True,
        )
        if hasattr(localizer, "track"):
            _install_midrun_visual_recheck(
                mission, localizer, args, client, speed_controller, camera,
                visual_recognizer, gpu, visual_directory or Path("search_runs"),
                event_callback, latest_scan, scan_iterator=scan_iterator, stopping=stopping,
            )
        return selected.pose, latest_scan

    def localize_with_recovery(args, client, scan_iterator, localizer, speed_controller,
                               stopping, **kwargs):
        try:
            return localize(args, client, scan_iterator, localizer, speed_controller,
                            stopping, **kwargs)
        except RuntimeError as exc:
            # Only localization-evidence failures seek a new view. Cancellation,
            # device failure or unexpected program errors must not move the car.
            recoverable = ('双摄复核未通过', '原地启动重试位姿发生变化',
                           '启动重试雷达验证失败', '启动重试雷达一致性不足',
                           '固定起点没有达到质量要求', '固定起点雷达扫描未达成',
                           '雷达候选超出固定起点范围', '固定起点雷达定位超时')
            if (stopping() or not any(text in str(exc) for text in recoverable)
                    or not callable(getattr(client, 'startup_recovery_pulse', None))):
                raise
            from grasp_range_integration.phase1.startup_active_recovery import recover_startup
            callback = kwargs.get('event_callback')
            if callback:
                callback('startup_static_fallback', reason=str(exc), next_action='safe_view_change')
            print('原地启动定位证据不足：结构清楚时先做一次全朝向复核；仍不足再按实时雷达安全走廊连续起步/换角度，积累新视角后复核；不是起点寻物。', flush=True)
            try:
                pose, scan = recover_startup(mission, args, client, scan_iterator,
                    localizer, speed_controller, stopping, prior,
                    camera=kwargs.get('camera'), recognizer=kwargs.get('visual_recognizer'),
                    gpu=kwargs.get('gpu'), directory=kwargs.get('visual_directory') or Path('search_runs'),
                    emit=callback)
            finally:
                client.stop()
                if callable(getattr(speed_controller, 'stopped', None)):
                    speed_controller.stopped()
            if kwargs.get('status_callback'):
                kwargs['status_callback']('localized', '安全换视角后定位通过，开始任务导航')
            if hasattr(localizer, 'track'):
                _install_midrun_visual_recheck(mission, localizer, args, client, speed_controller,
                    kwargs.get('camera'), kwargs.get('visual_recognizer'), kwargs.get('gpu'),
                    kwargs.get('visual_directory') or Path('search_runs'), callback, scan,
                    scan_iterator=scan_iterator, stopping=stopping)
            return pose, scan

    mission._localize = localize_with_recovery


def main() -> None:
    from relationship_object_search_entry import install_relationship_planner
    from grasp_range_integration.phase1.turn_policy import install_turn_policy
    from grasp_range_integration.phase1.observation_recovery import install_observation_adapter

    install_relationship_planner()
    from grasp_range_integration.phase1.visibility_route_planner import (
        install_visibility_planner,
    )

    install_visibility_planner()
    import raspberry_pi.object_search_mission as mission
    from grasp_range_integration.phase1.ble_startup_recovery import install_ble_startup_recovery
    from raspberry_pi.map_camera_inspection import CameraCapture
    from deepseek_visual_localization import install as install_deepseek_vision
    from relationship_localization_guard import install as install_localization_guard

    CameraCapture._stable_usb_capture_nodes = staticmethod(_camera_nodes)
    install_ble_startup_recovery(mission)
    recovery = install_observation_adapter(mission, CameraCapture)
    from grasp_range_integration.phase1.fresh_motion_feedback import install_fresh_motion_feedback
    install_fresh_motion_feedback(mission)
    from grasp_range_integration.phase1.localization_performance import install_localization_performance
    install_localization_performance(mission)
    install_turn_policy(mission, dynamic_feedback=True)
    _install_waypoint_obstacle_guard(mission)
    _install_phase1_supervisor(mission, recovery)
    start_prior = install_localization_guard(mission)
    mission._phase1_deepseek_enabled = install_deepseek_vision(mission)
    _install_stationary_start_localization(mission, start_prior)
    from grasp_range_integration.phase1.startup_route_recovery import install_startup_route_recovery
    install_startup_route_recovery(mission)
    args = mission.build_argument_parser().parse_args()
    from grasp_range_integration.phase1.glass_boundary_guard import (
        DEFAULT_CONFIG, load_verified_boundaries, install_glass_boundary_guard,
    )
    boundary_file = os.environ.get("PHASE1_GLASS_BOUNDARIES", str(DEFAULT_CONFIG))
    boundaries = load_verified_boundaries(boundary_file, args.map, args.map_yaml)
    install_glass_boundary_guard(mission, boundaries)
    if boundaries.get("boundary_policy") == "user_confirmed_left_side_half_plane":
        print("玻璃墙策略：地图左侧斜边全段（含门位置）及外侧禁行；用户指定，地图估计+0.20m余量，非现场实测。", flush=True)
    args.real_test_max_distance = 0.0
    from grasp_range_integration.phase1.motion_experience import configure_motion_experience
    configure_motion_experience(mission, args)
    args.navigation_progress_distance = 0.025
    args.turn_speed = 2600
    if args.camera_stream_base_url:
        raise RuntimeError(
            "第一阶段必须传入空的--camera-stream-base-url，避免旧8080左右顺序覆盖训练契约"
        )
    print(
        "第一阶段相机契约：LEFT={} RIGHT={}（直连V4L2）".format(*_camera_nodes()),
        flush=True,
    )
    print("第一阶段导航保护：已启用已知地图42度前向走廊；停车距离包络未修改。", flush=True)
    print("前置N10车身过滤v2：保留0917基础掩膜，分别复核107–117°及-108–-102°窄安装边缘；新增遮挡必须有可靠定位及0.44m地图旋转净空支持。", flush=True)
    print("旋转地图支持不足：新帧/定位复核后改选其他观察站；不重复规划同一受阻站位，不新增假道路障碍，真近障碍保护不变。", flush=True)
    print("第一阶段雷达：后台持续读取，仅使用最新完整扫描；转向ACK及脉冲结束后新扫描配对。", flush=True)
    print("定位计算后扫描过时：停车新扫描验证≤3cm/2°，再复核障碍及玻璃边界；有位移则零速等待下一轮定位，不因单次计算耗时直接退出。", flush=True)
    print("第一阶段自适应转向：左右近期反馈独立；低进展逐步加力加时，不沿用历史最快值锁死短脉冲；有改善的超调允许减力反向纠正。", flush=True)
    print("转向执行：单次不续租，期间新雷达可提前刹车；转后新扫描验证仍强制执行，真实障碍/玻璃墙/异常反馈不绕过。", flush=True)
    print("导航进展：可靠实测转动且剩余角度减少2°也刷新35秒计时；单路径段仅转向最多120秒。左右成功参数保留，当前低进展不会反而降低已继承的轮输出。", flush=True)
    print("沿途单摄近期目标：只引导已验证通行且保持工作区域退距的短段；停车后的单摄复查仍原地小角度对准，不凭单目框推算物距。", flush=True)
    print("任务区域门检：接近区域沿途可后台识别；实际像素与模型一致可提前复拍，不因临界地图覆盖漏掉目标；远处无关候选不结束任务。", flush=True)
    print("停车双摄原图及带框结果独立保存到本次任务 observations/；index.html 可逐张复核。", flush=True)
    print("观察站以实际视野验收：临界覆盖先留照复查，仍不足则有界调整末段站位；临界单摄不能直接报成功。", flush=True)
    print("定位计算优化：同一对原始扫描复用ICP，地图射线采用相同距离采样批量计算；扫描时效、原始回波和转后保护阈值不放宽。", flush=True)
    region_job = os.environ.get("FINAL_WORK_REGION_JOB")
    if region_job:
        from final_work.pi.region_navigation import install_region_skill
        install_region_skill(mission, args, region_job)
    else:
        from grasp_range_integration.phase1.live_observation import install_live_observation
        observer=install_live_observation(mission)
        print('沿途双摄观察：{}秒采样、模型最小间隔{}秒；单摄近期目标引导安全短段，双摄候选提前停车复拍，入口确认后交接夹取。'.format(
            observer.settings.capture_interval_s,observer.settings.model_interval_s),flush=True)
    mission.run(args)


if __name__ == "__main__":
    main()
