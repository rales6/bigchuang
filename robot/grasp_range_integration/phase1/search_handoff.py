"""Live target admission and independently measured absolute-map anchors.

These are handoff checks, not motor commands or grasp-success shortcuts.
"""
import math
import time


def near_region_geometry(route, pose):
    """Approximate task membership only, not object identity or motor clearance."""
    result = {'near': False}
    if route.get('mode') != 'semantic_visibility_regions':
        return result
    region = route.get('target_region') or {}
    try:
        x, y, yaw = map(float, pose)
        cx, cy = map(float, region['center_xy_m'])
        radius = float(region['radius_m'])
        low, high = map(float, region['recognition_distance_m'])
        fov = float(region.get('camera_horizontal_fov_deg', 82))
    except (TypeError, ValueError, KeyError):
        return result
    if not all(math.isfinite(v) for v in (x,y,yaw,cx,cy,radius,low,high,fov)):
        return result
    if not radius > 0 or not 0 < low < high <= 3 or not 45 <= fov <= 120:
        return result
    distance = math.hypot(x-cx,y-cy)
    # 25cm allowance applies only to approximate region membership; it does
    # not shrink obstacle/glass clearance or estimate image-object depth.
    limit = radius + min(high, 1.45) + .25
    error = abs((math.atan2(cy-y,cx-x)-yaw+math.pi)%(2*math.pi)-math.pi)
    region_angle = math.asin(min(1., radius/max(distance,1e-6)))
    if distance > limit or error > math.radians(fov/2+10)+region_angle:
        return result
    return dict(near=True, region_center_distance_m=round(distance,3),
                admission_distance_limit_m=limit, semantic_position_allowance_m=.25)


def boxes_overlap(box, peer):
    if len(box or [])!=4 or len(peer or [])!=4:
        return False
    ix = max(0.,min(box[2],peer[2])-max(box[0],peer[0]))
    iy = max(0.,min(box[3],peer[3])-max(box[1],peer[1]))
    area = (box[2]-box[0])*(box[3]-box[1])
    other = (peer[2]-peer[0])*(peer[3]-peer[1])
    union = area+other-ix*iy
    return union > 0 and ix*iy/union >= .15


def ground_colour_boxes(checked, cues):
    """A model's green/red box must overlap independently observed pixels."""
    from copy import deepcopy
    result = deepcopy(checked)
    for item in result:
        if item.get('single_camera_match') and not any(
                c.get('camera_id')==item.get('camera_id') and
                boxes_overlap(item.get('bbox_1000'),c.get('bbox_1000')) for c in cues):
            item.update(single_camera_match=False,confirmed_match=False,dual_camera_confirmed=False)
            item.pop('stereo_pair_id',None)
            item.setdefault('search_evidence',{}).update(
                bbox_valid=False,pixel_rejected_reason='model_box_misses_target_colour')
    # One rejected member also invalidates its peer's stereo claim.
    valid_pairs={i.get('stereo_pair_id') for i in result if i.get('dual_camera_confirmed')}
    for pair in valid_pairs:
        if {i.get('camera_id') for i in result if i.get('stereo_pair_id')==pair
                and i.get('single_camera_match')} != {'left','right'}:
            for i in result:
                if i.get('stereo_pair_id')==pair:
                    i.update(confirmed_match=False,dual_camera_confirmed=False)
                    i.pop('stereo_pair_id',None)
    return result


def live_target_gate(gate, route, pose, checked, pixel_cues):
    """Near, pixel-grounded single eye permits recheck, never stereo success."""
    if gate.get('eligible') is not False or gate.get('reason') != 'approaching_task_region':
        return gate
    geometry = near_region_geometry(route,pose)
    if not geometry['near']:
        return gate
    eyes = set()
    for item in checked:
        evidence = item.get('search_evidence') or {}
        if (not item.get('single_camera_match') or not item.get('complete_in_frame')
                or not all(evidence.get(k) is True for k in
                           ('source_ok','class_ok','colour_ok','bbox_valid','bbox_complete'))
                or float(item.get('confidence') or 0) < .75):
            continue
        box = item.get('bbox_1000') or []
        for cue in pixel_cues:
            peer = cue.get('bbox_1000') or []
            if cue.get('camera_id') != item.get('camera_id') or len(box)!=4 or len(peer)!=4:
                continue
            if boxes_overlap(box,peer):
                eyes.add(item['camera_id'])
    if not eyes:
        return gate
    return dict(gate, eligible=True, reason='live_target_near_task_region',
                **geometry, visual_evidence_cameras=sorted(eyes),
                visual_evidence_requires_confirmation=True, object_depth_estimated=False)


def absolute_map_range_evidence(localizer, candidate, scan, deadline, *, clock=time.monotonic):
    """Raycast THIS absolute pose against THIS raw scan, not unset fit fields."""
    profile_fn = getattr(localizer, 'range_profile', None)
    probe = getattr(localizer, '_range_consistency', None)
    result = dict(accepted=False, source='absolute_map_raw_ranges',
                  sequence=getattr(scan,'sequence',None), range_agreement=None,
                  range_mean_error_m=None, range_coverage=None)
    if not callable(profile_fn) or not callable(probe):
        return dict(result,reason='absolute_range_probe_unavailable')
    if clock() >= deadline:
        return dict(result,reason='absolute_range_probe_deadline')
    try:
        measured = probe(candidate.pose, profile_fn(scan,maximum=36),
                         deadline_s=min(deadline,clock()+.35))
    except TimeoutError:
        return dict(result,reason='absolute_range_probe_deadline')
    if measured is None:
        return dict(result,reason='absolute_range_probe_incomplete')
    agreement, mean, coverage = measured
    result.update(range_agreement=agreement,range_mean_error_m=mean,range_coverage=coverage)
    accepted = (all(math.isfinite(v) for v in measured) and candidate.inlier_ratio >= .70
                and agreement >= .70 and coverage >= .50 and clock() < deadline)
    return dict(result,accepted=accepted,
                reason='absolute_map_ranges_verified' if accepted else 'absolute_map_ranges_disagree')
