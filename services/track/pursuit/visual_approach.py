"""Multi-cue approach evidence, never a grasp gate or metric distance."""
import math


def box_size(observation,side):
    boxes=observation.get('target_boxes') or {}
    if not isinstance(boxes,dict):
        return None
    box=boxes.get(side)
    try:
        x1,y1,x2,y2=map(float,box)
        cx,cy=map(float,observation['centers'][side])
        if (not all(math.isfinite(v) for v in (x1,y1,x2,y2,cx,cy))
                or not 1<x1<x2<639 or not 1<y1<y2<479
                or not x1-2<=cx<=x2+2 or not y1-2<=cy<=y2+2):
            return None  # clipped edge or invalid/different-object geometry
        return x2-x1,y2-y1
    except (TypeError,ValueError,KeyError):
        return None


def recent_forward_valid(step,last_step,now,last_at,max_age_s=30):
    return 0<=step-last_step<=3 and 0<=now-last_at<=max_age_s


def approach_evidence(before,after,feedback,*,recent_forward=False,jitter_px=6,growth_ratio=1.03):
    changes=[]
    ratios={}
    invalid_geometry=False
    for side in ('left','right'):
        changes.append(float(after['centers'][side][1])-float(before['centers'][side][1]))
        a,b=box_size(before,side),box_size(after,side)
        for sample,size in ((before,a),(after,b)):
            boxes=sample.get('target_boxes') or {}
            if (not isinstance(boxes,dict) or side in boxes and size is None):
                invalid_geometry=True
        if a and b:
            wr,hr=b[0]/a[0],b[1]/a[1]
            ratios[side]=dict(width=wr,height=hr,area=wr*hr)
    disparity_delta=(after['centers']['left'][0]-after['centers']['right'][0]
                     -before['centers']['left'][0]+before['centers']['right'][0])
    visual=feedback.get('visual_feedback') or {}
    bg=visual.get('movement_px') or {}
    roi=visual.get('target_region_movement_px') or {}
    values=[float(bg.get(s,0)) for s in ('left','right')]
    locals_=[float(roi.get(s,0)) for s in ('left','right')]
    finite=(all(math.isfinite(v) and v>=0 for v in values+locals_)
            and all(math.isfinite(v) for v in changes+[disparity_delta]))
    strong_flow=finite and min(values)>=.8
    light_flow=(finite and all(max(a,b)>=.35 for a,b in zip(values,locals_))
                and max(values)>=.8)
    size_supported=(len(ratios)==2 and all(r['area']>=growth_ratio
        and min(r['width'],r['height'])>=.98 and max(r['width'],r['height'])<=1.5
        for r in ratios.values()) and min(changes)>=-2*jitter_px)
    size_contradiction=(any(r['area']<.88 or max(r['width'],r['height'])>1.5
                            for r in ratios.values())
                        or len(ratios)==2 and all(r['area']<.94 for r in ratios.values()))
    rows_supported=min(changes)>=3 and disparity_delta>=-4
    jitter_compatible=(min(changes)>=-jitter_px and disparity_delta>=-jitter_px)
    supported=not size_contradiction and not invalid_geometry and (
        rows_supported and strong_flow or size_supported and light_flow
        or recent_forward and jitter_compatible and light_flow)
    # Only reliable current evidence may replace the learned 4D response.
    learnable=not size_contradiction and not invalid_geometry and strong_flow and (rows_supported or size_supported)
    return dict(supported=bool(supported),learnable=bool(learnable),
        rows_supported=bool(rows_supported),size_supported=bool(size_supported),
        size_contradiction=bool(size_contradiction),recent_forward=bool(recent_forward),
        invalid_geometry=bool(invalid_geometry),
        jitter_compatible=bool(jitter_compatible),row_changes_px=changes,
        box_ratios=ratios,strong_flow=bool(strong_flow),light_flow=bool(light_flow))
