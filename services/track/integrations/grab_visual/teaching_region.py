"""Per-eye pixel hulls, independent of Torch, hardware and model weights."""
import math

REGION_MODE = 'dual_2d_shell'
GRASP_MARGIN_PX = 25.0


def convex_polygon(points):
    points=sorted(set(tuple(map(float,p)) for p in points))
    if not points or any(len(p)!=2 or not all(math.isfinite(v) for v in p) for p in points):
        raise ValueError('invalid teaching polygon')
    if len(points)<3:
        return points
    def cross(o,a,b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    low,high=[],[]
    for p in points:
        while len(low)>=2 and cross(low[-2],low[-1],p)<=0:
            low.pop()
        low.append(p)
    for p in reversed(points):
        while len(high)>=2 and cross(high[-2],high[-1],p)<=0:
            high.pop()
        high.append(p)
    return low[:-1]+high[:-1]


def polygon_distance(point,polygon):
    p=tuple(map(float,point))
    if len(p)!=2 or not all(math.isfinite(v) for v in p):
        raise ValueError('invalid target center')
    polygon=convex_polygon(polygon)
    if len(polygon)==1:
        return math.dist(p,polygon[0]),polygon[0]
    edges=list(zip(polygon,polygon[1:]+polygon[:1]))
    if len(polygon)>=3 and all(
        (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])>=-1e-8 for a,b in edges):
        return 0.0,p
    candidates=[]
    for a,b in edges:
        dx,dy=b[0]-a[0],b[1]-a[1]
        length=dx*dx+dy*dy
        t=max(0,min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/length))
        q=(a[0]+t*dx,a[1]+t*dy)
        candidates.append((math.dist(p,q),q))
    return min(candidates)


def dual_view_region(centers,hull,width=640,height=480):
    if not hull or any(len(v)!=4 or not all(math.isfinite(float(x)) for x in v) for v in hull):
        raise ValueError('invalid joint teaching data')
    distances,nearest,polygons={}, {}, {}
    for side,offset in (('left',0),('right',2)):
        center=tuple(map(float,centers[side]))
        if len(center)!=2 or not 0<=center[0]<=width or not 0<=center[1]<=height:
            raise ValueError('target center outside camera frame')
        polygon=convex_polygon([(float(v[offset])*width,float(v[offset+1])*height) for v in hull])
        d,q=polygon_distance(center,polygon)
        distances[side]=float(d)
        nearest[side]=list(q)
        polygons[side]=[list(p) for p in polygon]
    return dict(grasp_region_mode=REGION_MODE,view_hull_distances_px=distances,
        view_hull_centers=nearest,view_hull_polygons_px=polygons,
        inside_view_hulls={s:distances[s]<=1e-8 for s in distances},
        grasp_region_ok=all(d<=GRASP_MARGIN_PX for d in distances.values()),
        grasp_margin_px=GRASP_MARGIN_PX)


def region_accepts(observation,maximum_px=GRASP_MARGIN_PX):
    """Require both measured distances and service verdict; no 4D veto."""
    if observation.get('grasp_region_mode')!=REGION_MODE or observation.get('grasp_region_ok') is not True:
        return False
    if observation.get('inside_teaching_hull') is not True and observation.get('projected_to_teaching_boundary') is not True:
        return False  # outside model inputs must use the original calibrated boundary
    try:
        # Reject a mixed deployment: the predictor and actuator must agree
        # on the exact same per-eye dilation, not just today's distances.
        margin=float(observation['grasp_margin_px'])
        if not math.isfinite(margin) or margin!=maximum_px:
            return False
        gaps=observation['view_hull_distances_px']
        return all(math.isfinite(float(gaps[s])) and 0<=float(gaps[s])<=maximum_px for s in ('left','right'))
    except (KeyError,TypeError,ValueError):
        return False
