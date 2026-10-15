"""Version-independent reciprocal raw-beam check for phase-one turn feedback.

Only uses recorded scan arrays and the established Pose2D fields. Neither
turn_range_evidence nor range_profile is required on the protected localizer.
"""
import math
import numpy as np


def turn_range_evidence(reference_scan, current_scan, transform, *, lidar_offset, maximum_range):
    offset = np.asarray(lidar_offset, dtype=float)
    if offset.shape != (2,) or not np.isfinite(offset).all() or not math.isfinite(maximum_range):
        return {"accepted": False, "reason": "invalid_sensor_geometry"}

    def profile(scan):
        angles = np.asarray(scan.angles_rad, dtype=float)
        ranges = np.asarray(scan.distances_m, dtype=float)
        if angles.ndim != 1 or angles.shape != ranges.shape:
            return np.empty((0,2))
        valid = np.isfinite(angles) & np.isfinite(ranges) & (ranges >= .25) & (ranges <= maximum_range)
        values = np.column_stack((angles[valid], ranges[valid]))
        if len(values)>360:
            values=values[np.linspace(0,len(values)-1,360,dtype=int)]
        return values

    yaw, x, y = float(transform.yaw_rad),float(transform.x_m),float(transform.y_m)
    if not all(math.isfinite(v) for v in (yaw,x,y)):
        return {"accepted":False,"reason":"invalid_transform"}
    rotation=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
    translation=np.array([x,y])

    def compare(before,after,reverse=False):
        reference, current=profile(before),profile(after)
        empty={"supported":0,"agreement":0.,"mean_error_m":None,
               "median_error_m":None,"p80_error_m":None,
               "supported_sectors":0,"bad_sectors":0,"overlap":0.}
        if not len(reference) or not len(current):return empty
        points=np.column_stack((reference[:,1]*np.cos(reference[:,0]),
                                reference[:,1]*np.sin(reference[:,0])))+offset
        projected=(points @ rotation.T + translation if reverse else (points-translation) @ rotation)-offset
        predicted_angles=np.arctan2(projected[:,1],projected[:,0])
        predicted_ranges=np.linalg.norm(projected,axis=1)
        gaps=np.abs((current[:,:1]-predicted_angles[None,:]+math.pi)%(2*math.pi)-math.pi)
        nearest=np.argmin(gaps,axis=1)
        supported=gaps[np.arange(len(current)),nearest]<=math.radians(2.5)
        errors=np.abs(current[:,1]-predicted_ranges[nearest])
        agrees=errors<=.10+.04*current[:,1]
        sectors=((current[:,0]+math.pi)%(2*math.pi)/(math.pi/4)).astype(int)
        ratios=[float(np.mean(agrees[supported & (sectors==i)]))
                for i in range(8) if np.count_nonzero(supported & (sectors==i))>=8]
        count=int(np.count_nonzero(supported))
        return {"supported":count,"agreement":float(np.mean(agrees[supported])) if count else 0.,
                "mean_error_m":float(np.mean(errors[supported])) if count else None,
                "median_error_m":float(np.median(errors[supported])) if count else None,
                "p80_error_m":float(np.percentile(errors[supported],80)) if count else None,
                "supported_sectors":len(ratios),"bad_sectors":sum(v<.60 for v in ratios),
                "overlap":count/len(current)}

    forward,reverse=compare(reference_scan,current_scan),compare(current_scan,reference_scan,True)
    accepted=all(e["supported"]>=60 and e["overlap"]>=.50 and e["agreement"]>=.75
                 and e["mean_error_m"] is not None and e["mean_error_m"]<=.16
                 and e["supported_sectors"]>=3 and e["bad_sectors"]<=1 for e in (forward,reverse))
    return {"accepted":accepted,"forward":forward,"reverse":reverse,"implementation":"phase1_raw_beam_v1"}
