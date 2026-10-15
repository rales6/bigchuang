"""Offline hypotheses for stationary field lidar; never controls hardware."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from raspberry_pi.known_map_navigation import KnownOccupancyMap, ScanMapLocalizer
from raspberry_pi.lidar import LaserScan
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import IcpScanMatcher
from raspberry_pi.joint_mapping import atomic_json, angle_distance


def result_dict(result):
    if result is None:
        return None
    return {"x_m": result.pose.x_m, "y_m": result.pose.y_m,
            "yaw_deg": math.degrees(result.pose.yaw_rad),
            "rmse_m": result.rmse_m, "inlier_ratio": result.inlier_ratio,
            "score": result.score,
            "ambiguity_margin": result.ambiguity_margin if math.isfinite(result.ambiguity_margin) else None}


def down_hypothesis(localizer, scan):
    points = localizer.points(scan, maximum=220)
    profile = localizer.range_profile(scan)
    candidates = [p for p in localizer._global_lattice(position_step_m=.14, yaw_step_deg=10)
                  if angle_distance(p.yaw_rad, -math.pi / 2) <= math.radians(10.01)]
    coarse = localizer._rank(points, candidates, keep=8, range_profile=profile)
    fine = []
    for _, pose, _ in coarse:
        fine.extend(localizer._pose_lattice(pose, translation_radius_m=.12,
                    translation_step_m=.02, yaw_radius_deg=4, yaw_step_deg=2))
    fine = [p for p in fine if angle_distance(p.yaw_rad, -math.pi / 2) <= math.radians(10.01)]
    return localizer._result_from_ranked(localizer._rank(points, fine, keep=12, range_profile=profile))


def render(known, localizer, scan, hypotheses, destination):
    occupied = np.argwhere(known.occupied)
    lo = occupied.min(axis=0) - 12
    hi = occupied.max(axis=0) + 12
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, (known.height - 1, known.width - 1))
    scale = 4
    width = int(hi[1] - lo[1] + 1) * scale
    height = int(hi[0] - lo[0] + 1) * scale
    canvas = Image.new("RGB", (width * len(hypotheses), height + 70), "white")
    def pixel(x, y):
        col = (x - known.origin_x_m) / known.resolution_m
        row = (y - known.origin_y_m) / known.resolution_m
        return ((col - lo[1]) * scale, (hi[0] - row) * scale)
    for index, (title, result) in enumerate(hypotheses):
        base = Image.fromarray(np.flipud(known.image[lo[0]:hi[0]+1, lo[1]:hi[1]+1])).convert("RGB")
        base = base.resize((width, height), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(base)
        if result is not None:
            for x, y in result.pose.transform_points(localizer.points(scan, maximum=360)):
                px, py = pixel(x, y)
                draw.ellipse((px-1, py-1, px+1, py+1), fill=(220, 50, 50))
            px, py = pixel(result.pose.x_m, result.pose.y_m)
            draw.ellipse((px-5, py-5, px+5, py+5), fill="blue")
            draw.line((px, py, px+28*math.cos(result.pose.yaw_rad), py-28*math.sin(result.pose.yaw_rad)), fill="blue", width=4)
            title += "\nyaw={:.1f} deg RMSE={:.3f}m inliers={:.1%}".format(math.degrees(result.pose.yaw_rad), result.rmse_m, result.inlier_ratio)
        canvas.paste(base, (index * width, 70))
        ImageDraw.Draw(canvas).text((index * width + 10, 10), title, fill="black")
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    directory = Path(args.directory)
    with gzip.open(directory / "scans.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [row for line in stream if (row := json.loads(line))["type"] == "scan"]
    scans = [LaserScan(np.array(r["angles_rad"], dtype=float), np.array(r["distances_m"], dtype=float), r["timestamp_s"]) for r in rows]
    known = KnownOccupancyMap.load(directory / "ble_scene_17.pgm")
    localizer = ScanMapLocalizer(known)
    matcher = IcpScanMatcher(max_correspondence_m=.3, min_points=35)
    stability = []
    reference = scans[0].points()
    for scan in scans[1::5]:
        match = matcher.match(reference, scan.points())
        stability.append({"translation_m": math.hypot(match.transform.x_m, match.transform.y_m),
                          "yaw_deg": math.degrees(match.transform.yaw_rad), "rmse_m": match.rmse_m})
    results = []
    for index in (0, len(scans)//2, len(scans)-1):
        scan = scans[index]
        free = localizer.global_localize(scan)
        down = down_hypothesis(localizer, scan)
        record = {"scan_index": index, "global": result_dict(free), "down_heading_hypothesis": result_dict(down)}
        results.append(record)
        print(json.dumps(record), flush=True)
        if index == len(scans)//2:
            render(known, localizer, scan, [("Unconstrained map match", free), ("Heading constrained to image DOWN +/-10 deg", down)], directory / "hypotheses.png")
    payload = {"scans": len(scans), "duration_s": scans[-1].timestamp_s-scans[0].timestamp_s,
               "map_sha256": hashlib.sha256((directory / "ble_scene_17.pgm").read_bytes()).hexdigest(),
               "stationary_scan_to_first": stability, "map_matches": results,
               "note": "Down heading is a user-description hypothesis, not measured ground truth. No robot motion commands."}
    atomic_json(directory / "analysis.json", payload)
    print("OUTPUT=" + str(directory / "analysis.json"), flush=True)


if __name__ == "__main__":
    main()
