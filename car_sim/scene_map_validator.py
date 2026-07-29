"""Compare a simulator occupancy map with the actual configured scene."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from urllib.request import urlopen

import numpy as np

from raspberry_pi.mapping.map_visualization import encode_rgb_png


def _read_pgm(path):
    with Path(path).open("rb") as stream:
        if stream.readline().strip() != b"P5":
            raise ValueError("only binary P5 PGM maps are supported")
        dimensions = stream.readline()
        while dimensions.startswith(b"#"):
            dimensions = stream.readline()
        width, height = map(int, dimensions.split())
        if int(stream.readline()) != 255:
            raise ValueError("PGM maximum value must be 255")
        image = np.frombuffer(stream.read(), dtype=np.uint8)
    return image.reshape(height, width)


def _read_map_metadata(prefix):
    text = Path(prefix).with_suffix(".yaml").read_text(encoding="utf-8")
    resolution = float(re.search(
        r"^resolution:\s*([-+0-9.eE]+)",
        text,
        re.MULTILINE,
    ).group(1))
    origin_match = re.search(
        r"^origin:\s*\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)",
        text,
        re.MULTILINE,
    )
    return resolution, (float(origin_match.group(1)), float(origin_match.group(2)))


def _sample_segment(start, end, spacing_m):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    count = max(
        2,
        int(math.ceil(float(np.linalg.norm(end - start)) / spacing_m)) + 1,
    )
    ratios = np.linspace(0.0, 1.0, count)
    return start[None, :] + ratios[:, None] * (end - start)[None, :]


def _scene_boundaries(scene, spacing_m=0.025):
    groups = []
    room = scene["room"]
    width = float(room["width"])
    height = float(room["height"])
    groups.append((
        "room",
        np.vstack((
            _sample_segment((0.0, 0.0), (width, 0.0), spacing_m),
            _sample_segment((width, 0.0), (width, height), spacing_m),
            _sample_segment((width, height), (0.0, height), spacing_m),
            _sample_segment((0.0, height), (0.0, 0.0), spacing_m),
        )),
    ))
    for obstacle in scene.get("obstacles", ()):
        label = "{}_{}".format(obstacle["type"], obstacle["id"])
        if obstacle["type"] == "rect":
            x_m = float(obstacle["x"])
            y_m = float(obstacle["y"])
            width = float(obstacle["w"])
            height = float(obstacle["h"])
            points = np.vstack((
                _sample_segment(
                    (x_m, y_m),
                    (x_m + width, y_m),
                    spacing_m,
                ),
                _sample_segment(
                    (x_m + width, y_m),
                    (x_m + width, y_m + height),
                    spacing_m,
                ),
                _sample_segment(
                    (x_m + width, y_m + height),
                    (x_m, y_m + height),
                    spacing_m,
                ),
                _sample_segment(
                    (x_m, y_m + height),
                    (x_m, y_m),
                    spacing_m,
                ),
            ))
        elif obstacle["type"] == "circle":
            radius = float(obstacle["r"])
            count = max(
                32,
                int(math.ceil(math.tau * radius / spacing_m)),
            )
            angles = np.linspace(0.0, math.tau, count, endpoint=False)
            points = np.column_stack((
                float(obstacle["x"]) + radius * np.cos(angles),
                float(obstacle["y"]) + radius * np.sin(angles),
            ))
        else:
            continue
        groups.append((label, points))
    return groups


def _to_initial_frame(points, initial_pose):
    points = np.asarray(points, dtype=np.float64)
    x_m, y_m, yaw_rad = initial_pose
    shifted = points - np.asarray((x_m, y_m))
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    inverse_rotation = np.asarray(((cosine, sine), (-sine, cosine)))
    return shifted @ inverse_rotation.T


def _nearest_distances(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    result = np.full(len(first), math.inf, dtype=np.float64)
    for start in range(0, len(first), 256):
        chunk = first[start:start + 256]
        squared = np.sum(
            (chunk[:, None, :] - second[None, :, :]) ** 2,
            axis=2,
        )
        result[start:start + len(chunk)] = np.sqrt(np.min(squared, axis=1))
    return result


def validate(prefix, scene, initial_pose=(1.0, 1.0, 0.0), tolerance_m=0.15):
    prefix = Path(prefix)
    pgm = _read_pgm(prefix.with_suffix(".pgm"))
    resolution_m, origin = _read_map_metadata(prefix)
    internal = np.flipud(pgm)
    rows, columns = np.where(internal < 65)
    occupied = np.column_stack((
        origin[0] + (columns + 0.5) * resolution_m,
        origin[1] + (rows + 0.5) * resolution_m,
    ))
    groups = [
        (label, _to_initial_frame(points, initial_pose))
        for label, points in _scene_boundaries(scene)
    ]
    expected = np.vstack(tuple(points for _label, points in groups))
    occupied_error = _nearest_distances(occupied, expected)
    expected_error = _nearest_distances(expected, occupied)

    shape_metrics = {}
    offset = 0
    for label, points in groups:
        distances = expected_error[offset:offset + len(points)]
        offset += len(points)
        shape_metrics[label] = {
            "coverage": float(np.mean(distances <= tolerance_m)),
            "median_error_m": float(np.median(distances)),
            "p90_error_m": float(np.percentile(distances, 90.0)),
        }
    report = {
        "map": str(prefix),
        "tolerance_m": float(tolerance_m),
        "occupied_cells": int(len(occupied)),
        "map_precision": float(np.mean(occupied_error <= tolerance_m)),
        "scene_coverage": float(np.mean(expected_error <= tolerance_m)),
        "occupied_median_error_m": float(np.median(occupied_error)),
        "occupied_p90_error_m": float(np.percentile(occupied_error, 90.0)),
        "shapes": shape_metrics,
    }

    image = np.repeat(pgm[:, :, None], 3, axis=2)
    bad = occupied_error > tolerance_m
    bad_rows = internal.shape[0] - 1 - rows[bad]
    image[bad_rows, columns[bad]] = (220, 30, 30)
    for point, distance in zip(expected, expected_error):
        column = int(math.floor((point[0] - origin[0]) / resolution_m))
        row = int(math.floor((point[1] - origin[1]) / resolution_m))
        image_row = internal.shape[0] - 1 - row
        if (
            0 <= image_row < image.shape[0]
            and 0 <= column < image.shape[1]
        ):
            color = (30, 180, 50) if distance <= tolerance_m else (255, 150, 0)
            image[image_row, column] = color
    overlay_path = prefix.with_name(prefix.name + "_scene_overlay.png")
    overlay_path.write_bytes(encode_rgb_png(image))
    report["overlay"] = str(overlay_path)
    report_path = prefix.with_name(prefix.name + "_scene_report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main():
    parser = argparse.ArgumentParser(
        description="compare a saved simulator map with its actual scene geometry"
    )
    parser.add_argument("output_prefix")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--start-x", type=float, default=1.0)
    parser.add_argument("--start-y", type=float, default=1.0)
    parser.add_argument("--start-yaw-deg", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=0.15)
    args = parser.parse_args()
    with urlopen(args.base_url.rstrip("/") + "/api/scene", timeout=3.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    report = validate(
        args.output_prefix,
        payload["scene"],
        (
            args.start_x,
            args.start_y,
            math.radians(args.start_yaw_deg),
        ),
        args.tolerance,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
