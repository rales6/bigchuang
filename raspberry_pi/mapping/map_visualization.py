"""Dependency-free colored map rendering for Raspberry Pi."""

from pathlib import Path
import math
import struct
import zlib

import numpy as np


TRAJECTORY_COLOR = (220, 30, 30)
START_COLOR = (20, 180, 40)
END_COLOR = (30, 90, 230)


def save_trajectory_png(grid, trajectory, output_path):
    image = build_trajectory_rgb(grid, trajectory)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_rgb_png(path, image)
    return path


def encode_trajectory_png(grid, trajectory):
    """Return the current occupancy map and trajectory as PNG bytes."""
    return encode_rgb_png(build_trajectory_rgb(grid, trajectory))


def build_trajectory_rgb(grid, trajectory):
    gray = np.flipud(grid.render_image())
    image = np.repeat(gray[:, :, None], 3, axis=2)
    points = []
    for sample in trajectory:
        if len(sample) < 3:
            continue
        cell = grid.world_to_grid(
            np.asarray(((sample[1], sample[2]),), dtype=np.float64)
        )[0]
        column, row = int(cell[0]), int(cell[1])
        if grid.in_bounds(column, row):
            points.append((column, grid.height - 1 - row))

    for start, end in zip(points, points[1:]):
        _draw_line(image, start, end, TRAJECTORY_COLOR, radius=1)

    if points:
        star_radius = max(
            5,
            int(round(0.22 / grid.resolution_m)),
        )
        _draw_star(
            image,
            points[0],
            star_radius + 2,
            START_COLOR,
        )
        _draw_star(
            image,
            points[-1],
            star_radius,
            END_COLOR,
        )
    return image


def _draw_line(image, start, end, color, radius=1):
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        _paint_disc(image, x0, y0, radius, color)
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += step_x
        if twice <= dx:
            error += dx
            y0 += step_y


def _paint_disc(image, center_x, center_y, radius, color):
    height, width = image.shape[:2]
    for y in range(center_y - radius, center_y + radius + 1):
        for x in range(center_x - radius, center_x + radius + 1):
            if (
                0 <= x < width
                and 0 <= y < height
                and (x - center_x) ** 2 + (y - center_y) ** 2
                <= radius ** 2
            ):
                image[y, x] = color


def _draw_star(image, center, outer_radius, color):
    center_x, center_y = center
    inner_radius = max(2.0, outer_radius * 0.42)
    vertices = []
    for index in range(10):
        angle = -math.pi / 2.0 + index * math.pi / 5.0
        radius = outer_radius if index % 2 == 0 else inner_radius
        vertices.append((
            center_x + radius * math.cos(angle),
            center_y + radius * math.sin(angle),
        ))
    _fill_polygon(image, vertices, color)


def _fill_polygon(image, vertices, color):
    height, width = image.shape[:2]
    min_y = max(0, int(math.floor(min(y for _x, y in vertices))))
    max_y = min(
        height - 1,
        int(math.ceil(max(y for _x, y in vertices))),
    )
    for y in range(min_y, max_y + 1):
        scan_y = y + 0.5
        intersections = []
        for index, first in enumerate(vertices):
            second = vertices[(index + 1) % len(vertices)]
            x1, y1 = first
            x2, y2 = second
            if (y1 <= scan_y < y2) or (y2 <= scan_y < y1):
                ratio = (scan_y - y1) / (y2 - y1)
                intersections.append(x1 + ratio * (x2 - x1))
        intersections.sort()
        for index in range(0, len(intersections) - 1, 2):
            start_x = max(0, int(math.ceil(intersections[index])))
            end_x = min(
                width - 1,
                int(math.floor(intersections[index + 1])),
            )
            if start_x <= end_x:
                image[y, start_x:end_x + 1] = color


def _write_rgb_png(path, image):
    path.write_bytes(encode_rgb_png(image))


def encode_rgb_png(image):
    """Encode an RGB uint8 array without requiring Pillow."""
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("PNG image must have shape (height, width, 3)")
    height, width = image.shape[:2]
    raw = b"".join(
        b"\x00" + image[row].tobytes()
        for row in range(height)
    )
    header = struct.pack(
        ">IIBBBBB",
        width,
        height,
        8,
        2,
        0,
        0,
        0,
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw, level=6))
        + _png_chunk(b"IEND", b"")
    )


def _png_chunk(kind, data):
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )
