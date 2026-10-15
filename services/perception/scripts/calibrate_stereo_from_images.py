from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def _find_corners(gray: np.ndarray, pattern_size: tuple[int, int]) -> tuple[bool, np.ndarray | None]:
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags)
    if ok:
        return True, corners.astype(np.float32)
    ok, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not ok:
        return False, None
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined.astype(np.float32)


def _object_points(pattern_size: tuple[int, int], square_mm: float) -> np.ndarray:
    cols, rows = pattern_size
    points = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points[:, :2] = grid * float(square_mm)
    return points


def _as_list(value: np.ndarray) -> list[Any]:
    return np.asarray(value, dtype=float).tolist()


def _mean_reprojection_error(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: tuple[np.ndarray, ...],
    tvecs: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    total_error = 0.0
    total_points = 0
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(img, projected, cv2.NORM_L2)
        total_error += error * error
        total_points += len(obj)
    return float(np.sqrt(total_error / max(total_points, 1)))


def _save_rectified_preview(
    left_image: np.ndarray,
    right_image: np.ndarray,
    calibration: dict[str, Any],
    output_path: Path,
) -> None:
    image_size = tuple(calibration["image_size"])
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        np.array(calibration["left"]["camera_matrix"]),
        np.array(calibration["left"]["dist_coeffs"]),
        np.array(calibration["rectification"]["R1"]),
        np.array(calibration["rectification"]["P1"]),
        image_size,
        cv2.CV_16SC2,
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        np.array(calibration["right"]["camera_matrix"]),
        np.array(calibration["right"]["dist_coeffs"]),
        np.array(calibration["rectification"]["R2"]),
        np.array(calibration["rectification"]["P2"]),
        image_size,
        cv2.CV_16SC2,
    )
    left_rect = cv2.remap(left_image, left_map1, left_map2, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_image, right_map1, right_map2, cv2.INTER_LINEAR)
    preview = np.hstack([left_rect, right_rect])
    for y in range(40, preview.shape[0], 40):
        cv2.line(preview, (0, y), (preview.shape[1], y), (0, 255, 255), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), preview)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate a left/right stereo camera pair from chessboard images.")
    parser.add_argument("--input", required=True, help="Folder containing left/ and right/ image folders.")
    parser.add_argument("--board-cols", type=int, default=9, help="Inner chessboard corners per row.")
    parser.add_argument("--board-rows", type=int, default=6, help="Inner chessboard corners per column.")
    parser.add_argument("--square-mm", type=float, required=True, help="Physical chessboard square size in millimeters.")
    parser.add_argument("--output", default="configs/stereo_calibration.yaml")
    parser.add_argument("--json-output", default="")
    parser.add_argument("--preview", default="outputs/stereo_rectified_preview.jpg")
    parser.add_argument("--min-pairs", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.0, help="stereoRectify alpha, 0 crops invalid pixels, 1 keeps all pixels.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    left_images = sorted((input_dir / "left").glob("*.jpg"))
    right_images = sorted((input_dir / "right").glob("*.jpg"))
    if not left_images or len(left_images) != len(right_images):
        raise RuntimeError("Expected the same number of .jpg files in left/ and right/.")

    pattern_size = (args.board_cols, args.board_rows)
    obj_template = _object_points(pattern_size, args.square_mm)
    object_points: list[np.ndarray] = []
    left_points: list[np.ndarray] = []
    right_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    first_valid_pair: tuple[np.ndarray, np.ndarray] | None = None

    for left_path, right_path in zip(left_images, right_images):
        left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left is None or right is None:
            print(f"[skip] unreadable pair: {left_path.name}, {right_path.name}")
            continue
        if left.shape[:2] != right.shape[:2]:
            print(f"[skip] size mismatch: {left_path.name}, {right_path.name}")
            continue
        current_size = (left.shape[1], left.shape[0])
        if image_size is None:
            image_size = current_size
        elif image_size != current_size:
            print(f"[skip] different image size: {left_path.name}, {right_path.name}")
            continue

        left_ok, left_corners = _find_corners(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), pattern_size)
        right_ok, right_corners = _find_corners(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), pattern_size)
        if not left_ok or left_corners is None or not right_ok or right_corners is None:
            print(f"[skip] chessboard not found in both images: {left_path.name}, {right_path.name}")
            continue
        object_points.append(obj_template.copy())
        left_points.append(left_corners)
        right_points.append(right_corners)
        if first_valid_pair is None:
            first_valid_pair = (left, right)
        print(f"[ok] {left_path.name}, {right_path.name}")

    if image_size is None:
        raise RuntimeError("No readable image pairs.")
    if len(object_points) < args.min_pairs:
        raise RuntimeError(f"Only {len(object_points)} valid pairs; collect at least {args.min_pairs}.")

    flags = cv2.CALIB_RATIONAL_MODEL
    left_rms, left_camera, left_dist, left_rvecs, left_tvecs = cv2.calibrateCamera(
        object_points,
        left_points,
        image_size,
        None,
        None,
        flags=flags,
    )
    right_rms, right_camera, right_dist, right_rvecs, right_tvecs = cv2.calibrateCamera(
        object_points,
        right_points,
        image_size,
        None,
        None,
        flags=flags,
    )
    stereo_flags = cv2.CALIB_FIX_INTRINSIC
    stereo_rms, left_camera, left_dist, right_camera, right_dist, R, T, E, F = cv2.stereoCalibrate(
        object_points,
        left_points,
        right_points,
        left_camera,
        left_dist,
        right_camera,
        right_dist,
        image_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=stereo_flags,
    )
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        left_camera,
        left_dist,
        right_camera,
        right_dist,
        image_size,
        R,
        T,
        alpha=float(args.alpha),
    )

    calibration = {
        "version": 1,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "board": {
            "inner_corners": [args.board_cols, args.board_rows],
            "square_mm": float(args.square_mm),
            "valid_pairs": len(object_points),
        },
        "quality": {
            "left_rms": float(left_rms),
            "right_rms": float(right_rms),
            "stereo_rms": float(stereo_rms),
            "left_mean_reprojection_px": _mean_reprojection_error(object_points, left_points, left_rvecs, left_tvecs, left_camera, left_dist),
            "right_mean_reprojection_px": _mean_reprojection_error(object_points, right_points, right_rvecs, right_tvecs, right_camera, right_dist),
        },
        "left": {
            "camera_matrix": _as_list(left_camera),
            "dist_coeffs": _as_list(left_dist.reshape(-1)),
        },
        "right": {
            "camera_matrix": _as_list(right_camera),
            "dist_coeffs": _as_list(right_dist.reshape(-1)),
        },
        "stereo": {
            "R": _as_list(R),
            "T_mm": _as_list(T.reshape(-1)),
            "E": _as_list(E),
            "F": _as_list(F),
            "baseline_mm_norm": float(np.linalg.norm(T)),
            "baseline_mm_x": float(abs(T.reshape(-1)[0])),
        },
        "rectification": {
            "R1": _as_list(R1),
            "R2": _as_list(R2),
            "P1": _as_list(P1),
            "P2": _as_list(P2),
            "Q": _as_list(Q),
            "roi1": [int(v) for v in roi1],
            "roi2": [int(v) for v in roi2],
        },
        "distance_formula": {
            "Z_mm": "Q based reprojection is preferred after rectification; rough formula: Z = fx_px * baseline_mm / disparity_px",
            "fx_px_rectified": float(P1[0, 0]),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(calibration, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    if first_valid_pair is not None:
        _save_rectified_preview(first_valid_pair[0], first_valid_pair[1], calibration, Path(args.preview))

    print(f"[done] valid_pairs={len(object_points)} stereo_rms={stereo_rms:.4f}")
    print(f"[done] baseline_norm={calibration['stereo']['baseline_mm_norm']:.2f} mm")
    print(f"[done] wrote {output}")
    if args.preview:
        print(f"[done] wrote rectified preview {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
