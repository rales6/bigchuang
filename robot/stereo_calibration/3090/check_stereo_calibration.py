from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid calibration file: {path}")
    return data


def _rectify_pair(left: np.ndarray, right: np.ndarray, calibration: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    image_size = tuple(calibration["image_size"])
    if (left.shape[1], left.shape[0]) != image_size or (right.shape[1], right.shape[0]) != image_size:
        raise RuntimeError(
            f"Image size must match calibration {image_size}; got left={left.shape[1]}x{left.shape[0]}, right={right.shape[1]}x{right.shape[0]}"
        )
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        np.array(calibration["left"]["camera_matrix"], dtype=np.float64),
        np.array(calibration["left"]["dist_coeffs"], dtype=np.float64),
        np.array(calibration["rectification"]["R1"], dtype=np.float64),
        np.array(calibration["rectification"]["P1"], dtype=np.float64),
        image_size,
        cv2.CV_16SC2,
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        np.array(calibration["right"]["camera_matrix"], dtype=np.float64),
        np.array(calibration["right"]["dist_coeffs"], dtype=np.float64),
        np.array(calibration["rectification"]["R2"], dtype=np.float64),
        np.array(calibration["rectification"]["P2"], dtype=np.float64),
        image_size,
        cv2.CV_16SC2,
    )
    return (
        cv2.remap(left, left_map1, left_map2, cv2.INTER_LINEAR),
        cv2.remap(right, right_map1, right_map2, cv2.INTER_LINEAR),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check stereo calibration by rectifying one left/right image pair.")
    parser.add_argument("--calibration", default="configs/stereo_calibration.yaml")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", default="outputs/stereo_check_rectified.jpg")
    parser.add_argument("--line-step", type=int, default=40)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    calibration = _load_yaml(Path(args.calibration))
    left = cv2.imread(args.left, cv2.IMREAD_COLOR)
    right = cv2.imread(args.right, cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise RuntimeError("Unable to read left/right images.")

    left_rect, right_rect = _rectify_pair(left, right, calibration)
    preview = np.hstack([left_rect, right_rect])
    for y in range(args.line_step, preview.shape[0], args.line_step):
        cv2.line(preview, (0, y), (preview.shape[1], y), (0, 255, 255), 1)
    cv2.putText(preview, "Rectified stereo check: matching points should lie on the same horizontal line", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), preview)
    print(f"[done] wrote {output}")
    print(f"[info] stereo_rms={calibration.get('quality', {}).get('stereo_rms')} baseline={calibration.get('stereo', {}).get('baseline_mm_norm')} mm")
    if args.show:
        cv2.imshow("stereo calibration check", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
