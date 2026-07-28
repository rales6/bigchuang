from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def _open_camera(source: str, backend: str, width: int, height: int, fps: int, fourcc: str) -> cv2.VideoCapture:
    api = cv2.CAP_V4L2 if backend.lower() == "v4l2" else cv2.CAP_ANY
    camera_id: int | str
    try:
        camera_id = int(source)
    except ValueError:
        camera_id = source
    cap = cv2.VideoCapture(camera_id, api)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera: {source}")
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


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


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _corner_motion(corners: np.ndarray | None, last_corners: np.ndarray | None) -> float:
    if corners is None or last_corners is None or len(corners) != len(last_corners):
        return 999.0
    return float(np.mean(np.linalg.norm(corners.reshape(-1, 2) - last_corners.reshape(-1, 2), axis=1)))


def _draw_status(
    frame: np.ndarray,
    label: str,
    ok: bool,
    corners: np.ndarray | None,
    pattern_size: tuple[int, int],
    sharpness: float,
) -> np.ndarray:
    out = frame.copy()
    if corners is not None:
        cv2.drawChessboardCorners(out, pattern_size, corners, ok)
    color = (40, 220, 40) if ok else (40, 40, 255)
    cv2.putText(out, f"{label} {'OK' if ok else 'NO'} sharp={sharpness:.0f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture synchronized left/right chessboard samples for stereo calibration.")
    parser.add_argument("--left-camera", required=True, help="Robot left camera index, e.g. 0 or /dev/video0.")
    parser.add_argument("--right-camera", required=True, help="Robot right camera index, e.g. 2 or /dev/video2.")
    parser.add_argument("--swap-cameras", action="store_true", help="Swap left/right after opening cameras.")
    parser.add_argument("--backend", default="v4l2")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--board-cols", type=int, default=9, help="Inner chessboard corners per row.")
    parser.add_argument("--board-rows", type=int, default=6, help="Inner chessboard corners per column.")
    parser.add_argument("--output", default="outputs/stereo_calibration_samples")
    parser.add_argument("--count", type=int, default=35)
    parser.add_argument("--min-interval", type=float, default=0.8)
    parser.add_argument("--min-motion-px", type=float, default=12.0)
    parser.add_argument("--min-sharpness", type=float, default=60.0)
    parser.add_argument("--manual", action="store_true", help="Press S to save instead of automatic capture.")
    parser.add_argument("--headless", action="store_true", help="Do not open a preview window.")
    args = parser.parse_args()

    output = Path(args.output)
    left_dir = output / "left"
    right_dir = output / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    left_cap = _open_camera(args.left_camera, args.backend, args.width, args.height, args.fps, args.fourcc)
    right_cap = _open_camera(args.right_camera, args.backend, args.width, args.height, args.fps, args.fourcc)
    if args.swap_cameras:
        left_cap, right_cap = right_cap, left_cap

    pattern_size = (args.board_cols, args.board_rows)
    saved = len(list(left_dir.glob("left_*.jpg")))
    last_saved_at = 0.0
    last_left_corners: np.ndarray | None = None
    print("[Stereo calibration] Move the chessboard to different positions, angles, and distances.")
    print("[Stereo calibration] Auto mode saves only when both cameras see the board and it moved enough.")

    try:
        while saved < args.count:
            ok_l, left = left_cap.read()
            ok_r, right = right_cap.read()
            if not ok_l or left is None or not ok_r or right is None:
                print("[Stereo calibration] camera read failed; retrying")
                time.sleep(0.05)
                continue

            left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            left_ok, left_corners = _find_corners(left_gray, pattern_size)
            right_ok, right_corners = _find_corners(right_gray, pattern_size)
            left_sharp = _sharpness(left_gray)
            right_sharp = _sharpness(right_gray)
            moved = _corner_motion(left_corners, last_left_corners)
            now = time.monotonic()

            should_save = (
                left_ok
                and right_ok
                and left_sharp >= args.min_sharpness
                and right_sharp >= args.min_sharpness
                and now - last_saved_at >= args.min_interval
                and moved >= args.min_motion_px
            )
            if args.manual:
                should_save = False

            preview_l = _draw_status(left, "LEFT", left_ok, left_corners, pattern_size, left_sharp)
            preview_r = _draw_status(right, "RIGHT", right_ok, right_corners, pattern_size, right_sharp)
            preview = np.hstack([preview_l, preview_r])
            cv2.putText(preview, f"saved={saved}/{args.count} moved={moved:.1f}px S=save Q=quit", (12, preview.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            key = -1
            if not args.headless:
                cv2.imshow("stereo calibration capture", preview)
                key = cv2.waitKey(1) & 0xFF
            if args.manual and key in {ord("s"), ord("S")}:
                should_save = left_ok and right_ok
            if key in {ord("q"), ord("Q"), 27}:
                break

            if should_save:
                saved += 1
                left_path = left_dir / f"left_{saved:03d}.jpg"
                right_path = right_dir / f"right_{saved:03d}.jpg"
                cv2.imwrite(str(left_path), left)
                cv2.imwrite(str(right_path), right)
                last_saved_at = now
                last_left_corners = None if left_corners is None else left_corners.copy()
                print(f"[Stereo calibration] saved pair {saved:03d}: {left_path.name}, {right_path.name}")
            elif left_ok and right_ok and last_left_corners is None:
                last_left_corners = None if left_corners is None else left_corners.copy()
    finally:
        left_cap.release()
        right_cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"[Stereo calibration] done, saved {saved} pairs in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
