from __future__ import annotations

import cv2
import numpy as np

from qwen_grounded_tracker.domain import (
    LidarObservation,
    ObstacleObservation,
    SafetyDecision,
    TrackObservation,
)


def _put_lines(frame: np.ndarray, lines: list[str]) -> None:
    y = 24
    for line in lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 21


def draw_overlay(
    frame: np.ndarray,
    track: TrackObservation,
    obstacles: ObstacleObservation,
    lidar: LidarObservation,
    safety: SafetyDecision,
    grounding_status: str,
    boundary_mode: str,
    emergency_stop: bool,
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]

    for detection in obstacles.detections:
        box = detection.bbox
        color = (0, 0, 255) if detection.in_danger_zone else (255, 128, 0)
        cv2.rectangle(
            output,
            (int(box.x1), int(box.y1)),
            (int(box.x2), int(box.y2)),
            color,
            2,
        )
        cv2.putText(
            output,
            f"OBS {detection.label} {detection.confidence:.2f}",
            (int(box.x1), max(18, int(box.y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    if track.visible and track.bbox is not None:
        box = track.bbox
        cv2.rectangle(
            output,
            (int(box.x1), int(box.y1)),
            (int(box.x2), int(box.y2)),
            (0, 255, 255),
            2,
        )
        if track.contour is not None:
            cv2.drawContours(output, [track.contour], -1, (0, 255, 0), 2)
        cx, cy = box.center
        cv2.drawMarker(
            output,
            (int(cx), int(cy)),
            (0, 255, 255),
            cv2.MARKER_CROSS,
            18,
            2,
        )
        cv2.putText(
            output,
            f"{track.logical_target_id or 'target'} identity={track.identity_score:.2f}",
            (int(box.x1), min(height - 10, int(box.y2) + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    center_x = width // 2
    cv2.line(output, (center_x, 0), (center_x, height), (80, 80, 80), 1)

    direction = safety.guidance.direction
    if direction == "TURN_LEFT":
        cv2.arrowedLine(output, (center_x, height - 35), (center_x - 90, height - 35), (0, 255, 255), 4)
    elif direction == "TURN_RIGHT":
        cv2.arrowedLine(output, (center_x, height - 35), (center_x + 90, height - 35), (0, 255, 255), 4)
    elif direction == "FORWARD":
        cv2.arrowedLine(output, (center_x, height - 15), (center_x, height - 95), (0, 255, 0), 4)
    elif direction == "BACKWARD":
        cv2.arrowedLine(output, (center_x, height - 95), (center_x, height - 15), (0, 165, 255), 4)
    else:
        cv2.putText(output, "STOP", (center_x - 55, height - 25), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

    _put_lines(
        output,
        [
            f"Grounding: {grounding_status}",
            f"Tracking: {track.status}",
            f"Boundary: {boundary_mode}",
            f"YOLO obstacles: {obstacles.status}",
            f"LiDAR: {lidar.status}",
            f"Guidance: {direction} lin={safety.guidance.linear:.3f} ang={safety.guidance.angular:.3f}",
            f"Safety blocked: {safety.blocked} | E-stop: {emergency_stop}",
            "Keys: G reground | I instruction | M manual ROI | R reset | C contour | SPACE e-stop | S save | Q quit",
        ],
    )
    return output
