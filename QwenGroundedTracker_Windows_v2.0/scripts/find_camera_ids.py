from __future__ import annotations

import cv2


for camera_id in range(10):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    try:
        if not cap.isOpened():
            print(f"camera {camera_id}: unavailable")
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            height, width = frame.shape[:2]
            print(f"camera {camera_id}: available, frame={width}x{height}")
        else:
            print(f"camera {camera_id}: opened but no frame")
    finally:
        cap.release()
