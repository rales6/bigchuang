from __future__ import annotations

import cv2
import numpy as np

from qwen_grounded_tracker.domain import BBox


class ContourRefiner:
    """Approximate a target boundary inside the tracked box with GrabCut.

    This is a lightweight Windows validation path, not a substitute for SAM 2.
    """

    def __init__(
        self,
        mode: str = "grabcut",
        update_interval_frames: int = 4,
        grabcut_iterations: int = 2,
    ) -> None:
        self.mode = mode
        self.update_interval_frames = max(1, update_interval_frames)
        self.grabcut_iterations = max(1, grabcut_iterations)
        self.frame_index = 0
        self.last_contour: np.ndarray | None = None
        self.last_bbox: BBox | None = None

    @staticmethod
    def _box_contour(bbox: BBox) -> np.ndarray:
        return np.array(
            [
                [int(bbox.x1), int(bbox.y1)],
                [int(bbox.x2), int(bbox.y1)],
                [int(bbox.x2), int(bbox.y2)],
                [int(bbox.x1), int(bbox.y2)],
            ],
            dtype=np.int32,
        ).reshape(-1, 1, 2)

    @staticmethod
    def _remap_contour(contour: np.ndarray, old_box: BBox, new_box: BBox) -> np.ndarray:
        points = contour.reshape(-1, 2).astype(np.float32)
        nx = (points[:, 0] - old_box.x1) / max(old_box.width, 1.0)
        ny = (points[:, 1] - old_box.y1) / max(old_box.height, 1.0)
        points[:, 0] = new_box.x1 + nx * new_box.width
        points[:, 1] = new_box.y1 + ny * new_box.height
        return np.rint(points).astype(np.int32).reshape(-1, 1, 2)

    def refine(self, frame: np.ndarray, bbox: BBox) -> np.ndarray:
        self.frame_index += 1
        if self.mode != "grabcut":
            self.last_contour = self._box_contour(bbox)
            self.last_bbox = bbox
            return self.last_contour

        if (
            self.last_contour is not None
            and self.last_bbox is not None
            and self.frame_index % self.update_interval_frames != 0
        ):
            self.last_contour = self._remap_contour(self.last_contour, self.last_bbox, bbox)
            self.last_bbox = bbox
            return self.last_contour

        height, width = frame.shape[:2]
        safe = bbox.clamp(width, height)
        x, y, w, h = safe.to_xywh()
        if w < 6 or h < 6:
            self.last_contour = self._box_contour(safe)
            self.last_bbox = safe
            return self.last_contour

        margin_x = max(1, int(w * 0.03))
        margin_y = max(1, int(h * 0.03))
        rect = (
            min(x + margin_x, width - 2),
            min(y + margin_y, height - 2),
            max(2, min(w - 2 * margin_x, width - x - margin_x)),
            max(2, min(h - 2 * margin_y, height - y - margin_y)),
        )

        mask = np.zeros((height, width), dtype=np.uint8)
        bg_model = np.zeros((1, 65), dtype=np.float64)
        fg_model = np.zeros((1, 65), dtype=np.float64)

        try:
            cv2.grabCut(
                frame,
                mask,
                rect,
                bg_model,
                fg_model,
                self.grabcut_iterations,
                cv2.GC_INIT_WITH_RECT,
            )
            foreground = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                255,
                0,
            ).astype(np.uint8)
            foreground[:y, :] = 0
            foreground[y + h :, :] = 0
            foreground[:, :x] = 0
            foreground[:, x + w :] = 0

            contours, _ = cv2.findContours(
                foreground,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                raise ValueError("GrabCut produced no contour")

            center_x, center_y = safe.center
            center_point = (float(center_x), float(center_y))
            containing = [
                contour
                for contour in contours
                if cv2.pointPolygonTest(contour, center_point, False) >= 0
            ]
            candidates = containing or contours
            contour = max(candidates, key=cv2.contourArea)
            if cv2.contourArea(contour) < safe.area * 0.05:
                raise ValueError("GrabCut contour is too small")
            self.last_contour = contour
        except Exception:
            self.last_contour = self._box_contour(safe)

        self.last_bbox = safe
        return self.last_contour

    def toggle_mode(self) -> str:
        self.mode = "box" if self.mode == "grabcut" else "grabcut"
        self.last_contour = None
        self.last_bbox = None
        return self.mode

    def reset(self) -> None:
        self.frame_index = 0
        self.last_contour = None
        self.last_bbox = None
