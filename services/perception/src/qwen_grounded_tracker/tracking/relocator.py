from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from qwen_grounded_tracker.domain import BBox


@dataclass
class RelocationResult:
    bbox: BBox
    score: float
    method: str


class ReferenceRelocator:
    """Relocate a Qwen-grounded snapshot box in the latest camera frame.

    Qwen CPU grounding can take many seconds, so the returned box belongs to an
    older snapshot. This helper first tries ORB feature matching, then multi-scale
    template matching, and finally falls back to the same normalized box.
    """

    def __init__(self, template_match_threshold: float = 0.52) -> None:
        self.template_match_threshold = template_match_threshold

    @staticmethod
    def _orb(reference: np.ndarray, frame: np.ndarray) -> RelocationResult | None:
        if min(reference.shape[:2]) < 20:
            return None

        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=800, fastThreshold=8)
        ref_kp, ref_desc = orb.detectAndCompute(ref_gray, None)
        frame_kp, frame_desc = orb.detectAndCompute(frame_gray, None)
        if ref_desc is None or frame_desc is None or len(ref_kp) < 8:
            return None

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(ref_desc, frame_desc, k=2)
        good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            return None

        src = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([frame_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if homography is None or inliers is None:
            return None

        h, w = reference.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        x1, y1 = projected.min(axis=0)
        x2, y2 = projected.max(axis=0)
        frame_h, frame_w = frame.shape[:2]
        bbox = BBox(float(x1), float(y1), float(x2), float(y2)).clamp(frame_w, frame_h)
        if bbox.width < 5 or bbox.height < 5:
            return None

        score = float(inliers.sum()) / max(len(inliers), 1)
        if score < 0.35:
            return None
        return RelocationResult(bbox, score, "orb_homography")

    def _template(self, reference: np.ndarray, frame: np.ndarray) -> RelocationResult | None:
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = frame_gray.shape[:2]

        best_score = -1.0
        best_bbox: BBox | None = None
        for scale in (0.70, 0.82, 0.92, 1.0, 1.08, 1.18, 1.32):
            width = int(round(ref_gray.shape[1] * scale))
            height = int(round(ref_gray.shape[0] * scale))
            if width < 8 or height < 8 or width >= frame_w or height >= frame_h:
                continue

            resized = cv2.resize(ref_gray, (width, height), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(frame_gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_value, _, max_location = cv2.minMaxLoc(result)
            if max_value > best_score:
                x, y = max_location
                best_score = float(max_value)
                best_bbox = BBox(x, y, x + width, y + height)

        if best_bbox is None or best_score < self.template_match_threshold:
            return None
        return RelocationResult(best_bbox, best_score, "template_match")

    def relocate(
        self,
        snapshot: np.ndarray,
        snapshot_bbox: BBox,
        current_frame: np.ndarray,
    ) -> RelocationResult:
        reference = snapshot_bbox.crop(snapshot)
        if reference.size > 0:
            orb_result = self._orb(reference, current_frame)
            if orb_result is not None:
                return orb_result

            template_result = self._template(reference, current_frame)
            if template_result is not None:
                return template_result

        snapshot_h, snapshot_w = snapshot.shape[:2]
        current_h, current_w = current_frame.shape[:2]
        normalized = BBox(
            snapshot_bbox.x1 / max(snapshot_w, 1),
            snapshot_bbox.y1 / max(snapshot_h, 1),
            snapshot_bbox.x2 / max(snapshot_w, 1),
            snapshot_bbox.y2 / max(snapshot_h, 1),
        )
        fallback = BBox(
            normalized.x1 * current_w,
            normalized.y1 * current_h,
            normalized.x2 * current_w,
            normalized.y2 * current_h,
        ).clamp(current_w, current_h)
        return RelocationResult(fallback, 0.0, "normalized_snapshot_fallback")
