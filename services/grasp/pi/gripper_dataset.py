"""Pi-side gripper pose collection and YOLO-Pose dataset export."""
from __future__ import annotations

import io
import json
from pathlib import Path
import random
import time
import uuid
import zipfile

import cv2
import numpy as np


class GripperDataset:
    """Store a box and two ordered jaw contact points per gripper image."""

    def __init__(self, root: str | Path, target_per_camera: int | None = None) -> None:
        self.root = Path(root).expanduser()
        self.target_per_camera = None if target_per_camera is None else int(target_per_camera)
        self.samples = self.root / "samples"
        self.samples.mkdir(parents=True, exist_ok=True)
        self._previous: dict[str, np.ndarray] = {}

    def _meta_path(self, sample_id: str) -> Path:
        return self.samples / f"{Path(sample_id).name}.json"

    def _image_path(self, sample_id: str) -> Path:
        return self.samples / f"{Path(sample_id).name}.jpg"

    @staticmethod
    def _candidate_boxes(frame: np.ndarray, previous: np.ndarray | None) -> list[dict]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dark = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY_INV)[1]
        mask = dark
        if previous is not None and previous.shape == frame.shape:
            prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
            motion = cv2.threshold(cv2.absdiff(gray, prev_gray), 16, 255, cv2.THRESH_BINARY)[1]
            motion = cv2.dilate(motion, np.ones((11, 11), np.uint8), iterations=2)
            overlap = cv2.bitwise_and(dark, motion)
            if cv2.countNonZero(overlap) > 80:
                mask = overlap
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        items = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            x, y, bw, bh = cv2.boundingRect(contour)
            ratio = area / max(float(w * h), 1.0)
            if area < 100 or ratio > 0.16 or bw < 8 or bh < 8:
                continue
            if bw > w * 0.55 or bh > h * 0.60:
                continue
            items.append({
                "x1": int(x), "y1": int(y), "x2": int(x + bw), "y2": int(y + bh),
                "width": int(bw), "height": int(bh), "score": round(area, 1),
            })
        items.sort(key=lambda box: box["score"], reverse=True)
        return items[:8]

    def _write_meta(self, meta: dict) -> None:
        self._meta_path(meta["sample_id"]).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def capture(self, camera: str, jpeg: bytes) -> dict:
        if camera not in ("left", "right"):
            raise ValueError("camera must be left or right")
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("invalid JPEG")
        sample_id = (
            f"{camera}_{time.strftime('%Y%m%d_%H%M%S')}_"
            f"{time.time_ns() % 1_000_000:06d}_{uuid.uuid4().hex[:8]}"
        )
        candidates = self._candidate_boxes(frame, self._previous.get(camera))
        self._previous[camera] = frame.copy()
        self._image_path(sample_id).write_bytes(jpeg)
        meta = {
            "sample_id": sample_id, "camera": camera, "width": int(frame.shape[1]),
            "height": int(frame.shape[0]), "created_at": time.time(),
            "confirmed": False, "negative": False, "bbox": None,
            "keypoints": None, "qwen_bbox": None, "qwen_keypoints": None,
            "candidates": candidates,
        }
        self._write_meta(meta)
        return meta

    def get(self, sample_id: str) -> dict:
        path = self._meta_path(sample_id)
        if not path.exists():
            raise FileNotFoundError(sample_id)
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta.setdefault("keypoints", None)
        meta.setdefault("qwen_keypoints", None)
        return meta

    @staticmethod
    def _normalise_keypoints(keypoints, width: int, height: int) -> list[dict]:
        if not isinstance(keypoints, list):
            keypoints = []
        result: list[dict] = []
        for item in keypoints[:2]:
            if isinstance(item, dict):
                x, y, v = item.get("x", 0), item.get("y", 0), item.get("v", 2)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                x, y = item[0], item[1]
                v = item[2] if len(item) > 2 else 2
            else:
                x = y = v = 0
            v = int(v or 0)
            if v <= 0:
                result.append({"x": 0.0, "y": 0.0, "v": 0})
            else:
                result.append({
                    "x": max(0.0, min(float(x), width - 1.0)),
                    "y": max(0.0, min(float(y), height - 1.0)),
                    "v": 2 if v >= 2 else 1,
                })
        while len(result) < 2:
            result.append({"x": 0.0, "y": 0.0, "v": 0})
        if sum(point["v"] > 0 for point in result) == 2:
            result.sort(key=lambda point: point["x"])
        return result

    def save(self, sample_id: str, bbox: dict | None, negative: bool = False,
             keypoints: list | None = None) -> dict:
        meta = self.get(sample_id)
        if not negative:
            if not isinstance(bbox, dict):
                raise ValueError("bbox required")
            x1, y1, x2, y2 = [float(bbox.get(k, 0)) for k in ("x1", "y1", "x2", "y2")]
            x1 = max(0.0, min(x1, meta["width"] - 1.0))
            y1 = max(0.0, min(y1, meta["height"] - 1.0))
            x2 = max(x1 + 2.0, min(x2, float(meta["width"])))
            y2 = max(y1 + 2.0, min(y2, float(meta["height"])))
            points = self._normalise_keypoints(keypoints, meta["width"], meta["height"])
            if not any(point["v"] > 0 for point in points):
                raise ValueError("at least one visible jaw contact point is required")
            meta["bbox"] = {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": x2 - x1, "height": y2 - y1,
            }
            meta["keypoints"] = points
        else:
            meta["bbox"] = None
            meta["keypoints"] = None
        meta["negative"] = bool(negative)
        meta["confirmed"] = True
        meta["confirmed_at"] = time.time()
        self._write_meta(meta)
        return meta

    def set_qwen(self, sample_id: str, bbox: dict | None, raw: str = "",
                 keypoints: list | None = None) -> dict:
        meta = self.get(sample_id)
        meta["qwen_bbox"] = bbox
        if keypoints is not None:
            meta["qwen_keypoints"] = self._normalise_keypoints(
                keypoints, meta["width"], meta["height"],
            )
        meta["qwen_raw"] = raw[:4000]
        self._write_meta(meta)
        return meta

    def delete(self, sample_id: str) -> None:
        self._meta_path(sample_id).unlink(missing_ok=True)
        self._image_path(sample_id).unlink(missing_ok=True)

    @staticmethod
    def _visible_count(meta: dict) -> int:
        return sum(
            1 for point in (meta.get("keypoints") or [])
            if isinstance(point, dict) and int(point.get("v", 0)) > 0
        )

    def status(self) -> dict:
        counts = {
            "left": 0, "right": 0, "negative": 0, "pending": 0,
            "one_point": 0, "two_points": 0, "legacy_bbox": 0,
        }
        for path in self.samples.glob("*.json"):
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not meta.get("confirmed"):
                counts["pending"] += 1
            elif meta.get("negative"):
                counts["negative"] += 1
            else:
                visible = self._visible_count(meta)
                if visible == 0:
                    counts["legacy_bbox"] += 1
                    continue
                counts[meta.get("camera", "left")] += 1
                counts["two_points" if visible >= 2 else "one_point"] += 1
        counts["target_per_camera"] = None
        counts["unlimited"] = True
        counts["ready"] = counts["left"] + counts["right"] >= 10 and counts["two_points"] >= 4
        return counts

    def list_samples(self, pose_pending_only: bool = False) -> list[dict]:
        """List saved images so legacy bbox samples can be annotated in-place."""
        records: list[dict] = []
        for path in self.samples.glob("*.json"):
            try:
                meta = self.get(path.stem)
            except Exception:
                continue
            if pose_pending_only and (
                meta.get("negative")
                or not meta.get("bbox")
                or self._visible_count(meta) > 0
            ):
                continue
            meta["visible_points"] = self._visible_count(meta)
            records.append(meta)
        records.sort(key=lambda item: (item.get("created_at", 0), item["sample_id"]))
        return records
    def build_zip(self) -> tuple[bytes, dict]:
        records = []
        for path in self.samples.glob("*.json"):
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if meta.get("confirmed") and (
                meta.get("negative") or self._visible_count(meta) > 0
            ):
                records.append(meta)
        positive_count = sum(not record.get("negative") for record in records)
        two_point_count = sum(self._visible_count(record) >= 2 for record in records)
        if len(records) < 10 or positive_count < 6:
            raise ValueError(
                "at least 10 usable samples including 6 positive pose samples are required"
            )
        if two_point_count < 4:
            raise ValueError("at least 4 samples with both jaw points visible are required")
        random.Random(42).shuffle(records)
        splits = {}
        for index, meta in enumerate(records):
            ratio = index / max(len(records), 1)
            splits[meta["sample_id"]] = (
                "train" if ratio < 0.8 else ("val" if ratio < 0.9 else "test")
            )
        output = io.BytesIO()
        yaml = (
            "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
            "kpt_shape: [2, 3]\nflip_idx: [1, 0]\nnames:\n  0: gripper_jaws\n"
        )
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("data.yaml", yaml)
            for meta in records:
                sid = meta["sample_id"]
                split = splits[sid]
                archive.write(self._image_path(sid), f"images/{split}/{sid}.jpg")
                label = ""
                box = meta.get("bbox")
                if box and not meta.get("negative"):
                    width, height = float(meta["width"]), float(meta["height"])
                    values = [
                        "0",
                        f"{(float(box['x1']) + float(box['x2'])) * 0.5 / width:.6f}",
                        f"{(float(box['y1']) + float(box['y2'])) * 0.5 / height:.6f}",
                        f"{(float(box['x2']) - float(box['x1'])) / width:.6f}",
                        f"{(float(box['y2']) - float(box['y1'])) / height:.6f}",
                    ]
                    points = self._normalise_keypoints(
                        meta.get("keypoints"), int(width), int(height),
                    )
                    for point in points:
                        if point["v"] > 0:
                            values.extend((
                                f"{point['x'] / width:.6f}",
                                f"{point['y'] / height:.6f}",
                                str(point["v"]),
                            ))
                        else:
                            values.extend(("0.000000", "0.000000", "0"))
                    label = " ".join(values) + "\n"
                archive.writestr(f"labels/{split}/{sid}.txt", label)
        return output.getvalue(), {
            "samples": len(records), "positive": positive_count,
            "two_points": two_point_count, **self.status(),
        }