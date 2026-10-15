"""Paired-camera grasp calibration data and lightweight arm mapping model."""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
import threading
import time
import uuid
import zipfile

import cv2
import numpy as np


ARM_OUTPUTS = ("forward_mm", "up_mm", "yaw_deg")


class ArmStateTracker:
    """Persist command-integrated arm state when no joint encoders are available."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._state = self._load()

    @staticmethod
    def _default() -> dict:
        return {
            "known": False,
            "forward_mm": 0.0,
            "up_mm": 0.0,
            "yaw_deg": 0.0,
            "claw_value": None,
            "sequence": 0,
            "last_command": "",
            "updated_at": time.time(),
            "reference": "unknown",
        }

    def _load(self) -> dict:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            state = self._default()
            state.update(value if isinstance(value, dict) else {})
            return state
        except Exception:
            return self._default()

    def _write(self) -> None:
        self.path.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)

    def mark_baseline(self, reference: str = "manual_baseline") -> dict:
        with self._lock:
            self._state.update(
                known=True,
                forward_mm=0.0,
                up_mm=0.0,
                yaw_deg=0.0,
                sequence=int(self._state.get("sequence", 0)) + 1,
                updated_at=time.time(),
                reference=str(reference),
                last_command="BASELINE",
            )
            self._write()
            return dict(self._state)

    def set_state(self, values: dict) -> dict:
        with self._lock:
            for key in ARM_OUTPUTS:
                if key in values:
                    self._state[key] = float(values[key])
            if "claw_value" in values and values["claw_value"] is not None:
                self._state["claw_value"] = int(values["claw_value"])
            self._state.update(
                known=True,
                sequence=int(self._state.get("sequence", 0)) + 1,
                updated_at=time.time(),
                reference="manual_values",
                last_command="SET_STATE",
            )
            self._write()
            return dict(self._state)

    def apply(self, command: str) -> dict:
        import re

        text = str(command or "").strip().upper()
        with self._lock:
            if text == "#HOME!":
                self._state.update(
                    known=True,
                    forward_mm=0.0,
                    up_mm=0.0,
                    yaw_deg=0.0,
                    reference="home",
                )
            elif text.startswith("#ARM") and text.endswith("!"):
                body = text[4:-1]
                for axis, raw in re.findall(r"([DUBFLRC])(-?\d+)", body):
                    value = int(raw)
                    if axis == "D":
                        self._state["forward_mm"] += value
                    elif axis == "U":
                        self._state["forward_mm"] -= value
                    elif axis == "B":
                        self._state["up_mm"] += value
                    elif axis == "F":
                        self._state["up_mm"] -= value
                    elif axis == "L":
                        self._state["yaw_deg"] += value
                    elif axis == "R":
                        self._state["yaw_deg"] -= value
                    elif axis == "C":
                        self._state["claw_value"] = max(0, min(100, value))
            self._state.update(
                sequence=int(self._state.get("sequence", 0)) + 1,
                updated_at=time.time(),
                last_command=text,
            )
            self._write()
            return dict(self._state)


class ArmMappingDataset:
    """Store synchronized left/right images, annotations and final arm labels."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self.samples = self.root / "samples"
        self.models = self.root / "models"
        self.samples.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _meta_path(self, sample_id: str) -> Path:
        return self.samples / f"{Path(sample_id).name}.json"

    def _image_path(self, sample_id: str, camera: str) -> Path:
        return self.samples / f"{Path(sample_id).name}_{camera}.jpg"

    def capture_pair(
        self,
        left_jpeg: bytes,
        right_jpeg: bytes,
        *,
        object_name: str,
        arm_state: dict,
        motion_steps: dict | None = None,
    ) -> dict:
        frames = {}
        for camera, jpeg in (("left", left_jpeg), ("right", right_jpeg)):
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"invalid {camera} JPEG")
            frames[camera] = frame
        sample_id = (
            f"pair_{time.strftime('%Y%m%d_%H%M%S')}_"
            f"{time.time_ns() % 1_000_000:06d}_{uuid.uuid4().hex[:8]}"
        )
        with self._lock:
            self._image_path(sample_id, "left").write_bytes(left_jpeg)
            self._image_path(sample_id, "right").write_bytes(right_jpeg)
            meta = {
                "sample_id": sample_id,
                "created_at": time.time(),
                "object_name": str(object_name or "").strip(),
                "confirmed": False,
                "grasp_ready": False,
                "jaw_verified": False,
                "arm_state": dict(arm_state or {}),
                "motion_steps": dict(motion_steps or {}),
                "views": {
                    camera: {
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                        "target_found": False,
                        "target_bbox": None,
                        "target_center": None,
                        "jaw_bbox": None,
                        "jaw_points": None,
                    }
                    for camera, frame in frames.items()
                },
                "analysis_raw": "",
                "notes": "",
                "gripper_sample_ids": {},
            }
            self._write(meta)
            return meta

    def _write(self, meta: dict) -> None:
        self._meta_path(meta["sample_id"]).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get(self, sample_id: str) -> dict:
        path = self._meta_path(sample_id)
        if not path.exists():
            raise FileNotFoundError(sample_id)
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _point(value, width: int, height: int) -> dict | None:
        if isinstance(value, dict):
            x, y = value.get("x"), value.get("y")
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            x, y = value[:2]
        else:
            return None
        try:
            return {
                "x": max(0.0, min(float(x), width - 1.0)),
                "y": max(0.0, min(float(y), height - 1.0)),
            }
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _box(value, width: int, height: int) -> dict | None:
        if not isinstance(value, dict):
            return None
        try:
            x1, y1, x2, y2 = [float(value.get(k, 0)) for k in ("x1", "y1", "x2", "y2")]
        except (TypeError, ValueError):
            return None
        x1 = max(0.0, min(x1, width - 2.0))
        y1 = max(0.0, min(y1, height - 2.0))
        x2 = max(x1 + 2.0, min(x2, float(width)))
        y2 = max(y1 + 2.0, min(y2, float(height)))
        return {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width": x2 - x1, "height": y2 - y1,
        }

    def set_analysis(self, sample_id: str, analysis: dict) -> dict:
        with self._lock:
            meta = self.get(sample_id)
            for camera in ("left", "right"):
                incoming = (analysis.get("views") or {}).get(camera) or {}
                view = meta["views"][camera]
                width, height = int(view["width"]), int(view["height"])
                view["target_found"] = bool(incoming.get("target_found"))
                view["target_bbox"] = self._box(incoming.get("target_bbox"), width, height)
                center = self._point(incoming.get("target_center"), width, height)
                if center is None and view["target_bbox"]:
                    box = view["target_bbox"]
                    center = {
                        "x": (box["x1"] + box["x2"]) * 0.5,
                        "y": (box["y1"] + box["y2"]) * 0.5,
                    }
                view["target_center"] = center
                view["jaw_bbox"] = self._box(incoming.get("jaw_bbox"), width, height)
                points = []
                for point in (incoming.get("jaw_points") or [])[:2]:
                    parsed = self._point(point, width, height)
                    if parsed is not None:
                        parsed["v"] = int(point.get("v", 2)) if isinstance(point, dict) else 2
                        parsed["confidence"] = float(point.get("confidence", 1.0)) if isinstance(point, dict) else 1.0
                        points.append(parsed)
                view["jaw_points"] = points
                view["target_confidence"] = float(incoming.get("target_confidence", 0.0) or 0.0)
                view["jaw_confidence"] = float(incoming.get("jaw_confidence", 0.0) or 0.0)
            meta["analysis_raw"] = str(analysis.get("raw", ""))[:8000]
            meta["analyzed_at"] = time.time()
            self._write(meta)
            return meta

    def confirm(
        self,
        sample_id: str,
        *,
        views: dict,
        arm_state: dict,
        grasp_ready: bool,
        jaw_verified: bool,
        notes: str = "",
        gripper_sample_ids: dict | None = None,
    ) -> dict:
        with self._lock:
            meta = self.get(sample_id)
            meta = self.set_analysis(sample_id, {"views": views, "raw": meta.get("analysis_raw", "")})
            if grasp_ready:
                if not bool((arm_state or {}).get("known")):
                    raise ValueError("arm state is unknown; HOME or mark a baseline first")
                for camera in ("left", "right"):
                    if not meta["views"][camera].get("target_center"):
                        raise ValueError(f"{camera} target center is required")
            meta["arm_state"] = dict(arm_state or {})
            meta["grasp_ready"] = bool(grasp_ready)
            meta["jaw_verified"] = bool(jaw_verified)
            meta["notes"] = str(notes or "")[:1000]
            meta["confirmed"] = True
            meta["confirmed_at"] = time.time()
            if gripper_sample_ids is not None:
                meta["gripper_sample_ids"] = dict(gripper_sample_ids)
            self._write(meta)
            return meta

    def list_samples(self, object_name: str = "") -> list[dict]:
        result = []
        wanted = str(object_name or "").strip()
        for path in self.samples.glob("*.json"):
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if wanted and meta.get("object_name") != wanted:
                continue
            result.append(meta)
        result.sort(key=lambda item: (item.get("created_at", 0), item["sample_id"]))
        return result

    def status(self) -> dict:
        records = self.list_samples()
        objects: dict[str, int] = {}
        for item in records:
            name = item.get("object_name") or "unnamed"
            objects[name] = objects.get(name, 0) + int(
                bool(item.get("confirmed") and item.get("grasp_ready"))
            )
        return {
            "samples": len(records),
            "confirmed": sum(bool(item.get("confirmed")) for item in records),
            "grasp_ready": sum(bool(item.get("grasp_ready")) for item in records),
            "jaw_verified": sum(bool(item.get("jaw_verified")) for item in records),
            "objects": objects,
            "latest_model": self.latest_model(),
        }

    @staticmethod
    def _features(meta: dict) -> list[float]:
        values = []
        for camera in ("left", "right"):
            view = meta["views"][camera]
            center = view.get("target_center")
            if not center:
                raise ValueError(f"{camera} target center missing")
            values.extend((
                float(center["x"]) / max(float(view["width"]), 1.0),
                float(center["y"]) / max(float(view["height"]), 1.0),
            ))
        lx, ly, rx, ry = values
        return [
            1.0, lx, ly, rx, ry,
            lx - rx, (lx + rx) * 0.5, (ly + ry) * 0.5,
            lx * lx, ly * ly, rx * rx, ry * ry,
            lx * rx, ly * ry,
        ]

    def _training_records(self, object_name: str) -> list[dict]:
        return [
            item for item in self.list_samples(object_name)
            if item.get("confirmed") and item.get("grasp_ready")
            and bool((item.get("arm_state") or {}).get("known"))
            and all((item.get("views") or {}).get(camera, {}).get("target_center")
                    for camera in ("left", "right"))
        ]

    def train(
        self,
        object_name: str,
        *,
        ridge: float = 0.01,
        quantization: dict | None = None,
    ) -> dict:
        records = self._training_records(object_name)
        if len(records) < 8:
            raise ValueError("at least 8 confirmed grasp-ready paired samples are required")
        x = np.asarray([self._features(item) for item in records], dtype=np.float64)
        y = np.asarray([
            [float(item["arm_state"][name]) for name in ARM_OUTPUTS]
            for item in records
        ], dtype=np.float64)
        order = np.random.default_rng(42).permutation(len(records))
        test_count = max(1, int(round(len(records) * 0.2)))
        test_idx, train_idx = order[:test_count], order[test_count:]
        x_train, y_train = x[train_idx], y[train_idx]
        regularizer = np.eye(x.shape[1], dtype=np.float64) * max(0.0, float(ridge))
        regularizer[0, 0] = 0.0
        coefficients = np.linalg.pinv(x_train.T @ x_train + regularizer) @ x_train.T @ y_train
        prediction = x[test_idx] @ coefficients
        mae = np.mean(np.abs(prediction - y[test_idx]), axis=0)
        quantum = {
            name: max(0.001, float((quantization or {}).get(name, 1.0)))
            for name in ARM_OUTPUTS
        }
        model = {
            "version": 1,
            "created_at": time.time(),
            "object_name": str(object_name or "").strip(),
            "feature_kind": "stereo_target_centers_polynomial_v1",
            "feature_names": [
                "bias", "left_x", "left_y", "right_x", "right_y",
                "disparity_x", "mean_x", "mean_y", "left_x2", "left_y2",
                "right_x2", "right_y2", "left_x_right_x", "left_y_right_y",
            ],
            "outputs": list(ARM_OUTPUTS),
            "coefficients": coefficients.tolist(),
            "quantization": quantum,
            "ridge": float(ridge),
            "samples": len(records),
            "test_samples": int(test_count),
            "test_mae": {name: round(float(mae[i]), 4) for i, name in enumerate(ARM_OUTPUTS)},
            "sample_ids": [item["sample_id"] for item in records],
        }
        path = self.models / f"arm_mapping_{int(time.time())}_{uuid.uuid4().hex[:6]}.json"
        path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.models / "latest.json").write_text(
            json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        return model

    def latest_model(self) -> dict | None:
        path = self.models / "latest.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def predict(self, views: dict, model: dict | None = None) -> dict:
        selected = model or self.latest_model()
        if not selected:
            raise ValueError("arm mapping model is not trained")
        meta = {"views": views}
        feature = np.asarray(self._features(meta), dtype=np.float64)
        coefficients = np.asarray(selected["coefficients"], dtype=np.float64)
        raw = feature @ coefficients
        result = {}
        for index, name in enumerate(ARM_OUTPUTS):
            quantum = float((selected.get("quantization") or {}).get(name, 1.0))
            value = round(float(raw[index]) / quantum) * quantum
            result[name] = round(value, 4)
        return {
            "object_name": selected.get("object_name", ""),
            "parameters": result,
            "raw_parameters": {
                name: round(float(raw[index]), 4) for index, name in enumerate(ARM_OUTPUTS)
            },
            "quantization": selected.get("quantization", {}),
            "test_mae": selected.get("test_mae", {}),
        }

    def build_zip(self) -> tuple[bytes, dict]:
        records = self.list_samples()
        output = io.BytesIO()
        csv_text = io.StringIO()
        writer = csv.writer(csv_text)
        writer.writerow([
            "sample_id", "object_name", "grasp_ready",
            "left_x", "left_y", "right_x", "right_y",
            "forward_mm", "up_mm", "yaw_deg", "claw_value",
        ])
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for meta in records:
                sample_id = meta["sample_id"]
                archive.write(self._meta_path(sample_id), f"samples/{sample_id}.json")
                for camera in ("left", "right"):
                    image = self._image_path(sample_id, camera)
                    if image.exists():
                        archive.write(image, f"images/{sample_id}_{camera}.jpg")
                centers = [
                    (meta.get("views") or {}).get(camera, {}).get("target_center") or {}
                    for camera in ("left", "right")
                ]
                arm = meta.get("arm_state") or {}
                writer.writerow([
                    sample_id, meta.get("object_name", ""), int(bool(meta.get("grasp_ready"))),
                    centers[0].get("x", ""), centers[0].get("y", ""),
                    centers[1].get("x", ""), centers[1].get("y", ""),
                    arm.get("forward_mm", ""), arm.get("up_mm", ""),
                    arm.get("yaw_deg", ""), arm.get("claw_value", ""),
                ])
            archive.writestr("arm_mapping_samples.csv", csv_text.getvalue())
            latest = self.models / "latest.json"
            if latest.exists():
                archive.write(latest, "models/latest.json")
        return output.getvalue(), self.status()
