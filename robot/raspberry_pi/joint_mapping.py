"""Session-local mapping with stopped photo evidence and asynchronous Qwen.

The old map is a read-only, UNREGISTERED reference, not a localization prior.
Visual observations are attached to camera poses, not guessed object positions.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import queue
import shutil
import threading
import time
import urllib.request
import uuid

import numpy as np
from PIL import Image, ImageDraw

from raspberry_pi import autonomous_mapping
from raspberry_pi.map_camera_inspection import CameraCapture, OnlineQwenComponentLabeler


def pose_values(pose):
    return [float(pose.x_m), float(pose.y_m), float(pose.yaw_rad)]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def angle_distance(a, b):
    return abs((a - b + math.pi) % math.tau - math.pi)


class QwenViews:
    def __init__(self, url, timeout=25):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def __call__(self, left, right, context):
        image = OnlineQwenComponentLabeler._jpeg_pair(left, right) if right else left.read_bytes()
        body, content_type = OnlineQwenComponentLabeler._multipart(image, context)
        request = urllib.request.Request(self.url + "/landmarks", data=body,
                                         headers={"Content-Type": content_type})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read(4 * 1024 * 1024))
        if not isinstance(payload, dict):
            raise ValueError("Qwen response is not an object")
        return payload


class JointMappingSession:
    """Only the control thread writes records. Worker results cannot mutate poses."""
    def __init__(self, directory, camera, infer, coarse_map=None,
                 spacing_m=0.65, angle_deg=55, clock=time.monotonic):
        self.directory = Path(directory)
        # A unique run directory is mandatory: never overwrite previous evidence.
        self.directory.mkdir(parents=True, exist_ok=False)
        self.camera, self.infer, self.clock = camera, infer, clock
        self.spacing_m, self.angle_rad = spacing_m, math.radians(angle_deg)
        self.records = []
        self.last_attempt = -math.inf
        self.jobs, self.results = queue.Queue(maxsize=2), queue.Queue()
        self.stop_event = threading.Event()
        self.closed = False
        self.frame_id = "session_" + self.directory.name
        self.metadata = {
            "schema_version": 1, "frame_id": self.frame_id,
            "origin": "first accepted lidar scan; +x is initial vehicle heading",
            "coarse_registration": "unregistered_reference_only",
            "coarse_used_for_localization_or_planning": False,
            "annotation_geometry": "camera_viewpoint_not_object_position",
            "exploration_complete": False,
            "completion": "not_started", "observations": self.records,
        }
        if coarse_map:
            coarse = Path(coarse_map)
            reference = self.directory / "coarse_reference"
            reference.mkdir()
            hashes = {}
            for source in (coarse, coarse.with_suffix(".yaml")):
                if not source.is_file():
                    raise FileNotFoundError(source)
                shutil.copy2(source, reference / source.name)
                hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
            self.metadata["coarse_sha256"] = hashes
        self.scans = gzip.open(self.directory / "scans.jsonl.gz", "wt", encoding="utf-8")
        self.worker = threading.Thread(target=self._work, name="joint-qwen", daemon=True)
        self.worker.start()
        self.save()

    def _work(self):
        while not self.stop_event.is_set():
            try:
                job = self.jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            index, left, right, context = job
            try:
                result = self.infer(left, right, context)
                if not isinstance(result, dict):
                    raise ValueError("Qwen response is not an object")
                # Validate serializability before returning to the motion thread.
                json.dumps(result, allow_nan=False)
                self.results.put((index, "analyzed", result))
            except Exception as exc:
                self.results.put((index, "inference_failed", {"error": str(exc)}))
            finally:
                self.jobs.task_done()

    def poll(self):
        changed = False
        while True:
            try:
                index, status, payload = self.results.get_nowait()
            except queue.Empty:
                break
            record = self.records[index]
            record["status"] = status
            record["qwen"] = payload
            # Even a confident visual category does not establish object depth.
            record["object_position_confirmed"] = False
            record["needs_revisit"] = True
            changed = True
        if changed:
            self.save()

    def due(self, pose):
        self.poll()
        if self.clock() - self.last_attempt < 8:
            return False
        current = pose_values(pose)
        return not any(
            r.get("image") and math.dist(current[:2], r["pose"][:2]) < self.spacing_m
            and angle_distance(current[2], r["pose"][2]) < self.angle_rad
            for r in self.records
        )

    def record_scan(self, scan, update, command):
        self.poll()
        finite = lambda values: [float(v) if math.isfinite(float(v)) else None for v in values]
        self.scans.write(json.dumps({
            "timestamp_s": float(scan.timestamp_s),
            "angles_rad": finite(scan.angles_rad), "distances_m": finite(scan.distances_m),
            "pose": pose_values(update.pose), "accepted": bool(update.accepted),
            "map_integrated": bool(update.map_integrated),
            "rejection_reason": update.rejection_reason,
            "last_command": None if command is None else [command.linear_mm_s, command.angular_mrad_s],
        }, allow_nan=False) + "\n")
        self.scans.flush()

    def capture(self, scan, update, grid):
        if not update.accepted or not update.map_integrated:
            raise ValueError("photo must follow accepted stationary fusion")
        self.last_attempt = self.clock()
        index = len(self.records)
        filename = "view_{:04d}.jpg".format(index + 1)
        record = {"id": index + 1, "frame_id": self.frame_id,
                  "pose": pose_values(update.pose), "scan_timestamp_s": float(scan.timestamp_s),
                  "capture_started_utc_s": time.time(), "status": "capturing",
                  "needs_revisit": True, "object_position_confirmed": False}
        self.records.append(record)
        self.save()
        try:
            info = self.camera.capture(self.directory / filename)
            record.update(image=filename, camera=info, capture_finished_utc_s=time.time())
            right = info.get("camera_2_image")
            context = {"frame_id": self.frame_id, "pose": record["pose"],
                       "inspection_mode": "joint_mapping_view", "lidar_features": [],
                       "note": "Describe visible objects only. No depth or map association is established; do not invent lidar feature IDs."}
            record["context"] = context
            try:
                self.jobs.put_nowait((index, self.directory / filename,
                                     self.directory / Path(right).name if right else None, context))
                record["status"] = "queued"
            except queue.Full:
                record["status"] = "deferred_inference"
            print("\n联合建图：已保存视点 {}；{}，不等待推理控制车辆。".format(index + 1, record["status"]))
        except Exception as exc:
            record.update(status="capture_failed", error=str(exc))
            print("\n联合建图拍照失败，保留扫描并继续建图：{}".format(exc))
        self.last_attempt = self.clock()
        self.render(grid)
        self.save()

    def render(self, grid):
        # Render camera locations/directions, not labels at invented object depth.
        image = Image.fromarray(np.flipud(grid.render_image())).convert("RGB")
        image = image.resize((grid.width * 2, grid.height * 2))
        draw = ImageDraw.Draw(image)
        for record in self.records:
            if not record.get("image"):
                continue
            x, y, yaw = record["pose"]
            px = (x - grid.origin_x_m) / grid.resolution_m * 2
            py = (grid.height - 1 - (y - grid.origin_y_m) / grid.resolution_m) * 2
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill="blue")
            draw.line((px, py, px + 18 * math.cos(yaw), py - 18 * math.sin(yaw)), fill="blue", width=2)
            types = []
            payload = record.get("qwen", {})
            items = []
            for key in ("landmarks", "objects"):
                if isinstance(payload.get(key), list):
                    items.extend(payload[key])
            for item in items:
                if isinstance(item, dict):
                    label = str(item.get("type", ""))
                    if label.isascii() and label and label not in types:
                        types.append(label)
            text = str(record["id"]) + (":" + "/".join(types[:2]) if types else "")
            draw.text((px + 4, py + 4), text, fill="red")
        image.save(self.directory / "photo_viewpoints.png")

    def save(self):
        atomic_json(self.directory / "observations.json", self.metadata)

    def finish_map(self, grid, finished=False):
        self.metadata["exploration_complete"] = bool(finished)
        self.poll()
        self.render(grid)
        self.save()

    def close(self, completed=False, wait_s=1):
        if self.closed:
            return
        self.closed = True
        self.stop_event.set()
        self.worker.join(timeout=wait_s)
        self.poll()
        for record in self.records:
            if record["status"] == "queued":
                record["status"] = "deferred_inference"
        self.metadata["completion"] = "mapping_returned" if completed else "interrupted_or_failed"
        self.metadata["all_views_analyzed"] = bool(self.records) and all(r["status"] == "analyzed" for r in self.records)
        self.scans.close()
        self.save()


def build_parser():
    parser = autonomous_mapping.build_argument_parser()
    parser.description = "联合建图与视点标注：启动点局部坐标，旧图只读参考"
    parser.add_argument("--session-root", default="mapping_sessions")
    parser.add_argument("--coarse-map", default="maps/ble_scene_17.pgm")
    parser.add_argument("--annotation-url", default="http://192.0.2.20:8010")
    parser.add_argument("--photo-spacing", type=float, default=0.65)
    parser.add_argument("--photo-angle", type=float, default=55)
    parser.add_argument("--camera-device", type=int, default=0)
    parser.add_argument("--single-camera", action="store_true")
    parser.add_argument("--resume-annotations", metavar="SESSION_DIRECTORY",
                        help="only retry saved photos; never open motor/lidar clients")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.resume_annotations:
        directory = Path(args.resume_annotations)
        manifest = directory / "observations.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        infer = QwenViews(args.annotation_url)
        for record in payload["observations"]:
            if record["status"] == "analyzed" or not record.get("image"):
                continue
            left = directory / Path(record["image"]).name
            right_name = record.get("camera", {}).get("camera_2_image")
            right = directory / Path(right_name).name if right_name else None
            try:
                result = infer(left, right, record["context"])
                json.dumps(result, allow_nan=False)
                record.update(qwen=result, status="analyzed", object_position_confirmed=False)
            except Exception as exc:
                record.update(status="inference_failed", error=str(exc))
            payload["all_views_analyzed"] = all(r["status"] == "analyzed" for r in payload["observations"])
            atomic_json(manifest, payload)
            print("视点 {}：{}".format(record["id"], record["status"]))
        return
    autonomous_mapping.validate_args(args, parser)
    if not math.isfinite(args.photo_spacing) or args.photo_spacing < 0.3:
        parser.error("photo spacing must be finite and >= 0.3 m")
    if not math.isfinite(args.photo_angle) or not 20 <= args.photo_angle <= 180:
        parser.error("photo angle must be between 20 and 180 degrees")
    directory = Path(args.session_root) / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    camera = CameraCapture("opencv", args.camera_device, 1280, 720, 8,
                           dual_usb_cameras=not args.single_camera, free_cameras=False)
    session = JointMappingSession(directory, camera, QwenViews(args.annotation_url),
                                  args.coarse_map, args.photo_spacing, args.photo_angle)
    # Deliberately ignore autonomous_mapping's default output path (old map).
    args.output = str(directory / "map")
    atomic_json(directory / "config.json", vars(args))
    print("联合建图输出：{}；旧图未配准，不参与定位/路径约束。".format(directory))
    completed = False
    try:
        # Camera preflight precedes any client construction / wheel command.
        camera.capture(directory / "camera_test.jpg")
        autonomous_mapping.run(args, mapping_observer=session)
        completed = True
    finally:
        session.close(completed)


if __name__ == "__main__":
    main()
