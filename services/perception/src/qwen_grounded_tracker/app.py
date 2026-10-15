from __future__ import annotations

import socket
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, strftime
from typing import Any

import cv2
import numpy as np

from qwen_grounded_tracker.camera.opencv_camera import OpenCVCamera
from qwen_grounded_tracker.config import section
from qwen_grounded_tracker.domain import (
    GroundingResult,
    MotionGuidance,
    ObstacleObservation,
    TrackObservation,
)
from qwen_grounded_tracker.grounding.qwen_grounder import Qwen3VLTargetGrounder
from qwen_grounded_tracker.grounding.worker import GroundingResponse, GroundingWorker
from qwen_grounded_tracker.lidar.null_lidar import NullLidarProvider
from qwen_grounded_tracker.navigation.direction_planner import DirectionPlanner
from qwen_grounded_tracker.perception.yolo_obstacles import YoloObstacleDetector
from qwen_grounded_tracker.safety.arbiter import SafetyArbiter
from qwen_grounded_tracker.tracking.contour_refiner import ContourRefiner
from qwen_grounded_tracker.tracking.csrt_tracker import CSRTTargetTracker
from qwen_grounded_tracker.tracking.identity_guard import IdentityGuard
from qwen_grounded_tracker.tracking.relocator import ReferenceRelocator
from qwen_grounded_tracker.ui.overlay import draw_overlay


class SingleInstanceLock:
    def __init__(self, port: int) -> None:
        self.port = port
        self.socket: socket.socket | None = None

    def acquire(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        try:
            sock.bind(("127.0.0.1", self.port))
            sock.listen(1)
        except OSError as exc:
            sock.close()
            raise RuntimeError(
                "Another demo instance may already be running. "
                f"Single-instance port {self.port} is occupied."
            ) from exc
        self.socket = sock

    def release(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None


@dataclass
class RuntimeTarget:
    logical_id: str
    instruction: str
    target_name: str
    bbox: Any | None = None
    reference_crop: np.ndarray | None = None
    last_seen_at: float | None = None
    lost_since: float | None = None
    identity_score: float = 0.0


class GroundedTrackingApp:
    def __init__(self, config: dict[str, Any], instruction: str) -> None:
        self.config = config
        self.instruction = instruction.strip()
        if not self.instruction:
            raise ValueError("Instruction must not be empty")

        app_cfg = section(config, "app")
        camera_cfg = section(config, "camera")
        qwen_cfg = section(config, "qwen")
        tracking_cfg = section(config, "tracking")
        boundary_cfg = section(config, "boundary")
        obstacle_cfg = section(config, "obstacles")
        lidar_cfg = section(config, "lidar")
        navigation_cfg = section(config, "navigation")
        safety_cfg = section(config, "safety")

        source_raw = camera_cfg.get("source", 0)
        try:
            camera_source: int | str = int(source_raw)
        except (TypeError, ValueError):
            camera_source = str(source_raw)

        self.window_name = str(app_cfg.get("window_name", "Qwen Grounded Tracker"))
        self.output_dir = Path(str(app_cfg.get("output_dir", "outputs")))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval_seconds = float(app_cfg.get("log_interval_seconds", 1.0))
        self.instance_lock = SingleInstanceLock(int(app_cfg.get("single_instance_port", 39521)))

        self.camera = OpenCVCamera(
            source=camera_source,
            width=int(camera_cfg.get("width", 640)),
            height=int(camera_cfg.get("height", 480)),
            fps=int(camera_cfg.get("fps", 30)),
            backend=str(camera_cfg.get("backend", "dshow")),
            side_by_side=str(camera_cfg.get("side_by_side", "auto")),
        )

        self.grounder = Qwen3VLTargetGrounder(
            model_path=str(qwen_cfg.get("model_path")),
            device=str(qwen_cfg.get("device", "cpu")),
            local_files_only=bool(qwen_cfg.get("local_files_only", True)),
            max_new_tokens=int(qwen_cfg.get("max_new_tokens", 64)),
            max_time_seconds=float(qwen_cfg.get("max_time_seconds", 90.0)),
            min_visual_tokens=int(qwen_cfg.get("min_visual_tokens", 64)),
            max_visual_tokens=int(qwen_cfg.get("max_visual_tokens", 256)),
            minimum_box_area_ratio=float(qwen_cfg.get("minimum_box_area_ratio", 0.001)),
        )
        self.grounding_worker = GroundingWorker(self.grounder)

        self.tracker = CSRTTargetTracker()
        self.identity_guard = IdentityGuard(
            minimum_score=float(tracking_cfg.get("identity_min_score", 0.32)),
            maximum_bad_frames=int(tracking_cfg.get("identity_bad_frames", 4)),
            reference_update_score=float(tracking_cfg.get("reference_update_score", 0.78)),
        )
        self.auto_reground = bool(tracking_cfg.get("auto_reground", True))
        self.auto_reground_after_seconds = float(
            tracking_cfg.get("auto_reground_after_seconds", 3.0)
        )
        self.hold_after_loss_seconds = float(
            tracking_cfg.get("hold_after_loss_seconds", 0.8)
        )
        self.relocator = ReferenceRelocator(
            template_match_threshold=float(
                tracking_cfg.get("template_match_threshold", 0.52)
            )
        )

        self.contour_refiner = ContourRefiner(
            mode=str(boundary_cfg.get("mode", "grabcut")),
            update_interval_frames=int(boundary_cfg.get("update_interval_frames", 4)),
            grabcut_iterations=int(boundary_cfg.get("grabcut_iterations", 2)),
        )

        self.obstacle_detector = YoloObstacleDetector(
            enabled=bool(obstacle_cfg.get("enabled", True)),
            model_path=str(obstacle_cfg.get("model_path", "models/yolo/yolo11n-seg.pt")),
            device=str(obstacle_cfg.get("device", "cpu")),
            image_size=int(obstacle_cfg.get("image_size", 416)),
            confidence=float(obstacle_cfg.get("confidence", 0.25)),
            frame_interval=int(obstacle_cfg.get("frame_interval", 3)),
            stop_labels=list(obstacle_cfg.get("stop_labels", [])),
            danger_zone=dict(obstacle_cfg.get("danger_zone", {})),
            exclude_target_iou=float(obstacle_cfg.get("exclude_target_iou", 0.60)),
        )

        self.lidar = NullLidarProvider()
        self.direction_planner = DirectionPlanner(
            center_x=float(navigation_cfg.get("center_x", 0.5)),
            center_tolerance=float(navigation_cfg.get("center_tolerance", 0.07)),
            stop_area_ratio=float(navigation_cfg.get("stop_area_ratio", 0.16)),
            too_close_area_ratio=float(navigation_cfg.get("too_close_area_ratio", 0.28)),
            linear_speed=float(navigation_cfg.get("linear_speed", 0.08)),
            backward_speed=float(navigation_cfg.get("backward_speed", 0.04)),
            angular_speed=float(navigation_cfg.get("angular_speed", 0.22)),
            enabled=bool(navigation_cfg.get("enabled", True)),
        )
        self.safety = SafetyArbiter(
            require_lidar_ready=bool(lidar_cfg.get("require_ready", False)),
            stop_on_tracking_loss=bool(safety_cfg.get("stop_on_tracking_loss", True)),
            stop_on_yolo_obstacle=bool(safety_cfg.get("stop_on_yolo_obstacle", True)),
        )

        self.emergency_stop = bool(safety_cfg.get("emergency_stop_default", False))
        self.target_counter = 0
        self.target: RuntimeTarget | None = None
        self.grounding_status = "waiting for first frame"
        self.last_log_at = 0.0
        self.latest_obstacles = ObstacleObservation(status="not run")

    def _new_logical_id(self) -> str:
        self.target_counter += 1
        return f"target_{self.target_counter:03d}"

    def _reset_tracking(self, keep_reference: bool = True) -> None:
        self.tracker.reset()
        self.identity_guard.reset()
        self.contour_refiner.reset()
        if self.target is not None:
            self.target.bbox = None
            self.target.last_seen_at = None
            self.target.lost_since = None
            self.target.identity_score = 0.0
            if not keep_reference:
                self.target.reference_crop = None

    def _submit_grounding(self, frame: np.ndarray, reason: str) -> None:
        reference = self.target.reference_crop if self.target is not None else None
        submitted = self.grounding_worker.submit(
            frame=frame,
            instruction=self.instruction,
            reference_image=reference,
        )
        if submitted:
            self.grounding_status = f"running: {reason}"
            print(f"[Grounding] {reason}; instruction={self.instruction}")

    def _install_grounding(self, response: GroundingResponse, current_frame: np.ndarray) -> None:
        result: GroundingResult = response.result
        if not result.found or result.bbox is None:
            self.grounding_status = f"failed: {result.message}"
            print(f"[Grounding failed] {result.message}; raw={result.raw_text}")
            return

        relocation = self.relocator.relocate(
            snapshot=response.snapshot,
            snapshot_bbox=result.bbox,
            current_frame=current_frame,
        )
        current_box = relocation.bbox
        reference_crop = result.bbox.crop(response.snapshot)

        if self.target is None or self.target.instruction != response.instruction:
            self.target = RuntimeTarget(
                logical_id=self._new_logical_id(),
                instruction=response.instruction,
                target_name=result.target_name,
            )
        self.target.target_name = result.target_name
        self.target.reference_crop = reference_crop
        self.target.bbox = current_box
        self.target.last_seen_at = monotonic()
        self.target.lost_since = None

        self.tracker.reset()
        self.tracker.initialize(current_frame, current_box)
        self.identity_guard.reset()
        self.identity_guard.initialize(current_frame, current_box)
        self.contour_refiner.reset()
        self.grounding_status = (
            f"ready {result.target_name}; relocation={relocation.method} {relocation.score:.2f}"
        )
        print(
            f"[Target installed] id={self.target.logical_id}, name={result.target_name}, "
            f"bbox={current_box.to_xywh()}, relocation={relocation.method}, "
            f"score={relocation.score:.2f}"
        )

    def _manual_roi(self, frame: np.ndarray) -> None:
        roi = cv2.selectROI("Manual target selection", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Manual target selection")
        x, y, w, h = [int(v) for v in roi]
        if w <= 1 or h <= 1:
            return

        from qwen_grounded_tracker.domain import BBox

        bbox = BBox(x, y, x + w, y + h)
        self.target = RuntimeTarget(
            logical_id=self._new_logical_id(),
            instruction=self.instruction,
            target_name="manual target",
            bbox=bbox,
            reference_crop=bbox.crop(frame),
            last_seen_at=monotonic(),
        )
        self.tracker.reset()
        self.tracker.initialize(frame, bbox)
        self.identity_guard.reset()
        self.identity_guard.initialize(frame, bbox)
        self.contour_refiner.reset()
        self.grounding_status = "manual ROI initialized"

    def _update_track(self, frame: np.ndarray) -> TrackObservation:
        now = monotonic()
        if self.target is None or not self.tracker.initialized:
            return TrackObservation(
                visible=False,
                logical_target_id=self.target.logical_id if self.target else None,
                bbox=None,
                contour=None,
                identity_score=0.0,
                status="waiting for Qwen target grounding",
                lost_seconds=0.0,
            )

        ok, bbox = self.tracker.update(frame)
        if ok and bbox is not None:
            identity = self.identity_guard.evaluate(frame, bbox)
            if identity.accepted:
                contour = self.contour_refiner.refine(frame, bbox)
                self.target.bbox = bbox
                self.target.last_seen_at = now
                self.target.lost_since = None
                self.target.identity_score = identity.score
                return TrackObservation(
                    visible=True,
                    logical_target_id=self.target.logical_id,
                    bbox=bbox,
                    contour=contour,
                    identity_score=identity.score,
                    status=identity.status,
                    lost_seconds=0.0,
                )
            status = identity.status
        else:
            status = "CSRT update failed"

        if self.target.lost_since is None:
            self.target.lost_since = now
        lost_seconds = now - self.target.lost_since
        return TrackObservation(
            visible=False,
            logical_target_id=self.target.logical_id,
            bbox=None,
            contour=None,
            identity_score=self.target.identity_score,
            status=f"target lost {lost_seconds:.1f}s: {status}",
            lost_seconds=lost_seconds,
        )

    def _maybe_auto_reground(self, frame: np.ndarray, track: TrackObservation) -> None:
        if (
            self.auto_reground
            and not track.visible
            and track.lost_seconds >= self.auto_reground_after_seconds
            and not self.grounding_worker.busy
        ):
            self._submit_grounding(frame, "automatic re-ground after target loss")

    def _handle_key(self, key: int, frame: np.ndarray) -> bool:
        if key in (ord("q"), 27):
            return False
        if key == ord("g"):
            self._submit_grounding(frame, "manual re-ground")
        elif key == ord("i"):
            print("\nEnter a new instruction. Camera refresh pauses while waiting for input.")
            new_instruction = input("instruction> ").strip()
            if new_instruction:
                self.instruction = new_instruction
                self.target = None
                self._reset_tracking(keep_reference=False)
                self._submit_grounding(frame, "new instruction")
        elif key == ord("m"):
            self._manual_roi(frame)
        elif key == ord("r"):
            self.target = None
            self._reset_tracking(keep_reference=False)
            self.grounding_status = "reset; press G to ground again"
        elif key == ord("c"):
            mode = self.contour_refiner.toggle_mode()
            print(f"[Boundary] mode={mode}")
        elif key == ord("s"):
            filename = self.output_dir / f"capture_{strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"[Saved] {filename}")
        elif key == 32:
            self.emergency_stop = not self.emergency_stop
            print(f"[Emergency stop] {self.emergency_stop}")
        return True

    def run(self) -> None:
        self.instance_lock.acquire()
        self.camera.open()
        self.lidar.open()
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        frame_count = 0
        running = True
        try:
            while running:
                frame = self.camera.read()
                if frame is None:
                    print("[Camera] frame missing")
                    continue
                frame_count += 1

                if frame_count == 3 and self.target is None and not self.grounding_worker.busy:
                    self._submit_grounding(frame, "initial target grounding")

                try:
                    response = self.grounding_worker.poll()
                except Exception as exc:
                    self.grounding_status = f"error: {exc!r}"
                    print(f"[Qwen grounding error] {exc!r}")
                    response = None
                if response is not None:
                    self._install_grounding(response, frame)

                track = self._update_track(frame)
                self._maybe_auto_reground(frame, track)

                if self.grounding_worker.busy:
                    self.grounding_status = (
                        f"running {self.grounding_worker.elapsed_seconds:.1f}s; chassis STOP"
                    )

                target_bbox = track.bbox if track.visible else None
                if self.grounding_worker.busy:
                    # The virtual chassis is already stopped. Pausing YOLO here prevents
                    # CPU contention from making Qwen grounding unnecessarily slower.
                    self.latest_obstacles = ObstacleObservation(
                        danger=False,
                        status="paused during Qwen grounding",
                    )
                else:
                    try:
                        self.latest_obstacles = self.obstacle_detector.detect(frame, target_bbox)
                    except Exception as exc:
                        self.latest_obstacles = ObstacleObservation(
                            danger=False,
                            status=f"YOLO error: {exc}",
                        )
                        if monotonic() - self.last_log_at >= self.log_interval_seconds:
                            print(f"[YOLO obstacle error] {exc!r}")

                lidar_observation = self.lidar.read()
                requested = self.direction_planner.plan(
                    bbox=target_bbox,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                )

                # During any Qwen grounding request the virtual chassis stays stopped.
                if self.grounding_worker.busy:
                    requested = MotionGuidance(
                        "STOP",
                        0.0,
                        0.0,
                        "Qwen grounding is in progress",
                    )

                safety = self.safety.decide(
                    requested=requested,
                    tracking_visible=track.visible and not self.grounding_worker.busy,
                    yolo_obstacles=self.latest_obstacles,
                    lidar=lidar_observation,
                    emergency_stop=self.emergency_stop,
                )

                output = draw_overlay(
                    frame=frame,
                    track=track,
                    obstacles=self.latest_obstacles,
                    lidar=lidar_observation,
                    safety=safety,
                    grounding_status=self.grounding_status,
                    boundary_mode=self.contour_refiner.mode,
                    emergency_stop=self.emergency_stop,
                )
                cv2.imshow(self.window_name, output)

                now = monotonic()
                if now - self.last_log_at >= self.log_interval_seconds:
                    print(
                        f"[Runtime] track={track.status}; guidance={safety.guidance.direction}; "
                        f"blocked={safety.blocked}; yolo={self.latest_obstacles.status}; "
                        f"lidar={lidar_observation.status}"
                    )
                    self.last_log_at = now

                key = cv2.waitKey(1) & 0xFF
                running = self._handle_key(key, frame)
        finally:
            self.grounding_worker.close()
            self.lidar.close()
            self.camera.close()
            cv2.destroyAllWindows()
            self.instance_lock.release()
