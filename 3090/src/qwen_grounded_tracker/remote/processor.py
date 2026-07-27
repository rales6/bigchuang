from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any

import numpy as np

from qwen_grounded_tracker.config import section
from qwen_grounded_tracker.domain import (
    BBox,
    GroundedCandidate,
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
from qwen_grounded_tracker.robot_task import RobotTaskController
from qwen_grounded_tracker.safety.arbiter import SafetyArbiter
from qwen_grounded_tracker.tracking.contour_refiner import ContourRefiner
from qwen_grounded_tracker.tracking.csrt_tracker import CSRTTargetTracker
from qwen_grounded_tracker.tracking.identity_guard import IdentityGuard
from qwen_grounded_tracker.tracking.motion_predictor import MotionPredictor
from qwen_grounded_tracker.tracking.relocator import ReferenceRelocator
from qwen_grounded_tracker.ui.overlay import draw_overlay


@dataclass
class RuntimeTarget:
    logical_id: str
    instruction: str
    target_name: str
    bbox: BBox | None = None
    reference_crop: np.ndarray | None = None
    last_seen_at: float | None = None
    lost_since: float | None = None
    identity_score: float = 0.0


@dataclass
class QueuedTarget:
    logical_id: str
    instruction: str
    target_name: str
    bbox: BBox
    reference_crop: np.ndarray
    status: str = "pending"


def _bbox_payload(bbox: BBox | None) -> dict[str, float] | None:
    if bbox is None:
        return None
    return {
        "x1": float(bbox.x1),
        "y1": float(bbox.y1),
        "x2": float(bbox.x2),
        "y2": float(bbox.y2),
        "width": float(bbox.width),
        "height": float(bbox.height),
    }


def _contour_payload(contour: np.ndarray | None, max_points: int = 180) -> list[dict[str, float]] | None:
    if contour is None:
        return None
    points = contour.reshape(-1, 2)
    if len(points) > max_points:
        step = max(1, len(points) // max_points)
        points = points[::step]
    return [{"x": float(x), "y": float(y)} for x, y in points]


def _track_payload(track: TrackObservation) -> dict[str, Any]:
    return {
        "visible": track.visible,
        "logical_target_id": track.logical_target_id,
        "bbox": _bbox_payload(track.bbox),
        "contour": _contour_payload(track.contour),
        "identity_score": float(track.identity_score),
        "status": track.status,
        "lost_seconds": float(track.lost_seconds),
    }


def _queue_payload(queue: list[QueuedTarget], active_index: int) -> list[dict[str, Any]]:
    return [
        {
            "id": item.logical_id,
            "target_name": item.target_name,
            "status": item.status,
            "active": index == active_index and item.status == "active",
            "bbox": _bbox_payload(item.bbox),
        }
        for index, item in enumerate(queue)
    ]


class RemoteFrameProcessor:
    """Process externally streamed camera frames with the original tracker pipeline."""

    def __init__(self, config: dict[str, Any], instruction: str) -> None:
        self.config = config
        self.instruction = instruction.strip()
        if not self.instruction:
            raise ValueError("Instruction must not be empty")

        app_cfg = section(config, "app")
        qwen_cfg = section(config, "qwen")
        tracking_cfg = section(config, "tracking")
        boundary_cfg = section(config, "boundary")
        obstacle_cfg = section(config, "obstacles")
        lidar_cfg = section(config, "lidar")
        navigation_cfg = section(config, "navigation")
        safety_cfg = section(config, "safety")
        task_cfg = section(config, "task")
        prediction_cfg = section(config, "prediction")

        self.log_interval_seconds = float(app_cfg.get("log_interval_seconds", 1.0))
        self.grounder = Qwen3VLTargetGrounder(
            model_path=str(qwen_cfg.get("model_path")),
            device=str(qwen_cfg.get("device", "cuda")),
            local_files_only=bool(qwen_cfg.get("local_files_only", True)),
            max_new_tokens=int(qwen_cfg.get("max_new_tokens", 64)),
            max_time_seconds=float(qwen_cfg.get("max_time_seconds", 30.0)),
            min_visual_tokens=int(qwen_cfg.get("min_visual_tokens", 128)),
            max_visual_tokens=int(qwen_cfg.get("max_visual_tokens", 512)),
            minimum_box_area_ratio=float(qwen_cfg.get("minimum_box_area_ratio", 0.001)),
        )
        fallback_cfg = qwen_cfg.get("fallback_box", {})
        if not isinstance(fallback_cfg, dict):
            fallback_cfg = {}
        self.qwen_fallback_enabled = bool(fallback_cfg.get("enabled", True))
        self.qwen_fallback_width_ratio = float(fallback_cfg.get("width_ratio", 0.35))
        self.qwen_fallback_height_ratio = float(fallback_cfg.get("height_ratio", 0.35))
        self.qwen_fallback_y_center = float(fallback_cfg.get("y_center", 0.50))
        self.qwen_fallback_min_failures = int(fallback_cfg.get("after_failures", 1))
        self.consecutive_grounding_failures = 0
        self.grounding_worker = GroundingWorker(self.grounder)

        self.tracker = CSRTTargetTracker()
        self.identity_guard = IdentityGuard(
            minimum_score=float(tracking_cfg.get("identity_min_score", 0.32)),
            maximum_bad_frames=int(tracking_cfg.get("identity_bad_frames", 4)),
            reference_update_score=float(tracking_cfg.get("reference_update_score", 0.78)),
        )
        self.auto_reground = bool(tracking_cfg.get("auto_reground", True))
        self.auto_reground_after_seconds = float(
            tracking_cfg.get("auto_reground_after_seconds", 2.0)
        )
        self.relocator = ReferenceRelocator(
            template_match_threshold=float(
                tracking_cfg.get("template_match_threshold", 0.52)
            )
        )
        self.contour_refiner = ContourRefiner(
            mode=str(boundary_cfg.get("mode", "grabcut")),
            update_interval_frames=int(boundary_cfg.get("update_interval_frames", 2)),
            grabcut_iterations=int(boundary_cfg.get("grabcut_iterations", 2)),
        )
        self.motion_predictor = MotionPredictor(
            enabled=bool(prediction_cfg.get("enabled", True)),
            history_size=int(prediction_cfg.get("history_size", 5)),
            max_prediction_seconds=float(prediction_cfg.get("max_prediction_seconds", 0.35)),
            velocity_damping=float(prediction_cfg.get("velocity_damping", 0.75)),
            max_velocity_px_per_second=float(
                prediction_cfg.get("max_velocity_px_per_second", 900.0)
            ),
            max_size_change_px_per_second=float(
                prediction_cfg.get("max_size_change_px_per_second", 500.0)
            ),
        )
        self.obstacle_detector = YoloObstacleDetector(
            enabled=bool(obstacle_cfg.get("enabled", True)),
            model_path=str(obstacle_cfg.get("model_path", "models/yolo/yolo11n-seg.pt")),
            device=str(obstacle_cfg.get("device", "0")),
            image_size=int(obstacle_cfg.get("image_size", 640)),
            confidence=float(obstacle_cfg.get("confidence", 0.25)),
            frame_interval=int(obstacle_cfg.get("frame_interval", 1)),
            stop_labels=list(obstacle_cfg.get("stop_labels", [])),
            danger_zone=dict(obstacle_cfg.get("danger_zone", {})),
            exclude_target_iou=float(obstacle_cfg.get("exclude_target_iou", 0.60)),
            safety_only=bool(obstacle_cfg.get("safety_only", True)),
            report_clear_detections=bool(obstacle_cfg.get("report_clear_detections", False)),
        )
        self.lidar = NullLidarProvider()
        self.direction_planner = DirectionPlanner(
            center_x=float(navigation_cfg.get("center_x", 0.5)),
            center_tolerance=float(navigation_cfg.get("center_tolerance", 0.06)),
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
        self.robot_task = RobotTaskController(task_cfg)
        self.robot_task.load_instruction(self.instruction)
        self.instruction = self.robot_task.current_grounding_instruction or self.instruction

        self.emergency_stop = bool(safety_cfg.get("emergency_stop_default", False))
        self.target_counter = 0
        self.target: RuntimeTarget | None = None
        self.target_queue: list[QueuedTarget] = []
        self.active_queue_index = -1
        self.grounding_status = "waiting for first streamed frame"
        self.latest_obstacles = ObstacleObservation(status="not run")
        self.frame_count = 0
        self.last_log_at = 0.0
        self.pending_manual_bbox: BBox | None = None
        self.grounding_requested = True
        self.latest_measured_track = TrackObservation(
            visible=False,
            logical_target_id=None,
            bbox=None,
            contour=None,
            identity_score=0.0,
            status="not run",
        )
        self.latest_control_track = self.latest_measured_track
        self.latest_prediction_state: dict[str, Any] = {
            "enabled": self.motion_predictor.enabled,
            "active": False,
            "age_seconds": 0.0,
        }
        self.latest_robot_task: dict[str, Any] = {
            "active": False,
            "task_type": "none",
            "phase": "idle",
            "phase_label": "No robot task",
            "completed": False,
        }
        self.latest_robot_command: dict[str, Any] = {
            "mode": "simulated",
            "subsystem": "vision",
            "action": "idle",
            "reason": "No active task",
        }

    def start(self) -> None:
        self.lidar.open()

    def close(self) -> None:
        self.grounding_worker.close()
        self.lidar.close()

    def set_instruction(self, instruction: str) -> None:
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("Instruction must not be empty")
        self.robot_task.load_instruction(instruction)
        self.instruction = self.robot_task.current_grounding_instruction or instruction
        self.target = None
        self.target_queue = []
        self.active_queue_index = -1
        self._reset_tracking(keep_reference=False)
        self.grounding_status = "instruction changed; waiting for next frame"
        self.grounding_requested = True
        self.consecutive_grounding_failures = 0

    def request_reground(self) -> None:
        self.grounding_status = "manual re-ground requested"
        self.grounding_requested = True

    def reset(self) -> None:
        self.target = None
        self.target_queue = []
        self.active_queue_index = -1
        self.robot_task.reset()
        self._reset_tracking(keep_reference=False)
        self.grounding_status = "reset; press G on the camera client to ground again"
        self.grounding_requested = False
        self.consecutive_grounding_failures = 0

    def set_emergency_stop(self, enabled: bool) -> None:
        self.emergency_stop = enabled

    def toggle_boundary(self) -> str:
        return self.contour_refiner.toggle_mode()

    def set_manual_roi_xywh(self, x: float, y: float, width: float, height: float) -> None:
        if width <= 1 or height <= 1:
            raise ValueError("Manual ROI must have width and height greater than 1")
        self.pending_manual_bbox = BBox(x, y, x + width, y + height)

    def _new_logical_id(self) -> str:
        self.target_counter += 1
        return f"target_{self.target_counter:03d}"

    def _reset_tracking(self, keep_reference: bool = True) -> None:
        self.tracker.reset()
        self.identity_guard.reset()
        self.contour_refiner.reset()
        self.motion_predictor.reset()
        empty_track = TrackObservation(
            visible=False,
            logical_target_id=self.target.logical_id if self.target else None,
            bbox=None,
            contour=None,
            identity_score=0.0,
            status="tracking reset",
        )
        self.latest_measured_track = empty_track
        self.latest_control_track = empty_track
        self.latest_prediction_state = {
            "enabled": self.motion_predictor.enabled,
            "active": False,
            "age_seconds": 0.0,
        }
        if self.target is not None:
            self.target.bbox = None
            self.target.last_seen_at = None
            self.target.lost_since = None
            self.target.identity_score = 0.0
            if not keep_reference:
                self.target.reference_crop = None

    def _has_next_queued_target(self) -> bool:
        return any(item.status == "pending" for item in self.target_queue)

    def _mark_active_queue_done(self) -> None:
        if 0 <= self.active_queue_index < len(self.target_queue):
            self.target_queue[self.active_queue_index].status = "done"

    def _activate_queued_target(self, index: int, frame: np.ndarray) -> bool:
        if index < 0 or index >= len(self.target_queue):
            return False
        queued = self.target_queue[index]
        if queued.status == "done":
            return False

        for item in self.target_queue:
            if item.status == "active":
                item.status = "pending"
        queued.status = "active"
        self.active_queue_index = index

        # 多目标队列只激活一个目标，CSRT 仍然保持单目标跟踪，控制逻辑更稳定。
        self.target = RuntimeTarget(
            logical_id=queued.logical_id,
            instruction=queued.instruction,
            target_name=queued.target_name,
            bbox=queued.bbox,
            reference_crop=queued.reference_crop,
            last_seen_at=monotonic(),
        )
        self.tracker.reset()
        self.tracker.initialize(frame, queued.bbox)
        self.identity_guard.reset()
        self.identity_guard.initialize(frame, queued.bbox)
        self.contour_refiner.reset()
        return True

    def _activate_next_queued_target(self, frame: np.ndarray) -> bool:
        for index, item in enumerate(self.target_queue):
            if item.status == "pending":
                return self._activate_queued_target(index, frame)
        return False

    def _build_target_queue(
        self,
        candidates: list[GroundedCandidate],
        response: GroundingResponse,
        current_frame: np.ndarray,
    ) -> None:
        count_limit = 1
        if self.robot_task.plan is not None:
            count_limit = max(1, int(self.robot_task.plan.requested_count))
        ordered = sorted(candidates, key=lambda item: item.bbox.x1)[:count_limit]
        queue: list[QueuedTarget] = []
        for candidate in ordered:
            relocation = self.relocator.relocate(
                snapshot=response.snapshot,
                snapshot_bbox=candidate.bbox,
                current_frame=current_frame,
            )
            bbox = relocation.bbox
            queue.append(
                QueuedTarget(
                    logical_id=self._new_logical_id(),
                    instruction=response.instruction,
                    target_name=candidate.target_name,
                    bbox=bbox,
                    reference_crop=candidate.bbox.crop(response.snapshot),
                )
            )
        self.target_queue = queue
        self.active_queue_index = -1

    def _submit_grounding(self, frame: np.ndarray, reason: str) -> None:
        self.instruction = self.robot_task.current_grounding_instruction or self.instruction
        reference = self.target.reference_crop if self.target is not None else None
        submitted = self.grounding_worker.submit(
            frame=frame,
            instruction=self.instruction,
            reference_image=reference,
        )
        if submitted:
            self.grounding_status = f"running: {reason}"
            print(f"[Remote grounding] {reason}; instruction={self.instruction}")

    def _fallback_grounding_result(
        self,
        response: GroundingResponse,
        current_frame: np.ndarray,
    ) -> GroundingResult | None:
        if not self.qwen_fallback_enabled:
            return None
        if self.consecutive_grounding_failures < self.qwen_fallback_min_failures:
            return None

        height, width = current_frame.shape[:2]
        box_width = max(16.0, width * max(0.05, min(self.qwen_fallback_width_ratio, 0.95)))
        box_height = max(16.0, height * max(0.05, min(self.qwen_fallback_height_ratio, 0.95)))
        center_x = width * 0.5
        center_y = height * max(0.05, min(self.qwen_fallback_y_center, 0.95))
        bbox = BBox(
            center_x - box_width / 2,
            center_y - box_height / 2,
            center_x + box_width / 2,
            center_y + box_height / 2,
        ).clamp(width, height)
        return GroundingResult(
            found=True,
            bbox=bbox,
            target_name="fallback target",
            confidence=0.0,
            raw_text=response.result.raw_text,
            message="fallback center box because Qwen returned no valid bbox",
            candidates=[
                GroundedCandidate(
                    target_name="fallback target",
                    bbox=bbox,
                    confidence=0.0,
                )
            ],
        )

    def _install_grounding(self, response: GroundingResponse, current_frame: np.ndarray) -> None:
        result: GroundingResult = response.result
        if not result.found or result.bbox is None:
            self.consecutive_grounding_failures += 1
            fallback = self._fallback_grounding_result(response, current_frame)
            if fallback is None:
                self.grounding_status = f"failed: {result.message}"
                print(f"[Remote grounding failed] {result.message}; raw={result.raw_text}")
                return
            result = fallback
            self.grounding_status = "fallback: Qwen returned no valid target box"
            print(
                "[Remote grounding fallback] "
                f"using center box after {self.consecutive_grounding_failures} failure(s); "
                f"raw={response.result.raw_text}"
            )
        else:
            self.consecutive_grounding_failures = 0
        fallback_used = result.message.startswith("fallback")

        if (
            self.robot_task.plan is not None
            and self.robot_task.plan.task_type == "pick_multiple"
            and result.candidates
        ):
            self._build_target_queue(result.candidates, response, current_frame)
            if not self._activate_queued_target(0, current_frame):
                self.grounding_status = "failed: could not activate first queued target"
                return
            self.robot_task.set_queue_state(len(self.target_queue), self.active_queue_index + 1)
            self.grounding_status = f"ready {len(self.target_queue)} queued targets; active=1"
            print(
                f"[Remote target queue] queued={len(self.target_queue)}, "
                f"active={self.target.logical_id if self.target else '-'}"
            )
            return

        if fallback_used:
            current_box = result.bbox.clamp(current_frame.shape[1], current_frame.shape[0])
            reference_crop = current_box.crop(current_frame)
            relocation_method = "none"
            relocation_score = 0.0
        else:
            relocation = self.relocator.relocate(
                snapshot=response.snapshot,
                snapshot_bbox=result.bbox,
                current_frame=current_frame,
            )
            current_box = relocation.bbox
            reference_crop = result.bbox.crop(response.snapshot)
            relocation_method = relocation.method
            relocation_score = relocation.score

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
            f"ready {result.target_name}; relocation={relocation_method} {relocation_score:.2f}"
        )
        if fallback_used:
            self.grounding_status = "ready fallback center box; adjust with manual ROI if needed"
        print(
            f"[Remote target installed] id={self.target.logical_id}, "
            f"name={result.target_name}, bbox={current_box.to_xywh()}, "
            f"relocation={relocation_method}, score={relocation_score:.2f}"
        )

    def _install_manual_roi(self, frame: np.ndarray, bbox: BBox) -> None:
        height, width = frame.shape[:2]
        safe_bbox = bbox.clamp(width, height)
        self.target = RuntimeTarget(
            logical_id=self._new_logical_id(),
            instruction=self.instruction,
            target_name="manual target",
            bbox=safe_bbox,
            reference_crop=safe_bbox.crop(frame),
            last_seen_at=monotonic(),
        )
        self.target_queue = []
        self.active_queue_index = -1
        self.tracker.reset()
        self.tracker.initialize(frame, safe_bbox)
        self.identity_guard.reset()
        self.identity_guard.initialize(frame, safe_bbox)
        self.contour_refiner.reset()
        self.grounding_status = "manual ROI initialized from camera client"
        self.consecutive_grounding_failures = 0

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

    def current_prediction_payload(self, frame_width: int, frame_height: int) -> dict[str, Any]:
        fallback = self.latest_measured_track
        if not fallback.visible:
            fallback = self.latest_control_track
        track, prediction_state = self.motion_predictor.predict(
            timestamp=monotonic(),
            frame_width=frame_width,
            frame_height=frame_height,
            fallback=fallback,
        )
        return {
            "track": _track_payload(track),
            "predicted_track": _track_payload(track),
            "prediction": prediction_state,
        }

    def process_frame(self, frame: np.ndarray, return_overlay: bool = True) -> tuple[dict[str, Any], np.ndarray | None]:
        self.frame_count += 1

        if self.pending_manual_bbox is not None:
            self._install_manual_roi(frame, self.pending_manual_bbox)
            self.pending_manual_bbox = None
        elif self.grounding_requested and not self.grounding_worker.busy:
            self.instruction = self.robot_task.current_grounding_instruction or self.instruction
            reason = "initial target grounding" if self.target is None else "manual re-ground"
            self._submit_grounding(frame, reason)
            self.grounding_requested = False

        try:
            response = self.grounding_worker.poll()
        except Exception as exc:
            self.grounding_status = f"error: {exc!r}"
            print(f"[Remote Qwen grounding error] {exc!r}")
            response = None
        if response is not None:
            self._install_grounding(response, frame)

        track = self._update_track(frame)
        self._maybe_auto_reground(frame, track)
        prediction_now = monotonic()
        self.motion_predictor.observe(track, prediction_now)
        control_track, prediction_state = self.motion_predictor.predict(
            timestamp=prediction_now,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            fallback=track,
        )
        self.latest_measured_track = track
        self.latest_control_track = control_track
        self.latest_prediction_state = dict(prediction_state)

        if self.grounding_worker.busy:
            self.grounding_status = (
                f"running {self.grounding_worker.elapsed_seconds:.1f}s; remote guidance STOP"
            )

        target_bbox = control_track.bbox if control_track.visible else None
        if self.grounding_worker.busy:
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

        lidar_observation = self.lidar.read()
        requested = self.direction_planner.plan(
            bbox=target_bbox,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
        )
        if self.grounding_worker.busy:
            requested = MotionGuidance("STOP", 0.0, 0.0, "Qwen grounding is in progress")

        safety = self.safety.decide(
            requested=requested,
            tracking_visible=control_track.visible and not self.grounding_worker.busy,
            yolo_obstacles=self.latest_obstacles,
            lidar=lidar_observation,
            emergency_stop=self.emergency_stop,
        )

        self.robot_task.set_queue_state(len(self.target_queue), self.active_queue_index + 1)
        task_update = self.robot_task.update(
            track=control_track,
            safety=safety,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            grounding_busy=self.grounding_worker.busy,
            has_next_target=self._has_next_queued_target(),
        )
        self.latest_robot_task = task_update.status
        self.latest_robot_command = task_update.command
        if task_update.complete_active_target:
            self._mark_active_queue_done()
            self.target = None
            self._reset_tracking(keep_reference=False)
        if task_update.activate_next_target:
            if self._activate_next_queued_target(frame):
                self.robot_task.set_queue_state(len(self.target_queue), self.active_queue_index + 1)
                self.grounding_status = (
                    f"activated queued target {self.active_queue_index + 1}/"
                    f"{len(self.target_queue)}"
                )
        if task_update.reset_target:
            self.target = None
            self.target_queue = []
            self.active_queue_index = -1
            self._reset_tracking(keep_reference=False)
        if task_update.next_grounding_instruction:
            self.instruction = task_update.next_grounding_instruction
        if task_update.request_grounding:
            self.grounding_status = (
                f"task phase {self.latest_robot_task.get('phase')}; waiting for Qwen grounding"
            )
            self.grounding_requested = True

        overlay = None
        if return_overlay:
            overlay = draw_overlay(
                frame=frame,
                track=control_track,
                obstacles=self.latest_obstacles,
                lidar=lidar_observation,
                safety=safety,
                grounding_status=self.grounding_status,
                boundary_mode=self.contour_refiner.mode,
                emergency_stop=self.emergency_stop,
                robot_task=self.latest_robot_task,
                robot_command=self.latest_robot_command,
                target_queue=_queue_payload(self.target_queue, self.active_queue_index),
            )

        now = monotonic()
        if now - self.last_log_at >= self.log_interval_seconds:
            print(
                f"[Remote runtime] track={control_track.status}; "
                f"guidance={safety.guidance.direction}; blocked={safety.blocked}; "
                f"task={self.latest_robot_task.get('phase')}; "
                f"robot={self.latest_robot_command.get('subsystem')}/"
                f"{self.latest_robot_command.get('action')}; "
                f"yolo={self.latest_obstacles.status}; lidar={lidar_observation.status}"
            )
            self.last_log_at = now

        payload: dict[str, Any] = {
            "type": "result",
            "frame_index": self.frame_count,
            "grounding_status": self.grounding_status,
            "track": _track_payload(control_track),
            "predicted_track": _track_payload(control_track),
            "measured_track": _track_payload(track),
            "prediction": prediction_state,
            "obstacles": {
                "danger": self.latest_obstacles.danger,
                "status": self.latest_obstacles.status,
                "detections": [
                    {
                        "label": detection.label,
                        "confidence": float(detection.confidence),
                        "bbox": _bbox_payload(detection.bbox),
                        "in_danger_zone": detection.in_danger_zone,
                    }
                    for detection in self.latest_obstacles.detections
                ],
            },
            "lidar": {
                "ready": lidar_observation.ready,
                "obstacle": lidar_observation.obstacle,
                "min_distance_m": lidar_observation.min_distance_m,
                "status": lidar_observation.status,
            },
            "guidance": {
                "direction": safety.guidance.direction,
                "linear": float(safety.guidance.linear),
                "angular": float(safety.guidance.angular),
                "reason": safety.guidance.reason,
            },
            "safety": {
                "blocked": safety.blocked,
                "reasons": safety.reasons,
            },
            "robot_task": self.latest_robot_task,
            "robot_command": self.latest_robot_command,
            "target_queue": _queue_payload(self.target_queue, self.active_queue_index),
            "boundary_mode": self.contour_refiner.mode,
            "emergency_stop": self.emergency_stop,
        }
        return payload, overlay
