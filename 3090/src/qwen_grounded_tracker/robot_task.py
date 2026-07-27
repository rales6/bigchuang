from __future__ import annotations

import re
from dataclasses import dataclass
from time import monotonic
from typing import Any

from qwen_grounded_tracker.domain import SafetyDecision, TrackObservation


PICK_AND_DISPOSE_PHASES = [
    "find_target",
    "approach_target",
    "grasp_target",
    "verify_grasp",
    "find_destination",
    "approach_destination",
    "release_target",
    "verify_done",
    "done",
]

PICK_MULTIPLE_PHASES = [
    "find_target",
    "approach_target",
    "grasp_target",
    "verify_grasp",
    "next_target",
    "done",
]


@dataclass
class TaskPlan:
    task_type: str
    original_instruction: str
    target_query: str
    destination_query: str = ""
    requested_count: int = 1


@dataclass
class TaskUpdate:
    status: dict[str, Any]
    command: dict[str, Any]
    reset_target: bool = False
    request_grounding: bool = False
    next_grounding_instruction: str | None = None
    complete_active_target: bool = False
    activate_next_target: bool = False


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    folded = text.lower()
    return any(word in folded for word in words)


def _requested_count(text: str) -> int:
    folded = text.lower()
    if _contains_any(folded, ("全部", "所有", "all", "every")):
        return 6
    for value in range(2, 10):
        if re.search(rf"\b{value}\b", folded):
            return value
    chinese_counts = {
        "两个": 2,
        "二个": 2,
        "两只": 2,
        "三个": 3,
        "三只": 3,
        "四个": 4,
        "五个": 5,
    }
    for word, value in chinese_counts.items():
        if word in text:
            return value
    return 1


def parse_task_plan(instruction: str) -> TaskPlan:
    text = instruction.strip()
    if not text:
        raise ValueError("Instruction must not be empty")

    count = _requested_count(text)
    has_trash = _contains_any(text, ("垃圾", "trash", "waste", "rubbish"))
    has_bin = _contains_any(text, ("垃圾桶", "trash bin", "dustbin", "bin"))
    has_bottle = _contains_any(text, ("水瓶", "瓶子", "bottle", "water bottle"))
    has_pick = _contains_any(text, ("捡", "拾", "拿起", "抓", "pick", "grab"))
    has_dispose = _contains_any(text, ("扔", "丢", "放进", "投放", "dispose", "throw", "put"))

    if count > 1 and (has_bottle or has_trash) and has_pick:
        target_name = "水瓶" if has_bottle else "垃圾"
        return TaskPlan(
            task_type="pick_multiple",
            original_instruction=text,
            target_query=f"找到画面中最多 {count} 个需要依次抓取的{target_name}，按从左到右返回多个框",
            requested_count=count,
        )

    if has_trash and (has_bin or has_dispose or has_pick):
        return TaskPlan(
            task_type="pick_and_dispose",
            original_instruction=text,
            target_query="找到需要捡起的垃圾，并框选最适合抓取的那个垃圾",
            destination_query="找到垃圾桶或可投放垃圾的容器开口，并框选它",
            requested_count=1,
        )

    return TaskPlan(task_type="visual_track", original_instruction=text, target_query=text)


class RobotTaskController:
    """把语言任务拆成串行阶段；真实机器人只需要订阅输出命令。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.center_tolerance = float(config.get("center_tolerance", 0.08))
        self.target_reached_area_ratio = float(config.get("target_reached_area_ratio", 0.13))
        self.destination_reached_area_ratio = float(
            config.get("destination_reached_area_ratio", 0.16)
        )
        self.grasp_seconds = float(config.get("grasp_seconds", 1.2))
        self.verify_grasp_seconds = float(config.get("verify_grasp_seconds", 0.6))
        self.release_seconds = float(config.get("release_seconds", 1.0))
        self.verify_done_seconds = float(config.get("verify_done_seconds", 0.6))
        self.plan: TaskPlan | None = None
        self.phase = "idle"
        self.phase_started_at = monotonic()
        self.completed = False
        self.completed_targets = 0
        self.queued_targets = 0
        self.active_target_index = 0

    def load_instruction(self, instruction: str) -> TaskPlan:
        self.plan = parse_task_plan(instruction)
        self.completed = False
        self.completed_targets = 0
        self.queued_targets = 0
        self.active_target_index = 0
        phase = (
            "find_target"
            if self.plan.task_type in {"pick_and_dispose", "pick_multiple"}
            else "track_target"
        )
        self._set_phase(phase)
        return self.plan

    def reset(self) -> None:
        self.plan = None
        self.phase = "idle"
        self.completed = False
        self.completed_targets = 0
        self.queued_targets = 0
        self.active_target_index = 0
        self.phase_started_at = monotonic()

    def set_queue_state(self, queued_targets: int, active_target_index: int) -> None:
        self.queued_targets = max(0, int(queued_targets))
        self.active_target_index = max(0, int(active_target_index))

    @property
    def active(self) -> bool:
        return self.plan is not None and self.plan.task_type in {
            "pick_and_dispose",
            "pick_multiple",
        }

    @property
    def current_grounding_instruction(self) -> str:
        if self.plan is None:
            return ""
        if self.plan.task_type == "pick_and_dispose" and self.phase in {
            "find_destination",
            "approach_destination",
            "release_target",
            "verify_done",
            "done",
        }:
            return self.plan.destination_query
        return self.plan.target_query

    def _set_phase(self, phase: str) -> None:
        if phase != self.phase:
            self.phase = phase
            self.phase_started_at = monotonic()

    def _elapsed(self) -> float:
        return monotonic() - self.phase_started_at

    def _target_ready(
        self,
        track: TrackObservation,
        frame_width: int,
        frame_height: int,
        area_ratio: float,
    ) -> bool:
        if not track.visible or track.bbox is None:
            return False
        center_x, _ = track.bbox.normalized_center(frame_width, frame_height)
        centered = abs(center_x - 0.5) <= self.center_tolerance
        close_enough = track.bbox.area_ratio(frame_width, frame_height) >= area_ratio
        return centered and close_enough

    def _base_status(self) -> dict[str, Any]:
        if self.plan is None:
            return {
                "active": False,
                "task_type": "none",
                "phase": "idle",
                "phase_label": "No robot task",
                "current_grounding_instruction": "",
                "steps": [],
                "completed": False,
                "requested_count": 0,
                "completed_targets": 0,
                "queued_targets": 0,
                "active_target_index": 0,
            }
        steps = PICK_MULTIPLE_PHASES if self.plan.task_type == "pick_multiple" else PICK_AND_DISPOSE_PHASES
        if self.plan.task_type == "visual_track":
            steps = ["track_target"]
        return {
            "active": self.active,
            "task_type": self.plan.task_type,
            "phase": self.phase,
            "phase_label": self.phase.replace("_", " "),
            "current_grounding_instruction": self.current_grounding_instruction,
            "target_query": self.plan.target_query,
            "destination_query": self.plan.destination_query,
            "steps": steps,
            "completed": self.completed,
            "requested_count": self.plan.requested_count,
            "completed_targets": self.completed_targets,
            "queued_targets": self.queued_targets,
            "active_target_index": self.active_target_index,
            "elapsed_seconds": round(self._elapsed(), 2),
        }

    def _command(
        self,
        subsystem: str,
        action: str,
        reason: str,
        safety: SafetyDecision | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": "simulated",
            "subsystem": subsystem,
            "action": action,
            "reason": reason,
        }
        if safety is not None:
            payload["direction"] = safety.guidance.direction
            payload["linear"] = float(safety.guidance.linear)
            payload["angular"] = float(safety.guidance.angular)
            payload["blocked"] = bool(safety.blocked)
        return payload

    def update(
        self,
        track: TrackObservation,
        safety: SafetyDecision,
        frame_width: int,
        frame_height: int,
        grounding_busy: bool,
        has_next_target: bool = False,
    ) -> TaskUpdate:
        if self.plan is None:
            return TaskUpdate(
                status=self._base_status(),
                command=self._command("vision", "idle", "No active task"),
            )

        if self.plan.task_type == "visual_track":
            return TaskUpdate(
                status=self._base_status(),
                command=self._command("chassis", "track_target", "Single target tracking", safety),
            )

        if self.plan.task_type == "pick_multiple":
            return self._update_pick_multiple(track, safety, frame_width, frame_height, grounding_busy, has_next_target)

        return self._update_pick_and_dispose(track, safety, frame_width, frame_height, grounding_busy)

    def _update_pick_multiple(
        self,
        track: TrackObservation,
        safety: SafetyDecision,
        frame_width: int,
        frame_height: int,
        grounding_busy: bool,
        has_next_target: bool,
    ) -> TaskUpdate:
        command = self._command("chassis", "stop", "Waiting")
        complete_active = False
        activate_next = False

        if self.phase == "find_target":
            command = self._command("vision", "ground_targets", "Locating multiple targets with Qwen")
            if track.visible and not grounding_busy:
                self._set_phase("approach_target")

        elif self.phase == "approach_target":
            command = self._command("chassis", "approach_target", "Approach current queued target", safety)
            if self._target_ready(track, frame_width, frame_height, self.target_reached_area_ratio):
                self._set_phase("grasp_target")
                command = self._command("arm", "close_gripper", "Current target is close enough")

        elif self.phase == "grasp_target":
            command = self._command("arm", "close_gripper", "Simulated grasp in progress")
            if self._elapsed() >= self.grasp_seconds:
                self._set_phase("verify_grasp")

        elif self.phase == "verify_grasp":
            command = self._command("arm", "hold", "Verifying simulated grasp")
            if self._elapsed() >= self.verify_grasp_seconds:
                self.completed_targets += 1
                complete_active = True
                if has_next_target and self.completed_targets < self.plan.requested_count:
                    self._set_phase("next_target")
                    activate_next = True
                    command = self._command("vision", "activate_next_target", "Switching to next queued target")
                else:
                    self._set_phase("done")
                    self.completed = True
                    command = self._command("all", "stop", "All queued targets completed")

        elif self.phase == "next_target":
            command = self._command("vision", "activate_next_target", "Preparing next queued target")
            self._set_phase("approach_target")

        elif self.phase == "done":
            command = self._command("all", "stop", "Task completed")

        return TaskUpdate(
            status=self._base_status(),
            command=command,
            complete_active_target=complete_active,
            activate_next_target=activate_next,
        )

    def _update_pick_and_dispose(
        self,
        track: TrackObservation,
        safety: SafetyDecision,
        frame_width: int,
        frame_height: int,
        grounding_busy: bool,
    ) -> TaskUpdate:
        reset_target = False
        request_grounding = False
        next_instruction: str | None = None
        command = self._command("chassis", "stop", "Waiting")

        if self.phase == "find_target":
            command = self._command("vision", "ground_target", "Locating trash with Qwen")
            if track.visible and not grounding_busy:
                self._set_phase("approach_target")

        elif self.phase == "approach_target":
            command = self._command("chassis", "approach_target", "Center and approach trash", safety)
            if self._target_ready(track, frame_width, frame_height, self.target_reached_area_ratio):
                self._set_phase("grasp_target")
                command = self._command("arm", "close_gripper", "Trash is centered and close enough")

        elif self.phase == "grasp_target":
            command = self._command("arm", "close_gripper", "Simulated grasp in progress")
            if self._elapsed() >= self.grasp_seconds:
                self._set_phase("verify_grasp")

        elif self.phase == "verify_grasp":
            command = self._command("arm", "hold", "Verifying simulated grasp")
            if self._elapsed() >= self.verify_grasp_seconds:
                self._set_phase("find_destination")
                reset_target = True
                request_grounding = True
                next_instruction = self.current_grounding_instruction
                command = self._command("vision", "ground_destination", "Switching to trash bin search")

        elif self.phase == "find_destination":
            command = self._command("vision", "ground_destination", "Locating trash bin with Qwen")
            if track.visible and not grounding_busy:
                self._set_phase("approach_destination")

        elif self.phase == "approach_destination":
            command = self._command("chassis", "approach_destination", "Center and approach trash bin", safety)
            if self._target_ready(track, frame_width, frame_height, self.destination_reached_area_ratio):
                self._set_phase("release_target")
                command = self._command("arm", "open_gripper", "Trash bin is centered and close enough")

        elif self.phase == "release_target":
            command = self._command("arm", "open_gripper", "Simulated release in progress")
            if self._elapsed() >= self.release_seconds:
                self._set_phase("verify_done")

        elif self.phase == "verify_done":
            command = self._command("vision", "verify_done", "Checking task completion")
            if self._elapsed() >= self.verify_done_seconds:
                self._set_phase("done")
                self.completed = True

        elif self.phase == "done":
            command = self._command("all", "stop", "Task completed")

        return TaskUpdate(
            status=self._base_status(),
            command=command,
            reset_target=reset_target,
            request_grounding=request_grounding,
            next_grounding_instruction=next_instruction,
        )
