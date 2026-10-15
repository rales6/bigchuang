from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol


ALLOWED_ACTIONS = {"navigate_next_view", "replan", "report_success", "report_failure"}
TERMINAL_ACTIONS = {"report_success", "report_failure"}


class TextBackend(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class MissionDirective:
    action: str
    reason: str
    next_view_index: int | None
    terminal: bool
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "object_search_mission_directive.v1",
            "action": self.action,
            "reason": self.reason,
            "next_view_index": self.next_view_index,
            "terminal": self.terminal,
            "source": self.source,
        }


def _json_object(raw: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Qwen response does not contain a JSON object")


def build_next_step_prompt(state: dict[str, Any]) -> str:
    compact = {
        "instruction": state.get("instruction"),
        "target": state.get("target"),
        "anchor": state.get("anchor"),
        "current_view_index": state.get("current_view_index"),
        "total_views": state.get("total_views"),
        "remaining_view_indices": state.get("remaining_view_indices", []),
        "latest_event": state.get("latest_event", {}),
        "detections": state.get("detections", []),
        "attempts": state.get("attempts", 0),
    }
    return f"""
你是室内寻物任务监督器。传统算法已经负责雷达定位、临时障碍物、A*路径和底盘控制；
你不能输出坐标、速度、角速度、电机命令或机械臂命令。你只决定下一种语义动作。

当前任务状态：{json.dumps(compact, ensure_ascii=False, separators=(',', ':'))}

只允许输出以下 action：
- report_success：detections 中已经有满足目标类别和颜色的 confirmed_match。
- navigate_next_view：目标还没找到且仍有未访问观察点。
- replan：最新事件是 temporary_obstacle 或 path_invalid，需要传统算法重算剩余路线。
- report_failure：所有观察点耗尽、定位不可恢复、感知服务不可恢复或超时。

禁止因为“可能看到了”而成功；只有 confirmed_match=true 才能 report_success。
严格只输出JSON：
{{"action":"navigate_next_view","reason":"简短原因","next_view_index":2}}
""".strip()


class MissionSupervisor:
    """Qwen proposes task progress; deterministic guards retain authority."""

    def __init__(self, backend: TextBackend | None = None) -> None:
        self.backend = backend

    @staticmethod
    def _guarded(state: dict[str, Any], proposal: dict[str, Any] | None, source: str) -> MissionDirective:
        detections = state.get("detections") or []
        confirmed = any(
            isinstance(item, dict) and bool(item.get("confirmed_match"))
            for item in detections
        )
        event = str((state.get("latest_event") or {}).get("type") or "")
        remaining = [int(value) for value in state.get("remaining_view_indices") or []]
        fatal = event in {"localization_failed", "perception_unavailable", "mission_timeout"}
        exhausted = not remaining

        # These conditions are authoritative. Qwen cannot override them.
        if confirmed:
            return MissionDirective("report_success", "目标类别和颜色已由视觉过滤确认", None, True, source)
        if fatal:
            return MissionDirective("report_failure", f"不可恢复事件：{event}", None, True, source)
        if event in {"temporary_obstacle", "path_invalid", "navigation_no_progress"}:
            return MissionDirective("replan", "地图会话层发生变化，需要重新规划", None, False, source)
        if exhausted:
            return MissionDirective("report_failure", "全部可达观察点已检查仍未发现目标", None, True, source)

        requested = str((proposal or {}).get("action") or "navigate_next_view")
        if requested not in ALLOWED_ACTIONS or requested in TERMINAL_ACTIONS:
            requested = "navigate_next_view"
        requested_index = (proposal or {}).get("next_view_index")
        try:
            requested_index = int(requested_index)
        except (TypeError, ValueError):
            requested_index = remaining[0]
        if requested_index not in remaining:
            requested_index = remaining[0]
        return MissionDirective(
            requested,
            str((proposal or {}).get("reason") or "继续检查下一个可达观察点"),
            requested_index if requested == "navigate_next_view" else None,
            False,
            source,
        )

    def next(self, state: dict[str, Any]) -> MissionDirective:
        if self.backend is None:
            return self._guarded(state, None, "deterministic")
        try:
            proposal = _json_object(self.backend.generate(build_next_step_prompt(state)))
            return self._guarded(state, proposal, "qwen_guarded")
        except Exception as exc:
            directive = self._guarded(state, None, "deterministic_fallback")
            return MissionDirective(
                directive.action,
                f"Qwen阶段判断不可用，采用安全状态机：{exc}",
                directive.next_view_index,
                directive.terminal,
                directive.source,
            )
