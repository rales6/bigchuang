"""Adapt a compiled plan to a fresh, read-only binocular observation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def normalize_web_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Apply current pursuit/grasp semantics to plans from any model backend."""
    steps = deepcopy(plan.get("steps") or [])
    if not steps:
        return plan
    actions = [str(step.get("action") or "") for step in steps]
    if "locate_object" in actions and "align_for_grasp" in actions:
        # align_for_grasp already launches the Qwen->YOLO pursuit session.
        # A preceding locate_object would launch that same session twice.
        steps = [step for step in steps if step.get("action") != "locate_object"]
    for index, step in enumerate(steps, 1):
        step["id"] = f"step_{index:02d}"
        step["depends_on"] = [] if index == 1 else [f"step_{index - 1:02d}"]
        if (plan.get("intent") in {"pick", "pick_and_place"} and step.get("action") == "patrol_search"
                and "align_for_grasp" in actions and "model_grasp" in actions
                and (step.get("target") or {}).get("target_query")
                and (step.get("target") or {}).get("query")
                != (step.get("target") or {}).get("target_query")):
            # The Pi integration selects the released DeepSeek + phase-1
            # semantic search only for this explicit, full pick workflow.
            # Other map/inspection actions keep their existing executor.
            target = dict(step.get("target") or {})
            target["integration_workflow"] = "phase1_deepseek_pick_v1"
            target["requested_intent"] = "pick"
            target["pickup_instruction"] = f"去{target['query']}边上夹取一个{target['target_query']}"
            step["target"] = target
        description = str(step.get("description") or "")
        step["description"] = description.replace("四维示教凸包", "左右各自二维示教区域（含 25px 外沿）")
        if step.get("action") == "drive_relative":
            target = step.get("target") or {}
            direction = str(target.get("direction") or "")
            if direction == "backward":
                step["description"] = (
                    f"单次限幅后退 {float(target.get('distance_m') or 0):.2f}m；"
                    "后向雷达会看到车身，执行前须人工确认后方净空"
                )
            if direction in {"left", "right"}:
                label = "左转" if direction == "left" else "右转"
                step["description"] = (
                    f"雷达保护下分段前进画弧{label} {float(target.get('angle_deg') or 0):.0f}°；"
                    "净空不足时最多短回退一次，复测仍不足则停车"
                )
    result = deepcopy(plan)
    result["steps"] = steps
    return result


def adapt_plan_to_live_view(plan: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    """Skip exploration only when the requested object is confirmed in both eyes.

    The observation is advisory: the pursuit and arm endpoints must still
    re-detect the target immediately before any hardware action.
    """
    intent = str(plan.get("intent") or "")
    if intent not in {"pick", "pick_and_place", "search"} or probe.get("status") != "visible_both":
        return plan
    original = list(plan.get("steps") or [])
    if not original:
        return plan
    if intent in {"pick", "pick_and_place"}:
        anchored = next((step for step in original
                         if step.get("action") == "patrol_search" and
                         (step.get("target") or {}).get("integration_workflow")
                         == "phase1_deepseek_pick_v1"), None)
        if anchored is not None:
            anchor = str((anchored.get("target") or {}).get("query") or "").strip()
            object_query = str((anchored.get("target") or {}).get("target_query") or "").strip()
            if anchor and object_query and anchor != object_query:
                # Two cameras can see the requested object, but the current-view probe
                # cannot establish that it is beside the named cabinet.
                result = deepcopy(plan)
                result["live_view_probe"] = {
                    "status": "visible_both",
                    "skip_suppressed": "named_anchor_not_geographically_verified",
                    "observed_at_ms": probe.get("observed_at_ms"),
                }
                return result
    inside = probe.get("inside_grasp_region") is True
    result = deepcopy(plan)
    steps = deepcopy(original)
    grasp_index = next((i for i, item in enumerate(steps) if item.get("action") == "model_grasp"), -1)
    if intent in {"pick", "pick_and_place"} and grasp_index < 0:
        return plan
    if intent == "search":
        kept = [step for step in steps if step.get("action") in ({"report_result"} if inside else {"align_for_grasp", "report_result"})]
    else:
        # Only discard the *pre-grasp* map/search/navigation actions.  The
        # destination navigation after verify_grasp must remain intact.
        tail = steps[grasp_index:]
        if inside:
            target = dict(steps[grasp_index].get("target") or {})
            target["allow_chassis_motion"] = False
            tail[0]["target"] = target
            tail[0]["description"] = "双目目标已在左右各自二维示教区域（含 25px 外沿）；重新核验后原地夹取"
            kept = [
                {"action": "prepare_stationary_grasp", "description": "检查双目与机械臂就绪，底盘保持停止",
                 "target": target, "sensors": ["camera"], "success_condition": "左右画面新鲜且底盘停止",
                 "on_failure": "safe_stop", "max_retries": 1, "condition": "always"},
                *tail,
            ]
        else:
            alignment = next((step for step in steps[:grasp_index] if step.get("action") == "align_for_grasp"), None)
            if alignment is None:
                return plan
            kept = [alignment, *tail]
    if not kept:
        return plan
    for index, step in enumerate(kept, 1):
        step["id"] = f"step_{index:02d}"
        step["depends_on"] = [] if index == 1 else [f"step_{index - 1:02d}"]
    result["steps"] = kept
    source_target = original[grasp_index].get("target") if grasp_index >= 0 else original[-1].get("target")
    target_query = str((source_target or {}).get("query") or "目标物")
    if intent == "search":
        result["summary"] = f"当前双目已看到“{target_query}”，" + ("无需巡航，报告当前位置。" if inside else "直接视觉追踪靠近，无需先巡航。")
    else:
        result["summary"] = f"当前双目已看到“{target_query}”，" + ("位于抓取区内，直接原地夹取。" if inside else "先视觉追踪靠近，再夹取。")
        if intent == "pick_and_place":
            result["summary"] += "夹取实物核验后再执行搬运。"
    result["source"] = f"{plan.get('source') or 'compiled'}+live_stereo"
    result["live_view_probe"] = {
        "status": "visible_both", "inside_grasp_region": inside,
        "ble_ready": probe.get("ble_ready"),
        "grasp_margin_px": probe.get("grasp_margin_px"),
        "observed_at_ms": probe.get("observed_at_ms"),
    }
    return result
