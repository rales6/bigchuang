"""Inject Qwen ranking between safe route generation and frozen execution.

The frozen car1.0 planner remains responsible for candidate coordinates,
occupancy clearance, visibility rays and A* paths.  Qwen may only return a
ranking of those numbered candidates.  The wrapper then recomputes every path
in the selected order before the original mission consumes it.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
from typing import Any
from urllib import request as urlrequest


def _argument(name: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _semantic_context(semantic: dict[str, Any], anchor_id: str) -> dict[str, Any]:
    objects = semantic.get("objects") if isinstance(semantic.get("objects"), list) else []
    related_ids = {anchor_id}
    relations = []
    for item in semantic.get("scene_relations") or []:
        if not isinstance(item, dict):
            continue
        if anchor_id and anchor_id in {str(item.get("subject_id") or ""), str(item.get("object_id") or "")}:
            relations.append(item)
            related_ids.update({str(item.get("subject_id") or ""), str(item.get("object_id") or "")})
    compact_objects = [
        {
            "id": item.get("id"), "name": item.get("name"),
            "category": item.get("category"), "map_xy": item.get("map_xy"),
            "behind_glass": item.get("behind_glass", False),
        }
        for item in objects
        if isinstance(item, dict) and (not anchor_id or str(item.get("id") or "") in related_ids)
    ]
    return {"objects": compact_objects[:12], "scene_relations": relations[:12]}


def _candidate_payload(known_map: Any, start_xy: tuple[float, float], route: dict[str, Any], route_request: dict[str, Any]) -> list[dict[str, Any]]:
    anchor = route_request.get("anchor_xy_m")
    anchor_xy = (
        (float(anchor[0]), float(anchor[1]))
        if isinstance(anchor, (list, tuple)) and len(anchor) >= 2 else None
    )
    candidates = []
    for index, view in enumerate(route.get("views") or []):
        point = (float(view["view_xy_m"][0]), float(view["view_xy_m"][1]))
        path = known_map.path(start_xy, point) or []
        length = sum(
            math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
            for a, b in zip([start_xy] + path[:-1], path)
        )
        candidates.append({
            "index": index,
            "view_xy_m": [round(point[0], 4), round(point[1], 4)],
            "path_length_from_start_m": round(length, 3),
            "distance_to_anchor_m": 0.0 if anchor_xy is None else round(math.hypot(point[0] - anchor_xy[0], point[1] - anchor_xy[1]), 3),
            "look_at_yaw_deg": 0.0 if anchor_xy is None else round(math.degrees(math.atan2(anchor_xy[1] - point[1], anchor_xy[0] - point[0])), 1),
        })
    return candidates


def _rebuild_in_ranked_order(known_map: Any, start_xy: tuple[float, float], route: dict[str, Any], ranked: list[int]) -> dict[str, Any]:
    original = list(route.get("views") or [])
    valid = []
    for value in ranked:
        if isinstance(value, int) and 0 <= value < len(original) and value not in valid:
            valid.append(value)
    valid.extend(index for index in range(len(original)) if index not in valid)
    current = start_xy
    ordered = []
    applied = []
    for old_index in valid:
        point = tuple(float(value) for value in original[old_index]["view_xy_m"])
        path = known_map.path(current, point)
        if not path:
            continue
        length = sum(
            math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
            for a, b in zip([current] + path[:-1], path)
        )
        value = dict(original[old_index])
        value["path_xy_m"] = [[round(float(x), 4), round(float(y), 4)] for x, y in path]
        value["path_length_m"] = round(length, 3)
        value["ai_original_index"] = old_index
        ordered.append(value)
        applied.append(old_index)
        current = point
    result = dict(route)
    result["views"] = ordered
    result["ai_ranked_original_indices"] = applied
    return result


class ObservationRanker:
    def __init__(self, decision: dict[str, Any], semantic: dict[str, Any]) -> None:
        self.decision = decision
        self.semantic = semantic
        self.url = os.environ.get("SEARCH_3090_URL", "http://192.0.2.20:8010").rstrip("/")
        self.cache: dict[str, dict[str, Any]] = {}

    def rank(self, known_map: Any, start_xy: tuple[float, float], route_request: dict[str, Any], route: dict[str, Any]) -> dict[str, Any]:
        candidates = _candidate_payload(known_map, start_xy, route, route_request)
        if len(candidates) <= 1:
            route["ai_observation_selection"] = {"source": "single_safe_candidate", "ranked_indices": list(range(len(candidates)))}
            return route
        cache_key = json.dumps({"start": [round(start_xy[0], 2), round(start_xy[1], 2)], "request": route_request, "candidates": candidates}, sort_keys=True)
        selection = self.cache.get(cache_key)
        if selection is None:
            anchor = self.decision.get("anchor") if isinstance(self.decision.get("anchor"), dict) else {}
            payload = json.dumps({
                "instruction": self.decision.get("instruction"),
                "target": self.decision.get("target"),
                "anchor": anchor,
                "semantic_context": _semantic_context(self.semantic, str(anchor.get("landmark_id") or "")),
                "candidates": candidates,
            }, ensure_ascii=False).encode("utf-8")
            request = urlrequest.Request(
                f"{self.url}/mission/select-observation", data=payload, method="POST",
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            with urlrequest.urlopen(request, timeout=90) as response:
                selection = json.loads(response.read().decode("utf-8"))
            if not isinstance(selection, dict) or not selection.get("ok"):
                raise RuntimeError(str(selection.get("error") if isinstance(selection, dict) else selection))
            self.cache[cache_key] = selection
        if not selection.get("confirmed"):
            route["ai_observation_selection"] = {
                "source": "qwen_unconfirmed_safe_order",
                "confirmed": False,
                "confidence": selection.get("confidence"),
                "reason": selection.get("reason"),
                "proposed_indices": selection.get("ranked_indices") or [],
                "ranked_indices": list(range(len(route.get("views") or []))),
            }
            print("AI 观察点未达到确认阈值，保持安全顺序：" + json.dumps(route["ai_observation_selection"], ensure_ascii=False), flush=True)
            return route
        result = _rebuild_in_ranked_order(known_map, start_xy, route, list(selection.get("ranked_indices") or []))
        result["ai_observation_selection"] = {
            "source": selection.get("source"),
            "confirmed": True,
            "confidence": selection.get("confidence"),
            "reason": selection.get("reason"),
            "ranked_indices": result.get("ai_ranked_original_indices"),
            "candidate_reasons": selection.get("candidate_reasons") or {},
        }
        print("AI 观察点确认：" + json.dumps(result["ai_observation_selection"], ensure_ascii=False), flush=True)
        return result


def main() -> None:
    decision = _read_json(Path(_argument("--decision")))
    release = Path(os.environ.get(
        "FINAL_WORK_SEMANTIC_RELEASE",
        "/home/robot/dachuang/car1.0/ble_scene_17_relationship_20260919",
    ))
    semantic = _read_json(release / "semantic_runtime.json")
    ranker = ObservationRanker(decision, semantic)

    from raspberry_pi import object_search_mission as mission
    from raspberry_pi import object_search_simulation as simulation

    original = mission.build_search_route

    def build_with_ai(known_map: Any, start_xy: tuple[float, float], route_request: dict[str, Any]) -> dict[str, Any]:
        route = original(known_map, start_xy, route_request)
        try:
            return ranker.rank(known_map, start_xy, route_request, route)
        except Exception as exc:
            route["ai_observation_selection"] = {
                "source": "safe_deterministic_fallback",
                "error": f"{type(exc).__name__}: {exc}",
                "ranked_indices": list(range(len(route.get("views") or []))),
            }
            print("AI 观察点确认失败，保持安全确定性顺序：" + str(exc), flush=True)
            return route

    mission.build_search_route = build_with_ai
    simulation.build_search_route = build_with_ai
    mission.main()


if __name__ == "__main__":
    main()
