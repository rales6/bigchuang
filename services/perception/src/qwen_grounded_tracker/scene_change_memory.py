from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any


SCHEMA = "scene_change_overlay.v1"
VALID_CHANGE_TYPES = {"moved", "missing", "unexpected", "appearance_changed", "rechecked"}


def _xy(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return [x, y] if math.isfinite(x) and math.isfinite(y) else None


class SceneChangeMemory:
    """Persistent overlay for suspicious places; never mutates the baseline map.

    A suspicion is intentionally sticky.  Rechecking updates its history and
    last-seen state, but future visits to the same place still require a camera
    observation because movable objects may change again.
    """

    def __init__(self, path: str | Path | None = None, *, map_id: str = "unknown") -> None:
        self.path = Path(path) if path else None
        self.data: dict[str, Any] = {
            "schema": SCHEMA,
            "map_id": str(map_id or "unknown"),
            "baseline_map_mutated": False,
            "suspicious_locations": [],
        }
        if self.path and self.path.exists():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if loaded.get("schema") != SCHEMA:
                raise ValueError("Unsupported scene-change overlay schema")
            self.data = loaded

    @property
    def locations(self) -> list[dict[str, Any]]:
        return list(self.data.get("suspicious_locations") or [])

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.data, ensure_ascii=False, indent=2)
        fd, temporary = tempfile.mkstemp(
            prefix=self.path.name + ".", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def record(
        self,
        *,
        change_type: str,
        expected_item_id: str | None,
        expected_query: str,
        map_xy: Any = None,
        grid_xy: Any = None,
        observed_item_id: str | None = None,
        confidence: float = 0.0,
        evidence: dict[str, Any] | None = None,
        timestamp_s: float | None = None,
    ) -> dict[str, Any]:
        change_type = str(change_type).strip().lower()
        if change_type not in VALID_CHANGE_TYPES:
            raise ValueError(f"Unsupported scene change type: {change_type}")
        expected_query = str(expected_query or "").strip()
        expected_item_id = str(expected_item_id or "") or None
        observed_item_id = str(observed_item_id or "") or None
        map_point = _xy(map_xy)
        grid_point = _xy(grid_xy)
        if map_point is None and grid_point is None:
            raise ValueError("A suspicious location requires map_xy or grid_xy")
        now = float(time.time() if timestamp_s is None else timestamp_s)

        def same_place(item: dict[str, Any]) -> bool:
            if expected_item_id and item.get("expected_item_id") == expected_item_id:
                return True
            left = _xy(item.get("map_xy")) or _xy(item.get("grid_xy"))
            right = map_point or grid_point
            return bool(
                left and right and math.hypot(left[0] - right[0], left[1] - right[1]) <= 0.60
                and expected_query
                and expected_query.lower() in str(item.get("expected_query") or "").lower()
            )

        locations = self.data.setdefault("suspicious_locations", [])
        location = next((item for item in locations if same_place(item)), None)
        if location is None:
            location = {
                "id": f"suspicious_{len(locations) + 1:04d}",
                "expected_item_id": expected_item_id,
                "expected_query": expected_query,
                "map_xy": map_point,
                "grid_xy": grid_point,
                "first_detected_s": now,
                "requires_recheck_on_every_visit": True,
                "history": [],
            }
            locations.append(location)
        if map_point is not None:
            location["map_xy"] = map_point
        if grid_point is not None:
            location["grid_xy"] = grid_point
        location.update(
            last_change_type=change_type,
            last_observed_item_id=observed_item_id,
            last_confidence=max(0.0, min(1.0, float(confidence))),
            last_checked_s=now,
            requires_recheck_on_every_visit=True,
        )
        location.setdefault("history", []).append({
            "timestamp_s": now,
            "change_type": change_type,
            "observed_item_id": observed_item_id,
            "confidence": location["last_confidence"],
            "evidence": dict(evidence or {}),
        })
        location["history"] = location["history"][-50:]
        self.data["updated_s"] = now
        self.data["baseline_map_mutated"] = False
        self._save()
        return copy.deepcopy(location)

    def overlay(self, semantic_map: dict[str, Any] | None) -> dict[str, Any]:
        result = copy.deepcopy(semantic_map if isinstance(semantic_map, dict) else {})
        result["scene_change_overlay_schema"] = SCHEMA
        merged = {
            str(item.get("id")): copy.deepcopy(item)
            for item in (result.get("suspicious_locations") or [])
            if isinstance(item, dict) and item.get("id")
        }
        for item in self.locations:
            merged[str(item.get("id"))] = copy.deepcopy(item)
        result["suspicious_locations"] = list(merged.values())
        result["baseline_map_mutated"] = False
        return result

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)
