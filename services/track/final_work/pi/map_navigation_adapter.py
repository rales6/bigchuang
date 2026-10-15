"""Read-only adapter for the frozen ``car1.0`` semantic search stack.

The adapter deliberately keeps all integration code inside ``final_work``.
It discovers published semantic artefacts, exposes a fixed map-layer catalog,
asks the shared 8010 Qwen process for a search contract, and writes only the
generated decision file used by the existing ``search_object.sh`` entrypoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any
from urllib import request as urlrequest
import uuid


CAR1_ROOT = Path(os.environ.get("FINAL_WORK_CAR1_ROOT", "/home/dachuang/dachuang/car1.0"))
TRACK_ROOT = Path(os.environ.get("FINAL_WORK_TRACK_ROOT", "/home/dachuang/dachuang/track"))
SEMANTIC_RELEASE = Path(os.environ.get(
    "FINAL_WORK_SEMANTIC_RELEASE",
    str(CAR1_ROOT / "ble_scene_17_relationship_20260919"),
))
SEARCH_API_URL = os.environ.get("FINAL_WORK_SEARCH_API_URL", "http://192.168.55.33:8010").rstrip("/")
RUNTIME_ROOT = Path(os.environ.get(
    "FINAL_WORK_NAVIGATION_DATA_DIR",
    str(TRACK_ROOT / "final_work" / "data" / "navigation"),
))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _entity_count(value: dict[str, Any]) -> int:
    entities = value.get("entities")
    if isinstance(entities, dict):
        return len(entities)
    for key in ("landmarks", "objects"):
        items = value.get(key)
        if isinstance(items, list):
            return len(items)
    return 0


class MapNavigationAdapter:
    """Discover and invoke stable, frozen map/navigation interfaces."""

    def __init__(self) -> None:
        self.car1_root = CAR1_ROOT
        self.release_root = SEMANTIC_RELEASE
        self.runtime_root = RUNTIME_ROOT
        self.search_api_url = SEARCH_API_URL

    def _semantic_candidates(self) -> list[tuple[str, Path]]:
        return [
            ("stable_release", self.release_root / "semantic_runtime.json"),
            ("compatibility_fallback", self.car1_root / "maps" / "ble_scene_17_semantics_qwen.json"),
            ("legacy_fallback", self.car1_root / "maps" / "ble_scene_17_semantics.json"),
        ]

    def semantic_map(self) -> dict[str, Any]:
        for status, path in self._semantic_candidates():
            value = _read_json(path)
            if value is None:
                continue
            result = dict(value)
            result["_status"] = status
            result["_source_path"] = str(path)
            result["_entity_count"] = _entity_count(value)
            result["_release_expected"] = str(self.release_root)
            result["_release_available"] = self.release_root.is_dir()
            if status != "stable_release":
                result["_warning"] = (
                    "正式关系语义发布包不存在；"
                    "当前只读使用 ble_scene_17_semantics_qwen.json 兼容数据。"
                )
            return result
        return {
            "objects": [],
            "landmarks": [],
            "_status": "missing",
            "_source_path": None,
            "_entity_count": 0,
            "_release_expected": str(self.release_root),
            "_release_available": False,
            "_warning": "未找到可用的语义地图 JSON。",
        }

    def _layer_specs(self) -> list[tuple[str, str, str, Path]]:
        release_base = self.release_root / "ble_scene_17.pgm"
        base_path = release_base if release_base.is_file() else self.car1_root / "maps" / "ble_scene_17.pgm"
        specs = [
            ("base", "正式雷达占据栅格" if release_base.is_file() else "雷达占据栅格（兼容）", "base", base_path),
            ("relationship_release", "正式关系语义地图", "semantic", self.release_root / "semantic_relationship_map_20260919.png"),
            ("observation_release", "成功巡检观测证据", "inspection", self.release_root / "semantic_observation_20260917_190706.png"),
            ("qwen_reannotation", "Qwen 重标注审核图", "review", self.release_root / "qwen_reannotation_20260918.png"),
            ("semantic_trace", "旧自动投影（仅追溯）", "trace", self.release_root / "semantic_combined_20260919.png"),
            ("semantic_compat", "当前语义标注（兼容）", "semantic", self.car1_root / "maps" / "ble_scene_17_semantic_qwen.png"),
            ("navigation_live", "当前导航与搜索路线", "navigation", self.car1_root / "search_runs" / "live_map.png"),
        ]
        # Release filenames may differ slightly between frozen deliveries. Add
        # only files inside the named release directory and never expose paths
        # received from the network.
        if self.release_root.is_dir():
            known = {path.resolve() for *_, path in specs if path.exists()}
            for index, path in enumerate(sorted(self.release_root.glob("*.png"))):
                if path.resolve() in known:
                    continue
                specs.append((f"release_extra_{index}", path.stem, "release", path))
        return specs

    def layers(self) -> dict[str, Any]:
        semantic = self.semantic_map()
        layers = []
        for layer_id, label, kind, path in self._layer_specs():
            if not path.is_file():
                continue
            layers.append({
                "id": layer_id,
                "label": label,
                "kind": kind,
                "source_path": str(path),
                "updated_at": int(path.stat().st_mtime * 1000),
                "is_default": layer_id == ("relationship_release" if (self.release_root / "semantic_relationship_map_20260919.png").is_file() else "semantic_compat"),
            })
        return {
            "ok": True,
            "layers": layers,
            "semantic": {
                "status": semantic.get("_status"),
                "source_path": semantic.get("_source_path"),
                "entity_count": semantic.get("_entity_count", 0),
                "release_available": semantic.get("_release_available", False),
                "release_expected": semantic.get("_release_expected"),
                "warning": semantic.get("_warning", ""),
            },
        }

    def layer_path(self, layer_id: str) -> Path | None:
        if not re.fullmatch(r"[a-z0-9_]{1,64}", str(layer_id or "")):
            return None
        for candidate_id, _label, _kind, path in self._layer_specs():
            if candidate_id == layer_id and path.is_file():
                return path
        return None

    def live_status(self) -> dict[str, Any]:
        path = self.car1_root / "search_runs" / "live_status.json"
        value = _read_json(path) or {}
        if value:
            value = dict(value)
            value["_source_path"] = str(path)
            try:
                value["_updated_at"] = int(path.stat().st_mtime * 1000)
            except OSError:
                pass
        return value

    @staticmethod
    def planner_semantic_map(semantic: dict[str, Any]) -> dict[str, Any]:
        """Return a planning-only copy with reviewed labels made authoritative.

        The compatibility map contains several stale geometric labels (for
        example a wall originally tagged ``cabinet``) alongside a later,
        high-confidence Qwen review.  The frozen planner intentionally indexes
        both fields, so leaving the conflict in place can select the wrong
        anchor.  Normalising the request copy preserves the source artefact and
        makes the reviewed annotation authoritative without inventing data.
        """
        result = json.loads(json.dumps({
            key: value for key, value in semantic.items() if not str(key).startswith("_")
        }, ensure_ascii=False))
        entities = result.get("entities")
        if not isinstance(entities, dict):
            return result
        for item in entities.values():
            if not isinstance(item, dict):
                continue
            annotation = item.get("qwen_annotation")
            if not isinstance(annotation, dict) or annotation.get("needs_human_review"):
                continue
            try:
                confidence = float(annotation.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            reviewed_label = str(
                annotation.get("semantic_label") or annotation.get("category") or ""
            ).strip()
            if confidence >= 0.80 and reviewed_label:
                stale_label = str(item.get("semantic_label") or "").strip()
                if stale_label and stale_label != reviewed_label:
                    item["original_semantic_label"] = stale_label
                item["semantic_label"] = reviewed_label
                item["semantic_label_confidence"] = confidence
        return result

    def request_decision(self, instruction: str, timeout: float = 90.0) -> dict[str, Any]:
        semantic = self.semantic_map()
        payload = json.dumps({
            "instruction": instruction,
            "semantic_map": self.planner_semantic_map(semantic),
        }, ensure_ascii=False).encode("utf-8")
        request = urlrequest.Request(
            f"{self.search_api_url}/mission/command",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        with urlrequest.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not isinstance(result, dict) or not result.get("ok") or not isinstance(result.get("decision"), dict):
            raise RuntimeError(str((result or {}).get("error") or "搜索规划服务返回无效结果"))
        decision = dict(result["decision"])
        decision.setdefault("integration", {})
        decision["integration"].update({
            "semantic_status": semantic.get("_status"),
            "semantic_source": semantic.get("_source_path"),
            "stable_release_available": semantic.get("_release_available", False),
        })
        return decision

    def save_decision(self, decision: dict[str, Any]) -> Path:
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        task_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(decision.get("task_id") or "search"))[:80]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.runtime_root / f"{stamp}_{task_id}.json"
        path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def prepare_phase1_pick(self, instruction: str) -> tuple[dict[str, Any], Path]:
        """Use the released DeepSeek/semantic-map compiler without starting hardware."""
        consent_file = self.runtime_root.parent / "deepseek_semantic_upload.allowed"
        try:
            persisted_consent = consent_file.read_text(encoding="utf-8").strip() == "1"
        except OSError:
            persisted_consent = False
        if (os.environ.get("FINAL_WORK_ALLOW_DEEPSEEK_SEMANTIC_UPLOAD") != "1"
                and not persisted_consent):
            raise RuntimeError(
                "DeepSeek 规划需明确允许发送指令及语义地图关系；未调用外部 API，也未启动底盘"
            )
        planner = self.release_root / "grasp_range_integration/phase1/plan_phase1_deepseek.sh"
        if not planner.is_file():
            raise FileNotFoundError(f"第一阶段规划入口不存在：{planner}")
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        path = self.runtime_root / (time.strftime("%Y%m%d_%H%M%S") +
                                    "_phase1_" + uuid.uuid4().hex[:8] + ".json")
        command = ["bash", str(planner), instruction, str(path)]
        try:
            # The 3090 patrol_search HTTP request expires after 105 seconds.
            # Return a clear planning error before that deadline instead of
            # leaving the web step timed out while the Pi starts a mission.
            result = subprocess.run(command, cwd=self.car1_root, capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=95)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("DeepSeek 第一阶段规划超时；没有启动底盘") from exc
        decision = _read_json(path)
        if decision is None:
            raise RuntimeError("DeepSeek 第一阶段未生成有效决策：" +
                               (result.stderr or result.stdout)[-500:])
        expected_code = 0 if decision.get("status") == "executable" else 2
        if result.returncode != expected_code:
            raise RuntimeError("DeepSeek 第一阶段规划失败：" +
                               (result.stderr or result.stdout)[-500:])
        if decision.get("instruction") != instruction:
            raise RuntimeError("规划输出的原始指令不匹配")
        if decision.get("status") == "executable":
            target = decision.get("target") or {}
            phase = decision.get("phase_control") or {}
            if (not str(target.get("query") or "").strip()
                    or not str(target.get("class_name") or "").strip()
                    or str(target.get("class_name") or "").strip().lower()
                    in {"any_yolo_object", "any_object"}
                    or int(target.get("requested_count") or 0) != 1
                    or phase.get("requested_final_action") != "grasp"):
                raise ValueError("夹取任务必须指定单个、明确类别的目标，并由模型在夹取前重新确认双目中心")
        return decision, path

    def phase1_pick_command(self, decision_path: Path) -> tuple[list[str], dict[str, str]]:
        """Find-only process exits zero only after a fresh stereo mission success."""
        resolved = decision_path.resolve(strict=True)
        try:
            resolved.relative_to(self.runtime_root.resolve(strict=True))
        except ValueError as exc:
            raise ValueError("第一阶段决策文件不在 final_work 运行目录内") from exc
        python = self.car1_root / ".venv-pi/bin/python3"
        module = self.release_root / "grasp_range_integration/phase1/verified_search.py"
        required = (python, module, self.release_root / "ble_scene_17.pgm",
                    self.release_root / "ble_scene_17.yaml")
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("完整夹取前的寻物接口缺少运行文件：" + "、".join(missing))
        return [str(python), "-u", "-m",
                "grasp_range_integration.phase1.verified_search",
                "--decision", str(resolved), "--execute-verified-search"], {
                    "PYTHONPATH": str(self.release_root) + os.pathsep + str(self.car1_root),
                    "CAR_ROOT": str(self.car1_root),
                }

    def mission_command(self, decision_path: Path, real: bool) -> tuple[list[str], dict[str, str]]:
        try:
            resolved = decision_path.resolve()
            resolved.relative_to(self.runtime_root.resolve())
        except (OSError, ValueError) as exc:
            raise ValueError("导航决策文件不在 final_work 运行目录内") from exc
        release_map = self.release_root / "ble_scene_17.pgm"
        release_yaml = self.release_root / "ble_scene_17.yaml"
        landmarks = self.release_root / "semantic_localization_landmarks.json"
        python = self.car1_root / ".venv-pi" / "bin" / "python"
        required = (release_map, release_yaml, landmarks, python)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("正式地图发布包缺少运行文件：" + "、".join(missing))
        command = [
            str(python), "-u", "-m", "final_work.pi.ai_observation_selector",
            "--decision", str(resolved),
            "--map", str(release_map),
            "--map-yaml", str(release_yaml),
            "--visual-localization-semantic-map", str(landmarks),
            "--output", str(self.car1_root / "search_runs" / "ble_scene_17"),
            "--3090-url", self.search_api_url,
        ]
        if not real:
            command.append("--simulate")
        else:
            command += [
                "--enable-motion", "--link", "ble",
                "--ble-name", "ESP32-Robot-Car", "--ble-address", "64:B7:08:69:9A:E6",
                "--lidar-port", "/dev/ttyACM0",
                "--speed", "240", "--slow-speed", "230",
                "--max-drive-speed", "460", "--drive-speed-step", "60",
                "--start-kick-speed", "400",
                "--turn-speed", "1250", "--max-turn-speed", "2600",
                "--relocalization-speed", "300", "--relocalization-max-speed", "460",
                "--relocalization-turn-speed", "1500", "--relocalization-max-turn-speed", "2600",
                "--relocalization-turn-step", "300", "--localization-required-moves", "1",
                "--stop-distance", "0.45", "--clearance", "0.30",
                "--camera-backend", "opencv", "--dual-usb-cameras",
                "--camera-width", "1280", "--camera-height", "720",
                "--real-test-max-distance", os.environ.get("SEARCH_TEST_MAX_DISTANCE", "0"),
                "--max-runtime-min", "15",
            ]
        return command, {
            "SEARCH_3090_URL": self.search_api_url,
            "PYTHONPATH": str(TRACK_ROOT),
            "FINAL_WORK_SEMANTIC_RELEASE": str(self.release_root),
        }

    def region_job(self, target: dict, *, carrying: bool, pose_hint: list,
                   carry_id: str = "") -> tuple[dict, Path]:
        from .region_navigation import entities_for
        candidates = entities_for(self.semantic_map(), target)
        # A nearest request must not accept a prematurely model-selected id.
        if target.get("selection") == "nearest_reachable":
            candidates = entities_for(self.semantic_map(), {**target, "landmark_id": None})
        job = {"schema": "robot_region_job.v1", "job_id": uuid.uuid4().hex,
               "status": "planned", "target": target, "candidates": candidates,
               "carrying": carrying, "carry_id": carry_id, "pose_hint": pose_hint,
               "created_at": time.time()}
        path = self.save_decision({**job, "task_id": "region_" + job["job_id"]})
        return job, path

    def region_command(self, job_path: Path) -> tuple[list[str], dict[str, str]]:
        resolved = job_path.resolve(strict=True)
        resolved.relative_to(self.runtime_root.resolve(strict=True))
        return [str(self.car1_root / ".venv-pi/bin/python3"), "-u", "-m",
                "final_work.pi.region_navigation", "--run-job", str(resolved)], {
                    "PYTHONPATH": os.pathsep.join((str(TRACK_ROOT), str(self.release_root), str(self.car1_root))),
                    "CAR_ROOT": str(self.car1_root),
                    "FINAL_WORK_SEMANTIC_RELEASE": str(self.release_root)}
