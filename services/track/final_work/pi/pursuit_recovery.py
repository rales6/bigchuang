"""Reacquire an identified target after pursuit/near-blind recovery stops.

This adapter owns no motors. A new attempt must reacquire current images;
neither an old bounding box nor an issued movement is evidence of progress.
"""
from __future__ import annotations

import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
import time


LOSS_DETAIL = "受限转向搜索仍未找回目标，已停车，禁止连续盲转"
ALIGNMENT_LOSS = "目标未同时出现在双目画面"
RETREAT_SCAN_DETAIL = "短回退期间雷达扫描失效或后方出现非固定近距障碍，已停车"
REACQUIRE_DETAILS = frozenset({
    LOSS_DETAIL, RETREAT_SCAN_DETAIL,
    "转向回溯找回超时，已停车；不得继续盲转或前进",
    "近端盲区双目找回超时，已停车；不得继续前进或盲转",
    "小步回退预算已用尽，双目仍未安全找回；停车等待重新观察",
    "近端盲区已尝试有限次短回退，双目仍未安全找回；停车等待人工检查",
    "停车重新识别超时，已保持停止；未授权抓取",
    "重新识别后双目未连续恢复，已停车",
})


class LossRecoveryBudget:
    """Two fresh restarts, sharing one recovery window across both stages."""

    def __init__(self, max_restarts=2, timeout_s=180.0, clock=time.monotonic):
        self.max_restarts = max_restarts
        self.timeout_s = timeout_s
        self.clock = clock
        self.started = None
        self.restarts = 0

    def reserve(self, cancelled) -> bool:
        now = self.clock()
        if cancelled.is_set():
            return False
        if self.started is None:
            self.started = now
        if self.restarts >= self.max_restarts or now - self.started >= self.timeout_s:
            return False
        self.restarts += 1
        return True


def identified_loss_run(track_root, *, target, child_pid, launched_at,
                        returncode, now=None) -> dict | None:
    """LATEST is only a locator; validate this child's identity and telemetry.

The frozen runner suppresses stop errors in cleanup. Metadata alone therefore
does NOT authorize movement: the caller must additionally confirm STOP and
fresh preview/BLE before restarting.
    """
    if returncode != 2 or not target:
        return None
    now = time.time() if now is None else now
    try:
        root = (Path(track_root) / "logs" / "pursuit").resolve()
        directory = Path((root / "LATEST.txt").read_text(encoding="utf-8").strip()).resolve()
        if not directory.is_relative_to(root):
            return None
        metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        started = datetime.fromisoformat(metadata["started_at"]).timestamp()
        ended = datetime.fromisoformat(metadata["ended_at"]).timestamp()
        if (metadata.get("pid") != child_pid or metadata.get("mode") != "esp32"
                or metadata.get("instruction") != target
                or metadata.get("active_instruction") != target
                or metadata.get("status") != "give_up"
                or metadata.get("final_state") != "GIVE_UP"
                or metadata.get("final_result") != "give_up"
                or metadata.get("final_detail") not in REACQUIRE_DETAILS
                or metadata.get("error") != ""
                or not all(math.isfinite(t) for t in (started, ended, launched_at, now))
                or not launched_at - 2 <= started <= ended <= now
                or not 0 <= now - ended <= 15):
            return None
        # Do not restart a never-recognized target's initial blind search.
        with (directory / "telemetry.csv").open(encoding="utf-8", newline="") as stream:
            seen = any(row.get("state") == "TRACKING" and any(
                row.get(key, "").lower() in {"true", "1"}
                for key in ("tracking", "tracking2")) for row in csv.DictReader(stream))
        if not seen:
            return None
        # Ending a blocked reverse pulse does not authorize another reverse.
        # Only a fresh perception session is requested. The caller confirms
        # STOP, restores cameras/BLE and (for scan stops) verifies new lidar.
        detail = metadata["final_detail"]
        return {"run_dir": str(directory), "reason": detail,
                "requires_fresh_lidar": detail == RETREAT_SCAN_DETAIL}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, csv.Error):
        return None


def alignment_target_lost(result) -> bool:
    return isinstance(result, dict) and result.get("ok") is False and result.get("reason") == ALIGNMENT_LOSS


def alignment_convergence_retry(result) -> bool:
    """Only mild optimisation failures; never classify a hardware stop as one."""
    if not isinstance(result, dict) or result.get('ok') is not False:
        return False
    reason = str(result.get('reason') or '')
    match = re.fullmatch(
        r'微调后凸包距离未改善 \(([0-9.]+)→([0-9.]+)px\)，已停止', reason)
    if match is None:
        return False
    try:
        before, after = map(float, match.groups())
        distance = float(result['distance_px'])
        pulses = int(result['pulses'])
        return (all(math.isfinite(v) for v in (before, after, distance))
                and 0 <= after <= 240 and after <= before + 8
                and abs(distance-after) <= 1.0 and pulses > 0)
    except (ValueError, TypeError, KeyError):
        return False


def align_with_loss_recovery(target, *, align, prepare, cancelled, real, emit,
                             budget=None, max_convergence_retries=3,
                             retry_target_loss=True, deadline=None, clock=time.monotonic):
    """Fresh alignment retries; single-step low progress is not task failure.

    Each callback obtains new stereo evidence and owns its guarded pulses.
    Do not renew retreat permissions or bypass the actual grasp-region check.
    """
    budget = budget or LossRecoveryBudget()
    convergence_retries = 0
    attempts = []
    while not cancelled.is_set():
        if deadline is not None and clock() + 12 >= deadline:
            if attempts:
                return {**attempts[-1], 'alignment_attempts': attempts,
                        'convergence_retries': convergence_retries}
            return {'ok': False, 'reason': '微调复核时间不足，未启动机械臂'}
        result = align()
        attempts.append(dict(result))
        if cancelled.is_set():
            break
        if not real:
            return result
        if alignment_convergence_retry(result):
            if convergence_retries >= max_convergence_retries:
                emit('hull_convergence_retry_exhausted', target=target,
                     retries=convergence_retries, reason=result['reason'])
                return {**result, 'alignment_attempts': attempts,
                        'convergence_retries': convergence_retries}
            convergence_retries += 1
            emit('hull_convergence_retry', target=target, retry=convergence_retries,
                 retry_limit=max_convergence_retries, reason=result['reason'],
                 previous_distance_px=result.get('distance_px'),
                 resume_from_current_position=True, fresh_stereo_required=True)
            print(f'[phase] 单步微调反馈不足：重新拍摄双目，从当前位置继续调整 '
                  f'（额外第{convergence_retries}/{max_convergence_retries}轮）', flush=True)
            prepare()
            continue
        if not retry_target_loss or not alignment_target_lost(result):
            return ({**result, 'alignment_attempts': attempts,
                     'convergence_retries': convergence_retries}
                    if convergence_retries else result)
        if not budget.reserve(cancelled):
            emit("pursuit_loss_recovery_exhausted", target=target, stage="hull",
                 restarts=budget.restarts, reason=result["reason"])
            return result
        emit("pursuit_loss_recovery", target=target, stage="hull",
             restart=budget.restarts, reason=result["reason"], resume_from_current_position=True)
        prepare()
    return {"ok": False, "reason": "任务已取消，未启动机械臂"}
