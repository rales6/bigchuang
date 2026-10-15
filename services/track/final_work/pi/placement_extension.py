"""Narrow grab_visual extension: queued ground placement without #HOME!.

Installed into arm_teach's existing blueprint by one registration line. Uses
its existing BLE pacing, servo limits, single queue and stop generation.
"""
import math
import re
import threading
import time
import json
from pathlib import Path


def validate_pose(params, servos):
    result = {}
    for spec in servos:
        key = spec["key"]
        value = float(params[key])
        if not math.isfinite(value) or value != int(value) or not spec["min"] <= value <= spec["max"]:
            raise ValueError("放置姿态超出已有舵机工作空间：" + key)
        result[key] = int(value)
    return result


def run_placement(arm, ble_send, target, status, generation, capture=None, persist=None):
    # Persist BEFORE sending a physical command: a lost reply is not proof
    # that a lowering/release did not happen.
    status.update(placement_protocol=2, lower_started=False, release_started=False)
    link_fault = None

    def cancelled():
        if generation != arm._exec_cancel_generation:
            raise RuntimeError("用户急停/取消放置")

    def guarded_send(command):
        nonlocal link_fault
        cancelled()
        if link_fault:
            raise ConnectionError(link_fault)
        result = ble_send(command)
        if result != "OK":
            link_fault = "放置 BLE 写入未确认：{} ({})".format(command, result)
            raise ConnectionError(link_fault)
        return result

    def phase(value):
        if guarded_send("#PING!") != "OK":
            raise ConnectionError("放置 BLE 心跳未确认")
        status.update(phase=value, updated_at=time.time())
        with arm._state_lock:
            arm._exec_status.update(active=value != "done", phase="place_" + value,
                                    params=target, user_stopped=False)
        if persist:
            persist(status)

    def mark_started(key):
        status[key] = True
        if persist:
            persist(status)

    def photograph(stage):
        if capture:
            try:
                capture(stage, status)
            except Exception as exc:
                # Evidence I/O is not permission to repeat a completed release.
                status.setdefault("image_errors", {})[stage] = str(exc)

    def establish_link():
        status.update(phase="handoff", updated_at=time.time(),
                      message="搬运链路交接：只用心跳等待 BLE 恢复，保持夹爪闭合")
        with arm._state_lock:
            arm._exec_status.update(active=True, phase="place_handoff", params=target,
                                    user_stopped=False)
        if persist:
            persist(status)
        deadline, confirmations = time.monotonic() + 20, 0
        attempts = 0
        while True:
            cancelled()
            attempts += 1
            try:
                result = ble_send("#PING!")
            except Exception as exc:
                result = type(exc).__name__
            cancelled()
            confirmations = confirmations + 1 if result == "OK" else 0
            status.update(link_attempts=attempts, link_last_result=result,
                          updated_at=time.time())
            if persist:
                persist(status)
            if confirmations >= 2:
                return
            if time.monotonic() >= deadline:
                raise ConnectionError("放置前 BLE 心跳仍未恢复；未下降、未松爪，可重试交接")
            time.sleep(.2)

    try:
        establish_link()
        # Admission already requires known, closed carry history. A redundant
        # ARMC0 was the first stale-GATT write that aborted the previous run.
        # Never HOME (opens claw), and never re-apply grasp nudges here.
        photograph("before")
        phase("lower")
        mark_started("lower_started")
        arm.send_grasp_pose(guarded_send, target)
        arm._heartbeat_wait(guarded_send, .6)
        phase("release")
        mark_started("release_started")
        arm.send_claw(guarded_send, arm.CLAW_OPEN)
        phase("lift")
        arm.send_grasp_pose(guarded_send, {s["key"]: s["home"] for s in arm.SERVOS})
        photograph("after")
        status.update(ok=True, physical_place_verified=False)
        phase("done")
    except Exception as exc:
        # No HOME or recovery opening on error: do not drop a carried item.
        status.update(ok=False, phase="error", error=str(exc), updated_at=time.time(),
                      retryable_before_lower=(not status["lower_started"]
                          and not status["release_started"]
                          and generation == arm._exec_cancel_generation))
        try:
            status["stop_result"] = ble_send("#STOP!")
        except Exception as stop_exc:
            status["stop_result"] = type(stop_exc).__name__
        finally:
            with arm._state_lock:
                arm._exec_status.update(active=False, phase="place_error", message=str(exc))
    finally:
        if persist:
            persist(status)


def register_placement(bp, arm, ble_send, jpegs=None):
    from flask import request, jsonify
    jobs, lock = {}, threading.RLock()
    evidence_root = Path("/home/dachuang/dachuang/track/final_work/data/placement_evidence")

    def persist(status):
        directory = evidence_root / status["job_id"]
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "status.json.tmp"
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(directory / "status.json")

    def capture(stage, status):
        directory = evidence_root / status["job_id"]
        directory.mkdir(parents=True, exist_ok=True)
        images = {}
        for camera in ("left", "right"):
            frame = (jpegs or {}).get(camera) or b""
            if isinstance(frame, bytes) and len(frame) > 1000:
                path = directory / f"{stage}_{camera}.jpg"
                path.write_bytes(frame)
                images[camera] = str(path)
        status.setdefault("images", {})[stage] = images
        status[stage + "_capture_at"] = time.time()

    @bp.post("/api/arm-teach/adopt-carry")
    def adopt_carry():
        # Read only OUR ledger, not a client-supplied pose or file. This
        # restores command history, not a hardware read-back claim.
        if request.remote_addr not in {"127.0.0.1", "::1"}:
            return jsonify(ok=False, error="携带交接仅允许本机桥服务"), 403
        path = Path("/home/dachuang/dachuang/track/final_work/data/robot_skills_state.json")
        try:
            proof = json.loads(path.read_text(encoding="utf-8"))
            if proof.get("status") != "carrying":
                raise ValueError("缺少当前持物记录")
            # An explicit, recent in-place release does not have a region
            # navigation job. Restore command history only; do not send HOME,
            # move a joint, or invent an arrival to authorize the arm queue.
            if proof.get("placement_mode") == "in_place":
                age = time.time() - float(proof.get("in_place_requested_at") or 0)
                if not 0 <= age <= 120:
                    raise ValueError("缺少当前原地放置请求")
                validate_pose(proof.get("ground_pose") or {}, arm.SERVOS)
                if not re.fullmatch(r"[a-f0-9]{32}", str(proof.get("carry_id") or "")):
                    raise ValueError("原地放置持物编号无效")
            else:
                job = json.loads(Path(proof["region_job"]).read_text(encoding="utf-8"))
                if (job.get("status") != "arrived" or job.get("carry_id") != proof.get("carry_id")
                        or time.time() - float(job.get("arrived_at") or 0) > 120):
                    raise ValueError("缺少当前搬运到达交接")
            with arm._state_lock:
                if arm._arm_jobs_pending or arm._exec_status.get("active"):
                    raise RuntimeError("机械臂队列忙，不能覆盖历史状态")
                if arm._state.get("known"):
                    return jsonify(ok=True, state=arm.snapshot())
                arm._state.update({s["key"]: s["home"] for s in arm.SERVOS})
                arm._state.update(known=True, claw_value=arm.CLAW_CLOSE,
                                  last_command="carrying_command_history_handoff", updated_at=time.time())
            return jsonify(ok=True, state=arm.snapshot(), hardware_readback=False)
        except (OSError, ValueError, KeyError, RuntimeError, TypeError, OverflowError) as exc:
            return jsonify(ok=False, error=str(exc)), 409

    @bp.post("/api/arm-teach/place")
    def place():
        data = request.get_json(silent=True) or {}
        job_id = str(data.get("job_id") or "")
        if not re.fullmatch(r"[a-f0-9_]{16,100}", job_id):
            return jsonify(ok=False, error="放置任务编号无效"), 400
        with lock:
            existing = evidence_root / job_id / "status.json"
            if job_id not in jobs and existing.is_file():
                jobs[job_id] = json.loads(existing.read_text(encoding="utf-8"))
            if job_id in jobs:
                return jsonify(dict(jobs[job_id]))
            try:
                params = validate_pose(data.get("params") or {}, arm.SERVOS)
                with arm._state_lock:
                    if arm._exec_status.get("active") or arm._arm_jobs_pending:
                        raise RuntimeError("机械臂队列已有其他动作")
                    if not arm._state.get("known") or arm._state.get("claw_value") != arm.CLAW_CLOSE:
                        raise RuntimeError("夹爪闭合与携带姿态未确认")
                    generation = arm._exec_cancel_generation
                status = {"ok": True, "job_id": job_id, "phase": "queued", "params": params}
                jobs[job_id] = status
                persist(status)
                arm.enqueue_command(run_placement, arm, ble_send, params, status, generation, capture, persist)
            except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                jobs.pop(job_id, None)
                return jsonify(ok=False, error=str(exc)), 409
            return jsonify(dict(status))

    @bp.get("/api/arm-teach/place-status")
    def place_status():
        with lock:
            job_id = str(request.args.get("job_id") or "")
            if not re.fullmatch(r"[a-f0-9_]{16,100}", job_id):
                return jsonify(ok=False, phase="unknown"), 400
            status = jobs.get(job_id)
            if status is None:
                try:
                    status = json.loads((evidence_root / job_id / "status.json").read_text(encoding="utf-8"))
                    safe_pre_lower = (status.get("phase") == "error"
                        and status.get("placement_protocol") == 2
                        and status.get("lower_started") is False
                        and status.get("release_started") is False)
                    if status.get("phase") != "done" and not safe_pre_lower:
                        status.update(ok=False, phase="unknown", message="服务重启后的未完成放置，不能重复动作")
                except (OSError, ValueError):
                    pass
            return jsonify(dict(status) if status else {"ok": False, "phase": "unknown",
                "message": "无此放置任务；可能预览服务已重启，不允许重复执行"})
