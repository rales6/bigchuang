"""Persistent pick/carry/arrival/place handoffs, with no direct motor control."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import threading
import time
import uuid

from .region_navigation import save_json


class RobotSkills:
    def __init__(self, bridge):
        self.bridge = bridge
        self.path = bridge.map_navigation.runtime_root.parent / "robot_skills_state.json"
        self.lock = threading.RLock()
        self.boot_id = uuid.uuid4().hex
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}
        if not isinstance(self.data, dict):
            self.data = {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        save_json(self.path, self.data)

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.data))

    def assert_empty(self):
        """An explicit new pick/HOME supersedes a previous demo's carry record.

        This is not a physical empty-gripper assertion. Keep this round's
        carry handoff until a new pick/HOME is requested, so placement works.
        A pending/unknown release still needs reconciliation, not duplication.
        """
        with self.lock:
            if self.data.get("status") in {"placing", "placement_unknown"}:
                raise RuntimeError("放置动作尚未结束或结果待查询，不能重复插入机械臂动作")
            if self.data.get("status") == "carrying":
                previous = {k: v for k, v in self.data.items() if k != "previous_carry"}
                self.data = {"status": "ready_for_pick", "reason": "new_pick_or_home_requested",
                             "superseded_at": time.time(), "previous_carry": previous,
                             "physical_empty_verified": False}
                self.save()

    def confirm_empty(self, payload, headers):
        """Operator confirms physical emptiness; metadata only, no arm command."""
        if payload.get('confirmed_empty') is not True:
            raise ValueError('请先确认夹爪已经没有物品，再清除持物状态')
        self.bridge._authorise_hardware(headers)
        if not self.bridge._grasp_lock.acquire(blocking=False):
            raise RuntimeError('机械臂任务仍在执行，暂不能复位持物状态')
        try:
            managed = self.bridge.process.snapshot()
            if managed.get('running') and managed.get('kind') != 'camera_preview':
                raise RuntimeError('任务仍在运行，请结束后再确认空夹爪')
            with self.lock:
                expected = payload.get('expected_carry_id')
                if expected != self.data.get('carry_id'):
                    raise RuntimeError('持物记录已改变，请刷新状态后再确认')
                previous = {k: self.data.get(k) for k in ('status', 'carry_id', 'target')}
                self.data = {'status': 'empty', 'reason': 'operator_confirmed_empty',
                             'confirmed_at': time.time(), 'previous_carry': previous}
                self.save()
                return {'ok': True, 'carry': self.snapshot(), 'hardware_commands_sent': False,
                        'message': '已确认空夹爪并清除旧持物记录；机械臂和底盘未动作'}
        finally:
            self.bridge._grasp_lock.release()

    def record_pick(self, target, result):
        with self.lock:
            execution = result.get("execution") or {}
            params = execution.get("params") or {}
            if execution.get("phase") != "done" or any(k not in params for k in ("s0", "s1", "s2")):
                raise RuntimeError("抓取执行完成但缺少本次落地姿态；不可衔接自动放置")
            self.data = {"status": "carrying", "carry_id": uuid.uuid4().hex,
                         "target": target, "ground_pose": {k: int(params[k]) for k in ("s0", "s1", "s2")},
                         "picked_at": time.time(), "physical_pick_verified": False,
                         "pose_hint": self.current_pose_hint()}
            self.save()
            return self.snapshot()

    def current_pose_hint(self):
        value = self.bridge.map_navigation.live_status().get("pose") or {}
        try:
            pose = [float(value["x_m"]), float(value["y_m"]), math.radians(float(value["yaw_deg"]))]
            return pose if all(math.isfinite(v) for v in pose) else []
        except (TypeError, ValueError, KeyError):
            return []

    def check_carry(self, request, base_url):
        with self.lock:
            if self.data.get("status") != "carrying":
                raise RuntimeError("没有可衔接的本次模型夹取记录；不会猜测舵机落地姿态")
            arm = request(base_url + "/api/arm-teach/state", timeout=4).get("state") or {}
            if arm.get("known") is not True or arm.get("claw_value") != 0:
                raise RuntimeError("夹爪闭合状态未确认；保留记录，不自动重复抓取或松爪")
            return {"ok": True, "carry": self.snapshot(), "physical_pick_verified": False,
                    "message": "夹取序列完成且夹爪命令仍闭合；无力传感器，实物夹住与否尚未独立确认"}

    def resolve(self, target):
        with self.lock:
            if target.get("surface", "floor") != "floor":
                raise ValueError("仅支持地点旁的地面放置；不能推断椅面/桌面高度")
            if self.data.get("status") != "carrying":
                raise RuntimeError("先完成抓取，再选择放置地点")
            # This is deliberately read-only w.r.t. hardware. The live stage
            # re-resolves nearest after acquiring a fresh radar pose.
            hint = self.current_pose_hint() or self.data.get("pose_hint") or []
            job, path = self.bridge.map_navigation.region_job(target, carrying=True, pose_hint=hint,
                                                            carry_id=self.data["carry_id"])
            self.data.update(region_job=str(path), region_boot_id=self.boot_id,
                             destination_query=target.get("query"))
            self.save()
            return {"ok": True, "selection": target.get("selection"), "candidates": job["candidates"],
                    "reference": "fresh_pose_in_carry_navigation", "job_id": job["job_id"],
                    "message": "仅绑定候选；起步定位后按当前位置的可达路径确定最近区域"}

    def navigate(self, target, headers):
        with self.lock:
            bridge = self.bridge
            bridge._authorise_hardware(headers)
            if bridge._grasp_lock.locked():
                raise RuntimeError("机械臂任务执行中不能导航")
            current = bridge.process.snapshot()
            if current.get("running") and current.get("kind") != "camera_preview":
                raise RuntimeError("其他底盘任务仍在执行")
            carrying = self.data.get("status") == "carrying"
            if target.get("reference") == "current_pose_after_pickup" and not carrying:
                raise RuntimeError("搬运步骤没有闭爪携带交接记录")
            if carrying:
                if (not self.data.get("region_job") or self.data.get("destination_query") != target.get("query")
                        or self.data.get("region_boot_id") != self.boot_id):
                    self.resolve(target)
                path = Path(self.data["region_job"])
            else:
                _, path = bridge.map_navigation.region_job(target, carrying=False, pose_hint=[])
            command, env = bridge.map_navigation.region_command(path)
            process = bridge.process.start("region_navigation", command,
                                           bridge.map_navigation.car1_root, extra_env=env)
            return {"ok": True, "process": process, "job_path": str(path), "carrying": carrying}

    def arrival(self):
        if self.data.get("region_boot_id") != self.boot_id:
            raise RuntimeError("服务重启后旧到达记录不能直接授权松爪；先重新完成区域导航")
        job = json.loads(Path(self.data["region_job"]).read_text(encoding="utf-8"))
        if (job.get("status") != "arrived" or job.get("carry_id") != self.data.get("carry_id")
                or (job.get("arrival") or {}).get("anchor_visual_confirmed") is not True
                or (job.get("arrival") or {}).get("placement_clearance") is not True):
            raise RuntimeError("尚未到达并复核本次放置区域；夹爪保持闭合，不在途中松爪")
        if time.time() - float(job.get("arrived_at") or 0) > 120:
            raise RuntimeError("到达证据已过期，请重新复核放置区域，物品保持夹住")
        return job

    def place(self, target, headers, *, request, base_url):
        bridge = self.bridge
        bridge._authorise_hardware(headers)
        with self.lock:
            if self.data.get("status") == "placed":
                return self.data.get("placement") or {"ok": True, "already_completed": True}
            if self.data.get("status") in {"placing", "placement_unknown"}:
                # A timeout is NOT permission to repeat a physical release.
                status = request(base_url + "/api/arm-teach/place-status?job_id=" + self.data["place_id"], timeout=4)
                return self._placement_result(status)
            if self.data.get("status") != "carrying":
                raise RuntimeError("没有本次携带记录，不能执行放置")
            if target.get("surface", "floor") != "floor":
                raise ValueError("仅支持地面放置；不能推断桌面或椅面高度")
            managed = bridge.process.snapshot()
            if managed.get("running") and managed.get("kind") != "camera_preview":
                raise RuntimeError("放置前底盘任务尚未退出；不能在行驶中松爪")
            in_place = target.get("selection") == "in_place"
            if in_place:
                if target.get("allow_chassis_motion") is not False or target.get("reference") != "current_pose":
                    raise ValueError("原地放置必须明确禁止底盘移动")
                # Explicit operator request authorizes the current floor, not
                # a semantic arrival. Never consult stale destination/pose data.
                job = {"job_id": uuid.uuid4().hex, "placement_mode": "in_place"}
                self.data.update(placement_mode="in_place", in_place_requested_at=time.time())
                self.save()  # Permit a metadata-only arm handoff after preview restart.
            else:
                if target.get("query") != self.data.get("destination_query"):
                    raise ValueError("放置步骤的目的区域与本次导航不一致")
                job = self.arrival()
            # Preview restarts for each exclusive navigation stage. Its
            # software arm state is initially unknown, although no HOME was
            # sent and the ESP32 still holds the closed/lifted pose.
            ready_deadline = time.monotonic() + 20
            while True:
                try:
                    ready = request(base_url + "/api/status", timeout=3)
                    cameras = ready.get("cameras") or {}
                    fresh = all(isinstance(cameras.get(side), dict)
                        and cameras[side].get("frames", 0) > 0
                        and 0 <= float(cameras[side].get("age_s", 99)) < 2.0
                        for side in ("left", "right"))
                    if ready.get("ble") is True and fresh:
                        break
                except Exception:
                    pass
                if time.monotonic() >= ready_deadline:
                    raise RuntimeError("放置双摄/BLE仍未就绪；物品保持夹住，可重试衔接步骤")
                time.sleep(.3)
            arm = request(base_url + "/api/arm-teach/state", timeout=4).get("state") or {}
            if not arm.get("known"):
                request(base_url + "/api/arm-teach/adopt-carry", {}, timeout=4)
            self.check_carry(request, base_url)
            managed = bridge.process.snapshot()
            if managed.get("running") and managed.get("kind") != "camera_preview":
                raise RuntimeError("放置前底盘导航尚未退出")
            if not bridge._grasp_lock.acquire(blocking=False):
                raise RuntimeError("机械臂已被其他任务占用")
            try:
                bridge._grasp_cancel.clear()
                # A proven pre-lower failure can be retried, but it must not
                # collide with the preview's idempotent completed/error job.
                place_id = self.data["carry_id"] + "_" + job["job_id"] + "_" + uuid.uuid4().hex[:8]
                self.data.update(status="placing", place_id=place_id,
                                 placement_mode="in_place" if in_place else "named_region")
                self.save()  # Before enqueue: never duplicate on lost reply.
                status = request(base_url + "/api/arm-teach/place", {
                    "job_id": place_id, "params": self.data["ground_pose"]}, timeout=8)
                deadline = time.monotonic() + 90
                while status.get("phase") in {"queued", "handoff", "lower", "release", "lift"}:
                    if bridge._grasp_cancel.is_set():
                        raise RuntimeError("放置已被急停取消")
                    if time.monotonic() >= deadline:
                        raise RuntimeError("放置回复超时；保留任务编号供查询，不重复下臂")
                    time.sleep(.25)
                    status = request(base_url + "/api/arm-teach/place-status?job_id=" + place_id, timeout=4)
                return self._placement_result(status)
            except Exception:
                self.data["status"] = "placement_unknown"
                self.save()
                raise
            finally:
                bridge._grasp_lock.release()

    def _placement_result(self, status):
        if status.get("phase") == "done" and status.get("ok"):
            self.data.update(status="placed", placement=status, placed_at=time.time())
            self.save()
            return {**status, "physical_place_verified": False,
                    "message": "下臂、松爪、抬臂序列已完成；实物落地结果需结合照片核实"}
        if status.get("phase") in {"error", "unknown"}:
            safe_pre_lower = (status.get("placement_protocol") == 2
                and status.get("lower_started") is False
                and status.get("release_started") is False
                and status.get("retryable_before_lower") is True)
            self.data.update(status="carrying" if safe_pre_lower else "placement_unknown",
                             placement=status, last_place_id=self.data.get("place_id"))
            self.save()
        return {**status, "ok": False}

    def invalidate(self, reason):
        with self.lock:
            self.data.pop("region_boot_id", None)
            self.data["invalidated_by"] = reason
            self.save()

    def cleared(self, reason):
        with self.lock:
            self.data = {"status": "empty", "reason": reason}
            self.save()
