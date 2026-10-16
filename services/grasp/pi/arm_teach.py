"""机械臂示教夹取 (独立系统) — 树莓派端 (v2: 三舵机脉宽直控)。

数据流:
  采集: 左右相机图 + Qwen 双目标中心 (3090:8022) + 三舵机脉宽 (s0/s1/s2) -> 样本
  训练: 3090 train_arm_mlp.py 读样本 -> MLP(4 中心 -> s0/s1/s2) + 4D 凸包
  夹取: 拼接帧 -> 3090 预测 -> 两眼二维凸包或25px外沿 -> #HOME! -> #ARMG -> 闭合 -> HOME

控制通道: ESP32 BLE NUS ASCII。固件 v19 (2026.08.16) 新增 #ARMG{s0}{s1}{s2}!:
  0/1/2 号舵机直达目标脉宽 (各限位内钳制), 3 号与 2 号镜像联动 (s3=3100-s2),
  4 号舵机不参与, 5 号夹爪走 #ARMC 指令。旧 #ARM 相对指令保留给队友系统。
每次发送都带全三轴当前值 (绝对指令, 幂等): 设值与逐步到达物理等价,
彻底消除旧相对积分方案的"设值100 ≠ +10x10"类不一致。
状态 = 树莓派最后发出的绝对脉宽 (无编码器, 无回读); #HOME! 锚定 (1500,1700,2000)。
"""
from __future__ import annotations

import base64
import hmac
import json
import math
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, Response, jsonify, request
from teaching_region import region_accepts,GRASP_MARGIN_PX

DEFAULT_TEACH_URL = "http://192.0.2.20:8022"
DATASET_ROOT = Path(__file__).resolve().parent / "arm_teach_dataset"


def _require_taught_camera_order() -> None:
    """Bind taught left/right to current physical video2/video0 (replay audited)."""
    args = sys.argv
    # Deployment includes the SAME canonical module beside web_app.py.
    try:
        from pursuit.camera_contract import model_devices
    except ImportError:
        from camera_contract import model_devices
    expected = model_devices(require_ports=sys.platform == 'linux')
    def device(flag: str, default: int) -> int:
        equals = next((arg.split("=", 1)[1] for arg in args
                       if arg.startswith(flag + "=")), None)
        if equals is not None:
            try:
                return int(equals)
            except ValueError as exc:
                raise RuntimeError(f"抓取服务相机参数 {flag} 无效") from exc
        if flag not in args:
            return default
        index = args.index(flag) + 1
        try:
            return int(args[index])
        except (IndexError, ValueError) as exc:
            raise RuntimeError(f"抓取服务相机参数 {flag} 无效") from exc
    left, right = device("--left-camera", 0), device("--right-camera", 2)
    if (left, right) != expected:
        raise RuntimeError(
            f"当前抓取相机为 video{left}/video{right}，"
            f"但模型绑定的物理左右为 video{expected[0]}/video{expected[1]}；禁止模型夹取")

# 三舵机参数: 0=底座左右, 1=关节1俯仰, 2=关节2俯仰 (3 号镜像联动, 4 号不参与)
SERVOS = [
    {"key": "s0", "name": "底座左右 (舵机0)", "home": 1500, "min": 500, "max": 2500,
     "steps": [10, 50], "presets": [1100, 1300, 1500, 1700, 1900]},
    {"key": "s1", "name": "关节1俯仰 (舵机1)", "home": 1700, "min": 800, "max": 1700,
     "steps": [10, 50], "presets": [900, 1100, 1300, 1500, 1700]},
    {"key": "s2", "name": "关节2俯仰 (舵机2)", "home": 2000, "min": 1500, "max": 2200,
     "steps": [10, 50], "presets": [1600, 1800, 2000, 2100, 2200]},
]
SERVO_KEYS = [s["key"] for s in SERVOS]
CLAW_CLOSE = 0    # #ARMC0! = 全闭 (1600us, 用户 2026-08-16: 10 太松)
CLAW_OPEN = 95    # #ARMC95! = 全开

_state_lock = threading.RLock()
_state = {
    "s0": 1500, "s1": 1700, "s2": 2000,
    "claw_value": None,
    "known": False,
    "last_command": "",
    "updated_at": 0.0,
}
# 节奏控制: 固件在"旧动作未完成"时会被新指令抢占停止。
# 每条指令发出后必须等它按固件速度走完, 否则连按会把动作掐断。
_move_lock = threading.Lock()
_move_done_at = 0.0


def _heartbeat_wait(ble_send, seconds: float) -> None:
    """等待期间每 400ms 发一次 #PING! 保活。

    ESP32 固件有 PI_LINK_TIMEOUT_MS=800ms 链路看门狗: 树莓派静默超过 800ms
    就停止一切动作 (#255PDST 冻结斜坡, 2026-08-16 踩坑根因)。
    运动/等待期间必须持续发心跳, 否则每次指令只走一小段。
    """
    deadline = time.time() + max(float(seconds), 0.0)
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return
        time.sleep(min(0.4, remaining))
        if ble_send:
            try:
                ble_send("#PING!")
            except Exception:
                pass


def _pace_and_wait(duration_ms: float, ble_send=None) -> None:
    """等上一条指令完成 (期间心跳保活), 再记录本条指令的预计完成时间。"""
    global _move_done_at
    with _move_lock:
        remaining = _move_done_at - time.time()
        if remaining > 0:
            _heartbeat_wait(ble_send, remaining)
        _move_done_at = time.time() + max(0.0, float(duration_ms)) / 1000.0


# 与固件一致的行程模型 (2026-08-16):
# 底板/舵机链路完好 (原始 #nnnPxxxxTxxxx! 单条一步到位, 实测 ~200us/s)。
# 之前"每次只走一段"的根因是 ESP32 的 800ms 链路看门狗在树莓派静默期间
# 发 #255PDST 冻结斜坡 — 由心跳保活解决 (见 _heartbeat_wait), 无需加大系数。
# 单条指令时长超上限的极端行程 (满量程 yaw) 仍按同一规则链式补发。
GRASP_MS_PER_US = 6
GRASP_MAX_T_MS = 9999
STEP_CAPACITY_US = (GRASP_MAX_T_MS - 700) // GRASP_MS_PER_US  # 1549us


def _step_toward(current: int, target: int):
    """按固件截断规则走一步: 返回 (新的模型位置, 需等待时长 ms)。"""
    delta = target - current
    if abs(delta) <= STEP_CAPACITY_US:
        return target, 700.0 + abs(delta) * GRASP_MS_PER_US + 120.0
    step = current + (STEP_CAPACITY_US if delta > 0 else -STEP_CAPACITY_US)
    return step, GRASP_MAX_T_MS + 120.0


# ── 指令执行线程 (所有机械臂指令串行执行, HTTP 立即返回) ──
_cmd_queue = queue.Queue(maxsize=100)
_queue_gate = threading.Lock()
_arm_jobs_pending = 0
_micro_motion_active = False
_micro_cancel = threading.Event()
_micro_state_lock = threading.Lock()
_micro_active_id = None
_micro_cancelled_ids: set[str] = set()

_MICRO_COMMANDS = {
    # Firmware: S is a percentage of 550 mm/s or 3500 mrad/s.  A generous
    # 400 mm / 30 degree goal only keeps MOV alive until our explicit STOP;
    # it is never used as the requested travel distance.
    "forward": "#MOVF400L000S074!",   # about 407 mm/s
    "backward": "#MOVB400L000S074!",
    "turn_left": "#MOVF000L120S013!", # timer watchdog, not requested rotation
    "turn_right": "#MOVF000R120S013!",
    "arc_left": None,  # V2 twist: independent linear/yaw speeds, NOT MOV's shared S
    "arc_right": None,
}


def _micro_command(action: str, speed_mm_s: float = 405,
                   angular_mrad_s: float = 450) -> str | bytes:
    if (action not in _MICRO_COMMANDS or not math.isfinite(speed_mm_s)
            or not 100 <= speed_mm_s <= 420):
        raise ValueError("非法微调速度或动作")
    if action in {'arc_left','arc_right'}:
        # MOV applies the same S percentage to 550mm/s and 3500mrad/s:
        # F400L120S074 would command ~2590mrad/s, NOT the requested 450.
        # Reuse the existing V2 protocol over this SAME BLE owner instead.
        if not math.isfinite(angular_mrad_s) or not 450 <= angular_mrad_s <= 900:
            raise ValueError('非法弧线微调转向速度')
        try:
            from car.protocol.frame import encode_frame, ADDR_RASPBERRY_PI, ADDR_ESP32
            from car.protocol.messages import encode_twist, MSG_SET_TWIST
        except ImportError:
            import sys
            from pathlib import Path
            track_root = Path('/home/robot/dachuang/track')
            if not (track_root/'car/protocol/frame.py').is_file():
                raise RuntimeError('弧线微调缺少现有 V2 协议模块，禁止降级为 MOV')
            sys.path.insert(0,str(track_root))
            from car.protocol.frame import encode_frame, ADDR_RASPBERRY_PI, ADDR_ESP32
            from car.protocol.messages import encode_twist, MSG_SET_TWIST
        import uuid
        angular = int(angular_mrad_s)*(1 if action=='arc_left' else -1)
        return encode_frame(MSG_SET_TWIST,uuid.uuid4().int & 0xffff,
                            encode_twist(int(speed_mm_s),angular,800),
                            src=ADDR_RASPBERRY_PI,dst=ADDR_ESP32,flags=0)
    if action not in {"forward", "backward"}:
        if not math.isfinite(angular_mrad_s) or not 450 <= angular_mrad_s <= 2400:
            raise ValueError("非法微调转向速度")
        percent = int(round(angular_mrad_s * 100.0 / 3500.0))
        direction = 'L' if action == 'turn_left' else 'R'
        return f"#MOVF000{direction}120S{percent:03d}!"
    percent = int(round(speed_mm_s * 100.0 / 550.0))
    direction = "F" if action == "forward" else "B"
    return f"#MOV{direction}400L000S{percent:03d}!"


def _run_chassis_pulse(ble_send, action: str, duration_s: float,
                       speed_mm_s: float = 405, angular_mrad_s: float = 450,
                       movement_feedback=None) -> dict:
    """Use the preview's sole BLE link; always terminate the MOV command."""
    if action not in _MICRO_COMMANDS or not 0.4 <= duration_s <= 0.8 \
            or not math.isfinite(duration_s):
        raise ValueError("非法底盘微调动作或时长")
    if action in {'arc_left','arc_right'} and movement_feedback is None:
        raise ValueError('弧线微调必须提供实时视觉反馈')
    command = _micro_command(action, speed_mm_s, angular_mrad_s)
    stop_result = None
    failure = None
    started = time.monotonic()
    feedback = None
    termination = 'time_limit'
    try:
        if _micro_cancel.is_set():
            raise RuntimeError("脉冲开始前已收到急停")
        result = ble_send(command) if ble_send else "NO_BLE"
        if result != "OK":
            raise RuntimeError(f"底盘 MOV 未写入 BLE：{result}")
        if _micro_cancel.is_set():
            raise RuntimeError("MOV 写入期间收到急停")
        # MOV may already act while its BLE write awaits acknowledgement.
        # Never add that latency to any requested micro-motion window.
        deadline = started + duration_s
        mov_write_s = time.monotonic() - started
        if movement_feedback is not None:
            feedback = movement_feedback()
            if feedback.get('stop'):
                termination = 'visual_motion_limit'
        last_ping = time.monotonic()
        while time.monotonic() < deadline and termination != 'visual_motion_limit':
            if _micro_cancel.is_set():
                raise RuntimeError("收到急停，底盘微调已取消")
            time.sleep(min(0.04 if movement_feedback else 0.17,
                           max(0.0, deadline - time.monotonic())))
            if movement_feedback is not None:
                feedback = movement_feedback()
                if feedback.get('stop'):
                    termination = 'visual_motion_limit'
                    break
            if deadline - time.monotonic() >= 0.12 and time.monotonic()-last_ping >= 0.17:
                if ble_send("#PING!") != "OK":
                    raise RuntimeError("底盘微调期间 BLE 心跳失败")
                last_ping = time.monotonic()
    except Exception as exc:
        failure = exc
    finally:
        until_stop_requested_s = time.monotonic() - started
        stop_result = ble_send("#STOP!") if ble_send else "NO_BLE"
    if stop_result != "OK":
        raise RuntimeError(f"底盘 STOP 未确认：{stop_result}") from failure
    if failure is not None:
        raise failure
    return {'termination':termination, 'until_stop_requested_s':until_stop_requested_s,
            'command_encoding':'v2_twist' if isinstance(command,bytes) else 'ascii_mov',
            'command_hex':command.hex() if isinstance(command,bytes) else None,
            'command_ascii':command if isinstance(command,str) else None,
            'mov_write_s':mov_write_s, 'requested_window_s':duration_s,
            'visual_feedback':feedback}


def _make_turn_feedback(jpegs: dict, movement_limit_px: float = 8.0,
                        target_centers: dict | None = None):
    """Fast, robust background flow: no Qwen and no target/depth assumption.

    Measure in 640x480 image pixels. RANSAC rejects local object/hand motion;
    missing frames or unreliable matches STOP instead of allowing blind yaw.
    This observes image motion, never claims a metric angle or rear clearance.
    """
    anchors = {}
    cache = {}
    local_cache = {}
    for side in ('left','right'):
        raw = jpegs.get(side)
        if not raw: raise RuntimeError('微调前缺少双目画面')
        image = cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        if image is None: raise RuntimeError('微调前画面解码失败')
        gray = cv2.resize(image,(640,480))
        mask=None
        if target_centers is not None:
            x,y=map(float,target_centers[side])
            mask=np.zeros_like(gray)
            mask[max(0,int(y)-50):min(480,int(y)+51),max(0,int(x)-50):min(640,int(x)+51)]=255
        points = cv2.goodFeaturesToTrack(gray,120,.02,8,
                                        mask=None if mask is None else 255-mask)
        if points is None or len(points)<16:
            raise RuntimeError('微调前视觉特征不足，禁止盲走')
        if target_centers is not None:
            local=cv2.goodFeaturesToTrack(gray,30,.02,5,mask=mask)
            if local is not None: points=np.concatenate((points,local))
        local_cache[side]=0.0
        anchors[side]=(gray,points)
        cache[side]=(raw,time.monotonic(),0.0)

    def measure():
        movement={}
        local_movement={}
        for side,(before,points) in anchors.items():
            raw=jpegs.get(side)
            old,last_updated,previous_motion=cache[side]
            if not raw: raise RuntimeError('微调中摄像头无帧，已停车')
            if raw==old:
                if time.monotonic()-last_updated>.45:
                    raise RuntimeError('微调中双目画面冻结，已停车')
                movement[side]=previous_motion
                local_movement[side]=local_cache[side]
                continue
            image=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
            if image is None: raise RuntimeError('微调中画面损坏，已停车')
            gray=cv2.resize(image,(640,480))
            after,status,_=cv2.calcOpticalFlowPyrLK(before,gray,points,None,
                winSize=(21,21),maxLevel=3)
            if after is None or status is None:
                raise RuntimeError('微调中光流无效，已停车')
            valid=status.reshape(-1).astype(bool)
            source=points.reshape(-1,2)[valid]
            target=after.reshape(-1,2)[valid]
            if len(source)<16: raise RuntimeError('微调中稳定特征不足，已停车')
            transform,inliers=cv2.estimateAffinePartial2D(source,target,
                method=cv2.RANSAC,ransacReprojThreshold=3.0)
            if transform is None or inliers is None or float(inliers.mean())<.55:
                raise RuntimeError('微调中画面运动不一致，已停车')
            accepted=inliers.reshape(-1).astype(bool)
            magnitude=float(np.median(np.linalg.norm(target[accepted]-source[accepted],axis=1)))
            if not math.isfinite(magnitude): raise RuntimeError('微调光流数值无效')
            cache[side]=(raw,time.monotonic(),magnitude)
            movement[side]=round(magnitude,2)
            if target_centers is not None:
                # Near-object parallax may be much larger than background
                # motion. Local flow is STOP-only, never target identification.
                x,y=map(float,target_centers[side])
                nearby=(np.abs(source[:,0]-x)<=50)&(np.abs(source[:,1]-y)<=50)
                if int(nearby.sum())>=4:
                    local_cache[side]=float(np.percentile(
                        np.linalg.norm(target[nearby]-source[nearby],axis=1),75))
                else:
                    local_cache[side]=0.0  # background guard still mandatory
            local_movement[side]=round(local_cache[side],2)
        values=list(movement.values())
        return {'stop':min(values)>=movement_limit_px or max(values)>=1.5*movement_limit_px
                       or target_centers is not None and max(local_movement.values())>=movement_limit_px,
                'movement_px':movement,'target_region_movement_px':local_movement}
    return measure


def _command_worker():
    global _arm_jobs_pending
    while True:
        job = _cmd_queue.get()
        if job is None:
            break
        func, args = job
        try:
            func(*args)
        except Exception as exc:
            with _state_lock:
                _state["last_error"] = str(exc)
        finally:
            with _queue_gate:
                _arm_jobs_pending -= 1


_command_worker_thread = threading.Thread(
    target=_command_worker, daemon=True, name="arm-command-worker",
)
_command_worker_thread.start()


def enqueue_command(func, *args) -> dict:
    """把机械臂指令交给后台串行执行线程, HTTP 立即返回。

    所有指令 (含夹取流程) 都走同一个队列, 顺序与点击顺序一致;
    #ARMG 是绝对脉宽指令, 即使中途被 #STOP!/复位打断也不会累积误差。
    """
    global _arm_jobs_pending
    with _queue_gate:
        if _micro_motion_active:
            raise RuntimeError("底盘正在短脉冲微调，机械臂指令暂不可入队")
        try:
            _cmd_queue.put_nowait((func, args))
            _arm_jobs_pending += 1
        except queue.Full:
            raise RuntimeError("指令队列已满, 请稍候再试")
    with _state_lock:
        pending = max(_cmd_queue.qsize() - 1, 0)
    return {"pending": pending, "state": snapshot()}


def enqueue_home(ble_send) -> dict:
    """复位: 清空待执行队列并立即执行 #HOME!。"""
    global _arm_jobs_pending
    with _queue_gate:
        if _micro_motion_active:
            raise RuntimeError("底盘正在微调，不能同时复位机械臂")
        with _cmd_queue.mutex:
            removed = len(_cmd_queue.queue)
            _cmd_queue.queue.clear()
            _arm_jobs_pending -= removed
        _cmd_queue.put_nowait((send_home, (ble_send,)))
        _arm_jobs_pending += 1
    with _state_lock:
        pending = max(_cmd_queue.qsize() - 1, 0)
    return {"pending": pending, "state": snapshot()}

_exec_lock = threading.Lock()
_exec_status = {
    "active": False, "phase": "idle", "message": "",
    "params": {"s0": 1500, "s1": 1700, "s2": 2000},
    "user_stopped": False,
}
_exec_cancel_generation = 0


# ── 状态与指令 ───────────────────────────────────────────

def snapshot() -> dict:
    with _state_lock:
        return dict(_state)


def reset_state(reason: str = "home") -> dict:
    with _state_lock:
        _state.update(
            s0=1500, s1=1700, s2=2000,
            claw_value=None, known=True, last_command="#HOME!",
            updated_at=time.time(),
        )
        return snapshot()


def send_home(ble_send) -> str:
    """链式发送 #HOME! 直到模型回到 HOME (旧底板单条指令行程受限)。"""
    home = {s["key"]: int(s["home"]) for s in SERVOS}
    with _state_lock:
        current = {k: int(_state[k]) for k in SERVO_KEYS}
    guard = 0
    while True:
        step = {}
        max_duration = 0.0
        for k in SERVO_KEYS:
            step[k], duration = _step_toward(current[k], home[k])
            max_duration = max(max_duration, duration)
        _pace_and_wait(max_duration, ble_send)
        result = ble_send("#HOME!") if ble_send else "NO_BLE"
        if result != "OK":
            return result
        with _state_lock:
            _state.update(step)
            _state["known"] = True
            _state["last_command"] = "#HOME!"
            _state["updated_at"] = time.time()
        if all(step[k] == home[k] for k in SERVO_KEYS):
            _heartbeat_wait(ble_send, max_duration / 1000.0)
            break
        current = step
        guard += 1
        if guard > 20:
            raise RuntimeError("#HOME! 链过长, 已中止")
    reset_state("home")
    return "OK"


def _build_grasp_command(state: dict) -> str:
    return "#ARMG{:04d}{:04d}{:04d}!".format(
        int(state["s0"]), int(state["s1"]), int(state["s2"]),
    )


def send_grasp_pose(ble_send, target: dict) -> dict:
    """链式发送 #ARMG 绝对脉宽指令直到模型到达 target。

    旧底板把每条指令当"限时行程", 单条走不完的大行程按固件同一截断规则
    分步补发 (绝对指令幂等, 链式重发安全收敛)。
    """
    target = {k: int(round(float(target[k]))) for k in SERVO_KEYS}
    with _state_lock:
        current = {k: int(_state[k]) for k in SERVO_KEYS}
    guard = 0
    while True:
        step = {}
        max_duration = 0.0
        for k in SERVO_KEYS:
            step[k], duration = _step_toward(current[k], target[k])
            max_duration = max(max_duration, duration)
        command = _build_grasp_command(step)
        _pace_and_wait(max_duration, ble_send)
        result = ble_send(command) if ble_send else "NO_BLE"
        if result != "OK":
            raise ConnectionError(f"机械臂指令未写入 (BLE: {result}): {command}")
        with _state_lock:
            _state.update(step)
            _state["known"] = True
            _state["last_command"] = command
            _state["updated_at"] = time.time()
        if all(step[k] == target[k] for k in SERVO_KEYS):
            # 最后一条指令的运动窗口内也要保活 (防看门狗冻结)
            _heartbeat_wait(ble_send, max_duration / 1000.0)
            break
        current = step
        guard += 1
        if guard > 20:
            raise RuntimeError("行程链过长, 已中止 (请先 #HOME! 复位)")
    return snapshot()


def send_servo_set(ble_send, key: str, value: float) -> dict:
    """把某个舵机设为绝对脉宽 (钳制到该舵机限位)。"""
    spec = next(s for s in SERVOS if s["key"] == key)
    value = int(min(max(int(value), spec["min"]), spec["max"]))
    with _state_lock:
        target = {k: int(_state[k]) for k in SERVO_KEYS}
    target[key] = value
    return send_grasp_pose(ble_send, target)


def send_servo_delta(ble_send, key: str, delta: float) -> dict:
    with _state_lock:
        value = int(_state[key]) + int(delta)
    return send_servo_set(ble_send, key, value)


def send_claw(ble_send, value: int) -> dict:
    value = max(0, min(100, int(value)))
    command = f"#ARMC{value}!"
    _pace_and_wait(900.0, ble_send)
    result = ble_send(command) if ble_send else "NO_BLE"
    if result != "OK":
        raise ConnectionError(f"夹爪指令未写入 (BLE: {result}): {command}")
    with _state_lock:
        _state["claw_value"] = value
        _state["last_command"] = command
        _state["updated_at"] = time.time()
    _heartbeat_wait(ble_send, 0.9)
    return snapshot()


# ── 样本存取 ────────────────────────────────────────────

def _dataset_dir() -> Path:
    path = DATASET_ROOT / "samples"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_sample(object_name: str, left_center, right_center,
                left_jpeg: bytes, right_jpeg: bytes,
                state: dict, notes: str = "") -> dict:
    sample_id = (
        f"teach_{time.strftime('%Y%m%d_%H%M%S')}_"
        f"{time.time_ns() % 1_000_000:06d}_{uuid.uuid4().hex[:8]}"
    )
    meta = {
        "sample_id": sample_id,
        "created_at": time.time(),
        "object_name": str(object_name or "").strip(),
        "left_center": [round(float(left_center[0]), 1), round(float(left_center[1]), 1)],
        "right_center": [round(float(right_center[0]), 1), round(float(right_center[1]), 1)],
        "s0": int(state["s0"]),
        "s1": int(state["s1"]),
        "s2": int(state["s2"]),
        "claw_value": state.get("claw_value"),
        "notes": str(notes or "")[:500],
    }
    (_dataset_dir() / f"{sample_id}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (_dataset_dir() / f"{sample_id}_left.jpg").write_bytes(left_jpeg)
    (_dataset_dir() / f"{sample_id}_right.jpg").write_bytes(right_jpeg)
    return meta


def list_samples() -> list[dict]:
    result = []
    for path in _dataset_dir().glob("*.json"):
        try:
            result.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    result.sort(key=lambda item: (item.get("created_at", 0), item["sample_id"]))
    return result


def delete_sample(sample_id: str) -> None:
    for suffix in ("", "_left.jpg", "_right.jpg"):
        (_dataset_dir() / f"{Path(sample_id).name}{suffix}").unlink(missing_ok=True)


def export_zip() -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for meta in list_samples():
            sample_id = meta["sample_id"]
            archive.writestr(
                f"samples/{sample_id}.json",
                json.dumps(meta, ensure_ascii=False, indent=2),
            )
            for camera in ("left", "right"):
                image = _dataset_dir() / f"{sample_id}_{camera}.jpg"
                if image.exists():
                    archive.write(image, f"samples/{sample_id}_{camera}.jpg")
    return output.getvalue()


# ── 拼接帧 (LEFT|RIGHT 标注, 与 Qwen 双目标 prompt 匹配) ──

def _halves_b64(jpegs: dict) -> dict:
    frames = []
    encoded_halves = {}
    for camera, label in (("left", "LEFT"), ("right", "RIGHT")):
        jpeg = jpegs.get(camera) or b""
        if not jpeg or len(jpeg) <= 1000:
            raise RuntimeError(f"{camera} 相机无可用帧")
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"{camera} 相机帧解码失败")
        frame = cv2.resize(frame, (640, 480))
        ok, half = cv2.imencode(".jpg", frame,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError(f"{camera} 相机帧编码失败")
        encoded_halves[camera] = base64.b64encode(half.tobytes()).decode("ascii")
        labeled = frame.copy()
        cv2.rectangle(labeled, (0, 0), (125, 38), (0, 0, 0), -1)
        cv2.putText(labeled, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)
        frames.append(labeled)
    combined = np.hstack(frames)
    ok, encoded = cv2.imencode(
        ".jpg", combined, [int(cv2.IMWRITE_JPEG_QUALITY), 85],
    )
    if not ok:
        raise RuntimeError("拼接帧编码失败")
    return {
        "left": encoded_halves["left"],
        "right": encoded_halves["right"],
        "stitched": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }


def _http_json(url: str, payload: dict, timeout: float = 90.0) -> dict:
    request_obj = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request_obj, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            body = json.loads(raw)
            detail = str(body.get("error") or body.get("detail") or raw)
        except (json.JSONDecodeError, AttributeError):
            detail = raw or str(exc)
        raise RuntimeError(detail) from exc


# ── 模型夹取执行 (后台线程) ──────────────────────────────

def _s2_preclose_nudge_target(target: dict) -> dict | None:
    """Apply operator-calibrated S0 and mirrored S2/S3 offsets once."""
    from grasp_pose_correction import corrected_grasp_pose
    return corrected_grasp_pose(target, SERVOS)


def _run_grasp_sequence(ble_send, params: dict) -> None:
    """#HOME! -> #ARMG 舵机就位 -> 闭合 -> 复位 #HOME!。任何失败优先复位。"""
    def set_status(**values):
        with _state_lock:
            _exec_status.update(values)

    def aborted() -> bool:
        with _state_lock:
            return bool(_exec_status.get("user_stopped"))

    target = {k: int(round(float(params.get(k, 1500)))) for k in SERVO_KEYS}
    for k in SERVO_KEYS:
        spec = next(s for s in SERVOS if s["key"] == k)
        target[k] = int(min(max(target[k], spec["min"]), spec["max"]))
    try:
        nudge_target = _s2_preclose_nudge_target(target)
        set_status(active=True, phase="home", message="复位 #HOME! 并等待 3s",
                   params={k: target[k] for k in SERVO_KEYS},
                   user_stopped=False)
        result = send_home(ble_send)
        if result != "OK":
            raise ConnectionError(f"#HOME! 未写入 (BLE: {result})")
        _heartbeat_wait(ble_send, 3.0)
        if aborted():
            raise RuntimeError("用户已停止")

        set_status(phase="move", message=(
            f"舵机就位: s0={target['s0']} s1={target['s1']} s2={target['s2']}"
        ))
        send_grasp_pose(ble_send, target)
        _heartbeat_wait(ble_send, 1.5)
        if aborted():
            raise RuntimeError("用户已停止")
        set_status(phase="settle", message="就位完成, 等待稳定")
        _heartbeat_wait(ble_send, 1.5)
        if aborted():
            raise RuntimeError("user stopped before pre-close nudge")
        if nudge_target is not None:
            set_status(phase="lower_nudge", params=dict(nudge_target),
                       message=f"Model pose reached; pre-close correction s0={nudge_target['s0']} s2={nudge_target['s2']}")
            send_grasp_pose(ble_send, nudge_target)
            _heartbeat_wait(ble_send, 0.4)
            if aborted():
                raise RuntimeError("user stopped after pre-close nudge")

        set_status(phase="grasp", message=f"闭合夹爪 (C{CLAW_CLOSE})")
        send_claw(ble_send, CLAW_CLOSE)
        _heartbeat_wait(ble_send, 1.2)

        # 抬升回中: 用 #ARMG 回 HOME 舵机值, 不碰夹爪 (夹爪保持闭合夹住物体)。
        # 注意不能用 #HOME! — 它的复位包含 j5=1200, 会马上把夹爪张开。
        set_status(phase="home_end", message="抬升回中 (保持夹爪闭合)")
        home_pose = {s["key"]: int(s["home"]) for s in SERVOS}
        send_grasp_pose(ble_send, home_pose)
        set_status(active=False, phase="done",
                   message="模型夹取流程完成 (夹爪保持闭合, 等待任务指令)")
    except Exception as exc:
        try:
            send_home(ble_send)
        except Exception:
            pass
        set_status(active=False, phase="error", user_stopped=False,
                   message=f"夹取流程中止, 已尝试复位: {exc}")


def start_predict_grasp(ble_send, teach_url: str,
                        object_name: str, jpegs: dict) -> str:
    """拼接帧 -> 3090 预测 -> 后台执行。返回给用户的消息。"""
    with _exec_lock:
        if _exec_status.get("active"):
            raise RuntimeError("已有夹取流程在执行")
        _require_taught_camera_order()
        with _state_lock:
            cancel_generation = _exec_cancel_generation
        halves = _halves_b64(jpegs)
        response = _http_json(
            f"{teach_url.rstrip('/')}/api/model_grasp/predict",
            {"image_b64": halves["stitched"], "target": object_name},
            timeout=120,
        )
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "预测失败"))
        region_ok = region_accepts(response)
        if not region_ok:
            raise RuntimeError(
                "目标尚未同时进入左右二维示教区或25px外沿，机械臂保持停止；"
                f"左右误差 {response.get('view_hull_distances_px', {})}px，"
                f"目标中心 {response.get('centers', {})}"
            )
        with _state_lock:
            if _exec_cancel_generation != cancel_generation:
                raise RuntimeError("预测期间收到急停或复位，取消本次模型夹取")
        params = response.get("parameters", {})
        centers = response.get("centers", {})
        # 夹取流程作为单个任务进队列: 期间用户点的手动指令排在其后, 不会插队
        enqueue_command(_run_grasp_sequence, ble_send, params)
        return (
            f"预测成功: 左中心 {centers.get('left')} 右中心 {centers.get('right')} "
            f"-> 舵机 {params}; 已入队, 后台开始执行"
        )


# ── HTML 页面 ───────────────────────────────────────────

ARM_TEACH_HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>机械臂示教夹取</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0b1320;color:#e9eef6;font-family:Arial,"Microsoft YaHei",sans-serif}
.top{display:flex;gap:14px;align-items:center;padding:10px 14px;background:#111d2c;position:sticky;top:0;z-index:3}.top b{color:#4fd1c5}
.grid{display:grid;grid-template-columns:minmax(560px,1fr) 400px;gap:12px;padding:12px}
.card{background:#111c2a;border:1px solid #33465d;border-radius:10px;padding:11px;margin-bottom:10px}
.cams{display:grid;grid-template-columns:1fr 1fr;gap:8px}.stage{position:relative;background:#050a13}
.stage img{width:100%;display:block}.stage canvas{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none}
.label{position:absolute;left:7px;top:7px;background:#111c2add;padding:3px 7px;color:#4fd1c5;z-index:2}
.row{display:flex;gap:7px;flex-wrap:wrap;align-items:center;margin:7px 0}
button,select,input{background:#172234;color:#edf2f7;border:1px solid #52677f;border-radius:6px;padding:7px}
.go{background:#065f46}.blue{background:#1e3a5f}.bad{background:#6b1830}
.status{white-space:pre-wrap;font-size:12px;max-height:200px;overflow:auto;background:#0a111d;padding:8px;font-family:Consolas,monospace}
.ok{color:#34d399;font-weight:700}.offline{color:#fb7185;font-weight:700}
.ctrl{border:1px solid #3a5268;border-radius:8px;padding:8px;margin-bottom:8px}
.ctrl b{color:#facc15}.small{font-size:11px;color:#a9b8ca}
</style></head><body>
<div class="top"><b>机械臂示教夹取（独立系统 v2·舵机直控）</b>
<span id="bleStatus" class="offline">BLE 检查中…</span><span id="modelStatus" class="small">模型状态…</span>
<a href="/">返回</a></div>
<div class="grid"><div>
<div class="card"><h3>① 放置物块 → 识别中心 → 调舵机 → 保存样本</h3>
<div class="row"><label>物体名称 <input id="objectName" value="红色小物块" style="width:130px"></label>
<button class="go" onclick="detect()">识别物块中心 (Qwen)</button>
<button onclick="freezeOff()">恢复实时预览</button></div>
<div class="cams"><div><div class="stage"><span class="label">物理左目 · video2（模型 right）</span><img id="img-right" src="/video/right"><canvas id="cv-right"></canvas></div></div>
<div><div class="stage"><span class="label">物理右目 · video0（模型 left）</span><img id="img-left" src="/video/left"><canvas id="cv-left"></canvas></div></div></div>
<div class="row"><span id="centers" class="small">尚未识别</span></div>
<div class="row"><button class="go" onclick="saveSample()">保存样本（左右图+中心+三舵机脉宽）</button>
<span id="sampleCount" class="small"></span></div>
</div>
<div class="card"><h3>③ 训练与夹取</h3>
<div class="row"><button class="blue" onclick="modelStatus()">刷新模型状态</button>
<button class="bad" onclick="predictGrasp()">模型预测并夹取</button>
<a href="/api/arm-teach/export"><button>下载样本 ZIP（供训练）</button></a></div>
<div id="modelInfo" class="status">模型状态…</div>
<div class="small">训练在 3090 上执行: python train_arm_mlp.py --dataset &lt;解压后的样本目录&gt;。
物块中心落在采集点凸包之外时, 夹取会直接拒绝("无法实现")。</div>
</div>
</div><div>
<div class="card"><h3>② 舵机手动调节 (0/1/2 直控, 3 号镜像联动)</h3>
<div id="controls"></div>
<div class="row"><button onclick="rawClaw(95)">夹爪张开</button>
<button class="bad" onclick="rawClaw(10)">夹爪闭合</button>
<button class="blue" onclick="armHome()">#HOME! 复位</button>
<button class="bad" onclick="armStop()">#STOP! 停止</button></div>
<div id="armState" class="status">状态…</div>
<div class="small">指令为 #ARMG 绝对脉宽 (s0 底座左右 / s1 关节1 / s2 关节2), 3 号舵机=3100-s2 自动镜像, 4 号不动。每次发送带全三轴当前值, 设值与逐步调完全等价。开机后请先 #HOME! 复位。</div>
</div>
<div class="card"><h3>样本列表</h3><div class="row"><button onclick="refreshSamples()">刷新</button></div>
<div id="samples" class="status">加载中…</div></div>
</div></div>
<script>
let FROZEN=false,CTX=null;
const E=id=>document.getElementById(id);
async function post(url,data){let r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data||{})});let j=await r.json();if(!r.ok||j.ok===false)throw Error(j.error||r.statusText);return j}
function say(x){E('armState').textContent=x}
async function detect(){try{say('识别中 (Qwen 约 2-5s)…');let j=await post('/api/arm-teach/detect',{target:E('objectName').value});FROZEN=true;CTX=j;let c=j.centers;E('centers').textContent='左中心 '+JSON.stringify(c.left)+' | 右中心 '+JSON.stringify(c.right)+' | 已锁定, 保存前请勿移动物块';['left','right'].forEach(x=>{let im=document.getElementById('img-'+x);im.onload=()=>drawCross(x);im.src='data:image/jpeg;base64,'+j.frames[x]});say('识别成功 (画面已冻结并标注十字)');}catch(e){say('识别失败: '+e.message)}}
function drawCross(c){let im=document.getElementById('img-'+c),cv=document.getElementById('cv-'+c);cv.width=im.clientWidth;cv.height=im.clientHeight;let ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);if(!CTX||!CTX.centers||!CTX.centers[c])return;let p=CTX.centers[c],sx=cv.width/640,sy=cv.height/480,x=p[0]*sx,y=p[1]*sy;ctx.strokeStyle='#22d3ee';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x-15,y);ctx.lineTo(x+15,y);ctx.moveTo(x,y-15);ctx.lineTo(x,y+15);ctx.stroke();ctx.beginPath();ctx.arc(x,y,11,0,Math.PI*2);ctx.stroke();ctx.fillStyle='#22d3ee';ctx.font='13px sans-serif';ctx.fillText(c.toUpperCase()+' ('+Math.round(p[0])+','+Math.round(p[1])+')',x+14,y-10)}
function freezeOff(){FROZEN=false;CTX=null;E('centers').textContent='实时预览中';['left','right'].forEach(x=>{let cv=document.getElementById('cv-'+x);cv.getContext('2d').clearRect(0,0,cv.width,cv.height);document.getElementById('img-'+x).src='/video/'+x})}
window.onresize=()=>{if(FROZEN){drawCross('left');drawCross('right')}}
async function saveSample(){if(!CTX)return say('请先识别物块中心');try{let j=await post('/api/arm-teach/save',{object_name:E('objectName').value,left_center:CTX.centers.left,right_center:CTX.centers.right});say('样本已保存: '+j.sample.sample_id);refreshSamples()}catch(e){say('保存失败: '+e.message)}}
async function nudge(key,delta){try{let j=await post('/api/arm-teach/move',{axis:key,delta:delta});renderState(j.state);say('已入队: '+key+' '+delta+(j.queued?' (前面还有 '+j.queued+' 条)':''))}catch(e){say('指令失败: '+e.message)}}
async function setAxis(key,target){try{let j=await post('/api/arm-teach/move',{axis:key,target:target});renderState(j.state);say('已入队: '+key+'='+target+(j.queued?' (前面还有 '+j.queued+' 条)':''))}catch(e){say('指令失败: '+e.message)}}
async function rawClaw(v){try{let j=await post('/api/arm-teach/claw',{value:v});renderState(j.state);say('已入队: 夹爪'+(v===95?'张开':'闭合'))}catch(e){say('夹爪失败: '+e.message)}}
async function armHome(){try{let j=await post('/api/arm-teach/home',{});renderState(j.state);say('已入队: #HOME! 复位 (清空待执行指令)')}catch(e){say('复位失败: '+e.message)}}
async function armStop(){try{let j=await post('/api/arm-teach/stop',{});say(j.result==='OK'?'已发送 #STOP!':'停止失败: '+j.result)}catch(e){say(e.message)}}
function renderState(s){E('armState').textContent='s0='+s.s0+'  s1='+s.s1+'  s2='+s.s2+'  claw='+s.claw_value+'\nknown='+s.known+'  last='+s.last_command+(s.last_error?'\n错误: '+s.last_error:'')}
async function pollState(){try{let r=await fetch('/api/arm-teach/state',{cache:'no-store'}).then(x=>x.json());renderState(r.state);let b=await fetch('/api/status',{cache:'no-store'}).then(x=>x.json());let e=E('bleStatus');e.textContent=b.ble?'BLE 已连接':'BLE 未连接';e.className=b.ble?'ok':'offline'}catch(e){}}
async function refreshSamples(){try{let j=await fetch('/api/arm-teach/samples',{cache:'no-store'}).then(x=>x.json());E('sampleCount').textContent='样本数 '+j.samples.length;let el=E('samples');el.innerHTML='';j.samples.forEach(s=>{let row=document.createElement('div');row.style.cssText='margin:3px 0;border-bottom:1px solid #22334a;padding-bottom:3px';row.innerHTML='<span style="color:#4fd1c5">'+s.sample_id.slice(6,18)+'</span> | L'+JSON.stringify(s.left_center)+' R'+JSON.stringify(s.right_center)+' | s0='+s.s0+' s1='+s.s1+' s2='+s.s2+' <button class="bad" onclick="delSample(\''+s.sample_id+'\')">删除</button>';el.appendChild(row)})}catch(e){E('samples').textContent='加载失败 '+e.message}}
async function delSample(id){if(!confirm('删除样本 '+id.slice(6,18)+'?'))return;try{await post('/api/arm-teach/delete',{sample_id:id});say('已删除');refreshSamples()}catch(e){say('删除失败: '+e.message)}}
async function modelStatus(){try{let j=await fetch('/api/arm-teach/model-status',{cache:'no-store'}).then(x=>x.json());let m=E('modelStatus');let t=E('modelInfo');if(j.trained){m.textContent='模型已训练 ('+j.num_samples+' 样本)';m.className='ok';t.textContent='物体: '+j.object_name+'\n样本: '+j.num_samples+' | 凸包点: '+j.hull_points+'\n测试MAE: '+JSON.stringify(j.test_mae)}else{m.textContent='模型未训练';m.className='offline';t.textContent='尚未训练: '+j.error}}catch(e){E('modelStatus').textContent='模型状态不可达';E('modelStatus').className='offline';E('modelInfo').textContent='错误: '+e.message}}
function dualRegionOK(j){let d=j.view_hull_distances_px||{};return j.grasp_region_mode==='dual_2d_shell'&&j.grasp_margin_px===25&&j.grasp_region_ok===true&&(j.inside_teaching_hull===true||j.projected_to_teaching_boundary===true)&&['left','right'].every(s=>typeof d[s]==='number'&&Number.isFinite(d[s])&&d[s]>=0&&d[s]<=25)}
async function predictGrasp(){try{say('预测中 (不动臂)…');let j=await post('/api/arm-teach/predict',{target:E('objectName').value});if(!dualRegionOK(j)){say('双目尚未同时进入二维抓取区或25px外沿；左右误差 '+JSON.stringify(j.view_hull_distances_px||{})+'px，机械臂不会启动');return}let p=j.parameters;if(!confirm('左右二维抓取区检查通过'+(j.inside_teaching_hull===true?'':'（使用已有示教边界姿态）')+'。预测舵机: s0='+p.s0+' s1='+p.s1+' s2='+p.s2+'\n确认执行? (将先 #HOME! 复位)'))return;let e=await post('/api/arm-teach/predict-grasp',{target:E('objectName').value});say(e.message);pollExec()}catch(e){say('预测失败: '+e.message)}}
async function pollExec(){let r=await fetch('/api/arm-teach/exec',{cache:'no-store'}).then(x=>x.json());say('['+r.status.phase+'] '+r.status.message);if(r.status.active)setTimeout(pollExec,2000)}
(async function(){let j=await fetch('/api/arm-teach/state',{cache:'no-store'}).then(x=>x.json());renderState(j.state);
let html='';j.servos.forEach((a,i)=>{let opts=a.steps.map(s=>'<option>'+s+'</option>').join('');let presets=(a.presets||[]).map(p=>'<button onclick="setAxis(\''+a.key+'\','+p+')">'+p+'</button>').join('');html+='<div class="ctrl"><b>'+a.name+'</b> <span class="small">(范围 '+a.min+'~'+a.max+', HOME '+a.home+', 当前 <span id="cur'+i+'">'+a.home+'</span>)</span><div class="row">常用值: '+presets+'</div><div class="row"><select id="st'+i+'">'+opts+'</select><button onclick="nudge(\''+a.key+'\',-1*+E(\'st'+i+'\').value)">-</button><button onclick="nudge(\''+a.key+'\',+1*+E(\'st'+i+'\').value)">+</button><input id="set'+i+'" type="number" min="'+a.min+'" max="'+a.max+'" style="width:80px" placeholder="绝对值"><button onclick="setAxis(\''+a.key+'\',+E(\'set'+i+'\').value)">设值</button><button onclick="nudge(\''+a.key+'\',-10)">-10</button><button onclick="nudge(\''+a.key+'\',+10)">+10</button></div></div>'});E('controls').innerHTML=html;
let orender=renderState;renderState=(s)=>{orender(s);j.servos.forEach((a,i)=>{let el=E('cur'+i);if(el)el.textContent=s[a.key]})};
refreshSamples();modelStatus();setInterval(pollState,1500);})();
</script></body></html>'''


# ── Flask Blueprint ─────────────────────────────────────

def register(app, jpegs: dict, ble_send, teach_url: str = DEFAULT_TEACH_URL):
    bp = Blueprint("arm_teach", __name__)

    def authorised_local_micro() -> bool:
        if request.remote_addr not in {"127.0.0.1", "::1"}:
            return False
        token_path = Path(os.environ.get(
            "FINAL_WORK_HARDWARE_TOKEN_FILE",
            "/home/robot/dachuang/track/final_work/data/hardware_token"))
        try:
            expected = token_path.read_text(encoding="utf-8").strip()
        except OSError:
            expected = ""
        supplied = str(request.headers.get("X-Robot-Safety-Token") or "")
        return bool(expected and hmac.compare_digest(supplied, expected))

    @bp.get("/arm-teach")
    def page():
        return Response(ARM_TEACH_HTML, mimetype="text/html")

    @bp.post("/api/arm-teach/chassis-pulse")
    def chassis_pulse():
        """Local, token-guarded pulse through this process's existing BLE."""
        global _micro_motion_active, _micro_active_id
        if not authorised_local_micro():
            return jsonify(ok=False, error="仅允许本机携带有效令牌调用"), 403
        data = request.get_json(silent=True) or {}
        action = str(data.get("action") or "")
        pulse_id = str(data.get("pulse_id") or "")
        try:
            if uuid.UUID(pulse_id).hex != pulse_id:
                raise ValueError("noncanonical pulse id")
        except ValueError:
            return jsonify(ok=False, error="脉冲 ID 无效"), 400
        try:
            duration = float(data.get("duration_s"))
            speed_mm_s = float(data.get("speed_mm_s", 405))
            angular_mrad_s = float(data.get("angular_mrad_s", 450))
            motion_limit = data.get("visual_motion_limit_px")
            target_centers = data.get('target_centers')
            if target_centers is not None:
                for side in ('left','right'):
                    xy=target_centers[side]
                    if len(xy)!=2 or not all(math.isfinite(float(v)) for v in xy) \
                            or not 0<=float(xy[0])<640 or not 0<=float(xy[1])<480:
                        raise ValueError('invalid target center')
            if motion_limit is not None:
                motion_limit = float(motion_limit)
                if not math.isfinite(motion_limit) or not 1 <= motion_limit <= 8:
                    raise ValueError("invalid visual limit")
            _micro_command(action, speed_mm_s, angular_mrad_s)
        except (TypeError, ValueError, KeyError):
            return jsonify(ok=False, error="微调速度或时长无效"), 400
        if action not in _MICRO_COMMANDS or not math.isfinite(duration) \
                or not 0.4 <= duration <= 0.8:
            return jsonify(ok=False, error="微调动作或时长越界"), 400
        if not _exec_lock.acquire(blocking=False):
            return jsonify(ok=False, error="机械臂正在预测或夹取"), 409
        try:
            with _micro_state_lock:
                if pulse_id in _micro_cancelled_ids:
                    return jsonify(ok=False, error="脉冲已取消"), 409
                with _queue_gate:
                    if (_micro_motion_active or _arm_jobs_pending > 0
                            or _exec_status.get("active")):
                        return jsonify(ok=False, error="机械臂或底盘已有动作"), 409
                    _micro_cancel.clear()
                    _micro_motion_active = True
                    _micro_active_id = pulse_id
            try:
                feedback = _make_turn_feedback(jpegs, motion_limit or 4.0, target_centers)
                pulse = _run_chassis_pulse(ble_send, action, duration, speed_mm_s,
                                           angular_mrad_s, movement_feedback=feedback)
                return jsonify(ok=True, action=action,
                               duration_s=duration, pulse_id=pulse_id, pulse=pulse)
            finally:
                with _micro_state_lock:
                    with _queue_gate:
                        _micro_motion_active = False
                        _micro_active_id = None
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 503
        finally:
            _exec_lock.release()

    @bp.post("/api/arm-teach/chassis-cancel")
    def chassis_cancel():
        """Cancel a pulse even if its POST has not reached the preview yet."""
        if not authorised_local_micro():
            return jsonify(ok=False, error="仅允许本机携带有效令牌调用"), 403
        pulse_id = str((request.get_json(silent=True) or {}).get("pulse_id") or "")
        try:
            if uuid.UUID(pulse_id).hex != pulse_id:
                raise ValueError("noncanonical pulse id")
        except ValueError:
            return jsonify(ok=False, error="脉冲 ID 无效"), 400
        with _micro_state_lock:
            if len(_micro_cancelled_ids) >= 128:
                _micro_cancelled_ids.pop()
            _micro_cancelled_ids.add(pulse_id)
            active = _micro_active_id == pulse_id
            if active:
                _micro_cancel.set()
        if active:
            result = ble_send("#STOP!") if ble_send else "NO_BLE"
            if result != "OK":
                return jsonify(ok=False, error=f"STOP 未确认：{result}"), 503
        return jsonify(ok=True, active=active)

    @bp.route("/api/arm-teach/state", methods=["GET", "POST"])
    def state():
        with _state_lock:
            pending = max(_cmd_queue.qsize() - 1, 0)
        return jsonify(ok=True, state=snapshot(), servos=SERVOS,
                       pending=pending)

    @bp.post("/api/arm-teach/move")
    def move():
        data = request.get_json(silent=True) or {}
        axis = str(data.get("axis", ""))
        if axis not in SERVO_KEYS:
            return jsonify(ok=False, error="未知舵机: 用 s0/s1/s2"), 400
        try:
            if "target" in data and data.get("target") not in (None, ""):
                spec = next(s for s in SERVOS if s["key"] == axis)
                value = int(min(max(int(float(data["target"])),
                                   spec["min"]), spec["max"]))
                result = enqueue_command(send_servo_set, ble_send, axis, value)
            else:
                delta = int(float(data.get("delta", 0)))
                result = enqueue_command(send_servo_delta, ble_send, axis, delta)
        except (RuntimeError, TypeError, ValueError) as exc:
            return jsonify(ok=False, error=str(exc)), 400
        return jsonify(ok=True, queued=result["pending"], state=result["state"])

    @bp.post("/api/arm-teach/claw")
    def claw():
        data = request.get_json(silent=True) or {}
        try:
            result = enqueue_command(
                send_claw, ble_send, int(data.get("value", CLAW_OPEN)),
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            return jsonify(ok=False, error=str(exc)), 400
        return jsonify(ok=True, queued=result["pending"], state=result["state"])

    @bp.post("/api/arm-teach/home")
    def home():
        global _exec_cancel_generation
        with _state_lock:
            _exec_cancel_generation += 1
        try:
            result = enqueue_home(ble_send)
        except RuntimeError as exc:
            return jsonify(ok=False, error=str(exc)), 503
        with _state_lock:
            if _exec_status.get("active"):
                _exec_status["user_stopped"] = True
        return jsonify(ok=True, queued=result["pending"], state=result["state"])

    @bp.post("/api/arm-teach/stop")
    def stop():
        global _exec_cancel_generation, _arm_jobs_pending
        _micro_cancel.set()
        with _state_lock:
            _exec_cancel_generation += 1
        # 紧急停止: 绕过队列直接发 #STOP!, 并清空待执行指令
        with _queue_gate:
            with _cmd_queue.mutex:
                removed = len(_cmd_queue.queue)
                _cmd_queue.queue.clear()
                _arm_jobs_pending -= removed
        result = ble_send("#STOP!") if ble_send else "NO_BLE"
        with _state_lock:
            if _exec_status.get("active"):
                _exec_status["user_stopped"] = True
        return jsonify(ok=True, result=result)

    @bp.post("/api/arm-teach/detect")
    def detect():
        data = request.get_json(silent=True) or {}
        try:
            halves = _halves_b64(jpegs)
            response = _http_json(
                f"{teach_url.rstrip('/')}/api/teach/detect",
                {"image_b64": halves["stitched"],
                 "target": str(data.get("target", "红色小物块"))},
                timeout=120,
            )
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 502
        if not response.get("ok"):
            return jsonify(ok=False, error=response.get("error")), 422
        return jsonify(ok=True, centers=response["centers"],
                       frames={"left": halves["left"], "right": halves["right"]})

    @bp.post("/api/arm-teach/save")
    def save():
        data = request.get_json(silent=True) or {}
        try:
            left = data.get("left_center")
            right = data.get("right_center")
            if not isinstance(left, (list, tuple)) or len(left) < 2 \
                    or not isinstance(right, (list, tuple)) or len(right) < 2:
                raise ValueError("left_center/right_center required")
            if not snapshot()["known"]:
                raise ValueError("机械臂状态未知: 请先 #HOME! 复位")
            meta = save_sample(
                str(data.get("object_name", "")),
                left, right,
                jpegs.get("left") or b"",
                jpegs.get("right") or b"",
                snapshot(),
                str(data.get("notes", "")),
            )
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 400
        return jsonify(ok=True, sample=meta)

    @bp.get("/api/arm-teach/samples")
    def samples():
        return jsonify(ok=True, samples=list_samples())

    @bp.post("/api/arm-teach/delete")
    def delete():
        data = request.get_json(silent=True) or {}
        delete_sample(str(data.get("sample_id", "")))
        return jsonify(ok=True)

    @bp.get("/api/arm-teach/export")
    def export():
        return Response(export_zip(), mimetype="application/zip",
                        headers={"Content-Disposition":
                                 "attachment; filename=arm_teach_samples.zip"})

    @bp.get("/api/arm-teach/model-status")
    def model_status_proxy():
        try:
            with urllib.request.urlopen(
                f"{teach_url.rstrip('/')}/api/model/status", timeout=8,
            ) as response:
                return jsonify(json.loads(response.read().decode("utf-8", "replace")))
        except Exception as exc:
            return jsonify({"trained": False, "error": f"模型服务不可达: {exc}"})

    @bp.post("/api/arm-teach/predict")
    def predict_only():
        """只预测不动臂: 返回三舵机脉宽供用户确认。"""
        data = request.get_json(silent=True) or {}
        try:
            _require_taught_camera_order()
            halves = _halves_b64(jpegs)
            response = _http_json(
                f"{teach_url.rstrip('/')}/api/model_grasp/predict",
                {"image_b64": halves["stitched"],
                 "target": str(data.get("target", "红色小物块"))},
                timeout=120,
            )
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 502
        if not response.get("ok"):
            return jsonify(ok=False, error=response.get("error")), 422
        return jsonify(
            ok=True, centers=response.get("centers"),
            target_boxes=response.get('target_boxes',{}),
            parameters=response.get("parameters"),
            inside_teaching_hull=response.get("inside_teaching_hull", False),
            grasp_region_ok=response.get("grasp_region_ok", False),
            grasp_margin_px=response.get("grasp_margin_px", 0),
            projected_to_teaching_boundary=response.get("projected_to_teaching_boundary", False),
            outside_distance_px=response.get("outside_distance_px"),
            nearest_hull_centers=response.get("nearest_hull_centers"),
            joint_hull_distance_px=response.get('joint_hull_distance_px'),
            grasp_region_mode=response.get('grasp_region_mode'),
            view_hull_distances_px=response.get('view_hull_distances_px'),
            view_hull_centers=response.get('view_hull_centers'),
            view_hull_polygons_px=response.get('view_hull_polygons_px'),
            inside_view_hulls=response.get('inside_view_hulls'),
        )

    @bp.post("/api/arm-teach/predict-grasp")
    def predict_grasp():
        data = request.get_json(silent=True) or {}
        try:
            message = start_predict_grasp(
                ble_send, teach_url,
                str(data.get("target", "红色小物块")), jpegs,
            )
        except Exception as exc:
            return jsonify(ok=False, error=str(exc)), 422
        return jsonify(ok=True, message=message)

    @bp.get("/api/arm-teach/exec")
    def exec_status():
        with _state_lock:
            return jsonify(ok=True, status=dict(_exec_status))

    from placement_extension import register_placement
    register_placement(bp, __import__('sys').modules[__name__], ble_send, jpegs)
    app.register_blueprint(bp)
    return bp
