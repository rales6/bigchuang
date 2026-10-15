"""视觉伺服抓取执行器 — 共用 BLE 连接。
通过 grasp_runner 模块级函数 start/stop 控制。
"""
import asyncio, json, threading, time

try:
    from .structured_log import EventLogger
except ImportError:
    from structured_log import EventLogger

EVENT_LOG = EventLogger("pi_runner", "grasp_runner.jsonl")

# 共享状态 (web_app 读取)
status = {"active":False,"phase":"idle","step":"","camera":"left","arm_cmd":{},"tracking":{},"error":"","log":[],"frame":0}
_lock = threading.RLock()

def _set(**kw):
    with _lock: status.update(kw)
def _log(msg):
    with _lock:
        status["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if len(status["log"])>50: status["log"]=status["log"][-50:]
    EVENT_LOG.event("runner_message", message=msg)

class Runner:
    def __init__(s, server, target, left_cam=0, w=640, h=480,
                 ble_send=None, capture=None, captures=None,
                 initial_camera="left"):
        s.server=server; s.target=target; s.lc=0; s.w=w; s.h=h; s._ble=ble_send; s._stop=False; s._release=False
        s._captures = dict(captures or {})
        if capture is not None and "left" not in s._captures:
            s._captures["left"] = capture
        s._active_camera = (initial_camera if initial_camera in s._captures
                            else next(iter(s._captures), "left"))
        s._compare_cameras = False
        s._last_no_frame_log = 0.0
        # 左右已交换: left=/dev/video2(MJPEG), right=/dev/video0(MJPEG+capture共享)
    def start(s):
        _set(active=True,phase="deploy",camera=s._active_camera,
             target=s.target,arm_cmd={},tracking={},error="",frame=0)
        _log(f"target: {s.target}")
        s._stop=False; threading.Thread(target=s._run, daemon=True).start()
    def stop(s):
        s._stop=True
        if s._ble: s._ble("#STOP!")
        _log("stopped")
    def request_release(s):
        s._release=True
        _log("release requested")

    def _run(s):
        try: asyncio.run(s._async())
        except Exception as e:
            _set(phase="error",error=str(e)); _log(f"err: {e}")
            EVENT_LOG.event("runner_error", error=repr(e))
        finally:
            _set(active=False)
            EVENT_LOG.event("runner_finished", phase=status.get("phase"), error=status.get("error"))
            if not getattr(s, "_homed_at_end", False):
                s._homed_at_end = True
                s._home("session ended")

    def _home(s, why=""):
        """复位机械臂到 HOME 位，卸载舵机力矩（防过热/防下次任务承接遗留位姿）。"""
        if not s._ble:
            return
        try:
            r = s._ble("#HOME!")
            EVENT_LOG.event("arm_home", why=why, result=r)
            _log(f"arm homed ({why})")
        except Exception as exc:
            EVENT_LOG.event("arm_home_error", why=why, error=repr(exc))

    async def _async(s):
        import websockets
        _log(f"connecting {s.server}")
        async with websockets.connect(s.server,max_size=5*1024*1024) as ws:
            await ws.send(json.dumps({
                "type": "start",
                "target_description": s.target,
                "active_camera": s._active_camera,
                "available_cameras": list(s._captures) or [s._active_camera],
            }))
            ack=json.loads(await ws.recv()); _log(f"3090: {ack.get('message',ack)}")
            EVENT_LOG.event("server_ack", response=ack)
            # 每次任务从 HOME 位开始，避免承接上一次报错遗留的前伸/低垂位姿
            # （否则部署阶段的 D20 前伸会把机械臂顶到地面）。
            s._home("task start")
            await asyncio.sleep(3.0)
            while not s._stop:
                if s._release:
                    s._release=False
                    await ws.send(json.dumps({"type":"release"}))
                    release_result=json.loads(await asyncio.wait_for(ws.recv(),timeout=10))
                    release_cmd=release_result.get("arm_cmd",{})
                    if release_cmd.get("action")=="release":
                        s._arm(release_cmd)
                        await asyncio.sleep(float(release_cmd.get("duration_ms",500))/1000)
                    continue
                f=(s._capture_comparison() if s._compare_cameras
                   else s._capture())
                if f is None:
                    now = time.time()
                    if now - s._last_no_frame_log >= 2.0:
                        s._last_no_frame_log = now
                        source_name = "both cameras" if s._compare_cameras else s._active_camera
                        _log(f"waiting for {source_name} MJPEG frame")
                        EVENT_LOG.event("frame_unavailable", source=source_name)
                    await asyncio.sleep(0.3)
                    continue
                EVENT_LOG.throttled(
                    "frame_sent", 5.0, "frame_sent",
                    jpeg_bytes=len(f), frame=status["frame"] + 1,
                )
                await ws.send(f)  # f 已经是 JPEG bytes
                raw=await asyncio.wait_for(ws.recv(),timeout=90)
                resp=json.loads(raw)
                cmd=resp.get("arm_cmd",{})
                ph=resp.get("phase",""); st=resp.get("step_label",""); tr=resp.get("tracking",{})
                response_camera = str(resp.get("active_camera", s._active_camera))
                _set(phase=ph,step=st,camera=response_camera,
                     arm_cmd=cmd,tracking=tr,frame=status["frame"]+1)
                act=cmd.get("action","idle"); rsn=cmd.get("reason","")
                EVENT_LOG.throttled(
                    "server_response", 1.0, "server_response",
                    phase=ph, step=st, arm_cmd=cmd, tracking=tr,
                )
                if act=="move":
                    _log(f"[{ph}] {rsn}")
                    s._arm(cmd)
                    dur=float(cmd.get("duration_ms",900))/1000
                    # 运动中快速送帧 → 3090 运动检测锁定爪子 (黑色+移动)
                    n=int(max(2, dur/0.4))
                    for _k in range(n):
                        await asyncio.sleep(dur/n)
                        f=s._capture()
                        if f is None: continue
                        try:
                            await ws.send(f)
                            raw=await asyncio.wait_for(ws.recv(),timeout=90)
                            resp=json.loads(raw)
                            _set(phase=resp.get("phase",ph),
                                 step=resp.get("step_label",st),
                                 arm_cmd=resp.get("arm_cmd",cmd),
                                 tracking=resp.get("tracking",tr),
                                 frame=status["frame"]+1)
                        except Exception:
                            pass
                    await asyncio.sleep(0.5)
                elif act in ("grasp","release"):
                    _log(f"[{ph}] {act}")
                    s._arm(cmd); await asyncio.sleep(0.8)
                elif act=="done": _log("done!"); _set(phase="done"); break
                elif act=="carry":
                    _log("grasp verified; holding object")
                    _set(phase="carry")
                    await asyncio.sleep(0.3)
                elif act=="error":
                    if cmd.get("emergency_stop") and s._ble: s._ble("#STOP!")
                    s._home("server error")  # 报错优先复位，防过热
                    _log(f"err: {rsn}"); _set(phase="error",error=rsn); break
                elif act=="reground": _log("reground..."); await asyncio.sleep(0.5)
                elif act=="compare_cameras":
                    if len(s._captures) < 2:
                        raise RuntimeError("camera comparison requires both left and right streams")
                    s._compare_cameras = True
                    _set(camera="both")
                    EVENT_LOG.throttled(
                        "camera_comparison", 2.0, "camera_comparison",
                        reason=rsn,
                    )
                    await asyncio.sleep(0.3)
                elif act=="switch_camera":
                    requested = str(cmd.get("camera", ""))
                    if requested not in s._captures:
                        raise RuntimeError(f"server requested unavailable camera: {requested}")
                    old = s._active_camera
                    s._active_camera = requested
                    s._compare_cameras = False
                    _set(camera=requested)
                    _log(f"camera switched: {old} -> {requested}")
                    EVENT_LOG.event(
                        "camera_switched", old_camera=old,
                        new_camera=requested, reason=rsn,
                    )
                    await asyncio.sleep(0.3)
                else: await asyncio.sleep(0.3)
            try:
                await ws.send(json.dumps({"type":"stop"}))
            except Exception:
                pass
        if s._stop and s._ble:
            s._ble("#STOP!")

    def _capture(s):
        # Return the original JPEG bytes. Sending a decoded ndarray over the
        # websocket produces raw pixel bytes that the 3090 JPEG decoder rejects.
        source = s._captures.get(s._active_camera)
        if source is not None:
            try:
                jpeg = source()
                if jpeg and len(jpeg) > 1000:
                    return bytes(jpeg)
            except Exception as exc:
                EVENT_LOG.throttled(
                    "capture_callback_error", 2.0,
                    "capture_callback_error", error=repr(exc),
                )
            return None
        try:
            try:
                from .web_app import _latest_jpeg
            except ImportError:
                from web_app import _latest_jpeg
            jpeg = _latest_jpeg.get("left", b"")
            if jpeg and len(jpeg) > 1000:
                return bytes(jpeg)
        except: pass
        return None

    def _capture_comparison(s):
        """Return one labelled LEFT|RIGHT JPEG for Qwen camera selection."""
        import cv2
        import numpy as np

        frames = []
        for camera, label in (("left", "LEFT"), ("right", "RIGHT")):
            source = s._captures.get(camera)
            if source is None:
                return None
            try:
                jpeg = source()
                if not jpeg or len(jpeg) <= 1000:
                    return None
                frame = cv2.imdecode(
                    np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR,
                )
                if frame is None:
                    return None
                frame = cv2.resize(frame, (s.w, s.h))
                cv2.rectangle(frame, (0, 0), (125, 38), (0, 0, 0), -1)
                cv2.putText(
                    frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA,
                )
                frames.append(frame)
            except Exception as exc:
                EVENT_LOG.throttled(
                    "comparison_capture_error", 2.0,
                    "comparison_capture_error", camera=camera, error=repr(exc),
                )
                return None
        combined = np.hstack(frames)
        ok, encoded = cv2.imencode(
            ".jpg", combined,
            [int(cv2.IMWRITE_JPEG_QUALITY), 82],
        )
        return encoded.tobytes() if ok else None

    def _arm(s,cmd):
        act=cmd.get("action",""); fwd=int(cmd.get("forward_mm",0)); up=int(cmd.get("up_mm",0))
        left=int(cmd.get("left_deg",0)); claw=cmd.get("claw_open"); dur=int(cmd.get("duration_ms",900))
        c=""
        if act=="grasp": c=f"#ARMC{max(0,min(100,int(claw or 10)))}!"
        elif act=="release": c=f"#ARMC{max(0,min(100,int(claw or 95)))}!"
        elif act=="move":
            p=[];
            # Firmware/web UI physical mapping:
            # D/U = forward/back, B/F = up/down.
            if fwd>0: p.append(f"D{fwd}")
            elif fwd<0: p.append(f"U{abs(fwd)}")
            if up>0: p.append(f"B{up}")
            elif up<0: p.append(f"F{abs(up)}")
            if left>0: p.append(f"L{left}")
            elif left<0: p.append(f"R{abs(left)}")
            if claw is not None and claw>=0: p.append(f"C{int(claw)}")
            if p: c=f"#ARM{''.join(p)}!"
        if c:
            result = s._ble(c) if s._ble else "NO_BLE_CALLBACK"
            EVENT_LOG.event(
                "arm_command",
                ascii=c,
                source_cmd=cmd,
                ble_result=result,
            )
            if result != "OK":
                raise ConnectionError(
                    f"arm command not sent because BLE is unavailable: {c}"
                )

_runner = None
def start(target, server="ws://192.168.55.33:8020/ws/grasp",
          ble_send=None, capture=None, captures=None,
          initial_camera="left"):
    global _runner
    if _runner and status.get("active"): return False
    _runner = Runner(
        server, target, ble_send=ble_send, capture=capture,
        captures=captures, initial_camera=initial_camera,
    )
    _runner.start(); return True
def stop():
    global _runner
    if _runner: _runner.stop()

def release():
    global _runner
    if not _runner or not status.get("active"):
        return False
    _runner.request_release()
    return True
