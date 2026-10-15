# -*- coding: utf-8 -*-
"""测试网页端到端验证：真实 uvicorn 服务器 + httpx + websocket-client，
走一遍 页面/帧/状态/重新识别/WS推送 全链路（模型用桩替换，
无论真机还是本机都跑桩，只验证 HTTP/WS 接口层）。

依赖：fastapi、uvicorn、python-multipart、websockets（服务器端）+
httpx、websocket-client（本测试用）。
运行：python test_web_page.py（3090 或本地均可）"""
import json
import sys
import threading
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# ---- 桩：qwen_grounded_tracker ----
class BBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = float(x1), float(y1), float(x2), float(y2)
    @property
    def area(self):
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)
    @property
    def center(self):
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    def clamp(self, w, h):
        return BBox(max(0, min(self.x1, w)), max(0, min(self.y1, h)),
                    max(0, min(self.x2, w)), max(0, min(self.y2, h)))
    def to_xywh(self):
        return (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

class GroundingResult:
    def __init__(self, found, bbox=None, target_name="蓝色瓶子",
                 confidence=0.9, message="", raw_text=""):
        self.found = found
        self.bbox = bbox
        self.target_name = target_name
        self.confidence = confidence
        self.message = message
        self.raw_text = raw_text

class Qwen3VLTargetGrounder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def ground(self, frame, instruction, reference_image=None):
        # 固定返回左目内一个 bbox（拼接坐标 500..700 < 1280）
        return GroundingResult(True, BBox(500, 300, 700, 450))

class CSRTTargetTracker:
    def __init__(self):
        self.initialized = False
    def initialize(self, frame, bbox):
        self.initialized = True
    def update(self, frame):
        # 2026-08-18 帧率重构后 CSRT 是 YOLO/模板未确认时的兜底观测；
        # 桩返回固定框模拟"兜底确认成功"（与 ground 锁定框一致）
        return True, BBox(500, 300, 700, 450)

domain = types.ModuleType("qwen_grounded_tracker.domain")
domain.BBox = BBox
domain.GroundingResult = GroundingResult
qgt = types.ModuleType("qwen_grounded_tracker")
grounding = types.ModuleType("qwen_grounded_tracker.grounding")
qgrounder = types.ModuleType("qwen_grounded_tracker.grounding.qwen_grounder")
qgrounder.Qwen3VLTargetGrounder = Qwen3VLTargetGrounder
tracking = types.ModuleType("qwen_grounded_tracker.tracking")
csrt = types.ModuleType("qwen_grounded_tracker.tracking.csrt_tracker")
csrt.CSRTTargetTracker = CSRTTargetTracker
sys.modules.update({
    "qwen_grounded_tracker": qgt,
    "qwen_grounded_tracker.domain": domain,
    "qwen_grounded_tracker.grounding": grounding,
    "qwen_grounded_tracker.grounding.qwen_grounder": qgrounder,
    "qwen_grounded_tracker.tracking": tracking,
    "qwen_grounded_tracker.tracking.csrt_tracker": csrt,
})

# ---- 桩：ultralytics（返回空检测） ----
class FakeResult:
    names = {}
    boxes = []
class FakeYOLO:
    def __init__(self, model):
        pass
    def __call__(self, *args, **kwargs):
        return [FakeResult()]
ultralytics = types.ModuleType("ultralytics")
ultralytics.YOLO = FakeYOLO
sys.modules["ultralytics"] = ultralytics

import cv2
import numpy as np
import httpx
import uvicorn
import websocket

import pursuit_server





CALIB = str(HERE / "configs" / "stereo_calibration.yaml")
# 8022 在 3090 上由 arm_teach_model 长期占用；网页回归使用独立端口，
# 避免测试误碰机械臂示教服务。
PORT = 8122


def make_synthetic_frame():
    # 平滑渐变背景：JPEG 压缩后白分界线/绿框仍可稳定检出
    ys, xs = np.mgrid[0:720, 0:2560]
    img = np.zeros((720, 2560, 3), dtype=np.uint8)
    img[:, :, 0] = (xs * 0.08 + ys * 0.05).astype(np.uint8)
    img[:, :, 1] = (ys * 0.12).astype(np.uint8)
    img[:, :, 2] = (xs * 0.03 + 90).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return buf.tobytes()


def main():
    session = pursuit_server.PursuitSession(
        qwen_model="fake", yolo_model="fake",
        calibration_yaml=CALIB, device="cpu")
    pursuit_server.app.state.pursuit_session = session

    config = uvicorn.Config(pursuit_server.app, host="127.0.0.1",
                            port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    base = f"http://127.0.0.1:{PORT}"
    for _ in range(100):
        try:
            httpx.get(f"{base}/health", timeout=0.3)
            break
        except Exception:
            time.sleep(0.1)
    print("[0] uvicorn 已启动")

    # 1) 首页
    r = httpx.get(f"{base}/")
    assert r.status_code == 200
    for keyword in ("左摄像头", "右摄像头", "重新识别", "frame.jpg"):
        assert keyword in r.text, f"页面缺少 {keyword}"
    # yolo_box 位于 frame_result 顶层；旧页面误读 track.yolo_box，
    # 导致始终画 KF/模板框而不是检测器真实物体框。
    assert "target.yolo_box" in r.text
    assert "target.yolo_box2" in r.text
    print("[1] GET / 页面 OK（双画面/输入框/重新识别按钮）")

    # 2) 无帧时各端点行为
    assert httpx.get(f"{base}/frame.jpg").status_code == 404
    assert httpx.get(f"{base}/status").json() == {"frame": False, "result": {}}
    r = httpx.post(f"{base}/reground", data={"instruction": "测试"})
    body = r.json()
    assert body["found"] is False and "尚无相机帧" in body["message"]
    print("[2] 无帧时 /frame.jpg 404、/status 空、/reground 友好报错 OK")

    # 3) 模拟树莓派 WS 接入
    ws = websocket.create_connection(f"ws://127.0.0.1:{PORT}/pursuit_ws",
                                     timeout=5)
    ws.send(json.dumps({"type": "pursuit_start", "instruction": "测试",
                        "frame_width": 2560, "frame_height": 720,
                        "upload_fps": 10.0}, ensure_ascii=False))
    assert json.loads(ws.recv())["type"] == "ack"
    print("[3] WS 接入 OK")

    # 4) 上传一帧 → frame_result；随后 /frame.jpg 有标注帧
    ws.send_binary(make_synthetic_frame())
    result = json.loads(ws.recv())
    assert result["type"] == "frame_result"
    print("[4] WS 帧处理 OK:", json.dumps(result["track"], ensure_ascii=False))

    r = httpx.get(f"{base}/frame.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    frame = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert frame is not None and frame.shape == (720, 2560, 3)
    print("[5] GET /frame.jpg 返回 2560x720 拼接帧 OK")

    # 5) 网页"重新识别"：锁定目标并把 ground_result 推给树莓派 WS
    r = httpx.post(f"{base}/reground", data={"instruction": "追踪蓝色瓶子"})
    body = r.json()
    assert body["found"] is True and body["target_name"] == "蓝色瓶子", body
    push = json.loads(ws.recv())
    assert push["type"] == "ground_result" and push["found"] is True
    assert push["target_name"] == "蓝色瓶子"
    print("[6] POST /reground → 锁定 → WS 推 ground_result OK")

    # 6) 再传一帧：tracking 状态 + 标注帧画上目标框和分界线
    # 树莓派上行逼近阶段状态（08-16 起画框要求 pi_state=TRACKING）
    ws.send_text(json.dumps({"type": "pi_state", "state": "TRACKING",
                             "cmd": [100, 0], "detail": "", "result": ""},
                            ensure_ascii=False))
    time.sleep(0.1)
    ws.send_binary(make_synthetic_frame())
    result = json.loads(ws.recv())
    assert result["type"] == "frame_result"
    track = result["track"]
    assert track["status"] == "tracking" and track["bbox"], track
    r = httpx.get(f"{base}/frame.jpg")
    frame = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    seam = frame[:, 1280, :]
    assert (seam > 240).all(), "分界线未画出"
    # bbox 绿色框线应出现在 bbox 边界附近（JPEG 压缩后非纯绿）
    edge = frame[300:302, 500:700, :]
    assert ((edge[:, :, 1] > 200) & (edge[:, :, 2] < 60)).any(), "目标框未画出"
    print("[7] tracking 状态 + /frame.jpg 目标框/分界线标注 OK")

    # 7) /status 反映最新结果
    st = httpx.get(f"{base}/status").json()
    assert st["frame"] is True
    assert st["result"]["track"]["name"] == "蓝色瓶子"
    print("[8] GET /status OK")

    # 8) WS 断开后 /reground 正常返回且不推送报错
    ws.close()
    time.sleep(0.3)
    r = httpx.post(f"{base}/reground", data={"instruction": "再来一次"})
    assert r.json()["found"] is True
    print("[9] WS 断开后 /reground 静默不推送 OK")

    # 9) display_swap：画面交换左右半幅，bbox 同步平移
    # 注意：/frame.jpg 有 (seq,state) 缓存——运行时切换 swap 必须上传
    # 新帧使缓存失效（真实运行 swap 是启动参数，不会运行时切换）
    ws4 = websocket.create_connection(f"ws://127.0.0.1:{PORT}/pursuit_ws",
                                      timeout=5)
    ws4.send(json.dumps({"type": "pursuit_start", "instruction": "测试",
                         "frame_width": 2560, "frame_height": 720,
                         "upload_fps": 10.0}, ensure_ascii=False))
    json.loads(ws4.recv())  # ack
    ws4.send_text(json.dumps({"type": "pi_state", "state": "TRACKING",
                              "cmd": [100, 0], "detail": "", "result": ""},
                             ensure_ascii=False))
    session.display_swap = True
    ws4.send_binary(make_synthetic_frame())
    json.loads(ws4.recv())  # frame_result（刷新 seq → /frame.jpg 缓存失效）
    r = httpx.get(f"{base}/frame.jpg")
    frame = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    # 原 bbox 在左半幅 (500..700) → 交换后应出现在右半幅 (1780..1980)
    edge_swapped = frame[300:302, 1780:1980, :]
    assert ((edge_swapped[:, :, 1] > 200) & (edge_swapped[:, :, 2] < 60)).any(), \
        "display_swap 后目标框未平移到右半幅"
    st = httpx.get(f"{base}/status").json()
    bbox = st["result"]["track"]["bbox"]
    assert 1280 <= bbox[0] < 2560, f"/status bbox 未随 swap 平移: {bbox}"
    ws4.close()
    print("[10] display_swap 画面与 bbox 同步交换 OK")





    # 10) /stream.mjpg：零重编码 MJPEG 流 + 连接即发当前帧
    # 注意：此时 display_swap=True（上一步），但流应保持原始字节不变
    uploaded_jpeg = make_synthetic_frame()
    with httpx.stream("GET", f"{base}/stream.mjpg", timeout=10) as r:
        assert r.status_code == 200
        ct = r.headers["content-type"]
        assert "multipart/x-mixed-replace" in ct, ct
        assert "boundary=" in ct
        assert r.headers.get("cache-control") == "no-store"
        buf = bytearray()
        jpeg_bytes = None
        for chunk in r.iter_bytes():
            buf += chunk
            if jpeg_bytes is None and b"\r\n\r\n" in buf:
                head_text = buf.decode("latin1").partition("\r\n\r\n")[0]
                length = None
                for line in head_text.split("\r\n"):
                    if line.lower().startswith("content-length:"):
                        length = int(line.split(":", 1)[1].strip())
                        break
                if length is not None:
                    header_end = len(head_text.encode("latin1")) + 4
                    if len(buf) >= header_end + length:
                        jpeg_bytes = bytes(buf[header_end:header_end + length])
            if jpeg_bytes is not None:
                break  # 必须在循环体末尾 break：顶部 break 会先阻塞等下一块
        assert jpeg_bytes is not None, "流未在超时内发出完整 part"
        assert jpeg_bytes == uploaded_jpeg, "流字节与上传帧不一致（被重编码？）"
    print("[10] /stream.mjpg 零重编码流 + 首帧即发 OK")

    # 11) /annotations：浏览器订阅 + 逐帧广播 + 双客户端 + 慢客户端不阻塞
    annot1 = websocket.create_connection(f"ws://127.0.0.1:{PORT}/annotations",
                                         timeout=5)
    hello = json.loads(annot1.recv())
    assert hello["type"] == "hello"
    assert hello["display_swap"] is True, "hello 未带 display_swap 信息"
    assert hello["frame_width"] == 2560 and hello["frame_height"] == 720
    # 慢客户端（不读消息，队列应 latest-wins 丢弃积压）
    annot2 = websocket.create_connection(f"ws://127.0.0.1:{PORT}/annotations",
                                         timeout=5)
    json.loads(annot2.recv())  # hello
    # 重接树莓派 WS（上一步已断开），上传帧 + pi_state
    ws3 = websocket.create_connection(f"ws://127.0.0.1:{PORT}/pursuit_ws",
                                      timeout=5)
    ws3.send(json.dumps({"type": "pursuit_start", "instruction": "测试",
                         "frame_width": 2560, "frame_height": 720,
                         "upload_fps": 0.0}, ensure_ascii=False))
    json.loads(ws3.recv())  # ack
    # 消费重连补发的 ground_result（服务端已改为补发后清空；防御性处理）
    # 注意：settimeout 重置必须放 except/finally——超时路径也要恢复 5s
    try:
        ws3.settimeout(0.2)
        pending = json.loads(ws3.recv())
        assert pending["type"] == "ground_result"
    except Exception:
        pass
    finally:
        ws3.settimeout(5)
    # annot1 用并发读取线程持续消费（保持队列清空，逐帧都送达）
    collect: list = []

    def annot_reader() -> None:
        try:
            while True:
                msg = json.loads(annot1.recv())
                collect.append(msg)
                types = {m.get("type") for m in collect}
                if "frame_result" in types and "pi_state" in types:
                    break
        except Exception:
            pass
    reader_thread = threading.Thread(target=annot_reader, daemon=True)
    reader_thread.start()
    for _ in range(30):  # 快速连发 30 帧：慢客户端队列应 latest-wins 不阻塞
        ws3.send_binary(uploaded_jpeg)
        result = json.loads(ws3.recv())
        assert result["type"] == "frame_result"
    ws3.send_text(json.dumps({"type": "pi_state", "state": "TEST_STATE",
                              "cmd": [100, 0], "detail": "x", "result": "r",
                              "left_fps": 25.0, "right_fps": 25.0,
                              "enc_fps": 24.0, "upload_fps": 24.0,
                              "skip_fps": 0.0, "send_ms": 5.0},
                             ensure_ascii=False))
    reader_thread.join(timeout=5)
    assert not reader_thread.is_alive(), "annotations 未及时收到 frame_result+pi_state"
    # 取最后一个 frame_result（对应最后一帧上传，与 Pi 侧 result 同帧）
    got_frame = next(m for m in reversed(collect) if m["type"] == "frame_result")
    got_pi = next(m for m in collect if m["type"] == "pi_state")
    assert got_pi["state"] == "TEST_STATE" and got_pi["left_fps"] == 25.0
    # 浏览器收到的 bbox = 管线 raw 拼接坐标（display_swap 不作用于新链路）
    assert got_frame["track"]["bbox"] == result["track"]["bbox"], \
        "annotations bbox 与管线结果不一致"
    # 慢客户端：队列 latest-wins，拿到的是最新消息（非 keepalive）
    last2 = None
    deadline = time.time() + 5
    while time.time() < deadline and last2 is None:
        m = json.loads(annot2.recv())
        if m["type"] in ("frame_result", "pi_state"):
            last2 = m
    assert last2 is not None, "慢客户端未收到任何消息（感知管线被阻塞？）"
    annot1.close()
    annot2.close()
    ws3.close()
    print("[11] /annotations 广播（hello/逐帧/pi_state/双客户端/慢客户端 latest-wins）OK")

    # 12) StreamHub 纯单元测试：latest-wins 覆盖、注销静默、跨线程通知
    import asyncio

    def test_streamhub_unit():
        import pursuit_server as ps

        async def run():
            hub = ps.StreamHub()
            loop = asyncio.get_running_loop()
            hub.attach(loop)
            # 未注册时通知静默（不抛异常）
            hub.notify_latest({"type": "frame_result", "n": 0})
            key, queue = hub.register_annot()
            hub.notify_latest({"type": "frame_result", "n": 1})
            hub.notify_latest({"type": "frame_result", "n": 2})
            m1 = await asyncio.wait_for(queue.get(), 1.0)
            assert m1["n"] == 2, f"latest-wins 失败: {m1}"
            hub.unregister_annot(key)
            hub.notify_latest({"type": "frame_result", "n": 3})
            await asyncio.sleep(0.05)
            assert queue.empty(), "注销后仍收到消息"
        asyncio.run(run())
    test_streamhub_unit()
    print("[12] StreamHub 单元测试（latest-wins/注销静默）OK")

    server.should_exit = True
    print("WEB_PAGE_TESTS_PASSED")


if __name__ == "__main__":
    main()
