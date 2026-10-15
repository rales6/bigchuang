# -*- coding: utf-8 -*-
"""pursuit_server.py 新增强的本地测试：卡尔曼门控 / ego-motion /
模板匹配 KF 中心 / YOLO 提案追回 / process_frame 冒烟。

AST 装配：剥掉 3090 专属 import（qwen_grounded_tracker/stereo_depth），
用本地桩替换后 exec 整个模块，直接实例化 PursuitSession 测方法。
"""
import math
import sys
from pathlib import Path

# 本地（Windows）用仓库路径；部署机上与 pursuit_server.py 同目录
TRACK_3090 = r"F:\bigchuang\track\3090"
if not Path(TRACK_3090).exists():
    TRACK_3090 = str(Path(__file__).resolve().parent)
if TRACK_3090 not in sys.path:
    sys.path.insert(0, TRACK_3090)

import cv2
import numpy as np

from pursuit_kalman import KalmanBoxTracker


class Box:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1 = float(x1), float(y1)
        self.x2, self.y2 = float(x2), float(y2)

    @property
    def area(self):
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def clamp(self, w, h):
        return Box(max(0.0, min(self.x1, w)), max(0.0, min(self.y1, h)),
                   max(0.0, min(self.x2, w)), max(0.0, min(self.y2, h)))


def iou(a, b):
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


# ---------------- 一、KalmanBoxTracker 纯单元测试 ----------------

def test_kalman():
    kf = KalmanBoxTracker(Box(400, 300, 480, 380))
    # A1) 预测后同位置测量 → 采纳，马氏距离小
    kf.predict(0.04, (0.0, 0.0))
    ok, dist = kf.update(Box(402, 302, 482, 382))
    assert ok, "同位置测量应被采纳"
    assert dist < 1.5, dist
    # A2) 远跳 300px → 门控拒绝；force 越过门控
    kf.predict(0.04, (0.0, 0.0))
    ok, dist = kf.update(Box(700, 600, 780, 680))
    assert not ok, "300px 远跳应被门控拒绝"
    assert dist > 3.0, dist
    ok, _ = kf.update(Box(700, 600, 780, 680), force=True)
    assert ok, "强观测应越过门控"
    # A3) ego-motion 控制输入：u=(+500,0) 0.04s → 预测中心 +20px
    kf2 = KalmanBoxTracker(Box(100, 100, 140, 140))
    kf2.predict(0.04, (500.0, 0.0))
    assert abs(kf2.predicted_center[0] - 140.0) < 1e-6, kf2.predicted_center
    # A4) 连续跟踪收敛：目标每帧右移 20px，预测应跟上且测量全被采纳
    kf3 = KalmanBoxTracker(Box(100, 100, 140, 140))
    rejected = 0
    for i in range(10):
        kf3.predict(0.05, (0.0, 0.0))
        ok, _ = kf3.update(Box(100 + 20 * (i + 1), 100, 140 + 20 * (i + 1), 140))
        if not ok:
            rejected += 1
    assert rejected == 0, f"匀速目标不应被门控拒绝 {rejected} 次"
    kf3.predict(0.05, (0.0, 0.0))
    pcx, _ = kf3.predicted_center
    assert abs(pcx - (120 + 200 + 20)) < 40.0, pcx  # 预测接近下一真值
    print("KALMAN_OK")


# ---------------- 二、AST 装配 PursuitSession ----------------

def _install_fastapi_stubs():
    """本地无 fastapi：注入桩（路由装饰器/参数注解桩，不执行请求）。"""
    import types

    fastapi = types.ModuleType("fastapi")

    def _route(self, *args, **kwargs):
        def deco(fn):
            return fn
        return deco

    class _FastAPI:
        def __init__(self, *args, **kwargs):
            self.state = types.SimpleNamespace()

        get = _route
        post = _route
        websocket = _route

    class _HTTPException(Exception):
        def __init__(self, status_code=None, detail=None, *a, **kw):
            super().__init__(detail)
            self.status_code = status_code

    fastapi.FastAPI = _FastAPI
    fastapi.HTTPException = _HTTPException
    fastapi.File = lambda *a, **k: None
    fastapi.Form = lambda *a, **k: None
    fastapi.UploadFile = object
    fastapi.WebSocket = object
    fastapi.WebSocketDisconnect = Exception
    responses = types.ModuleType("fastapi.responses")
    responses.HTMLResponse = object
    responses.Response = object
    responses.StreamingResponse = object
    fastapi.responses = responses
    sys.modules["fastapi"] = fastapi
    sys.modules["fastapi.responses"] = responses


def build_session_module():
    _install_fastapi_stubs()
    src = (Path(TRACK_3090) / "pursuit_server.py").read_text(encoding="utf-8")
    stub = '''class BBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = float(x1), float(y1), float(x2), float(y2)
    @property
    def area(self):
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)
    def clamp(self, w, h):
        return BBox(max(0.0, min(self.x1, w)), max(0.0, min(self.y1, h)),
                    max(0.0, min(self.x2, w)), max(0.0, min(self.y2, h)))
    def __repr__(self):
        return f"BBox({self.x1:.0f},{self.y1:.0f},{self.x2:.0f},{self.y2:.0f})"
GroundingResult = BBox
class Qwen3VLTargetGrounder:
    def __init__(self, **kwargs): pass
class CSRTTargetTracker:
    def __init__(self): self.update_ret = (False, None)
    def initialize(self, image, bbox): pass
    def update(self, image): return self.update_ret
class StereoDepth:
    def __init__(self, *args, **kwargs): self.image_size = (1280, 720)
'''
    import re
    for pat in (
        re.compile(r"from qwen_grounded_tracker\.domain import BBox, "
                   r"GroundingResult  # noqa: E402\n"),
        re.compile(r"from qwen_grounded_tracker\.grounding\.qwen_grounder "
                   r"import \(  # noqa: E402\n"
                   r"    Qwen3VLTargetGrounder,\n\)\n"),
        re.compile(r"from qwen_grounded_tracker\.tracking\.csrt_tracker "
                   r"import \(  # noqa: E402\n"
                   r"    CSRTTargetTracker,\n\)\n"),
        re.compile(r"from stereo_depth import StereoDepth  # noqa: E402\n"),
        re.compile(r"from pursuit_kalman import KalmanBoxTracker  # noqa: E402\n"),
    ):
        src, n = pat.subn("", src)
        assert n == 1, f"import 剥离失败: {pat.pattern}"
    # 桩定义放在被剥离位置，并重新引入本地 pursuit_kalman
    # （from __future__ 必须挪到文件最开头，否则语法错误）
    src = src.replace("from __future__ import annotations\n\n", "", 1)
    body = ("from __future__ import annotations\n" + stub
            + "\nfrom pursuit_kalman import KalmanBoxTracker\n" + src)
    ns: dict = {"__name__": "pursuit_server_stub"}
    exec(compile(body, "pursuit_server_stub.py", "exec"), ns)
    return ns


def make_noise_patch(rng, size=40):
    """随机噪声纹理块（唯一性强，匹配不会误认）。"""
    patch = rng.randint(40, 220, (size, size, 3), dtype=np.uint8)
    return patch


def build_test_image(rng, w=640, h=480):
    image = np.full((h, w, 3), 60, dtype=np.uint8)
    return image


def test_session_methods():
    ns = build_session_module()
    PursuitSession = ns["PursuitSession"]
    # 首帧安全门：空画面实测曾把 floor/window 拼成双目目标并连续追踪。
    assert not ns["_scene_region_allowed"]("floor", "追踪目标物")
    assert ns["_scene_region_allowed"]("floor", "靠近地板区域")
    assert not ns["_target_names_compatible"](
        "floor", "window", "追踪目标物")
    assert ns["_target_names_compatible"](
        "红色小物块", "red block", "追踪红色小物块")
    assert ns["_instruction_target_classes"]("小车追踪蓝色瓶子") == {"bottle"}
    assert not ns["_initial_box_complete"](Box(0, 10, 80, 90), 640, 480)
    assert ns["_initial_box_complete"](Box(10, 10, 80, 90), 640, 480)
    session = PursuitSession("qwen", "yolo", "calib", device="cpu",
                             display_swap=False, use_depth=False)
    rng = np.random.RandomState(42)
    image = build_test_image(rng)

    class _GroundResult:
        def __init__(self, name, bbox, confidence=0.99):
            self.found = bbox is not None
            self.bbox = bbox
            self.target_name = name
            self.confidence = confidence
            self.message = ""
            self.raw_text = name

    class _FakeGrounder:
        def __init__(self, results):
            self.results = list(results)
            self.references = []
        def ground(self, frame, instruction, reference_image=None):
            self.references.append(reference_image is not None)
            return self.results.pop(0)

    # 空画面幻觉 floor/window 必须整组拒绝，不能再让车追场景边缘。
    session.grounder = _FakeGrounder([
        _GroundResult("floor", Box(300, 100, 500, 180)),
        _GroundResult("window", Box(300, 100, 500, 180)),
    ])
    rejected = session._ground_locked(image, image.copy(), "追踪目标物")
    assert not rejected["found"], rejected
    # 正确目标必须双目确认；第二次调用确实带第一路参考裁块。
    good_grounder = _FakeGrounder([
        _GroundResult("blue bottle", Box(300, 180, 360, 320)),
        _GroundResult("blue bottle", Box(280, 182, 340, 322)),
    ])
    session.grounder = good_grounder
    accepted = session._ground_locked(image, image.copy(), "追踪蓝色瓶子")
    assert accepted["found"] and accepted["bbox2"], accepted
    assert good_grounder.references == [False, True], good_grounder.references
    # 后续追踪场景使用干净会话，避免 grounding 测试的外观锚点串场。
    session = PursuitSession("qwen", "yolo", "calib", device="cpu",
                             display_swap=False, use_depth=False)

    # B1) _apply_measurement 门控
    kf = KalmanBoxTracker(Box(400, 300, 480, 380))
    session._kf["left"] = kf
    adopted = session._apply_measurement("left", Box(405, 305, 485, 385))
    assert adopted is not None, "近邻测量应采纳"
    # 远跳 → 拒绝；force → 采纳
    adopted = session._apply_measurement("left", Box(100, 100, 180, 180))
    assert adopted is None, "远跳测量应被门控拒绝"
    adopted = session._apply_measurement("left", Box(100, 100, 180, 180),
                                         force=True)
    assert adopted is not None, "force 应越过门控"
    # 采纳结果应介于测量与预测之间（EMA）
    assert 100.0 < adopted.x1 < 400.0, adopted

    # B2) _match_template 搜索中心用卡尔曼预测：目标真实在 (320,220)，
    #     锁框在 (300,200)，但 KF 已预测到 (430,220)（车转动后的位置）
    #     ——匹配应找回真实位置，而不是被旧锁框拖住。
    #     噪声块大于 bbox：模板 padding 区也有纹理（贴近真实场景
    #     "目标周围纹理与目标相似"），否则低纹理背景会稀释相关系数。
    image2 = build_test_image(rng)
    patch = make_noise_patch(rng, size=80)
    image2[180:260, 280:360] = patch
    box_true = Box(300, 200, 340, 240)
    session._templates = {}
    session._histograms = {}
    session._store_template("left", image2, box_true)
    session._kf["left"] = KalmanBoxTracker(Box(420, 210, 440, 230))
    session._kf["left"].predict(0.04, (0.0, 0.0))
    matched, score = session._match_template(image2, box_true, "left")
    assert matched is not None, "模板应匹配成功"
    assert score > 0.6, score
    mcx = (matched.x1 + matched.x2) / 2.0
    mcy = (matched.y1 + matched.y2) / 2.0
    assert abs(mcx - 320.0) < 25.0 and abs(mcy - 220.0) < 25.0, \
        (mcx, mcy, matched)
    # 初始外观锚点不被后续 3 张近期模板挤掉。
    anchor = session._anchor_templates["left"].copy()
    for dx in (2, 4, 6, 8):
        session._store_template("left", image2,
                                Box(300 + dx, 200, 340 + dx, 240))
    bank, _ = session._appearance_bank("left")
    assert len(bank) == 4 and np.array_equal(bank[-1], anchor)

    # B3) _yolo_proposal_recover：真目标 + 干扰块，YOLO 提案都能提出来，
    #     但只有外观验证通过的真目标被采纳（非 COCO 类阈值 0.72）
    image3 = build_test_image(rng)
    patch_a = make_noise_patch(rng, size=80)
    patch_b = make_noise_patch(rng)
    image3[220:300, 380:460] = patch_a    # 真目标（模板来自它）
    image3[80:120, 80:120] = patch_b      # 干扰块
    session._templates = {}
    session._histograms = {}
    session._store_template("left", image3, Box(400, 240, 440, 280))
    session.target_name = "红色方块"   # 非 COCO 类
    session._frame_count = 4          # 偶数帧才执行提案追回
    session._pi_state = {}

    class _FakeBox:
        def __init__(self, xyxy, conf):
            self.xyxy = [np.asarray(xyxy, dtype=np.float64)]
            self.conf = [np.asarray(conf, dtype=np.float64)]
            self.cls = [np.asarray(0)]

    class _FakeYolo:
        def __init__(self, dets):
            self._dets = dets

        def __call__(self, image, conf=0.25, iou=0.45, verbose=False,
                     device=None):
            class R:
                pass
            r = R()
            r.names = {0: "thing"}
            r.boxes = [_FakeBox(xy, c) for xy, c in self._dets]
            return [r]

    session.yolo = _FakeYolo([((398, 238, 442, 282), 0.5),
                              ((78, 78, 122, 122), 0.55)])
    recovered, rec_score = session._yolo_proposal_recover(image3, "left")
    assert recovered is not None, "真目标应被提案追回"
    assert rec_score >= 0.72, rec_score
    assert iou(recovered, Box(400, 240, 440, 280)) > 0.5, recovered
    # 只有干扰块 → 外观验证不通过 → 无追回
    session.yolo = _FakeYolo([((78, 78, 122, 122), 0.9)])
    recovered, _ = session._yolo_proposal_recover(image3, "left")
    assert recovered is None, "干扰块不应被采纳"
    # 无检测 → 无追回
    session.yolo = _FakeYolo([])
    recovered, _ = session._yolo_proposal_recover(image3, "left")
    assert recovered is None
    # 奇数帧跳过（省时）
    session._frame_count = 5
    session.yolo = _FakeYolo([((398, 238, 442, 282), 0.5)])
    recovered, _ = session._yolo_proposal_recover(image3, "left")
    assert recovered is None, "奇数帧应跳过提案追回"

    # B4) process_frame 冒烟：CSRT 持续确认 → tracking + bbox 坐标合理；
    #     顺带验证 _match_yolo 二元组解包、KF 预测等签名改动
    image4 = build_test_image(rng)
    patch_c = make_noise_patch(rng, size=80)
    image4[180:260, 280:360] = patch_c
    stitched = np.hstack([image4, image4.copy()])
    session._templates = {}
    session._histograms = {}
    session.lock_bbox = Box(300, 200, 340, 240)
    session.lock_view = "left"
    session.lost_frames = 0
    session.lock2_active = False
    session._store_template("left", image4, session.lock_bbox)
    session._kf = {"left": KalmanBoxTracker(session.lock_bbox)}
    session._frame_count = 4   # process_frame 内 +1 → 5（奇数帧跳过 YOLO）
    session.csrt.update_ret = (True, Box(302, 202, 342, 242))
    result = session.process_frame(stitched)
    track = result["track"]
    assert track["status"] == "tracking", track
    assert len(track["bbox"]) == 4
    assert 0.0 <= track["bbox"][0] < 640.0, track["bbox"]
    # set_pi_motion 钳位与容错
    session.set_pi_motion({"v": "abc", "w": 99999})
    assert session._pi_motion["w_mrad_s"] == 3000.0
    session.set_pi_motion({"v": 200, "w": -500})
    assert session._pi_motion == {"v_mm_s": 200.0, "w_mrad_s": -500.0}

    # B4b) 偶数帧 YOLO 匹配路径（_match_yolo 二元组解包 + 门控采纳回归）
    session._frame_count = 3   # process 内 +1 → 4（偶数帧跑 YOLO）
    session._templates = {}
    session._histograms = {}
    session.lock_bbox = Box(302, 202, 342, 242)
    session.lost_frames = 0
    session._kf = {"left": KalmanBoxTracker(session.lock_bbox)}
    session.yolo = _FakeYolo([((305, 205, 345, 245), 0.7)])   # IoU 高
    session.csrt.update_ret = (False, None)
    result = session.process_frame(stitched)
    track = result["track"]
    assert track["status"] == "tracking", track
    assert track["source"] in ("yolo", "tpl"), track
    bx = (track["bbox"][0] + track["bbox"][2]) / 2.0
    assert 280.0 <= bx <= 380.0, track["bbox"]

    # B5) CSRT 漂移 → 模板纠偏（无条件采纳回归）：卡尔曼被 CSRT 拖到
    #     漂移位置（相距 ~150px）后，模板在真实位置找回必须仍被采纳，
    #     把锁拉回真实目标而不是继续跟着漂移
    image5 = build_test_image(rng)
    patch_d = make_noise_patch(rng, size=80)
    image5[180:260, 280:360] = patch_d
    stitched5 = np.hstack([image5, image5.copy()])
    session._templates = {}
    session._histograms = {}
    session.lock_bbox = Box(300, 200, 340, 240)
    session.lock_view = "left"
    session.lost_frames = 0
    session.lock2_active = False
    session._store_template("left", image5, session.lock_bbox)
    drifted = Box(420, 280, 460, 320)   # CSRT 漂移后的错误位置
    session._kf = {"left": KalmanBoxTracker(drifted)}
    session._frame_count = 5
    session.yolo = _FakeYolo([])   # YOLO 每帧跑（重构后）；清空检测让模板纠偏路径生效
    session._yolo_miss_streak = 2  # 模板触发条件：YOLO 连续 2 帧未确认（2026-08-18）
    session.csrt.update_ret = (True, drifted)
    result = session.process_frame(stitched5)
    track = result["track"]
    assert track["status"] == "tracking", track
    assert track["source"] == "tpl", track
    cx = (track["bbox"][0] + track["bbox"][2]) / 2.0
    cy = (track["bbox"][1] + track["bbox"][3]) / 2.0
    assert abs(cx - 320.0) < 30.0 and abs(cy - 220.0) < 30.0, \
        (cx, cy, track["bbox"])

    # C1) CSRT 远跳门控拒绝（2026-08-17 漂移修复）：CSRT 报告与 KF
    #     预测相距 300px 的框 → 不采纳，输出 KF 预测框（source="kf"），
    #     计入丢失。旧逻辑无条件采纳+force 更新 KF，漂移形成正反馈。
    session._templates = {}
    session._histograms = {}
    session.lock_bbox = Box(300, 200, 340, 240)
    session.lock_view = "left"
    session.lost_frames = 0
    session.lock2_active = False
    session._kf = {"left": KalmanBoxTracker(session.lock_bbox)}
    session._frame_count = 4   # → 5 奇数帧跳过 YOLO
    session.csrt.update_ret = (True, Box(600, 200, 640, 240))
    result = session.process_frame(stitched)
    track = result["track"]
    assert track["source"] == "kf", track
    assert session.lost_frames >= 1, session.lost_frames
    bx = (track["bbox"][0] + track["bbox"][2]) / 2.0
    by = (track["bbox"][1] + track["bbox"][3]) / 2.0
    assert abs(bx - 320.0) < 15.0 and abs(by - 220.0) < 15.0, \
        track["bbox"]   # 预测框在真实位置附近，未漂走

    # C2) 分叉检测（2026-08-17 修复）：dx 小但 y 一致 → 不丢第二路；
    #     y 差 >150 → 持续 10 帧后丢第二路。
    session._templates = {}
    session._histograms = {}
    session.lock_bbox = Box(300, 200, 340, 240)
    session.lock_view = "left"
    session.lost_frames = 0
    session.lock2_active = True
    session.lock_bbox2 = Box(288, 200, 328, 240)   # dx=12, y 一致
    session.lock_view2 = "right"
    session.lost_frames2 = 0
    session._kf = {"left": KalmanBoxTracker(session.lock_bbox),
                   "right": KalmanBoxTracker(session.lock_bbox2)}
    session.csrt.update_ret = (True, Box(302, 202, 342, 242))
    session.csrt2.update_ret = (True, Box(290, 202, 330, 242))
    session.yolo = _FakeYolo([])
    for _ in range(12):
        session._frame_count += 1
        session.process_frame(stitched)
    assert session.lock2_active, "dx=12 且 y 一致不应丢第二路"
    # y 差 180px → 持续 10 帧丢第二路
    session.lock2_active = True
    session.lock_bbox2 = Box(288, 400, 328, 440)
    session._y_divergence_frames = 0
    session.csrt2.update_ret = (True, Box(290, 402, 330, 442))
    for _ in range(12):
        session._frame_count += 1
        session.process_frame(stitched)
    assert not session.lock2_active, "y 差大应丢第二路"

    # D1) 近距大目标模板匹配（2026-08-17 尺度修复回归）：旧模板
    #     上限 64px + 固定尺度 0.5~2.0× 最多匹配 128px，300px 目标
    #     永远匹配不上（相关系数在目标内部拉平）→ 近距漂移无法
    #     纠偏。现在存储上限 256，应能匹配整幅目标并回到中心。
    image_d1 = build_test_image(rng, w=800, h=600)
    cv2.rectangle(image_d1, (200, 100), (500, 400), (200, 200, 200), -1)
    cv2.circle(image_d1, (350, 250), 12, (255, 255, 255), -1)  # 中心白点
    session_d1 = PursuitSession("qwen", "yolo", "calib", device="cpu",
                                display_swap=False, use_depth=False)
    session_d1._templates = {}
    session_d1._histograms = {}
    box_d1 = Box(200, 100, 500, 400)
    session_d1._store_template("left", image_d1, box_d1)
    assert max(session_d1._templates["left"][0].shape[:2]) == \
        session_d1._tpl_max_size, session_d1._templates["left"][0].shape
    matched, score = session_d1._match_template(image_d1, box_d1, "left")
    assert matched is not None, "近距大目标应被模板匹配找到"
    mcx = (matched.x1 + matched.x2) / 2.0
    mcy = (matched.y1 + matched.y2) / 2.0
    assert abs(mcx - 350.0) < 30.0 and abs(mcy - 250.0) < 30.0, \
        (mcx, mcy, matched)
    assert score > 0.6, score

    # D2) 自适应尺度：模板 100px（旧固定尺度最大 2×100=200px 匹配
    #     不上 400px 目标），锁框 400px → 自适应 base=4 才能匹配
    image_d2 = build_test_image(rng, w=800, h=600)
    cv2.rectangle(image_d2, (200, 100), (600, 500), (180, 180, 180), -1)
    cv2.circle(image_d2, (400, 300), 16, (255, 255, 255), -1)
    session_d1._templates = {}
    session_d1._histograms = {}
    session_d1._store_template("left", image_d2, Box(370, 270, 430, 330))
    box_d2 = Box(200, 100, 600, 500)
    session_d1._kf["left"] = KalmanBoxTracker(box_d2)
    matched2, score2 = session_d1._match_template(image_d2, box_d2, "left")
    assert matched2 is not None, "自适应尺度应能匹配 400px 大目标"
    mcx2 = (matched2.x1 + matched2.x2) / 2.0
    mcy2 = (matched2.y1 + matched2.y2) / 2.0
    assert abs(mcx2 - 400.0) < 40.0 and abs(mcy2 - 300.0) < 40.0, \
        (mcx2, mcy2, matched2)

    # D3) 平滑漂移看门狗（2026-08-17）：无模板可印证、无 YOLO 时，
    #     CSRT 每帧 +25px 平滑漂移（近距 475px/s 的测试等价）→ 旧
    #     代码一直采纳漂移直到 15 帧外观缺失才清锁；新代码看门狗
    #     ~6 帧内介入（source=kf、lost_frames≥3），最终清锁。
    image_d3 = build_test_image(rng)
    stitched_d3 = np.hstack([image_d3, image_d3.copy()])
    session_d1._templates = {}
    session_d1._histograms = {}
    session_d1.lock_bbox = Box(300, 200, 340, 240)
    session_d1.lock_view = "left"
    session_d1.lost_frames = 0
    session_d1.lock2_active = False
    session_d1._kf = {"left": KalmanBoxTracker(session_d1.lock_bbox)}
    session_d1._drift = {}
    session_d1._drift_frames = {}
    session_d1._last_centers = {}
    session_d1._frame_count = 4
    session_d1.yolo = _FakeYolo([])
    saw_kf_watchdog = False
    drifted_box = Box(300, 200, 340, 240)
    for _ in range(25):
        drifted_box = Box(drifted_box.x1 + 25, drifted_box.y1,
                          drifted_box.x2 + 25, drifted_box.y2)
        session_d1.csrt.update_ret = (True, drifted_box)
        result = session_d1.process_frame(stitched_d3)
        if result["track"].get("source") == "kf" \
                and session_d1.lost_frames >= 3:
            saw_kf_watchdog = True
        if session_d1.lock_bbox is None:
            break
    assert saw_kf_watchdog, "漂移看门狗应介入（输出 kf 预测框）"
    assert session_d1.lock_bbox is None, \
        "平滑漂移且无法外观确认最终应清锁"

    # D6) 小目标模板匹配（2026-08-17 实测红色方块 30-70px 频繁跌出
    #     0.6 阈值）：小目标（<64px）粗搜/精搜阈值放宽（0.50/0.55）
    #     + 粗搜半径按目标尺寸缩小——30px 目标应能稳定匹配。
    image_d6 = build_test_image(rng)
    patch_d6 = make_noise_patch(rng, size=30)
    image_d6[220:250, 300:330] = patch_d6
    session_d1._templates = {}
    session_d1._histograms = {}
    session_d1._store_template("left", image_d6, Box(305, 225, 325, 245))
    box_d6 = Box(305, 225, 325, 245)
    session_d1._kf["left"] = KalmanBoxTracker(box_d6)
    matched6, score6 = session_d1._match_template(image_d6, box_d6, "left")
    assert matched6 is not None, "30px 小目标应被模板匹配找到"
    mcx6 = (matched6.x1 + matched6.x2) / 2.0
    assert abs(mcx6 - 315.0) < 20.0, (mcx6, matched6)
    assert score6 >= 0.55, score6
    # miss_limit 按目标尺寸缩放
    session_d1.lock_bbox = Box(305, 225, 325, 245)      # 20px → 25 帧
    assert session_d1._miss_limit() == 25, session_d1._miss_limit()
    session_d1.lock_bbox = Box(100, 100, 200, 200)      # 100px → 20 帧
    assert session_d1._miss_limit() == 20, session_d1._miss_limit()
    session_d1.lock_bbox = Box(100, 100, 500, 500)      # 400px → 15 帧
    assert session_d1._miss_limit() == 15, session_d1._miss_limit()

    # D4) 近距裁剪框分叉容忍（2026-08-17）：第二路框贴画面下边缘
    #     （近距目标比画面大，中心被裁剪位移污染）→ y 差 >150px
    #     是裁剪假象不是认错锁，不能丢第二路；旧代码 10 帧后丢。
    image_d4 = build_test_image(rng, w=1280, h=720)
    stitched_d4 = np.hstack([image_d4, image_d4.copy()])
    session_d1._templates = {}
    session_d1._histograms = {}
    session_d1.lock_bbox = Box(300, 100, 980, 620)
    session_d1.lock_view = "left"
    session_d1.lost_frames = 0
    session_d1.lock2_active = True
    session_d1.lock_bbox2 = Box(280, 480, 960, 719)  # 贴下边缘
    session_d1.lock_view2 = "right"
    session_d1.lost_frames2 = 0
    session_d1._kf = {"left": KalmanBoxTracker(session_d1.lock_bbox),
                      "right": KalmanBoxTracker(session_d1.lock_bbox2)}
    session_d1._drift = {}
    session_d1._drift_frames = {}
    session_d1._last_centers = {}
    session_d1.csrt.update_ret = (True, Box(302, 102, 982, 622))
    session_d1.csrt2.update_ret = (True, Box(282, 482, 962, 721))
    session_d1.yolo = _FakeYolo([])
    session_d1._pi_state = {"state": "TRACKING"}
    for _ in range(12):
        session_d1._frame_count += 1
        session_d1.process_frame(stitched_d4)
    assert session_d1.lock2_active, "裁剪框不应触发分叉丢锁"

    print("SESSION_OK")


def test_kalman_size_follow():
    """D5) 尺寸跟随（2026-08-17 流畅性修复）：位置拒收时尺寸照收——
    目标快速变大但位置远跳被门控拒 → 预测框位置不动、尺寸变大
    （输出跟本帧）；旧代码拒收整组测量，尺寸冻结在旧值。"""
    kf = KalmanBoxTracker(Box(400, 300, 480, 380))
    kf.predict(0.04, (0.0, 0.0))
    ok, _ = kf.update(Box(700, 600, 900, 800))   # 远跳 300px + 尺寸 200
    assert not ok, "远跳应被拒"
    _, _, w, h = kf.predicted_box()
    assert w > 100, w      # 尺寸已跟随（旧代码冻结在 80）
    assert h > 100, h
    cx, cy = kf.predicted_center
    assert abs(cx - 440.0) < 60.0 and abs(cy - 340.0) < 60.0, \
        (cx, cy)           # 位置未被拉走
    # update_size 独立调用（服务端印证失败回滚后的尺寸保留路径）
    kf2 = KalmanBoxTracker(Box(100, 100, 140, 140))
    kf2.predict(0.04, (0.0, 0.0))
    kf2.update_size(Box(100, 100, 200, 220))
    _, _, w2, h2 = kf2.predicted_box()
    assert 95 < w2 < 205 and 95 < h2 < 225, (w2, h2)
    print("KALMAN_SIZE_OK")


if __name__ == "__main__":
    test_kalman()
    test_kalman_size_follow()
    test_session_methods()
    print("ALL_SERVER_TESTS_PASSED")
