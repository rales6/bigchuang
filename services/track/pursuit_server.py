# -*- coding: utf-8 -*-
"""视觉追踪逼近服务（3090 端）。

职责：
- POST /ground：接收左右两帧 + 指令，Qwen3-VL-2B 做一次性视觉定位
  （复用 ~/3090/src/qwen_grounded_tracker 的 Qwen3VLTargetGrounder）；
- WS /pursuit_ws：持续接收左右拼接 JPEG，逐帧执行
  YOLO11n 目标匹配（IoU+类别）→ CSRT 兜底 → SGBM 双目深度，
  返回 {track, bbox, depth_m, depth_conf} 给树莓派；
- 测试网页：浏览器打开 GET / 可看左右双画面（带目标框/深度标注），
  输入框随时重新提交识别（POST /reground，结果经 WS 推给树莓派
  驱动其状态机）。

模型与虚拟环境复用 ~/3090 内的现成资源；本服务独立运行在 8021 端口
（8020 已被 arm_grasp_server 占用），不影响 stage1 的 run_3090_server.py
（8011/8000）。

运行（复用 ~/3090/.venv）：
  ~/3090/.venv/bin/python ~/track/3090/pursuit_server.py --port 8021
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import (FastAPI, File, Form, HTTPException, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.responses import HTMLResponse, Response, StreamingResponse

HOME = Path.home()
SRC_3090 = HOME / "3090" / "src"
if str(SRC_3090) not in sys.path:
    sys.path.insert(0, str(SRC_3090))

from qwen_grounded_tracker.domain import BBox, GroundingResult  # noqa: E402
from qwen_grounded_tracker.grounding.qwen_grounder import (  # noqa: E402
    Qwen3VLTargetGrounder,
)
from qwen_grounded_tracker.tracking.csrt_tracker import (  # noqa: E402
    CSRTTargetTracker,
)

from stereo_depth import StereoDepth  # noqa: E402
from pursuit_kalman import KalmanBoxTracker  # noqa: E402
_PURSUIT_FILE = Path(globals().get("__file__", ".")).resolve()
for _candidate_root in (_PURSUIT_FILE.parent, _PURSUIT_FILE.parent.parent):
    if (_candidate_root / "final_work").is_dir() and str(_candidate_root) not in sys.path:
        sys.path.insert(0, str(_candidate_root))
try:
    from final_work.gpu.remote_grounder import RemoteQwenGrounder  # noqa: E402
except ModuleNotFoundError:  # source-extraction unit tests do not execute this adapter
    class RemoteQwenGrounder:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("final_work remote Qwen adapter is unavailable")

# 各相机原始照片空间近似焦距（像素，≈标定 K[0][2]）：ego-motion
# 补偿（车体转动 → 像面位移）与视线偏角换算用；与树莓派
# StereoConfig.cx_raw_left_px / cx_raw_right_px 的标定保持一致。
_FOCAL_RAW_PX = {"left": 573.41, "right": 653.81}


# Qwen target_name 关键词 → COCO 80 类名（用于 YOLO 类别优先匹配）。
_CLASS_KEYWORDS = {
    "bottle": "bottle", "瓶": "bottle", "water bottle": "bottle",
    "plastic bottle": "bottle", "水瓶": "bottle", "矿泉水": "bottle",
    "cup": "cup", "杯": "cup", "mug": "cup", "milk tea": "cup",
    "奶茶": "cup", "drink container": "cup", "takeaway cup": "cup",
    "chair": "chair", "椅": "chair",
    "person": "person", "人": "person", "man": "person", "woman": "person",
    "dog": "dog", "狗": "dog",
    "cat": "cat", "猫": "cat",
    "car": "car", "车": "car",
    "box": "box", "箱": "box", "carton": "box",
    "plant": "potted plant", "花": "potted plant", "pot": "potted plant",
    "backpack": "backpack", "包": "backpack", "bag": "backpack",
    "book": "book", "书": "book",
    "phone": "cell phone", "手机": "cell phone",
    "laptop": "laptop", "电脑": "laptop",
    "keyboard": "keyboard", "键盘": "keyboard",
    "mouse": "mouse", "鼠标": "mouse",
    "tv": "tv", "电视": "tv", "monitor": "tv",
    "sink": "sink", "洗手池": "sink",
    "toilet": "toilet", "马桶": "toilet",
    "door": "door", "门": "door",
    "table": "dining table", "桌子": "dining table", "desk": "dining table",
    "桌": "dining table",
    "bowl": "bowl", "碗": "bowl",
    "banana": "banana", "香蕉": "banana",
    "apple": "apple", "苹果": "apple",
    "ball": "sports ball", "球": "sports ball",
    "teddy": "teddy bear", "熊": "teddy bear", "玩偶": "teddy bear",
}

# 不属于 COCO 80 类、但本项目常被用作追踪目标的语义别名。不能仅因
# YOLO 没有该类别就跳过语义校验，否则“我的手”可能被 Qwen 幻觉成
# 画面里更显眼的红方块并被系统接受。
_NON_COCO_TARGET_ALIASES = (
    ("hand", "hands", "palm", "手掌", "手部", "手"),
    ("cube", "block", "方块", "立方体", "物块"),
    ("square", "正方形"),
)

# Qwen 的通用 grounding 提示词允许返回 floor/wall/window 等场景区域，
# 这对导航任务有用，但对本任务（追踪一个可接近的具体物体）会造成危险的
# “替代目标”幻觉。只有用户指令明确点名这些区域时才允许采纳。
_SCENE_REGION_ALIASES = {
    "floor": ("floor", "ground", "地板", "地面"),
    "wall": ("wall", "墙"),
    "ceiling": ("ceiling", "天花板", "顶棚"),
    "window": ("window", "窗"),
    "door": ("door", "门"),
    "tabletop": ("tabletop", "桌面"),
}

_COLOR_ALIASES = (
    ("red", "红"), ("blue", "蓝"), ("green", "绿"),
    ("yellow", "黄"), ("black", "黑"), ("white", "白"),
    ("orange", "橙"), ("purple", "紫"),
)


def _mapped_coco_classes(target_name: str) -> set[str]:
    lowered = target_name.lower()
    return {coco for keyword, coco in _CLASS_KEYWORDS.items()
            if keyword.lower() in lowered}


def _instruction_target_classes(instruction: str) -> set[str]:
    """从指令中提取目标类，先移除“机器人/小车”主语避免误判 car。"""
    cleaned = (instruction or "").lower()
    for subject in ("机器人小车", "机器人", "小车", "robot car", "robot"):
        cleaned = cleaned.replace(subject, " ")
    return _mapped_coco_classes(cleaned)


def _target_name_matches_instruction(target_name: str,
                                     instruction: str) -> bool:
    """Qwen 返回名称是否与用户要求的目标类别一致。"""
    name = (target_name or "").strip().lower()
    request = (instruction or "").strip().lower()
    if not name or not request:
        return False
    requested_classes = _instruction_target_classes(request)
    if requested_classes and not (
            requested_classes & _mapped_coco_classes(name)):
        return False
    for aliases in _NON_COCO_TARGET_ALIASES:
        # “手机”里的“手”不是 hand 指令，先消除这个中文包含关系。
        request_probe = request.replace("手机", "") \
            if "hand" in aliases else request
        if any(alias in request_probe for alias in aliases) \
                and not any(alias in name for alias in aliases):
            return False
    return True


def _scene_region_allowed(target_name: str, instruction: str) -> bool:
    """拒绝 Qwen 用未被点名的场景区域替代用户目标。"""
    name = (target_name or "").lower()
    request = (instruction or "").lower()
    for aliases in _SCENE_REGION_ALIASES.values():
        if any(alias.lower() in name for alias in aliases):
            return any(alias.lower() in request for alias in aliases)
    return True


def _target_names_compatible(first: str, second: str,
                             instruction: str) -> bool:
    """双目两次 Qwen 输出是否仍指向同一类目标。

    优先用 COCO 类映射；非 COCO 小物体允许共享颜色词或较长名称互为
    子串。其余情况宁可退回单目，也不能把 floor/window 之类拼成一对。
    """
    a = (first or "").strip().lower()
    b = (second or "").strip().lower()
    if not a or not b:
        return False
    if a == b or (min(len(a), len(b)) >= 3 and (a in b or b in a)):
        return True
    mapped_a = _mapped_coco_classes(a)
    mapped_b = _mapped_coco_classes(b)
    if mapped_a and mapped_b:
        return bool(mapped_a & mapped_b)
    request = (instruction or "").lower()
    for aliases in _COLOR_ALIASES:
        requested = any(alias in request for alias in aliases)
        if requested and any(alias in a for alias in aliases) \
                and any(alias in b for alias in aliases):
            return True
    return False


def _initial_box_complete(bbox: BBox, width: int, height: int) -> bool:
    """新锁必须完整落在画面内，避免用半个目标/场景边缘初始化追踪器。"""
    margin = max(2.0, min(width, height) * 0.003)
    return (bbox.x1 > margin and bbox.y1 > margin
            and bbox.x2 < width - margin and bbox.y2 < height - margin)


def _strict_ground_instruction(instruction: str) -> str:
    """在复用通用 Qwen grounder 时覆盖其“近似一个区域”倾向。"""
    return (
        f"{instruction}\n\n"
        "本任务只允许锁定指令所指的那个具体、可移动的目标物。"
        "如果该目标物没有清晰完整地出现在当前图像中，必须返回 "
        "found=false；不要用地板、墙、窗、门、家具或任意相似物替代，"
        "不要为了给出答案而猜测。边界框必须紧贴目标物本身。"
    )


def _ground_instruction_variants(instruction: str) -> tuple[str, ...]:
    """通用的两阶段 Qwen 提示，不对任何目标类别写专用逻辑。"""
    strict = _strict_ground_instruction(instruction)
    scan = (
        f"{strict}\n"
        "请先从左到右、从上到下完整扫描图像，再按用户原始描述定位。"
        "目标可能很小、透明、低对比、部分遮挡、角度变化或不是常见"
        "数据集类别；描述中的颜色、材质、形状、文字、用途和附属结构"
        "都属于身份线索。只要能看到符合描述的具体目标，就返回紧贴其"
        "完整可见部分的边界框；确实看不到才返回 found=false。"
    )
    return strict, scan


def _bbox_iou(a: BBox, b: BBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _instruction_requires_red(instruction: str) -> bool:
    lowered = instruction.lower()
    return ("红" in instruction) or ("red" in lowered)


def _red_mask(region: np.ndarray) -> np.ndarray:
    """红色掩码（鲜艳红才认，与 _region_has_red 同一套判据）。

    2026-08-18 加固：旧判据 (r>70, 1.25×) 会把暗红木门/偏红物体当
    红色（Qwen 连续锁"door"过校验）；鲜艳红方块要求高亮度 + 强红
    主导（r>100、1.5×、差>40）。
    """
    r = region[:, :, 2].astype(np.int32)
    g = region[:, :, 1].astype(np.int32)
    b = region[:, :, 0].astype(np.int32)
    mask = (r > 100) & (r > g * 1.5) & (r > b * 1.5) & (r - g > 40)
    return mask.astype(np.uint8) * 255


def _region_has_red(image: np.ndarray, bbox: BBox,
                    min_fraction: float = 0.08) -> bool:
    """框内红色像素占比是否达标（宽容差，允许框松/部分遮挡）。

    实测教训：目标不在画面里时，Qwen 会以 0.99 置信度锁定无关的
    灰白物体；红色指令必须校验框内真的有红色，否则幻觉锁定会让
    车追着墙跑。
    """
    height, width = image.shape[:2]
    x1 = int(max(0, min(width - 1, round(bbox.x1))))
    x2 = int(max(x1 + 1, min(width, round(bbox.x2))))
    y1 = int(max(0, min(height - 1, round(bbox.y1))))
    y2 = int(max(y1 + 1, min(height, round(bbox.y2))))
    region = image[y1:y2, x1:x2].astype(np.int32)
    if region.size == 0:
        return False
    r, g, b = (region[:, :, 2], region[:, :, 1], region[:, :, 0])
    # 与 _red_mask 同一套严格判据（鲜艳红；暗红木门等不再误认）
    red = (r > 100) & (r > g * 1.5) & (r > b * 1.5) & (r - g > 40)
    return bool(float(red.mean()) >= min_fraction)


class StreamHub:
    """浏览器流推送枢纽：/stream.mjpg（视频）与 /annotations（标注）的事件源。

    线程安全纪律（跨线程通知的统一入口）：
    - 所有注册/注销/put/set 只在事件循环线程执行（WS 处理器协程内）；
    - notify_* 从任意线程调用，内部统一 loop.call_soon_threadsafe 归一
      （从事件循环线程调用也合法，开销极小）；
    - 标注队列 maxsize=1 latest-wins（put_nowait 满则丢旧），感知管线
      永不阻塞、永不 await 客户端——慢浏览器只影响自己的队列。
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._annot: dict[int, asyncio.Queue] = {}
        self._mjpeg: dict[int, asyncio.Event] = {}
        self._next_key = 0

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._loop is None:
            self._loop = loop

    # ---- 事件循环线程内调用（WS 处理器协程） ----

    def register_annot(self) -> tuple[int, asyncio.Queue]:
        key = self._next_key
        self._next_key += 1
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._annot[key] = queue
        return key, queue

    def unregister_annot(self, key: int) -> None:
        self._annot.pop(key, None)

    def register_mjpeg(self) -> tuple[int, asyncio.Event]:
        key = self._next_key
        self._next_key += 1
        event = asyncio.Event()
        self._mjpeg[key] = event
        return key, event

    def unregister_mjpeg(self, key: int) -> None:
        self._mjpeg.pop(key, None)

    # ---- 任意线程可调的通知入口 ----

    def notify_latest(self, result: dict[str, Any]) -> None:
        """新帧结果：唤醒所有 mjpg 生成器 + 推给所有标注队列。"""
        self._dispatch(self._notify_latest, result)

    def notify_pi_state(self, state: dict[str, Any]) -> None:
        """树莓派状态更新（含 fps 统计），原样转发给浏览器。"""
        self._dispatch(self._notify_pi_state, state)

    def notify_ground(self, message: dict[str, Any]) -> None:
        """网页重新识别结果，广播给浏览器。"""
        self._dispatch(self._notify_ground, message)

    def notify_push_fps(self, fps: float) -> None:
        """MJPEG 流推送帧率（每 5s），广播给浏览器 HUD。"""
        self._dispatch(self._notify_push_fps, {"type": "stream_stats",
                                               "push_fps": fps})

    def _dispatch(self, fn: Any, payload: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(fn, payload)

    # ---- 事件循环线程内执行（call_soon_threadsafe 目标） ----

    @staticmethod
    def _push(queue: asyncio.Queue, message: dict[str, Any]) -> None:
        """latest-wins 入队：满则丢旧消息，绝不阻塞。"""
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            pass

    def _notify_latest(self, result: dict[str, Any]) -> None:
        for event in self._mjpeg.values():
            event.set()
        for queue in self._annot.values():
            self._push(queue, result)

    def _notify_pi_state(self, state: dict[str, Any]) -> None:
        for queue in self._annot.values():
            self._push(queue, dict(state))

    def _notify_ground(self, message: dict[str, Any]) -> None:
        for queue in self._annot.values():
            self._push(queue, dict(message))

    def _notify_push_fps(self, message: dict[str, Any]) -> None:
        for queue in self._annot.values():
            self._push(queue, dict(message))


class PursuitSession:
    """一个任务会话：锁定目标 → YOLO/CSRT 逐帧追踪 → 双目深度。"""

    def _red_candidate_boxes(
            self, image: np.ndarray,
            center: tuple[float, float] | None = None,
            search_radius: int | None = None) -> list[BBox]:
        """返回搜索区域内全部可信红色连通域，而不是只取最大块。

        手遮挡方块时，最大红色连通域可能来自肤色/衣物；旧实现先取
        最大块再做身份校验，一旦最大块未通过就会漏掉旁边仍可见的
        真目标。这里保留全部候选，交给命令属性、初始锚点和运动预测
        联合排序。
        """
        height, width = image.shape[:2]
        if center is not None and search_radius is not None:
            cx = max(0.0, min(float(width - 1), center[0]))
            cy = max(0.0, min(float(height - 1), center[1]))
            x1 = int(max(0, cx - search_radius))
            x2 = int(min(width, cx + search_radius))
            y1 = int(max(0, cy - search_radius))
            y2 = int(min(height, cy + search_radius))
        else:
            x1, y1, x2, y2 = 0, 0, width, height
        if x2 - x1 < 8 or y2 - y1 < 8:
            return []
        region = image[y1:y2, x1:x2]
        mask = _red_mask(region)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[BBox] = []
        for contour in contours:
            # 允许远处/运动模糊后只剩一部分的目标进入候选；后续身份门
            # 会淘汰噪点。旧固定 300px 会直接漏掉快速移动的小方块。
            if cv2.contourArea(contour) < 80.0:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bw < 6 or bh < 6:
                continue
            boxes.append(BBox(float(x1 + bx), float(y1 + by),
                              float(x1 + bx + bw), float(y1 + by + bh)))
        return boxes

    def _red_target_box(self, image: np.ndarray, center: tuple[float, float],
                        search_radius: int) -> BBox | None:
        """红色目标精确边界框：预测中心附近红色掩码分割（2026-08-18）。

        红色方块是纯色目标，颜色分割比模板/CSRT/KF 都贴合实际物体
        （用户反复反馈框不贴物体）。搜索半径内最大连通域作为目标，
        返回 BBox；无有效红色区域返回 None。
        """
        boxes = self._red_candidate_boxes(image, center, search_radius)
        if not boxes:
            return None
        # grounding 尚无身份锚点时，优先离 Qwen 框中心最近且面积较大的
        # 连通域；追踪阶段使用下面带身份分数的排序函数。
        return min(boxes, key=lambda box: (
            math.hypot((box.x1 + box.x2) / 2.0 - center[0],
                       (box.y1 + box.y2) / 2.0 - center[1])
            / max(math.sqrt(box.area), 1.0)))

    def _rank_red_candidates(
            self, image: np.ndarray, view: str,
            predicted_center: tuple[float, float] | None = None,
            search_radius: int | None = None,
            reference_bbox: BBox | None = None) -> list[tuple[BBox, float]]:
        """命令为红色目标时，对局部或全画面候选做身份保持排序。"""
        height, width = image.shape[:2]
        diag = max(1.0, math.hypot(width, height))
        ranked: list[tuple[BBox, float]] = []
        for bbox in self._red_candidate_boxes(
                image, predicted_center, search_radius):
            if not self._command_candidate_ok(image, bbox, view):
                continue
            anchor = self._score_anchor_at_bbox(image, bbox, view)
            proximity = 0.5
            if predicted_center is not None:
                distance = math.hypot(
                    (bbox.x1 + bbox.x2) / 2.0 - predicted_center[0],
                    (bbox.y1 + bbox.y2) / 2.0 - predicted_center[1])
                scale = float(search_radius or int(diag * 0.45))
                proximity = math.exp(-distance / max(scale, 1.0))
            size_score = 0.5
            if reference_bbox is not None and reference_bbox.area > 0:
                ratio = bbox.area / reference_bbox.area
                size_score = math.exp(-abs(math.log(max(ratio, 1e-6))))
            # 初始 Qwen 外观是主证据，运动预测只是排序提示，不能让靠近
            # 旧框的手覆盖更远但身份正确的目标。
            score = 0.68 * anchor + 0.22 * proximity + 0.10 * size_score
            ranked.append((bbox, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def _red_recover_candidate(
            self, image: np.ndarray, view: str,
            predicted_center: tuple[float, float] | None,
            search_radius: int | None,
            reference_bbox: BBox | None) -> tuple[BBox | None, float]:
        ranked = self._rank_red_candidates(
            image, view, predicted_center, search_radius, reference_bbox)
        return ranked[0] if ranked else (None, 0.0)

    def _ground_yolo_pair(
            self, left_image: np.ndarray, right_image: np.ndarray,
            instruction: str,
            preferred: dict[str, BBox] | None = None
            ) -> tuple[BBox, BBox, str, float] | None:
        """Qwen 框缺失时，用指令类别筛选并双目配对 YOLO 候选。

        只对命令能映射到 COCO 类别（如 cup/bottle）启用。两路必须
        类别相同、y 位置一致，且裁块外观/尺寸相近；因此不是“看到任意
        瓶子就追”，而是命令约束下的双目初始化兜底。
        """
        expected = _instruction_target_classes(instruction)
        if not expected:
            return None
        preferred = preferred or {}
        height, width = left_image.shape[:2]
        scale = width / 1280.0

        def collect(view: str, image: np.ndarray
                    ) -> list[tuple[BBox, str, float]]:
            candidates: list[tuple[BBox, str, float]] = []
            preferred_box = preferred.get(view)
            preferred_class = next(iter(expected))
            if preferred_box is not None:
                candidates.append((preferred_box, preferred_class, 1.0))
            for bbox, class_name, conf in self._yolo_detect(image, conf=0.12):
                if class_name not in expected:
                    continue
                bbox = bbox.clamp(width, height)
                if not _initial_box_complete(bbox, width, height):
                    continue
                cx = (bbox.x1 + bbox.x2) / 2.0
                plausible = (250.0 * scale <= cx <= float(width)
                             if view == "left"
                             else 0.0 <= cx <= 1000.0 * scale)
                if plausible:
                    candidates.append((bbox, class_name, float(conf)))
            return candidates

        left_candidates = collect("left", left_image)
        right_candidates = collect("right", right_image)
        best: tuple[BBox, BBox, str, float] | None = None
        best_score = -1.0
        for left_box, left_class, left_conf in left_candidates:
            for right_box, right_class, right_conf in right_candidates:
                if left_class != right_class:
                    continue
                if not self._cross_view_y_ok(left_box, right_box):
                    continue
                left_tpl, left_hist = self._extract_template(
                    left_image, left_box)
                right_tpl, right_hist = self._extract_template(
                    right_image, right_box)
                if left_tpl.size == 0 or right_tpl.size == 0:
                    continue
                resized_right = cv2.resize(
                    right_tpl, (left_tpl.shape[1], left_tpl.shape[0]),
                    interpolation=cv2.INTER_AREA)
                corr = float(cv2.matchTemplate(
                    left_tpl, resized_right, cv2.TM_CCOEFF_NORMED)[0, 0])
                hist_sim = float(cv2.compareHist(
                    left_hist, right_hist, cv2.HISTCMP_CORREL))
                appearance = 0.5 * max(0.0, corr) \
                    + 0.5 * max(0.0, hist_sim)
                left_h = max(1.0, left_box.y2 - left_box.y1)
                right_h = max(1.0, right_box.y2 - right_box.y1)
                size_sim = math.exp(-abs(math.log(left_h / right_h)))
                left_y = (left_box.y1 + left_box.y2) / 2.0
                right_y = (right_box.y1 + right_box.y2) / 2.0
                y_sim = math.exp(-abs(left_y - right_y) / 100.0)
                score = (left_conf + right_conf + 0.65 * appearance
                         + 0.20 * size_sim + 0.20 * y_sim)
                if score > best_score:
                    best_score = score
                    best = (left_box, right_box, left_class, score)
        # 两个最低置信候选(0.12+0.12)仍必须有较好的双目外观/几何；
        # 正常 0.4+ 检测会明显超过该阈值。
        return best if best is not None and best_score >= 0.75 else None

    def _command_candidate_ok(self, image: np.ndarray, bbox: BBox,
                              view: str) -> bool:
        """候选是否满足命令属性与 Qwen 初始身份。

        颜色/形状只在命令明确提出时启用；对瓶子、书等其他命令不会
        强行套用红色规则。初始锚点始终作为身份底线。
        """
        width = max(1.0, bbox.x2 - bbox.x1)
        height = max(1.0, bbox.y2 - bbox.y1)
        if self._square_target:
            ratio = width / height
            # 透视、倾斜和部分遮挡下放宽，但排除细长手臂/人体框。
            if ratio < 0.45 or ratio > 2.2:
                return False
        if self._red_target and not _region_has_red(
                image, bbox, min_fraction=0.22):
            return False
        if view in self._anchor_templates:
            # 阈值低于模板正式采纳阈值，允许光照/尺度变化；手与方块
            # 的色彩和纹理差异远大于此阈值。
            anchor_score = self._score_anchor_at_bbox(image, bbox, view)
            if anchor_score < 0.28:
                # 实车 2026-09-25：同一红色小方块由 30px 变为 70px
                # 后，左目初始外观分数降到 0.208，右目仍为 0.364。
                # 只对指令明确要求的红色方块，在颜色纯度和方形轮廓
                # 同时很强时接受这个尺度变化；肤色手、红色细条以及
                # 其他目标仍走原来的身份阈值。
                if not (self._red_target and self._square_target
                        and anchor_score >= 0.18
                        and 0.60 <= width / height <= 1.67
                        and _region_has_red(
                            image, bbox, min_fraction=0.55)):
                    return False
        return True

    def _refine_command_candidate(self, image: np.ndarray, bbox: BBox,
                                  view: str) -> BBox | None:
        """按命令属性把通用候选收紧到真正目标，再做身份校验。"""
        refined = bbox
        if self._red_target:
            center = ((bbox.x1 + bbox.x2) / 2.0,
                      (bbox.y1 + bbox.y2) / 2.0)
            radius = int(max(80.0, bbox.x2 - bbox.x1,
                             bbox.y2 - bbox.y1) * 1.5)
            refined = self._red_target_box(image, center, radius)
            if refined is None:
                return None
        return refined if self._command_candidate_ok(
            image, refined, view) else None

    def __init__(self, qwen_model: str, yolo_model: str,
                 calibration_yaml: str, device: str = "cpu",
                 display_swap: bool = False, use_depth: bool = True,
                 qwen_hub_url: str = "") -> None:
        self.qwen_model = qwen_model
        self.yolo_model = yolo_model
        self.device = device
        self.qwen_hub_url = qwen_hub_url.rstrip("/")
        # 摄像头位置改动后双目标定失效：--no-depth 关闭深度计算，
        # 距离只由树莓派雷达提供。
        self.use_depth = bool(use_depth)
        # 仅影响测试网页展示：实车摄像头物理左右与标定命名相反时置
        # True（/frame.jpg 交换左右半幅并同步平移 bbox）。管线内部
        # （校正/追踪/树莓派控制）始终保持标定顺序，深度不受影响。
        self.display_swap = bool(display_swap)
        self.frame_width = 0
        self.frame_height = 0
        self.stereo = StereoDepth(calibration_yaml)

        # /ground（线程池）与 WS 帧处理之间的互斥：ground 期间目标位置
        # 即将刷新，帧结果全是旧视角的陈旧数据，直接返回 busy 丢弃。
        self._ground_lock = threading.Lock()

        # 浏览器流推送枢纽（/stream.mjpg + /annotations）
        self.hub = StreamHub()
        # CSRT 兜底用：锁定建立时初始化一次；yolo/tpl 采纳不再每帧重置
        # （2026-08-18 帧率重构：每帧 csrt.initialize 实测 104ms 是 2fps 主因）
        self._csrt_ready = False
        self._csrt2_ready = False
        self._yolo_miss_streak = 0   # YOLO 连续未确认计数（模板触发条件）
        self._yolo_miss_streak2 = 0
        self._red_target = False     # 指令含红/red 时启用颜色分割精确框
        self._square_target = False  # 指令含方块/cube/square 时启用形状约束
        # 摄像头分辨率相对 1280 宽的缩放（pursuit_start hello 里按
        # frame_width 设置）：ego-motion 焦距按比例缩放。
        self._focal_scale = 1.0

        # 测试网页：保存最新拼接帧 JPEG 与逐帧结果（HTTP 轮询用）
        self._latest_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_result: dict[str, Any] | None = None
        # 网页帧缓存（2026-08-17 帧率修复）：/frame.jpg 每次请求都
        # 重新解码+画框+编码（720p ~25ms）。网页 50ms 轮询 = 每秒 20
        # 次 ≈ 50% 单核，GIL 拖垮 WS 帧处理（实测 frame_ms 390-450ms、
        # 服务 CPU 59%）。帧内容未变时直接返回缓存。
        self._frame_seq = 0
        self._cache_seq = -1
        self._cache_state: str | None = None
        self._cache_jpeg: bytes | None = None
        # 树莓派状态机状态（1Hz 上行），网页实时显示
        self._pi_state: dict[str, Any] = {}
        # 树莓派 WS 连接（网页 /reground 的结果推给它）
        self._ws: WebSocket | None = None
        self._loop: Any = None
        # 最近一次网页识别结果：树莓派重连后补发，避免指令丢失
        self._last_ground_push: dict[str, Any] | None = None

        self.grounder: Qwen3VLTargetGrounder | RemoteQwenGrounder | None = None
        self.yolo: Any | None = None
        self.csrt = CSRTTargetTracker()
        # 第二路（另一摄像头）独立追踪：示教停车需要两路目标中心对齐
        self.csrt2 = CSRTTargetTracker()

        self.target_name = ""
        self.lock_bbox: BBox | None = None      # 主视图（左目优先）内坐标
        self.lock_view = "left"
        self.lock_source = ""
        self.lost_frames = 0
        # 外观模板（每视图最多 3 张近期快照 + HSV 直方图）：
        # 任意目标通用的"每帧重检"，多模板抗尺度/光照变化，
        # 锁漂移时用模板匹配把锁拉回真正的目标。
        self._templates: dict[str, list[np.ndarray]] = {}
        self._histograms: dict[str, list[np.ndarray]] = {}
        # 初始 Qwen 锁定外观永久保留一份。旧逻辑每次 YOLO/模板采纳都
        # 刷新三张快照，3 帧后初始身份即被挤掉；一次误测会把错误外观
        # 逐帧写回，形成“模板自我污染”。锚点只在重新 ground 时更换。
        self._anchor_templates: dict[str, np.ndarray] = {}
        self._anchor_histograms: dict[str, np.ndarray] = {}
        # 模板保持原始像素尺度（只对超大裁块降采样防匹配开销爆炸）：
        # 2026-08-16 实测固定 48px 会引入 ~9% 尺度失配，细纹理目标
        # 半分辨率粗搜相关系数从 ~1.0 崩到 0.26，粗搜直接失效。
        # 2026-08-17 上限 64→256（近距漂移根治的一环）：旧上限下
        # 模板最大 64px、精搜尺度 0.5~2.0× → 最多匹配 128px，而近距
        # 目标 300~600px，模板匹配永远找不到真实位置（相关系数在
        # 大目标内部平坦），唯一能拉回锁的机制在近距静默失效。
        self._tpl_max_size = 256
        self.lock2_active = False
        self.lock_bbox2: BBox | None = None     # 第二路（另一视图）内坐标
        self.lock_view2 = ""
        self.lock_source2 = ""
        self.lost_frames2 = 0
        # 2026-08-16 调快：30 帧(~3s)的陈旧锁会让树莓派持续朝错误方向
        # 转；15 帧(~1-1.5s)内恢复不了就尽早判 lost 交给重搜。
        self.lost_threshold = 15
        # 逐帧计数器：YOLO 隔帧跑（省时给模板匹配），模板/CSRT 每帧跑
        self._frame_count = 0
        # 上帧锁中心（模板搜索运动预测用，卡尔曼未就绪时的退化）
        self._last_centers: dict[str, tuple[float, float]] = {}
        self._tpl_miss_frames = 0
        # 双路 y 持续分叉计数：超过阈值丢掉第二路（认错锁不会自愈）
        self._y_divergence_frames = 0
        # 卡尔曼门控/平滑（防随机跳框）+ ego-motion 补偿（2026-08-16）
        self._kf: dict[str, KalmanBoxTracker] = {}
        self._pi_motion: dict[str, float] = {"v_mm_s": 0.0, "w_mrad_s": 0.0}
        self._last_frame_t: float | None = None
        # 平滑漂移看门狗（2026-08-17 近距 475px/s 漂移根治）：
        # 输出中心横向速度 EMA（扣除 ego-motion 期望）与连续帧计数
        self._drift: dict[str, float] = {}
        self._drift_frames: dict[str, int] = {}
        # 完全丢锁后不再永久遗忘 Qwen 首次确认的目标。每隔若干帧在
        # 两路画面做一次低频全局重识别；只有双目同时命中才恢复，避免
        # 单路模板把相似背景误认成目标。
        # 普通外观全局搜索较重，3 帧一次；命令属性（如红色）是轻量
        # 分割，丢锁后每帧搜索，避免小车一转动目标跨出局部窗后等待。
        self._passive_reid_period = 3
        self._reid_notice_frames = 0

    # ---------------- 识别 ----------------

    def ground(self, left_image: np.ndarray, right_image: np.ndarray,
               instruction: str) -> dict[str, Any]:
        with self._ground_lock:
            return self._ground_locked(left_image, right_image, instruction)

    def _ground_locked(self, left_image: np.ndarray, right_image: np.ndarray,
                       instruction: str) -> dict[str, Any]:
        if self.grounder is None:
            if self.qwen_hub_url:
                print(f"[pursuit] 复用共享 Qwen Hub {self.qwen_hub_url}")
                self.grounder = RemoteQwenGrounder(self.qwen_hub_url)
            else:
                print(f"[pursuit] 加载 Qwen 模型 {self.qwen_model} …")
                self.grounder = Qwen3VLTargetGrounder(
                    model_path=self.qwen_model,
                    device=self.device,
                    local_files_only=True,
                    max_new_tokens=160,
                    max_time_seconds=30.0,
                    min_visual_tokens=128,
                    max_visual_tokens=512,
                    minimum_box_area_ratio=0.0002,
                )
        # 分路识别避免超宽拼接图压缩定位精度；第二路使用第一路目标裁块
        # 作为 reference_image，防止两次独立问答分别返回 floor/window。
        height, width = left_image.shape[:2]
        last: GroundingResult | None = None
        require_red = _instruction_requires_red(instruction)
        # 重新识别是一次事务：同一任务已有可靠身份时，先保存不可变
        # 锚点。如果这次 Qwen 因运动模糊/暂时遮挡失败，恢复旧身份，
        # 不能让一次失败复检把后续实时重识别的依据全部抹掉。
        preserve_previous = bool(self._anchor_templates) and \
            _target_name_matches_instruction(self.target_name, instruction)
        previous_identity = None
        if preserve_previous:
            previous_identity = {
                "target_name": self.target_name,
                "lock_view": self.lock_view,
                "lock_view2": self.lock_view2,
                "templates": {k: list(v) for k, v in self._templates.items()},
                "histograms": {k: list(v) for k, v in self._histograms.items()},
                "anchors": dict(self._anchor_templates),
                "anchor_hists": dict(self._anchor_histograms),
            }

        def restore_previous_identity() -> None:
            if previous_identity is None:
                self._clear_lock()
                return
            self._clear_lock()
            self.target_name = previous_identity["target_name"]
            self.lock_view = previous_identity["lock_view"]
            self.lock_view2 = previous_identity["lock_view2"]
            self._templates = previous_identity["templates"]
            self._histograms = previous_identity["histograms"]
            self._anchor_templates = previous_identity["anchors"]
            self._anchor_histograms = previous_identity["anchor_hists"]
            print("[pursuit] Qwen 复检失败，保留原目标身份供实时重识别")

        self._square_target = any(token in instruction.lower() for token in (
            "方块", "物块", "正方形", "立方体", "cube", "block", "square"))
        ground_prompts = _ground_instruction_variants(instruction)
        self._red_target = require_red
        self._clear_lock()
        self._csrt_ready = False
        self._csrt2_ready = False
        self._last_centers = {}

        def candidate(view: str, image: np.ndarray,
                      reference: np.ndarray | None = None
                      ) -> tuple[GroundingResult, BBox] | None:
            nonlocal last
            image_h, image_w = image.shape[:2]
            attempts: list[tuple[np.ndarray, int, int, str, str]] = [
                (image, 0, 0, ground_prompts[0], "full"),
                (image, 0, 0, ground_prompts[-1], "full-scan"),
            ]
            # 全图视觉 token 会把小目标压得过小；全图失败后使用四个
            # 有重叠的通用瓦片放大细节。瓦片与类别无关，任意描述均可用。
            tile_w = int(image_w * 0.62)
            tile_h = int(image_h * 0.68)
            for x0 in (0, image_w - tile_w):
                for y0 in (0, image_h - tile_h):
                    attempts.append((
                        image[y0:y0 + tile_h, x0:x0 + tile_w],
                        x0, y0, ground_prompts[-1], f"tile-{x0}-{y0}"))

            for attempt_image, off_x, off_y, prompt, attempt_name in attempts:
                result: GroundingResult = self.grounder.ground(
                    attempt_image, prompt, reference_image=reference)
                last = result
                if not result.found or result.bbox is None:
                    continue
                name = str(result.target_name or "")
                confidence = float(
                    getattr(result, "confidence", 0.0) or 0.0)
                if confidence < 0.3:
                    print(f"[pursuit] Qwen 置信度过低 {confidence:.2f} "
                          f"view={view} attempt={attempt_name}，跳过")
                    continue
                if not _scene_region_allowed(name, instruction):
                    print(f"[pursuit] Qwen 用场景区域 {name!r} 替代目标 "
                          f"view={view} attempt={attempt_name}，跳过")
                    continue
                if not _target_name_matches_instruction(name, instruction):
                    print(f"[pursuit] Qwen 类别与指令不一致 view={view} "
                          f"instruction={instruction!r} got={name!r}，跳过")
                    continue
                crop_h, crop_w = attempt_image.shape[:2]
                crop_bbox = result.bbox.clamp(crop_w, crop_h)
                if not _initial_box_complete(crop_bbox, crop_w, crop_h):
                    # 瓦片边缘只看到半个目标时不建立污染锚点；重叠瓦片
                    # 或另一相机仍有机会获得完整框。
                    continue
                raw_bbox = BBox(
                    crop_bbox.x1 + off_x, crop_bbox.y1 + off_y,
                    crop_bbox.x2 + off_x, crop_bbox.y2 + off_y).clamp(
                        width, height)
                if raw_bbox.area <= 0 \
                        or raw_bbox.area > width * height * 0.40:
                    print(f"[pursuit] Qwen 框面积不合理 view={view} "
                          f"ratio={raw_bbox.area / (width * height):.1%}，跳过")
                    continue
                if require_red:
                    if not _region_has_red(
                            image, raw_bbox, min_fraction=0.2):
                        print(f"[pursuit] 红色校验未过 view={view} "
                              f"bbox={raw_bbox}")
                        continue
                    center = ((raw_bbox.x1 + raw_bbox.x2) / 2.0,
                              (raw_bbox.y1 + raw_bbox.y2) / 2.0)
                    radius = int(max(80.0, raw_bbox.x2 - raw_bbox.x1,
                                     raw_bbox.y2 - raw_bbox.y1) * 1.5)
                    segmented = self._red_target_box(image, center, radius)
                    if segmented is not None:
                        raw_bbox = segmented.clamp(width, height)
                if not _initial_box_complete(raw_bbox, width, height):
                    continue
                cx = (raw_bbox.x1 + raw_bbox.x2) / 2.0
                scale = width / 1280.0
                plausible = (250.0 * scale <= cx <= float(width)
                             if view == "left"
                             else 0.0 <= cx <= 1000.0 * scale)
                if not plausible:
                    print(f"[pursuit] 初始锁定位置不合理 view={view} "
                          f"cx={cx:.0f}，跳过")
                    continue
                print(f"[pursuit] Qwen 定位成功 view={view} "
                      f"attempt={attempt_name}")
                return result, raw_bbox
            return None

        def crop(image: np.ndarray, bbox: BBox) -> np.ndarray:
            x1, y1 = int(bbox.x1), int(bbox.y1)
            x2, y2 = int(math.ceil(bbox.x2)), int(math.ceil(bbox.y2))
            return image[y1:y2, x1:x2].copy()

        primary_view = ""
        primary_image: np.ndarray | None = None
        primary: tuple[GroundingResult, BBox] | None = candidate(
            "left", left_image)
        secondary: tuple[GroundingResult, BBox] | None = None
        secondary_view = ""
        secondary_image: np.ndarray | None = None
        if primary is not None:
            primary_view, primary_image = "left", left_image
            secondary = candidate("right", right_image,
                                  crop(left_image, primary[1]))
            secondary_view, secondary_image = "right", right_image
        else:
            primary = candidate("right", right_image)
            if primary is not None:
                primary_view, primary_image = "right", right_image
                secondary = candidate("left", left_image,
                                      crop(right_image, primary[1]))
                secondary_view, secondary_image = "left", left_image

        # Qwen 是身份入口；当它已尽力扫描但仍缺一目/两目框时，如果原始
        # 描述恰能映射到 YOLO 已知类别，就用 Qwen 已有框（若有）和两目
        # 检测候选做外观+几何配对。未知类别仍完全由 Qwen 处理。
        if primary is None or secondary is None:
            preferred: dict[str, BBox] = {}
            if primary is not None:
                preferred[primary_view] = primary[1]
            if secondary is not None:
                preferred[secondary_view] = secondary[1]
            fallback = self._ground_yolo_pair(
                left_image, right_image, instruction, preferred)
            if fallback is not None:
                left_box, right_box, class_name, pair_score = fallback

                class _FallbackResult:
                    found = True
                    bbox = None
                    target_name = instruction
                    confidence = min(0.99, pair_score / 3.0)
                    message = f"Qwen+YOLO stereo fallback: {class_name}"
                    raw_text = message

                fallback_result = _FallbackResult()
                primary = (fallback_result, left_box)
                primary_view, primary_image = "left", left_image
                secondary = (fallback_result, right_box)
                secondary_view, secondary_image = "right", right_image
                print(f"[pursuit] Qwen 后备双目候选锁定 "
                      f"class={class_name} score={pair_score:.2f}")

        if primary is None or primary_image is None:
            restore_previous_identity()
            return {"found": False,
                    "target_name": getattr(last, "target_name", "unknown"),
                    "message": (getattr(last, "message", "") if last else "")
                    or (getattr(last, "raw_text", "")[:200] if last else "")
                    or "Qwen did not return a valid target box"}

        primary_result, primary_bbox = primary
        self.target_name = str(primary_result.target_name or "target")
        self.lock_bbox = primary_bbox
        self.lock_view = primary_view
        self.lock_source = "qwen"
        self.lost_frames = 0
        self.csrt.initialize(primary_image, primary_bbox)
        self._csrt_ready = True
        self._store_template(primary_view, primary_image, primary_bbox,
                             anchor=True)
        self._kf[primary_view] = KalmanBoxTracker(primary_bbox)
        print(f"[pursuit] 锁定目标 {self.target_name!r} "
              f"view={primary_view} raw={primary_bbox}")

        if secondary is not None and secondary_image is not None:
            second_result, second_bbox = secondary
            same_name = _target_names_compatible(
                self.target_name, str(second_result.target_name or ""),
                instruction)
            y1 = (primary_bbox.y1 + primary_bbox.y2) / 2.0
            y2 = (second_bbox.y1 + second_bbox.y2) / 2.0
            y_limit = max(150.0, 0.35 * min(
                primary_bbox.y2 - primary_bbox.y1,
                second_bbox.y2 - second_bbox.y1))
            if not same_name:
                print(f"[pursuit] 第二路语义不一致 "
                      f"{self.target_name!r}/{second_result.target_name!r}，"
                      "退回单目")
            elif abs(y1 - y2) > y_limit:
                print(f"[pursuit] 第二路 y 与主路不一致 "
                      f"({y2:.0f} vs {y1:.0f})，退回单目")
            else:
                self.lock2_active = True
                self.lock_bbox2 = second_bbox
                self.lock_view2 = secondary_view
                self.lock_source2 = "qwen"
                self.lost_frames2 = 0
                self.csrt2.initialize(secondary_image, second_bbox)
                self._csrt2_ready = True
                self._store_template(secondary_view, secondary_image,
                                     second_bbox, anchor=True)
                self._kf[secondary_view] = KalmanBoxTracker(second_bbox)
                print(f"[pursuit] 第二路锁定 view={secondary_view} "
                      f"raw={second_bbox}")
        # 本任务的终点由左右两张掩码共同定义；单目锁无法证明是同一
        # 目标，也永远不能完成示教停车。宁可继续旋转重搜，不在单目
        # 猜测上驱动车辆。
        if not self.lock2_active or self.lock_bbox2 is None:
            rejected_name = self.target_name
            restore_previous_identity()
            return {
                "found": False,
                "target_name": rejected_name,
                "message": "目标未通过左右双目一致性确认，请调整视角后重试",
            }
        return {
            "found": True,
            "target_name": self.target_name,
            "confidence": float(primary_result.confidence or 0.0),
            "view": self.lock_view,
            "bbox": [float(primary_bbox.x1), float(primary_bbox.y1),
                     float(primary_bbox.x2), float(primary_bbox.y2)],
            "view2": self.lock_view2,
            "bbox2": ([float(self.lock_bbox2.x1), float(self.lock_bbox2.y1),
                       float(self.lock_bbox2.x2), float(self.lock_bbox2.y2)]
                      if self.lock2_active and self.lock_bbox2 else None),
        }

    def _clear_lock(self, preserve_identity: bool = False) -> None:
        """清除实时锁。

        跟踪自然丢失时保留 Qwen 初始外观与两路视图身份，使目标离开
        画面后再次进入时可以自动找回；开始新识别或识别失败时仍彻底
        清空，绝不把上一任务的目标带到下一任务。
        """
        self.lock_bbox = None
        self.lost_frames = 0
        self.lock_source = ""
        if not preserve_identity:
            self._templates = {}
            self._histograms = {}
            self._anchor_templates = {}
            self._anchor_histograms = {}
        self.lock2_active = False
        self.lock_bbox2 = None
        if not preserve_identity:
            self.lock_view2 = ""
        self.lock_source2 = ""
        self.lost_frames2 = 0
        self._tpl_miss_frames = 0
        self._kf = {}
        self._drift = {}
        self._drift_frames = {}
        self._last_centers = {}
        self._csrt_ready = False
        self._csrt2_ready = False
        self._pi_motion = {"v_mm_s": 0.0, "w_mrad_s": 0.0}
        if not preserve_identity:
            self._reid_notice_frames = 0

    # ---------------- 逐帧处理 ----------------

    def process_frame(self, stitched: np.ndarray) -> dict[str, Any]:
        # ground 期间（可能数十秒）跳过帧处理：既不阻塞 Qwen 推理，
        # 也不返回旧视角的陈旧结果。
        if not self._ground_lock.acquire(blocking=False):
            return {"type": "frame_result",
                    "track": {"status": "busy", "bbox": None,
                              "name": self.target_name, "source": ""},
                    "depth_m": None, "depth_conf": 0.0, "frame_ms": 0.0}
        try:
            return self._process_frame_locked(stitched)
        finally:
            self._ground_lock.release()

    def _process_frame_locked(self, stitched: np.ndarray) -> dict[str, Any]:
        started = time.perf_counter()
        self._frame_count += 1
        yolo_raw_box: BBox | None = None    # 当帧 YOLO 检测框（网页画框用）
        yolo_raw_box2: BBox | None = None
        # 帧耗时分解（2026-08-17 帧率诊断，每 60 帧打印一次分布）
        t_yolo = t_csrt = t_tpl1 = t_tpl2 = t_misc = 0.0
        t_yolo2 = t_csrt2 = t_score = 0.0  # 2026-08-18 诊断
        t_csrt_init = t_accept = t_red = 0.0
        _t0 = time.perf_counter()
        height, full_width = stitched.shape[:2]
        self.frame_width = int(full_width)
        self.frame_height = int(height)
        width = full_width // 2
        left_image = stitched[:, :width]
        right_image = stitched[:, width:]
        # 目标完全出画后，实时锁已经清掉，但 Qwen 初始身份仍保留。
        # 低频双路全局搜索；必须两路同时找到且 y 几何一致才恢复。
        reid_period = 1 if self._red_target else self._passive_reid_period
        if (self.lock_bbox is None
                and self._frame_count % reid_period == 0):
            self._try_passive_reidentify(left_image, right_image,
                                         width, height)
        # ego-motion 卡尔曼预测：车转动时目标在像面的位移大部分由车
        # 自身运动引起；把树莓派逐帧上行的角速度换算成像面速度作为
        # 控制输入，转弯/旋转重搜时预测中心才跟得上（丢失时尤其关键）。
        now_t = time.perf_counter()
        dt = (now_t - self._last_frame_t) if self._last_frame_t is not None \
            else 0.05
        self._last_frame_t = now_t
        w_rad_s = float(self._pi_motion.get("w_mrad_s") or 0.0) / 1000.0
        # angular 正 = 车左转（CCW）：静止目标在画面里向右移动（+x）。
        for view, active in ((self.lock_view, self.lock_bbox is not None),
                             (self.lock_view2,
                              self.lock2_active and self.lock_bbox2 is not None)):
            if active and view:
                kf = self._kf.get(view)
                if kf is not None:
                    kf.predict(dt, (_FOCAL_RAW_PX.get(view, 600.0)
                                    * self._focal_scale * w_rad_s, 0.0))
        # 追踪在 raw 空间；只有启用深度时才做校正（默认 --no-depth）
        left_rect: np.ndarray | None = None
        right_rect: np.ndarray | None = None
        if self.use_depth:
            left_rect, right_rect = self.stereo.rectify(left_image, right_image)

        track: dict[str, Any] = {"status": "lost", "bbox": None,
                                 "name": self.target_name, "source": ""}
        depth_m: float | None = None
        depth_conf = 0.0

        if self.lock_bbox is not None:
            view_image = left_image if self.lock_view == "left" else right_image
            kf = self._kf.get(self.lock_view)
            kf_pred = BBox(*kf.predicted_box()) if kf is not None else None
            yolo_adopted = False
            matched = None
            matched_force = False
            # ---- 命令属性快速通道 ----
            # 命令明确包含颜色时先在 KF 预测区做属性分割。命中后既比
            # YOLO 更贴物体，又省掉两路 YOLO；这不是固定红色追踪，
            # 只有当前命令提出该属性才启用。
            if self._red_target and kf is not None:
                _lock_long = max(self.lock_bbox.x2 - self.lock_bbox.x1,
                                 self.lock_bbox.y2 - self.lock_bbox.y1)
                # 随丢失帧数和车体角速度扩大局部窗；局部未命中后同帧
                # 立即做全画面轻量分割，不再先等 4 帧模板/YOLO。
                _local_radius = max(
                    220, int(_lock_long * 2.5),
                    int(220 + self.lost_frames * 70
                        + abs(w_rad_s) * _FOCAL_RAW_PX.get(
                            self.lock_view, 600.0) * 0.8))
                _t1 = time.perf_counter()
                seg_box, seg_score = self._red_recover_candidate(
                    view_image, self.lock_view, kf.predicted_center,
                    _local_radius, self.lock_bbox)
                global_recover = False
                if seg_box is None:
                    seg_box, seg_score = self._red_recover_candidate(
                        view_image, self.lock_view, kf.predicted_center,
                        None, self.lock_bbox)
                    global_recover = seg_box is not None
                t_red = time.perf_counter() - _t1
                if seg_box is not None \
                        and self._command_candidate_ok(
                            view_image, seg_box, self.lock_view):
                    adopted = self._apply_measurement(
                        self.lock_view, seg_box, force=True)
                    if adopted is not None:
                        yolo_adopted = True   # 复用标记：跳过模板/CSRT
                        yolo_raw_box = seg_box   # 画框用精确框（真实大小）
                        self.lock_bbox = adopted
                        self.lock_source = (
                            "red-recover" if global_recover else "red")
                        self.lost_frames = 0
                        self._store_template(self.lock_view, view_image,
                                             adopted)
                else:
                    # 遮挡时先记丢失；下方通用检测仍可提出候选，但必须
                    # 再经过命令属性收紧，手框不会被直接采纳。
                    self.lost_frames += 1
            # ---- 主观测：YOLO ----
            if not yolo_adopted:
                _t1 = time.perf_counter()
                detections = self._yolo_detect(view_image)
                t_yolo = time.perf_counter() - _t1
                matched, matched_force = self._match_yolo(
                    detections, view_image, self.lock_bbox,
                    self.lock_view, kf_pred)
                if matched is not None:
                    matched = self._refine_command_candidate(
                        view_image, matched, self.lock_view)
                if matched is not None:
                    adopted = self._apply_measurement(
                        self.lock_view, matched, force=matched_force)
                    if adopted is not None:
                        yolo_adopted = True
                        yolo_raw_box = matched
                        self.lock_bbox = adopted
                        self.lock_source = "yolo"
                        self.lost_frames = 0
                        self._store_template(
                            self.lock_view, view_image, adopted)
            if self._red_target and not yolo_adopted:
                # 上一帧的 red 不能继续冒充“本帧有强观测”。明确切到
                # 预测源，使外观丢失计数正常增长并最终进入被动重识别。
                self.lock_source = "kf"
            # ---- 外观确认：模板在 YOLO 连续未确认时触发（2026-08-18
            # 帧率优化：锁定后正常路径纯 YOLO ~15ms/帧；模板 80ms 只
            # 在 YOLO 连续 2 帧失手时介入，防非 COCO 目标盲区）----
            tpl_box = None
            tpl_score = 0.0
            self._yolo_miss_streak = self._yolo_miss_streak + 1                 if not yolo_adopted else 0
            if not yolo_adopted and not self._red_target and self._yolo_miss_streak >= 2:
                _t1 = time.perf_counter()
                tpl_box, tpl_score = self._match_template(view_image,
                                                          self.lock_bbox,
                                                          self.lock_view)
                t_tpl1 = time.perf_counter() - _t1
                _t_accept0 = time.perf_counter()
                if tpl_box is not None:
                    # 框尺寸用 KF 当前尺寸：模板匹配框 = 模板缩放尺寸
                    # （含 0.55 外扩 padding + 离散 5 档尺度），永远不
                    # 等于目标真实大小——2026-08-18 用户反复反馈"框选
                    # 要体现物品大小"。模板只提供位置，尺寸跟 KF。
                    if kf is not None:
                        _kb = kf.predicted_box()   # (x1, y1, x2, y2)
                        tpl_box = BBox(tpl_box.x1, tpl_box.y1,
                                       tpl_box.x1 + (_kb[2] - _kb[0]),
                                       tpl_box.y1 + (_kb[3] - _kb[1]))
                    # 近期模板可能在遮挡过程中被污染；最终仍必须通过
                    # 不可更新的 Qwen 初始锚点，不能从方块切换到手。
                    if self._score_anchor_at_bbox(
                            view_image, tpl_box, self.lock_view) < 0.38:
                        tpl_box = None
                if tpl_box is not None and self._cross_view_y_ok(
                        tpl_box, self.lock_bbox2 if self.lock2_active else None) \
                        and (not self._tracking_phase()
                             or self._plausible_region_ok(tpl_box, self.lock_view,
                                                          width)):
                    # 模板匹配有粗精两段空间约束 + 双模态打分把关，无条件
                    # 采纳（force）：CSRT 漂移时卡尔曼跟着 CSRT 走，若再对
                    # 模板门控，真实位置的纠偏会被拒（门太严锁死错误路径）。
                    adopted = self._apply_measurement(
                        self.lock_view, tpl_box, force=True)
                    if adopted is not None:
                        self.lock_bbox = adopted
                        self.lock_source = "tpl"
                        self.lost_frames = 0
                        # 近距目标快速变大：模板按当前匹配尺度刷新，
                        # 旧小模板失配是漂移纠偏失效的根源之一
                        self._store_template(self.lock_view, view_image, adopted)
                t_accept += time.perf_counter() - _t_accept0
            # ---- CSRT 兜底：YOLO/模板都没确认才跑（独立观测防漂移）。
            # 兜底前 ensure 初始化一次；正常锁定期间不重置（每帧
            # initialize 实测 104ms——2026-08-18 帧率重构）----
            if not yolo_adopted and tpl_box is None and not self._red_target:
                if not self._csrt_ready:
                    _t1 = time.perf_counter()
                    self.csrt.initialize(view_image, self.lock_bbox)
                    t_csrt_init += time.perf_counter() - _t1
                    self._csrt_ready = True
                _t1 = time.perf_counter()
                ok, csrt_bbox = self.csrt.update(view_image)
                t_csrt = time.perf_counter() - _t1
                # 近距大目标（>15% 画面）：尺度变化剧烈，CSRT 框抖动
                # 绝对值大，靠 KF 尺寸自适应门控宽放（不再 force——
                # 旧逻辑 force 让 CSRT 漂移与 KF 互相喂成正反馈，
                # 实测 1s 漂 475px）。大框测量还需模板在框附近印证：
                # 两个独立观测同时通过才采纳，CSRT 自证漂移的路被堵死。
                near_large = ((self.lock_bbox.x2 - self.lock_bbox.x1)
                              > width * 0.15
                              or (self.lock_bbox.y2 - self.lock_bbox.y1)
                              > height * 0.15)
                if ok and csrt_bbox is not None and \
                        self._sane(csrt_bbox, width, height):
                    ok_kf = True
                    if kf is not None:
                        snap = kf.snapshot()
                        ok_kf, _dist = kf.update(csrt_bbox, force=False)
                        anchor_score = self._score_anchor_at_bbox(
                            view_image, csrt_bbox, self.lock_view)
                        threshold = 0.48 if near_large else 0.38
                        if ok_kf and anchor_score < threshold:
                            # 印证失败：回滚位置/速度（漂移测量不许
                            # 污染预测，否则下一帧门控失效），但尺寸
                            # 保留——目标在逼近中快速变大，框尺寸要
                            # 跟本帧走（"框大小不跟图片"的根源之一）。
                            kf.restore(snap)
                            kf.update_size(csrt_bbox)
                            ok_kf = False
                    if ok_kf:
                        self.lock_bbox = csrt_bbox
                        self.lock_source = "csrt"
                        self.lost_frames = 0
                    else:
                        self.lock_bbox = BBox(*kf.predicted_box())
                        self.lock_source = "kf"
                        self.lost_frames += 1
                else:
                    self.lost_frames += 1
            # 视觉追回：连续失败过半阈值 → 全画面搜索模板 + YOLO 提案
            # 外观验证，目标还在画面里就能原地找回，不必等树莓派旋转重搜。
            if self.lost_frames >= 4 and not self._red_target:
                recovered, rec_score = self._recover_template(
                    view_image, self.lock_view)
                rec_source = "tpl-recover"
                if recovered is None:
                    recovered, rec_score = self._yolo_proposal_recover(
                        view_image, self.lock_view)
                    rec_source = "yolo-recover"
                if recovered is not None:
                    recovered = self._refine_command_candidate(
                        view_image, recovered, self.lock_view)
                if recovered is not None and self._cross_view_y_ok(
                        recovered,
                        self.lock_bbox2 if self.lock2_active else None) \
                        and self._score_anchor_at_bbox(
                            view_image, recovered, self.lock_view) >= 0.45 \
                        and (not self._tracking_phase()
                             or self._plausible_region_ok(
                                 recovered, self.lock_view, width)):
                    adopted = self._apply_measurement(
                        self.lock_view, recovered, force=rec_score >= 0.8)
                    if adopted is not None:
                        self.lock_bbox = adopted
                        _t1 = time.perf_counter()
                        self.csrt.initialize(view_image, adopted)
                        t_csrt_init += time.perf_counter() - _t1
                        self.lock_source = rec_source
                        self.lost_frames = 0
                        self._store_template(self.lock_view, view_image,
                                             adopted)
            # ---- 平滑漂移看门狗（2026-08-17 近距 475px/s 漂移根治）----
            # 单帧跳变保护（PI 侧 120px）能被 19px/帧 的平滑漂移骗过：
            # 门控只拦"跳变"，拦不住"每帧小幅漂"。对输出中心的横向
            # 速度做 EMA（扣除 ego-motion 期望），持续高速且外观未
            # 确认 → 判漂移：输出回到 KF 预测框并快速计入丢失，触发
            # 模板/YOLO 外观追回重新锚定。
            # 2026-08-17 实测调灵敏：0.7/0.3 EMA 爬升太慢，与分叉
            # 检测（10 帧）竞态时总是分叉先丢第二路；改 0.5 EMA +
            # 5 帧计数 + 首帧直接取 residual。
            motion_confirmed = self._tracking_source_confirmed(
                self.lock_view, self.lock_source, self.lock_bbox, 0.0)
            prev_c = self._last_centers.get(self.lock_view)
            if kf is not None and prev_c is not None:
                out_cx = (self.lock_bbox.x1 + self.lock_bbox.x2) / 2.0
                dtd = max(0.01, min(0.2, dt))
                residual = (out_cx - prev_c[0]) / dtd \
                    - _FOCAL_RAW_PX.get(self.lock_view, 600.0) \
                    * self._focal_scale * w_rad_s
                ema = 0.5 * self._drift.get(self.lock_view, residual) \
                    + 0.5 * residual
                self._drift[self.lock_view] = ema
                # 红色分割也是强观测；双目 CSRT 同向移动/几何一致时，
                # 更可能是真实移动目标而非单路漂移。只有高速且没有任何
                # 这些证据时才累计漂移，避免人手快速移动被误杀。
                motion_confirmed = self._tracking_source_confirmed(
                    self.lock_view, self.lock_source, self.lock_bbox, ema)
                if abs(ema) > 220.0 and not motion_confirmed:
                    self._drift_frames[self.lock_view] = \
                        self._drift_frames.get(self.lock_view, 0) + 1
                else:
                    self._drift_frames[self.lock_view] = 0
                if self._drift_frames.get(self.lock_view, 0) >= 5:
                    self._drift_frames[self.lock_view] = 0
                    self.lock_bbox = BBox(*kf.predicted_box())
                    self.lock_source = "kf"
                    self.lost_frames += 3
                    print(f"[pursuit] 漂移看门狗触发 view={self.lock_view} "
                          f"residual={residual:.0f}px/s")
            self._last_centers[self.lock_view] = (
                (self.lock_bbox.x1 + self.lock_bbox.x2) / 2.0,
                (self.lock_bbox.y1 + self.lock_bbox.y2) / 2.0)
            # 外观确认计数（2026-08-17 防"假装锁定"）：目标真丢后
            # CSRT 会漂到错误位置继续报 tracking。模板匹配 ≥0.6 才
            # 有输出，连续 N 帧模板/YOLO/追回都未确认 = 锁定内容已
            # 不是目标外观 → 宣告丢失，树莓派走旋转重搜 + Qwen 复检。
            big_near = ((self.lock_bbox.x2 - self.lock_bbox.x1) > width * 0.6
                        or (self.lock_bbox.y2 - self.lock_bbox.y1) > height * 0.6)
            if yolo_adopted or tpl_box is not None or motion_confirmed or \
                    (big_near and self.lock_source in ("yolo", "tpl")):
                self._tpl_miss_frames = 0
            else:
                self._tpl_miss_frames += 1
            if self._tpl_miss_frames >= self._miss_limit():
                print("[pursuit] 外观确认连续缺失，判定丢失并清锁")
                self._clear_lock(preserve_identity=True)

        # 第二路独立追踪（示教停车需要两路目标中心）
        track2: dict[str, Any] = {"status": "lost", "bbox": None,
                                  "name": self.target_name, "source": ""}
        if self.lock2_active and self.lock_bbox2 is not None:
            view_image2 = (left_image if self.lock_view2 == "left"
                           else right_image)
            raw_observation2: BBox | None = None
            kf2 = self._kf.get(self.lock_view2)
            kf_pred2 = BBox(*kf2.predicted_box()) if kf2 is not None else None
            yolo_adopted2 = False
            matched2 = None
            matched_force2 = False
            # ---- 第二路命令属性快速通道 ----
            if self._red_target and kf2 is not None:
                _lock_long2 = max(self.lock_bbox2.x2 - self.lock_bbox2.x1,
                                  self.lock_bbox2.y2 - self.lock_bbox2.y1)
                _local_radius2 = max(
                    220, int(_lock_long2 * 2.5),
                    int(220 + self.lost_frames2 * 70
                        + abs(w_rad_s) * _FOCAL_RAW_PX.get(
                            self.lock_view2, 600.0) * 0.8))
                _t1 = time.perf_counter()
                seg_box2, seg_score2 = self._red_recover_candidate(
                    view_image2, self.lock_view2, kf2.predicted_center,
                    _local_radius2, self.lock_bbox2)
                global_recover2 = False
                if seg_box2 is None:
                    seg_box2, seg_score2 = self._red_recover_candidate(
                        view_image2, self.lock_view2, kf2.predicted_center,
                        None, self.lock_bbox2)
                    global_recover2 = seg_box2 is not None
                t_red += time.perf_counter() - _t1
                if seg_box2 is not None \
                        and self._command_candidate_ok(
                            view_image2, seg_box2, self.lock_view2):
                    adopted2 = self._apply_measurement(
                        self.lock_view2, seg_box2, force=True)
                    if adopted2 is not None:
                        yolo_adopted2 = True
                        yolo_raw_box2 = seg_box2
                        self.lock_bbox2 = adopted2
                        self.lock_source2 = (
                            "red-recover" if global_recover2 else "red")
                        self.lost_frames2 = 0
                        self._store_template(self.lock_view2, view_image2,
                                             adopted2)
                else:
                    self.lost_frames2 += 1
            # ---- 第二路 YOLO；属性任务只有快速通道失败才运行 ----
            if not yolo_adopted2:
                _t1 = time.perf_counter()
                detections2 = self._yolo_detect(view_image2)
                t_yolo2 = time.perf_counter() - _t1
                matched2, matched_force2 = self._match_yolo(
                    detections2, view_image2, self.lock_bbox2,
                    self.lock_view2, kf_pred2)
                if matched2 is not None:
                    matched2 = self._refine_command_candidate(
                        view_image2, matched2, self.lock_view2)
                if matched2 is not None:
                    adopted2 = self._apply_measurement(
                        self.lock_view2, matched2, force=matched_force2)
                    if adopted2 is not None:
                        yolo_adopted2 = True
                        yolo_raw_box2 = matched2
                        self.lock_bbox2 = adopted2
                        self.lock_source2 = "yolo"
                        self.lost_frames2 = 0
                        self._store_template(
                            self.lock_view2, view_image2, adopted2)
            if self._red_target and not yolo_adopted2:
                self.lock_source2 = "kf"
            tpl_box2 = None
            tpl_score2 = 0.0
            self._yolo_miss_streak2 = self._yolo_miss_streak2 + 1                 if not yolo_adopted2 else 0
            if not yolo_adopted2 and not self._red_target and self._yolo_miss_streak2 >= 2:
                _t1 = time.perf_counter()
                tpl_box2, tpl_score2 = self._match_template(view_image2,
                                                            self.lock_bbox2,
                                                            self.lock_view2)
                t_tpl2 = time.perf_counter() - _t1
                if tpl_box2 is not None:
                    # 与主路一致：模板只提供位置，框尺寸跟 KF（2026-08-18）
                    if kf2 is not None:
                        _kb2 = kf2.predicted_box()   # (x1, y1, x2, y2)
                        tpl_box2 = BBox(tpl_box2.x1, tpl_box2.y1,
                                        tpl_box2.x1 + (_kb2[2] - _kb2[0]),
                                        tpl_box2.y1 + (_kb2[3] - _kb2[1]))
                    if self._score_anchor_at_bbox(
                            view_image2, tpl_box2,
                            self.lock_view2) < 0.38:
                        tpl_box2 = None
                if tpl_box2 is not None and self._cross_view_y_ok(
                        tpl_box2,
                        self.lock_bbox if self.lock_bbox is not None else None) \
                        and (not self._tracking_phase()
                             or self._plausible_region_ok(
                                 tpl_box2, self.lock_view2, width)):
                    # 与主路一致：模板匹配无条件采纳（force），保留对
                    # CSRT 漂移的纠偏能力
                    adopted2 = self._apply_measurement(
                        self.lock_view2, tpl_box2, force=True)
                    if adopted2 is not None:
                        self.lock_bbox2 = adopted2
                        self.lock_source2 = "tpl"
                        self.lost_frames2 = 0
                        self._store_template(self.lock_view2, view_image2,
                                             adopted2)
            if not yolo_adopted2 and tpl_box2 is None and not self._red_target:
                # 兜底前 ensure 初始化一次（正常锁定期间不重置）
                if not self._csrt2_ready:
                    _t1 = time.perf_counter()
                    self.csrt2.initialize(view_image2, self.lock_bbox2)
                    t_csrt_init += time.perf_counter() - _t1
                    self._csrt2_ready = True
                _t1 = time.perf_counter()
                ok2, csrt_bbox2 = self.csrt2.update(view_image2)
                t_csrt2 = time.perf_counter() - _t1
                # 与主路一致的漂移修复：CSRT 过 KF 门控（近距大目标
                # 尺寸自适应宽放、不再 force）+ 大框模板印证才采纳，
                # 被拒输出预测框并回滚 KF。
                near_large2 = ((self.lock_bbox2.x2 - self.lock_bbox2.x1)
                               > width * 0.15
                               or (self.lock_bbox2.y2 - self.lock_bbox2.y1)
                               > height * 0.15)
                if ok2 and csrt_bbox2 is not None and \
                        self._sane(csrt_bbox2, width, height):
                    raw_observation2 = csrt_bbox2
                    ok_kf2 = True
                    if kf2 is not None:
                        snap2 = kf2.snapshot()
                        ok_kf2, _dist2 = kf2.update(csrt_bbox2, force=False)
                        anchor_score2 = self._score_anchor_at_bbox(
                            view_image2, csrt_bbox2, self.lock_view2)
                        threshold2 = 0.48 if near_large2 else 0.38
                        if ok_kf2 and anchor_score2 < threshold2:
                            kf2.restore(snap2)
                            kf2.update_size(csrt_bbox2)
                            ok_kf2 = False
                    if ok_kf2:
                        self.lock_bbox2 = csrt_bbox2
                        self.lock_source2 = "csrt"
                        self.lost_frames2 = 0
                    else:
                        self.lock_bbox2 = BBox(*kf2.predicted_box())
                        self.lock_source2 = "kf"
                        self.lost_frames2 += 1
                else:
                    self.lost_frames2 += 1
            if (self.lost_frames2 >= self._miss_limit() // 2
                    and not self._red_target):
                recovered2, rec_score2 = self._recover_template(
                    view_image2, self.lock_view2)
                rec_source2 = "tpl-recover"
                if recovered2 is None:
                    recovered2, rec_score2 = self._yolo_proposal_recover(
                        view_image2, self.lock_view2)
                    rec_source2 = "yolo-recover"
                if recovered2 is not None:
                    recovered2 = self._refine_command_candidate(
                        view_image2, recovered2, self.lock_view2)
                if recovered2 is not None and self._cross_view_y_ok(
                        recovered2,
                        self.lock_bbox if self.lock_bbox is not None else None) \
                        and self._score_anchor_at_bbox(
                            view_image2, recovered2,
                            self.lock_view2) >= 0.45 \
                        and (not self._tracking_phase()
                             or self._plausible_region_ok(
                                 recovered2, self.lock_view2, width)):
                    adopted2 = self._apply_measurement(
                        self.lock_view2, recovered2,
                        force=rec_score2 >= 0.8)
                    if adopted2 is not None:
                        self.lock_bbox2 = adopted2
                        _t1 = time.perf_counter()
                        self.csrt2.initialize(view_image2, adopted2)
                        t_csrt_init += time.perf_counter() - _t1
                        self.lock_source2 = rec_source2
                        self.lost_frames2 = 0
                        self._store_template(self.lock_view2, view_image2,
                                             adopted2)
            # 与主路一致的平滑漂移看门狗（0.5 EMA + 5 帧 + 外观确认过滤）
            prev_c2 = self._last_centers.get(self.lock_view2)
            if kf2 is not None and prev_c2 is not None:
                out_cx2 = (self.lock_bbox2.x1 + self.lock_bbox2.x2) / 2.0
                dtd = max(0.01, min(0.2, dt))
                residual2 = (out_cx2 - prev_c2[0]) / dtd \
                    - _FOCAL_RAW_PX.get(self.lock_view2, 600.0) \
                    * self._focal_scale * w_rad_s
                ema2 = 0.5 * self._drift.get(self.lock_view2, residual2) \
                    + 0.5 * residual2
                self._drift[self.lock_view2] = ema2
                motion_confirmed2 = self._tracking_source_confirmed(
                    self.lock_view2, self.lock_source2,
                    self.lock_bbox2, ema2)
                if abs(ema2) > 220.0 and not motion_confirmed2:
                    self._drift_frames[self.lock_view2] = \
                        self._drift_frames.get(self.lock_view2, 0) + 1
                else:
                    self._drift_frames[self.lock_view2] = 0
                if self._drift_frames.get(self.lock_view2, 0) >= 5:
                    self._drift_frames[self.lock_view2] = 0
                    self.lock_bbox2 = BBox(*kf2.predicted_box())
                    self.lock_source2 = "kf"
                    self.lost_frames2 += 3
                    print(f"[pursuit] 漂移看门狗触发 view={self.lock_view2} "
                          f"residual={residual2:.0f}px/s")
            self._last_centers[self.lock_view2] = (
                (self.lock_bbox2.x1 + self.lock_bbox2.x2) / 2.0,
                (self.lock_bbox2.y1 + self.lock_bbox2.y2) / 2.0)
            if self.lost_frames2 < self._miss_limit():
                offset2 = 0.0 if self.lock_view2 == "left" else float(width)
                bbox2 = self.lock_bbox2
                # 即使 KF 因大跳变拒绝了第二路测量，也用独立追踪器的
                # 原始框做双目分叉判定；否则错误第二路会永远藏在 KF
                # 预测框后面，既不更新也不被剔除。
                geometry_bbox2 = raw_observation2 or bbox2
                track2["status"] = "tracking"
                track2["source"] = self.lock_source2
                track2["bbox"] = [bbox2.x1 + offset2, bbox2.y1,
                                  bbox2.x2 + offset2, bbox2.y2]
                # 双路持续分叉 → 第二路锁错了（初始认错不会自愈），
                # 丢掉第二路，树莓派回退单路模式。
                # 两个信号：y 差过大；或横向视差绝对值几乎跨满画面。
                # 左右物理设备交换后，合法目标的 dx 符号也会改变，
                # 不能再用 dx<-200 当作第二路锁错证据。
                if self.lock_bbox is not None \
                        and self.lost_frames < self._miss_limit():
                    y1 = (self.lock_bbox.y1 + self.lock_bbox.y2) / 2.0
                    y2 = (geometry_bbox2.y1 + geometry_bbox2.y2) / 2.0
                    l_cx = (self.lock_bbox.x1 + self.lock_bbox.x2) / 2.0
                    r_cx = (geometry_bbox2.x1 + geometry_bbox2.x2) / 2.0
                    # 只在车逼近阶段检查：旋转重搜时目标可以在画面
                    # 任意位置，几何约束不适用。
                    diverged = (self._tracking_phase()
                                and self._stereo_boxes_diverged(
                                    self.lock_bbox, geometry_bbox2,
                                    width, height))
                    if diverged:
                        self._y_divergence_frames += 1
                        if self._y_divergence_frames >= 10:
                            print(f"[pursuit] 双路持续分叉 "
                                  f"(y {y1:.0f}/{y2:.0f}, "
                                  f"dx {l_cx - r_cx:.0f})，丢弃第二路锁定")
                            self.lock2_active = False
                            self.lock_bbox2 = None
                            track2["status"] = "lost"
                            track2["bbox"] = None
                            self._y_divergence_frames = 0
                    else:
                        self._y_divergence_frames = 0

        # 主路状态赋值放在 lock2 块外面：2026-08-16 修复——旧代码
        # 把这段缩进在 `if self.lock2_active` 内，第二路锁未建立/被
        # 丢弃（单路模式）时主路明明在追踪，上报的却是默认 lost，
        # 树莓派会误判目标丢失进入重搜。
        if self.lock_bbox is not None and self.lost_frames < self._miss_limit():
            # 只有最新一帧确认过才算 tracking；连续未确认（CSRT/
            # 模板都失败）进入 stale——网页不再画框，树莓派也会
            # 更快进入保持/重搜，不再对着陈旧锁转向。
            track["status"] = "tracking" if self.lost_frames == 0 \
                else "stale"
            track["source"] = self.lock_source
            # 返回整幅拼接图坐标（raw 空间，与网页画面一致）
            offset = 0.0 if self.lock_view == "left" else float(width)
            bbox = self.lock_bbox
            track["bbox"] = [bbox.x1 + offset, bbox.y1,
                             bbox.x2 + offset, bbox.y2]
            if self.use_depth and left_rect is not None:
                # bbox 贴着画面边缘（目标部分出画）时视差不可靠且
                # 多档 SGBM 回退昂贵，跳过深度，交给树莓派雷达测距。
                at_edge = (bbox.x1 <= 4.0 or bbox.x2 >= width - 4.0
                           or bbox.y1 <= 4.0 or bbox.y2 >= height - 4.0)
                if at_edge:
                    depth_m, depth_conf = None, 0.0
                else:
                    # raw 框 → 该路校正框 → 深度；右目锁定把两图
                    # 水平翻转后仍走左参考管线（翻转后匹配方向与
                    # minDisparity=0 一致），框同步翻转。
                    rect_width = int(self.stereo.image_size[0])
                    if self.lock_view == "left":
                        rect_bbox = self.stereo.bbox_raw_to_rect(
                            (bbox.x1, bbox.y1, bbox.x2, bbox.y2), "left")
                        if rect_bbox is not None:
                            depth_m, depth_conf = \
                                self.stereo.depth_in_region(
                                    left_rect, right_rect, rect_bbox)
                    else:
                        rect_bbox = self.stereo.bbox_raw_to_rect(
                            (bbox.x1, bbox.y1, bbox.x2, bbox.y2), "right")
                        if rect_bbox is not None:
                            depth_m, depth_conf = \
                                self.stereo.depth_in_region(
                                    right_rect[:, ::-1],
                                    left_rect[:, ::-1],
                                    (rect_width - rect_bbox[2],
                                     rect_bbox[1],
                                     rect_width - rect_bbox[0],
                                     rect_bbox[3]))
        else:
            track["status"] = "lost"
            track["source"] = ""

        frame_ms = (time.perf_counter() - started) * 1000.0
        if self._frame_count % 60 == 0:
            _sum = t_yolo + t_csrt + t_tpl1 + t_tpl2 + t_yolo2 + t_csrt2 + t_score                 + t_csrt_init + t_accept
            t_misc = max(0.0, frame_ms / 1000.0 - _sum)
            print(f"[perf] frame_ms={frame_ms:.0f} yolo={t_yolo*1000:.0f} "
                  f"csrt={t_csrt*1000:.0f} tpl1={t_tpl1*1000:.0f} "
                  f"tpl2={t_tpl2*1000:.0f} yolo2={t_yolo2*1000:.0f} "
                  f"csrt2={t_csrt2*1000:.0f} score={t_score*1000:.0f} "
                  f"init={t_csrt_init*1000:.0f} accept={t_accept*1000:.0f} "
                  f"misc={t_misc*1000:.0f}ms red={self._red_target} "
                  f"src={self.lock_source} tgt={self.target_name[:10]}")
        # 网页画框优先用 YOLO 检测框（真实目标大小）；yolo_raw_box 是
        # 该路局部 raw 坐标，转拼接坐标（2026-08-18 用户要求框体现物品大小）
        yolo_box_out = None
        if yolo_raw_box is not None and self.lock_view:
            _off = 0.0 if self.lock_view == "left" else float(width)
            yolo_box_out = [yolo_raw_box.x1 + _off, yolo_raw_box.y1,
                            yolo_raw_box.x2 + _off, yolo_raw_box.y2]
        yolo_box2_out = None
        if yolo_raw_box2 is not None and self.lock_view2:
            _off2 = 0.0 if self.lock_view2 == "left" else float(width)
            yolo_box2_out = [yolo_raw_box2.x1 + _off2, yolo_raw_box2.y1,
                             yolo_raw_box2.x2 + _off2, yolo_raw_box2.y2]
        reidentified = self._reid_notice_frames > 0
        if self._reid_notice_frames > 0:
            self._reid_notice_frames -= 1
        return {
            "type": "frame_result",
            "frame_id": self._frame_count,
            "track": track,
            "track2": track2,
            "yolo_box": yolo_box_out,
            "yolo_box2": yolo_box2_out,
            "depth_m": None if depth_m is None else round(float(depth_m), 3),
            "depth_conf": round(float(depth_conf), 3),
            "frame_ms": round(frame_ms, 1),
            "reidentified": reidentified,
        }

    def _yolo_detect(self, image: np.ndarray,
                     conf: float = 0.25) -> list[tuple[BBox, str, float]]:
        """返回 [(bbox, coco类名, 置信度)]。conf 低阈值用于提案追回。"""
        if self.yolo is None:
            from ultralytics import YOLO
            print(f"[pursuit] 加载 YOLO 模型 {self.yolo_model} …")
            # 2026-08-17 帧率修复：旧代码未指定 device，YOLO 在 CPU 上
            # 跑（每帧 100-300ms），服务端整体 ~7fps——网页延迟严重、
            # 追踪帧率低的直接根源。跟随 --device（cuda/cpu）。
            # 注意：本版 ultralytics 的 YOLO 构造不支持 device 参数，
            # 用 .to() 移动模型。
            self.yolo = YOLO(self.yolo_model)
            if self.device and self.device != "cpu":
                self.yolo.to(self.device)
        # 推理时显式传 device（对齐 ~/3090/src/qwen_grounded_tracker/
        # perception/yolo_obstacles.py 的用法：predict kwargs 带 device，
        # 避免 ultralytics 每次探测设备）
        results = self.yolo(image, conf=conf, iou=0.45, verbose=False,
                            device=self.device)
        detections: list[tuple[BBox, str, float]] = []
        for result in results:
            names = result.names or {}
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                conf = float(box.conf[0])
                class_name = str(names.get(int(box.cls[0]), ""))
                detections.append((BBox(x1, y1, x2, y2), class_name, conf))
        return detections

    @staticmethod
    def _strong_tracking_source(source: str) -> bool:
        """由检测、颜色或模板直接验证过的观测源。"""
        return source in {
            "qwen", "yolo", "red", "tpl", "tpl-recover",
            "red-recover", "yolo-recover", "reid",
        }

    def _tracking_source_confirmed(self, view: str, source: str,
                                   bbox: BBox | None, ema: float) -> bool:
        """判断当前运动是否有足够证据，不把真实移动误当成跟踪漂移。

        强观测直接可信。CSRT 只在另一目也有非 KF 实时框且两路 y 一致
        时获得宽限；高速运动还要求两路横向速度方向一致、幅值同量级。
        """
        if self._strong_tracking_source(source):
            return True
        if source != "csrt" or bbox is None or not self.lock2_active:
            return False
        if view == self.lock_view:
            other_view, other_bbox, other_source = (
                self.lock_view2, self.lock_bbox2, self.lock_source2)
        else:
            other_view, other_bbox, other_source = (
                self.lock_view, self.lock_bbox, self.lock_source)
        if not other_view or other_bbox is None or other_source in ("", "kf"):
            return False
        if not self._cross_view_y_ok(bbox, other_bbox):
            return False
        # 两路独立 CSRT 同时仍锁在相同极线上，本身就是比单路速度更强
        # 的证据；真实目标快速横移时两路线程更新时间有先后，不能要求
        # 当帧 EMA 已严格同速，否则先处理的一路仍会被误杀。
        if other_source == "csrt":
            return True
        if abs(ema) <= 220.0:
            return True
        other_ema = self._drift.get(other_view)
        if other_ema is None or ema * other_ema <= 0.0:
            return False
        large = max(abs(ema), abs(other_ema))
        small = min(abs(ema), abs(other_ema))
        return large > 0.0 and small / large >= 0.30

    def _try_passive_reidentify(self, left_image: np.ndarray,
                                right_image: np.ndarray,
                                width: int, height: int) -> bool:
        """锁已清除后用保留身份做双目被动重识别。"""
        if (not self.lock_view or not self.lock_view2
                or self.lock_view == self.lock_view2
                or self.lock_view not in self._anchor_templates
                or self.lock_view2 not in self._anchor_templates):
            return False
        images = {"left": left_image, "right": right_image}
        if self._red_target:
            # 颜色指令使用全画面轻量候选的“双目配对”而不是两路各自
            # 贪心取第一名。小车转动后旧预测已不可靠，但同一目标在两
            # 路 y 仍应一致；配对可快速找回，也不会因另一目旧框拒绝。
            ranked1 = self._rank_red_candidates(
                images[self.lock_view], self.lock_view)
            ranked2 = self._rank_red_candidates(
                images[self.lock_view2], self.lock_view2)
            best_pair: tuple[BBox, BBox, float, float] | None = None
            best_pair_score = -1.0
            for candidate1, candidate_score1 in ranked1[:8]:
                for candidate2, candidate_score2 in ranked2[:8]:
                    if not self._cross_view_y_ok(candidate1, candidate2):
                        continue
                    h1 = max(1.0, candidate1.y2 - candidate1.y1)
                    h2 = max(1.0, candidate2.y2 - candidate2.y1)
                    size_consistency = math.exp(-abs(math.log(h1 / h2)))
                    pair_score = (candidate_score1 + candidate_score2
                                  + 0.15 * size_consistency)
                    if pair_score > best_pair_score:
                        best_pair_score = pair_score
                        best_pair = (candidate1, candidate2,
                                     candidate_score1, candidate_score2)
            if best_pair is None:
                return False
            box1, box2, score1, score2 = best_pair
        else:
            box1, score1 = self._recover_template(
                images[self.lock_view], self.lock_view)
            box2, score2 = self._recover_template(
                images[self.lock_view2], self.lock_view2)
            if box1 is not None:
                box1 = self._refine_command_candidate(
                    images[self.lock_view], box1, self.lock_view)
            if box2 is not None:
                box2 = self._refine_command_candidate(
                    images[self.lock_view2], box2, self.lock_view2)
        # 全画面单路模板容易碰到相似纹理；双路均超过较高阈值并满足
        # 双目 y 几何后才恢复，宁可晚几帧也不追错物体。
        score_threshold = 0.28 if self._red_target else 0.64
        anchor_threshold = 0.28 if self._red_target else 0.45
        anchor1 = (self._score_anchor_at_bbox(
            images[self.lock_view], box1, self.lock_view)
                   if box1 is not None else 0.0)
        anchor2 = (self._score_anchor_at_bbox(
            images[self.lock_view2], box2, self.lock_view2)
                   if box2 is not None else 0.0)
        anchors_ok = (min(anchor1, anchor2) >= 0.28
                      or (self._red_target and self._square_target
                          and min(anchor1, anchor2) >= 0.18
                          and max(anchor1, anchor2) >= 0.28))
        if (box1 is None or box2 is None
                or score1 < score_threshold or score2 < score_threshold
                or not self._sane(box1, width, height)
                or not self._sane(box2, width, height)
                or not (anchors_ok if self._red_target
                        else min(anchor1, anchor2) >= anchor_threshold)
                or not self._cross_view_y_ok(box1, box2)):
            return False
        self.lock_bbox = box1
        self.lock_source = "reid"
        self.lost_frames = 0
        self.lock2_active = True
        self.lock_bbox2 = box2
        self.lock_source2 = "reid"
        self.lost_frames2 = 0
        self._tpl_miss_frames = 0
        self._yolo_miss_streak = 0
        self._yolo_miss_streak2 = 0
        self._kf = {
            self.lock_view: KalmanBoxTracker(box1),
            self.lock_view2: KalmanBoxTracker(box2),
        }
        self.csrt.initialize(images[self.lock_view], box1)
        self.csrt2.initialize(images[self.lock_view2], box2)
        self._csrt_ready = True
        self._csrt2_ready = True
        # 给树莓派足够的连续帧窗口退出 SEARCHING/GIVE_UP；实际框源在
        # 本帧稍后可能变成 tpl/csrt/yolo，不能只依赖 source="reid"。
        self._reid_notice_frames = 12
        self._last_centers = {
            self.lock_view: ((box1.x1 + box1.x2) / 2.0,
                             (box1.y1 + box1.y2) / 2.0),
            self.lock_view2: ((box2.x1 + box2.x2) / 2.0,
                              (box2.y1 + box2.y2) / 2.0),
        }
        print(f"[pursuit] 双目被动重识别恢复目标 "
              f"score={score1:.2f}/{score2:.2f}")
        return True

    def _extract_template(self, image: np.ndarray,
                          bbox: BBox) -> tuple[np.ndarray, np.ndarray]:
        """从锁定框提取外观模板（外扩、保持原始像素尺度）+ HSV 直方图。

        尺度严格一致才能让相关系数匹配细纹理目标：裁块不放大，
        只有超过 _tpl_max_size 才等比降采样（防大框匹配开销爆炸）。
        """
        height, width = image.shape[:2]
        cx = (bbox.x1 + bbox.x2) / 2.0
        cy = (bbox.y1 + bbox.y2) / 2.0
        # 少量背景 padding 提升匹配鲁棒性；太大会让框远超目标大小
        half_w = max(4.0, (bbox.x2 - bbox.x1) * 0.55)
        half_h = max(4.0, (bbox.y2 - bbox.y1) * 0.55)
        x1 = int(max(0, min(width - 1, cx - half_w)))
        x2 = int(max(x1 + 1, min(width, cx + half_w)))
        y1 = int(max(0, min(height - 1, cy - half_h)))
        y2 = int(max(y1 + 1, min(height, cy + half_h)))
        patch = image[y1:y2, x1:x2]
        if max(patch.shape[0], patch.shape[1]) > self._tpl_max_size:
            ratio = self._tpl_max_size / max(patch.shape[0], patch.shape[1])
            patch = cv2.resize(
                patch, (max(1, int(patch.shape[1] * ratio)),
                        max(1, int(patch.shape[0] * ratio))),
                interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16],
                            [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
        return gray, hist

    def _store_template(self, view: str, image: np.ndarray,
                        bbox: BBox, anchor: bool = False) -> None:
        """保存外观快照（3 张近期模板 + 1 张不可被污染的初始锚点）。"""
        tpl, hist = self._extract_template(image, bbox)
        if anchor or view not in self._anchor_templates:
            self._anchor_templates[view] = tpl.copy()
            self._anchor_histograms[view] = hist.copy()
        self._templates.setdefault(view, []).insert(0, tpl)
        self._histograms.setdefault(view, []).insert(0, hist)
        del self._templates[view][3:]
        del self._histograms[view][3:]

    def _appearance_bank(
            self, view: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """近期外观后追加初始锚点；相同对象引用不重复计算。"""
        templates = list(self._templates.get(view) or [])
        histograms = list(self._histograms.get(view) or [])
        anchor = self._anchor_templates.get(view)
        if anchor is not None:
            templates.append(anchor)
            histograms.append(self._anchor_histograms[view])
        return templates, histograms

    @staticmethod
    def _patch_hist_sim(image: np.ndarray, hist: np.ndarray,
                        x1: int, y1: int, size: int) -> float:
        """匹配位置小块的 HSV 直方图与模板直方图的相似度（0~1）。"""
        patch = image[y1:y1 + size, x1:x1 + size]
        patch = cv2.resize(patch, (size, size),
                           interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        cand = cv2.calcHist([hsv], [0, 1], None, [16, 16],
                            [0, 180, 0, 256])
        cv2.normalize(cand, cand, 0, 1.0, cv2.NORM_MINMAX)
        return float(cv2.compareHist(hist, cand, cv2.HISTCMP_CORREL))

    def _adaptive_scales(self, template: np.ndarray,
                         box: BBox) -> tuple[float, ...]:
        """近距尺度自适应：模板按锁框当前尺寸匹配。

        旧固定尺度 0.5~2.0× 只覆盖模板 2 倍以内，近距目标比模板大
        2 倍以上时（目标 300~600px、模板 ≤256px）匹配尺度永远对不
        上，相关系数在目标内部拉平——模板纠偏在近距静默失效（漂移
        失控的根源之一）。框尺寸与模板尺寸接近时不追加（避免无谓
        开销）。钳到 [0.25, 4] 防极端值。
        """
        t_long = max(template.shape[:2])
        box_long = max(box.x2 - box.x1, box.y2 - box.y1)
        base = box_long / max(t_long, 1.0)
        if 0.6 <= base <= 1.4:
            return ()
        base = max(0.25, min(4.0, base))
        return (base * 0.8, base, base * 1.2)

    def _match_template(self, image: np.ndarray, lock_bbox: BBox,
                        view: str) -> tuple[BBox | None, float]:
        """任意目标通用的每帧重检：多模板 × 多尺度 × 直方图双模态。

        目标不是 COCO 类时 YOLO 匹配不上、CSRT 漂移丢锁；外观模型
        每帧把锁拉回真正目标。粗搜半分辨率大范围（等效 ±200px，含
        运动预测），命中后精搜 ±24px、5 档尺度；分数 = 0.5×像素相关
        + 0.5×HSV 直方图相似度（抗光照/尺度变化）。≥0.6 才采纳。
        返回 (bbox, 分数)。
        """
        templates, hists = self._appearance_bank(view)
        if not templates:
            return None, 0.0
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cx = (lock_bbox.x1 + lock_bbox.x2) / 2.0
        cy = (lock_bbox.y1 + lock_bbox.y2) / 2.0
        # 搜索中心：卡尔曼预测（含 ego-motion 控制输入）优先——车
        # 转动时目标在像面的位移大部分由车自身运动引起，预测比
        # 上帧位移外推准得多；无滤波器时退回旧的外推。
        kf = self._kf.get(view)
        if kf is not None:
            pcx, pcy = kf.predicted_center
            cx = max(0.0, min(float(width - 1), pcx))
            cy = max(0.0, min(float(height - 1), pcy))
        else:
            last = self._last_centers.get(view)
            if last is not None:
                pred_cx = cx + (cx - last[0])
                pred_cy = cy + (cy - last[1])
                cx = max(0.0, min(float(width - 1), pred_cx))
                cy = max(0.0, min(float(height - 1), pred_cy))
        # 粗搜：半分辨率，半径 100（等效全分辨率 200px），多模板取最好。
        # 小目标（<64px）半径与阈值都放宽/缩小：200px 窗口里 15px
        # 模板的噪声局部峰太多，误匹配会把小目标拉走（实测红色方块
        # 30-70px 模板分数波动大、频繁跌出 0.6）。
        lock_long = max(lock_bbox.x2 - lock_bbox.x1,
                        lock_bbox.y2 - lock_bbox.y1)
        small_target = lock_long < 64.0
        small_img = cv2.resize(gray, None, fx=0.5, fy=0.5,
                               interpolation=cv2.INTER_AREA)
        hx, hy = cx * 0.5, cy * 0.5
        radius = max(60, int(lock_long * 2)) if small_target else 100
        coarse_val = 0.50 if small_target else 0.55
        coarse_cx = hx
        coarse_cy = hy
        for template in templates:
            small_tpl = cv2.resize(template, None, fx=0.5, fy=0.5,
                                   interpolation=cv2.INTER_AREA)
            th, tw = small_tpl.shape[:2]
            if th + 4 >= small_img.shape[0] or tw + 4 >= small_img.shape[1]:
                continue
            x1 = int(max(0, min(small_img.shape[1] - tw - 1,
                                hx - tw / 2 - radius)))
            x2 = int(max(x1 + tw, min(small_img.shape[1],
                                      hx + tw / 2 + radius)))
            y1 = int(max(0, min(small_img.shape[0] - th - 1,
                                hy - th / 2 - radius)))
            y2 = int(max(y1 + th, min(small_img.shape[0],
                                      hy + th / 2 + radius)))
            search = small_img[y1:y2, x1:x2]
            if search.shape[0] < th or search.shape[1] < tw:
                continue
            result = cv2.matchTemplate(search, small_tpl,
                                       cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > coarse_val:
                coarse_val = max_val
                coarse_cx = (x1 + max_loc[0] + tw / 2.0) * 2.0
                coarse_cy = (y1 + max_loc[1] + th / 2.0) * 2.0
        if coarse_val < (0.50 if small_target else 0.55):
            return None, 0.0
        # 精搜：全分辨率，粗位置 ±24px，多模板 × (固定 5 尺度 +
        # 近距自适应尺度)，双模态打分
        best_score = 0.55 if small_target else 0.6
        best_box: BBox | None = None
        for idx, template in enumerate(templates):
            hist = hists[idx] if idx < len(hists) else None
            for scale in (0.5, 0.7, 1.0, 1.4, 2.0) \
                    + self._adaptive_scales(template, lock_bbox):
                tpl = (cv2.resize(template, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_AREA)
                       if scale != 1.0 else template)
                t_h, t_w = tpl.shape[:2]
                if t_h + 4 >= height or t_w + 4 >= width:
                    continue
                fx1 = int(max(0, min(width - t_w - 1,
                                     coarse_cx - t_w / 2 - 24)))
                fx2 = int(max(fx1 + t_w, min(width,
                                             coarse_cx + t_w / 2 + 24)))
                fy1 = int(max(0, min(height - t_h - 1,
                                     coarse_cy - t_h / 2 - 24)))
                fy2 = int(max(fy1 + t_h, min(height,
                                             coarse_cy + t_h / 2 + 24)))
                fine = gray[fy1:fy2, fx1:fx2]
                if fine.shape[0] < t_h or fine.shape[1] < t_w:
                    continue
                fine_res = cv2.matchTemplate(fine, tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(fine_res)
                if max_val < (0.45 if small_target else 0.5):
                    continue
                score = max_val
                if hist is not None:
                    hist_sim = self._patch_hist_sim(
                        image, hist, fx1 + max_loc[0], fy1 + max_loc[1],
                        max(t_h, t_w))
                    score = 0.5 * max_val + 0.5 * hist_sim
                if score > best_score:
                    best_score = score
                    best_box = BBox(
                        float(fx1 + max_loc[0]), float(fy1 + max_loc[1]),
                        float(fx1 + max_loc[0] + t_w),
                        float(fy1 + max_loc[1] + t_h))
        return best_box, best_score if best_box is not None else 0.0

    def _miss_limit(self) -> int:
        """外观确认/丢失容忍帧数按目标尺寸缩放。

        实测（2026-08-17 红色方块，30-70px 小目标）：固定 15 帧
        （~2s）清锁对小目标太激进——小目标模板分数受噪声/运动模糊
        影响波动大，两段 TRACKING 都以"外观确认连续缺失 15 帧"清锁
        告终，车永远追不到近距。小目标放宽到 25 帧（~3.5s），
        给模板/追回更多机会。
        """
        long_side = 0.0
        if self.lock_bbox is not None:
            long_side = max(self.lock_bbox.x2 - self.lock_bbox.x1,
                            self.lock_bbox.y2 - self.lock_bbox.y1)
        if long_side < 64.0:
            return 25
        if long_side < 256.0:
            return 20
        return 15

    def _recover_template(self, image: np.ndarray,
                          view: str) -> tuple[BBox | None, float]:
        """丢锁后的全画面追回：四分之一分辨率整幅搜索（多模板）。

        目标可能还在画面里，只是 CSRT/局部匹配都断了；整幅粗搜
        一次找回（相关系数 ≥0.68，小目标放宽到 0.60——30px 模板
        在 0.25× 分辨率下噪声大，固定 0.68 实测追不回小目标），
        比树莓派旋转重搜+Qwen 快得多。返回 (bbox, 分数)。
        """
        templates, _hists = self._appearance_bank(view)
        if not templates:
            return None, 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=0.25, fy=0.25,
                           interpolation=cv2.INTER_AREA)
        best_val = 0.60 if self._miss_limit() >= 25 else 0.68
        best_box: BBox | None = None
        # 近距大目标：模板按锁框当前尺寸先缩放到匹配尺度再降采样，
        # 否则整幅 0.25× 搜索里模板（≤256px→64px）远小于画面中的
        # 大目标，相关系数拉平、追回失效。
        lock_long = 0.0
        if self.lock_bbox is not None:
            lock_long = max(self.lock_bbox.x2 - self.lock_bbox.x1,
                            self.lock_bbox.y2 - self.lock_bbox.y1)
        for template in templates:
            base = 1.0
            if lock_long > 0.0:
                base = max(0.25, min(4.0,
                                     lock_long / max(max(template.shape[:2]),
                                                     1.0)))
            # 已有实时锁时围绕当前尺度搜；完全丢锁后不知道重新入画
            # 的远近，补充宽尺度搜索，允许手/目标以不同大小再次出现。
            scales = ((base * 0.8, base, base * 1.2)
                      if lock_long > 0.0
                      else (0.5, 0.7, 1.0, 1.4, 2.0))
            for scale in scales:
                # OpenCV 5 对极小 fx/fy 会四舍五入为 0×0 并抛异常；
                # 显式计算且钳位尺寸，丢锁恢复不能因此打断整个 WS。
                resized_w = max(1, int(round(
                    template.shape[1] * 0.25 * scale)))
                resized_h = max(1, int(round(
                    template.shape[0] * 0.25 * scale)))
                tpl = cv2.resize(template, (resized_w, resized_h),
                                 interpolation=cv2.INTER_AREA)
                th, tw = tpl.shape[:2]
                if (th < 3 or tw < 3
                        or th + 4 >= small.shape[0]
                        or tw + 4 >= small.shape[1]):
                    continue
                result = cv2.matchTemplate(small, tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                # 全画面搜索容易认错相似纹理，阈值高于局部匹配
                if max_val > best_val:
                    best_val = max_val
                    best_box = BBox(max_loc[0] * 4.0,
                                    max_loc[1] * 4.0,
                                    (max_loc[0] + tw) * 4.0,
                                    (max_loc[1] + th) * 4.0)
        return best_box, best_val if best_box is not None else 0.0

    def _apply_measurement(self, view: str, measured: BBox,
                           force: bool = False) -> BBox | None:
        """卡尔曼门控 + EMA 平滑。返回采纳后的框（供 CSRT 初始化与
        输出）；门控拒绝返回 None（调用方按"未确认"处理）。

        EMA 0.75 测量 + 0.25 预测：消掉检测抖动（Roboflow Detections
        Stabilizer 同款思路），又不至于滞后于真实运动。
        """
        kf = self._kf.get(view)
        if kf is None:
            self._kf[view] = KalmanBoxTracker(measured)
            return measured
        pred = kf.predicted_box()
        ok, _dist = kf.update(measured, force=force)
        if not ok:
            return None
        return BBox(0.75 * measured.x1 + 0.25 * pred[0],
                    0.75 * measured.y1 + 0.25 * pred[1],
                    0.75 * measured.x2 + 0.25 * pred[2],
                    0.75 * measured.y2 + 0.25 * pred[3])

    def _score_candidate(self, image: np.ndarray, bbox: BBox,
                         view: str) -> float:
        """外观模型对候选框打分：各模板在候选中心邻域的相关系数
        最大值 × 3 尺度，加 HSV 直方图相似度（0.5+0.5）。"""
        templates, hists = self._appearance_bank(view)
        if not templates:
            return 0.0
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cx = (bbox.x1 + bbox.x2) / 2.0
        cy = (bbox.y1 + bbox.y2) / 2.0
        best = 0.0
        for idx, template in enumerate(templates):
            hist = hists[idx] if idx < len(hists) else None
            for scale in (0.7, 1.0, 1.4) \
                    + self._adaptive_scales(template, bbox):
                tpl = (cv2.resize(template, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_AREA)
                       if scale != 1.0 else template)
                t_h, t_w = tpl.shape[:2]
                if t_h + 4 >= height or t_w + 4 >= width:
                    continue
                half_win = max(t_w, t_h) // 2
                x1 = int(max(0, min(width - t_w - 1,
                                    cx - t_w / 2 - half_win)))
                x2 = int(max(x1 + t_w, min(width, cx + t_w / 2 + half_win)))
                y1 = int(max(0, min(height - t_h - 1,
                                    cy - t_h / 2 - half_win)))
                y2 = int(max(y1 + t_h, min(height, cy + t_h / 2 + half_win)))
                search = gray[y1:y2, x1:x2]
                if search.shape[0] < t_h or search.shape[1] < t_w:
                    continue
                res = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                score = float(max_val)
                if hist is not None:
                    score = 0.5 * score + 0.5 * self._patch_hist_sim(
                        image, hist, x1 + max_loc[0], y1 + max_loc[1],
                        max(t_h, t_w))
                best = max(best, score)
        return best

    def _score_anchor_at_bbox(self, image: np.ndarray, bbox: BBox,
                              view: str) -> float:
        """候选框本身与 Qwen 初始身份锚点的相似度。

        与 `_score_candidate` 的“候选附近搜索目标”不同，本方法只评估
        框内内容。这样手和方块重叠时，即使方块仍在手框附近，也不会
        因邻域里找得到方块而把整只手的框采纳为新目标。
        """
        anchor = self._anchor_templates.get(view)
        anchor_hist = self._anchor_histograms.get(view)
        if anchor is None or anchor_hist is None or bbox.area <= 0:
            return 0.0
        candidate, candidate_hist = self._extract_template(image, bbox)
        if candidate.size == 0:
            return 0.0
        resized = cv2.resize(candidate,
                             (anchor.shape[1], anchor.shape[0]),
                             interpolation=cv2.INTER_AREA)
        corr = float(cv2.matchTemplate(
            resized, anchor, cv2.TM_CCOEFF_NORMED)[0, 0])
        if not math.isfinite(corr):
            corr = 0.0
        hist_sim = float(cv2.compareHist(
            anchor_hist, candidate_hist, cv2.HISTCMP_CORREL))
        if not math.isfinite(hist_sim):
            hist_sim = 0.0
        return 0.5 * max(0.0, corr) + 0.5 * max(0.0, hist_sim)

    def _yolo_proposal_recover(self, image: np.ndarray,
                               view: str) -> tuple[BBox | None, float]:
        """丢锁后的 YOLO 提案追回（文献：ARGUS 的 adaptive
        re-detection 思路）。

        把 YOLO 全部高于低阈值的检测当作候选提案，用已存外观模板
        （相关+直方图）打分验证后采纳：
        - COCO 类目标：类别命中 + 外观双确认，阈值 0.65，秒级找回；
        - 任意非 COCO 目标：纯外观验证，阈值 0.72，抗相似纹理误认。
        隔帧执行（YOLO 耗时让给模板匹配，保持实时性）。
        返回 (bbox, 分数)。
        """
        if self._frame_count % 2 != 0:
            return None, 0.0
        if not self._appearance_bank(view)[0]:
            return None, 0.0
        height, width = image.shape[:2]
        detections = self._yolo_detect(image, conf=0.15)
        expected = _mapped_coco_classes(self.target_name)
        best_score = 0.0
        best_box: BBox | None = None
        for bbox, class_name, conf in detections:
            if self._tracking_phase() and not self._plausible_region_ok(
                    bbox, view, width):
                continue
            score = self._score_candidate(image, bbox, view)
            anchor_score = self._score_anchor_at_bbox(image, bbox, view)
            class_hit = bool(class_name in expected)
            if class_hit:
                if anchor_score < 0.35:
                    continue
                score = 0.5 * max(score, float(conf)) + 0.5 * anchor_score
            else:
                if anchor_score < 0.45:
                    continue
                score = 0.5 * score + 0.5 * anchor_score
            threshold = 0.55 if class_hit else 0.60
            if score >= threshold and score > best_score:
                best_score = score
                best_box = bbox.clamp(width, height)
        return best_box, best_score if best_box is not None else 0.0

    def _cross_view_y_ok(self, candidate: BBox,
                         other_bbox: BBox | None) -> bool:
        """跨视角一致性：两相机俯仰接近（内八字朝前），同一目标在
        两路的 y 应该相近；差太多说明某一路锁错了。阈值按两路框高
        缩放（近距大框中心抖动绝对值大，固定 150px 会把合法近距
        模板纠偏误杀）。另一路无锁时跳过。"""
        if other_bbox is None:
            return True
        y_this = (candidate.y1 + candidate.y2) / 2.0
        y_other = (other_bbox.y1 + other_bbox.y2) / 2.0
        h = min(candidate.y2 - candidate.y1,
                other_bbox.y2 - other_bbox.y1)
        return abs(y_this - y_other) <= max(150.0, 0.35 * h)

    @staticmethod
    def _stereo_boxes_diverged(primary: BBox, secondary: BBox,
                               width: int, height: int) -> bool:
        """追踪中判定双目是否真分叉；横向视差不依赖相机左右顺序。"""
        clip_eps = 4.0
        for box in (primary, secondary):
            # 贴边框的中心会因裁剪偏移，不能用来否决双目锁。
            if (box.x1 <= clip_eps or box.x2 >= width - clip_eps
                    or box.y1 <= clip_eps or box.y2 >= height - clip_eps):
                return False
        y1 = (primary.y1 + primary.y2) / 2.0
        y2 = (secondary.y1 + secondary.y2) / 2.0
        h = min(primary.y2 - primary.y1,
                secondary.y2 - secondary.y1)
        dx = ((primary.x1 + primary.x2)
              - (secondary.x1 + secondary.x2)) / 2.0
        return (abs(y1 - y2) > max(150.0, 0.35 * h)
                or abs(dx) > 0.86 * width)

    def _plausible_region_ok(self, bbox: BBox, view: str,
                             width: int) -> bool:
        """内八字朝前：车在逼近（TRACKING）时目标应落在各路的
        内侧区域——左 x∈[250,width]、右局部 x∈[0,1000]。
        实测锁错时模板会把锁带到外侧边缘（右局部 1176 等）。"""
        cx = (bbox.x1 + bbox.x2) / 2.0
        if view == "left":
            return 250.0 <= cx <= float(width)
        return 0.0 <= cx <= 1000.0

    def _tracking_phase(self) -> bool:
        """树莓派是否处于逼近阶段（由 pi_state 上行消息得知）。"""
        return self._pi_state.get("state") == "TRACKING"

    def _match_yolo(self, detections: list[tuple[BBox, str, float]],
                    image: np.ndarray,
                    lock_bbox: BBox,
                    view: str,
                    kf_pred: BBox | None = None) -> tuple[BBox | None, bool]:
        """在检测里找与当前锁匹配的目标框。

        返回 (bbox, class_confirmed)：类别命中是强观测（可越卡尔曼
        门控），纯 IoU 匹配只算普通测量。
        """
        if not detections or lock_bbox is None:
            return None, False
        height, width = image.shape[:2]
        expected = _mapped_coco_classes(self.target_name)
        best_score = 0.0
        best_box: BBox | None = None
        best_force = False
        for bbox, class_name, conf in detections:
            # 与当前锁框及 KF 预测框都算 IoU：锁漂后当前框已在错误
            # 位置，只对当前框匹配的话类别命中的 YOLO 永远无法纠偏
            # （漂移后 IoU<0.15 直接不匹配）——预测框才是目标期望位置。
            iou = max(_bbox_iou(bbox, lock_bbox),
                      _bbox_iou(bbox, kf_pred) if kf_pred is not None else 0.0)
            area_ratio = bbox.area / max(lock_bbox.area, 1.0)
            anchor_score = self._score_anchor_at_bbox(image, bbox, view)
            if class_name in expected:
                # 类别命中是独立检测器的强观测：面积比宽放（近距
                # 目标尺度变化大，旧 0.25~4 会把合法检测拒掉）
                if not (0.1 <= area_ratio <= 25.0):
                    continue
                # 类别相同不代表实例相同（多个人/瓶子）；仍需初始身份
                # 的最低外观印证，避免遮挡者抢锁。
                if anchor_score < 0.32:
                    continue
                score = conf * (0.6 + 0.4 * iou) if iou >= 0.15 else 0.0
                force = True
            else:
                if not (0.25 <= area_ratio <= 4.0):
                    continue
                if iou >= 0.45 and anchor_score >= 0.45:
                    score = 0.5 * conf * iou + 0.5 * anchor_score
                    force = False
                else:
                    continue
            if score > best_score:
                best_score = score
                best_box = bbox
                best_force = force
        if best_box is None:
            return None, False
        return best_box.clamp(width, height), best_force

    @staticmethod
    def _sane(bbox: BBox, width: int, height: int) -> bool:
        if bbox.area <= 0:
            return False
        if bbox.x1 < -width * 0.1 or bbox.x2 > width * 1.1:
            return False
        if bbox.y1 < -height * 0.1 or bbox.y2 > height * 1.1:
            return False
        return True

    # ---------------- 测试网页支持 ----------------

    def set_latest(self, jpeg: bytes, result: dict[str, Any]) -> None:
        with self._latest_lock:
            self._latest_jpeg = jpeg
            self._latest_result = result
            self._frame_seq += 1
        # 广播给浏览器：result 的 track/track2.bbox 已是 raw 拼接图坐标
        # （_process_frame_locked 794 行注释），与 /stream.mjpg 画面一致，
        # 网页 canvas 直接按原始像素画，无需坐标换算。
        self.hub.notify_latest(result)

    def set_pi_state(self, state: dict[str, Any]) -> None:
        with self._latest_lock:
            self._pi_state = state
        self.hub.notify_pi_state(dict(state))

    def latest_seq(self) -> int:
        with self._latest_lock:
            return self._frame_seq

    def set_pi_motion(self, motion: dict[str, Any]) -> None:
        """树莓派逐帧上行当前运动指令（ego-motion 补偿用），
        钳位防异常值；单字段非法不影响另一字段。"""

        def _num(value: Any, lo: float, hi: float) -> float:
            try:
                v = float(value)
            except (TypeError, ValueError):
                return 0.0
            return max(lo, min(hi, v))

        self._pi_motion = {
            "v_mm_s": _num(motion.get("v"), -1000.0, 1000.0),
            "w_mrad_s": _num(motion.get("w"), -3000.0, 3000.0),
        }

    def pi_state(self) -> dict[str, Any]:
        with self._latest_lock:
            return dict(self._pi_state)

    def latest(self) -> tuple[bytes | None, dict[str, Any] | None]:
        with self._latest_lock:
            return self._latest_jpeg, self._latest_result

    def _bbox_for_display(self, stitched_width: int,
                          bbox: list | None) -> list | None:
        """网页显示 bbox（管线已在原始拼接图坐标，恒等；仅处理左右交换）。"""
        if not bbox or len(bbox) != 4:
            return None
        bx1, by1, bx2, by2 = (float(v) for v in bbox)
        if self.display_swap:
            half = max(1, int(stitched_width) // 2)
            bx1 = bx1 - half if bx1 >= half else bx1 + half
            bx2 = bx2 - half if bx2 >= half else bx2 + half
        return [bx1, by1, bx2, by2]

    def annotated_jpeg(self, jpeg: bytes, result: dict[str, Any]) -> bytes:
        """在拼接帧上画目标框/深度/分界线后重新编码（仅网页请求时执行）。"""
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jpeg
        track = result.get("track") or {}
        bbox = self._bbox_for_display(image.shape[1], track.get("bbox"))
        track2 = result.get("track2") or {}
        bbox2 = self._bbox_for_display(image.shape[1], track2.get("bbox"))
        if self.display_swap:
            half = image.shape[1] // 2
            image = np.hstack([image[:, half:], image[:, :half]])
        # 只有树莓派处于逼近阶段（TRACKING）且最新一帧确认过才画框：
        # 旋转重搜/停止时锁定已无意义，画框就是误导（2026-08-16 用户要求）。
        if bbox and track.get("status") == "tracking" \
                and self._tracking_phase():
            x1, y1, x2, y2 = (int(v) for v in bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{track.get('name') or ''} {track.get('status') or ''}"
            if result.get("depth_m") is not None:
                label += f" {result['depth_m']:.2f}m"
            cv2.putText(image, label.strip(), (x1, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if bbox2 and track2.get("status") == "tracking" \
                and self._tracking_phase():
            x1, y1, x2, y2 = (int(v) for v in bbox2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(image, f"{track2.get('status') or ''}",
                        (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
        height, width = image.shape[:2]
        cv2.line(image, (width // 2, 0), (width // 2, height),
                 (255, 255, 255), 2)
        ok, buffer = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return buffer.tobytes() if ok else jpeg

    def reground_latest(self, instruction: str) -> dict[str, Any]:
        """用最新拼接帧执行一次 Qwen 识别（测试网页"重新识别"按钮）。
        无可用帧时返回带 frame_missing=True 的结果（不推送树莓派）。"""
        jpeg, _ = self.latest()
        if jpeg is None:
            return {"found": False, "frame_missing": True,
                    "message": "尚无相机帧，请先启动树莓派端"}
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[1] < 2:
            return {"found": False, "frame_missing": True,
                    "message": "相机帧解码失败"}
        half = image.shape[1] // 2
        return self.ground(image[:, :half], image[:, half:], instruction)

    def push_ground_result(self, result: dict[str, Any],
                           instruction: str = "") -> None:
        """把网页识别结果推送给树莓派，驱动其状态机；
        无连接时暂存，重连后由 WS 处理器补发。"""
        message = {"type": "ground_result", "found": bool(result.get("found")),
                   "target_name": result.get("target_name", ""),
                   "instruction": instruction}
        with self._latest_lock:
            self._last_ground_push = message
        ws, loop = self._ws, self._loop
        if ws is None or loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                ws.send_text(json.dumps(message, ensure_ascii=False)), loop)
        except Exception:
            pass
        # 广播给浏览器（无 Pi 连接时浏览器也应看到重新识别结果）
        self.hub.notify_ground(message)


# ---------------- FastAPI 应用 ----------------

app = FastAPI(title="Pursuit Tracking Server")


def build_session(args: argparse.Namespace) -> PursuitSession:
    session = PursuitSession(
        qwen_model=args.qwen_model,
        yolo_model=args.yolo_model,
        calibration_yaml=args.calibration,
        device=args.device,
        display_swap=bool(args.swap_display),
        use_depth=not args.no_depth,
        qwen_hub_url=getattr(args, "qwen_hub_url", ""),
    )
    # YOLO 预热：首次推理的 CUDA 编译/权重上传耗时数百毫秒，若发生在
    # 追踪首帧会导致帧处理变慢、WS 队列积压。提前在启动时消化掉。
    try:
        print("[pursuit] YOLO 预热中 …")
        session._yolo_detect(np.zeros((128, 128, 3), dtype=np.uint8))
        print("[pursuit] YOLO 预热完成")
    except Exception as exc:
        print(f"[pursuit] YOLO 预热失败（将在首帧重试）: {exc}")
    app.state.pursuit_session = session
    return session


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "models": {
        "qwen": app.state.pursuit_session.qwen_model,
        "yolo": app.state.pursuit_session.yolo_model,
    }}


@app.post("/ground")
def ground(left: UploadFile = File(...),
           right: UploadFile = File(...),
           instruction: str = Form(...)) -> dict[str, Any]:
    # 注意：必须用同步 def（FastAPI 自动放线程池）——Qwen 首次加载
    # 可能耗时数十秒，async def 会阻塞事件循环，卡住 WS 帧流。
    session: PursuitSession = app.state.pursuit_session
    left_image = cv2.imdecode(np.frombuffer(left.file.read(), np.uint8),
                              cv2.IMREAD_COLOR)
    right_image = cv2.imdecode(np.frombuffer(right.file.read(), np.uint8),
                               cv2.IMREAD_COLOR)
    if left_image is None or right_image is None:
        return {"found": False, "message": "图像解码失败"}
    if left_image.shape != right_image.shape:
        right_image = cv2.resize(right_image,
                                 (left_image.shape[1], left_image.shape[0]))
    return session.ground(left_image, right_image, instruction.strip())


# ---------------- 测试网页 ----------------

_PAGE_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>pursuit 测试台</title>
<style>
body{background:#14161a;color:#d7dbe0;font-family:system-ui;margin:16px}
h1{font-size:18px;margin:0 0 4px}
.sub{font-size:12px;color:#6b7682;margin-bottom:12px}
#hud{display:flex;flex-wrap:wrap;gap:4px 20px;font-size:13px;margin:10px 0;
     background:#0b0d10;border:1px solid #2a2f36;border-radius:6px;padding:8px 12px}
#hud .lbl{color:#6b7682}
#hud .bad{color:#ff5252}
.stage{position:relative;aspect-ratio:32/9;background:#0b0d10;
        border:1px solid #2a2f36;overflow:hidden}
.stage img{position:absolute;inset:0;width:100%;height:100%;display:block}
#ovl{position:absolute;inset:0;width:100%;height:100%;z-index:1;pointer-events:none}
.corner{position:absolute;top:6px;background:#000a;padding:2px 8px;
        font-size:12px;border-radius:4px;z-index:2}
.corner.l{left:6px}
.corner.r{right:6px}
.ctl{display:flex;gap:8px;margin:12px 0}
.ctl input{flex:1;padding:8px 10px;background:#0b0d10;
           border:1px solid #2a2f36;color:#d7dbe0;border-radius:4px}
.ctl button{padding:8px 18px;background:#2f6fed;border:none;color:#fff;
            border-radius:4px;cursor:pointer}
#note{font-size:12px;color:#9aa4af;margin-top:6px;min-height:16px}
</style>
</head>
<body>
<h1>pursuit 测试台</h1>
<div class="sub">树莓派端启动 run_visual_pursuit.py 后：视频经 /stream.mjpg 流式显示，
标注经 WebSocket 在画布上绘制。流不可用时降级为 /frame.jpg 轮询模式。</div>
<div id="hud">
  <span><span class="lbl">显示</span> <b id="hud_disp">-</b>fps</span>
  <span><span class="lbl">采集</span> <b id="hud_cam">-</b>fps</span>
  <span><span class="lbl">上传</span> <b id="hud_up">-</b>fps</span>
  <span><span class="lbl">感知</span> <b id="hud_perc">-</b></span>
  <span><span class="lbl">车</span> <b id="hud_car">-</b></span>
  <span><span class="lbl">目标</span> <b id="hud_tgt">-</b></span>
  <span><span class="lbl">深度</span> <b id="hud_depth">-</b></span>
  <span><span class="lbl">bbox</span> <b id="hud_bbox">-</b></span>
</div>
<div class="stage">
  <img id="stream" src="/stream.mjpg" alt="等待帧…">
  <canvas id="ovl"></canvas>
  <span class="corner l">左摄像头</span>
  <span class="corner r">右摄像头</span>
</div>
<div class="ctl">
  <input id="cmd" placeholder="目标指令，如：追踪那个蓝色瓶子" />
  <button id="go">重新识别</button>
</div>
<div id="note"></div>
<script>
// 排障：JS 运行时错误显示在页面底部（用户反馈看不到框时快速定位）
window.onerror = (msg, src, line) => {
  try {
    document.getElementById('note').textContent =
      'JS错误: ' + msg + ' @' + line;
  } catch (e) {}
};
// ---- MJPEG 流（显示帧率 = Pi 上传率 = 相机帧率） ----
const img = document.getElementById('stream');
const canvas = document.getElementById('ovl');
const ctx = canvas.getContext('2d');
let naturalW = 2560, naturalH = 720;   // 帧自然分辨率 = canvas 内部像素
// canvas DOM 默认只有 300x150。即使视频恰好也是默认的 2560x720，
// 也必须立刻显式设置 backing store，否则 bbox 坐标全部落在画布之外。
canvas.width = naturalW; canvas.height = naturalH;
// 注意：img.onload 对 MJPEG 流只触发首帧（浏览器行为），不能数帧率。
// 显示帧率用服务端统计（/annotations 的 stream_stats 消息，5s 一次）；
// 断流检测用"最后活动时间"（onload 首帧 + frame_result 持续更新）。
let lastActivity = 0, retryDelay = 500, streamDead = true;
let pushFps = null;                    // 服务端推送帧率（stream_stats）
function touchActivity() { lastActivity = Date.now(); }
function connStream() {
  img.onload = () => {
    streamDead = false; retryDelay = 500; touchActivity();
    if (img.naturalWidth > 0) {
      naturalW = img.naturalWidth;
      naturalH = img.naturalHeight || naturalH;
      if (canvas.width !== naturalW || canvas.height !== naturalH) {
        canvas.width = naturalW; canvas.height = naturalH;
      }
    }
  };
  img.onerror = () => {
    streamDead = true;
    setTimeout(connStream, retryDelay);
    retryDelay = Math.min(retryDelay * 2, 5000);
  };
  img.src = '/stream.mjpg?t=' + Date.now();
}
connStream();
setInterval(() => {                    // 断流 watchdog：2.5s 无活动自动重连
  if (!streamDead && Date.now() - lastActivity > 2500) {
    streamDead = true; connStream();
  }
}, 1000);

// ---- 标注 WebSocket（画布绘制数据源） ----
// 注意：/stream.mjpg 与 /annotations 的 bbox 都是标定顺序的 raw 拼接
// 坐标，天然对齐，无需任何镜像/换算（display_swap 只影响 /frame.jpg
// 降级模式，新页面不用）。
let ws = null, lastWsMsg = 0, annotOnline = false;
let target = null;                     // 最新 frame_result（rAF 每帧消化一个）
let piState = null, lastTrack = null, lastDepth = null;
let percEma = null, lastResultAt = 0, lastPiAt = 0;
function clearPerceptionState() {
  target = null; cur = null; lastTrack = null; lastDepth = null;
}
function connAnnot() {
  ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://')
                     + location.host + '/annotations');
  ws.onopen = () => { annotOnline = true; };
  ws.onmessage = (ev) => {
    annotOnline = true;
    lastWsMsg = Date.now();
    let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
    if (m.type === 'hello') {
      if (m.frame_width) { naturalW = m.frame_width; canvas.width = naturalW; }
      if (m.frame_height) { naturalH = m.frame_height; canvas.height = naturalH; }
    } else if (m.type === 'frame_result') {
      target = m; lastResultAt = Date.now(); touchActivity();
      const ms = parseFloat(m.frame_ms);
      if (ms > 0) percEma = (percEma == null) ? ms : percEma * 0.8 + ms * 0.2;
      lastTrack = m.track || null;
      if (m.depth_m != null) lastDepth = m.depth_m;
    } else if (m.type === 'pi_state') {
      piState = m; lastPiAt = Date.now();
    } else if (m.type === 'stream_stats') {
      pushFps = m.push_fps;
    } else if (m.type === 'ground_result') {
      note(m.found ? ('已锁定: ' + (m.target_name || ''))
                   : ('未找到: ' + (m.instruction || '')));
    }
  };
  ws.onclose = () => {
    annotOnline = false; clearPerceptionState();
    setTimeout(connAnnot, 1000);
  };
}
connAnnot();
setInterval(() => {                    // 死连检测：8s 无任何消息主动重连
  if (ws && Date.now() - lastWsMsg > 8000) {
    try { ws.close(); } catch (e) {}
  }
}, 2000);

// ---- 画布绘制（rAF + EMA 平滑：感知 ~10fps 下框也流畅） ----
let cur = null;                        // {box, color, dashed, label}
let lastTargetAt = 0, lastFrame = 0;
function drawBox(b, color, dashed, label) {
  // 裁剪出画面部分：目标脱离视野时框不溢出画面（2026-08-18 用户
  // 反馈"脱离视觉后出现半个方框"）
  const x1 = Math.max(0, Math.min(naturalW, b[0]));
  const y1 = Math.max(0, Math.min(naturalH, b[1]));
  const x2 = Math.max(0, Math.min(naturalW, b[2]));
  const y2 = Math.max(0, Math.min(naturalH, b[3]));
  if (x2 - x1 < 2 || y2 - y1 < 2) return;   // 完全出画面不画
  const w = x2 - x1, h = y2 - y1;
  // 外发光描边（细线两遍：半透明宽 + 实色细）
  ctx.save();
  ctx.strokeStyle = color; ctx.globalAlpha = 0.38; ctx.lineWidth = 10;
  ctx.strokeRect(x1, y1, w, h);
  ctx.globalAlpha = 1; ctx.lineWidth = 4;
  ctx.setLineDash(dashed ? [8, 6] : []);
  ctx.strokeRect(x1, y1, w, h);
  ctx.setLineDash([]);
  // 目标中心十字准星（细线 + 中心点）
  const cx = x1 + w / 2, cy = y1 + h / 2;
  ctx.globalAlpha = 0.85; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx - 12, cy); ctx.lineTo(cx - 4, cy);
  ctx.moveTo(cx + 4, cy); ctx.lineTo(cx + 12, cy);
  ctx.moveTo(cx, cy - 12); ctx.lineTo(cx, cy - 4);
  ctx.moveTo(cx, cy + 4); ctx.lineTo(cx, cy + 12);
  ctx.stroke();
  ctx.beginPath(); ctx.arc(cx, cy, 2, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
  // 标签：半透明深底 + 白字 + 小圆角
  if (label) {
    ctx.font = '13px system-ui';
    const tw = ctx.measureText(label).width + 12;
    const ty = Math.max(0, y1 - 26);
    ctx.globalAlpha = 0.55;
    ctx.fillStyle = '#000';
    // 不用 roundRect（较新 API，旧浏览器会抛异常中断绘制循环）
    ctx.fillRect(x1, ty, tw, 20);
    ctx.globalAlpha = 1;
    ctx.fillStyle = color;
    ctx.fillText(label, x1 + 6, ty + 14);
  }
  ctx.restore();
}
function drawTaughtPoint(x, y, label) {
  ctx.save();
  ctx.strokeStyle = '#00bcd4'; ctx.fillStyle = '#00bcd4';
  ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.arc(x, y, 18, 0, Math.PI * 2); ctx.stroke();
  ctx.setLineDash([]); ctx.beginPath();
  ctx.moveTo(x - 25, y); ctx.lineTo(x + 25, y);
  ctx.moveTo(x, y - 25); ctx.lineTo(x, y + 25); ctx.stroke();
  ctx.font = '12px system-ui'; ctx.fillText(label, x + 22, y - 10);
  ctx.restore();
}
function tick(now) {
  const dt = Math.min(0.1, (now - lastFrame) / 1000); lastFrame = now;
  ctx.clearRect(0, 0, naturalW, naturalH);
  // 左右分界线
  ctx.strokeStyle = 'rgba(255,255,255,0.45)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(naturalW / 2, 0);
  ctx.lineTo(naturalW / 2, naturalH); ctx.stroke();
  // left_picture/right_picture 掩码质心；按实际分辨率等比缩放。
  const sx = (naturalW / 2) / 1280, sy = naturalH / 720;
  drawTaughtPoint(771.36 * sx, 544.48 * sy, 'left taught');
  drawTaughtPoint(naturalW / 2 + 526.25 * sx, 420.99 * sy, 'right taught');
  if (target) {
    const tr = target.track || {};
    if (tr.bbox && tr.bbox.length === 4) {
      // 画框优先用 YOLO 检测框（真实目标大小，2026-08-18 用户要求）；
      // 无 YOLO 框（模板/KF 路径）才用管线框
      const b = (target.yolo_box && target.yolo_box.length === 4)
        ? target.yolo_box : tr.bbox;
      const k = 1 - Math.exp(-dt / 0.09);   // EMA 时间常数 ~90ms
      const box = cur ? cur.box.map((v, i) => v + (b[i] - v) * k) : b;
      const st = tr.status || '';
      cur = { box,
              color: st === 'tracking' ? '#00e676'
                   : st === 'stale' ? '#ffd600' : '#ff5252',
              dashed: st === 'lost',
              label: ((tr.name || '') + ' ' + st).trim() };
      lastTargetAt = now;
      const tr2 = target.track2 || {};
      if (tr2.bbox && tr2.bbox.length === 4 && tr2.status === 'tracking') {
        const b2 = (target.yolo_box2 && target.yolo_box2.length === 4)
          ? target.yolo_box2 : tr2.bbox;
        drawBox(b2, '#ffd600', false, tr2.status);
      }
    }
    target = null;
  } else if (cur && now - lastTargetAt > 600) {
    cur = null;                        // 600ms 无新结果 → 清框
  }
  if (cur) drawBox(cur.box, cur.color, cur.dashed, cur.label);
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

// ---- HUD 刷新（250ms） ----
function hud(id, text, bad) {
  const el = document.getElementById(id);
  el.textContent = text;
  el.className = bad ? 'bad' : '';
}
setInterval(() => {
  // 显示帧率 = 服务端推送统计（img.onload 对 MJPEG 只触发首帧，不可用）
  hud('hud_disp', pushFps != null ? pushFps.toFixed(1) : '-');
  const piDead = lastPiAt === 0 || Date.now() - lastPiAt > 2500;
  if (piState && !piDead) {
    hud('hud_cam', 'L' + (piState.left_fps != null ? piState.left_fps.toFixed(0) : '-')
                 + '/R' + (piState.right_fps != null ? piState.right_fps.toFixed(0) : '-'));
    hud('hud_up', (piState.upload_fps != null ? piState.upload_fps.toFixed(1) : '-')
        + (piState.skip_fps ? ' 丢' + piState.skip_fps.toFixed(1) : ''));
  } else {
    hud('hud_cam', '-'); hud('hud_up', '-', true);
  }
  const percDead = !annotOnline || lastResultAt === 0
                   || Date.now() - lastResultAt > 2000;
  if (percDead) clearPerceptionState();
  hud('hud_perc', percDead ? '离线'
      : (percEma != null ? Math.round(percEma) + 'ms (≈'
         + Math.round(1000 / percEma) + 'fps)' : '-'), percDead);
  if (piState && piState.state && !piDead) {
    const cmd = (piState.cmd || []).join(',');
    hud('hud_car', piState.state + (cmd ? ' (' + cmd + ')' : ''));
  } else {
    hud('hud_car', '离线', true);
  }
  const tr = lastTrack || {};
  hud('hud_tgt', tr.status ? (tr.status + (tr.source ? '/' + tr.source : '')) : '-',
      tr.status === 'lost');
  hud('hud_depth', lastDepth != null ? lastDepth + 'm' : '-');
  hud('hud_bbox', cur ? cur.box.map(v => Math.round(v)).join(',') : '-');
}, 250);

// ---- 重新识别 ----
function note(text) {
  document.getElementById('note').textContent = text;
}
document.getElementById('go').onclick = async () => {
  const cmd = document.getElementById('cmd').value.trim();
  if (!cmd) return;
  note('识别中…（Qwen 首次加载可能需数十秒）');
  try {
    const body = new FormData();
    body.append('instruction', cmd);
    const r = await fetch('/reground', {method: 'POST', body});
    const j = await r.json();
    note(j.found ? ('已锁定: ' + (j.target_name || '') + ' view=' + (j.view || '')
         + (j.confidence != null ? ' conf=' + j.confidence : ''))
         : ('未找到: ' + (j.message || j.target_name || '')));
  } catch (e) {
    note('请求失败: ' + e);
  }
};
</script>
</body>
</html>
"""


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(_PAGE_HTML)


@app.get("/frame.jpg")
def frame_jpg() -> Response:
    session: PursuitSession = app.state.pursuit_session
    jpeg, result = session.latest()
    if jpeg is None:
        raise HTTPException(status_code=404, detail="尚无相机帧")
    with session._latest_lock:
        seq = session._frame_seq
        state = session._pi_state.get("state")
        cached = (seq == session._cache_seq
                  and state == session._cache_state
                  and session._cache_jpeg is not None)
        if cached:
            return Response(content=session._cache_jpeg,
                            media_type="image/jpeg",
                            headers={"Cache-Control": "no-store"})
    # 帧或状态变化：重新编码并缓存（网页轮询不再每 50ms 吃 25ms CPU）
    if result is not None:
        jpeg = session.annotated_jpeg(jpeg, result)
    with session._latest_lock:
        session._cache_seq = seq
        session._cache_state = state
        session._cache_jpeg = jpeg
    return Response(content=jpeg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


async def _mjpeg_frames(session: PursuitSession, boundary: str) -> Any:
    """MJPEG 流生成器：2ms 轮询最新帧序号，零重编码转发 Pi 上传的原始字节。

    2026-08-18 从"事件驱动"改为轮询：事件驱动依赖 hub.notify 链路，
    实测唤醒频率低于处理频率（curl 只收到 7.4fps，感知 27fps）——
    中间帧被 latest-wins 合并。轮询 2ms/次（500Hz 轻量检查）保证每帧
    必推（帧率 = set_latest 频率 = 感知帧率）；Pi 离线时无新帧不发
    内容，断流检测交给 JS 侧 watchdog。客户端断开 → 生成器被 cancel。
    """
    last_seq = -1
    _yielded = 0
    _t0 = time.monotonic()
    try:
        while True:
            seq = session.latest_seq()
            if seq != last_seq:
                last_seq = seq
                jpeg, _ = session.latest()
                if jpeg is not None:
                    part = (f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                            f"Content-Length: {len(jpeg)}\r\n\r\n").encode()
                    yield part + jpeg + b"\r\n"
                    _yielded += 1
                    if time.monotonic() - _t0 >= 5.0:
                        _fps = _yielded / 5.0
                        print(f"[stream] push {_fps:.1f} fps")
                        # 广播给浏览器 HUD（img.onload 对 MJPEG 流只触发
                        # 一次，浏览器侧数帧不可靠——2026-08-18 用户实测
                        # 显示 1fps 但画面流畅）
                        session.hub.notify_push_fps(round(_fps, 1))
                        _yielded = 0
                        _t0 = time.monotonic()
            await asyncio.sleep(0.002)
    finally:
        pass


@app.get("/stream.mjpg")
async def stream_mjpg() -> Response:
    """浏览器视频流：multipart/x-mixed-replace，帧率 = Pi 上传率 = 相机帧率。"""
    session: PursuitSession = app.state.pursuit_session
    if session.latest()[0] is None:
        raise HTTPException(status_code=404, detail="尚无相机帧")
    # 每连接独立 boundary，防 JPEG 内容与分隔串碰撞
    boundary = "----pursuit" + uuid.uuid4().hex
    return StreamingResponse(
        _mjpeg_frames(session, boundary),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={"Cache-Control": "no-store", "Connection": "keep-alive"},
    )


@app.get("/status")
def status() -> dict[str, Any]:
    session: PursuitSession = app.state.pursuit_session
    _, result = session.latest()
    if not result:
        return {"frame": False, "result": {}}
    # 网页展示用：把管线 bbox（校正坐标）换算为原始拼接图坐标，
    # 与 /frame.jpg 上画的框一致；树莓派经 WS 收到的仍是校正坐标。
    result = json.loads(json.dumps(result))
    bbox = (result.get("track") or {}).get("bbox")
    if bbox:
        result["track"]["bbox"] = session._bbox_for_display(
            session.frame_width, bbox)
    return {"frame": True, "result": result,
            "pi_state": session.pi_state()}


@app.post("/reground")
def reground(instruction: str = Form(...)) -> dict[str, Any]:
    session: PursuitSession = app.state.pursuit_session
    instruction = instruction.strip()
    result = session.reground_latest(instruction)
    if not result.get("frame_missing"):
        session.push_ground_result(result, instruction)
    return result


@app.websocket("/pursuit_ws")
async def pursuit_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    session: PursuitSession = app.state.pursuit_session
    session._ws = websocket
    session._loop = asyncio.get_running_loop()
    try:
        hello = json.loads(await websocket.receive_text())
        if hello.get("type") != "pursuit_start":
            await websocket.send_text(json.dumps(
                {"type": "error", "message": "首条消息必须是 pursuit_start"}))
            await websocket.close()
            return
        print(f"[pursuit] 相机流已连接: {hello.get('frame_width')}x"
              f"{hello.get('frame_height')} fps={hello.get('upload_fps')}")
        # 摄像头分辨率自动探测后可能非 1280 宽：ego-motion 焦距按比例缩放
        try:
            half_width = float(hello.get("frame_width") or 0) / 2.0
            if half_width > 0:
                session._focal_scale = half_width / 1280.0
        except (TypeError, ValueError):
            pass
        await websocket.send_text(json.dumps(
            {"type": "ack", "message": "pursuit session started"},
            ensure_ascii=False))
        # 重连后补发最近的网页识别结果，保证用户新指令不因断线丢失；
        # 补发后清空——否则每次重连都会重放旧指令，干扰 Pi 状态机
        # （2026-08-18 测试发现：步骤 8 的 reground 在步骤 11 重连后仍被补发）
        with session._latest_lock:
            pending_push = session._last_ground_push
            session._last_ground_push = None
        if pending_push is not None:
            await websocket.send_text(json.dumps(pending_push,
                                                 ensure_ascii=False))
        def _handle_text(text: str) -> bool:
            """处理文本消息（pi_state/motion/close）；返回 True=应断开。"""
            if text == "close":
                return True
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                return False
            if data.get("type") == "pi_state":
                session.set_pi_state(data)
            elif data.get("type") == "motion":
                session.set_pi_motion(data)
            return False

        while True:
            # 阻塞等首条消息（按帧到达节奏驱动，不忙等）
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            frame_bytes: bytes | None = None
            if message.get("bytes") is not None:
                frame_bytes = message["bytes"]
            elif message.get("text"):
                if _handle_text(message["text"]):
                    break
                continue   # 文本消息不驱动帧处理
            # 吸干处理间隔内积压的消息：逐帧处理是状态管线（KF/模板
            # 每帧推进），处理陈旧帧既浪费算力又让结果滞后真实画面。
            # 只保留最新一帧；motion/pi_state 与帧同流保序，立即应用
            # 即保持最新配对。
            disconnect = False
            while True:
                try:
                    newer = await asyncio.wait_for(
                        websocket.receive(), timeout=0.002)
                except asyncio.TimeoutError:
                    break
                except WebSocketDisconnect:
                    disconnect = True
                    break
                if newer.get("type") == "websocket.disconnect":
                    disconnect = True
                    break
                if newer.get("bytes") is not None:
                    frame_bytes = newer["bytes"]
                elif newer.get("text"):
                    if _handle_text(newer["text"]):
                        disconnect = True
                        break
            if disconnect:
                break
            frame = cv2.imdecode(
                np.frombuffer(frame_bytes, np.uint8),
                cv2.IMREAD_COLOR)
            if frame is not None and frame.shape[1] >= 2:
                result = session.process_frame(frame)
                session.set_latest(frame_bytes, result)
                await websocket.send_text(
                    json.dumps(result, ensure_ascii=False))
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        import traceback
        print(f"[pursuit] WS 异常: {exc}")
        traceback.print_exc()
    finally:
        if session._ws is websocket:
            session._ws = None
            session._loop = None
    print("[pursuit] 相机流已断开")


@app.websocket("/annotations")
async def annotations_ws(websocket: WebSocket) -> None:
    """浏览器标注订阅：逐帧 frame_result / pi_state / ground_result。

    感知管线与浏览器之间隔着"每浏览器一个队列 + 独立发送协程"，
    感知侧只有 put_nowait（latest-wins），永不 await 客户端——
    慢浏览器只阻塞自己，拖不垮 pursuit_ws 主循环。
    """
    await websocket.accept()
    session: PursuitSession = app.state.pursuit_session
    session.hub.attach(asyncio.get_running_loop())
    key, queue = session.hub.register_annot()
    try:
        await websocket.send_text(json.dumps({
            "type": "hello",
            "display_swap": session.display_swap,
            "frame_width": session.frame_width or None,
            "frame_height": session.frame_height or None,
        }))
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                # 空闲保活：既是浏览器侧活性信号（JS 8s 无消息判死重连），
                # 也让空闲断连的浏览器能在下一次 send 时被发现并清理。
                await websocket.send_text(json.dumps({"type": "keepalive"}))
                continue
            await websocket.send_text(json.dumps(message, ensure_ascii=False))
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        session.hub.unregister_annot(key)


def _port_in_use(port: int) -> bool:
    """探测端口是否已被占用（防重复实例：2026-08-17 实测双实例
    并存导致 CUDA OOM——旧实例未杀净时新实例应直接拒绝启动）。"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Pursuit tracking server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--qwen-model",
                        default=str(HOME / "3090/models/qwen/Qwen3-VL-2B-Instruct"))
    parser.add_argument(
        "--qwen-hub-url", default="http://127.0.0.1:8010",
        help="共享 Qwen Hub；非空时 8021 不本地加载 Qwen 模型",
    )
    parser.add_argument("--yolo-model",
                        default=str(HOME / "3090/models/yolo/yolo11n-seg.pt"))
    parser.add_argument("--calibration",
                        default=str(Path(__file__).resolve().parent
                                    / "configs" / "stereo_calibration.yaml"))
    parser.add_argument("--device", default="cpu",
                        help="Qwen 推理设备。3090 显存常被其他服务占满"
                             "（OOM），默认 cpu 慢但稳；显存富余时可改 cuda")
    parser.add_argument("--swap-display", action="store_true",
                        help="网页显示交换左右半幅（实车物理左右与标定"
                             "命名相反时开启；不影响管线与深度）")
    parser.add_argument("--no-depth", action="store_true",
                        help="禁用双目深度（摄像头位置改动导致标定失效时"
                             "使用；距离只由树莓派雷达提供）")
    args = parser.parse_args()

    if _port_in_use(args.port):
        print(f"[pursuit] 端口 {args.port} 已被占用——已有实例在运行？"
              f"拒绝启动（双实例会耗尽显存 OOM）。"
              f"如需重启：先杀旧实例再启动。")
        raise SystemExit(1)

    build_session(args)
    import uvicorn
    print(f"[pursuit] 启动服务 http://{args.host}:{args.port} "
          f"(Qwen={args.qwen_model}, YOLO={args.yolo_model}, "
          f"calibration={args.calibration})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
