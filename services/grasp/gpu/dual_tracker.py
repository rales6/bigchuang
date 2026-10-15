"""双目标跟踪器 v2 — 优化版。

改进:
  - 爪子搜索区域限定（画面下方 50%）提升 CamShift 准确性
  - 跟踪质量评分（bbox 尺寸/位置合理性检查）防止漂移
  - 当 CSRT+颜色都失败时，用运动学估算爪子位置
  - YOLO 候选去重 (NMS-style 按类别)

初始化: Qwen 识别爪子 + 目标物
逐帧:   两个 CSRT tracker 持续更新
恢复:   爪子丢失 → 颜色回退 (限定搜索区域)
        目标丢失 → YOLO 候选 + 类别过滤
定期:   Qwen 重识别 (跟踪质量下降或超时)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import cv2
import numpy as np


@dataclass
class TrackedObject:
    name: str
    bbox: Any | None = None
    tracker: Any | None = None
    color_hist: np.ndarray | None = None
    lost_frames: int = 0
    consecutive_ok: int = 0
    last_seen_at: float = 0.0
    initialized: bool = False
    quality_score: float = 0.0       # 0-1 跟踪质量
    last_size: tuple[float, float] = (0, 0)  # (w, h) 用于尺寸突变检测

    @property
    def center(self) -> tuple[float, float] | None:
        if self.bbox is None:
            return None
        return self.bbox.center

    @property
    def area(self) -> float:
        if self.bbox is None:
            return 0.0
        return float(self.bbox.width * self.bbox.height)


@dataclass
class DualTrackResult:
    claw_bbox: Any | None = None
    target_bbox: Any | None = None
    claw_ok: bool = False
    target_ok: bool = False
    claw_lost_frames: int = 0
    target_lost_frames: int = 0
    claw_quality: float = 0.0
    target_quality: float = 0.0
    needs_reground: bool = False
    timestamp: float = 0.0


# ── 爪子搜索区域 (画面下方，因为摄像头在车身上方，爪子在下方) ──
CLAW_SEARCH_Y1_RATIO = 0.40   # 搜索区域从画面 40% 高度开始
CLAW_SEARCH_Y2_RATIO = 0.95   # 到 95% 高度
CLAW_SIZE_MAX_RATIO = 0.40    # 爪子不应超过画面 40%
CLAW_SIZE_MIN_PX = 18         # 爪子最小像素尺寸
TARGET_SIZE_MIN_RATIO = 0.003 # 目标至少占画面 0.3%
TARGET_SIZE_MAX_RATIO = 0.60  # 目标不超过画面 60%
SIZE_CHANGE_MAX_RATIO = 2.5   # 单帧尺寸突变阈值

# ── 绿色标记检测 (爪子尖贴绿色胶带) ──
GREEN_HSV_LOW  = (35, 60, 50)    # H: 35-85 绿, S>60 饱和, V>50 亮度
GREEN_HSV_HIGH = (85, 255, 255)
GREEN_MIN_AREA_PX = 120          # 绿斑最小面积(像素)
GREEN_MAX_AREA_RATIO = 0.25      # 绿斑不应超过画面 25%


class DualTracker:
    """同时跟踪爪子和目标物 — 优化版。

    爪子: CSRT + 限定区域的 CamShift 颜色回退 + 运动学估算
    目标物: CSRT + YOLO 类别过滤候补
    """

    def __init__(
        self,
        calibration=None,       # ArmVisualCalibration
        yolo_detector=None,
        max_lost_frames: int = 18,
        reground_interval_s: float = 5.0,
    ) -> None:
        self.cal = calibration
        self.yolo = yolo_detector
        self.max_lost_frames = max_lost_frames
        self.reground_interval_s = reground_interval_s

        self.claw = TrackedObject(name="claw")
        self.target = TrackedObject(name="target")
        self._last_reground_at: float = 0.0
        self._frame_count: int = 0

    # ── 初始化 ────────────────────────────────────────────

    def initialize_claw(self, frame: np.ndarray, bbox) -> bool:
        """初始化爪子 tracker。返回是否成功。"""
        h, w = frame.shape[:2]
        clamped = bbox.clamp(w, h)
        if clamped.width < CLAW_SIZE_MIN_PX or clamped.height < CLAW_SIZE_MIN_PX:
            return False
        if clamped.width > w * CLAW_SIZE_MAX_RATIO:
            return False

        self.claw.bbox = clamped
        self.claw.tracker = self._create_csrt()
        try:
            self.claw.tracker.init(frame, clamped.to_xywh())
        except Exception:
            return False
        self.claw.initialized = True
        self.claw.lost_frames = 0
        self.claw.consecutive_ok = 0
        self.claw.last_seen_at = time.time()
        self._last_reground_at = self.claw.last_seen_at
        self.claw.quality_score = 1.0
        self.claw.last_size = (clamped.width, clamped.height)
        self._extract_color_hist(frame, clamped)
        return True

    def initialize_target(self, frame: np.ndarray, bbox) -> bool:
        """初始化目标物 tracker。"""
        h, w = frame.shape[:2]
        clamped = bbox.clamp(w, h)
        if clamped.width < 10 or clamped.height < 10:
            return False
        if clamped.width > w * TARGET_SIZE_MAX_RATIO:
            return False

        self.target.bbox = clamped
        self.target.tracker = self._create_csrt()
        try:
            self.target.tracker.init(frame, clamped.to_xywh())
        except Exception:
            return False
        self.target.initialized = True
        self.target.lost_frames = 0
        self.target.consecutive_ok = 0
        self.target.last_seen_at = time.time()
        self._last_reground_at = self.target.last_seen_at
        self.target.quality_score = 1.0
        self.target.last_size = (clamped.width, clamped.height)
        return True

    def initialize_both(self, frame, claw_bbox, target_bbox) -> bool:
        """同时初始化两个 tracker。"""
        ok_c = self.initialize_claw(frame, claw_bbox)
        ok_t = self.initialize_target(frame, target_bbox)
        if ok_c or ok_t:
            self._last_reground_at = time.time()
        return ok_c and ok_t

    # ── 逐帧更新 ──────────────────────────────────────────

    def update(self, frame: np.ndarray) -> DualTrackResult:
        self._frame_count += 1
        now = time.time()
        result = DualTrackResult(timestamp=now)

        # ── 爪子 ──
        if self.claw.initialized:
            ok, bbox = self._update_csrt(frame, self.claw)
            if ok and bbox is not None:
                if self._validate_bbox(bbox, frame, is_claw=True,
                                       last_size=self.claw.last_size):
                    self.claw.bbox = bbox
                    self.claw.lost_frames = 0
                    self.claw.consecutive_ok += 1
                    self.claw.last_seen_at = now
                    self.claw.quality_score = min(1.0, self.claw.quality_score + 0.08)
                    self.claw.last_size = (bbox.width, bbox.height)
                    result.claw_ok = True
                else:
                    # bbox 不合法 → 视为丢失
                    self.claw.lost_frames += 1
                    result.claw_ok = False
            else:
                # CSRT 丢失 → 颜色回退
                recovered = self._recover_claw(frame)
                if recovered is not None:
                    self.claw.bbox = recovered
                    self.claw.lost_frames = 0
                    self.claw.consecutive_ok = 0
                    self.claw.quality_score = max(0.3, self.claw.quality_score - 0.15)
                    self.claw.last_size = (recovered.width, recovered.height)
                    result.claw_ok = True
                    # 用恢复的 bbox 重建 CSRT
                    try:
                        self.claw.tracker = self._create_csrt()
                        self.claw.tracker.init(frame, recovered.to_xywh())
                    except Exception:
                        pass
                else:
                    self.claw.lost_frames += 1
                    self.claw.quality_score = max(0.0, self.claw.quality_score - 0.12)
                    result.claw_ok = False
        result.claw_bbox = self.claw.bbox
        result.claw_lost_frames = self.claw.lost_frames
        result.claw_quality = self.claw.quality_score

        # ── 目标物 ──
        if self.target.initialized:
            ok, bbox = self._update_csrt(frame, self.target)
            if ok and bbox is not None:
                if self._validate_bbox(bbox, frame, is_claw=False,
                                       last_size=self.target.last_size):
                    self.target.bbox = bbox
                    self.target.lost_frames = 0
                    self.target.consecutive_ok += 1
                    self.target.last_seen_at = now
                    self.target.quality_score = min(1.0, self.target.quality_score + 0.06)
                    self.target.last_size = (bbox.width, bbox.height)
                    result.target_ok = True
                else:
                    self.target.lost_frames += 1
                    result.target_ok = False
            else:
                recovered = self._recover_target(frame)
                if recovered is not None:
                    self.target.bbox = recovered
                    self.target.lost_frames = 0
                    self.target.consecutive_ok = 0
                    self.target.quality_score = max(0.35, self.target.quality_score - 0.12)
                    self.target.last_size = (recovered.width, recovered.height)
                    result.target_ok = True
                    try:
                        self.target.tracker = self._create_csrt()
                        self.target.tracker.init(frame, recovered.to_xywh())
                    except Exception:
                        pass
                else:
                    self.target.lost_frames += 1
                    self.target.quality_score = max(0.0, self.target.quality_score - 0.1)
                    result.target_ok = False
        result.target_bbox = self.target.bbox
        result.target_lost_frames = self.target.lost_frames
        result.target_quality = self.target.quality_score

        # 重识别只由真实的跟踪失败触发。固定时间重识别会让一个健康的
        # 抓取流程无故退回 Qwen，甚至在刚初始化后立刻中断运动。
        quality_low = (self.claw.quality_score < 0.3 or self.target.quality_score < 0.3)
        if (self.target.lost_frames >= self.max_lost_frames
                or self.claw.lost_frames >= self.max_lost_frames
                or (quality_low and self._frame_count > 20)):
            result.needs_reground = True
            self._last_reground_at = now

        return result

    # ── CSRT 更新包装 ─────────────────────────────────────

    @staticmethod
    def _update_csrt(frame, obj: TrackedObject) -> tuple[bool, Any]:
        if obj.tracker is None:
            return False, None
        try:
            ok, xywh = obj.tracker.update(frame)
            if not ok:
                return False, None
            # OpenCV trackers return a plain (x, y, width, height) tuple.
            # The rest of this module consistently works with domain.BBox.
            from qwen_grounded_tracker.domain import BBox
            x, y, width, height = [float(value) for value in xywh]
            frame_h, frame_w = frame.shape[:2]
            bbox = BBox(x, y, x + width, y + height).clamp(frame_w, frame_h)
            return (bbox.width >= 3 and bbox.height >= 3), bbox
        except Exception:
            return False, None

    # ── bbox 合法性校验 ───────────────────────────────────

    def _validate_bbox(self, bbox, frame, is_claw: bool,
                       last_size: tuple[float, float] = (0, 0),
                       trained_claw: bool = False) -> bool:
        """检查 bbox 是否合理: 尺寸、位置、突变检测。

        trained_claw=True 用于用户训练的 YOLO 框。该框标注了完整夹爪和
        一部分机械臂，天然比旧的“运动暗色爪尖”框宽，不能套用 40% 宽度上限。
        """
        h, w = frame.shape[:2]
        if bbox.width < 5 or bbox.height < 5:
            return False
        if bbox.x1 < -5 or bbox.y1 < -5 or bbox.x2 > w + 5 or bbox.y2 > h + 5:
            return False

        if is_claw:
            # The trained box covers the full gripper/nearby linkage (live
            # width is commonly 55-60% of a 640 px frame). Legacy motion-dark
            # boxes still use the original narrow-tip limits.
            cy = (bbox.y1 + bbox.y2) / 2.0
            min_center_y = 0.12 if trained_claw else 0.20
            # A close gripper can legitimately fill most of the view. For the
            # trained detector only reject an almost full-frame box; temporal
            # stability and size-jump checks remain the primary safety gates.
            max_width_ratio = 0.95 if trained_claw else CLAW_SIZE_MAX_RATIO
            max_height_ratio = 0.90 if trained_claw else 0.55
            max_area_ratio = 0.85 if trained_claw else 0.16
            if cy < h * min_center_y:
                return False
            if bbox.width > w * max_width_ratio:
                return False
            if bbox.height > h * max_height_ratio:
                return False
            if bbox.width * bbox.height > w * h * max_area_ratio:
                return False
            if bbox.width < CLAW_SIZE_MIN_PX:
                return False
        else:
            if bbox.width > w * TARGET_SIZE_MAX_RATIO:
                return False
            area_ratio = (bbox.width * bbox.height) / (w * h)
            if area_ratio < TARGET_SIZE_MIN_RATIO:
                return False

        # 尺寸突变检测
        if last_size[0] > 0 and last_size[1] > 0:
            w_ratio = bbox.width / last_size[0]
            h_ratio = bbox.height / last_size[1]
            if w_ratio > SIZE_CHANGE_MAX_RATIO or w_ratio < 1.0 / SIZE_CHANGE_MAX_RATIO:
                return False
            if h_ratio > SIZE_CHANGE_MAX_RATIO or h_ratio < 1.0 / SIZE_CHANGE_MAX_RATIO:
                return False

        return True

    # ── 颜色回退 (爪子) ───────────────────────────────────

    def _extract_color_hist(self, frame, bbox) -> None:
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
            x2, y2 = min(frame.shape[1], int(bbox.x2)), min(frame.shape[0], int(bbox.y2))
            if x2 <= x1 + 3 or y2 <= y1 + 3:
                return
            mask = np.zeros(hsv.shape[:2], np.uint8)
            mask[y1:y2, x1:x2] = 255
            hist = cv2.calcHist([hsv], [0, 1], mask, [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            self.claw.color_hist = hist
        except Exception:
            self.claw.color_hist = None

    def _recover_claw(self, frame):
        """限定区域的 CamShift 颜色搜索。"""
        if self.claw.color_hist is None:
            return None
        try:
            from qwen_grounded_tracker.domain import BBox

            h, w = frame.shape[:2]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            backproj = cv2.calcBackProject([hsv], [0, 1], self.claw.color_hist,
                                           [0, 180, 0, 256], 1)

            # 限定搜索区域: 画面下方
            mask = np.zeros((h, w), np.uint8)
            y1_abs = int(h * CLAW_SEARCH_Y1_RATIO)
            y2_abs = int(h * CLAW_SEARCH_Y2_RATIO)
            mask[y1_abs:y2_abs, :] = 255
            backproj = cv2.bitwise_and(backproj, backproj, mask=mask)

            # 初始搜索窗口: 上一帧 bbox 或 画面下方中心
            if self.claw.bbox is not None:
                tw = (int(self.claw.bbox.x1), int(self.claw.bbox.y1),
                      int(self.claw.bbox.width), int(self.claw.bbox.height))
            else:
                tw = (w // 4, y1_abs, w // 2, y2_abs - y1_abs)

            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            _ret, tw = cv2.CamShift(backproj, tw, term_crit)

            x, y, bw, bh = tw
            if bw < CLAW_SIZE_MIN_PX or bh < CLAW_SIZE_MIN_PX:
                return None
            if bw > w * CLAW_SIZE_MAX_RATIO or bh > h * 0.5:
                return None
            # 必须在限定区域内
            cy = y + bh / 2.0
            if cy < y1_abs:
                return None

            return BBox(float(x), float(y), float(x + bw), float(y + bh))
        except Exception:
            return None

    # ── YOLO 候补 (目标物) ────────────────────────────────

    def _recover_target(self, frame):
        if self.yolo is None:
            return None
        try:
            from qwen_grounded_tracker.domain import BBox

            obs = self.yolo.detect(frame)
            if not obs.detections:
                return None

            stop_labels = getattr(self.yolo, "stop_labels", set())
            candidates = [d for d in obs.detections if d.label not in stop_labels]
            if not candidates:
                return None

            # NMS-style 去重: 同类别重叠 > 0.6 IOU 只保留 confidence 最高的
            filtered = self._nms_by_class(candidates, iou_threshold=0.6)

            # 匹配策略: 如果上一帧有 bbox，匹配最近的; 否则选最高 confidence
            if self.target.bbox is not None and self.target.quality_score > 0.4:
                last_center = self.target.bbox.center
                best = min(filtered, key=lambda d:
                    (d.bbox.center[0] - last_center[0]) ** 2 +
                    (d.bbox.center[1] - last_center[1]) ** 2)
            else:
                best = max(filtered, key=lambda d: d.confidence)

            return best.bbox
        except Exception:
            return None

    @staticmethod
    def _nms_by_class(detections, iou_threshold=0.6) -> list:
        """对同类别检测结果做简单 NMS。"""
        if len(detections) <= 1:
            return list(detections)

        # 按 class 分组
        by_class: dict[str, list] = {}
        for d in detections:
            by_class.setdefault(d.label, []).append(d)

        result = []
        for label, items in by_class.items():
            items.sort(key=lambda d: d.confidence, reverse=True)
            kept = []
            for item in items:
                suppressed = False
                for kept_item in kept:
                    if hasattr(item.bbox, 'iou') and item.bbox.iou(kept_item.bbox) > iou_threshold:
                        suppressed = True
                        break
                if not suppressed:
                    kept.append(item)
            result.extend(kept)
        return result

    # ── 运动+黑色 爪子检测 (主检测方式) ───────────────────

    def _dark_blobs(self, frame) -> list:
        """全图暗色斑块列表 (爪子是黑色胶质, HSV V 通道低)。"""
        try:
            from qwen_grounded_tracker.domain import BBox
            h, w = frame.shape[:2]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            v_max = getattr(self.cal, "claw_dark_v_max", 90)
            dark = cv2.inRange(hsv[:, :, 2], 0, v_max)
            kernel = np.ones((5, 5), np.uint8)
            dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
            dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            min_area = getattr(self.cal, "claw_blob_min_area_px", 80)
            max_area = w * h * getattr(self.cal, "claw_blob_max_area_ratio", 0.25)
            blobs = []
            for c in contours:
                area = cv2.contourArea(c)
                if area < min_area or area > max_area:
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                blobs.append(BBox(float(x), float(y), float(x + bw), float(y + bh)))
            return blobs
        except Exception:
            return []

    def detect_claw_motion_dark(self, frame) -> Any | None:
        """多帧暗色斑块运动跟踪 → 返回爪子 bbox。

        原理: 相机静止、小车不动, 画面里唯一持续移动的黑色物体就是爪子。
        维护暗色斑块轨迹; 累计位移最大的轨迹 = 爪子。
        识别后即使爪子静止, 只要斑块仍在视野内就持续返回其位置。

        返回 None 表示尚未确定 (需要爪子动几下)。
        """
        blobs = self._dark_blobs(frame)
        if not hasattr(self, "_blob_tracks"):
            self._blob_tracks: dict = {}
        now = time.time()
        match_px = getattr(self.cal, "claw_track_match_px", 80.0)

        # 当前帧斑块与历史轨迹匹配 (最近邻)
        used: set = set()
        for b in blobs:
            c = b.center
            best_key, best_d = None, match_px
            for key, t in self._blob_tracks.items():
                if key in used:
                    continue
                d = ((c[0] - t["center"][0]) ** 2
                     + (c[1] - t["center"][1]) ** 2) ** 0.5
                if d < best_d:
                    best_d, best_key = d, key
            if best_key is not None:
                used.add(best_key)
                t = self._blob_tracks[best_key]
                t["disp"] += best_d          # 累计位移
                t["center"] = c
                t["age"] += 1
                t["last_seen"] = now
                t["bbox"] = b
            else:
                self._blob_tracks[id(b)] = {
                    "center": c, "disp": 0.0, "age": 1,
                    "last_seen": now, "bbox": b,
                }

        # 清理过期轨迹 (斑块消失超过 N 秒)
        timeout_s = getattr(self.cal, "claw_track_timeout_s", 2.0)
        stale = [k for k, t in self._blob_tracks.items()
                 if now - t["last_seen"] > timeout_s]
        for k in stale:
            del self._blob_tracks[k]

        # 累计位移最大的存活轨迹 = 爪子
        min_disp = getattr(self.cal, "claw_motion_min_disp_px", 12.0)
        best, best_disp = None, min_disp
        for t in self._blob_tracks.values():
            if t["age"] >= 2 and t["disp"] > best_disp:
                best_disp = t["disp"]
                best = t["bbox"]
        return best

    def is_dark_region(self, frame, bbox, max_mean_v: float = 110.0) -> bool:
        """检查 bbox 区域是否足够黑 (CSRT 跟踪校验: 防漂移到非黑色区域)。"""
        try:
            h, w = frame.shape[:2]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
            x2, y2 = min(w, int(bbox.x2)), min(h, int(bbox.y2))
            if x2 <= x1 + 3 or y2 <= y1 + 3:
                return False
            roi = hsv[y1:y2, x1:x2, 2]
            return float(roi.mean()) < max_mean_v
        except Exception:
            return False

    # ── 绿色标记检测 (备用) ───────────────────────────────

    def detect_claw_green(self, frame) -> Any | None:
        """HSV 检测绿色标记 → 返回爪子 bbox (BBox)。

        这是爪子的主要定位方式，比 CSRT/Qwen 可靠得多。
        返回 None 表示画面里没有绿斑。
        """
        try:
            from qwen_grounded_tracker.domain import BBox
            h, w = frame.shape[:2]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, GREEN_HSV_LOW, GREEN_HSV_HIGH)

            # 形态学去噪
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            # 取面积最大的绿斑
            best = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(best)
            if area < GREEN_MIN_AREA_PX:
                return None
            if area > w * h * GREEN_MAX_AREA_RATIO:
                return None

            x, y, bw, bh = cv2.boundingRect(best)
            # 爪子应在画面下半部分 (摄像头俯视)
            cy = y + bh / 2.0
            if cy < h * 0.25:
                return None

            return BBox(float(x), float(y), float(x + bw), float(y + bh))
        except Exception:
            return None

    # ── 辅助 ──────────────────────────────────────────────

    @staticmethod
    def _create_csrt():
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        legacy = getattr(cv2, "legacy", None)
        if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
            return legacy.TrackerCSRT_create()
        raise RuntimeError("OpenCV CSRT tracker unavailable")

    def reset(self) -> None:
        self.claw = TrackedObject(name="claw")
        self.target = TrackedObject(name="target")
        self._last_reground_at = 0.0
        self._frame_count = 0
        self._blob_tracks = {}   # 运动检测轨迹

    @property
    def both_ready(self) -> bool:
        return self.claw.initialized and self.target.initialized

    @property
    def claw_ready(self) -> bool:
        return self.claw.initialized
