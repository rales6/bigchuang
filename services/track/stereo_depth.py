# -*- coding: utf-8 -*-
"""双目校正与 SGBM 深度估计。

读取 stereo_calibration.yaml（camera_lidar_experiments_20260805_192453
标定结果），构建左右校正映射；对目标 bbox 区域做半分辨率 SGBM 视差，
取中位数换算深度。

z = fx_rect * B / disparity
    fx_rect=1251.497px, B=141.107mm → z = 0.1766 / disparity(px, 全分辨率)
    半分辨率（640 宽）下 disparity 减半，z = 0.1766 / (2*d_half)
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import yaml


class StereoDepth:
    def __init__(self, calibration_yaml: str, scale: float = 0.5) -> None:
        with open(calibration_yaml, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        stereo = data["stereo"]
        left = data["left"]
        right = data["right"]
        rect = data["rectification"]
        image_size = tuple(int(v) for v in data["image_size"])
        self.image_size = image_size
        self.fx_rect = float(rect["P1"][0][0])
        self.baseline_m = float(stereo["baseline_mm_norm"]) / 1000.0
        self.scale = float(scale)

        def matrix(values):
            return np.asarray(values, dtype=np.float64)

        k1, d1, k2, d2 = (matrix(left["camera_matrix"]),
                          matrix(left["dist_coeffs"]),
                          matrix(right["camera_matrix"]),
                          matrix(right["dist_coeffs"]))
        r1, r2 = matrix(rect["R1"]), matrix(rect["R2"])
        p1, p2 = matrix(rect["P1"]), matrix(rect["P2"])

        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            k1, d1, r1, p1, image_size, cv2.CV_32FC1)
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            k2, d2, r2, p2, image_size, cv2.CV_32FC1)

        half_width = max(64, int(round(image_size[0] * self.scale / 16.0) * 16))
        half_height = max(64, int(round(image_size[1] * self.scale)))
        self.half_size = (half_width, half_height)

        # mode 必须用 HH：cv2 5.0 实测 3WAY 在整幅合成图上几乎无有效输出，
        # HH 全图有效且中位数正确；OpenCV 4.x 下两者均可用，HH 同样稳定。
        # 不同 numDisparities 的 matcher 按需创建：cv2 5.0 的 SGBM 在
        # 大 numDisparities 下画面左侧大面积无输出（有效区随 nd 减小而
        # 扩大），depth_in_region 会 512→256→128 自适应回退。
        self._matchers: dict[int, Any] = {}
        self._last_good_nd = 512

    def _matcher(self, num_disparities: int):
        if num_disparities not in self._matchers:
            self._matchers[num_disparities] = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=num_disparities,
                blockSize=9,
                P1=8 * 3 * 9 * 9,
                P2=32 * 3 * 9 * 9,
                disp12MaxDiff=2,
                uniquenessRatio=8,
                speckleWindowSize=100,
                speckleRange=2,
                mode=cv2.STEREO_SGBM_MODE_HH,
            )
        return self._matchers[num_disparities]

    def rectify(self, left_frame: np.ndarray,
                right_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left = cv2.remap(left_frame, self.map1_left, self.map2_left,
                         cv2.INTER_LINEAR)
        right = cv2.remap(right_frame, self.map1_right, self.map2_right,
                          cv2.INTER_LINEAR)
        return left, right

    def bbox_rect_to_raw(self, bbox_xyxy: tuple[float, float, float, float],
                         view: str) -> tuple[float, float, float, float]:
        """校正图 bbox → 原始图 bbox（把追踪框画回原始拼接帧用）。

        反向查 rect→raw 映射：取 bbox 覆盖的校正像素，其原始坐标的
        外接框即为结果。畸变校正把主点从 640 挪到 ~674、裁剪了四周
        大边框，两坐标系差可达 170px，画框必须先换算。
        """
        map1, map2 = ((self.map1_left, self.map2_left) if view == "left"
                      else (self.map1_right, self.map2_right))
        x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
        height, width = map1.shape[:2]
        rx1 = int(max(0, min(width - 1, math.floor(x1))))
        rx2 = int(max(rx1 + 1, min(width, math.ceil(x2))))
        ry1 = int(max(0, min(height - 1, math.floor(y1))))
        ry2 = int(max(ry1 + 1, min(height, math.ceil(y2))))
        raw_x = map1[ry1:ry2, rx1:rx2]
        raw_y = map2[ry1:ry2, rx1:rx2]
        return (float(raw_x.min()), float(raw_y.min()),
                float(raw_x.max()) + 1.0, float(raw_y.max()) + 1.0)

    def bbox_raw_to_rect(self, bbox_xyxy: tuple[float, float, float, float],
                         view: str) -> tuple[float, float, float, float] | None:
        """原始图像 bbox → 校正后视图 bbox（利用 rect→raw 映射反向求）。

        追踪（YOLO/CSRT）在校正视图上进行，Qwen 返回的是原始图坐标，
        必须换算后才能初始化 CSRT；否则初始化与逐帧更新坐标系不一致，
        小目标会漂移锁错。
        """
        map1, map2 = ((self.map1_left, self.map2_left) if view == "left"
                      else (self.map1_right, self.map2_right))
        x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
        mask = ((map1 >= x1) & (map1 <= x2)
                & (map2 >= y1) & (map2 <= y2))
        rows, cols = np.nonzero(mask)
        if cols.size < 8:
            return None
        return (float(cols.min()), float(rows.min()),
                float(cols.max()) + 1.0, float(rows.max()) + 1.0)

    def disparity(self, left_rect: np.ndarray,
                  right_rect: np.ndarray,
                  num_disparities: int | None = None) -> np.ndarray:
        small_left = cv2.resize(left_rect, self.half_size,
                                interpolation=cv2.INTER_AREA)
        small_right = cv2.resize(right_rect, self.half_size,
                                 interpolation=cv2.INTER_AREA)
        matcher = self._matcher(num_disparities or self._last_good_nd)
        disparity = matcher.compute(
            cv2.cvtColor(small_left, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(small_right, cv2.COLOR_BGR2GRAY),
        ).astype(np.float32) / 16.0
        # 无效视差（SGBM 输出 -16/16=-1 或 0）置 NaN
        disparity[disparity <= 0.0] = np.nan
        return disparity

    def depth_in_region(self, left_rect: np.ndarray, right_rect: np.ndarray,
                        bbox_xyxy: tuple[float, float, float, float],
                        min_disp_px: float = 4.0) -> tuple[float | None, float]:
        """返回 (深度米, 置信度)。置信度为 bbox 内有效视差点占比。

        numDisparities 覆盖范围：512→z≥0.17m、256→z≥0.35m、128→z≥0.7m；
        先试上次成功的档位，失败按 512→256→128 回退（cv2 5.0 下
        512 的左侧大面积无效，回退扩大有效区；4.x 首轮 512 即成功）。
        """
        x1, y1, x2, y2 = bbox_xyxy
        width, height = self.half_size
        rx1 = int(max(0, min(width - 1, round(x1 * self.scale))))
        rx2 = int(max(rx1 + 1, min(width, round(x2 * self.scale))))
        ry1 = int(max(0, min(height - 1, round(y1 * self.scale))))
        ry2 = int(max(ry1 + 1, min(height, round(y2 * self.scale))))
        candidates = [self._last_good_nd] + \
            [nd for nd in (512, 256, 128) if nd != self._last_good_nd]
        for num_disp in candidates:
            disparity = self.disparity(left_rect, right_rect, num_disp)
            region = disparity[ry1:ry2, rx1:rx2]
            valid = region[np.isfinite(region) & (region >= min_disp_px)]
            if valid.size < 8:
                continue
            confidence = float(valid.size) / float(max(1, region.size))
            median_disp = float(np.median(valid))
            if median_disp < min_disp_px:
                continue
            self._last_good_nd = num_disp
            # 半分辨率下有效视差 = 全分辨率视差 * scale
            depth = (self.fx_rect * self.baseline_m) / (median_disp / self.scale)
            return depth, confidence
        return None, 0.0
